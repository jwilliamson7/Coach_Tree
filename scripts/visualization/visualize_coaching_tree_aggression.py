#!/usr/bin/env python3
"""
Visualize NFL Coaching Tree with Aggression Gene Overlay

This script creates an interactive visualization of the NFL coaching tree using
NetworkX with Kamada-Kawai layout, colored by each coach's aggression gene.
Focuses on coordinator-to-head-coach promotions.

Usage:
    python visualize_coaching_tree_aggression.py [--output tree.html]
"""

import argparse
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.offline as pyo
import networkx as nx
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AggressionGeneTreeVisualizer:
    """Visualize coaching tree with aggression gene overlay"""
    
    def __init__(self, tree_dir: str = "data/processed/coaching_tree",
                 gene_dir: str = "data/processed/coaching_genes"):
        self.tree_dir = Path(tree_dir)
        self.gene_dir = Path(gene_dir)
        self.relationships_df = None
        self.coaches_data = None
        self.aggression_data = None
        self.G = None
        
    def load_data(self) -> None:
        """Load coaching tree and aggression gene data"""
        logger.info("Loading coaching tree data...")
        
        # Load relationships
        relationships_file = self.tree_dir / "relationships.csv"
        if not relationships_file.exists():
            raise FileNotFoundError(f"Relationships file not found: {relationships_file}")
        
        self.relationships_df = pd.read_csv(relationships_file)
        logger.info(f"Loaded {len(self.relationships_df):,} relationships")
        
        # Load coaches data
        coaches_file = self.tree_dir / "coaches.json"
        if not coaches_file.exists():
            raise FileNotFoundError(f"Coaches file not found: {coaches_file}")
            
        with open(coaches_file, 'r') as f:
            self.coaches_data = json.load(f)
        logger.info(f"Loaded {len(self.coaches_data):,} coaches")
        
        # Load aggression gene data
        aggression_file = self.gene_dir / "aggression_gene_by_year.csv"
        if not aggression_file.exists():
            logger.warning(f"Aggression gene file not found: {aggression_file}")
            self.aggression_data = pd.DataFrame()
        else:
            self.aggression_data = pd.read_csv(aggression_file)
            logger.info(f"Loaded aggression data for {len(self.aggression_data):,} coach-years")
            
            # Calculate average aggression per coach
            if 'composite_aggression' in self.aggression_data.columns:
                self.avg_aggression = self.aggression_data.groupby('head_coach')['composite_aggression'].mean()
                logger.info(f"Calculated average aggression for {len(self.avg_aggression)} coaches")
            else:
                logger.warning("No composite_aggression column found")
                self.avg_aggression = pd.Series()
    
    def build_coordinator_network(self, years_filter: Tuple[int, int] = None) -> nx.DiGraph:
        """
        Build NetworkX graph with only coordinator-to-HC relationships
        
        Args:
            years_filter: Optional year range filter (defaults to 1970-present)
            
        Returns:
            NetworkX directed graph
        """
        logger.info("Building coordinator-to-HC network...")
        
        # Filter for coordinator to head coach relationships only
        df = self.relationships_df[
            self.relationships_df['relationship_type'] == 'coordinator_to_hc'
        ].copy()
        
        # Default to filtering out pre-1970 coaches
        if years_filter:
            start_year, end_year = years_filter
        else:
            start_year = 1970  # Default minimum year
            end_year = 2025    # Future-proof max year
            logger.info(f"Using default year filter: {start_year}-{end_year}")
        
        df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]
        
        logger.info(f"Using {len(df):,} coordinator-to-HC relationships")
        
        # Create directed graph
        self.G = nx.DiGraph()
        
        # Track all coaches involved
        all_coaches = set()
        
        # Build edges and nodes
        for _, row in df.iterrows():
            parent = row['parent_name']
            child = row['child_name']
            
            all_coaches.add(parent)
            all_coaches.add(child)
            
            # Add nodes with attributes
            if not self.G.has_node(parent):
                self.G.add_node(parent, 
                              id=row['parent_id'],
                              role='Head Coach')
                              
            if not self.G.has_node(child):
                self.G.add_node(child,
                              id=row['child_id'], 
                              role=row['child_role'])
            
            # Add edge (mentor HC -> protege who became HC)
            self.G.add_edge(parent, child,
                          year=row['year'],
                          team=row['team'])
        
        # Add aggression gene data to nodes
        for node in self.G.nodes():
            if hasattr(self, 'avg_aggression') and node in self.avg_aggression.index:
                self.G.nodes[node]['aggression'] = self.avg_aggression[node]
                self.G.nodes[node]['has_gene_data'] = True
            else:
                self.G.nodes[node]['aggression'] = 0
                self.G.nodes[node]['has_gene_data'] = False
            
            # Add career data
            coach_id = self.G.nodes[node].get('id', node.lower().replace(' ', '_'))
            if coach_id in self.coaches_data:
                coach_data = self.coaches_data[coach_id]
                if coach_data.get('career'):
                    years = list(coach_data['career'].keys())
                    if years:
                        self.G.nodes[node]['career_start'] = min(int(y) for y in years)
                        self.G.nodes[node]['career_end'] = max(int(y) for y in years)
        
        # Calculate metrics
        for node in self.G.nodes():
            self.G.nodes[node]['n_mentors'] = self.G.in_degree(node)
            self.G.nodes[node]['n_proteges'] = self.G.out_degree(node)
            descendants = nx.descendants(self.G, node)
            self.G.nodes[node]['n_descendants'] = len(descendants)
        
        logger.info(f"Created graph with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")
        return self.G
    
    def create_kamada_layout(self) -> Dict[str, Tuple[float, float]]:
        """Create Kamada-Kawai layout for the graph"""
        logger.info("Creating Kamada-Kawai layout...")
        pos = nx.kamada_kawai_layout(self.G)
        logger.info("Layout complete")
        return pos
    
    def create_aggression_plot(self, pos: Dict[str, Tuple[float, float]]) -> go.Figure:
        """
        Create plot colored by aggression gene
        
        Args:
            pos: Node positions from layout algorithm
            
        Returns:
            Plotly figure
        """
        logger.info("Creating aggression gene visualization...")
        
        fig = go.Figure()
        
        # Draw edges (thinner, more subtle)
        edge_traces = []
        for edge in self.G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=0.5, color='rgba(200, 200, 200, 0.5)'),
                hoverinfo='none',
                showlegend=False
            )
            edge_traces.append(edge_trace)
        
        for trace in edge_traces:
            fig.add_trace(trace)
        
        # Prepare node data
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        hover_text = []
        
        # Separate nodes with and without gene data
        nodes_with_data = []
        nodes_without_data = []
        
        for node in self.G.nodes():
            if self.G.nodes[node].get('has_gene_data', False):
                nodes_with_data.append(node)
            else:
                nodes_without_data.append(node)
        
        # Process all nodes
        for node in self.G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            
            # Get node attributes
            node_attrs = self.G.nodes[node]
            aggression = node_attrs.get('aggression', 0)
            has_data = node_attrs.get('has_gene_data', False)
            
            # Create hover text
            hover_info = f"<b>{node}</b><br>"
            
            if has_data:
                hover_info += f"<b>Aggression Gene: {aggression:.4f}</b><br>"
                
                # Add component genes if available
                if hasattr(self, 'aggression_data') and not self.aggression_data.empty:
                    coach_data = self.aggression_data[
                        self.aggression_data['head_coach'] == node
                    ]
                    if not coach_data.empty:
                        avg_4th = coach_data['fourth_down_aggression'].mean() if 'fourth_down_aggression' in coach_data else 0
                        avg_pass = coach_data['pass_heavy_aggression'].mean() if 'pass_heavy_aggression' in coach_data else 0
                        avg_deep = coach_data['deep_pass_aggression'].mean() if 'deep_pass_aggression' in coach_data else 0
                        avg_2pt = coach_data['two_point_aggression'].mean() if 'two_point_aggression' in coach_data else 0
                        
                        hover_info += f"  - 4th Down: {avg_4th:.4f}<br>"
                        hover_info += f"  - Pass Heavy: {avg_pass:.4f}<br>"
                        hover_info += f"  - Deep Pass: {avg_deep:.4f}<br>"
                        hover_info += f"  - Two Point: {avg_2pt:.4f}<br>"
            else:
                hover_info += f"<i>No aggression data available</i><br>"
            
            # Add mentor information
            mentors = list(self.G.predecessors(node))
            if mentors:
                hover_info += f"<br><b>Mentors ({len(mentors)}):</b><br>"
                for mentor in mentors:
                    # Get mentor's aggression gene if available
                    mentor_aggression = ""
                    if hasattr(self, 'avg_aggression') and mentor in self.avg_aggression.index:
                        mentor_aggression = f" (Gene: {self.avg_aggression[mentor]:+.4f})"
                    hover_info += f"  • {mentor}{mentor_aggression}<br>"
            else:
                hover_info += f"<br>Mentors: 0<br>"
                
            hover_info += f"HC Protégés: {node_attrs.get('n_proteges', 0)}<br>"
            hover_info += f"Total HC Descendants: {node_attrs.get('n_descendants', 0)}<br>"
            
            if 'career_start' in node_attrs:
                hover_info += f"Career: {node_attrs['career_start']}-{node_attrs.get('career_end', 'present')}"
            
            hover_text.append(hover_info)
            
            # Color by aggression (use gray for no data)
            if has_data:
                node_color.append(aggression)
            else:
                node_color.append(None)  # Will handle separately
            
            # Size by total connections
            total_connections = node_attrs.get('n_mentors', 0) + node_attrs.get('n_proteges', 0)
            node_size.append(max(8, min(25, 10 + total_connections * 2)))
        
        # Add nodes WITH aggression data
        data_mask = [self.G.nodes[node].get('has_gene_data', False) for node in self.G.nodes()]
        
        if any(data_mask):
            node_trace = go.Scatter(
                x=[node_x[i] for i, has_data in enumerate(data_mask) if has_data],
                y=[node_y[i] for i, has_data in enumerate(data_mask) if has_data],
                mode='markers+text',
                text=[node_text[i] for i, has_data in enumerate(data_mask) if has_data],
                textposition='top center',
                textfont=dict(size=9),
                hovertext=[hover_text[i] for i, has_data in enumerate(data_mask) if has_data],
                hoverinfo='text',
                marker=dict(
                    size=[node_size[i] for i, has_data in enumerate(data_mask) if has_data],
                    color=[c for c, has_data in zip(node_color, data_mask) if has_data and c is not None],
                    colorscale='RdBu',  # Red (aggressive) to Blue (conservative)
                    cmin=-0.05,  # Set range for better color distribution
                    cmax=0.05,
                    showscale=True,
                    colorbar=dict(
                        title='Aggression<br>Gene',
                        thickness=15,
                        len=0.7,
                        x=1.02
                    ),
                    line=dict(width=1, color='black')
                ),
                showlegend=False,
                name='Coaches with Gene Data'
            )
            fig.add_trace(node_trace)
        
        # Add nodes WITHOUT aggression data (gray)
        if any(not d for d in data_mask):
            no_data_trace = go.Scatter(
                x=[node_x[i] for i, has_data in enumerate(data_mask) if not has_data],
                y=[node_y[i] for i, has_data in enumerate(data_mask) if not has_data],
                mode='markers+text',
                text=[node_text[i] for i, has_data in enumerate(data_mask) if not has_data],
                textposition='top center',
                textfont=dict(size=9),
                hovertext=[hover_text[i] for i, has_data in enumerate(data_mask) if not has_data],
                hoverinfo='text',
                marker=dict(
                    size=[node_size[i] for i, has_data in enumerate(data_mask) if not has_data],
                    color='lightgray',
                    line=dict(width=1, color='gray')
                ),
                showlegend=False,
                name='Coaches without Gene Data'
            )
            fig.add_trace(no_data_trace)
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="NFL Coaching Tree: Coordinator to Head Coach Promotions<br>" +
                     "<sub>Colored by Aggression Gene (Red=Aggressive, Blue=Conservative, Gray=No Data)</sub>",
                x=0.5,
                font=dict(size=18)
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=60),
            annotations=[
                dict(
                    text="Node size = total HC connections | Kamada-Kawai layout",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=-0.05,
                    xanchor="center", yanchor="top",
                    font=dict(size=10, color="gray")
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            width=1400,
            height=900
        )
        
        return fig
    
    def analyze_gene_propagation(self) -> Dict:
        """
        Analyze how aggression genes propagate through the tree
        
        Returns:
            Dictionary with propagation analysis
        """
        logger.info("Analyzing aggression gene propagation...")
        
        if not hasattr(self, 'avg_aggression') or self.avg_aggression.empty:
            logger.warning("No aggression data available for analysis")
            return {}
        
        propagation_data = {
            'parent_child_correlation': [],
            'aggressive_lineages': [],
            'conservative_lineages': []
        }
        
        # Analyze parent-child aggression correlation
        for parent, child in self.G.edges():
            if (parent in self.avg_aggression.index and 
                child in self.avg_aggression.index):
                parent_agg = self.avg_aggression[parent]
                child_agg = self.avg_aggression[child]
                propagation_data['parent_child_correlation'].append({
                    'parent': parent,
                    'child': child,
                    'parent_aggression': parent_agg,
                    'child_aggression': child_agg,
                    'difference': child_agg - parent_agg
                })
        
        # Find most aggressive lineages
        for node in self.G.nodes():
            if node in self.avg_aggression.index:
                descendants = nx.descendants(self.G, node)
                if descendants:
                    desc_with_data = [d for d in descendants if d in self.avg_aggression.index]
                    if desc_with_data:
                        avg_desc_agg = np.mean([self.avg_aggression[d] for d in desc_with_data])
                        if self.avg_aggression[node] > 0.02:  # Aggressive threshold
                            propagation_data['aggressive_lineages'].append({
                                'root': node,
                                'root_aggression': self.avg_aggression[node],
                                'avg_descendant_aggression': avg_desc_agg,
                                'n_descendants': len(desc_with_data)
                            })
                        elif self.avg_aggression[node] < -0.02:  # Conservative threshold
                            propagation_data['conservative_lineages'].append({
                                'root': node,
                                'root_aggression': self.avg_aggression[node],
                                'avg_descendant_aggression': avg_desc_agg,
                                'n_descendants': len(desc_with_data)
                            })
        
        return propagation_data


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Visualize coaching tree with aggression genes')
    parser.add_argument('--output', type=str, default='outputs/visualizations/coaching_tree_aggression.html',
                       help='Output HTML file name')
    parser.add_argument('--filter_years', type=str,
                       help='Year range filter (e.g., "2010-2024")')
    parser.add_argument('--analyze', action='store_true',
                       help='Perform gene propagation analysis')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("COACHING TREE WITH AGGRESSION GENE VISUALIZATION")
    print("=" * 80)
    
    try:
        # Initialize visualizer
        visualizer = AggressionGeneTreeVisualizer()
        
        # Load data
        visualizer.load_data()
        
        # Parse year filter
        years_filter = None
        if args.filter_years:
            try:
                start_year, end_year = map(int, args.filter_years.split('-'))
                years_filter = (start_year, end_year)
                print(f"Filtering years: {start_year}-{end_year}")
            except ValueError:
                print(f"Invalid year format: {args.filter_years}")
                return
        
        # Build network (coordinator to HC only)
        G = visualizer.build_coordinator_network(years_filter)
        
        # Create Kamada-Kawai layout
        pos = visualizer.create_kamada_layout()
        
        # Create visualization
        print("Creating aggression gene visualization...")
        fig = visualizer.create_aggression_plot(pos)
        
        # Save to HTML
        output_path = Path(args.output)
        pyo.plot(fig, filename=str(output_path), auto_open=True)
        
        print(f"Visualization saved to: {output_path}")
        print("Opening in browser...")
        
        # Analyze gene propagation if requested
        if args.analyze:
            print("\n" + "=" * 80)
            print("AGGRESSION GENE PROPAGATION ANALYSIS")
            print("=" * 80)
            
            analysis = visualizer.analyze_gene_propagation()
            
            if analysis.get('parent_child_correlation'):
                correlations = analysis['parent_child_correlation']
                if correlations:
                    # Calculate correlation
                    parent_aggs = [c['parent_aggression'] for c in correlations]
                    child_aggs = [c['child_aggression'] for c in correlations]
                    
                    if len(parent_aggs) > 1:
                        correlation = np.corrcoef(parent_aggs, child_aggs)[0, 1]
                        print(f"\nParent-Child Aggression Correlation: {correlation:.3f}")
                        print(f"(Based on {len(correlations)} mentor-protégé pairs with data)")
                    
                    # Show biggest increases/decreases
                    sorted_by_diff = sorted(correlations, key=lambda x: x['difference'])
                    
                    print("\nBiggest Aggression Increases (Protégé > Mentor):")
                    for item in sorted_by_diff[-5:]:
                        print(f"  {item['parent']:20} ({item['parent_aggression']:+.4f}) -> "
                              f"{item['child']:20} ({item['child_aggression']:+.4f}) "
                              f"Delta={item['difference']:+.4f}")
                    
                    print("\nBiggest Aggression Decreases (Protégé < Mentor):")
                    for item in sorted_by_diff[:5]:
                        print(f"  {item['parent']:20} ({item['parent_aggression']:+.4f}) -> "
                              f"{item['child']:20} ({item['child_aggression']:+.4f}) "
                              f"Delta={item['difference']:+.4f}")
            
            if analysis.get('aggressive_lineages'):
                print("\nMost Aggressive Coaching Trees:")
                for lineage in sorted(analysis['aggressive_lineages'], 
                                     key=lambda x: x['root_aggression'], 
                                     reverse=True)[:5]:
                    print(f"  {lineage['root']:20} (Gene: {lineage['root_aggression']:+.4f}) "
                          f"-> {lineage['n_descendants']} descendants "
                          f"(Avg: {lineage['avg_descendant_aggression']:+.4f})")
            
            if analysis.get('conservative_lineages'):
                print("\nMost Conservative Coaching Trees:")
                for lineage in sorted(analysis['conservative_lineages'], 
                                     key=lambda x: x['root_aggression'])[:5]:
                    print(f"  {lineage['root']:20} (Gene: {lineage['root_aggression']:+.4f}) "
                          f"-> {lineage['n_descendants']} descendants "
                          f"(Avg: {lineage['avg_descendant_aggression']:+.4f})")
        
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        raise


if __name__ == "__main__":
    main()