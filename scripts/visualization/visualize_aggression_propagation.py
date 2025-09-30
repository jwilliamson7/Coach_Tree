#!/usr/bin/env python3
"""
Visualize Aggression Gene Propagation Through Coaching Lineages

This script creates a line chart showing how aggression genes change with 
distance from root coaches. Each line represents a coaching "school" 
(connected component of the coaching tree).

Usage:
    python visualize_aggression_propagation.py [--output propagation.html]
"""

import argparse
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
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


class AggressionPropagationAnalyzer:
    """Analyze aggression gene propagation through coaching lineages"""
    
    def __init__(self, tree_dir: str = "data/processed/coaching_tree",
                 gene_dir: str = "data/processed/coaching_genes"):
        self.tree_dir = Path(tree_dir)
        self.gene_dir = Path(gene_dir)
        self.relationships_df = None
        self.coaches_data = None
        self.aggression_data = None
        self.G = None
        self.avg_aggression = None
        
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
            raise FileNotFoundError(f"Aggression gene file not found: {aggression_file}")
            
        self.aggression_data = pd.read_csv(aggression_file)
        logger.info(f"Loaded aggression data for {len(self.aggression_data):,} coach-years")
        
        # Calculate average aggression per coach
        if 'composite_aggression' in self.aggression_data.columns:
            self.avg_aggression = self.aggression_data.groupby('head_coach')['composite_aggression'].mean()
            logger.info(f"Calculated average aggression for {len(self.avg_aggression)} coaches")
        else:
            raise ValueError("No composite_aggression column found in aggression data")
    
    def build_coordinator_network(self, years_filter: Tuple[int, int] = None) -> nx.DiGraph:
        """Build NetworkX graph with only coordinator-to-HC relationships"""
        logger.info("Building coordinator-to-HC network...")
        
        # Filter for coordinator to head coach relationships only
        df = self.relationships_df[
            self.relationships_df['relationship_type'] == 'coordinator_to_hc'
        ].copy()
        
        # Default to filtering out pre-1970 coaches
        if years_filter:
            start_year, end_year = years_filter
        else:
            start_year = 1970
            end_year = 2025
            logger.info(f"Using default year filter: {start_year}-{end_year}")
        
        df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]
        logger.info(f"Using {len(df):,} coordinator-to-HC relationships")
        
        # Create directed graph
        self.G = nx.DiGraph()
        
        # Build edges and nodes
        for _, row in df.iterrows():
            parent = row['parent_name']
            child = row['child_name']
            
            # Add nodes
            if not self.G.has_node(parent):
                self.G.add_node(parent, id=row['parent_id'])
            if not self.G.has_node(child):
                self.G.add_node(child, id=row['child_id'])
            
            # Add edge (mentor HC -> protege who became HC)
            self.G.add_edge(parent, child, year=row['year'], team=row['team'])
        
        # Add aggression gene data to nodes
        for node in self.G.nodes():
            if node in self.avg_aggression.index:
                self.G.nodes[node]['aggression'] = self.avg_aggression[node]
                self.G.nodes[node]['has_gene_data'] = True
            else:
                self.G.nodes[node]['aggression'] = None
                self.G.nodes[node]['has_gene_data'] = False
        
        logger.info(f"Created graph with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")
        return self.G
    
    def find_coaching_lineages(self, max_lineages: int = 8) -> List[Dict]:
        """
        Find coaching lineages from the most influential root coaches
        
        Args:
            max_lineages: Maximum number of lineages to analyze
            
        Returns:
            List of lineage dictionaries
        """
        logger.info("Finding coaching lineages from influential root coaches...")
        
        # Find root nodes (coaches with no predecessors)
        roots = [n for n in self.G.nodes() if self.G.in_degree(n) == 0]
        logger.info(f"Found {len(roots)} root coaches")
        
        # If we have many roots, select the most influential ones
        if len(roots) > max_lineages:
            # Calculate influence metrics for roots
            root_influence = {}
            
            for root in roots:
                # Number of total descendants
                descendants = nx.descendants(self.G, root)
                desc_with_data = [d for d in descendants if self.G.nodes[d]['has_gene_data']]
                
                # Influence score: descendants with gene data
                influence = len(desc_with_data)
                root_influence[root] = influence
            
            # Select top roots by influence
            top_roots = sorted(root_influence.items(), key=lambda x: x[1], reverse=True)[:max_lineages]
            selected_roots = [root for root, _ in top_roots]
            
            logger.info(f"Selected top {len(selected_roots)} most influential roots")
        else:
            selected_roots = roots
        
        lineages = []
        
        # Analyze each selected root
        for i, root in enumerate(selected_roots):
            logger.info(f"Analyzing lineage from {root}...")
            
            # Get all descendants and their distances
            lineage_data = self._analyze_lineage_from_root(self.G, root)
            
            if lineage_data:
                # Get lineage metadata
                coaches_with_data = [l for l in lineage_data if l['has_data']]
                
                descendants = nx.descendants(self.G, root)
                desc_with_data = [d for d in descendants if self.G.nodes[d]['has_gene_data']]
                
                avg_lineage_aggression = None
                if coaches_with_data:
                    avg_lineage_aggression = np.mean([l['aggression'] for l in coaches_with_data])
                
                lineages.append({
                    'lineage_id': i + 1,
                    'root': root,
                    'root_aggression': self.G.nodes[root].get('aggression'),
                    'total_descendants': len(descendants),
                    'descendants_with_data': len(desc_with_data),
                    'avg_aggression': avg_lineage_aggression,
                    'lineage_data': lineage_data
                })
        
        logger.info(f"Analyzed {len(lineages)} coaching lineages")
        return lineages
    
    def _analyze_lineage_from_root(self, graph: nx.DiGraph, root: str) -> List[Dict]:
        """
        Analyze lineage distances from a root coach using BFS
        
        Args:
            graph: School subgraph
            root: Root coach name
            
        Returns:
            List of lineage dictionaries
        """
        lineages = []
        
        # Use BFS to efficiently calculate distances from root
        try:
            distances = nx.single_source_shortest_path_length(graph, root)
        except nx.NetworkXError:
            return lineages
        
        # Create lineage points for all reachable nodes
        for node, distance in distances.items():
            aggression = graph.nodes[node].get('aggression')
            lineage_point = {
                'coach': node,
                'distance': distance,
                'aggression': aggression,
                'has_data': graph.nodes[node].get('has_gene_data', False)
            }
            lineages.append(lineage_point)
        
        return lineages
    
    def create_propagation_plot(self, lineages: List[Dict], max_lineages: int = 10) -> go.Figure:
        """
        Create line plot showing aggression propagation by distance from root
        
        Args:
            lineages: List of coaching lineage data
            max_lineages: Maximum number of lineages to display
            
        Returns:
            Plotly figure
        """
        logger.info("Creating aggression propagation visualization...")
        
        fig = go.Figure()
        
        # Color palette for different lineages (use hex colors)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        lineage_stats = []
        
        # Process each lineage
        for i, lineage in enumerate(lineages[:max_lineages]):
            lineage_id = lineage['lineage_id']
            lineage_data = lineage['lineage_data']
            
            # Filter to coaches with aggression data
            valid_points = [l for l in lineage_data if l['has_data']]
            
            if len(valid_points) < 2:  # Need at least 2 points for a line
                continue
            
            # Group by distance and calculate statistics
            distance_groups = {}
            for point in valid_points:
                dist = point['distance']
                if dist not in distance_groups:
                    distance_groups[dist] = []
                distance_groups[dist].append(point['aggression'])
            
            # Calculate mean aggression at each distance
            distances = sorted(distance_groups.keys())
            mean_aggressions = []
            std_aggressions = []
            coach_counts = []
            
            for dist in distances:
                values = distance_groups[dist]
                mean_aggressions.append(np.mean(values))
                std_aggressions.append(np.std(values) if len(values) > 1 else 0)
                coach_counts.append(len(values))
            
            # Create hover text
            hover_text = []
            for j, dist in enumerate(distances):
                coaches_at_dist = [p['coach'] for p in valid_points if p['distance'] == dist]
                hover_info = f"<b>{lineage['root']} Lineage</b><br>"
                hover_info += f"Distance: {dist}<br>"
                hover_info += f"Mean Aggression: {mean_aggressions[j]:.4f}<br>"
                hover_info += f"Coaches: {coach_counts[j]}<br>"
                hover_info += f"Names: {', '.join(coaches_at_dist[:3])}"
                if len(coaches_at_dist) > 3:
                    hover_info += f" (+{len(coaches_at_dist)-3} more)"
                hover_text.append(hover_info)
            
            # Add line for this lineage
            color = colors[i % len(colors)]
            
            fig.add_trace(go.Scatter(
                x=distances,
                y=mean_aggressions,
                mode='lines+markers',
                name=f'{lineage["root"]} (n={lineage["descendants_with_data"]})',
                line=dict(color=color, width=2),
                marker=dict(
                    size=[max(5, min(15, 5 + count)) for count in coach_counts],
                    color=color,
                    line=dict(width=1, color='black')
                ),
                hovertext=hover_text,
                hoverinfo='text'
            ))
            
            # Add error bars if we have standard deviations
            if any(std > 0 for std in std_aggressions):
                fig.add_trace(go.Scatter(
                    x=distances + distances[::-1],
                    y=[m + s for m, s in zip(mean_aggressions, std_aggressions)] + 
                      [m - s for m, s in zip(mean_aggressions[::-1], std_aggressions[::-1])],
                    fill='toself',
                    fillcolor=f'rgba({",".join(map(str, px.colors.hex_to_rgb(color)))},0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=False,
                    name=f'{lineage["root"]} ±1σ'
                ))
            
            # Store stats for summary
            lineage_stats.append({
                'lineage_id': lineage_id,
                'root': lineage['root'],
                'descendants_with_data': lineage['descendants_with_data'],
                'max_distance': max(distances),
                'root_aggression': mean_aggressions[0] if distances else None,
                'final_aggression': mean_aggressions[-1] if distances else None,
                'aggression_change': (mean_aggressions[-1] - mean_aggressions[0]) if len(mean_aggressions) > 1 else 0
            })
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="Aggression Gene Propagation Through Coaching Lineages<br>" +
                     "<sub>Each line = coaching school, X = distance from root coach, Y = mean aggression gene</sub>",
                x=0.5,
                font=dict(size=16)
            ),
            xaxis=dict(
                title="Distance from Root Coach (Generations)",
                tickmode='linear',
                tick0=0,
                dtick=1
            ),
            yaxis=dict(
                title="Mean Aggression Gene",
                zeroline=True,
                zerolinecolor='black',
                zerolinewidth=1
            ),
            hovermode='closest',
            showlegend=True,
            legend=dict(
                x=1.02,
                y=1,
                bgcolor='rgba(255,255,255,0.8)'
            ),
            margin=dict(b=50, l=60, r=150, t=80),
            width=1200,
            height=700,
            plot_bgcolor='white',
            grid=dict(
                rows=1, columns=1,
                pattern='independent'
            )
        )
        
        # Add horizontal line at y=0 (neutral aggression)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                     annotation_text="Neutral Aggression", 
                     annotation_position="bottom right")
        
        return fig, lineage_stats


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Visualize aggression gene propagation through lineages')
    parser.add_argument('--output', type=str, default='outputs/visualizations/aggression_propagation.html',
                       help='Output HTML file name')
    parser.add_argument('--filter_years', type=str,
                       help='Year range filter (e.g., "2010-2024")')
    parser.add_argument('--max_lineages', type=int, default=8,
                       help='Maximum number of lineages to display')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("AGGRESSION GENE PROPAGATION ANALYZER")
    print("=" * 80)
    
    try:
        # Initialize analyzer
        analyzer = AggressionPropagationAnalyzer()
        
        # Load data
        analyzer.load_data()
        
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
        
        # Build network
        G = analyzer.build_coordinator_network(years_filter)
        
        # Find coaching lineages
        print(f"Finding coaching lineages...")
        lineages = analyzer.find_coaching_lineages(args.max_lineages)
        
        print(f"\nFound {len(lineages)} coaching lineages:")
        for lineage in lineages:
            print(f"  {lineage['root']:20}: {lineage['total_descendants']} total descendants "
                  f"({lineage['descendants_with_data']} with gene data), "
                  f"Avg aggression: {lineage['avg_aggression']:.4f}" if lineage['avg_aggression'] else "N/A")
        
        # Create visualization
        print(f"\nCreating propagation visualization for {len(lineages)} lineages...")
        fig, lineage_stats = analyzer.create_propagation_plot(lineages, args.max_lineages)
        
        # Save to HTML
        output_path = Path(args.output)
        pyo.plot(fig, filename=str(output_path), auto_open=True)
        
        print(f"Visualization saved to: {output_path}")
        print("Opening in browser...")
        
        # Print summary statistics
        print("\n" + "=" * 80)
        print("PROPAGATION ANALYSIS SUMMARY")
        print("=" * 80)
        
        for stat in lineage_stats:
            print(f"\n{stat['root']} Lineage ({stat['descendants_with_data']} coaches with data):")
            print(f"  Max distance: {stat['max_distance']} generations")
            if stat['root_aggression'] is not None:
                print(f"  Root aggression: {stat['root_aggression']:+.4f}")
                print(f"  Final aggression: {stat['final_aggression']:+.4f}")
                print(f"  Change: {stat['aggression_change']:+.4f}")
        
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        raise


if __name__ == "__main__":
    main()