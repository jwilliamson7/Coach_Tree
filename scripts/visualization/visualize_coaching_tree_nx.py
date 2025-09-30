#!/usr/bin/env python3
"""
Visualize NFL Coaching Tree with NetworkX

This script creates an interactive visualization of the NFL coaching tree using
Plotly and NetworkX. The NetworkX library provides superior graph algorithms
for layout and analysis.

Usage:
    python visualize_coaching_tree_nx.py [--output tree.html] [--layout dot]
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


class NetworkXCoachingTreeVisualizer:
    """Create interactive visualizations of the NFL coaching tree using NetworkX"""
    
    def __init__(self, data_dir: str = "data/processed/coaching_tree"):
        self.data_dir = Path(data_dir)
        self.relationships_df = None
        self.coaches_data = None
        self.G = None  # NetworkX graph
        
    def load_data(self) -> None:
        """Load coaching tree data from files"""
        logger.info("Loading coaching tree data...")
        
        # Load relationships
        relationships_file = self.data_dir / "relationships.csv"
        if not relationships_file.exists():
            raise FileNotFoundError(f"Relationships file not found: {relationships_file}")
        
        self.relationships_df = pd.read_csv(relationships_file)
        logger.info(f"Loaded {len(self.relationships_df):,} relationships")
        
        # Load coaches data
        coaches_file = self.data_dir / "coaches.json"
        if not coaches_file.exists():
            raise FileNotFoundError(f"Coaches file not found: {coaches_file}")
            
        with open(coaches_file, 'r') as f:
            self.coaches_data = json.load(f)
        logger.info(f"Loaded {len(self.coaches_data):,} coaches")
    
    def build_network_graph(self, relationship_types: List[str] = None, 
                           years_filter: Tuple[int, int] = None) -> nx.DiGraph:
        """
        Build NetworkX directed graph from coaching relationships
        
        Args:
            relationship_types: List of relationship types to include
            years_filter: Tuple of (start_year, end_year) to filter by (defaults to 1970-present)
            
        Returns:
            NetworkX directed graph
        """
        logger.info("Building NetworkX graph...")
        
        # Filter relationships
        df = self.relationships_df.copy()
        
        if relationship_types:
            df = df[df['relationship_type'].isin(relationship_types)]
        
        # Default to filtering out pre-1970 coaches    
        if years_filter:
            start_year, end_year = years_filter
        else:
            start_year = 1970  # Default minimum year
            end_year = 2025    # Future-proof max year
            logger.info(f"Using default year filter: {start_year}-{end_year}")
        
        df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]
        
        logger.info(f"Using {len(df):,} filtered relationships")
        
        # Create directed graph
        self.G = nx.DiGraph()
        
        # Build edges and nodes from relationships
        for _, row in df.iterrows():
            # Add nodes with attributes
            if not self.G.has_node(row['parent_name']):
                self.G.add_node(row['parent_name'], 
                              id=row['parent_id'],
                              role=row['parent_role'],
                              type='mentor')
                              
            if not self.G.has_node(row['child_name']):
                self.G.add_node(row['child_name'],
                              id=row['child_id'], 
                              role=row['child_role'],
                              type='protege')
            
            # Add edge (mentor -> protege)
            self.G.add_edge(row['parent_name'], row['child_name'],
                          year=row['year'],
                          team=row['team'],
                          relationship_type=row['relationship_type'])
        
        # Add coach career data to nodes
        for node in self.G.nodes():
            coach_id = self.G.nodes[node].get('id', node.lower().replace(' ', '_'))
            if coach_id in self.coaches_data:
                coach_data = self.coaches_data[coach_id]
                if coach_data.get('career'):
                    years = list(coach_data['career'].keys())
                    if years:
                        self.G.nodes[node]['career_start'] = min(int(y) for y in years)
                        self.G.nodes[node]['career_end'] = max(int(y) for y in years)
        
        logger.info(f"Created graph with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")
        
        # Calculate graph metrics
        self._calculate_metrics()
        
        return self.G
    
    def _calculate_metrics(self):
        """Calculate various graph metrics for each node"""
        logger.info("Calculating graph metrics...")
        
        # Degree centrality (connections)
        degree_centrality = nx.degree_centrality(self.G)
        for node, centrality in degree_centrality.items():
            self.G.nodes[node]['degree_centrality'] = centrality
        
        # Betweenness centrality (bridge importance)
        betweenness = nx.betweenness_centrality(self.G)
        for node, centrality in betweenness.items():
            self.G.nodes[node]['betweenness'] = centrality
        
        # PageRank (influence)
        pagerank = nx.pagerank(self.G)
        for node, rank in pagerank.items():
            self.G.nodes[node]['pagerank'] = rank
        
        # Coaching tree metrics
        for node in self.G.nodes():
            # Number of direct mentors
            self.G.nodes[node]['n_mentors'] = self.G.in_degree(node)
            # Number of direct proteges
            self.G.nodes[node]['n_proteges'] = self.G.out_degree(node)
            # Total descendants in tree
            descendants = nx.descendants(self.G, node)
            self.G.nodes[node]['n_descendants'] = len(descendants)
    
    def create_layout(self, layout_type: str = 'spring') -> Dict[str, Tuple[float, float]]:
        """
        Create layout using NetworkX algorithms
        
        Args:
            layout_type: Type of layout ('spring', 'hierarchical', 'kamada', 'circular')
            
        Returns:
            Dictionary mapping node names to (x, y) positions
        """
        logger.info(f"Creating {layout_type} layout...")
        
        if layout_type == 'hierarchical':
            # Try to use graphviz for true hierarchical layout
            try:
                import pygraphviz
                pos = nx.nx_agraph.graphviz_layout(self.G, prog='dot')
                logger.info("Used Graphviz 'dot' hierarchical layout")
            except ImportError:
                logger.warning("Pygraphviz not available, using multipartite layout")
                # Create hierarchy levels for multipartite layout
                levels = self._compute_hierarchy_levels()
                pos = nx.multipartite_layout(self.G, subset_key='level')
        
        elif layout_type == 'spring':
            # Force-directed layout - good for seeing clusters
            pos = nx.spring_layout(self.G, k=2, iterations=50, seed=42)
            logger.info("Used spring force-directed layout")
        
        elif layout_type == 'kamada':
            # Kamada-Kawai layout - minimizes stress
            pos = nx.kamada_kawai_layout(self.G)
            logger.info("Used Kamada-Kawai layout")
        
        elif layout_type == 'circular':
            # Circular layout - good for seeing connections
            pos = nx.circular_layout(self.G)
            logger.info("Used circular layout")
        
        elif layout_type == 'spectral':
            # Spectral layout - uses eigenvectors
            pos = nx.spectral_layout(self.G)
            logger.info("Used spectral layout")
        
        else:
            # Default to spring
            pos = nx.spring_layout(self.G, k=2, iterations=50, seed=42)
            logger.info("Used default spring layout")
        
        return pos
    
    def _compute_hierarchy_levels(self) -> Dict[str, int]:
        """Compute hierarchy levels for nodes using BFS from roots"""
        levels = {}
        
        # Find root nodes (no incoming edges)
        roots = [n for n in self.G.nodes() if self.G.in_degree(n) == 0]
        
        # BFS to assign levels
        from collections import deque
        queue = deque([(root, 0) for root in roots])
        
        while queue:
            node, level = queue.popleft()
            if node not in levels:
                levels[node] = level
                # Set as node attribute for multipartite layout
                self.G.nodes[node]['level'] = level
                # Add successors
                for successor in self.G.successors(node):
                    queue.append((successor, level + 1))
        
        return levels
    
    def create_interactive_plot(self, pos: Dict[str, Tuple[float, float]], 
                               color_by: str = 'influence',
                               title: str = "NFL Coaching Tree") -> go.Figure:
        """
        Create interactive Plotly visualization
        
        Args:
            pos: Node positions from NetworkX layout
            color_by: Metric to color nodes by ('influence', 'pagerank', 'betweenness', 'descendants')
            title: Plot title
            
        Returns:
            Plotly figure
        """
        logger.info(f"Creating interactive plot (coloring by {color_by})...")
        
        fig = go.Figure()
        
        # Draw edges
        edge_traces = []
        for edge in self.G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=0.5, color='rgba(125, 125, 125, 0.5)'),
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
        
        for node in self.G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            
            # Get node attributes
            node_attrs = self.G.nodes[node]
            
            # Create hover text
            hover_info = f"<b>{node}</b><br>"
            hover_info += f"Role: {node_attrs.get('role', 'Unknown')}<br>"
            hover_info += f"Mentors: {node_attrs.get('n_mentors', 0)}<br>"
            hover_info += f"Direct Protégés: {node_attrs.get('n_proteges', 0)}<br>"
            hover_info += f"Total Descendants: {node_attrs.get('n_descendants', 0)}<br>"
            
            if 'career_start' in node_attrs:
                hover_info += f"Career: {node_attrs['career_start']}-{node_attrs.get('career_end', 'present')}<br>"
            
            hover_info += f"<br><b>Metrics:</b><br>"
            hover_info += f"PageRank: {node_attrs.get('pagerank', 0):.4f}<br>"
            hover_info += f"Betweenness: {node_attrs.get('betweenness', 0):.4f}<br>"
            hover_info += f"Degree Centrality: {node_attrs.get('degree_centrality', 0):.4f}"
            
            hover_text.append(hover_info)
            
            # Determine node color based on selected metric
            if color_by == 'influence':
                color_value = node_attrs.get('n_proteges', 0)
            elif color_by == 'pagerank':
                color_value = node_attrs.get('pagerank', 0) * 100
            elif color_by == 'betweenness':
                color_value = node_attrs.get('betweenness', 0) * 100
            elif color_by == 'descendants':
                color_value = node_attrs.get('n_descendants', 0)
            else:
                color_value = node_attrs.get('n_proteges', 0)
            
            node_color.append(color_value)
            
            # Size by total connections
            total_connections = node_attrs.get('n_mentors', 0) + node_attrs.get('n_proteges', 0)
            node_size.append(max(5, min(20, 5 + total_connections)))
        
        # Add node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition='top center',
            textfont=dict(size=8),
            hovertext=hover_text,
            hoverinfo='text',
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale='YlOrRd',
                showscale=True,
                colorbar=dict(
                    title=color_by.capitalize(),
                    thickness=15,
                    len=0.7,
                    x=1.02
                ),
                line=dict(width=1, color='black')
            ),
            showlegend=False
        )
        
        fig.add_trace(node_trace)
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=18)
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text=f"Node size = total connections | Color = {color_by}",
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
            width=1200,
            height=800
        )
        
        return fig
    
    def analyze_coaching_schools(self) -> Dict:
        """
        Identify coaching 'schools' or communities using network analysis
        
        Returns:
            Dictionary of community analysis results
        """
        logger.info("Analyzing coaching schools/communities...")
        
        # Find communities using Louvain method
        communities = nx.community.greedy_modularity_communities(self.G.to_undirected())
        
        # Analyze each community
        community_data = {}
        for i, community in enumerate(communities):
            if len(community) < 3:  # Skip tiny communities
                continue
                
            # Find most central coach in community
            subgraph = self.G.subgraph(community)
            pagerank = nx.pagerank(subgraph)
            central_coach = max(pagerank, key=pagerank.get)
            
            community_data[f"School_{i+1}"] = {
                'size': len(community),
                'central_figure': central_coach,
                'members': sorted(list(community)),
                'top_5': sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]
            }
        
        logger.info(f"Found {len(community_data)} coaching schools")
        return community_data
    
    def find_coaching_lineages(self, max_display: int = 10) -> List[Dict]:
        """
        Find the longest coaching lineages (chains of mentorship)
        
        Args:
            max_display: Maximum number of lineages to return
            
        Returns:
            List of lineage dictionaries
        """
        logger.info("Finding longest coaching lineages...")
        
        # Find all simple paths between roots and leaves
        roots = [n for n in self.G.nodes() if self.G.in_degree(n) == 0]
        leaves = [n for n in self.G.nodes() if self.G.out_degree(n) == 0]
        
        longest_paths = []
        
        for root in roots:
            for leaf in leaves:
                try:
                    # Find all simple paths
                    paths = list(nx.all_simple_paths(self.G, root, leaf, cutoff=10))
                    for path in paths:
                        longest_paths.append({
                            'length': len(path),
                            'path': path,
                            'start': root,
                            'end': leaf
                        })
                except nx.NetworkXNoPath:
                    continue
        
        # Sort by length and return top lineages
        longest_paths.sort(key=lambda x: x['length'], reverse=True)
        
        logger.info(f"Found {len(longest_paths)} total lineages")
        return longest_paths[:max_display]


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Visualize NFL coaching tree with NetworkX')
    parser.add_argument('--output', type=str, default='coaching_tree_nx.html',
                       help='Output HTML file name')
    parser.add_argument('--layout', type=str, default='spring',
                       choices=['spring', 'hierarchical', 'kamada', 'circular', 'spectral'],
                       help='Layout algorithm to use')
    parser.add_argument('--color_by', type=str, default='influence',
                       choices=['influence', 'pagerank', 'betweenness', 'descendants'],
                       help='Metric to color nodes by')
    parser.add_argument('--filter_years', type=str,
                       help='Year range filter (e.g., "2010-2024")')
    parser.add_argument('--relationship_types', type=str, nargs='+',
                       help='Relationship types to include',
                       choices=['position_to_coordinator', 'position_to_hc', 'coordinator_to_hc'])
    parser.add_argument('--analyze', action='store_true',
                       help='Perform additional network analysis')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("NFL COACHING TREE VISUALIZER (NetworkX)")
    print("=" * 80)
    
    try:
        # Initialize visualizer
        visualizer = NetworkXCoachingTreeVisualizer()
        
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
        
        # Build graph
        G = visualizer.build_network_graph(
            relationship_types=args.relationship_types,
            years_filter=years_filter
        )
        
        # Create layout
        pos = visualizer.create_layout(args.layout)
        
        # Create visualization
        print(f"Creating {args.layout} visualization colored by {args.color_by}...")
        title = f"NFL Coaching Tree ({args.layout.capitalize()} Layout)"
        if years_filter:
            title += f" [{years_filter[0]}-{years_filter[1]}]"
        
        fig = visualizer.create_interactive_plot(pos, args.color_by, title)
        
        # Save to HTML
        output_path = Path(args.output)
        pyo.plot(fig, filename=str(output_path), auto_open=True)
        
        print(f"Visualization saved to: {output_path}")
        print("Opening in browser...")
        
        # Perform additional analysis if requested
        if args.analyze:
            print("\n" + "=" * 80)
            print("NETWORK ANALYSIS")
            print("=" * 80)
            
            # Basic stats
            print(f"\nGraph Statistics:")
            print(f"  Nodes: {G.number_of_nodes()}")
            print(f"  Edges: {G.number_of_edges()}")
            print(f"  Density: {nx.density(G):.4f}")
            
            # Most influential coaches (by PageRank)
            pagerank = nx.pagerank(G)
            top_coaches = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
            print(f"\nTop 10 Most Influential Coaches (PageRank):")
            for coach, rank in top_coaches:
                print(f"  {coach:30} {rank:.4f}")
            
            # Coaching schools
            schools = visualizer.analyze_coaching_schools()
            print(f"\nCoaching Schools/Communities Found: {len(schools)}")
            for school_name, data in list(schools.items())[:5]:
                print(f"\n  {school_name}:")
                print(f"    Size: {data['size']} coaches")
                print(f"    Central Figure: {data['central_figure']}")
                print(f"    Top Members: {', '.join([c[0] for c in data['top_5'][:3]])}")
            
            # Longest lineages
            lineages = visualizer.find_coaching_lineages(5)
            print(f"\nLongest Coaching Lineages:")
            for i, lineage in enumerate(lineages, 1):
                print(f"\n  Lineage {i} ({lineage['length']} generations):")
                print(f"    {' → '.join(lineage['path'][:5])}")
                if lineage['length'] > 5:
                    print(f"    ... → {lineage['path'][-1]}")
        
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        raise


if __name__ == "__main__":
    main()