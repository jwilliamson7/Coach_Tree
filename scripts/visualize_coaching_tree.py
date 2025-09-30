#!/usr/bin/env python3
"""
Visualize NFL Coaching Tree

This script creates an interactive visualization of the NFL coaching tree using
Plotly and NetworkX. The tree shows mentor-protégé relationships between coaches
with interactive features for exploring the network.

Usage:
    python visualize_coaching_tree.py [--output tree.html] [--filter_years 2010-2024]
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CoachingTreeVisualizer:
    """Create interactive visualizations of the NFL coaching tree"""
    
    def __init__(self, data_dir: str = "data/processed/coaching_tree"):
        self.data_dir = Path(data_dir)
        self.relationships_df = None
        self.coaches_data = None
        
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
            years_filter: Tuple of (start_year, end_year) to filter by
            
        Returns:
            NetworkX directed graph
        """
        logger.info("Building network graph...")
        
        # Filter relationships
        df = self.relationships_df.copy()
        
        if relationship_types:
            df = df[df['relationship_type'].isin(relationship_types)]
            
        if years_filter:
            start_year, end_year = years_filter
            df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]
        
        logger.info(f"Using {len(df):,} filtered relationships")
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add edges (mentor -> protégé relationships)
        for _, row in df.iterrows():
            # Add nodes with attributes
            if not G.has_node(row['parent_name']):
                G.add_node(row['parent_name'], 
                          id=row['parent_id'],
                          role=row['parent_role'])
                          
            if not G.has_node(row['child_name']):
                G.add_node(row['child_name'],
                          id=row['child_id'], 
                          role=row['child_role'])
            
            # Add edge with relationship info
            G.add_edge(row['parent_name'], row['child_name'],
                      year=row['year'],
                      team=row['team'],
                      relationship_type=row['relationship_type'])
        
        logger.info(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    def create_hierarchical_layout(self, G: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
        """
        Create hierarchical layout for the coaching tree
        
        Args:
            G: NetworkX directed graph
            
        Returns:
            Dictionary mapping node names to (x, y) positions
        """
        logger.info("Creating hierarchical layout...")
        
        # Try different layout algorithms
        try:
            # Try graphviz hierarchical layout (requires pygraphviz)
            try:
                import pygraphviz
                pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
                logger.info("Used Graphviz 'dot' layout")
                return pos
            except ImportError:
                logger.warning("Pygraphviz not available, falling back to spring layout")
        except:
            logger.warning("Graphviz layout failed, using spring layout")
        
        # Fallback to spring layout with hierarchical bias
        # Create artificial hierarchy based on graph structure
        levels = {}
        
        # Find root nodes (no predecessors)
        roots = [n for n in G.nodes() if G.in_degree(n) == 0]
        
        # BFS to assign levels
        from collections import deque
        queue = deque([(root, 0) for root in roots])
        
        while queue:
            node, level = queue.popleft()
            if node not in levels:
                levels[node] = level
                # Add successors to queue
                for successor in G.successors(node):
                    queue.append((successor, level + 1))
        
        # Create position constraints based on levels
        max_level = max(levels.values()) if levels else 0
        
        # Use spring layout with fixed y-coordinates based on hierarchy
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        
        # Adjust y-coordinates to create hierarchy
        for node in pos:
            level = levels.get(node, max_level // 2)
            pos[node] = (pos[node][0], max_level - level)
        
        logger.info("Used spring layout with hierarchical adjustment")
        return pos
    
    def create_interactive_plot(self, G: nx.DiGraph, pos: Dict[str, Tuple[float, float]], 
                               title: str = "NFL Coaching Tree") -> go.Figure:
        """
        Create interactive Plotly visualization
        
        Args:
            G: NetworkX directed graph
            pos: Node positions
            title: Plot title
            
        Returns:
            Plotly figure
        """
        logger.info("Creating interactive plot...")
        
        fig = go.Figure()
        
        # Add edges
        edge_x = []
        edge_y = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=1, color='rgba(125, 125, 125, 0.5)'),
            hoverinfo='none',
            showlegend=False,
            name='Relationships'
        ))
        
        # Add nodes
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_text = list(G.nodes())
        
        # Create hover text with coach information
        hover_text = []
        node_colors = []
        node_sizes = []
        
        for node in G.nodes():
            # Get coach data
            coach_id = G.nodes[node].get('id', node.lower().replace(' ', '_'))
            coach_data = self.coaches_data.get(coach_id, {})
            
            # Count coaching relationships
            in_degree = G.in_degree(node)  # Number of mentors
            out_degree = G.out_degree(node)  # Number of protégés
            
            # Create hover text
            hover_info = f"<b>{node}</b><br>"
            hover_info += f"Mentors: {in_degree}<br>"
            hover_info += f"Protégés: {out_degree}<br>"
            
            # Add career span if available
            if coach_data.get('career'):
                years = list(coach_data['career'].keys())
                if years:
                    start_year = min(int(y) for y in years)
                    end_year = max(int(y) for y in years)
                    hover_info += f"Career: {start_year}-{end_year}<br>"
            
            # Current role
            current_role = G.nodes[node].get('role', 'Unknown')
            hover_info += f"Role: {current_role}"
            
            hover_text.append(hover_info)
            
            # Color by influence (number of protégés)
            if out_degree == 0:
                node_colors.append('lightblue')  # No protégés
            elif out_degree <= 2:
                node_colors.append('orange')     # Few protégés
            elif out_degree <= 5:
                node_colors.append('red')        # Many protégés
            else:
                node_colors.append('darkred')    # Highly influential
            
            # Size by total connections
            total_connections = in_degree + out_degree
            node_sizes.append(max(8, min(20, 8 + total_connections * 2)))
        
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition='middle center',
            textfont=dict(size=8),
            hovertext=hover_text,
            hoverinfo='text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=1, color='black'),
                opacity=0.8
            ),
            showlegend=False,
            name='Coaches'
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=16)
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Node size = total connections, Color = influence (# of protégés)",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor="left", yanchor="bottom",
                    font=dict(size=10, color="gray")
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        return fig
    
    def create_visualization(self, relationship_types: List[str] = None,
                           years_filter: Tuple[int, int] = None,
                           title: str = "NFL Coaching Tree") -> go.Figure:
        """
        Create complete coaching tree visualization
        
        Args:
            relationship_types: Types of relationships to include
            years_filter: Year range to filter by
            title: Plot title
            
        Returns:
            Plotly figure
        """
        # Build graph
        G = self.build_network_graph(relationship_types, years_filter)
        
        # Create layout
        pos = self.create_hierarchical_layout(G)
        
        # Create plot
        fig = self.create_interactive_plot(G, pos, title)
        
        return fig


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Visualize NFL coaching tree')
    parser.add_argument('--output', type=str, default='coaching_tree.html',
                       help='Output HTML file name')
    parser.add_argument('--filter_years', type=str, 
                       help='Year range filter (e.g., "2010-2024")')
    parser.add_argument('--relationship_types', type=str, nargs='+',
                       help='Relationship types to include',
                       choices=['position_to_coordinator', 'position_to_hc', 'coordinator_to_hc'])
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("NFL COACHING TREE VISUALIZER")
    print("=" * 80)
    
    try:
        # Initialize visualizer
        visualizer = CoachingTreeVisualizer()
        
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
        
        # Create visualization
        print("Creating visualization...")
        title = "NFL Coaching Tree"
        if years_filter:
            title += f" ({years_filter[0]}-{years_filter[1]})"
        
        fig = visualizer.create_visualization(
            relationship_types=args.relationship_types,
            years_filter=years_filter,
            title=title
        )
        
        # Save to HTML
        output_path = Path(args.output)
        pyo.plot(fig, filename=str(output_path), auto_open=True)
        
        print(f"Visualization saved to: {output_path}")
        print("Opening in browser...")
        
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        raise


if __name__ == "__main__":
    main()