#!/usr/bin/env python3
"""
Visualize NFL Coaching Tree (Simple Version)

This script creates an interactive visualization of the NFL coaching tree using
Plotly without NetworkX dependency. Shows mentor-protégé relationships with
interactive features.

Usage:
    python visualize_coaching_tree_simple.py [--output tree.html]
"""

import argparse
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.offline as pyo
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Set
import logging
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleCoachingTreeVisualizer:
    """Create interactive visualizations of the NFL coaching tree without NetworkX"""
    
    def __init__(self, data_dir: str = "data/processed/coaching_tree"):
        self.data_dir = Path(data_dir)
        self.relationships_df = None
        self.coaches_data = None
        self.graph = None  # Our own graph representation
        
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
    
    def build_graph(self, relationship_types: List[str] = None, 
                   years_filter: Tuple[int, int] = None) -> Dict:
        """
        Build graph representation from coaching relationships
        
        Args:
            relationship_types: List of relationship types to include
            years_filter: Tuple of (start_year, end_year) to filter by
            
        Returns:
            Dictionary representing the graph
        """
        logger.info("Building graph...")
        
        # Filter relationships
        df = self.relationships_df.copy()
        
        if relationship_types:
            df = df[df['relationship_type'].isin(relationship_types)]
            
        if years_filter:
            start_year, end_year = years_filter
            df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]
        
        logger.info(f"Using {len(df):,} filtered relationships")
        
        # Build graph as adjacency lists
        graph = {
            'nodes': set(),
            'edges': [],
            'successors': defaultdict(set),  # parent -> children
            'predecessors': defaultdict(set),  # child -> parents
            'node_data': {}
        }
        
        # Process relationships
        for _, row in df.iterrows():
            parent = row['parent_name']
            child = row['child_name']
            
            # Add nodes
            graph['nodes'].add(parent)
            graph['nodes'].add(child)
            
            # Add edges
            graph['edges'].append((parent, child))
            graph['successors'][parent].add(child)
            graph['predecessors'][child].add(parent)
            
            # Store node data
            if parent not in graph['node_data']:
                graph['node_data'][parent] = {
                    'id': row['parent_id'],
                    'role': row['parent_role']
                }
            if child not in graph['node_data']:
                graph['node_data'][child] = {
                    'id': row['child_id'],
                    'role': row['child_role']
                }
        
        logger.info(f"Created graph with {len(graph['nodes'])} nodes and {len(graph['edges'])} edges")
        self.graph = graph
        return graph
    
    def create_hierarchical_layout(self) -> Dict[str, Tuple[float, float]]:
        """
        Create hierarchical layout using our own algorithm
        
        Returns:
            Dictionary mapping node names to (x, y) positions
        """
        logger.info("Creating hierarchical layout...")
        
        if not self.graph:
            raise ValueError("Graph not built. Call build_graph() first.")
        
        # Find root nodes (coaches with no mentors)
        roots = []
        for node in self.graph['nodes']:
            if len(self.graph['predecessors'][node]) == 0:
                roots.append(node)
        
        logger.info(f"Found {len(roots)} root coaches")
        
        # Assign levels using BFS
        levels = {}
        queue = deque([(root, 0) for root in roots])
        
        while queue:
            node, level = queue.popleft()
            if node not in levels:
                levels[node] = level
                # Add children to queue
                for child in self.graph['successors'][node]:
                    queue.append((child, level + 1))
        
        # Group nodes by level
        level_groups = defaultdict(list)
        for node, level in levels.items():
            level_groups[level].append(node)
        
        max_level = max(levels.values()) if levels else 0
        
        # Position nodes
        positions = {}
        
        for level in range(max_level + 1):
            nodes_at_level = level_groups[level]
            n_nodes = len(nodes_at_level)
            
            if n_nodes == 0:
                continue
            
            # Spread nodes horizontally at this level
            if n_nodes == 1:
                x_positions = [0]
            else:
                x_positions = np.linspace(-n_nodes/2, n_nodes/2, n_nodes)
            
            for i, node in enumerate(nodes_at_level):
                positions[node] = (x_positions[i], max_level - level)
        
        logger.info(f"Positioned {len(positions)} nodes across {max_level + 1} levels")
        return positions
    
    def create_interactive_plot(self, positions: Dict[str, Tuple[float, float]], 
                               title: str = "NFL Coaching Tree") -> go.Figure:
        """
        Create interactive Plotly visualization
        
        Args:
            positions: Node positions
            title: Plot title
            
        Returns:
            Plotly figure
        """
        logger.info("Creating interactive plot...")
        
        fig = go.Figure()
        
        # Add edges
        edge_x = []
        edge_y = []
        
        for parent, child in self.graph['edges']:
            if parent in positions and child in positions:
                x0, y0 = positions[parent]
                x1, y1 = positions[child]
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
        node_x = []
        node_y = []
        node_text = []
        hover_text = []
        node_colors = []
        node_sizes = []
        
        for node in self.graph['nodes']:
            if node not in positions:
                continue
                
            x, y = positions[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            
            # Get coach data
            coach_id = self.graph['node_data'][node].get('id', node.lower().replace(' ', '_'))
            coach_data = self.coaches_data.get(coach_id, {})
            
            # Count relationships
            n_mentors = len(self.graph['predecessors'][node])
            n_proteges = len(self.graph['successors'][node])
            
            # Create hover text
            hover_info = f"<b>{node}</b><br>"
            hover_info += f"Mentors: {n_mentors}<br>"
            hover_info += f"Protégés: {n_proteges}<br>"
            
            # Add career span if available
            if coach_data.get('career'):
                years = list(coach_data['career'].keys())
                if years:
                    start_year = min(int(y) for y in years)
                    end_year = max(int(y) for y in years)
                    hover_info += f"Career: {start_year}-{end_year}<br>"
            
            # Current role
            current_role = self.graph['node_data'][node].get('role', 'Unknown')
            hover_info += f"Role: {current_role}"
            
            hover_text.append(hover_info)
            
            # Color by influence (number of protégés)
            if n_proteges == 0:
                node_colors.append('lightblue')  # No protégés
            elif n_proteges <= 2:
                node_colors.append('orange')     # Few protégés
            elif n_proteges <= 5:
                node_colors.append('red')        # Many protégés
            else:
                node_colors.append('darkred')    # Highly influential
            
            # Size by total connections
            total_connections = n_mentors + n_proteges
            node_sizes.append(max(8, min(25, 8 + total_connections)))
        
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
            margin=dict(b=40, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Node size = total connections, Color = influence (# of protégés)<br>" +
                         "Blue: No protégés, Orange: 1-2 protégés, Red: 3-5 protégés, Dark Red: 6+ protégés",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=-0.05,
                    xanchor="center", yanchor="top",
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
        self.build_graph(relationship_types, years_filter)
        
        # Create layout
        positions = self.create_hierarchical_layout()
        
        # Create plot
        fig = self.create_interactive_plot(positions, title)
        
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
        visualizer = SimpleCoachingTreeVisualizer()
        
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