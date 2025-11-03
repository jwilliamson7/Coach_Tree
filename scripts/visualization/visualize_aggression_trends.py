#!/usr/bin/env python3
"""
Visualize Aggression Gene Trends Over Time

This script creates a line chart showing how aggregate coaching aggression
and its four subcomponents have evolved over time across the NFL. Aggregates
all coaches per year to show league-wide trends.

Four aggression components:
1. 4th Down Aggression: Going for it on 4th down vs predicted
2. Pass-Heavy Aggression: Passing vs running relative to predictions
3. Deep Pass Aggression: Targeting beyond the sticks vs predicted
4. Two-Point Aggression: Attempting 2-point conversions vs predicted

Usage:
    python visualize_aggression_trends.py [--output_dir outputs/visualizations/trends]
    python visualize_aggression_trends.py --use_matplotlib
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AggressionTrendVisualizer:
    """Visualize aggression gene trends over time"""

    def __init__(self, gene_dir: str = "data/processed/coaching_genes",
                 output_dir: str = "outputs/visualizations/trends",
                 use_matplotlib: bool = False):
        self.gene_dir = Path(gene_dir)
        self.output_dir = Path(output_dir)
        self.use_matplotlib = use_matplotlib
        self.aggression_data = None
        self.yearly_trends = None

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> None:
        """Load aggression gene data"""
        logger.info("Loading aggression gene data...")

        aggression_file = self.gene_dir / "aggression_gene_by_year.csv"
        if not aggression_file.exists():
            raise FileNotFoundError(
                f"Aggression gene file not found: {aggression_file}\n"
                "Please run: python scripts/analysis/calculate_aggression_gene.py"
            )

        self.aggression_data = pd.read_csv(aggression_file)
        logger.info(f"Loaded {len(self.aggression_data):,} coach-year records")

        required_cols = [
            'season', 'composite_aggression', 'fourth_down_aggression',
            'pass_heavy_aggression', 'deep_pass_aggression', 'two_point_aggression'
        ]
        missing_cols = [col for col in required_cols if col not in self.aggression_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def calculate_yearly_trends(self) -> pd.DataFrame:
        """Calculate aggregate aggression metrics by year"""
        logger.info("Calculating yearly trends...")

        yearly_agg = self.aggression_data.groupby('season').agg({
            'composite_aggression': ['mean', 'std', 'count'],
            'fourth_down_aggression': 'mean',
            'pass_heavy_aggression': 'mean',
            'deep_pass_aggression': 'mean',
            'two_point_aggression': 'mean'
        }).reset_index()

        yearly_agg.columns = [
            'season', 'composite_aggression_mean', 'composite_aggression_std',
            'coach_count', 'fourth_down_aggression', 'pass_heavy_aggression',
            'deep_pass_aggression', 'two_point_aggression'
        ]

        self.yearly_trends = yearly_agg
        logger.info(f"Calculated trends for {len(yearly_agg)} seasons ({yearly_agg['season'].min()}-{yearly_agg['season'].max()})")

        return yearly_agg

    def plot_with_plotly(self) -> None:
        """Create visualization using Plotly"""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        logger.info("Creating Plotly visualization...")

        df = self.yearly_trends

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Composite Aggression Gene Over Time',
                          'Aggression Components Over Time'),
            vertical_spacing=0.12,
            row_heights=[0.4, 0.6]
        )

        fig.add_trace(
            go.Scatter(
                x=df['season'],
                y=df['composite_aggression_mean'] + df['composite_aggression_std'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=df['season'],
                y=df['composite_aggression_mean'] - df['composite_aggression_std'],
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(68, 68, 68, 0.2)',
                fill='tonexty',
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=df['season'],
                y=df['composite_aggression_mean'],
                mode='lines+markers',
                name='Composite Aggression',
                line=dict(color='#2E86AB', width=3),
                marker=dict(size=6),
                hovertemplate='<b>%{x}</b><br>Aggression: %{y:.4f}<extra></extra>'
            ),
            row=1, col=1
        )

        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)

        components = [
            ('fourth_down_aggression', '4th Down', '#A23B72'),
            ('pass_heavy_aggression', 'Pass-Heavy', '#F18F01'),
            ('deep_pass_aggression', 'Deep Pass', '#C73E1D'),
            ('two_point_aggression', '2-Point Conv.', '#6A994E')
        ]

        for col_name, label, color in components:
            fig.add_trace(
                go.Scatter(
                    x=df['season'],
                    y=df[col_name],
                    mode='lines+markers',
                    name=label,
                    line=dict(color=color, width=2),
                    marker=dict(size=5),
                    hovertemplate=f'<b>%{{x}}</b><br>{label}: %{{y:.4f}}<extra></extra>'
                ),
                row=2, col=1
            )

        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)

        fig.update_xaxes(title_text="Season", row=2, col=1)
        fig.update_yaxes(title_text="Aggression Score", row=1, col=1)
        fig.update_yaxes(title_text="Component Score", row=2, col=1)

        fig.update_layout(
            title={
                'text': 'NFL Coaching Aggression Genes: Trends Over Time',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#1a1a1a'}
            },
            hovermode='x unified',
            template='plotly_white',
            height=900,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5
            ),
            font=dict(size=12)
        )

        png_path = self.output_dir / "aggression_trends.png"
        try:
            fig.write_image(str(png_path), width=1200, height=900, scale=2)
            logger.info(f"Saved PNG image: {png_path}")
        except Exception as e:
            logger.warning(f"Could not save PNG (kaleido may not be installed): {e}")
            logger.info("To enable PNG export, install: pip install -U kaleido")

    def plot_with_matplotlib(self) -> None:
        """Create visualization using Matplotlib"""
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MultipleLocator, FuncFormatter

        logger.info("Creating Matplotlib visualization...")

        plt.rcParams['font.family'] = 'Cambria'

        df = self.yearly_trends

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle('NFL Coaching Aggression Genes: Trends Over Time',
                    fontsize=16, fontweight='bold', y=0.99, family='Cambria')

        ax1.plot(df['season'], df['composite_aggression_mean'],
                color='#2E86AB', linewidth=2.5, marker='o', markersize=5,
                label='Composite Aggression', zorder=3)

        ax1.fill_between(
            df['season'],
            df['composite_aggression_mean'] - df['composite_aggression_std'],
            df['composite_aggression_mean'] + df['composite_aggression_std'],
            alpha=0.2, color='#2E86AB', label='±1 Std Dev'
        )

        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=1)
        ax1.set_ylabel('POE (Percent over Expected)', fontsize=12, fontweight='bold')
        ax1.set_title('Composite Aggression Gene Over Time', fontsize=13, pad=15)
        ax1.legend(loc='upper left', framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle=':')
        ax1.set_xlim(df['season'].min() - 0.5, df['season'].max() + 0.5)

        components = [
            ('fourth_down_aggression', '4th Down', '#A23B72', 'o'),
            ('pass_heavy_aggression', 'Pass-Heavy', '#F18F01', 's'),
            ('deep_pass_aggression', 'Deep Pass', '#C73E1D', '^'),
            ('two_point_aggression', '2-Point Conv.', '#6A994E', 'D')
        ]

        for col_name, label, color, marker in components:
            ax2.plot(df['season'], df[col_name],
                    color=color, linewidth=2, marker=marker, markersize=5,
                    label=label, alpha=0.85)

        ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=1)
        ax2.set_xlabel('Season', fontsize=12, fontweight='bold')
        ax2.set_ylabel('POE (Percent over Expected)', fontsize=12, fontweight='bold')
        ax2.set_title('Aggression Components Over Time', fontsize=13, pad=10)
        ax2.legend(loc='best', framealpha=0.9, ncol=2)
        ax2.grid(True, alpha=0.3, linestyle=':')
        ax2.set_xlim(df['season'].min() - 0.5, df['season'].max() + 0.5)

        for ax in [ax1, ax2]:
            ax.xaxis.set_major_locator(MultipleLocator(2))
            ax.xaxis.set_minor_locator(MultipleLocator(1))

        def percent_formatter(x, pos):
            """Format y-axis as +/-XX.X%"""
            return f"{x*100:+.1f}%"

        ax1.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
        ax2.yaxis.set_major_formatter(FuncFormatter(percent_formatter))

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        png_path = self.output_dir / "aggression_trends.png"
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved PNG image: {png_path}")

        plt.close()

    def create_summary_stats(self) -> None:
        """Create a summary statistics file"""
        logger.info("Creating summary statistics...")

        df = self.yearly_trends

        summary = {
            'overall_stats': {
                'seasons': int(len(df)),
                'year_range': f"{int(df['season'].min())}-{int(df['season'].max())}",
                'avg_coaches_per_year': float(df['coach_count'].mean()),
                'total_coach_years': int(df['coach_count'].sum())
            },
            'composite_aggression': {
                'overall_mean': float(df['composite_aggression_mean'].mean()),
                'overall_std': float(df['composite_aggression_mean'].std()),
                'min_year': int(df.loc[df['composite_aggression_mean'].idxmin(), 'season']),
                'min_value': float(df['composite_aggression_mean'].min()),
                'max_year': int(df.loc[df['composite_aggression_mean'].idxmax(), 'season']),
                'max_value': float(df['composite_aggression_mean'].max()),
                'trend': 'increasing' if df['composite_aggression_mean'].iloc[-1] > df['composite_aggression_mean'].iloc[0] else 'decreasing'
            },
            'components': {}
        }

        for component in ['fourth_down_aggression', 'pass_heavy_aggression',
                         'deep_pass_aggression', 'two_point_aggression']:
            summary['components'][component] = {
                'mean': float(df[component].mean()),
                'std': float(df[component].std()),
                'min': float(df[component].min()),
                'max': float(df[component].max())
            }

        import json
        summary_path = self.output_dir / "aggression_trends_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary statistics: {summary_path}")

        csv_path = self.output_dir / "aggression_trends_yearly.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved yearly trends data: {csv_path}")

    def run(self) -> None:
        """Execute the full visualization pipeline"""
        logger.info("Starting aggression trends visualization...")

        self.load_data()
        self.calculate_yearly_trends()

        if self.use_matplotlib:
            self.plot_with_matplotlib()
        else:
            self.plot_with_plotly()

        self.create_summary_stats()

        logger.info("Visualization complete!")
        logger.info(f"Output directory: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize aggression gene trends over time'
    )
    parser.add_argument(
        '--gene_dir',
        type=str,
        default='data/processed/coaching_genes',
        help='Directory containing aggression gene data'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/visualizations/trends',
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--use_matplotlib',
        action='store_true',
        help='Use Matplotlib instead of Plotly'
    )

    args = parser.parse_args()

    visualizer = AggressionTrendVisualizer(
        gene_dir=args.gene_dir,
        output_dir=args.output_dir,
        use_matplotlib=args.use_matplotlib
    )

    visualizer.run()


if __name__ == "__main__":
    main()
