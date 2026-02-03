"""
Shared plot configuration for consistent styling across all visualizations.
Uses fonts compatible with the LaTeX lmodern template.
"""

import matplotlib.pyplot as plt


def configure_plots():
    """Configure matplotlib to use fonts matching the LaTeX lmodern template."""
    # Use serif fonts to match Latin Modern in LaTeX
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = [
        'Latin Modern Roman',
        'Computer Modern Roman',
        'DejaVu Serif',
        'Times New Roman',
        'serif'
    ]
    plt.rcParams['font.size'] = 13

    # Use LaTeX-style math text
    plt.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern for math

    # Additional styling for publication quality
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
