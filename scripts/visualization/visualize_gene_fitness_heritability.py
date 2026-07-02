#!/usr/bin/env python3
"""
Heritability x Selection Map for Coaching Genes

The keystone figure of the evolutionary reading. Each coaching gene is placed in
a two-dimensional space:
    x = transmission fidelity (heritability proxy): coordinator -> head coach
        era-adjusted correlation (does the trait pass down the apprenticeship?)
    y = selection (fitness gradient): gene -> WAR era-adjusted inverse-variance
        weighted correlation (does the trait win games?)

The four quadrants formalize the paper's dissociation thesis:
    top-right    (heritable, adaptive)   "Adaptive and heritable"
    bottom-right (heritable, neutral)     "Heritable but neutral"
    top-left     (not heritable, adaptive)"Adaptive but personal"
    bottom-left  (neither)                "Neither (drift)"

Error bars are the respective confidence intervals; a bar that crosses a zero
line means the gene is statistically indistinguishable from neutral on that axis.

All numbers are read from committed result JSONs (no new statistics here).

Usage:
    python scripts/visualization/visualize_gene_fitness_heritability.py
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt

from plot_config import configure_plots

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Each gene: how to find its heritability (coord->HC) and selection (gene->WAR).
#   inherit_key -> key in gene_inheritance_summary.json["statistics"]
#   war_key/war_label -> key path in gene_war_correlation_results.json
GENES = [
    {
        'name': 'Offensive\naggression',
        'short': 'aggression',
        'inherit_key': 'aggression',
        'war_key': 'aggression',
        'war_label': 'Composite Aggression',
        'color': '#C1272D',
    },
    {
        'name': 'Defensive\nscheme',
        'short': 'defensive',
        'inherit_key': 'defensive_scheme',
        'war_key': 'defensive_scheme',
        'war_label': 'Defensive Scheme',
        'color': '#2E86AB',
    },
    {
        'name': 'Shotgun',
        'short': 'shotgun',
        'inherit_key': 'shotgun',
        'war_key': 'shotgun',
        'war_label': 'Shotgun Formation',
        'color': '#6A4C93',
    },
    {
        'name': 'Tempo',
        'short': 'tempo',
        'inherit_key': 'tempo',
        'war_key': 'tempo',
        'war_label': 'Composite Tempo',
        'color': '#6A994E',
    },
]

# Manual label offsets (data units) so text does not collide with points/bars.
LABEL_OFFSETS = {
    'aggression': (0.045, 0.012),
    'defensive': (0.02, 0.022),
    'shotgun': (0.0, -0.040),
    'tempo': (0.02, -0.030),
}


def load_numbers():
    """Read the confirmatory heritability (C1 posterior h^2, HDI) and selection
    (component/composite era-adjusted IVW S, bootstrap CI) for each gene-level
    composite from the frozen-protocol result JSONs."""
    repo = Path(__file__).resolve().parents[2]
    h2_path = repo / "outputs/analysis/ehs_heritability_results.json"
    S_path = repo / "outputs/analysis/ehs_selection_results.json"

    with open(h2_path) as f:
        h2 = json.load(f)["composites"]
    with open(S_path) as f:
        Sj = json.load(f)["composites"]

    rows = []
    for g in GENES:
        c1 = h2[g['short']]['c1']
        s = Sj[g['short']]
        rows.append({
            **g,
            # heritability = C1 Bayesian parent-offspring h^2 (posterior median, HDI)
            'h2': c1['h2_median'], 'h2_lo': c1['hdi_low'], 'h2_hi': c1['hdi_high'],
            'h2_p': 1.0 - c1['p_positive'],
            # selection = era-adjusted IVW gene->WAR correlation (bootstrap CI)
            's': s['S'], 's_lo': s['ci_low'], 's_hi': s['ci_high'], 's_p': s['p_bootstrap'],
        })
        logger.info("%-11s h2=%.2f [%.2f,%.2f] | S=%.2f [%.2f,%.2f]",
                    g['short'], rows[-1]['h2'], rows[-1]['h2_lo'], rows[-1]['h2_hi'],
                    rows[-1]['s'], rows[-1]['s_lo'], rows[-1]['s_hi'])
    return rows


def create_map(rows):
    configure_plots()
    fig, ax = plt.subplots(figsize=(11, 9))

    # Data-driven limits with padding around the CI extents.
    xs = [r['h2_lo'] for r in rows] + [r['h2_hi'] for r in rows] + [0]
    ys = [r['s_lo'] for r in rows] + [r['s_hi'] for r in rows] + [0]
    xpad = 0.12 * (max(xs) - min(xs)); ypad = 0.18 * (max(ys) - min(ys))
    xlim = (min(xs) - xpad, max(xs) + xpad)
    ylim = (min(ys) - ypad, max(ys) + ypad)
    zx = (0 - xlim[0]) / (xlim[1] - xlim[0])   # fraction of x-axis where h2 = 0

    # Quadrant shading (very light).
    ax.axhspan(0, ylim[1], xmin=zx, xmax=1.0, color='#2E86AB', alpha=0.05, zorder=0)
    ax.axhspan(ylim[0], 0, xmin=zx, xmax=1.0, color='#6A4C93', alpha=0.05, zorder=0)
    ax.axhspan(0, ylim[1], xmin=0.0, xmax=zx, color='#C1272D', alpha=0.05, zorder=0)
    ax.axhspan(ylim[0], 0, xmin=0.0, xmax=zx, color='gray', alpha=0.05, zorder=0)

    # Zero reference lines (neutral on each axis).
    ax.axhline(0, color='gray', lw=1.2, ls='-', alpha=0.6, zorder=1)
    ax.axvline(0, color='gray', lw=1.2, ls='-', alpha=0.6, zorder=1)

    quad_labels = [
        (xlim[1] - 0.01, ylim[1] - 0.004, 'ADAPTIVE AND HERITABLE', 'right', 'top', '#1B5E7A'),
        (xlim[1] - 0.01, ylim[0] + 0.006, 'HERITABLE BUT NEUTRAL', 'right', 'bottom', '#4A3568'),
        (xlim[0] + 0.01, ylim[1] - 0.004, 'ADAPTIVE BUT PERSONAL', 'left', 'top', '#8A1B20'),
        (xlim[0] + 0.01, ylim[0] + 0.006, 'NEITHER (DRIFT)', 'left', 'bottom', '#666666'),
    ]
    for x, y, txt, ha, va, c in quad_labels:
        ax.text(x, y, txt, ha=ha, va=va, fontsize=11.5, fontweight='bold',
                color=c, alpha=0.55, style='italic', zorder=1)

    # Plot each gene with asymmetric CI error bars.
    for r in rows:
        xerr = [[r['h2'] - r['h2_lo']], [r['h2_hi'] - r['h2']]]
        yerr = [[r['s'] - r['s_lo']], [r['s_hi'] - r['s']]]
        ax.errorbar(
            r['h2'], r['s'], xerr=xerr, yerr=yerr,
            fmt='o', ms=15, color=r['color'], ecolor=r['color'],
            elinewidth=1.8, capsize=4, capthick=1.8, alpha=0.9,
            markeredgecolor='black', markeredgewidth=1.0, zorder=4
        )
        dx, dy = LABEL_OFFSETS[r['short']]
        ax.annotate(
            r['name'], (r['h2'], r['s']), (r['h2'] + dx, r['s'] + dy),
            fontsize=13, fontweight='bold', color=r['color'],
            ha='left' if dx >= 0 else 'right',
            va='center', zorder=5
        )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Heritability  $h^{2}$  (parent-offspring, posterior median)',
                  fontsize=14, fontweight='bold')
    ax.set_ylabel('Selection  $S$  (gene $\\rightarrow$ WAR $r$, era-adjusted IVW)',
                  fontsize=14, fontweight='bold')
    ax.set_title('Heritability $\\times$ Selection Map of Coaching Genes',
                 fontsize=17, fontweight='bold', pad=34)

    ax.grid(True, alpha=0.25, ls=':', zorder=0)
    ax.tick_params(labelsize=12)

    plt.tight_layout()

    out_dir = Path(__file__).resolve().parents[2] / "outputs/visualizations/performance"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "gene_fitness_heritability_map.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info("Saved figure: %s", out_file)
    plt.close()


def main():
    logger.info("Building heritability x selection map...")
    rows = load_numbers()
    create_map(rows)
    logger.info("Done.")


if __name__ == "__main__":
    main()
