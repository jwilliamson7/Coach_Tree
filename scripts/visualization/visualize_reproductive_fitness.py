#!/usr/bin/env python3
"""
Reproductive Fitness of NFL Head Coaches -- two-panel figure

Left:  distribution of reproductive fitness R (number of head-coaching offspring)
       across all NFL head coaches. A heavy right tail = a few supersires.
Right: the two fitnesses. Ecological fitness (career WAR) vs reproductive fitness
       (R), point size = number of head-coaching seasons (tenure). The positive
       trend is carried by tenure: long-tenured coaches both win more total and
       cycle more future head coaches through their staffs.

Reads outputs/analysis/reproductive_fitness_coaches.csv and
reproductive_fitness_results.json (produced by analyze_reproductive_fitness.py).

Usage:
    python scripts/visualization/visualize_reproductive_fitness.py
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from plot_config import configure_plots

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
ANALYSIS = REPO / "outputs/analysis"

# Supersires to label on the scatter (kept short so the panel stays readable).
LABEL_COACHES = {
    "Bill Belichick", "Bill Parcells", "Andy Reid", "Marty Schottenheimer",
    "Don Shula", "Mike Holmgren", "Bill Walsh", "Sean McVay", "Mike Shanahan",
}


def main():
    df = pd.read_csv(ANALYSIS / "reproductive_fitness_coaches.csv")
    with open(ANALYSIS / "reproductive_fitness_results.json") as f:
        res = json.load(f)
    d = res["distribution"]
    w = res["winning_begets_offspring"]

    configure_plots()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.6))

    # ---- Panel 1: distribution of R ---------------------------------------- #
    rmax = int(df["repro_fitness"].max())
    counts = df["repro_fitness"].value_counts().reindex(range(rmax + 1), fill_value=0)
    bars = ax1.bar(counts.index, counts.values, color='#2E86AB',
                   edgecolor='black', linewidth=0.6, width=0.85, zorder=3)
    # Fade the bars from common (light) to rare (dark) to emphasize the tail.
    for i, b in enumerate(bars):
        b.set_alpha(0.45 + 0.55 * (i / max(1, rmax)))

    ax1.set_xlabel('Reproductive fitness $R$  (head-coaching offspring)',
                   fontsize=13, fontweight='bold')
    ax1.set_ylabel('Number of head coaches', fontsize=13, fontweight='bold')
    ax1.set_title('A few supersires dominate reproduction',
                  fontsize=15, fontweight='bold', pad=10)
    ax1.set_xticks(range(rmax + 1))
    ax1.grid(True, axis='y', alpha=0.25, ls=':', zorder=0)

    stats_txt = (
        f"n = {d['n_mentors']} head coaches\n"
        f"{d['pct_with_offspring']:.0f}% leave $\\geq$1 offspring\n"
        f"mean $R$ = {d['mean']:.2f},  max = {d['max']}\n"
        f"Gini = {d['gini']:.2f}\n"
        f"top 10% produce {d['top_decile_share_of_offspring']*100:.0f}% of all offspring"
    )
    ax1.text(0.97, 0.95, stats_txt, transform=ax1.transAxes, fontsize=11.5,
             va='top', ha='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.92,
                       edgecolor='black'))

    # ---- Panel 2: two fitnesses -------------------------------------------- #
    sc = df.dropna(subset=["career_mean_war", "repro_fitness", "n_hc_seasons"]).copy()
    sizes = 18 + 14 * sc["n_hc_seasons"]
    # Light vertical jitter on integer R so overlapping points are visible.
    rng = np.random.default_rng(0)
    sc["r_jit"] = sc["repro_fitness"] + rng.uniform(-0.14, 0.14, len(sc))
    ax2.scatter(sc["career_mean_war"], sc["r_jit"], s=sizes,
                c='#6A4C93', alpha=0.42, edgecolors='black', linewidth=0.5,
                zorder=3)

    # Label supersires; place to the left for high-WAR points so text stays
    # inside the panel, to the right otherwise.
    xhi = sc["career_mean_war"].quantile(0.80)
    for r in sc.itertuples():
        if r.coach_name in LABEL_COACHES and r.repro_fitness >= 4:
            if r.career_mean_war >= xhi:
                dx, ha = -7, 'right'
            else:
                dx, ha = 7, 'left'
            ax2.annotate(r.coach_name.split()[-1],
                         (r.career_mean_war, r.repro_fitness),
                         (dx, 4), textcoords='offset points',
                         fontsize=9.5, fontweight='bold', color='#3A2A52', ha=ha)

    ax2.set_xlim(sc["career_mean_war"].min() - 0.02, sc["career_mean_war"].max() + 0.05)
    ax2.set_ylim(-0.6, rmax + 0.9)
    ax2.axvline(0, color='gray', lw=1, alpha=0.5, zorder=1)
    ax2.set_xlabel('Ecological fitness:  career mean WAR per season',
                   fontsize=13, fontweight='bold')
    ax2.set_ylabel('Reproductive fitness $R$', fontsize=13, fontweight='bold')
    ax2.set_title('Winning and reproducing are linked through tenure',
                  fontsize=15, fontweight='bold', pad=10)
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x*16:+.0f}"))
    ax2.grid(True, alpha=0.25, ls=':', zorder=0)
    ax2.text(0.5, -0.135, 'Games above replacement per season',
             transform=ax2.transAxes, fontsize=10, ha='center')

    rmean = w["repro_vs_career_mean_war"]["r"]
    rten = w["repro_vs_n_hc_seasons"]["r"]
    rpart = w["repro_vs_mean_war_partial_tenure"]["r"]
    txt2 = (
        f"$R$ vs mean WAR:  $r$ = {rmean:.2f}\n"
        f"$R$ vs tenure:     $r$ = {rten:.2f}\n"
        f"$R$ vs WAR | tenure: $r$ = {rpart:.2f}\n"
        f"(point size $\\propto$ seasons coached)"
    )
    ax2.text(0.03, 0.95, txt2, transform=ax2.transAxes, fontsize=11.5,
             va='top', ha='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.92,
                       edgecolor='black'))

    plt.suptitle('Two Fitnesses of NFL Head Coaches: Performance vs Progeny',
                 fontsize=17, fontweight='bold', y=1.02)
    plt.tight_layout()

    out_dir = REPO / "outputs/visualizations/performance"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "reproductive_fitness.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info("Saved figure: %s", out_file)
    plt.close()


if __name__ == "__main__":
    main()
