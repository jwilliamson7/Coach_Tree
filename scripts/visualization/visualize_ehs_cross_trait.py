#!/usr/bin/env python3
"""C3 cross-trait heritability-selection scatter (the Kruuk / Mousseau-Roff test).

Each of the ten frozen sub-traits is a point in (h^2, S) space: transmissibility on
the x-axis, selection on the y-axis, colored by family (approach vs schematic
identity), with 95\% intervals on both axes. This is the direct cultural analog of
the heritability-fitness scatter used in wild-population quantitative genetics
(Kruuk et al. 2000; Mousseau and Roff 1987); the annotated Spearman rank
correlation is the preregistered H3 statistic.

Reads outputs/analysis/ehs_heritability_results.json + ehs_selection_results.json
and outputs/analysis/ehs_confirmatory_summary.json. Writes
outputs/visualizations/performance/ehs_cross_trait_h2_S.png. ASCII only.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_config import configure_plots

REPO = Path(__file__).resolve().parents[2]
ANALYSIS = REPO / "outputs/analysis"

# sub-trait -> (label, family)  approach = risk/attitude; identity = schematic
SUBTRAITS = {
    "fourth_down": ("Fourth down", "approach"),
    "pass_heavy": ("Pass-heavy", "approach"),
    "deep_pass": ("Deep pass", "approach"),
    "two_point": ("Two-point", "approach"),
    "no_huddle": ("No-huddle", "identity"),
    "pace": ("Pace", "identity"),
    "box_stacking": ("Box stacking", "approach"),
    "pass_rush": ("Pass rush", "approach"),
    "man_coverage": ("Man coverage", "approach"),
    "shotgun": ("Shotgun", "identity"),
}
# Single colour for all sub-traits; the approach / identity distinction is drawn in
# the text, not by segmenting the figure.
PT_COLOR = "#2E86AB"

# manual label nudges (data units) to avoid collisions
NUDGE = {
    "fourth_down": (0.015, 0.006), "pass_heavy": (0.015, 0.006),
    "deep_pass": (0.015, -0.004), "two_point": (0.015, -0.010),
    "no_huddle": (0.015, -0.010), "pace": (0.015, 0.008),
    "box_stacking": (0.015, -0.018), "pass_rush": (0.015, 0.006),
    "man_coverage": (0.015, 0.008), "shotgun": (0.018, 0.016),
}


def main():
    h2 = json.load(open(ANALYSIS / "ehs_heritability_results.json"))["subtraits"]
    S = json.load(open(ANALYSIS / "ehs_selection_results.json"))["subtraits"]
    conf = json.load(open(ANALYSIS / "ehs_confirmatory_summary.json"))["C3_H3"]

    configure_plots()
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axhline(0, color="gray", lw=1.0, ls="-", alpha=0.5, zorder=1)
    ax.axvline(0, color="gray", lw=1.0, ls="-", alpha=0.5, zorder=1)

    for key, (label, fam) in SUBTRAITS.items():
        c1 = h2[key]["c1"]; s = S[key]
        x, xl, xh = c1["h2_median"], c1["hdi_low"], c1["hdi_high"]
        y, yl, yh = s["S"], s["ci_low"], s["ci_high"]
        ax.errorbar(x, y, xerr=[[x - xl], [xh - x]], yerr=[[y - yl], [yh - y]],
                    fmt="o", ms=11, color=PT_COLOR, ecolor=PT_COLOR, elinewidth=1.4,
                    capsize=3, alpha=0.85, markeredgecolor="black",
                    markeredgewidth=0.8, zorder=4)
        dx, dy = NUDGE[key]
        ax.annotate(label, (x, y), (x + dx, y + dy), fontsize=11,
                    color="black", va="center", zorder=5)

    rho = conf["spearman_rho"]; lo = conf["ci_low"]; hi = conf["ci_high"]
    ax.text(0.03, 0.97,
            f"Spearman $r_s(h^2, S) = {rho:+.2f}$\n95% bootstrap [{lo:+.2f}, {hi:+.2f}]\n"
            f"(H3 predicted $<0$; not supported)",
            transform=ax.transAxes, va="top", ha="left", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9))

    ax.set_xlabel("Heritability  $h^{2}$  (parent-offspring, posterior median)",
                  fontsize=14, fontweight="bold")
    ax.set_ylabel("Selection  $S$  (gene $\\rightarrow$ WAR, era-adjusted IVW)",
                  fontsize=14, fontweight="bold")
    ax.set_title("The heritability-fitness relationship across ten coaching sub-traits",
                 fontsize=15, fontweight="bold", pad=14)
    ax.grid(True, alpha=0.25, ls=":", zorder=0)

    plt.tight_layout()
    out = REPO / "outputs/visualizations/performance/ehs_cross_trait_h2_S.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print("Saved:", out)
    plt.close()


if __name__ == "__main__":
    main()
