#!/usr/bin/env python3
"""Diffusion of composite offensive aggression over time.

Plots the league-wide composite offensive aggression gene (actual minus the
season-agnostic model expectation, reliability-weighted across the four
sub-components) by season, with a 95% CI band, and the cross-coach dispersion.
Because the baseline model excludes season, the rising league mean is the
diffusion signal: coaches move from below the time-invariant baseline early to
above it in the modern era. Writes
outputs/visualizations/performance/aggression_adoption_curve.png. ASCII only.
"""

import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_config import configure_plots

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

GENE_CSV = "data/processed/coaching_genes/aggression_gene_by_year.csv"
COL = "composite_aggression"


def main():
    configure_plots()
    df = pd.read_csv(GENE_CSV)
    df = df.dropna(subset=[COL, "season"])

    rows = []
    for season, g in df.groupby("season"):
        vals = g[COL].to_numpy()
        n = len(vals)
        mean = float(vals.mean())
        sd = float(vals.std(ddof=1)) if n > 1 else 0.0
        se = sd / np.sqrt(n) if n > 0 else 0.0
        rows.append({
            "season": int(season), "mean": mean, "sd": sd,
            "lo": mean - 1.96 * se, "hi": mean + 1.96 * se,
            "p25": float(np.percentile(vals, 25)), "p75": float(np.percentile(vals, 75)),
            "n": n,
        })
    s = pd.DataFrame(rows).sort_values("season")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Cross-coach dispersion (inter-quartile band) behind the mean
    ax.fill_between(s["season"], s["p25"], s["p75"], color="#FF6B35", alpha=0.12,
                    label="Inter-quartile range across coaches")
    # 95% CI of the league mean
    ax.fill_between(s["season"], s["lo"], s["hi"], color="#FF6B35", alpha=0.35,
                    label="95% CI of league mean")
    ax.plot(s["season"], s["mean"], color="#B8400E", linewidth=2.5, marker="o",
            markersize=5, label="League-mean composite offensive aggression")

    ax.axhline(0, color="gray", linestyle="--", linewidth=1.2)
    ax.text(s["season"].min(), 0.0008, "model expectation (season-agnostic baseline)",
            fontsize=10, color="gray", va="bottom")

    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:+.3f}"))
    ax.set_xlabel("Season", fontsize=13, fontweight="bold")
    ax.set_ylabel("Composite aggression (actual minus expected)", fontsize=13,
                  fontweight="bold")
    ax.set_title("Diffusion of composite offensive aggression, 2006-2024",
                 fontsize=14, fontweight="bold", pad=10)
    ax.set_xticks(range(int(s["season"].min()), int(s["season"].max()) + 1, 2))
    ax.legend(loc="upper left", framealpha=0.95, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=":")
    plt.tight_layout()

    out_dir = Path("outputs/visualizations/performance")
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "aggression_adoption_curve.png"
    plt.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Saved: {png}")

    early = s[s["season"] <= 2011]["mean"].mean()
    late = s[s["season"] >= 2018]["mean"].mean()
    sd_early = s[s["season"] <= 2011]["sd"].mean()
    sd_late = s[s["season"] >= 2018]["sd"].mean()
    logger.info(f"mean residual early(<=2011)={early:+.4f}  late(>=2018)={late:+.4f}")
    logger.info(f"dispersion (SD) early={sd_early:.4f}  late={sd_late:.4f}  "
                f"growth={100*(sd_late/sd_early-1):+.0f}%")


if __name__ == "__main__":
    main()
