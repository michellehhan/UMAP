"""
Experiment 1: Actionable Target Identification (RQ1)

Compares how humans and LLMs identify actionable targets for change
across the 5 stages of the Gross Process Model of Emotion Regulation.

Source notebook: IUI_addCount_perStage_actionableCompare.ipynb
Produces: Figure 2 from the paper
"""

import argparse
import ast
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

# Add parent dir to path for utils import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.helpers import (
    PHASE_LABELS,
    compute_actionables,
    compute_count_array,
    ensure_str_keys,
    normalize_cf_map,
    parse_deep,
    to_percent,
)


# ============================================================
# Configuration
# ============================================================

# Column names for each condition's staged counterfactual JSON
CONDITIONS = {
    "Human": "human_alt_staged_json",
    "LLM-B": "LLMB_alt_staged_json",
    "LLM-C": "LLMC_alt_staged_json",
    "LLM-CT": "LLMCT_alt_staged_json",
}

# Also track raw-input variants for LLM conditions
RAW_CONDITIONS = {
    "LLM-B (raw)": "LLMB_alt_staged_json_raw",
    "LLM-C (raw)": "LLMC_alt_staged_json_raw",
    "LLM-CT (raw)": "LLMCT_alt_staged_json_raw",
}

STAGE_LABELS = [
    "Situation\nSelection",
    "Situation\nModification",
    "Attention\nDeployment",
    "Cognitive\nChange",
    "Response\nModulation",
]

COLORS = {
    "Human": "#ff6b6b",
    "LLM-B": "#ffe66d",
    "LLM-C": "#4ecdc4",
    "LLM-CT": "#1a535c",
}


# ============================================================
# Core computation
# ============================================================

def compute_stage_arrays(df: pd.DataFrame, col: str):
    """
    For a given condition column, compute:
    - count_arr: total counterfactuals per stage across all scenarios
    - coverage_arr: total scenarios that address each stage (binary)
    """
    count_arr = np.zeros(5, dtype=int)
    coverage_arr = np.zeros(5, dtype=int)

    for _, row in df.iterrows():
        raw = parse_deep(row.get(col, ""), default={})
        cf_map = normalize_cf_map(raw)

        counts = compute_count_array(cf_map)
        actionables = compute_actionables(raw)

        count_arr += np.array(counts)
        coverage_arr += np.array(actionables)

    return count_arr, coverage_arr


def compute_all_conditions(df: pd.DataFrame, conditions: dict):
    """Compute stage arrays for all conditions."""
    results = {}
    for name, col in conditions.items():
        if col not in df.columns:
            print(f"  Warning: column '{col}' not found, skipping {name}")
            continue
        count_arr, coverage_arr = compute_stage_arrays(df, col)
        results[name] = {
            "counts": count_arr,
            "counts_pct": to_percent(count_arr),
            "coverage": coverage_arr,
            "coverage_pct": to_percent(coverage_arr),
        }
    return results


# ============================================================
# Visualization (Figure 2)
# ============================================================

def annotate_bars(ax, data, xpos, fontsize=9):
    """Add percentage labels above/below bars."""
    for i, val in enumerate(data):
        ax.text(
            xpos[i],
            val + (1.0 if val >= 0 else -1.5),
            f"{val:.1f}%",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=fontsize,
        )


def plot_figure2(results: dict, output_path: str):
    """
    Generate the 4-panel Figure 2:
    (a) Count by stage (%)
    (b) Count difference from human (%)
    (c) Binary coverage (%)
    (d) Coverage difference from human (%)
    """
    cond_names = [n for n in ["Human", "LLM-B", "LLM-C", "LLM-CT"] if n in results]
    llm_names = [n for n in cond_names if n != "Human"]
    colors = [COLORS[n] for n in cond_names]
    diff_colors = [COLORS[n] for n in llm_names]

    width = 0.18
    x = np.arange(5)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), sharey=False)

    # --- Panel (a): Count distribution (%) ---
    count_pcts = np.vstack([results[n]["counts_pct"] for n in cond_names]).T
    for i, name in enumerate(cond_names):
        axes[0].bar(x + i * width, count_pcts[:, i], width, label=name, color=colors[i])
    axes[0].set_xticks(x + 1.5 * width)
    axes[0].set_xticklabels(STAGE_LABELS, rotation=25, ha="right")
    axes[0].set_ylabel("Percentage (%)")
    axes[0].set_title("(a) Count by Stage (%)")

    # --- Panel (b): Count differences from human ---
    human_counts_pct = results["Human"]["counts_pct"]
    for i, name in enumerate(llm_names):
        diff = results[name]["counts_pct"] - human_counts_pct
        xpos = x + i * width
        axes[1].bar(xpos, diff, width, label=f"{name} − Human", color=diff_colors[i])
        annotate_bars(axes[1], diff, xpos)
    axes[1].axhline(0, color="gray", linewidth=1, linestyle="--")
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(STAGE_LABELS, rotation=25, ha="right")
    axes[1].set_ylabel("Difference from Human (%)")
    axes[1].set_title("(b) Count Difference from Human (%)")
    axes[1].set_ylim(-20, 20)

    # --- Panel (c): Binary coverage (%) ---
    cov_pcts = np.vstack([results[n]["coverage_pct"] for n in cond_names]).T
    for i, name in enumerate(cond_names):
        axes[2].bar(x + i * width, cov_pcts[:, i], width, label=name, color=colors[i])
    axes[2].set_xticks(x + 1.5 * width)
    axes[2].set_xticklabels(STAGE_LABELS, rotation=25, ha="right")
    axes[2].set_ylabel("Coverage (%)")
    axes[2].set_title("(c) Stage Coverage (%)")

    # --- Panel (d): Coverage differences from human ---
    human_cov_pct = results["Human"]["coverage_pct"]
    for i, name in enumerate(llm_names):
        diff = results[name]["coverage_pct"] - human_cov_pct
        xpos = x + i * width
        axes[3].bar(xpos, diff, width, label=f"{name} − Human", color=diff_colors[i])
        annotate_bars(axes[3], diff, xpos)
    axes[3].axhline(0, color="gray", linewidth=1, linestyle="--")
    axes[3].set_xticks(x + width)
    axes[3].set_xticklabels(STAGE_LABELS, rotation=25, ha="right")
    axes[3].set_ylabel("Difference from Human (%)")
    axes[3].set_title("(d) Coverage Difference from Human (%)")
    axes[3].set_ylim(-20, 20)

    # --- Shared legend ---
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=12)

    fig.suptitle(
        "Comparison of Human- and LLM-Generated Counterfactuals\n"
        "Across the Five Stages of the Gross Process Model",
        fontsize=14, fontweight="bold", y=1.05,
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 2 → {output_path}")


# ============================================================
# Print summary table
# ============================================================

def print_summary(results: dict):
    """Print the stage distribution numbers matching the paper."""
    print("\n=== Stage Distribution Summary ===\n")
    for name in ["Human", "LLM-B", "LLM-C", "LLM-CT"]:
        if name not in results:
            continue
        r = results[name]
        pct = np.round(r["counts_pct"], 1)
        cov = np.round(r["coverage_pct"], 1)
        print(f"{name:8s}  Counts(%): {pct}  Coverage(%): {cov}")

    print("\n=== Differences from Human (Counts %) ===\n")
    human_pct = results["Human"]["counts_pct"]
    for name in ["LLM-B", "LLM-C", "LLM-CT"]:
        if name not in results:
            continue
        diff = np.round(results[name]["counts_pct"] - human_pct, 1)
        print(f"{name:8s} − Human: {diff}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Target Identification")
    parser.add_argument("--input", required=True, help="Path to processed CSV with all conditions")
    parser.add_argument("--output", default="outputs/", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input, dtype=str).fillna("")

    print("Computing stage arrays for all conditions...")
    results = compute_all_conditions(df, CONDITIONS)

    print_summary(results)

    fig_path = os.path.join(args.output, "figure2_target_comparison.png")
    plot_figure2(results, fig_path)

    # Save raw numbers
    rows = []
    for name, r in results.items():
        for i in range(5):
            rows.append({
                "condition": name,
                "stage": PHASE_LABELS[i],
                "count": int(r["counts"][i]),
                "count_pct": round(r["counts_pct"][i], 2),
                "coverage": int(r["coverage"][i]),
                "coverage_pct": round(r["coverage_pct"][i], 2),
            })
    csv_path = os.path.join(args.output, "stage_distributions.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved stage distributions → {csv_path}")


if __name__ == "__main__":
    main()
