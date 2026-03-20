"""
Experiment 2-2: Human Evaluation Analysis (RQ2)

Statistical analysis of expert ratings from CloudResearch/Qualtrics survey.
Computes ICC(2,k), Friedman test, pairwise t-tests, Cohen's d, ordinal
regression, and generates Figure 4 boxplots.

Source notebook: [ICC] AI-HumanEvaluate-Data.ipynb
Produces: Table 1 (human metrics), Figure 4, ICC table
"""

import argparse
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, ttest_rel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# Configuration
# ============================================================

MODELS = ["human", "LLMB", "LLMC", "LLMCT"]
MODEL_DISPLAY = {"human": "Human", "LLMB": "LLM-B", "LLMC": "LLM-C", "LLMCT": "LLM-CT"}

METRIC_MAP = {
    "1": "Feasibility",
    "2": "Action Clarity",
    "3": "Goal Alignment",
    "4": "Overall Effectiveness",
}
METRICS = list(METRIC_MAP.values())

COLORS = {
    "human": "#ff6b6b",
    "LLMB": "#ffe66d",
    "LLMC": "#4ecdc4",
    "LLMCT": "#1a535c",
}

# Regex to parse Qualtrics column names
# Pattern: {scenario}_Q_{model}[_raw]_0_{metric}
COLUMN_PATTERN = re.compile(
    r"(?P<scenario>\d+)_Q_(?P<model>[A-Za-z]+)(?:_raw)?_0_(?P<metric>[1-4])"
)


# ============================================================
# Step 1: Reshape Qualtrics wide format -> long format
# ============================================================

def reshape_qualtrics_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert wide Qualtrics export (835 columns) to long format with columns:
    scenario, model, metric, rater, rating
    """
    rows = []
    for rater_id, row in df.iterrows():
        for col in df.columns:
            m = COLUMN_PATTERN.match(col)
            if m and pd.notna(row[col]):
                rows.append({
                    "scenario": int(m.group("scenario")),
                    "model": m.group("model"),
                    "metric": METRIC_MAP[m.group("metric")],
                    "rater": f"rater_{rater_id}",
                    "rating": float(row[col]),
                })

    long_df = pd.DataFrame(rows)
    print(f"Reshaped to long format: {long_df.shape[0]} ratings")
    print(f"  Raters: {long_df['rater'].nunique()}")
    print(f"  Scenarios: {long_df['scenario'].nunique()}")
    print(f"  Models: {long_df['model'].unique().tolist()}")
    return long_df


# ============================================================
# Step 2: ICC(2,k) -- two-way random-effects, average measures
# ============================================================

def anova_icc2k(sub_df: pd.DataFrame) -> float:
    """
    Compute ICC(2,k) via two-way random-effects ANOVA.
    Expects columns: scenario, rater, rating
    """
    mean_per_scenario = sub_df.groupby("scenario")["rating"].mean()
    mean_per_rater = sub_df.groupby("rater")["rating"].mean()
    grand_mean = sub_df["rating"].mean()

    a = sub_df["scenario"].nunique()  # number of targets
    k = sub_df["rater"].nunique()     # number of raters

    MS_scenario = k * np.var(mean_per_scenario, ddof=1)
    MS_rater = a * np.var(mean_per_rater, ddof=1)

    residuals = []
    for s in sub_df["scenario"].unique():
        for r in sub_df["rater"].unique():
            val = sub_df.loc[
                (sub_df["scenario"] == s) & (sub_df["rater"] == r), "rating"
            ]
            if not val.empty:
                residuals.append(
                    val.values[0]
                    - mean_per_scenario[s]
                    - mean_per_rater[r]
                    + grand_mean
                )
    MS_resid = np.var(residuals, ddof=1)

    icc2k = (MS_scenario - MS_resid) / (
        MS_scenario + (MS_rater - MS_resid) / a
    )
    return icc2k


def compute_all_icc(long_df: pd.DataFrame) -> pd.DataFrame:
    """Compute ICC(2,k) for each model x metric combination."""
    rows = []
    for model in MODELS:
        for metric in METRICS:
            sub = long_df[
                (long_df["model"] == model) & (long_df["metric"] == metric)
            ].copy()
            sub["rating"] = pd.to_numeric(sub["rating"], errors="coerce")

            if sub["rater"].nunique() > 1 and sub["scenario"].nunique() > 1:
                try:
                    icc = anova_icc2k(sub)
                    rows.append({
                        "Model": MODEL_DISPLAY.get(model, model),
                        "Metric": metric,
                        "ICC(2,k)": round(icc, 3),
                        "n_raters": sub["rater"].nunique(),
                        "n_scenarios": sub["scenario"].nunique(),
                    })
                except Exception as e:
                    rows.append({
                        "Model": MODEL_DISPLAY.get(model, model),
                        "Metric": metric,
                        "ICC(2,k)": np.nan,
                        "note": f"failed: {e}",
                    })

    icc_df = pd.DataFrame(rows).sort_values(["Metric", "Model"]).reset_index(drop=True)
    return icc_df


# ============================================================
# Step 3: Descriptive statistics
# ============================================================

def compute_descriptives(long_df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean +/- SD for each model x metric."""
    rows = []
    for metric in METRICS:
        for model in MODELS:
            vals = long_df[
                (long_df["model"] == model) & (long_df["metric"] == metric)
            ]["rating"]
            rows.append({
                "Metric": metric,
                "Model": MODEL_DISPLAY.get(model, model),
                "Mean": round(vals.mean(), 2),
                "SD": round(vals.std(), 2),
                "N": len(vals),
            })
    return pd.DataFrame(rows)


# ============================================================
# Step 4: Friedman test + pairwise t-tests + Cohen's d
# ============================================================

def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d for paired samples."""
    diff = x - y
    return diff.mean() / diff.std(ddof=1)


def run_statistical_tests(long_df: pd.DataFrame):
    """Run Friedman omnibus test and pairwise dependent t-tests."""
    print("\n" + "=" * 70)
    print("STATISTICAL TESTS")
    print("=" * 70)

    # --- Friedman test ---
    print("\n--- Friedman Omnibus Test ---")
    for metric in METRICS:
        pivot = long_df[long_df["metric"] == metric].pivot_table(
            index="scenario", columns="model", values="rating", aggfunc="mean"
        )
        available = [m for m in MODELS if m in pivot.columns]
        if len(available) < 3:
            print(f"  {metric}: insufficient conditions")
            continue

        clean = pivot[available].dropna()
        arrays = [clean[m].values for m in available]

        stat, p = friedmanchisquare(*arrays)
        print(f"  {metric}: chi2({len(available)-1}) = {stat:.2f}, p = {p:.2e}")

    # --- Pairwise t-tests: LLM vs Human ---
    print("\n--- Pairwise t-tests (LLM vs Human) ---")
    for metric in METRICS:
        print(f"\n  {metric}:")
        pivot = long_df[long_df["metric"] == metric].pivot_table(
            index="scenario", columns="model", values="rating", aggfunc="mean"
        )
        if "human" not in pivot.columns:
            continue

        for llm in ["LLMB", "LLMC", "LLMCT"]:
            if llm not in pivot.columns:
                continue
            paired = pivot[["human", llm]].dropna()
            t, p = ttest_rel(paired[llm], paired["human"])
            d = cohens_d(paired[llm].values, paired["human"].values)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(
                f"    {MODEL_DISPLAY[llm]:8s} vs Human: "
                f"t = {t:.3f}, p = {p:.2e}, d = {d:.2f} {sig}"
            )

    # --- Pairwise t-tests among LLM variants ---
    print("\n--- Pairwise t-tests (LLM vs LLM) ---")
    llm_pairs = [("LLMB", "LLMC"), ("LLMB", "LLMCT"), ("LLMC", "LLMCT")]
    for metric in METRICS:
        print(f"\n  {metric}:")
        pivot = long_df[long_df["metric"] == metric].pivot_table(
            index="scenario", columns="model", values="rating", aggfunc="mean"
        )
        for a, b in llm_pairs:
            if a not in pivot.columns or b not in pivot.columns:
                continue
            paired = pivot[[a, b]].dropna()
            t, p = ttest_rel(paired[a], paired[b])
            d = cohens_d(paired[a].values, paired[b].values)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(
                f"    {MODEL_DISPLAY[a]:8s} vs {MODEL_DISPLAY[b]:8s}: "
                f"t = {t:.3f}, p = {p:.3f}, d = {d:.2f} {sig}"
            )


# ============================================================
# Step 5: Ordinal logistic regression (supplementary)
# ============================================================

def run_ordinal_regression(long_df: pd.DataFrame):
    """
    Run ordinal logistic regression with cluster-robust SEs (rater clusters).
    Requires: pip install statsmodels
    """
    try:
        from statsmodels.miscmodels.ordinal_model import OrderedModel
    except ImportError:
        print("\nSkipping ordinal regression (install statsmodels to enable)")
        return None

    print("\n" + "=" * 70)
    print("ORDINAL REGRESSION (vs Human baseline, cluster-robust SEs)")
    print("=" * 70)

    rows = []
    for metric in METRICS:
        sub_df = long_df[long_df["metric"] == metric].copy()
        sub_df = sub_df.dropna(subset=["rating"])
        sub_df["rating"] = sub_df["rating"].astype(int)
        sub_df["model"] = pd.Categorical(
            sub_df["model"],
            categories=["human", "LLMB", "LLMC", "LLMCT"],
            ordered=True,
        )

        X = pd.get_dummies(sub_df["model"], drop_first=True)
        y = sub_df["rating"]

        model = OrderedModel(y, X, distr="logit")
        res = model.fit(
            method="bfgs",
            disp=False,
            cov_type="cluster",
            cov_kwds={"groups": sub_df["rater"]},
        )

        keep = [k for k in res.params.index if k in ["LLMB", "LLMC", "LLMCT"]]
        ci = res.conf_int().loc[keep]

        for k in keep:
            beta = res.params[k]
            pval = res.pvalues[k]
            lo, hi = ci.loc[k, 0], ci.loc[k, 1]
            rows.append({
                "Metric": metric,
                "Model vs Human": MODEL_DISPLAY.get(k, k),
                "beta": round(beta, 3),
                "OR": round(np.exp(beta), 3),
                "95% CI (beta)": f"[{lo:.3f}, {hi:.3f}]",
                "95% CI (OR)": f"[{np.exp(lo):.3f}, {np.exp(hi):.3f}]",
                "p": f"{pval:.2e}" if pval < 1e-3 else f"{pval:.3f}",
            })

    result_df = (
        pd.DataFrame(rows)
        .sort_values(["Metric", "Model vs Human"])
        .reset_index(drop=True)
    )
    print(result_df.to_string(index=False))
    return result_df


# ============================================================
# Step 6: Figure 4 -- boxplots
# ============================================================

def plot_figure4(long_df: pd.DataFrame, output_path: str):
    """Generate Figure 4: boxplots of expert ratings across conditions."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)

    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        data = []
        labels_plot = []

        for model in MODELS:
            # Average per scenario for boxplot (as in paper Figure 4)
            scenario_means = (
                long_df[
                    (long_df["model"] == model) & (long_df["metric"] == metric)
                ]
                .groupby("scenario")["rating"]
                .mean()
                .values
            )
            data.append(scenario_means)
            labels_plot.append(MODEL_DISPLAY[model])

        bp = ax.boxplot(data, labels=labels_plot, patch_artist=True, widths=0.6)

        for patch, model in zip(bp["boxes"], MODELS):
            patch.set_facecolor(COLORS[model])
            patch.set_alpha(0.7)

        ax.set_title(metric, fontsize=12)
        ax.set_ylim(0.5, 5.5)
        ax.set_ylabel("Rating (1-5)" if idx == 0 else "")
        ax.tick_params(axis="x", rotation=25)

    fig.suptitle(
        "Distribution of Human Evaluation Scores Across Models",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved Figure 4 -> {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 2-2: Human Evaluation Analysis"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to Qualtrics CSV export (wide format)",
    )
    parser.add_argument("--output", default="outputs/", help="Output directory")
    parser.add_argument(
        "--skip-rows", type=int, nargs="*", default=[1, 2],
        help="Qualtrics metadata rows to skip (default: 1 2)",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load Qualtrics data
    print(f"Loading Qualtrics data from {args.input}...")
    df = pd.read_csv(args.input, skiprows=args.skip_rows)
    print(f"  Shape: {df.shape}")

    # Reshape to long format
    long_df = reshape_qualtrics_to_long(df)
    long_path = os.path.join(args.output, "ratings_long_format.csv")
    long_df.to_csv(long_path, index=False)
    print(f"Saved long format -> {long_path}")

    # Descriptive statistics
    print("\n" + "=" * 70)
    print("DESCRIPTIVE STATISTICS (Mean +/- SD)")
    print("=" * 70)
    desc_df = compute_descriptives(long_df)
    print(desc_df.to_string(index=False))
    desc_df.to_csv(os.path.join(args.output, "descriptives.csv"), index=False)

    # ICC
    print("\n" + "=" * 70)
    print("ICC(2,k) -- Inter-Rater Reliability")
    print("=" * 70)
    icc_df = compute_all_icc(long_df)
    print(icc_df.to_string(index=False))
    icc_df.to_csv(os.path.join(args.output, "icc_results.csv"), index=False)

    # Statistical tests (Friedman + t-tests)
    run_statistical_tests(long_df)

    # Ordinal regression (supplementary)
    ord_df = run_ordinal_regression(long_df)
    if ord_df is not None:
        ord_df.to_csv(os.path.join(args.output, "ordinal_regression.csv"), index=False)

    # Figure 4
    fig_path = os.path.join(args.output, "figure4_human_eval_boxplots.png")
    plot_figure4(long_df, fig_path)

    print("\nAll analyses complete.")


if __name__ == "__main__":
    main()
