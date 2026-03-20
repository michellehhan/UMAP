"""
Experiment 2-2: Survey Data Preparation

Prepares the blinded, randomized survey for human evaluation on CloudResearch.
For each scenario, randomly selects counterfactuals from each condition.

Source notebook: IUI-HumanEvalChosen.ipynb
"""

import argparse
import ast
import os
import random
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# Column definitions
# ============================================================

CONDITION_COLS = {
    "human": [f"s{i}_human_alt" for i in range(5)],
    "LLMB": [f"s{i}_LLMB_alt" for i in range(5)],
    "LLMC": [f"s{i}_LLMC_alt" for i in range(5)],
    "LLMCT": [f"s{i}_LLMCT_alt" for i in range(5)],
}


# ============================================================
# Selection logic
# ============================================================

def pick_random_indices(row: pd.Series, col_group: list, n: int = 2) -> list:
    """Select up to n random indices among non-empty columns for this row."""
    available = [
        i for i, col in enumerate(col_group)
        if pd.notna(row.get(col)) and str(row.get(col, "")).strip()
    ]
    if len(available) >= n:
        return random.sample(available, n)
    return available


def extract_content(row: pd.Series, indices: list, col_group: list) -> list:
    """Fetch content at the given indices from col_group."""
    values = []
    for i in indices:
        if 0 <= i < len(col_group):
            val = row.get(col_group[i], "")
            values.append(str(val).strip() if pd.notna(val) else "")
    return values


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Experiment 2-2: Survey Data Prep")
    parser.add_argument("--input", required=True, help="Path to processed CSV with all conditions")
    parser.add_argument("--output", required=True, help="Output CSV path for survey data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)

    # For each condition, pick random indices and extract content
    for cond_name, cols in CONDITION_COLS.items():
        # Pick indices
        df[f"rand_{cond_name}"] = df.apply(
            lambda r: str(pick_random_indices(r, cols)), axis=1
        )
        # Extract content
        def _extract(row, cond=cond_name, col_group=cols):
            indices = ast.literal_eval(row[f"rand_{cond}"])
            content = extract_content(row, indices, col_group)
            while len(content) < 2:
                content.append("")
            return content[0], content[1]

        df[[f"{cond_name}_0", f"{cond_name}_1"]] = df.apply(
            _extract, axis=1, result_type="expand"
        )

    # Select columns for survey output
    output_cols = [
        "userId", "sessionNumber",
        "target_goal_paraphrased",
        "transcript_paraphrased_staged_merged",
    ]
    # Add paraphrased transcript stages
    for i in range(5):
        col = f"transcript_paraphrased_{i}"
        if col in df.columns:
            output_cols.append(col)

    # Add selected counterfactuals and their indices
    for cond_name in CONDITION_COLS:
        output_cols.extend([f"{cond_name}_0", f"{cond_name}_1", f"rand_{cond_name}"])

    # Filter to available columns
    output_cols = [c for c in output_cols if c in df.columns]

    df_out = df[output_cols]
    df_out.to_csv(args.output, index=False)
    print(f"Saved survey data ({len(df_out)} scenarios) → {args.output}")


if __name__ == "__main__":
    main()
