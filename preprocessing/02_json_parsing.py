"""
Preprocessing Step 2: JSON Parsing

Parses JSON stage-assignment columns into individual per-stage columns
(e.g., s0_human_alt, s1_LLMB_alt) for downstream analysis.

Source notebook: IUI_JsonParsing.ipynb
"""

import argparse
import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.helpers import parse_deep, split_into_sentences


# ============================================================
# Parsing functions
# ============================================================

def extract_first_item(val):
    """Extract the first non-empty string from a cell that may be a list or string."""
    if pd.isna(val):
        return ""
    s = str(val).strip()
    if not s:
        return ""

    # Try parsing as list
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list) and len(parsed) > 0:
            return str(parsed[0]).strip()
        elif isinstance(parsed, str) and parsed.strip():
            return parsed.strip()
    except Exception:
        pass

    # Try ast.literal_eval for Python-style lists
    try:
        import ast
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list) and len(parsed) > 0:
            return str(parsed[0]).strip()
        elif isinstance(parsed, str) and parsed.strip():
            return parsed.strip()
    except Exception:
        pass

    return s


def expand_staged_json(df: pd.DataFrame, json_col: str, prefix: str) -> pd.DataFrame:
    """
    Expand a JSON column with keys '0'-'4' into individual columns
    named {prefix}_0 through {prefix}_4.
    """
    for i in range(5):
        def _extract(val, idx=i):
            parsed = parse_deep(val, default={})
            return str(parsed.get(str(idx), "")).strip()
        df[f"s{i}_{prefix}"] = df[json_col].apply(_extract)

    return df


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Preprocessing: JSON Parsing")
    parser.add_argument("--input", required=True, help="Path to staged CSV")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)

    # Define which JSON columns to expand and their prefixes
    # Adjust these based on which columns exist in your data
    json_columns = {
        "human_alt_staged_json": "human_alt",
        "LLMB_alt_staged_json": "LLMB_alt",
        "LLMB_alt_staged_json_raw": "LLMB_alt_raw",
        "LLMC_alt_staged_json": "LLMC_alt",
        "LLMC_alt_staged_json_raw": "LLMC_alt_raw",
        "LLMCT_alt_staged_json": "LLMCT_alt",
        "LLMCT_alt_staged_json_raw": "LLMCT_alt_raw",
    }

    for json_col, prefix in json_columns.items():
        if json_col in df.columns:
            print(f"  Expanding {json_col} → s0_{prefix}...s4_{prefix}")
            df = expand_staged_json(df, json_col, prefix)
        else:
            print(f"  Skipping {json_col} (not found)")

    # Clean up: extract first item from any list-valued cells in the stage columns
    suffixes = [v for v in json_columns.values() if any(f"s0_{v}" in c for c in df.columns)]
    for suffix in suffixes:
        for i in range(5):
            col = f"s{i}_{suffix}"
            if col in df.columns:
                df[col] = df[col].apply(extract_first_item)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved parsed data ({len(df)} rows, {len(df.columns)} cols) → {args.output}")


if __name__ == "__main__":
    main()
