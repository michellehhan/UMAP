"""
Experiment 2-1: Automatic Evaluation of Counterfactual Quality (RQ2)

Computes perplexity (GPT-2), goal alignment, contextual coherence (BERT),
and diversity for each condition.

Source notebook: IUI_CondA_AutomatricEvaluation.ipynb
Produces: Automatic metrics in Table 1
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, GPT2LMHeadModel, GPT2TokenizerFast

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.helpers import (
    PHASE_KEYS,
    MIN_WORDS,
    flatten_counterfactuals,
    normalize_cf_map,
    normalize_phases_map,
    parse_deep,
    split_into_sentences,
    word_count,
)


# ============================================================
# Device setup
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Model loading
# ============================================================

def load_gpt2():
    """Load GPT-2 model and tokenizer for perplexity computation."""
    print("Loading GPT-2...")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model.eval()
    return model, tokenizer


def load_bert():
    """Load BERT model and tokenizer for embedding computation."""
    print("Loading BERT...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").to(device)
    model.eval()
    return model, tokenizer


# ============================================================
# Metric computation
# ============================================================

@torch.inference_mode()
def compute_perplexity(sentence: str, gpt2_model, gpt2_tokenizer) -> float:
    """Compute GPT-2 perplexity for a sentence."""
    encodings = gpt2_tokenizer(sentence, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    if input_ids.shape[1] < 2:
        return float("nan")

    outputs = gpt2_model(input_ids=input_ids)
    logits = outputs.logits[:, :-1, :]
    target_ids = input_ids[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)
    target_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    mean_nll = -target_log_probs.mean()
    perplexity = torch.exp(mean_nll)

    return perplexity.item()


@torch.inference_mode()
def embed_sentence(text: str, bert_model, bert_tokenizer, max_len: int = 64) -> torch.Tensor:
    """Compute mean-pooled BERT embedding for a sentence."""
    encoded = bert_tokenizer(
        text, return_tensors="pt", padding="max_length",
        truncation=True, max_length=max_len,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
    last_hidden = outputs.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    sum_embeddings = (last_hidden * mask_expanded).sum(1)
    sum_mask = mask_expanded.sum(1)
    mean_pooled = sum_embeddings / sum_mask

    return mean_pooled.squeeze(0)


def cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    """Compute cosine similarity between two embeddings."""
    return torch.cosine_similarity(emb1, emb2, dim=0).item()


def compute_diversity(sentences: List[str], bert_model, bert_tokenizer) -> float:
    """Compute average pairwise cosine distance among a set of sentences."""
    if len(sentences) < 2:
        return 0.0

    embeddings = [embed_sentence(s, bert_model, bert_tokenizer) for s in sentences]
    n = len(embeddings)
    total_distance = 0.0
    count = 0

    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            total_distance += (1 - sim)
            count += 1

    return total_distance / count if count > 0 else 0.0


# ============================================================
# Per-condition evaluation
# ============================================================

@torch.inference_mode()
def evaluate_condition(
    df: pd.DataFrame,
    condition_col: str,
    gpt2_model, gpt2_tokenizer,
    bert_model, bert_tokenizer,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate a single condition column. Returns:
    - metrics_df: per-counterfactual metrics (perplexity, target sim, phase sim)
    - diversity_df: per-scenario diversity scores
    """
    metric_rows = []
    diversity_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {condition_col}", leave=False):
        user_id = row.get("userId", "")
        session = row.get("sessionNumber", "")
        target_text = (row.get("target_goal_paraphrased", "") or "").strip()

        # Parse counterfactual map
        raw_json = parse_deep(row.get(condition_col, ""), default={})

        # Special handling for human condition: values are strings, not lists
        if condition_col == "human_alt_staged_json":
            raw_json = {
                str(k): split_into_sentences(v) if isinstance(v, str) else []
                for k, v in raw_json.items()
            }

        phases_map = normalize_phases_map(raw_json)
        cf_map = normalize_cf_map(raw_json)
        all_cfs = flatten_counterfactuals(cf_map)

        # Diversity
        div_score = compute_diversity(all_cfs, bert_model, bert_tokenizer) if len(all_cfs) >= 2 else 0.0
        diversity_rows.append({
            "userId": user_id,
            "sessionNumber": session,
            "model_source": condition_col,
            "cf_count": len(all_cfs),
            "diversity": div_score,
        })

        # Target embedding
        target_emb = embed_sentence(target_text, bert_model, bert_tokenizer) if target_text else None

        # Phase embeddings
        phase_embs = {}
        for k in PHASE_KEYS:
            pt = phases_map.get(k, "").strip()
            phase_embs[k] = embed_sentence(pt, bert_model, bert_tokenizer) if pt else None

        # Per-counterfactual metrics
        for k in PHASE_KEYS:
            cf_list = cf_map.get(k, [])
            for idx, cf in enumerate(cf_list):
                if word_count(cf) < MIN_WORDS:
                    continue

                try:
                    ppl = compute_perplexity(cf, gpt2_model, gpt2_tokenizer)
                except Exception:
                    ppl = float("nan")

                try:
                    cf_emb = embed_sentence(cf, bert_model, bert_tokenizer)
                except Exception:
                    cf_emb = None

                sim_target = cosine_similarity(cf_emb, target_emb) if (cf_emb is not None and target_emb is not None) else float("nan")
                sim_phase = cosine_similarity(cf_emb, phase_embs.get(k)) if (cf_emb is not None and phase_embs.get(k) is not None) else float("nan")

                metric_rows.append({
                    "userId": user_id,
                    "sessionNumber": session,
                    "model_source": condition_col,
                    "phase": k,
                    "cf_index": idx,
                    "cf_text": cf,
                    "perplexity": ppl,
                    "similarity_to_target": sim_target,
                    "similarity_to_phase": sim_phase,
                })

    return pd.DataFrame(metric_rows), pd.DataFrame(diversity_rows)


# ============================================================
# Summary printing
# ============================================================

def print_summary(all_metrics: Dict[str, pd.DataFrame], all_diversity: Dict[str, pd.DataFrame]):
    """Print average metrics matching Table 1 format."""
    print("\n" + "=" * 70)
    print(f"{'Condition':<25} {'Perplexity↓':>12} {'TargetSim↑':>12} {'PhaseSim↑':>12} {'Diversity↑':>12}")
    print("=" * 70)

    for name, metrics_df in all_metrics.items():
        if metrics_df.empty:
            continue
        avg_ppl = metrics_df["perplexity"].mean(skipna=True)
        avg_target = metrics_df["similarity_to_target"].mean(skipna=True)
        avg_phase = metrics_df["similarity_to_phase"].mean(skipna=True)

        div_df = all_diversity[name]
        valid_div = div_df[div_df["cf_count"] > 1]
        avg_div = valid_div["diversity"].mean(skipna=True) if not valid_div.empty else float("nan")

        print(f"{name:<25} {avg_ppl:>12.2f} {avg_target:>12.2f} {avg_phase:>12.2f} {avg_div:>12.2f}")

    print("=" * 70)


# ============================================================
# Main
# ============================================================

# Conditions to evaluate (column name → display name)
MODEL_COLS = {
    "human_alt_staged_json": "Human",
    "LLMB_alt_staged_json": "LLM-B (paraphrased)",
    "LLMB_alt_staged_json_raw": "LLM-B (raw)",
    "LLMC_alt_staged_json": "LLM-C (paraphrased)",
    "LLMC_alt_staged_json_raw": "LLM-C (raw)",
    "LLMCT_alt_staged_json": "LLM-CT (paraphrased)",
    "LLMCT_alt_staged_json_raw": "LLM-CT (raw)",
}


def main():
    parser = argparse.ArgumentParser(description="Experiment 2-1: Automatic Evaluation")
    parser.add_argument("--input", required=True, help="Path to CSV with generated counterfactuals")
    parser.add_argument("--output", default="outputs/", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input, dtype=str).fillna("")

    # Load models
    gpt2_model, gpt2_tokenizer = load_gpt2()
    bert_model, bert_tokenizer = load_bert()

    # Evaluate each condition
    all_metrics = {}
    all_diversity = {}

    for col, display_name in MODEL_COLS.items():
        if col not in df.columns:
            print(f"  Skipping {display_name}: column '{col}' not found")
            continue

        print(f"\nEvaluating {display_name}...")
        metrics_df, diversity_df = evaluate_condition(
            df, col, gpt2_model, gpt2_tokenizer, bert_model, bert_tokenizer,
        )

        all_metrics[display_name] = metrics_df
        all_diversity[display_name] = diversity_df

        # Save per-condition results
        metrics_path = os.path.join(args.output, f"metrics_{col}.csv")
        diversity_path = os.path.join(args.output, f"diversity_{col}.csv")
        metrics_df.to_csv(metrics_path, index=False)
        diversity_df.to_csv(diversity_path, index=False)
        print(f"  Saved → {metrics_path}, {diversity_path}")

    # Print summary table
    print_summary(all_metrics, all_diversity)


if __name__ == "__main__":
    main()
