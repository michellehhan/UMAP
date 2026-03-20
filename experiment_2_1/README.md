# Experiment 2-1 — Automatic Evaluation (RQ2)

Evaluates counterfactual quality using automatic NLP metrics across all conditions.

## Script

### `automatic_evaluation.py`
**Source notebook:** `IUI_CondA_AutomatricEvaluation.ipynb`

**Produces:** Automatic metrics in Table 1

Computes four metrics for each counterfactual across all conditions (Human, LLM-B, LLM-C, LLM-CT) on both paraphrased and raw inputs:

1. **Fluency (Perplexity ↓)** — GPT-2 log-likelihood perplexity. Lower = more fluent.
2. **Goal Alignment (Target Similarity ↑)** — Cosine similarity between BERT embeddings of counterfactual and target goal.
3. **Contextual Coherence (Phase Similarity ↑)** — Cosine similarity between counterfactual and corresponding scenario phase text.
4. **Diversity (↑)** — Average pairwise cosine distance among counterfactuals for the same scenario.

```bash
python automatic_evaluation.py \
    --input ../data/processed/A_LLMGenerateOutputs.csv \
    --output outputs/
```

## Model Details

| Component | Model | Purpose |
|---|---|---|
| Perplexity | GPT-2 (`gpt2`) | Fluency measurement |
| Embeddings | BERT (`bert-base-uncased`) | Mean-pooled sentence embeddings for similarity/diversity |

**GPU recommended** — BERT and GPT-2 inference runs on CUDA if available.

## Outputs

- `outputs/condA_metrics_{condition}.csv` — Per-counterfactual metrics for each condition
- `outputs/condA_diversity_{condition}.csv` — Per-scenario diversity scores for each condition

## Expected Results (from paper, Table 1)

| Condition | Perplexity ↓ | Target Sim ↑ | Phase Sim ↑ | Diversity ↑ |
|---|---|---|---|---|
| Human | 104.28 | 0.69 | 0.85 | 0.27 |
| LLM-B | 94.94 | 0.76 | 0.87 | 0.18 |
| LLM-C | 81.34 | 0.77 | 0.88 | 0.17 |
| LLM-CT | 90.77 | 0.78 | 0.88 | 0.17 |
