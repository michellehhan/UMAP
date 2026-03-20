# Counterfactual Generation

Generates LLM counterfactuals under three prompt conditions using GPT-4.1, then assigns each to Gross Model stages.

## Script

### `generate_counterfactuals.py`
**Source notebook:** `IUI_Cleaned_LLMGeneration.ipynb`

For each scenario, generates counterfactuals under three conditions:
1. **LLM-B (Baseline):** "What could the person have done differently to be closer to the goal?"
2. **LLM-C (Counterfactual):** Adds explicit counterfactual definition + black-box classifier framing
3. **LLM-CT (Two-Step):** First identifies actionable targets, then generates counterfactuals

Each condition generates on both **paraphrased** and **raw** transcript inputs.

After generation, counterfactuals are assigned to Gross Model stages using GPT-4.1 (temperature=0).

**Requires:** OpenAI API key (`OPENAI_API_KEY` env var)

```bash
python generate_counterfactuals.py \
    --input ../data/processed/A_Plan_parsed.csv \
    --output ../data/processed/A_LLMGenerateOutputs.csv
```

## Model Configuration

| Parameter | Value |
|---|---|
| Model | `gpt-4.1-2025-04-14` |
| Temperature | 0.2 (generation) |
| Max output tokens | 500 |
| Max counterfactuals | 5 per scenario per condition |

## Output Columns Added

- `LLMB_alt_staged_json`, `LLMB_alt_staged_json_raw` — Baseline counterfactuals (paraphrased/raw input)
- `LLMC_alt_staged_json`, `LLMC_alt_staged_json_raw` — Counterfactual-definition counterfactuals
- `LLMCT_alt_staged_json`, `LLMCT_alt_staged_json_raw` — Two-step counterfactuals
