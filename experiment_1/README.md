# Experiment 1 — Actionable Target Identification (RQ1)

Compares how humans and LLMs identify actionable targets for change, analyzed through the Gross Process Model of Emotion Regulation.

## Script

### `target_comparison.py`
**Source notebook:** `IUI_addCount_perStage_actionableCompare.ipynb`

**Produces:** Figure 2 from the paper (4-panel comparison)

For each condition (Human, LLM-B, LLM-C, LLM-CT), computes:
1. **Count arrays** — number of counterfactuals targeting each of the 5 Gross Model stages
2. **Binary coverage arrays** — which stages were addressed (1) vs. not (0)
3. **Normalized percentages** — stage distributions as proportions
4. **Differences from human baseline** — how each LLM condition deviates from human patterns

```bash
python target_comparison.py \
    --input ../data/processed/A_LLMGenerateOutputs.csv \
    --output outputs/
```

## Outputs

- `outputs/figure2_target_comparison.png` — 4-panel figure (count %, count diff, coverage %, coverage diff)
- `outputs/stage_distributions.csv` — raw numbers for all conditions

## Key Findings (from paper)

- Humans concentrate counterfactuals on Situation Selection (30.8%) and Cognitive Change (22.2%)
- LLMs distribute more evenly, with emphasis on Situation Modification and Cognitive Change
- LLM-B most closely mirrors human distribution; LLM-CT emphasizes the most action-oriented stages
