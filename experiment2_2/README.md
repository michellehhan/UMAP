# Experiment 2-2 — Human Evaluation (RQ2)

Expert evaluation of counterfactual quality by mental health professionals.

## Scripts

### `survey_data_prep.py`
**Source notebook:** `IUI-HumanEvalChosen.ipynb`

Prepares the blinded survey for CloudResearch Connect:
1. For each scenario, randomly selects counterfactuals from each condition (Human, LLM-B, LLM-C, LLM-CT)
2. Formats into a survey-ready CSV with goal, scenario, and 4 blinded action plans
3. Each evaluator rates 15 randomly assigned scenarios

```bash
python survey_data_prep.py \
    --input ../data/processed/A_LLMGenerateOutputs.csv \
    --output outputs/survey_data.csv
```

### `human_eval_analysis.py`
**Source notebook:** `[ICC] AI-HumanEvaluate-Data.ipynb`

Performs statistical analysis on collected survey responses:
1. **Inter-rater reliability:** ICC(2,k) two-way random-effects model
2. **Omnibus test:** Friedman test for main effect of condition
3. **Pairwise comparisons:** Dependent t-tests (LLM vs Human, LLM vs LLM)
4. **Effect sizes:** Cohen's d for each comparison
5. **Visualization:** Boxplots of expert ratings (Figure 4)
6. **Ordinal regression:** Logistic regression with cluster-robust SEs (supplementary analysis)

```bash
# Input: Qualtrics CSV export (wide format, ~835 columns)
# The script auto-detects columns matching pattern: {scenario}_Q_{model}_raw_0_{metric}
python human_eval_analysis.py \
    --input outputs/AI-HumanEvaluate-Data.csv \
    --output outputs/ \
    --skip-rows 1 2
```

## Survey Design

- **Evaluators:** 35 mental health professionals (CloudResearch Connect)
- **Eligibility:** English fluency, age 18+, experience in mental health or psychology
- **Task:** Rate each of 4 counterfactuals on 5-point Likert scales
- **Metrics:** Feasibility, Action Clarity, Goal Alignment, Overall Effectiveness
- **Load:** 15 scenarios per evaluator (~38 min avg), 525 total evaluations

## Expected Results (from paper, Table 1)

| Condition | Action Clarity ↑ | Feasibility ↑ | Goal Align. ↑ | Effectiveness ↑ |
|---|---|---|---|---|
| Human | 3.29 ± .66 | 3.47 ± .69 | 3.09 ± .74 | 3.04 ± .70 |
| LLM-B | 4.07 ± .36 | 4.06 ± .36 | 4.06 ± .39 | 3.89 ± .42 |
| LLM-C | 4.02 ± .45 | 4.03 ± .42 | 4.07 ± .49 | 3.88 ± .50 |
| LLM-CT | 4.05 ± .38 | 3.97 ± .37 | 4.02 ± .39 | 3.84 ± .38 |

## Statistical Results

- Friedman omnibus: χ²(3) > 38, p < .001 for all metrics
- All LLM vs Human comparisons: p < 10⁻⁵, Cohen's d = 0.86–1.32
- No significant differences among LLM variants (p > .28)
