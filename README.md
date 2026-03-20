# Actionable Counterfactuals Codebase

**Paper:** *Actionable Counterfactuals: Augmenting Human What-If Reasoning with LLM-Generated Recommendations*

This repository contains the analysis scripts for all experiments reported in the paper. The codebase is organized by experiment, with shared preprocessing and generation modules.

---

## Repository Structure

```
umap/
├── README.md
├── data/
│   ├── raw/                           ← Raw anonymized study data (not tracked in git)
│   └── processed/                     ← Cleaned CSVs produced by preprocessing
├── preprocessing/
│   ├── 01_data_prep.py                ← Clean raw data, segment journals into Gross Model stages, paraphrase goals
│   ├── 02_json_parsing.py             ← Parse JSON columns into per-stage counterfactual columns
│   └── README.md
├── generation/
│   ├── generate_counterfactuals.py    ← Generate LLM-B, LLM-C, LLM-CT counterfactuals via GPT-4.1
│   └── README.md
├── experiment_1/
│   ├── target_comparison.py           ← Compute count/coverage arrays across Gross Model stages (Figure 2)
│   └── README.md
├── experiment_2_1/
│   ├── automatic_evaluation.py        ← Perplexity (GPT-2), SBERT similarity, diversity metrics (Table 1)
│   └── README.md
├── experiment_2_2/
│   ├── survey_data_prep.py            ← Prepare blinded survey for CloudResearch human evaluation
│   ├── human_eval_analysis.py         ← Statistical analysis: Friedman, t-tests, ICC, Cohen's d (Table 1, Figure 4)
│   └── README.md
└── utils/
    └── helpers.py                     ← Shared utility functions
```

---

## Mapping: Paper → Scripts

| Paper Section | Experiment | Script |
|---|---|---|
| §3 Data Collection | Formative study | `preprocessing/01_data_prep.py` |
| §3 Data Preparation | JSON parsing | `preprocessing/02_json_parsing.py` |
| §4 Base Model | LLM generation | `generation/generate_counterfactuals.py` |
| §4.1 Experiment 1 | Target identification (Figure 2) | `experiment_1/target_comparison.py` |
| §4.2.1 Experiment 2-1 | Automatic evaluation (Table 1, auto metrics) | `experiment_2_1/automatic_evaluation.py` |
| §4.2.2 Experiment 2-2 | Human evaluation survey prep | `experiment_2_2/survey_data_prep.py` |
| §4.2.2 Experiment 2-2 | Human evaluation analysis (Table 1, human metrics; Figure 4) | `experiment_2_2/human_eval_analysis.py` |

---

## Prerequisites

```bash
pip install openai pandas numpy torch transformers sentence-transformers scipy matplotlib tqdm tenacity
```

- **OpenAI API key** required for: `preprocessing/01_data_prep.py`, `generation/generate_counterfactuals.py`
- **GPU recommended** for: `experiment_2_1/automatic_evaluation.py` (GPT-2 perplexity + BERT embeddings)

Set your API key:
```bash
export OPENAI_API_KEY="sk-..."
```

---

## How to Run Each Experiment

### Step 0: Data Preparation
```bash
# Clean raw data, segment journals into Gross Model stages, paraphrase goals
python preprocessing/01_data_prep.py --input data/raw/A_Plan_unparaphrased_targetGoal.csv --output data/processed/A_Plan_staged.csv

# Parse JSON columns into per-stage columns
python preprocessing/02_json_parsing.py --input data/processed/A_Plan_staged.csv --output data/processed/A_Plan_parsed.csv
```

### Step 1: Generate Counterfactuals
```bash
# Generate LLM-B, LLM-C, LLM-CT counterfactuals + assign to Gross Model stages
python generation/generate_counterfactuals.py --input data/processed/A_Plan_parsed.csv --output data/processed/A_LLMGenerateOutputs.csv
```

### Step 2: Experiment 1 — Target Identification (RQ1)
```bash
# Compute count/coverage arrays, generate Figure 2
python experiment_1/target_comparison.py --input data/processed/A_LLMGenerateOutputs.csv --output experiment_1/outputs/
```

### Step 3: Experiment 2-1 — Automatic Evaluation (RQ2)
```bash
# Compute perplexity, SBERT similarity, diversity → Table 1 automatic metrics
python experiment_2_1/automatic_evaluation.py --input data/processed/A_LLMGenerateOutputs.csv --output experiment_2_1/outputs/
```

### Step 4: Experiment 2-2 — Human Evaluation (RQ2)
```bash
# Prepare blinded survey data for CloudResearch
python experiment_2_2/survey_data_prep.py --input data/processed/A_LLMGenerateOutputs.csv --output experiment_2_2/outputs/survey_data.csv

# After collecting Qualtrics survey responses, run statistical analysis
python experiment_2_2/human_eval_analysis.py --input experiment_2_2/outputs/AI-HumanEvaluate-Data.csv --output experiment_2_2/outputs/ --skip-rows 1 2
```

---

## Model & Hyperparameters

| Parameter | Value |
|---|---|
| Base LLM | GPT-4.1 (`gpt-4.1-2025-04-14`) |
| Temperature | 0.2 (generation), 0.0 (stage assignment), 0.3 (paraphrasing) |
| Max counterfactuals per scenario | 5 |
| Perplexity model | GPT-2 (HuggingFace `gpt2`) |
| Embedding model | BERT (`bert-base-uncased`, mean pooling) |
| Human evaluators | N=35 mental health professionals via CloudResearch Connect |

---

## Data

Raw study data is not included in this repository for privacy reasons. The `data/raw/` directory should contain:
- `A_Plan_unparaphrased_targetGoal.csv` — Condition A participant data with userId, sessionNumber, transcriptionText, answers, target_goal
- `B_Plan_unparaphrased_targetGoal.csv` — Condition B data

Processed intermediates are written to `data/processed/` by the preprocessing scripts.
