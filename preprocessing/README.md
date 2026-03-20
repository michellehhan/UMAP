# Preprocessing

Scripts for preparing raw study data into the format needed by downstream experiments.

## Scripts

### `01_data_prep.py`
**Source notebook:** `IUI_CleanedFinalDataPrep.ipynb`

Performs the following steps using GPT-4.1:
1. Loads raw participant data (userId, sessionNumber, transcriptionText, answers, target_goal)
2. Segments journal transcripts into the 5 stages of the Gross Process Model (via GPT-4.1, temperature=0.3)
3. Assigns human-authored counterfactuals to their corresponding Gross Model stages (via GPT-4.1, temperature=0.3)
4. Paraphrases target goals into concise first-person sentences (via GPT-4.1, temperature=0.3)

**Requires:** OpenAI API key (`OPENAI_API_KEY` env var)

```bash
python 01_data_prep.py --input ../data/raw/A_Plan_unparaphrased_targetGoal.csv --output ../data/processed/A_Plan_staged.csv
```

### `02_json_parsing.py`
**Source notebook:** `IUI_JsonParsing.ipynb`

Parses the JSON stage-assignment columns into individual per-stage columns (s0_human_alt, s1_human_alt, etc.) for downstream analysis.

```bash
python 02_json_parsing.py --input ../data/processed/A_Plan_staged.csv --output ../data/processed/A_Plan_parsed.csv
```

## Output Columns

After preprocessing, each row contains:
- `userId`, `sessionNumber` — participant identifiers
- `transcriptionText` — raw journal transcript
- `target_goal`, `target_goal_paraphrased` — original and paraphrased goals
- `transcript_paraphrased_0` through `transcript_paraphrased_4` — journal segmented into 5 Gross Model stages
- `transcript_paraphrased_staged_merged` — all stages merged into one text
- `s0_human_alt` through `s4_human_alt` — human counterfactuals assigned to stages
