"""
Preprocessing Step 1: Data Preparation

Cleans raw study data and uses GPT-4.1 to:
1. Segment journal transcripts into 5 Gross Process Model stages
2. Assign human counterfactuals to their corresponding stages
3. Paraphrase target goals into concise first-person sentences

Source notebook: IUI_CleanedFinalDataPrep.ipynb
"""

import argparse
import json
import os
import re
import sys
import time

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.helpers import compute_actionables, is_nonempty


# ============================================================
# Configuration
# ============================================================

MODEL = "gpt-4.1-2025-04-14"
TEMPERATURE_STAGE = 0.3
TEMPERATURE_GOAL = 0.3
MAX_RETRIES = 3
SLEEP_BASE = 2


# ============================================================
# Prompts
# ============================================================

STAGE_SYSTEM_PROMPT = (
    "You are an expert in paraphrasing and segmenting reflective journal entries "
    "into the five stages of the Gross Process Model of Emotion Regulation."
)

STAGE_USER_PROMPT = """
Your task is to label the journal entries into five stages.
From the following journal entry, extract what the user explicitly described doing, feeling,
or deciding for each of the five phases below.

Do not infer or add new information beyond what is clearly stated.
Try to keep as much detail and in original tone.

Guidelines:
- Use only details explicitly present in the journal.
- Each phase should be expressed in natural first-person language.
- Preserve order and context; do not summarize or generalize.
- Each phase can include multiple short sentences if needed (max 3 per phase).
- If a phase is not represented, set its value to an empty string "".
- Always return valid minified JSON with keys "0" through "4".
- No code fences, no explanations, no extra text.

Phases:
0. Situation Selection — Choosing to approach or avoid situations or people.
1. Situation Modification — Changing the environment to alter its emotional impact.
2. Attentional Deployment — Directing or shifting attention to influence emotions.
3. Cognitive Change — Reframing or reinterpreting the meaning of a situation.
4. Response Modulation — Managing emotional expression, behavior, or physiology.

Journal entry: {journal}
"""

CF_ASSIGNMENT_SYSTEM_PROMPT = (
    "You are an expert in labeling and paraphrasing counterfactual sentences into the "
    "five stages of the Gross Process Model of Emotion Regulation."
)

CF_ASSIGNMENT_USER_PROMPT = """
Your task is to label counterfactual statements into five stages.

Phases:
0. Situation Selection — Choosing to approach or avoid situations or people.
1. Situation Modification — Changing the environment to alter its emotional impact.
2. Attentional Deployment — Directing or shifting attention to influence emotions.
3. Cognitive Change — Reframing or reinterpreting the meaning of a situation.
4. Response Modulation — Managing emotional expression, behavior, or physiology.

There may be multiple counterfactuals per phase or none for some phases.
Return only a JSON object with keys 0-4; use "" for any phase not found.
No code fences, no explanation, no extra text—just minified JSON.

Journal entry (for context): {journal}
Counterfactual text: {answers}
"""

GOAL_SYSTEM_PROMPT = (
    "You are an expert at paraphrasing user-written goals into a single, clear, "
    "first-person sentence that captures the main focus or change the user wants to make."
)

GOAL_USER_PROMPT = """
Paraphrase the following reflection into one concise first-person sentence that expresses
the user's overall goal or focus for improvement.

Guidelines:
- Capture the main behavioral or emotional change the user wants to make.
- Preserve the user's tone.
- Avoid repeating examples or minor details.
- Keep it under 25 words.
- Output only one natural-sounding first-person sentence.

Original goal text:
{goal}
"""


# ============================================================
# API call with retry
# ============================================================

def gpt_call(client: OpenAI, system_prompt: str, user_prompt: str, temperature: float) -> str:
    """Make a GPT-4.1 API call with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  Retry {attempt + 1}/{MAX_RETRIES}: {e}")
            time.sleep(SLEEP_BASE * (2 ** attempt))
    return json.dumps({"0": "", "1": "", "2": "", "3": "", "4": ""})


# ============================================================
# Processing steps
# ============================================================

def segment_transcripts(client: OpenAI, df: pd.DataFrame) -> pd.DataFrame:
    """Step 1: Segment journal transcripts into Gross Model stages."""
    print("\n--- Step 1: Segmenting transcripts into stages ---")
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Segmenting"):
        journal = str(row.get("transcriptionText", "")).strip()
        if not journal or journal.lower() == "nan":
            results.append("")
            continue

        prompt = STAGE_USER_PROMPT.format(journal=journal)
        result = gpt_call(client, STAGE_SYSTEM_PROMPT, prompt, TEMPERATURE_STAGE)

        try:
            parsed = json.loads(result)
            results.append(json.dumps(parsed, ensure_ascii=False))
        except json.JSONDecodeError:
            results.append(result)

    df["transcript_paraphrased_staged_json"] = results

    # Also create merged version and individual stage columns
    for i in range(5):
        df[f"transcript_paraphrased_{i}"] = df["transcript_paraphrased_staged_json"].apply(
            lambda x: json.loads(x).get(str(i), "") if x else ""
        )

    df["transcript_paraphrased_staged_merged"] = df.apply(
        lambda r: " ".join(
            str(r.get(f"transcript_paraphrased_{i}", "")).strip()
            for i in range(5)
        ).strip(),
        axis=1,
    )
    return df


def assign_human_counterfactuals(client: OpenAI, df: pd.DataFrame) -> pd.DataFrame:
    """Step 2: Assign human counterfactuals to Gross Model stages."""
    print("\n--- Step 2: Assigning human counterfactuals to stages ---")
    results_json = []
    results_actionables = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Assigning CFs"):
        journal = str(row.get("transcript_paraphrased_staged_json", ""))
        answers = str(row.get("answers", ""))

        prompt = CF_ASSIGNMENT_USER_PROMPT.format(journal=journal, answers=answers)
        result = gpt_call(client, CF_ASSIGNMENT_SYSTEM_PROMPT, prompt, TEMPERATURE_STAGE)

        try:
            parsed = json.loads(result)
            result_min = json.dumps(parsed, separators=(",", ":"))
            actionables = compute_actionables(parsed)
        except json.JSONDecodeError:
            result_min = result
            actionables = [0, 0, 0, 0, 0]

        results_json.append(result_min)
        results_actionables.append(actionables)

    df["human_alt_staged_json"] = results_json
    df["actionables_human"] = results_actionables

    # Create individual stage columns
    for i in range(5):
        df[f"s{i}_human_alt"] = df["human_alt_staged_json"].apply(
            lambda x: json.loads(x).get(str(i), "") if x and x.startswith("{") else ""
        )

    return df


def paraphrase_goals(client: OpenAI, df: pd.DataFrame) -> pd.DataFrame:
    """Step 3: Paraphrase target goals."""
    print("\n--- Step 3: Paraphrasing target goals ---")
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Paraphrasing goals"):
        goal = str(row.get("target_goal", "")).strip()
        if not goal or goal.lower() == "nan":
            results.append("")
            continue

        prompt = GOAL_USER_PROMPT.format(goal=goal)
        result = gpt_call(client, GOAL_SYSTEM_PROMPT, prompt, TEMPERATURE_GOAL)
        results.append(result)

    df["target_goal_paraphrased"] = results
    return df


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Preprocessing: Data Preparation")
    parser.add_argument("--input", required=True, help="Path to raw CSV")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY environment variable")
        return
    client = OpenAI(api_key=api_key)

    print(f"Loading raw data from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows")

    df = segment_transcripts(client, df)
    df = assign_human_counterfactuals(client, df)
    df = paraphrase_goals(client, df)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved preprocessed data → {args.output}")


if __name__ == "__main__":
    main()
