"""
Preprocessing Step 1: Data Preparation

Cleans raw study data and uses GPT-4.1 to:
1. Segment journal transcripts into 5 Gross Process Model stages
2. Assign human counterfactuals to their corresponding stages
3. Paraphrase target goals into concise first-person sentences

Source notebook: IUI_CleanedFinalDataPrep.ipynb

IMPORTANT: All prompts below are copied verbatim from the original notebook.
The notebook reassigns SYSTEM_PROMPT and USER_PROMPT_TEMPLATE at different
stages — here we use separate named variables for clarity, but the text is
identical to the original.
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

# -----------------------------------------------------------
# Used for: Transcript segmentation (Paraphrasing TargetGoals
# + Temperature Testing section / Appending to OUTPUT_CSV)
# -----------------------------------------------------------

TRANSCRIPT_SYSTEM_PROMPT = (
    "You are an expert in paraphrasing and segmenting reflective journal entries into the five stages of the Gross Process Model of Emotion Regulation."
)

TRANSCRIPT_USER_PROMPT = """
Your task is to label the journal entries into five stages.
From the following journal entry, extract what the user explicitly described doing, feeling, or deciding for each of the five phases below.
Do not infer or add new information beyond what is clearly stated. Try to keep as much as detail and in original tone.
If there are multiple actions, events, or reflections related to one phase, include them all — either as separate short sentences or joined naturally.
Each phase can have multiple sentences if necessary for completeness or clarity.

Guidelines:
- Use only details explicitly present in the journal.
- Each phase should be expressed in natural first-person language that reflects both the context and emotional nuance of the journal entry
- Preserve order and context; do not summarize or generalize.
- Each phase can include multiple short sentences if needed, but try to keep at most 3 sentences per phase.
- If a phase is not represented, set its value to an empty string "".
- Always return valid minified JSON with keys "0" through "4".
- No code fences, no explanations, no extra text.

Counterfactuals refer to a statement on alternative way a situation could have unfolded or an action one could have taken differently.

The following texts are user a journal entry separated into the phases of the gross model of emotion regulation and user counterfactuals about what they could've done differently.
Separate the counterfactuals text into paraphrased individual counterfactuals and then assign them to the proper phase of the journal entry based on the context of the journal entry and the phases as defined below:
0. Situation Selection — Choosing to approach or avoid situations or people to regulate emotions.
   Example: taking a different route to avoid an unpleasant neighbor, or seeking out a supportive friend.
1. Situation Modification — Changing the environment to alter its emotional impact.
   Example: asking someone to lower loud music, or moving a stressful in-person meeting to a phone call.
2. Attentional Deployment — Directing or shifting attention to influence emotions.
   Example: distracting oneself, focusing on a non-emotional detail, or using an engaging task to break rumination.
3. Cognitive Change — Reframing or reinterpreting the meaning of a situation.
   Example: viewing stage fright as excitement, or comparing oneself to others less fortunate to feel better.
4. Response Modulation — Managing emotional expression, behavior, or physiology.
   Example: deep breathing to reduce arousal, suppressing inappropriate laughter, or using alcohol to numb emotions.

There can be multiple counterfactuals per phase, and none for some phases.
Return the result as a JSON with keys: 0-4 for each corresponding phase. In the case a phase isn't present, make "" the default filler for that phase.
No code fences, no explanation, no extra text—just minified JSON.

Journal entry (from extracted_phases_json): {journal}
"""

# -----------------------------------------------------------
# Used for: Human counterfactual assignment to stages
# (Assigning Stages to CondA Human-Generated Counterfactuals)
# -----------------------------------------------------------

CF_ASSIGNMENT_SYSTEM_PROMPT = (
    "You are an expert in labeling and paraphrasing the given counterfactual sentence into the relevant five stages of how the event unfolded based on the Gross Process Model of Emotion Regulation."
)

CF_ASSIGNMENT_USER_PROMPT = """
Your task is to label the journal entries into five stages.
From the following journal entry, extract what the user explicitly described doing, feeling, or deciding for each of the five phases below.
Do not infer or add new information beyond what is clearly stated. Try to keep as much as detail and in original tone.
If there are multiple actions, events, or reflections related to one phase, include them all — either as separate short sentences or joined naturally.
Each phase can have multiple sentences if necessary for completeness or clarity.

Guidelines:
- Use only details explicitly present in the journal.
- Each phase should be expressed in natural first-person language that reflects both the context and emotional nuance of the journal entry
- Preserve order and context; do not summarize or generalize.
- Each phase can include multiple short sentences if needed, but try to keep at most 3 sentences per phase.
- If a phase is not represented, set its value to an empty string "".
- Always return valid minified JSON with keys "0" through "4".
- No code fences, no explanations, no extra text.

Counterfactuals refer to a statement on alternative way a situation could have unfolded or an action one could have taken differently.

The following texts are user a journal entry separated into the phases of the gross model of emotion regulation and user counterfactuals about what they could've done differently.
Separate the counterfactuals text into paraphrased individual counterfactuals and then assign them to the proper phase of the journal entry based on the context of the journal entry and the phases as defined below:
0. Situation Selection — Choosing to approach or avoid situations or people to regulate emotions.
   Example: taking a different route to avoid an unpleasant neighbor, or seeking out a supportive friend.
1. Situation Modification — Changing the environment to alter its emotional impact.
   Example: asking someone to lower loud music, or moving a stressful in-person meeting to a phone call.
2. Attentional Deployment — Directing or shifting attention to influence emotions.
   Example: distracting oneself, focusing on a non-emotional detail, or using an engaging task to break rumination.
3. Cognitive Change — Reframing or reinterpreting the meaning of a situation.
   Example: viewing stage fright as excitement, or comparing oneself to others less fortunate to feel better.
4. Response Modulation — Managing emotional expression, behavior, or physiology.
   Example: deep breathing to reduce arousal, suppressing inappropriate laughter, or using alcohol to numb emotions.

There can be multiple counterfactuals per phase, and none for some phases.
Return the result as a JSON with keys: 0-4 for each corresponding phase. In the case a phase isn't present, make "" the default filler for that phase.
No code fences, no explanation, no extra text—just minified JSON.

Journal entry (from extracted_phases_json): {journal}
Counterfactual text (from answers): {answers}
"""

# -----------------------------------------------------------
# Used for: Splitting Answers into Human Alt Stages
# (Temperature = 0.3 through testing section)
# -----------------------------------------------------------

HUMAN_ALT_SYSTEM_PROMPT = (
    "You are an expert in analyzing and paraphrasing counterfactual statements about emotional experiences. "
    "Your goal is to accurately label each statement into one of the five stages of the Gross Process Model of Emotion Regulation. "
    "Do not infer, elaborate, or redistribute meaning beyond what the user explicitly expressed."
)

HUMAN_ALT_USER_PROMPT = """
Your task is to paraphrase and label the following counterfactual statements into the five stages of the Gross Process Model of Emotion Regulation.

From the counterfactual text below, extract what the user explicitly described doing, feeling, or deciding for each of the five phases.
Keep the content faithful to the user's original intent and tone; do not infer unstated emotions or causes.
If multiple actions, events, or reflections relate to one phase, include them all—either as short separate sentences or joined naturally.
Each phase can include multiple short sentences if needed, but limit to three per phase.

Guidelines:
- Use only information explicitly present in the counterfactual text.
- Express each phase in first-person language consistent with the user's tone.
- Preserve order and emotional context; avoid summarizing or generalizing.
- If a phase is not represented, set its value to an empty string "".
- Always return valid minified JSON with keys "0" through "4".
- No code fences, explanations, or extra text.

Phases:
0. Situation Selection — Choosing to approach or avoid situations or people to regulate emotions.
   Example: taking a different route to avoid an unpleasant neighbor, or seeking out a supportive friend.
1. Situation Modification — Changing the environment to alter its emotional impact.
   Example: asking someone to lower loud music, or moving a stressful in-person meeting to a phone call.
2. Attentional Deployment — Directing or shifting attention to influence emotions.
   Example: distracting oneself, focusing on a neutral detail, or using an engaging task to break rumination.
3. Cognitive Change — Reinterpreting or reassessing the meaning of a situation to alter its emotional impact.
   Example: viewing stage fright as excitement, or realizing that a setback is a chance to learn.
4. Response Modulation — Managing emotional expression, behavior, or physiology.
   Example: deep breathing to reduce arousal, suppressing laughter, or using relaxation to calm down.

There may be multiple counterfactuals per phase or none for some phases.
Return only a JSON object with keys 0–4; use "" for any phase not found.

Counterfactual text (from answers): {answers}
"""

# -----------------------------------------------------------
# Used for: Goal paraphrasing (Target Goal Paraphrased section)
# -----------------------------------------------------------

GOAL_SYSTEM_PROMPT = (
    "You are an expert at paraphrasing user-written goals into a single, clear, first-person sentence "
    "that captures the main focus or change the user wants to make."
)

GOAL_USER_PROMPT = """
Paraphrase the following reflection into one concise first-person sentence that expresses
the user's overall goal or focus for improvement.

Guidelines:
- Capture the *main behavioral or emotional change* the user wants to make.
- Preserve the user's tone (e.g., sincere, reflective, goal-oriented).
- Avoid repeating examples or minor details.
- Keep it under 25 words.
- Output only one natural-sounding first-person sentence. No explanations, no bullet points.

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

        prompt = TRANSCRIPT_USER_PROMPT.format(journal=journal)
        result = gpt_call(client, TRANSCRIPT_SYSTEM_PROMPT, prompt, TEMPERATURE_STAGE)

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

        # The notebook processes each CF individually then combines
        answers_str = answers.strip()
        if answers_str.startswith("[") and answers_str.endswith("]"):
            try:
                cf_list = json.loads(answers_str)
            except json.JSONDecodeError:
                cf_list = [answers_str]
        else:
            cf_list = [answers_str]

        combined = {"0": [], "1": [], "2": [], "3": [], "4": []}
        for cf in cf_list:
            if not cf or str(cf).lower() == "nan":
                continue
            prompt = HUMAN_ALT_USER_PROMPT.format(answers=cf)
            result = gpt_call(client, HUMAN_ALT_SYSTEM_PROMPT, prompt, TEMPERATURE_STAGE)
            try:
                parsed = json.loads(result)
            except json.JSONDecodeError:
                parsed = {"0": "", "1": "", "2": "", "3": "", "4": ""}
            for k in combined.keys():
                if parsed.get(k):
                    combined[k].append(parsed[k])

        staged_json = {k: " ".join(map(str, v)) if v else "" for k, v in combined.items()}
        result_min = json.dumps(staged_json, ensure_ascii=False)
        actionables = compute_actionables(staged_json)

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
    print(f"\nSaved preprocessed data -> {args.output}")


if __name__ == "__main__":
    main()