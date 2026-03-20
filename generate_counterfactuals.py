"""
Counterfactual Generation: LLM-B, LLM-C, LLM-CT

Generates counterfactuals under three prompt conditions using GPT-4.1,
then assigns each to Gross Process Model stages.

Source notebook: IUI_Cleaned_LLMGeneration.ipynb
"""

import argparse
import json
import os
import sys
from typing import List

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.helpers import coerce_goal, coerce_transcript


# ============================================================
# Configuration
# ============================================================

MODEL = "gpt-4.1-2025-04-14"
TEMPERATURE = 0.2
MAX_TOKENS = 500
MAX_CF = 5


# ============================================================
# Prompt templates
# ============================================================

def prompt_LLMB(goal: str, transcript: str) -> str:
    """Baseline prompt: simple 'what could have been done differently' framing."""
    return f"""
You are to return ONLY valid JSON.

INPUT:
OVERALL GOAL:
"{goal}"

SCENARIO (Transcript):
"{transcript}"

TASK: What could the person have done differently in the <SCENARIO> to be closer to the OVERALL GOAL?

Return the output as a strict JSON array of strings with no commentary, no extra keys,
no prose, no markdown, and a maximum of {MAX_CF} items. For example:
["alternative 1", "alternative 2", "alternative 3"]

Note: PLEASE GIVE alternatives IN FIRST PERSON.
Your response MUST be valid JSON and nothing else.
"""


def prompt_LLMC(goal: str, transcript: str) -> str:
    """Counterfactual prompt: adds explicit counterfactual definition + classifier framing."""
    return f"""
You are to return ONLY valid JSON.

INPUT:
OVERALL GOAL:
"{goal}"

SCENARIO (Transcript):
"{transcript}"

GOAL ALIGNMENT LABEL:
'FALSE'

Assume that a trained black-box classifier correctly predicted <GOAL ALIGNMENT LABEL>,
where the classifier measures whether <SCENARIO (Transcript)> is aligned with <OVERALL GOAL>.

TASK: What could the person have done differently in the <SCENARIO> with minimal change
to be closer to the OVERALL GOAL?

Given the following overall goal and SCENARIO, generate counterfactual explanations,
where each should reflect a minimal change in action the person could have taken.

The purpose is to understand what could be changed to receive a desired result in the future,
based on the current decision-making model.

Note: PLEASE GIVE COUNTERFACTUAL EXPLANATION IN FIRST PERSON.
(e.g. I could have, I would have, I will, I won't, I should, Could, Should ...)

Use the following definition of 'counterfactual explanation':
'A counterfactual explanation reveals what should have been different in an instance
to change the prediction of a classifier.'

In this case, generate counterfactual explanations to any part of SCENARIO (Transcript)
to flip the label <GOAL ALIGNMENT LABEL> to 'TRUE'.

Return the output as a strict JSON array of strings with no commentary, no extra keys,
no prose, no markdown, and a maximum of {MAX_CF} items. For example:
["counterfactual 1", "counterfactual 2", "counterfactual 3"]

Your response MUST be valid JSON and nothing else.
"""


def prompt_LLMCT(goal: str, transcript: str) -> str:
    """Two-step prompt: first identifies actionable targets, then generates counterfactuals."""
    return f"""
You are to return ONLY valid JSON.

INPUT:
OVERALL GOAL:
"{goal}"

SCENARIO (Transcript):
"{transcript}"

GOAL ALIGNMENT LABEL:
'FALSE'

Step 1: Assume that a trained black-box classifier correctly predicted <GOAL ALIGNMENT LABEL>,
where the classifier measures whether <SCENARIO (Transcript)> is aligned with <OVERALL GOAL>.

Given the following overall goal and SCENARIO, first IDENTIFY all the actionable elements
in the transcribed scenario that might have caused the misalignment with the overall goal.
Do not output these; consider them internally to guide the next step.

Step 2: What could the person have done differently in the <SCENARIO> with minimal change
to be closer to the OVERALL GOAL?

Given the following overall goal and SCENARIO, generate *counterfactual explanations*,
where each should reflect a minimal change in action the person could have taken.

The purpose is to understand what could be changed to receive a desired result in the future,
based on the current decision-making model.

Note: PLEASE GIVE COUNTERFACTUAL EXPLANATION IN FIRST PERSON.
(e.g. I could have, I would have, I will, I won't, I should, Could, Should ...)

Use the following definition of 'counterfactual explanation':
'A counterfactual explanation reveals what should have been different in an instance
to change the prediction of a classifier.'

In this case, generate counterfactual explanations to any part of SCENARIO (Transcript)
to flip the label <GOAL ALIGNMENT LABEL> to 'TRUE'.

Return the output as a strict JSON array of strings with no commentary, no extra keys,
no prose, no markdown, and a maximum of {MAX_CF} items. For example:
["counterfactual 1", "counterfactual 2", "counterfactual 3"]

Your response MUST be valid JSON and nothing else.
"""


# ============================================================
# Stage assignment prompt
# ============================================================

STAGE_SYSTEM_PROMPT = (
    "You are assigning user counterfactuals to the relevant phase of the "
    "gross model of emotion regulation."
)

STAGE_USER_PROMPT = """You are given a json of journal entry texts.
Separate the counterfactuals text into paraphrased individual counterfactuals
and then assign them to the proper phase of the journal entry based on the context
and the phases as defined below:

Each index corresponds to one of the five phases of the Gross Model of Emotion Regulation:
0. Situation Selection — Choosing to approach or avoid situations or people to regulate emotions.
1. Situation Modification — Changing the environment to alter its emotional impact.
2. Attentional Deployment — Directing or shifting attention to influence emotions.
3. Cognitive Change — Reframing or reinterpreting the meaning of a situation.
4. Response Modulation — Managing emotional expression, behavior, or physiology.

There can be multiple counterfactuals per phase, and none for some phases.
If there are multiple counterfactuals for a phase, make a list of strings.
Return the result as a JSON with keys: 0-4 for each corresponding phase.
In the case a phase isn't present, make "" the default filler for that key.
No code fences, no explanation, no extra text—just minified JSON.

Journal entry: {transcript}
Counterfactual text: {counterfactuals}
"""


# ============================================================
# Generation functions
# ============================================================

def generate_counterfactuals(client: OpenAI, prompt: str) -> List[str]:
    """Call GPT-4.1 and parse the JSON array response."""
    try:
        resp = client.responses.create(
            model=MODEL,
            input=prompt,
            max_output_tokens=MAX_TOKENS,
        )
        output = resp.output_text
        parsed = json.loads(output)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed[:MAX_CF] if str(x).strip()]
    except Exception as e:
        print(f"  Generation failed: {e}")
    return []


def assign_to_stages(client: OpenAI, counterfactuals: list, transcript: str) -> dict:
    """Assign counterfactuals to Gross Model stages via GPT-4.1."""
    try:
        resp = client.responses.create(
            model=MODEL,
            input=[
                {"role": "system", "content": STAGE_SYSTEM_PROMPT},
                {"role": "user", "content": STAGE_USER_PROMPT.format(
                    transcript=transcript, counterfactuals=str(counterfactuals)
                )},
            ],
            max_output_tokens=MAX_TOKENS,
        )
        return json.loads(resp.output_text)
    except Exception as e:
        print(f"  Stage assignment failed: {e}")
        return {"0": "", "1": "", "2": "", "3": "", "4": ""}


# ============================================================
# Main pipeline
# ============================================================

def run_condition(client, df, prompt_fn, transcript_col, goal_col, desc):
    """Generate counterfactuals for one condition and assign to stages."""
    cf_results = []
    staged_results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        transcript = coerce_transcript(row.get(transcript_col, ""))
        goal = coerce_goal(row.get(goal_col, ""))

        prompt = prompt_fn(goal, transcript)
        cfs = generate_counterfactuals(client, prompt)
        cf_results.append(cfs)

        # Assign to stages
        transcript_staged = row.get("transcript_paraphrased_staged_json", transcript)
        staged = assign_to_stages(client, cfs, str(transcript_staged))
        staged_results.append(staged)

    return cf_results, staged_results


def main():
    parser = argparse.ArgumentParser(description="Generate LLM counterfactuals")
    parser.add_argument("--input", required=True, help="Path to preprocessed CSV")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY environment variable")
        return
    client = OpenAI(api_key=api_key)

    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)

    # Run each condition on paraphrased input
    for condition, prompt_fn, prefix in [
        ("LLM-B", prompt_LLMB, "LLMB"),
        ("LLM-C", prompt_LLMC, "LLMC"),
        ("LLM-CT", prompt_LLMCT, "LLMCT"),
    ]:
        print(f"\n{'='*50}")
        print(f"Generating {condition} (paraphrased input)...")
        cfs, staged = run_condition(
            client, df, prompt_fn,
            transcript_col="transcript_paraphrased_staged_merged",
            goal_col="target_goal_paraphrased",
            desc=f"{condition} (paraphrased)",
        )
        df[f"{prefix}_alt_staged_json"] = [json.dumps(s) for s in staged]

        print(f"Generating {condition} (raw input)...")
        cfs_raw, staged_raw = run_condition(
            client, df, prompt_fn,
            transcript_col="transcriptionText",
            goal_col="target_goal_paraphrased",
            desc=f"{condition} (raw)",
        )
        df[f"{prefix}_alt_staged_json_raw"] = [json.dumps(s) for s in staged_raw]

    # Save
    df.to_csv(args.output, index=False)
    print(f"\nSaved all conditions → {args.output}")


if __name__ == "__main__":
    main()
