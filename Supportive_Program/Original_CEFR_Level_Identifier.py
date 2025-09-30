import re
import pandas as pd
from typing import Callable, Optional

import os
print("Current working directory:", os.getcwd())


# ---------- 1) Prompt builder ----------
CEFR_PROMPT_TEMPLATE = (
    "You are an expert in language proficiency classification based on the "
    "Common European Framework of Reference for Languages (CEFR). Your task is to analyze "
    "the given text or narrative and determine its CEFR level [A1, A2, B1, B2, C1, or C2] "
    "based on vocabulary complexity, grammar, and overall language proficiency. "
    "Provide only the CEFR level as output, without explanation or justification.\n\n"
    "Text: {text}\n\n"
    "Answer:"
)

def build_cefr_prompt(text: str) -> str:
    return CEFR_PROMPT_TEMPLATE.format(text=text)

# ---------- 2) Output parser ----------
_VALID_LABELS = ["A1","A2","B1","B2","C1","C2"]
_LABEL_RE = re.compile(r"\b(A1|A2|B1|B2|C1|C2)\b", re.IGNORECASE)

def parse_cefr_label(model_output: str) -> Optional[str]:
    m = _LABEL_RE.search(model_output or "")
    return m.group(1).upper() if m else None

# ---------- 3) Classifier ----------
def classify_cefr(text: str, llm_call: Callable[[str], str]) -> Optional[str]:
    prompt = build_cefr_prompt(text)
    raw = llm_call(prompt)
    return parse_cefr_label(raw)

# ---------- 4) JSONL handler ----------
def annotate_jsonl_with_original_cefr(
    input_path: str = "Summarization.jsonl",
    text_column: str = "original",
    output_jsonl: str = "Final_Summarization.jsonl",
    llm_call: Callable[[str], str] = None,
):
    assert llm_call is not None, "Please provide an llm_call(prompt)->str function."

    # Read JSONL into DataFrame
    df = pd.read_json(input_path, lines=True)

    # Cache repeated queries
    cache = {}
    def _predict(text):
        if pd.isna(text) or not str(text).strip():
            return None
        key = str(text)
        if key not in cache:
            cache[key] = classify_cefr(key, llm_call)
        return cache[key]

    # Add new column
    df["Original_CEFR_Level"] = df[text_column].apply(_predict)

    # Save back to JSONL (preserve line-by-line JSON objects)
    df.to_json(output_jsonl, orient="records", lines=True, force_ascii=False)
    print(f"Saved JSONL with CEFR levels: {output_jsonl}")
# ---------- Example usage ----------
def openai_llm_call(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key="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")   # 👈 replace YOUR_KEY with your real key
    resp = client.chat.completions.create(
        model="gpt-4o-mini",              # you can use gpt-4o-mini or another model
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return resp.choices[0].message.content.strip()

annotate_jsonl_with_original_cefr(
    input_path="Summarization.jsonl",
    text_column="original",
    output_jsonl="Final_Summarization.jsonl",
    llm_call=openai_llm_call
)
