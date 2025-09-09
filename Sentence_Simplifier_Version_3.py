#!/usr/bin/env python3
import json
import os
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from pathlib import Path


import re,codecs;
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# ---------------------------
# Fixed I/O
# ---------------------------
INPUT_PATH = "tsar2025_trialdata.jsonl"
OUTPUT_PATH = "output_version_3_second.jsonl"

# ---------------------------
# CEFR Labelers (3 models)
# ---------------------------
LABELER_IDS = [
    "AbdullahBarayan/ModernBERT-base-doc_en-Cefr",
    "AbdullahBarayan/ModernBERT-base-doc_sent_en-Cefr",
    "AbdullahBarayan/ModernBERT-base-reference_AllLang2-Cefr2",
]

# ---------------------------
# Defaults (can be tweaked)
# ---------------------------
DEFAULT_GENERATOR = os.environ.get("CEFR_SIMPLIFIER_GENERATOR_MODEL", "HuggingFaceH4/zephyr-7b-beta")
DEFAULT_DEVICE = 0 if torch.cuda.is_available() else -1
DEFAULT_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SIMILARITY_FLOOR = 0.80
DEFAULT_MAX_ITERS = 6

TARGET_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]

@dataclass
class SimplifyResult:
    original: str
    simplified: str
    predicted_level: str
    similarity: float
    iterations: int
    levels_vote: Dict[str, int]

def build_prompt_first(original_text: str, target_cefr: str) -> str:
    return (
        f"Simplify the following text to CEFR level {target_cefr}.\n"
        f"Keep the meaning, make it natural and readable.\n"
        f"Return only the simplified text.\n\n"
        f"Text:\n{original_text}"
    )

def build_prompt_followup(current_text: str, original_text: str, target_cefr: str) -> str:
    return (
        f"You are simplifying text to CEFR level {target_cefr}.\n"
        f"Further simplify the text below, while preserving the original meaning as much as possible.\n"
        f"Prefer simpler vocabulary and structure; avoid adding new facts.\n"
        f"Return only the simplified text.\n\n"
        f"Text:\n{current_text}"
    )

def majority_vote(levels: List[str]) -> Tuple[str, Dict[str, int]]:
    counts: Dict[str, int] = {}
    for lvl in levels:
        counts[lvl] = counts.get(lvl, 0) + 1
    max_count = max(counts.values())
    winners = [k for k, v in counts.items() if v == max_count]
    ordering = {lvl: i for i, lvl in enumerate(TARGET_LEVELS)}
    winners.sort(key=lambda x: ordering.get(x, 999))
    return winners[0], counts

def normalize_label(label: str) -> str:
    label = label.strip().upper()
    for lvl in TARGET_LEVELS:
        if lvl in label:
            return lvl
    mapping = {"A0": "A1", "C2+": "C2"}
    return mapping.get(label, label)

def predict_cefr(pipes: List, text: str) -> Tuple[str, Dict[str, int]]:
    levels = []
    for p in pipes:
        out = p(text, truncation=True)
        if isinstance(out, list) and len(out) > 0 and "label" in out[0]:
            levels.append(normalize_label(out[0]["label"]))
        else:
            levels.append("B1")
    return majority_vote(levels)

def make_generator(model_id: str):
    gen = pipeline(
        task="text-generation",
        model=model_id,
        device=DEFAULT_DEVICE,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
    )
    return gen

def generate_text(gen, prompt: str, max_new_tokens: int = 256) -> str:
    out = gen(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1,
        eos_token_id=None,
    )
    text = out[0]["generated_text"]
    if text.startswith(prompt):
        text = text[len(prompt):]
    return text.strip().split("\n\n")[0].strip()

def iterative_simplify(
    original_text: str,
    target_cefr: str,
    gen,
    labelers: List,
    st_model,
    sim_floor: float = DEFAULT_SIMILARITY_FLOOR,
    max_iters: int = DEFAULT_MAX_ITERS,
) -> SimplifyResult:
    prompt1 = build_prompt_first(original_text, target_cefr)
    simplified = generate_text(gen, prompt1)
    simplified= deescape_unicode_literals(simplified)

    pred, votes = predict_cefr(labelers, simplified)
    sim = util.cos_sim(
        st_model.encode([original_text], convert_to_tensor=True, normalize_embeddings=True),
        st_model.encode([simplified], convert_to_tensor=True, normalize_embeddings=True),
    ).item()

    iters = 1

    while pred != target_cefr and iters < max_iters:
        prompt2 = build_prompt_followup(simplified, original_text, target_cefr)
        candidate = generate_text(gen, prompt2)

        sim_new = util.cos_sim(
            st_model.encode([original_text], convert_to_tensor=True, normalize_embeddings=True),
            st_model.encode([candidate], convert_to_tensor=True, normalize_embeddings=True),
        ).item()

        if sim_new < sim_floor:
            break

        simplified = candidate
        sim = sim_new
        pred, votes = predict_cefr(labelers, simplified)
        iters += 1

    return SimplifyResult(
        original=original_text,
        simplified=simplified,
        predicted_level=pred,
        similarity=sim,
        iterations=iters,
        levels_vote=votes,
    )

def load_labelers():
    pipes = []
    for mid in LABELER_IDS:
        p = pipeline(task="text-classification", model=mid, device=DEFAULT_DEVICE)
        pipes.append(p)
    return pipes

COMMON_TEXT_KEYS = ["text", "sentence", "original", "src", "source", "input", "content"]

def extract_text(obj: Dict[str, Any], text_field_candidates: List[str]) -> str:
    for key in text_field_candidates + COMMON_TEXT_KEYS:
        if key in obj and isinstance(obj[key], str):
            return obj[key]
    # last resort: if there's exactly one string value, use it
    string_vals = [v for v in obj.values() if isinstance(v, str)]
    if len(string_vals) == 1:
        return string_vals[0]
    raise KeyError(f"Could not find a text field in object: {obj}")

def extract_text_id(obj: Dict[str, Any]) -> Any:
    # Prefer 'text_id' if present, then common id-like keys, else index will be used by caller
    for key in ["text_id", "id", "uid", "idx"]:
        if key in obj:
            return obj[key]
    return None

def extract_target_cefr(obj: Dict[str, Any]) -> str:
    """
    Try multiple common variants for a per-record CEFR target.
    Defaults to 'B1' if not found.
    """
    candidates = [
        "target", "target_cefr", "target_level", "targetCEFR", "targetCEFR_LEVEL",
        "cefr_target", "cefr", "desired_level"
    ]
    for k in candidates:
        if k in obj and isinstance(obj[k], str):
            lvl = obj[k].strip().upper()
            # normalize common noise
            lvl = lvl.replace("-", "").replace(" ", "")
            # allow 'A1', 'A2', ..., 'C2' anywhere in the string
            for valid in TARGET_LEVELS:
                if valid in lvl:
                    return valid
    return "B1"

def deescape_unicode_literals(s: str) -> str:
    """Turn literal \\u00e1 style escapes into real Unicode."""
    if not isinstance(s, str):
        return s
    # Fast path: only try when typical patterns appear
    if "\\u" not in s and "\\x" not in s:
        return s
    try:
        # Decode common escapes, including \uXXXX
        return codecs.decode(s, "unicode_escape")
    except Exception:
        # Fallback: only replace \uXXXX patterns
        return re.sub(r'\\u([0-9a-fA-F]{4})',
                      lambda m: chr(int(m.group(1), 16)),
                      s)

def main():
    # ---- Init models ----
    print(f"[Init] Loading generator: {DEFAULT_GENERATOR}")
    gen = make_generator(DEFAULT_GENERATOR)

    print("[Init] Loading CEFR labelers (3 pipelines)...")
    labelers = load_labelers()

    print(f"[Init] Loading embedder: {DEFAULT_EMBEDDER}")
    st_model = SentenceTransformer(DEFAULT_EMBEDDER, device=f"cuda:{DEFAULT_DEVICE}" if DEFAULT_DEVICE >= 0 else "cpu")

    # ---- Read input (JSONL only, fixed name) ----
    items: List[Tuple[Any, str, Dict[str, Any], str]] = []  # (text_id, text, obj, target_cefr)
    in_path = Path(INPUT_PATH)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=0):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text_id = extract_text_id(obj)
            if text_id is None:
                text_id = i
            # Try to extract text; prefer explicit 'text'/'sentence', but be robust
            text = extract_text(obj, text_field_candidates=["text", "sentence"])
            target_cefr = extract_target_cefr(obj)
            items.append((text_id, text, obj, target_cefr))

    # ---- Process ----
    outputs: List[Tuple[Any, str]] = []  # (text_id, simplified_sentence)
    for text_id, text, obj, target in items:
        print(f"[{text_id}] Simplifying to target {target} ...")
        res = iterative_simplify(
            original_text=text,
            target_cefr=target,
            gen=gen,
            labelers=labelers,
            st_model=st_model,
            sim_floor=DEFAULT_SIMILARITY_FLOOR,
            max_iters=DEFAULT_MAX_ITERS,
        )
        outputs.append((text_id, res.simplified))

    # ---- Write output (JSONL with exactly two keys) ----
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for tid, simplified in outputs:
            out_obj = {
                "text_id": tid,
                "simplified": simplified
            }
            f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    print(f"[Done] Wrote JSONL to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
