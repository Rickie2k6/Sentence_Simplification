#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Predict CEFR level for each JSON object’s original sentence and output
the same objects with an added field: `original_level`.

Input formats supported:
- JSONL: one object per line, e.g. {"text_id": "01", "original": "...."}
- JSON array file: [ {"text_id": "...", "original": "..."}, ... ]

Accepted original-sentence keys (first match wins):
- "original", "original_sentence", "sentence", "text"

Usage:
  python cefr_original_level.py --in input.jsonl --out output.jsonl

Requires: transformers>=4.x, torch (or another backend supported by transformers)
"""

import argparse, io, json, sys
from collections import Counter, defaultdict
from typing import List, Dict, Any

from transformers import pipeline

# ----------------------------- Config -------------------------------- #

MODEL_NAMES = [
    "AbdullahBarayan/ModernBERT-base-doc_en-Cefr",
    "AbdullahBarayan/ModernBERT-base-doc_sent_en-Cefr",
    "AbdullahBarayan/ModernBERT-base-reference_AllLang2-Cefr2",
]

# Order for tie-breaking
CEFR_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2"]
CEFR_RANK = {lbl: i for i, lbl in enumerate(CEFR_ORDER)}

# Keys we’ll try for the original sentence
ORIG_KEYS = ["original", "original_sentence", "sentence", "text"]

# --------------------------- Utilities -------------------------------- #

def load_items(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file or a JSON array file into a list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        start = f.read(1)
        f.seek(0)
        if start.strip().startswith("["):  # JSON array
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Top-level JSON is not a list.")
            return data
        else:  # JSONL
            items = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
            return items

def save_items_jsonl(items: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def get_original_text(obj: Dict[str, Any]) -> str:
    for k in ORIG_KEYS:
        if k in obj and isinstance(obj[k], str):
            return obj[k]
    raise KeyError(
        f"Could not find the original sentence in any of keys {ORIG_KEYS} for object with keys {list(obj.keys())}"
    )

# ------------------------- Voting logic -------------------------------- #

def vote_label(preds: List[Dict[str, float]]) -> str:
    """
    preds: list of {"label": "B1", "score": 0.92} from 3 models (top-1 each).
    Rule:
      1) Majority vote by label.
      2) Tie -> pick label with highest mean confidence among tied labels.
      3) Still tie -> pick by CEFR order (lower rank first).
    """
    counts = Counter(p["label"] for p in preds)
    most = counts.most_common()
    max_votes = most[0][1]
    tied = [lbl for lbl, c in most if c == max_votes]

    if len(tied) == 1:
        return tied[0]

    # mean confidence among tied labels
    mean_conf = defaultdict(list)
    for p in preds:
        if p["label"] in tied:
            mean_conf[p["label"]].append(p["score"])
    best_lbl, best_mean = None, -1.0
    for lbl, arr in mean_conf.items():
        m = sum(arr) / max(1, len(arr))
        if m > best_mean:
            best_lbl, best_mean = lbl, m

    # if another tie (rare), use CEFR rank
    tied2 = [lbl for lbl, arr in mean_conf.items() if abs(sum(arr)/len(arr) - best_mean) < 1e-12]
    if len(tied2) == 1:
        return best_lbl
    return sorted(tied2, key=lambda x: CEFR_RANK.get(x, 999))[0]

# --------------------------- Main flow --------------------------------- #

def build_pipelines(device: int = -1):
    """Create three text-classification pipelines (top-1)."""
    models = []
    for m in MODEL_NAMES:
        clf = pipeline(task="text-classification", model=m, device=device)
        models.append(clf)
    return models

def predict_original_level(text: str, models) -> str:
    preds = []
    for mdl in models:
        out = mdl(text)
        # `pipeline` returns a list with one dict: [{'label': 'B1', 'score': 0.93}]
        top = out[0]
        preds.append({"label": top["label"], "score": float(top["score"])})
    return vote_label(preds)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Path to input JSONL or JSON array file")
    ap.add_argument("--out", dest="outp", required=True, help="Path to output JSONL")
    ap.add_argument("--device", type=int, default=-1, help="Transformers device id (-1=CPU, 0=CUDA:0, ...)")
    ap.add_argument("--orig_key", default=None, help="Force a specific key for original sentence (optional)")
    ap.add_argument("--level_key", default="original_level", help="Output field name for the level")
    args = ap.parse_args()

    items = load_items(args.inp)
    models = build_pipelines(device=args.device)

    used_key = args.orig_key
    out_items = []
    for obj in items:
        try:
            text = obj[used_key] if used_key else get_original_text(obj)
        except KeyError as e:
            print(f"[WARN] {e}", file=sys.stderr)
            continue

        level = predict_original_level(text, models)
        new_obj = dict(obj)
        new_obj[args.level_key] = level
        out_items.append(new_obj)

    # Keep the output as JSONL for robust downstream use
    save_items_jsonl(out_items, args.outp)
    print(f"Done. Wrote {len(out_items)} objects to: {args.outp}")

if __name__ == "__main__":
    main()
