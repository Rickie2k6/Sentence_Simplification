#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CEFR-targeted sentence simplifier with LLM-first + ModernBERT/SBERT pipeline.

Pipeline (high level)
---------------------
1) **LLM-first attempt** (exact prompt):
   Prompt:
     Simplify the following text to CEFR level {target_cefr}.
     Keep meaning, make it natural and readable. Return only the simplified text.

   If the LLM output passes CEFR ≤ target and semantic checks, accept immediately.
   Otherwise, the LLM output is added as a candidate to be scored.

2) **Rule-based + verification** (existing):
   - Three ModernBERT CEFR classifiers (majority voting)
   - SBERT cosine similarity floors
   - Lightweight SARI vs reference
   - Candidate search with increasingly strong edits
   - Reconciliation retries with relaxed thresholds
   - Final nearest-level fill and conservative fallback
   - Sorted JSONL output {"text_id": "...", "simplified": "..."}

Run:
  python New_Sentence_Simplification_with_CEFR_LEVEL_VERIFICATION.py --input tsar2025_trialdata.jsonl --output Trial_Simplification.jsonl \
    [--llm-provider openai|ollama|none] [--llm-model gpt-4o-mini] [--llm-timeout 60] [--no-llm]
"""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable
from collections import Counter, defaultdict

import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util as sbert_util

# --------------------------- LLM utilities -------------------------------

# The used prompt
def build_prompt(original_text: str, target_cefr: str) -> str:
    return (
        f"Simplify the following text to CEFR level {target_cefr}."
        f" Keep meaning, make it natural and readable. "
        f"Return only the simplified text.\n\nText:\n{original_text}"
    )

def _openai_chat(prompt: str, model: str, api_key: Optional[str], timeout: float = 60.0) -> str:
    if not api_key:
        return ""
    try:
        import requests  # lazy import for environments without requests preinstalled
    except Exception:
        return ""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model or "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant for CEFR-controlled text simplification."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 512,
    }
    try:
        r = requests.post(url, headers=headers, json=body, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
    except Exception:
        return ""

def _ollama_generate(prompt: str, model: str, host: Optional[str] = None, timeout: float = 60.0) -> str:
    try:
        import requests
    except Exception:
        return ""
    host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
    url = f"{host}/api/generate"
    body = {"model": model or "mistral", "prompt": prompt, "stream": False, "options": {"temperature": 0.2}}
    try:
        r = requests.post(url, json=body, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "") or ""
    except Exception:
        return ""

def call_llm(provider: str, model: str, prompt: str, timeout: float = 60.0) -> str:
    provider = (provider or "openai").lower().strip()
    if provider == "openai":
        return _openai_chat(prompt, model=model, api_key=os.environ.get("OPENAI_API_KEY"), timeout=timeout)
    if provider == "ollama":
        return _ollama_generate(prompt, model=model, host=os.environ.get("OLLAMA_HOST"), timeout=timeout)
    return ""

def clean_response(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t).rstrip("`").strip()
    if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
        t = t[1:-1].strip()
    return " ".join(t.split())

# ============================ CEFR & models ==============================
VALID_LEVELS = ("A1","A2","B1","B2","C1","C2")
LEVEL_INDEX = {lvl:i for i,lvl in enumerate(VALID_LEVELS)}

def _normalize_cefr_label(raw: str) -> str:
    u = raw.strip().upper()
    for lvl in VALID_LEVELS:
        if lvl in u:
            return lvl
    return "C2"

# Three ModernBERT heads
_cefr1 = pipeline("text-classification", model="AbdullahBarayan/ModernBERT-base-doc_en-Cefr")
_cefr2 = pipeline("text-classification", model="AbdullahBarayan/ModernBERT-base-doc_sent_en-Cefr")
_cefr3 = pipeline("text-classification", model="AbdullahBarayan/ModernBERT-base-reference_AllLang2-Cefr2")
_CEFR_MODELS = (_cefr1, _cefr2, _cefr3)

#This method predicts the CEFR level of a given text using majority voting across three ModernBERT models.
def predict_cefr(text: str) -> str:
    """
    Majority voting across the 3 heads.
    Tie-breaks: sum(score) desc, max(score) desc, prefer simpler (A1..C2).
    """
    votes = []
    score_sum = defaultdict(float)
    score_max = defaultdict(float)
    for m in _CEFR_MODELS:
        preds = m(text)
        if not preds:
            continue
        top = preds[0]
        lab = _normalize_cefr_label(top["label"])
        sc  = float(top.get("score", 0.0))
        votes.append(lab)
        score_sum[lab] += sc
        score_max[lab] = max(score_max[lab], sc)
    if not votes:
        return "C2"
    counts = Counter(votes)
    def key(lab):
        return (counts[lab], score_sum[lab], score_max[lab], -LEVEL_INDEX[lab])
    return sorted(set(votes), key=key, reverse=True)[0]

def hits_target_or_below(pred: str, target: str) -> bool:
    return LEVEL_INDEX[pred] <= LEVEL_INDEX[target]

# ===================== Semantics (SBERT) & lightweight SARI ==============
_SBERT = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def cosine_sim(a: Optional[str], b: Optional[str]) -> float:
    if not a or not b:
        return 0.0
    emb = _SBERT.encode([a, b], convert_to_tensor=True, normalize_embeddings=True)
    return float(sbert_util.cos_sim(emb[0], emb[1]).item())

def _ngram_counts(tokens, n):
    return {tuple(tokens[i:i+n]): 1 for i in range(len(tokens)-n+1)}

def _sari_1ref(sys: str, src: str, ref: Optional[str]) -> float:
    if not ref:
        return 0.0
    def toks(s): return s.lower().split()
    sys_t, src_t, ref_t = toks(sys), toks(src), toks(ref)
    def f1(n):
        sys_n = _ngram_counts(sys_t, n); src_n = _ngram_counts(src_t, n); ref_n = _ngram_counts(ref_t, n)
        keep_sys = set(sys_n) & set(src_n); keep_ref = set(ref_n) & set(src_n)
        if not keep_sys and not keep_ref: keepF = 1.0
        else:
            kp_prec = len(keep_sys & keep_ref) / max(1, len(keep_sys))
            kp_rec  = len(keep_sys & keep_ref) / max(1, len(keep_ref))
            keepF = 0 if (kp_prec+kp_rec)==0 else 2*kp_prec*kp_rec/(kp_prec+kp_rec)
        add_sys = set(sys_n) - set(src_n); add_ref = set(ref_n) - set(src_n)
        if not add_sys and not add_ref: addF = 1.0
        else:
            ad_prec = len(add_sys & add_ref) / max(1, len(add_sys))
            ad_rec  = len(add_sys & add_ref) / max(1, len(add_ref))
            addF = 0 if (ad_prec+ad_rec)==0 else 2*ad_prec*ad_rec/(ad_prec+ad_rec)
        del_src_minus_ref = set(src_n) - set(ref_n); del_src_minus_sys = set(src_n) - set(sys_n)
        if not del_src_minus_ref and not del_src_minus_sys: delF = 1.0
        else:
            dl_prec = len(del_src_minus_sys & del_src_minus_ref) / max(1, len(del_src_minus_sys))
            dl_rec  = len(del_src_minus_sys & del_src_minus_ref) / max(1, len(del_src_minus_ref))
            delF = 0 if (dl_prec+dl_rec)==0 else 2*dl_prec*dl_rec/(dl_prec+dl_rec)
        return (keepF + addF + delF) / 3.0
    return float(np.mean([f1(n) for n in (1,2,3,4)]))

# =========================== Rule-based candidates =======================
_REPLACEMENTS = [
    ("might","can"), ("could","can"), ("would","will"),
    ("however","but"), ("therefore","so"), ("in order to","to"),
    ("utilize","use"), ("approximately","about"),
    ("individuals","people"), ("demonstrate","show"),
    ("numerous","many"), ("subsequently","then"),
    ("comprehend","understand"), ("assist","help"),
    ("prior to","before"), ("subsequent to","after"),
]
REL_STARTS = r"(which|that|who|whom|whose|where|when|although|though|because|since|while|whereas|however)"
ADJ_ADV = r"\b(very|extremely|significantly|particularly|highly|relatively|somewhat|considerably)\b"

def replace_words(t: str) -> str:
    for k, v in _REPLACEMENTS:
        t = re.sub(rf"\b{re.escape(k)}\b", v, t, flags=re.IGNORECASE)
    return t

def numbers_to_plain(t: str) -> str:
    return re.sub(r'(?<=\d),(?=\d)', "", t)

def simplify_numbers_units(t: str) -> str:
    t = numbers_to_plain(t)
    t = re.sub(r"\bmetres?\b", "meters", t, flags=re.I)
    return t

def keep_shortest_clause(t: str) -> str:
    parts = re.split(r'[;:–—]|(?<=[.!?])\s+', t.strip())
    parts = [p.strip() for p in parts if p.strip()]
    return min(parts, key=len) if parts else t.strip()

def strip_relative_clauses(t: str) -> str:
    t = re.sub(r",\s*"+REL_STARTS+r".*?(,|\.|;|$)", ". ", t, flags=re.IGNORECASE)
    t = re.sub(r"\b"+REL_STARTS+r"\b.*?(,|\.|;|$)", " ", t, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", t).strip()

def strip_modifiers(t: str) -> str:
    return re.sub(ADJ_ADV, "", t, flags=re.IGNORECASE)

def trim_to_limit(t: str, limit: int) -> str:
    words = t.split()
    return t if len(words) <= limit else " ".join(words[:limit]).rstrip(",;:") + "."

def sentence_split(t: str) -> List[str]:
    parts = re.split(r'(?<=[.!?])\s+', t.strip())
    return [p.strip() for p in parts if p.strip()]

def basic_candidates(text: str, step_idx: int) -> List[str]:
    base = text.strip()
    lim = max(8, 28 - 2*step_idx)
    cands = [
        replace_words(base),
        simplify_numbers_units(base),
        keep_shortest_clause(base),
        strip_relative_clauses(base),
        trim_to_limit(replace_words(base), lim),
        trim_to_limit(simplify_numbers_units(base), lim),
        trim_to_limit(strip_relative_clauses(base), lim),
        trim_to_limit(keep_shortest_clause(replace_words(base)), lim),
    ]
    for s in sentence_split(strip_relative_clauses(base)):
        cands.append(trim_to_limit(replace_words(s), max(10, lim-4)))
    seen, out = set(), []
    for c in cands:
        c = c.strip().replace(" ,", ",").replace(" .", ".")
        if c and c not in seen:
            out.append(c); seen.add(c)
    return out

# ========================== LLM-first candidate ==========================
def llm_candidate(original: str, target_cefr: str, provider: str, model: str, timeout: float) -> str:
    prompt = build_prompt(original, target_cefr)
    raw = call_llm(provider=provider, model=model, prompt=prompt, timeout=timeout)
    return clean_response(raw)

# =============================== Search ==================================
def search_target_with_reference(
    original: str,
    target_cefr: str,
    reference: Optional[str],
    sim_floor: float = 0.88,
    max_steps: int = 8,
    w_hit: float = 10.0,
    w_ref: float = 2.5,
    w_sari: float = 2.0,
    w_orig: float = 0.5,
    # LLM controls
    llm_provider: str = "openai",
    llm_model: str = "gpt-4o-mini",
    llm_timeout: float = 60.0,
    use_llm: bool = True,
) -> Optional[str]:
    """Return a simplified text that meets target-or-below, else None."""
    if not original:
        return None
    orig = original.strip()
    target_cefr = target_cefr.strip().upper()
    assert target_cefr in VALID_LEVELS

    # Quick accept if already at/below level
    pred0 = predict_cefr(orig)
    if hits_target_or_below(pred0, target_cefr):
        return orig

    # ---------- LLM-first attempt ----------
    first_llm_text = ""
    if use_llm:
        first_llm_text = llm_candidate(orig, target_cefr, provider=llm_provider, model=llm_model, timeout=llm_timeout)
        if first_llm_text:
            sim_o = cosine_sim(orig, first_llm_text)
            pred  = predict_cefr(first_llm_text)
            if sim_o >= sim_floor and hits_target_or_below(pred, target_cefr):
                if not reference or cosine_sim(reference, first_llm_text) >= 0.95:
                    return first_llm_text  # immediate accept

    best_text, best_score = None, -math.inf
    cur_seed = orig

    for step in range(1, max_steps + 1):
        # Rule-based candidates
        cands = basic_candidates(cur_seed, step_idx=step)
        # Include the LLM output once (only at the first step) if present
        if use_llm and step == 1 and first_llm_text:
            cands = [first_llm_text] + cands

        step_best_text, step_best_score = None, -math.inf

        for cand in cands:
            sim_o = cosine_sim(orig, cand)
            if sim_o < sim_floor:
                continue

            pred = predict_cefr(cand)

            # If above target, simplify further
            if LEVEL_INDEX[pred] > LEVEL_INDEX[target_cefr]:
                cand2 = trim_to_limit(strip_modifiers(cand), max(8, 24 - step*2))
                sim2 = cosine_sim(orig, cand2)
                if sim2 >= sim_floor:
                    cand = cand2
                    pred = predict_cefr(cand)
                    sim_o = sim2

            sim_r = cosine_sim(reference, cand) if reference else 0.0
            sari  = _sari_1ref(cand, orig, reference) if reference else 0.0

            base = w_ref*sim_r + w_sari*sari + w_orig*sim_o
            score = (w_hit + base) if hits_target_or_below(pred, target_cefr) else base

            if score > step_best_score:
                step_best_text, step_best_score = cand, score

        if step_best_text:
            cur_seed = step_best_text
            if step_best_score > best_score:
                best_text, best_score = step_best_text, step_best_score
            if hits_target_or_below(predict_cefr(step_best_text), target_cefr) and \
               (not reference or cosine_sim(reference, step_best_text) >= 0.95):
                return step_best_text

    if best_text and hits_target_or_below(predict_cefr(best_text), target_cefr):
        return best_text
    return None

# ======================== Nearest-level fill (final) =====================
def nearest_level_fill(row: Dict, sim_floor: float, max_steps: int, llm_provider: str, llm_model: str, llm_timeout: float, use_llm: bool) -> Optional[str]:
    """
    Return a candidate whose predicted CEFR is CLOSEST to target (by distance),
    while preserving meaning (relaxed floor). Returns only the text.
    """
    original = row.get("original", "") or ""
    target = (row.get("target_cefr") or row.get("target_cefr_level") or "").strip().upper()
    reference = row.get("reference")
    if not original or target not in VALID_LEVELS:
        return None

    orig = original.strip()
    best_text = None
    best_key  = None

    # Optionally try LLM once here as well
    if use_llm:
        llm_text = llm_candidate(orig, target, provider=llm_provider, model=llm_model, timeout=llm_timeout)
        if llm_text:
            pred = predict_cefr(llm_text)
            dist = abs(LEVEL_INDEX[pred] - LEVEL_INDEX[target])
            sim_o = cosine_sim(orig, llm_text)
            sim_r = cosine_sim(reference, llm_text) if reference else 0.0
            sari  = _sari_1ref(llm_text, orig, reference) if reference else 0.0
            best_text = llm_text
            best_key  = (-dist, sim_r, sari, sim_o)

    for step in range(1, max_steps + 1):
        seed = orig if best_text is None else best_text
        for cand in basic_candidates(seed, step_idx=step):
            sim_o = cosine_sim(orig, cand)
            if sim_o < sim_floor:
                continue
            pred = predict_cefr(cand)
            dist = abs(LEVEL_INDEX[pred] - LEVEL_INDEX[target])
            sim_r = cosine_sim(reference, cand) if reference else 0.0
            sari  = _sari_1ref(cand, orig, reference) if reference else 0.0
            key = (-dist, sim_r, sari, sim_o)  # better = larger key
            if best_key is None or key > best_key:
                best_key = key
                best_text = cand

    return best_text

# ================================ I/O ====================================
def read_jsonl(path: Path) -> List[Dict]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def write_jsonl_records(path: Path, recs: Iterable[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps({"text_id": r["text_id"], "simplified": r["simplified"]}, ensure_ascii=False) + "\n")

# ============================== Batch pass ===============================
def simplify_batch(items: List[Dict], sim_floor: float, max_steps: int, debug: bool,
                   llm_provider: str, llm_model: str, llm_timeout: float, use_llm: bool) -> List[Dict]:
    out, reasons = [], []
    for row in items:
        tid = row.get("text_id")
        original = row.get("original", "")
        target = (row.get("target_cefr") or row.get("target_cefr_level") or "").strip().upper()
        reference = row.get("reference")
        if not tid or not original or target not in VALID_LEVELS:
            continue

        final_text = search_target_with_reference(
            original=original, target_cefr=target, reference=reference,
            sim_floor=sim_floor, max_steps=max_steps,
            llm_provider=llm_provider, llm_model=llm_model, llm_timeout=llm_timeout, use_llm=use_llm
        )

        # Last-resort: consider reference if valid and at/below target
        if final_text is None and reference:
            if cosine_sim(original, reference) >= max(0.75, sim_floor - 0.05) and hits_target_or_below(predict_cefr(reference), target):
                final_text = reference

        if final_text is not None:
            out.append({"text_id": tid, "simplified": final_text})
        else:
            if debug:
                try_pred = predict_cefr(trim_to_limit(strip_relative_clauses(original), 18))
                reasons.append((tid, f"no_match; try_pred={try_pred}"))

    if debug and reasons:
        print("[DEBUG] first 15 failures:")
        for tid, msg in reasons[:15]:
            print(f"  {tid}: {msg}")
    return out

# ============================== Sorting helper ===========================
CEFR_ORDER = {lvl:i for i,lvl in enumerate(("A1","A2","B1","B2","C1","C2"))}

def sort_key_text_id(text_id: str) -> Tuple[int, int]:
    m = re.match(r"^(\d+)-([a-c][12])$", text_id, re.IGNORECASE)
    if not m:
        return (10**9, 10**9)
    group = int(m.group(1))
    cefr = (m.group(2)[0] + m.group(2)[1]).upper()
    return (group, CEFR_ORDER.get(cefr, 10**6))

# ===================== Reconcile loop + Final nearest fill ===============
def process_file_with_reconcile(
    in_path: Path,
    out_path: Path,
    similarity_floor: float = 0.88,
    max_steps: int = 8,
    max_retries: int = 6,
    floor_step: float = 0.03,
    steps_step: int = 6,
    debug: bool = True,
    # LLM controls
    llm_provider: str = "openai",
    llm_model: str = "gpt-4o-mini",
    llm_timeout: float = 60.0,
    use_llm: bool = True,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = read_jsonl(in_path)

    # First pass
    current_records: Dict[str, Dict] = {}
    first_out = simplify_batch(
        data, sim_floor=similarity_floor, max_steps=max_steps, debug=debug,
        llm_provider=llm_provider, llm_model=llm_model, llm_timeout=llm_timeout, use_llm=use_llm
    )
    for r in first_out:
        current_records[r["text_id"]] = {"text_id": r["text_id"], "simplified": r["simplified"]}

    input_ids = {row.get("text_id") for row in data if row.get("text_id")}
    output_ids = set(current_records.keys())

    retry, cur_floor, cur_steps = 0, similarity_floor, max_steps
    while retry < max_retries and output_ids != input_ids:
        missing_ids = sorted(list(input_ids - output_ids))
        missing_items = [row for row in data if row.get("text_id") in missing_ids]

        cur_floor = max(0.75, cur_floor - floor_step)
        cur_steps = min(36, cur_steps + steps_step)

        if debug:
            print(f"[Retry {retry+1}] missing={len(missing_items)}  floor={cur_floor:.2f}  steps={cur_steps}")

        retry_out = simplify_batch(
            missing_items, sim_floor=cur_floor, max_steps=cur_steps, debug=debug,
            llm_provider=llm_provider, llm_model=llm_model, llm_timeout=llm_timeout, use_llm=use_llm
        )
        for r in retry_out:
            current_records[r["text_id"]] = {"text_id": r["text_id"], "simplified": r["simplified"]}
        output_ids = set(current_records.keys())
        retry += 1

    # ---- Summary ----
    output_ids = set(current_records.keys())
    total_in, total_out = len(input_ids), len(output_ids)
    remaining_ids = sorted(list(input_ids - output_ids))
    print(f"[Summary] Input objects: {total_in} | Output objects: {total_out}")
    if remaining_ids:
        print(f"[Info] Proceeding to final nearest-level fill for {len(remaining_ids)} remaining item(s).")

    # ---- Final nearest-level fill ----
    if remaining_ids:
        final_floor = max(0.72, cur_floor - 0.03)
        final_steps = min(40, cur_steps + 4)
        id_to_row = {row["text_id"]: row for row in data if row.get("text_id")}
        for miss_id in remaining_ids:
            row = id_to_row.get(miss_id)
            if not row:
                continue
            best_text = nearest_level_fill(
                row, sim_floor=final_floor, max_steps=final_steps,
                llm_provider=llm_provider, llm_model=llm_model, llm_timeout=llm_timeout, use_llm=use_llm
            )
            if best_text:
                current_records[miss_id] = {"text_id": miss_id, "simplified": best_text}
            else:
                orig = (row.get("original") or "").strip()
                if orig:
                    fallback = trim_to_limit(strip_relative_clauses(replace_words(orig)), 28)
                    current_records[miss_id] = {"text_id": miss_id, "simplified": fallback}

        output_ids = set(current_records.keys())
        total_out = len(output_ids)
        still_missing = sorted(list(input_ids - output_ids))
        print(f"[Final Fill] Output objects: {total_out} (missing {len(still_missing)})")
        if still_missing and debug:
            print("[Final Fill] Still missing:", ", ".join(still_missing[:40]))

    # ---- Sort and write output ----
    sorted_items = sorted(current_records.values(), key=lambda r: sort_key_text_id(r["text_id"]))
    write_jsonl_records(out_path, sorted_items)

# ================================= CLI ===================================
def main():
    ap = argparse.ArgumentParser(description="LLM-first CEFR simplifier + ModernBERT/SBERT reconciliation pipeline.")
    ap.add_argument("--input", required=True, help="Path to input JSONL")
    ap.add_argument("--output", required=True, help="Path to output JSONL")
    ap.add_argument("--similarity_floor", type=float, default=0.88)
    ap.add_argument("--max_steps", type=int, default=8)
    ap.add_argument("--max_retries", type=int, default=6)
    ap.add_argument("--floor_step", type=float, default=0.03)
    ap.add_argument("--steps_step", type=int, default=6)
    ap.add_argument("--no_debug", action="store_true")

    # New LLM options
    ap.add_argument("--llm-provider", default=os.environ.get("LLM_PROVIDER", "openai"), choices=["openai", "ollama", "none"], help="LLM provider")
    ap.add_argument("--llm-model", default=os.environ.get("LLM_MODEL", "gpt-4o-mini"), help="LLM model (e.g., gpt-4o-mini, mistral, llama3)")
    ap.add_argument("--llm-timeout", type=float, default=float(os.environ.get("LLM_TIMEOUT", "60")), help="LLM request timeout (sec)")
    ap.add_argument("--no-llm", action="store_true", help="Disable LLM-first attempt entirely")

    args = ap.parse_args()

    use_llm = not args.no_llm and args.llm_provider.lower() != "none"
    if not args.no_debug:
        print(f"[cfg] LLM provider={args.llm_provider} model={args.llm_model} use_llm={use_llm}")

    process_file_with_reconcile(
        in_path=Path(args.input),
        out_path=Path(args.output),
        similarity_floor=args.similarity_floor,
        max_steps=args.max_steps,
        max_retries=args.max_retries,
        floor_step=args.floor_step,
        steps_step=args.steps_step,
        debug=not args.no_debug,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_timeout=args.llm_timeout,
        use_llm=use_llm,
    )

if __name__ == "__main__":
    main()
