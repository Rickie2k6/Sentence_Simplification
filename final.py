#!/usr/bin/env python3
import os
import json
import time
import argparse
from typing import Dict, Any, Optional
from openai import OpenAI, APIConnectionError, RateLimitError, BadRequestError

# ---------- CEFR descriptions (optional helper) ----------
cefr_descriptions = {
    "A1": "Can understand very short, simple texts a single phrase at a time, picking up familiar names, words and basic phrases and rereading as required.",
    "A2": "Can understand short, simple texts containing the highest frequency vocabulary, including a proportion of shared international vocabulary items.",
    "B1": "Can read straightforward factual texts on subjects related to their field of interest with a satisfactory level of comprehension.",
    "B2": "Can read with a large degree of independence, adapting style and speed of reading to different texts and purposes, and using appropriate reference sources selectively. Has a broad active reading vocabulary, but may experience some difficulty with low-frequency idioms.",
    "C1": "Can understand a wide variety of texts including literary writings, newspaper or magazine articles, and specialised academic or professional publications, provided there are opportunities for rereading and they have access to reference tools. Can understand in detail lengthy, complex texts, whether or not these relate to their own area of speciality, provided they can reread difficult sections.",
    "C2": "Can understand a wide range of long and complex texts, appreciating subtle distinctions of style and implicit as well as explicit meaning. Can understand virtually all types of texts including abstract, structurally complex, or highly colloquial literary and non-literary writings."
}

# ---------- Core simplify function using OpenAI v1 ----------
def simplify_sentence(
    client: OpenAI,
    model: str,
    target_cefr: str,
    original_text: str,
    max_retries: int = 5,
    backoff_base: float = 1.5
) -> str:
    """Call the chat.completions API with simple retry logic."""
    prompt = (
        f"Simplify the following text to CEFR level {target_cefr}."
        f" Keep meaning, make it natural and readable. "
        f"Return only the simplified text.\n\nText:\n{original_text}"
    )

    attempt = 0
    while True:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that simplifies English text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            return resp.choices[0].message.content.strip()

        except (RateLimitError, APIConnectionError) as e:
            attempt += 1
            if attempt > max_retries:
                raise RuntimeError(f"Failed after {max_retries} retries: {e}") from e
            sleep_s = backoff_base ** attempt
            time.sleep(sleep_s)

        except BadRequestError as e:
            # Usually unrecoverable (e.g., invalid model, content issues)
            raise RuntimeError(f"Bad request to OpenAI API: {e}") from e

        except Exception as e:
            # Catch-all to avoid crashing the whole batch; re-raise for visibility
            raise

# ---------- CLI / main ----------
def main():
    parser = argparse.ArgumentParser(description="Batch text simplification to target CEFR levels.")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL path (one object per line).")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL path.")
    parser.add_argument("--model", "-m", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                        help="OpenAI model name (default: gpt-4o-mini; or set OPENAI_MODEL env var).")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. In your terminal run:\n"
            '  export OPENAI_API_KEY="sk-..."\n'
            "and re-run this script."
        )

    client = OpenAI(api_key=api_key)

    total = 0
    ok = 0
    with open(args.input, "r", encoding="utf-8") as infile, \
         open(args.output, "w", encoding="utf-8") as outfile:

        for line in infile:
            total += 1
            line = line.strip()
            if not line:
                continue

            try:
                data: Dict[str, Any] = json.loads(line)
                text_id: str = data["text_id"]
                target_cefr: str = data["target_cefr"]
                original: str = data["original"]

                simplified = simplify_sentence(client, args.model, target_cefr, original)

                out = {"text_id": text_id, "simplified_sentence": simplified}
                outfile.write(json.dumps(out, ensure_ascii=False) + "\n")
                ok += 1

            except Exception as e:
                # Write a minimal error record so you can review failures later
                err_out = {
                    "error": str(e),
                    "raw_line": line[:5000]
                }
                outfile.write(json.dumps(err_out, ensure_ascii=False) + "\n")

    print(f"Done. Processed: {total}, Succeeded: {ok}, Failed: {total - ok}")
    print(f"Output saved to: {args.output}")

if __name__ == "__main__":
    main()

