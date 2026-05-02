#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assign taxonomy labels to split AI-review opinions produced by the evaluation pipeline.

Input JSONL
-----------
- Records produced by `src/evaluation/_2_split.py`, containing:
  - `sentence_texts`
  - `opinions`

Processing
----------
1) Reconstruct each opinion by concatenating the referenced `sentence_texts`.
2) Query the classify model for taxonomy labels.
3) Retry malformed or low-quality local vLLM responses with different presence penalties.

Output JSONL
------------
- Same records with an added `category` field:
  - `category[i]` is the predicted label list for `opinions[i]`

Notes
-----
- When `--base_url` points to localhost, the script waits for the vLLM server before
  submitting requests.
- The current evaluation pipeline only supports the AI-review classification path;
  there is no separate `--human` mode in this script.
"""

import argparse
import json
import re
from typing import List, Dict, Tuple, Optional, Set, Any
from collections import defaultdict
import os
from urllib.parse import urlparse
import unicodedata

import sys
from pathlib import Path

# Project root directory (two levels up from src/evaluation/)
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
# Ensure we can import `src.benchmark.*` when running this file directly.
sys.path.insert(0, PROJECT_ROOT)
from src.utils import (
    LLMClient,
    classify_categories,
    wait_until_ready,
)
from src.benchmark.prompt_registry import classify_reviews_prompt

client: Optional[LLMClient] = None


def _extract_ints(x: Any, out: List[int]) -> None:
    if isinstance(x, int):
        out.append(x)
    elif isinstance(x, list):
        for it in x:
            _extract_ints(it, out)
    elif isinstance(x, dict):
        for v in x.values():
            _extract_ints(v, out)


def _label_to_fuzzy_regex(label: str) -> str:
    """
    Compile a category label into a more robust matching regex:
    - Case-insensitive (controlled by re.I at compile time)
    - Inside the label, spaces/underscores/hyphens are interchangeable and repeatable
      e.g. "data_science" matches "data science" / "data-science" / "data__science"
    - Use non-alphanumeric boundaries on both sides to avoid matching 'sport' in 'esports'
    """
    s = (label or "").strip()
    if not s:
        return r"(?!x)x"  # Never match

    parts = [p for p in re.split(r"[\s_\-]+", s) if p]
    if not parts:
        return r"(?!x)x"

    body = r"[\s_\-]+".join(re.escape(p) for p in parts)
    return rf"(?<![0-9A-Za-z]){body}(?![0-9A-Za-z])"


def parse_labels_from_llm(raw_response: str, allowed_categories: Set[str]) -> Tuple[Set[str], bool]:
    """
    Convert raw model output text into a predicted label set P (set[str]) and return
    a flag indicating whether the output format is acceptable.

    - Only use the text after </think>; if </think> is missing, treat it as empty prediction and format OK
    - Do not rely on comma/semicolon/newline splitting; directly match allowed_categories in the full text
    - If the same label appears multiple times, treat as format error

    Returns: (predicted_labels, is_format_ok)
    """
    if raw_response is None:
        return set(), True  # Empty prediction is not additionally treated as "format error"
    cleaned = raw_response.split("</think>")[-1].strip()
    if cleaned == "":
        return set(), True  # Empty prediction is not additionally treated as "format error"

    # Normalize common dash/minus variants and map "_" to "-"
    cleaned = unicodedata.normalize("NFKC", cleaned)
    dash_map = str.maketrans({
        "\u2010": "-",  # hyphen
        "\u2011": "-",  # non-breaking hyphen
        "\u2012": "-",  # figure dash
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2212": "-",  # minus sign
        "\uFE63": "-",  # small hyphen-minus
        "\uFF0D": "-",  # fullwidth hyphen-minus
        "_": "-",       # underscore
    })
    cleaned = cleaned.translate(dash_map)

    # 1) Directly match allowed_categories (sort by length descending: longer labels first)
    cats_sorted = sorted(allowed_categories, key=len, reverse=True)

    # Use named groups to recover which category matched (no normalize/suffix correction needed).
    opinion_to_label = {}
    opinion_parts = []
    for i, cat in enumerate(cats_sorted):
        g = f"L{i}"
        opinion_to_label[g] = cat
        opinion_parts.append(rf"(?P<{g}>{_label_to_fuzzy_regex(cat)})")

    cat_re = re.compile("|".join(opinion_parts), flags=re.IGNORECASE)

    counts = defaultdict(int)
    for m in cat_re.finditer(cleaned):
        label = opinion_to_label.get(m.lastgroup)
        if not label:
            continue
        counts[label] += 1

    parsed = {lab for lab, c in counts.items() if c >= 1}
    has_duplicates = any(c > 1 for c in counts.values())

    is_format_ok = not has_duplicates
    return sorted(list(parsed)), is_format_ok


def _is_na_labelset(s: set) -> bool:
    """Treat empty set or a set that only contains N/A as NA."""
    if not s:
        return True
    return (len(s) == 1 and ("N/A" in s or "n/a" in {x.lower() for x in s}))


def classify_opinions(args):
    if not client:
        raise RuntimeError("LLMClient has not been initialized.")

    input_jsonl = args.input_jsonl
    if not os.path.exists(input_jsonl):
        raise FileNotFoundError(
            f"Input file not found: {input_jsonl}."
        )
    output_path = args.output_jsonl
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    sys_prompt, main_prompt_body = classify_reviews_prompt()
    allowed_categories, label_defs = classify_categories()

    # ---- Phase 1: read input and directly build flat tasks
    records: List[Dict[str, Any]] = []
    flat_tasks: List[Tuple[Tuple[int, int], str]] = []
    print(f"Reading {input_jsonl} and preparing flat tasks...")

    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            parts = obj.get("sentence_texts", [])
            opinions = obj.get("opinions", [])
            pi = len(records)
            records.append({
                "obj": obj,
                "opinion_count": len(opinions),
            })

            for ti, opinion in enumerate(opinions):
                comment = "\n".join([parts[i] for i in opinion if 0 <= i < len(parts)]).strip()
                prompt_text = main_prompt_body.strip() + "\n\n\nReviewer Comment:\n" + comment
                flat_tasks.append(((pi, ti), prompt_text))

    print(f"Prepared {len(flat_tasks)} tasks for {len(records)} records.")

    # ---- Phase 2: batch get_result (collect all first-pass results, then handle retries)
    print(f"Collecting results and writing to {output_path}...")
    retry_count = 0

    result_map: Dict[Tuple[int, int], List[str]] = {}
    retry_candidates: Dict[Tuple[int, int], List[List[str]]] = defaultdict(list)
    pending_keys: List[Tuple[int, int]] = [key for key, _ in flat_tasks]
    retry_prompt_map: Dict[Tuple[int, int], str] = dict(flat_tasks)

    for round_idx, penalty in enumerate((0.0, 0.5, 1.0), start=1):
        if penalty > 0.0 and not pending_keys:
            print(f"[Round {round_idx}] presence_penalty={penalty}: no pending tasks, skip.")
            continue
        print(f"[Round {round_idx}] presence_penalty={penalty}, processing {len(pending_keys)} tasks...")

        round_task_map: Dict[Tuple[int, int], str] = {}
        for key in pending_keys:
            task_id = client.submit_task(
                prompt=retry_prompt_map[key],
                system_prompt=(
                    "You are given a reviewer comment. Your task is to classify the comment using the "
                    "following taxonomy. The output should only contain the sub-category name."
                ),
                temperature=0.6,
                presence_penalty=penalty,
                max_completion_tokens=12288,
                reasoning_effort=args.effort,
            )
            round_task_map[key] = task_id

        next_pending_keys: List[Tuple[int, int]] = []
        accepted_count = 0
        for key, rid in round_task_map.items():
            r = client.get_result(rid)
            content = r.get("content", "")
            labels, is_format_ok = parse_labels_from_llm(content, allowed_categories)
            is_retry_ok = content and is_format_ok and "</think>" in content
            labels_candidate = labels if labels else ["N/A"]
            if is_retry_ok:
                retry_candidates[key].append(labels_candidate)
                accepted_count += 1
            if (not is_retry_ok) or _is_na_labelset(labels_candidate) or len(labels_candidate) > 3:
                next_pending_keys.append(key)
            retry_count += 1

        pending_keys = next_pending_keys
        if penalty == 0.0:
            print(f"[Round {round_idx}] presence_penalty={penalty}, need_retry={len(pending_keys)}")
        else:
            print(
                f"[Round {round_idx}] presence_penalty={penalty}, "
                f"accepted={accepted_count}, next_pending={len(pending_keys)}"
            )

    for key, candidates in retry_candidates.items():
        if candidates:
            if any(_is_na_labelset(c) for c in candidates):
                result_map[key] = ["N/A"]
            else:
                result_map[key] = min(candidates, key=len)

    # 2.3 Write file: assemble opinion_labels per pack (retry results override first-pass)
    with open(output_path, "w", encoding="utf-8") as w:
        for pi, record in enumerate(records):
            obj = record["obj"]
            opinion_count = record["opinion_count"]
            opinion_labels: List[List[str]] = []
            for ti in range(opinion_count):
                labs = result_map.get((pi, ti), ["N/A"])
                opinion_labels.append(labs if labs else ["N/A"])

            obj["category"] = opinion_labels
            w.write(json.dumps(obj, ensure_ascii=False) + "\n")
            w.flush()

    client.shutdown()
    print("Done.")
    print("Retry count:", retry_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify opinions produced by the split script (parts+opinions JSONL).")
    parser.add_argument("--input_jsonl", required=True, help="Path to input JSONL.")
    parser.add_argument("--output_jsonl", required=True, help="Path to output JSONL.")
    parser.add_argument("--model", default="Qwen3-8B-Classify", help="LLM model name used in the classify step")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--max_workers", type=int, default=128)
    parser.add_argument("--base_url", default="http://localhost:13098/v1")
    parser.add_argument("--cache_version", default="v1")

    args = parser.parse_args()

    _host = (urlparse(args.base_url).hostname or "").lower()
    _is_vllm = _host in {"localhost", "127.0.0.1", "::1"}
    args._is_vllm = _is_vllm
    vllm_proc = None
    if _is_vllm:
        print("Waiting for vLLM server to be ready...")
        wait_until_ready(args.base_url, "dummy", args.model, timeout=3600, interval=20.0)
        print("vLLM server is ready")
        client = LLMClient(
            api_key="dummy",
            base_url=args.base_url,
            model_name=args.model,
            max_workers=args.max_workers,
            cache_version=args.cache_version
        )
    else:
        raise NotImplementedError

    classify_opinions(args)
    client.shutdown()
