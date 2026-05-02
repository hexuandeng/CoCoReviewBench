#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose
-------
Batch-classify split discussion/comment threads into a fixed label taxonomy using an LLM.

This script reads an input JSONL (produced by an upstream "split" step). For each record,
it submits every split segment (except the final summary item) to an LLM classifier prompt,
then collects results and writes a new JSONL with an added `classify` field.

Label parsing policy
--------------------
- Accept exact labels if they appear in the allowed label set.
- Strip any <think>...</think> blocks from model output.
- Apply *limited* auto-correction:
  - Only correct by suffix if that suffix maps to exactly one allowed label.
  - If ambiguous, return None for that token (do not guess).

Inputs
------
- JSONL file determined by (org, year) conventions in the code.
- Prompt templates are provided by `prompt_registry.classify_reviews_prompt(...)`.
- Allowed labels are extracted from the prompt via `utils.extract_labels_from_prompt_text(...)`.

Outputs
-------
- A JSONL file mirroring the input objects, with:
  - `classify`: list of parsed label lists, aligned with input `split_texts` (excluding final summary).
  - Removes legacy `parsed` field if present.

Notes
-----
- The LLM client is expected to support async submission (`submit_task`) and blocking retrieval
  (`get_result`) with an internal worker pool.
- For open-sourcing, do NOT hardcode credentials. Pass them via --api_key.
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Set

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from prompt_registry import classify_reviews_prompt
from src.utils import BaseArguments, LLMClient, extract_labels_from_prompt_text

# Initialized in `__main__`
client: Optional[LLMClient] = None


def parse_llm_output(response: str, allowed_categories: Set[str]) -> List[Optional[str]]:
    """
    Parse model output into labels.

    The model may return comma/newline/semicolon-separated tokens. We:
      1) Strip <think>...</think> blocks.
      2) Normalize tokens (upper, hyphens, punctuation trimming).
      3) Keep exact matches to `allowed_categories`.
      4) If not exact, attempt *suffix-based* correction only when unambiguous.
         Otherwise, return None for that token.

    Returns a de-duplicated list of parsed labels (order not guaranteed).
    """
    # Build suffix index from allowed categories (e.g., "QUAL-EXP" -> suffix "EXP")
    suffix_index = defaultdict(list)
    for lab in allowed_categories:
        if "-" in lab:
            suf = lab.split("-")[-1]
            suffix_index[suf].append(lab)

    def normalize_label(tok: str) -> Optional[str]:
        s = (tok or "").strip()
        if not s:
            return None
        t = s.split()[0].strip().upper()
        t = (
            t.replace("\u2011", "-")
            .replace("\u2013", "-")
            .replace("\u2014", "-")
            .replace("\u2212", "-")
            .replace("_", "-")
        )
        t = t.strip(".,;:][}{)(")
        t = re.sub(r"-{2,}", "-", t)
        return t

    def correct_by_suffix(bad: Optional[str]) -> Optional[str]:
        if not bad or "-" not in bad:
            return None
        parts = bad.split("-")
        # If model returns something like "QUAL-EXP-SOMETHING", try first two components.
        if len(parts) > 2:
            cand = "-".join(parts[:2])
            if cand in allowed_categories:
                return cand
        suffix = parts[-1]
        cands = suffix_index.get(suffix, [])
        # Only auto-correct when the suffix maps to exactly one allowed label.
        if len(cands) == 1:
            return cands[0]
        return None  # ambiguous → do not guess

    # Strip <think>...</think> blocks robustly
    response = re.sub(r"(?is)<think>.*?</think>", "", response or "").strip()
    if response == "":
        return [None]

    parsed_labels: List[Optional[str]] = []
    for label_part in re.split(r"[,\n;]+", response):
        label = normalize_label(label_part)
        if not label:
            parsed_labels.append(None)
            continue

        if label not in allowed_categories:
            fixed = correct_by_suffix(label)
            label = fixed if (fixed and fixed in allowed_categories) else None

        parsed_labels.append(label)

    return list(set(parsed_labels))


def classify_all(args) -> None:
    """
    End-to-end pipeline:
      1) Determine input/output paths.
      2) Build prompts and allowed label set.
      3) Submit all segment classification tasks.
      4) Collect results and write output JSONL.

    This is intentionally a two-phase process to maximize client-side parallelism.
    """
    if not client:
        raise RuntimeError("LLMClient has not been initialized.")

    # Resolve paths
    if args.org == "eval":
        jsonl_path = "pre_eval/split/evaluation_gpt-5_medium_split.jsonl"
        output_dir = "pre_eval/classify"
    else:
        jsonl_path = f"{args.org}.cc_{args.year}/split/{args.org}.cc_{args.year}_sample_gpt-5_medium_split.jsonl"
        output_dir = f"{args.org}.cc_{args.year}/classify"
    os.makedirs(output_dir, exist_ok=True)

    model_tag = args.model.replace("/", "_")
    if args.org == "eval":
        output_path = f"{output_dir}/evaluation_{model_tag}_{args.effort}_classify.jsonl"
    else:
        output_path = f"{output_dir}/{args.org}.cc_{args.year}_{model_tag}_{args.effort}_classify.jsonl"

    # Prompts and allowed labels
    sys_prompt, main_prompt_body = classify_reviews_prompt()
    allowed_categories, label_defs = extract_labels_from_prompt_text(main_prompt_body)
    allowed_categories.add("N/A")
    label_defs.setdefault("N/A", "Polite text or pure paper summary; contains no substantive technical/content point.")

    # --- Phase 1: Submit all tasks ---
    tasks_to_collect = []
    total_records = 0
    total_tasks = 0

    print("Submitting classification tasks...")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            total_records += 1
            data_obj = json.loads(line)

            # The last item is a summary segment; skip it for classification.
            split_texts = data_obj.get("split_texts", [])[:-1]

            task_ids_for_line = []
            for item_data in split_texts:
                prompt_text = ""
                for role, text in item_data:
                    prompt_text += f"{role}\n\n{text}\n\n\n"

                task_id = client.submit_task(
                    prompt=prompt_text + main_prompt_body,
                    system_prompt=sys_prompt,
                    temperature=0.6,
                    reasoning_effort=args.effort,  # Custom kwarg supported by your client
                )
                task_ids_for_line.append(task_id)
                total_tasks += 1

            tasks_to_collect.append((data_obj, task_ids_for_line))

    print(f"Submission complete: {total_records} records, {total_tasks} tasks queued.")

    # --- Phase 2: Collect results and write output ---
    print("Collecting results and writing output...")
    processed_records = 0
    with open(output_path, "w", encoding="utf-8") as w:
        for data_obj, task_ids in tasks_to_collect:
            all_results = []
            for task_id in task_ids:
                result = client.get_result(task_id)  # Blocks until done
                parsed = parse_llm_output(result.get("content", ""), allowed_categories)
                all_results.append(parsed)

            data_obj["classify"] = all_results
            if "parsed" in data_obj:
                del data_obj["parsed"]  # Remove legacy field if present

            w.write(json.dumps(data_obj, ensure_ascii=False) + "\n")
            w.flush()

            processed_records += 1
            if processed_records % 100 == 0:
                print(f"  Written: {processed_records}/{total_records} records")

    print("Shutting down client...")
    client.shutdown()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch-classify split comment threads into a fixed taxonomy using an LLM."
    )
    BaseArguments.add_to_parser(parser, model_default="gpt-5-mini", effort_default="medium")
    args = parser.parse_args()
    BaseArguments.apply(args)
    client = LLMClient(
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model,
        max_workers=args.max_workers,
        cache_version=args.cache_version,
    )

    classify_all(args)
    client.shutdown()
