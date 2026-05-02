#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose
-------
Resolve *previously detected* conflicts among reviewer opinions for each discussion point,
and decide which opinion block(s) are more correct/valid.

This script expects input records that already contain:
- conflict_analysis: {label: {"point_conflicts": {point_id: {"has_conflict": bool}}}}
and will only run LLM-based resolution for points where has_conflict == True.

Resolution policy
-----------------
- For each conflicted point_id (within each label), gather all associated reviewer opinion blocks
  (identified by block_id) and provide them to the LLM along with optional meta-review context.
- The LLM returns:
  - correct_blocks:   ["block_X", ...]
  - incorrect_blocks: ["block_Y", ...]

Output fields (added per record)
-------------------------------
- conflict_resolution: {label: {point_id: {"resolution": {...}, "opinions": [...], "num_opinions": int}}}

Notes
-----
- Uses a threaded worker pool to process records in parallel.
- The JSON parser is resilient to markdown fences and minor formatting noise.
- **Security**: Do NOT hardcode real API keys in open-source code. Provide them via
  CLI flags.
"""

import argparse
import json
import re
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from prompt_registry import validate_conflict_reviews_system_prompt, validate_conflict_reviews_prompt
from src.utils import BaseArguments, LLMClient

logger = logging.getLogger(__name__)


def parse_conflict_resolution_response(response: str) -> Dict[str, Any]:
    """
    Parse the LLM JSON response (conflict resolution / adjudication).

    Expected JSON format:
      {
        "correct_blocks":   ["block_X", ...],
        "incorrect_blocks": ["block_Y", ...]
      }

    This parser is resilient to:
    - ```json fenced blocks
    - extra whitespace
    - cases where the JSON object is embedded in other text
    """
    response = re.sub(r"^```json\s*", "", response)
    response = re.sub(r"\s*```$", "", response)
    response = response.strip()

    try:
        data = json.loads(response, strict=False)
        return {
            "correct_blocks": data.get("correct_blocks", []),
            "incorrect_blocks": data.get("incorrect_blocks", []),
        }
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from LLM response: {e}")
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return {
                    "correct_blocks": data.get("correct_blocks", []),
                    "incorrect_blocks": data.get("incorrect_blocks", []),
                }
            except Exception:
                pass
        return {"correct_blocks": [], "incorrect_blocks": []}


def extract_meta_review_text(record: Dict[str, Any]) -> str:
    """
    Extract meta-review text from a record (with defensive error handling).

    The meta-review structure may vary across venues/exports, so this function tries
    a few common patterns and returns an empty string if not found.
    """
    meta_review = record.get("metareview", "")
    record_id = record.get("id", "")

    try:
        if isinstance(meta_review, list) and len(meta_review) > 0:
            if "comment" in meta_review[0][0][1]:
                meta_review_text = meta_review[0][0][1]["comment"]
                if isinstance(meta_review_text, dict):
                    meta_review_text = meta_review_text.get("value", "")
            elif "metareview:_summary,_strengths_and_weaknesses" in meta_review[0][0][1]:
                meta_review_text = meta_review[0][0][1][
                    "metareview:_summary,_strengths_and_weaknesses"
                ]
            else:
                meta_review_text = meta_review[0][0][1]["metareview"]
                if isinstance(meta_review_text, dict):
                    meta_review_text = meta_review_text.get("value", "")
            return meta_review_text
    except (IndexError, KeyError, TypeError) as e:
        logger.warning(f"Record {record_id}: failed to extract meta-review text: {e}")
        return ""

    return ""


def extract_role_texts(
    block: List[Any], sentence_texts: List[str], role_prefix: str = "Reviewer"
) -> Dict[str, str]:
    """
    Extract reviewer texts from a block (roles starting with `role_prefix`).

    Returns:
      {reviewer_name: combined_text}
    """
    role_texts = {}

    if not isinstance(block, list) or len(block) != 2:
        return role_texts

    for role_item in block[0]:
        role_name = role_item[0]
        sentence_ids = role_item[1]

        if not role_name.startswith(role_prefix):
            continue

        sorted_ids = sorted(sentence_ids)
        texts = []
        for sid in sorted_ids:
            if 0 <= sid < len(sentence_texts):
                texts.append(sentence_texts[sid].strip())

        combined_text = "\n".join(texts) if texts else ""
        if combined_text:
            role_texts[role_name] = combined_text

    return role_texts


def process_single_record(
    record: Dict[str, Any], client: LLMClient, effort: str
) -> Tuple[Dict[str, Any], int]:
    """
    Process a single record: adjudicate conflicts for discussion points marked as conflicted.
    """
    conflict_analysis = record.get("conflict_analysis", {})
    if not conflict_analysis:
        logger.warning(f"Record {record.get('id')} is missing 'conflict_analysis'.")
        return record, 0

    sentence_texts = record.get("sentence_texts", [])
    new_split_texts = record.get("new_split_texts", [])
    point_assignments = record.get("discussion_point_assignments", {})

    if not sentence_texts or not new_split_texts or not point_assignments:
        logger.warning(f"Record {record.get('id')} is missing required fields.")
        return record, 0

    # Meta-review provides additional global context (optional)
    meta_review = extract_meta_review_text(record)

    resolution_analysis = {}
    total_llm_calls = 0

    for label, label_data in conflict_analysis.items():
        point_conflicts = label_data.get("point_conflicts", {})
        if not point_conflicts:
            continue

        label_assignments = point_assignments.get(label, {}).get("blocks", [])

        # Build mapping: block_id -> [(reviewer_name, point_id), ...]
        block_to_points = defaultdict(list)
        for block_info in label_assignments:
            block_id = int(block_info.get("block_id"))
            reviewer_assignments = block_info.get("reviewer_assignments", {})
            for reviewer_name, point_id in reviewer_assignments.items():
                if isinstance(point_id, list):
                    point_id = int(point_id[0])
                elif isinstance(point_id, dict):
                    point_id = int(list(point_id.values())[0])
                else:
                    point_id = int(point_id)
                block_to_points[block_id].append((reviewer_name, point_id))

        label_resolutions = {}

        for point_id_str, conflict_info in point_conflicts.items():
            if not conflict_info.get("has_conflict", False):
                continue

            opinions = []
            for block_id, reviewer_point_pairs in block_to_points.items():
                if block_id >= len(new_split_texts):
                    continue

                block = new_split_texts[block_id]
                reviewer_texts = extract_role_texts(block, sentence_texts)

                for reviewer_name, assigned_pid in reviewer_point_pairs:
                    if int(assigned_pid) == int(point_id_str) and reviewer_name in reviewer_texts:
                        opinions.append(
                            {
                                "block_id": block_id,
                                "reviewer": reviewer_name,
                                "text": reviewer_texts[reviewer_name],
                            }
                        )

            if len(opinions) < 2:
                logger.warning(
                    f"Point {point_id_str} is marked conflicted but has <2 opinions: {len(opinions)}"
                )
                continue

            prompt = validate_conflict_reviews_prompt(point_id_str, opinions, meta_review)

            try:
                system_prompt = validate_conflict_reviews_system_prompt()

                logger.info(
                    f"Resolving conflicts for record {record.get('id')} / point {point_id_str} "
                    f"({len(opinions)} opinion blocks)."
                )

                key = client.submit_task(
                    prompt,
                    system_prompt=system_prompt,
                    temperature=0.6,
                    reasoning_effort=effort,
                )

                result = client.get_result(key)
                total_llm_calls += 1

                if result.get("content"):
                    parsed_result = parse_conflict_resolution_response(result["content"])
                    label_resolutions[point_id_str] = {
                        "resolution": parsed_result,
                        "opinions": opinions,
                        "num_opinions": len(opinions),
                    }
                else:
                    logger.error(f"LLM returned no content for point {point_id_str}.")
                    label_resolutions[point_id_str] = {
                        "resolution": {"correct_blocks": [], "incorrect_blocks": []},
                        "opinions": opinions,
                        "num_opinions": len(opinions),
                    }

            except Exception as e:
                logger.error(f"LLM call failed for point {point_id_str}: {e}")
                label_resolutions[point_id_str] = {
                    "resolution": {"correct_blocks": [], "incorrect_blocks": []},
                    "opinions": opinions,
                    "num_opinions": len(opinions),
                }

        if label_resolutions:
            resolution_analysis[label] = label_resolutions

    record["conflict_resolution"] = resolution_analysis
    return record, total_llm_calls


def process_all_records(
    input_path: str,
    output_path: str,
    client: LLMClient,
    effort: str,
    max_workers: int = 32,
) -> None:
    """
    Process all records from an input JSONL and write an output JSONL.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse a JSONL line: {e}")
                continue

    logger.info(f"Loaded {len(records)} records from: {input_path}")

    # Estimate how many conflicted points require adjudication
    total_conflict_points = 0
    for record in records:
        conflict_analysis = record.get("conflict_analysis", {})
        for _, label_data in conflict_analysis.items():
            for _, conflict_info in label_data.get("point_conflicts", {}).items():
                if conflict_info.get("has_conflict", False):
                    total_conflict_points += 1

    logger.info(f"Estimated conflicted discussion points to resolve: {total_conflict_points}")

    processed_records = []
    total_llm_calls = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_record = {
            executor.submit(process_single_record, record, client, effort): record
            for record in records
        }

        for future in as_completed(future_to_record):
            try:
                result, calls = future.result()
                processed_records.append(result)
                total_llm_calls += calls
            except Exception as e:
                logger.error(f"Failed to process a record: {e}")
                record = future_to_record[future]
                processed_records.append(record)

    processed_records.sort(key=lambda x: x.get("id", 0))

    with open(output_path, "w", encoding="utf-8") as f:
        for record in processed_records:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

    logger.info(f"Done. Total LLM calls: {total_llm_calls}")
    logger.info(f"Wrote output to: {output_path}")

    # Summary statistics
    analyzed_points = 0
    total_correct_blocks = 0
    total_incorrect_blocks = 0

    for record in processed_records:
        resolution_analysis = record.get("conflict_resolution", {})
        for _, label_data in resolution_analysis.items():
            for _, point_data in label_data.items():
                analyzed_points += 1
                resolution = point_data["resolution"]
                total_correct_blocks += len(resolution.get("correct_blocks", []))
                total_incorrect_blocks += len(resolution.get("incorrect_blocks", []))

    logger.info("=" * 60)
    logger.info("Summary:")
    logger.info(f"Conflicted points adjudicated: {analyzed_points}")
    logger.info(f"Blocks marked correct:        {total_correct_blocks}")
    logger.info(f"Blocks marked incorrect:      {total_incorrect_blocks}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Adjudicate conflicts between reviewer opinions and decide which block(s) are more correct "
            "(block_id-based, with meta-review context when available)."
        )
    )
    BaseArguments.add_to_parser(parser, model_default="deepseek-reasoner", effort_default="medium",)
    args = parser.parse_args()
    BaseArguments.apply(args)
    model_name = args.model.replace("/", "-")
    args.input = f"{args.paper_series}/conflicts/{args.paper_series}_deepseek-reasoner_medium_split_clean_with_conflicts.jsonl"
    args.output = f"{args.paper_series}/conflicts_resolved/{args.paper_series}_{model_name}_medium_split_clean_conflicts_resolved.jsonl"
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    client = LLMClient(
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model,
        max_workers=args.max_workers,
        cache_version=args.cache_version,
    )

    process_all_records(
        input_path=args.input,
        output_path=args.output,
        client=client,
        effort=args.effort,
        max_workers=args.max_workers,
    )
    client.shutdown()
