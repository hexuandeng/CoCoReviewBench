#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose
-------
Detect *genuine* conflicts among reviewer opinions within each discussion point.

This script reads a JSONL file where each line is a record containing:
- sentence_texts: list[str]
- new_split_texts: list[Any]  (blocks; each block contains role -> sentence_id mappings)
- discussion_point_assignments: dict[label, {...}]  (block- and reviewer-level point assignments)

For each label, the script groups reviewer texts by point_id and calls an LLM **only**
for points that have 2+ reviewer opinions. Points with 0/1 opinion are marked as
no-conflict by default.

Output fields (added per record)
-------------------------------
- conflict_analysis: {label: {"point_conflicts": {point_id: {"has_conflict": bool}}}}

Notes
-----
- Uses a threaded worker pool to process records in parallel.
- LLM results are parsed from JSON; the parser is resilient to markdown fences and
  minor formatting noise.
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

from prompt_registry import conflict_reviews_system_prompt, conflict_reviews_prompt
from src.utils import BaseArguments, LLMClient

logger = logging.getLogger(__name__)


def parse_conflict_detection_response(response: str) -> Dict[str, Any]:
    """
    Parse the LLM JSON response (conflict detection only).

    Expected JSON format:
      {"point_conflicts": {point_id: {"has_conflict": bool}}}

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
        if "point_conflicts" in data:
            return data
        return {"point_conflicts": {}}
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from LLM response: {e}")
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                if "point_conflicts" in data:
                    return data
                return {"point_conflicts": {}}
            except Exception:
                pass
        return {"point_conflicts": {}}


def extract_role_texts(
    block: List[Any],
    sentence_texts: List[str],
    role_prefix: str = "Reviewer",
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Extract texts for roles whose names start with `role_prefix` from a single block.

    Returns:
      (reviewer_texts, other_texts)
        - reviewer_texts: {role_name: combined_text}
        - other_texts:    {role_name: combined_text}
    """
    reviewer_texts = {}
    other_texts = {}

    for role_item in block[0]:  # block[0] is the role list
        role_name = role_item[0]
        sentence_ids = role_item[1]

        # Sort sentence IDs and concatenate corresponding texts
        sorted_ids = sorted(sentence_ids)
        texts = []
        for sid in sorted_ids:
            if 0 <= sid < len(sentence_texts):
                texts.append(sentence_texts[sid].strip())

        combined_text = "\n".join(texts) if texts else ""

        if role_name.startswith(role_prefix):
            reviewer_texts[role_name] = combined_text
        else:
            other_texts[role_name] = combined_text

    return reviewer_texts, other_texts


def process_single_record(
    record: Dict[str, Any], client: LLMClient, effort: str
) -> Tuple[Dict[str, Any], int]:
    """
    Process a single record: conflict detection only.

    Returns:
      (processed_record, num_llm_calls)
    """
    point_assignments = record.get("discussion_point_assignments", {})
    if not point_assignments:
        logger.warning(
            f"Record {record.get('id')} is missing 'discussion_point_assignments'."
        )
        return record, 0

    sentence_texts = record.get("sentence_texts", [])
    new_split_texts = record.get("new_split_texts", [])

    if not sentence_texts or not new_split_texts:
        logger.warning(
            f"Record {record.get('id')} is missing 'sentence_texts' or 'new_split_texts'."
        )
        return record, 0

    conflict_analysis = {}
    total_llm_calls = 0

    for label, assignment_data in point_assignments.items():
        # Build a mapping: point_id -> list of opinions
        point_groups = defaultdict(list)

        for block_data in assignment_data.get("blocks", []):
            block_id = int(block_data["block_id"])
            reviewer_assignments = block_data["reviewer_assignments"]

            if block_id >= len(new_split_texts):
                continue

            block = new_split_texts[block_id]
            reviewer_texts, _ = extract_role_texts(block, sentence_texts)

            for reviewer_name, point_id in reviewer_assignments.items():
                if isinstance(point_id, list) and len(point_id) > 0:
                    point_id = int(point_id[0])
                elif isinstance(point_id, list) and len(point_id) == 0:
                    logger.warning(
                        f"Record {record.get('id')} has an invalid empty point_id list: {point_id}"
                    )
                    continue
                elif isinstance(point_id, Dict):
                    point_id = int(list(point_id.values())[0])
                else:
                    point_id = int(point_id)

                if reviewer_name in reviewer_texts:
                    point_groups[str(point_id)].append(
                        {
                            "reviewer": reviewer_name,
                            "block_id": block_id,
                            "text": reviewer_texts[reviewer_name],
                        }
                    )

        if not point_groups:
            continue

        # Separate points with single opinion vs. multiple opinions
        multi_opinion_groups = {}
        single_opinion_results = {}

        for point_id, opinions in point_groups.items():
            if len(opinions) <= 1:
                single_opinion_results[point_id] = {"has_conflict": False}
            else:
                multi_opinion_groups[point_id] = opinions

        # Only call the LLM if there are points with multiple opinions
        if multi_opinion_groups:
            logger.info(
                f"Detecting conflicts for record {record.get('id')} / label '{label}' "
                f"({len(multi_opinion_groups)}/{len(point_groups)} points require LLM)."
            )

            prompt = conflict_reviews_prompt(multi_opinion_groups)

            try:
                system_prompt = conflict_reviews_system_prompt()

                key = client.submit_task(
                    prompt,
                    system_prompt=system_prompt,
                    temperature=0.6,
                    reasoning_effort=effort,
                )

                result = client.get_result(key)
                total_llm_calls += 1

                if result.get("content"):
                    parsed_result = parse_conflict_detection_response(result["content"])
                    all_results = {
                        **parsed_result.get("point_conflicts", {}),
                        **single_opinion_results,
                    }
                    conflict_analysis[label] = {"point_conflicts": all_results}
                else:
                    logger.error(f"LLM returned no content for label '{label}'.")
                    conflict_analysis[label] = {"point_conflicts": single_opinion_results}

            except Exception as e:
                logger.error(f"LLM call failed for label '{label}': {e}")
                conflict_analysis[label] = {"point_conflicts": single_opinion_results}
        else:
            logger.info(
                f"Record {record.get('id')} / label '{label}': all points have <=1 opinion; skipping LLM."
            )
            conflict_analysis[label] = {"point_conflicts": single_opinion_results}

    record["conflict_analysis"] = conflict_analysis
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect conflicts within discussion points (per label) using an LLM."
    )
    BaseArguments.add_to_parser(parser, model_default="deepseek-reasoner", effort_default="medium")
    args = parser.parse_args()
    BaseArguments.apply(args)

    input_path = f"{args.paper_series}/point_ids/{args.paper_series}_deepseek-reasoner_medium_split_clean_point_ids.jsonl"
    output_path = f"{args.paper_series}/conflicts/{args.paper_series}_{args.model.replace("/", "-")}_medium_split_clean_with_conflicts.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    client = LLMClient(
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model,
        max_workers=args.max_workers,
        cache_version=args.cache_version,
    )

    process_all_records(
        input_path=input_path,
        output_path=output_path,
        client=client,
        effort=args.effort,
        max_workers=args.max_workers,
    )
    client.shutdown()
