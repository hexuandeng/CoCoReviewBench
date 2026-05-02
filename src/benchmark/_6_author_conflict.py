#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose
-------
Detect whether authors *explicitly refute* reviewer opinions in peer review discussions,
grouped by reviewer, using an LLM.

Refutation detection policy
---------------------------
An opinion is marked as "refuted" (True) only when the author explicitly argues that the
reviewer's statement is factually incorrect, irrelevant, or based on a misunderstanding
(e.g., "We disagree", "This is incorrect", "This is a misunderstanding").

Not refutation (avoid false positives)
--------------------------------------
- Compliance + differentiation: the author performs the requested action (e.g., adds a baseline)
  and then explains differences; this is NOT a refutation.
- Technical clarification without rejecting the validity of the reviewer’s point is NOT a refutation.
- Pure citation/metadata lines are not arguments and cannot be refuted (always False).

Inputs
------
- JSONL records containing:
  - new_split_texts
  - sentence_texts
  - metareview (optional / venue-dependent structure)

Outputs
-------
- A JSONL file with:
  - author_refutation_analysis: per reviewer, per opinion refutation results
  - llm_call_stats: basic call statistics

Notes
-----
- This script assumes a specific on-disk dataset layout for default paths; override as needed.
- Be mindful of request volume and provider rate limits when using high parallelism.
- Do NOT hardcode API keys in open-source code; pass via CLI flags.
"""

import argparse
import json
import re
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from prompt_registry import author_conflict_system_prompt, author_conflict_prompt
from src.utils import BaseArguments, LLMClient
import traceback
logger = logging.getLogger(__name__)


def extract_text_by_role(block: List[Any],
                        sentence_texts: List[str],
                        role_prefix: str) -> Dict[str, str]:
    """Extract all texts for a given role prefix from a block (grouped by role name)."""
    if not isinstance(block, list) or len(block) != 2:
        return {}

    role_texts = {}
    for role_item in block[0]:
        role_name = role_item[0]
        sentence_ids = role_item[1]

        if not role_name.startswith(role_prefix):
            continue

        sorted_ids = sorted(sentence_ids)
        texts = []
        for sid in sorted_ids:
            if 0 <= sid < len(sentence_texts):
                text = sentence_texts[sid].strip()
                if text:
                    texts.append(text)

        if texts:
            role_texts[role_name] = "\n".join(texts)

    return role_texts


def group_by_reviewer(record: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Group content by reviewer and collect:
      - all opinions from that reviewer
      - all author responses that appear in the SAME blocks as that reviewer's opinions
      - meta-review text (as global context)

    Returns:
      {
        "Reviewer 1": {
          "opinions": [{"block_id": 0, "text": "..."}],
          "author_responses_combined": "all author responses from related blocks",
          "meta_review": "meta-review text"
        },
        ...
      }
    """
    new_split_texts = record.get("new_split_texts", [])
    sentence_texts = record.get("sentence_texts", [])
    meta_review = record.get("metareview", "")
    record_id = record.get("id", "")

    # Safely extract meta-review text across different possible schemas.
    try:
        if "comment" in meta_review[0][0][1]:
            meta_review_text = meta_review[0][0][1]["comment"]
            if isinstance(meta_review_text, Dict):
                meta_review_text = meta_review_text["value"]
        elif "metareview:_summary,_strengths_and_weaknesses" in meta_review[0][0][1]:
            meta_review_text = meta_review[0][0][1]["metareview:_summary,_strengths_and_weaknesses"]
        else:
            meta_review_text = meta_review[0][0][1]["metareview"]
            if isinstance(meta_review_text, Dict):
                meta_review_text = meta_review_text["value"]
    except IndexError as e:
        meta_review_text = ""
        print("Meta-review text not found (IndexError):", record_id)
    except KeyError as e:
        meta_review_text = ""
        print("Meta-review key not found (KeyError):", record_id)
    except TypeError as e:
        meta_review_text = ""
        print("Meta-review not found / invalid structure (TypeError):", record_id)

    if not new_split_texts or not sentence_texts:
        return {}

    # Initialize reviewer groups
    reviewer_groups = defaultdict(lambda: {
        "opinions": [],
        "author_responses_combined": "",
        "meta_review": meta_review_text
    })

    # Track which blocks each reviewer appears in
    reviewer_blocks = defaultdict(set)

    # Step 1: collect all reviewer opinions and record block IDs
    for block_idx, block in enumerate(new_split_texts):
        if not isinstance(block, list) or len(block) != 2:
            continue

        reviewer_texts = extract_text_by_role(block, sentence_texts, "Reviewer")

        for reviewer_name, text in reviewer_texts.items():
            reviewer_groups[reviewer_name]["opinions"].append({
                "block_id": block_idx,
                "text": text
            })
            reviewer_blocks[reviewer_name].add(block_idx)

    # Step 2: for each reviewer, collect all author responses in the same blocks
    for reviewer_name, block_ids in reviewer_blocks.items():
        author_responses_list = []

        for block_id in block_ids:
            assert block_id < len(new_split_texts), "block_id should less than len(new_split_texts)"
            block = new_split_texts[block_id]
            if not isinstance(block, list) or len(block) != 2:
                continue

            author_texts = extract_text_by_role(block, sentence_texts, "Author")

            for author_name, text in author_texts.items():
                new_author_text = f"\n{text}"
                if text.strip() and (new_author_text not in author_responses_list):
                    author_responses_list.append(f"\n{text}")

        combined_responses = "\n\n".join(author_responses_list) if author_responses_list else ""
        reviewer_groups[reviewer_name]["author_responses_combined"] = "[Author]" + combined_responses

    return dict(reviewer_groups)


def parse_batch_refutation_response(response: str, num_opinions: int) -> List[Dict[str, Any]]:
    """Parse the batched LLM response."""
    response = re.sub(r'^```json\s*', '', response)
    response = re.sub(r'\s*```$', '', response)
    response = response.strip()

    try:
        data = json.loads(response, strict=False)

        # Ensure it is a list and the length matches the number of opinions
        if isinstance(data, list) and len(data) == num_opinions:
            parsed_results = []
            for item in data:
                parsed_results.append({
                    "refutes": bool(item.get("refutes", False))
                })
            return parsed_results
        else:
            logger.warning(
                f"Response format mismatch: expected {num_opinions} opinions, got "
                f"{len(data) if isinstance(data, list) else 'non-list'}"
            )
            return [{"refutes": False} for _ in range(num_opinions)]

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                if isinstance(data, list) and len(data) == num_opinions:
                    parsed_results = []
                    for item in data:
                        parsed_results.append({
                            "refutes": bool(item.get("refutes", False))
                        })
                    return parsed_results
            except:
                pass

        return [{"refutes": False} for _ in range(num_opinions)]


def process_single_record(record: Dict[str, Any],
                         client: LLMClient,
                         effort: str) -> Tuple[Dict[str, Any], int]:
    """
    Process a single record:
      - group by reviewer
      - batch-evaluate whether author responses refute each opinion
    """
    record_id = record.get("id", record.get("record_id"))
    logger.info(f"Processing record {record_id}")

    reviewer_groups = group_by_reviewer(record)
    if not reviewer_groups:
        logger.warning(f"Record {record_id} has no reviewer opinions")
        return record, 0

    logger.info(f"Record {record_id} contains {len(reviewer_groups)} reviewers")

    total_llm_calls = 0
    all_refutation_results = {}

    for reviewer_name, group_data in reviewer_groups.items():
        opinions = group_data["opinions"]
        author_responses_combined = group_data["author_responses_combined"]
        meta_review = group_data["meta_review"]

        if not opinions:
            continue

        logger.info(f"  Reviewer '{reviewer_name}': {len(opinions)} opinions")

        prompt = author_conflict_prompt(
            reviewer_name=reviewer_name,
            opinions=opinions,
            author_responses_combined=author_responses_combined,
            meta_review=meta_review
        )

        try:
            system_prompt = author_conflict_system_prompt()

            key = client.submit_task(
                prompt,
                system_prompt=system_prompt,
                temperature=0.6,
                reasoning_effort=effort
            )

            result = client.get_result(key)
            total_llm_calls += 1

            if result.get("content"):
                batch_results = parse_batch_refutation_response(
                    result["content"],
                    len(opinions)
                )

                reviewer_results = []
                for idx, opinion in enumerate(opinions):
                    reviewer_results.append({
                        "block_id": opinion["block_id"],
                        "opinion_text": opinion["text"],
                        "author_refutes": batch_results[idx]["refutes"],
                        "llm_call_success": True
                    })

                all_refutation_results[reviewer_name] = {
                    "num_opinions": len(opinions),
                    "author_responses_combined": author_responses_combined,
                    "meta_review": meta_review,
                    "results": reviewer_results
                }
            else:
                logger.error(f"No LLM response for reviewer '{reviewer_name}'")
                reviewer_results = []
                for opinion in opinions:
                    reviewer_results.append({
                        "block_id": opinion["block_id"],
                        "opinion_text": opinion["text"],
                        "author_refutes": False,
                        "llm_call_success": False
                    })

                all_refutation_results[reviewer_name] = {
                    "num_opinions": len(opinions),
                    "author_responses_combined": author_responses_combined,
                    "meta_review": meta_review,
                    "results": reviewer_results
                }

        except Exception as e:
            logger.error(f"LLM call failed for reviewer '{reviewer_name}': {e}")
            reviewer_results = []
            for opinion in opinions:
                reviewer_results.append({
                    "block_id": opinion["block_id"],
                    "opinion_text": opinion["text"],
                    "author_refutes": False,
                    "llm_call_success": False,
                    "error": str(e)
                })

            all_refutation_results[reviewer_name] = {
                "num_opinions": len(opinions),
                "author_responses_combined": author_responses_combined,
                "meta_review": meta_review,
                "results": reviewer_results
            }

    record["author_refutation_analysis"] = all_refutation_results
    record["llm_call_stats"] = {
        "total_calls": total_llm_calls,
        "num_reviewers_processed": len(reviewer_groups)
    }

    return record, total_llm_calls


def process_all_records(input_path: str,
                       output_path: str,
                       client: LLMClient,
                       effort: str,
                       max_workers: int = 32) -> None:
    """Process all records in the input JSONL."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    records = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {e}")
                continue

    logger.info(f"Loaded {len(records)} records")

    # Example for quick testing (keep commented out):
    # num_test = 1000
    # records = records[0:num_test]
    # logger.info(f"Test mode: only processing the first {num_test} records")

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
                logger.error(
                    f"Error type: {type(e).__name__}\n"
                    f"Error message: {str(e)}\n"
                    f"Traceback:\n{traceback.format_exc()}"
                )
                record = future_to_record[future]
                processed_records.append(record)

    # Sort back to the original order by record id (best-effort)
    processed_records.sort(key=lambda x: x.get('id', 0))

    with open(output_path, 'w', encoding='utf-8') as f:
        for record in processed_records:
            json.dump(record, f, ensure_ascii=False)
            f.write('\n')

    logger.info(f"Done! Total LLM calls: {total_llm_calls}")
    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect whether authors explicitly refute reviewer opinions (grouped by reviewer; includes meta-review context)."
    )
    BaseArguments.add_to_parser(parser, model_default="gpt-5-mini", effort_default="medium")
    args = parser.parse_args()
    BaseArguments.apply(args)

    # Default input path: expects Stage-1 point_id assignment output as input here.
    input_path = f"{args.paper_series}/point_ids/{args.paper_series}_deepseek-reasoner_medium_split_clean_point_ids.jsonl"
    output_path = f"{args.paper_series}/refutations/{args.paper_series}_{args.model.replace("/","-")}_medium_split_clean_refutations.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Initialize LLM client
    client = LLMClient(
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model,
        max_workers=args.max_workers,
        cache_version=args.cache_version
    )

    try:
        process_all_records(
            input_path=input_path,
            output_path=output_path,
            client=client,
            effort=args.effort,
            max_workers=args.max_workers
        )
    finally:
        client.shutdown()
