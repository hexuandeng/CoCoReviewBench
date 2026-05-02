#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose
-------
Validate whether reviewer criticisms are *actually wrong* in cases where **two different models**
both detected that the author explicitly refuted the reviewer.

This script is intended as a second-stage filter:
1) Load two refutation-detection outputs (from two models).
2) Take the intersection of opinions where both models predicted `author_refutes=True`.
3) Use an LLM to judge whether each intersected reviewer opinion is truly incorrect
   (factually/logically), based on the author response and the meta-review.

Selection policy (intersection)
-------------------------------
- Only keep records that exist in BOTH input files.
- For each reviewer within a record:
  - Only keep block_ids where BOTH models marked `author_refutes=True`.
- Only keep records that still have at least one intersected refuted opinion.

Outputs
-------
- JSONL (validated):
  - reviewer_opinion_validation: per reviewer, validated judgments for refuted opinions
  - validation_stats: aggregate processing stats
- JSON (summary):
  - totals and rates over all validated opinions (wrong vs. not wrong)

Notes
-----
- This script assumes a project-local `utils.LLMClient` and a specific dataset schema.
- Be mindful of rate limits and request volume when using high parallelism.
- Do NOT hardcode API keys in open-source code; pass via CLI flags.
"""

import argparse
import json
import logging
import re
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from prompt_registry import validate_author_conflict_system_prompt, validate_author_conflict_prompt
from src.utils import BaseArguments, LLMClient

logger = logging.getLogger(__name__)


def extract_meta_review_text(meta_review: Any, record_id) -> str:
    """Safely extract meta-review text across different possible schemas."""
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
        print("Meta-review text not found:", record_id)
    except KeyError as e:
        meta_review_text = ""
        print("Meta-review key not found:", record_id)
    except TypeError as e:
        meta_review_text = ""
        print("Meta-review key not found:", record_id)
    return meta_review_text


def load_refutation_results(input_path1: str, input_path2: str) -> List[Dict[str, Any]]:
    """
    Load two model outputs and take the intersection:
      - Only keep opinions where BOTH models predict `author_refutes=True`.
    Also track each reviewer's total number of opinions (for later statistics).
    """
    records = []

    # Load file 1
    records1 = {}
    try:
        with open(input_path1, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    record_id = record.get("id", record.get("record_id", f"unknown_{line_num}"))
                    records1[record_id] = record
                except json.JSONDecodeError as e:
                    logger.error(f"File 1 line {line_num} JSON parse failed: {e}")
                    continue
    except FileNotFoundError:
        logger.error(f"File not found: {input_path1}")
        sys.exit(1)

    # Load file 2
    records2 = {}
    try:
        with open(input_path2, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    record_id = record.get("id", record.get("record_id", f"unknown_{line_num}"))
                    records2[record_id] = record
                except json.JSONDecodeError as e:
                    logger.error(f"File 2 line {line_num} JSON parse failed: {e}")
                    continue
    except FileNotFoundError:
        logger.error(f"File not found: {input_path2}")
        sys.exit(1)

    # Intersection by record_id
    common_record_ids = set(records1.keys()) & set(records2.keys())
    logger.info(
        f"File 1 records: {len(records1)}, File 2 records: {len(records2)}, "
        f"common records: {len(common_record_ids)}"
    )

    for record_id in common_record_ids:
        record1 = records1[record_id]
        record2 = records2[record_id]

        # Safely extract meta-review (from file 1)
        meta_review_raw = record1.get("metareview", "")
        meta_review_text = extract_meta_review_text(meta_review_raw, record_id)

        # Get refutation analyses from both models
        refutation_analysis1 = record1.get("author_refutation_analysis", {})
        refutation_analysis2 = record2.get("author_refutation_analysis", {})

        if not refutation_analysis1 or not refutation_analysis2:
            logger.warning(f"Record {record_id} missing refutation analysis; skipping")
            continue

        processed_record = {
            "record_id": record_id,
            "meta_review": meta_review_text,
            "reviewers": {},
            "original_record": record1  # keep original record from file 1
        }

        # Common reviewers
        common_reviewers = set(refutation_analysis1.keys()) & set(refutation_analysis2.keys())
        for reviewer_name in common_reviewers:
            opinions1 = refutation_analysis1[reviewer_name].get("results", [])
            opinions2 = refutation_analysis2[reviewer_name].get("results", [])

            # Total number of opinions for this reviewer (assumes both models extracted the same count)
            total_opinions = len(opinions1)

            # Build block_id -> opinion map, only for author_refutes=True
            def build_opinion_map(opinions):
                op_map = {}
                for op in opinions:
                    if op.get("author_refutes", False) and op.get("opinion_text"):
                        block_id = op.get("block_id")
                        op_map[block_id] = op
                return op_map

            opinion_map1 = build_opinion_map(opinions1)
            opinion_map2 = build_opinion_map(opinions2)

            # Block IDs where both models agree on refutation
            common_block_ids = set(opinion_map1.keys()) & set(opinion_map2.keys())

            if common_block_ids:
                refuted_opinions = []
                for block_id in common_block_ids:
                    op1 = opinion_map1[block_id]
                    refuted_opinions.append({
                        "block_id": block_id,
                        "text": op1.get("opinion_text", ""),
                        "author_responses_combined": refutation_analysis1[reviewer_name].get("author_responses_combined", ""),
                        "llm_call_success": op1.get("llm_call_success", False)
                    })

                processed_record["reviewers"][reviewer_name] = {
                    "refuted_opinions": refuted_opinions,
                    "num_refuted": len(refuted_opinions),
                    "total_opinions": total_opinions
                }
            else:
                print("No intersected block_ids found")

        # Keep only records with at least one intersected refuted opinion
        if processed_record["reviewers"]:
            records.append(processed_record)

    logger.info(f"Loaded {len(records)} records with intersected refuted opinions")
    total_refuted = sum(sum(r["reviewers"][rev]["num_refuted"] for rev in r["reviewers"]) for r in records)
    total_all = sum(sum(r["reviewers"][rev]["total_opinions"] for rev in r["reviewers"]) for r in records)
    logger.info(f"Total opinions: {total_all}; intersected refuted opinions: {total_refuted}")

    return records


def generate_summary(processed_records: List[Dict], output_path: str) -> None:
    """Generate an aggregate summary over all validated results."""
    summary = {
        "total_records": len(processed_records),
        "total_reviewers": 0,
        "total_all_opinions": 0,
        "total_refuted_opinions": 0,
        "reviewer_actually_wrong_cases": 0,
        "reviewer_correct_cases": 0,
        "model_success_rate": 0
    }

    for record in processed_records:
        validation_data = record.get("reviewer_opinion_validation", {})
        summary["total_reviewers"] += len(validation_data)

        for reviewer_name, data in validation_data.items():
            opinions = data.get("refuted_opinions_validated", [])
            summary["total_refuted_opinions"] += len(opinions)
            summary["total_all_opinions"] += data.get("total_opinions", 0)

            for op in opinions:
                if op.get("validation_result", {}).get("is_reviewer_wrong"):
                    summary["reviewer_actually_wrong_cases"] += 1
                else:
                    summary["reviewer_correct_cases"] += 1

    if summary["total_refuted_opinions"] > 0:
        summary["wrong_rate"] = summary["reviewer_actually_wrong_cases"] / summary["total_refuted_opinions"]
        summary["correct_rate"] = summary["reviewer_correct_cases"] / summary["total_refuted_opinions"]

    summary_path = output_path.replace(".jsonl", "_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("Validation summary")
    logger.info("=" * 60)
    logger.info(f"Total records: {summary['total_records']}")
    logger.info(f"Total reviewers: {summary['total_reviewers']}")
    logger.info(f"Total opinions (all): {summary['total_all_opinions']}")
    logger.info(f"Total refuted opinions (intersection): {summary['total_refuted_opinions']}")
    logger.info(
        f"Reviewer actually wrong: {summary['reviewer_actually_wrong_cases']} "
        f"({summary.get('wrong_rate', 0):.2%})"
    )
    logger.info(
        f"Reviewer not wrong: {summary['reviewer_correct_cases']} "
        f"({summary.get('correct_rate', 0):.2%})"
    )
    logger.info(f"Summary saved to: {summary_path}")


def parse_batch_validation_response(response: str, expected_block_ids: List[str]) -> List[Dict[str, Any]]:
    """Parse batch validation output, robust to multiple formats and block_id variants."""
    response = re.sub(r'^```json\s*', '', response)
    response = re.sub(r'\s*```$', '', response)
    response = response.strip()

    def normalize_block_id(block_id: Any) -> str:
        """Extract the numeric core to normalize block_id variants."""
        import re

        s = str(block_id).lower().replace(" ", "")
        match = re.search(r'\d+', s)
        return match.group() if match else s

    try:
        data = json.loads(response, strict=False)
        judgments = data.get("judgments", [])

        # Compatibility: LLM returns dict instead of list
        if isinstance(judgments, dict):
            logger.warning("LLM returned dict judgments; attempting automatic conversion...")
            temp_list = []
            for k, v in judgments.items():
                if isinstance(v, dict):
                    if "block_id" not in v:
                        v["block_id"] = k
                    temp_list.append(v)
                else:
                    temp_list.append({"block_id": k, "is_reviewer_wrong": bool(v)})
            judgments = temp_list

        if not isinstance(judgments, list):
            raise ValueError(f"Judgments must be a list, got {type(judgments)}")

        judgment_map = {}
        for idx, j in enumerate(judgments):
            block_id_raw = j.get("block_id")
            if not block_id_raw:
                logger.warning(f"Judgment #{idx} missing block_id; skipping: {j}")
                continue

            normalized_key = normalize_block_id(block_id_raw)

            if normalized_key in judgment_map:
                logger.warning(f"Duplicate block_id detected: {block_id_raw} (normalized: {normalized_key})")

            judgment_map[normalized_key] = j

        results = []
        for block_id in expected_block_ids:
            normalized_expected = normalize_block_id(block_id)

            if normalized_expected in judgment_map:
                j = judgment_map[normalized_expected]
                results.append({
                    "is_reviewer_wrong": bool(j.get("is_reviewer_wrong", False)),
                    "parse_success": True,
                    "raw_block_id": j.get("block_id")
                })
            else:
                logger.warning(
                    f"No validation result for block_id '{block_id}' (normalized: '{normalized_expected}'). "
                    f"Available normalized keys: {list(judgment_map.keys())}"
                )
                results.append({
                    "is_reviewer_wrong": False,
                    "parse_success": False,
                    "raw_block_id": None
                })

        return results

    except Exception as e:
        logger.error(f"Failed to parse response: {e}")
        logger.error(f"Response preview: {response[:500]}...")
        logger.error(traceback.format_exc())

        return [
            {
                "is_reviewer_wrong": False,
                "parse_success": False,
                "raw_block_id": None
            }
            for _ in expected_block_ids
        ]


def process_single_reviewer_batch(
    record_id: str,
    reviewer_name: str,
    opinions: List[Dict[str, Any]],
    meta_review: str,
    client: LLMClient,
    effort: str
) -> Tuple[List[Dict[str, Any]], int]:
    """Batch-process all intersected refuted opinions for a single reviewer."""
    if not opinions:
        return [], 0

    # Use batch logic even for a single opinion (consistent interface).
    author_response = opinions[0]["author_responses_combined"]

    prompt = validate_author_conflict_prompt(
        meta_review=meta_review,
        author_response=author_response,
        reviewer_opinions=opinions
    )

    system_prompt = validate_author_conflict_system_prompt()

    try:
        key = client.submit_task(
            prompt,
            system_prompt=system_prompt,
            temperature=0.6,
            reasoning_effort=effort
        )

        result = client.get_result(key)
        if result.get("content"):
            expected_block_ids = [op.get("block_id") for op in opinions]
            parsed_results = parse_batch_validation_response(result["content"], expected_block_ids)

            batch_results = []
            for opinion, validation_result in zip(opinions, parsed_results):
                batch_results.append({
                    "record_id": record_id,
                    "reviewer_name": reviewer_name,
                    "opinion_text": opinion["text"],
                    "block_id": opinion["block_id"],
                    "validation_result": validation_result,
                    "llm_call_success": True
                })

            return batch_results, 1
        else:
            raise ValueError("Empty LLM response")

    except Exception as e:
        logger.error(f"Validation LLM call failed for record {record_id}, reviewer {reviewer_name}: {e}")

        error_results = []
        for opinion in opinions:
            error_results.append({
                "record_id": record_id,
                "reviewer_name": reviewer_name,
                "opinion_text": opinion["text"],
                "block_id": opinion["block_id"],
                "validation_result": {
                    "is_reviewer_wrong": False,
                    "parse_success": False
                },
                "llm_call_success": False,
                "error": str(e)
            })

        return error_results, 1


def process_single_record(
    record: Dict[str, Any],
    client: LLMClient,
    effort: str
) -> Tuple[Dict[str, Any], int]:
    """Process one record's intersected refuted opinions (batched by reviewer)."""
    record_id = record["record_id"]
    meta_review = record["meta_review"]
    total_llm_calls = 0

    logger.info(f"Processing record {record_id} - {len(record['reviewers'])} reviewers have intersected refuted opinions")

    validation_results = defaultdict(lambda: {
        "refuted_opinions_validated": [],
        "total_opinions": 0,
        "summary": {
            "total_refuted": 0,
            "reviewer_actually_wrong": 0,
        }
    })

    for reviewer_name, reviewer_data in record["reviewers"].items():
        opinions_to_validate = reviewer_data["refuted_opinions"]

        logger.info(f"  Reviewer '{reviewer_name}': {len(opinions_to_validate)} intersected refuted opinions")

        batch_results, calls = process_single_reviewer_batch(
            record_id=record_id,
            reviewer_name=reviewer_name,
            opinions=opinions_to_validate,
            meta_review=meta_review,
            client=client,
            effort=effort
        )

        validation_results[reviewer_name]["refuted_opinions_validated"].extend(batch_results)
        validation_results[reviewer_name]["total_opinions"] = reviewer_data.get("total_opinions", 0)
        total_llm_calls += calls

    for reviewer_name, data in validation_results.items():
        opinions = data["refuted_opinions_validated"]
        if opinions:
            wrong_count = sum(1 for op in opinions if op["validation_result"]["is_reviewer_wrong"])

            data["summary"] = {
                "total_refuted": len(opinions),
                "reviewer_actually_wrong": wrong_count,
                "wrong_rate": wrong_count / len(opinions)
            }

    updated_record = record["original_record"]
    updated_record["reviewer_opinion_validation"] = dict(validation_results)
    updated_record["validation_stats"] = {
        "total_llm_calls": total_llm_calls,
        "total_refuted_opinions_processed": sum(
            len(r["refuted_opinions_validated"])
            for r in validation_results.values()
        )
    }

    return updated_record, total_llm_calls


def process_all_records(
    input_path1: str,
    input_path2: str,
    output_path: str,
    client: LLMClient,
    effort: str,
    max_workers: int = 16
) -> None:
    """Run validation over all eligible records."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    records = load_refutation_results(input_path1, input_path2)

    if not records:
        logger.warning("No records to process")
        return

    # Optional test slicing (keep commented out):
    # records = records[:180]

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
                logger.info(f"Finished: {result.get('id', 'unknown')} - {calls} LLM call(s)")
            except Exception as e:
                logger.error(
                    f"Error type: {type(e).__name__}\n"
                    f"Error message: {str(e)}\n"
                    f"Traceback:\n{traceback.format_exc()}"
                )
                record = future_to_record[future]
                processed_records.append(record["original_record"])

    processed_records.sort(key=lambda x: x.get('id', 0))

    with open(output_path, 'w', encoding='utf-8') as f:
        for record in processed_records:
            json.dump(record, f, ensure_ascii=False)
            f.write('\n')

    generate_summary(processed_records, output_path)

    logger.info(f"Done! Total LLM calls: {total_llm_calls}")
    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 2: Validate whether intersected refuted reviewer opinions are actually wrong (intersection of two models)."
    )
    parser.add_argument("--input_path1", default=None,
                        help="Path to the first model output JSONL")
    parser.add_argument("--input_path2", default=None,
                        help="Path to the second model output JSONL")
    parser.add_argument("--output_path", help="Output path (optional; auto-generated if not provided)")
    BaseArguments.add_to_parser(parser, model_default="deepseek-reasoner", effort_default="medium")

    args = parser.parse_args()
    BaseArguments.apply(args)

    input_path1 = args.input_path1
    if input_path1 is None:
        input_path1 = f"{args.paper_series}/refutations/{args.paper_series}_gpt-5-mini_medium_split_clean_refutations.jsonl"
    input_path2 = args.input_path2
    if input_path2 is None:
        input_path2 = f"{args.paper_series}/refutations/{args.paper_series}_gemini-2.5-flash_medium_split_clean_refutations.jsonl"
    if args.output_path:
        output_path = args.output_path
    else:
        model_name = args.model.replace("/", "-")
        output_path = input_path1.replace("_refutations.jsonl", "_validated.jsonl")
        output_path = output_path.replace("refutations/", "validations/")
        dir_name = os.path.dirname(output_path)
        base_name = os.path.basename(output_path)
        output_path = os.path.join(dir_name, f"intersection_{model_name}_{base_name}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    client = LLMClient(
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model,
        max_workers=args.max_workers,
        cache_version=args.cache_version
    )

    process_all_records(
        input_path1=input_path1,
        input_path2=input_path2,
        output_path=output_path,
        client=client,
        effort=args.effort,
        max_workers=args.max_workers
    )
    client.shutdown()
