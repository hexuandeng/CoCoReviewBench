#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose
-------
Assign fine-grained “discussion point” IDs to reviewer comments using an LLM, then
reformat/group the results for manual inspection.

Point ID assignment policy
--------------------------
1) Identify the most specific core subject discussed in each reviewer opinion
   (e.g., “SGD learning rate value”, “F1-score metric”, “Random Forest selection”).
2) Reuse the same point_id for opinions that discuss the exact same subject,
   even if the stance is contradictory.
3) Use different point_ids for different specific subjects, even if related.

Outputs
-------
Stage 1 (JSONL):
- discussion_point_assignments: per-record mapping of label -> {blocks: [...]}
- llm_call_stats: basic call statistics for that record

Stage 2 (JSON + TXT summary):
- A grouped JSON for review: groups by (record_id, label, point_id)
- A human-readable summary text file alongside the JSON

Notes
-----
- This script is designed to run on existing preprocessed JSONL inputs.
- Provide your own API endpoint and key via CLI flags; do NOT hardcode secrets.
- Parallelism can increase request volume; be mindful of provider rate limits.
- Default paths assume a particular dataset layout; for open-source usage, pass
  explicit paths via: --input_jsonl, --point_ids_jsonl, --output_json
"""

import argparse
import json
import re
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from prompt_registry import review_clustering_system_prompt, review_clustering_prompt
from src.utils import BaseArguments, LLMClient

logger = logging.getLogger(__name__)

# =============================================================================
# Stage 1: Assign discussion point IDs (LLM)
# =============================================================================

def extract_role_texts(
    block: List[Any],
    sentence_texts: List[str],
    role_prefix: str = "Reviewer"
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Extract per-role texts from one annotated block.

    Expected block format (from `new_split_texts`):
      [
        [ [role_name, [sentence_ids...]], ... ],
        [label_info...]
      ]

    Args:
      block:
        A single element from `new_split_texts`.
      sentence_texts:
        Global list of sentence strings; indices in sentence_ids refer here.
      role_prefix:
        Roles starting with this prefix are treated as reviewers.

    Returns:
      (reviewer_texts, other_texts)
      - reviewer_texts: {reviewer_name: combined_text}
      - other_texts:    {other_role_name: combined_text}

    Notes:
      - Sentence IDs are sorted to restore natural reading order.
      - Out-of-range sentence IDs are ignored.
    """
    reviewer_texts = {}
    other_texts = {}

    for role_item in block[0]:  # block[0] is the role list
        role_name = role_item[0]
        sentence_ids = role_item[1]

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


def parse_point_assignment_response(response: str) -> Dict[str, Any]:
    """
    Parse the LLM JSON response for point_id assignments.

    Expected structure:
      {"blocks": [{"block_id": ..., "reviewer_assignments": {...}}, ...]}

    Robustness:
      - Tolerates code fences (```json ... ```).
      - Falls back to extracting the first JSON object-like span.
      - Returns {"blocks": []} on any parsing failure.
    """
    response = re.sub(r'^```json\s*', '', response)
    response = re.sub(r'\s*```$', '', response)
    response = response.strip()

    try:
        data = json.loads(response, strict=False)
        if "blocks" in data:
            return data
        return {"blocks": []}
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from model output: {e}")
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                if "blocks" in data:
                    return data
                return {"blocks": []}
            except Exception:
                pass
        return {"blocks": []}


def process_single_record(
    record: Dict[str, Any],
    client: LLMClient,
    effort: str
) -> Tuple[Dict[str, Any], int]:
    """
    Process a single record and assign discussion point IDs.

    Inputs (expected in `record`):
      - new_split_texts: block-structured annotations
      - sentence_texts: full sentence strings

    Behavior:
      - Point IDs are assigned independently within each label bucket.
      - If a label bucket contains exactly one reviewer block, assign point_id=1
        without calling the LLM (fast path).

    Returns:
      (processed_record, num_llm_calls_for_this_record)
    """
    new_split_texts = record.get("new_split_texts", [])
    sentence_texts = record.get("sentence_texts", [])

    record_id = record.get("id")

    if not new_split_texts or not sentence_texts:
        logger.warning(f"Record {record_id} missing required fields: new_split_texts and/or sentence_texts")
        return record, 0

    # Group blocks by label (e.g., "CLAR-WRT") so point_ids are assigned within each label bucket.
    label_to_blocks = defaultdict(list)
    for block_idx, block in enumerate(new_split_texts):
        if not isinstance(block, list) or len(block) != 2:
            continue

        labels = block[1]
        if not isinstance(labels, list) or not labels:
            continue

        label = labels[0]
        if label == "N/A":
            continue

        reviewer_texts, author_texts = extract_role_texts(block, sentence_texts)

        if not reviewer_texts:
            continue

        label_to_blocks[label].append({
            "block_idx": block_idx,
            "reviewer_texts": reviewer_texts,
            "author_texts": author_texts,
            "original_block": block
        })

    if not label_to_blocks:
        return record, 0

    point_assignments = {}
    total_llm_calls = 0

    for label, blocks in label_to_blocks.items():
        logger.info(f"Record {record_id} | label='{label}' | reviewer_blocks={len(blocks)}")

        # Fast path: one block in this label -> assign point_id=1 without calling the LLM.
        if len(blocks) == 1:
            block = blocks[0]
            reviewer_name = next(iter(block["reviewer_texts"].keys()))
            point_assignments[label] = {
                "blocks": [
                    {
                        "block_id": block["block_idx"],
                        "reviewer_assignments": {
                            reviewer_name: 1
                        }
                    }
                ]
            }
            logger.info(f"Record {record_id} | label='{label}' | fast-path (single block): assigned point_id=1")
            continue

        prompt = review_clustering_prompt(blocks)

        try:
            system_prompt = review_clustering_system_prompt()

            key = client.submit_task(
                prompt,
                system_prompt=system_prompt,
                temperature=0.6,
                reasoning_effort=effort
            )

            result = client.get_result(key)
            total_llm_calls += 1

            if result.get("content"):
                parsed_result = parse_point_assignment_response(result["content"])
                point_assignments[label] = parsed_result
            else:
                logger.error(f"Record {record_id} | label='{label}' | empty model response")
                point_assignments[label] = {"blocks": []}

        except Exception as e:
            logger.error(f"Record {record_id} | label='{label}' | model call failed: {e}")
            point_assignments[label] = {"blocks": []}

    record["discussion_point_assignments"] = point_assignments
    record["llm_call_stats"] = {
        "num_labels_processed": len(label_to_blocks),
        "total_calls": total_llm_calls
    }

    return record, total_llm_calls


def process_all_records(
    input_path: str,
    output_path: str,
    client: LLMClient,
    effort: str,
    max_workers: int = 32
) -> None:
    """
    Process all records from a JSONL input and write a JSONL output including
    `discussion_point_assignments`.

    Args:
      input_path:  JSONL input file
      output_path: JSONL output file (Stage 1)
      client:      LLM client wrapper
      effort:      reasoning effort parameter forwarded to the client
      max_workers: thread pool parallelism
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    records = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.error(f"Input JSONL parse error at line {line_num}: {e}")
                continue

    logger.info(f"Loaded {len(records)} records from: {input_path}")

    processed_records = []
    total_llm_calls = 0

    logger.info(f"Stage 1 starting | workers={max_workers} | effort='{effort}' | output={output_path}")

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
                record = future_to_record[future]
                logger.error(f"Record processing failed (record_id={record.get('id')}): {e}")
                processed_records.append(record)

    processed_records.sort(key=lambda x: x.get('id', 0))

    with open(output_path, 'w', encoding='utf-8') as f:
        for record in processed_records:
            json.dump(record, f, ensure_ascii=False)
            f.write('\n')

    logger.info(f"Stage 1 completed | total_llm_calls={total_llm_calls} | saved={output_path}")


# =============================================================================
# Stage 2: Format point_id assignments for manual review
# =============================================================================

def extract_reviewer_texts(
    block: List[Any],
    sentence_texts: List[str],
    role_prefix: str = "Reviewer"
) -> Dict[str, str]:
    """
    Extract reviewer texts from a block.

    Returns:
      {reviewer_name: combined_text}

    Notes:
      - Sentence IDs are sorted to restore natural reading order.
      - Out-of-range sentence IDs are ignored.
    """
    reviewer_texts = {}

    if not isinstance(block, list) or len(block) != 2:
        return reviewer_texts

    for role_item in block[0]:  # block[0] is the role list
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
            reviewer_texts[role_name] = combined_text

    return reviewer_texts


def get_block_label(block: List[Any]) -> str:
    """
    Extract the label from a block.

    Convention:
      new_split_texts[block_id][1] is a label list such as ["CLAR-WRT"].
    """
    if not isinstance(block, list) or len(block) != 2:
        return None

    label_info = block[1]
    if isinstance(label_info, list) and len(label_info) > 0:
        return label_info[0]
    return None


def load_and_group_by_point_id(input_path: str, model_name: str) -> List[Dict[str, Any]]:
    """
    Load Stage 1 results and group comments by (record_id, label, point_id).

    Coverage rule:
      - Ensure every reviewer block is represented.
      - Any unclassified reviewer comment becomes its own singleton group.

    Returns:
      A flat list of group entries, each with:
        - record_id, label, point_id (or None), model_name
        - num_comments, comments[]
        - is_classified: True/False
    """
    grouped_results = []

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line, strict=False)
                    record_id = record.get("id", f"unknown_{line_num}")

                    sentence_texts = record.get("sentence_texts", [])
                    new_split_texts = record.get("new_split_texts", [])
                    point_assignments = record.get("discussion_point_assignments", {})

                    if not point_assignments:
                        logger.warning(f"Stage 2: record {record_id} has no discussion_point_assignments; skipping")
                        continue

                    if not new_split_texts:
                        logger.warning(f"Stage 2: record {record_id} has no new_split_texts; skipping")
                        continue

                    # Map: block_id -> list of (reviewer_name, point_id, actual_label)
                    block_to_point_assignments = defaultdict(list)
                    global_classified_block_ids: Set[int] = set()

                    for label, label_data in point_assignments.items():
                        blocks_info = label_data.get("blocks", [])

                        for block_info in blocks_info:
                            try:
                                block_id = int(block_info.get("block_id"))
                                reviewer_assignments = block_info.get("reviewer_assignments", {})

                                if block_id >= len(new_split_texts):
                                    logger.warning(
                                        f"Stage 2: record {record_id} block_id {block_id} out of range "
                                        f"(max index: {len(new_split_texts)-1}); skipping"
                                    )
                                    continue

                                actual_label = get_block_label(new_split_texts[block_id])
                                if actual_label != label:
                                    logger.warning(
                                        f"Stage 2: record {record_id} block_id {block_id} label mismatch: "
                                        f"assigned_label={label}, actual_label={actual_label}"
                                    )

                                for reviewer_name, point_id in reviewer_assignments.items():
                                    try:
                                        if isinstance(point_id, list) and len(point_id) > 0:
                                            point_id = int(point_id[0])
                                        elif isinstance(point_id, list) and len(point_id) == 0:
                                            logger.warning(f"Stage 2: record {record_id} invalid point_id: {point_id}")
                                            continue
                                        elif isinstance(point_id, dict):
                                            point_id = int(list(point_id.values())[0])
                                        else:
                                            point_id = int(point_id)
                                    except (ValueError, TypeError):
                                        logger.warning(f"Stage 2: record {record_id} invalid point_id: {point_id}")
                                        continue

                                    block_to_point_assignments[block_id].append(
                                        (reviewer_name, point_id, actual_label)
                                    )
                                    global_classified_block_ids.add(block_id)

                            except (ValueError, TypeError):
                                logger.warning(
                                    f"Stage 2: record {record_id} invalid block_id: {block_info.get('block_id')}"
                                )
                                continue

                    # For each label: emit classified groups + unclassified singleton groups.
                    for label, label_data in point_assignments.items():
                        # 1) Classified groups: group by point_id
                        point_groups = defaultdict(list)

                        for block_id, assignments in block_to_point_assignments.items():
                            label_assignments = [
                                (reviewer_name, point_id)
                                for reviewer_name, point_id, actual_label in assignments
                                if actual_label == label
                            ]

                            if not label_assignments:
                                continue

                            if block_id >= len(new_split_texts):
                                logger.warning(
                                    f"Stage 2: record {record_id} block_id {block_id} out of range "
                                    f"(max index: {len(new_split_texts)-1}); skipping"
                                )
                                continue

                            reviewer_texts = extract_reviewer_texts(
                                new_split_texts[block_id],
                                sentence_texts
                            )

                            for reviewer_name, point_id in label_assignments:
                                if reviewer_name not in reviewer_texts:
                                    logger.warning(
                                        f"Stage 2: record {record_id} assigned reviewer '{reviewer_name}' "
                                        f"(block_id={block_id}) has no extracted text; skipping"
                                    )
                                    continue

                                point_groups[point_id].append({
                                    "block_id": block_id,
                                    "reviewer_name": reviewer_name,
                                    "text": reviewer_texts[reviewer_name]
                                })

                        for point_id, comments in point_groups.items():
                            if not comments:
                                continue

                            entry = {
                                "record_id": record_id,
                                "label": label,
                                "point_id": int(point_id),
                                "model_name": model_name,
                                "num_comments": len(comments),
                                "comments": comments,
                                "is_classified": True
                            }
                            grouped_results.append(entry)

                        # 2) Unclassified groups: each reviewer comment becomes its own group
                        label_block_ids = {
                            block_id for block_id, block in enumerate(new_split_texts)
                            if get_block_label(block) == label
                        }

                        unclassified_block_ids = label_block_ids - global_classified_block_ids

                        for block_id in sorted(unclassified_block_ids):
                            reviewer_texts = extract_reviewer_texts(
                                new_split_texts[block_id],
                                sentence_texts
                            )

                            for reviewer_name, text in reviewer_texts.items():
                                entry = {
                                    "record_id": record_id,
                                    "label": label,
                                    "point_id": None,
                                    "model_name": model_name,
                                    "num_comments": 1,
                                    "comments": [{
                                        "block_id": block_id,
                                        "reviewer_name": reviewer_name,
                                        "text": text
                                    }],
                                    "is_classified": False
                                }
                                grouped_results.append(entry)

                        logger.info(
                            f"Stage 2: record {record_id} | label='{label}' | "
                            f"blocks_in_label={len(label_block_ids)} | "
                            f"classified_blocks={len(label_block_ids & global_classified_block_ids)} | "
                            f"unclassified_blocks={len(unclassified_block_ids)} | "
                            f"classified_point_groups={len(point_groups)}"
                        )

                except json.JSONDecodeError as e:
                    logger.error(f"Stage 2: JSON parse failed at line {line_num}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Stage 2: error processing line {line_num}: {e}", exc_info=True)
                    continue

    except FileNotFoundError:
        logger.error(f"Stage 2 input not found: {input_path}")
        sys.exit(1)

    logger.info(f"Loaded Stage 1 results: groups={len(grouped_results)} | model={model_name} | input={input_path}")
    return grouped_results


def format_groups_to_outputs(
    input_jsonl: str,
    output_json: str,
    model_name: str,
    indent: int = 2,
    min_comments: int = 1
) -> Tuple[str, Dict[str, Any]]:
    """
    Load grouped results and write:
      - A formatted JSON file (output_json)
      - A human-readable summary TXT file (output_json with _summary.txt)

    Args:
      input_jsonl:  Stage 1 output JSONL
      output_json:  Stage 2 output JSON
      model_name:   Stored in output metadata
      indent:       JSON indentation
      min_comments: Filter out groups smaller than this size

    Returns:
      (summary_path, stats)
    """
    os.makedirs(os.path.dirname(output_json) if os.path.dirname(output_json) else ".", exist_ok=True)

    grouped_results = load_and_group_by_point_id(input_jsonl, model_name)

    filtered_results = [r for r in grouped_results if r["num_comments"] >= min_comments]
    filtered_results.sort(key=lambda x: (
        x["record_id"],
        x["label"],
        (x["point_id"] if x["point_id"] is not None else float('inf'))
    ))

    stats = {
        "total_point_groups": len(grouped_results),
        "filtered_point_groups": len(filtered_results),
        "total_comments": sum(r["num_comments"] for r in grouped_results),
        "total_classified_groups": sum(1 for r in grouped_results if r.get("is_classified")),
        "total_unclassified_groups": sum(1 for r in grouped_results if not r.get("is_classified")),
        "model_name": model_name,
        "min_comments_threshold": min_comments
    }

    output_data = {
        "metadata": {
            "stats": stats,
            "filter": {"min_comments": min_comments}
        },
        "point_groups": filtered_results
    }

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=indent)

    summary_path = output_json.replace(".json", "_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"Discussion point ID review summary (model: {model_name})\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Groups (after filter): {stats['filtered_point_groups']}\n")
        f.write(f"  - Classified groups: {stats['total_classified_groups']}\n")
        f.write(f"  - Unclassified groups: {stats['total_unclassified_groups']}\n")
        f.write(f"Total comments (all groups): {stats['total_comments']}\n")
        f.write(f"Filter: min_comments >= {min_comments}\n")
        f.write("=" * 80 + "\n\n")

        current_record = None
        current_label = None

        for group in filtered_results:
            if group["record_id"] != current_record:
                current_record = group["record_id"]
                f.write(f"\n{'#' * 80}\n")
                f.write(f"# Record ID: {current_record}\n")
                f.write(f"{'#' * 80}\n")

            if group["label"] != current_label:
                current_label = group["label"]
                f.write(f"\n## Label: {current_label}\n")
                f.write("-" * 80 + "\n")

            point_id_str = group["point_id"] if group["point_id"] is not None else "UNCATEGORIZED"
            classification_status = "[CLASSIFIED]" if group.get("is_classified") else "[UNCLASSIFIED]"

            f.write(f"\n### Point ID: {point_id_str} {classification_status} ({group['num_comments']} comments)\n")
            f.write("-" * 80 + "\n")

            for comment in group["comments"]:
                f.write(f"\n**Block {comment['block_id']} | {comment['reviewer_name']}:**\n")
                for line in comment["text"].split("\n"):
                    f.write(f"> {line}\n")
                f.write("\n")

            f.write("\n" + "=" * 80 + "\n")

    logger.info(
        f"Stage 2 completed | total_groups={stats['total_point_groups']} | "
        f"filtered_groups={stats['filtered_point_groups']} | "
        f"min_comments>={min_comments} | "
        f"json={output_json} | summary={summary_path}"
    )

    return summary_path, stats


# =============================================================================
# Combined main: run Stage 1 -> Stage 2 on the same paper_series
# =============================================================================

def build_default_paths(paper_series: str, model_name: str) -> Tuple[str, str, str]:
    """
    Build default file paths.

    IMPORTANT:
      These defaults assume a specific on-disk dataset layout.
      This script does not currently expose CLI overrides for these paths.
      Change the defaults in code if your local layout differs.

    Notes:
      - Stage 1 input naming convention expects 'gpt-5-mini_medium' in the filename.
      - Stage 1 output path is derived from input path and model name (sanitized).
    """
    # Preserve the original Stage-1 input naming convention:
    # it expects 'gpt-5-mini_medium' in the input filename.
    input_path = f"{paper_series}/split_final/{paper_series}_gpt-5-mini_medium_split_clean.jsonl"
    output_path_stage1 = f"{paper_series}/point_ids/{paper_series}_{model_name.replace("/", "-")}_medium_split_clean_point_ids.jsonl"
    output_path_stage2 = f"{paper_series}/visual_output/assign_{model_name.replace('/', '-')}_point_ids.json"

    return input_path, output_path_stage1, output_path_stage2


def main():
    parser = argparse.ArgumentParser(
        description="Assign discussion point IDs (LLM) and format results for manual review."
    )

    # Shared + LLM
    parser.add_argument("--log_file", default="logs/assign_points.log", help="Path to the log file")
    BaseArguments.add_to_parser(parser, model_default="deepseek-reasoner", effort_default="medium")

    # Flow control
    parser.add_argument("--skip_assign", action="store_true", help="Skip Stage 1 (assumes point_ids_jsonl exists)")
    parser.add_argument("--skip_format", action="store_true", help="Skip Stage 2 formatting")
    parser.add_argument("--min_comments", type=int, default=1, help="Filter groups with fewer comments than this")

    args = parser.parse_args()
    BaseArguments.apply(args)

    # Logging configuration
    os.makedirs(os.path.dirname(args.log_file) if os.path.dirname(args.log_file) else ".", exist_ok=True)
    input_path, point_ids_path, output_json = build_default_paths(args.paper_series, args.model)

    # Stage 1
    if not args.skip_assign:
        os.makedirs(os.path.dirname(point_ids_path), exist_ok=True)

        client = LLMClient(
            api_key=args.api_key,
            base_url=args.base_url,
            model_name=args.model,
            max_workers=args.max_workers,
            cache_version=args.cache_version
        )

        process_all_records(
            input_path=input_path,
            output_path=point_ids_path,
            client=client,
            effort=args.effort,
            max_workers=args.max_workers,
        )
        client.shutdown()
    else:
        logger.info("Stage 1 skipped (--skip_assign set)")

    # Stage 2
    if not args.skip_format:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        format_groups_to_outputs(
            input_jsonl=point_ids_path,
            output_json=output_json,
            model_name=args.model.replace("/", "-"),
            min_comments=args.min_comments
        )
    else:
        logger.info("Stage 2 skipped (--skip_format set)")


if __name__ == "__main__":
    main()
