#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose
-------
Format and merge multi-model outputs for:
1) reviewer-opinion refutation validation, and
2) conflict-resolution adjudication,
then assemble a final JSONL dataset with per-block validation labels.

Pipeline
--------
1) Refutation validation formatting
   - Read per-model validation JSONL.
   - Extract meta-review text safely.
   - Normalize block_id variants for cross-model matching.
   - Merge results per (record_id, reviewer_name) and compute simple consensus stats.
   - Write a JSON summary plus a human-readable TXT excerpt.

2) Conflict resolution formatting
   - Read per-model conflict-resolution JSONL.
   - Extract meta-review text safely.
   - Merge results per (record_id, label, point_id) and compute per-block vote counts.
   - Write a JSON summary plus a human-readable TXT excerpt.

3) Final dataset assembly
   - Start from split_final JSONL (per-paper blocks).
   - Attach grouping info from assign_point_ids.json.
   - Attach:
     - conflicts_validation: per-block labels from conflict resolutions
     - rebuttal_validation: per-block labels from refutation validations
   - Write a final JSONL suitable for downstream analysis/visualization.

Output artifacts
----------------
- refutation_validation_results_for_review.json (+ _summary.txt)
- conflict_resolutions_for_review.json (+ _summary.txt)
- *_with_reviewer_validations.jsonl

Notes
-----
- This script is a formatter/merger only; it does NOT run any LLM calls.
- Defaults assume a local AIReviewBenchmark directory layout.
- Logging is designed to be informative but not overly verbose.
"""

import argparse
import json
import logging
import sys
import os
import re
from typing import Dict, Any, List, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


# ============================================================
# Part A: format_validate_author_refutations_for_review.py
# ============================================================

def normalize_block_id(block_id: Any) -> str:
    """
    Extract the numeric core from block_id and normalize common variants.

    Examples
    --------
    - 12            -> "12"
    - "block_12"    -> "12"
    - "Block 12"    -> "12"
    - "b12-extra"   -> "12"
    """
    s = str(block_id).lower().replace(" ", "")
    match = re.search(r"\d+", s)
    return match.group() if match else s


def extract_meta_review_text_from_metareview(meta_review: Any, record_id: Any) -> str:
    """
    Safely extract meta-review text from OpenReview-style structures.

    This preserves the original access pattern (nested list + dict), with broad
    error handling to avoid breaking formatting.
    """
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
    except IndexError:
        meta_review_text = ""
        logger.warning(f"Record {record_id}: meta-review text not found (IndexError).")
    except KeyError:
        meta_review_text = ""
        logger.warning(f"Record {record_id}: meta-review key not found (KeyError).")
    except TypeError:
        meta_review_text = ""
        logger.warning(f"Record {record_id}: unexpected meta-review type (TypeError).")
    return meta_review_text


def safe_truncate(text: str, max_length: int = 150) -> str:
    """
    Truncate text safely without cutting too aggressively mid-word.
    """
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    truncated = text[:max_length]
    if len(truncated) < len(text) and not text[len(truncated):].startswith((" ", "\n", ".", ",")):
        last_space = truncated.rfind(" ")
        if last_space > max_length * 0.8:
            truncated = truncated[:last_space]
    return truncated + "..."


def load_validation_results(model_name: str, paper_series: str) -> List[Dict[str, Any]]:
    """
    Load validation results for a single model.

    Expected input
    --------------
    JSONL file containing records with keys such as:
    - id / record_id
    - reviewer_opinion_validation
    - metareview
    - author_refutation_analysis
    """
    input_path = f"{paper_series}/validations/intersection_{model_name}_{paper_series}_gpt-5-mini_medium_split_clean_validated.jsonl"
    results: List[Dict[str, Any]] = []

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    record_id = record.get("id", record.get("record_id", f"unknown_{line_num}"))

                    validation_data = record.get("reviewer_opinion_validation", {})
                    meta_review = extract_meta_review_text_from_metareview(record.get("metareview", ""), record_id)

                    refutation_analysis = record.get("author_refutation_analysis", {})

                    if not validation_data:
                        logger.warning(f"Record {record_id}: missing reviewer_opinion_validation.")
                        continue

                    for reviewer_name, reviewer_validation_data in validation_data.items():
                        reviewer_refutation_data = refutation_analysis.get(reviewer_name, {})
                        author_responses_combined = reviewer_refutation_data.get("author_responses_combined", "")

                        opinions = reviewer_validation_data.get("refuted_opinions_validated", [])
                        if not opinions:
                            continue

                        processed_opinions = []
                        for opinion in opinions:
                            raw_block_id = opinion.get("block_id")
                            normalized_id = normalize_block_id(raw_block_id)

                            validation_result = opinion.get("validation_result", {})
                            processed_opinions.append({
                                "block_id": raw_block_id,  # keep original for display
                                "block_id_normalized": normalized_id,  # normalized for internal matching
                                "text": opinion.get("opinion_text", ""),
                                "is_reviewer_wrong": validation_result.get("is_reviewer_wrong", False),
                                "llm_call_success": opinion.get("llm_call_success", False),
                            })

                        review_entry = {
                            "record_id": record_id,
                            "reviewer_name": reviewer_name,
                            "meta_review": meta_review,
                            "author_responses_combined": author_responses_combined,
                            "num_opinions": len(processed_opinions),
                            "opinions": processed_opinions,
                            "model": model_name,
                            "validation_summary": reviewer_validation_data.get("summary", {}),
                        }

                        results.append(review_entry)

                except json.JSONDecodeError as e:
                    logger.error(f"Line {line_num}: JSON decode failed: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Line {line_num}: unexpected error: {e}")
                    logger.error(f"Record ID: {record_id}")
                    continue

    except FileNotFoundError as e:
        logger.error(f"Input file not found: {input_path}")
        logger.error(f"Error: {e}")
        sys.exit(1)

    logger.info(f"Loaded validation results | model={model_name} | reviewer_entries={len(results)}")
    return results


def merge_multi_model_results(model_results: Dict[str, List[Dict]]) -> List[Dict[str, Any]]:
    """
    Merge validation results across models, grouped by (record_id, reviewer_name).

    Within each group, opinions are matched by:
    - normalized_block_id
    - cleaned opinion text
    """
    merged_groups = defaultdict(lambda: {
        "record_id": "",
        "reviewer_name": "",
        "meta_review": "",
        "author_responses_combined": "",
        "models": set(),
        "opinion_results": {}  # key: (normalized_block_id, opinion_text) -> aggregated validations
    })

    for model_name, records in model_results.items():
        for record in records:
            key = (record["record_id"], record["reviewer_name"])

            if not merged_groups[key]["record_id"]:
                merged_groups[key]["record_id"] = record["record_id"]
                merged_groups[key]["reviewer_name"] = record["reviewer_name"]
                merged_groups[key]["meta_review"] = record["meta_review"]
                merged_groups[key]["author_responses_combined"] = record["author_responses_combined"]

            merged_groups[key]["models"].add(model_name)

            if merged_groups[key]["author_responses_combined"] != record["author_responses_combined"]:
                logger.warning(
                    f"Record {key}: author_responses_combined differs across models; keeping the first seen value."
                )

            for opinion in record["opinions"]:
                normalized_block_id = opinion.get("block_id_normalized", normalize_block_id(opinion.get("block_id")))
                clean_text = opinion["text"].strip().replace("\r\n", "\n")
                opinion_key = (normalized_block_id, clean_text)

                if opinion_key not in merged_groups[key]["opinion_results"]:
                    merged_groups[key]["opinion_results"][opinion_key] = {
                        "block_id": opinion["block_id"],  # keep original for output
                        "block_id_normalized": normalized_block_id,
                        "text": opinion["text"],
                        "validations_by_model": {}
                    }

                merged_groups[key]["opinion_results"][opinion_key]["validations_by_model"][model_name] = {
                    "is_reviewer_wrong": opinion["is_reviewer_wrong"],
                    "llm_call_success": opinion["llm_call_success"]
                }

    merged_list = []
    for _, group in merged_groups.items():
        opinions_list = []
        conflict_count = 0
        total_multi_model = 0
        wrong_consensus_count = 0

        for data in group["opinion_results"].values():
            validations = list(data["validations_by_model"].values())
            num_models = len(validations)
            wrong_values = [v["is_reviewer_wrong"] for v in validations]

            if num_models >= 2:
                all_agree_wrong = len(set(wrong_values)) == 1
                wrong_agreement_rate = sum(wrong_values) / num_models
                conflict_count += 0 if all_agree_wrong else 1
                total_multi_model += 1

                consensus_wrong = wrong_agreement_rate >= 0.5
                if consensus_wrong:
                    wrong_consensus_count += 1
            else:
                all_agree_wrong = True
                wrong_agreement_rate = 1.0 if wrong_values[0] else 0.0
                if wrong_values[0]:
                    wrong_consensus_count += 1

            num_wrong = sum(wrong_values)

            opinions_list.append({
                "block_id": data["block_id"],
                "text": data["text"],
                "validations_by_model": data["validations_by_model"],
                "consensus": {
                    "all_agree_is_wrong": all_agree_wrong,
                    "wrong_agreement_rate": wrong_agreement_rate,
                    "num_models": num_models,
                    "num_marked_as_wrong": num_wrong,
                    "consensus_is_wrong": wrong_agreement_rate >= 0.5 if num_models >= 2 else wrong_values[0]
                }
            })

        def sort_key(x):
            try:
                match = re.search(r"(\d+)", str(x["block_id"]))
                return int(match.group(1)) if match else 0
            except Exception:
                return str(x["block_id"])

        opinions_list.sort(key=sort_key)

        total_opinions = len(opinions_list)
        consensus_rate = (total_multi_model - conflict_count) / total_multi_model if total_multi_model > 0 else 1.0

        entry = {
            "record_id": group["record_id"],
            "reviewer_name": group["reviewer_name"],
            "meta_review": group["meta_review"],
            "author_responses_combined": group["author_responses_combined"],
            "num_models": len(group["models"]),
            "model_names": sorted(list(group["models"])),
            "opinions": opinions_list,
            "consensus_summary": {
                "total_opinions": total_opinions,
                "multi_model_opinions": total_multi_model,
                "conflicting_opinions": conflict_count,
                "consensus_rate": consensus_rate,
                "num_consensus_wrong": wrong_consensus_count,
                "wrong_rate": wrong_consensus_count / total_opinions if total_opinions > 0 else 0.0
            }
        }

        merged_list.append(entry)

    return merged_list


def run_format_validate_author_refutations_for_review(
    paper_series: str,
    models: List[str],
    output_path: str,
    indent: int = 2,
    min_opinions: int = 1,
    verbose: bool = False,
) -> Tuple[str, str]:
    """
    Run the "refutation validation" formatting stage.

    Returns
    -------
    (json_output_path, summary_txt_path)
    """
    if not models:
        logger.error("At least one validation model must be provided.")
        sys.exit(1)

    if verbose:
        logger.setLevel(logging.DEBUG)

    logger.info(f"Refutation validation formatting | paper_series={paper_series} | models={models}")

    all_model_results: Dict[str, List[Dict[str, Any]]] = {}
    for model_name in models:
        logger.info(f"Loading validation input for model: {model_name}")
        formatted_results = load_validation_results(model_name, paper_series)

        if formatted_results:
            all_model_results[model_name] = formatted_results
        else:
            logger.warning(f"No usable validation entries found for model: {model_name}")

    if not all_model_results:
        logger.error("No model results loaded successfully; cannot continue.")
        sys.exit(1)

    logger.info(f"Merging validation results across {len(all_model_results)} model(s)...")
    merged_results = merge_multi_model_results(all_model_results)

    filtered_results = [r for r in merged_results if len(r["opinions"]) >= min_opinions]

    filtered_results.sort(key=lambda x: (
        -x["consensus_summary"]["wrong_rate"],
        -len(x["opinions"]),
        x["record_id"],
        x["reviewer_name"]
    ))

    stats = {
        "total_reviewers": len(merged_results),
        "filtered_reviewers": len(filtered_results),
        "total_opinions": sum(len(r["opinions"]) for r in merged_results),
        "model_coverage": {model: len(results) for model, results in all_model_results.items()},
        "consensus_summary": {
            "total_opinions_with_multi_models": sum(
                len([o for o in r["opinions"] if o["consensus"]["num_models"] >= 2])
                for r in merged_results
            ),
            "conflicting_opinions": sum(
                r["consensus_summary"]["conflicting_opinions"]
                for r in merged_results
            ),
            "total_marked_as_wrong": sum(
                r["consensus_summary"]["num_consensus_wrong"]
                for r in merged_results
            )
        }
    }

    final_output = {
        "metadata": {
            "models": models,
            "min_opinions_threshold": min_opinions,
            "stats": stats
        },
        "reviewers": filtered_results
    }

    output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else "."
    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=indent)

    logger.info(
        "Validation formatting complete | "
        f"reviewers_total={stats['total_reviewers']} | "
        f"reviewers_kept={stats['filtered_reviewers']} | "
        f"opinions_total={stats['total_opinions']} | "
        f"saved={output_path}"
    )

    summary_path = output_path.replace(".json", "_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Refutation Validation Results (Human-Review Summary)\n")
        f.write("=" * 80 + "\n\n")

        for result in filtered_results[:30]:
            f.write(f"Record ID: {result['record_id']}\n")
            f.write(f"Reviewer: {result['reviewer_name']}\n")
            f.write(f"Opinions: {len(result['opinions'])}\n")
            f.write(f"Models: {result['num_models']} ({', '.join(result['model_names'])})\n")
            f.write(f"Consensus rate: {result['consensus_summary']['consensus_rate']:.2f}\n")
            f.write(f"Consensus-wrong rate: {result['consensus_summary']['wrong_rate']:.2%}\n")

            if result["meta_review"]:
                f.write(f"Meta-review: {safe_truncate(result['meta_review'])}\n")

            if result["author_responses_combined"]:
                f.write(f"Author response: {safe_truncate(result['author_responses_combined'])}\n")

            f.write("\nOpinion checks (first 5):\n")
            for idx, opinion in enumerate(result["opinions"][:5]):
                consensus = opinion["consensus"]
                f.write(f"  [{idx}] Block: {opinion['block_id']}\n")
                f.write(f"       Text: {safe_truncate(opinion['text'], 100)}\n")
                f.write(
                    f"       Agreement: {'OK' if consensus['all_agree_is_wrong'] else 'CONFLICT'} | "
                    f"Wrong votes: {consensus['num_marked_as_wrong']}/{consensus['num_models']} | "
                    f"Final: {'WRONG' if consensus['consensus_is_wrong'] else 'NOT WRONG'}\n"
                )
            f.write("\n" + "-" * 80 + "\n\n")

    logger.info(f"Validation summary saved: {summary_path}")
    return output_path, summary_path


# ============================================================
# Part B: format_conflicts_resolutions.py
# ============================================================

def extract_meta_review_text_from_record(record: Dict[str, Any]) -> str:
    """
    Extract meta-review text from a single record with defensive error handling.
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
                meta_review_text = meta_review[0][0][1]["metareview:_summary,_strengths_and_weaknesses"]
            else:
                meta_review_text = meta_review[0][0][1]["metareview"]
                if isinstance(meta_review_text, dict):
                    meta_review_text = meta_review_text.get("value", "")
            return meta_review_text
    except (IndexError, KeyError, TypeError) as e:
        logger.warning(f"Record {record_id}: failed to extract meta-review text: {e}")
        return ""

    return ""


def load_and_format_resolutions(input_path: str, model_name: str) -> List[Dict[str, Any]]:
    """
    Load conflict-resolution outputs for a single model and format for human review.
    """
    formatted_records: List[Dict[str, Any]] = []

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    record_id = record.get("id", record.get("record_id", f"unknown_{line_num}"))

                    sentence_texts = record.get("sentence_texts", [])
                    new_split_texts = record.get("new_split_texts", [])
                    conflict_resolution = record.get("conflict_resolution", {})
                    meta_review = extract_meta_review_text_from_record(record)

                    if not conflict_resolution:
                        logger.warning(f"Record {record_id}: missing conflict_resolution.")
                        continue

                    if not sentence_texts or not new_split_texts:
                        logger.warning(f"Record {record_id}: missing sentence_texts or new_split_texts.")
                        continue

                    for label, label_data in conflict_resolution.items():
                        for point_id, point_data in label_data.items():
                            resolution = point_data.get("resolution", {})
                            opinions = point_data.get("opinions", [])

                            if not resolution or not opinions:
                                continue

                            reviewer_opinions = []
                            for opinion in opinions:
                                reviewer_opinions.append({
                                    "block_id": opinion["block_id"],
                                    "reviewer_name": opinion["reviewer"],
                                    "text": opinion["text"]
                                })

                            review_entry = {
                                "record_id": record_id,
                                "label": label,
                                "point_id": point_id,
                                "meta_review": meta_review,
                                "model_name": model_name,
                                "conflict_resolution": {
                                    "correct_blocks": resolution.get("correct_blocks", []),
                                    "incorrect_blocks": resolution.get("incorrect_blocks", [])
                                },
                                "reviewer_opinions": reviewer_opinions,
                                "num_opinions": len(reviewer_opinions),
                                "raw_data": {
                                    "has_meta_review": bool(meta_review.strip())
                                }
                            }

                            formatted_records.append(review_entry)

                except json.JSONDecodeError as e:
                    logger.error(f"Line {line_num}: JSON decode failed: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Line {line_num}: unexpected error: {e}")
                    continue

    except FileNotFoundError:
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    logger.info(f"Loaded conflict resolutions | model={model_name} | points={len(formatted_records)}")
    return formatted_records


def merge_multi_model_resolutions(model_results: Dict[str, List[Dict]]) -> List[Dict[str, Any]]:
    """
    Merge conflict-resolution results across models, grouped by (record_id, label, point_id).
    """
    merged_groups = defaultdict(lambda: {
        "record_id": "",
        "label": "",
        "point_id": "",
        "meta_review": "",
        "reviewer_opinions": [],
        "model_resolutions": [],
        "models": set()
    })

    for model_name, records in model_results.items():
        for record in records:
            key = (record["record_id"], record["label"], record["point_id"])
            group = merged_groups[key]

            if not group["record_id"]:
                group["record_id"] = record["record_id"]
                group["label"] = record["label"]
                group["point_id"] = record["point_id"]
                group["meta_review"] = record["meta_review"]
                group["reviewer_opinions"] = record["reviewer_opinions"]

            group["model_resolutions"].append({
                "model_name": model_name,
                "correct_blocks": record["conflict_resolution"]["correct_blocks"],
                "incorrect_blocks": record["conflict_resolution"]["incorrect_blocks"]
            })
            group["models"].add(model_name)

    merged_list = []
    for _, group in merged_groups.items():
        all_correct_blocks = []
        all_incorrect_blocks = []

        for resolution in group["model_resolutions"]:
            all_correct_blocks.extend(resolution["correct_blocks"])
            all_incorrect_blocks.extend(resolution["incorrect_blocks"])

        correct_counts = defaultdict(int)
        incorrect_counts = defaultdict(int)

        for block in all_correct_blocks:
            correct_counts[block] += 1
        for block in all_incorrect_blocks:
            incorrect_counts[block] += 1

        entry = {
            "record_id": group["record_id"],
            "label": group["label"],
            "point_id": group["point_id"],
            "meta_review": group["meta_review"],
            "reviewer_opinions": group["reviewer_opinions"],
            "num_opinions": len(group["reviewer_opinions"]),
            "num_models": len(group["models"]),
            "model_resolutions": group["model_resolutions"],
            "consensus_analysis": {
                "all_models": list(group["models"]),
                "block_agreement": {
                    block: {
                        "correct_count": correct_counts[block],
                        "incorrect_count": incorrect_counts[block],
                        "consensus": (
                            "agree_correct" if correct_counts[block] > incorrect_counts[block]
                            else "agree_incorrect" if incorrect_counts[block] > correct_counts[block]
                            else "disagree"
                        )
                    }
                    for block in set(list(correct_counts.keys()) + list(incorrect_counts.keys()))
                }
            }
        }

        merged_list.append(entry)

    return merged_list


def run_format_conflicts_resolutions(
    paper_series: str,
    models: List[str],
    output_path: str,
    input_pattern: str,
    indent: int = 2,
    min_models: int = 1,
) -> Tuple[str, str]:
    """
    Run the "conflict resolutions" formatting stage.

    Returns
    -------
    (json_output_path, summary_txt_path)
    """
    if not models:
        logger.error("At least one conflict-resolution model must be provided.")
        sys.exit(1)

    logger.info(f"Conflict resolution formatting | paper_series={paper_series} | models={models}")

    all_model_results: Dict[str, List[Dict[str, Any]]] = {}

    for model_name in models:
        input_path = input_pattern.format(model=model_name.replace("/", "-"), paper_series=paper_series)

        if not os.path.exists(input_path):
            logger.error(f"Input file does not exist: {input_path}")
            continue

        logger.info(f"Loading conflict-resolution input for model: {model_name}")
        formatted_results = load_and_format_resolutions(input_path, model_name)

        if formatted_results:
            all_model_results[model_name] = formatted_results

    if not all_model_results:
        logger.error("No model results loaded successfully; cannot continue.")
        sys.exit(1)

    logger.info(f"Merging conflict resolutions across {len(all_model_results)} model(s)...")
    merged_results = merge_multi_model_resolutions(all_model_results)

    filtered_results = [r for r in merged_results if r["num_models"] >= min_models]

    filtered_results.sort(key=lambda x: (
        -x["num_models"],
        -(len([b for b in x["consensus_analysis"]["block_agreement"].values() if b["consensus"] != "disagree"])),
        x["record_id"],
        x["label"],
        x["point_id"]
    ))

    stats = {
        "total_resolution_points": len(merged_results),
        "filtered_resolution_points": len(filtered_results),
        "model_coverage": {model: len(results) for model, results in all_model_results.items()},
        "total_reviewer_opinions": sum(r["num_opinions"] for r in merged_results),
        "consensus_breakdown": defaultdict(int)
    }

    for result in merged_results:
        num_models_here = result["num_models"]
        if num_models_here >= 2:
            has_consensus = any(
                block_data["consensus"] != "disagree"
                for block_data in result["consensus_analysis"]["block_agreement"].values()
            )
            if has_consensus:
                stats["consensus_breakdown"]["has_consensus"] += 1
            else:
                stats["consensus_breakdown"]["all_disagree"] += 1
        else:
            stats["consensus_breakdown"]["single_model"] += 1

    final_output = {
        "metadata": {
            "models": models,
            "input_pattern": input_pattern,
            "min_models_threshold": min_models,
            "stats": dict(stats)
        },
        "conflict_resolutions": filtered_results
    }

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=indent)

    logger.info(
        "Conflict-resolution formatting complete | "
        f"points_total={stats['total_resolution_points']} | "
        f"points_kept={stats['filtered_resolution_points']} | "
        f"reviewer_opinions_total={stats['total_reviewer_opinions']} | "
        f"saved={output_path}"
    )

    if stats["consensus_breakdown"]:
        logger.info("Consensus breakdown:")
        for key, count in stats["consensus_breakdown"].items():
            logger.info(f"  - {key}: {count}")

    summary_path = output_path.replace(".json", "_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Conflict Resolution Results (Human-Review Summary)\n")
        f.write("=" * 80 + "\n\n")

        for idx, result in enumerate(filtered_results[:100]):
            f.write(f"[{idx + 1}] Record ID: {result['record_id']}\n")
            f.write(f"Label: {result['label']}\n")
            f.write(f"Point ID: {result['point_id']}\n")
            f.write(f"Models: {result['num_models']}\n")
            f.write(f"Reviewer opinions: {result['num_opinions']}\n")

            if result["meta_review"].strip():
                f.write(f"Meta-review: {result['meta_review'][:200]}...\n")
            else:
                f.write("Meta-review: (none)\n")

            f.write("\nModel decisions:\n")
            for model_res in result["model_resolutions"]:
                f.write(f"  - {model_res['model_name']}:\n")
                f.write(f"    correct: {model_res['correct_blocks']}\n")
                f.write(f"    incorrect: {model_res['incorrect_blocks']}\n")

            f.write("\nReviewer opinions:\n")
            for opinion in result["reviewer_opinions"]:
                f.write(f"  [block_{opinion['block_id']}] {opinion['reviewer_name']}: {opinion['text'][:150]}...\n")

            f.write("\nAgreement by block:\n")
            for block, block_data in result["consensus_analysis"]["block_agreement"].items():
                if block_data["correct_count"] > 0 or block_data["incorrect_count"] > 0:
                    f.write(
                        f"  {block}: {block_data['consensus']} "
                        f"(correct={block_data['correct_count']}, incorrect={block_data['incorrect_count']})\n"
                    )

            f.write("\n" + "-" * 80 + "\n\n")

    logger.info(f"Conflict-resolution summary saved: {summary_path}")
    return output_path, summary_path


# ============================================================
# Part C: final_processed.py
# ============================================================

def _safe_block_id_to_int(block_id: Any) -> Any:
    """
    Minimal, safe block_id conversion to int for indexing:
    - If already int: keep as-is
    - If like 'block_12' / '12': extract digits and convert to int
    - Otherwise: return the original value
    """
    if isinstance(block_id, int):
        return block_id
    try:
        s = str(block_id)
        m = re.search(r"(\d+)", s)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return block_id


def run_final_processed(
    split_input_path: str,
    assign_point_ids_path: str,
    conflict_json_path: str,
    refutation_json_path: str,
    output_path: str,
) -> str:
    """
    Assemble the final JSONL by attaching:
    - opinion_groups
    - conflicts_validation
    - rebuttal_validation

    Returns
    -------
    output_path
    """
    # Step 1: Load base split records (expects 'new_split_texts')
    data = []
    with open(split_input_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line, strict=False))

    # Step 2: Load grouping info (point groups)
    with open(assign_point_ids_path, "r", encoding="utf-8") as f:
        point_groups = json.load(f)["point_groups"]

    grouped_data: Dict[str, List[List[Any]]] = {}
    for point_group in point_groups:
        record_id = point_group["record_id"]
        block_ids = [comment["block_id"] for comment in point_group["comments"]]
        grouped_data.setdefault(record_id, []).append(block_ids)

    # Ensure all blocks are present in groups; missing blocks become singleton groups
    for record in data:
        record_id = record["id"]

        total_blocks = len(record["new_split_texts"])
        need_block_ids = set(range(total_blocks))

        included_block_ids = set()
        for block_ids_list in grouped_data[record_id]:
            included_block_ids.update(block_ids_list)

        missing_block_ids = sorted(list(need_block_ids - included_block_ids))

        for missing_id in missing_block_ids:
            grouped_data[record_id].append([missing_id])
        record["opinion_groups"] = grouped_data[record_id]

    # Step 3: Load conflict adjudication output and build per-record lookup
    with open(conflict_json_path, "r", encoding="utf-8") as f:
        conflict_data = json.load(f)

    conflict_dict: Dict[str, Dict[str, set]] = {}
    for item in conflict_data["conflict_resolutions"]:
        record_id = item["record_id"]

        if record_id not in conflict_dict:
            conflict_dict[record_id] = {
                "correct_blocks": set(),
                "incorrect_blocks": set()
            }

        for model_resolution in item["model_resolutions"]:
            for block_str in model_resolution["correct_blocks"]:
                block_id = int(block_str.split("_")[1])
                conflict_dict[record_id]["correct_blocks"].add(block_id)

            for block_str in model_resolution["incorrect_blocks"]:
                block_id = int(block_str.split("_")[1])
                conflict_dict[record_id]["incorrect_blocks"].add(block_id)

    # Generate conflicts_validation for each record (default 'correct')
    for record in data:
        record_id = record["id"]

        total_blocks = len(record["new_split_texts"])
        result_list = ["correct"] * total_blocks

        if record_id in conflict_dict:
            conflict_info = conflict_dict[record_id]

            for block_id in conflict_info["correct_blocks"]:
                if 0 <= block_id < total_blocks:
                    result_list[block_id] = "correct"

            for block_id in conflict_info["incorrect_blocks"]:
                if 0 <= block_id < total_blocks and result_list[block_id] == "correct":
                    result_list[block_id] = "incorrect"
        record["conflicts_validation"] = result_list

    # Step 4: Load refutation-validation output and build per-record lookup
    with open(refutation_json_path, "r", encoding="utf-8") as f:
        refutations_data = json.load(f)

    refutations_dict: Dict[str, Dict[Any, str]] = {}
    for item in refutations_data["reviewers"]:
        record_id = item["record_id"]

        if record_id not in refutations_dict:
            refutations_dict[record_id] = {}

        for opinion in item["opinions"]:
            block_id = opinion["block_id"]

            # Legacy note: the original script referenced "wrong_agreement_rated" (typo).
            # We keep the same branch structure to avoid semantic changes.
            if "consensus" in opinion and "wrong_agreement_rated" in opinion["consensus"]:
                is_wrong = (opinion["consensus"]["wrong_agreement_rate"] == 1.0)
                refutations_dict[record_id][block_id] = "incorrect" if is_wrong else "correct"
            elif "validations_by_model" in opinion and opinion["validations_by_model"]:
                first_model = list(opinion["validations_by_model"].values())[0]
                if "is_reviewer_wrong" in first_model:
                    is_wrong = first_model["is_reviewer_wrong"]
                    refutations_dict[record_id][block_id] = "incorrect" if is_wrong else "correct"

    # Step 5: Generate rebuttal_validation per record (default 'correct')
    for record in data:
        record_id = record["id"]

        total_blocks = len(record["new_split_texts"])
        result_list = ["correct"] * total_blocks

        if record_id in refutations_dict:
            block_results = refutations_dict[record_id]
            for block_id, label in block_results.items():
                idx = _safe_block_id_to_int(block_id)
                if isinstance(idx, int) and 0 <= idx < total_blocks:
                    result_list[idx] = label

        record["rebuttal_validation"] = result_list

        # Remove intermediate fields used upstream (kept from original behavior)
        del record["split_texts"]
        del record["classify_new"]
        del record["for_judge"]

        # Rename to a simpler key for downstream consumers
        record["opinions"] = record["new_split_texts"]
        del record["new_split_texts"]

    with open(output_path, "w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Print a compact summary (kept as print to match original behavior)
    logger.info("Processing complete. Summary:")
    total_conflicts_blocks = 0
    incorrect_conflicts_blocks = 0
    total_rebuttal_blocks = 0
    incorrect_rebuttal_blocks = 0

    for record in data:
        if "conflicts_validation" in record:
            total_conflicts_blocks += len(record["conflicts_validation"])
            incorrect_conflicts_blocks += sum(1 for x in record["conflicts_validation"] if x == "incorrect")

        if "rebuttal_validation" in record:
            total_rebuttal_blocks += len(record["rebuttal_validation"])
            incorrect_rebuttal_blocks += sum(1 for x in record["rebuttal_validation"] if x == "incorrect")

    if total_conflicts_blocks > 0:
        conflicts_ratio = incorrect_conflicts_blocks / total_conflicts_blocks
        logger.info("Reviewer conflict validation (conflicts_validation):")
        logger.info(f"  - Total blocks: {total_conflicts_blocks}")
        logger.info(f"  - Incorrect blocks: {incorrect_conflicts_blocks}")
        logger.info(f"  - Incorrect ratio: {conflicts_ratio:.2%}")
    else:
        logger.info("Reviewer conflict validation (conflicts_validation): no data")

    if total_rebuttal_blocks > 0:
        rebuttal_ratio = incorrect_rebuttal_blocks / total_rebuttal_blocks
        logger.info("Author refutation validation (rebuttal_validation):")
        logger.info(f"  - Total blocks: {total_rebuttal_blocks}")
        logger.info(f"  - Incorrect blocks: {incorrect_rebuttal_blocks}")
        logger.info(f"  - Incorrect ratio: {rebuttal_ratio:.2%}")
    else:
        logger.info("Author refutation validation (rebuttal_validation): no data")

    return output_path


# ============================================================
# Unified main (single argparse) - sequential pipeline
# ============================================================

def _resolve_template_path(path_template: str, paper_series: str) -> str:
    """
    Resolve a path template with either:
    - '{paper_series}' formatting, or
    - legacy replacement of 'ICLR.cc_2020' with the actual paper_series.
    """
    if "{paper_series}" in path_template:
        return path_template.format(paper_series=paper_series)
    return path_template.replace("ICLR.cc_2020", paper_series)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Single-file pipeline that runs:\n"
            "1) Refutation validation formatting\n"
            "2) Conflict resolution formatting\n"
            "3) Final JSONL assembly with validation labels\n"
        )
    )

    # Shared
    parser.add_argument("--org", default="ICLR", help="Organization name, e.g. ICLR or NeurIPS")
    parser.add_argument("--year", type=int, default=2020, help="Conference year (e.g., 2020)")

    # Stage 1: validation formatting
    parser.add_argument("--validation-models", nargs="+", default=["deepseek-reasoner"],
                        help="Model names for refutation validation formatting")
    parser.add_argument(
        "--validation-output",
        default="{paper_series}/visual_output/refutation_validation_results_for_review.json",
        help="Output JSON path for formatted refutation validation results"
    )
    parser.add_argument("--validation-min-opinions", type=int, default=1, help="Minimum #opinions to keep a reviewer")

    # Stage 2: conflict formatting
    parser.add_argument("--conflict-models", nargs="+", default=["deepseek-reasoner"],
                        help="Model names for conflict resolution formatting")
    parser.add_argument("--conflict-output", default="{paper_series}/visual_output/conflict_resolutions_for_review.json",
                        help="Output JSON path for formatted conflict resolutions")
    parser.add_argument(
        "--conflict-input-pattern",
        default="{paper_series}/conflicts_resolved/{paper_series}_deepseek-reasoner_medium_split_clean_conflicts_resolved.jsonl",
        help="Input path template for conflict-resolution JSONL (per model)"
    )
    parser.add_argument("--conflict-min-models", type=int, default=1, help="Minimum #models to keep a resolution point")

    # Stage 3: final assembly
    parser.add_argument(
        "--split-input",
        default="{paper_series}/split_final/{paper_series}_gpt-5-mini_medium_split_clean.jsonl",
        help="Input split_final JSONL (expects 'new_split_texts')"
    )
    parser.add_argument(
        "--assign-point-ids",
        default="{paper_series}/visual_output/assign_deepseek-reasoner_point_ids.json",
        help="Path to point-id grouping JSON (assign_point_ids output)"
    )
    parser.add_argument(
        "--final-output",
        default="{paper_series}/split_final/{paper_series}_gpt-5-mini_medium_split_clean_with_reviewer_validations.jsonl",
        help="Output JSONL path for final dataset with validation labels"
    )

    args = parser.parse_args()
    if args.org.endswith(".cc"):
        args.org = args.org[:-3]
    args.paper_series = f"{args.org}.cc_{args.year}"

    paper_series = args.paper_series

    # Resolve path templates
    validation_output = _resolve_template_path(args.validation_output, paper_series)
    conflict_output = _resolve_template_path(args.conflict_output, paper_series)
    conflict_input_pattern = _resolve_template_path(args.conflict_input_pattern, paper_series)

    split_input = _resolve_template_path(args.split_input, paper_series)
    assign_point_ids = _resolve_template_path(args.assign_point_ids, paper_series)
    final_output = _resolve_template_path(args.final_output, paper_series)

    # Ensure output directories exist
    os.makedirs(os.path.dirname(validation_output), exist_ok=True)
    os.makedirs(os.path.dirname(conflict_output), exist_ok=True)
    os.makedirs(os.path.dirname(final_output), exist_ok=True)

    # 1) Validation formatting
    run_format_validate_author_refutations_for_review(
        paper_series=paper_series,
        models=args.validation_models,
        output_path=validation_output,
        min_opinions=args.validation_min_opinions,
    )

    # 2) Conflict formatting
    run_format_conflicts_resolutions(
        paper_series=paper_series,
        models=args.conflict_models,
        output_path=conflict_output,
        input_pattern=conflict_input_pattern,
        min_models=args.conflict_min_models,
    )

    # 3) Final dataset assembly
    run_final_processed(
        split_input_path=split_input,
        assign_point_ids_path=assign_point_ids,
        conflict_json_path=conflict_output,
        refutation_json_path=validation_output,
        output_path=final_output,
    )


if __name__ == "__main__":
    main()
