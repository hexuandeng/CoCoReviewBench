#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Purpose
Analyze conflict-resolution outputs across models and report block-level agreement metrics.

Notes
- Reads per-model `conflicts_resolved.jsonl` files and merges by record/label/point.
- Computes a soft-gold majority vote (tie = 0.5) and prints summaries.
- Writes no output files.
"""

import argparse
import json
import logging
import math
import itertools
import re
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, Counter

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import eval_binary_lists_by_majority_vote, fmt, print_table, derive_bal_acc_and_mcc

logger = logging.getLogger(__name__)

# Block ID normalization for consistent grouping.
def extract_block_number(block_id: Any) -> Optional[int]:
    """Extract the numeric portion of a block id (e.g., "block_17", "17", 17 -> 17)."""
    if block_id is None:
        return None
    s = str(block_id)
    m = re.search(r"block_(\d+)", s)
    if m:
        return int(m.group(1))
    m2 = re.search(r"(\d+)", s)
    return int(m2.group(1)) if m2 else None


def canon_block_id(x: Any) -> Optional[str]:
    """Normalize a block id to the form "block_{n}"."""
    n = extract_block_number(x)
    return f"block_{n}" if n is not None else None


def load_and_format_resolutions(input_path: str, model_name: str) -> List[Dict[str, Any]]:
    """
    Load a model's conflict resolutions and normalize them for review.
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

                    if not conflict_resolution:
                        logger.warning(f"Record {record_id} has no conflict resolution")
                        continue

                    if not sentence_texts or not new_split_texts:
                        logger.warning(f"Record {record_id} is missing text fields")
                        continue

                    # Nested schema: label -> point_id.
                    for label, label_data in conflict_resolution.items():
                        if not isinstance(label_data, dict):
                            continue

                        for point_id, point_data in label_data.items():
                            if not isinstance(point_data, dict):
                                continue

                            resolution = point_data.get("resolution", {}) or {}
                            opinions = point_data.get("opinions", []) or []

                            if not resolution or not opinions:
                                continue

                            # Collect reviewer opinions; keep raw and canonical block ids.
                            reviewer_opinions = []
                            for op in opinions:
                                if not isinstance(op, dict):
                                    continue
                                blk_raw = op.get("block_id")
                                blk_c = canon_block_id(blk_raw)
                                reviewer_opinions.append({
                                    "block_id": blk_raw,
                                    "block_id_canon": blk_c,
                                    "reviewer_name": op.get("reviewer"),
                                    "text": op.get("text", "")
                                })

                            # Normalize correct/incorrect block ids.
                            correct_blocks_raw = resolution.get("correct_blocks", []) or []
                            incorrect_blocks_raw = resolution.get("incorrect_blocks", []) or []
                            correct_blocks = [b for b in (canon_block_id(x) for x in correct_blocks_raw) if b]
                            incorrect_blocks = [b for b in (canon_block_id(x) for x in incorrect_blocks_raw) if b]

                            review_entry = {
                                "record_id": record_id,
                                "label": label,
                                "point_id": point_id,
                                "model_name": model_name,
                                "conflict_resolution": {
                                    "correct_blocks": correct_blocks,
                                    "incorrect_blocks": incorrect_blocks
                                },
                                "reviewer_opinions": reviewer_opinions,
                                "num_opinions": len(reviewer_opinions),
                            }

                            formatted_records.append(review_entry)

                except json.JSONDecodeError as e:
                    logger.error(f"JSON parse failed on line {line_num}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing line {line_num}: {e}")
                    continue

    except FileNotFoundError:
        logger.error(f"File not found: {input_path}")
        sys.exit(1)

    logger.info(f"Loaded model {model_name}: {len(formatted_records)} conflict resolutions")
    return formatted_records


def merge_multi_model_resolutions(model_results: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Merge model results by (record_id, label, point_id).
    """
    merged_groups = defaultdict(lambda: {
        "record_id": "",
        "label": "",
        "point_id": "",
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
                group["reviewer_opinions"] = record["reviewer_opinions"]
            else:
                if group["reviewer_opinions"] != record["reviewer_opinions"]:
                    logger.warning(
                        "Reviewer opinions mismatch across models for "
                        f"{(record['record_id'], record['label'], record['point_id'])}; keeping the first."
                    )

            group["model_resolutions"].append({
                "model_name": model_name,
                "correct_blocks": record["conflict_resolution"]["correct_blocks"],
                "incorrect_blocks": record["conflict_resolution"]["incorrect_blocks"]
            })
            group["models"].add(model_name)

    merged_list: List[Dict[str, Any]] = []
    for _, group in merged_groups.items():
        correct_counts = defaultdict(int)
        incorrect_counts = defaultdict(int)

        for mr in group["model_resolutions"]:
            for blk in (mr.get("correct_blocks") or []):
                if blk:
                    correct_counts[blk] += 1
            for blk in (mr.get("incorrect_blocks") or []):
                if blk:
                    incorrect_counts[blk] += 1

        all_blocks = set(correct_counts.keys()) | set(incorrect_counts.keys())

        entry = {
            "record_id": group["record_id"],
            "label": group["label"],
            "point_id": group["point_id"],
            "reviewer_opinions": group["reviewer_opinions"],
            "num_opinions": len(group["reviewer_opinions"]),
            "num_models": len(group["models"]),
            "model_resolutions": group["model_resolutions"],
            "consensus_analysis": {
                "all_models": sorted(list(group["models"])),
                "block_agreement": {
                    blk: {
                        "correct_count": correct_counts.get(blk, 0),
                        "incorrect_count": incorrect_counts.get(blk, 0),
                        "consensus": (
                            "agree_correct" if correct_counts.get(blk, 0) > incorrect_counts.get(blk, 0) else
                            "agree_incorrect" if incorrect_counts.get(blk, 0) > correct_counts.get(blk, 0) else
                            "disagree"
                        )
                    }
                    for blk in sorted(all_blocks, key=lambda x: extract_block_number(x) or -1)
                }
            }
        }
        merged_list.append(entry)

    return merged_list


def build_model_vote_maps(model_resolutions: List[Dict[str, Any]]) -> Dict[str, Dict[str, bool]]:
    """
    Return vote_maps[model_name][block_id_canon] = is_wrong.
    incorrect_blocks => True
    correct_blocks => False
    """
    vote_maps: Dict[str, Dict[str, bool]] = {}
    for mr in model_resolutions or []:
        m = mr.get("model_name")
        if not m:
            continue
        mp = vote_maps.setdefault(m, {})

        for blk in (mr.get("correct_blocks") or []):
            blk_c = canon_block_id(blk) or blk
            if blk_c:
                if blk_c in mp and mp[blk_c] is True:
                    logger.warning(
                        f"Model {m} marks {blk_c} as both incorrect and correct; keeping the last value."
                    )
                mp[blk_c] = False

        for blk in (mr.get("incorrect_blocks") or []):
            blk_c = canon_block_id(blk) or blk
            if blk_c:
                if blk_c in mp and mp[blk_c] is False:
                    logger.warning(
                        f"Model {m} marks {blk_c} as both correct and incorrect; keeping the last value."
                    )
                mp[blk_c] = True

    return vote_maps

# Analysis entrypoint: majority-vote soft gold (tie = 0.5).
def analyze_conflict_blocks_threshold_gold(
    merged_obj: Dict[str, Any],
    threshold_k: int = 4,  # Kept for CLI compatibility; not used for gold.
    top_k_pairs: int = 15,
    print_skipped_limit: int = 50,
) -> None:
    cr_list = merged_obj.get("conflict_resolutions", merged_obj)
    if not isinstance(cr_list, list):
        raise ValueError(
            "Invalid JSON structure: expected a conflict_resolutions list or a list input."
        )

    expected_models = (merged_obj.get("metadata") or {}).get("models")
    if not expected_models:
        ms = set()
        for cr in cr_list:
            for mr in (cr.get("model_resolutions") or []):
                if mr.get("model_name"):
                    ms.add(mr["model_name"])
        expected_models = sorted(ms)

    expected_models = list(expected_models)
    M = len(expected_models)

    print(f"\nExpected models: {M}")
    print("Expected model list:", expected_models)
    print(
        "Gold rule: majority True (> half) => 1.0; majority False (< half) => 0.0; tie (== half) => 0.5"
    )
    if threshold_k is not None:
        print(f"(Note) --threshold-k={threshold_k} is kept for compatibility; not used for gold.")

    dist_wrong_count = Counter()
    skipped: List[Tuple[Any, Any, str, List[str]]] = []

    # Collect per-model predictions; keep only blocks with full model coverage.
    preds_by_model: Dict[str, List[bool]] = {m: [] for m in expected_models}

    total_blocks_seen = 0
    total_blocks_used = 0

    for cr in cr_list:
        record_id = cr.get("record_id")
        point_id = cr.get("point_id")

        block_agreement = ((cr.get("consensus_analysis") or {}).get("block_agreement")) or {}
        if not block_agreement:
            continue

        vote_maps = build_model_vote_maps(cr.get("model_resolutions") or [])

        # Deduplicate by canonical block id.
        canon_blocks = []
        seen = set()
        for blk_raw in block_agreement.keys():
            blk_c = canon_block_id(blk_raw) or (blk_raw if isinstance(blk_raw, str) else None)
            if not blk_c:
                continue
            if blk_c in seen:
                continue
            seen.add(blk_c)
            canon_blocks.append(blk_c)

        for blk_c in canon_blocks:
            total_blocks_seen += 1

            votes = {}
            missing = []
            for m in expected_models:
                mv = vote_maps.get(m, {})
                if blk_c not in mv:
                    missing.append(m)
                else:
                    votes[m] = bool(mv[blk_c])

            if missing:
                skipped.append((record_id, point_id, blk_c, missing))
                continue

            total_blocks_used += 1
            wrong_count = sum(votes.values())
            dist_wrong_count[wrong_count] += 1

            for m in expected_models:
                preds_by_model[m].append(votes[m])

    print(f"\nTotal blocks (from block_agreement, canonicalized): {total_blocks_seen}")
    print(f"Blocks with full model coverage ({M} models): {total_blocks_used}")
    print(f"Skipped blocks: {len(skipped)}")

    if skipped:
        print(
            "\n=== Skipped blocks (missing model votes; showing first {}) ===".format(
                print_skipped_limit if print_skipped_limit >= 0 else len(skipped)
            )
        )
        to_show = skipped if print_skipped_limit < 0 else skipped[:print_skipped_limit]
        for rid, pid, blk_c, missing in to_show:
            print(f"- record_id={rid}, point_id={pid}, block={blk_c}, missing={missing}")
        if print_skipped_limit >= 0 and len(skipped) > print_skipped_limit:
            print(f"... {len(skipped) - print_skipped_limit} more not shown")

    print("\nDistribution of models voting is_wrong=True per block (count):")
    dist_full = {i: dist_wrong_count.get(i, 0) for i in range(M + 1)}
    print(dist_full)

    if total_blocks_used == 0:
        print("\nNo blocks with full model coverage; cannot compute metrics.")
        return

    # All-model majority-vote evaluation (soft gold: 0/0.5/1).
    pred_lists = [preds_by_model[m] for m in expected_models]
    eval_out = eval_binary_lists_by_majority_vote(pred_lists, names=expected_models, gold=None)
    gold_out = eval_out.get("gold", [])
    summary = eval_out.get("summary", {})
    per_model = eval_out.get("per_model", {})

    print(
        f"\nSoft-gold distribution: "
        f"gold=1.0({summary.get('num_gold_true', 0)}), "
        f"gold=0.0({summary.get('num_gold_false', 0)}), "
        f"gold=0.5({summary.get('num_gold_tie', 0)}), "
        f"n={summary.get('n', 0)}"
    )

    # Single-model metrics table.
    model_rows = []
    for m in expected_models:
        met = per_model.get(m, {})
        tp = float(met.get("tp", 0.0))
        fp = float(met.get("fp", 0.0))
        fn = float(met.get("fn", 0.0))
        tn = float(met.get("tn", 0.0))
        bal_acc, mcc = derive_bal_acc_and_mcc(tp, fp, fn, tn)

        model_rows.append([
            m,
            met.get("n", 0),
            fmt(met.get("acc", 0.0)),
            fmt(met.get("precision", 0.0)),
            fmt(met.get("recall", 0.0)),
            fmt(met.get("f1", 0.0)),
            fmt(bal_acc),
            fmt(mcc),
            f"{tp:.1f}/{fp:.1f}/{fn:.1f}/{tn:.1f}",
        ])

    print("\n=== Single model vs soft-gold (majority vote; tie=0.5), sorted by F1 ===")
    model_rows.sort(key=lambda r: float(r[5]), reverse=True)
    print_table(
        model_rows,
        headers=["model", "n", "acc", "prec", "recall", "f1", "bal_acc", "mcc", "tp/fp/fn/tn"]
    )

    # Two-model OR/AND combinations, evaluated against the same soft gold.
    pair_results = []
    for m1, m2 in itertools.combinations(expected_models, 2):
        p1 = preds_by_model[m1]
        p2 = preds_by_model[m2]
        y_or = [a or b for a, b in zip(p1, p2)]
        y_and = [a and b for a, b in zip(p1, p2)]

        # Evaluate against the same gold_out.
        r_or = eval_binary_lists_by_majority_vote([y_or], names=[f"OR({m1},{m2})"], gold=gold_out)["per_model"][f"OR({m1},{m2})"]
        r_and = eval_binary_lists_by_majority_vote([y_and], names=[f"AND({m1},{m2})"], gold=gold_out)["per_model"][f"AND({m1},{m2})"]

        for rule_name, met in [("OR", r_or), ("AND", r_and)]:
            tp = float(met.get("tp", 0.0))
            fp = float(met.get("fp", 0.0))
            fn = float(met.get("fn", 0.0))
            tn = float(met.get("tn", 0.0))
            bal_acc, mcc = derive_bal_acc_and_mcc(tp, fp, fn, tn)

            pair_results.append([
                rule_name,
                f"{m1} + {m2}",
                met.get("n", 0),
                fmt(met.get("acc", 0.0)),
                fmt(met.get("precision", 0.0)),
                fmt(met.get("recall", 0.0)),
                fmt(met.get("f1", 0.0)),
                fmt(bal_acc),
                fmt(mcc),
                f"{tp:.1f}/{fp:.1f}/{fn:.1f}/{tn:.1f}",
            ])

    pair_results.sort(key=lambda r: float(r[6]), reverse=True)
    top_pairs = pair_results[:top_k_pairs]

    print(f"\n=== Top {top_k_pairs} two-model combos (by F1; gold=soft majority vote) ===")
    print_table(
        top_pairs,
        headers=["rule", "pair", "n", "acc", "prec", "recall", "f1", "bal_acc", "mcc", "tp/fp/fn/tn"]
    )

    if model_rows:
        best_single = model_rows[0]
        print("\n=== Best recommendation (by F1) ===")
        print(f"Best single model: {best_single[0]} (F1={best_single[5]}, Acc={best_single[2]})")
    if pair_results:
        best_pair = pair_results[0]
        print(f"Best two-model combo: {best_pair[0]}({best_pair[1]}) (F1={best_pair[6]}, Acc={best_pair[3]})")


# CLI entrypoint (prints results only).
def main():
    parser = argparse.ArgumentParser(
        description="Conflict resolutions: load models -> merge -> print stats + block analysis (no files written)."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gemini-2.5-flash", "gpt-5-mini", "gpt-5", "gemini-2.5-pro", "deepseek-reasoner", "deepseek-chat"],
        help="Model name list to process."
    )
    parser.add_argument(
        "--input-pattern",
        default="pre_eval/conflicts_resolved/evaluation_{model}_medium_split_clean_conflicts_resolved.jsonl",
        help="Input path pattern; {model} will be replaced by the model name."
    )
    parser.add_argument(
        "--min-models",
        type=int,
        default=1,
        help="Minimum number of models required to keep a conflict point.",
    )
    parser.add_argument(
        "--threshold-k",
        type=int,
        default=4,
        help="Compatibility flag; no longer used for gold (majority vote with tie=0.5).",
    )
    parser.add_argument("--top-k-pairs", type=int, default=15, help="Top-K two-model combinations to display.")
    parser.add_argument("--target-model", default="gpt-5", help="Target model for mismatch exports.")
    parser.add_argument(
        "--print-skipped-limit",
        type=int,
        default=50,
        help="Max skipped blocks to print (-1 prints all).",
    )
    parser.add_argument(
        "--dump-mismatch-json",
        action="store_true",
        help="Whether to print mismatch groups as JSON.",
    )
    parser.add_argument(
        "--dump-groups-limit",
        type=int,
        default=5,
        help="Max groups to print in JSON dump (<=0 prints all).",
    )
    args = parser.parse_args()

    if not args.models:
        logger.error("At least one model must be specified.")
        sys.exit(1)

    all_model_results: Dict[str, List[Dict[str, Any]]] = {}
    for model_name in args.models:
        input_path = args.input_pattern.format(model=model_name.replace("/", "-"))
        if not os.path.exists(input_path):
            logger.error(f"File does not exist: {input_path}")
            continue

        logger.info(f"Loading model {model_name}...")
        formatted_results = load_and_format_resolutions(input_path, model_name)
        if formatted_results:
            all_model_results[model_name] = formatted_results

    if not all_model_results:
        logger.error("No model results were loaded.")
        sys.exit(1)

    logger.info(f"\nMerging results from {len(all_model_results)} models...")
    merged_results = merge_multi_model_resolutions(all_model_results)

    filtered_results = [r for r in merged_results if r["num_models"] >= args.min_models]

    filtered_results.sort(key=lambda x: (
        -x["num_models"],
        -(len([b for b in x["consensus_analysis"]["block_agreement"].values() if b["consensus"] != "disagree"])),
        str(x["record_id"]),
        str(x["label"]),
        str(x["point_id"]),
    ))

    stats = {
        "total_resolution_points": len(merged_results),
        "filtered_resolution_points": len(filtered_results),
        "model_coverage": {model: len(results) for model, results in all_model_results.items()},
        "total_reviewer_opinions": sum(r["num_opinions"] for r in merged_results),
        "consensus_breakdown": defaultdict(int),
    }

    for r in merged_results:
        if r["num_models"] >= 2:
            has_consensus = any(
                block_data["consensus"] != "disagree"
                for block_data in r["consensus_analysis"]["block_agreement"].values()
            )
            stats["consensus_breakdown"]["has_consensus" if has_consensus else "all_disagree"] += 1
        else:
            stats["consensus_breakdown"]["single_model"] += 1

    logger.info("\n" + "=" * 60)
    logger.info("Conflict resolutions merged and summarized (no files written).")
    logger.info("=" * 60)
    logger.info(f"\nTotal resolution points: {stats['total_resolution_points']}")
    logger.info(
        f"Filtered points: {stats['filtered_resolution_points']} (min models: {args.min_models})"
    )
    logger.info(f"Total reviewer opinions: {stats['total_reviewer_opinions']}")
    logger.info(f"Model coverage: {dict(stats['model_coverage'])}")

    logger.info("\nConsensus breakdown:")
    for k, v in stats["consensus_breakdown"].items():
        logger.info(f"  {k}: {v}")

    merged_obj = {
        "metadata": {
            "models": args.models,
            "input_pattern": args.input_pattern,
            "min_models_threshold": args.min_models,
            "stats": {
                **{k: v for k, v in stats.items() if k != "consensus_breakdown"},
                "consensus_breakdown": dict(stats["consensus_breakdown"]),
            }
        },
        "conflict_resolutions": filtered_results
    }

    analyze_conflict_blocks_threshold_gold(
        merged_obj,
        threshold_k=args.threshold_k,
        top_k_pairs=args.top_k_pairs,
        print_skipped_limit=args.print_skipped_limit,
    )
    logger.info("\nDone: printed summary and analysis results (no extra files written).")


if __name__ == "__main__":
    main()
