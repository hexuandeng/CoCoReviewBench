#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose
-------
Aggregate and analyze *reviewer opinion validation* outputs from multiple LLMs.

Given per-model JSONL files that contain:
- reviewer_opinion_validation (per reviewer)
- author_refutation_analysis (optional context: author_responses_combined)

This script:
1) Loads each model's validated outputs (JSONL).
2) Normalizes and merges results across models by (record_id, reviewer_name).
3) Computes opinion-level agreement/conflict statistics across models.
4) Evaluates each model (and OR/AND pairs) against a *soft majority-vote gold*:
   - gold=1.0 if True votes > half
   - gold=0.0 if True votes < half
   - gold=0.5 if tie

It prints summary statistics and evaluation tables to stdout and does not write files.

Input
-----
- Expected files (per model):
  pre_eval/validations/evaluation_{model_name}_medium_split_clean_validated.jsonl

Output
------
- Printed logs and tables only (stdout)

Notes
-----
- Requires helper functions in `utils`:
  eval_binary_lists_by_majority_vote, fmt, print_table, derive_bal_acc_and_mcc
- Paths are hard-coded for the current repo layout; adjust if you reorganize files.
"""

import argparse
import json
import logging
import itertools
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import eval_binary_lists_by_majority_vote, fmt, print_table, derive_bal_acc_and_mcc

logger = logging.getLogger(__name__)


# ============================================================
# 0) Utilities
# ============================================================
def normalize_block_id(block_id: Any) -> str:
    """Extract the numeric core to normalize block_id format variants."""
    s = str(block_id).lower().replace(" ", "")
    match = re.search(r"\d+", s)
    return match.group() if match else s


# ============================================================
# 1) Load per-model validated.jsonl (keep the "first-file" logic)
# ============================================================
def load_validation_results(model_name: str) -> List[Dict[str, Any]]:
    """Load a single model's validation results (validated.jsonl)."""
    input_path = f"pre_eval/validations/evaluation_{model_name}_medium_split_clean_validated.jsonl"
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
                    refutation_analysis = record.get("author_refutation_analysis", {})

                    if not validation_data:
                        logger.warning(f"Record {record_id}: missing 'reviewer_opinion_validation'; skipping.")
                        continue

                    for reviewer_name, reviewer_validation_data in validation_data.items():
                        reviewer_refutation_data = refutation_analysis.get(reviewer_name, {})
                        author_responses_combined = reviewer_refutation_data.get("author_responses_combined", "")

                        opinions = reviewer_validation_data.get("refuted_opinions_validated", [])
                        if not isinstance(opinions, list) or not opinions:
                            continue

                        processed_opinions = []
                        for opinion in opinions:
                            if not isinstance(opinion, dict):
                                continue
                            raw_block_id = opinion.get("block_id")
                            normalized_id = normalize_block_id(raw_block_id) if raw_block_id is not None else None

                            validation_result = opinion.get("validation_result", {})
                            if not isinstance(validation_result, dict):
                                validation_result = {}

                            processed_opinions.append(
                                {
                                    "block_id": raw_block_id,
                                    "block_id_normalized": normalized_id,
                                    "text": opinion.get("opinion_text", ""),
                                    "is_reviewer_wrong": bool(validation_result.get("is_reviewer_wrong", False)),
                                    "llm_call_success": bool(opinion.get("llm_call_success", False)),
                                }
                            )

                        review_entry = {
                            "record_id": record_id,
                            "reviewer_name": reviewer_name,
                            "author_responses_combined": author_responses_combined,
                            "num_opinions": len(processed_opinions),
                            "opinions": processed_opinions,
                            "model": model_name,
                            "validation_summary": reviewer_validation_data.get("summary", {}),
                        }
                        results.append(review_entry)

                except json.JSONDecodeError as e:
                    logger.error(f"Line {line_num}: JSON parse failed: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Line {line_num}: failed to process record: {e}")
                    try:
                        logger.error(f"Record ID: {record_id}")
                    except Exception:
                        pass
                    continue

    except FileNotFoundError as e:
        logger.error(f"File not found: {input_path}")
        logger.error(f"Error: {e}")
        sys.exit(1)

    logger.info(f"Loaded model '{model_name}': {len(results)} reviewers.")
    return results


# ============================================================
# 2) Merge multi-model results (keep opinion_key logic)
# ============================================================
def merge_multi_model_results(model_results: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Merge multiple models' validation results by (record_id, reviewer_name)."""
    merged_groups = defaultdict(
        lambda: {
            "record_id": "",
            "reviewer_name": "",
            "author_responses_combined": "",
            "models": set(),
            "opinion_results": {},  # key: (normalized_block_id, clean_text)
        }
    )

    for model_name, records in model_results.items():
        for record in records:
            key = (record["record_id"], record["reviewer_name"])

            if not merged_groups[key]["record_id"]:
                merged_groups[key]["record_id"] = record["record_id"]
                merged_groups[key]["reviewer_name"] = record["reviewer_name"]
                merged_groups[key]["author_responses_combined"] = record["author_responses_combined"]

            merged_groups[key]["models"].add(model_name)

            if merged_groups[key]["author_responses_combined"] != record["author_responses_combined"]:
                logger.warning(
                    f"Record {key}: 'author_responses_combined' differs across models; keeping the first value."
                )

            for opinion in record["opinions"]:
                normalized_block_id = opinion.get("block_id_normalized")
                if normalized_block_id is None:
                    normalized_block_id = normalize_block_id(opinion.get("block_id"))

                clean_text = (opinion.get("text", "") or "").strip().replace("\r\n", "\n")
                opinion_key = (normalized_block_id, clean_text)

                if opinion_key not in merged_groups[key]["opinion_results"]:
                    merged_groups[key]["opinion_results"][opinion_key] = {
                        "block_id": opinion.get("block_id"),
                        "block_id_normalized": normalized_block_id,
                        "text": opinion.get("text", ""),
                        "validations_by_model": {},
                    }

                merged_groups[key]["opinion_results"][opinion_key]["validations_by_model"][model_name] = {
                    "is_reviewer_wrong": bool(opinion.get("is_reviewer_wrong", False)),
                    "llm_call_success": bool(opinion.get("llm_call_success", False)),
                }

    merged_list: List[Dict[str, Any]] = []
    for _, group in merged_groups.items():
        opinions_list = []
        conflict_count = 0
        total_multi_model = 0
        wrong_consensus_count = 0

        for data in group["opinion_results"].values():
            validations = list(data["validations_by_model"].values())
            num_models = len(validations)
            wrong_values = [bool(v.get("is_reviewer_wrong", False)) for v in validations]

            if num_models >= 2:
                all_agree_wrong = len(set(wrong_values)) == 1
                wrong_agreement_rate = sum(wrong_values) / num_models
                conflict_count += 0 if all_agree_wrong else 1
                total_multi_model += 1
                if wrong_agreement_rate >= 0.5:
                    wrong_consensus_count += 1
            else:
                all_agree_wrong = True
                wrong_agreement_rate = 1.0 if (wrong_values[0] if wrong_values else False) else 0.0
                if wrong_values and wrong_values[0]:
                    wrong_consensus_count += 1

            num_wrong = sum(wrong_values)
            opinions_list.append(
                {
                    "block_id": data.get("block_id"),
                    "text": data.get("text", ""),
                    "validations_by_model": data.get("validations_by_model", {}),
                    "consensus": {
                        "all_agree_is_wrong": all_agree_wrong,
                        "wrong_agreement_rate": wrong_agreement_rate,
                        "num_models": num_models,
                        "num_marked_as_wrong": num_wrong,
                        "consensus_is_wrong": (wrong_agreement_rate >= 0.5)
                        if num_models >= 2
                        else (wrong_values[0] if wrong_values else False),
                    },
                }
            )

        def sort_key(x):
            try:
                bid = x.get("block_id")
                m = re.search(r"(\d+)", str(bid))
                return int(m.group(1)) if m else 0
            except Exception:
                return 0

        opinions_list.sort(key=sort_key)

        total_opinions = len(opinions_list)
        consensus_rate = (total_multi_model - conflict_count) / total_multi_model if total_multi_model > 0 else 1.0

        merged_list.append(
            {
                "record_id": group["record_id"],
                "reviewer_name": group["reviewer_name"],
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
                    "wrong_rate": (wrong_consensus_count / total_opinions) if total_opinions > 0 else 0.0,
                },
            }
        )

    return merged_list


# ============================================================
# 3) Opinion traversal + prediction extraction (unchanged)
# ============================================================
def iter_opinions(reviewers: List[Dict[str, Any]]):
    """Iterate strictly over reviewers -> opinions to avoid duplicate counting."""
    for rev in reviewers:
        if not isinstance(rev, dict):
            continue
        for op in (rev.get("opinions") or []):
            if isinstance(op, dict) and isinstance(op.get("validations_by_model"), dict):
                yield rev, op


def get_model_preds_from_opinion(opinion: Dict[str, Any], require_success: bool = True) -> Dict[str, bool]:
    """
    Extract model predictions (is_reviewer_wrong) from opinion['validations_by_model'].

    require_success=True: if llm_call_success is present and False, drop that model's prediction.
    """
    vbm = opinion.get("validations_by_model", {})
    preds: Dict[str, bool] = {}
    if not isinstance(vbm, dict):
        return preds

    for m, info in vbm.items():
        if not isinstance(info, dict):
            continue
        if "is_reviewer_wrong" not in info:
            continue
        if require_success and ("llm_call_success" in info) and (info["llm_call_success"] is False):
            continue
        preds[m] = bool(info["is_reviewer_wrong"])
    return preds


# ============================================================
# 4) Analysis: soft majority-vote gold over all models (tie=0.5)
#    OR/AND pair rules are evaluated against the same gold.
# ============================================================
def analyze_two_model_only(
    reviewers: List[Dict[str, Any]],
    threshold_k: int = 4,  # kept for CLI compatibility; not used for gold anymore
    top_k_pairs: int = 60,
    require_success: bool = True,
    require_all_models: bool = False,
    expected_models: List[str] | None = None,
) -> None:
    """
    Gold is derived from a majority vote over all models:
      > half => 1.0, < half => 0.0, tie => 0.5

    Per-model metrics come directly from eval_binary_lists_by_majority_vote.
    OR/AND pair rules are also evaluated using the same gold.
    """
    blocks: List[Dict[str, bool]] = []
    skipped: List[Tuple[Any, Any, str]] = []

    for rev, op in iter_opinions(reviewers):
        preds = get_model_preds_from_opinion(op, require_success=require_success)
        if not preds:
            skipped.append((rev.get("record_id"), op.get("block_id"), "no_valid_preds"))
            continue
        blocks.append(preds)

    if skipped:
        print(f"\n[Skipped opinions] {len(skipped)} total (showing first 50):")
        for rid, bid, reason in skipped[:50]:
            print(f"- record_id={rid}, block_id={bid}, reason={reason}")
        if len(skipped) > 50:
            print(f"... plus {len(skipped) - 50} more not shown")

    if not blocks:
        print("No opinions available for analysis (possibly filtered out by require_success).")
        return

    # Choose model set to evaluate + alignment filtering (each index corresponds to the same opinion)
    if expected_models:
        models_eval = list(expected_models)
    else:
        # If expected_models is not provided, use the intersection across all opinions to avoid misalignment.
        models_eval = sorted(set.intersection(*[set(p.keys()) for p in blocks])) if blocks else []

    if not models_eval:
        print("Unable to determine the evaluation model set (models_eval is empty).")
        return

    aligned_blocks = []
    align_skipped = []

    for preds in blocks:
        missing = [m for m in models_eval if m not in preds]
        if missing:
            if require_all_models:
                align_skipped.append(missing)
            continue
        aligned_blocks.append(preds)

    if require_all_models and align_skipped:
        print(f"\n[Alignment skipped opinions] {len(align_skipped)} total (missing models; counts only).")

    if not aligned_blocks:
        print("No opinions have complete model coverage after alignment (aligned set is empty).")
        return

    # Build pred_lists in models_eval order
    pred_lists = [[p[m] for p in aligned_blocks] for m in models_eval]

    # One-shot evaluation across all models: soft gold by majority vote (tie=0.5)
    eval_out = eval_binary_lists_by_majority_vote(pred_lists, names=models_eval, gold=None)
    gold_out = eval_out.get("gold", [])
    summary = eval_out.get("summary", {})
    per_model = eval_out.get("per_model", {})

    n_blocks = summary.get("n", 0)
    print(f"\nTotal aligned opinions: {n_blocks}")
    print(
        "Soft-gold distribution: "
        f"gold=1.0({summary.get('num_gold_true', 0)}), "
        f"gold=0.0({summary.get('num_gold_false', 0)}), "
        f"gold=0.5({summary.get('num_gold_tie', 0)})"
    )

    # Per-model table
    model_rows = []
    for m in models_eval:
        met = per_model.get(m, {})
        tp = float(met.get("tp", 0.0))
        fp = float(met.get("fp", 0.0))
        fn = float(met.get("fn", 0.0))
        tn = float(met.get("tn", 0.0))
        bal_acc, mcc = derive_bal_acc_and_mcc(tp, fp, fn, tn)

        model_rows.append(
            [
                m,
                met.get("n", 0),
                fmt(met.get("acc", 0.0)),
                fmt(met.get("precision", 0.0)),
                fmt(met.get("recall", 0.0)),
                fmt(met.get("f1", 0.0)),
                fmt(bal_acc),
                fmt(mcc),
                f"{tp:.1f}/{fp:.1f}/{fn:.1f}/{tn:.1f}",
            ]
        )

    print("\n=== Single-model vs soft-gold (sorted by Accuracy) ===")
    print_table(
        sorted(model_rows, key=lambda r: float(r[2]), reverse=True),
        headers=["model", "n", "acc", "prec", "recall", "f1", "bal_acc", "mcc", "tp/fp/fn/tn"],
    )

    # OR/AND combinations evaluated against the same gold_out
    pair_results = []
    idx = {m: i for i, m in enumerate(models_eval)}

    for m1, m2 in itertools.combinations(models_eval, 2):
        i1, i2 = idx[m1], idx[m2]
        p1 = pred_lists[i1]
        p2 = pred_lists[i2]
        y_or = [a or b for a, b in zip(p1, p2)]
        y_and = [a and b for a, b in zip(p1, p2)]

        name_or = f"OR({m1},{m2})"
        name_and = f"AND({m1},{m2})"

        met_or = eval_binary_lists_by_majority_vote([y_or], names=[name_or], gold=gold_out)["per_model"][name_or]
        met_and = eval_binary_lists_by_majority_vote([y_and], names=[name_and], gold=gold_out)["per_model"][name_and]

        for rule, pair_name, met in [("OR", f"{m1} + {m2}", met_or), ("AND", f"{m1} + {m2}", met_and)]:
            tp = float(met.get("tp", 0.0))
            fp = float(met.get("fp", 0.0))
            fn = float(met.get("fn", 0.0))
            tn = float(met.get("tn", 0.0))
            bal_acc, mcc = derive_bal_acc_and_mcc(tp, fp, fn, tn)

            pair_results.append(
                [
                    rule,
                    pair_name,
                    met.get("n", 0),
                    fmt(met.get("acc", 0.0)),
                    fmt(met.get("precision", 0.0)),
                    fmt(met.get("recall", 0.0)),
                    fmt(met.get("f1", 0.0)),
                    fmt(bal_acc),
                    fmt(mcc),
                    f"{tp:.1f}/{fp:.1f}/{fn:.1f}/{tn:.1f}",
                ]
            )

    top_by_f1 = sorted(pair_results, key=lambda r: float(r[6]), reverse=True)[:top_k_pairs]
    top_by_acc = sorted(pair_results, key=lambda r: float(r[3]), reverse=True)[:top_k_pairs]

    print(f"\n=== Two-model rules vs gold: Top {top_k_pairs} (sorted by F1) ===")
    print_table(
        top_by_f1,
        headers=["rule", "pair", "n", "acc", "prec", "recall", "f1", "bal_acc", "mcc", "tp/fp/fn/tn"],
    )

    print(f"\n=== Two-model rules vs gold: Top {top_k_pairs} (sorted by Accuracy) ===")
    print_table(
        top_by_acc,
        headers=["rule", "pair", "n", "acc", "prec", "recall", "f1", "bal_acc", "mcc", "tp/fp/fn/tn"],
    )

    # Best-by-accuracy (kept consistent with the original selection criterion)
    best_single = max(model_rows, key=lambda r: float(r[2])) if model_rows else None
    best_pair = max(pair_results, key=lambda r: float(r[3])) if pair_results else None

    print("\n=== Recommendation (based on Accuracy) ===")
    if best_single:
        print(f"Best single model: {best_single[0]} (Acc={best_single[2]}, F1={best_single[5]})")
    else:
        print("Single model: unavailable.")
    if best_pair:
        print(f"Best two-model rule: {best_pair[0]}({best_pair[1]}) (Acc={best_pair[3]}, F1={best_pair[6]})")


# ============================================================
# 5) Main: no file outputs; print logs + stats + analysis only
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Validation aggregation: load multi-model validated.jsonl -> merge -> print stats + analysis (no files written)."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "gemini-2.5-flash",
            "gpt-5-mini",
            "gpt-5",
            "gemini-2.5-pro",
            "deepseek-reasoner",
            "deepseek-chat",
        ],
        help="List of validation model names to load.",
    )
    parser.add_argument(
        "--min-opinions",
        type=int,
        default=1,
        help="Minimum number of opinions required to keep a merged reviewer entry.",
    )
    parser.add_argument(
        "--threshold-k",
        type=int,
        default=4,
        help="(Compatibility parameter) No longer used for gold; gold = majority vote (tie=0.5).",
    )
    parser.add_argument(
        "--top-k-pairs",
        type=int,
        default=60,
        help="Top-K two-model OR/AND combinations to display.",
    )
    parser.add_argument(
        "--require-success",
        action="store_true",
        help="Use only predictions with llm_call_success=True (per model).",
    )
    parser.add_argument(
        "--analysis-require-all-models",
        action="store_true",
        help="During analysis, drop any opinion missing at least one expected model prediction.",
    )
    args = parser.parse_args()

    if not args.models:
        logger.error("You must specify at least one model name.")
        sys.exit(1)

    logger.info(f"Models to process: {args.models}")

    all_model_results: Dict[str, List[Dict[str, Any]]] = {}
    for model_name in args.models:
        logger.info(f"Loading model '{model_name}'...")
        formatted_results = load_validation_results(model_name)
        if formatted_results:
            all_model_results[model_name] = formatted_results
        else:
            logger.warning(f"Model '{model_name}': no valid data loaded.")

    if not all_model_results:
        logger.error("No model results were successfully loaded.")
        sys.exit(1)

    logger.info(f"Merging results from {len(all_model_results)} models...")
    merged_results = merge_multi_model_results(all_model_results)

    # Filter + sort (keep original ordering intent)
    filtered_results = [r for r in merged_results if len(r.get("opinions", [])) >= args.min_opinions]
    filtered_results.sort(
        key=lambda x: (
            -(x.get("consensus_summary", {}).get("wrong_rate", 0.0)),
            -len(x.get("opinions", [])),
            str(x.get("record_id", "")),
            str(x.get("reviewer_name", "")),
        )
    )

    # Summary stats (logs only)
    stats = {
        "total_reviewers": len(merged_results),
        "filtered_reviewers": len(filtered_results),
        "total_opinions": sum(len(r.get("opinions", [])) for r in merged_results),
        "model_coverage": {model: len(results) for model, results in all_model_results.items()},
        "consensus_summary": {
            "total_opinions_with_multi_models": sum(
                len([o for o in r.get("opinions", []) if (o.get("consensus", {}).get("num_models", 0) >= 2)])
                for r in merged_results
            ),
            "conflicting_opinions": sum(
                int(r.get("consensus_summary", {}).get("conflicting_opinions", 0)) for r in merged_results
            ),
            "total_marked_as_wrong": sum(
                int(r.get("consensus_summary", {}).get("num_consensus_wrong", 0)) for r in merged_results
            ),
        },
    }

    logger.info("\n" + "=" * 60)
    logger.info("Reviewer opinion validation: merge + summary (no files written)")
    logger.info("=" * 60)
    logger.info(f"Total reviewers:        {stats['total_reviewers']}")
    logger.info(f"After filtering:        {stats['filtered_reviewers']} (min_opinions={args.min_opinions})")
    logger.info(f"Total opinions:         {stats['total_opinions']}")
    logger.info(f"Model coverage:         {dict(stats['model_coverage'])}")
    logger.info(f"Multi-model opinions:   {stats['consensus_summary']['total_opinions_with_multi_models']}")
    logger.info(f"Conflicting opinions:   {stats['consensus_summary']['conflicting_opinions']}")
    logger.info(f"Consensus 'wrong':      {stats['consensus_summary']['total_marked_as_wrong']}")

    # Analysis: majority-vote soft gold (tie=0.5); metrics come from eval_binary_lists_by_majority_vote
    analyze_two_model_only(
        filtered_results,
        threshold_k=args.threshold_k,
        top_k_pairs=args.top_k_pairs,
        require_success=args.require_success,
        require_all_models=args.analysis_require_all_models,
        expected_models=args.models,
    )


if __name__ == "__main__":
    main()
