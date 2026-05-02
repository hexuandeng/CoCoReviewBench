#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose
-------
Aggregate and analyze *author refutation detection* outputs from multiple LLMs.

Given per-model JSONL files (one record per paper) that contain
`author_refutation_analysis`, this script:
1) Loads each model's outputs and normalizes them into a reviewer-centric format.
2) Merges results across models by (record_id, reviewer_name).
3) Computes per-opinion cross-model agreement and conflict rates.
4) Evaluates each model (and OR/AND pairs) against a *soft majority-vote gold*:
   - gold=1.0 if True votes > half
   - gold=0.0 if True votes < half
   - gold=0.5 if tie

This script prints summary statistics and evaluation tables to stdout and does
not write any output files.

Input
-----
- Expected files (per model):
  pre_eval/refutations/evaluation_{model_name}_medium_split_clean_refutations.jsonl

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
import sys
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict, Counter
import itertools

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import eval_binary_lists_by_majority_vote, fmt, print_table, derive_bal_acc_and_mcc

logger = logging.getLogger(__name__)


def load_model_results(model_name: str) -> List[Dict[str, Any]]:
    """Load a single model's refutation detection results from JSONL."""
    input_path = f"pre_eval/refutations/evaluation_{model_name}_medium_split_clean_refutations.jsonl"
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

                    refutation_analysis = record.get("author_refutation_analysis", {})
                    if not refutation_analysis:
                        logger.warning(f"Record {record_id}: missing 'author_refutation_analysis'; skipping.")
                        continue

                    # Create one entry per reviewer
                    for reviewer_name, reviewer_data in refutation_analysis.items():
                        if not reviewer_data.get("results"):
                            continue

                        review_entry = {
                            "record_id": record_id,
                            "reviewer_name": reviewer_name,
                            "num_opinions": reviewer_data.get("num_opinions", 0),
                            "author_responses_combined": reviewer_data.get("author_responses_combined", ""),
                            "opinions": [],
                            "model": model_name,
                        }

                        for result_item in reviewer_data["results"]:
                            review_entry["opinions"].append(
                                {
                                    "block_id": result_item.get("block_id"),
                                    "text": result_item.get("opinion_text"),
                                    "author_refutes": result_item.get("author_refutes", False),
                                }
                            )

                        results.append(review_entry)

                except json.JSONDecodeError as e:
                    logger.error(f"Line {line_num}: JSON parse failed: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Line {line_num}: failed to process record: {e}")
                    continue

    except FileNotFoundError:
        logger.error(f"File not found: {input_path}")
        sys.exit(1)

    logger.info(f"Loaded model '{model_name}': {len(results)} reviewers.")
    return results


def merge_multi_model_results(model_results: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Merge multiple models' results, grouped by (record_id, reviewer_name).

    The merged output keeps opinion-level results keyed by opinion text.
    """
    merged_groups = defaultdict(
        lambda: {
            "record_id": "",
            "reviewer_name": "",
            "author_responses_combined": "",
            "models": set(),
            "opinion_results": {},  # key: opinion_text
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

            for opinion in record["opinions"]:
                opinion_text = opinion["text"]
                if opinion_text not in merged_groups[key]["opinion_results"]:
                    merged_groups[key]["opinion_results"][opinion_text] = {
                        "block_id": opinion["block_id"],
                        "refutes_by_model": {},
                    }

                merged_groups[key]["opinion_results"][opinion_text]["refutes_by_model"][model_name] = opinion[
                    "author_refutes"
                ]

    merged_list: List[Dict[str, Any]] = []
    for _, group in merged_groups.items():
        opinions_list: List[Dict[str, Any]] = []
        conflict_count = 0
        total_count = 0

        for opinion_text, data in group["opinion_results"].items():
            refute_values = list(data["refutes_by_model"].values())

            if len(refute_values) >= 2:
                all_agree = len(set(refute_values)) == 1
                agreement_rate = sum(bool(v) for v in refute_values) / len(refute_values)
                conflict_count += 0 if all_agree else 1
                total_count += 1
            else:
                all_agree = True
                agreement_rate = 0.0

            opinions_list.append(
                {
                    "text": opinion_text,
                    "block_id": data["block_id"],
                    "refutes_by_model": data["refutes_by_model"],
                    "consensus": {
                        "all_agree": all_agree,
                        "agreement_rate": agreement_rate,
                        "num_models": len(data["refutes_by_model"]),
                        "num_refutes": sum(bool(v) for v in refute_values),
                    },
                }
            )

        opinions_list.sort(key=lambda x: (x["block_id"] if x["block_id"] is not None else 10**18))

        entry = {
            "record_id": group["record_id"],
            "reviewer_name": group["reviewer_name"],
            "author_responses_combined": group["author_responses_combined"],
            "num_models": len(group["models"]),
            "opinions": opinions_list,
            "consensus_summary": {
                "total_opinions": len(opinions_list),
                "conflicting_opinions": conflict_count,
                "consensus_rate": (total_count - conflict_count) / total_count if total_count > 0 else 1.0,
            },
        }

        merged_list.append(entry)

    return merged_list


# ============================================================
# Analysis helpers
# ============================================================
def extract_blocks_from_merged_reviewers(merged_reviewers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract opinion blocks from merged reviewer entries.

    This strictly iterates over merged_reviewers[*].opinions[*] to avoid accidental
    over-counting. Additionally deduplicates by (record_id, reviewer_name, block_id, text).
    """
    blocks: List[Dict[str, Any]] = []
    seen = set()

    for r in merged_reviewers:
        rid = r.get("record_id", "")
        rname = r.get("reviewer_name", "")
        for op in r.get("opinions", []) or []:
            rbm = op.get("refutes_by_model")
            if not isinstance(rbm, dict) or not rbm:
                continue

            block = {
                "record_id": rid,
                "reviewer_name": rname,
                "block_id": op.get("block_id"),
                "text": op.get("text"),
                "refutes_by_model": rbm,
            }

            key = (rid, rname, block["block_id"], block["text"])
            if key in seen:
                continue
            seen.add(key)

            blocks.append(block)

    blocks.sort(
        key=lambda b: (
            b["record_id"],
            b["reviewer_name"],
            b["block_id"] if b["block_id"] is not None else 10**18,
        )
    )
    return blocks


def analyze_blocks(blocks: List[Dict[str, Any]], top_k_pairs: int = 15) -> None:
    """
    Evaluate model predictions against a soft majority-vote gold label.

    Key behaviors:
      - Gold is derived by majority vote over *all models* (tie => 0.5).
      - Metrics (acc/prec/recall/f1) are computed via eval_binary_lists_by_majority_vote.
      - OR/AND pair rules are evaluated using the same gold.
    """
    if not blocks:
        print("No opinion blocks with 'refutes_by_model' were found. Please check merged data.")
        return

    all_models = sorted({m for b in blocks for m in b["refutes_by_model"].keys()})
    M = len(all_models)
    if M == 0:
        print("No model names were found in 'refutes_by_model'.")
        return

    print(f"\nNumber of models: {M}")
    print("Models:", all_models)
    print("Gold rule: True votes > half => gold=1.0; < half => gold=0.0; == half => gold=0.5")

    # Keep only blocks with predictions from *all* models (alignment + valid majority vote)
    used_blocks = []
    skipped = 0
    for b in blocks:
        r = b["refutes_by_model"]
        if all(m in r for m in all_models):
            used_blocks.append(b)
        else:
            skipped += 1

    if skipped:
        print(f"\nSkipped blocks (missing at least one model prediction): {skipped}")
    print(f"Blocks used for evaluation (complete coverage): {len(used_blocks)}")

    if not used_blocks:
        print("No blocks have complete model coverage; cannot run evaluation.")
        return

    # Build aligned prediction lists
    pred_lists = [[bool(b["refutes_by_model"][m]) for b in used_blocks] for m in all_models]

    # Distribution of True vote counts
    vote_counts = [sum(pl[i] for pl in pred_lists) for i in range(len(used_blocks))]
    print("\nDistribution of 'True' votes per block (count -> frequency):")
    print(dict(sorted(Counter(vote_counts).items(), key=lambda x: x[0])))

    # Evaluate all models at once (soft gold; tie=0.5)
    eval_out = eval_binary_lists_by_majority_vote(pred_lists, names=all_models, gold=None)
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

    # Single-model table
    model_rows: List[List[Any]] = []
    for m in all_models:
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

    print("\n=== Per-model agreement with soft majority-vote gold (sorted by F1) ===")
    model_rows.sort(key=lambda r: float(r[5]), reverse=True)
    print_table(
        model_rows,
        headers=["model", "n", "acc", "prec", "recall", "f1", "bal_acc", "mcc", "tp/fp/fn/tn"],
    )

    # OR/AND pair rules evaluated against the same gold
    pair_rows: List[List[Any]] = []
    for m1, m2 in itertools.combinations(all_models, 2):
        p1 = [bool(b["refutes_by_model"][m1]) for b in used_blocks]
        p2 = [bool(b["refutes_by_model"][m2]) for b in used_blocks]
        y_or = [a or b for a, b in zip(p1, p2)]
        y_and = [a and b for a, b in zip(p1, p2)]

        r_or = eval_binary_lists_by_majority_vote(
            [y_or], names=[f"OR({m1},{m2})"], gold=gold_out
        )["per_model"][f"OR({m1},{m2})"]
        r_and = eval_binary_lists_by_majority_vote(
            [y_and], names=[f"AND({m1},{m2})"], gold=gold_out
        )["per_model"][f"AND({m1},{m2})"]

        for rule_name, met in [("OR", r_or), ("AND", r_and)]:
            tp = float(met.get("tp", 0.0))
            fp = float(met.get("fp", 0.0))
            fn = float(met.get("fn", 0.0))
            tn = float(met.get("tn", 0.0))
            bal_acc, mcc = derive_bal_acc_and_mcc(tp, fp, fn, tn)

            pair_rows.append(
                [
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
                ]
            )

    top_by_f1 = sorted(pair_rows, key=lambda r: float(r[6]), reverse=True)[:top_k_pairs]
    top_by_acc = sorted(pair_rows, key=lambda r: float(r[3]), reverse=True)[:top_k_pairs]

    print(f"\n=== Two-model rules vs gold: Top {top_k_pairs} (sorted by F1; gold=tie=0.5) ===")
    print_table(
        top_by_f1,
        headers=["rule", "pair", "n", "acc", "prec", "recall", "f1", "bal_acc", "mcc", "tp/fp/fn/tn"],
    )

    print(f"\n=== Two-model rules vs gold: Top {top_k_pairs} (sorted by Accuracy; gold=tie=0.5) ===")
    print_table(
        top_by_acc,
        headers=["rule", "pair", "n", "acc", "prec", "recall", "f1", "bal_acc", "mcc", "tp/fp/fn/tn"],
    )

    best_single = model_rows[0] if model_rows else None
    best_pair = top_by_f1[0] if top_by_f1 else None

    print("\n=== Recommendation (based on F1) ===")
    if best_single:
        print(f"Best single model: {best_single[0]}  (F1={best_single[5]}, Acc={best_single[2]})")
    if best_pair:
        print(f"Best two-model rule: {best_pair[0]}({best_pair[1]})  (F1={best_pair[6]}, Acc={best_pair[3]})")


# ============================================================
# Main: print stats and analysis; do not write any files
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load and merge multi-model refutation detection outputs, then print statistics and agreement analysis (no files written)."
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "gpt-5-mini",
            "gpt-5",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "deepseek-reasoner",
            "deepseek-chat",
        ],
        help="List of model names to load (maps to expected JSONL paths).",
    )
    parser.add_argument(
        "--min-opinions",
        type=int,
        default=1,
        help="Minimum number of opinions required to keep a merged reviewer entry.",
    )
    parser.add_argument(
        "--top-k-pairs",
        type=int,
        default=15,
        help="Top-K OR/AND model pairs to display.",
    )
    parser.add_argument(
        "--input-merged-json",
        default="",
        help="Reserved for future use: if provided, would read an already-merged JSON and skip JSONL loading/merging.",
    )

    args = parser.parse_args()
    logger.info(f"Models to process: {args.models}")

    all_model_results: Dict[str, List[Dict[str, Any]]] = {}
    for model_name in args.models:
        formatted_results = load_model_results(model_name)
        if formatted_results:
            all_model_results[model_name] = formatted_results

    if not all_model_results:
        logger.error("No model results were successfully loaded.")
        sys.exit(1)

    logger.info(f"Merging results from {len(all_model_results)} models...")
    merged_results = merge_multi_model_results(all_model_results)

    # Filter and sort
    filtered_results = [r for r in merged_results if len(r.get("opinions", [])) >= args.min_opinions]
    filtered_results.sort(key=lambda x: (x.get("record_id", ""), x.get("reviewer_name", "")))

    # Summary stats (printed via logs; no files written)
    model_coverage = {model: len(results) for model, results in all_model_results.items()}

    stats = {
        "total_reviewers": len(merged_results),
        "filtered_reviewers": len(filtered_results),
        "total_opinions": sum(len(r.get("opinions", [])) for r in merged_results),
        "model_coverage": model_coverage,
        "consensus_summary": {
            "total_opinions_with_multi_models": sum(
                len([o for o in r.get("opinions", []) if o.get("consensus", {}).get("num_models", 0) >= 2])
                for r in merged_results
            ),
            "conflicting_opinions": sum(r.get("consensus_summary", {}).get("conflicting_opinions", 0) for r in merged_results),
        },
    }

    logger.info("\n" + "=" * 60)
    logger.info("Refutation detection aggregation: merge + summary (no files written)")
    logger.info("=" * 60)
    logger.info(f"Total reviewers:        {stats['total_reviewers']}")
    logger.info(f"After filtering:        {stats['filtered_reviewers']} (min_opinions={args.min_opinions})")
    logger.info(f"Total opinions:         {stats['total_opinions']}")
    logger.info(f"Model coverage:         {dict(stats['model_coverage'])}")
    logger.info(f"Multi-model opinions:   {stats['consensus_summary']['total_opinions_with_multi_models']}")
    logger.info(f"Conflicting opinions:   {stats['consensus_summary']['conflicting_opinions']}")

    # Analysis: extract blocks (avoids over-counting) + soft majority-vote evaluation
    blocks = extract_blocks_from_merged_reviewers(filtered_results)
    analyze_blocks(blocks, top_k_pairs=args.top_k_pairs)


if __name__ == "__main__":
    main()
