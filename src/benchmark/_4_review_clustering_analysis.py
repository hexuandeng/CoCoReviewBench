#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose
-------
Evaluate cross-model consistency of LLM-assigned “discussion point” IDs for reviewer
comments, using the Omega Index for (potentially) overlapping clusterings.

This script:
1) Loads per-model Stage-1 JSONL outputs that contain `discussion_point_assignments`.
2) Reconstructs reviewer comment texts from `new_split_texts` + `sentence_texts`.
3) Computes pairwise Omega Index scores across models on shared records/labels.
4) Reports each model’s average pairwise consistency, and the average number of
   clusters (unique point_ids) produced per (record, label).
5) Optionally exports disagreements (same comment, different assigned point_id)
   for manual inspection.

Input Assumptions
-----------------
Each record in the JSONL is expected to include:
- id (or record_id)
- sentence_texts
- new_split_texts
- discussion_point_assignments:
    {label: {"blocks": [{"block_id": ..., "reviewer_assignments": {...}}, ...]}}

Output
------
- Console logs:
  - Pairwise Omega Index scores (model vs model)
  - Per-model average Omega score (mean over its pairwise scores)
  - Per-model average number of clusters (unique point_ids)
- A JSON file of disagreements for manual review (default: point_assignment_disagreements.json)

Notes
-----
- This script does NOT change or reassign point IDs; it only evaluates consistency.
- Overlapping partitions are supported via `omega_index_overlapping` (imported from utils).
- File paths are constructed from model names; override in code if your layout differs.
"""

import argparse
import json
import logging
from typing import Dict, Any, Set, Tuple, List, Optional
from collections import defaultdict
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import omega_index_overlapping

logger = logging.getLogger(__name__)


def load_model_results(model_names: List[str]) -> Dict[str, Dict[str, Any]]:
    """Load multiple model result files (JSONL).

    Expected path convention (per model name):
      pre_eval/point_ids/evaluation_{name}_medium_split_clean_point_ids.jsonl

    Returns:
      model_results[model_name][record_id] = record_dict
    """
    model_results = {}

    for name in model_names:
        path = f"pre_eval/point_ids/evaluation_{name}_medium_split_clean_point_ids.jsonl"
        records = {}

        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        records[record.get("id", record.get("record_id"))] = record
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.error(f"Failed to parse JSONL | file={path} | line={line_num} | error={e}")
                        continue
        except FileNotFoundError:
            logger.error(f"File not found: {path}")
            continue

        model_results[name] = records
        logger.info(f"Loaded model results | model={name} | records={len(records)}")

    return model_results


def get_comment_key(reviewer_name: str, text: str) -> Optional[Tuple[str, str]]:
    """Build a stable key for matching the *same* comment across model outputs."""
    if not reviewer_name or not text:
        return None
    return (reviewer_name, text.strip())


def _normalize_point_id(x: Any) -> Any:
    """Normalize reviewer_assignments[role] variants (list/dict/scalar) into a scalar-like value."""
    if isinstance(x, list) and len(x) > 0:
        return x[0]
    if isinstance(x, dict) and len(x) > 0:
        return list(x.values())[0]
    return x


def _extract_label_comment_to_pid(record: Dict[str, Any]) -> Dict[str, Dict[Tuple[str, str], Any]]:
    """
    Extract mapping:
      out[label][(reviewer_name, reconstructed_text)] = point_id

    Reconstruction uses `new_split_texts` + `sentence_texts` in the same way as
    the original assignment pipeline.
    """
    out: Dict[str, Dict[Tuple[str, str], Any]] = defaultdict(dict)

    assignments = record.get("discussion_point_assignments", {}) or {}
    sentence_texts = record.get("sentence_texts", []) or []
    new_split_texts = record.get("new_split_texts", []) or []

    for label, data in assignments.items():
        for block in (data.get("blocks", []) or []):
            block_id = block.get("block_id")
            if block_id is None:
                continue
            try:
                block_id = int(block_id)
            except Exception:
                continue

            reviewer_assignments = block.get("reviewer_assignments", {}) or {}

            if not (0 <= block_id < len(new_split_texts)):
                continue

            block_data = new_split_texts[block_id]
            if not (isinstance(block_data, list) and len(block_data) > 0 and isinstance(block_data[0], list)):
                continue

            for role_item in block_data[0]:
                if not (isinstance(role_item, list) and len(role_item) >= 2):
                    continue
                role_name = role_item[0]
                sentence_ids = role_item[1]

                if role_name not in reviewer_assignments:
                    continue

                texts = []
                for sid in sorted(sentence_ids):
                    if 0 <= sid < len(sentence_texts):
                        texts.append(sentence_texts[sid].strip())
                text = "\n".join(texts)

                key = get_comment_key(role_name, text)
                if not key:
                    continue

                pid = _normalize_point_id(reviewer_assignments[role_name])
                out[label][key] = pid

    return out


def compute_avg_num_clusters_per_model(model_results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute each model's average number of clusters (unique point_ids).

    Definition:
      - Restrict to record_ids shared by *all* models.
      - For each shared record and each label in that record:
          clusters = number of unique point_ids among comments for that label.
      - Report the average clusters over all (record, label) instances.

    Returns:
      {model_name: avg_num_clusters}
    """
    model_names = list(model_results.keys())
    if not model_names:
        return {}

    common_records = set.intersection(*[set(results.keys()) for results in model_results.values()])
    if not common_records:
        logger.warning("No shared record_ids across models; cannot compute avg #clusters.")
        return {m: 0.0 for m in model_names}

    per_model_counts: Dict[str, List[int]] = defaultdict(list)

    for model_name, records in model_results.items():
        for record_id in common_records:
            record = records.get(record_id)
            if not record:
                continue
            label_map = _extract_label_comment_to_pid(record)
            for label, cmap in label_map.items():
                if not cmap:
                    continue
                unique_pids = set(cmap.values())
                per_model_counts[model_name].append(len(unique_pids))

    per_model_avg = {
        m: (float(np.mean(per_model_counts[m])) if per_model_counts.get(m) else 0.0)
        for m in model_names
    }
    return per_model_avg


def compute_point_assignment_consistency(model_results: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Compute point-assignment consistency using Omega Index, comparing models'
    point_id assignments for identical comments.

    Returns:
      {
        "pairwise": { "modelA vs modelB": avg_omega, ... },
        "per_model_average": { model: avg_over_pairs, ... },
        "per_model_avg_num_clusters": { model: avg_unique_point_ids, ... },
      }
    """
    logger.info("=" * 72)
    logger.info("Computing cross-model discussion-point consistency (Omega Index)")
    logger.info("=" * 72)

    pair_omega_scores = {}
    model_names = list(model_results.keys())
    common_records = set.intersection(*[set(results.keys()) for results in model_results.values()])

    if not common_records:
        logger.warning("No shared record_ids across models.")
        return {"pairwise": {}, "per_model_average": {}, "per_model_avg_num_clusters": {}}

    logger.info(f"Shared records across all models: {len(common_records)}")

    for i, model1 in enumerate(model_names):
        for model2 in model_names[i + 1:]:
            all_omegas = []

            for record_id in common_records:
                # Collect: {label: {comment_key: point_id}}
                model1_data = defaultdict(dict)
                model2_data = defaultdict(dict)

                # --------------------------
                # Collect model1 data
                # --------------------------
                record1 = model_results[model1][record_id]
                assignments1 = record1.get("discussion_point_assignments", {})

                for label, data in assignments1.items():
                    for block in data.get("blocks", []):
                        block_id = block.get("block_id")
                        if block_id is None:
                            continue
                        try:
                            block_id = int(block_id)
                        except Exception:
                            continue

                        reviewer_assignments = block.get("reviewer_assignments", {})

                        # Reconstruct original text
                        sentence_texts = record1.get("sentence_texts", [])
                        new_split_texts = record1.get("new_split_texts", [])

                        if 0 <= block_id < len(new_split_texts):
                            block_data = new_split_texts[block_id]
                            for role_item in block_data[0]:
                                role_name = role_item[0]
                                sentence_ids = role_item[1]

                                if role_name in reviewer_assignments:
                                    texts = []
                                    for sid in sorted(sentence_ids):
                                        if 0 <= sid < len(sentence_texts):
                                            texts.append(sentence_texts[sid].strip())
                                    text = "\n".join(texts)

                                    key = get_comment_key(role_name, text)
                                    if key:
                                        model1_data[label][key] = _normalize_point_id(reviewer_assignments[role_name])

                # --------------------------
                # Collect model2 data
                # --------------------------
                record2 = model_results[model2][record_id]
                assignments2 = record2.get("discussion_point_assignments", {})

                for label, data in assignments2.items():
                    for block in data.get("blocks", []):
                        block_id = block.get("block_id")
                        if block_id is None:
                            continue
                        try:
                            block_id = int(block_id)
                        except Exception:
                            continue

                        reviewer_assignments = block.get("reviewer_assignments", {})

                        # Reconstruct original text
                        sentence_texts = record2.get("sentence_texts", [])
                        new_split_texts = record2.get("new_split_texts", [])

                        if 0 <= block_id < len(new_split_texts):
                            block_data = new_split_texts[block_id]
                            for role_item in block_data[0]:
                                role_name = role_item[0]
                                sentence_ids = role_item[1]

                                if role_name in reviewer_assignments:
                                    texts = []
                                    for sid in sorted(sentence_ids):
                                        if 0 <= sid < len(sentence_texts):
                                            texts.append(sentence_texts[sid].strip())
                                    text = "\n".join(texts)

                                    key = get_comment_key(role_name, text)
                                    if key:
                                        model2_data[label][key] = _normalize_point_id(reviewer_assignments[role_name])

                # --------------------------
                # Compute Omega per shared label
                # --------------------------
                for label in set(model1_data.keys()) & set(model2_data.keys()):
                    common_keys = set(model1_data[label].keys()) & set(model2_data[label].keys())
                    if len(common_keys) < 2:
                        continue

                    key_to_idx = {k: ii for ii, k in enumerate(sorted(common_keys))}

                    partition1 = defaultdict(set)
                    partition2 = defaultdict(set)
                    for key in common_keys:
                        pid1 = model1_data[label][key]
                        pid2 = model2_data[label][key]
                        idx = key_to_idx[key]
                        partition1[pid1].add(idx)
                        partition2[pid2].add(idx)

                    universe = set(range(len(common_keys)))
                    omega = omega_index_overlapping(
                        list(partition1.values()),
                        list(partition2.values()),
                        universe
                    )
                    all_omegas.append(omega)

            pair_name = f"{model1} vs {model2}"
            if all_omegas:
                avg_omega = np.mean(all_omegas)
                pair_omega_scores[pair_name] = float(avg_omega)
                logger.info(f"Pairwise Omega | {pair_name} | omega={avg_omega:.4f} | labels_evaluated={len(all_omegas)}")
            else:
                logger.warning(f"Pairwise Omega | {pair_name} | no valid label-level comparisons")
                pair_omega_scores[pair_name] = 0.0

    # Per-model average of pairwise scores
    per_model_avg = {}
    model_values = defaultdict(list)

    for pair_str, value in pair_omega_scores.items():
        model1, model2 = pair_str.split(" vs ")
        model_values[model1].append(value)
        model_values[model2].append(value)

    for model, values in model_values.items():
        per_model_avg[model] = float(np.mean(values)) if values else 0.0

    # Per-model average number of clusters (unique point_ids)
    per_model_avg_num_clusters = compute_avg_num_clusters_per_model(model_results)

    return {
        "pairwise": pair_omega_scores,
        "per_model_average": per_model_avg,
        "per_model_avg_num_clusters": per_model_avg_num_clusters,
    }


def extract_disagreements_for_review(model_results: Dict[str, Dict], output_path: str):
    """
    Extract comments where two models assign different point_ids to the *same* comment.
    This is intended for manual audit and debugging.

    Notes:
      - For simplicity, this exports disagreements only between the first two models
        in `--models` (same as the original script behavior).
    """
    logger.info("Extracting disagreements for manual review...")

    model_names = list(model_results.keys())
    if len(model_names) < 2:
        logger.error("Need at least 2 models to extract disagreements.")
        return

    # Preserve original behavior: only compare the first two models.
    model1, model2 = model_names[0], model_names[1]
    common_records = set(model_results[model1].keys()) & set(model_results[model2].keys())

    disagreements = []

    for record_id in common_records:
        sentence_texts = model_results[model1][record_id].get("sentence_texts", [])
        new_split_texts = model_results[model1][record_id].get("new_split_texts", [])

        assignments1 = model_results[model1][record_id].get("discussion_point_assignments", {})
        assignments2 = model_results[model2][record_id].get("discussion_point_assignments", {})

        for label in set(assignments1.keys()) & set(assignments2.keys()):
            comment_to_pid1 = {}
            comment_to_pid2 = {}

            for block in assignments1[label].get("blocks", []):
                block_id = block.get("block_id")
                if block_id is None:
                    continue
                try:
                    block_id = int(block_id)
                except Exception:
                    continue

                reviewer_assignments = block.get("reviewer_assignments", {})

                if 0 <= block_id < len(new_split_texts):
                    block_data = new_split_texts[block_id]
                    for role_item in block_data[0]:
                        role_name = role_item[0]
                        sentence_ids = role_item[1]

                        if role_name in reviewer_assignments:
                            texts = []
                            for sid in sorted(sentence_ids):
                                if 0 <= sid < len(sentence_texts):
                                    texts.append(sentence_texts[sid].strip())
                            text = "\n".join(texts)

                            key = get_comment_key(role_name, text)
                            if key:
                                comment_to_pid1[key] = _normalize_point_id(reviewer_assignments[role_name])

            for block in assignments2[label].get("blocks", []):
                block_id = block.get("block_id")
                if block_id is None:
                    continue
                try:
                    block_id = int(block_id)
                except Exception:
                    continue

                reviewer_assignments = block.get("reviewer_assignments", {})

                if 0 <= block_id < len(new_split_texts):
                    block_data = new_split_texts[block_id]
                    for role_item in block_data[0]:
                        role_name = role_item[0]
                        sentence_ids = role_item[1]

                        if role_name in reviewer_assignments:
                            texts = []
                            for sid in sorted(sentence_ids):
                                if 0 <= sid < len(sentence_texts):
                                    texts.append(sentence_texts[sid].strip())
                            text = "\n".join(texts)

                            key = get_comment_key(role_name, text)
                            if key:
                                comment_to_pid2[key] = _normalize_point_id(reviewer_assignments[role_name])

            common_keys = set(comment_to_pid1.keys()) & set(comment_to_pid2.keys())

            for key in common_keys:
                pid1 = comment_to_pid1[key]
                pid2 = comment_to_pid2[key]

                if pid1 != pid2:
                    disagreements.append({
                        "record_id": record_id,
                        "label": label,
                        "comment": {
                            "role": key[0],
                            "text": key[1]
                        },
                        "models": {
                            model1: {"point_id": pid1},
                            model2: {"point_id": pid2}
                        }
                    })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(disagreements, f, ensure_ascii=False, indent=2)

    logger.info(f"Disagreements exported | count={len(disagreements)} | output={output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute cross-model consistency of discussion point ID assignments (Omega Index)."
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
        help="List of model names to compare (must have corresponding JSONL files)."
    )
    args = parser.parse_args()
    if len(args.models) < 2:
        logger.error("At least two models are required.")
        sys.exit(1)

    logger.info(f"Loading models: {args.models}")
    model_results = load_model_results(args.models)
    if not model_results:
        logger.error("No model results were loaded successfully.")
        sys.exit(1)

    consistency_results = compute_point_assignment_consistency(model_results)

    logger.info("=" * 72)
    logger.info("Consistency results (Omega Index)")
    logger.info("=" * 72)
    logger.info("Pairwise consistency:")
    for pair, score in consistency_results["pairwise"].items():
        logger.info(f"  {pair}: {score:.4f}")

    logger.info("Per-model average consistency (mean over pairwise scores):")
    for model, score in consistency_results["per_model_average"].items():
        logger.info(f"  {model}: {score:.4f}")

    logger.info("Per-model average number of clusters (unique point_ids):")
    for model, avg_k in consistency_results.get("per_model_avg_num_clusters", {}).items():
        logger.info(f"  {model}: {avg_k:.4f}")


if __name__ == "__main__":
    main()
