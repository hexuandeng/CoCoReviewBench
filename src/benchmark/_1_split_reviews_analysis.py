#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose
-------
Compare multiple JSONL outputs that contain clustering/partition predictions (field: `parsed`)
and report:
1) Pairwise Omega Index between models (on elements that both models consider "non-noisy"),
2) Multi-model NA (noisy/last-class) binary evaluation using a majority-vote pseudo-gold,
3) A combined ranking metric: mean(Omega_avg, F1),
4) The ratio of "last class" assignments over all assigned elements.

This script does NOT crawl OpenReview or download PDFs. It is an analysis utility
used after partitioning has already been produced.

Input format (per JSONL line)
-----------------------------
Each line is expected to be a JSON object containing:
- parsed: list[list[int]] (or list of index lists)
    The final cluster (parsed[-1]) is treated as the "noisy / NA" class.

Key metrics
-----------
- Omega Index (overlapping): pairwise agreement between two partitions, computed only
  over the intersection of elements both models place in non-last clusters.
- NA binary metrics (acc/precision/recall/F1): computed for each model against a
  majority-vote pseudo-gold over all models (ties yield gold=0.5).

Notes
-----
- The accuracy/F1 logic is delegated to shared utilities (utils.py).
- File discovery uses a glob pattern. Ensure all matched files have aligned line counts
  and compatible JSON schema.
"""

import json
import glob
import os
import sys
from itertools import combinations
from collections import defaultdict
from pathlib import Path
import numpy as np
import argparse

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import eval_binary_lists_by_majority_vote, fmt, print_table, omega_index_overlapping


# ==============================================================================
# Core helpers (behavior intentionally preserved)
# ==============================================================================

def get_last_class_ratio(file_path):
    """
    Compute the fraction of elements assigned to the final cluster (last class)
    over all assigned elements, aggregated across all lines in a JSONL file.
    """
    total_last = 0
    total_all = 0
    for obj in read_jsonl(file_path):
        parsed = obj.get('parsed')
        if not parsed:
            continue
        partition = [set(s) for s in parsed]
        total_all += sum(len(s) for s in partition)
        total_last += len(partition[-1]) if partition else 0
    return (total_last / total_all) if total_all > 0 else 0.0


def read_jsonl(file_path):
    """
    Read a JSONL file line-by-line as a generator.

    Invalid JSON lines are skipped with a warning.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                yield json.loads(line.strip())
            except json.JSONDecodeError:
                print(f"[WARN] Skipping invalid JSON line in {file_path}: {line.strip()}")
                continue


# ==============================================================================
# Omega computation (pairwise)
# ==============================================================================

def calculate_omega_for_line(parsed1, parsed2):
    partition1 = [set(s) for s in (parsed1 or [])]
    partition2 = [set(s) for s in (parsed2 or [])]

    # Treat the final cluster as noisy/NA and exclude it from "clean" partition sets
    non_noisy_sets1 = partition1[:-1] if partition1 else []
    non_noisy_sets2 = partition2[:-1] if partition2 else []

    non_noisy_els1 = set().union(*non_noisy_sets1) if non_noisy_sets1 else set()
    non_noisy_els2 = set().union(*non_noisy_sets2) if non_noisy_sets2 else set()

    # Only evaluate omega on elements that BOTH models consider non-noisy
    common_clean = non_noisy_els1 & non_noisy_els2
    filt1 = [s & common_clean for s in non_noisy_sets1 if (s & common_clean)]
    filt2 = [s & common_clean for s in non_noisy_sets2 if (s & common_clean)]

    # Degenerate case: omega is ill-defined for <2 elements; treat as perfect agreement
    if len(common_clean) < 2:
        return 1.0
    return omega_index_overlapping(filt1, filt2, common_clean)


def get_comparison_stats(file1_path, file2_path):
    total_lines = 0
    omega_non_noisy_list = []

    gen1 = read_jsonl(file1_path)
    gen2 = read_jsonl(file2_path)

    for i, (obj1, obj2) in enumerate(zip(gen1, gen2)):
        line_num = i + 1
        try:
            parsed1 = obj1.get('parsed')
            parsed2 = obj2.get('parsed')
            if parsed1 is None or parsed2 is None:
                continue

            omega = calculate_omega_for_line(parsed1, parsed2)
            total_lines += 1
            omega_non_noisy_list.append(omega)
        except Exception as e:
            print(f"[ERROR] Failed at line {line_num} comparing {file1_path} vs {file2_path}: {e}")
            continue

    if total_lines == 0:
        return {
            'average_omega_non_noisy': 0.0,
            'lines_processed': 0
        }

    return {
        'average_omega_non_noisy': float(np.mean(omega_non_noisy_list)) if omega_non_noisy_list else 0.0,
        'lines_processed': total_lines
    }


def normalize_file(f):
    """
    Create a short, stable model/file label from a filename.

    Note: logic preserved from the original script for compatibility with existing naming.
    """
    return os.path.basename(f).split(".cc")[-1].split("split")[0].strip("_")


# ==============================================================================
# NA binary prediction construction for all models at once
# ==============================================================================

def build_global_na_pred_lists(files):
    """
    Build NA (noisy/last-class) binary predictions for every model over a common alignment.

    Returns
    -------
    names: list[str]
        Display names derived from filenames.
    pred_lists: list[list[bool]]
        For each model, a flattened boolean list indicating whether each element
        (for each line) is in that model's last cluster.

    Alignment policy
    ----------------
    For each line, we take the union of all elements appearing in any cluster across
    all models. We then sort these elements and append per-model NA membership.
    """
    names = [normalize_file(f) for f in files]
    preds_by_file = {f: [] for f in files}

    gens_all = [read_jsonl(f) for f in files]

    for objs in zip(*gens_all):
        parsed_list = [(o.get('parsed') or []) for o in objs]

        # Union of elements across ALL clusters across ALL models for this line
        all_elements = set()
        noisy_sets = []
        for parsed in parsed_list:
            part = [set(s) for s in parsed] if parsed else []
            if part:
                noisy_sets.append(part[-1])
                for s in part:
                    all_elements.update(s)
            else:
                noisy_sets.append(set())

        if not all_elements:
            continue

        elements_sorted = sorted(all_elements)

        # Append per-model NA membership aligned to elements_sorted
        for f, noisy_set in zip(files, noisy_sets):
            preds_by_file[f].extend([(e in noisy_set) for e in elements_sorted])

    pred_lists = [preds_by_file[f] for f in files]
    return names, pred_lists


if __name__ == '__main__':
    # 1) Discover input files
    parser = argparse.ArgumentParser(description="Compare partition outputs across JSONL files.")
    parser.add_argument(
        "--file_pattern",
        default="pre_eval/split/evaluation_*_split.jsonl",
        help="Glob pattern for input JSONL files."
    )
    args = parser.parse_args()

    file_pattern = args.file_pattern
    files = sorted(glob.glob(file_pattern))
    for it in glob.glob("pre_eval/split/evaluation_Qwen3-8B*_split.jsonl"):
        files.remove(it)

    if len(files) < 2:
        print(f"[ERROR] Need at least 2 files to compare (pattern: '{file_pattern}').")
        exit()

    print(f"[INFO] Found {len(files)} file(s) for comparison:")
    for f in files:
        print(f"  - {normalize_file(f)}")
    print("-" * 60)

    # ======================================================================
    # A) Pairwise Omega Index (pairwise by definition)
    # ======================================================================

    pairwise_results = defaultdict(dict)  # Omega matrix
    file_scores = {f: {'total_omega': 0.0, 'count': 0} for f in files}

    for file1, file2 in combinations(files, 2):
        print(f"[INFO] Omega (pairwise): {normalize_file(file1)} vs {normalize_file(file2)}")
        stats = get_comparison_stats(file1, file2)
        omega = stats['average_omega_non_noisy']

        pairwise_results[file1][file2] = omega
        pairwise_results[file2][file1] = omega

        file_scores[file1]['total_omega'] += omega
        file_scores[file1]['count'] += 1
        file_scores[file2]['total_omega'] += omega
        file_scores[file2]['count'] += 1

    print("\n=== Pairwise Omega Index (non-noisy intersection only) ===")
    basenames = [normalize_file(f) for f in files]
    col_width = max(len(name) for name in basenames) + 2

    header = f"{'':<{col_width}}" + " | " + " | ".join([f"{name:^{col_width}}" for name in basenames])
    print(header)
    print("-" * len(header))

    for i, file_row in enumerate(files):
        row_str = f"{basenames[i]:<{col_width}}"
        for j, file_col in enumerate(files):
            if i == j:
                cell = fmt(1.0)
            else:
                score = pairwise_results[file_row].get(file_col, None)
                cell = fmt(score) if isinstance(score, float) else "N/A"
            row_str += f" | {cell:^{col_width}}"
        print(row_str)

    # ======================================================================
    # B) NA binary evaluation via majority-vote pseudo-gold (multi-model)
    # ======================================================================

    names, pred_lists = build_global_na_pred_lists(files)
    eval_out = eval_binary_lists_by_majority_vote(pred_lists, names=names, gold=None)

    summary = eval_out.get("summary", {})
    per_model = eval_out.get("per_model", {})

    print("\n=== NA Binary Evaluation (majority-vote pseudo-gold; tie=0.5) ===")
    print(f"[INFO] n              = {summary.get('n', 0)}")
    print(f"[INFO] gold=True     = {summary.get('num_gold_true', 0)}")
    print(f"[INFO] gold=False    = {summary.get('num_gold_false', 0)}")
    print(f"[INFO] gold=0.5 (tie)= {summary.get('num_gold_tie', 0)}")

    rows = []
    for name in names:
        met = per_model.get(name, {})
        rows.append([
            name,
            met.get("n", 0),
            fmt(met.get("acc", 0.0)),
            fmt(met.get("precision", 0.0)),
            fmt(met.get("recall", 0.0)),
            fmt(met.get("f1", 0.0)),
        ])

    # Sort by F1 (desc)
    rows.sort(key=lambda r: float(r[5]), reverse=True)
    print_table(rows, headers=["Model", "n", "acc", "prec", "recall", "f1"])

    # ======================================================================
    # C) Combined ranking: Omega(avg) + majority-vote acc/f1
    # ======================================================================

    omega_rows = []
    for f, name in zip(files, names):
        avg_omega = (file_scores[f]['total_omega'] / file_scores[f]['count']) if file_scores[f]['count'] else None
        met = per_model.get(name, {})
        acc = met.get("acc", 0.0)
        f1 = met.get("f1", 0.0)
        mean_val = (avg_omega + f1) / 2.0 if avg_omega is not None else None

        omega_rows.append([
            name,
            file_scores[f]['count'],
            fmt(avg_omega) if avg_omega is not None else "N/A",
            fmt(acc),
            fmt(f1),
            fmt(mean_val) if mean_val is not None else "N/A",
        ])

    omega_rows.sort(key=lambda r: (float(r[5]) if r[5] != "N/A" else -1e18), reverse=True)

    print("\n=== Combined Score (Omega_avg + F1, and (Omega+F1)/2) ===")
    print_table(omega_rows, headers=["File", "#Compared(Omega)", "Omega(avg)", "acc", "f1", "Mean(O+F)/2"])

    # ======================================================================
    # D) Last-class ratio (noisy/NA prevalence)
    # ======================================================================

    ratio_rows = []
    for f, name in zip(files, names):
        ratio_rows.append([name, fmt(get_last_class_ratio(f))])

    print("\n=== Last-Class Ratio (fraction of elements in final cluster) ===")
    print_table(ratio_rows, headers=["File", "LastClassRatio"])
