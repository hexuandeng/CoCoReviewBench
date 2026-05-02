#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose
-------
Compare agreement of a list-of-lists field across multiple JSONL files on an
ELEMENT-WISE basis.

For each aligned record, this script compares corresponding elements of the target
outer list (e.g., classify[i] vs classify[i]) between files.

Element-level metrics
---------------------
1) Strict agreement:
   The normalized inner lists must match exactly after sorting (order-insensitive
   within the inner list, but order-preserving across elements).
2) Partial agreement:
   The two inner lists must share at least one common label.

Additional analyses
-------------------
- Co-occurring label pairs: labels that frequently appear together within the same
  inner list (sublist) across unique records.
- Misclassified label pairs: label pairs observed between mismatching corresponding
  elements across files (cartesian product between inner lists when elements differ).

Inputs
------
- Multiple JSONL files matched by a glob pattern.
- Records are aligned by `--key` (default: "id"). If omitted, line index is used.
- The compared field is `--hash-key` (default: "classify").

Outputs
-------
Printed report including:
- Pairwise strict agreement matrix (percent).
- Pairwise partial agreement matrix (percent).
- Top-N co-occurring label pairs.
- Top-N misclassified label pairs.
- Top-N overall label frequencies.
- Per-file weighted averages vs a peer pool, optionally excluding "reference-only"
  files from other rows' averages.

Usage examples
--------------
  python compare_classify_elementwise.py \
    --key id \
    --hash-key classify \
    --file-pattern 'pre_eval/classify/evaluation_*_*_classify.jsonl'

Notes
-----
- Outer list order is treated as meaningful and compared element-wise.
- Inner lists are normalized by sorting and dropping None values.
- "Reference-only" files are still reported, but can be excluded from others'
  peer averages via `--peer-exclude-pattern`.
"""

import argparse
import json
import os
import sys
import glob
from collections import Counter, defaultdict
from itertools import combinations, product
from typing import Dict, Any, List, Optional, Iterable, Tuple


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """Yield parsed JSON objects from a JSONL file (one JSON object per line)."""
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{ln} is not valid JSON: {e}") from e


def normalize_classification(classify_list: List[List[str]]) -> Tuple[Tuple[str, ...], ...]:
    """
    Normalize a list-of-lists classification for consistent comparison.

    - Validates that the field is a list.
    - For each inner list:
        - drops None values
        - sorts labels
        - converts to a tuple
    - Preserves outer list order for element-wise alignment.
    """
    if not isinstance(classify_list, list):
        raise TypeError(f"Expected a list for target field, got {type(classify_list)}")

    return tuple(tuple(sorted([i for i in sublist if i is not None])) for sublist in classify_list)


def build_map(path: str, key_field: Optional[str], hash_key: str, skip_missing: bool) -> Dict[Any, Any]:
    """
    Build a mapping from record id -> (normalized_target_field, split_texts_without_summary).

    This enforces that:
    - the record has `hash_key`
    - the record has `split_texts`
    - lengths of `hash_key` and `split_texts` (excluding summary) match
    """
    mapping = {}
    for idx, obj in enumerate(read_jsonl(path)):
        rec_id = obj.get(key_field) if key_field else idx
        if rec_id is None:
            if skip_missing:
                continue
            raise KeyError(f"{path}: record {idx} missing key_field '{key_field}'")

        if hash_key not in obj:
            if skip_missing:
                continue
            raise KeyError(f"{path}: record {rec_id} missing hash_key '{hash_key}'")

        if "split_texts" not in obj:
            if skip_missing:
                continue
            raise KeyError(f"{path}: record {rec_id} missing 'split_texts'")

        try:
            normalized_class = normalize_classification(obj[hash_key])
            split_texts = obj["split_texts"][:-1]  # last item assumed to be a summary
            if len(normalized_class) != len(split_texts):
                raise ValueError(
                    f"'{hash_key}' and 'split_texts' have mismatched lengths "
                    f"({len(normalized_class)} vs {len(split_texts)})"
                )
            mapping[rec_id] = (normalized_class, split_texts)
        except Exception as e:
            raise RuntimeError(f"{path}: record {rec_id} - failed to process: {e}") from e

    return mapping


def analyze_cooccurrence(all_maps: List[Dict[Any, Any]]) -> Counter:
    """
    Count label co-occurrences within the same inner list (sublist).

    Each record id is counted only once across all_maps (first occurrence wins) to avoid
    double-counting the same example across multiple input files.
    """
    cooccurrence_counts = Counter()
    seen_keys = set()
    for m in all_maps:
        for key, (classified_item, _) in m.items():
            if key not in seen_keys:
                seen_keys.add(key)
                for sublist in classified_item:
                    if len(sublist) > 1:
                        for pair in combinations(sorted(sublist), 2):
                            cooccurrence_counts[pair] += 1
    return cooccurrence_counts


def format_ratio(numer: int, denom: int) -> str:
    """Format a ratio as a percentage string, or N/A if denom==0."""
    if denom == 0:
        return "N/A"
    pct = 100.0 * numer / denom
    return f"{pct:.2f}%"


def analyze_label_counts(all_maps: List[Dict[Any, Any]]) -> Counter:
    """Count frequency of individual labels across all (file, record, element) occurrences."""
    label_counts: Counter = Counter()
    for m in all_maps:
        for key, (classified_item, _) in m.items():
            for sublist in classified_item:
                label_counts.update(sublist)
    return label_counts


def main():
    ap = argparse.ArgumentParser(
        description="Element-wise comparison of a list-of-lists classification field across JSONL files."
    )
    ap.add_argument(
        "--file-pattern",
        default="pre_eval/classify/evaluation_*_*_classify.jsonl",
        help="Glob pattern matching input JSONL files.",
    )
    ap.add_argument("--key", default="id", help="Record ID field to align records (default: 'id').")
    ap.add_argument("--hash-key", default="classify", help="Field name to compare (default: 'classify').")
    ap.add_argument("--skip-missing", action="store_true", help="Skip records missing the key/hash-key/split_texts.")
    ap.add_argument("--top-n-stats", type=int, default=20, help="Number of top items to show in statistics tables.")
    args = ap.parse_args()

    files = sorted(glob.glob(args.file_pattern))
    print(f"[INFO] File pattern : {args.file_pattern}")
    print(f"[INFO] Matched      : {len(files)} file(s)")
    for p in files:
        print(f"       - {p}")

    if len(files) < 2:
        ap.error("File pattern must match at least 2 JSONL files.")

    names = [os.path.basename(p) for p in files]
    maps = [build_map(p, args.key, args.hash_key, args.skip_missing) for p in files]
    key_sets = [set(m.keys()) for m in maps]
    avg_len = []

    strict_numer = defaultdict(lambda: defaultdict(int))
    partial_numer = defaultdict(lambda: defaultdict(int))
    denom = defaultdict(lambda: defaultdict(int))  # total elements compared
    misclassified_pairs = Counter()

    # --- Pairwise ELEMENT-WISE Comparison ---
    for i, j in combinations(range(len(files)), 2):
        map_i, map_j = maps[i], maps[j]
        common_keys = key_sets[i] & key_sets[j]

        total_elements_compared = 0
        for k in common_keys:
            list_i, texts = map_i[k]
            list_j, _ = map_j[k]

            if len(list_i) != len(list_j):
                print(
                    f"[WARN] Key '{k}': element-count mismatch between "
                    f"'{names[i]}' ({len(list_i)}) and '{names[j]}' ({len(list_j)}); skipping record.",
                    file=sys.stderr,
                )
                continue

            total_elements_compared += len(list_i)

            for elem_idx in range(len(list_i)):
                elem_i = list_i[elem_idx]
                elem_j = list_j[elem_idx]

                # 1) Strict element agreement
                if elem_i == elem_j:
                    strict_numer[i][j] += 1

                # 2) Partial element agreement
                if not set(elem_i).isdisjoint(set(elem_j)):
                    partial_numer[i][j] += 1

                # 3) Misclassification pairs between corresponding elements
                if elem_i != elem_j:
                    avg_len.append(len(elem_i))
                    avg_len.append(len(elem_j))
                    for pair in product(sorted(elem_i), sorted(elem_j)):
                        if pair[0] is None or pair[1] is None:
                            misclassified_pairs[tuple(pair)] += 1
                        else:
                            if pair[0] != pair[1]:
                                misclassified_pairs[tuple(sorted(pair))] += 1

        # Symmetrically fill results
        denom[i][j] = denom[j][i] = total_elements_compared
        strict_numer[j][i] = strict_numer[i][j]
        partial_numer[j][i] = partial_numer[i][j]

    # --- Co-occurrence & Frequency Analysis ---
    cooccurrence_stats = analyze_cooccurrence(maps)
    label_frequency_stats = analyze_label_counts(maps)

    # --- Reporting ---
    print("\n=== FINAL REPORT ===")
    print(f"Files compared : {len(files)}")
    print(f"Align key      : {args.key!r}")
    print(f"Target field   : {args.hash_key!r}")
    print("Comparison     : element-wise (outer list index aligned)")
    print("Normalization  : inner lists sorted; None removed\n")

    # Matrices only include non-excluded columns (to match existing behavior)
    header = [""] + [name for idx, name in enumerate(names)]

    print("--- Pairwise Strict Agreement (element-level; exact match after normalization) ---")
    print(",".join(header))
    for i in range(len(files)):
        row = [names[i]]
        for j in range(len(files)):
            row.append("—" if i == j else format_ratio(strict_numer[i][j], denom[i][j]))
        print(",".join(row))

    print("\n--- Pairwise Partial Agreement (element-level; at least one shared label) ---")
    print(",".join(header))
    for i in range(len(files)):
        row = [names[i]]
        for j in range(len(files)):
            row.append("—" if i == j else format_ratio(partial_numer[i][j], denom[i][j]))
        print(",".join(row))

    print(f"\n--- Top {args.top_n_stats} Co-occurring Label Pairs (within the same element) ---")
    if not cooccurrence_stats:
        print("No co-occurring pairs found.")
    else:
        for pair, count in cooccurrence_stats.most_common(args.top_n_stats):
            print(f"{count:<5}  {pair[0]} + {pair[1]}")

    print(f"\n--- Top {args.top_n_stats} Misclassified Label Pairs (between corresponding elements) ---")
    if not misclassified_pairs:
        print("No misclassifications found between differing elements.")
    else:
        for pair, count in misclassified_pairs.most_common(args.top_n_stats):
            sorted_pair = tuple(sorted(pair))
            print(f"{count:<5}  {sorted_pair[0]} vs {sorted_pair[1]}")

    print(f"\n--- Top {args.top_n_stats} Most Frequent Labels (overall) ---")
    if not label_frequency_stats:
        print("No labels found.")
    else:
        for label, count in label_frequency_stats.most_common(args.top_n_stats):
            print(f"{count:<5}  {label}")

    if avg_len:
        print(f"\n[INFO] Average label count in mismatched elements: {sum(avg_len) / len(avg_len):.2f}")

    # --- [Summary] Per-File Weighted Averages (sorted by mean desc) ---
    # Fixed peer pool: all non-reference-only files
    peer_pool = [j for j in range(len(files))]

    rows = []
    for i in range(len(files)):
        peers = [j for j in peer_pool if j != i]
        s_num = 0
        s_den = 0
        p_num = 0

        for j in peers:
            if denom[i][j] <= 0:
                continue
            s_num += strict_numer[i][j]
            s_den += denom[i][j]
            p_num += partial_numer[i][j]
        p_den = s_den

        s_pct = (100.0 * s_num / s_den) if s_den else None
        p_pct = (100.0 * p_num / p_den) if p_den else None
        if s_pct is None and p_pct is None:
            mean_pct = None
        elif s_pct is None:
            mean_pct = p_pct
        elif p_pct is None:
            mean_pct = s_pct
        else:
            mean_pct = (s_pct + p_pct) / 2.0

        rows.append(
            {
                "name": names[i],
                "strict": s_pct,
                "partial": p_pct,
                "mean": mean_pct,
            }
        )

    # Sort: higher mean first; N/A at the bottom
    rows.sort(key=lambda r: (r["mean"] is None, -(r["mean"] or 0)))

    name_w = max([len("Model")] + [len(r["name"]) for r in rows])
    col_w = 15

    print("\n--- Per-File Averages vs Peers (weighted; sorted by mean desc) ---")
    print("[INFO] Rows marked '*' are reference-only and do NOT count as peers for other rows.")
    header = (
        f'{"Model".ljust(name_w)}  '
        f'{"Strict Avg (%)".rjust(col_w)}  '
        f'{"Partial Avg (%)".rjust(col_w)}  '
        f'{"Mean of Two (%)".rjust(col_w)}'
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        s = f'{r["strict"]:.2f}' if r["strict"] is not None else "N/A"
        p = f'{r["partial"]:.2f}' if r["partial"] is not None else "N/A"
        m = f'{r["mean"]:.2f}' if r["mean"] is not None else "N/A"
        print(f'{(r["name"]).ljust(name_w)}  {s.rjust(col_w)}  {p.rjust(col_w)}  {m.rjust(col_w)}')


if __name__ == "__main__":
    main()
