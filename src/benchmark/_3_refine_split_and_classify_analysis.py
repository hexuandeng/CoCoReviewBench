#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose
-------
Compute pairwise agreement statistics across multiple JSONL annotation outputs.

This script expects each JSONL record to include:
- `id`: paper/sample identifier
- `for_judge`: a mapping from sentence_id -> list of labels

It loads multiple JSONL files and computes:
- Pairwise agreement for every file pair:
  - full: exact label-set match
  - partial: one label-set is a subset of the other (but not equal)
  - none: all other cases
- Per-file descriptive stats:
  - average number of labels per sentence
- Per-file summary aggregated from pairwise stats:
  - average full/partial/any-overlap ratios against other files
  - average number of commonly compared sentences (per pair)

Notes
-----
- The script only compares sentences that exist in BOTH files for the SAME paper_id.
- Output is printed as:
  1) JSON summary (machine-friendly)
  2) A Markdown table ranking files by a simple composite score
"""

import os
import glob
import json
from collections import defaultdict
from typing import List, Dict, Set, Tuple


def load_annotations(jsonl_paths: List[str]) -> Dict[str, Dict[str, Dict[str, Set[str]]]]:
    """
    Load multiple JSONL files and build a nested mapping:

        data[file_path][paper_id][sentence_id] -> set(labels)

    Expected JSONL schema per line:
      - id: str (paper/sample id)
      - for_judge: dict[sentence_id, list[str]]
    """
    data: Dict[str, Dict[str, Dict[str, Set[str]]]] = {}

    for path in jsonl_paths:
        per_file: Dict[str, Dict[str, Set[str]]] = defaultdict(dict)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)

                sample_id = obj.get("id")
                if sample_id is None:
                    continue

                for_judge = obj.get("for_judge", {})
                # for_judge example: {"28": ["CLAR-WRT"], "190": ["CLAR-WRT", "QUAL-EXP"], ...}
                for sent_id, labels in for_judge.items():
                    sent_id_str = str(sent_id)
                    per_file[sample_id][sent_id_str] = set(labels)

        data[path] = per_file

    return data


def compute_pairwise_agreement(
    data: Dict[str, Dict[str, Dict[str, Set[str]]]]
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    Compute pairwise agreement stats for every (fi, fj).

    Definitions (strict):
      - full: label sets are exactly equal
      - partial: label sets are not equal, but one is a subset of the other
      - none: all other cases
    """
    files = list(data.keys())
    pair_stats: Dict[Tuple[str, str], Dict[str, float]] = {}

    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            fi, fj = files[i], files[j]
            samples_i = data[fi]
            samples_j = data[fj]

            total = 0
            full = 0
            partial = 0
            none = 0

            # Compare only shared paper_ids.
            common_sample_ids = set(samples_i.keys()) & set(samples_j.keys())
            for sample_id in common_sample_ids:
                sents_i = samples_i[sample_id]
                sents_j = samples_j[sample_id]

                # Compare only shared sentence_ids.
                common_sent_ids = set(sents_i.keys()) & set(sents_j.keys())
                for sent_id in common_sent_ids:
                    labels_i = sents_i[sent_id]
                    labels_j = sents_j[sent_id]

                    total += 1
                    if labels_i == labels_j:
                        full += 1
                    elif labels_i.issubset(labels_j) or labels_j.issubset(labels_i):
                        partial += 1
                    else:
                        none += 1

            if total > 0:
                full_ratio = full / total
                partial_ratio = partial / total
                any_overlap_ratio = (full + partial) / total
            else:
                full_ratio = partial_ratio = any_overlap_ratio = 0.0

            pair_stats[(fi, fj)] = {
                "num_common_sentences": total,
                "full_count": full,
                "partial_count": partial,
                "none_count": none,
                "full_ratio": full_ratio,
                "partial_ratio": partial_ratio,
                "any_overlap_ratio": any_overlap_ratio,
            }

    return pair_stats


def compute_labelset_stats_per_file(
    data: Dict[str, Dict[str, Dict[str, Set[str]]]]
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-file descriptive stats:
      - avg_labels_per_sentence: average label-set size per sentence
    """
    stats_per_file: Dict[str, Dict[str, float]] = {}

    for fpath, samples in data.items():
        total_sentences = 0
        total_labels = 0

        for sent_map in samples.values():
            for label_set in sent_map.values():
                total_sentences += 1
                total_labels += len(label_set)

        avg_labels = total_labels / total_sentences if total_sentences else 0.0
        stats_per_file[fpath] = {"avg_labels_per_sentence": avg_labels}

    return stats_per_file


def summarize_per_file_from_pairs(
    pair_stats: Dict[Tuple[str, str], Dict[str, float]],
    files: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate pairwise stats into per-file averages across all file-pairs involving that file.

    Metrics:
      - avg_full_ratio
      - avg_partial_ratio
      - avg_any_overlap_ratio
      - avg_num_common_sentences
      - avg_relative_accuracy: uses avg_full_ratio as a proxy "accuracy" score
    """
    summary: Dict[str, Dict[str, float]] = {}

    for f in files:
        related_stats = []
        for (fi, fj), st in pair_stats.items():
            if f == fi or f == fj:
                related_stats.append(st)

        if not related_stats:
            continue

        num_pairs = len(related_stats)

        sum_full_ratio = sum(s["full_ratio"] for s in related_stats)
        sum_partial_ratio = sum(s["partial_ratio"] for s in related_stats)
        sum_any_overlap_ratio = sum(s["any_overlap_ratio"] for s in related_stats)
        sum_num_common = sum(s["num_common_sentences"] for s in related_stats)

        sum_full_count = sum(s["full_count"] for s in related_stats)
        sum_partial_count = sum(s["partial_count"] for s in related_stats)
        sum_none_count = sum(s["none_count"] for s in related_stats)

        summary[f] = {
            "num_pairs": num_pairs,
            "avg_full_ratio": sum_full_ratio / num_pairs,
            "avg_partial_ratio": sum_partial_ratio / num_pairs,
            "avg_any_overlap_ratio": sum_any_overlap_ratio / num_pairs,
            "avg_relative_accuracy": sum_full_ratio / num_pairs,
            "avg_num_common_sentences": sum_num_common / num_pairs,
            # Aggregate counts across all related pairs (useful for sanity checks).
            "total_common_sentences_across_pairs": sum_num_common,
            "total_full_count_across_pairs": sum_full_count,
            "total_partial_count_across_pairs": sum_partial_count,
            "total_none_count_across_pairs": sum_none_count,
        }

    return summary


if __name__ == "__main__":
    jsonl_files = sorted(glob.glob("pre_eval/split_final/evaluation_*_medium_split_clean.jsonl"))
    data = load_annotations(jsonl_files)

    # Pairwise agreement across all file pairs.
    pair_stats = compute_pairwise_agreement(data)

    # Per-file: average label-set size.
    labelset_stats = compute_labelset_stats_per_file(data)

    # Per-file: averages derived from pairwise comparisons.
    per_file_pair_summary = summarize_per_file_from_pairs(pair_stats, list(data.keys()))

    def short_name(path: str) -> str:
        return os.path.basename(path)

    # Machine-readable JSON output.
    output = {
        "pairwise_stats": {
            f"{short_name(fi)}|||{short_name(fj)}": stats
            for (fi, fj), stats in pair_stats.items()
        },
        "per_file_summary": {
            short_name(fpath): {
                **per_file_pair_summary.get(fpath, {}),
                **labelset_stats.get(fpath, {}),
            }
            for fpath in data.keys()
        },
    }

    print(json.dumps(output, indent=2, ensure_ascii=False))

    # Human-readable ranking table (Markdown).
    rows = []
    for fname, st in output["per_file_summary"].items():
        full = float(st.get("avg_full_ratio", 0.0) or 0.0)
        any_overlap = float(st.get("avg_any_overlap_ratio", 0.0) or 0.0)
        avg = (full + any_overlap) / 2.0
        rows.append((fname, full, any_overlap, avg))

    rows.sort(key=lambda x: x[3], reverse=True)

    print("\n\n| file | avg_full_ratio | avg_any_overlap_ratio | mean |")
    print("|---|---:|---:|---:|")
    for fname, full, any_overlap, avg in rows:
        print(f"| {fname} | {full:.4f} | {any_overlap:.4f} | {avg:.4f} |")
