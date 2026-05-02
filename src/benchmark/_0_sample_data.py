#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose
-------
Sample a balanced subset of OpenReview-derived paper records after filtering,
and write the selected records back out as raw JSONL lines.

This script assumes each input record already contains a PDF correctness flag
(e.g., produced by an upstream crawler/downloader):

PDF correctness field (assumed precomputed)
------------------------------------------
- PDF_version_correct: "openreview", "arxiv", or False (tri-state)
- PDF_path: local path to the chosen PDF, or None if not available/readable

Sampling overview
-----------------
For each (org, year):
1) Load a cleaned JSONL input file.
2) Filter out records that:
   - are withdrawn/desk-rejected (based on `decision` text),
   - have already been used in a previous evaluation set (by title),
   - have fewer than N reviewer overall scores,
   - do not have sufficient author replies to reviewer blocks,
   - lack overall score data, or do not fall into a valid score bracket,
   - do not have a "correct" PDF (PDF_version_correct is falsy).
3) Group remaining records by score bracket.
4) Allocate a total quota K evenly across brackets.
5) Within each bracket, select up to the bracket quota while approximately
   preserving the bracket-internal decision distribution.
6) Write selected records to `{org}.cc_{year}/{org}.cc_{year}_sample.jsonl`,
   adding a `score_bracket` field to each record.
"""

import sys
import json
import math
import random
import argparse
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict


BRACKETS = ("0-3", "3-5", "5-7", "7-10")


def determine_bracket(score: float) -> Optional[str]:
    """Map a numeric score to a bracket label."""
    if score is None:
        return None
    if 0.0 <= score <= 3.0:
        return "0-3"
    if 3.0 < score <= 5.0:
        return "3-5"
    if 5.0 < score <= 7.0:
        return "5-7"
    if 7.0 < score <= 10.0:
        return "7-10"
    return None


def has_at_least_n_reviewers(clean_record: Dict[str, Any], n: int = 3) -> bool:
    """
    Count reviewer mains (first post role == 'Reviewer' with a 'scores' dict)
    across threads and check if there are at least n.
    """
    count = 0
    for thread in clean_record.get("reviews", []):
        if not thread or not isinstance(thread, list):
            continue
        first = thread[0]
        if (
            isinstance(first, (list, tuple)) and
            len(first) == 2 and
            first[0] == "Reviewer" and
            isinstance(first[1], dict) and
            isinstance(first[1].get("scores", None), dict)
        ):
            count += 1
    return count >= n


def normalize_overall(score: float, org: str, year: int) -> float:
    if org == "ICLR" and year == 2020:
        norm = (score - 1.0) * (10.0 / 7.0)   # 1–8  -> 0–10
    else:
        norm = (score - 1.0) * (10.0 / 9.0)   # 1–10 -> 0–10
    return max(0.0, min(10.0, norm))          # clamp


def extract_reviewer_overall_scores(clean_record: Dict[str, Any], org: str, year: int) -> List[float]:
    scores: List[float] = []
    for thread in clean_record.get("reviews", []):
        if not thread or not isinstance(thread, list):
            continue
        first = thread[0]
        if (
            isinstance(first, (list, tuple)) and
            len(first) == 2 and
            first[0] == "Reviewer" and
            isinstance(first[1], dict)
        ):
            scores_dict = first[1].get("scores", {})
            overall = scores_dict.get("Overall", None)
            if isinstance(overall, (int, float)):
                scores.append(normalize_overall(float(overall), org, year))  # <-- normalize here
    return scores


def read_lines(path: Optional[str]) -> List[str]:
    """Read all lines from a file path, or stdin if path is '-' or None."""
    if path and path != "-":
        with open(path, "r", encoding="utf-8") as f:
            return f.readlines()
    return sys.stdin.readlines()


def is_withdrawn(obj: Dict[str, Any]) -> bool:
    """
    Treat withdrawn / desk-rejected papers as excluded.

    Heuristic: check `decision` string for keywords.
    """
    decision = str(obj.get("decision") or "").lower()
    return ("withdraw" in decision) or ("desk" in decision)


def reply_all_ok(obj: Dict[str, Any]) -> Tuple[List[float], bool]:
    """
    Check that author replies exist for (most) reviewer review blocks.

    Definition (as implemented)
    ---------------------------
    For each review block:
    - If it contains any non-empty Reviewer text, count it as a "reviewer block".
    - If it contains any non-empty Author text, count it as an "author-replied block".

    The function returns True if:
        author_replied_blocks / reviewer_blocks > 0.75

    Notes
    -----
    - The return type annotation is preserved from the original code.
    - Upstream filtering enforces at least N reviewer blocks, so division-by-zero
      should not occur in normal usage.
    """
    overalls: List[float] = [0, 0]  # [author_replied_blocks, reviewer_blocks]
    reviews = obj.get("reviews") or []
    for block in reviews:
        if not isinstance(block, list):
            continue
        has_reviewer = False
        has_author_reply = False
        for pair in block:
            if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                continue
            role, payload = pair
            flat = "".join(payload["value"].values()).strip()
            role_l = str(role).lower()
            if role_l == "reviewer" and len(flat):
                has_reviewer = True
            elif role_l == "author" and len(flat):
                has_author_reply = True
        if has_reviewer:
            overalls[-1] += 1
        if has_author_reply:
            overalls[0] += 1

    return overalls[0] / overalls[-1] > 0.75


def allocate_quota_evenly(counts: Dict[str, int], k: int) -> Dict[str, int]:
    """
    Allocate a total quota `k` as evenly as possible across BRACKETS.

    Strategy
    --------
    Round-robin allocation in BRACKETS order:
    - Each round, give +1 to each bracket that still has remaining capacity,
      until we reach k or all brackets are saturated.
    """
    alloc = {b: 0 for b in BRACKETS}
    if k <= 0:
        return alloc
    total_capacity = sum(counts.get(b, 0) for b in BRACKETS)
    target = min(k, total_capacity)
    given = 0

    while given < target and any(alloc[b] < counts.get(b, 0) for b in BRACKETS):
        progressed = False
        for b in BRACKETS:
            if given >= target:
                break
            cap = counts.get(b, 0)
            if alloc[b] < cap:
                alloc[b] += 1
                given += 1
                progressed = True
        if not progressed:
            break
    return alloc


def _stratified_select_by_decision(items: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """
    Select k items while keeping the selection's decision distribution close to the
    original items' decision distribution.

    Important
    ---------
    - The input order of `items` is preserved as the tie-breaker.
    - No extra sorting is performed here (so "earlier" items are preferred).
    """
    n = len(items)
    if k <= 0:
        return []
    if k >= n:
        return items[:k]

    # Original distribution
    per_decision = defaultdict(int)
    for it in items:
        d = str(it.get("decision", "")).lower()
        per_decision[d] += 1

    # Target quotas by proportional allocation (largest remainder method)
    targets: Dict[str, int] = {}
    remainders = []
    total_assigned = 0
    for d, cnt in per_decision.items():
        q = cnt * k / n
        base = int(math.floor(q))
        targets[d] = min(base, cnt)
        total_assigned += targets[d]
        remainders.append((q - base, d))

    remainders.sort(reverse=True)
    idx = 0
    safety_loops = 0
    while total_assigned < k and remainders and safety_loops < 2:
        progressed = False
        while total_assigned < k and idx < len(remainders):
            _, d = remainders[idx]
            if targets[d] < per_decision[d]:
                targets[d] += 1
                total_assigned += 1
                progressed = True
            idx += 1
        if not progressed:
            break
        idx = 0
        safety_loops += 1

    # Pick in input order while meeting per-decision targets
    chosen: List[Dict[str, Any]] = []
    chosen_cnt = defaultdict(int)
    leftovers: List[Dict[str, Any]] = []
    for it in items:
        d = str(it.get("decision", "")).lower()
        if chosen_cnt[d] < targets.get(d, 0):
            chosen.append(it)
            chosen_cnt[d] += 1
            if len(chosen) >= k:
                return chosen
        else:
            leftovers.append(it)

    # If some decision buckets had insufficient availability, fill remaining from leftovers
    need = k - len(chosen)
    if need > 0:
        chosen.extend(leftovers[:need])
    return chosen


def main():
    parser = argparse.ArgumentParser(
        description="Filter OpenReview-style JSONL and sample evenly across score brackets; output raw JSONL lines."
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2024, 2025],
        help="Years to process (e.g., --years 2020 2021 2022)."
    )
    parser.add_argument(
        "--orgs",
        type=str,
        nargs="+",
        default=["ICLR"],
        help="Organizations to process (e.g., --orgs NeurIPS ICLR)."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=300,
        help="Total samples to output per (org, year). If <= 0, output all remaining after filtering."
    )
    args = parser.parse_args()
    random.seed(0)

    # Titles that have already been used in a prior evaluation set (avoid re-sampling)
    used = set()
    with open("pre_eval/evaluation.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            title = line["content"]["title"]
            if isinstance(title, dict):
                title = title["value"]
            used.add(title)

    for org in args.orgs:
        for year in args.years:
            out_path = f"{org}.cc_{year}/{org}.cc_{year}_sample.jsonl"
            in_path = f"{org}.cc_{year}/{org}.cc_{year}_clean.jsonl"

            write = open(out_path, "w", encoding="utf-8")
            raw_lines = read_lines(in_path)

            # 1) Filter and bucket by score bracket
            grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

            filtered_in = 0
            for raw in raw_lines:
                obj = json.loads(raw)
                title = obj["content"]["title"]
                if isinstance(title, dict):
                    title = title["value"]
                if not isinstance(obj, Dict):
                    continue
                if is_withdrawn(obj):
                    continue
                if title in used:
                    continue
                # Require at least 3 reviewer overall scores
                if not has_at_least_n_reviewers(obj, n=3):
                    continue
                # Require sufficient author replies
                reply_ok = reply_all_ok(obj)
                if not reply_ok:
                    continue
                # Require overall scores for bracket assignment
                scores = extract_reviewer_overall_scores(obj, org, year)
                if not scores:
                    continue
                avg = sum(scores) / len(scores)
                bracket = determine_bracket(avg)
                if bracket is None:
                    continue
                # Require an already-verified "correct" PDF
                if not obj.get("PDF_version_correct", False):
                    continue

                grouped[bracket].append(deepcopy(obj))
                filtered_in += 1

            # 2) Determine bracket quotas (evenly across all brackets)
            remaining_total = sum(len(grouped[b]) for b in BRACKETS)
            target_k = args.k if args.k and args.k > 0 else remaining_total
            target_k = min(target_k, remaining_total)

            counts = {b: len(grouped[b]) for b in BRACKETS}
            quota = allocate_quota_evenly(counts, target_k)

            # 3) Select within each bracket while approximating decision distribution
            for b in BRACKETS:
                v = quota.get(b, 0)
                if v <= 0:
                    grouped[b] = []
                    continue
                grouped[b] = _stratified_select_by_decision(grouped[b], v)

            # 4) Write output and collect summary stats
            decision_counts = defaultdict(int)
            bracket_counts = defaultdict(int)

            for k_b, v_list in grouped.items():
                for it in v_list:
                    it["score_bracket"] = k_b
                    decision = str(it.get("decision", "")).lower()
                    decision_counts[decision] += 1
                    bracket_counts[k_b] += 1
                    json.dump(it, write, ensure_ascii=False)
                    write.write("\n")
                    write.flush()
            write.close()

            total_selected = sum(bracket_counts.values())

            # Console summary (kept as prints to match the original style)
            print(f"[{org} {year}] Input: {in_path}")
            print(f"[{org} {year}] Output: {out_path}")
            print(f"[{org} {year}] Filter-passed: {filtered_in} | Selected: {total_selected} (k={args.k})")

            print(f"[{org} {year}] Selected by score bracket:")
            for b in BRACKETS:
                print(f"  {b}: {bracket_counts.get(b, 0)}")

            print(f"[{org} {year}] Selected decision distribution:")
            for decision, count in decision_counts.items():
                print(f"  {decision}: {count}")
            print()


if __name__ == "__main__":
    main()
