#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose
-------
Split peer-review discussion threads into small parts, ask an LLM to assign each part
to an "atomic discussion point" label, then post-process the labels into merged groups
suitable for downstream clustering/visualization.

This script does NOT crawl OpenReview or download PDFs. It operates on JSONL records
that already contain review/rebuttal threads.

High-level workflow
-------------------
1) Read each record from a JSONL dataset (evaluation or sampled venue/year).
2) Convert nested review blocks into a linear list of (role, text) pairs.
3) Chunk each text into smaller parts and build a numbered prompt ("ID: k ...").
4) Call an LLM to assign labels ("Label 1", "Label 2", ..., "Label N/A") to each part.
5) Parse the LLM output and merge highly similar numeric groups via Jaccard/union-find,
   while removing "hub" sentences that appear in many groups to prevent spurious merges.
6) Rebuild output blocks with an author-aware splitting policy and write per-record JSONL:
   - split_texts: grouped text blocks after merging/splitting
   - parsed: the corresponding part-id lists for each block

Output
------
- For --org eval:
    pre_eval/split/evaluation_<model>_<effort>_split.jsonl
- Otherwise:
    <org>.cc_<year>/split/<org>.cc_<year>_sample_<model>_<effort>_split.jsonl

Notes
-----
- Accuracy/quality depends on the LLM and prompt adherence.
- The script warns when label coverage is low (e.g., missing IDs in the LLM output).
- Logging is kept lightweight for open-source readability.
"""

import argparse
import json
import re
import math
import logging
import sys
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple, Dict, Union

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from prompt_registry import split_reviews_system_prompt
from src.utils import BaseArguments, LLMClient, split_text_into_chunks
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# LLM output parsing
# ------------------------------------------------------------------------------

def parse_llm_output(response: str) -> Dict[Union[int, str], List[int]]:
    """
    Parse the LLM's textual response into a mapping: label -> list[part_id].

    Expected format (exactly one line per part, in order):
        Part k: Label X
        Part k: Label X, Y    (multi-label)

    Notes
    -----
    - "Part" and "Label" are treated case-insensitively.
    - Labels may be integers or strings (e.g., "N/A").
    - Any content before a "</think>" token is discarded.
    """
    label2id = defaultdict(list)
    response = response.split("</think>")[-1]

    for line in response.splitlines():
        line = line.strip().replace(".", ":")
        if line.startswith("ID: "):
            line = line[3:].strip()
        if not line.lower().startswith("part "):
            line = "Part " + line
        if not line.lower().startswith("part ") or ":" not in line:
            print(line)
            continue

        left, right = line.split(":", 1)
        try:
            part_id = int(left.split(maxsplit=1)[1])
            labels = right.strip().split(maxsplit=1)[-1].split(",")
            for label in labels:
                label = label.strip()
                try:
                    label = int(label)
                except Exception:
                    pass
                label2id[label].append(part_id)
        except Exception:
            print(left, right)

    return label2id


# ------------------------------------------------------------------------------
# Label-group merging
# ------------------------------------------------------------------------------

def merge_similar_parsed(
    parsed: Dict,
    jaccard_threshold: float = 0.8,
    hub_frac: float = 0.5,  # A sentence in >= 50% of numeric groups is treated as a hub
) -> Dict:
    """
    Merge similar numeric-labeled groups via Jaccard similarity and union-find.

    Design
    ------
    - "Hub" sentence IDs (very frequent across numeric groups) are removed from all groups
      before similarity checks to prevent spurious merges.
      hub_cut = max(5, ceil(hub_frac * #numeric_groups)).
    - Only numeric labels are candidates for merging.
    - Merge criteria (on hub-removed sets):
        (a) subset-like or differ by at most one sentence, OR
        (b) Jaccard(A, B) >= jaccard_threshold
    - Merges are transitive (union-find).

    Note
    ----
    Hub sentences are removed permanently from returned numeric groups. If you need to
    track hubs, do so outside this function (e.g., send them to an "N/A" bucket).
    """

    # 1) Materialize groups and identify numeric-label indices.
    groups = [[lbl, set(ids)] for lbl, ids in parsed.items()]
    idx_numeric = [i for i, (lbl, _) in enumerate(groups) if isinstance(lbl, int)]
    if len(idx_numeric) < 2:
        return {lbl: sorted(list(s)) for lbl, s in parsed.items()}

    # 2) Compute hub sentence IDs from original sets.
    df = Counter()
    for i in idx_numeric:
        for sid in groups[i][1]:
            df[sid] += 1

    hub_cut = max(5, int(math.ceil(hub_frac * len(idx_numeric))))
    hubs = {sid for sid, c in df.items() if c >= hub_cut}

    # 3) Remove hubs from all groups before similarity checks.
    groups = [[lbl, s - hubs] for lbl, s in groups]

    def jaccard_similarity(A: set, B: set) -> float:
        """Jaccard similarity on hub-removed sets."""
        union = A | B
        if not union:
            return 1.0
        inter = A & B
        return len(inter) / len(union)

    def one_sentence_diff(A: set, B: set) -> bool:
        """True if subset-like or differs by at most one element."""
        if A <= B or B <= A:
            return True
        if len(A) <= 1 or len(B) <= 1:
            return False
        if len(A ^ B) <= 2:
            return True
        return False

    # 4) Collect candidate merge pairs.
    pairs = []
    for a in range(len(idx_numeric)):
        i = idx_numeric[a]
        _, Ai = groups[i]
        for b in range(a + 1, len(idx_numeric)):
            j = idx_numeric[b]
            _, Bj = groups[j]
            if one_sentence_diff(Ai, Bj) or jaccard_similarity(Ai, Bj) >= jaccard_threshold:
                pairs.append((i, j))

    if not pairs:
        return {lbl: sorted(list(s)) for lbl, s in groups}

    # 5) Union-find merge (pointer-based).
    pairs.sort(key=lambda p: max(p[0], p[1]), reverse=True)
    for i, j in pairs:
        logger.debug("%s %s", groups[i], groups[j])
    logger.debug("")
    remap_pointers = list(range(len(groups)))

    def find_root(k):
        while k != remap_pointers[k]:
            k = remap_pointers[k]
        return k

    for i, j in pairs:
        root_i = find_root(i)
        root_j = find_root(j)

        if root_i == root_j:
            continue

        small_root, large_root = (root_i, root_j) if root_i < root_j else (root_j, root_i)

        groups[small_root][1].update(groups[large_root][1])
        if isinstance(groups[small_root][0], int) and isinstance(groups[large_root][0], int):
            groups[small_root][0] = min(groups[small_root][0], groups[large_root][0])

        remap_pointers[large_root] = small_root
        groups[large_root][1] = set()
        groups[large_root][0] = groups[small_root][0]

    # 6) Build final mapping; coalesce duplicate labels.
    final_output = {}
    for label, item_set in groups:
        if item_set or not isinstance(label, int):
            if label in final_output:
                final_output[label].update(item_set)
            else:
                final_output[label] = item_set

    return {lbl: sorted(list(s)) for lbl, s in final_output.items()}


# ------------------------------------------------------------------------------
# Prompt construction
# ------------------------------------------------------------------------------

def build_prompt(
    all_texts_pairs,
    start: int = 0,
    min_words: int = 4,
    max_words: int = 150,
    strict_commas: bool = False
):
    """
    Build an LLM prompt from a sequence of (role, text) pairs by chunking text
    and assigning incremental part IDs.

    Returns
    -------
    to_text: dict[int, str]
        Part ID -> chunk text.
    to_role: dict[int, str]
        Part ID -> speaker role.
    prompt_text: str
        The full prompt with "ID: k" prefixes.
    cnt_idx: int
        The last used part ID.
    """
    to_text = {}
    to_role = {}
    prompt_text = ""
    cnt_idx = start

    for role, text in all_texts_pairs:
        if role is None:
            role = "Public Comment"

        prompt_text += role + "\n\n"
        chunks = split_text_into_chunks(
            text,
            min_words=min_words,
            max_words=max_words,
            strict_commas=strict_commas
        )

        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue

            cnt_idx += 1
            to_text[cnt_idx] = chunk
            to_role[cnt_idx] = role

            if "\n" in chunk:
                prompt_text += f"ID: {cnt_idx}.\n{chunk}\n"
            else:
                prompt_text += f"ID: {cnt_idx}. {chunk}\n"

        prompt_text += "\n\n\n"

    return to_text, to_role, prompt_text, cnt_idx


# ------------------------------------------------------------------------------
# Thread flattening
# ------------------------------------------------------------------------------

def reviews_to_text_pairs(replies):
    """
    Convert raw 'reviews' blocks into a flat iterable of [role, text] pairs.

    Reviewer roles are expanded into "Reviewer i" (or "Reviewer i Further Reply") to
    help the model separate reviewer threads.
    """
    for reviewer_id, reply in enumerate(replies):
        for role, content in reply:
            if role == "Reviewer" and "scores" in content:
                role = f"Reviewer {reviewer_id + 1}"
            elif role == "Reviewer":
                role = f"Reviewer {reviewer_id + 1} Further Reply"
            assert isinstance(content["value"], dict)
            content_text = "\n\n".join([f"{k}:\n{v}" for k, v in content["value"].items()])
            yield [role, content_text]


# ------------------------------------------------------------------------------
# Main classification loop
# ------------------------------------------------------------------------------

def classify_all(args) -> List[List[str]]:
    """
    Run labeling for every record in the selected JSONL dataset.

    Output format (per line)
    ------------------------
    {
      "id": <record_id>,
      "content": <original content>,
      "split_texts": <merged text blocks>,
      "parsed": <list of part-id lists aligned to split_texts>
    }

    Returns
    -------
    List[List[str]]
        Not used by the current implementation (kept for compatibility with earlier versions).
    """
    if args.org == "eval":
        jsonl_path = f"pre_eval/evaluation.jsonl"
        output_path = f"pre_eval/split/evaluation_{args.model.replace('/', '_')}_{args.effort}_split.jsonl"
    else:
        jsonl_path = f"{args.org}.cc_{args.year}/{args.org}.cc_{args.year}_sample.jsonl"
        output_path = (
            f"{args.org}.cc_{args.year}/split/"
            f"{args.org}.cc_{args.year}_sample_{args.model.replace('/', '_')}_{args.effort}_split.jsonl"
        )

    w = open(output_path, "w", encoding="utf-8")
    all_keys = []
    all_details = []
    all_to_text = {}
    all_to_role = {}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for file_id, line in enumerate(f):
            data = json.loads(line)
            replies = data.get("reviews", [])
            all_texts_pairs: List[Tuple[str, str]] = list(reviews_to_text_pairs(replies))

            to_text, to_role, prompt_text, _ = build_prompt(all_texts_pairs)
            key = client.submit_task(
                prompt_text,
                system_prompt=split_reviews_system_prompt(len(to_text)),
                temperature=0.6,
                max_completion_tokens=16384,
                reasoning_effort=args.effort
            )

            all_keys.append(key)
            all_details.append((data["id"], data["content"]))
            all_to_text[key] = deepcopy(to_text)
            all_to_role[key] = deepcopy(to_role)

    print(f"[INFO] Submitted {len(all_keys)} LLM request(s).")
    print(f"[INFO] Writing outputs to: {output_path}")

    # Fetch results and post-process
    for key, (rec_id, content) in zip(all_keys, all_details):
        result = client.get_result(key)
        response = result["content"]
        if response is None:
            continue

        parsed = parse_llm_output(response)

        # Coverage check: how many part IDs are assigned to any label?
        total_parts = len(all_to_text[key])
        labeled_ids = set()
        for ids in parsed.values():
            for pid in ids:
                if isinstance(pid, int):
                    labeled_ids.add(pid)

        coverage = (len(labeled_ids) / total_parts) if total_parts else 1.0
        if coverage < 0.95:
            missing = sorted(set(all_to_text[key].keys()) - labeled_ids)
            print(
                f"[WARN] Low label coverage: rec_id={rec_id} "
                f"{len(labeled_ids)}/{total_parts}={coverage:.1%}; "
                f"missing IDs (first 20): {missing[:20]}"
            )

        # Remove any existing 'N/A' label before merging numeric groups, then re-add later.
        if 'N/A' in parsed:
            del parsed['N/A']

        parsed = merge_similar_parsed(parsed, jaccard_threshold=0.8)

        # Add any uncovered IDs into N/A.
        na = set(all_to_text[key].keys())
        for _, v in parsed.items():
            na -= set(v)
        parsed['N/A'] = sorted(na)

        def _lbl_sort_key(k):
            return (0, k) if isinstance(k, int) else (1, str(k))

        ordered_items = sorted(parsed.items(), key=lambda kv: _lbl_sort_key(kv[0]))

        def _is_author(role: str) -> bool:
            return role.strip().lower().startswith("author")

        _author_reviewer_id = re.compile(r"^reviewer\s+(\d+)", re.IGNORECASE)

        def _rev_id(role: str):
            m = _author_reviewer_id.match(role.strip())
            return int(m.group(1)) if m else None

        # --- Rebuild split_texts based on merged groups (author-aware splitting) ---
        new_split_texts = []
        new_parsed = []

        for label, id_list in ordered_items:
            if label == "N/A":
                # Preserve role boundaries within N/A: merge adjacent chunks with the same role.
                block_roles = [all_to_role[key][it] for it in id_list]
                block_texts = [all_to_text[key][it] for it in id_list]
                merged_block = []
                last_role = None
                acc_text = []

                for r, t in zip(block_roles, block_texts):
                    if last_role is None:
                        last_role = r
                        acc_text = [t]
                    elif r == last_role:
                        acc_text.append(t)
                    else:
                        merged_block.append([last_role, "\n".join(acc_text)])
                        last_role = r
                        acc_text = [t]
                if last_role is not None:
                    merged_block.append([last_role, "\n".join(acc_text)])

                new_split_texts.append(deepcopy(merged_block))
                new_parsed.append(deepcopy(sorted(id_list)))
                continue

            seq_ids = sorted([it for it in id_list if it in all_to_text[key].keys()])
            if not len(seq_ids):
                continue
            roles = [all_to_role[key][it] for it in seq_ids]
            texts = [all_to_text[key][it] for it in seq_ids]

            # Candidate cut points when reviewer ID switches inside the same numeric label group
            candidate_by_rev_switch = [False] * len(seq_ids)
            prev_rev = _rev_id(roles[0])
            for i in range(1, len(seq_ids)):
                cur_rev = _rev_id(roles[i])
                if prev_rev is not None and cur_rev is not None and cur_rev != prev_rev:
                    candidate_by_rev_switch[i] = True
                prev_rev = cur_rev

            has_author = any(_is_author(r) for r in roles)

            cut_points = []
            blocked_due_to_author = False

            for i in range(1, len(seq_ids)):
                if not has_author:
                    if candidate_by_rev_switch[i]:
                        cut_points.append(i)
                    continue

                # With author text present: only allow cuts right after an Author segment,
                # and the next segment must not be Author.
                is_author_boundary = _is_author(roles[i - 1]) and not _is_author(roles[i])

                if candidate_by_rev_switch[i]:
                    if is_author_boundary:
                        assert not _is_author(roles[i]), f"Invalid split at {i}: next role is Author"
                        cut_points.append(i)
                        blocked_due_to_author = False
                    else:
                        blocked_due_to_author = True
                else:
                    if blocked_due_to_author and is_author_boundary:
                        assert not _is_author(roles[i]), f"Invalid split at {i}: next role is Author"
                        cut_points.append(i)
                        blocked_due_to_author = False

            cut_points = sorted(set(cp for cp in cut_points if 0 < cp < len(seq_ids)))
            slices = []
            start = 0
            for cp in cut_points:
                if cp > start:
                    slices.append((start, cp))
                    start = cp
            slices.append((start, len(seq_ids)))
            if len(slices) > 1:
                logger.debug("\n".join(([" ".join(roles[slice[0]: slice[1]]) for slice in slices])))

            # Merge adjacent items with the same role within each slice.
            for (s, e) in slices:
                block_roles = roles[s:e]
                block_texts = texts[s:e]
                block_ids = seq_ids[s:e]

                merged_block = []
                last_role = None
                acc_text = []
                acc_ids = []

                for r, t, iid in zip(block_roles, block_texts, block_ids):
                    if last_role is None:
                        last_role = r
                        acc_text = [t]
                        acc_ids = [iid]
                    elif r == last_role:
                        acc_text.append(t)
                        acc_ids.append(iid)
                    else:
                        merged_block.append([last_role, "\n".join(acc_text)])
                        last_role = r
                        acc_text = [t]
                        acc_ids = [iid]

                if last_role is not None:
                    merged_block.append([last_role, "\n".join(acc_text)])

                new_split_texts.append(deepcopy(merged_block))
                new_parsed.append(deepcopy(block_ids))
        # --- end author-aware rebuilding ---

        # Keep non-N/A blocks ordered by first ID; keep N/A as the last block.
        pairs = list(zip(new_split_texts, new_parsed))
        if pairs:
            head = pairs[:-1]
            tail = pairs[-1:]
            head.sort(key=lambda p: (p[1][0] if p[1] else float('inf')))
            pairs = head + tail
            new_split_texts, new_parsed = map(list, zip(*pairs))

        json.dump(
            {
                "id": rec_id,
                "content": content,
                "split_texts": new_split_texts,
                "parsed": new_parsed
            },
            w,
            ensure_ascii=False
        )
        w.write('\n')
        w.flush()
    w.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Label peer-review discussion parts into atomic discussion points."
    )
    BaseArguments.add_to_parser(parser, model_default="gpt-5", effort_default="medium")
    args = parser.parse_args()
    BaseArguments.apply(args)
    client = LLMClient(
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model,
        max_workers=args.max_workers,
        cache_version=args.cache_version
    )

    classify_all(args)
    client.shutdown()
