#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split free-form AI review text into opinion groups for downstream classification.

Input JSONL
-----------
- Records produced by `src/evaluation/_1_query_ai_reviewer.py`, typically containing:
  - `id`
  - `text`

Processing
----------
1) Read each review `text`.
2) Chunk it into shorter passages and build a numbered prompt.
3) Ask the split model to group chunk ids into atomic opinions.
4) Sanitize, split, and merge the predicted groups into stable 0-based opinion indices.

Output JSONL
------------
- Same records with:
  - `sentence_texts`: list[str] of chunked review text
  - `opinions`: list[list[int]] indexing groups inside `sentence_texts`

Notes
-----
- When `--base_url` points to localhost, the script waits for the vLLM server before
  submitting requests.
- This script only operates on already-generated AI review text.
"""

import argparse
import json
from copy import deepcopy
import math
import logging
from urllib.parse import urlparse
from tqdm import tqdm
import sys
from pathlib import Path
import os
from typing import List, Tuple, Dict, Union, Iterable, Any, Optional
from collections import defaultdict, Counter

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, PROJECT_ROOT)

from src.utils import (
    LLMClient,
    split_text_into_chunks,
    _is_markdown_item_start,
    dedup_by_word_sequence,
    wait_until_ready,
)
from src.benchmark.prompt_registry import split_ai_reviewer_prompt

logger = logging.getLogger(__name__)


# For parsing model outputs
def parse_llm_output(response: str) -> Dict[Union[int, str], List[int]]:
    """
    Parse the LLM's textual output into a mapping from label -> list of part IDs.

    The function expects the model to produce exactly N lines like either of:
        "Part k: Label X"
        "Part k: Label X, Y"  (multiple labels)

    "Part" and "Label" are case-insensitive, and labels may be integers or strings
    (e.g., "N/A").
    """
    label2id = defaultdict(list)
    response = response.split("</think>")[-1]

    for line in response.splitlines():
        line = line.strip().replace(".", ":")
        if line.startswith("ID: "):
            line = line[3: ].strip()
        if not line.lower().startswith("part "):
            line = "Part " + line
        if not line.lower().startswith("part ") or ":" not in line:
            continue
        # Expected example: "Part 3: Label 1, 5"
        left, right = line.split(":", 1)
        try:
            part_id = int(left.split(maxsplit=1)[1])
            labels = right.strip().split(maxsplit=1)[-1].split(",")
            for label in labels:
                label = label.strip()
                try:
                    label = int(label)
                except:
                    pass
                label2id[label].append(part_id)
        except:
            pass

    return label2id


def merge_similar_parsed(
    parsed: Dict,
    jaccard_threshold: float = 0.8,
    hub_frac: float = 0.5,  # A sentence appearing in >= 50% of numeric groups is treated as a hub
) -> Dict:
    """
    Merge similar numeric-labeled groups based on Jaccard similarity and a union-find scheme.

    Behavior:
      • Identify "hub" sentence IDs that occur in many numeric groups and permanently remove
        them from all groups before similarity checks (to avoid spurious merges driven by
        very common sentences). Hubs are defined by count >= max(5, ceil(hub_frac * #numeric_groups)).
      • Only consider merges between groups with *numeric* labels.
      • Merge if (a) the sets differ by at most one sentence (subset or near-substitution), or
        (b) their Jaccard similarity >= jaccard_threshold.
      • Merges are transitive.
      • All items are unweighted; no TF–IDF weighting is used.

    Note:
      The returned groups do not include the removed hub sentences. If you want to keep track
      of those, handle them outside this function (e.g., assign them to an "N/A" group).
    """

    # 1) Build working list of [label, set(sentence_ids)] and track numeric-label indices.
    groups = [[lbl, set(ids)] for lbl, ids in parsed.items()]
    idx_numeric = [i for i, (lbl, s) in enumerate(groups) if isinstance(lbl, int)]
    if len(idx_numeric) < 2:
        return {lbl: sorted(list(s)) for lbl, s in parsed.items()}

    # 2) Identify hub sentence IDs from the original (pre-filter) sets.
    df = Counter()
    for i in idx_numeric:
        for sid in groups[i][1]:
            df[sid] += 1
    
    hub_cut = max(5, int(math.ceil(hub_frac * len(idx_numeric))))
    hubs = {sid for sid, c in df.items() if c >= hub_cut}

    # 3) Permanently remove hubs from *all* groups before computing similarity.
    groups = [[lbl, s - hubs] for lbl, s in groups]

    # 4) Similarity/compatibility helpers (operate on hub-removed sets).
    def jaccard_similarity(A: set, B: set) -> float:
        """Standard Jaccard similarity on the pre-filtered sets."""
        union = A | B
        if not union:
            return 1.0  # Both sets became empty after hub removal
        inter = A & B
        return len(inter) / len(union)

    def one_sentence_diff(A: set, B: set) -> bool:
        """Return True if A and B are subset-like or differ by at most one element."""
        if A <= B or B <= A:
            return True
        if len(A) <= 1 or len(B) <= 1:
            return False
        if len(A ^ B) <= 2:
            return True
        return False

    # 5) Collect candidate pairs that meet either merge condition.
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
        # No merges: return hub-removed groups as-is.
        return {lbl: sorted(list(s)) for lbl, s in groups}

    # 6) Merge using a simple pointer-based union-find; handle transitivity.
    pairs.sort(key=lambda p: max(p[0], p[1]), reverse=True)
    for i, j in pairs:
        logger.debug("%s %s", groups[i], groups[j])
    logger.debug("")
    remap_pointers = list(range(len(groups)))

    def find_root(k):
        # Follow pointers to the representative index of k
        while k != remap_pointers[k]:
            k = remap_pointers[k]
        return k

    for i, j in pairs:
        root_i = find_root(i)
        root_j = find_root(j)

        if root_i == root_j:
            continue

        # Always merge the larger index into the smaller one for stability
        small_root, large_root = (root_i, root_j) if root_i < root_j else (root_j, root_i)
        
        # Union the content and keep the smaller numeric label if both are numeric
        groups[small_root][1].update(groups[large_root][1])
        if isinstance(groups[small_root][0], int) and isinstance(groups[large_root][0], int):
            groups[small_root][0] = min(groups[small_root][0], groups[large_root][0])
        
        # Redirect pointer and clear the old container
        remap_pointers[large_root] = small_root
        groups[large_root][1] = set()
        groups[large_root][0] = groups[small_root][0]  # keep label consistent

    # 8) Build final mapping from merged working list. Duplicate labels will be coalesced.
    final_output = {}
    for label, item_set in groups:
        # Keep non-empty sets, or any non-numeric labels.
        if item_set or not isinstance(label, int):
            if label in final_output:
                final_output[label].update(item_set)
            else:
                final_output[label] = item_set
    
    return {lbl: sorted(list(s)) for lbl, s in final_output.items()}


def split_groups_on_markdown_items(
    parsed: Dict[Union[int, str], List[int]],
    to_text_for_key,
) -> Dict[Union[int, str], List[int]]:
    """
    Rewrite `parsed` once to force split points on markdown list-item boundaries.

    For each label's ID list, when we encounter a sentence whose text is recognized
    as the start of a markdown item (via `_is_markdown_item_start`), we *force* a
    group cut at that position, ensuring the markdown item does not connect the
    text "before" and "after" into the same part.

    Procedure:
      - Sort each label's `id_list`.
      - Sweep IDs in increasing order:
          * If the current ID's text is a markdown item start:
                - If the current segment is non-empty, close it first
                - Put this ID alone as a segment (so it won't merge with the previous one)
          * Otherwise, append to the current segment.
      - If one label is split into multiple segments:
          * The first segment keeps the original label
          * Later segments get derived new labels:
            - If the original label is `int`, allocate new `int` labels starting from (max_int_label + 1)
            - If the original label is `str`, append suffixes like "SUM_1"
    """
    # Helper: support both list and dict `to_text` containers.
    def get_text(pid: int) -> str:
        if isinstance(to_text_for_key, dict):
            return to_text_for_key.get(pid, "") or ""
        # Assume list is indexed by pid-1
        if isinstance(to_text_for_key, list):
            idx = pid - 1
            if 0 <= idx < len(to_text_for_key):
                return to_text_for_key[idx] or ""
        return ""

    # Find the current max numeric label to allocate new numeric labels.
    numeric_labels = [lbl for lbl in parsed.keys() if isinstance(lbl, int)]
    next_numeric = (max(numeric_labels) if numeric_labels else 0) + 1

    # For string labels, track derivation counts.
    str_label_cnt = defaultdict(int)

    new_parsed: Dict[Union[int, str], List[int]] = {}

    for label, id_list in parsed.items():
        if not id_list:
            continue

        ids = sorted(set(id_list))  # De-duplicate and keep stable order.

        # Collect segments after splitting.
        segments: List[List[int]] = []
        cur: List[int] = []

        for pid in ids:
            txt = get_text(pid).strip()
            if _is_markdown_item_start(txt, strict=True):
                # Close current segment first.
                if cur:
                    segments.append(cur)
                    cur = []
                # This markdown item becomes its own segment (so the two sides must be different parts).
                segments.append([pid])
            else:
                cur.append(pid)

        if cur:
            segments.append(cur)

        # If there's only one segment, keep label -> ids as-is.
        if len(segments) == 1:
            new_parsed[label] = segments[0]
            continue

        # Multiple segments: first keeps original label; remaining segments get derived new labels.
        new_parsed[label] = segments[0]
        for seg in segments[1:]:
            if not seg:
                continue
            if isinstance(label, int):
                new_label = next_numeric
                next_numeric += 1
            else:
                str_label_cnt[label] += 1
                new_label = f"{label}_{str_label_cnt[label]}"
            new_parsed[new_label] = seg

    return new_parsed


def build_prompt(
    text,
    start: int = 0,
    min_words: int = 4,
    max_words: int = 150,
    strict_commas: bool = False
):
    """
    Build a prompt from text chunks and assign incremental part IDs.

    Args:
      text: either a raw string or a pre-split list of chunks
      start: starting ID (useful for appending)
      min_words, max_words, strict_commas: passed through to `split_text_into_chunks`

    Returns:
      to_text: list[str], chunk texts in ID order (1-based in the prompt)
      prompt_text: concatenated prompt text with "Part k:" prefixes
      cnt_idx: the last ID used
    """
    to_text = []
    prompt_text = ""
    cnt_idx = start

    if isinstance(text, list):
        chunks = text
    else:
        chunks_old = split_text_into_chunks(
            text,
            min_words=min_words,
            max_words=max_words,
            strict_commas=strict_commas
        )
        chunks = dedup_by_word_sequence(chunks_old)
        if len(chunks) != len(chunks_old):
            pass

    # Assign IDs chunk-by-chunk and build the prompt text.
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        to_text.append(chunk)
        cnt_idx += 1
        if "\n" in chunk:
            prompt_text += f"Part {cnt_idx}:\n{chunk}\n"
        else:
            prompt_text += f"Part {cnt_idx}: {chunk}\n"
        
    return to_text, prompt_text, cnt_idx


def normalize_groups(
    parsed_values: Iterable[Any],
    missing: Optional[Iterable[Any]] = None,
    offset: int = -1,          # Original logic used i-1; default offset=-1
) -> List[List[int]]:
    missing = missing or []

    group_set = set()

    def _as_tuple_group(xs, off=0) -> Tuple[int, ...]:
        if xs is None:
            return tuple()
        # Support int / iterable[int]
        if isinstance(xs, int):
            items = [xs]
        else:
            items = list(xs)
        # Within-group: de-duplicate + apply offset + sort.
        g = tuple(sorted({int(x) + off for x in items}))
        return g

    # parsed.values(): apply offset (historically i-1) to each list
    for lst in parsed_values:
        g = _as_tuple_group(lst, off=offset)
        if g:  # Skip empty groups
            group_set.add(g)

    # missing: typically already 0-based (if you also want -1, change off=offset)
    g = _as_tuple_group(missing, off=offset)
    if g:
        group_set.add(g)

    # Inter-group sorting: stable and reproducible
    groups_sorted = sorted(group_set, key=lambda g: g)

    # Output list[list[int]] (keep tuple if you prefer)
    return [list(g) for g in groups_sorted]


def sanitize_parsed_ids(
    parsed: Dict[Union[int, str], List[int]],
    total_parts: int,
) -> Tuple[Dict[Union[int, str], List[int]], List[int]]:
    """
    Keep only part_ids in 1..total_parts; return (cleaned, extras).

    extras: part_ids out of range (de-duplicated and sorted), useful for penalties/logging.
    """
    cleaned = defaultdict(list)
    extras = set()

    for lbl, ids in parsed.items():
        if ids is None:
            continue
        # In some cases ids may not be a list.
        if isinstance(ids, int):
            ids = [ids]

        try:
            it = list(ids)
        except TypeError:
            continue

        for pid in it:
            if not isinstance(pid, int):
                continue
            if 1 <= pid <= total_parts:
                cleaned[lbl].append(pid)
            else:
                extras.add(pid)

    # De-duplicate/sort within groups; drop empty groups to avoid downstream pollution.
    cleaned = {lbl: sorted(set(v)) for lbl, v in cleaned.items() if v}
    return cleaned, sorted(extras)


def compute_coverage(parsed, total_parts: int, extra_penalty: float = 1.0):
    """
    Return (score, missing, extras).

    score = coverage - penalty
      coverage = (# covered in-range parts) / total_parts
      penalty  = extra_penalty * (# out-of-range parts) / total_parts

    extra_penalty=1.0 means: one invalid ID penalizes as much as missing one valid ID.
    """
    in_range = set()
    extras = set()

    for ids in parsed.values():
        for pid in ids:
            if not isinstance(pid, int):
                continue
            if 1 <= pid <= total_parts:
                in_range.add(pid)
            else:
                extras.add(pid)

    base = (len(in_range) / total_parts) if total_parts else 1.0
    penalty = (extra_penalty * len(extras) / total_parts) if total_parts else 0.0
    score = max(0.0, min(1.0, base - penalty))

    missing = sorted(set(range(1, total_parts + 1)) - in_range)
    return score, missing


def split_all(args) -> List[List[str]]:
    jsonl_path = args.input_jsonl
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(
            f"Input file not found: {jsonl_path}. "
        )
    output_path = args.output_jsonl
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # ---- 1) Batch submit (first pass)
    jobs = []  # One work item per record
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in tqdm(enumerate(f)):
            data = json.loads(line)
            to_text, prompt_text, _ = build_prompt(data["text"])
            keys = []
            for p in [0.0, 0.5, 1.0]:
                keys.append(client.submit_task(
                    prompt_text,
                    system_prompt=split_ai_reviewer_prompt(len(to_text)).strip(),
                    temperature=0.6,
                    max_completion_tokens=32768,
                    reasoning_effort=args.effort,
                    presence_penalty=p
                ))
            jobs.append({
                "data": deepcopy(data),
                "to_text": deepcopy(to_text),   # list[str]
                "prompt_text": prompt_text,
                "keys": keys,
            })

    print("Total Requests:", len(jobs * 4))

    # ---- 2) Batch get_result
    with open(output_path, "w", encoding="utf-8") as w:
        for job in tqdm(jobs):
            to_text = job["to_text"]
            results = []
            failed = []
            for key in job["keys"]:
                result = client.get_result(key)
                response = result["content"]
                parsed = parse_llm_output(response)
                cov, missing = compute_coverage(parsed, len(to_text))
                parsed, _ = sanitize_parsed_ids(parsed, len(to_text))
                if cov == 1.0:
                    parsed = split_groups_on_markdown_items(parsed, to_text)
                    parsed = merge_similar_parsed(parsed, jaccard_threshold=0.8)
                    results.append(deepcopy(parsed))
                else:
                    parsed["##N/A"] = missing
                    parsed = split_groups_on_markdown_items(parsed, to_text)
                    parsed = merge_similar_parsed(parsed, jaccard_threshold=0.8)
                    failed.append((cov, deepcopy(parsed)))
            if not len(results):
                failed.sort(key=lambda x: x[0])
                results = [failed[-1][-1]]
            results.sort(key=len)
            data = job["data"]
            data["sentence_texts"] = to_text
            data["opinions"] = normalize_groups(results[-1].values(), offset=-1)
            data['opinions'] = sorted([sorted(it) for it in data["opinions"]])
            w.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Split AI Reviews into parts."
    )
    parser.add_argument(
        "--input_jsonl",
        default="review_result/Qwen3-8B-result/result_sample.jsonl",
        help="Path to input JSONL (records with `review` or `sentence_texts` + `reviews`).",
    )
    parser.add_argument(
        "--output_jsonl",
        default="review_result/Qwen3-8B-result/result_sample_split.jsonl",
        help="Path to output JSONL (will be created/overwritten).",
    )
    parser.add_argument(
        "--model",
        default="Qwen3-8B-Split",
        help="LLM model name",
    )
    parser.add_argument(
        "--effort",
        default="medium",
        help="The reasoning effort parameter for the LLM.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=64,
        help="The reasoning effort parameter for the LLM.",
    )
    parser.add_argument(
        "--cache_version",
        default="v1",
        help="Cache version for LLM responses"
    )
    parser.add_argument(
        "--base_url",
        default="http://localhost:13099/v1",
        help="LLM API base URL",
    )
    args = parser.parse_args()

    _host = (urlparse(args.base_url).hostname or "").lower()
    args._is_vllm = _host in {"localhost", "127.0.0.1", "::1"}
    if args._is_vllm:
        print("Waiting for vLLM server to be ready...")
        wait_until_ready(args.base_url, "dummy", args.model, timeout=3600, interval=20.0)
        print("vLLM server is ready")
        client = LLMClient(
            api_key="dummy",
            base_url=args.base_url, 
            max_workers=args.max_workers,
            model_name=args.model,
            cache_version=args.cache_version
        )
    else:
        raise NotImplementedError

    split_all(args)
    client.shutdown()
