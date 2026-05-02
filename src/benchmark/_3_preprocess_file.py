#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose
-------
Build a sentence-level benchmark JSONL from preprocessed OpenReview-style paper objects.

This script:
1) Splits reviewer/author text fields into sentence-like chunks.
2) Rewrites `reviews[*][*][1]["value"][field]` from raw strings to lists of sentence IDs.
3) Rewrites `split_texts` segments from raw strings to the best-matching reviewer sentence IDs.
4) Optionally merges short "marker" fragments (e.g., headings/bullets) into adjacent sentences.
5) Produces a final JSON object per paper with:
   - reviews: reviewer entries with sentence-id references
   - sentence_texts: canonical sentence strings (index == sentence_id)
   - split_texts: discussion threads with sentence-id references

Inputs
------
- {conf}.cc_{year}/{conf}.cc_{year}_sample.jsonl
  Contains the original paper objects (decision/reviews/metareview/etc.).
- {conf}.cc_{year}/classify/.../{conf}.cc_{year}_gpt-5-mini_medium_classify.jsonl
  Contains classification results plus paper IDs.

Outputs
-------
- {conf}.cc_{year}/{conf}.cc_{year}_benchmark.jsonl
  One JSON object per paper with sentence-level structure.

Notes
-----
- Logging is controlled by the module-level logger and DEBUG flag below.
- This script assumes `utils.py` provides:
  - english_word_count
  - merge_similar_reviewer_threads
  - _is_markdown_item_start
  - split_text_into_chunks
  - GROUPS
"""

import json
import logging
import re
import sys
import unicodedata
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from tqdm import tqdm
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import (
    GROUPS,
    _is_markdown_item_start,
    english_word_count,
    merge_similar_reviewer_threads,
    split_text_into_chunks,
)

logger = logging.getLogger(__name__)
DEBUG = False  # Set True to print diagnostics about imperfect text alignment.


def process_paper(paper, min_words=0, strict_commas=True):
    """
    Transform a single paper object into a sentence-indexed representation.

    Input:
      - `paper`: JSON dict with keys including:
        - reviews: nested list of [role, payload] entries
        - split_texts: discussion threads as [role, text] pairs
        - classify: per-thread labels (removed in output)

    Output (`paper_out`):
      - reviews[*][*][1]["value"][key]: string -> [sentence_ids]
      - split_texts: long strings -> best-matched reviewer sentence id lists
      - sentence_texts: list[str] containing only canonical (reviewer/author) sentences
      - classify: removed
    """
    reviews = paper["reviews"]
    split_texts = paper["split_texts"]

    # -------------------------------------------------------------------------
    # 1) Split all review text fields into sentence-like chunks and index them.
    # -------------------------------------------------------------------------
    sentence_texts = []   # canonical_id -> sentence string
    sentence_roles = []   # canonical_id -> role ("Reviewer", "Author", ...)
    new_reviews = []

    # group_ids[group_idx] = all sentence ids that originated from reviews[group_idx]
    group_ids = defaultdict(list)

    for g_idx, group in enumerate(reviews):
        new_group = []
        for _, entry in enumerate(group):
            role, payload = entry
            v = payload["value"]

            new_value = {}
            for fname, text in v.items():
                # Split text into chunks; register each chunk into the global sentence table.
                chunks = split_text_into_chunks(text, min_words=0, strict_commas=False)
                canonical_ids = []
                for seg in chunks:
                    new_id = len(sentence_texts)
                    sentence_texts.append(seg)
                    sentence_roles.append(role)
                    canonical_ids.append(new_id)

                # Replace raw strings with canonical sentence IDs.
                group_ids[g_idx] += canonical_ids
                new_value[fname] = canonical_ids

            new_payload = dict(payload)
            new_payload["value"] = new_value
            new_group.append([role, new_payload])

        if new_group:
            new_reviews.append(new_group)

    # -------------------------------------------------------------------------
    # 2) Text normalization helpers for alignment between split_texts and reviews.
    # -------------------------------------------------------------------------
    def normalize_text(s: str) -> str:
        # NFKC normalization + normalize smart quotes.
        s = unicodedata.normalize("NFKC", s)
        s = s.replace("“", '"').replace("”", '"').replace("’", "'")
        return s

    def normalize_for_match(s: str) -> str:
        s = normalize_text(s or "")
        # Remove punctuation (keep alnum/underscore/space), then collapse whitespace.
        s = re.sub(r"[^\w\s]", " ", s)
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    def match_text_to_ids(content, role):
        """
        Given a text fragment from split_texts, find the best-matching set of
        reviewer sentence IDs by greedily marking covered character spans.

        Returns:
          - matched_ids: sorted unique sentence IDs
          - leftover_str: uncovered remainder after matching (normalized)
        """
        content_norm = normalize_for_match(content.strip())
        best_leftover = None
        best_leftover_str = content_norm
        best_ids = []

        # Try each review group; pick the one leaving the least uncovered residue.
        for _, cand_ids in group_ids.items():
            if not cand_ids:
                continue

            # Precompute normalized sentences for this group.
            seg_norms = {}
            for sid in cand_ids:
                seg = normalize_for_match(sentence_texts[sid])
                if seg:
                    seg_norms[sid] = seg
            if not seg_norms:
                continue

            n = len(content_norm)
            covered = [False] * n
            matched_ids = []

            for sid, seg in seg_norms.items():
                if sentence_roles[sid] not in role:
                    continue
                pos = content_norm.find(seg)
                if pos == -1:
                    continue

                end = pos + len(seg)
                for p in range(pos, min(end, n)):
                    covered[p] = True
                matched_ids.append(sid)

            # Score by uncovered non-space characters.
            leftover = 0
            for idx, ch in enumerate(content_norm):
                if ch != " " and not covered[idx]:
                    leftover += 1

            if best_leftover is None or leftover < best_leftover:
                best_leftover = leftover
                best_leftover_str = "".join(
                    content_norm[i] for i in range(n) if not covered[i]
                )
                best_ids = matched_ids

        return sorted(set(best_ids)), best_leftover_str

    # -------------------------------------------------------------------------
    # 3) Rewrite split_texts: keep original thread structure, replace strings with IDs.
    # -------------------------------------------------------------------------
    new_split_texts = []
    for group in split_texts:
        new_group = []
        for role, content in group:
            leftover_str = content
            best_ids = []
            # Two passes: match, then try to match again against leftover.
            for _ in range(2):
                cnt_ids, leftover_str = match_text_to_ids(leftover_str, role)
                best_ids += list(cnt_ids)

            if DEBUG and len(leftover_str) > 75:
                logger.debug("Large unmatched remainder (>75 chars): %s", leftover_str)

            new_group.append([role, sorted(set(best_ids))])
        new_split_texts.append(new_group)

    # -------------------------------------------------------------------------
    # 4) If some reviewer sentences were never referenced by split_texts, attach them
    #    to the nearest existing thread of the same role (best-effort coverage).
    # -------------------------------------------------------------------------
    used_ids = set()
    for group in new_split_texts:
        for _, sid_list in group:
            used_ids.update(sid_list)

    all_ids = set(range(len(sentence_texts)))
    unused_ids = all_ids - used_ids

    if len(unused_ids):
        # Track where each role already appears in split_texts.
        role_pos = defaultdict(list)  # role -> list[(sid, group_idx, thread_idx)]
        for g_idx, group in enumerate(new_split_texts):
            for t_idx, (role, sid_list) in enumerate(group):
                for sid in sid_list:
                    if "Reviewer" in role:
                        role_pos["Reviewer"].append((sid, g_idx, t_idx))
                    elif "Author" in role:
                        role_pos["Author"].append((sid, g_idx, t_idx))

        # Insert each unused sentence into the closest same-role thread by ID distance.
        for uid in sorted(unused_ids):
            role = sentence_roles[uid]
            if not role_pos.get(role):
                continue

            candidates = role_pos[role]
            _, best_g, best_t = min(candidates, key=lambda x: abs(x[0] - uid))

            new_split_texts[best_g][best_t][1].append(uid)
            new_split_texts[best_g][best_t][1] = sorted(new_split_texts[best_g][best_t][1])
            role_pos[role].append((uid, best_g, best_t))

        unused_ids = set()

    # -------------------------------------------------------------------------
    # 5) Reformat split_texts into per-reviewer threads and merge similar threads.
    # -------------------------------------------------------------------------
    final_result = []
    for split_text, classify in zip(new_split_texts[:-1], paper["classify"]):
        cnt_text = []
        for role, text in split_text:
            # If a new Reviewer segment starts and we already have content, flush a thread.
            if "Reviewer" in role and cnt_text:
                final_result.append([deepcopy(cnt_text), classify])
                cnt_text = []
            cnt_text.append([role, text])

        if cnt_text:
            # Drop empty/unknown-only fragments.
            if cnt_text[0][0] == "Unknown":
                cnt_text = []
                continue

            # Keep at most two roles; remove Unknown segments where possible.
            if len(cnt_text) > 2:
                if cnt_text[2][0] == "Unknown":
                    cnt_text = cnt_text[:2]
                else:
                    cnt_text = [i for i in cnt_text if i[0] != "Unknown"]
                    if len(cnt_text) > 2:
                        cnt_text = cnt_text[:2]

            final_result.append([deepcopy(cnt_text), classify])
            cnt_text = []

    merged_final_result = merge_similar_reviewer_threads(
        final_result, sentence_texts, threshold=0.3
    )
    # Preserve the last split_texts group as a "tail" bucket (often metadata / trailing content).
    merged_final_result.append([new_split_texts[-1], ["N/A"]])

    # -------------------------------------------------------------------------
    # 6) Merge very short "marker-like" sentences into neighbors (same role only).
    #    This reduces fragmented headings like "Strengths:" or short bullets.
    # -------------------------------------------------------------------------
    N = len(sentence_texts)
    if N > 0:
        # Candidate: very short English fragments without tables.
        candidate = [False] * N
        for i, txt in enumerate(sentence_texts):
            if english_word_count(txt.strip()) <= 4 and len(txt.strip()) < 50 and "|" not in txt:
                candidate[i] = True

        # Decide target sentence for each candidate (itself, prev, or next).
        target_for = list(range(N))
        merges_by_target = defaultdict(list)  # target_idx -> list of (source_idx, pos)

        for i in range(N):
            if not candidate[i]:
                continue

            role = sentence_roles[i]
            prev_idx = i - 1 if i > 0 else None
            next_idx = i + 1 if i + 1 < N else None

            options = []
            # Allow merging only into same-role non-candidate neighbors with markdown constraints.
            if (
                prev_idx is not None
                and sentence_roles[prev_idx] == role
                and not candidate[prev_idx]
                and not _is_markdown_item_start(sentence_texts[i])  # marker itself can't merge backward
            ):
                options.append(("prev", prev_idx))

            if (
                next_idx is not None
                and sentence_roles[next_idx] == role
                and not candidate[next_idx]
                and not _is_markdown_item_start(sentence_texts[next_idx])  # next is a marker => don't merge forward
            ):
                options.append(("next", next_idx))

            if not options:
                continue

            if len(options) == 1:
                best_kind, best_target = options[0]
            else:
                # Heuristics: punctuation/format cues decide whether to attach before/after.
                s = (
                    sentence_texts[i]
                    .rstrip()
                    .rstrip("*")
                    .rstrip()
                    .lstrip()
                    .lstrip("*")
                    .lstrip()
                )
                first_char = s[0] if s else ""
                last_char = s[-1] if s else ""

                if last_char in ":," or first_char in ">#-":
                    best_kind, best_target = [opt for opt in options if opt[0] == "next"][0]
                elif (
                    last_char in ".!?;)]}"
                    or "\\end" in s
                    or s.strip().lower().startswith("please")
                    or "above" in s.lower()
                ):
                    best_kind, best_target = [opt for opt in options if opt[0] == "prev"][0]
                else:
                    best_kind, best_target = [opt for opt in options if opt[0] == "next"][0]
                    if DEBUG and english_word_count(s.strip()) > 2:
                        logger.debug("Ambiguous short fragment (defaulting next): %r", sentence_texts[i].rstrip())

            target_for[i] = best_target
            pos = "before" if best_kind == "next" else "after"
            merges_by_target[best_target].append((i, pos))

        # Build merged sentence strings.
        combined_texts = {}
        for target, items in merges_by_target.items():
            items_sorted = sorted(items, key=lambda x: x[0])
            before_parts = [sentence_texts[src] for src, pos in items_sorted if pos == "before"]
            after_parts = [sentence_texts[src] for src, pos in items_sorted if pos == "after"]

            base = sentence_texts[target]
            parts = []
            if before_parts:
                parts.append("\n".join(before_parts))
            parts.append(base)
            if after_parts:
                parts.append("\n".join(after_parts))
            combined_texts[target] = "\n".join(parts)

        # Re-index sentences: keep only targets; map old_id -> new_id.
        old2new = {}
        new_sentence_texts = []
        new_sentence_roles = []

        for i in range(N):
            if target_for[i] != i:
                continue
            new_idx = len(new_sentence_texts)
            old2new[i] = new_idx
            txt = combined_texts.get(i, sentence_texts[i])
            new_sentence_texts.append(txt)
            new_sentence_roles.append(sentence_roles[i])

        # Update reviews' sentence-id lists.
        for group in new_reviews:
            for e_idx, (_, payload) in enumerate(group):
                v = payload.get("value", {})
                new_value = {}
                for fname, id_list in v.items():
                    new_ids = sorted(set(old2new[sid] for sid in id_list if sid in old2new))
                    new_value[fname] = new_ids
                payload["value"] = new_value
                group[e_idx][1] = payload

        # Update split_texts' sentence-id lists.
        for item in merged_final_result:
            thread = item[0]  # [ [role, [ids]], ... ]
            for seg in thread:
                _, sid_list = seg
                new_ids = sorted(set(old2new[sid] for sid in sid_list if sid in old2new))
                seg[1] = new_ids

        sentence_texts = new_sentence_texts
        sentence_roles = new_sentence_roles  # kept for completeness; only used within this function

    # -------------------------------------------------------------------------
    # 7) Final output object (keep schema stable; remove classify).
    # -------------------------------------------------------------------------
    paper_out = dict(paper)
    paper_out["reviews"] = new_reviews
    paper_out["sentence_texts"] = sentence_texts
    paper_out["split_texts"] = merged_final_result
    del paper_out["classify"]

    return paper_out


# --------------------------- Main: batch processing ---------------------------

def main():
    for conf, year in GROUPS:
        # Load the "ground truth" sample objects keyed by paper id.
        mem = {}
        with open(f"{conf}.cc_{year}/{conf}.cc_{year}_sample.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                mem[obj["id"]] = obj

        out_dir = f"{conf}.cc_{year}"
        in_cls = f"{out_dir}/classify/{conf}.cc_{year}_gpt-5-mini_medium_classify.jsonl"
        out_bm = f"{out_dir}/classify/{conf}.cc_{year}_benchmark.jsonl"

        with open(in_cls, "r", encoding="utf-8") as f, open(out_bm, "w", encoding="utf-8") as w:
            for raw in tqdm(f):
                line = json.loads(raw)
                old = mem[line["id"]]

                # Keep decision/reviews/metareview from the original sample object.
                line["decision"] = old["decision"]
                line["reviews"] = old["reviews"]
                line["metareview"] = old["metareview"]

                new_obj = process_paper(line, min_words=0, strict_commas=True)
                json.dump(new_obj, w, ensure_ascii=False)
                w.write("\n")

        logger.info("Wrote benchmark: %s", out_bm)


if __name__ == "__main__":
    main()
