#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose
-------
Disambiguate sentence-level labels for peer-review discussion data.

This script reads a JSONL benchmark where each record contains:
- `sentence_texts`: a list of sentence strings (indexed by sentence_id)
- `split_texts`: discussion blocks, each with role->sentence_id mapping and one-or-more candidate labels

For blocks with a single candidate label (or where the role is Author), the label is inherited directly.
For blocks with multiple candidate labels, the script calls an LLM to classify each sentence, then
re-splits the original blocks based on the refined sentence labels.

Output
------
Writes a JSONL file where each record is augmented with:
- `classify_new`: sentence-level classification results grouped by original block_idx
- `for_judge`: a compact sentence_id -> labels mapping (for downstream inspection)
- `new_split_texts`: updated blocks after applying the re-splitting rules

Notes
-----
- The implementation uses a thread pool to process JSONL records in parallel. Output order is
  therefore **not guaranteed** to match input order.
- LLM calls are cached (via `LLMClient`), and may run against a local vLLM endpoint or a remote API.
- For open-source usage, do **not** hardcode API keys. Pass them via --api_key.
"""

import argparse
import json
import copy
import re
import sys
from typing import List, Dict, Any, Tuple, Set, DefaultDict
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from prompt_registry import refine_split_and_classify_system_prompt
from src.utils import (
    BaseArguments,
    LLMClient,
    _is_markdown_item_start,
    merge_similar_reviewer_threads,
    english_word_count,
)
from collections import defaultdict
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed  # Thread-pool parallelism

logger = logging.getLogger(__name__)


def normalize_label(raw_label: str, candidate_set: Set[str]) -> str:
    """
    Normalize an LLM-produced label to match the provided candidate label set.

    This function tries a series of lightweight transformations:
    - Strip common prefixes like "Label:" / "Category:"
    - Unwrap quotes/brackets/parentheses
    - Handle "X (…)" suffixes
    - Case-insensitive matching

    If no match is found, returns the stripped raw label unchanged.
    """
    label = raw_label.strip()

    if label in candidate_set:
        return label

    # (pattern, replacement) transformations
    transformations = [
        (r'^[Ll]abel[:\s-]+', ''),                # Remove common prefixes
        (r'^[Cc]ategory[:\s-]+', ''),
        (r'^[Cc]lass[:\s-]+', ''),
        (r'^[Tt]ype[:\s-]+', ''),
        (r'^["\'](.+?)["\']$', r'\1'),            # Unwrap quotes
        (r'^\[(.+?)\]$', r'\1'),                  # Unwrap square brackets
        (r'^\((.+?)\)$', r'\1'),                  # Unwrap parentheses
    ]

    for pattern, replacement in transformations:
        if re.match(pattern, label):
            new_label = re.sub(pattern, replacement, label).strip()
            if new_label in candidate_set:
                logger.debug("Normalized label %r -> %r", raw_label, new_label)
                return new_label

    # Handle "LABEL (extra info)"
    if "(" in label:
        base_label = label.split("(")[0].strip()
        if base_label in candidate_set:
            return base_label

    # Case-insensitive match
    for cand in candidate_set:
        if cand.lower() == label.lower():
            return cand

    return raw_label.strip()


def parse_sentence_labels(
    response: str,
    candidate_labels: List[str],
    expected_sentence_ids: List[int] = None,  # Keep order for positional remapping
) -> Tuple[Dict[int, List[str]], Dict[int, str], Dict[str, Any]]:
    """
    Parse sentence-level labels from an LLM response.

    Supported line formats (case-insensitive):
      - "Sentence {id}: {labels}"
      - "ID {id}: {labels}"
      - "{id}: {labels}"

    Robustness features:
    - If the model outputs IDs that do not match `expected_sentence_ids`, results are remapped
      **by position** (output order -> expected ID order).
    - Labels are normalized via `normalize_label()` and validated against the candidate set.

    Returns
    -------
    sentence2labels : Dict[int, List[str]]
        Parsed labels for each sentence_id.
    failed_sentences : Dict[int, str]
        Error messages for sentences that could not be parsed/validated.
    stats : Dict[str, Any]
        Parsing diagnostics (counts, missing IDs, remapping count, etc.).
    """
    sentence2labels = defaultdict(list)
    failed_sentences = {}
    stats = {
        "total_lines": 0,
        "parsed_lines": 0,
        "failed_lines": 0,
        "invalid_labels": 0,
        "normalized_labels": 0,
        "id_remapped": 0,  # Number of IDs remapped by position
        "missing_ids": [],
    }

    # Drop any chain-of-thought remnants if present.
    response = response.split("</think>")[-1]
    candidate_set = set(candidate_labels)

    parsed_outputs = []  # List[(output_id, labels_part)]

    patterns = [
        r"^Sentence\s+(\d+)\s*:\s*(.+)$",
        r"^ID\s+(\d+)\s*:\s*(.+)$",
        r"^(\d+)\s*:\s*(.+)$",
    ]

    for line in response.splitlines():
        line = line.strip()
        if not line:
            continue

        stats["total_lines"] += 1

        # Skip suspiciously short lines (typically not useful output).
        if len(line) < 5:
            continue

        matched = False
        output_id = None
        labels_part = None

        for pattern in patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                try:
                    output_id = int(match.group(1))
                    labels_part = match.group(2).strip()
                    matched = True
                    break
                except (IndexError, ValueError) as e:
                    logger.warning("Line format error: %r (%s)", line, e)
                    stats["failed_lines"] += 1

        if not matched:
            # Very loose fallback: "... {id}: {labels}"
            loose_match = re.match(r".*?(\d+)\s*[:.]\s*(.+)$", line)
            if loose_match:
                try:
                    output_id = int(loose_match.group(1))
                    labels_part = loose_match.group(2).strip()
                    matched = True
                except Exception:
                    pass

        if not matched or output_id is None:
            stats["failed_lines"] += 1
            logger.debug("Unrecognized output line: %r", line)
            continue

        parsed_outputs.append((output_id, labels_part))
        stats["parsed_lines"] += 1

    expected_len = len(expected_sentence_ids)
    actual_len = len(parsed_outputs)

    if expected_len != actual_len:
        logger.warning(
            "Sentence count mismatch: expected %d lines, got %d. Remapping by position.",
            expected_len,
            actual_len,
        )

    # Map output IDs to expected IDs by output position.
    id_mapping = {}
    for idx, (_, _) in enumerate(parsed_outputs):
        if idx < expected_len:
            expected_id = expected_sentence_ids[idx]
            original_id = parsed_outputs[idx][0]

            if original_id != expected_id:
                stats["id_remapped"] += 1
                logger.debug("ID remap: output %d -> expected %d", original_id, expected_id)

            id_mapping[original_id] = expected_id

    for output_id, labels_part in parsed_outputs:
        sentence_id = id_mapping.get(output_id, output_id)

        # Handle explicit empty label markers.
        if labels_part.upper() in ["N/A", "NA", "NONE", "NULL", ""]:
            sentence2labels[sentence_id] = ["N/A"]
            continue

        raw_labels = [
            lbl.strip()
            for lbl in re.split(r'[,;|]\s*', labels_part)
            if lbl.strip()
        ]

        if not raw_labels:
            failed_sentences[sentence_id] = "No labels found"
            stats["failed_lines"] += 1
            continue

        valid_labels = []
        invalid_labels = []

        for raw_label in raw_labels:
            normalized = normalize_label(raw_label, candidate_set)

            if normalized in candidate_set:
                valid_labels.append(normalized)
                if normalized != raw_label:
                    stats["normalized_labels"] += 1
            else:
                invalid_labels.append(raw_label)
                stats["invalid_labels"] += 1

        if valid_labels:
            sentence2labels[sentence_id] = valid_labels
        elif invalid_labels:
            failed_sentences[sentence_id] = f"All labels invalid: {invalid_labels}"
        else:
            failed_sentences[sentence_id] = "No valid labels extracted"

    if expected_sentence_ids:
        parsed_ids = set(sentence2labels.keys()) | set(failed_sentences.keys())
        stats["missing_ids"] = [sid for sid in expected_sentence_ids if sid not in parsed_ids]

        for missing_id in stats["missing_ids"]:
            failed_sentences[missing_id] = "Sentence ID missing from LLM response"

    if stats["id_remapped"] > 0:
        logger.info("Remapped %d sentence IDs by output position.", stats["id_remapped"])

    return sentence2labels, failed_sentences, stats


def build_sentence_classification_prompt(
    sentence_ids: List[int],
    sentence_texts: List[str],
    roles: Dict[int, str],
) -> str:
    """
    Build the user prompt for a batch of sentence IDs.

    Each entry is formatted as:
      ID: {sid}
      [Role] Sentence text
    """
    prompt_parts = []
    for sid in sentence_ids:
        role_prefix = f"[{roles[sid]}] " if sid in roles else ""
        prompt_parts.append(f"ID: {sid}\n{role_prefix}{sentence_texts[sid]}")

    return "\n\n".join(prompt_parts) + "\n"


def update_split_texts_with_classification(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Re-split `record['split_texts']` based on sentence-level classifications in `record['classify_new']`.

    High-level rules
    ----------------
    For each original block (in the original `split_texts` order):
    1) Collect all Reviewer sentences in that block (from `classify_new`, grouped by block_idx).
    2) Sort those Reviewer sentences by `sentence_id`, and split into sub-blocks when:
       - The label set changes, OR
       - The sentence_id gap to the previous Reviewer sentence is > 2, OR
       - The sentence is a Markdown list-item start (`_is_markdown_item_start`).
       Additional guards based on `english_word_count()` are applied (kept as-is).
    3) For each new sub-block:
       - Reviewer roles include only sentences in that sub-block.
       - Non-Reviewer roles from the original block (e.g., Author) are copied into each sub-block.
       - The category list is the union of all Reviewer labels in the sub-block (deduped, stable order).
    4) If an original block contains no Reviewer entries, the block is preserved as-is.

    Returns
    -------
    record : Dict[str, Any]
        A deep-copied record with `record['new_split_texts']` added.
    """
    # Deepcopy to avoid mutating the input record in-place.
    record = copy.deepcopy(record)

    split_texts: List[Any] = record.get("split_texts", [])
    sentence_texts: List[str] = record.get("sentence_texts", [])
    classify_new: List[List[Dict[str, Any]]] = record.get("classify_new", [])

    # Identify the (optional) block whose category includes "N/A".
    na_block_indices: List[int] = []
    for idx, blk in enumerate(split_texts):
        if (
            isinstance(blk, list)
            and len(blk) == 2
            and isinstance(blk[1], list)
            and "N/A" in blk[1]
        ):
            na_block_indices.append(idx)

    # The downstream logic assumes at most one "N/A" category block.
    assert len(na_block_indices) <= 1, f"Found {len(na_block_indices)} blocks containing 'N/A' label"
    na_block_idx = na_block_indices[0] if na_block_indices else None

    # Flatten classify_new into a mapping: block_idx -> list[entry]
    block_to_entries: DefaultDict[int, List[Dict[str, Any]]] = defaultdict(list)
    for group in classify_new:
        for item in group:
            block_idx = item.get("block_idx")
            if block_idx is None:
                continue
            block_to_entries[block_idx].append(item)

    new_split_texts: List[Any] = []

    # Process blocks in original order.
    for block_idx, old_block in enumerate(split_texts):
        if na_block_idx is not None and block_idx == na_block_idx:
            continue

        old_roles = old_block[0]  # [[role, [sid...]], ...]
        old_cat = old_block[1]    # e.g. ["QUAL-EXP"]
        if None in old_cat:
            continue

        entries = block_to_entries.get(block_idx, [])

        reviewer_entries = [e for e in entries if str(e.get("role", "")).startswith("Reviewer")]

        non_reviewer_roles = []
        for role, sids in old_roles:
            if not str(role).startswith("Reviewer"):
                # Preserve original sentence order for non-reviewer roles.
                non_reviewer_roles.append([role, list(sids)])

        # If there are no reviewer entries in this block, keep it unchanged.
        if not reviewer_entries:
            new_split_texts.append(old_block)
            continue

        # Sort reviewer entries by sentence_id, then split into groups by rules.
        reviewer_entries_sorted = sorted(
            reviewer_entries,
            key=lambda e: int(e.get("sentence_id"))
        )

        groups: List[List[Dict[str, Any]]] = []
        current_group: List[Dict[str, Any]] = []

        for i, entry in enumerate(reviewer_entries_sorted):
            sid = int(entry.get("sentence_id"))
            labels = entry.get("label", [])
            if not isinstance(labels, list):
                labels = [labels] if labels is not None else []

            # Start a new group for the first reviewer sentence.
            if i == 0:
                current_group = [entry]
                continue

            prev = reviewer_entries_sorted[i - 1]
            prev_sid = int(prev.get("sentence_id"))
            prev_labels = prev.get("label", [])
            if not isinstance(prev_labels, list):
                prev_labels = [prev_labels] if prev_labels is not None else []

            text = sentence_texts[sid] if 0 <= sid < len(sentence_texts) else ""
            is_item_start = _is_markdown_item_start(text)

            split_here = False
            txt = "\n".join([sentence_texts[i["sentence_id"]] for i in current_group])

            # 1) Label set changes (compare as sets to ignore ordering)
            if set(labels) != set(prev_labels) and english_word_count(txt) > 1:
                split_here = True
            # 2) Sentence-id gap too large
            elif sid - prev_sid > 2 and english_word_count(txt) > 5:
                split_here = True
            # 3) Markdown list-item boundary
            elif is_item_start and english_word_count(txt) > 5:
                split_here = True

            if split_here:
                if current_group:
                    groups.append(current_group)
                current_group = [entry]
            else:
                current_group.append(entry)

        if current_group:
            groups.append(current_group)

        # Convert each reviewer-group into a new split_texts block.
        for g in groups:
            reviewer_role_to_sids: DefaultDict[str, List[int]] = defaultdict(list)
            label_union: List[str] = []  # Stable-order union of labels inside this sub-block.

            for e in g:
                role = str(e.get("role"))
                sid = int(e.get("sentence_id"))
                reviewer_role_to_sids[role].append(sid)

                labels = e.get("label", [])
                assert isinstance(labels, list)
                for lab in labels:
                    if lab not in label_union:
                        label_union.append(lab)

            reviewer_role_pairs = []
            for role, sids in reviewer_role_to_sids.items():
                sids_sorted = sorted(sids)
                reviewer_role_pairs.append((role, sids_sorted))

            reviewer_role_pairs.sort(key=lambda x: x[1][0])  # Sort by earliest sentence_id.

            new_roles = []
            for role, sids in reviewer_role_pairs:
                new_roles.append([role, sids])

            # Copy non-reviewer roles (Author, etc.) into each new sub-block.
            for nr in non_reviewer_roles:
                new_roles.append([nr[0], list(nr[1])])

            assert label_union
            new_cat = label_union
            assert None not in new_cat
            new_split_texts.append([new_roles, new_cat])

    # Merge adjacent/near-duplicate reviewer threads, then append the original "N/A" block (kept as-is).
    new_split_texts = merge_similar_reviewer_threads(new_split_texts, sentence_texts, threshold=0.3)
    new_split_texts.append(copy.deepcopy(split_texts[na_block_idx]))
    record["new_split_texts"] = new_split_texts
    return record


def process_single_record(args, file_id: int, line: str):
    """
    Process a single JSONL record (one paper) to enable parallel execution.

    Returns
    -------
    (data, num_local) :
      - data: processed record dict (or None if skipped)
      - num_local: number of multi-label blocks encountered in this record
    """
    num_local = 0
    data = json.loads(line)
    paper_id = data.get("id", file_id)
    split_texts = data.get("split_texts", [])
    sentence_texts = data.get("sentence_texts", [])
    for_judge: Dict[int, List[str]] = {}

    if not split_texts or not sentence_texts:
        logger.warning(
            "Skipping record %r: missing split_texts or sentence_texts",
            data.get("id", file_id),
        )
        return None, 0

    classify_new = []  # Final sentence-level results

    # Track outstanding LLM tasks for multi-label blocks.
    task_mapping = {}  # task_key -> task_info

    # Step 1: Build tasks and/or directly assign labels.
    for block_idx, block in enumerate(split_texts):
        candidate_labels = block[1]
        if None in candidate_labels:
            continue

        block_sentences = []
        for role_sentence in block[0]:
            role = role_sentence[0]

            # Authors are not re-classified. Also, single-label blocks do not need LLM.
            if role.lower() == "author" or len(candidate_labels) <= 1:
                for sid in role_sentence[1]:
                    classify_new.append({
                        "sentence_id": sid,
                        "role": role,
                        "label": candidate_labels,
                        "candidate_labels": candidate_labels,
                        "block_idx": block_idx
                    })
                continue

            sents = role_sentence[1]
            block_sentences.extend([(sid, role) for sid in sents])

        sentence_ids = [sid for sid, _ in block_sentences]
        roles = {sid: role for sid, role in block_sentences}

        # Case 1: Single label -> direct assignment.
        if len(candidate_labels) <= 1:
            for sid in sentence_ids:
                classify_new.append({
                    "sentence_id": sid,
                    "role": roles[sid],
                    "label": candidate_labels,
                    "candidate_labels": candidate_labels,
                    "block_idx": block_idx
                })
            continue

        # Case 2: Multiple labels -> create an LLM classification task.
        batch_size = 20
        if len(sentence_ids) > batch_size:
            logger.info(
                "Record %r, block %d: %d sentences; batching with size=%d",
                data.get("id", file_id),
                block_idx,
                len(sentence_ids),
                batch_size,
            )

        # Count multi-label blocks (matches original behavior).
        num_local += 1

        for batch_start in range(0, len(sentence_ids), batch_size):
            batch_ids = sentence_ids[batch_start: batch_start + batch_size]
            prompt_text = build_sentence_classification_prompt(batch_ids, sentence_texts, roles)

            logger.info(
                "Submitting LLM task: record=%r block=%d batch_start=%d batch_size=%d",
                data.get("id", file_id),
                block_idx,
                batch_start,
                len(batch_ids),
            )

            key = client.submit_task(
                prompt_text,
                system_prompt=refine_split_and_classify_system_prompt(candidate_labels, len(batch_ids)),
                temperature=0.6,
                reasoning_effort=args.effort
            )

            task_mapping[key] = {
                "paper_id": paper_id,
                "block_idx": block_idx,
                "sentence_ids": batch_ids,
                "roles": roles,
                "candidate_labels": candidate_labels
            }

    # Step 2: Collect LLM results.
    if task_mapping:
        logger.info("Record %r: collecting %d LLM task(s)", data.get("id", file_id), len(task_mapping))

    all_failed = {}  # task_key -> failed_sentences

    for key, task_info in task_mapping.items():
        result = client.get_result(key)
        response = result["content"]

        if response is None:
            logger.warning("Task %r returned no content; assigning N/A to batch.", key)
            for sid in task_info["sentence_ids"]:
                classify_new.append({
                    "sentence_id": sid,
                    "role": task_info["roles"][sid],
                    "label": ["N/A"],
                    "candidate_labels": task_info["candidate_labels"],
                    "parsing_status": "no_response",
                    "block_idx": task_info["block_idx"]
                })
            continue

        expected_ids = task_info["sentence_ids"]
        sentence2labels, failed_sentences, stats = parse_sentence_labels(
            response,
            task_info["candidate_labels"],
            expected_ids
        )

        for k, v in sentence2labels.items():
            for_judge[k] = v

        if stats["failed_lines"] > 0:
            logger.warning(
                "Task %r: %d/%d output lines failed to parse",
                key,
                stats["failed_lines"],
                stats["total_lines"],
            )

        if failed_sentences:
            all_failed[key] = failed_sentences
            logger.error(
                "Task %r: failed to parse %d sentence(s): %s",
                key,
                len(failed_sentences),
                failed_sentences,
            )

        for sid in task_info["sentence_ids"]:
            if sid in failed_sentences:
                classify_new.append({
                    "sentence_id": sid,
                    "role": task_info["roles"][sid],
                    "label": ["N/A"],
                    "candidate_labels": task_info["candidate_labels"],
                    "parsing_status": "failed",
                    "parsing_error": failed_sentences[sid],
                    "block_idx": task_info["block_idx"]
                })
            else:
                labels = sentence2labels.get(sid, ["N/A"])
                classify_new.append({
                    "sentence_id": sid,
                    "role": task_info["roles"][sid],
                    "label": labels,
                    "candidate_labels": task_info["candidate_labels"],
                    "parsing_status": "success",
                    "block_idx": task_info["block_idx"]
                })

    # Group classifications by block_idx (stable per block), then sort by sentence_id inside each group.
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in classify_new:
        if None in item["label"]:
            continue
        key = str(item["block_idx"])
        grouped.setdefault(key, []).append(item)

    for key in grouped:
        grouped[key].sort(key=lambda x: x["sentence_id"])
    classify_new_sorted = list(grouped.values())

    # Step 3: Build output record.
    data["classify_new"] = classify_new_sorted
    data["for_judge"] = for_judge
    data = update_split_texts_with_classification(data)

    if all_failed:
        logger.error(
            "Record %r: total failed sentences across all tasks: %d",
            data.get("id", file_id),
            sum(len(v) for v in all_failed.values()),
        )

    return data, num_local


def classify_all(args) -> None:
    """
    Main entry point.

    Processing logic
    ---------------
    - Single-label blocks: direct inheritance
    - Multi-label blocks: LLM sentence classification
    - Post-process: re-split blocks based on sentence-level labels

    Parallelism
    -----------
    Each JSONL record is processed in a thread pool. Results are written as soon as each task finishes.
    Output order is therefore not guaranteed.

    Note
    ----
    The thread pool size here is currently fixed to 32 (kept as-is to avoid changing behavior).
    """
    if args.org == "eval":
        jsonl_path = f"pre_eval/classify/eval_benchmark.jsonl"
        output_path = f"pre_eval/split_final/evaluation_{args.model.replace('/', '_')}_{args.effort}_split_clean.jsonl"
    else:
        jsonl_path = f"{args.org}.cc_{args.year}/classify/{args.org}.cc_{args.year}_benchmark.jsonl"
        output_path = f"{args.org}.cc_{args.year}/split_final/{args.org}.cc_{args.year}_{args.model.replace('/', '_')}_{args.effort}_split_clean.jsonl"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    total_num = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        with open(jsonl_path, "r", encoding="utf-8") as in_f:
            with ThreadPoolExecutor(max_workers=32) as executor:
                futures = []
                for file_id, line in enumerate(in_f):
                    futures.append(executor.submit(process_single_record, args, file_id, line))

                for future in as_completed(futures):
                    result, num_local = future.result()
                    total_num += num_local

                    if result is None:
                        continue
                    json.dump(result, out_f, ensure_ascii=False)
                    out_f.write('\n')
                    out_f.flush()

    logger.info("Finished. Total multi-label blocks processed (proxy for LLM usage): %d", total_num)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Disambiguate multi-label sentence classifications in peer-review data."
    )
    BaseArguments.add_to_parser(parser, model_default="gpt-5-mini", effort_default="medium")
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
