#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Purpose
-------
Provide shared utilities for review preprocessing, LLM calls, and evaluation metrics.
"""

import asyncio
import inspect
import os
import re
import sys
import json
import time
import requests
import hashlib
import threading
import logging
import subprocess
import signal
import math
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Set, Any
from urllib.parse import urlparse
try:
    from openai import AsyncOpenAI, OpenAI
except ImportError:
    from openai import OpenAI
    AsyncOpenAI = None
from itertools import combinations
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

GROUPS = [("ICLR", year) for year in range(2017, 2026)]
GROUPS += [("NeurIPS", year) for year in range(2021, 2025)]

MAX_NUM_TOKENS = 32768


class BaseArguments:
    """
    Shared CLI argument builder for scripts that use LLMClient.

    This keeps org/year/model/base_url/etc. consistent across scripts and
    provides a post-parse normalization hook.
    """

    @staticmethod
    def add_to_parser(
        parser,
        *,
        org_default: str = "ICLR",
        year_default: int = 2025,
        model_default: str = "gpt-5-mini",
        effort_default: str = "medium",
        max_workers_default: int = 128,
        base_url_default: str = "https://api.openai.com/v1",
        api_key_default: str = "YOUR_API_KEY",
        cache_version_default: str = "v1",
        include_org_year: bool = True,
        include_model: bool = True,
        include_effort: bool = True,
        include_max_workers: bool = True,
        include_base_url: bool = True,
        include_api_key: bool = True,
        include_cache_version: bool = True,
    ) -> None:
        if include_org_year:
            parser.add_argument("--org", default=org_default, help="Organization name, e.g. ICLR or NeurIPS")
            parser.add_argument("--year", type=int, default=year_default, help="Conference year (e.g., 2020)")
        if include_model:
            parser.add_argument("--model", default=model_default, help="LLM model name")
        if include_effort:
            parser.add_argument("--effort", default=effort_default, help="Reasoning effort level")
        if include_max_workers:
            parser.add_argument("--max_workers", type=int, default=max_workers_default, help="Maximum parallel workers")
        if include_base_url:
            parser.add_argument("--base_url", default=base_url_default, help="LLM API base URL")
        if include_api_key:
            parser.add_argument("--api_key", default=api_key_default, help="LLM API key")
        if include_cache_version:
            parser.add_argument("--cache_version", default=cache_version_default, help="Cache version for LLM responses")

    @staticmethod
    def apply(args) -> None:
        """Normalize org/year and populate paper_series after parsing."""
        if hasattr(args, "org") and isinstance(args.org, str) and args.org.endswith(".cc"):
            args.org = args.org[:-3]
        if hasattr(args, "org") and hasattr(args, "year"):
            args.paper_series = f"{args.org}.cc_{args.year}"

CUE_TERMS = [
    # Sequence & enumeration
    "first", "firstly", "first of all", "to begin with", "to start with", "for starters",
    "in the first place", "for one thing", "for another",
    "second", "secondly", "in the second place",
    "third", "thirdly", "in the third place",
    "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
    "next", "then", "afterwards", "afterward", "subsequently", "thereafter",
    "earlier", "previously", "beforehand", "before that", "prior to that",
    "later", "soon", "immediately", "eventually", "ultimately",
    "finally", "last", "lastly", "last but not least",
    "meanwhile", "in the meantime", "at the same time",
    "to conclude", "to finish", "in the end",

    # Addition
    "also", "in addition", "additionally", "moreover", "furthermore", "further",
    "besides", "what's more", "what is more", "as well", "as well as",
    "too", "plus", "on top of that", "equally", "equally important",
    "another", "a further", "a related point",

    # Contrast / concession
    "however", "but", "yet", "nevertheless", "nonetheless", "still", "even so",
    "by contrast", "in contrast", "on the other hand", "on the contrary",
    "conversely", "instead",
    "whereas", "while", "although", "though", "even though", "albeit",
    "despite this", "in spite of this", "that said", "having said that",
    "notwithstanding",

    # Cause → effect / result
    "so", "therefore", "thus", "hence", "accordingly", "consequently",
    "as a result", "as such", "for this reason", "for that reason",
    "because of this", "it follows that", "which means", "leading to", "resulting in",

    # Clarification / restatement / specification
    "that is", "i.e.", "namely", "in other words", "to put it differently",
    "to be more specific", "specifically", "more precisely", "to be precise",
    "in particular", "put differently", "that is to say",

    # Examples
    "for example", "e.g.", "for instance", "such as", "including", "like",

    # Comparison / similarity
    "similarly", "likewise", "in the same way", "by the same token", "correspondingly", "simultaneously",

    # Condition / alternative
    "if", "unless", "otherwise", "provided that", "as long as",
    "on condition that", "assuming that", "in case",
    "when", "whenever",
    "alternatively", "or else", "else",

    # Summary / conclusion
    "overall", "in sum", "in summary", "to summarize", "to sum up",
    "in short", "in brief", "briefly", "all in all", "on the whole",
    "to conclude", "in conclusion", "ultimately", "finally",

    # Emphasis / signposting
    "importantly", "notably", "in fact", "indeed", "above all",
    "crucially", "significantly", "key point", "the main point",

    # Topic shift / section cues
    "regarding", "as for", "with respect to", "concerning", "about",
    "speaking of", "in terms of", "turning to", "moving on to",
    "on the topic of", "on the subject of", "as to", "as regards",

    # Exception / restriction
    "except", "except that", "apart from", "other than", "save that", "excluding",

    # Common review-discourse hedges and rhetorical transitions
    "admittedly",
    "granted",
    "of course",
    "after all",
    "be that as it may",
    "in reality",
    "in practice",
    "in theory",
    "on the one hand",
    "even then",
    "then again",
    "all the same",
    "by way of contrast",
    "by comparison",
    "in comparison",
    "contrary to",
    "despite",
    "in spite of",
    "unlike",
    "rather",
    "rather than",
    "as much as",
    "even if",
    "even when",
    "regardless",
    "regardless of",
    "for all that",
    "strangely enough",
    "ironically",

    # Softened concession → pivot phrases
    "to be sure",
    "to be fair",
    "it is true that",
    "it is true",
    "I admit",
    "I concede",
    "I grant",
    "it should be noted",
    "it is worth noting",
    "it should be acknowledged",
    "even still",

    # Partial agreement / partial validity cues
    "to some extent",
    "to a certain extent",
    "to a lesser extent",
    "to a large extent",
    "to an extent",
    "in part",
    "at least",
    "at least in part",
    "at least to some extent",

    # Discourse closure / alternate framing
    "either way",
    "in either case",
    "in either event",
    "in any case",
    "in any event",
    "by and large",

    # Perspective shifts common in peer reviews
    "from a different perspective",
    "from another perspective",
    "from a different angle",
    "from another angle",
    "from a theoretical perspective",
    "from an empirical perspective",
    "from the reviewer's perspective",
    "from the reader's perspective",

    # Typical "praise-then-critique" review starters
    "while this is a strength",
    "while this is a limitation",
    "while this is promising",
    "while the idea is interesting",
    "while the results are encouraging",
    "while the empirical results are promising",
    "while the approach is novel",
    "while the paper is well written",
]

MARKER_RE = re.compile(
    r"""
    ^
    \*{0,3}                             # Optional leading '*' (0–3), e.g., '*', '**', '***'
    \(?\[?(                             # Optional '(' and/or '[' before the marker
        \d+                             #   Pure digits
    | [A-Za-z>\-\*](\d+)?               #   A letter/symbol optionally followed by digits
    | (?:strength|weakness|question)\d* #   strength/weakness/question with optional digits
    )
    [.:)\]]?                            # Optional trailing punctuation: '.', ':', ')', or ']'
    $
    """,
    re.IGNORECASE | re.VERBOSE,
)

STRICT_MARKER_RE = re.compile(
    r"""
    ^
    (?:                     # Two accepted forms:
        \((?P<num>\d+)\)     #   (123)
      | (?P<num2>\d+)        #   123
    )
    [.:)]?                   # Optional trailing punctuation: '.', ':', or ')'
    $
    """,
    re.VERBOSE,
)

TABLE_SEP_RE = re.compile(
    r'^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$'
)

_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def dedup_by_word_sequence(paragraphs: List[str]) -> List[str]:
    """
    Deduplicate paragraphs by their extracted English word sequence.

    Two paragraphs are considered duplicates if the sequence of ASCII words
    (lowercased, in-order) is identical, ignoring punctuation and whitespace.

    Examples:
      "Hello, world!" -> ["hello", "world"]
      "hello   world" -> ["hello", "world"]  => duplicate
      "world hello"   -> ["world", "hello"]  => not a duplicate (order differs)
    """
    seen: set[Tuple[str, ...]] = set()
    out: List[str] = []

    for p in paragraphs:
        words = tuple(w.lower() for w in _WORD_RE.findall(p or ""))
        if words in seen:
            continue
        seen.add(words)
        out.append(p)

    return out


def english_word_count(s: str) -> int:
    """Count ASCII-alphabetic word tokens (used for lightweight length heuristics)."""
    return len(re.findall(r"[A-Za-z]+", s))


def _is_markdown_item_start(text: str, strict=False) -> bool:
    """
    Check whether a line looks like the start of a Markdown list item or marker.

    Supports:
    - Unordered list: '* xxx', '- xxx', '+ xxx', '• xxx' (with following whitespace)
    - Ordered list: '1. xxx', '2) xxx'
    - Marker tokens: '(1)', '[a]', 'strength2:', 'question3)' etc.
    """
    if not text:
        return False

    s = text.lstrip()

    # Unordered list bullets
    if re.match(r'^(\*{1,3}|-|\+|•)\s+', s):
        return True

    # Ordered list or marker token
    first_token = s.split(maxsplit=1)[0]
    if STRICT_MARKER_RE.match(first_token):
        return True
    if not strict and MARKER_RE.match(first_token):
        return True

    return False


def is_table_header_and_sep(lines: List[str]) -> bool:
    """
    Detect a Markdown table header + separator pattern at the start of `lines`.

    Requires:
      - At least 2 lines
      - Line 0 contains '|'
      - Line 1 matches a typical Markdown table separator row
    """
    if len(lines) < 2:
        return False
    head = lines[0].rstrip('\n')
    sep  = lines[1].rstrip('\n')
    return ('|' in head) and bool(TABLE_SEP_RE.match(sep))


def merge_short_texts(
    sentence_texts: List[str],
    max_short_words: int = 4,
    max_short_len: int = 60
) -> List[str]:
    """
    Merge very short English lines into adjacent non-short neighbors.

    Heuristic rules:
    - A "short" line is one with <= max_short_words English words AND < max_short_len chars.
    - Avoid merging across Markdown list markers.
    - Avoid merging into a Markdown table header/separator region.
    - Prefer attaching "intro-like" fragments to the next line; otherwise attach to the previous line.

    Returns a new list of lines with merges applied while preserving overall order.
    """
    N = len(sentence_texts)
    if N == 0:
        return []

    # Pass 1: identify candidate short lines
    candidate = [False] * N
    for i, txt in enumerate(sentence_texts):
        s = (txt or "").strip()
        if not s or "|" in s:   # Do not merge table-ish lines
            continue
        if english_word_count(s) <= max_short_words and len(s) < max_short_len:
            candidate[i] = True

    # Pass 2: decide a merge target for each candidate (self by default)
    target_for = list(range(N))
    merges_by_target = defaultdict(list)  # target_idx -> list[(source_idx, pos)]

    for i in range(N):
        if not candidate[i]:
            continue

        prev_idx = i - 1 if i > 0 else None
        next_idx = i + 1 if i + 1 < N else None
        options = []

        # Merge into previous if previous is not short and current is not a marker
        if (
            prev_idx is not None
            and not candidate[prev_idx]
            and not _is_markdown_item_start(sentence_texts[i])
        ):
            options.append(("prev", prev_idx))

        # Merge into next if next is not short, next is not a marker, and next is not a table header/separator
        if (
            next_idx is not None
            and not candidate[next_idx]
            and not _is_markdown_item_start(sentence_texts[next_idx])
            and not is_table_header_and_sep([i.strip() for i in sentence_texts[next_idx].splitlines() if i.strip()])
        ):
            options.append(("next", next_idx))

        if not options:
            continue

        # Select best target
        if len(options) == 1:
            best_kind, best_target = options[0]
        else:
            s = sentence_texts[i]
            s = s.rstrip().rstrip("*").rstrip()
            s = s.lstrip().lstrip("*").lstrip()
            first_char = s[0] if s else ""
            last_char = s[-1] if s else ""

            # Intro-like fragments: attach forward
            if last_char in ":," or first_char in ">#-":
                best_kind, best_target = [opt for opt in options if opt[0] == "next"][0]
            # Sentence-like endings: attach backward
            elif (
                last_char in ".!?;)]}"
                or "\\end" in s
                or s.strip().lower().startswith("please")
                or "above" in s.lower()
            ):
                best_kind, best_target = [opt for opt in options if opt[0] == "prev"][0]
            else:
                best_kind, best_target = [opt for opt in options if opt[0] == "next"][0]

        target_for[i] = best_target
        pos = "before" if best_kind == "next" else "after"
        merges_by_target[best_target].append((i, pos))

    # Pass 3: materialize merged texts
    combined_texts = {}
    for target, items in merges_by_target.items():
        items_sorted = sorted(items, key=lambda x: x[0])
        before_parts = [sentence_texts[src] for src, pos in items_sorted if pos == "before"]
        after_parts  = [sentence_texts[src] for src, pos in items_sorted if pos == "after"]

        base = sentence_texts[target]
        parts = []
        if before_parts:
            parts.append("\n".join(before_parts))
        parts.append(base)
        if after_parts:
            parts.append("\n".join(after_parts))

        combined_texts[target] = "\n".join(parts)

    # Pass 4: emit final list in original order
    result = []
    for i in range(N):
        if i in combined_texts:
            result.append(combined_texts[i])
        elif target_for[i] == i:
            result.append(sentence_texts[i])
        else:
            continue

    return result


def _split_long_segment_no_tables(text_block: str, max_w: int):
    """
    Sentence-level splitting for a block assumed to contain no Markdown tables.

    If the block exceeds `max_w` English words, it is split by sentence-ending
    punctuation into approximately equal-sized chunks (by word count).
    """
    if max_w is None:
        return [text_block]

    total_words = english_word_count(text_block)
    if total_words <= max_w:
        return [text_block]

    sentence_end_re = re.compile(r'([.!?;。！？])')
    tokens = sentence_end_re.split(text_block)

    sentences = []
    for i in range(0, len(tokens), 2):
        chunk = tokens[i]
        if not chunk:
            continue
        delim = tokens[i + 1] if i + 1 < len(tokens) else ""
        sent = (chunk + delim).strip()
        if sent:
            sentences.append(sent)

    if not sentences:
        return [text_block]

    word_counts = [english_word_count(s) for s in sentences]
    total_words = sum(word_counts)

    n_chunks = (total_words + max_w - 1) // max_w
    if n_chunks <= 1:
        return [text_block]

    ideal_per_chunk = total_words / n_chunks
    thresholds = [ideal_per_chunk * i for i in range(1, n_chunks)]

    segments = []
    current_sentences = []
    current_words = 0
    cum_words = 0
    threshold_idx = 0

    for sent, w in zip(sentences, word_counts):
        current_sentences.append(sent)
        current_words += w
        cum_words += w

        if threshold_idx < len(thresholds) and cum_words >= thresholds[threshold_idx]:
            segments.append("\n".join(s.strip() for s in current_sentences).strip())
            current_sentences = []
            current_words = 0
            threshold_idx += 1
            while threshold_idx < len(thresholds) and cum_words >= thresholds[threshold_idx]:
                threshold_idx += 1

    if current_sentences:
        segments.append("\n".join(s.strip() for s in current_sentences).strip())
    return segments


def split_long_segment_by_sentences(text_block: str, max_w: int):
    """
    Enhanced splitter that preserves Markdown tables as atomic blocks.

    Steps:
      1) Scan `text_block` line-by-line and identify Markdown tables
         (header + separator row + subsequent '|' rows).
      2) Split the content into interleaved segments:
           - kind="text": split via _split_long_segment_no_tables
           - kind="table": kept intact (never split)
      3) Return a flat list of chunks in original order.
    """
    if max_w is None:
        return [text_block]

    lines = text_block.splitlines()
    if not lines:
        return [text_block]

    segments = []  # (kind, text) where kind in {"text", "table"}
    buf_lines = []
    i = 0
    n = len(lines)

    while i < n:
        if is_table_header_and_sep(lines[i: ]):
            if buf_lines:
                text_seg = "\n".join(buf_lines).strip()
                if text_seg:
                    segments.append(("text", text_seg))
                buf_lines = []

            table_lines = [lines[i].rstrip('\n'), lines[i + 1].rstrip('\n')]
            i += 2
            while i < n:
                row = lines[i].rstrip('\n')
                if row.strip() and ('|' in row):
                    table_lines.append(row)
                    i += 1
                else:
                    break

            table_text = "\n".join(table_lines)
            segments.append(("table", table_text))
        else:
            buf_lines.append(lines[i])
            i += 1

    if buf_lines:
        text_seg = "\n".join(buf_lines).strip()
        if text_seg:
            segments.append(("text", text_seg))

    if all(kind == "text" for kind, _ in segments):
        return _split_long_segment_no_tables(text_block, max_w)

    result_chunks = []
    for kind, seg_text in segments:
        if kind == "table":
            result_chunks.append(seg_text)
        else:
            result_chunks.extend(_split_long_segment_no_tables(seg_text, max_w))

    return result_chunks


def split_text_into_chunks(
    text: str,
    min_words: int = 4,
    max_words: int = 150,
    strict_commas: bool = False
):
    """
    Split a long text into chunks suitable for prompting.

    Pipeline:
      - Pass 1: line-wise parsing with optional cue-term splitting and table preservation
      - Pass 2: merge very short lines (heuristic)
      - Pass 3: sentence-level splitting for overlong segments (table-safe)

    Parameters:
      - min_words: short-line threshold (<= min_words attempts to merge forward/backward)
      - max_words: maximum English word count per chunk; longer chunks are sentence-split
      - strict_commas: if True, require a comma after cue terms (e.g., "First,") to trigger a split
    """
    has_english_re = re.compile(r'[A-Za-z]')

    def cues_to_pattern(cues):
        parts = []
        for c in sorted(cues, key=len, reverse=True):
            esc = re.escape(c).replace(r'\ ', r'\s+')
            parts.append(esc)
        return r'(?:' + '|'.join(parts) + r')'

    cues_pat = cues_to_pattern(CUE_TERMS)

    if strict_commas:
        cue_boundary_re = re.compile(
            r'([.!?;…。！？])\s+(?=(?:' + cues_pat + r')\b\s*,)', re.IGNORECASE)
    else:
        cue_boundary_re = re.compile(
            r'([.!?;…。！？])\s+(?=(?:' + cues_pat + r')\b)', re.IGNORECASE)

    def split_on_cue_markers(line_text: str):
        segs = []
        last = 0
        text = line_text
        for m in cue_boundary_re.finditer(text):
            k = m.end()
            while k < len(text) and text[k].isspace():
                k += 1
            split_idx = k
            left = text[last:split_idx].rstrip()
            if left:
                segs.append((left, False))
            last = split_idx

        tail = text[last:].strip()
        if tail:
            forced = bool(re.match(r'^\s*(?:' + cues_pat + r')\b', tail, re.IGNORECASE))
            segs.append((tail, forced))
        if not segs:
            return [(text.strip(), False)]

        for i in range(len(segs)):
            if re.match(r'^\s*(?:' + cues_pat + r')\b', segs[i][0], re.IGNORECASE):
                segs[i] = (segs[i][0], True)
        return segs

    # Pass 1: collect line/table blocks
    lines = text.splitlines()
    i = 0
    blocks = []

    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        if not line:
            i += 1
            continue

        # Markdown table: header + separator row + subsequent '|' rows
        if is_table_header_and_sep(lines[i: ]):
            block_lines = [lines[i].rstrip('\n'), lines[i + 1].rstrip('\n')]
            i += 2
            while i < len(lines):
                row = lines[i].rstrip('\n')
                if row.strip() and ('|' in row):
                    block_lines.append(row)
                    i += 1
                else:
                    break
            table_text = "\n".join(block_lines).strip()
            if table_text:
                blocks.append(table_text)
            continue

        # Skip decorative lines with neither English letters nor CJK characters
        if not has_english_re.search(line) and not re.search(r'[\u4e00-\u9fff]', line):
            i += 1
            continue

        for seg, forced in split_on_cue_markers(line):
            if seg:
                blocks.append(seg)
        i += 1

    # Pass 2: merge short blocks into neighbors
    blocks = merge_short_texts(blocks, max_short_words=min_words, max_short_len=min_words*15)

    # Pass 3: split oversized blocks by sentences (table-safe)
    chunks = []
    for b in blocks:
        content = b.strip()
        pieces = split_long_segment_by_sentences(content, max_words)
        for p in pieces:
            if p.strip():
                chunks.append(p.strip())

    return chunks


def extract_labels_from_prompt_text(main_prompt_text: str) -> Tuple[Set[str], Dict[str, str]]:
    """
    Extract allowed label tokens and their definitions from bullet lines.

    Supported formats (examples):
      - "- SUM (Summary) – What was done ..."
      - "- QUAL-EXP (Experimental ...): Assesses ..."
      - "- SUM" followed by a non-bullet explanatory paragraph.

    Returns:
      allowed: Set[str] of label tokens
      label_defs: Dict[label -> definition text]
    """
    lines = main_prompt_text.splitlines()
    allowed: Set[str] = set()
    label_defs: Dict[str, str] = {}

    bullet_re = re.compile(r"^\s*[-*]\s+")
    head_re = re.compile(
        r"^\s*[-*]\s+([A-Z]{2,}(?:-[A-Z]+)*)\s*(?:\(([^)]*)\))?\s*(?:(?::|[–—-])\s*(.*\S))?\s*$"
    )
    label_only_re = re.compile(r"^\s*[-*]\s+([A-Z]{2,}(?:-[A-Z]+)*)\s*$")

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        m = head_re.match(line)
        if m:
            lab = m.group(1).upper()
            subtitle = (m.group(2) or "").strip()
            tail = (m.group(3) or "").strip()
            allowed.add(lab)
            definition = tail
            if subtitle:
                definition = (subtitle + (": " if tail else "") + tail).strip()
            if not definition:
                j = i + 1
                acc = []
                while j < n:
                    nxt = lines[j]
                    if not nxt.strip():
                        break
                    if bullet_re.match(nxt):
                        break
                    acc.append(nxt.strip())
                    j += 1
                if acc:
                    definition = " ".join(acc).strip()
                i = j
            else:
                i += 1
            label_defs[lab] = definition
            continue

        m2 = label_only_re.match(line)
        if m2:
            lab = m2.group(1).upper()
            allowed.add(lab)
            j = i + 1
            acc = []
            while j < n:
                nxt = lines[j]
                if not nxt.strip():
                    break
                if bullet_re.match(nxt):
                    break
                acc.append(nxt.strip())
                j += 1
            if acc:
                label_defs[lab] = " ".join(acc).strip()
            i = j
            continue

        i += 1

    return allowed, label_defs


def classify_categories(exp: Optional[str] = "gpt5-v6-13-21") -> Tuple[str, str]:
    """Load category labels/definitions from the fixed classify prompt and append a default 'N/A' label."""
    try:
        # When imported as a package module: src.benchmark.utils
        from .prompt_registry import classify_reviews_prompt  # type: ignore
    except Exception:
        try:
            # When running from repo root with src on sys.path
            from src.benchmark.prompt_registry import classify_reviews_prompt  # type: ignore
        except Exception:
            # When cwd/sys.path points at src/benchmark
            from prompt_registry import classify_reviews_prompt  # type: ignore

    sys_prompt, main_prompt_body = classify_reviews_prompt()
    allowed_categories, label_defs = extract_labels_from_prompt_text(main_prompt_body)
    allowed_categories.add("N/A")
    label_defs.setdefault("N/A", "Polite text or pure paper summary; contains no substantive technical/content point.")
    return allowed_categories, label_defs


def wait_until_ready(base_url: str,
                     api_key: str = "",
                     expect_model: Optional[str] = None,
                     timeout: int = 3600,
                     interval: float = 1.0) -> None:
    """
    Poll an OpenAI-compatible /models endpoint until the service is available.

    If `expect_model` is provided, also require that model to appear in the returned list.
    """
    url = base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    end = time.time() + timeout
    last_err = None

    while time.time() < end:
        try:
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json() if resp.content else {}
                if not expect_model:
                    return
                names = {m.get("id") or m.get("name") for m in (data.get("data") or []) if isinstance(m, dict)}
                if expect_model in names:
                    return
                last_err = f"model '{expect_model}' not in {names}"
            else:
                last_err = f"HTTP {resp.status_code}"
        except Exception as e:
            last_err = str(e)
        time.sleep(interval)

    raise RuntimeError(f"Base URL not ready: {base_url} | last_error={last_err}")


def start_service(script_path: str, args: Optional[List[str]] = None):
    """
    Start a local vLLM service via a shell script.

    Uses start_new_session=True so the service runs in its own process group,
    enabling clean termination of the entire group later.
    """
    command = ["bash", script_path]
    if args:
        command.extend(str(arg) for arg in args)
    print(" ".join(command))

    proc = subprocess.Popen(
        command,
        stdout=None,
        stderr=None,
        start_new_session=True,
    )
    return proc


def stop_service(proc, grace_sec: int = 10):
    """
    Stop a vLLM service process group.

    Sends SIGTERM first, then SIGKILL after a grace period if necessary.
    """
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        proc.wait(timeout=grace_sec)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=grace_sec)
        except Exception:
            pass


def safe_div(a: float, b: float) -> float:
    """Safe division helper (returns 0.0 when denominator is 0)."""
    return a / b if b else 0.0


def eval_binary_lists_by_majority_vote(
    pred_lists: List[List[bool]],
    names: Optional[List[str]] = None,
    gold: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Evaluate binary predictions using a soft majority-vote gold label.

    Inputs:
      - pred_lists: list of boolean prediction lists (one list per model/system)
      - gold: optional soft gold list; if not provided, derived by majority vote:
          True votes > half  => 1.0
          True votes < half  => 0.0
          tie                => 0.5

    Soft counting for gold=0.5 (tie cases):
      - pred=True  => TP += 0.5, FP += 0.5
      - pred=False => TN += 0.5, FN += 0.5

    Returns:
      {
        "gold": [0.0|0.5|1.0] * n,
        "summary": {...},
        "per_model": {name: {acc, precision, recall, f1, tp, fp, fn, tn, n}}
      }
    """
    if not pred_lists:
        return {"gold": [], "summary": {}, "per_model": {}}

    m = len(pred_lists)
    lens = [len(x) for x in pred_lists]
    n = min(lens) if lens else 0
    if gold is not None:
        n = min(n, len(gold))

    if n == 0:
        return {"gold": [], "summary": {}, "per_model": {}}

    if len(set(lens)) != 1:
        logger.warning(f"[majority_vote_eval] Prediction lengths differ: {lens}; truncating to n={n}")
    if gold is not None and len(gold) != n:
        logger.warning(f"[majority_vote_eval] Gold length is {len(gold)}; truncating to n={n}")

    if names is None or len(names) != m:
        names = [f"model_{i}" for i in range(m)]

    if gold is None:
        gold_out: List[float] = []
        for i in range(n):
            true_votes = sum(1 for k in range(m) if bool(pred_lists[k][i]))
            if true_votes * 2 > m:
                gold_out.append(1.0)
            elif true_votes * 2 < m:
                gold_out.append(0.0)
            else:
                gold_out.append(0.5)
    else:
        gold_out = list(gold[:n])

    per_model = {}
    for name, preds in zip(names, pred_lists):
        tp = tn = fp = fn = 0.0
        for i in range(n):
            g = gold_out[i]
            p = bool(preds[i])

            if g == 1.0:
                if p:
                    tp += 1.0
                else:
                    fn += 1.0
            elif g == 0.0:
                if p:
                    fp += 1.0
                else:
                    tn += 1.0
            else:
                if p:
                    tp += 0.5
                    fp += 0.5
                else:
                    tn += 0.5
                    fn += 0.5

        acc = (tp + tn) / n
        prec = safe_div(tp, tp + fp)
        rec = safe_div(tp, tp + fn)
        f1 = safe_div(2 * prec * rec, prec + rec) if (prec + rec) else 0.0

        per_model[name] = {
            "n": n,
            "acc": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "tp": float(tp), "fp": float(fp), "fn": float(fn), "tn": float(tn),
        }

    summary = {
        "n": n,
        "num_gold_true": sum(1 for g in gold_out if g == 1.0),
        "num_gold_false": sum(1 for g in gold_out if g == 0.0),
        "num_gold_tie": sum(1 for g in gold_out if g == 0.5),
    }

    return {"gold": gold_out, "summary": summary, "per_model": per_model}


def derive_bal_acc_and_mcc(tp: float, fp: float, fn: float, tn: float) -> Tuple[float, float]:
    """Compute balanced accuracy and Matthews correlation coefficient from confusion counts."""
    tpr = safe_div(tp, tp + fn)
    tnr = safe_div(tn, tn + fp)
    bal_acc = 0.5 * (tpr + tnr)
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = safe_div(tp * tn - fp * fn, denom)
    return bal_acc, mcc


def fmt(x: float) -> str:
    """Format a float metric for tabular printing."""
    return f"{x:.4f}"


def print_table(rows: List[List[Any]], headers: List[str]) -> None:
    """Print a simple aligned ASCII table for the provided rows."""
    col_widths = [len(h) for h in headers]
    for r in rows:
        for i, v in enumerate(r):
            col_widths[i] = max(col_widths[i], len(str(v)))

    line = " | ".join(headers[i].ljust(col_widths[i]) for i in range(len(headers)))
    sep = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    print(line)
    print(sep)
    for r in rows:
        print(" | ".join(str(r[i]).ljust(col_widths[i]) for i in range(len(headers))))


def _pair_comembership_counts(partition_sets, universe):
    """
    For overlapping clusterings, count pair co-membership multiplicities over `universe`.

    Returns:
      - pair2c: dict[(i, j) -> c], where c is the number of clusters in which (i, j) co-occur
      - hist: Counter, where hist[c] is the number of pairs with co-membership count == c (excluding c==0)
    """
    pair2c = {}
    hist = Counter()
    for S in partition_sets:
        T = sorted(S & universe)
        if len(T) < 2:
            continue
        for i, j in combinations(T, 2):
            key = (i, j)
            pair2c[key] = pair2c.get(key, 0) + 1
    for c in pair2c.values():
        hist[c] += 1
    return pair2c, hist


def omega_index_overlapping(part1_sets, part2_sets, universe):
    """
    Compute the Omega Index for overlapping clusterings (Collins & Dent, 1988).

    Omega = (u - e) / (1 - e)
      - u: observed agreement
      - e: expected agreement by chance (based on co-membership histograms)
    """
    n = len(universe)
    if n < 2:
        return 1.0

    P = n * (n - 1) // 2

    c1, h1_nonzero = _pair_comembership_counts(part1_sets, universe)
    c2, h2_nonzero = _pair_comembership_counts(part2_sets, universe)

    h1 = dict(h1_nonzero)
    h2 = dict(h2_nonzero)
    h1[0] = P - sum(h1_nonzero.values())
    h2[0] = P - sum(h2_nonzero.values())

    keys1 = set(c1.keys())
    keys2 = set(c2.keys())
    union_pairs = keys1 | keys2
    both_zero = P - len(union_pairs)
    equal_nonzero = sum(1 for k in (keys1 & keys2) if c1[k] == c2[k])
    u = (both_zero + equal_nonzero) / P

    all_js = set(h1.keys()) | set(h2.keys())
    e = sum(h1.get(j, 0) * h2.get(j, 0) for j in all_js) / (P * P)

    denom = (1.0 - e)
    if denom == 0.0:
        return 1.0 if u == 1.0 else 0.0
    return (u - e) / denom


class LLMClient:
    """LLM client with an on-disk cache and a background async event loop.

    Design overview:
    - Disk cache can be enabled/disabled via `cache`. Results are persisted to disk.
    - submit_task(...) schedules work on a dedicated event loop and returns a task_id.
    - get_result(task_id) blocks until completion and then reads from cache.
    - `max_workers` is the maximum number of simultaneous completion requests.
    - Requests with identical cache keys are de-duplicated while already running.

    Task bookkeeping:
    - _tasks: key -> Future or None (None means already cached/done).
    - _task_status: key -> {"status": "running"|"done"|"error", "error": Optional[str]}.
      (Errors are tracked in memory; not persisted to disk.)
    """
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        cache: bool = True,
        cache_dir: str | Path = ".llm_cache",
        max_workers: int = 32,
        cache_version: str = "v1"
    ):
        self._use_async_openai = AsyncOpenAI is not None
        client_cls = AsyncOpenAI if self._use_async_openai else OpenAI
        self.client = client_cls(api_key=api_key, base_url=base_url, timeout=3600)
        self.model_name = model_name
        self.base_url = base_url
        self.cache_version = cache_version
        self.max_workers = max(1, int(max_workers))

        # Cache configuration:
        # If the endpoint appears to be local vLLM, disable disk caching by default.
        _host = (urlparse(base_url).hostname or "").lower()
        self._is_vllm = _host in {"localhost", "127.0.0.1", "::1"}
        if self._is_vllm:
            cache = False
        self.cache = cache
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Background async loop and task bookkeeping
        self._lock = threading.Lock()
        self._tasks: Dict[str, Any] = {}
        self._task_status: Dict[str, Dict[str, Any]] = {}
        self._loop = asyncio.new_event_loop()
        self._request_semaphore = None
        self._loop_ready = threading.Event()
        self._loop_thread = threading.Thread(
            target=self._run_event_loop,
            name="llm-async-loop",
            daemon=True,
        )
        self._shutdown_started = False
        self._loop_thread.start()
        if not self._loop_ready.wait(timeout=10):
            raise RuntimeError("Failed to start the LLM async event loop.")

        # Token usage counters (aggregated across calls)
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0

    def _run_event_loop(self) -> None:
        """Own a dedicated event loop so sync callers can submit async LLM work."""
        asyncio.set_event_loop(self._loop)
        self._request_semaphore = asyncio.Semaphore(self.max_workers)
        self._loop_ready.set()

        try:
            self._loop.run_forever()
        finally:
            pending = [task for task in asyncio.all_tasks(self._loop) if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self._loop.run_until_complete(self._close_client_async())
            self._loop.close()

    async def _close_client_async(self) -> None:
        """Close the OpenAI client if the SDK exposes a close hook."""
        close_fn = getattr(self.client, "close", None)
        if close_fn is None:
            return
        try:
            close_result = close_fn()
            if inspect.isawaitable(close_result):
                await close_result
        except Exception:
            logger.exception("Failed to close LLM client cleanly.")


    @staticmethod
    def _json_dumps_canonical(obj: Any) -> str:
        """Canonical JSON serialization (sorted keys, compact separators) for stable hashing."""
        return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def _build_cache_key(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_completion_tokens: int,
        **kwargs
    ) -> str:
        """Build a cache key by hashing all request-affecting fields."""
        payload = {
            "cache_version": self.cache_version,
            "model": self.model_name,
            "messages": messages,
            "temperature": float(temperature),
            "max_completion_tokens": int(max_completion_tokens),
            **kwargs,
        }
        s = self._json_dumps_canonical(payload)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        """Return the filesystem path for a given cache key."""
        return self.cache_dir / f"{key}.json"

    def _cache_load(self, key: str) -> Optional[Tuple[Any, List[Dict[str, str]]]]:
        """Load (content, new_msg_history) from cache, or return None if missing/invalid."""
        p = self._cache_path(key)
        if not p.exists():
            return None
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(e)
            return None
        if not len(data["content"]):
            return None
        return data["content"], data["new_msg_history"]

    def _cache_store(self, key: str, content: Any, new_msg_history: List[Dict[str, str]]):
        """Atomically write a cached result (write .tmp then replace)."""
        p = self._cache_path(key)
        tmp = p.with_suffix(".json.tmp")
        payload = {
            "cache_version": self.cache_version,
            "created": time.time(),
            "content": content,
            "new_msg_history": new_msg_history,
        }
        with self._lock:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            tmp.replace(p)

    def _prepare_request(
        self,
        prompt: str,
        system_prompt: str | None = None,
        msg_history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_completion_tokens: int = MAX_NUM_TOKENS,
        **kwargs
    ) -> Tuple[str, List[Dict[str, str]], Dict[str, Any]]:
        """Prepare messages, cache key, and API payload for a chat completion request."""
        if msg_history is None:
            history = []
        else:
            history = list(msg_history)

        messages = history + [{"role": "user", "content": prompt}]
        if system_prompt is not None:
            messages = [{"role": "system", "content": system_prompt}] + messages

        hash_kwargs = kwargs
        key = self._build_cache_key(
            messages=messages, temperature=temperature, max_completion_tokens=max_completion_tokens, **hash_kwargs
        )

        api_payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
            **kwargs,
        }
        api_payload.pop("_TIME", None)
        if getattr(self, "_is_vllm", False):
            api_payload["max_tokens"] = max_completion_tokens
            api_payload.pop("max_completion_tokens", None)

            if "reasoning_effort" in api_payload:
                ctk = {"enable_thinking": api_payload["reasoning_effort"] != "none"}
                api_payload["extra_body"] = {"chat_template_kwargs": ctk}
            api_payload.pop("reasoning_effort", None)

        return key, messages, api_payload

    async def _create_completion_async(self, api_payload: Dict[str, Any]) -> Any:
        """Run one completion call while respecting the global concurrency cap."""
        async with self._request_semaphore:
            if self._use_async_openai:
                return await self.client.chat.completions.create(**api_payload)
            return await asyncio.to_thread(self.client.chat.completions.create, **api_payload)

    async def get_response(self, key, messages, api_payload) -> Tuple[Any, List[Dict[str, str]]]:
        """
        Asynchronous request execution:
        - Return cached result if available (when cache is enabled).
        - Otherwise call the API, cache the response, and return the content + updated history.

        Notes:
        - If `n > 1`, content may be a List[str]; otherwise content is a single str.
        - Only the first candidate is appended to the message history to keep it linear.
        """
        if self.cache:
            cached = self._cache_load(key)
            if cached is not None:
                return cached

        response = {}
        content: Any = ""
        for _ in range(3):
            try:
                response = await self._create_completion_async(api_payload)
                _choices = getattr(response, "choices", []) or []
                _texts = [ch.message.content for ch in _choices if getattr(ch, "message", None)]
                content = _texts[0] if len(_texts) == 1 else _texts
                break
            except Exception as e:
                print(f"[LLM Error] {type(e).__name__}: {e}")
                content = f"[LLM Error] {type(e).__name__}: {e}"
                await asyncio.sleep(5)

        # Accumulate token usage (supports both prompt/completion and input/output naming conventions)
        usage = getattr(response, "usage", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            if prompt_tokens is None:
                prompt_tokens = getattr(usage, "input_tokens", 0)
            if completion_tokens is None:
                completion_tokens = getattr(usage, "output_tokens", 0)

            try:
                pt = int(prompt_tokens or 0)
                ct = int(completion_tokens or 0)
            except Exception:
                pt, ct = 0, 0

            with self._lock:
                self._total_prompt_tokens += pt
                self._total_completion_tokens += ct

        if isinstance(content, list):
            assistant_content = content[0] if content else ""
        else:
            assistant_content = content

        final_msg_history = messages + [{"role": "assistant", "content": assistant_content}]
        self._cache_store(key, content, final_msg_history)
        if isinstance(content, str) and "[LLM Error]" in content:
            print(content, final_msg_history)

        return content, final_msg_history

    def submit_task(
        self,
        prompt: str,
        system_prompt: str | None = None,
        msg_history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_completion_tokens: int = MAX_NUM_TOKENS,
        clear_cache: bool = False,
        **kwargs
    ) -> str:
        """
        Submit a request to the async worker loop and return its task_id (cache key).

        If an identical request is already cached (and clear_cache is False), returns immediately.
        """
        if clear_cache:
            kwargs["_TIME"] = time.time_ns()
        key, messages, api_payload = self._prepare_request(
            prompt=prompt,
            system_prompt=system_prompt,
            msg_history=msg_history,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            **kwargs,
        )

        with self._lock:
            if not clear_cache and self._cache_load(key) is not None:
                self._tasks[key] = None
                self._task_status[key] = {"status": "done", "error": None}
                return key

            existing_fut = self._tasks.get(key)
            existing_status = self._task_status.get(key, {})
            if not clear_cache and existing_fut is not None and existing_status.get("status") == "running":
                return key

            fut = asyncio.run_coroutine_threadsafe(
                self._worker_run_call_get_response(key, messages, api_payload),
                self._loop,
            )
            self._tasks[key] = fut
            self._task_status[key] = {"status": "running", "error": None}
            return key

    def get_result(self, key: str, clear_cache: bool = False) -> Dict[str, Any]:
        """
        Block until a submitted task completes, then return its cached result (if available).

        Returns:
          {
            'status': 'done' | <error string> | None,
            'key': <cache_key>,
            'content': <str | List[str] | None>,
            'msg_history': <list | None>
          }
        """
        with self._lock:
            fut = self._tasks.get(key)

        if fut is not None:
            try:
                fut.result()
            except Exception:
                logger.exception("Unexpected exception bubbling from worker for key=%s", key)

        cached = self._cache_load(key)
        if cached is not None:
            content, msg_history = cached
            with self._lock:
                self._task_status[key] = {"status": "done", "error": None}
                if clear_cache:
                    cache_path = self._cache_path(key)
                    if cache_path.exists():
                        os.remove(cache_path)
            return {
                "status": "done",
                "key": key,
                "content": content,
                "msg_history": msg_history,
            }

        with self._lock:
            status_entry = self._task_status.get(key)
        error_msg = status_entry.get("error") if status_entry else "missing request keys"
        logger.error("Failed for key=%s: %s", key, error_msg)

        return {
            "status": error_msg,
            "key": key,
            "content": None,
            "msg_history": None,
        }

    async def _worker_run_call_get_response(self, key, messages, api_payload, **kwargs):
        """
        Worker routine with retries.

        - Checks cache before calling the API.
        - Uses get_response(...) to populate cache on cache miss.
        - Retries on exceptions; records a final error in memory after exhausting retries.
        """
        last_err: Optional[BaseException] = None

        for i in range(5):
            try:
                cached = self._cache_load(key)
                if cached is None:
                    await self.get_response(key, messages, api_payload)

                with self._lock:
                    self._task_status[key] = {"status": "done", "error": None}
                return
            except asyncio.CancelledError:
                msg = f"Task cancelled for key={key}"
                logger.warning(msg)
                with self._lock:
                    self._task_status[key] = {"status": "error", "error": msg}
                raise
            except Exception as e:
                last_err = e
                logger.exception("Worker attempt %d failed for key=%s", i + 1, key)
                await asyncio.sleep(10)

        msg = f"All retries failed for key={key}: {repr(last_err)}"
        logger.error(msg)
        with self._lock:
            self._task_status[key] = {"status": "error", "error": msg}

    def shutdown(self, wait: bool = True):
        """Shut down the async loop and log aggregated token usage."""
        with self._lock:
            total_prompt = self._total_prompt_tokens
            total_completion = self._total_completion_tokens
            total = total_prompt + total_completion

        logger.info(
            "LLM token usage — prompt: %d, completion: %d, total: %d",
            total_prompt,
            total_completion,
            total,
        )
        print(f"[LLM usage] prompt={total_prompt}, completion={total_completion}, total={total}")

        if wait:
            with self._lock:
                pending_futures = [fut for fut in self._tasks.values() if fut is not None]
            for fut in pending_futures:
                try:
                    fut.result()
                except Exception:
                    logger.exception("Unexpected exception while waiting for LLM task shutdown.")

        with self._lock:
            if self._shutdown_started:
                return
            self._shutdown_started = True

        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join()

def merge_similar_reviewer_threads(final_result, sentence_texts, threshold=0.5):
    """
    Merge highly similar reviewer threads based on sentence-id overlap.

    For each item in `final_result`, collect the set of sentence IDs that appear under roles
    containing "Reviewer". For two items A and B, define similarity as:

        sim = |A ∩ B| / min(|A|, |B|)

    This implementation uses a length-weighted proxy where overlap and set sizes are measured
    in total character length of the referenced sentences, and then applies:
      - Merge if sim >= threshold
      - Also merge if one set is a subset of the other (A ⊆ B or B ⊆ A)
      - Uses a union-find structure to build clusters

    Merge strategy (within each cluster):
      - Merge per-role sentence IDs by set union, then sort IDs.
      - For `classify`, flatten across cluster members, deduplicate, and sort.
    """
    n = len(final_result)
    if n <= 1:
        return final_result

    reviewer_sets = []
    for idx, (thread, classify) in enumerate(final_result):
        r_ids = set()
        for role, sid_list in thread:
            if "Reviewer" in role:
                r_ids.update(sid_list)
        reviewer_sets.append(r_ids)

    parent = list(range(n))
    to_remove = set()

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        A = reviewer_sets[i]
        if not A:
            continue
        for j in range(i + 1, n):
            B = reviewer_sets[j]
            if not B:
                continue

            inter = len(A & B)
            inter_len = sum([len(sentence_texts[idx]) for idx in A & B])
            if inter == 0:
                continue

            denom = min(len(A), len(B))
            denom_len = min(sum([len(sentence_texts[idx]) for idx in A]), sum([len(sentence_texts[idx]) for idx in B]))
            if denom == 0:
                continue

            sim = inter_len / denom_len
            if sim >= threshold or A <= B or B <= A:
                if A != B and not ((A <= B or B <= A) and set(final_result[i][1]) == set(final_result[j][1])):
                    if set(final_result[i][1]) <= set(final_result[j][1]) or set(final_result[j][1]) <= set(final_result[i][1]):
                        union(i, j)
                    else:
                        if A <= B and A != B:
                            to_remove.add(i)
                        elif B <= A and A != B:
                            to_remove.add(j)
                        else:
                            sen_A = " ".join([sentence_texts[idx] for idx in A])
                            sen_B = " ".join([sentence_texts[idx] for idx in B])
                            pass
                else:
                    union(i, j)

    clusters = {}
    for i in range(n):
        if i in to_remove:
            continue
        root = find(i)
        clusters.setdefault(root, []).append(i)

    merged_final = []

    for root, idxs in sorted(clusters.items()):
        if len(idxs) == 1:
            merged_final.append(final_result[idxs[0]])
            continue

        role2ids = {}
        for idx in idxs:
            thread, _ = final_result[idx]
            for role, sid_list in thread:
                if role not in role2ids:
                    role2ids[role] = set()
                role2ids[role].update(sid_list)

        merged_thread = []
        for role, id_set in role2ids.items():
            if not id_set:
                continue
            merged_thread.append([role, sorted(id_set)])

        flat = []
        for c in [final_result[idx][1] for idx in idxs]:
            flat.extend(list(c))
        merged_classify = sorted(set(flat))
        if len(merged_classify) == 1 and A != B:
            pass

        merged_final.append([merged_thread, merged_classify])

    return merged_final
