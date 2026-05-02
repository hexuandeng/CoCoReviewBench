#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Purpose
-------
Normalize OpenReview JSONL review threads into a compact schema with reviewer
and metareview blocks.

Inputs
------
- JSONL file at <org>.cc_<year>/<org>.cc_<year>_PDF.jsonl with a "replies" field.

Outputs
-------
- JSONL file at <org>.cc_<year>/<org>.cc_<year>_<postfix>.jsonl.

Notes
-----
- Reviewer main objects are created only when an overall score is present.
- Threads whose first role is "Unknown" are skipped.
"""

from typing import Any, Dict, Optional, List, Tuple
from copy import deepcopy
import re
import argparse
import json
import os
from collections import defaultdict

COUNTS = defaultdict(int)

# ---------------------------
# Utilities
# ---------------------------

def get_value(data: Dict[str, Any], key: str, default=None):
    """
    Safely retrieve a value from a possibly nested dict entry.

    Many scraped sources store actual values under a 'value' sub-key, e.g.:
        data['rating'] = "4: Good"
        data['rating'] = {"value": "4: Good"}

    This helper returns:
      - data[key]['value'] if data[key] is a dict with a 'value' key
      - data[key] directly otherwise
      - default if key is missing or data is not a dict
    """
    if not isinstance(data, dict):
        return default
    item = data.get(key)
    if item is None:
        return default
    if isinstance(item, dict) and 'value' in item:
        return item['value']
    return item

def get_first_value(data: Dict[str, Any], keys: List[str], default=None):
    """
    Try keys in order and return the first non-None value via get_value().

    Added invariant (assertion):
      - At most one key in `keys` may yield a non-None value. If multiple keys
        produce non-None values, an AssertionError is raised. This protects
        against ambiguous schemas where multiple aliases accidentally coexist.

    Args:
        data: Mapping containing potentially nested fields.
        keys: Candidate key names, in priority order.
        default: Value to return if no key yields a value.

    Returns:
        The first found non-None value, or `default` if none.
    """
    found_key = None
    found_value = None

    for k in keys:
        v = get_value(data, k, None)
        if v is not None:
            if found_key is None:
                found_key, found_value = k, v
            else:
                raise AssertionError(
                    f"get_first_value: multiple keys matched non-None values "
                    f"(first='{found_key}', second='{k}')."
                )

    return found_value if found_key is not None else default

_SCORE_RE = re.compile(r'^\s*([0-9]+(?:\.[0-9]+)?)')

def parse_numeric_score(value) -> Optional[float]:
    """
    Parse a numeric score from heterogeneous inputs.

    Accepts examples like:
      - "4: Good", "4 - Good"      -> 4.0
      - "8/10"                      -> 8.0
      - "3.5"                       -> 3.5
      - 5                           -> 5.0
      - {"value": "4: ..."}         -> 4.0 (by upstream get_value usage)

    Returns:
        float value if successfully parsed, else None.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value)
    # Normalize common ratio-like inputs, e.g., "8/10" -> take head "8"
    if '/' in s:
        head = s.split('/', 1)[0].strip()
        try:
            return float(head)
        except ValueError:
            pass
    m = _SCORE_RE.match(s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None

def parse_novelty(review_data: Dict[str, Any]) -> Optional[float]:
    """
    Compute the novelty score as a float if possible.

    Strategy:
      1) Average 'technical_novelty_and_significance' and
         'empirical_novelty_and_significance' if both parse.
      2) Otherwise, fall back to parsing 'contribution'.

    Returns:
        float or None.
    """
    tech_novelty = get_value(review_data, 'technical_novelty_and_significance')
    emp_novelty = get_value(review_data, 'empirical_novelty_and_significance')

    t = parse_numeric_score(tech_novelty)
    e = parse_numeric_score(emp_novelty)
    if t is not None and e is not None:
        return (t + e) / 2.0
    # Fallback: sometimes novelty is captured under 'contribution'
    return parse_numeric_score(get_value(review_data, 'contribution'))

def select_by_keys(keys_to_keep: list, source_dict: dict, to_str: bool = False) -> dict:
    """
    Select a subset of entries from `source_dict` constrained by `keys_to_keep`.

    The function preserves original values by default; set `to_str=True` to
    convert each selected value to a flattened string (lists/tuples joined,
    dicts unwrapped via 'value', None -> "").
    """
    keys_set = set(keys_to_keep)
    dct = {key: value for key, value in source_dict.items() if key in keys_set}
    if to_str:
        for k, v in list(dct.items()):
            dct[k] = _flatten_to_str(v)
    return dct

def _flatten_to_str(x) -> str:
    """
    Convert nested/iterable values into a displayable single string.

    Rules:
      - None -> ""
      - dict with 'value' -> unwrap to that value
      - list/tuple -> join recursively flattened elements with a space
      - other -> str(x)
    """
    if x is None:
        return ""
    if isinstance(x, dict) and 'value' in x:
        x = x['value']
    if isinstance(x, (list, tuple)):
        return " ".join(_flatten_to_str(e) for e in x)
    assert isinstance(x, str)
    return x

def extract_comments(thread: list) -> list:
    """
    Extract subsequent comments from a thread (excluding the first post).

    A thread is a list of (role, content_dict) tuples.
    This function returns a list of (role, {'comment':..., 'rebuttal':...}) pairs
    for each remaining post, preserving only the 'comment' and 'rebuttal' fields.
    """
    comments = []
    for i in range(len(thread)):
        comment_role = thread[i][0]
        comment_content = select_by_keys(['comment', 'rebuttal', 'question'], thread[i][1])
        for k, v in comment_content.items():
            comment_content[k] = _flatten_to_str(v)
        comment_obj = (comment_role, {"value": comment_content})
        comments.append(comment_obj)
    return comments

# ---------------------------
# Core transformation
# ---------------------------

def transform_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform a single JSON object representing a paper and its review threads.

    Minimal-change rules (as requested):
      - If a reviewer post does not include an overall score (from
        'recommendation' or 'rating'), do NOT construct the reviewer main
        object; only preserve follow-up comments in that thread.
      - If the first post role is "Unknown", skip this thread entirely.
      - Metareview (PC/AC) field names are preserved exactly (no renaming).
      - All scores are consistently parsed as float or None.
      - Ethics fields are concatenated into a single string (flag + details).

    Output schema (added fields on top of input):
      - "reviews":    list of per-thread lists; each contains a main reviewer
                      tuple plus subsequent comments (or just comments).
      - "metareview": list of per-thread lists for PC/AC posts with fields and
                      their subsequent comments.

    Returns:
        A deep-copied and augmented object with "reviews" and "metareview".
    """
    COUNTS[data["PDF_version_correct"]] += 1
    output = deepcopy(data)
    output["reviews"] = []
    output["metareview"] = []

    # Iterate over all conversation threads under "replies"
    for thread in data.get("replies", []):
        cnt_data: List[Tuple[str, Dict[str, Any]]] = []
        metareview_cnt_data: List[Tuple[str, Dict[str, Any]]] = []

        if not thread:
            continue

        first_post = thread[0]
        role = first_post[0]
        content = first_post[1]

        # Skip threads whose leading post role is unknown
        if role == "Unknown":
            continue

        # Handle PC/AC (area chair / program committee) threads as metareview
        if role in ["PC", "AC"]:
            metareview_keys = [
                'metareview',
                'metareview:_summary,_strengths_and_weaknesses',
                'comment',
                'desk_reject_comments',
                'additional_comments_on_reviewer_discussion',
                'justification_for_why_not_higher_score',  # ICLR 2024 23
                'justification_for_why_not_lower_score',   # ICLR 2024 23
                'summary_of_AC-reviewer_meeting'           # ICLR 2023
            ]
            metareview_obj = (
                role,
                select_by_keys(metareview_keys, content)
            )
            metareview_cnt_data.append(metareview_obj)

            # Include subsequent author/reviewer comments in the same thread
            rest = thread[1: ]
            cmts = extract_comments(rest)
            if cmts:
                metareview_cnt_data += cmts

            if metareview_cnt_data:
                output["metareview"].append(deepcopy(metareview_cnt_data))
            continue

        # Handle main reviewer post (+ its follow-up comments)
        elif role == "Reviewer":
            # Presence of an overall score gates creation of the reviewer object
            overall_score = parse_numeric_score(
                get_first_value(content, ['recommendation', 'rating'])
            )

            if overall_score is not None:
                reviewer_keys = [
                    'strengths',
                    'weaknesses',
                    'limitations',
                    'strength_and_weaknesses',
                    'strengths_and_weaknesses',
                    'clarity,_quality,_novelty_and_reproducibility',
                    'main_review',
                    'review',
                    # 'summary_of_the_review',
                    'question',
                    'questions',
                    # 'code_of_conduct',
                    'justification_for_why_not_higher_score',  # ICLR 2024
                    'justification_for_why_not_lower_score',   # ICLR 2024
                    'desk_reject_comments',                     # keep DR too
                ]
                values = select_by_keys(reviewer_keys, content)

                # Optional concise paper summary
                summary_of_paper = get_first_value(content, ['summary_of_the_paper', 'summary'])

                # Ethics fields are flattened/concatenated into one string
                ethics_flag = get_first_value(
                    content,
                    ['flag_for_ethics_review', 'needs_ethics_review', 'ethics_flag', 'ethical_issues']  # NIPS 2021
                )
                ethics_review = _flatten_to_str(ethics_flag)
                ethics_detail = get_first_value(content, ['details_of_ethics_concerns', 'ethics_review'])
                if ethics_detail is not None:
                    ethics_review = (ethics_review + ("\n" if ethics_review else "")) + _flatten_to_str(ethics_detail)
                
                if len(ethics_review.strip()):
                    values["needs_ethics_review"] = ethics_review 
                for k, v in values.items():
                    values[k] = _flatten_to_str(v)

                review_obj = (
                    "Reviewer",
                    {
                        "summary_of_the_paper": summary_of_paper,
                        "value": values,
                        "scores": {
                            "Solid": parse_numeric_score(get_first_value(content, ['correctness', 'soundness'])),
                            "Presentation": parse_numeric_score(get_value(content, 'presentation')),
                            "Novelty": parse_novelty(content),
                            "Overall": overall_score,
                            "Confidence": parse_numeric_score(get_value(content, 'confidence')),
                        }
                    }
                )
                cnt_data.append(review_obj)
                thread = thread[1: ]  # consume the first reviewer post

        # Always append subsequent comments for reviewer threads
        comments = extract_comments(thread)
        if comments:
            cnt_data += comments

        if cnt_data:
            output["reviews"].append(deepcopy(cnt_data))
    if "replies" in output:
        del output["replies"]
        
    return output

# ---------------------------
# IO
# ---------------------------

def process_file(input_path: str, output_path: str):
    """
    Stream-process a JSONL file:
      - Read each line as a JSON object
      - Transform via `transform_json`
      - Write one transformed object per line to `output_path`

    Lines that are not valid JSON are skipped with a console notice.
    """
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        for line in infile:
            try:
                data = json.loads(line)
                processed_data = transform_json(data)
                outfile.write(json.dumps(processed_data, ensure_ascii=False) + '\n')
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line in {input_path}")
                continue

# ---------------------------
# CLI
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract and format review data from a JSONL file."
    )
    parser.add_argument("--org", type=str, default="ICLR", help="Conference org prefix, e.g., NeurIPS")
    parser.add_argument("--year", type=str, default="2017", help="Year, e.g., 2021")
    parser.add_argument("--postfix", type=str, default="clean", help="Output filename postfix, e.g., clean")

    args = parser.parse_args()

    folder = f"{args.org}.cc_{args.year}"
    base = f"{args.org}.cc_{args.year}"
    input_path = os.path.join(folder, f"{base}_PDF.jsonl")
    output_path = os.path.join(folder, f"{base}_{args.postfix}.jsonl")

    if not os.path.exists(input_path):
        print(f"Error: Input file not found at '{input_path}'")
    else:
        os.makedirs(folder, exist_ok=True)
        process_file(input_path, output_path)
        print(f"Processing complete!\n  Input : {input_path}\n  Output: {output_path}")
    print(COUNTS)
