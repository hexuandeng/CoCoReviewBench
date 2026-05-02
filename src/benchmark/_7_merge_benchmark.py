"""
Purpose
-------
Format and merge multi-model outputs for:
1) reviewer-opinion refutation validation, and
2) conflict-resolution adjudication,
then assemble a final JSONL dataset with per-block validation labels.

Pipeline
--------
1) Refutation validation formatting
   - Read per-model validation JSONL.
   - Extract meta-review text safely.
   - Normalize block_id variants for cross-model matching.
   - Merge results per (record_id, reviewer_name) and compute simple consensus stats.
   - Write a JSON summary plus a human-readable TXT excerpt.

2) Conflict resolution formatting
   - Read per-model conflict-resolution JSONL.
   - Extract meta-review text safely.
   - Merge results per (record_id, label, point_id) and compute per-block vote counts.
   - Write a JSON summary plus a human-readable TXT excerpt.

3) Final dataset assembly
   - Start from split_final JSONL (per-paper blocks).
   - Attach grouping info from assign_point_ids.json.
   - Attach:
     - conflicts_validation: per-block labels from conflict resolutions
     - rebuttal_validation: per-block labels from refutation validations
   - Write a final JSONL suitable for downstream analysis/visualization.

Output artifacts
----------------
- refutation_validation_results_for_review.json (+ _summary.txt)
- conflict_resolutions_for_review.json (+ _summary.txt)
- *_with_reviewer_validations.jsonl
"""
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple, Optional

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import GROUPS

# ===== Loose reference line matcher (no bracket required) =====

PREFIX_RE = re.compile(
    r"^\s*(?:\[(?:[0-9]+|[A-Za-z])\]\s*)?"
    r"(?:\(?\d{1,3}\)?[.)]\s*)?"
    r"(?:[-*•\u2022]\s*)?"
)

YEAR4_RE = re.compile(r"\b(?:19|20)\d{2}[a-z]?\b")  # 2024 / 2024a
ARXIV_ID_RE = re.compile(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b")
CONFYEAR_RE = re.compile(
    r"\b(?:ICLR|NeurIPS|NIPS|ICML|CVPR|ECCV|ICCV|AAAI|IJCAI|ACL|EMNLP|NAACL|"
    r"WACV|TMLR|UAI|AISTATS|KDD|SIGIR|OSDI|USENIX|PKC|SaTML|MLSys|ICRA|ICME|ICC)"
    r"\s*['’]?\s*\d{2,4}\b",
    re.IGNORECASE,
)

QUOTED_TITLE_RE = re.compile(r"[\"“][^\"”]{6,}[\"”]")
VENUE_KW_RE = re.compile(
    r"\b(arxiv|preprint|proceedings|proc\.|conference|workshop|journal|transactions|"
    r"foundations and trends|springer|pmlr|ieee|acm|neurips|nips|iclr|icml|cvpr|eccv|iccv|"
    r"aaai|ijcai|acl|emnlp|naacl|wacv|tmlr|uai|aistats|kdd|sigir|icra|icme|icc)\b",
    re.IGNORECASE,
)

NAME_CH = r"A-Za-z\u00C0-\u024F"

AUTHOR_TOKEN = rf"""
(?: 
    # Last, First / Last, F. / Last, T.-Y.
    [A-Z][{NAME_CH}'’\-]{{1,40}}
    (?:\s+(?:de|da|van|von|der|di|la|le|[A-Z][{NAME_CH}'’\-]{{1,40}})){{0,3}}
    ,\s*
    (?:[A-Z][{NAME_CH}'’\-]{{1,40}}|(?:[A-Z](?:\.-[A-Z])?\.){{1,3}})
    (?:\s+(?:[A-Z][{NAME_CH}'’\-]{{1,40}}|(?:[A-Z](?:\.-[A-Z])?\.){{1,3}}))* 
  |
    # First Last / First M. Last / Michael I. Jordan
    [A-Z][{NAME_CH}'’\-]{{1,40}}
    (?:\s+(?:[A-Z](?:\.-[A-Z])?\.){{1,2}})?
    \s+[A-Z][{NAME_CH}'’\-]{{1,40}}
    (?:\s+[A-Z][{NAME_CH}'’\-]{{1,40}}){{0,2}}
  |
    # Initials. Last
    (?:[A-Z](?:\.-[A-Z])?\.\s*){{1,3}}[A-Z][{NAME_CH}'’\-]{{1,40}}
  |
    # Last + Initial (Li Y)
    [A-Z][{NAME_CH}'’\-]{{1,40}}\s+[A-Z](?:\.)?
)
"""

AUTHOR_PREFIX_RE = re.compile(
    rf"""
    (?<![{NAME_CH}])
    {AUTHOR_TOKEN}
    (?:\s*,\s*{AUTHOR_TOKEN})*
    (?:\s*,?\s*(?:and|&)\s*{AUTHOR_TOKEN})?
    (?:\s*,?\s*et\s+al\.?)?
    """,
    re.VERBOSE,
)

BAD_LINE_RE = re.compile(r"^\s*(?:n/?a|N/?A|yes|no)\s*$")

URL_ONLY_RE = re.compile(
    r"^(?:https?://|www\.)[A-Za-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+$",
    re.IGNORECASE,
)

DOI_ONLY_RE = re.compile(
    r"^10\.\d{4,9}/[A-Za-z0-9\-._;()/:]+$",
    re.IGNORECASE,
)

URL_TRAIL_PUNCT = ".,;:)]}>\"'"

SEP_CHARS_RE = re.compile("[\\s\\.,;:()\\[\\]{}\\-\\u2013\\u2014/\\\\&]+")

BRACKET_LABEL_RE = re.compile(r"^\s*\[(?P<label>[0-9]+|[A-Za-z])\]\s*")
HAS_ALNUM_RE = re.compile(r"[A-Za-z0-9]")
REVIEWER_ROLE_RE = re.compile(r"\breviewer\b", re.IGNORECASE)


def leftover_ok(s: str, max_leftover_chars: int = 12) -> bool:
    """Ensure leftover text outside core spans is short."""
    x = SEP_CHARS_RE.sub("", (s or "")).strip()
    return len(x) <= max_leftover_chars


def is_reference_line_loose(line: str) -> bool:
    """Return True if a line looks like a reference entry under loose heuristics."""
    s = (line or "").strip()
    if not s or BAD_LINE_RE.match(s):
        return False

    rest = PREFIX_RE.sub("", s).strip()

    cand = rest.strip().rstrip(URL_TRAIL_PUNCT)

    if cand and (URL_ONLY_RE.fullmatch(cand) or DOI_ONLY_RE.fullmatch(cand)):
        return True

    has_yearish = bool(YEAR4_RE.search(rest) or CONFYEAR_RE.search(rest) or ARXIV_ID_RE.search(rest))
    if not has_yearish and ("arxiv" not in rest.lower()):
        return False

    # ---- span-based core blocks (author/title/venue) ----
    pos0 = 0
    spans = []  # (name, start, end)

    m_author0 = AUTHOR_PREFIX_RE.match(rest)
    if m_author0:
        dot = rest.find(".")
        if 0 <= dot <= 220:
            spans.append(("author", 0, dot + 1))
            pos0 = dot + 1
        else:
            return False

    m_title = QUOTED_TITLE_RE.search(rest, pos0)
    if m_title:
        spans.append(("title", m_title.start(), m_title.end()))
        pos0 = m_title.end()
        if QUOTED_TITLE_RE.search(rest, pos0):
            return False

    m_venue = None
    for pat in (CONFYEAR_RE, VENUE_KW_RE):
        m = pat.search(rest, pos0)
        if m and (m_venue is None or m.start() < m_venue.start()):
            m_venue = m

    if m_venue is None:
        idx = rest.lower().find("arxiv", pos0)
        if idx != -1:
            class _M:
                def __init__(self, s): self._s = s
                def start(self): return self._s
            m_venue = _M(idx)

    if m_venue:
        vstart = m_venue.start()
        spans.append(("venue", vstart, len(rest)))
        pos0 = len(rest)

    if not spans:
        return False

    spans_sorted = sorted(spans, key=lambda x: x[1])
    if spans_sorted != spans:
        return False
    for i in range(len(spans) - 1):
        if spans[i][2] > spans[i + 1][1]:
            return False

    covered = [""] * len(rest)
    for _, a, b in spans:
        for k in range(a, b):
            covered[k] = "X"
    left = "".join(ch for ch, mk in zip(rest, covered) if mk != "X")
    if not leftover_ok(left, max_leftover_chars=100):
        return False

    # Require at least two core signals (author/title/venue).
    score = 0
    if m_author0:
        score += 1
    if m_title:
        score += 1
    if m_venue:
        score += 1
    return score >= 2


def has_scores(sentence_ids: Any) -> bool:
    """Return True when a review payload contains a non-null scores field."""
    return isinstance(sentence_ids, dict) and ("scores" in sentence_ids) and (sentence_ids["scores"] is not None)


def normalize_reviewer_roles(obj: Dict[str, Any]) -> None:
    """
    Normalize reviewer role labels in reviews and propagate them to opinions.
    - Scored reviewers are numbered in appearance order.
    - Unscored reviewers are labeled "Reviewer".
    - Follow-up replies after a scored review are labeled "Reviewer k Further Reply".
    """
    reviews = obj.get("reviews")
    if not isinstance(reviews, list):
        opinions = obj.get("opinions")
        if isinstance(opinions, list):
            for op in opinions:
                if isinstance(op, dict):
                    data = op.get("data")
                    if isinstance(data, list):
                        for turn in data:
                            if isinstance(turn, dict) and is_reviewer_role(turn.get("role")):
                                turn["role"] = "Reviewer"
        return

    sid_to_role: Dict[int, str] = {}
    reviewer_no = 0

    for thread in reviews:
        if not isinstance(thread, list):
            continue

        thread_no: Optional[int] = None
        for turn in thread:
            if isinstance(turn, dict) and is_reviewer_role(turn.get("role")) and has_scores(turn.get("sentence_ids")):
                reviewer_no += 1
                thread_no = reviewer_no
                break

        seen_scored = False
        for turn in thread:
            if not isinstance(turn, dict):
                continue
            if not is_reviewer_role(turn.get("role")):
                continue

            if thread_no is None:
                label = "Reviewer"
            else:
                if has_scores(turn.get("sentence_ids")):
                    label = f"Reviewer {thread_no}"
                    seen_scored = True
                elif seen_scored:
                    label = f"Reviewer {thread_no} Further Reply"
                else:
                    label = "Reviewer"

            turn["role"] = label

            sids: List[int] = []
            extract_ints(turn.get("sentence_ids"), sids)
            for sid in sids:
                sid_to_role[sid] = label

    opinions = obj.get("opinions")
    if not isinstance(opinions, list):
        return

    for op in opinions:
        if isinstance(op, dict):
            data = op.get("data")
            if isinstance(data, list):
                for turn in data:
                    if isinstance(turn, dict) and is_reviewer_role(turn.get("role")):
                        sids: List[int] = []
                        extract_ints(turn.get("sentence_ids"), sids)
                        label = None
                        for sid in sids:
                            if sid in sid_to_role:
                                label = sid_to_role[sid]
                                break
                        turn["role"] = label or "Reviewer"
            continue

        if isinstance(op, list) and len(op) >= 1 and isinstance(op[0], list):
            for turn in op[0]:
                if isinstance(turn, dict) and is_reviewer_role(turn.get("role")):
                    sids: List[int] = []
                    extract_ints(turn.get("sentence_ids"), sids)
                    label = None
                    for sid in sids:
                        if sid in sid_to_role:
                            label = sid_to_role[sid]
                            break
                    turn["role"] = label or "Reviewer"
                elif isinstance(turn, (list, tuple)) and len(turn) == 2:
                    role, sids_any = turn[0], turn[1]
                    if is_reviewer_role(role):
                        sids: List[int] = []
                        extract_ints(sids_any, sids)
                        label = None
                        for sid in sids:
                            if sid in sid_to_role:
                                label = sid_to_role[sid]
                                break
                        turn[0] = label or "Reviewer"


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON parse error at {path}:{line_no}: {e}") from e
            if not isinstance(obj, dict):
                raise TypeError(f"Each JSONL line must be a dict, got {type(obj)} at {path}:{line_no}")
            yield obj


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def extract_ints(x: Any, out: List[int]) -> None:
    """Collect all integers from nested lists/dicts into out."""
    if isinstance(x, int):
        out.append(x)
    elif isinstance(x, list):
        for it in x:
            extract_ints(it, out)
    elif isinstance(x, dict):
        for v in x.values():
            extract_ints(v, out)


def is_increment(prev_label: str, next_label: str) -> bool:
    """Return True if labels increment (1->2 or a->b, case-insensitive)."""
    if prev_label.isdigit() and next_label.isdigit():
        return int(next_label) == int(prev_label) + 1
    if prev_label.isalpha() and next_label.isalpha() and len(prev_label) == 1 and len(next_label) == 1:
        return ord(next_label.lower()) == ord(prev_label.lower()) + 1
    return False


def is_filler_line(s: str) -> bool:
    """Return True for blank lines or lines without any alphanumerics."""
    s = (s or "").strip()
    if not s:
        return True
    return HAS_ALNUM_RE.search(s) is None


def find_reference_blocks(sentence_texts: List[str], min_labeled: int = 3) -> List[List[int]]:
    """
    Find contiguous reference blocks.
    - Start with [1] or [a].
    - Require strictly increasing labels.
    - Allow filler lines, plus at most one non-increment line if the next
      non-filler line resumes the sequence.
    - Require at least min_labeled labeled lines.
    """
    blocks: List[List[int]] = []
    n = len(sentence_texts)
    i = 0

    while i < n:
        rest = PREFIX_RE.sub("", sentence_texts[i]).strip()
        cand = rest.strip().rstrip(URL_TRAIL_PUNCT)
        if cand and (URL_ONLY_RE.fullmatch(cand) or DOI_ONLY_RE.fullmatch(cand)):
            blocks.append([i])
            i += 1
            continue

        m = BRACKET_LABEL_RE.match(sentence_texts[i] or "")
        if not m:
            i += 1
            continue

        start_label = m.group("label")
        if not (start_label == "1" or start_label.lower() == "a"):
            i += 1
            continue

        run = [i]
        labeled_count = 1
        prev_label = start_label
        j = i + 1

        while j < n:
            line = sentence_texts[j] or ""

            if is_filler_line(line):
                run.append(j)
                j += 1
                continue

            m2 = BRACKET_LABEL_RE.match(line)

            if not m2:
                k = j + 1
                while k < n and is_filler_line(sentence_texts[k] or ""):
                    k += 1
                if k < n:
                    m3 = BRACKET_LABEL_RE.match(sentence_texts[k] or "")
                    if m3 and is_increment(prev_label, m3.group("label")):
                        run.append(j)
                        j += 1
                        continue
                break

            lab2 = m2.group("label")
            if not is_increment(prev_label, lab2):
                k = j + 1
                while k < n and is_filler_line(sentence_texts[k] or ""):
                    k += 1
                if k < n:
                    m3 = BRACKET_LABEL_RE.match(sentence_texts[k] or "")
                    if m3 and is_increment(prev_label, m3.group("label")):
                        run.append(j)
                        j += 1
                        continue
                break

            run.append(j)
            labeled_count += 1
            prev_label = lab2
            j += 1

        if labeled_count >= min_labeled:
            blocks.append(run)
            i = j
        else:
            i += 1

    return blocks


SUPP_NUM_LEAD_RE = re.compile(r"^\s*\(?\d{1,3}\)?[.)]?\s+")
SUPP_NUM_PARSE_RE = re.compile(r"^\s*\(?(\d{1,3})\)?")
SUPP_STAR_LEAD_RE = re.compile(r"^\s*\*+")
SUPP_REF_HEADER_CHARS_RE = re.compile(r"[^A-Za-z]")

def _supp_is_blank_line(s: str) -> bool:
    return (s or "").strip() == ""

def _supp_is_url_or_doi_only(line: str) -> bool:
    rest = PREFIX_RE.sub("", (line or "")).strip()
    cand = rest.strip().rstrip(URL_TRAIL_PUNCT)
    return bool(cand and (URL_ONLY_RE.fullmatch(cand) or DOI_ONLY_RE.fullmatch(cand)))

def _supp_is_num_lead(line: str) -> bool:
    return bool(SUPP_NUM_LEAD_RE.match(line or ""))

def _supp_first_num_is_one(span_lines: List[str]) -> bool:
    for ln in span_lines:
        if _supp_is_blank_line(ln):
            continue
        m = SUPP_NUM_PARSE_RE.match(ln or "")
        if not m:
            return False
        return int(m.group(1)) == 1
    return False

def _supp_is_reference_header(line: str) -> bool:
    """
    Detect a "Reference(s)" header line.
    Allows surrounding whitespace and punctuation.
    """
    t = SUPP_REF_HEADER_CHARS_RE.sub("", (line or "")).lower()
    return t in ("reference", "references")

def supplement_reference_blocks(
    sentence_texts: List[str],
    blocks: List[List[int]]
) -> List[List[int]]:
    """
    Add supplemental reference blocks based on loose reference heuristics.
    Rules:
    1) Numeric-leading sequences that start at 1 and are majority reference-like.
    2) Bracket/bullet sequences with majority reference-like lines.
    3) "Reference(s)" header followed by contiguous reference-like lines, with
       single-line gaps allowed if both neighbors are references.
    4) Other contiguous runs of reference-like lines (length >= 2).
    """
    n = len(sentence_texts)
    existing: Set[int] = set(i for b in blocks for i in b)
    occupied: Set[int] = set(existing)
    new_blocks: List[List[int]] = []
    BRACKETISH_RE = re.compile(r"^\s*[^A-Za-z0-9]*\[")

    def is_bracketish_line(ln: str) -> bool:
        return bool(BRACKETISH_RE.match(ln or ""))

    def is_bullet_refish_line(ln: str) -> bool:
        t = (ln or "")
        s = t.lstrip()
        if not s:
            return False
        if s[0] not in ("*", "-"):
            return False
        return _supp_is_url_or_doi_only(t) or is_reference_line_loose(t)

    def in_bracket_seq(ln: str) -> bool:
        return _supp_is_blank_line(ln) or is_bracketish_line(ln) or is_bullet_refish_line(ln)

    def add_block(idxs: List[int]) -> None:
        nonlocal new_blocks, occupied
        if not idxs:
            return
        newly = [i for i in idxs if i not in occupied]
        if not newly:
            return
        new_blocks.append(idxs)
        for i2 in idxs:
            occupied.add(i2)

    i = 0
    while i < n:
        if i in occupied:
            i += 1
            continue

        line = sentence_texts[i] or ""

        if _supp_is_reference_header(line):
            start = i + 1
            included: Set[int] = set()
            hard_gap = 0
            saw_any = False
            k = start

            while k < n:
                if k in existing:
                    saw_any = True
                    hard_gap = 0
                    k += 1
                    continue

                lk = sentence_texts[k] or ""
                is_ref_k = _supp_is_url_or_doi_only(lk) or is_reference_line_loose(lk)

                if is_ref_k:
                    included.add(k)
                    saw_any = True
                    hard_gap = 0
                else:
                    if is_filler_line(lk):
                        pass
                    else:
                        if saw_any:
                            hard_gap += 1
                            if hard_gap >= 3:
                                break
                        else:
                            hard_gap += 1
                            if hard_gap >= 3:
                                break
                k += 1

            end = k - 1

            if included and start + 2 <= end:
                base = set(included)
                for j in range(start + 1, end):
                    if j not in base and (j - 1 in base) and (j + 1 in base):
                        included.add(j)

            if included:
                sorted_inc = sorted(included)
                seg = [sorted_inc[0]]
                for idx in sorted_inc[1:]:
                    if idx - seg[-1] <= 2:
                        seg.append(idx)
                    else:
                        a, b = seg[0], seg[-1]
                        add_block(list(range(a, b + 1)))
                        seg = [idx]
                if seg:
                    a, b = seg[0], seg[-1]
                    add_block(list(range(a, b + 1)))

            i = i + 1
            continue

        is_loose = is_reference_line_loose(line) or _supp_is_url_or_doi_only(line)
        if not is_loose:
            i += 1
            continue

        if _supp_is_num_lead(line):
            a = i
            while a - 1 >= 0 and (a - 1) not in occupied:
                prev = sentence_texts[a - 1] or ""
                if _supp_is_blank_line(prev):
                    a -= 1
                    continue
                if _supp_is_num_lead(prev):
                    mnum = SUPP_NUM_PARSE_RE.match(prev)
                    if mnum and int(mnum.group(1)) == 1:
                        a -= 1
                        break
                    a -= 1
                    continue
                break

            b = i
            while b + 1 < n and (b + 1) not in occupied:
                nxt = sentence_texts[b + 1] or ""
                if _supp_is_blank_line(nxt) or _supp_is_num_lead(nxt):
                    b += 1
                    continue
                break

            span = sentence_texts[a:b + 1]
            if _supp_first_num_is_one(span):
                denom = 0
                votes = 0
                for t in span:
                    if _supp_is_blank_line(t):
                        continue
                    denom += 1
                    if is_reference_line_loose(t) or _supp_is_url_or_doi_only(t):
                        votes += 1
                if denom > 0 and votes > (denom / 1.5):
                    add_block(list(range(a, b + 1)))
                    i = b + 1
                    continue

        if is_bracketish_line(line) or is_bullet_refish_line(line):
            a = i
            while a - 1 >= 0 and (a - 1) not in occupied:
                prev = sentence_texts[a - 1] or ""
                if in_bracket_seq(prev):
                    a -= 1
                    continue
                break

            b = i
            while b + 1 < n and (b + 1) not in occupied:
                nxt = sentence_texts[b + 1] or ""
                if in_bracket_seq(nxt):
                    b += 1
                    continue
                break

            span = sentence_texts[a:b + 1]
            denom = 0
            votes = 0
            for t in span:
                if _supp_is_blank_line(t):
                    continue
                denom += 1
                if is_reference_line_loose(t) or _supp_is_url_or_doi_only(t):
                    votes += 1
            if denom > 0 and votes > (denom / 1.5) and len(span) > 1:
                add_block(list(range(a, b + 1)))
                i = b + 1
                continue

        j = i
        run: List[int] = []
        while j < n and j not in occupied:
            lj = sentence_texts[j] or ""
            if is_reference_line_loose(lj) or _supp_is_url_or_doi_only(lj):
                run.append(j)
                j += 1
                continue
            break
        if len(run) >= 2:
            add_block(run)
            i = j
            continue

        i += 1

    return blocks + new_blocks


def is_reviewer_role(role: Any) -> bool:
    if not isinstance(role, str):
        return False
    return REVIEWER_ROLE_RE.search(role) is not None


def get_reviewer_ids_from_opinion(op: Any) -> List[int]:
    """
    Collect reviewer sentence ids from supported opinion formats.
    - dict: {"data":[{"role":..., "sentence_ids":...}, ...], "category":[...]}
    - list: [ [ [role, sentence_ids], ... ], category ]
    """
    ids: List[int] = []

    if isinstance(op, dict):
        data = op.get("data")
        if isinstance(data, list):
            for turn in data:
                if isinstance(turn, dict):
                    if is_reviewer_role(turn.get("role")) and "sentence_ids" in turn:
                        extract_ints(turn["sentence_ids"], ids)
                elif isinstance(turn, (list, tuple)) and len(turn) == 2:
                    role, sids = turn[0], turn[1]
                    if is_reviewer_role(role):
                        extract_ints(sids, ids)
        return ids

    if isinstance(op, list) and len(op) >= 1:
        data = op[0]
        if isinstance(data, list):
            for turn in data:
                if isinstance(turn, dict):
                    if is_reviewer_role(turn.get("role")) and "sentence_ids" in turn:
                        extract_ints(turn.get("sentence_ids"), ids)
                elif isinstance(turn, (list, tuple)) and len(turn) == 2:
                    role, sids = turn[0], turn[1]
                    if is_reviewer_role(role):
                        extract_ints(sids, ids)
        return ids

    return ids


def set_opinion_category_na(op: Any) -> None:
    """Set the opinion category/categories to ['N/A'] for dict or list formats."""
    if isinstance(op, dict):
        if "category" in op:
            op["category"] = ["N/A"]
        elif "categories" in op:
            op["categories"] = ["N/A"]
        else:
            op["category"] = ["N/A"]
        return

    if isinstance(op, list):
        if len(op) >= 2:
            op[1] = ["N/A"]
        else:
            op.append(["N/A"])


def process_one_sample(obj: Dict[str, Any], min_block_len: int = 3) -> Tuple[bool, Set[int]]:
    """
    Return (changed, moved_ids) for a single sample.
    """
    normalize_reviewer_roles(obj)

    sentence_texts = obj.get("sentence_texts")
    opinions = obj.get("opinions")
    if not isinstance(sentence_texts, list) or not isinstance(opinions, list):
        return False, set()

    blocks = find_reference_blocks(sentence_texts, min_labeled=min_block_len)
    blocks = supplement_reference_blocks(sentence_texts, blocks)
    ref_idx: Set[int] = set(i for b in blocks for i in b)
    if not ref_idx:
        return False, set()

    moved_ids: Set[int] = set()
    changed = False
    max_sid = len(sentence_texts) - 1

    for op_i, op in enumerate(opinions):
        reviewer_ids = get_reviewer_ids_from_opinion(op)

        if not reviewer_ids:
            continue

        if any((sid < 0 or sid > max_sid) for sid in reviewer_ids):
            continue

        if set(reviewer_ids).issubset(ref_idx):
            set_opinion_category_na(op)
            moved_ids.add(op_i)
            changed = True

    if moved_ids:
        for field in ("conflicts_validation", "rebuttal_validation"):
            lst = obj.get(field)
            if isinstance(lst, list):
                for idx in moved_ids:
                    if 0 <= idx < len(lst):
                        lst[idx] = "correct"

    if moved_ids:
        og = obj.get("opinion_groups")
        new_group = sorted(moved_ids)

        if isinstance(og, list):
            new_og = []
            for g in og:
                if isinstance(g, list):
                    g2 = [sid for sid in g if sid not in moved_ids]
                    if g2:
                        new_og.append(g2)
                else:
                    new_og.append(g)
            new_og.append(new_group)
            obj["opinion_groups"] = new_og
        else:
            obj["opinion_groups"] = [new_group]

        changed = True

    return changed, moved_ids


def convert_pairs(x):
    """
    Convert [role, payload] lists into {"role": role, "sentence_ids": payload}.
    Other lists and dicts are processed recursively.
    """
    if isinstance(x, list):
        if len(x) == 2 and isinstance(x[0], str) and not isinstance(x[1], str):
            return {"role": x[0], "sentence_ids": convert_pairs(x[1])}
        return [convert_pairs(i) for i in x]
    if isinstance(x, dict):
        return {k: convert_pairs(v) for k, v in x.items()}
    return x


def main(out_path: str = "benchmark/benchmark.jsonl", min_block_len: int = 1) -> None:
    out_rows: List[Dict[str, Any]] = []
    total = 0
    changed_samples = 0
    total_na_opinions = 0

    for conf, year in GROUPS:
        path = f"{conf}.cc_{year}/split_final/{conf}.cc_{year}_gpt-5-mini_medium_split_clean_with_reviewer_validations.jsonl"
        conference = f"{conf} {year}"
        
        for obj in iter_jsonl(path):
            total += 1
            for f in ("reviews", "metareview", "opinions"):
                if f in obj:
                    obj[f] = convert_pairs(obj[f])
            for cnt, it in enumerate(obj["opinions"]):
                assert len(it) == 2
                obj["opinions"][cnt] = {"data": it[0], "category": it[1]}
            obj["conference"] = conference

            changed, moved_ids = process_one_sample(obj, min_block_len=min_block_len)
            if changed:
                changed_samples += 1
                total_na_opinions += len(moved_ids)
            obj['opinion_groups'] = sorted([sorted(it) for it in obj['opinion_groups']])
            out_rows.append(obj)

    write_jsonl(out_path, out_rows)
    print(
        f"Done. total_samples={total}, changed_samples={changed_samples}, "
        f"na_opinions={total_na_opinions}, out={out_path}"
    )


if __name__ == "__main__":
    main()

