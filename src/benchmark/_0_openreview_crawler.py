#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose
-------
Crawl submissions + discussion threads from OpenReview (ICLR / NeurIPS),
and download the "correct" PDF version for each paper.

PDF selection policy
--------------------
1) Prefer OpenReview PDF if it is the last edit strictly before the first public comment
   (or other allowed rules for certain venues).
2) If OpenReview PDF is deemed not correct, fall back to arXiv:
   - Search arXiv by paper title.
   - Pick the latest arXiv version at or before the review/rebuttal cutoff date.
   - Download that specific arXiv version into the same PDF directory
     (filename is the arXiv id, e.g. 2410.16208v3.pdf).

Output fields
-------------
- PDF_version_correct: "openreview", "arxiv", or False (tri-state)
- PDF_path: local path to the chosen PDF, or None if not available/readable

Notes
-----
- This script uses local disk caching under ./cache/ to reduce repeated API calls.
- Be mindful of rate limits and request volume when crawling large venues.
"""

import os
import argparse
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json
import requests
import pickle
import openreview
import string
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import faulthandler
faulthandler.dump_traceback_later(120, repeat=True)

# -----------------------------------------------------------------------------
# Minimal arXiv integration (adapted from a separate implementation).
# Only dependencies needed for:
#   - title-based search
#   - version selection by cutoff timestamp
#   - robust PDF download + readability checks
# -----------------------------------------------------------------------------
import re
import shutil
import time
import difflib
import hashlib
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse
from urllib.error import HTTPError
from PyPDF2 import PdfReader

logger = logging.getLogger(__name__)

_ARXIV_ID_RE = re.compile(
    r"""(?ix)
    (?:                                     # Optional URL prefix
        (?:https?://)?(?:arxiv\.org/)?
        (?:abs|pdf|format)/?
    )?
    (?:arXiv:)?                             # Optional 'arXiv:' prefix
    (?P<id>
        (?:\d{4}\.\d{4,5})(?:v\d+)?         # New: 2410.16208 or 2410.16208v4
        |
        (?:[a-z\-]+(?:\.[a-z]{2})?/\d{7})(?:v\d+)?  # Old: hep-th/9112002v1, cs/0703056, cs.AI/0703056v2
    )
    (?:\.pdf)?                              # Optional '.pdf' suffix
    """,
)


def _is_pdf_readable(path: Path) -> bool:
    """
    Robust PDF sanity check to avoid caching HTML error pages / truncated downloads.

    Checks:
      1) file exists and size >= 1KB
      2) header starts with '%PDF-'
      3) tail contains 'startxref' and '%%EOF'
      4) (optional) PyPDF2 can parse pages
    """
    try:
        if not path.exists() or not path.is_file():
            return False
        size = path.stat().st_size
        if size < 1024:
            return False

        with open(path, "rb") as f:
            head = f.read(8)
            if not head.startswith(b"%PDF-"):
                return False
            tail_seek = max(0, size - 2048)
            f.seek(tail_seek, os.SEEK_SET)
            tail = f.read()
            if b"%%EOF" not in tail:
                return False
            if b"startxref" not in tail:
                return False

        if PdfReader is not None:
            try:
                reader = PdfReader(str(path))
                _ = len(reader.pages)
                if _ > 0:
                    try:
                        _ = reader.pages[0].extract_text()
                    except Exception:
                        pass
                return True
            except Exception as e:
                logger.warning(f"[PDF CHECK] Parse failed (treat as unreadable): {path} -> {e}")
                return False

        return True
    except Exception as e:
        logger.warning(f"[PDF CHECK] Unexpected error: {path} -> {e}")
        return False


def _parse_arxiv_identifier(obj: Any) -> Tuple[str, Optional[int], str]:
    """
    Parse an arXiv identifier (or URL) and return:
      - base_id (without version suffix)
      - version number (int) or None
      - full_id (possibly includes vN)
    """
    def _parse_from_str(s: str) -> Tuple[str, Optional[int], str]:
        s = (s or "").strip()
        if not s:
            raise ValueError("Empty string is not a valid arXiv identifier")
        m = _ARXIV_ID_RE.search(s.replace(".pdf", ""))
        if not m:
            try:
                name = Path(urlparse(s).path).name
            except Exception:
                raise ValueError(f"Cannot parse arXiv identifier: {s!r}")
            name = name[:-4] if name.endswith(".pdf") else name
            m = _ARXIV_ID_RE.fullmatch(name)
            if not m:
                raise ValueError(f"Cannot extract arXiv ID: {s!r}")
        full_id = m.group("id")
        base_id = re.sub(r"v\d+$", "", full_id)
        mv = re.search(r"v(\d+)$", full_id)
        version = int(mv.group(1)) if mv else None
        return base_id, version, full_id

    if isinstance(obj, dict):
        for key in ("id_url", "abs_url", "pdf_url"):
            s = (obj.get(key) or "").strip()
            if not s:
                continue
            try:
                return _parse_from_str(s)
            except Exception:
                continue
        return "", None, ""
    return _parse_from_str(str(obj))


def download_arxiv_to_cache(url_or_id: str, cache_dir: str,
                            timeout: float = 300.0, max_retries: int = 2) -> Optional[str]:
    """
    Download arXiv PDF into <cache_dir>/<arxiv_id>.pdf.

    Behavior:
      - If a specific version is requested and returns 404, it will fall back to earlier versions.
      - Cached files are validated; unreadable files are deleted and re-downloaded.

    Returns:
      - local path string on success
      - False/None on failure
    """
    base_id, v, full_id = _parse_arxiv_identifier(url_or_id)
    if not base_id:
        raise ValueError(f"Invalid arXiv identifier: {url_or_id!r}")

    dest = Path(cache_dir) / f"{full_id}.pdf"
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Cache hit: validate first; if unreadable, delete and re-download.
    if dest.exists():
        if _is_pdf_readable(dest):
            logger.warning("Using caching " + str(dest))
            return str(dest)
        try:
            logger.warning(f"[INFO] Found corrupted/unreadable PDF; delete and re-download: {dest}")
            dest.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"[WARN] Failed to delete unreadable cache: {dest} -> {e}")

    def _try_download_to(path: Path, pdf_url: str, retries: int) -> Tuple[bool, bool]:
        """
        Download with a temp file, validate readability, then atomically replace.
        Returns (ok, is_404).
        """
        logger.warning("Downloading " + pdf_url + " to " + str(path))
        pdf_url = pdf_url.replace("://arxiv.org/", "://export.arxiv.org/")
        tmp = path.with_suffix(".part")
        last_exc = None
        for attempt in range(1, retries):
            try:
                req = urllib.request.Request(pdf_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as f:
                    shutil.copyfileobj(resp, f)

                # Validate before replacing the final file.
                if not _is_pdf_readable(tmp):
                    last_exc = ValueError("Downloaded file is not a readable PDF")
                    try:
                        tmp.unlink(missing_ok=True)
                    except Exception:
                        pass
                    if attempt < retries:
                        time.sleep(attempt)
                    continue

                tmp.replace(path)
                return True, False

            except HTTPError as he:
                if he.code == 404:
                    try:
                        tmp.unlink(missing_ok=True)
                    except Exception:
                        pass
                    return False, True
                last_exc = he
            except Exception as e:
                last_exc = e

            # Cleanup and backoff.
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            if attempt < retries:
                time.sleep(attempt)

        if isinstance(last_exc, HTTPError) and last_exc.code == 404:
            return False, True
        raise TimeoutError(
            f"Failed downloading arXiv PDF (retried {retries} times): {pdf_url}"
        ) from last_exc

    # No explicit version: download the latest.
    if v is None:
        pdf_url = f"https://arxiv.org/pdf/{base_id}.pdf"
        ok, _ = _try_download_to(dest, pdf_url, max_retries)
        return str(dest) if ok else False

    # With vN: try that version first.
    pdf_url = f"https://arxiv.org/pdf/{base_id}v{v}.pdf"
    ok, is404 = _try_download_to(dest, pdf_url, max_retries)
    if ok:
        return str(dest)
    if not is404:
        return False  # Only fall back when confirmed 404.

    # Fall back to earlier versions (v-1 ... v1).
    for k in range(v - 1, 0, -1):
        alt_id = f"{base_id}v{k}"
        alt_dest = Path(cache_dir) / f"{alt_id}.pdf"
        if alt_dest.exists():
            if _is_pdf_readable(alt_dest):
                return str(alt_dest)
            try:
                logger.warning(f"[INFO] Found corrupted/unreadable PDF; delete and re-download: {alt_dest}")
                alt_dest.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"[WARN] Failed to delete unreadable cache: {alt_dest} -> {e}")
        alt_url = f"https://arxiv.org/pdf/{alt_id}.pdf"
        try:
            ok2, is404_2 = _try_download_to(alt_dest, alt_url, retries=2)
        except TimeoutError:
            ok2, is404_2 = False, False
        if ok2:
            return str(alt_dest)
        if is404_2:
            continue
        continue

    return False


def _normalize_title_for_match(s: str) -> str:
    """Normalize title text for robust matching across formatting variants."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = s.replace("\u200b", "").replace("\ufeff", "")
    s = "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")
    s = re.sub(r"\[[^\]]+\]", " ", s)
    s = re.sub(r"\((?:extended|revised|camera[- ]?ready|supplementary|appendix|note)\b[^)]*\)", " ", s, flags=re.I)
    greek_map = {
        "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon",
        "θ": "theta", "λ": "lambda", "μ": "mu", "π": "pi", "σ": "sigma",
        "φ": "phi", "ψ": "psi", "ω": "omega",
    }
    s = "".join(greek_map.get(ch, ch) for ch in s)
    s = s.lower()
    s = re.sub(r"[^0-9a-z]+", " ", s)
    s = " ".join(s.split())
    return s


def _tokenize_title(s: str) -> List[str]:
    """Tokenize a normalized title and drop common stopwords/noisy tokens."""
    toks = _normalize_title_for_match(s).split()
    stop = {
        "a", "an", "the", "of", "on", "for", "and", "to", "in", "with", "via", "from", "by",
        "under", "over", "at", "as", "is", "are", "be", "into", "between", "towards",
        "using", "based", "toward", "without", "beyond", "against", "within",
        "learning", "neural", "model", "models", "method", "methods", "approach", "approaches",
        "deep", "data", "task", "tasks", "analysis", "study"
    }
    return [t for t in toks if len(t) >= 3 and t not in stop and not t.isdigit()]


def _score_pair(a: str, b: str) -> float:
    """
    Heuristic similarity score between titles:
      - character-level similarity (SequenceMatcher)
      - token overlap F1
      - Jaccard token overlap
    """
    if not a or not b:
        return 0.0
    try:
        char_sim = difflib.SequenceMatcher(None, a, b).ratio()
    except Exception:
        char_sim = 0.0
    A, B = set(_tokenize_title(a)), set(_tokenize_title(b))
    inter = len(A & B)
    f1 = (2 * inter) / (len(A) + len(B)) if (A and B) else 0.0
    jacc = inter / len(A | B) if (A and B) else 0.0
    return max(char_sim, f1, jacc)


def _parse_any_datetime(v) -> Optional[datetime]:
    """Parse common datetime encodings into a timezone-aware UTC datetime."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        ts = float(v)
        if ts > 1e12:
            ts /= 1000.0
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        if re.fullmatch(r"\d{10,13}", s):
            ts = float(s)
            if len(s) > 10:
                ts /= 1000.0
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        iso = s
        if iso.endswith("Z"):
            iso = iso[:-1] + "+00:00"
        if iso.endswith(" UTC"):
            iso = iso[:-4] + "+00:00"
        if iso.endswith(" GMT"):
            iso = iso[:-4] + "+00:00"
        try:
            dt = datetime.fromisoformat(iso)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt
        except Exception:
            pass
        try:
            dt = parsedate_to_datetime(s)
            if dt is not None:
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = dt.astimezone(timezone.utc)
                return dt
        except Exception:
            pass
        for fmt in ("%a, %d %b %Y %H:%M:%S %Z", "%d %b %Y %H:%M:%S %Z", "%Y-%m-%d", "%Y/%m/%d"):
            try:
                dt = datetime.strptime(s, fmt)
                dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except Exception:
                continue
    return None


def _aoe_end_of_day_utc(y: int, m: int, d: int) -> datetime:
    """
    Convert an Anywhere-on-Earth (AoE) deadline date into an approximate UTC cutoff.
    AoE is UTC-12; adding 12 hours to end-of-day UTC is a common approximation.
    """
    return datetime(y, m, d, 23, 59, 59, tzinfo=timezone.utc) + timedelta(hours=12)


def _infer_review_cutoff_dt(org: Optional[str], year: Optional[int]) -> Optional[datetime]:
    """
    Infer a venue-specific cutoff datetime (UTC) for selecting arXiv versions.
    This is a hard-coded table; extend as needed for more venues/years.
    """
    if not isinstance(year, int) or not org:
        return None
    org_u = org.upper()
    ICLR_AOE_DATES = {
        2017: (2016, 12, 17),
        2018: (2017, 11, 27),
        2019: (2018, 12, 4),
        2020: (2019, 11, 4),
        2021: (2020, 11, 10),
        2022: (2021, 11, 8),
        2023: (2022, 11, 4),
        2024: (2023, 11, 10),
        2025: (2024, 11, 13),
        2026: (2025, 11, 11),
    }
    NEURIPS_AOE_DATES = {
        2021: (2021, 8, 3),
        2022: (2022, 7, 26),
        2023: (2023, 8, 2),
        2024: (2024, 7, 30),
        2025: (2025, 7, 24),
    }
    if org_u == "ICLR" and year in ICLR_AOE_DATES:
        y, m, d = ICLR_AOE_DATES[year]
        return _aoe_end_of_day_utc(y, m, d)
    if org_u in ("NEURIPS", "NIPS") and year in NEURIPS_AOE_DATES:
        y, m, d = NEURIPS_AOE_DATES[year]
        return _aoe_end_of_day_utc(y, m, d)
    return None


def _arxiv_cache_path(title: str, max_results: int, timeout: int, page_size: int, max_pages: int) -> str:
    """Build a deterministic cache filename for arXiv title queries."""
    key = {
        "title": title or "",
        "max_results": int(max_results),
        "timeout": int(timeout),
        "page_size": int(page_size),
        "max_pages": int(max_pages),
    }
    raw = json.dumps(key, sort_keys=True, ensure_ascii=False)
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"arxiv_query_{h}.json")


def _parse_entry(e) -> Dict[str, Any]:
    """Parse one Atom <entry> from arXiv API response into a dict."""
    ns = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    eid = (e.findtext("a:id", default="", namespaces=ns) or "").strip()
    title = (e.findtext("a:title", default="", namespaces=ns) or "").strip().replace("\n", " ").strip()
    published = (e.findtext("a:published", default="", namespaces=ns) or "").strip()
    updated = (e.findtext("a:updated", default="", namespaces=ns) or "").strip()
    abs_url, pdf_url = "", ""
    for link in e.findall("a:link", ns):
        rel = link.attrib.get("rel", "")
        typ = link.attrib.get("type", "")
        href = link.attrib.get("href", "")
        title_attr = link.attrib.get("title", "")
        if not abs_url and rel == "alternate":
            abs_url = href
        if (typ == "application/pdf") or (title_attr.lower() == "pdf"):
            pdf_url = href
    versions: List[Dict[str, str]] = []
    for v in e.findall("arxiv:version", ns):
        vtag = (v.attrib.get("version") or "").strip()
        vdate = (v.text or "").strip()
        if not vdate:
            vdate = v.findtext("a:date", default="", namespaces=ns) or ""
        if vtag:
            versions.append({"version": vtag, "date": vdate})
    base_id = ""
    try:
        path = urlparse(eid).path
        name = Path(path).name
        base_id = re.sub(r"v\d+$", "", name)
    except Exception:
        pass
    if not pdf_url and base_id:
        pdf_url = f"https://arxiv.org/pdf/{base_id}.pdf"
    return {
        "id_url": eid,
        "base_id": base_id,
        "title": title,
        "published": published,
        "updated": updated,
        "abs_url": abs_url,
        "pdf_url": pdf_url,
        "versions": versions,
    }


def _fetch_once(q: str, start: int, limit: int, timeout: int) -> List[Dict[str, Any]]:
    """Fetch one page of arXiv API results for a given query string."""
    params = {
        "search_query": q,
        "max_results": str(limit),
        "start": str(start),
        "sortBy": "submittedDate",
        "sortOrder": "ascending",
    }
    base = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    req = urllib.request.Request(base, headers={"User-Agent": "pdf-fixer/1.2 (+https://arxiv.org)"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
    except Exception:
        return []
    try:
        root = ET.fromstring(data)
    except Exception:
        return []
    ns = {"a": "http://www.w3.org/2005/Atom"}
    entries = []
    for e in root.findall("a:entry", ns):
        entries.append(_parse_entry(e))
    return entries


def _arxiv_query_entries_by_title(title: str,
                                  max_results: int = 50,
                                  timeout: int = 10,
                                  page_size: int = 50,
                                  max_pages: int = 2) -> List[Dict[str, Any]]:
    """
    Query arXiv by title using multiple progressively looser strategies.
    Results are cached on disk to avoid re-querying.
    """
    _cache_file = _arxiv_cache_path(title or "", max_results, timeout, page_size, max_pages)
    try:
        if os.path.exists(_cache_file):
            with open(_cache_file, "r", encoding="utf-8") as _cf:
                _cached = json.load(_cf)
                if isinstance(_cached, list) and len(_cached):
                    logger.warning("Using caching " + str(_cache_file))
                    return _cached
    except Exception:
        pass

    logger.warning("Searching " + str(title))
    toks = _tokenize_title(title)[:12]
    queries: List[str] = []
    if title.strip():
        queries.append(f'ti:"{title.strip()}"')
    if toks:
        queries.append(" AND ".join(f"ti:{t}" for t in toks))
        queries.append(" OR ".join(f"ti:{t}" for t in toks))

    seen_ids = set()
    results: List[Dict[str, Any]] = []
    target = max_results

    for q in queries:
        if len(results) >= target:
            break
        for pg in range(max_pages):
            if len(results) >= target:
                break
            take = min(page_size, target - len(results))
            start = pg * page_size
            page_entries = _fetch_once(q, start=start, limit=take, timeout=timeout)
            if not page_entries:
                break
            new_any = False
            for en in page_entries:
                key = (en.get("id_url") or "") + "#" + (en.get("updated") or "")
                if key and key not in seen_ids:
                    results.append(en)
                    seen_ids.add(key)
                    new_any = True
            if not new_any:
                break

    with open(_cache_file, "w", encoding="utf-8") as _cf:
        json.dump(results, _cf, ensure_ascii=False, indent=2)
    if not len(results):
        logger.error("Searching " + str(title) + " Failed!")

    return results


def _fetch_versions_from_api(base_id: str, timeout: int = 60) -> List[Dict[str, str]]:
    """
    Fetch arXiv version list (version tag + timestamp) via OAI-PMH arXivRaw.
    Cached to disk similarly to title query caching.
    """
    key = {
        "base_id": base_id or "",
        "timeout": int(timeout),
    }
    raw = json.dumps(key, sort_keys=True, ensure_ascii=False)
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    _cache_file = os.path.join(cache_dir, f"arxiv_versions_{h}.json")

    if os.path.exists(_cache_file):
        with open(_cache_file, "r", encoding="utf-8") as _cf:
            _cached = json.load(_cf)
            if isinstance(_cached, list) and len(_cached):
                return _cached

    versions: List[Dict[str, str]] = []
    logger.warning("Searching versions for " + str(base_id))

    # Primary source: OAI-PMH arXivRaw (provides per-version timestamps).
    try:
        params = {
            "verb": "GetRecord",
            "identifier": f"oai:arXiv.org:{base_id}",
            "metadataPrefix": "arXivRaw",
        }
        url = "https://oaipmh.arxiv.org/oai?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "pdf-fixer/1.3 (+https://example.org/contact)"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()

        root = ET.fromstring(data)
        ns = {
            "oai": "http://www.openarchives.org/OAI/2.0/",
            "raw": "http://arxiv.org/OAI/arXivRaw/",
        }
        raw_root = root.find(".//oai:GetRecord/oai:record/oai:metadata/raw:arXivRaw", ns)
        if raw_root is None:
            raw_root = root.find(".//raw:arXivRaw", ns)

        if raw_root is not None:
            for v in raw_root.findall("raw:version", ns):
                vtag = (
                    (v.attrib.get("version") or (v.findtext("raw:version", default="", namespaces=ns) or ""))
                    .strip()
                )
                vdate = (
                    (v.attrib.get("date") or v.findtext("raw:date", default="", namespaces=ns) or (v.text or ""))
                    .strip()
                )
                if vtag:
                    versions.append({"version": vtag, "date": vdate})

        if versions:
            def _vnum(rec: Dict[str, str]) -> int:
                m = re.search(r"v(\d+)$", rec.get("version", ""))
                return int(m.group(1)) if m else 0

            versions.sort(key=_vnum)
    except Exception:
        versions = []

    if versions:
        with open(_cache_file, "w", encoding="utf-8") as _cf:
            json.dump(versions, _cf, ensure_ascii=False, indent=2)

    return versions


def _choose_version_at_or_before_cutoff(entry: Dict[str, Any], cutoff_dt: datetime) -> Optional[Tuple[str, datetime]]:
    """
    Given an arXiv entry, choose the latest version with timestamp <= cutoff_dt.
    If version timestamps are unavailable, fall back to published time for v1.
    """
    base_id, _, _ = _parse_arxiv_identifier(entry)
    versions = _fetch_versions_from_api(base_id) or []
    best = None
    best_dt = None
    for v in versions:
        vtag = (v.get("version") or "").strip()
        vdt = _parse_any_datetime(v.get("date"))
        if not vtag or not vdt:
            continue
        if vdt <= cutoff_dt and (best_dt is None or vdt > best_dt):
            best = vtag
            best_dt = vdt
    if best and best_dt:
        return best, best_dt
    pub = _parse_any_datetime(entry.get("published"))
    if pub and pub <= cutoff_dt:
        return "v1", pub
    return None


def search_arxiv_pdf_at_cutoff(title: str, org: Optional[str], year: Optional[int],
                               max_results: int = 50, timeout: int = 10,
                               page_size: int = 50, max_pages: int = 2) -> Optional[Dict[str, Any]]:
    """
    Search arXiv by title and pick a version that is:
      - at or before the inferred review cutoff datetime
      - within a recent window (<= ~270 days before cutoff) to reduce mismatches
    """
    cutoff_dt = _infer_review_cutoff_dt(org, year)
    if cutoff_dt is None:
        return None

    # Only consider arXiv versions within ~270 days before cutoff.
    min_dt = cutoff_dt - timedelta(days=270)

    norm_query = _normalize_title_for_match(title or "")
    entries = _arxiv_query_entries_by_title(
        title, max_results=max_results, timeout=timeout, page_size=page_size, max_pages=max_pages,
    )

    best_entry = None
    best_score = -1.0
    best_version = None
    best_vdt = None

    for en in entries:
        cand_title = (en.get("title") or "").strip()
        cand_norm = _normalize_title_for_match(cand_title)
        score = _score_pair(norm_query, cand_norm)
        if not (cand_norm == norm_query or score >= 0.8):
            continue

        ver_pick = _choose_version_at_or_before_cutoff(en, cutoff_dt)
        if not ver_pick:
            continue

        vtag, vdt = ver_pick

        # Window filter (recent relative to cutoff).
        if vdt is None or not (min_dt <= vdt <= cutoff_dt):
            continue

        if score > best_score:
            best_entry = en
            best_score = score
            best_version = vtag
            best_vdt = vdt

    if not best_entry or not best_version:
        return None

    base_id, _, _ = _parse_arxiv_identifier(best_entry)
    if not base_id:
        return None

    pdf_url = f"https://arxiv.org/pdf/{base_id}{best_version}.pdf"
    abs_url = f"https://arxiv.org/abs/{base_id}{best_version}"
    return {
        "pdf_url": pdf_url,
        "abs_url": abs_url,
        "arxiv_id": f"{base_id}{best_version}",
        "arxiv_version": best_version,
        "arxiv_published": (best_vdt.isoformat() if isinstance(best_vdt, datetime) else (best_entry.get("published") or "")),
    }


# -----------------------------------------------------------------------------
# OpenReview crawler
# -----------------------------------------------------------------------------
class VenueCrawler:
    """
    Crawl submissions and discussion threads from an OpenReview venue.

    This crawler encapsulates the difference between OpenReview API v1 and v2:
      - older venues/years use api.openreview.net (v1)
      - newer venues/years use api2.openreview.net (v2)

    It also supports PDF downloading with OpenReview-first and optional arXiv fallback.
    """

    def __init__(self, *, username: Optional[str] = None,
                 password: Optional[str] = None,
                 token: Optional[str] = None,
                 verbose: bool = True,
                 download_PDF: bool = True) -> None:
        self.username = username
        self.password = password
        self.token = token
        self.verbose = verbose
        self.download = download_PDF
        self.finished: List[str] = []
        self.org = ""
        self.year = 0
        self.client = None

        # Concurrency control (ProcessPoolExecutor worker count).
        self.max_workers = 8  # Adjust based on network and CPU.

    # ------------------------------------------------------------------
    # API selection and invitation patterns
    # ------------------------------------------------------------------
    def _create_client(self, org: str, year: int) -> Any:
        """Instantiate an OpenReview client for the requested venue/year."""
        self.use_v1 = False
        if (org == "ICLR.cc" and year <= 2023) or \
           (org == "NeurIPS.cc" and year <= 2022):
            self.use_v1 = True

        if self.use_v1:
            baseurl = "https://api.openreview.net"
            if self.verbose:
                logger.warning(f"Using APIv1 for {org} {year}")
            try:
                client = openreview.Client(
                    baseurl=baseurl,
                    username=self.username,
                    password=self.password,
                    token=self.token,
                )
            except Exception:
                client = openreview.Client(baseurl=baseurl)
        else:
            baseurl = "https://api2.openreview.net"
            if self.verbose:
                logger.warning(f"Using APIv2 for {org} {year}")
            try:
                client = openreview.api.OpenReviewClient(  # type: ignore[attr-defined]
                    baseurl=baseurl,
                    username=self.username,
                    password=self.password,
                    token=self.token,
                )
            except Exception:
                client = openreview.api.OpenReviewClient(baseurl=baseurl)  # type: ignore[attr-defined]
        return client

    def _submission_invites(self, org: str, year: int) -> List[str]:
        """Return invitation identifiers to try for submissions."""
        invitations = []
        queries = ['Submission', 'submission', 'Blind_Submission',
                   'Withdrawn_Submission', 'Rejected_Submission',
                   'Desk_Rejected_Submission']

        if org == "ICLR.cc":
            for query in queries:
                if year <= 2017:
                    invitations.append(f'{org}/{year}/conference/-/{query}')
                else:
                    invitations.append(f'{org}/{year}/Conference/-/{query}')
        elif org == "NeurIPS.cc":
            for query in queries:
                invitations.append(f'{org}/{year}/Conference/-/{query}')
        else:
            raise NotImplementedError(f"Invitation patterns for organization '{org}' are not defined.")
        return invitations

    # ------------------------------------------------------------------
    # Utility helpers for roles and caching
    # ------------------------------------------------------------------
    def normalize_string(self, input_str: str) -> str:
        """
        Convert unicode-heavy names to a simplified ASCII-ish signature,
        keeping only first and last name when possible.
        """
        nfkd_form = unicodedata.normalize('NFD', input_str)
        ascii_string = nfkd_form.encode('ASCII', 'ignore').decode('utf-8')
        name_parts = ascii_string.split()
        if len(name_parts) > 2:
            processed_name = f"{name_parts[0]} {name_parts[-1]}"
            return processed_name.lower()
        else:
            return ascii_string.lower()

    def _to_name(self, invitation: Optional[str], authors: Optional[List[str]] = []) -> str:
        """Map OpenReview signature/invitation to a coarse role label."""
        if not invitation:
            return "Unknown"
        invitation_lower = invitation.lower()
        if "reviewer" in invitation_lower:
            return "Reviewer"
        if "author" in invitation_lower:
            return "Author"
        if "area_chair" in invitation_lower or "ac" in invitation_lower:
            return "AC"
        if "program_chair" in invitation_lower or "pc" in invitation_lower:
            return "PC"

        formatted_string = invitation_lower.lstrip('~').rstrip(string.digits).replace('_', ' ')
        formatted_string = self.normalize_string(formatted_string)
        if formatted_string in (authors or []):
            return "Author"
        return "Unknown"

    def get_note_edits_cached(self, note_id: str) -> List[Any]:
        """Fetch note edits from OpenReview with on-disk caching."""
        cache_dir = "cache"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"edits_{self.org}_{self.year}_{note_id}.pkl")

        if os.path.exists(cache_file):
            if self.verbose:
                logger.warning(f"Loading note edits from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        else:
            if self.verbose:
                logger.warning(f"Cache not found. Fetching edits for note: {note_id}")
            try:
                if self.use_v1:
                    edits = self.client.get_all_references(referent=note_id, original=True)
                else:
                    edits = self.client.get_note_edits(note_id=note_id)
                with open(cache_file, 'wb') as f:
                    pickle.dump(edits, f)
                if self.verbose:
                    logger.warning(f"Note edits saved to cache: {cache_file}")
                return edits
            except Exception as e:
                logger.warning(f"Could not fetch edits for note {note_id}. Error: {e}")
                return []

    def get_all_notes_cached(self, invitation, details="replies"):
        """Fetch all notes for an invitation with on-disk caching."""
        cache_dir = "cache"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, invitation.replace("/", "_") + ".pkl")
        if os.path.exists(cache_file):
            if self.verbose:
                logger.warning(f"Loading notes from cache file: {cache_file}")
            with open(cache_file, 'rb') as f:
                notes = pickle.load(f)
            return notes
        else:
            if self.verbose:
                logger.warning(f"Cache file not found. Fetching notes from OpenReview for invitation: {invitation}")
            notes = self.client.get_all_notes(invitation=invitation, details=details)
            with open(cache_file, 'wb') as f:
                pickle.dump(notes, f)
            if self.verbose:
                logger.warning(f"Notes have been saved to cache file: {cache_file}")
            return notes

    # ------------------------------------------------------------------
    # Per-paper processing
    # ------------------------------------------------------------------
    def _process_one(self, note, inv) -> Optional[Dict[str, Any]]:
        """
        Process one submission note:
          - build nested reply threads
          - infer decision
          - download OpenReview PDF (pre-comment edit) and validate
          - optionally fall back to arXiv with cutoff-based version selection
        """
        try:
            replies = note.details.get('replies', [])
            replies = sorted(replies, key=lambda x: x.get("tcdate", 0))
            clean_replies = []
            for it in replies:
                if it.get("replyto") == note.id:
                    clean_replies.append(([it["id"]], [(it.get("signatures", [None])[-1], it["tcdate"], it["content"])]))
                else:
                    flag = False
                    for idx, (k, v) in enumerate(clean_replies):
                        if it.get("replyto") in k:
                            k.append(it["id"])
                            v.append((it.get("signatures", [None])[-1], it["tcdate"], it["content"]))
                            clean_replies[idx] = (k, v)
                            flag = True
                            break

            authors = [self.normalize_string(i) for i in note.content.get('authors', [])] if 'authors' in note.content else []
            final_replies = [[(self._to_name(i, authors), k) for (i, j, k) in y] for (x, y) in clean_replies]

            decision = "Unknown"
            if self.use_v1:
                invs = [note.invitation, inv]
            else:
                invs = list(note.invitations) + [inv]
            for it in invs:
                if "withdrawn_submission" in it.lower():
                    decision = "Withdrawal"
                if "desk_rejected_submission" in it.lower():
                    decision = "Desk Reject"
            if 'withdrawal' in note.content and note.content['withdrawal'] == 'Confirmed':
                decision = "Withdrawal"

            for thread in final_replies:
                if not thread:
                    continue
                k, v = thread[0]
                if v is None:
                    continue
                if self.org == "ICLR.cc" and self.year == 2019 and k == "AC" and 'recommendation' in v:
                    decision = v['recommendation']
                    break
                elif k == "PC":
                    if 'decision' in v:
                        if decision != "Unknown" and decision != "Withdrawal":
                            pass
                        if isinstance(v['decision'], dict):
                            decision = v['decision'].get('value', 'Unknown')
                            break
                        elif isinstance(v['decision'], str):
                            decision = v['decision']

            if decision == "Unknown" and 'decision' in note.content:
                decision = note.content['decision']

            # For OpenReview PDF selection: use the first reply time as a proxy for "comment start".
            tcdate_for_pdf = replies[0]["tcdate"] if replies else float("inf")
            or_correct, or_path = self.download_pdf(note.id, tcdate_for_pdf)
            or_exists = bool(or_path and os.path.exists(or_path))

            correct_source: Any = False
            pdf_path: Optional[str] = None

            # NeurIPS special-case: rejected/withdrawn papers can still accept OpenReview PDFs.
            is_neurips_rej_or_withdraw = (
                self.org == 'NeurIPS.cc'
                and isinstance(decision, str)
                and ("reject" in decision.lower() or "withdrawal" in decision.lower())
            )

            if or_exists and (or_correct or is_neurips_rej_or_withdraw):
                correct_source = "openreview"
                pdf_path = or_path
            else:
                title = note.content.get('title')
                if isinstance(title, dict):
                    title = title.get('value', '')
                if isinstance(title, str) and title.strip():
                    short_org = (self.org.split('.', 1)[0] if '.' in self.org else self.org)
                    found = search_arxiv_pdf_at_cutoff(
                        title=title.strip(),
                        org=short_org,
                        year=self.year,
                        max_results=50,
                        timeout=10,
                        page_size=50,
                        max_pages=2
                    )
                    if found and found.get("pdf_url"):
                        output_dir = f'{self.org}_{self.year}/PDF'
                        local_pdf = download_arxiv_to_cache(found["pdf_url"], cache_dir=output_dir)
                        if local_pdf:
                            correct_source = "arxiv"
                            pdf_path = str(local_pdf)
                        else:
                            correct_source = False
                            pdf_path = None

            # Final safeguard: validate readability again before returning.
            if pdf_path is not None:
                if not _is_pdf_readable(Path(pdf_path)):
                    if self.verbose:
                        logger.warning(f"[WARN] PDF not readable, mark as missing: {pdf_path}")
                    pdf_path = None
                    correct_source = False

            return {
                "id": note.id,
                "content": note.content,
                "replies": final_replies,
                "tcdate": replies[0]["tcdate"] if len(replies) else None,
                "PDF_version_correct": correct_source,
                "PDF_path": pdf_path,
                "decision": decision
            }
        except Exception as e:
            if self.verbose:
                logger.warning(f"[ERROR] Failed processing note {getattr(note, 'id', 'UNKNOWN')}: {e}")
            return None

    def crawl(self, org: str, year: int) -> Iterable[Dict[str, Any]]:
        """
        Yield structured results for all submissions in a venue.

        Implementation detail:
          - Uses ProcessPoolExecutor to parallelize per-paper processing.
        """
        self.org = org
        self.year = year
        self.client = self._create_client(org, year)
        if self.client is None:
            raise RuntimeError("openreview library is not installed; crawling cannot proceed")

        invitations = self._submission_invites(org, year)
        with ProcessPoolExecutor(max_workers=self.max_workers) as ex:
            for inv in invitations:
                try:
                    notes = self.get_all_notes_cached(invitation=inv, details="replies")
                except Exception as e:
                    logger.warning(f"Could not fetch notes for invitation {inv}. Error: {e}")
                    continue

                if self.verbose and notes:
                    logger.warning(f"Fetched {len(notes)} notes for invitation {inv}")

                futures = []
                for note in notes:
                    futures.append(ex.submit(self._process_one, note, inv))

                for fut in as_completed(futures):
                    res = fut.result()
                    if res is None:
                        continue
                    yield res

    def download_pdf(self, paper_id, tcdate) -> Tuple[bool, Optional[str]]:
        """
        Download the OpenReview PDF version that is strictly before tcdate.

        Returns:
          (correct, pdf_path)
        where:
          - correct=True means we found an edit that satisfies the cutoff rule
          - correct=False means we had to fall back (e.g., earliest available edit)
        """
        correct = True
        output_dir = f'{self.org}_{self.year}/PDF'
        os.makedirs(output_dir, exist_ok=True)
        pdf_path = os.path.join(output_dir, f'{paper_id}.pdf')

        note_edits = self.get_note_edits_cached(paper_id)
        note_edits = sorted(note_edits, key=lambda x: x.tcdate)
        edits = [
            i for i in note_edits
            if i.tcdate < tcdate and ('pdf' in i.content if self.use_v1 else 'pdf' in i.note.content)
        ]

        # If no edit before tcdate exists, use the earliest edit and mark as not correct.
        if not len(edits) and len(note_edits):
            edits = [note_edits[0]]
            correct = False
        if self.org == 'NeurIPS.cc' and not len(note_edits):
            correct = False

        if len(edits):
            oldest = edits[-1].id
            if self.use_v1:
                oldest_url = f"https://openreview.net/references/pdf?id={oldest}"
            else:
                oldest_url = f'https://openreview.net/notes/edits/attachment?id={oldest}&name=pdf'
        else:
            oldest = paper_id
            oldest_url = f'https://openreview.net/pdf?id={oldest}'

        # If PDF downloads are disabled, return the expected path for compatibility.
        if not self.download:
            return correct, pdf_path

        # Cache hit: validate; re-download if unreadable.
        if os.path.exists(pdf_path):
            if _is_pdf_readable(Path(pdf_path)):
                return correct, pdf_path
            if self.verbose:
                logger.warning(f"[INFO] Existing PDF not readable, re-downloading: {pdf_path}")

        for retry in range(3):
            try:
                response = requests.get(oldest_url, timeout=300)
                response.raise_for_status()
                with open(pdf_path, 'wb') as f:
                    f.write(response.content)

                if not _is_pdf_readable(Path(pdf_path)):
                    raise ValueError("Downloaded file is not a readable PDF")

                if self.verbose:
                    logger.warning(f"Successfully downloaded {pdf_path}")
                return correct, pdf_path
            except Exception as e:
                logger.warning(f"Attempt {retry+1} failed for {pdf_path}: {e}")

        logger.warning(f"Failed to download PDF for paper {paper_id} after multiple retries.")
        return correct, pdf_path


def main(org: str, year: int, verbose: bool, download_pdf: bool):
    """Run the crawler and write one JSON record per line (jsonl)."""
    logger.warning(f"Starting crawler for {org} {year}...")

    # SECURITY NOTE:
    # Do NOT hardcode credentials in open-source code. Use env vars, config files,
    # or interactive login. Placeholders are used here intentionally.
    crawler = VenueCrawler(
        username="YOUR_OPENREVIEW_EMAIL",
        password="YOUR_OPENREVIEW_PASSWORD",
        verbose=verbose,
        download_PDF=download_pdf
    )

    output_dir = f"{org}_{year}"
    os.makedirs(output_dir, exist_ok=True)
    suffix = "_PDF" if download_pdf else ""
    output_file = os.path.join(output_dir, f"{org}_{year}{suffix}.jsonl")

    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for result in crawler.crawl(org, year):
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')
            count += 1
            if count % 100 == 0:
                logger.warning(f"Processed {count} papers...")

    logger.warning(f"\nCrawling complete. Processed a total of {count} papers.")
    logger.warning(f"Results saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawl submission data from OpenReview for a specific venue and year.")
    parser.add_argument('--org', type=str, default="ICLR.cc", choices=['ICLR.cc', 'NeurIPS.cc'],
                        help='The root organisation, e.g., "ICLR.cc" or "NeurIPS.cc".')
    parser.add_argument('--year', type=int, default=2025,
                        help='The four-digit year for the conference.')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output during crawling.')
    parser.add_argument('--no-pdf', action='store_true',
                        help='Disable PDF downloads.')

    args = parser.parse_args()
    args.download_pdf = not args.no_pdf

    main(org=args.org, year=args.year, verbose=args.verbose, download_pdf=args.download_pdf)
