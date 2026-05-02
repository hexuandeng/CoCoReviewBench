#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


project_root = str(Path(__file__).resolve().parents[2])


def iter_jsonl_objects(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def resolve_benchmark_files(benchmark_patterns: List[str]) -> List[Path]:
    benchmark_dir = Path(project_root) / "benchmark"
    resolved_files: List[Path] = []
    seen_paths = set()

    for benchmark_pattern in benchmark_patterns:
        matched_files = [path.resolve() for path in Path(project_root).glob(benchmark_pattern) if path.is_file()]
        if not matched_files and "/" not in benchmark_pattern and "\\" not in benchmark_pattern:
            matched_files = [path.resolve() for path in benchmark_dir.glob(benchmark_pattern) if path.is_file()]
        if not matched_files:
            raise FileNotFoundError(f"No benchmark files matched pattern: {benchmark_pattern}")

        for matched_file in sorted(matched_files):
            if matched_file in seen_paths:
                continue
            resolved_files.append(matched_file)
            seen_paths.add(matched_file)

    if not resolved_files:
        raise FileNotFoundError("No benchmark files were resolved.")

    return resolved_files


def extract_reviewer_name(role: str) -> Optional[str]:
    match = re.match(r"(Reviewer \d+)", role or "")
    return match.group(1) if match else None


def extract_review_group_text(
    review_group: List[Dict[str, Any]],
    sentence_texts: List[str],
    section_separator: str = "\n\n",
) -> Tuple[Optional[str], str]:
    if not review_group:
        raise ValueError("review_group is empty")

    first_element = review_group[0]
    reviewer_name = extract_reviewer_name(first_element["role"])
    if not reviewer_name:
        return None, None

    sentence_ids = first_element["sentence_ids"]["value"]
    review_text_parts: List[str] = []
    for key, value in sentence_ids.items():
        key_text = "\n".join(
            sentence_texts[sentence_id]
            for sentence_id in value
            if isinstance(sentence_id, int) and 0 <= sentence_id < len(sentence_texts)
        ).strip()
        if not key_text:
            continue
        review_text_parts.append(f"## {key.replace('_', ' ').capitalize()}\n{key_text}")

    return reviewer_name, section_separator.join(review_text_parts).strip()


def load_benchmark_records(
    paper_ids: Optional[List[str]], benchmark_files: List[Path]
) -> Dict[str, Dict[str, Any]]:
    wanted_ids = None if paper_ids is None else {str(paper_id) for paper_id in paper_ids}
    benchmark_records: Dict[str, Dict[str, Any]] = {}

    for benchmark_path in benchmark_files:
        if not benchmark_path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")
        for obj in iter_jsonl_objects(benchmark_path):
            paper_id = str(obj["id"])
            if wanted_ids is None or paper_id in wanted_ids:
                benchmark_records[paper_id] = obj

    missing_ids = [] if wanted_ids is None else sorted(wanted_ids - set(benchmark_records))
    if missing_ids:
        raise FileNotFoundError(
            f"Could not find {len(missing_ids)} paper ids in benchmark files {benchmark_files}. "
            f"Examples: {missing_ids[:10]}"
        )

    return benchmark_records


def collect_reviewer_texts(benchmark_records: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    reviewer_texts_by_paper: Dict[str, Dict[str, str]] = {}

    for paper_id, obj in benchmark_records.items():
        blocks_by_reviewer: Dict[str, List[str]] = {}
        for review_group in obj["reviews"]:
            reviewer_name, review_text = extract_review_group_text(review_group, obj["sentence_texts"])
            if reviewer_name is None or not review_text:
                continue
            blocks_by_reviewer.setdefault(reviewer_name, []).append(review_text)

        reviewer_texts = {
            reviewer_name: "\n\n".join(blocks).strip()
            for reviewer_name, blocks in sorted(blocks_by_reviewer.items())
            if any(block.strip() for block in blocks)
        }
        if reviewer_texts:
            reviewer_texts_by_paper[paper_id] = reviewer_texts

    return reviewer_texts_by_paper


def collect_all_reviewer_role_texts(benchmark_records: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    all_reviewer_texts_by_paper: Dict[str, List[str]] = {}

    for paper_id, obj in benchmark_records.items():
        sentence_texts = obj["sentence_texts"]
        reviewer_texts: List[str] = []

        for review_group in obj["reviews"]:
            for item in review_group:
                role = str(item.get("role", ""))
                if "reviewer" not in role.lower():
                    continue

                sentence_id_container = item.get("sentence_ids", {})
                value = sentence_id_container.get("value") if isinstance(sentence_id_container, dict) else None
                text_parts: List[str] = []
                for section_name, sentence_ids in value.items():
                    section_text = "\n".join(
                        sentence_texts[sentence_id]
                        for sentence_id in sentence_ids
                        if isinstance(sentence_id, int) and 0 <= sentence_id < len(sentence_texts)
                    ).strip()
                    if not section_text:
                        continue
                    text_parts.append(f"## {section_name.replace('_', ' ').capitalize()}\n{section_text}")

                merged_text = "\n\n".join(text_parts).strip()
                if merged_text:
                    reviewer_texts.append(merged_text)

        if reviewer_texts:
            all_reviewer_texts_by_paper[paper_id] = "\n\n".join(reviewer_texts)

    return all_reviewer_texts_by_paper


def load_cached_selection(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return {str(k): str(v) for k, v in json.load(f).items()}


def load_selection_from_left_out_file(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}

    selection: Dict[str, str] = {}
    for obj in iter_jsonl_objects(path):
        paper_id = str(obj["id"])
        reviewer_name = str(obj["reviewer"])
        selection[paper_id] = reviewer_name
    return selection


def select_left_out_reviewers(
    reviewer_texts_by_paper: Dict[str, Dict[str, str]],
    cached_selection: Dict[str, str],
) -> Dict[str, str]:
    selection: Dict[str, str] = {}

    for paper_id, reviewer_texts in sorted(reviewer_texts_by_paper.items()):
        reviewers = sorted(reviewer_texts)
        if not reviewers:
            raise ValueError(f"No reviewers found for paper {paper_id}")

        cached_reviewer = cached_selection.get(paper_id)
        if cached_reviewer in reviewer_texts:
            selection[paper_id] = cached_reviewer
            continue

        selection[paper_id] = random.choice(reviewers)

    return selection


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def iter_all_human_records(all_reviewer_texts_by_paper: Dict[str, List[str]]) -> Iterable[Dict[str, Any]]:
    for paper_id, reviewer_texts in sorted(all_reviewer_texts_by_paper.items()):
        if not reviewer_texts:
            continue
        yield {
            "id": paper_id,
            "text": reviewer_texts,
        }


def iter_left_out_records(
    reviewer_texts_by_paper: Dict[str, Dict[str, str]],
    selection: Dict[str, str],
) -> Iterable[Dict[str, Any]]:
    for paper_id, reviewer_name in sorted(selection.items()):
        review_text = reviewer_texts_by_paper[paper_id][reviewer_name].strip()
        if not review_text:
            raise ValueError(f"Empty left-out human review text for paper {paper_id}")
        yield {
            "id": paper_id,
            "text": review_text,
            "reviewer": reviewer_name,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build benchmark_all_human.jsonl, benchmark_left_out_human.jsonl, and benchmark_reviewer_selection.json."
    )
    parser.add_argument(
        "--benchmark_file",
        type=str,
        nargs="+",
        default=["benchmark/benchmark_*_20*.jsonl"],
        help="Benchmark JSONL file path(s) or glob pattern(s).",
    )
    parser.add_argument(
        "--all_output_file",
        type=str,
        default="benchmark/benchmark_all_human.jsonl",
        help="Output path for all human reviews.",
    )
    parser.add_argument(
        "--left_out_output_file",
        type=str,
        default="benchmark/benchmark_left_out_human.jsonl",
        help="Output path for one selected human review per paper.",
    )
    parser.add_argument(
        "--selection_output_file",
        type=str,
        default="benchmark/benchmark_reviewer_selection.json",
        help="Output path for the selected reviewer mapping.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when a paper is missing a cached reviewer selection.",
    )

    args = parser.parse_args()
    random.seed(args.seed)

    benchmark_files = resolve_benchmark_files(args.benchmark_file)
    all_output_file = Path(args.all_output_file).expanduser().resolve()
    left_out_output_file = Path(args.left_out_output_file).expanduser().resolve()
    selection_output_file = Path(args.selection_output_file).expanduser().resolve()

    benchmark_records = load_benchmark_records(None, benchmark_files)
    reviewer_texts_by_paper = collect_reviewer_texts(benchmark_records)
    all_reviewer_texts_by_paper = collect_all_reviewer_role_texts(benchmark_records)
    cached_selection = load_cached_selection(selection_output_file)
    if not cached_selection:
        cached_selection = load_selection_from_left_out_file(left_out_output_file)
    selection = select_left_out_reviewers(reviewer_texts_by_paper, cached_selection)

    if selection_output_file.exists():
        print(f"Skip existing selection file: {selection_output_file}")
    else:
        write_json(selection_output_file, selection)
        print(f"Wrote selection file: {selection_output_file}")

    if all_output_file.exists():
        all_count = 0
        print(f"Skip existing all-human file: {all_output_file}")
    else:
        all_count = write_jsonl(all_output_file, iter_all_human_records(all_reviewer_texts_by_paper))
        print(f"Wrote all-human file: {all_output_file}")

    if left_out_output_file.exists():
        left_out_count = 0
        print(f"Skip existing left-out file: {left_out_output_file}")
    else:
        left_out_count = write_jsonl(left_out_output_file, iter_left_out_records(reviewer_texts_by_paper, selection))
        print(f"Wrote left-out file: {left_out_output_file}")

    print(f"Benchmark files: {len(benchmark_files)}")
    print(f"Papers with reviewer text: {len(reviewer_texts_by_paper)}")
    print(f"All human records written: {all_count}")
    print(f"Left-out human records written: {left_out_count}")


if __name__ == "__main__":
    main()
