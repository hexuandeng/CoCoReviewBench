#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from collections import defaultdict
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from _0_human_review import (
    extract_review_group_text,
    iter_jsonl_objects,
    load_benchmark_records,
    resolve_benchmark_files,
)
from src.utils import LLMClient, english_word_count


benchmark_reviewer_selection_file = Path(project_root) / "benchmark" / "benchmark_reviewer_selection.json"


table_old_metric_map = {
    "BLEU": "bleu",
    "ROUGE-L": "rougeL",
    "BERT-F1": "bertscore_f1",
}

table_judge_metric_map = {
    "Correctness": "alignment",
    "Thoroughness": "completeness",
    "Grounding": "grounding_specificity",
    "Verifiability": "verifiability",
    "Clarity": "clarity",
}


def parse_llm_json_response(judge_resp: str) -> Dict[str, Any]:
    def _extract_first_json_object(text: str) -> Optional[str]:
        start = text.find("{")
        if start < 0:
            return None

        depth = 0
        in_str = False
        escape = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue

            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
        return None

    json_str = _extract_first_json_object(judge_resp.strip())
    assert json_str is not None, f"Cannot find JSON in response: {judge_resp[:200]}"
    parsed_scores = json.loads(json_str)
    assert isinstance(parsed_scores, dict), f"Expected a JSON object, got: {type(parsed_scores).__name__}"
    return parsed_scores


def calculate_scores_from_parsed(parsed_scores: Dict[str, Any]) -> Dict[str, Any]:
    if "claim_extraction" in parsed_scores or "grounding_specificity" in parsed_scores:
        claim_extraction = parsed_scores.get("claim_extraction")
        actionability = parsed_scores.get("actionability")
        grounding_specificity = parsed_scores.get("grounding_specificity")
        verifiability = parsed_scores.get("verifiability")
        alignment = parsed_scores.get("alignment")
        completeness = parsed_scores.get("completeness")
        clarity = parsed_scores.get("clarity")

        numeric_scores: List[float] = []
        for score in [
            actionability,
            grounding_specificity,
            verifiability,
            alignment,
            completeness,
            clarity,
        ]:
            if score == "X":
                continue
            if isinstance(score, (int, float)):
                numeric_scores.append(float(score))
        avg_score = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 0.0

        return {
            "claim_extraction": claim_extraction,
            "actionability": actionability,
            "grounding_specificity": grounding_specificity,
            "verifiability": verifiability,
            "alignment": alignment,
            "completeness": completeness,
            "clarity": clarity,
            "average": avg_score,
        }

    if "quality" in parsed_scores or "constructive" in parsed_scores:
        quality = parsed_scores.get("quality")
        constructive = parsed_scores.get("constructive")
        accuracy = parsed_scores.get("accuracy")
        completeness = parsed_scores.get("completeness")
        clarity = parsed_scores.get("clarity")

        numeric_scores = [
            float(score)
            for score in [quality, constructive, accuracy, completeness, clarity]
            if isinstance(score, (int, float))
        ]
        avg_score = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 0.0

        return {
            "quality": quality,
            "constructive": constructive,
            "accuracy": accuracy,
            "completeness": completeness,
            "clarity": clarity,
            "average": avg_score,
        }

    numeric_scores = [float(value) for value in parsed_scores.values() if isinstance(value, (int, float))]
    avg_score = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 0.0
    return {**parsed_scores, "average": avg_score}


def get_judge_system_prompt() -> str:
    prompt_file = Path(__file__).parent / "prompt" / "judge.txt"
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read().strip()


def get_reviewer_sentence_ids(opinion: Dict[str, Any]) -> List[int]:
    sentence_ids: List[int] = []
    for item in opinion["data"]:
        role = item["role"]
        if "reviewer" not in role.lower():
            continue
        sentence_ids.extend(
            sentence_id
            for sentence_id in item["sentence_ids"]
            if isinstance(sentence_id, int)
        )
    return sentence_ids


def build_reviewer_opinion_text(opinion: Dict[str, Any], sentence_texts: List[str]) -> str:
    return "\n".join(
        sentence_texts[sentence_id]
        for sentence_id in sorted(set(get_reviewer_sentence_ids(opinion)))
        if 0 <= sentence_id < len(sentence_texts) and sentence_texts[sentence_id]
    )


def is_figure_comment(category_tag: str, opinion_text: str) -> bool:
    return category_tag == "CLAR-FIG" and bool(re.search(r"[Ff]igure", opinion_text))
def load_ai_review_data(input_jsonl_classify: str) -> Dict[str, Dict[str, List[str]]]:
    id_review_category_dict_ai: Dict[str, Dict[str, List[str]]] = {}

    for obj in iter_jsonl_objects(Path(input_jsonl_classify)):
        paper_id = str(obj["id"])

        parts = obj["sentence_texts"]
        labels_by_opinion = obj["category"]
        opinions = obj["opinions"]

        category_dict: Dict[str, List[str]] = {}
        for opinion_idx in range(len(opinions)):
            category_tags = labels_by_opinion[opinion_idx]
            part_indices = opinions[opinion_idx]
            merged_text = "\n\n".join(parts[part_idx] for part_idx in part_indices if 0 <= part_idx < len(parts)).strip()
            if english_word_count(merged_text) <= 3:
                print(merged_text)
                continue

            for category_tag in category_tags:
                if category_tag not in category_dict:
                    category_dict[category_tag] = []
                category_dict[category_tag].append(merged_text)

        id_review_category_dict_ai[paper_id] = category_dict

    return id_review_category_dict_ai


def load_ai_full_text_data(input_jsonl_ai: str) -> Dict[str, str]:
    id_review_full_dict_ai: Dict[str, str] = {}
    input_path = Path(input_jsonl_ai)
    if not input_path.exists():
        raise FileNotFoundError(f"AI data file not found: {input_path}")

    for obj in iter_jsonl_objects(input_path):
        paper_id = str(obj["id"])

        review_text = obj["text"]
        id_review_full_dict_ai[paper_id] = review_text

    return id_review_full_dict_ai


def load_ai_paper_level_data(input_jsonl_classify: str) -> Dict[str, str]:
    id_review_full_dict_ai: Dict[str, str] = {}

    for obj in iter_jsonl_objects(Path(input_jsonl_classify)):
        paper_id = str(obj["id"])

        parts = obj["sentence_texts"]
        labels_by_opinion = obj["category"]
        opinions = obj["opinions"]

        paper_texts: List[str] = []
        for opinion_idx, part_indices in enumerate(opinions):
            category_tags = labels_by_opinion[opinion_idx] if opinion_idx < len(labels_by_opinion) else ["N/A"]
            valid_tags = [category_tag for category_tag in category_tags if category_tag and category_tag != "N/A"]
            if not valid_tags:
                continue

            merged_text = " ".join(parts[part_idx] for part_idx in part_indices if 0 <= part_idx < len(parts)).strip()
            if merged_text:
                paper_texts.append(merged_text)

        id_review_full_dict_ai[paper_id] = "\n\n".join(paper_texts).strip()

    return id_review_full_dict_ai


def left_out_one(
    benchmark_files: List[Path],
) -> Dict[str, str]:
    benchmark_records = load_benchmark_records(None, benchmark_files)
    if not benchmark_reviewer_selection_file.exists():
        raise FileNotFoundError(f"Reviewer selection file not found: {benchmark_reviewer_selection_file}")

    with open(benchmark_reviewer_selection_file, "r", encoding="utf-8") as f:
        cached_selection = {str(k): str(v) for k, v in json.load(f).items()}

    missing_ids = sorted(set(benchmark_records) - set(cached_selection))
    if missing_ids:
        raise ValueError(
            f"Reviewer selection file is missing {len(missing_ids)} paper ids. "
            f"Examples: {missing_ids[:10]}"
        )

    test_reviewers_dict = {
        paper_id: cached_selection[paper_id]
        for paper_id in benchmark_records
    }

    invalid_selection: List[str] = []
    for paper_id, obj in benchmark_records.items():
        selected_reviewer = test_reviewers_dict[paper_id]
        available_reviewers = {
            reviewer_name
            for reviewer_name, _ in (
                extract_review_group_text(review_group, obj["sentence_texts"])
                for review_group in obj["reviews"]
            )
            if reviewer_name is not None
        }
        if selected_reviewer not in available_reviewers:
            invalid_selection.append(f"{paper_id}:{selected_reviewer}")

    if invalid_selection:
        raise ValueError(
            f"Reviewer selection file contains {len(invalid_selection)} invalid reviewer assignments. "
            f"Examples: {invalid_selection[:10]}"
        )

    return test_reviewers_dict


def load_human_full_text_data(
    benchmark_records: Dict[str, Dict[str, Any]],
    benchmark_files: List[Path]
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    print("Loading human full-text data...")

    test_reviewers_dict = left_out_one(benchmark_files)
    id_review_full_dict_human_test: Dict[str, str] = {}
    id_review_full_dict_human_ref: Dict[str, List[str]] = {}

    for paper_id, obj in benchmark_records.items():
        test_reviewer = test_reviewers_dict[paper_id]
        paper_ref_texts: List[str] = []
        for review_group in obj["reviews"]:
            reviewer_name, full_text = extract_review_group_text(
                review_group,
                obj["sentence_texts"],
            )
            if reviewer_name == test_reviewer:
                id_review_full_dict_human_test[paper_id] = full_text
            elif reviewer_name is not None:
                paper_ref_texts.append(full_text)

        if paper_id not in id_review_full_dict_human_test:
            raise ValueError(f"Missing test reviewer full text for paper {paper_id}")
        if not paper_ref_texts:
            raise ValueError(f"Missing reference full text for paper {paper_id}")
        id_review_full_dict_human_ref[paper_id] = paper_ref_texts

    print(
        f"Loaded full-text for {len(id_review_full_dict_human_test)} test papers "
        f"and {len(id_review_full_dict_human_ref)} ref papers."
    )
    return id_review_full_dict_human_test, id_review_full_dict_human_ref


def load_human_review_data(
    benchmark_records: Dict[str, Dict[str, Any]],
    benchmark_files: List[Path],
) -> Dict[str, Dict[str, Any]]:
    def _is_correct(opinion_idx: int, conflicts_validation: List[Any], rebuttal_validation: List[Any]) -> bool:
        return (
            opinion_idx < len(conflicts_validation)
            and conflicts_validation[opinion_idx] == "correct"
            and opinion_idx < len(rebuttal_validation)
            and rebuttal_validation[opinion_idx] == "correct"
        )

    def _opinion_text_len(opinion: Any, sentence_texts: List[str]) -> int:
        return len(build_reviewer_opinion_text(opinion, sentence_texts))

    def _role_bucket_strict(opinion: Any, test_reviewer: str) -> str:
        for item in opinion["data"]:
            role = item["role"]
            match = re.match(r"(Reviewer \d+)", role)
            if match:
                return "test" if match.group(1) == test_reviewer else "ref"
        return "skip"

    print("Loading human review data...")

    test_reviewers_dict = left_out_one(benchmark_files)
    id_review_category_dict_human_ref: Dict[str, Dict[str, Any]] = {}

    for paper_id, obj in benchmark_records.items():
        conflicts_validation = obj["conflicts_validation"]
        rebuttal_validation = obj["rebuttal_validation"]
        opinions = obj["opinions"]
        opinion_groups = obj["opinion_groups"]
        sentence_texts = obj["sentence_texts"]
        test_reviewer = test_reviewers_dict[paper_id]

        keep_opinion: List[int] = []
        error_opinion: List[int] = []
        for group in opinion_groups:
            assert isinstance(group, list) and group, f"Invalid opinion group for paper {paper_id}: {group}"

            ref_candidates: List[int] = []
            for opinion_idx in group:
                bucket = _role_bucket_strict(opinions[opinion_idx], test_reviewer)
                if bucket != "ref":
                    continue

                if _is_correct(opinion_idx, conflicts_validation, rebuttal_validation):
                    ref_candidates.append(opinion_idx)
                else:
                    error_opinion.append(opinion_idx)

            if ref_candidates:
                keep_ref_idx = max(ref_candidates, key=lambda idx: _opinion_text_len(opinions[idx], sentence_texts))
                keep_opinion.append(keep_ref_idx)

        def _build_paper_dict(opinion_ids: List[int]) -> Tuple[Dict[str, List[str]], List[str]]:
            category_dict_ref: Dict[str, List[str]] = {}
            paper_texts: List[str] = []

            for opinion_idx in opinion_ids:
                opinion = opinions[opinion_idx]
                category_tags = opinion["category"]
                opinion_text = build_reviewer_opinion_text(opinion, sentence_texts)
                if not opinion_text:
                    continue

                keep_for_paper = False

                for category_tag in category_tags:
                    if category_tag == "N/A":
                        continue
                    if is_figure_comment(category_tag, opinion_text):
                        continue
                    if category_tag not in category_dict_ref:
                        category_dict_ref[category_tag] = []
                    category_dict_ref[category_tag].append(opinion_text)
                    keep_for_paper = True

                if keep_for_paper:
                    paper_texts.append(opinion_text)

            return category_dict_ref, paper_texts

        category_dict_correct_ref, paper_correct_texts = _build_paper_dict(keep_opinion)
        category_dict_incorrect_ref, paper_incorrect_texts = _build_paper_dict(error_opinion)

        id_review_category_dict_human_ref[paper_id] = {
            "category_correct": category_dict_correct_ref,
            "paper_correct": "\n\n".join(paper_correct_texts),
            "category_incorrect": category_dict_incorrect_ref,
            "paper_incorrect": "\n\n".join(paper_incorrect_texts),
        }

    return id_review_category_dict_human_ref


def compute_full_text_metrics(hyps: List[str], refs: List[List[str]]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    print(f"Computing metrics for {len(hyps)} samples...")
    assert len(hyps) == len(refs), "The number of hypotheses and references must match."
    assert importlib.util.find_spec("nltk") is not None, "nltk is required for BLEU evaluation."
    assert importlib.util.find_spec("rouge_score") is not None, "rouge_score is required for ROUGE evaluation."
    assert importlib.util.find_spec("bert_score") is not None, "bert_score is required for BertScore evaluation."

    from bert_score import score as bert_score_fn
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from rouge_score import rouge_scorer

    results_detail: Dict[str, List[float]] = {
        "bleu": [],
        "rougeL": [],
        "bertscore_precision": [],
        "bertscore_recall": [],
        "bertscore_f1": [],
    }

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    smoothing = SmoothingFunction().method1

    for hyp, ref_list in tqdm(zip(hyps, refs), total=len(hyps), desc="BLEU/ROUGE"):
        max_bleu = 0.0
        hyp_tokens = hyp.split()
        for ref in ref_list:
            ref_tokens = ref.split()
            bleu_score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)
            max_bleu = max(max_bleu, bleu_score)
        results_detail["bleu"].append(max_bleu)

        max_rouge_l = 0.0
        for ref in ref_list:
            rouge_scores = scorer.score(ref, hyp)
            max_rouge_l = max(max_rouge_l, rouge_scores["rougeL"].fmeasure)
        results_detail["rougeL"].append(max_rouge_l)

    precision_scores, recall_scores, f1_scores = bert_score_fn(hyps, refs, lang="en", verbose=True)
    results_detail["bertscore_precision"] = precision_scores.tolist()
    results_detail["bertscore_recall"] = recall_scores.tolist()
    results_detail["bertscore_f1"] = f1_scores.tolist()

    avg_metrics = {key: float(np.mean(values)) if values else 0.0 for key, values in results_detail.items()}
    return avg_metrics, results_detail


def compute_table_average_from_scores(judge_scores: Dict[str, Any]) -> Optional[float]:
    table_scores = [
        float(judge_scores[metric_name])
        for metric_name in table_judge_metric_map.values()
        if is_numeric_score(judge_scores.get(metric_name))
    ]
    if not table_scores:
        return None
    return sum(table_scores) / len(table_scores)


def is_numeric_score(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def extract_numeric_metrics(metric_dict: Dict[str, Any], metric_map: Dict[str, str]) -> Dict[str, float]:
    return {
        display_name: float(metric_dict[metric_name])
        for display_name, metric_name in metric_map.items()
        if is_numeric_score(metric_dict.get(metric_name))
    }


def build_metric_summary_from_records(metric_records: List[Dict[str, float]]) -> Dict[str, Any]:
    metric_values: Dict[str, List[float]] = defaultdict(list)
    for metric_record in metric_records:
        for metric_name, metric_value in metric_record.items():
            metric_values[metric_name].append(float(metric_value))

    return {
        "item_count": len(metric_records),
        "metric_counts": {
            metric_name: len(values)
            for metric_name, values in sorted(metric_values.items())
        },
        "metrics": {
            metric_name: sum(values) / len(values)
            for metric_name, values in sorted(metric_values.items())
        },
    }


def build_metric_summary_from_average(metric_record: Dict[str, float], item_count: int) -> Dict[str, Any]:
    return {
        "item_count": int(item_count),
        "metric_counts": {
            metric_name: int(item_count)
            for metric_name in sorted(metric_record)
        },
        "metrics": {
            metric_name: float(metric_record[metric_name])
            for metric_name in sorted(metric_record)
        },
    }


def build_old_metric_record(metric_dict: Dict[str, Any]) -> Dict[str, float]:
    return extract_numeric_metrics(metric_dict, table_old_metric_map)


def build_judge_metric_record(judge_scores: Dict[str, Any], average_label: str) -> Dict[str, float]:
    metric_record = extract_numeric_metrics(judge_scores, table_judge_metric_map)
    table_average = compute_table_average_from_scores(judge_scores)
    if table_average is not None:
        metric_record[average_label] = table_average
    return metric_record


def get_major_category_tag(category_tag: str) -> str:
    return category_tag.split("-", 1)[0] if "-" in category_tag else category_tag


def get_ai_category_set(category_dict: Dict[str, List[str]]) -> List[str]:
    return sorted(
        category_tag
        for category_tag, texts in category_dict.items()
        if category_tag != "N/A" and any(text.strip() for text in texts)
    )


def load_human_complete_category_sets(
    human_classify_file: Path,
    input_ids: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    human_category_sets: Dict[str, List[str]] = {}
    target_ids = {str(paper_id) for paper_id in input_ids} if input_ids is not None else None

    for obj in iter_jsonl_objects(human_classify_file):
        paper_id = str(obj["id"])
        if target_ids is not None and paper_id not in target_ids:
            continue

        category_set = set()
        labels_by_opinion = obj.get("category", [])
        for opinion_categories in labels_by_opinion:
            for category_tag in opinion_categories:
                if not category_tag or category_tag == "N/A":
                    continue
                category_set.add(category_tag)

        human_category_sets[paper_id] = category_set

    if target_ids is not None:
        missing_ids = sorted(target_ids - set(human_category_sets))
        if missing_ids:
            raise KeyError(
                "Missing papers in complete human classify file: "
                + ", ".join(missing_ids[:10])
                + ("..." if len(missing_ids) > 10 else "")
            )

    return human_category_sets


def attach_category_coverage_details(
    processed_result_category: Dict[str, Dict[str, Any]],
    ai_category_data: Dict[str, Dict[str, List[str]]],
    human_category_sets: Dict[str, List[str]],
) -> Dict[str, Dict[str, Any]]:
    coverage_details: Dict[str, Dict[str, Any]] = {}

    for paper_id, paper_result in processed_result_category.items():
        ai_categories = set(get_ai_category_set(ai_category_data[paper_id]))
        if "N/A" in ai_categories:
            ai_categories.remove("N/A")
        human_categories = set(human_category_sets[paper_id])
        if not human_categories:
            raise ValueError(f"No human full-score categories found for paper {paper_id}")

        only_ref_tag = sorted(human_categories - ai_categories)
        only_test_tag = sorted(ai_categories - human_categories)
        paper_result["only_ref_tag"] = only_ref_tag
        paper_result["only_test_tag"] = only_test_tag

        coverage_details[paper_id] = {
            "ai_categories": sorted(ai_categories),
            "human_categories": sorted(human_categories),
            "only_ref_tag": only_ref_tag,
            "only_test_tag": only_test_tag,
            "complete_score": len(ai_categories) / len(human_categories) * 100.0,
        }

    return coverage_details


def summarize_category_coverage(
    processed_result_category: Dict[str, Dict[str, Any]],
    ai_category_data: Dict[str, Dict[str, List[str]]],
    human_category_sets: Dict[str, List[str]],
) -> Dict[str, Any]:
    coverage_details = attach_category_coverage_details(
        processed_result_category,
        ai_category_data,
        human_category_sets,
    )
    complete_scores = [paper_data["complete_score"] for paper_data in coverage_details.values()]
    average_complete_score = sum(complete_scores) / len(complete_scores) if complete_scores else None
    return {
        "item_count": len(complete_scores),
        "metric_counts": {"Complete.": len(complete_scores)} if complete_scores else {},
        "metrics": {"Complete.": average_complete_score} if average_complete_score is not None else {},
    }


def summarize_subcategory_complete_scores(coverage_details: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    human_counts: Dict[str, int] = defaultdict(int)
    ai_counts: Dict[str, int] = defaultdict(int)
    paper_count = len(coverage_details)

    for paper_data in coverage_details.values():
        for category_tag in paper_data["human_categories"]:
            human_counts[category_tag] += 1
        for category_tag in paper_data["ai_categories"]:
            ai_counts[category_tag] += 1

    subcategory_scores: Dict[str, Any] = {}
    for category_tag in sorted(human_counts):
        human_count = human_counts[category_tag]
        ai_count = ai_counts.get(category_tag, 0)
        subcategory_scores[category_tag] = {
            "paper_count": paper_count,
            "human_paper_count": human_count,
            "ai_paper_count": ai_count,
            "metric_counts": {"Complete.": human_count},
            "metrics": {"Complete.": (ai_count / human_count * 100.0) if human_count else None},
        }

    return subcategory_scores


def summarize_category_metric_breakdown(processed_result_category: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    all_metric_records: List[Dict[str, float]] = []
    records_by_major_category: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    records_by_subcategory: Dict[str, List[Dict[str, float]]] = defaultdict(list)

    for paper_data in processed_result_category.values():
        for classify_tag, category_data in paper_data["category_scores"].items():
            judge_scores = category_data["judge_scores"]
            if not judge_scores.get("success"):
                continue

            metric_record = build_judge_metric_record(judge_scores, average_label="Average")
            if not metric_record:
                continue

            all_metric_records.append(metric_record)
            records_by_major_category[get_major_category_tag(classify_tag)].append(metric_record)
            records_by_subcategory[classify_tag].append(metric_record)

    return {
        "overall": build_metric_summary_from_records(all_metric_records),
        "by_major_category": {
            category_tag: build_metric_summary_from_records(metric_records)
            for category_tag, metric_records in sorted(records_by_major_category.items())
        },
        "by_subcategory": {
            category_tag: build_metric_summary_from_records(metric_records)
            for category_tag, metric_records in sorted(records_by_subcategory.items())
        },
    }


def summarize_category_results(processed_result_category: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Optional[float]], Dict[str, int]]:
    paper_scores_by_column: Dict[str, List[float]] = {column_name: [] for column_name in list(table_judge_metric_map) + ["Average"]}

    for paper_data in processed_result_category.values():
        category_scores = paper_data["category_scores"]
        per_paper_values: Dict[str, List[float]] = {column_name: [] for column_name in paper_scores_by_column}

        for category_data in category_scores.values():
            judge_scores = category_data["judge_scores"]
            if not judge_scores["success"]:
                continue

            for column_name, metric_name in table_judge_metric_map.items():
                score = judge_scores[metric_name]
                if is_numeric_score(score):
                    per_paper_values[column_name].append(float(score))

            table_average = compute_table_average_from_scores(judge_scores)
            if table_average is not None:
                per_paper_values["Average"].append(table_average)

        for column_name, values in per_paper_values.items():
            if values:
                paper_scores_by_column[column_name].append(sum(values) / len(values))

    summary_scores = {
        column_name: (sum(values) / len(values) if values else None)
        for column_name, values in paper_scores_by_column.items()
    }
    score_counts = {column_name: len(values) for column_name, values in paper_scores_by_column.items()}
    return summary_scores, score_counts


def summarize_full_judge_results(processed_result_full_judge: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Optional[float]], Dict[str, int]]:
    paper_scores_by_column: Dict[str, List[float]] = {column_name: [] for column_name in list(table_judge_metric_map) + ["Paper."]}

    for paper_data in processed_result_full_judge.values():
        judge_scores = paper_data["judge_scores"]
        if not judge_scores["success"]:
            continue

        for column_name, metric_name in table_judge_metric_map.items():
            score = judge_scores[metric_name]
            if is_numeric_score(score):
                paper_scores_by_column[column_name].append(float(score))

        table_average = compute_table_average_from_scores(judge_scores)
        if table_average is not None:
            paper_scores_by_column["Paper."].append(table_average)

    summary_scores = {
        column_name: (sum(values) / len(values) if values else None)
        for column_name, values in paper_scores_by_column.items()
    }
    score_counts = {column_name: len(values) for column_name, values in paper_scores_by_column.items()}
    return summary_scores, score_counts


def build_table_1_summary(
    input_label: str,
    processed_result_full: Optional[Dict[str, Any]],
    processed_result_category: Optional[Dict[str, Any]],
    processed_result_full_judge: Optional[Dict[str, Any]],
    ai_category_data: Optional[Dict[str, Dict[str, List[str]]]],
    complete_human_category_sets: Optional[Dict[str, List[str]]],
) -> Dict[str, Any]:
    absolute_scores: Dict[str, Optional[float]] = {
        column_name: None
        for column_name in [
            "BLEU",
            "ROUGE-L",
            "BERT.",
            "Correct.",
            "Thoro.",
            "Ground.",
            "Verify.",
            "Clarity",
            "Average",
            "Paper.",
            "Complete.",
        ]
    }
    paper_counts = {column_name: 0 for column_name in absolute_scores}
    category_coverage = None

    if processed_result_full is not None:
        avg_metrics = processed_result_full["average"]
        detail_count = int(processed_result_full["count"])
        for column_name, metric_name in table_old_metric_map.items():
            metric_value = avg_metrics[metric_name]
            if is_numeric_score(metric_value):
                absolute_scores[column_name] = float(metric_value) * 100.0
                paper_counts[column_name] = detail_count

    if processed_result_category is not None:
        category_scores, category_counts = summarize_category_results(processed_result_category)
        for column_name in table_judge_metric_map:
            absolute_scores[column_name] = category_scores[column_name]
            paper_counts[column_name] = category_counts[column_name]
        absolute_scores["Average"] = category_scores["Average"]
        paper_counts["Average"] = category_counts["Average"]

        if ai_category_data is None:
            raise ValueError("ai_category_data is required to compute category coverage.")
        if complete_human_category_sets is None:
            raise ValueError("complete_human_category_sets is required to compute category coverage.")
        category_coverage = summarize_category_coverage(
            processed_result_category,
            ai_category_data,
            complete_human_category_sets,
        )
        absolute_scores["Complete."] = category_coverage["metrics"].get("Complete.")
        paper_counts["Complete."] = category_coverage["metric_counts"].get("Complete.", 0)

    if processed_result_full_judge is not None:
        full_judge_scores, full_judge_counts = summarize_full_judge_results(processed_result_full_judge)
        absolute_scores["Paper."] = full_judge_scores["Paper."]
        paper_counts["Paper."] = full_judge_counts["Paper."]

    return {
        "input_label": input_label,
        "absolute_scores": absolute_scores,
        "paper_counts": paper_counts,
        "category_coverage": category_coverage,
    }


def summarize_full_judge_metric_details(processed_result_full_judge: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    metric_records: List[Dict[str, float]] = []

    for paper_data in processed_result_full_judge.values():
        judge_scores = paper_data["judge_scores"]
        if not judge_scores.get("success"):
            continue

        metric_record = build_judge_metric_record(judge_scores, average_label="Paper.")
        if metric_record:
            metric_records.append(metric_record)

    return build_metric_summary_from_records(metric_records)


def build_output_summary(
    input_label: str,
    processed_result_full: Optional[Dict[str, Any]],
    processed_result_category: Optional[Dict[str, Any]],
    processed_result_category_incorrect: Optional[Dict[str, Any]],
    processed_result_full_judge: Optional[Dict[str, Any]],
    ai_category_data: Optional[Dict[str, Dict[str, List[str]]]],
    benchmark_records: Dict[str, Dict[str, Any]],
    complete_human_category_sets: Optional[Dict[str, List[str]]],
    table_1_summary: Dict[str, Any],
) -> Dict[str, Any]:
    overall_metrics: Dict[str, Any] = {}
    category_breakdown: Dict[str, Any] = {}

    if processed_result_full is not None:
        full_text_metric_record = build_old_metric_record(processed_result_full["average"])
        overall_metrics["full_text"] = build_metric_summary_from_average(
            full_text_metric_record,
            processed_result_full["count"],
        )

    if processed_result_category is not None:
        if ai_category_data is None:
            raise ValueError("ai_category_data is required to summarize category-level outputs.")
        if complete_human_category_sets is None:
            raise ValueError("complete_human_category_sets is required to summarize category-level outputs.")

        category_summary = summarize_category_metric_breakdown(processed_result_category)
        category_summary["coverage"] = summarize_category_coverage(
            processed_result_category,
            ai_category_data,
            complete_human_category_sets,
        )
        coverage_details = attach_category_coverage_details(
            processed_result_category,
            ai_category_data,
            complete_human_category_sets,
        )
        subcategory_complete_scores = summarize_subcategory_complete_scores(coverage_details)
        overall_metrics["category_correct"] = {
            **category_summary["overall"],
            "coverage": category_summary["coverage"],
        }
        by_subcategory = dict(category_summary["by_subcategory"])
        for category_tag, coverage_summary in subcategory_complete_scores.items():
            if category_tag in by_subcategory:
                by_subcategory[category_tag]["coverage"] = coverage_summary
            else:
                by_subcategory[category_tag] = {
                    **build_metric_summary_from_records([]),
                    "coverage": coverage_summary,
                }
        category_breakdown["category_correct"] = {
            "by_major_category": category_summary["by_major_category"],
            "by_subcategory": by_subcategory,
        }

    if processed_result_category_incorrect is not None:
        incorrect_summary = summarize_category_metric_breakdown(processed_result_category_incorrect)
        overall_metrics["category_incorrect"] = incorrect_summary["overall"]
        category_breakdown["category_incorrect"] = {
            "by_major_category": incorrect_summary["by_major_category"],
            "by_subcategory": incorrect_summary["by_subcategory"],
        }

    if processed_result_full_judge is not None:
        overall_metrics["paper_level"] = summarize_full_judge_metric_details(processed_result_full_judge)

    return {
        "input_label": input_label,
        "overall_metrics": overall_metrics,
        "category_breakdown": category_breakdown,
        "table_1_summary": table_1_summary,
    }


def print_table_1_summary(table_1_summary: Dict[str, Any]) -> None:
    absolute_scores = table_1_summary["absolute_scores"]
    input_label = table_1_summary["input_label"]
    columns = list(absolute_scores)

    def _format_score(value: Optional[float]) -> str:
        return "N/A" if value is None else f"{value:.2f}"

    print("\n" + "=" * 140)
    print("Table 1 Summary (absolute scores)")
    print("=" * 140)
    print("Model".ljust(24) + "".join(column_name.rjust(11) for column_name in columns))
    print(input_label[:24].ljust(24) + "".join(_format_score(absolute_scores[column_name]).rjust(11) for column_name in columns))
    print("=" * 140)


def submit_judge_task(
    judge_client: LLMClient,
    ref_text: str,
    test_text: str,
    clear_cache: bool = False,
) -> str:
    return judge_client.submit_task(
        f"REFERENCE:\n{ref_text}\n\nCANDIDATE:\n{test_text}",
        system_prompt=get_judge_system_prompt(),
        temperature=0,
        reasoning_effort="medium",
        clear_cache=clear_cache,
    )


def get_judge_result(judge_client: LLMClient, task_key: str) -> Dict[str, Any]:
    result = judge_client.get_result(task_key)
    judge_resp = result["content"]
    if not isinstance(judge_resp, str) or not judge_resp.strip():
        raise ValueError(f"LLM judge returned invalid content for task {task_key}")
    parsed_scores = parse_llm_json_response(judge_resp)
    scores = calculate_scores_from_parsed(parsed_scores)
    return {**scores, "raw_response": judge_resp, "success": True}


def submit_judge_task_with_retry(
    judge_client: LLMClient,
    ref_text: str,
    test_text: str,
    context: str,
    max_attempts: int = 3,
) -> Optional[str]:
    for attempt in range(1, max_attempts + 1):
        try:
            return submit_judge_task(
                judge_client,
                ref_text,
                test_text
            )
        except Exception as exc:
            print(f"\nLLM judge submission failed for {context} (attempt {attempt}/{max_attempts}): {exc}")
    print(f"Skipping {context} after {max_attempts} failed submission attempts.")
    return None


def get_judge_result_with_retry(
    judge_client: LLMClient,
    task_key: str,
    ref_text: str,
    test_text: str,
    context: str,
    max_attempts: int = 3,
) -> Optional[Dict[str, Any]]:
    current_task_key: Optional[str] = task_key
    for attempt in range(1, max_attempts + 1):
        try:
            if current_task_key is None:
                current_task_key = submit_judge_task(
                    judge_client,
                    ref_text,
                    test_text
                )
            return get_judge_result(judge_client, current_task_key)
        except Exception as exc:
            print(f"\nLLM judge failed for {context} (attempt {attempt}/{max_attempts}): {exc}")
            current_task_key = None
    print(f"Skipping {context} after {max_attempts} failed judge attempts.")
    return None


def evaluate_categories_batch(
    id_review_category_dict_ref: Dict[str, Dict[str, Any]],
    id_review_category_dict_test: Dict[str, Dict[str, List[str]]],
    judge_client: LLMClient,
    ref_label_for_eval: str = "category_correct",
) -> Dict[str, Dict[str, Any]]:
    print("\nStarting category-level judging")
    print("=" * 80)
    if ref_label_for_eval not in ("category_correct", "category_incorrect"):
        raise ValueError("ref_label_for_eval must be 'category_correct' or 'category_incorrect'")

    print("Phase 1: preparing and submitting all judge tasks...")
    jobs = []
    processed_result_category: Dict[str, Dict[str, Any]] = {}
    skipped_submission: List[str] = []
    skipped_judge: List[str] = []

    for paper_id in tqdm(id_review_category_dict_ref, desc="Preparing category-level tasks"):
        processed_result_category[paper_id] = {
            "category_scores": {},
            "only_ref_tag": [],
            "only_test_tag": [],
        }
        ref_dict = id_review_category_dict_ref[paper_id][ref_label_for_eval]
        test_dict = id_review_category_dict_test[paper_id]

        for classify_tag in ref_dict:
            if classify_tag == "N/A":
                continue
            if classify_tag in test_dict:
                ref_parts = "\n\n".join(ref_dict[classify_tag])
                test_parts = "\n\n".join(test_dict[classify_tag])
                assert ref_parts.strip(), f"Empty reference text for category '{classify_tag}' in paper '{paper_id}'"
                assert test_parts.strip(), f"Empty candidate text for category '{classify_tag}' in paper '{paper_id}'"
                context = f"category '{classify_tag}' in ID '{paper_id}'"
                task_key = submit_judge_task_with_retry(judge_client, ref_parts, test_parts, context)
                if task_key is None:
                    skipped_submission.append(context)
                    continue

                jobs.append(
                    {
                        "paper_id": paper_id,
                        "classify_tag": classify_tag,
                        "ref_parts": ref_parts,
                        "test_parts": test_parts,
                        "task_key": task_key,
                        "context": context,
                    }
                )

    print(f"Total tasks submitted: {len(jobs)}")
    if skipped_submission:
        print(f"Category-level submission skips: {len(skipped_submission)}")
        print(f"  Examples (up to 10): {skipped_submission[:10]}")
    print("\nPhase 2: collecting all judge results...")

    for job in tqdm(jobs, desc="Getting category-level results"):
        paper_id = job["paper_id"]
        classify_tag = job["classify_tag"]
        ref_parts = job["ref_parts"]
        test_parts = job["test_parts"]
        judge_result = get_judge_result_with_retry(
            judge_client,
            job["task_key"],
            ref_parts,
            test_parts,
            job["context"],
        )
        if judge_result is None:
            skipped_judge.append(job["context"])
            continue

        processed_result_category[paper_id]["category_scores"][classify_tag] = {
            "ref_text": ref_parts,
            "test_text": test_parts,
            "judge_scores": judge_result,
        }

    if skipped_judge:
        print(f"\nCategory-level result skips after retries: {len(skipped_judge)}")
        print(f"  Examples (up to 10): {skipped_judge[:10]}")
    print(f"\nProcessed {len(processed_result_category)} papers for category-level evaluation")
    return processed_result_category


def evaluate_full_judge_batch(
    id_review_full_dict_ref: Dict[str, Dict[str, Any]],
    id_review_full_dict_test: Dict[str, str],
    judge_client: LLMClient,
    ref_label_for_eval: str,
) -> Dict[str, Dict[str, Any]]:
    print("\nStarting full-text judge evaluation (full_judge)")
    print("=" * 80)

    print("Phase 1: preparing and submitting all judge tasks...")
    jobs = []
    processed_result_full_judge: Dict[str, Dict[str, Any]] = {}
    skipped_empty_text: List[str] = []
    skipped_submission: List[str] = []
    skipped_judge: List[str] = []

    for paper_id in tqdm(id_review_full_dict_test, desc="Preparing full-judge tasks"):
        if paper_id not in id_review_full_dict_ref:
            raise KeyError(f"Missing ref_text for paper {paper_id}")

        ref_text = id_review_full_dict_ref[paper_id][ref_label_for_eval]
        test_text = id_review_full_dict_test[paper_id]
        ref_text = ref_text if isinstance(ref_text, str) else ""
        test_text = test_text if isinstance(test_text, str) else ""
        if not ref_text.strip() or not test_text.strip():
            context = f"ID '{paper_id}'"
            print(f"[full_judge] Skipping {context}: empty ref/test text.")
            skipped_empty_text.append(context)
            continue

        context = f"ID '{paper_id}'"
        task_key = submit_judge_task_with_retry(judge_client, ref_text, test_text, context)
        if task_key is None:
            skipped_submission.append(context)
            continue
        jobs.append(
            {
                "paper_id": paper_id,
                "ref_text": ref_text,
                "test_text": test_text,
                "task_key": task_key,
                "context": context,
            }
        )

    print(f"Total tasks submitted: {len(jobs)}")
    if skipped_empty_text or skipped_submission:
        print("\n[full_judge] Submission skips (explaining why tasks < papers):")
        if skipped_empty_text:
            print(f"  - Empty ref/test text: {len(skipped_empty_text)}")
            print(f"    Examples (up to 10): {skipped_empty_text[:10]}")
        if skipped_submission:
            print(f"  - API submission failed after retries: {len(skipped_submission)}")
            print(f"    Examples (up to 10): {skipped_submission[:10]}")

    print("\nPhase 2: collecting all judge results...")
    for job in tqdm(jobs, desc="Getting full-judge results"):
        paper_id = job["paper_id"]
        ref_text = job["ref_text"]
        test_text = job["test_text"]
        judge_result = get_judge_result_with_retry(
            judge_client,
            job["task_key"],
            ref_text,
            test_text,
            job["context"],
        )
        if judge_result is None:
            skipped_judge.append(job["context"])
            continue

        processed_result_full_judge[paper_id] = {
            "ref_text": ref_text,
            "test_text": test_text,
            "judge_scores": judge_result,
        }

    if skipped_judge:
        print(f"\n[full_judge] Result skips after retries: {len(skipped_judge)}")
        print(f"  Examples (up to 10): {skipped_judge[:10]}")
    print(f"\nProcessed {len(processed_result_full_judge)} papers for full-judge evaluation")
    return processed_result_full_judge


def save_evaluation_results(
    output_file: Path,
    output_summary: Dict[str, Any],
) -> None:
    print("\n" + "=" * 80)
    print("Saving evaluation results")
    print("=" * 80)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_summary, f, ensure_ascii=False, indent=2)
    print(f"Saved summary results to: {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Review Evaluation Script")
    parser.add_argument("--input_file", type=str, required=True, help="Processed AI-review classify JSONL produced by src/evaluation/_3_classify.py.")
    parser.add_argument("--output_file", type=str, default="benchmark/benchmark_left_out_human_result.jsonl", help="Evaluation summary JSON path.")
    parser.add_argument("--benchmark_file", type=str, nargs="+", default=["benchmark/benchmark_*_20*.jsonl"], help="Benchmark JSONL file path(s) or filename regex pattern(s).")
    parser.add_argument(
        "--complete_human_classify_file",
        type=str,
        default="benchmark/benchmark_all_human_classify.jsonl",
        help="Human classify JSONL used as the denominator source for Complete.",
    )
    parser.add_argument("--model", type=str, default="gpt-5-mini", help="Judge model name.")
    parser.add_argument("--base_url", type=str, default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"), help="Judge API base URL.")
    parser.add_argument("--api_key", type=str, default=os.environ.get("OPENAI_API_KEY", ""), help="Judge API key.")
    parser.add_argument("--max_workers", type=int, default=128, help="Max worker threads for judge requests.")
    parser.add_argument("--cache_version", type=str, default="v1", help="Cache version for judge requests.")
    parser.add_argument("--effort", type=str, default="medium", help="Reasoning effort for the judge model.")
    parser.add_argument("--category_level_eval", action="store_true", help="Run category-level evaluation.")
    parser.add_argument("--category_incorrect_eval", action="store_true", help="Run incorrect-category evaluation.")
    parser.add_argument("--calc_old_metrics", action="store_true", help="Run full-text metrics evaluation.")
    parser.add_argument("--paper_level_eval", action="store_true", help="Run full-text judge evaluation.")

    args = parser.parse_args()

    if not any([args.category_level_eval, args.category_incorrect_eval, args.calc_old_metrics, args.paper_level_eval]):
        args.category_level_eval = True
        args.calc_old_metrics = True
        args.paper_level_eval = True
        print("No evaluation flags were provided. Enabled Table-1 evaluations: old metrics, category-level, and paper-level.")

    input_file = Path(args.input_file).expanduser().resolve()
    output_file = Path(args.output_file).expanduser().resolve()
    complete_human_classify_file = Path(args.complete_human_classify_file).expanduser().resolve()
    benchmark_files = resolve_benchmark_files(args.benchmark_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if not complete_human_classify_file.exists():
        raise FileNotFoundError(f"Complete human classify file not found: {complete_human_classify_file}")

    input_ids: List[str] = []
    for obj in iter_jsonl_objects(input_file):
        paper_id = obj.get("id")
        if paper_id is not None:
            input_ids.append(str(paper_id))
    if not input_ids:
        raise ValueError(f"No paper ids found in input file: {input_file}")

    benchmark_records = load_benchmark_records(input_ids, benchmark_files)
    complete_human_category_sets = (
        load_human_complete_category_sets(complete_human_classify_file, input_ids)
        if args.category_level_eval
        else None
    )

    print("=" * 80)
    print("Evaluation configuration:")
    print(f"  input_file: {input_file}")
    print(f"  output_file: {output_file}")
    print(f"  complete_human_classify_file: {complete_human_classify_file}")
    print(f"  benchmark_patterns: {args.benchmark_file}")
    print(f"  benchmark_files: {len(benchmark_files)}")
    print(f"  benchmark_file_examples: {[str(path) for path in benchmark_files[:5]]}")
    print(f"  model: {args.model}")
    print(f"  base_url: {args.base_url}")
    print(f"  category_level_eval: {args.category_level_eval}")
    print(f"  category_incorrect_eval: {args.category_incorrect_eval}")
    print(f"  calc_old_metrics: {args.calc_old_metrics}")
    print(f"  paper_level_eval: {args.paper_level_eval}")
    print(f"  benchmark_records: {len(benchmark_records)}")
    print("=" * 80)

    needs_judge_client = args.category_level_eval or args.category_incorrect_eval or args.paper_level_eval
    judge_client: Optional[LLMClient] = None
    if needs_judge_client:
        judge_client = LLMClient(
            api_key=args.api_key,
            base_url=args.base_url,
            model_name=args.model,
            max_workers=args.max_workers,
            cache_version=args.cache_version,
        )

    processed_result_category = None
    processed_result_category_incorrect = None
    processed_result_full = None
    processed_result_full_judge = None
    ai_category_data: Optional[Dict[str, Dict[str, List[str]]]] = None
    ai_full_text_data: Optional[Dict[str, str]] = None
    ai_paper_level_data: Optional[Dict[str, str]] = None

    if args.calc_old_metrics:
        print("\nStarting full-text metrics evaluation (BLEU, ROUGE, BertScore)")
        ai_full_text_data = load_ai_full_text_data(str(input_file))
        _, id_review_full_dict_ref = load_human_full_text_data(benchmark_records, benchmark_files)

        eval_ids = []
        hyps = []
        refs = []
        for paper_id in id_review_full_dict_ref:
            if paper_id not in ai_full_text_data:
                continue
            eval_ids.append(paper_id)
            hyps.append(ai_full_text_data[paper_id])
            refs.append(id_review_full_dict_ref[paper_id])

        avg_metrics, _ = compute_full_text_metrics(hyps, refs)
        processed_result_full = {
            "average": avg_metrics,
            "count": len(eval_ids),
        }

        print("\nFull-text metrics (averages):")
        for key, value in avg_metrics.items():
            print(f"  {key}: {value:.4f}")

    human_ref_data: Optional[Dict[str, Dict[str, Any]]] = None
    if needs_judge_client:
        human_ref_data = load_human_review_data(benchmark_records, benchmark_files)
    if args.category_level_eval or args.category_incorrect_eval:
        ai_category_data = load_ai_review_data(str(input_file))

    if args.category_level_eval:
        processed_result_category = evaluate_categories_batch(
            human_ref_data,
            ai_category_data,
            judge_client,
            ref_label_for_eval="category_correct",
        )

    if args.category_incorrect_eval:
        processed_result_category_incorrect = evaluate_categories_batch(
            human_ref_data,
            ai_category_data,
            judge_client,
            ref_label_for_eval="category_incorrect",
        )

    if args.paper_level_eval:
        if ai_paper_level_data is None:
            ai_paper_level_data = load_ai_paper_level_data(str(input_file))
        processed_result_full_judge = evaluate_full_judge_batch(
            human_ref_data,
            ai_paper_level_data,
            judge_client,
            ref_label_for_eval="paper_correct",
        )

    table_1_summary = build_table_1_summary(
        input_file.stem,
        processed_result_full,
        processed_result_category,
        processed_result_full_judge,
        ai_category_data,
        complete_human_category_sets,
    )
    print_table_1_summary(table_1_summary)

    output_summary = build_output_summary(
        input_file.stem,
        processed_result_full,
        processed_result_category,
        processed_result_category_incorrect,
        processed_result_full_judge,
        ai_category_data,
        benchmark_records,
        complete_human_category_sets,
        table_1_summary,
    )

    save_evaluation_results(
        output_file,
        output_summary=output_summary,
    )

    if judge_client is not None:
        judge_client.shutdown()


if __name__ == "__main__":
    main()
