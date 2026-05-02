import re


def _extract_first_number(text):
    """Return the first numeric value found in text, or None."""
    if not text:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _parse_cycle_reviewer_output(generated_text, review_separator, heading_prefix):
    """Parse CycleReviewer-style output with configurable heading markers."""
    fields_config = [
        ("summary", f"{heading_prefix} Summary\n\n", heading_prefix, False),
        ("soundness", f"{heading_prefix} Soundness\n\n", heading_prefix, False),
        ("presentation", f"{heading_prefix} Presentation\n\n", heading_prefix, False),
        ("contribution", f"{heading_prefix} Contribution\n\n", heading_prefix, False),
        ("strengths", f"{heading_prefix} Strengths\n\n", heading_prefix, False),
        ("weaknesses", f"{heading_prefix} Weaknesses\n\n", heading_prefix, False),
        ("questions", f"{heading_prefix} Questions\n\n", heading_prefix, False),
        ("flag_for_ethics_review", f"{heading_prefix} Flag For Ethics Review\n\n", heading_prefix, False),
        ("rating", f"{heading_prefix} Rating\n\n", heading_prefix, True),
        ("confidence", f"{heading_prefix} Confidence\n\n", "******", False),
    ]

    reviews = []
    field_data = {name: [] for name, _, _, _ in fields_config}
    rating_values = []
    paper_decision = ""
    meta_review = ""

    for chunk in generated_text.split(review_separator):
        review = chunk.strip()
        if not review:
            continue

        if "## Paper Decision\n\n" in review:
            review, decision_part = review.split("## Paper Decision\n\n", 1)
            decision_line = decision_part.splitlines()[0] if decision_part else ""
            paper_decision = "Accept" if "accept" in decision_line.lower() else "Reject"

        if "## Meta Review\n\n" in review:
            review, meta_part = review.split("## Meta Review\n\n", 1)
            meta_review = meta_part.strip()

        required_sections = [
            f"{heading_prefix} Summary\n\n",
            f"{heading_prefix} Soundness\n\n",
            f"{heading_prefix} Presentation\n\n",
        ]
        if not all(section in review for section in required_sections):
            continue

        reviews.append(review)

        for field_name, prefix, suffix, extract_num in fields_config:
            if prefix in review:
                content = review.split(prefix, 1)[1].split(suffix, 1)[0].strip()
                field_data[field_name].append(content)
                if extract_num:
                    rating_values.append(_extract_first_number(content) or 0.0)
            else:
                field_data[field_name].append("")
                if extract_num:
                    rating_values.append(0.0)

    if not paper_decision:
        return {}

    return {
        "raw_text": generated_text,
        "reviews": reviews,
        "summary": field_data["summary"],
        "review_rate": field_data["rating"],
        "rating": rating_values,
        "soundness": field_data["soundness"],
        "presentation": field_data["presentation"],
        "contribution": field_data["contribution"],
        "strength": field_data["strengths"],
        "weaknesses": field_data["weaknesses"],
        "questions": field_data["questions"],
        "flag_for_ethics_review": field_data["flag_for_ethics_review"],
        "confidence": field_data["confidence"],
        "paper_decision": paper_decision,
        "meta_review": meta_review,
        "avg_rating": sum(rating_values) / len(rating_values) if rating_values else 0,
    }


def get_reviewer_score_7B(generated_text):
    try:
        return _parse_cycle_reviewer_output(generated_text, "**********\n", "##")
    except Exception:
        return {}


def get_reviewer_score_123B(generated_text):
    try:
        return _parse_cycle_reviewer_output(generated_text, "## Reviewer\n", "###")
    except Exception:
        return {}


def parse_review_cyclereviewer(generated_text):
    pred = get_reviewer_score_7B(generated_text)
    if not pred or not pred.get("rating") or all(value == 0 for value in pred["rating"]):
        pred = get_reviewer_score_123B(generated_text)
    return pred


def parse_review_deepreviewer(generated_text):
    """Parse DeepReviewer output into a structured dict."""
    result = {
        "raw_text": generated_text,
        "reviews": [],
        "meta_review": {},
        "decision": "",
    }

    meta_review_match = re.search(r"\\boxed_review\{(.*?)\n}", generated_text, re.DOTALL)
    if meta_review_match:
        result["meta_review"]["content"] = meta_review_match.group(1).strip()
        section = meta_review_match.group(1).strip()

        summary_match = re.search(r"## Summary:\s+(.*?)(?=##|\Z)", section, re.DOTALL)
        if summary_match:
            result["meta_review"]["summary"] = summary_match.group(1).strip()

        rating_match = re.search(r"## Rating:\s+(.*?)(?=##|\Z)", section, re.DOTALL)
        if rating_match:
            rating_text = rating_match.group(1).strip()
            result["meta_review"]["rating"] = _extract_first_number(rating_text) or rating_text

        for section_name in [
            "Soundness",
            "Presentation",
            "Contribution",
            "Strengths",
            "Weaknesses",
            "Suggestions",
            "Questions",
            "Confidence",
        ]:
            section_match = re.search(rf"## {section_name}:\s+(.*?)(?=##|\Z)", section, re.DOTALL)
            if section_match:
                result["meta_review"][section_name.lower()] = section_match.group(1).strip()

    simreviewer_match = re.search(r"\\boxed_simreviewers\{(.*?)\n}", generated_text, re.DOTALL)
    if simreviewer_match:
        simreviewer_text = simreviewer_match.group(1).strip()
        reviewer_sections = re.split(r"## Reviewer \d+", simreviewer_text)
        if reviewer_sections and not reviewer_sections[0].strip():
            reviewer_sections = reviewer_sections[1:]

        for index, section in enumerate(reviewer_sections, start=1):
            review = {
                "reviewer_id": index,
                "text": section.strip(),
            }

            summary_match = re.search(r"## Summary:\s+(.*?)(?=##|\Z)", section, re.DOTALL)
            if summary_match:
                review["summary"] = summary_match.group(1).strip()

            rating_match = re.search(r"## Rating:\s+(.*?)(?=##|\Z)", section, re.DOTALL)
            if rating_match:
                rating_text = rating_match.group(1).strip()
                review["rating"] = _extract_first_number(rating_text) or rating_text

            for section_name in [
                "Soundness",
                "Presentation",
                "Contribution",
                "Strengths",
                "Weaknesses",
                "Suggestions",
                "Questions",
            ]:
                section_match = re.search(rf"## {section_name}:\s+(.*?)(?=##|\Z)", section, re.DOTALL)
                if section_match:
                    review[section_name.lower()] = section_match.group(1).strip()

            result["reviews"].append(review)

    decision_match = re.search(r"## Decision:\s*\n\s*(\w+)", generated_text)
    if decision_match:
        result["decision"] = decision_match.group(1).strip()

    return result


def parse_review_ai_scientist(text: str):
    """
    Extract Strengths, Weaknesses, Questions, and Limitations from AI-Scientist output.

    The parser first looks for a REVIEW JSON block and falls back to section-based
    text extraction when JSON is missing or malformed.
    """
    json_content = None

    def is_valid_json_content(content):
        if not content:
            return False
        if not (
            re.search(r'["\']?Strengths["\']?', content, re.IGNORECASE)
            and re.search(r'["\']?Weaknesses["\']?', content, re.IGNORECASE)
        ):
            return False
        placeholders = ['"..."', "[...]", '"Summary": "..."']
        placeholder_count = sum(1 for placeholder in placeholders if placeholder in content)
        return placeholder_count < 2

    candidates = []

    candidates.extend(
        re.findall(r"REVIEW JSON:\s*```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    )

    for match in reversed(list(re.finditer("REVIEW JSON:", text, re.IGNORECASE))):
        after_prefix = text[match.end():].strip()
        first_brace = after_prefix.find("{")
        last_brace = after_prefix.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            candidates.append(after_prefix[first_brace:last_brace + 1])

    candidates.extend(re.findall(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE))

    all_braces = [match.start() for match in re.finditer(r"\{", text)]
    last_brace_overall = text.rfind("}")
    if last_brace_overall != -1:
        for first_brace in reversed(all_braces):
            if first_brace < last_brace_overall:
                candidates.append(text[first_brace:last_brace_overall + 1])
                if len(candidates) > 20:
                    break

    for candidate in reversed(candidates):
        if is_valid_json_content(candidate):
            json_content = candidate
            break

    results = {"Strengths": None, "Weaknesses": None, "Questions": None, "Limitations": None}

    if json_content:
        try:
            array_patterns = {
                "Strengths": r'["\']Strengths["\']:\s*\[(.*?)\]',
                "Weaknesses": r'["\']Weaknesses["\']:\s*\[(.*?)\]',
                "Questions": r'["\']Questions["\']:\s*\[(.*?)\]',
                "Limitations": r'["\']Limitations["\']:\s*\[(.*?)\]',
            }

            for field, pattern in array_patterns.items():
                match = re.search(pattern, json_content, re.DOTALL | re.IGNORECASE)
                if not match:
                    continue

                array_content = match.group(1)
                items = re.findall(r'"((?:[^"\\]|\\.)*)"', array_content)
                if not items:
                    items = re.findall(r"'((?:[^'\\]|\\.)*)'", array_content)
                if not items:
                    items = [item.strip().strip("\"'") for item in array_content.split(",") if item.strip()]

                results[field] = [item for item in items if any(char.isalnum() for char in item)]
        except Exception:
            pass

    if results["Strengths"] is None or results["Weaknesses"] is None or not results["Strengths"]:
        content_to_search = text
        if "</think>" in text:
            content_to_search = text.split("</think>")[-1]
        elif "THOUGHT:" in text:
            content_to_search = text.split("THOUGHT:")[-1]

        section_headers = (
            "Strengths|Weaknesses|Questions|Limitations|Originality|Quality|Clarity|"
            "Significance|Overall|Confidence|Decision|Soundness|Presentation|Contribution|Summary"
        )

        for field in ["Strengths", "Weaknesses", "Questions", "Limitations"]:
            if results[field]:
                continue

            pattern = (
                rf"(?:^|\n)(?:#+\s*|\*\*)?{field}(?:\*\*|:|\s*:)*\s*(.*?)"
                rf"(?=(?:\n|$)(?:#+\s*|\*\*)?(?:{section_headers})(?:\*\*|:|\s*:)*|$)"
            )
            match = re.search(pattern, content_to_search, re.DOTALL | re.IGNORECASE)
            if not match:
                continue

            field_content = match.group(1).strip()
            items = re.findall(
                r"^\s*[-*\u2022]\s*(.*)$|^\s*\d+\.\s*(.*)$",
                field_content,
                re.MULTILINE,
            )
            extracted = [
                first.strip() if first else second.strip()
                for first, second in items
                if (first and first.strip()) or (second and second.strip())
            ]

            if extracted:
                results[field] = extracted
            elif field_content:
                lines = [line.strip() for line in re.split(r"\n\n+", field_content) if line.strip()]
                if lines:
                    results[field] = lines

    return results
