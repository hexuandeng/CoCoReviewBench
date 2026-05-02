#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Purpose
-------
Provide prompt templates for review-splitting, classification, clustering, and conflict adjudication.
"""

from typing import List, Tuple, Dict, Any


def classify_reviews_prompt() -> Tuple[str, str]:
    """Return fixed classify-review prompts embedded in code (no file IO)."""
    system_prompt = "You are given a reviewer comment (and optionally the authors’ response). Your task is to classify the comment using the following taxonomy. The output should only contain the sub-category name."
    main_prompt_body = """
# Taxonomy

1. QUAL (Quality): Is the submission technically sound? Are claims well supported (e.g., by theoretical analysis or experimental results)? Are the methods used appropriate? Is this a complete piece of work or work in progress? Are the authors careful and honest about evaluating both the strengths and weaknesses of their work?
- QUAL-MET (Methodological Soundness): Evaluates whether the proposed algorithms, models or system architectures are technically correct and free of conceptual or implementation errors.
    - Strength: Highlights that the method is well-formulated, mathematically consistent and correctly implemented; no major flaws are apparent.
    - Weakness: Points out conceptual errors, mis-specified objectives, algorithmic flaws or misuses of optimisation methods.
- QUAL-EXP (Experimental Design & Evaluation): Assesses the adequacy of the experimental setup: choice of datasets, baselines, evaluation metrics, ablation studies, and statistical tests.
    - Strength: Notes comprehensive experiments, appropriate baselines, sufficiently large data and proper metrics; experiments convincingly validate claims.
    - Weakness: Points out inadequate baselines, missing standard benchmarks, inappropriate metrics or lack of ablation studies.
- QUAL-REP (Reproducibility & Implementation Details): Considers whether the submission provides enough detail (e.g., hyperparameters, code availability, dataset splits) to replicate the work and whether the implementation follows best practices.
    - Strength: Commends open-sourced code, detailed hyperparameters, clear dataset descriptions and comprehensive training details enabling easy reproduction.
    - Weakness: Notes lack of code, incomplete hyperparameters, unspecified data splits or other missing details that hinder reproduction.
- QUAL-CMP (Comparisons to Prior Work): Evaluates whether the paper sufficiently compares against relevant prior work and state-of-the-art methods.
    - Strength: Acknowledges thorough comparisons against recent and appropriate baselines and proper citation of related work.
    - Weakness: Notes missing comparisons to well-known recent methods, outdated baselines, or incomplete literature review.
- QUAL-STA (Statistical Rigor & Validation): Evaluates whether statistical analyses (e.g., significance tests, confidence intervals) are properly applied and whether reported improvements are statistically meaningful.
    - Strength: Praises appropriate statistical tests, robust validation, confidence intervals and transparent reporting of variance.
    - Weakness: Criticises absence of statistical tests, reliance on single runs, or misuse of statistical methods.

2. CLAR (Clarity): Is the submission clearly written? Is it well organized? (If not, please make constructive suggestions for improving its clarity.) Does it adequately inform the reader? (Note that a superbly written paper provides enough information for an expert reader to reproduce its results.)
- CLAR-WRT (Writing, Terminology & Algorithm Presentation): Covers overall writing quality and organization, precision of terminology and key concepts, and the clarity of algorithm presentations (code snippets, pseudocode, workflow diagrams).
    - Strength: Well-structured sections and smooth narrative; concise, readable prose; all specialized terms and assumptions are clearly defined and consistently used; pseudocode/code/workflows are self-contained, well-documented (clear variable names, inputs/outputs), and easy to follow step-by-step.
    - Weakness: Unclear or disorganized exposition, excessive jargon or redundancy; missing/ambiguous definitions or misuse of terms; confusing or incomplete pseudocode/code/workflows (undefined parameters, missing steps, opaque variable names).
- CLAR-NOT (Notation & Mathematical Explanation Clarity): Considers whether mathematical notation is consistent, well defined and explained.
    - Strength: Highlights clear definitions, consistent notation and well-explained derivations.
    - Weakness: Notes inconsistent symbols, missing definitions or unexplained mathematical steps.
- CLAR-FIG (Figures & Visual Aids Clarity): Evaluates whether figures, tables, plots and visualisations are legible, properly labelled, and aid understanding.
    - Strength: Praises informative and well-designed figures that clearly illustrate results or architectures.
    - Weakness: Points out illegible plots, missing labels, confusing colour schemes or misleading visualisations.

3. SIGN (Significance): Are the results impactful for the community? Are others (researchers or practitioners) likely to use the ideas or build on them? Does the submission address a difficult task in a better way than previous work? Does it advance our understanding/knowledge on the topic in a demonstrable way? Does it provide unique data, unique conclusions about existing data, or a unique theoretical or experimental approach?
- SIGN-BRD (Broad Research Impact): Evaluates whether the work addresses a broadly important problem or advances understanding in a way that is likely to influence multiple research areas.
    - Strength: Notes that the work tackles a significant, widely relevant challenge or introduces a paradigm that may influence many researchers.
    - Weakness: Suggests that the work addresses a niche or already well-explored problem with limited broader interest.
- SIGN-DOM (Domain/Applied Impact): Assesses the significance of the contribution for a specific application domain (e.g., NLP, robotics, healthcare).
    - Strength: Highlights how the method outperforms existing approaches or addresses an important challenge in a specialised field.
    - Weakness: Notes limited domain relevance, inadequate demonstration on domain-specific benchmarks or unconvincing domain benefits.
- SIGN-SOT (Improvement Over State of the Art): Focuses on the magnitude and importance of improvements relative to current state-of-the-art methods.
    - Strength: Praises substantial, consistent performance gains over strong baselines with appropriate statistical support.
    - Weakness: Criticises marginal or statistically insignificant improvements, or improvements only on favourable datasets.
- SIGN-IMP (Real-World & Societal Impact): Considers the potential practical and societal ramifications of the work, including benefits, risks, fairness, environmental impacts and accessibility.
    - Strength: Recognises thoughtful discussion of societal benefits, harms, fairness, environmental impact and possible mitigations.
    - Weakness: Points out a lack of consideration of societal impact or failure to discuss who benefits and who may be harmed.

4. ORIG (Originality): Does the work provide new insights, deepen understanding, or highlight important properties of existing methods? Is it clear how this work differs from previous contributions, with relevant citations provided? Does the work introduce novel tasks or methods that advance the field? Does this work offer a novel combination of existing techniques, and is the reasoning behind this combination well-articulated? As the questions above indicates, originality does not necessarily require introducing an entirely new method. Rather, a work that provides novel insights by evaluating existing methods, or demonstrates improved efficiency, fairness, etc. is also equally valuable.
- ORIG-PROB (Novel Problem Formulation): Comments on the introduction of a new problem, task, or dataset. Includes innovative problem definitions that highlight previously unaddressed challenges.
    - Strength: Highlights creative and meaningful new problem formulations or datasets that open new avenues for research.
    - Weakness: Notes that the problem is a minor variation of existing tasks or lacks clear motivation.
- ORIG-MTH (Novel Methodology or Algorithm): Assesses whether the paper introduces a genuinely new algorithmic approach or architectural design rather than a minor tweak of existing methods.
    - Strength: Recognises innovative algorithmic designs with clear conceptual advances.
    - Weakness: Points out that the method is a straightforward modification or re-implementation of existing techniques.
- ORIG-ANL (Novel Analysis or Insights): Pertains to original theoretical insights or analyses that deepen understanding of existing methods or phenomena.
    - Strength: Notes insightful analysis that sheds new light on known methods or uncovers unexpected behaviours.
    - Weakness: Suggests that the analysis is trivial, already known, or does not yield meaningful insights.
- ORIG-EXP (Novel Experimental Setup or Data): Evaluates whether the paper proposes new experimental setups, benchmarks, evaluation protocols or collects new datasets.
    - Strength: Praises introduction of meaningful new datasets, benchmarks or evaluation protocols.
    - Weakness: Notes that the experimental setup is derivative or does not add value beyond existing benchmarks.
- ORIG-COM (Creative Combination of Existing Methods): Considers whether the paper combines well-known techniques in an original way and whether the rationale behind the combination is compelling.
    - Strength: Recognises innovative synergies or architectures that combine methods in a way that yields new capabilities.
    - Weakness: Suggests that the combination is superficial or lacks a clear justification for being novel.
- ORIG-NEG (Negative Results or Critical Assessments): Covers papers that provide critical evaluations, ablations or negative results showing limitations of existing methods.
    - Strength: Commends thoughtful critical analysis or the presentation of negative results that challenge assumptions.
    - Weakness: Notes superficial or poorly justified criticism, or negative results that lack rigour.

5. POL (Policy/Compliance): Encompasses policy- or compliance-related concerns such as ethics, data/privacy compliance, anonymity rules, plagiarism, licensing, and broader impact. These issues often require checking adherence to conference or legal policies and may involve ethical considerations.
- POL-ETH (Ethics & Responsible AI Compliance): Addresses ethical concerns, including fairness, bias, harms to marginalised populations, dual use and whether the authors appropriately discuss and mitigate ethical risks.
    - Strength: Notes thorough consideration of potential harms, fairness, privacy and mitigation strategies.
    - Weakness: Flags unaddressed ethical issues, potential harms or insufficient discussion of biases.
- POL-DAT (Data Usage & Privacy Compliance): Considers whether the data used in the paper comply with privacy regulations and licensing terms (e.g., consent, personally identifiable information).
    - Strength: Praises adherence to data-use policies, proper anonymisation and appropriate licensing.
    - Weakness: Points out potential privacy violations, insufficient data consent or misuse of licensed data.
- POL-ANO (Double-Blind or Anonymity Violations): Concerns whether the submission inadvertently reveals author identities or institutional affiliations, violating double-blind review policies.
    - Strength: Confirms that the paper appropriately anonymises authors and affiliations.
    - Weakness: Highlights self-identifying text, acknowledgements or URLs that compromise anonymity.
- POL-PLG (Plagiarism & Dual Submission): Addresses issues of plagiarism, self-plagiarism, or dual submission to multiple venues.
    - Strength: Confirms originality of content, proper citations and no evidence of duplicate submission.
    - Weakness: Notes textual or figure overlap, uncredited reuse of material or simultaneous submission.
- POL-IMP (Broader Impact & Societal Considerations): Evaluates whether the authors complete required checklists and discuss the broader impact of their work on society.
    - Strength: Comprehensive and honest broader-impact statements and required checklists; documented IRB/ethics approvals where applicable; explicit and correct software/data licences (and usage within licence terms); clear consent/approval identifiers and compliance evidence.
    - Weakness: Missing or boilerplate impact statements; absent/unclear required checklists; missing IRB or ethics approval where needed; ambiguous or incompatible licensing; evidence of non-compliant use (e.g., violating dataset or code licence terms).

### Boundary Rules for Disambiguation
1. CLAR-WRT vs. QUAL-EXP: Comments about missing baselines or inadequate experiments belong under QUAL-EXP, even when poor writing makes experiments hard to follow; issues solely about readability go under CLAR-WRT.
2. CLAR-FIG vs. QUAL-EXP: Critiques about illegible or unclear plots go to CLAR-FIG; critiques about missing plots or missing baseline curves go to QUAL-EXP.
3. QUAL-REP vs. CLAR-NOT: Missing code, hyperparameters or data splits fall under QUAL-REP; undefined variables or inconsistent notation fall under CLAR-NOT.
4. SIGN-IMP vs. POL-IMP: General discussion of societal benefits or harms without mandatory checklists goes to SIGN-IMP; comments about compliance with required impact statements go to POL-IMP.
5. ORIG-EXP vs. QUAL-EXP: Introducing a new dataset or benchmark is ORIG-EXP; critiquing the thoroughness of experiments on existing datasets is QUAL-EXP.
6. POL-ETH vs. SIGN-IMP: Ethical violations, bias or harm to marginalised groups belong in POL-ETH; high-level impact comments without compliance issues belong in SIGN-IMP.
7. QUAL-CMP vs. SIGN-SOT: Missing baselines or incomplete literature reviews are QUAL-CMP; judging whether reported improvements are meaningful goes to SIGN-SOT.
8. ORIG-COM vs. ORIG-MTH: Combining existing methods in an original way is ORIG-COM; proposing entirely new algorithms is ORIG-MTH.
9. QUAL-CMP vs. QUAL-EXP: Missing/irrelevant or outdated baselines and literature mapping → QUAL-CMP; evaluation protocol design (metrics, splits, tuning fairness, test coverage) → QUAL-EXP.
10. ORIG-MTH vs. QUAL-EXP: Whether the algorithm/architecture is truly novel vs. a minor tweak → ORIG-MTH; whether experiments sufficiently validate the (novel or not) method → QUAL-EXP.

---

**Output:** Return **only** the applicable label(s). If the reviewer mentions multiple distinct comments, assign different labels, separated by commas. If the reviewer mentions only one comment, assign the most reasonable single label. Do not include any other words or explanation in the output.
"""

    return system_prompt.strip(), main_prompt_body.strip()


def split_reviews_system_prompt(num: int) -> str:
    """
    System instruction for labeling parts into atomic discussion points.
    """
    return f"""
    You are given a peer-review discussion split into {num} parts, each prefixed with "ID: {{id}}". Each part is either a reviewer comment or an author reply.

    # GOAL
    Assign a numeric label to EVERY part so that all parts about the same discussion Point share the same label.
    - Point: one atomic strength, weakness, or question raised by a reviewer.
    - Labels start at 1 ("Label 1", "Label 2", …). Use the same number for the reviewer’s Point and all replies addressing it.
    - Use "Label N/A" only for purely polite text with no technical or content value (e.g., "Thank you") or for pure paper summaries of the paper.
    - If a part is a citation (e.g., "[1] …" but clearly supports or questions a known Point, use that Point’s label. Do not use "Label N/A".

    # PROCEDURE
    1. Identify atomic Points in reviewer comments and create a new label for each.
    - If a single comment contains multiple Points, assign separate labels.
    - If a reviewer raises new Points later, give them new labels.

    2. For each author reply, assign the label(s) of the Point(s) it responds to.
    - If one reply addresses multiple Points, list all labels separated by commas (e.g., "Label 2, 5".
    - 'Strength' typically does not require a reply. Focus on matching replies to 'weaknesses' and 'questions'.

    # OUTPUT FORMAT
    Produce exactly {num} lines, one per part, in order:
    Part k: Label X
    or, if multiple:
    Part k: Label X, Y
    """


def refine_split_and_classify_system_prompt(candidate_labels: List[str], sen_num: int) -> str:
    """
    Build the system prompt for sentence classification.

    Candidate label explanations are extracted from the shared review taxonomy prompt.
    """
    try:
        from src.utils import extract_labels_from_prompt_text  # type: ignore
    except Exception:
        from utils import extract_labels_from_prompt_text  # type: ignore

    _, main_prompt_body = classify_reviews_prompt()
    _, label_explanations = extract_labels_from_prompt_text(main_prompt_body)
    label_explanations.setdefault(
        "N/A",
        "Polite text or pure paper summary; contains no substantive technical/content point.",
    )

    candidate_labels_with_explain = candidate_labels.copy()
    for lbl in candidate_labels:
        if lbl in label_explanations:
            candidate_labels_with_explain[candidate_labels.index(lbl)] = (
                f"Label: {lbl}, Explaination: {label_explanations[lbl]}"
            )

    labels_str = "\n".join(candidate_labels_with_explain)
    return f"""
You are given a list of {sen_num} sentences from a peer-review discussion, each prefixed with "ID: {{id}}".

TASK:
Classify each sentence into one or more of the following candidate labels:
{labels_str}

GUIDELINES:
- A sentence can have multiple labels if it discusses multiple aspects simultaneously
- Use only the provided candidate labels; do not invent new ones
- For all sentences within the same opinion, assign exactly the same label or set of labels to every sentence in that opinion
- Be precise and concise in your judgment

OUTPUT FORMAT:
For each of the {sen_num} sentences, output exactly one line:
Sentence k: Label X
or if multiple labels apply:
Sentence k: Label X, Y

Process exactly {sen_num} sentences in the order provided.
""".strip()

def split_ai_reviewer_prompt(num):
    return f"""
    You are given a peer-review discussion split into {num} parts, each prefixed with "Part {{id}}:". Each part is one comment in the discussion.

    # GOAL
    Assign numeric labels to EVERY part so that all parts about the same discussion Point share the same label.

    Definitions and rules:
    - A Point is one atomic strength, weakness, or question raised by a reviewer (or discussed in replies).
    - Labels are integers starting from 1 ("Label 1", "Label 2", …).
    - If a reviewer raises new Points later in the discussion, assign new labels to those Points. Try to avoid using the same label for Points that are very far apart in the discussion, even if they are related.
    - For summary-only parts that are neither strengths nor weaknesses, treat the entire summary as one single Point and assign one label to the whole summary.

    # OUTPUT FORMAT
    Produce exactly {num} lines, one per part, in input order:
    Part k: Label X
    """


def review_clustering_system_prompt() -> str:
    return (
        "You are a professional academic peer review analysis assistant. "
        "Your ONLY task is to assign discussion point IDs based on the *identical specific subject* "
        "being discussed in each opinion. Extract maximally granular topics (e.g., 'Random Forest algorithm', "
        "'SGD learning rate value'). CRITICAL: Opinions with contradictory stances on the EXACT SAME subject "
        "MUST share the same ID. Different subjects receive different IDs, regardless of opinion similarity. "
        "Focus exclusively on subject identity, not reviewer names or opinion agreement."
    )


def review_clustering_prompt(blocks: List[Dict[str, Any]]) -> str:
    """
    Build the LLM prompt for assigning discussion point IDs (reviewer texts only).

    The model is asked to:
      - extract maximally specific subjects,
      - assign stable IDs for identical subjects across blocks,
      - ignore stance (positive/negative) and reviewer identity for ID assignment.

    IMPORTANT:
      - This function intentionally only includes reviewer opinions in the prompt.
      - Do not change the prompt unless you intend to change labeling behavior.
    """
    prompt_parts = [
        "Please assign discussion point IDs to reviewer opinions based on *identical discussion content*.\n",
        "Task Requirements:",
        "1. Review ALL reviewer opinions across ALL blocks below",
        "2. Extract the *most specific core subject* from each opinion (e.g., 'Random Forest algorithm selection', 'SGD learning rate value')",
        "3. Assign an ID (starting from 1) to each *unique specific subject*",
        "4. **CRITICAL**: Reuse the same ID for opinions discussing the *exact same subject*, even if their evaluations are opposite/contradictory",
        "5. Different specific subjects receive different IDs, even if semantically related",
        "\nContent Identity Rules:",
        "- **Subject must match precisely**: 'SGD is suitable' vs. 'SGD is unsuitable' = SAME ID (identical subject: SGD suitability)",
        "- **Maximal specificity required**: Use 'F1-score metric' or 'L2 regularization strength', NOT broad terms like 'methodology' or 'evaluation'",
        "- **Different subjects, different IDs**: Comments on 'data augmentation' vs. 'train-test split' = DIFFERENT IDs",
        "- ID assignment depends *solely* on the identity of the discussed subject, completely independent of opinion stance or reviewer name",
        "",
        "",
        "Output Format (JSON):",
        "- Return a mapping of block_id -> reviewer_name -> point_id",
        "- Only include blocks that have reviewer opinions",
        "",
        "Example:",
        "```json",
        "{",
        '  "blocks": [',
        "    {",
        '      "block_id": 0,',
        '      "reviewer_assignments": {',
        '        "Reviewer 1": 1,',
        "      }",
        "    }",
        "  ]",
        "}",
        "```",
        "",
        "Text Blocks to Analyze:\n"
    ]

    for block_info in blocks:
        block_idx = block_info["block_idx"]
        prompt_parts.append(f'<block id="{block_idx}">')

        for reviewer_name, text in block_info["reviewer_texts"].items():
            if text.strip():
                prompt_parts.append(f'<reviewer name="{reviewer_name}">')
                prompt_parts.append(text)
                prompt_parts.append('</reviewer>')

        prompt_parts.append('</block>\n')

    return "\n".join(prompt_parts)


def conflict_reviews_system_prompt() -> str:
    return (
        "You are a professional academic peer review analysis assistant. "
        "Focus ONLY on detecting conflicts between reviewer opinions within each discussion point. "
        "Be precise about whether viewpoints are truly contradictory vs. complementary."
    )


def conflict_reviews_prompt(point_groups: Dict[str, List[Dict[str, Any]]]) -> str:
    """
    Build the prompt for conflict detection (grouped by point_id).
    """
    prompt_parts = [
        "Please analyze reviewer opinions within each discussion point to identify GENUINE conflicts.\n",
        "### Rules for Conflict Detection:",
        "1. **Definition of Conflict:**",
        "   - Direct contradiction on the IDENTICAL aspect (e.g., Reviewer A says 'method is novel', Reviewer B says 'method lacks novelty').",
        "   - Opposing sentiments (Positive vs. Negative) regarding the same specific feature.",
        "",
        "2. **What is NOT a Conflict (Crucial):**",
        "   - **Granularity Differences:** Specific examples (e.g., 'works on large images') vs. General statements (e.g., 'generalizes well') are COMPLEMENTARY, not conflicting.",
        "   - **Different Aspects:** Opinions on different dimensions (e.g., 'efficiency' vs 'accuracy') are not conflicts.",
        "   - **Non-Argumentative Text:** Ignore conflicts involving purely factual lists, citations, references, or formatting artifacts (e.g., 'NO.', block IDs).",
        "   - **Same Reviewer:** Opinions from the same reviewer name typically refine or list details for their own points and should not be flagged as conflicts.",
        "",
        "Task:",
        "For each discussion point ID, examine all associated reviewer opinions.",
        "Step 1: Filter out non-opinion text (citations, meta-data).",
        "Step 2: Check if opinions express opposing views on the same core issue.",
        "Step 3: Output conflict status.",
        "",
        "Output Format (JSON):",
        '- point_conflicts: {point_id: {"has_conflict": bool}}',
        "",
        "Discussion Point Groups:\n",
    ]

    for point_id, opinions in point_groups.items():
        prompt_parts.append(f'\n<discussion-point id="{point_id}">')
        for opinion in opinions:
            prompt_parts.append(
                f'<reviewer name="{opinion["reviewer"]}" block="{opinion["block_id"]}">'
            )
            prompt_parts.append(opinion["text"])
            prompt_parts.append("</reviewer>")
        prompt_parts.append("</discussion-point>")

    return "\n".join(prompt_parts)


def validate_conflict_reviews_system_prompt() -> str:
    return (
        "You are a professional academic peer review analysis assistant. "
        "Focus on resolving conflicts between opinions. Use ONLY block_id for identification. "
        "Respond with ONLY the required JSON format."
    )


def validate_conflict_reviews_prompt(
    point_id: str, opinions: List[Dict[str, Any]], meta_review: str
) -> str:
    """
    Build the prompt for conflict adjudication.

    IMPORTANT: The model must reference opinions by block_id (e.g., "block_5") only.
    """
    prompt_parts = [
        "You are an expert academic peer review analysis assistant. Your task is to resolve conflicts between reviewer opinions on a specific discussion point.\n",
        "### Conflict Resolution Guidelines:",
        "1. **Identify the Core Issue**: Determine the exact technical/academic point of disagreement.",
        "2. **Evaluate Each Position**: Assess the validity of each opinion based on logical consistency, technical soundness, and alignment with academic standards.",
        "3. **Make a Judgment**: Determine which opinion block(s) have the more correct/valid position.",
        "",
        "**Important:** Use block_id (e.g., 'block_5') to identify opinions in your response.",
        "",
        "=== META-REVIEW (Overall Context) ===",
    ]

    if meta_review.strip():
        prompt_parts.append(meta_review)
    else:
        prompt_parts.append("(No meta-review available)")

    prompt_parts.extend(["", f"--- Discussion Point {point_id} ---"])

    for opinion in opinions:
        prompt_parts.append(f"\n[block_{opinion['block_id']}] Opinion:")
        prompt_parts.append(opinion["text"])
        prompt_parts.append("")

    prompt_parts.extend(
        [
            "\n--- Output Format (JSON ONLY) ---",
            "{",
            '  "correct_blocks": ["block_X", ...],',
            '  "incorrect_blocks": ["block_Y", ...]',
            "}",
        ]
    )

    return "\n".join(prompt_parts)


def author_conflict_system_prompt() -> str:
    return (
        "You are a professional academic peer review analysis assistant. "
        "Focus on detecting if authors explicitly refute reviewer opinions. "
        "Analyze carefully but only output the JSON array as requested."
    )


def author_conflict_prompt(reviewer_name: str,
                          opinions: List[Dict],
                          author_responses_combined: str,
                          meta_review: str) -> str:
    """
    Build a batched prompt for a single reviewer's opinions.
    Includes meta-review as global context (if available).
    """
    prompt_parts = [
        "You are analyzing a peer review conversation.",
        f"Reviewer '{reviewer_name}' has {len(opinions)} opinions in this paper.",
        "Your task is to determine if the author's responses explicitly refute each opinion.",
        "",
        "### CRITICAL GUIDELINES FOR 'REFUTATION':",
        "1. **What is a Refutation?**",
        "   - The author explicitly argues that the reviewer's comment is **factually incorrect**, **irrelevant**, or based on a **misunderstanding**.",
        "   - Signals: 'We disagree', 'The reviewer is incorrect', 'This is a misunderstanding', 'We do not think this is necessary'.",
        "",
        "2. **What is NOT a Refutation (False Positives to AVOID):**",
        "   - **Acceptance + Differentiation:** If the reviewer asks to compare with a baseline (e.g., ConViT [6]), and the author adds the comparison but explains *why* their method is different/better (e.g., 'Unlike ConViT, we do X...'), this is **COMPLIANCE**, not refutation.",
        "   - **Technical Clarification:** Phrases like 'However, our method...' used to explain technical boundaries vs. baselines are NOT refutations of the reviewer.",
        "   - **Citation/Metadata:** If the 'opinion' text is merely a reference string (e.g., '- [6] d’Ascoli...'), it is data, not an argument. It CANNOT be refuted. **Always return False for citations.**",
        "",
        "Task:",
        "Check if the author explicitly rejects the validity of the reviewer's specific point. If they accepted the task (even with caveats) or if the text is just a citation, return False.",
        "",
        "=== META-REVIEW (for overall context) ==="
    ]

    if meta_review.strip():
        prompt_parts.append(meta_review)
    else:
        prompt_parts.append("(No meta-review available)")

    prompt_parts.extend([
        "",
        "=== AUTHOR'S RESPONSES (All combined) ==="
    ])

    if not author_responses_combined.strip():
        prompt_parts.append("(No author responses found)")
    else:
        prompt_parts.append(author_responses_combined)

    prompt_parts.extend([
        "",
        "=== REVIEWER OPINIONS TO ANALYZE ==="
    ])

    for idx, opinion in enumerate(opinions):
        prompt_parts.append(f"\nOpinion {idx} (Block {opinion['block_id']}): {opinion['text']}")

    prompt_parts.extend([
        "",
        "=== OUTPUT FORMAT ===",
        "Return a JSON array where each element corresponds to each opinion above:",
        "[",
        "  {\"opinion_index\": 0, \"refutes\": true},",
        "  {\"opinion_index\": 1, \"refutes\": false},",
        "  ...",
        "]",
        "",
        "Only output the JSON array, no explanations."
    ])

    return "\n".join(prompt_parts)


def validate_author_conflict_system_prompt() -> str:
    return (
        "You are a precise academic validator. Focus on factual accuracy and logical soundness. "
        "Your judgment should be based on evidence in the text, not assumptions. "
        "Output strictly in JSON format with a 'judgments' array containing one entry per opinion."
    )


def validate_author_conflict_prompt(meta_review: str, author_response: str, reviewer_opinions: List[Dict[str, Any]]) -> str:
    """
    Build a batch prompt to judge whether each reviewer opinion is truly incorrect.

    Core task:
      Based on meta-review context and the author's refutation, judge whether the reviewer is
      actually wrong (factually/logically) for each opinion in the batch.
    """
    opinions_str = ""
    for idx, opinion in enumerate(reviewer_opinions, 1):
        opinions_str += f"\n{idx}. **Block {opinion['block_id']}**: {opinion['text']}\n"
    prompt_parts = [
        "You are an expert academic peer review validator.",
        "Your task is to determine which of the reviewer's criticisms are **factually/logically incorrect** based on the author's response and meta-review.",
        "",
        "### CONTEXT:",
        f"**Meta-Review:** {meta_review if meta_review.strip() else '(No meta-review available)'}",
        "",
        f"**Author's Response (the refutation):** {author_response}",
        "",
        f"**Reviewer's Opinions (being refuted):** {opinions_str}",
        "",
        "### INSTRUCTIONS:",
        "1. **Reviewer is WRONG (true)** if:",
        "   - The reviewer's claim is based on factual errors about the paper",
        "   - The reviewer misunderstood the methodology/results",
        "   - The reviewer's request is logically inconsistent or impossible",
        "   - Meta-review explicitly or implicitly supports the author's position",
        "",
        "2. **Reviewer is NOT wrong (false)** if:",
        "   - The reviewer's concern is valid but author disagrees on priority/scope",
        "   - The author is making excuses without addressing the core issue",
        "   - The meta-review sides with the reviewer or remains neutral",
        "   - It's a matter of interpretation rather than factual error",
        "",
        "### OUTPUT FORMAT:",
        "Return ONLY a JSON object with a 'judgments' array:",
        "{",
        '  "judgments": [',
        "    {",
        '      "block_id": "block_id",',
        '      "is_reviewer_wrong": true/false,',
        "    },",
        "    ...",
        "  ]",
        "}"
    ]

    return "\n".join(prompt_parts)
