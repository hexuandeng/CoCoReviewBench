"""
Generate AI review text for benchmark papers with an OpenAI-compatible endpoint.

Inputs
------
- A benchmark JSONL path or glob pattern passed via `--benchmark_file`
  (or the script's fallback benchmark pattern `benchmark/benchmark_*_20*.jsonl`).
  Each paper must provide `id` and either `MD_path` directly or a resolvable Markdown lookup.
- Local Markdown files referenced by `MD_path`.
- Model endpoint configuration such as `--base_url`, `--api_key`, `--mode`, and `--model`.

Outputs
-------
- `review_result/<mode>{suffix}-result/result{sample_suffix}.jsonl`
  containing records with `id` and generated review `text`.
- `review_result/.../truncated_cases{sample_suffix}.txt` for papers truncated to fit
  the configured input token limit.
- Query logs under `logs/`.

Notes
-----
- Local localhost endpoints are started through `scripts/vllm.sh`; hosted API endpoints
  skip local service startup.
- Existing `result*.jsonl` files are loaded so interrupted runs can resume.
- `--model` may be a local checkpoint path, an open-source model name, or a hosted
  API model name.
- `--mode` is the reviewer identifier used for server matching, output folders, and
  downstream `scripts/run_process_review.sh <REVIEWER_MODEL>`.
- If `--mode` is omitted, it defaults to `basename(--model)`.
- Modes not listed in `specialized_modes` automatically use the AI-Scientist branch.
- AI-Scientist modes use helpers from `src/evaluation/ai_scientist/`.
"""

import argparse
import glob
import json
import re
import os
import sys
from typing import Dict, Optional, Any, List
from pathlib import Path
import time
import requests
from datasets import Dataset
import logging
from datetime import datetime
from transformers import AutoTokenizer
import unicodedata
from urllib.parse import urlparse

# Project root directory (two levels up from src/evaluation/)
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
# Ensure we can import `src.benchmark.utils` when running this file directly.
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from src.utils import LLMClient, start_service, stop_service

from parse_result import parse_review_cyclereviewer, parse_review_deepreviewer

perform_review = None
neurips_form = None
reviewer_system_prompt_base = None
get_review_fewshot_examples = None
extract_json_between_markers = None

DEFAULT_BENCHMARK_PATTERN = "benchmark/benchmark_*_20*.jsonl"
HOSTED_MODEL_NAME_ALIASES = {
    "gpt-52": "gpt-5.2-2025-12-11",
    "gpt-5-mini": "gpt-5-mini-2025-08-07",
    "gemini-3-flash": "gemini-3-flash-preview",
    "gemini-3-pro": "gemini-3-pro-preview",
}
specialized_modes = [
    "CycleReviewer-Llama-3.1-8B", "CycleReviewer-Llama-3.1-70B", 
    "DeepReviewer-7B", "DeepReviewer-14B", 
    "Llama-OpenReviewer-8B", 
    "SEA-E"
]

def ensure_ai_scientist_imports() -> None:
    """Import optional AI-Scientist helpers only when an AI-Scientist mode is used."""
    global perform_review
    global neurips_form
    global reviewer_system_prompt_base
    global get_review_fewshot_examples
    global extract_json_between_markers

    if reviewer_system_prompt_base is not None and extract_json_between_markers is not None:
        return

    try:
        from ai_scientist.perform_llm_review import (
            perform_review as _perform_review,
            neurips_form as _neurips_form,
            reviewer_system_prompt_base as _reviewer_system_prompt_base,
        )
        from ai_scientist.perform_llm_review import (
            get_review_fewshot_examples_wo_paper as _get_review_fewshot_examples,
        )
        from ai_scientist.llm import extract_json_between_markers as _extract_json_between_markers
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "AI-Scientist review modes depend on packages listed in requirements.txt "
            "for `src/evaluation/ai_scientist/` (for example `pypdf`). "
            "Install the repository requirements, or use a specialized / non-AI-Scientist mode."
        ) from exc

    perform_review = _perform_review
    neurips_form = _neurips_form
    reviewer_system_prompt_base = _reviewer_system_prompt_base
    get_review_fewshot_examples = _get_review_fewshot_examples
    extract_json_between_markers = _extract_json_between_markers


def _arg_was_explicitly_provided(flag: str) -> bool:
    """Return True if a CLI flag was explicitly set by the user."""
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in sys.argv[1:])


def populate_runtime_args(args):
    """Populate derived runtime flags used by both local vLLM and hosted APIs."""
    if args.mode is None:
        args.mode = os.path.basename(args.model)

    parsed_base_url = urlparse(args.base_url)
    args._host = parsed_base_url.hostname or ""
    args._port = parsed_base_url.port
    args._is_vllm = args._host in {"localhost", "127.0.0.1", "::1"}
    args._model_explicit = _arg_was_explicitly_provided("--model")

    if args._is_vllm:
        args._resolved_client_model = args.mode
    else:
        hosted_model_name = args.model if args._model_explicit else args.mode
        args._resolved_client_model = HOSTED_MODEL_NAME_ALIASES.get(hosted_model_name, hosted_model_name)

    return args


def setup_logger(mode: str, log_dir: str = f"{PROJECT_ROOT}/logs"):
    """
    Setup logger to output to both console and file.
    
    Args:
        mode (str): Review mode
        log_dir (str): Directory to save log files
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"query_vllm_{mode}_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging to: {log_path}")
    logger.info("="*60)
    
    return logger


def wait_until_ready(base_url: str,
                     api_key: str = "",
                     expect_model: Optional[str] = None,
                     timeout: int = 600,
                     interval: float = 2.0) -> None:
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


def generate_system_prompt(mode="Standard Mode", reviewer_num=4):
    """
    Generate the system prompt based on the review mode and number of reviewers.
    
    Args:
        mode (str): Review mode. Options: "Fast Mode", "Standard Mode", "Best Mode"
        reviewer_num (int): Number of reviewers to simulate
    
    Returns:
        str: System prompt for the specified mode
    """
    simreviewer_prompt = "When you simulate different reviewers, write the sections in this order: Summary, Soundness, Presentation, Contribution, Strengths, Weaknesses, Suggestions, Questions, Rating and Confidence."
    
    if mode == "Best Mode":
        prompt = f"""You are an expert academic reviewer tasked with providing a thorough and balanced evaluation of research papers. Your thinking mode is Best Mode. In this mode, you should aim to provide the most reliable review results by conducting a thorough analysis of the paper. I allow you to use search tools to obtain background knowledge about the paper - please provide three different questions. I will help you with the search. After you complete your thinking, you should review by simulating {reviewer_num} different reviewers, and use self-verification to double-check any paper deficiencies identified. Finally, provide complete review results."""
        return prompt + simreviewer_prompt
    elif mode == "DeepReviewer-7B Mode" or mode == "DeepReviewer-14B Mode":
        prompt = f"""You are an expert academic reviewer tasked with providing a thorough and balanced evaluation of research papers. Your thinking mode is Standard Mode. In this mode, you should review by simulating {reviewer_num} different reviewers, and use self-verification to double-check any paper deficiencies identified. Finally, provide complete review results."""
        return prompt + simreviewer_prompt
    elif mode == "Fast Mode":
        return "You are an expert academic reviewer tasked with providing a thorough and balanced evaluation of research papers. Your thinking mode is Fast Mode. In this mode, you should quickly provide the review results."
    elif mode == "Llama-OpenReviewer-8B Mode":
        review_fields = \
            """## Summary
            Briefly summarize the paper and its contributions. This is not the place to critique the paper; the authors should generally agree with a well-written summary.

            ## Soundness
            Please assign the paper a numerical rating on the following scale to indicate the soundness of the technical claims, experimental and research methodology and on whether the central claims of the paper are adequately supported with evidence. Choose from the following:
            4: excellent
            3: good
            2: fair
            1: poor

            ## Presentation
            Please assign the paper a numerical rating on the following scale to indicate the quality of the presentation. This should take into account the writing style and clarity, as well as contextualization relative to prior work. Choose from the following:
            4: excellent
            3: good
            2: fair
            1: poor

            ## Contribution
            Please assign the paper a numerical rating on the following scale to indicate the quality of the overall contribution this paper makes to the research area being studied. Are the questions being asked important? Does the paper bring a significant originality of ideas and/or execution? Are the results valuable to share with the broader ICLR community? Choose from the following:
            4: excellent
            3: good
            2: fair
            1: poor

            ## Strengths
            A substantive assessment of the strengths of the paper, touching on each of the following dimensions: originality, quality, clarity, and significance. We encourage reviewers to be broad in their definitions of originality and significance. For example, originality may arise from a new definition or problem formulation, creative combinations of existing ideas, application to a new domain, or removing limitations from prior results.

            ## Weaknesses
            A substantive assessment of the weaknesses of the paper. Focus on constructive and actionable insights on how the work could improve towards its stated goals. Be specific, avoid generic remarks. For example, if you believe the contribution lacks novelty, provide references and an explanation as evidence; if you believe experiments are insufficient, explain why and exactly what is missing, etc.

            ## Questions
            Please list up and carefully describe any questions and suggestions for the authors. Think of the things where a response from the author can change your opinion, clarify a confusion or address a limitation. This is important for a productive rebuttal and discussion phase with the authors.

            ## Flag For Ethics Review
            If there are ethical issues with this paper, please flag the paper for an ethics review and select area of expertise that would be most useful for the ethics reviewer to have. Please select all that apply. Choose from the following:
            No ethics review needed.
            Yes, Discrimination / bias / fairness concerns
            Yes, Privacy, security and safety
            Yes, Legal compliance (e.g., GDPR, copyright, terms of use)
            Yes, Potentially harmful insights, methodologies and applications
            Yes, Responsible research practice (e.g., human subjects, data release)
            Yes, Research integrity issues (e.g., plagiarism, dual submission)
            Yes, Unprofessional behaviors (e.g., unprofessional exchange between authors and reviewers)
            Yes, Other reasons (please specify below)

            ## Details Of Ethics Concerns
            Please provide details of your concerns.

            ## Rating
            Please provide an "overall score" for this submission. Choose from the following:
            1: strong reject
            3: reject, not good enough
            5: marginally below the acceptance threshold
            6: marginally above the acceptance threshold
            8: accept, good paper
            10: strong accept, should be highlighted at the conference


            """
        return \
            f"""You are an expert reviewer for AI conferences. You follow best practices and review papers according to the reviewer guidelines.

            Reviewer guidelines:
            1. Read the paper: It's important to carefully read through the entire paper, and to look up any related work and citations that will help you comprehensively evaluate it. Be sure to give yourself sufficient time for this step.
            2. While reading, consider the following:
                - Objective of the work: What is the goal of the paper? Is it to better address a known application or problem, draw attention to a new application or problem, or to introduce and/or explain a new theoretical finding? A combination of these? Different objectives will require different considerations as to potential value and impact.
                - Strong points: is the submission clear, technically correct, experimentally rigorous, reproducible, does it present novel findings (e.g. theoretically, algorithmically, etc.)?
                - Weak points: is it weak in any of the aspects listed in b.?
                - Be mindful of potential biases and try to be open-minded about the value and interest a paper can hold for the community, even if it may not be very interesting for you.
            3. Answer four key questions for yourself, to make a recommendation to Accept or Reject:
                - What is the specific question and/or problem tackled by the paper?
                - Is the approach well motivated, including being well-placed in the literature?
                - Does the paper support the claims? This includes determining if results, whether theoretical or empirical, are correct and if they are scientifically rigorous.
                - What is the significance of the work? Does it contribute new knowledge and sufficient value to the community? Note, this does not necessarily require state-of-the-art results. Submissions bring value to the community when they convincingly demonstrate new, relevant, impactful knowledge (incl., empirical, theoretical, for practitioners, etc).
            4. Write your review including the following information: 
                - Summarize what the paper claims to contribute. Be positive and constructive.
                - List strong and weak points of the paper. Be as comprehensive as possible.
                - Clearly state your initial recommendation (accept or reject) with one or two key reasons for this choice.
                - Provide supporting arguments for your recommendation.
                - Ask questions you would like answered by the authors to help you clarify your understanding of the paper and provide the additional evidence you need to be confident in your assessment.
                - Provide additional feedback with the aim to improve the paper. Make it clear that these points are here to help, and not necessarily part of your decision assessment.

            Your write reviews in markdown format. Your reviews contain the following sections:

            # Review

            {review_fields}

            Your response must only contain the review in markdown format with sections as defined above.
            """
    elif mode == "CycleReviewer-Llama-3.1-8B Mode" or mode == "CycleReviewer-Llama-3.1-70B Mode":
        return \
        """You are an expert academic reviewer tasked with providing a thorough and balanced evaluation of research papers. For each paper submitted, conduct a comprehensive review addressing the following aspects:
    
        1. Summary: Briefly outline main points and objectives.
        2. Soundness: Assess methodology and logical consistency.
        3. Presentation: Evaluate clarity, organization, and visual aids.
        4. Contribution: Analyze significance and novelty in the field.
        5. Strengths: Identify the paper's strongest aspects.
        6. Weaknesses: Point out areas for improvement.
        7. Questions: Pose questions for the authors.
        8. Rating: Score 1-10, justify your rating.
        9. Meta Review: Provide overall assessment and recommendation (Accept/Reject).

        Maintain objectivity and provide specific examples from the paper to support your evaluation.

        You need to fill out **4** review opinions."""
    elif mode == "SEA-S Mode":
        return "As an experienced academic paper reviewer, you are presented with different review contents for the same paper. Please analyze these contents carefully and consolidate them into a single review. The review should be organized into nine sections: Summary, Strengths, Weaknesses, Questions, Soundness, Presentation, Contribution, Rating and Paper Decision. Below is a description of each section:\n1. Summary: Combine the 'Summary' sections from all reviews into a cohesive summary, aiming for a length of about 100-150 words.\n2. Strengths/Weaknesses/Questions: Combine the Strengths/Weaknesses/Questions sections from all reviews into a unified, cohesive bullet-point list that avoids redundancy while preserving the specific details and depth of each point.\n3. Soundness/Presentation/Contribution: Aggregate the Contribution/Soundness/Presentation score from each review to determine a suitable overall score (the score must be an **integer**), then, match this integer score to the corresponding description from the list below and provide the result. For example, if the score is 3, the result should be '3 good'. The possible scores and their descriptions are: \n    1 poor\n    2 fair\n    3 good\n    4 excellent\n4. Rating: Aggregate the 'Rating' from each review to determine a suitable overall Rating (the Rating must be an **integer**), then, match this integer Rating to the corresponding description from the list below and provide the result. For example, if the Rating is 1, the result should be '1 strong reject'. The possible Ratings and their descriptions are: \n    1 strong reject\n    2 reject, significant issues present\n    3 reject, not good enough\n    4 possibly reject, but has redeeming facets\n    5 marginally below the acceptance threshold\n    6 marginally above the acceptance threshold\n    7 accept, but needs minor improvements \n    8 accept, good paper\n    9 strong accept, excellent work\n    10 strong accept, should be highlighted at the conference    \n5. Paper Decision: It must include the Decision itself(Accept or Reject) and the reasons for this decision, based on Metareview, the criteria of originality, methodological soundness, significance of results, and clarity and logic of presentation, etc. Please ensure your Decision (Accept/Reject) matches the value of the 'Decision' key in the JSON, if present. \n\nHere is the template for a review format. You must follow this format to output the integrated review results:\n**Summary:**\nSummary content\n\n**Strengths:**\n- Strength 1\n- Strength 2\n- ...\n\n**Weaknesses:**\n- Weakness 1\n- Weakness 2\n- ...\n\n**Questions:**\n- Question 1\n- Question 2\n- ...\n\n**Soundness:**\nSoundness result\n\n**Presentation:**\nPresentation result\n\n**Contribution:**\nContribution result\n\n**Rating:**\nRating result\n\n**Paper Decision:**\n- Decision: Accept/Reject\n- Reasons: reasons content\n\n"    
    elif mode == "SEA-E Mode":
        return "You are a highly experienced, conscientious, and fair academic reviewer, please help me review this paper. The review should be organized into nine sections: \n1. Summary: A summary of the paper in 100-150 words.\n2. Strengths/Weaknesses/Questions: The Strengths/Weaknesses/Questions of paper, which should be listed in bullet points, with each point supported by specific examples from the article where possible.\n3. Soundness/Contribution/Presentation: Rate the paper's Soundness/Contribution/Presentation, and match this score to the corresponding description from the list below and provide the result. The possible scores and their descriptions are: \n    1 poor\n    2 fair\n    3 good\n    4 excellent\n4. Rating: Give this paper an appropriate rating, match this rating to the corresponding description from the list below and provide the result. The possible Ratings and their descriptions are: \n    1 strong reject\n    2 reject, significant issues present\n    3 reject, not good enough\n    4 possibly reject, but has redeeming facets\n    5 marginally below the acceptance threshold\n    6 marginally above the acceptance threshold\n    7 accept, but needs minor improvements \n    8 accept, good paper\n    9 strong accept, excellent work\n    10 strong accept, should be highlighted at the conference   \n5. Paper Decision: It must include the Decision itself(Accept or Reject) and the reasons for this decision, based on the criteria of originality, methodological soundness, significance of results, and clarity and logic of presentation.\n\nHere is the template for a review format, you must follow this format to output your review result:\n**Summary:**\nSummary content\n\n**Strengths:**\n- Strength 1\n- Strength 2\n- ...\n\n**Weaknesses:**\n- Weakness 1\n- Weakness 2\n- ...\n\n**Questions:**\n- Question 1\n- Question 2\n- ...\n\n**Soundness:**\nSoundness result\n\n**Presentation:**\nPresentation result\n\n**Contribution:**\nContribution result\n\n**Rating:**\nRating result\n\n**Paper Decision:**\n- Decision: Accept/Reject\n- Reasons: reasons content\n\n\nPlease ensure your feedback is objective and constructive. The paper is as follows:"
    else:
        raise NotImplementedError(f"Invalid mode: {mode}")


def _build_prompts(ai_scientist_mode: bool = False, args: Any = None):
    """Build base_prompt and wrapper for different modes."""
    if ai_scientist_mode:
        ensure_ai_scientist_imports()
        # In ai_scientist_mode, combine prompts as: system_prompt + (ai_scientist_base_prompt + wrapper + paper_text)
        # Build ai_scientist_base_prompt as in lines 611-612
        fs_prompt = get_review_fewshot_examples(num_fs_examples=args.num_fs_examples)
        # Build the wrapper text that will surround the paper text (format matches lines 619-625)
        wrapper_prefix = neurips_form + fs_prompt + "\n\nHere is the paper you are asked to review:\n```\n"
        if args.no_think:
            wrapper_suffix = "\n```\nPlease put your final peer review of the above paper in json format."
        else:
            wrapper_suffix = "\n```\nPlease reason step by step, and put your final peer review of the above paper in json format."
        
        return wrapper_prefix, wrapper_suffix
    
    return "", ""

def truncate_paper_text(
    tokenizer,
    paper_text: str,
    system_prompt: str,
    max_input_tokens: int,
    ai_scientist_mode: bool = False,
    args: Any = None,
):
    """
    Truncate paper text to fit within token limit while keeping system prompt intact.
    """
    if tokenizer is None:
        return paper_text, 0, False
    
    def _count_tokens(system_msg, user_content):
        """Helper: count tokens for given messages."""
        messages = []
        if system_msg is not None:
            messages.append({"role": "system", "content": system_msg})
        messages.append({"role": "user", "content": user_content})
        
        try:
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return len(tokenizer.encode(formatted))
        except Exception:
            fallback = (system_msg or "") + "\n\n" + user_content
            return len(tokenizer.encode(fallback)) + 10
    
    # Build prompts
    wrapper_prefix, wrapper_suffix = _build_prompts(ai_scientist_mode=ai_scientist_mode, args=args)
    
    # Calculate full user prompt and check total tokens
    full_user_prompt = wrapper_prefix + paper_text + wrapper_suffix
    original_token_count = _count_tokens(system_prompt, full_user_prompt)
    
    if original_token_count <= max_input_tokens:
        return paper_text, original_token_count, False
    
    # Calculate overhead (everything except paper text)
    overhead_prompt = wrapper_prefix + wrapper_suffix
    overhead_tokens = _count_tokens(system_prompt, overhead_prompt)
    
    # Calculate available tokens for paper
    safety_margin = 16
    available_tokens = max_input_tokens - overhead_tokens - safety_margin
    
    if available_tokens <= 0:
        raise ValueError(f"Prompt overhead too long: {overhead_tokens} tokens, max: {max_input_tokens}")
    
    # Truncate paper text
    paper_tokens = tokenizer.encode(paper_text, add_special_tokens=False)
    truncated_paper_tokens = paper_tokens[:available_tokens]
    truncated_paper_text = tokenizer.decode(truncated_paper_tokens, skip_special_tokens=True)
    
    return truncated_paper_text, original_token_count, True


def _read_text_file(path: str) -> str:
    """Read a UTF-8 text file. Return empty string if missing/unreadable."""
    if not path or not isinstance(path, str):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def build_dataset_from_benchmark_jsonl(
    benchmark_jsonl_paths: List[Path],
    md_lookup_jsonl_paths: Optional[List[Path]] = None,
    base_dir: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
):
    """
    Build a HF Dataset with columns: id, inputs from a benchmark JSONL.

    Expected per-line format:
      - id: str
      - MD_path: str (optional in sampled files)

    If MD_path is missing in a benchmark file, this function will try to
    look it up from md_lookup_jsonl_paths by matching id.
    """
    # Build id -> MD_path map (optional)
    id_to_md_path: Dict[str, str] = {}
    for md_lookup_jsonl_path in md_lookup_jsonl_paths or []:
        try:
            with open(md_lookup_jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = (line or "").strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    pid = rec.get("id")
                    mdp = rec.get("MD_path")
                    if isinstance(pid, str) and pid and isinstance(mdp, str) and mdp:
                        id_to_md_path[pid] = mdp
        except Exception as e:
            if logger:
                logger.warning(f"Failed to read md_lookup_jsonl_path={md_lookup_jsonl_path}: {e}")

    rows = []
    seen_paper_ids = set()
    for benchmark_jsonl_path in benchmark_jsonl_paths:
        with open(benchmark_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = (line or "").strip()
                if not line:
                    continue
                rec = json.loads(line)
                paper_id = rec.get("id")
                if not isinstance(paper_id, str) or not paper_id or paper_id in seen_paper_ids:
                    continue

                md_path = rec.get("MD_path")
                if not md_path and paper_id in id_to_md_path:
                    md_path = id_to_md_path[paper_id]

                if not isinstance(md_path, str) or not md_path:
                    raise ValueError(
                        f"Missing MD_path for paper id={paper_id} in {benchmark_jsonl_path} "
                        f"(and not found in md_lookup_jsonl_paths={md_lookup_jsonl_paths})"
                    )

                resolved_md_path = md_path
                if base_dir and not os.path.isabs(resolved_md_path):
                    resolved_md_path = os.path.join(base_dir, resolved_md_path)

                md_content = _read_text_file(resolved_md_path)
                if not md_content:
                    raise ValueError(f"Failed to read MD file for paper id={paper_id}: {resolved_md_path}")

                # Keep the downstream contract: later code does `eval(inputs)[1]['content']`.
                inputs_obj = [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": md_content},
                ]
                rows.append(
                    {
                        "id": paper_id,
                        "inputs": repr(inputs_obj),
                    }
                )
                seen_paper_ids.add(paper_id)

    return Dataset.from_list(rows)


def resolve_benchmark_jsonl_paths(benchmark_pattern: str) -> List[Path]:
    expanded_pattern = os.path.expanduser(benchmark_pattern)
    search_pattern = (
        expanded_pattern
        if os.path.isabs(expanded_pattern)
        else os.path.join(PROJECT_ROOT, expanded_pattern)
    )

    matched_files = [
        Path(path).resolve()
        for path in sorted(glob.glob(search_pattern))
        if Path(path).is_file()
    ]

    if not matched_files:
        raise FileNotFoundError(f"No benchmark files matched pattern: {benchmark_pattern}")

    return matched_files


def apply_config_overrides(args, logger):
    """Apply original configuration overrides based on mode."""
    # allowed_modes = ["CycleReviewer-8B", "CycleReviewer-70B", "DeepReviewer-7B", "DeepReviewer-14B", "OpenReviewer", "SEA-E"]
    
    if args.mode is None:
        args.mode = os.path.basename(args.model)

    args.ori_config = args.mode in specialized_modes and not args.general_config
    if args.mode in specialized_modes:
        if args.general_config:
            logger.info(f"Using general configuration for specialized mode: {args.mode}")
        else:
            logger.info(f"Enabled original configuration by default for specialized mode: {args.mode}")

    if args.ori_config:
        if args.mode not in specialized_modes:
            logger.error(f"Internal ori_config state is only valid for: {', '.join(specialized_modes)}")
            logger.error(f"Current mode: {args.mode}")
            exit(1)

        if args.mode in ["CycleReviewer-Llama-3.1-8B", "CycleReviewer-Llama-3.1-70B"]:
            args.max_tokens = 7000
            args.max_input_tokens = 43000
            args.temperature = 0.4
            logger.info(
                f"Using original configuration (CycleReviewer): "
                f"max_tokens={args.max_tokens}, max_input_tokens={args.max_input_tokens}, temperature={args.temperature}"
            )
        elif args.mode in ["DeepReviewer-7B", "DeepReviewer-14B"]:
            args.max_tokens = 35000
            args.max_input_tokens = 55000
            args.temperature = 0.4
            logger.info(
                f"Using original configuration ({args.mode}): "
                f"max_tokens={args.max_tokens}, max_input_tokens={args.max_input_tokens}, temperature={args.temperature}"
            )
        elif args.mode in ["Llama-OpenReviewer-8B"]:
            args.max_tokens = 4096
        elif args.mode in ["SEA-E"]:
            args.max_tokens = 8192
    
    return args


def get_tokenizer_path(args):
    """Get tokenizer path based on mode."""
    return args.model


def initialize_client(args, logger):
    """Initialize LLM client based on mode."""
    cache_dir = str(Path(PROJECT_ROOT) / ".llm_cache")
    if not args._is_vllm:
        logger.info(
            f"Initializing hosted LLMClient for reviewer={args.mode}, api_model={args._resolved_client_model}..."
        )
        client = LLMClient(
            api_key=args.api_key,
            base_url=args.base_url,
            model_name=args._resolved_client_model,
            max_workers=args.max_workers,
            cache_dir=cache_dir
        )
    else:
        logger.info(f"Waiting for server at {args.base_url} to be ready...")
        wait_until_ready(args.base_url, api_key="dummy", expect_model=args.mode, timeout=3600, interval=20.0)
        logger.info("Server is ready!")
        
        client = LLMClient(
            api_key="dummy",
            base_url=args.base_url,
            model_name=args.mode,
            max_workers=args.max_workers,
            cache_dir=cache_dir
        )
    
    return client


def build_dataset_and_paths(args, logger):
    """Load dataset and build save paths."""
    sample_suffix = "_sample" if args.sample_data_path else ""
    # eval_model_full_name = model_full_name_dict[args.mode]
    
    # if args.mode in ["Fast", "Best"]:
    #     eval_model_full_name = f"{model_full_name_dict[args.mode]}-{args.mode}"

    _is_ai_scientist = args.mode not in specialized_modes
    
    # Build paths
    general_config_suffix = "-general-config" if args.mode in specialized_modes and args.general_config else ""
    no_think_suffix = "-no-think" if args.no_think else ""
    
    if args.benchmark_file:
        benchmark_jsonl_paths = resolve_benchmark_jsonl_paths(args.benchmark_file)
        md_lookup_jsonl_paths = benchmark_jsonl_paths
    elif args.sample_data_path:
        benchmark_jsonl_paths = [(Path(PROJECT_ROOT) / "benchmark" / args.sample_data_path).resolve()]
        md_lookup_jsonl_paths = benchmark_jsonl_paths
    else:
        benchmark_jsonl_paths = resolve_benchmark_jsonl_paths(DEFAULT_BENCHMARK_PATTERN)
        md_lookup_jsonl_paths = benchmark_jsonl_paths

    default_lookup_path = Path(PROJECT_ROOT) / "benchmark" / "benchmark.jsonl"
    if default_lookup_path.exists():
        md_lookup_jsonl_paths = [default_lookup_path.resolve(), *md_lookup_jsonl_paths]

    logger.info(f"Loading benchmark JSONL from {len(benchmark_jsonl_paths)} file(s)...")
    logger.info(f"Benchmark file examples: {[str(path) for path in benchmark_jsonl_paths[:5]]}")
    test_dataset = build_dataset_from_benchmark_jsonl(
        benchmark_jsonl_paths=benchmark_jsonl_paths,
        md_lookup_jsonl_paths=md_lookup_jsonl_paths,
        base_dir=PROJECT_ROOT,
        logger=logger,
    )
    logger.info(f"Loaded {len(test_dataset)} papers (from MD files)")
    
    save_dir = f"{PROJECT_ROOT}/review_result/{args.mode}{general_config_suffix}{no_think_suffix}-result"
    save_path = os.path.join(save_dir, f"result{sample_suffix}.jsonl")
    truncated_cases_path = os.path.join(save_dir, f"truncated_cases{sample_suffix}.txt")

    os.makedirs(save_dir, exist_ok=True)
    
    return test_dataset, save_path, truncated_cases_path, _is_ai_scientist, args.mode


def load_existing_results(save_path, logger):
    """Load existing results for resume capability."""
    if os.path.exists(save_path):
        all_review_results = []
        with open(save_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_review_results.append(json.loads(line))
        logger.info(f"Loaded {len(all_review_results)} existing results from {save_path}")
    else:
        all_review_results = []
    
    processed_ids = set()
    for result in all_review_results:
        if "id" in result and result["id"]:
            processed_ids.add(result["id"])
    logger.info(f"Found {len(processed_ids)} already processed paper IDs")
    
    return all_review_results, processed_ids


def prepare_batch_data(batch, processed_ids, mode):
    """Extract and filter batch data."""
    batch_sft_inputs = [eval(i) for i in batch["inputs"]]
    
    if mode == "Llama-OpenReviewer-8B":
        batch_paper_text = [
            f"""Review the following paper:

            {i[1]["content"]}
            """ for i in batch_sft_inputs]
    else:
        batch_paper_text = [i[1]["content"] for i in batch_sft_inputs]
    
    batch_ids = [i for i in batch["id"]]
    
    # Filter out already processed papers
    unprocessed_indices = [i for i, paper_id in enumerate(batch_ids) if paper_id not in processed_ids]
    
    if not unprocessed_indices:
        return [], [], []
    
    batch_paper_text = [batch_paper_text[i] for i in unprocessed_indices]
    batch_ids = [batch_ids[i] for i in unprocessed_indices]
    
    return batch_ids, batch_paper_text, unprocessed_indices


def process_ai_scientist_batch(batch_ids, batch_paper_text, client, tokenizer, args, logger, eval_model_full_name):
    """Process batch using AI Scientist mode with parallel retry mechanism and best result selection."""
    ensure_ai_scientist_imports()
    logger.info(f"Processing {len(batch_paper_text)} papers with AI Scientist based on {eval_model_full_name}...")
    # Truncate papers
    truncated_papers = []
    for i, paper_text in enumerate(batch_paper_text):
        paper_id = batch_ids[i]
        truncated_text, original_token_count, was_truncated = truncate_paper_text(
            tokenizer, paper_text, reviewer_system_prompt_base, args.max_input_tokens, ai_scientist_mode=True, args=args
        )
        batch_paper_text[i] = truncated_text

        if was_truncated:
            truncated_papers.append({
                "paper_id": paper_id,
                "original_token_count": original_token_count,
                "index": i
            })
            logger.warning(
                f"Truncated paper {paper_id}: {original_token_count} tokens -> {args.max_input_tokens} limit"
            )

    def check_content_issues(text):
        """Detect malformed AI-Scientist outputs and describe the failure mode."""
        if not text:
            return True, "empty"

        json_tag_count = text.count("```json")
        think_tag_count = text.count("</think>")
        extracted = extract_json_between_markers(text)

        issues = []
        if json_tag_count > 1:
            issues.append(f"multiple_json_tags({json_tag_count})")
        if think_tag_count > 0:
            issues.append(f"multiple_think_tags({think_tag_count})")
        if extracted is None:
            issues.append("json_extraction_failed")

        if issues:
            return True, ";".join(issues)
        return False, "ok"

    def calculate_repetition_score(text):
        """Return a simple repetition score; higher means more repeated content."""
        if not text or len(text) < 10:
            return 1.0

        from collections import Counter

        pattern = r'(\b\w+\b)(?:\s+\1){3,}'
        repeat_patterns = re.findall(pattern, text)
        repeat_count = len(repeat_patterns)

        words = text.split()
        if len(words) < 6:
            return 0.0

        windows = [' '.join(words[i:i + 6]) for i in range(len(words) - 5)]
        window_counts = Counter(windows)
        repeated_windows = sum(count - 1 for count in window_counts.values() if count > 1)

        return repeat_count + repeated_windows

    def select_best_result(attempts, paper_id):
        """Pick the cleanest usable attempt, preferring long low-repetition outputs."""
        if not attempts:
            return None

        valid_attempts = []
        for i, att in enumerate(attempts):
            length_ok = len(att["text"]) > 256
            repetition_score = calculate_repetition_score(att["text"])

            candidate = {
                "index": i,
                "text": att["text"],
                "text_with_think": att["text_with_think"],
                "length": len(att["text"]),
                "has_issue": att["has_issue"],
                "issue_type": att["issue_type"],
                "repetition_score": repetition_score,
                "is_valid": length_ok and not att["has_issue"],
            }
            valid_attempts.append(candidate)

            logger.debug(
                f"Paper {paper_id} attempt {i}: len={len(att['text'])}, "
                f"issues={att['issue_type']}, repetition={repetition_score:.3f}"
            )

        no_issue = [c for c in valid_attempts if not c["has_issue"]]
        if no_issue:
            long_no_issue = [c for c in no_issue if c["length"] > 256]
            if long_no_issue:
                best = min(long_no_issue, key=lambda x: x["repetition_score"])
                logger.info(
                    f"Clean result selected for paper {paper_id} "
                    f"(attempt {best['index']}, len={best['length']}, rep={best['repetition_score']:.3f})"
                )
                return best

            best = max(no_issue, key=lambda x: x["length"])
            logger.info(
                f"Shortest clean fallback selected for paper {paper_id} "
                f"(attempt {best['index']}, len={best['length']})"
            )
            return best

        long_attempts = [c for c in valid_attempts if c["length"] > 256]
        if long_attempts:
            best = min(long_attempts, key=lambda x: x["repetition_score"])
            logger.warning(
                f"Issue-containing result selected for paper {paper_id} "
                f"(attempt {best['index']}, issues={best['issue_type']}, rep={best['repetition_score']:.3f})"
            )
            return best

        best = max(valid_attempts, key=lambda x: x["length"])
        logger.error(
            f"Longest fallback selected for paper {paper_id} "
            f"(attempt {best['index']}, len={best['length']})"
        )
        return best

    # Build task parameters for all papers
    paper_tracking = {}  # original_idx -> tracking info
    task_map = {}  # task_id -> task info
    
    for i, paper_text in enumerate(batch_paper_text):
        wrapper_prefix, wrapper_suffix = _build_prompts(ai_scientist_mode=True, args=args)
        full_prompt = wrapper_prefix + paper_text + wrapper_suffix
        task_params = {
            "prompt": full_prompt,
            "system_prompt": reviewer_system_prompt_base,
            "temperature": args.temperature,
            "max_completion_tokens": args.max_tokens,
            "top_p": 0.95,
            "clear_cache": False
        }
        if args.no_think:
            task_params["reasoning_effort"] = "none"
        else:
            task_params["reasoning_effort"] = "medium"

        paper_tracking[i] = {
            "paper_id": batch_ids[i],
            "attempts": [],
            "attempt_count": 0,
            "max_attempts": 3,
            "task_params": task_params,
            "completed": False,
            "best_result": None
        }

        task_id = client.submit_task(**task_params)
        task_map[task_id] = {
            "original_idx": i,
            "is_retry": False,
            "attempt_num": 0
        }

    logger.info(f"Submitted {len(task_map)} initial tasks for {eval_model_full_name}")

    # Collect results with parallel retry
    pending_tasks = set(task_map.keys())
    
    while pending_tasks:
        newly_done = []
        
        # Check all pending tasks
        logger.info(f"Submitted {len(pending_tasks)} pending tasks for {eval_model_full_name}")
        for task_id in list(pending_tasks):
            result = client.get_result(task_id, clear_cache=False)
            
            if result["status"] == "done":
                newly_done.append((task_id, result))

        # Process completed tasks
        for task_id, result in newly_done:
            pending_tasks.discard(task_id)
            info = task_map[task_id]
            original_idx = info["original_idx"]
            attempt_num = info["attempt_num"]
            tracking = paper_tracking[original_idx]
            paper_id = tracking["paper_id"]
            
            llm_review_text_with_think = result["content"]
            
            if "</think>" in llm_review_text_with_think:
                llm_review_text = llm_review_text_with_think.split("</think>", 1)[1].strip()
            else:
                llm_review_text = llm_review_text_with_think.strip()
            
            has_issue, issue_type = check_content_issues(llm_review_text)
            
            attempt_record = {
                "text": llm_review_text,
                "text_with_think": llm_review_text_with_think,
                "has_issue": has_issue,
                "issue_type": issue_type,
                "attempt_num": attempt_num
            }
            tracking["attempts"].append(attempt_record)
            tracking["attempt_count"] += 1
            
            need_retry = has_issue and tracking["attempt_count"] < tracking["max_attempts"]
            
            if need_retry:
                logger.warning(
                    f"Paper {paper_id} attempt {attempt_num} has issues: {issue_type}. "
                    f"Retrying ({tracking['attempt_count']}/{tracking['max_attempts']})..."
                )
                
                retry_params = tracking["task_params"].copy()
                retry_params["temperature"] = min(args.temperature + 0.1 * (attempt_num + 1), 1.0)
                logger.debug(f"Retrying with temperature={retry_params['temperature']}")
                
                new_task_id = client.submit_task(**retry_params)
                task_map[new_task_id] = {
                    "original_idx": original_idx,
                    "is_retry": True,
                    "attempt_num": tracking["attempt_count"]
                }
                pending_tasks.add(new_task_id)
            else:
                tracking["completed"] = True
                
                if not has_issue:
                    logger.info(f"Paper {paper_id} attempt {attempt_num} successful (no issues)")
                else:
                    logger.warning(f"Paper {paper_id} reached max attempts with issues: {issue_type}")

    review_results = [None] * len(batch_ids)
    
    for original_idx, tracking in paper_tracking.items():
        paper_id = tracking["paper_id"]
        best = select_best_result(tracking["attempts"], paper_id)
        selected_text = best["text"]
        if len(best["text"]) < 256:
            selected_text = best["text_with_think"]
        review = build_ai_scientist_review(selected_text, paper_id)
        
        review_results[original_idx] = review

    return review_results, truncated_papers


def build_ai_scientist_review(llm_review_text, paper_id):
    """Build review dict from AI Scientist output."""
    
    if not llm_review_text:
        return {
            "raw_text": "",
            "reviews": [],
            "meta_review": {},
            "decision": "",
            "id": paper_id
        }

    review = {
        "raw_text": llm_review_text,
        "reviews": [],
        "meta_review": {},
        "decision": "",
        "id": paper_id
    }

    return review



def verify_generated_text_for_specific_models(generated_text, args):
    need_to_regenerated = False
    if args.mode in ["DeepReviewer-7B", "DeepReviewer-14B"]:
        review = parse_review_deepreviewer(generated_text)
        meta = review.get("meta_review", {})
        strengths = meta.get("strengths", "")
        weaknesses = meta.get("weaknesses", "")
        need_to_regenerated = len(strengths) == 0 or len(weaknesses) == 0
    elif args.mode in ["CycleReviewer-Llama-3.1-8B", "CycleReviewer-Llama-3.1-70B"]:
        review = parse_review_cyclereviewer(generated_text)
        need_to_regenerated =  (len(review) == 0) or (len(review["summary"]) == 0) or (len(review["review_rate"]) == 0)
    elif args.mode in ["SEA-E"]:
        need_to_regenerated =  "\n**Strengths:**\n" in generated_text and "\n**Soundness:**\n" in generated_text
        review = {"raw_text": generated_text}
    else:
        if "\n## Strengths\n" in generated_text:
            review = {"raw_text": generated_text, "strengths": generated_text.split("\n## Strengths\n")[-1]}
        else:
            review = {"raw_text": generated_text}
        need_to_regenerated = "\n## Strengths\n" not in generated_text

    return need_to_regenerated, review


def process_specialized_model_batch(batch_ids, batch_paper_text, client, tokenizer, args, logger):
    """Process batch using standard DeepReviewer modes with parallel retry mechanism."""
    system_prompt = generate_system_prompt(mode=f"{args.mode} Mode", reviewer_num=args.reviewer_num)
    
    # Truncate papers
    truncated_papers = []
    for i, paper_text in enumerate(batch_paper_text):
        paper_id = batch_ids[i]
        truncated_text, original_token_count, was_truncated = truncate_paper_text(
            tokenizer, paper_text, system_prompt, args.max_input_tokens, args=args
        )
        batch_paper_text[i] = truncated_text
        
        if was_truncated:
            truncated_papers.append({
                "paper_id": paper_id,
                "original_token_count": original_token_count,
                "index": i
            })
            logger.warning(
                f"Truncated paper {paper_id}: {original_token_count} tokens -> {args.max_input_tokens} limit"
            )
    
    # Build task parameters
    task_params = {
        "prompt": "",
        "system_prompt": system_prompt,
        "temperature": args.temperature,
        "max_completion_tokens": args.max_tokens,
        "clear_cache": False
    }
    
    if args.ori_config:
        if args.mode in ["CycleReviewer-Llama-3.1-8B", "CycleReviewer-Llama-3.1-70B"]:
            task_params.update({"top_p": 0.95, "temperature": 0.4, "max_completion_tokens": 7000})
        elif args.mode in ["DeepReviewer-7B", "DeepReviewer-14B"]:
            task_params.update({"top_p": 0.95, "temperature": 0.4, "max_completion_tokens": 35000})
        elif args.mode == "Llama-OpenReviewer-8B":
            task_params.update({"top_p": 0.9, "temperature": 0.6, "max_completion_tokens": 4096})
    
    if args.no_think:
        task_params["enable_thinking"] = False
    
    # First batch submission
    task_map = {}  # task_id -> {"original_idx": int, "is_retry": bool, "retry_count": int, "fallback_text": str}
    
    for i in range(len(batch_paper_text)):
        params = task_params.copy()
        params["prompt"] = batch_paper_text[i]
        task_id = client.submit_task(**params)
        task_map[task_id] = {
            "original_idx": i,
            "is_retry": False,
            "retry_count": 0,
            "paper_id": batch_ids[i],
            "fallback_text": None  # Store first attempt's generated text
        }
    
    logger.info(f"Submitted {len(task_map)} initial tasks")
    
    # Collect results with parallel retry
    review_results = [None] * len(batch_ids)  # Pre-allocate result slots
    pending_tasks = set(task_map.keys())
    retry_tasks = {}  # Tasks waiting to be retried: original_idx -> retry_params
    
    # Poll all pending tasks in parallel
    while pending_tasks or retry_tasks:
        # Check current pending tasks
        newly_done = []
        failed_tasks = []
        
        for task_id in list(pending_tasks):
            result = client.get_result(task_id, clear_cache=False)
            
            if result["status"] == "done":
                newly_done.append((task_id, result))
            else:
                failed_tasks.append((task_id, result))
        
        # Process completed tasks
        for task_id, result in newly_done:
            pending_tasks.discard(task_id)
            info = task_map[task_id]
            original_idx = info["original_idx"]
            paper_id = info["paper_id"]
            
            generated_text = result["content"]
            generated_text_with_think = generated_text
            if "</think>" in generated_text:
                generated_text = generated_text.split("</think>", 1)[1].strip()
            
            need_regenerate = not generated_text
            review = {}
            if not need_regenerate:
                need_regenerate, review = verify_generated_text_for_specific_models(generated_text, args)
            
            if need_regenerate and not info["is_retry"]:
                # Mark for retry (only retry once)
                logger.warning(f"Paper {paper_id} needs regeneration, queuing retry...")
                retry_params = task_params.copy()
                retry_params["prompt"] = batch_paper_text[original_idx]
                retry_params["temperature"] = min(float(task_params.get("temperature", args.temperature)) + 0.1, 1.0)
                retry_tasks[original_idx] = {
                    "params": retry_params,
                    "paper_id": paper_id,
                    "original_idx": original_idx,
                    "fallback_text": generated_text,
                    "fallback_text_with_think": generated_text_with_think  # Save first attempt's text
                }
            else:
                # Final result (success or failed retry)
                if need_regenerate and info["is_retry"]:
                    fallback_text = info.get("fallback_text")
                    fallback_text_with_think = info.get("fallback_text_with_think")
                    
                    if generated_text and fallback_text:
                        generated_text = generated_text if len(generated_text) > len(fallback_text) else fallback_text
                    elif fallback_text and not generated_text:
                        generated_text = fallback_text
                        
                    if generated_text_with_think and fallback_text_with_think:
                        generated_text_with_think = generated_text_with_think if len(generated_text_with_think) > len(fallback_text_with_think) else fallback_text_with_think
                    elif fallback_text_with_think and not generated_text_with_think:
                        generated_text_with_think = fallback_text_with_think
                
                if len(review) == 0:
                    review = {
                        "raw_text": generated_text if len(generated_text) > 30 else generated_text_with_think,
                        "reviews": [],
                        "meta_review": {},
                        "decision": ""
                    }
                review["id"] = paper_id
                review_results[original_idx] = review
        
        # Process failed tasks
        for task_id, result in failed_tasks:
            pending_tasks.discard(task_id)
            info = task_map[task_id]
            original_idx = info["original_idx"]
            paper_id = info["paper_id"]
            
            if not info["is_retry"]:
                # Retry failed tasks once
                logger.warning(f"Task failed for paper {paper_id}, queuing retry...")
                retry_params = task_params.copy()
                retry_params["prompt"] = batch_paper_text[original_idx]
                retry_params["temperature"] = min(float(task_params.get("temperature", args.temperature)) + 0.1, 1.0)
                retry_tasks[original_idx] = {
                    "params": retry_params,
                    "paper_id": paper_id,
                    "original_idx": original_idx,
                    "fallback_text": info.get("fallback_text")  # Pass through if exists
                }
            else:
                # Retry failed, try to use fallback text from first attempt
                fallback_text = info.get("fallback_text")
                fallback_text_with_think = info.get("fallback_text_with_think")
                
                logger.error(f"Retry failed for paper {paper_id}")
                
                # Try to extract from fallback if available
                final_text = ""
                if len(fallback_text) < 30:
                    final_text = fallback_text_with_think
                else:
                    final_text = fallback_text
                
                review_results[original_idx] = {
                    "id": paper_id,
                    "raw_text": final_text,
                    "reviews": [],
                    "meta_review": {},
                    "decision": ""
                }
        
        # Submit all queued retry tasks in batch
        if retry_tasks:
            logger.info(f"Submitting {len(retry_tasks)} retry tasks...")
            for original_idx, retry_info in list(retry_tasks.items()):
                task_id = client.submit_task(**retry_info["params"])
                task_map[task_id] = {
                    "original_idx": original_idx,
                    "is_retry": True,
                    "retry_count": 1,
                    "paper_id": retry_info["paper_id"],
                    "fallback_text": retry_info.get("fallback_text"),  # Carry forward fallback text
                    "fallback_text_with_think": retry_info.get("fallback_text_with_think")
                }
                pending_tasks.add(task_id)
            retry_tasks.clear()
        
    
    return review_results, truncated_papers


def save_truncated_log(truncated_papers, truncated_cases_path, max_input_tokens):
    """Save truncated papers log."""
    with open(truncated_cases_path, 'w') as f:
        f.write(f"Truncated cases (original token count > {max_input_tokens}):\n")
        f.write("="*60 + "\n")
        for truncated_info in truncated_papers:
            f.write(f"Paper ID: {truncated_info['paper_id']}, Original tokens: {truncated_info['original_token_count']}\n")
        f.write(f"\nTotal truncated: {len(truncated_papers)} cases\n")


def print_final_summary(logger, all_review_results, all_truncated_papers, save_path, truncated_cases_path, max_input_tokens):
    """Print final processing summary."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing complete!")
    logger.info(f"Total papers processed: {len(all_review_results)}")
    logger.info(f"Results saved to: {save_path}")
    
    if all_truncated_papers:
        logger.info(f"Truncated {len(all_truncated_papers)} cases due to token limit")
        logger.info(f"Truncated case details saved to: {truncated_cases_path}")
    else:
        logger.info(f"All cases processed without truncation (no cases exceeded {max_input_tokens} tokens)")
    logger.info(f"{'='*60}")


def extract_review_text(it: dict, model: str) -> str:
    """Extract review text using the original rules."""
    raw_text = it.get("raw_text", "")
    if "SEA-E" in model:
        if "**Strengths:**" in raw_text and "**Soundness:**" in raw_text:
            text = "**Strengths:**\n" + raw_text.split("**Strengths:**")[-1].strip()
            text = text.split("**Soundness:**")[0].strip()
        elif "**Strengths:**" in raw_text:
            text = "**Strengths:**\n" + raw_text.split("**Strengths:**")[-1].strip()
        else:
            text = raw_text

    elif "OpenReviewer" in model:
        if "## Strengths" in raw_text:
            text = "## Strengths\n" + raw_text.split("## Strengths")[-1]
            if "No ethics review needed." in text:
                if "## Flag For Ethics Review" in text:
                    text = text.split("## Flag For Ethics Review")[0].strip()
                else:
                    text = text
            else:
                if "## Rating" in text:
                    text = text.split("## Rating")[0].strip()
                else:
                    text = text
        else:
            text = raw_text

    elif "DeepReviewer" in model:
        meta = it.get("meta_review", {})
        text = ""
        if meta:
            strengths = meta.get("strengths", "")
            weaknesses = meta.get("weaknesses", "")
            questions = meta.get("questions", "")
            if len(strengths) > 0:
                text += "## Strengths\n" + strengths
            if len(weaknesses) > 0:
                text += "\n## Weaknesses\n" + weaknesses
            if questions:
                text += "\n## Questions\n" + questions
        
        if len(text) == 0 and "## Strengths:" in raw_text:
            text = "## Strengths:" + raw_text.split("## Strengths:", 1)[-1].strip()
        
        if len(text) == 0:
            text = raw_text

    elif "CycleReviewer" in model:
        if "\n##" in raw_text:
            raw_parts = raw_text.split("\n##")
            texts = []
            for t in raw_parts:
                if "Review" in t.strip()[:12] and len(texts):
                    break
                head = t[:20].lower()
                if (
                    "No ethics review needed" in t
                    or "summary" in head
                    or "abstract" in head
                    or "intro" in head
                    or "related work" in head
                    or "conclusion" in head
                    or "rating" in head
                    or "score" in head
                    or ("soundness" in head and len(t) < 30)
                    or ("correctness" in head and len(t) < 30)
                    or ("clarity" in head and len(t) < 30)
                    or ("originality" in head and len(t) < 30)
                    or ("significance" in head and len(t) < 30)
                    or ("quality" in head and len(t) < 30)
                    or ("presentation" in head and len(t) < 30)
                    or ("contribution" in head and len(t) < 30)
                    or "confidence" in head
                    or "overall" in head
                    or "recommendation" in head
                    or "decision" in head
                    or "references" in head
                    or "reviewer" in head
                ):
                    continue
                texts.append(t)
            text = "##" + "\n##".join(texts) if texts else raw_text
        else:
            text = raw_text

    else:  # AI scientist
        ensure_ai_scientist_imports()
        review_text_dict = extract_json_between_markers(raw_text)
        if review_text_dict is not None and isinstance(review_text_dict, Dict):
            has_valid_strengths = review_text_dict.get("Strengths") is not None and review_text_dict.get("Strengths") != []
            has_valid_weaknesses = review_text_dict.get("Weaknesses") is not None and review_text_dict.get("Weaknesses") != []
        else:
            has_valid_strengths = False
            has_valid_weaknesses = False
        
        if has_valid_strengths and has_valid_weaknesses:
            strengths = review_text_dict.get("Strengths", [])
            weaknesses = review_text_dict.get("Weaknesses", [])
            questions = review_text_dict.get("Questions", [])
            limitation = review_text_dict.get("Limitations", [])
            
            strengths_string = "".join([f"{i_item+1}. {item}\n\n" for i_item, item in enumerate(strengths)])
            weaknesses_string = "".join([f"{i_item+1}. {item}\n\n" for i_item, item in enumerate(weaknesses)])
            
            text = "## Strengths\n" + strengths_string + "\n## Weaknesses\n" + weaknesses_string
            
            if questions:
                try:
                    questions_string = "".join([f"{i_item+1}. {item}\n\n" for i_item, item in enumerate(questions)])
                    text += "\n## Questions\n" + questions_string
                except:
                    pass

            if limitation:
                try:
                    limitation_string = "".join([f"{i_item+1}. {item}\n\n" for i_item, item in enumerate(limitation)])
                    text += "\n## Limitations\n" + limitation_string
                except:
                    pass
                
        else:
            text = raw_text

    # Optional: normalize to reduce odd Unicode variants.
    text = unicodedata.normalize("NFKC", text)
    return text



def parse_and_validate_args():
    """Parse command line arguments and validate business rules."""
    parser = argparse.ArgumentParser(description="DeepReviewer inference via vLLM API.")
    parser.add_argument("--max_workers", type=int, default=256, help="Thread pool size for LLMClient.")
    parser.add_argument("--base_url", default="http://localhost:40000/v1", help="OpenAI-compatible base URL (vLLM).")
    parser.add_argument("--api_key", default="", help="API key for OpenAI-compatible model.")
    parser.add_argument('--temperature', type=float, default=0.6, help='Temperature for generation (default: 0.6)')
    parser.add_argument('--max_tokens', type=int, default=32768, help='Max tokens for generation (default: 32768)')
    parser.add_argument('--max_input_tokens', type=int, default=65536, help='Max input tokens per paper (default: 65536)')
    parser.add_argument('--sample_data_path', type=str, default=None, help='Use 1300-sample benchmark (default: None; if unset, use full benchmark)')
    parser.add_argument('--benchmark_file', type=str, default=None, help=f'Path or glob pattern for benchmark JSONL files. If unset, use --sample_data_path or {DEFAULT_BENCHMARK_PATTERN}.')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Underlying local checkpoint path, open-source model name, or hosted API model name',
    )
    parser.add_argument('--mode', type=str, default=None, 
                        # choices=['OpenReviewer', 'CycleReviewer-8B', 'CycleReviewer-70B', 'SEA-E', 'DeepReviewer-7B', 'Qwen25-7B', 'DeepReviewer-14B', "Qwen3-8B", "Qwen3-32B", "QwQ-32B", "llama-31-8B", "llama-33-70B", "Nemotron-3-30B", "gpt-52", "gpt-5-mini", "gemini-3-flash", "gemini-3-pro",
                        help='Reviewer identifier / served model name. Defaults to basename(--model); unmatched modes use the AI-Scientist branch.')
    parser.add_argument('--no_think', action='store_true', default=False,
                        help='Disable thinking mode in model generation (default: False)')
    
    parser.add_argument("--num_fs_examples", type=int, default=3, help="num few shot examples for ai scientist")
    parser.add_argument('--reviewer_num', type=int, default=1, help='Number of reviewers to simulate (default: 1)')
    parser.add_argument('--general_config', action='store_true', default=False,
                        help='Use general configuration; specialized modes use original configuration by default')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Parse and validate arguments
    args = parse_and_validate_args()
    args = populate_runtime_args(args)
    
    # Apply configuration overrides
    logger = setup_logger(mode=f"{args.mode}")
    args = apply_config_overrides(args, logger)

    if args._is_vllm:
        proc = start_service(
            str((Path(PROJECT_ROOT) / "scripts" / "vllm.sh").resolve()), [
                "--model", str(args.model),
                "--base-name", str(args.mode),
                "--host", str(args._host),
                "--port", str(args._port)
            ])

        # Initialize tokenizer
        tokenizer_path = get_tokenizer_path(args)
        logger.info(f"Loading tokenizer from {tokenizer_path}...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        logger.info("Tokenizer loaded successfully!")
    else:
        tokenizer = None
        logger.info("Hosted API Do Not load Tokenizer")
    
    # Load dataset and build paths
    test_dataset, save_path, truncated_cases_path, _is_ai_scientist, eval_model_full_name = build_dataset_and_paths(args, logger)
    
    # Load existing results
    all_review_results, processed_ids = load_existing_results(save_path, logger)
    
    # Initialize tracking
    all_truncated_papers = []
    
    logger.info(f"\nStarting processing...")
    logger.info(f"Configuration: reviewer_num={args.reviewer_num}, temperature={args.temperature}")
        
    # Prepare batch data
    batch_ids, batch_paper_text, unprocessed_indices = prepare_batch_data(test_dataset, processed_ids, args.mode)
    
    logger.info(f"{len(unprocessed_indices)} unprocessed papers to generate")
    
    # Initialize client
    client = initialize_client(args, logger)
    # Route to the matching processor.
    if _is_ai_scientist:
        review_results, truncated_papers = process_ai_scientist_batch(
            batch_ids, batch_paper_text, client, tokenizer, args, logger, eval_model_full_name
        )
    else:
        review_results, truncated_papers = process_specialized_model_batch(
            batch_ids, batch_paper_text, client, tokenizer, args, logger
        )

    # Update results and tracking
    for result in review_results:
        if "id" in result and result["id"]:
            processed_ids.add(result["id"])
            all_review_results.append({"id":result["id"], "text": extract_review_text(result, args.mode)})
        
    all_truncated_papers.extend(truncated_papers)
    
    # Save progress
    logger.info(f" Processed {len(review_results)} papers, total: {len(all_review_results)}")

    with open(save_path, "w", encoding="utf-8") as f:
        for review_result in all_review_results:
            f.write(json.dumps(review_result, ensure_ascii=False) + "\n")
    
    if all_truncated_papers:
        save_truncated_log(all_truncated_papers, truncated_cases_path, args.max_input_tokens)
    
    # Final save and summary
    print_final_summary(logger, all_review_results, all_truncated_papers, save_path, truncated_cases_path, args.max_input_tokens)
    
    # Shutdown client
    if client is not None:
        client.shutdown()

    if args._is_vllm:
        stop_service(proc)
