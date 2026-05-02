<div align="center">
  <h1>CoCoReviewBench</h1>
  <p><b>A Completeness- and Correctness-Oriented Benchmark for AI Reviewers</b></p>

  <p>
    <a href="#installation"><img alt="Python 3.12+" src="https://img.shields.io/badge/Python-3.12%2B-3776AB?style=flat-square&logo=python&logoColor=white" /></a>
    <a href="#license"><img alt="License CC BY 4.0" src="https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey?style=flat-square" /></a>
    <a href="#evaluation-pipeline"><img alt="Pipeline Evaluation" src="https://img.shields.io/badge/Pipeline-Evaluation-2EA44F?style=flat-square" /></a>
    <a href="https://huggingface.co/collections/hexuandeng/suprareviewbench-694aa27412658329fca6577f"><img alt="Hugging Face Collection" src="https://img.shields.io/badge/Hugging%20Face-Collection-FFD21E?style=flat-square&logo=huggingface&logoColor=black" /></a>
  </p>

  <p>
    <a href="#overview">Overview</a> |
    <a href="#evaluation-pipeline">Evaluation</a> |
    <a href="#benchmark-construction">Benchmark</a> |
    <a href="#license">License</a> |
    <a href="#citation">Citation</a>
  </p>
</div>

## Overview

CoCoReviewBench evaluates AI-generated paper reviews against correctness-filtered human references derived from OpenReview discussions.

The current codebase has two main parts:

- `src/evaluation/`: generate AI reviews, split them into atomic opinions, classify them into the taxonomy, and evaluate them.
- `src/benchmark/`: build the benchmark itself from OpenReview discussions.

For AI-review evaluation, the recommended workflow is:

1. Run `scripts/run_process_review.sh` with your reviewer model. The benchmark-file argument is optional.
2. The script generates the AI-review JSONL, prepares human references, splits the review into atomic opinions, classifies them into the taxonomy, and optionally evaluates them.
3. Use the lower-level Python entrypoints in `src/evaluation/` only if you need to customize or debug individual stages.

<p align="center">
  <img src="pic/framework.png" width="900" alt="CoCoReviewBench framework" />
</p>

What "CoCo" means:

- **Co**mpleteness: evaluate by category and avoid penalizing a model for categories that are absent from the human reference.
- **Co**rrectness: use reviewer-reviewer and reviewer-author disagreements, with meta-review context, to filter out incorrect human opinions.

## What Is Actually In This Repo

The source tree is the part intended to be shared. Large local artifacts are part of the workflow, but they are not clean source files:

- `benchmark/`: benchmark JSONL files, markdown caches, and human-reference artifacts expected by the code.
- `review_result/`: generated AI reviews and processed outputs.
- `model/`: local split/classify checkpoints used by the vLLM launcher scripts.
- `logs/`, PDFs, caches, and other experiment artifacts.

In other words, the code assumes these paths can exist locally, but a source-only release should not assume they are committed.

The PDFs corresponding to the benchmark papers are available in this [Google Drive folder](https://drive.google.com/drive/folders/1QIRH7G7Me4t1dhR5pjTvPAQg9eH8u2x1?usp=sharing).

### Model Checkpoints

The split and classify stages use two local vLLM-served checkpoints. They are available from the [SupraReviewBench Hugging Face collection](https://huggingface.co/collections/hexuandeng/suprareviewbench):

- Split model: [hexuandeng/ReviewSplit](https://huggingface.co/hexuandeng/ReviewSplit)
- Classify model: [hexuandeng/ReviewClassify](https://huggingface.co/hexuandeng/ReviewClassify)

Download them to the default paths expected by `scripts/vllm_split.sh` and `scripts/vllm_classify.sh`:

```bash
pip install -U "huggingface_hub[cli]"

hf download hexuandeng/ReviewSplit \
  --local-dir model/split_model

hf download hexuandeng/ReviewClassify \
  --local-dir model/classify_model
```

## Installation

Requirements:

- Python 3.12+
- Bash for `scripts/*.sh`
- For local vLLM workflows: a CUDA-compatible GPU environment and a Unix-like shell environment

Bash:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Notes:

- `requirements.txt` covers the current evaluation/runtime pipeline, including the AI-Scientist reviewer path under `src/evaluation/ai_scientist/`, plus benchmark crawling and metric packages.
- It does **not** install the training stack used by `src/benchmark/_9_sft.py`, `scripts/ReviewSplit_train.sh`, or `scripts/ReviewClassify_train.sh`.

## Evaluation Pipeline

The recommended entrypoint for AI-review evaluation is `scripts/run_process_review.sh`. This README treats it as the main command users should run. In the current repo, the script first calls `src/evaluation/_1_query_ai_reviewer.py` to generate the AI-review JSONL, then prepares human references, runs split and classify, and optionally runs evaluation.

### 1. Entry Point: `scripts/run_process_review.sh`

Before using the shell scripts, make sure `scripts/run_process_review.local.conf` is already configured:

```bash
# edit scripts/run_process_review.local.conf before running
```

You can also override the config path with `RUN_PROCESS_REVIEW_CONFIG=/path/to/your.conf`.

The `BENCHMARK_FILE` positional argument is optional: if omitted, the script uses the value from `scripts/run_process_review.local.conf`, or falls back to `benchmark/benchmark_*_20*.jsonl`.

Example:

```bash
bash scripts/run_process_review.sh gpt-5-mini
```

That command first generates:

- `review_result/gpt-5-mini-result/result.jsonl`

It then runs the internal processing stages for human-reference preparation, split, and classify.

It produces:

- `benchmark/benchmark_all_human.jsonl`
- `benchmark/benchmark_left_out_human.jsonl`
- `benchmark/benchmark_reviewer_selection.json`
- `review_result/gpt-5-mini-result/result_split.jsonl`
- `review_result/gpt-5-mini-result/result_classify.jsonl`

In the current shell script, the processed review path is derived from a small set of config and CLI switches:

- `REVIEW_MODEL` is the first positional argument. For external API generation, it is typically the remote model name; for `--is_vllm`, it is typically a local model path or local served model name. The script uses it for generation and to derive the base reviewer folder name.
- `--is_vllm` switches generation from the external `BASE_URL`/`API_KEY` path to the local vLLM path (`VLLM_BASE_URL`). In that mode, the script also uses `basename(REVIEW_MODEL)` as the reviewer folder name, since `REVIEW_MODEL` is typically a local checkpoint path.
- `--review_no_think` tells the generation step to use non-thinking mode. This is mainly relevant for reasoning-capable models, but it can also be used with models that already behave as non-thinking models. In the output layout, it appends `-no-think` to the reviewer folder name.

If you want evaluation in the same command, pass at least one evaluation flag:

```bash
bash scripts/run_process_review.sh \
  --eval_category_level \
  --eval_old_metrics \
  --eval_paper_level \
  gpt-5-mini
```

If no `--eval_*` flag is provided, the shell script skips evaluation entirely.

The evaluation step uses the configured `EVAL_MODEL`, `EVAL_COMPLETE_HUMAN_CLASSIFY_FILE`, `EVAL_EFFORT`, and optional `EVAL_OUTPUT_FILE`. If `EVAL_OUTPUT_FILE` is left unset, the default output path is derived from the same reviewer folder and sample suffix.

That additionally writes:

- `review_result/gpt-5-mini-result/evaluation.json`

### 2. Low-Level Review Generation Details

If you want to generate the initial AI-review JSONL inside this repo instead of supplying it from elsewhere, `src/evaluation/_1_query_ai_reviewer.py` reads benchmark JSONL files, loads each paper's markdown, and writes AI-review text to:

- `review_result/<mode>{suffix}-result/result{sample_suffix}.jsonl`
- `review_result/<mode>{suffix}-result/truncated_cases{sample_suffix}.txt`

Important behavior from the current code:

- If `--mode` is one of `CycleReviewer-Llama-3.1-8B`, `CycleReviewer-Llama-3.1-70B`, `DeepReviewer-7B`, `DeepReviewer-14B`, `Llama-OpenReviewer-8B`, or `SEA-E`, the script uses specialized parsing/config rules.
- Any other mode falls back to the AI-Scientist branch in `src/evaluation/ai_scientist/`.
- If `--mode` is omitted, it defaults to `basename(--model)`.
- In practice, the current CLI usage should pass `--model` explicitly.
- Existing `result*.jsonl` files are loaded so interrupted runs can resume.
- Output-folder suffixes are controlled by `--general_config` and `--no_think`.
- `--sample_data_path` changes the output filename from `result.jsonl` to `result_sample.jsonl`.

Examples:

```bash
# Local vLLM reviewer
python src/evaluation/_1_query_ai_reviewer.py \
  --benchmark_file benchmark/benchmark_ICLR_2025.jsonl \
  --model /path/to/your/model \
  --mode qwen3-32b-local \
  --base_url http://localhost:40000/v1
```

```bash
# Hosted API reviewer
python src/evaluation/_1_query_ai_reviewer.py \
  --benchmark_file benchmark/benchmark_ICLR_2025.jsonl \
  --model gpt-5-mini \
  --mode gpt-5-mini \
  --base_url https://your-api.example/v1 \
  --api_key YOUR_API_KEY
```

### 3. Run Evaluation Directly

`src/evaluation/_4_evaluation.py` evaluates processed AI reviews against the benchmark references.

Current inputs used by the code:

- `--input_file`: usually `review_result/<reviewer>-result/result_classify.jsonl`
- `--benchmark_file`: one or more benchmark JSONL files
- `--complete_human_classify_file`: usually `benchmark/benchmark_all_human_classify.jsonl`
- `benchmark/benchmark_reviewer_selection.json`: created by `_0_human_review.py` and used to define the left-out reviewer selection

Example:

```bash
python src/evaluation/_4_evaluation.py \
  --input_file review_result/gpt-5-mini-result/result_classify.jsonl \
  --output_file review_result/gpt-5-mini-result/evaluation_iclr2025.json \
  --benchmark_file benchmark/benchmark_ICLR_2025.jsonl \
  --complete_human_classify_file benchmark/benchmark_all_human_classify.jsonl \
  --category_level_eval \
  --calc_old_metrics \
  --paper_level_eval \
  --model gpt-5-mini \
  --base_url https://your-api.example/v1 \
  --api_key YOUR_API_KEY
```

Important behavior difference:

- If you call `_4_evaluation.py` directly **without** any evaluation flags, it automatically enables category-level evaluation, old metrics, and paper-level evaluation.
- `scripts/run_process_review.sh` does **not** do that. It skips evaluation unless you explicitly pass at least one `--eval_*` flag.

## Benchmark Construction

The scripts in `src/benchmark/` are the research pipeline used to build the benchmark. They are useful, but they are not packaged as a single polished release command. Several stages assume the venue/year directory layout encoded in the files, and some stages iterate over `src.utils.GROUPS`.

Current stage mapping:

0. `src/benchmark/_0_openreview_crawler.py`: crawl OpenReview submissions/discussions and pick a review-time PDF.
0. `src/benchmark/_0_format_reviews.py`: normalize raw OpenReview replies into reviewer/metareview thread format.
0. `src/benchmark/_0_sample_data.py`: filter and sample per-venue/year subsets.
1. `src/benchmark/_1_split_reviews.py`: split review discussions into atomic discussion points.
2. `src/benchmark/_2_classify_reviews.py`: classify split blocks into taxonomy labels.
3. `src/benchmark/_3_preprocess_file.py`: convert text spans into sentence-id form and build benchmark-style objects. This script iterates over `src.utils.GROUPS`.
3. `src/benchmark/_3_refine_split_and_classify.py`: refine multi-label blocks at the sentence level.
4. `src/benchmark/_4_review_clustering.py`: assign discussion point IDs across reviewer comments.
5. `src/benchmark/_5_conflict_reviews.py`: detect reviewer-reviewer conflicts.
5. `src/benchmark/_5_validate_conflict_reviews.py`: adjudicate conflicted reviewer opinions.
6. `src/benchmark/_6_author_conflict.py`: detect explicit author refutations of reviewer opinions.
6. `src/benchmark/_6_validate_author_conflict.py`: validate whether intersected refutations correspond to actually wrong reviewer opinions.
7. `src/benchmark/_7_postprocess_benchmark.py`: merge multi-model validation outputs and assemble `*_with_reviewer_validations.jsonl`.
7. `src/benchmark/_7_merge_benchmark.py`: merge all venue/year outputs into a final benchmark JSONL. This script also iterates over `src.utils.GROUPS`.

Representative commands for one venue/year:

```bash
python src/benchmark/_0_openreview_crawler.py --org ICLR.cc --year 2024
python src/benchmark/_0_format_reviews.py --org ICLR --year 2024 --postfix clean
python src/benchmark/_0_sample_data.py --orgs ICLR --years 2024 --k 300

python src/benchmark/_1_split_reviews.py --org ICLR --year 2024 --model gpt-5 --base_url https://your-api.example/v1 --api_key YOUR_API_KEY
python src/benchmark/_2_classify_reviews.py --org ICLR --year 2024 --model gpt-5-mini --base_url https://your-api.example/v1 --api_key YOUR_API_KEY
python src/benchmark/_3_refine_split_and_classify.py --org ICLR --year 2024 --model gpt-5-mini --base_url https://your-api.example/v1 --api_key YOUR_API_KEY

python src/benchmark/_4_review_clustering.py --org ICLR --year 2024 --model deepseek-reasoner --base_url https://your-api.example/v1 --api_key YOUR_API_KEY
python src/benchmark/_5_conflict_reviews.py --org ICLR --year 2024 --model deepseek-reasoner --base_url https://your-api.example/v1 --api_key YOUR_API_KEY
python src/benchmark/_5_validate_conflict_reviews.py --org ICLR --year 2024 --model deepseek-reasoner --base_url https://your-api.example/v1 --api_key YOUR_API_KEY

python src/benchmark/_6_author_conflict.py --org ICLR --year 2024 --model gpt-5-mini --base_url https://your-api.example/v1 --api_key YOUR_API_KEY
python src/benchmark/_6_validate_author_conflict.py --org ICLR --year 2024 --model deepseek-reasoner --base_url https://your-api.example/v1 --api_key YOUR_API_KEY
python src/benchmark/_7_postprocess_benchmark.py --org ICLR --year 2024
```

Notes:

- The benchmark pipeline uses the prompts defined in `src/benchmark/prompt_registry.py`.
- Scripts that use `BaseArguments` share the same `--org`, `--year`, `--model`, `--effort`, `--max_workers`, `--base_url`, `--api_key`, and `--cache_version` conventions.
- For `_3_preprocess_file.py` and `_7_merge_benchmark.py`, update `src/utils.py` if you want to limit processing to a subset of `GROUPS`.

## Training Helpers

The repository includes training helpers, but they are templates rather than turnkey scripts:

- `src/benchmark/_9_sft.py`: generic HuggingFace/LoRA SFT entrypoint for text or chat-style data.
- `src/benchmark/_9_split_reward.py`: reward function used by split-model RL training.
- `scripts/ReviewSplit_train.sh`: local training template for split-model RL.
- `scripts/ReviewClassify_train.sh`: local training template for the classify model.

Notes:

- These training helpers require additional packages such as `torch`, `peft`, `deepspeed`, or `verl`.
- They also contain local-path assumptions and should be edited for your environment before use.

## License

- Repository license: see `LICENSE`.
- `src/benchmark/_9_sft.py` is adapted from HuggingFace Transformers example code and keeps its own attribution in the file header.
- OpenReview data, cached PDFs, markdown files, and other third-party content remain subject to their own terms.

## Citation

```bibtex
@inproceedings{deng2026cocoreviewbench,
  title     = {{CoCoReviewBench}: A Completeness- and Correctness-Oriented Benchmark for {AI} Reviewers},
  author    = {Deng, Hexuan and Li, Yichen and Ke, Xiaopeng and Hu, Ruina and Wong, Derek F. and Wang, Yue and Liu, Xuebo and Huang, Dehao and Zhang, Min},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  year      = {2026},
  note      = {To appear},
  url       = {https://github.com/hexuandeng/CoCoReviewBench}
}
```
