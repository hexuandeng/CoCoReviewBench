#!/usr/bin/env bash
set -euo pipefail
export VLLM_CONFIGURE_LOGGING=0

show_help() {
    cat <<'EOF'
Usage: bash scripts/run_process_review.sh [OPTIONS] REVIEW_MODEL [BENCHMARK_FILE]

Run the full review pipeline:
  1. src/evaluation/_1_query_ai_reviewer.py
  2. src/evaluation/_0_human_review.py
  3. src/evaluation/_2_split.py
  4. src/evaluation/_3_classify.py
  5. src/evaluation/_4_evaluation.py

Positional arguments:
  REVIEW_MODEL     Reviewer model / mode passed to _1_query_ai_reviewer.py
                   and used as the output folder name. This is required.
  BENCHMARK_FILE   Benchmark JSONL file path or glob pattern. Default:
                   the value in scripts/run_process_review.local.conf, or
                   benchmark/benchmark_*_20*.jsonl

Options:
  --is_vllm                    Include to use local vLLM generation at
                               http://localhost:40000/v1. Omit to use
                               BASE_URL/API_KEY.
  --review_no_think            Include to pass --no_think to generation.
  --eval_category_level        Include to pass --category_level_eval.
  --eval_category_incorrect    Include to pass --category_incorrect_eval.
  --eval_old_metrics           Include to pass --calc_old_metrics.
  --eval_paper_level           Include to pass --paper_level_eval.
                               If no --eval_* flags are provided, evaluation
                               is skipped.
  -h, --help                   Show this help message

Config file:
  By default the script sources:
    scripts/run_process_review.local.conf
  You can override that path with:
    RUN_PROCESS_REVIEW_CONFIG=/path/to/your.conf
  The config file uses normal bash variable assignments, for example:
    BENCHMARK_FILE="benchmark/benchmark_*_20*.jsonl"
    SAMPLE_DATA_PATH=""
    BASE_URL="https://your-api.example/v1"
    API_KEY="..."
    GPU_NUM=8

Common environment variables:
  BASE_URL                           Shared hosted API base URL for generation/eval.
                                     Default: $OPENAI_BASE_URL
  API_KEY                            Shared hosted API key for generation/eval.
                                     Default: $OPENAI_API_KEY
  REVIEWER_NUM                       Passed to _1_query_ai_reviewer.py --reviewer_num
  REVIEW_GENERAL_CONFIG              Truthy to add --general_config

  GPU_NUM                            Shared GPU count for split/classify vLLM
  SPLIT_MODEL / CLASSIFY_MODEL       Local split/classify served model names
  SPLIT_PORT / CLASSIFY_PORT         Local split/classify server ports
  SPLIT_EFFORT / CLASSIFY_EFFORT     Reasoning effort passed to split/classify
  SPLIT_MAX_WORKERS / CLASSIFY_MAX_WORKERS
                                     Worker counts for split/classify
  CACHE_VERSION                      Cache version shared by split/classify/eval

  EVAL_MODEL                         Judge model for _4_evaluation.py.
  EVAL_OUTPUT_FILE                   Evaluation JSON output path
  EVAL_COMPLETE_HUMAN_CLASSIFY_FILE  Human classify reference file
EOF
}

POSITIONAL_ARGS=()
IS_VLLM="0"
REVIEW_NO_THINK="0"
EVAL_CATEGORY_LEVEL="0"
EVAL_CATEGORY_INCORRECT="0"
EVAL_OLD_METRICS="0"
EVAL_PAPER_LEVEL="0"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --is_vllm)
            IS_VLLM="1"
            shift
            ;;
        --review_no_think)
            REVIEW_NO_THINK="1"
            shift
            ;;
        --eval_category_level)
            EVAL_CATEGORY_LEVEL="1"
            shift
            ;;
        --eval_category_incorrect)
            EVAL_CATEGORY_INCORRECT="1"
            shift
            ;;
        --eval_old_metrics)
            EVAL_OLD_METRICS="1"
            shift
            ;;
        --eval_paper_level)
            EVAL_PAPER_LEVEL="1"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        --)
            shift
            while [ "$#" -gt 0 ]; do
                POSITIONAL_ARGS+=("$1")
                shift
            done
            ;;
        -*)
            echo "Unknown option: $1" >&2
            show_help >&2
            exit 1
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_CONFIG_FILE="${SCRIPT_DIR}/run_process_review.local.conf"
CONFIG_FILE="${RUN_PROCESS_REVIEW_CONFIG:-$DEFAULT_CONFIG_FILE}"

if [ -f "$CONFIG_FILE" ]; then
    # shellcheck disable=SC1090
    source "$CONFIG_FILE"
elif [ -n "${RUN_PROCESS_REVIEW_CONFIG:-}" ]; then
    echo "Config file not found: $CONFIG_FILE" >&2
    exit 1
fi

normalize_bool() {
    case "${1:-}" in
        1|true|TRUE|yes|YES|on|ON)
            printf '1\n'
            ;;
        0|false|FALSE|no|NO|off|OFF)
            printf '0\n'
            ;;
        *)
            echo "Invalid boolean value: ${1:-<empty>}" >&2
            exit 1
            ;;
    esac
}

sanitize_mode_name() {
    local value="$1"
    value="${value//\//_}"
    printf '%s\n' "$value"
}

is_specialized_review_mode() {
    case "${1:-}" in
        CycleReviewer-Llama-3.1-8B|CycleReviewer-Llama-3.1-70B|DeepReviewer-7B|DeepReviewer-14B|Llama-OpenReviewer-8B|SEA-E)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

is_truthy() {
    [ "$(normalize_bool "${1:-0}")" = "1" ]
}

has_eval_flags() {
    [ "$EVAL_CATEGORY_LEVEL" = "1" ] || \
    [ "$EVAL_CATEGORY_INCORRECT" = "1" ] || \
    [ "$EVAL_OLD_METRICS" = "1" ] || \
    [ "$EVAL_PAPER_LEVEL" = "1" ]
}

REVIEW_MODEL="${POSITIONAL_ARGS[0]:-}"
if [ -z "$REVIEW_MODEL" ]; then
    echo "Missing required positional argument: REVIEW_MODEL" >&2
    show_help >&2
    exit 1
fi

BASE_URL="${BASE_URL:-${OPENAI_BASE_URL:-}}"
API_KEY="${API_KEY:-${OPENAI_API_KEY:-}}"
REVIEWER_NUM="${REVIEWER_NUM:-1}"
: "${VLLM_BASE_URL:?VLLM_BASE_URL must be set in $CONFIG_FILE}"
: "${GPU_NUM:?GPU_NUM must be set in $CONFIG_FILE}"
: "${SPLIT_MODEL:?SPLIT_MODEL must be set in $CONFIG_FILE}"
: "${CLASSIFY_MODEL:?CLASSIFY_MODEL must be set in $CONFIG_FILE}"
: "${SPLIT_PORT:?SPLIT_PORT must be set in $CONFIG_FILE}"
: "${CLASSIFY_PORT:?CLASSIFY_PORT must be set in $CONFIG_FILE}"
: "${SPLIT_EFFORT:?SPLIT_EFFORT must be set in $CONFIG_FILE}"
: "${CLASSIFY_EFFORT:?CLASSIFY_EFFORT must be set in $CONFIG_FILE}"
: "${SPLIT_MAX_WORKERS:?SPLIT_MAX_WORKERS must be set in $CONFIG_FILE}"
: "${CLASSIFY_MAX_WORKERS:?CLASSIFY_MAX_WORKERS must be set in $CONFIG_FILE}"
: "${CACHE_VERSION:?CACHE_VERSION must be set in $CONFIG_FILE}"
: "${EVAL_COMPLETE_HUMAN_CLASSIFY_FILE:?EVAL_COMPLETE_HUMAN_CLASSIFY_FILE must be set in $CONFIG_FILE}"
: "${EVAL_MODEL:?EVAL_MODEL must be set in $CONFIG_FILE}"
: "${EVAL_MAX_WORKERS:?EVAL_MAX_WORKERS must be set in $CONFIG_FILE}"
: "${EVAL_EFFORT:?EVAL_EFFORT must be set in $CONFIG_FILE}"

BENCHMARK_FILE="${POSITIONAL_ARGS[1]:-${BENCHMARK_FILE:-benchmark/benchmark_*_20*.jsonl}}"
SAMPLE_DATA_PATH="${SAMPLE_DATA_PATH:-}"

REVIEW_MODE="$REVIEW_MODEL"
if [ "$IS_VLLM" = "1" ]; then
    REVIEW_MODE="$(basename -- "$REVIEW_MODEL")"
fi
REVIEW_MODE="$(sanitize_mode_name "$REVIEW_MODE")"

REVIEW_OUTPUT_SUFFIX=""
if is_truthy "${REVIEW_GENERAL_CONFIG:-0}" && is_specialized_review_mode "$REVIEW_MODE"; then
    REVIEW_OUTPUT_SUFFIX="${REVIEW_OUTPUT_SUFFIX}-general-config"
fi
if [ "$REVIEW_NO_THINK" = "1" ]; then
    REVIEW_OUTPUT_SUFFIX="${REVIEW_OUTPUT_SUFFIX}-no-think"
fi

SAMPLE_SUFFIX=""
if [ -n "$SAMPLE_DATA_PATH" ]; then
    SAMPLE_SUFFIX="_sample"
fi

INPUT_DIR="${ROOT_DIR}/review_result/${REVIEW_MODE}${REVIEW_OUTPUT_SUFFIX}-result"
INPUT_JSONL="${INPUT_DIR}/result${SAMPLE_SUFFIX}.jsonl"
SPLIT_JSONL="${INPUT_DIR}/result${SAMPLE_SUFFIX}_split.jsonl"
CLASSIFY_JSONL="${INPUT_DIR}/result${SAMPLE_SUFFIX}_classify.jsonl"
EVAL_OUTPUT_FILE="${EVAL_OUTPUT_FILE:-${INPUT_DIR}/evaluation${SAMPLE_SUFFIX}.json}"
EVAL_BENCHMARK_FILE="${EVAL_BENCHMARK_FILE:-${BENCHMARK_FILE}}"

VLLM_SPLIT_SCRIPT="${ROOT_DIR}/scripts/vllm_split.sh"
VLLM_CLASSIFY_SCRIPT="${ROOT_DIR}/scripts/vllm_classify.sh"
HUMAN_REVIEW_PY="${ROOT_DIR}/src/evaluation/_0_human_review.py"
QUERY_AI_REVIEWER_PY="${ROOT_DIR}/src/evaluation/_1_query_ai_reviewer.py"
SPLIT_PY="${ROOT_DIR}/src/evaluation/_2_split.py"
CLASSIFY_PY="${ROOT_DIR}/src/evaluation/_3_classify.py"
EVALUATION_PY="${ROOT_DIR}/src/evaluation/_4_evaluation.py"

cleanup_vllm() {
    pkill -f "vllm.entrypoints.openai.api_server" >/dev/null 2>&1 || true
}

trap cleanup_vllm EXIT INT TERM

run_query_ai_reviewer() {
    local review_base_url="$BASE_URL"
    local cmd=(
        python "$QUERY_AI_REVIEWER_PY"
        --model "$REVIEW_MODEL"
        --mode "$REVIEW_MODE"
        --reviewer_num "$REVIEWER_NUM"
    )

    if [ -n "$SAMPLE_DATA_PATH" ]; then
        cmd+=(--sample_data_path "$SAMPLE_DATA_PATH")
    else
        cmd+=(--benchmark_file "$BENCHMARK_FILE")
    fi

    if [ "$IS_VLLM" = "1" ]; then
        review_base_url="$VLLM_BASE_URL"
    elif [ -z "$review_base_url" ]; then
        echo "BASE_URL is required when --is_vllm is omitted" >&2
        exit 1
    fi

    cmd+=(--base_url "$review_base_url")

    if [ -n "$API_KEY" ]; then
        cmd+=(--api_key "$API_KEY")
    fi

    if [ -n "${REVIEW_MAX_WORKERS:-}" ]; then
        cmd+=(--max_workers "$REVIEW_MAX_WORKERS")
    fi

    if [ -n "${REVIEW_TEMPERATURE:-}" ]; then
        cmd+=(--temperature "$REVIEW_TEMPERATURE")
    fi

    if [ -n "${REVIEW_MAX_TOKENS:-}" ]; then
        cmd+=(--max_tokens "$REVIEW_MAX_TOKENS")
    fi

    if [ -n "${REVIEW_MAX_INPUT_TOKENS:-}" ]; then
        cmd+=(--max_input_tokens "$REVIEW_MAX_INPUT_TOKENS")
    fi

    if [ -n "${REVIEW_NUM_FS_EXAMPLES:-}" ]; then
        cmd+=(--num_fs_examples "$REVIEW_NUM_FS_EXAMPLES")
    fi

    if [ "$REVIEW_NO_THINK" = "1" ]; then
        cmd+=(--no_think)
    fi

    if is_truthy "${REVIEW_GENERAL_CONFIG:-0}"; then
        cmd+=(--general_config)
    fi

    echo "Running AI reviewer generation..."
    "${cmd[@]}"
}

run_evaluation() {
    if ! has_eval_flags; then
        echo "Skipping evaluation: no --eval_* flags were provided."
        return 0
    fi

    local cmd=(
        python "$EVALUATION_PY"
        --input_file "$CLASSIFY_JSONL"
        --output_file "$EVAL_OUTPUT_FILE"
        --benchmark_file "$EVAL_BENCHMARK_FILE"
        --complete_human_classify_file "$EVAL_COMPLETE_HUMAN_CLASSIFY_FILE"
        --model "$EVAL_MODEL"
        --max_workers "$EVAL_MAX_WORKERS"
        --cache_version "$CACHE_VERSION"
        --effort "$EVAL_EFFORT"
    )

    if [ -n "$BASE_URL" ]; then
        cmd+=(--base_url "$BASE_URL")
    fi

    if [ -n "$API_KEY" ]; then
        cmd+=(--api_key "$API_KEY")
    fi

    if [ "$EVAL_CATEGORY_LEVEL" = "1" ]; then
        cmd+=(--category_level_eval)
    fi

    if [ "$EVAL_CATEGORY_INCORRECT" = "1" ]; then
        cmd+=(--category_incorrect_eval)
    fi

    if [ "$EVAL_OLD_METRICS" = "1" ]; then
        cmd+=(--calc_old_metrics)
    fi

    if [ "$EVAL_PAPER_LEVEL" = "1" ]; then
        cmd+=(--paper_level_eval)
    fi

    echo "Running evaluation..."
    "${cmd[@]}"
}

cd "$ROOT_DIR"
mkdir -p "${ROOT_DIR}/logs"
export GPU_NUM="$GPU_NUM"
export SPLIT_DATA_PARALLEL_SIZE="$GPU_NUM"
export CLASSIFY_DATA_PARALLEL_SIZE="$GPU_NUM"

echo "Config file: $CONFIG_FILE"
echo "is_vllm: $IS_VLLM"
echo "Benchmark file: $BENCHMARK_FILE"
echo "Sample data path: ${SAMPLE_DATA_PATH:-<unset>}"
echo "GPU num: $GPU_NUM"
echo "Review model: $REVIEW_MODEL"
echo "Review mode: $REVIEW_MODE"
if [ "$IS_VLLM" = "1" ]; then
    echo "Generation base URL: $VLLM_BASE_URL"
else
    echo "Generation base URL: ${BASE_URL:-<unset>}"
fi
echo "AI review output: $INPUT_JSONL"
echo "Split output: $SPLIT_JSONL"
echo "Classify output: $CLASSIFY_JSONL"
if has_eval_flags; then
    echo "Evaluation output: $EVAL_OUTPUT_FILE"
else
    echo "Evaluation: skipped"
fi

run_query_ai_reviewer

echo "Building human-review files..."
python "$HUMAN_REVIEW_PY" --benchmark_file "$BENCHMARK_FILE"

if [ ! -f "$INPUT_JSONL" ]; then
    echo "Input file not found: $INPUT_JSONL" >&2
    exit 1
fi

echo "Input JSONL: $INPUT_JSONL"

cleanup_vllm

echo "Starting split server..."
bash "$VLLM_SPLIT_SCRIPT" &

echo "Running split..."
python "$SPLIT_PY" \
    --input_jsonl "$INPUT_JSONL" \
    --output_jsonl "$SPLIT_JSONL" \
    --model "$SPLIT_MODEL" \
    --base_url "http://localhost:${SPLIT_PORT}/v1" \
    --max_workers "$SPLIT_MAX_WORKERS" \
    --effort "$SPLIT_EFFORT" \
    --cache_version "$CACHE_VERSION"

cleanup_vllm

echo "Starting classify server..."
bash "$VLLM_CLASSIFY_SCRIPT" &

echo "Running classify..."
python "$CLASSIFY_PY" \
    --input_jsonl "$SPLIT_JSONL" \
    --output_jsonl "$CLASSIFY_JSONL" \
    --model "$CLASSIFY_MODEL" \
    --base_url "http://localhost:${CLASSIFY_PORT}/v1" \
    --max_workers "$CLASSIFY_MAX_WORKERS" \
    --effort "$CLASSIFY_EFFORT" \
    --cache_version "$CACHE_VERSION"

cleanup_vllm

if has_eval_flags; then
    run_evaluation
fi

echo "Done."
echo "Classified JSONL: $CLASSIFY_JSONL"
if has_eval_flags; then
    echo "Evaluation JSON: $EVAL_OUTPUT_FILE"
fi
