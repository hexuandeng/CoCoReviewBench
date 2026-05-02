#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${RUN_PROCESS_REVIEW_CONFIG:-${SCRIPT_DIR}/run_process_review.local.conf}"

if [ -f "$CONFIG_FILE" ]; then
    # shellcheck disable=SC1090
    source "$CONFIG_FILE"
elif [ -n "${RUN_PROCESS_REVIEW_CONFIG:-}" ]; then
    echo "Config file not found: $CONFIG_FILE" >&2
    exit 1
fi

show_help() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
    --model MODEL_PATH      Path to the classify model
    --base-name NAME        Served model name
    --host HOST             Host to bind to
    --port PORT             Port to listen on
    -h, --help              Show this help message

Defaults come from:
    $CONFIG_FILE

CLI arguments override config values.
EOF
    exit 0
}

MODEL="${CLASSIFY_MODEL_DIR:-${SCRIPT_DIR}/../model/classify_model}"
BASE_NAME="${CLASSIFY_MODEL:-Qwen3-8B-Classify}"
HOST="${CLASSIFY_HOST:-0.0.0.0}"
PORT="${CLASSIFY_PORT:-13098}"
: "${GPU_NUM:?GPU_NUM must be set in $CONFIG_FILE}"
DATA_PARALLEL_SIZE="${GPU_NUM}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --base-name)
            BASE_NAME="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1" >&2
            show_help
            ;;
    esac
done

if ! [[ "$DATA_PARALLEL_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid gpu/data-parallel size: $DATA_PARALLEL_SIZE" >&2
    exit 1
fi

export VLLM_ATTENTION_BACKEND=FLASH_ATTN

echo "vLLM OpenAI server is up at: http://localhost:${PORT}/v1"
echo "Config file: ${CONFIG_FILE}"
echo "Model path: ${MODEL}"
echo "Base model name: ${BASE_NAME}"
echo "Host: ${HOST}"
echo "GPU/Data parallel size: ${DATA_PARALLEL_SIZE}"

python -m vllm.entrypoints.openai.api_server \
  --host "${HOST}" --port "${PORT}" \
  --model "${MODEL}" \
  --served-model-name "${BASE_NAME}" \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --data-parallel-size "${DATA_PARALLEL_SIZE}" \
  --max-log-len 10 \
  --gpu-memory-utilization 0.95 \
  --enable-prefix-caching
