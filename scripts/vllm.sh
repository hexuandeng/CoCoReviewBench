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
    --model MODEL_PATH      Path to the model
    --base-name NAME        Base model name for OpenAI client
    --host HOST             Host to bind to
    --port PORT             Port to listen on
    -h, --help              Show this help message

Defaults come from:
    $CONFIG_FILE

CLI arguments override config values.
EOF
    exit 0
}

parse_host_from_url() {
    local url="$1"
    url="${url#*://}"
    url="${url%%/*}"
    printf '%s\n' "${url%%:*}"
}

parse_port_from_url() {
    local url="$1"
    url="${url#*://}"
    url="${url%%/*}"
    if [[ "$url" == *:* ]]; then
        printf '%s\n' "${url##*:}"
    fi
}

MODEL="${VLLM_MODEL:-${MODEL:-}}"
BASE_NAME="${VLLM_BASE_NAME:-${BASE_NAME:-}}"
HOST="${VLLM_HOST:-}"
PORT="${VLLM_PORT:-}"
if [ -z "$HOST" ] && [ -n "${VLLM_BASE_URL:-}" ]; then
    HOST="$(parse_host_from_url "$VLLM_BASE_URL")"
fi
if [ -z "$PORT" ] && [ -n "${VLLM_BASE_URL:-}" ]; then
    PORT="$(parse_port_from_url "$VLLM_BASE_URL")"
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-40000}"

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

if [ -z "$MODEL" ]; then
    echo "MODEL is required. Pass --model or set VLLM_MODEL in $CONFIG_FILE" >&2
    exit 1
fi

if [ -z "$BASE_NAME" ]; then
    BASE_NAME="$(basename "$MODEL")"
fi

: "${GPU_NUM:?GPU_NUM must be set in $CONFIG_FILE}"
if ! [[ "$GPU_NUM" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid GPU count: $GPU_NUM" >&2
    exit 1
fi

pick_tensor_parallel_size() {
    local gpu_num="$1"
    shift
    local candidate

    for candidate in "$@"; do
        if [ "$candidate" -le "$gpu_num" ] && [ $((gpu_num % candidate)) -eq 0 ]; then
            printf '%s\n' "$candidate"
            return
        fi
    done

    for candidate in "$@"; do
        if [ "$candidate" -le "$gpu_num" ]; then
            printf '%s\n' "$candidate"
            return
        fi
    done

    printf '1\n'
}

echo "=========================================="
echo "vLLM Server Configuration"
echo "=========================================="
echo "Config File: ${CONFIG_FILE}"
echo "Model Path:  ${MODEL}"
echo "Base Name:   ${BASE_NAME}"
echo "Host:        ${HOST}"
echo "Port:        ${PORT}"
echo "GPU Count:   ${GPU_NUM}"
echo "=========================================="

export VLLM_ATTENTION_BACKEND=FLASH_ATTN

MODEL_NAME="$(basename -- "${MODEL}")"

if [ "${MODEL_NAME}" = "DeepReviewer-14B" ]; then
    TENSOR_PARALLEL_SIZE="$(pick_tensor_parallel_size "$GPU_NUM" 2 1)"
    DATA_PARALLEL_SIZE=$((GPU_NUM / TENSOR_PARALLEL_SIZE))
    echo "Detected DeepReviewer-14B: adapting parallelism for ${GPU_NUM} GPU(s)"
elif [ "${MODEL_NAME}" = "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16" ]; then
    TENSOR_PARALLEL_SIZE="$(pick_tensor_parallel_size "$GPU_NUM" 8 4 2 1)"
    DATA_PARALLEL_SIZE=1
    echo "Detected Nemotron-H (Mamba hybrid): adapting tensor parallelism for ${GPU_NUM} GPU(s), DP disabled"
elif [ "${MODEL_NAME}" = "Qwen3-32B" ] || [ "${MODEL_NAME}" = "QwQ-32B" ]; then
    TENSOR_PARALLEL_SIZE="$(pick_tensor_parallel_size "$GPU_NUM" 4 2 1)"
    DATA_PARALLEL_SIZE=$((GPU_NUM / TENSOR_PARALLEL_SIZE))
    echo "Detected Qwen3-32B/QwQ-32B: adapting parallelism for ${GPU_NUM} GPU(s)"
elif [ "${MODEL_NAME}" = "Llama-3.3-70B-Instruct" ]; then
    TENSOR_PARALLEL_SIZE="$(pick_tensor_parallel_size "$GPU_NUM" 8 4 2 1)"
    DATA_PARALLEL_SIZE=$((GPU_NUM / TENSOR_PARALLEL_SIZE))
    echo "Detected Llama-3.3-70B-Instruct: adapting parallelism for ${GPU_NUM} GPU(s)"
else
    TENSOR_PARALLEL_SIZE=1
    DATA_PARALLEL_SIZE="${GPU_NUM}"
fi

ROPE_SCALING_ARGS=""
if [ "${MODEL_NAME}" = "Qwen3-8B" ] || [ "${MODEL_NAME}" = "Qwen3-32B" ] || [ "${MODEL_NAME}" = "QwQ-32B" ] || [ "${MODEL_NAME}" = "SEA-E" ]; then
    ROPE_SCALING_ARGS='--rope-scaling {"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'
    echo "Detected Qwen3: Using rope-scaling with yarn"
fi

TRUST_REMOTE_CODE=""
if [ "${MODEL_NAME}" = "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16" ]; then
    TRUST_REMOTE_CODE="--trust-remote-code True"
fi

echo ""
echo "Starting vLLM OpenAI API server..."
echo "Server URL: http://localhost:${PORT}/v1"
echo "Tensor Parallel Size: ${TENSOR_PARALLEL_SIZE}"
echo "Data Parallel Size: ${DATA_PARALLEL_SIZE}"
echo ""

python -m vllm.entrypoints.openai.api_server \
  --host "${HOST}" --port "${PORT}" \
  --model "${MODEL}" \
  --served-model-name "${BASE_NAME}" \
  --dtype bfloat16 \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --data-parallel-size "${DATA_PARALLEL_SIZE}" \
  --max-model-len 98304 \
  --max-log-len 50 \
  ${TRUST_REMOTE_CODE} \
  ${ROPE_SCALING_ARGS}
