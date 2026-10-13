#!/usr/bin/env bash
# Source this helper, never the .env data files. Requires Bash 4 or newer.
VLLM_BUILD_KEYS=(VLLM_VERSION TORCH_VERSION TORCHVISION_VERSION TORCHAUDIO_VERSION
  TRITON_VERSION TORCHCODEC_VERSION RUST_VERSION CUDA_BASE
  VLLM_COMMIT TORCH_COMMIT TORCHVISION_COMMIT TORCHAUDIO_COMMIT
  VLLM_REQUIRED_EXTENSIONS TORCH_CUDA_ARCH_LIST MAX_JOBS TORCH_MAX_JOBS VLLM_MAX_JOBS NVCC_THREADS)
VLLM_CONFIG_KEYS=("${VLLM_BUILD_KEYS[@]}" VLLM_IMAGE_REPOSITORY VLLM_IMAGE VLLM_MODEL
  VLLM_PORT VLLM_CONTAINER_NAME VLLM_CACHE_DIR VLLM_MODELS_DIR HF_TOKEN)
declare -A VLLM_CONFIG

vllm_known_key() {
  local candidate
  for candidate in "${VLLM_CONFIG_KEYS[@]}"; do
    [[ "$candidate" == "$1" ]] && return 0
  done
  return 1
}

vllm_read_config_file() {
  local file="$1" line key value number=0
  local -A seen_keys=()
  while IFS= read -r line || [[ -n "$line" ]]; do
    number=$((number + 1))
    [[ $number == 1 ]] && line="${line#$'\xef\xbb\xbf'}"
    line="${line%$'\r'}"
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "$line" || "$line" == \#* ]] && continue
    if [[ ! "$line" =~ ^([A-Z][A-Z0-9_]*)[[:space:]]*=(.*)$ ]]; then
      echo "$file:$number: expected KEY=VALUE" >&2; return 2
    fi
    key="${BASH_REMATCH[1]}"
    value="${BASH_REMATCH[2]}"
    if ! vllm_known_key "$key"; then
      echo "$file:$number: unknown setting $key" >&2; return 2
    fi
    if [[ -n "${seen_keys[$key]:-}" ]]; then
      echo "$file:$number: duplicate setting $key" >&2; return 2
    fi
    seen_keys["$key"]=1
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    if [[ "$value" == \"* || "$value" == \'* ]]; then
      if [[ ${#value} -lt 2 || "${value: -1}" != "${value:0:1}" ]]; then
        echo "$file:$number: unmatched quote" >&2; return 2
      fi
      value="${value:1:${#value}-2}"
    fi
    VLLM_CONFIG["$key"]="$value"
  done < "$file"
}

vllm_load_config() {
  local root="$1" key value limit
  for key in "${VLLM_CONFIG_KEYS[@]}"; do VLLM_CONFIG["$key"]=''; done
  VLLM_CONFIG[VLLM_PORT]=8000
  VLLM_CONFIG[VLLM_CONTAINER_NAME]=vllm-cu128
  VLLM_CONFIG[VLLM_CACHE_DIR]="$root/.cache/huggingface"
  vllm_read_config_file "$root/versions.env" || return
  if [[ -f "$root/.env" ]]; then vllm_read_config_file "$root/.env" || return; fi
  for key in "${VLLM_CONFIG_KEYS[@]}"; do
    if [[ -v "$key" ]]; then VLLM_CONFIG["$key"]="${!key}"; fi
  done
  for key in VLLM_VERSION TORCH_VERSION TORCHVISION_VERSION TORCHAUDIO_VERSION TRITON_VERSION TORCHCODEC_VERSION; do
    if [[ ! "${VLLM_CONFIG[$key]}" =~ ^[0-9]+\.[0-9]+\.[0-9]+((a|b|rc)[0-9]+|\.post[0-9]+)?$ ]]; then
      echo "Invalid $key release version: ${VLLM_CONFIG[$key]}" >&2; return 2
    fi
  done
  if [[ ! "${VLLM_CONFIG[RUST_VERSION]}" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ ]]; then
    echo 'RUST_VERSION must be a numeric release such as 1.95 or 1.95.0' >&2; return 2
  fi
  for key in VLLM_COMMIT TORCH_COMMIT TORCHVISION_COMMIT TORCHAUDIO_COMMIT; do
    value="${VLLM_CONFIG[$key]}"
    if [[ -n "$value" && ! "$value" =~ ^[0-9a-fA-F]{40}$ ]]; then
      echo "$key must be empty or a full 40-character commit SHA" >&2; return 2
    fi
  done
  for key in MAX_JOBS TORCH_MAX_JOBS VLLM_MAX_JOBS NVCC_THREADS VLLM_PORT; do
    value="${VLLM_CONFIG[$key]}"
    limit=1024
    [[ "$key" == VLLM_PORT ]] && limit=65535
    if [[ ! "$value" =~ ^[1-9][0-9]*$ || ${#value} -gt 5 ]] || (( value > limit )); then
      echo "$key must be an integer between 1 and $limit" >&2; return 2
    fi
  done
  if [[ ! "${VLLM_CONFIG[TORCH_CUDA_ARCH_LIST]}" =~ ^[0-9]+\.[0-9]+[a-z]?(\+PTX)?(\;[0-9]+\.[0-9]+[a-z]?(\+PTX)?)*$ ]]; then
    echo 'TORCH_CUDA_ARCH_LIST must contain semicolon-separated CUDA architectures' >&2; return 2
  fi
  if [[ ! "${VLLM_CONFIG[VLLM_REQUIRED_EXTENSIONS]}" =~ ^[A-Za-z_][A-Za-z0-9_]*(,[A-Za-z_][A-Za-z0-9_]*)*$ ]]; then
    echo 'VLLM_REQUIRED_EXTENSIONS must contain comma-separated module names' >&2; return 2
  fi
  for key in CUDA_BASE VLLM_IMAGE_REPOSITORY VLLM_CACHE_DIR VLLM_CONTAINER_NAME; do
    if [[ -z "${VLLM_CONFIG[$key]}" ]]; then echo "$key must not be empty" >&2; return 2; fi
  done
  if [[ -z "${VLLM_CONFIG[VLLM_IMAGE]}" ]]; then
    VLLM_CONFIG[VLLM_IMAGE]="${VLLM_CONFIG[VLLM_IMAGE_REPOSITORY]}:v${VLLM_CONFIG[VLLM_VERSION]}-cu128"
  fi
}

vllm_print_config() {
  local key value
  for key in "${VLLM_CONFIG_KEYS[@]}"; do
    value="${VLLM_CONFIG[$key]}"
    if [[ "$key" == HF_TOKEN && -n "$value" ]]; then value='[redacted]'; fi
    printf '%s=%s\n' "$key" "$value"
  done
}
