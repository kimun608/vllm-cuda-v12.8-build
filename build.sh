#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
project_dir="$PWD"
source "$project_dir/scripts/load-config.sh"
vllm_load_config "$project_dir"
preflight_only=0 skip_preflight=0 print_config=0
docker_extra=()
for arg in "$@"; do
  case "$arg" in
    --preflight-only) preflight_only=1 ;;
    --skip-preflight) skip_preflight=1 ;;
    --print-config) print_config=1 ;;
    --build-arg|--build-arg=*)
      echo '--build-arg overrides are not accepted here. Use versions.env, .env or shell variables so preflight and compilation use the same settings.' >&2
      exit 2 ;;
    *) docker_extra+=("$arg") ;;
  esac
done
if (( print_config )); then vllm_print_config; exit 0; fi
if (( preflight_only && skip_preflight )); then
  echo '--preflight-only and --skip-preflight cannot be combined' >&2; exit 2
fi
command -v docker >/dev/null || { echo 'Docker is required.' >&2; exit 1; }
docker_os="$(docker info --format '{{.OSType}}')"
[[ "$docker_os" == linux ]] || { echo 'Use a Linux Docker engine (Windows: Docker Desktop with WSL2).' >&2; exit 1; }
mkdir -p artifacts
export DOCKER_BUILDKIT=1
common_args=(build "${docker_extra[@]}" --platform linux/amd64 --progress plain)
for key in "${VLLM_BUILD_KEYS[@]}"; do common_args+=(--build-arg "$key=${VLLM_CONFIG[$key]}"); done
logged_build() {
  local log="$1"
  shift
  local -a result
  set +e
  docker "$@" 2>&1 | tee "$log"
  result=("${PIPESTATUS[@]}")
  set -e
  if (( result[0] != 0 )); then
    echo "Build failed (exit ${result[0]}). See $log." >&2; exit "${result[0]}"
  fi
  if (( result[1] != 0 )); then echo "Could not save $log." >&2; exit "${result[1]}"; fi
}
if (( ! skip_preflight )); then
  echo "Checking dependency compatibility for vLLM ${VLLM_CONFIG[VLLM_VERSION]}"
  logged_build artifacts/preflight.log "${common_args[@]}" --file Dockerfile.preflight \
    --tag "local/vllm-cu128-dependency-preflight:${VLLM_CONFIG[VLLM_VERSION]}" .
fi
if (( preflight_only )); then echo 'Dependency preflight passed. No source compilation was started.'; exit 0; fi
echo "Building ${VLLM_CONFIG[VLLM_IMAGE]} for CUDA architectures ${VLLM_CONFIG[TORCH_CUDA_ARCH_LIST]}"
logged_build artifacts/build.log "${common_args[@]}" --file Dockerfile --target vllm-openai --tag "${VLLM_CONFIG[VLLM_IMAGE]}" .
echo "Built ${VLLM_CONFIG[VLLM_IMAGE]}. Log: artifacts/build.log"
