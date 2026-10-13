#!/usr/bin/env bash
set -euo pipefail
source /opt/source-utils.sh
repository=https://github.com/vllm-project/vllm.git
commit="$(resolve_release_commit "$repository" "$VLLM_VERSION" "${VLLM_COMMIT:-}")"
checkout_release /src/vllm "$repository" "$VLLM_VERSION" "$commit"
mkdir -p /wheels
printf '%s\n' "$commit" > /wheels/vllm-commit.txt
# Incompatible new CUDA dependencies are rejected by dependency resolution.
sed -i 's/nvidia-cutlass-dsl\[cu13\]/nvidia-cutlass-dsl/;s/humming-kernels\[cu13\]/humming-kernels[cu12]/' /src/vllm/requirements/cuda.txt
