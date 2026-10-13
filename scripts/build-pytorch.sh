#!/usr/bin/env bash
set -euo pipefail
source /opt/source-utils.sh
repository=https://github.com/pytorch/pytorch.git
commit="$(resolve_release_commit "$repository" "$TORCH_VERSION" "${TORCH_COMMIT:-}")"
key="$(compile_cache_key "$commit" /opt/build-pytorch.sh "triton=$TRITON_VERSION")"
source_dir="/build/pytorch/$TORCH_VERSION-$key"
checkout_release "$source_dir" "$repository" "$TORCH_VERSION" "$commit"
cd "$source_dir"
git submodule update --init --recursive --depth 1 --jobs 4
uv pip install -r requirements-build.txt filelock fsspec networkx sympy optree
export PYTORCH_BUILD_VERSION="${TORCH_VERSION}+cu128" PYTORCH_BUILD_NUMBER=1
export USE_CUDA=1 USE_CUDNN=1 USE_DISTRIBUTED=1 USE_NCCL=1
export USE_SYSTEM_NCCL=0 BUILD_TEST=0 USE_CUSPARSELT=0 USE_CUDSS=0
export USE_NVSHMEM=0 USE_CUFILE=0 USE_MSLK=0 USE_MPI=0
export CMAKE_BUILD_TYPE=Release BLAS=OpenBLAS
export CMAKE_C_COMPILER_LAUNCHER=ccache CMAKE_CXX_COMPILER_LAUNCHER=ccache
export CMAKE_CUDA_COMPILER_LAUNCHER=ccache
mkdir -p /wheels
/opt/.venv/bin/python setup.py bdist_wheel --dist-dir=/wheels
printf '%s\n' "$commit" > /wheels/pytorch-commit.txt
