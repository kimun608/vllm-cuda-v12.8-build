# syntax=docker/dockerfile:1.7
# Wrappers/Compose pass versions.env values. Defaults keep direct docker build usable.
ARG CUDA_BASE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
ARG VLLM_VERSION=0.32.0
ARG TORCH_VERSION=2.13.0
ARG TORCHVISION_VERSION=0.28.0
ARG TORCHAUDIO_VERSION=2.11.0
ARG TRITON_VERSION=3.7.1
ARG TORCHCODEC_VERSION=0.14.0
ARG RUST_VERSION=1.95
ARG VLLM_REQUIRED_EXTENSIONS=_C_stable_libtorch,_moe_C_stable_libtorch

FROM ${CUDA_BASE} AS base
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG CUDA_BASE
ARG TRITON_VERSION
ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/.venv \
    PATH=/opt/.venv/bin:/root/.cargo/bin:/usr/local/cuda/bin:${PATH} \
    CUDA_HOME=/usr/local/cuda CUDA_BASE=${CUDA_BASE} \
    EXPECTED_TRITON_VERSION=${TRITON_VERSION} \
    UV_LINK_MODE=copy UV_HTTP_TIMEOUT=500 \
    UV_CONSTRAINT=/opt/cu128-constraints.txt \
    PIP_CONSTRAINT=/opt/cu128-constraints.txt \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas
RUN nvcc --version | grep -q 'release 12.8,' \
    && sed -i 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g; s|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' /etc/apt/sources.list.d/ubuntu.sources \
    && apt-get -o Acquire::Retries=5 update && apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
        build-essential git curl ca-certificates python3 python3-dev \
        libopenblas-dev libomp-dev libnuma-dev libssl-dev libffi-dev \
        libjpeg-dev libpng-dev libsox-dev libsndfile1 ffmpeg \
        pkg-config patch ccache unzip zip ninja-build libdw-dev libelf-dev \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.10.12 /uv /usr/local/bin/uv
RUN uv venv --python /usr/bin/python3 /opt/.venv \
    && python -c "import sys; assert sys.version_info[:2] == (3, 12), 'This recipe requires Python 3.12 (Ubuntu 24.04)'"
# This file pins the CUDA 12.8 policy only; package versions are added per stage.
COPY scripts/cu128-constraints.txt /opt/cu128-constraints.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    printf '\ntriton==%s\n' "${TRITON_VERSION}" >> /opt/cu128-constraints.txt \
    && uv pip install 'setuptools>=77.0.3,<81' wheel 'cmake>=3.27,<4' \
        ninja packaging pyyaml requests six typing-extensions numpy \
        setuptools-scm setuptools-rust 'build[uv]' jinja2 regex psutil \
        "triton==${TRITON_VERSION}"
COPY scripts/source-utils.sh /opt/source-utils.sh

FROM base AS torch-build
ARG TORCH_VERSION
ARG TORCH_COMMIT=""
ARG TORCH_CUDA_ARCH_LIST="9.0;12.0"
ARG MAX_JOBS=2
ARG TORCH_MAX_JOBS=${MAX_JOBS}
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} MAX_JOBS=${TORCH_MAX_JOBS}
COPY scripts/build-pytorch.sh /opt/build-pytorch.sh
RUN --mount=type=cache,id=vllm-cu128-pytorch-src,target=/build/pytorch,sharing=locked \
    --mount=type=cache,id=vllm-cu128-ccache,target=/root/.cache/ccache,sharing=locked \
    --mount=type=cache,target=/root/.cache/uv \
    bash /opt/build-pytorch.sh

FROM base AS torch-installed
ARG TORCH_VERSION
ENV EXPECTED_TORCH_VERSION=${TORCH_VERSION}
COPY --from=torch-build /wheels /wheels
RUN --mount=type=cache,target=/root/.cache/uv \
    printf '\ntorch==%s+cu128\n' "${TORCH_VERSION}" >> /opt/cu128-constraints.txt \
    && uv pip install /wheels/torch-*.whl \
    && python -c "import os, torch; assert torch.__version__ == os.environ['EXPECTED_TORCH_VERSION'] + '+cu128'; assert torch.version.cuda == '12.8'; assert torch.version.git_version == open('/wheels/pytorch-commit.txt').read().strip(); assert torch.distributed.is_nccl_available()"

FROM torch-installed AS vision-audio-build
ARG TORCHVISION_VERSION
ARG TORCHAUDIO_VERSION
ARG TORCHVISION_COMMIT=""
ARG TORCHAUDIO_COMMIT=""
ARG TORCH_CUDA_ARCH_LIST="9.0;12.0"
ARG MAX_JOBS=2
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} MAX_JOBS=${MAX_JOBS}
COPY scripts/build-media.sh /opt/build-media.sh
RUN --mount=type=cache,target=/root/.cache/uv \
    bash /opt/build-media.sh

FROM torch-installed AS vllm-build
ARG VLLM_VERSION
ARG VLLM_COMMIT=""
ARG TORCHVISION_VERSION
ARG TORCHAUDIO_VERSION
ARG TORCHCODEC_VERSION
ARG RUST_VERSION
ARG VLLM_REQUIRED_EXTENSIONS
ARG TORCH_CUDA_ARCH_LIST="9.0;12.0"
ARG MAX_JOBS=2
ARG VLLM_MAX_JOBS=${MAX_JOBS}
ARG NVCC_THREADS=1
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=${VLLM_MAX_JOBS} NVCC_THREADS=${NVCC_THREADS} \
    CARGO_BUILD_JOBS=${VLLM_MAX_JOBS} CMAKE_BUILD_TYPE=Release VLLM_TARGET_DEVICE=cuda \
    VLLM_USE_PRECOMPILED=0 VLLM_VERSION_OVERRIDE=${VLLM_VERSION}+cu128 \
    EXPECTED_VLLM_VERSION=${VLLM_VERSION} \
    EXPECTED_TORCHVISION_VERSION=${TORCHVISION_VERSION} \
    EXPECTED_TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION} \
    EXPECTED_TORCHCODEC_VERSION=${TORCHCODEC_VERSION} \
    VLLM_REQUIRED_EXTENSIONS=${VLLM_REQUIRED_EXTENSIONS}
COPY --from=vision-audio-build /wheels /wheels
RUN --mount=type=cache,target=/root/.cache/uv \
    printf '\ntorchvision==%s+cu128\ntorchaudio==%s+cu128\ntorchcodec==%s+cpu\n' \
        "${TORCHVISION_VERSION}" "${TORCHAUDIO_VERSION}" "${TORCHCODEC_VERSION}" >> /opt/cu128-constraints.txt \
    && uv pip install /wheels/torchvision-*.whl /wheels/torchaudio-*.whl
COPY scripts/prepare-vllm.sh /opt/prepare-vllm.sh
COPY scripts/prepare-cudnn-frontend.py /opt/prepare-cudnn-frontend.py
RUN bash /opt/prepare-vllm.sh
WORKDIR /src/vllm
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-deps --index-url https://download.pytorch.org/whl/cpu "torchcodec==${TORCHCODEC_VERSION}+cpu" \
    && python /opt/prepare-cudnn-frontend.py --constraints /opt/cu128-constraints.txt --output-dir /wheels \
    && uv pip install /wheels/nvidia_cudnn_frontend-*.whl -r requirements/build/cuda.txt -r requirements/cuda.txt
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain "${RUST_VERSION}"
# Ninja tracks the changed header; unchanged CUDA objects remain reusable.
COPY scripts/apply-cu128-patches.py scripts/cu128-ptx-compat.cuh /opt/
RUN python /opt/apply-cu128-patches.py /src/vllm --report /wheels/source-patches.json
COPY scripts/build-vllm.sh /opt/build-vllm.sh
RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    --mount=type=cache,id=vllm-cu128-ccache,target=/root/.cache/ccache,sharing=locked \
    --mount=type=cache,id=vllm-cu128-build,target=/build/vllm-cache,sharing=locked \
    --mount=type=cache,target=/root/.cache/uv \
    bash /opt/build-vllm.sh
COPY scripts/verify-stack.py scripts/write-build-info.py /opt/
RUN uv pip install --no-deps /wheels/vllm-*.whl \
    && cd /tmp && python /opt/verify-stack.py \
    && uv pip check \
    && uv pip freeze > /wheels/installed-packages.txt \
    && python /opt/write-build-info.py

FROM base AS vllm-openai
ARG VLLM_VERSION
ARG TORCH_VERSION
ARG TORCHVISION_VERSION
ARG TORCHAUDIO_VERSION
ARG TORCHCODEC_VERSION
ARG VLLM_REQUIRED_EXTENSIONS
ARG TORCH_CUDA_ARCH_LIST="9.0;12.0"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    EXPECTED_VLLM_VERSION=${VLLM_VERSION} EXPECTED_TORCH_VERSION=${TORCH_VERSION} \
    EXPECTED_TORCHVISION_VERSION=${TORCHVISION_VERSION} \
    EXPECTED_TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION} \
    EXPECTED_TORCHCODEC_VERSION=${TORCHCODEC_VERSION} \
    VLLM_REQUIRED_EXTENSIONS=${VLLM_REQUIRED_EXTENSIONS} \
    VLLM_USAGE_SOURCE=production-docker-image \
    VLLM_NO_USAGE_STATS=1 HF_HOME=/root/.cache/huggingface \
    VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLREDUCE_USE_FLASHINFER=0 \
    VLLM_USE_DEEP_GEMM=0 VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0
# COPY merges directories; remove the base venv to discard upgraded-package metadata.
RUN rm -rf /opt/.venv
COPY --from=vllm-build /opt/.venv /opt/.venv
COPY --from=vllm-build /opt/cu128-constraints.txt /opt/cu128-constraints.txt
COPY --from=vllm-build /wheels/installed-packages.txt /wheels/build-info.json /opt/
COPY --from=vllm-build /wheels/source-patches.json /opt/source-patches.json
COPY --from=vllm-build /wheels/nvidia-cudnn-frontend-wheel.json /opt/nvidia-cudnn-frontend-wheel.json
COPY scripts/verify-stack.py /opt/verify-stack.py
WORKDIR /vllm-workspace
# These repository assets are not included in the vLLM wheel.
COPY --from=vllm-build /src/vllm/examples ./examples
COPY --from=vllm-build /src/vllm/benchmarks ./benchmarks
COPY --from=vllm-build /src/vllm/vllm/collect_env.py ./collect_env.py
COPY --from=vllm-build /src/vllm/docker/entrypoints/vllm-nonroot-entrypoint.sh /usr/local/bin/vllm-nonroot-entrypoint.sh
COPY scripts/runtime-requirements.txt scripts/runtime-extra-constraints.txt scripts/install-runtime-extras.sh /opt/
RUN apt-get -o Acquire::Retries=5 update \
    && apt-get -o Acquire::Retries=5 install -y --no-install-recommends libsm6 libxext6 libgl1 libibverbs-dev numactl \
    && rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,target=/root/.cache/uv bash /opt/install-runtime-extras.sh
COPY scripts/install-runtime-gdrcopy.sh /opt/install-runtime-gdrcopy.sh
RUN bash /opt/install-runtime-gdrcopy.sh
COPY scripts/verify-runtime-assets.py /opt/verify-runtime-assets.py
# Runtime-only opt-in. Keep native Torch/vLLM build stages independent of this flag.
ARG VLLM_CUDA128_GEMM_AUTOTUNE=0
COPY scripts/patch-small-gemm-autotune-v3.py /opt/patch-small-gemm-autotune-v3.py
RUN case "${VLLM_CUDA128_GEMM_AUTOTUNE}" in 0|1) ;; *) echo 'VLLM_CUDA128_GEMM_AUTOTUNE must be 0 or 1' >&2; exit 2 ;; esac \
    && printf '%s  %s\n' 'e8e62249d17b370db42b088caefd2677b78b18247c1dbdf2db97f90ec6d34e2c' /opt/patch-small-gemm-autotune-v3.py | sha256sum --check --status \
    && if [ "${VLLM_CUDA128_GEMM_AUTOTUNE}" = 1 ]; then \
        python3 -B -E /opt/patch-small-gemm-autotune-v3.py --mode apply \
            --compilation-source /opt/.venv/lib/python3.12/site-packages/vllm/config/compilation.py \
            --compiler-source /opt/.venv/lib/python3.12/site-packages/vllm/compilation/compiler_interface.py \
            --metadata /opt/small-gemm-autotune-v3-patch.json; \
    else \
        python3 -B -E -c 'import hashlib,json,pathlib; h=pathlib.Path("/opt/patch-small-gemm-autotune-v3.py"); pathlib.Path("/opt/small-gemm-autotune-v3-disabled.json").write_text(json.dumps({"patch_id":"lottecard-small-capture-gemm-v3","status":"disabled","build_flag":"VLLM_CUDA128_GEMM_AUTOTUNE","build_flag_value":"0","source_modified":False,"helper_sha256":hashlib.sha256(h.read_bytes()).hexdigest()},indent=2)+"\n",encoding="utf-8")'; \
    fi
ENV LOTTECARD_GEMM_AUTOTUNE_SIZES=${VLLM_CUDA128_GEMM_AUTOTUNE}
ENV VLLM_ENABLE_CUDA_COMPATIBILITY=0
LABEL org.opencontainers.image.title="vLLM ${VLLM_VERSION} custom CUDA 12.8 build" \
      org.opencontainers.image.version="${VLLM_VERSION}+cu128" \
      org.opencontainers.image.ref.name="v${VLLM_VERSION}" \
      org.opencontainers.image.source="https://github.com/vllm-project/vllm"
EXPOSE 8000
ENTRYPOINT ["vllm", "serve"]
