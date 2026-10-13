#!/usr/bin/env bash
set -euo pipefail
source /opt/source-utils.sh
repository=https://github.com/pytorch/vision.git
commit="$(resolve_release_commit "$repository" "$TORCHVISION_VERSION" "${TORCHVISION_COMMIT:-}")"
checkout_release /build/vision "$repository" "$TORCHVISION_VERSION" "$commit"
cd /build/vision
BUILD_VERSION="${TORCHVISION_VERSION}+cu128" PYTORCH_VERSION="${TORCH_VERSION}+cu128" FORCE_CUDA=1 \
    /opt/.venv/bin/python setup.py bdist_wheel --dist-dir=/wheels
printf '%s\n' "$commit" > /wheels/torchvision-commit.txt
repository=https://github.com/pytorch/audio.git
commit="$(resolve_release_commit "$repository" "$TORCHAUDIO_VERSION" "${TORCHAUDIO_COMMIT:-}")"
checkout_release /build/audio "$repository" "$TORCHAUDIO_VERSION" "$commit"
cd /build/audio
git submodule update --init --recursive --depth 1 --jobs 4
BUILD_VERSION="${TORCHAUDIO_VERSION}+cu128" USE_CUDA=1 BUILD_SOX=0 \
    /opt/.venv/bin/python setup.py bdist_wheel --dist-dir=/wheels
printf '%s\n' "$commit" > /wheels/torchaudio-commit.txt
