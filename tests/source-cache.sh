#!/usr/bin/env bash
# Run in the existing local PyTorch image; no compilation, GPU or network required.
set -euo pipefail
cp /test-scripts/source-utils.sh /test-scripts/build-pytorch.sh /opt/
source /opt/source-utils.sh
repository=/tmp/release-fixture
git init --quiet "$repository"
git -C "$repository" -c user.name=Test -c user.email=test@localhost commit --quiet --allow-empty -m first
git -C "$repository" tag v1.2.3
first="$(git -C "$repository" rev-parse HEAD)"
git -C "$repository" -c user.name=Test -c user.email=test@localhost commit --quiet --allow-empty -m second
git -C "$repository" -c user.name=Test -c user.email=test@localhost tag -a v1.2.4 -m release
second="$(git -C "$repository" rev-parse HEAD)"
[[ "$(resolve_release_commit "$repository" 1.2.3)" == "$first" ]]
[[ "$(resolve_release_commit "$repository" 1.2.4)" == "$second" ]]
if resolve_release_commit "$repository" 1.2.4 "$first"; then exit 1; fi
checkout_release /tmp/checkout-old "$repository" 1.2.3 "$first"
if checkout_release /tmp/checkout-old "$repository" 1.2.4 "$second"; then exit 1; fi
# Interrupted fetches can be retried without deleting the checkout directory.
git init --quiet /tmp/checkout-new
git -C /tmp/checkout-new remote add origin "$repository"
checkout_release /tmp/checkout-new "$repository" 1.2.4 "$second"
[[ "$(git -C /tmp/checkout-new rev-parse HEAD)" == "$second" ]]
export TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=2
original="$(compile_cache_key "$first" /opt/build-pytorch.sh test-abi)"
export MAX_JOBS=4
[[ "$(compile_cache_key "$first" /opt/build-pytorch.sh test-abi)" == "$original" ]]
export TORCH_CUDA_ARCH_LIST=12.0
[[ "$(compile_cache_key "$first" /opt/build-pytorch.sh test-abi)" != "$original" ]]
export TORCH_CUDA_ARCH_LIST=9.0
[[ "$(compile_cache_key "$second" /opt/build-pytorch.sh test-abi)" != "$original" ]]
[[ "$(compile_cache_key "$first" /opt/build-pytorch.sh changed-abi)" != "$original" ]]
printf 'PASS: release identity, SHA mismatch, partial checkout recovery, version/ABI/architecture isolation, job-count reuse\n'
