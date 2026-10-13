#!/usr/bin/env bash
# Shared release checkout and cache identities. Never reset or delete old caches.
set -euo pipefail

resolve_release_commit() {
    local repository="$1" version="$2" expected="${3:-}" refs commit
    [[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+[a-zA-Z0-9.]*$ ]] || {
        echo "Invalid release version: $version (use a version without v or +cu128)" >&2; return 1;
    }
    [[ -z "$expected" || "$expected" =~ ^[a-fA-F0-9]{40}$ ]] || {
        echo 'Source commit must be empty or a full 40-character SHA.' >&2; return 1;
    }
    refs="$(git ls-remote --exit-code --tags "$repository" "refs/tags/v$version" "refs/tags/v$version^{}")"
    commit="$(printf '%s\n' "$refs" | awk '$2 ~ /\^\{\}$/ {print $1; exit}')"
    if [[ -z "$commit" ]]; then commit="$(printf '%s\n' "$refs" | awk 'NR==1 {print $1}')"; fi
    [[ "$commit" =~ ^[a-f0-9]{40}$ ]] || { echo "Release tag v$version not found in $repository" >&2; return 1; }
    [[ -z "$expected" || "$commit" == "${expected,,}" ]] || {
        echo "Release v$version resolves to $commit, not requested commit $expected" >&2; return 1;
    }
    printf '%s\n' "$commit"
}

checkout_release() {
    local directory="$1" repository="$2" version="$3" commit="$4"
    if [[ ! -d "$directory/.git" ]]; then
        mkdir -p "$directory"
        git init --quiet "$directory"
        git -C "$directory" remote add origin "$repository"
    fi
    # A cancelled first fetch may leave .git without HEAD; retry that fetch safely.
    if ! git -C "$directory" rev-parse --verify HEAD >/dev/null 2>&1; then
        [[ "$(git -C "$directory" remote get-url origin)" == "$repository" ]] || {
            echo "Unexpected repository in incomplete cache: $directory" >&2; return 1;
        }
        git -C "$directory" fetch --depth 1 origin "refs/tags/v$version"
        git -C "$directory" checkout --quiet --detach FETCH_HEAD
    fi
    local actual
    actual="$(git -C "$directory" rev-parse HEAD)"
    [[ "$actual" == "$commit" ]] || {
        echo "Source/cache mismatch in $directory: expected $commit, found $actual" >&2; return 1;
    }
}

compile_cache_key() {
    local source_commit="$1" recipe="$2" abi="${3:-}"
    # Job counts affect scheduling only and deliberately do not invalidate objects.
    {
        printf '%s\n' "$source_commit" "$abi" "${CUDA_BASE:-}" "${TORCH_CUDA_ARCH_LIST:-}" "${CMAKE_BUILD_TYPE:-}" "${CMAKE_ARGS:-}"
        /usr/local/cuda/bin/nvcc --version
        gcc --version
        /opt/.venv/bin/python --version
        cmake --version
        dpkg-query -W 'libcudnn*' 'libcublas*' 'libcusparse*' 'libcusolver*' 'cuda-*' 2>/dev/null || true
        sha256sum /opt/source-utils.sh "$recipe"
    } | sha256sum | cut -c1-24
}
