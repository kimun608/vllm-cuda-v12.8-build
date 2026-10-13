#!/usr/bin/env bash
set -euo pipefail
source /opt/source-utils.sh
cd /src/vllm
commit="$(git rev-parse HEAD)"
abi="$(cat /wheels/pytorch-commit.txt; uv pip freeze)"
key="$(compile_cache_key "$commit" /opt/build-vllm.sh "$abi")"
cache_dir="/build/vllm-cache/$VLLM_VERSION-$key"
for path in build .deps rust/target; do
    mkdir -p "$cache_dir/$path" "$(dirname "$path")"
    if [[ -e "$path" || -L "$path" ]]; then
        [[ "$(readlink -f "$path")" == "$cache_dir/$path" ]] || {
            echo "Unexpected existing build directory: /src/vllm/$path" >&2; exit 1;
        }
    else
        ln -s "$cache_dir/$path" "$path"
    fi
done
/opt/.venv/bin/python setup.py bdist_wheel --dist-dir=/wheels
