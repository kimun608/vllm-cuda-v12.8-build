#!/usr/bin/env bash
set -Eeuo pipefail
# The official v0.31 CUDA 12.9 image also uses this CUDA 12.8 GDRCopy build.
gdrapi_deb=/tmp/vllm-runtime-libgdrapi.deb
trap 'rm -f "$gdrapi_deb"' EXIT
curl --fail --location --retry 5 \
    'https://developer.download.nvidia.com/compute/redist/gdrcopy/CUDA%2012.8/ubuntu24_04/x64/libgdrapi_2.5.1-1_amd64.Ubuntu24_04.deb' \
    --output "$gdrapi_deb"
apt-get install -y --no-install-recommends "$gdrapi_deb"
