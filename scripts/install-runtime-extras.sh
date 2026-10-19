#!/usr/bin/env bash
set -Eeuo pipefail

# Freeze the validated CUDA 12.8 environment before resolving serving extras.
python - <<'PY'
from importlib import metadata
from pathlib import Path

# The build's direct_url metadata points to /wheels, which is not in runtime.
pins = sorted(f'{dist.metadata["Name"]}=={dist.version}' for dist in metadata.distributions())
Path('/opt/cu128-runtime-base.txt').write_text('\n'.join(pins) + '\n')
PY
uv pip install --constraint /opt/cu128-runtime-base.txt \
    --constraint /opt/runtime-extra-constraints.txt \
    --requirement /opt/runtime-requirements.txt
uv pip check
python - <<'PY'
from importlib import metadata
from pathlib import Path
import os

for line in Path('/opt/cu128-runtime-base.txt').read_text().splitlines():
    if '==' in line:
        name, expected = line.split('==', 1)
        assert metadata.version(name) == expected, (name, expected, metadata.version(name))

dist = metadata.distribution('vllm')
binary = Path(dist.locate_file('vllm/vllm-rs'))
assert binary.is_file() and os.access(binary, os.X_OK), binary
link = Path('/usr/local/bin/vllm-rs')
if link.is_symlink():
    assert link.resolve() == binary.resolve(), link
elif link.exists():
    raise RuntimeError(f'Unexpected existing binary: {link}')
else:
    link.symlink_to(binary)

versions = ''.join(f'{name}=={metadata.version(name)}\n' for name in ('torch', 'torchaudio', 'torchvision'))
Path('/vllm-workspace/torch_lib_versions.txt').write_text(versions)
PY
uv pip freeze > /opt/installed-packages.txt
python /opt/verify-stack.py
