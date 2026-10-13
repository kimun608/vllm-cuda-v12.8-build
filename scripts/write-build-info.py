"""Record actual installed versions and source commits in the distributable image."""
import datetime
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess

packages = ("vllm", "torch", "torchvision", "torchaudio", "triton", "torchcodec")
info = {
    "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "packages": {name: metadata.version(name) for name in packages},
    "source_commits": {
        path.stem.removesuffix("-commit"): path.read_text().strip()
        for path in Path("/wheels").glob("*-commit.txt")
    },
    "cuda_base": os.environ["CUDA_BASE"],
    "cuda_toolkit": subprocess.check_output(["/usr/local/cuda/bin/nvcc", "--version"], text=True).strip(),
    "gpu_architectures": os.environ["TORCH_CUDA_ARCH_LIST"].split(";"),
    "required_extensions": os.environ["VLLM_REQUIRED_EXTENSIONS"].split(","),
    "source_patches": json.loads(Path("/wheels/source-patches.json").read_text(encoding="utf-8")),
    "cudnn_frontend": {
        "installed_version": metadata.version("nvidia-cudnn-frontend"),
        "wheel_provenance": json.loads(Path("/wheels/nvidia-cudnn-frontend-wheel.json").read_text(encoding="utf-8")),
    },
}
Path("/wheels/build-info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
