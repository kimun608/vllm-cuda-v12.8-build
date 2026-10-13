"""Validate the CUDA build; --gpu also exercises real GPU kernels without a model."""
import argparse
import importlib
import importlib.metadata as metadata
import json
import os
import re
import subprocess

DEFAULT_VERSIONS = {
    "torch": "2.13.0", "vllm": "0.31.0", "torchvision": "0.28.0",
    "torchaudio": "2.11.0", "triton": "3.7.1", "torchcodec": "0.14.0",
}


def expected_version(package):
    version = os.environ.get(f"EXPECTED_{package.upper()}_VERSION", DEFAULT_VERSIONS[package])
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+|\.post\d+)?", version):
        raise ValueError(f"EXPECTED_{package.upper()}_VERSION must be a base release version: {version!r}")
    suffix = "+cpu" if package == "torchcodec" else "" if package == "triton" else "+cu128"
    return version + suffix


def check_version(package):
    actual = metadata.version(package)
    expected = expected_version(package)
    assert actual == expected, f"{package}: expected {expected}, installed {actual}"
    return actual


parser = argparse.ArgumentParser()
parser.add_argument("--torch-only", action="store_true")
parser.add_argument("--gpu", action="store_true")
args = parser.parse_args()

import torch

assert torch.__version__ == expected_version("torch"), torch.__version__
check_version("torch")
check_version("triton")
assert torch.version.cuda == "12.8", torch.version.cuda
assert torch.distributed.is_nccl_available(), "NCCL is needed for H200 tensor parallelism"
nvcc = subprocess.check_output(["/usr/local/cuda/bin/nvcc", "--version"], text=True)
assert "release 12.8," in nvcc, nvcc
for dist in metadata.distributions():
    name = dist.metadata["Name"].lower().replace("_", "-")
    assert not name.endswith("-cu13"), f"Unexpected CUDA 13 package: {name}"
    if name in {
        "nvidia-cuda-runtime-cu12", "nvidia-cuda-nvcc-cu12",
        "nvidia-cuda-nvrtc-cu12", "nvidia-cuda-cccl-cu12",
        "nvidia-cuda-runtime", "nvidia-cuda-nvcc", "nvidia-cuda-nvrtc",
        "nvidia-cuda-cccl", "nvidia-cuda-nvvm", "nvidia-cuda-cupti",
        "cuda-python", "cuda-bindings",
        "cuda-toolkit",
    }:
        assert dist.version.startswith("12.8."), (name, dist.version)
report = {"torch": torch.__version__, "triton": metadata.version("triton"), "cuda": torch.version.cuda}
if not args.torch_only:
    report["cudnn_frontend"] = metadata.version("nvidia-cudnn-frontend")
    assert report["cudnn_frontend"].endswith("+cu128"), "Expected the reviewed CUDA 12 frontend wheel adaptation"
    importlib.import_module("cudnn")
    for name in ("vllm", "torchvision", "torchaudio", "torchcodec"):
        report[name] = check_version(name)
    for name in ("vllm", "torchvision", "torchaudio"):
        importlib.import_module(name)
    # GPU extensions link libcuda.so.1, supplied by the host at docker run time.
    # Check their presence during a GPU-free build; load them in the GPU test.
    wheel_files = [str(path) for path in metadata.files("vllm") or []]
    extension_setting = os.environ.get("VLLM_REQUIRED_EXTENSIONS", "_C_stable_libtorch,_moe_C_stable_libtorch")
    extensions = [name.strip() for name in extension_setting.split(",")]
    assert all(re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", name) for name in extensions), "VLLM_REQUIRED_EXTENSIONS must contain comma-separated module names"
    for name in extensions:
        assert any(
            path.startswith(f"vllm/{name.replace('.', '/')}.") and path.endswith(".so")
            for path in wheel_files
        ), f"Missing compiled extension: {name}"
        if args.gpu:
            importlib.import_module(f"vllm.{name}")
    report["required_extensions"] = extensions
if args.gpu:
    assert torch.cuda.is_available(), "GPU unavailable; check --gpus all and NVIDIA driver"
    capability = torch.cuda.get_device_capability()
    arches = torch.cuda.get_arch_list()
    assert f"sm_{capability[0]}{capability[1]}" in arches, (capability, arches)
    x = torch.ones((128, 128), device="cuda")
    torch.testing.assert_close(x @ x, torch.full_like(x, 128))
    def pointwise(value):
        return value.sin() + value

    compiled = torch.compile(pointwise, fullgraph=True)
    torch.testing.assert_close(compiled(x), pointwise(x))
    if not args.torch_only:
        y = torch.randn(8, 128, dtype=torch.float16, device="cuda")
        out = torch.empty((8, 64), dtype=y.dtype, device="cuda")
        torch.ops._C.silu_and_mul(out, y)
        torch.testing.assert_close(out, torch.nn.functional.silu(y[:, :64]) * y[:, 64:], atol=2e-3, rtol=2e-3)
    torch.cuda.synchronize()
    report.update(gpu=torch.cuda.get_device_name(), capability=capability, arches=arches)
print(json.dumps(report, indent=2))
