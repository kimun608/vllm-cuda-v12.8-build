#!/usr/bin/env python3
"""Check the CUDA 12.8 cooperative TopK kernel on H200/H100 (SM90).

Run inside the built image on the H200 server:
    python /tests/cooperative-topk-runtime.py --require-supported --output /results/topk.json

SM12x is skipped: vLLM 0.31.0 excludes that family from this cluster kernel.
Use --metadata-only to inspect the cases without importing Torch or using a GPU.
"""

import argparse
import datetime as dt
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys


WORKSPACE_BYTES = 1024 * 1024
REPEATS = 2
UPSTREAM_VERSION = "0.31.0"
UPSTREAM_COMMIT = "db9527a46873454610df6dbedf79a36d6bf1a7f6"
OPERATOR = "torch.ops._C.cooperative_topk"


def cases():
    for k in (512, 1024, 2048):
        yield dict(
            name=f"mixed_lengths_k{k}", k=k, stride=65536, distribution="unique",
            lengths=[-3, 0, 1, k - 1, k, k + 1, 16383, 16385],
        )
        yield dict(
            name=f"fused_cluster4_k{k}", k=k, stride=65536, distribution="unique",
            lengths=[16385, 20001, 32767, 40003, 49151, 65535] * 2,
        )
        yield dict(
            name=f"streaming_cluster4_k{k}", k=k, stride=262144,
            distribution="unique",
            lengths=[65537, 131073, 196609, 262141] * 3,
        )
        yield dict(
            name=f"fused_cluster2_k{k}", k=k, stride=65536, distribution="unique",
            lengths=[16385, 20001, 32767, 40003, 65535] * 8,
        )
    yield dict(
        name="tie_workspace_is_per_row_k512", k=512, stride=65536,
        distribution="ties", lengths=[40960, 65535],
    )


def describe_case(case):
    rows = len(case["lengths"])
    cluster = 8 if rows <= 8 else (4 if rows <= 33 else 2)
    per_block = (case["stride"] + cluster - 1) // cluster
    capacity = (2 if cluster in (8, 16) else 3) * 16384
    return dict(
        case, rows=rows, cluster_size=cluster,
        large_path="fused_tma" if per_block <= capacity else "streaming_tma",
        large_rows=sum(length > 16384 for length in case["lengths"]),
        required_workspace_bytes=rows * max(1024, case["k"]) * 8,
        repeats=REPEATS,
    )


def skip_reason(capability):
    major, minor = capability
    if major == 12:
        return "vLLM 0.31.0 explicitly excludes SM12x from cooperative TopK"
    if capability == (9, 0):
        return None
    return f"This test is scoped to H200/H100 SM90; SM{major}{minor} is not validated"


def validate_source(version, build_info):
    if version.split("+", 1)[0] != UPSTREAM_VERSION:
        raise RuntimeError(
            f"Only vLLM {UPSTREAM_VERSION} was reviewed; this test must be "
            "reviewed again before running another release"
        )
    commits = build_info.get("source_commits", {})
    if "vllm" in commits and commits["vllm"] != UPSTREAM_COMMIT:
        raise RuntimeError("Image vLLM source commit differs from the reviewed source")


def assert_result(torch, logits, lengths, indices, storage, k):
    ends = lengths.reshape(-1, 1).clamp_min(0)
    valid = (indices >= 0) & (indices < ends)
    if not bool(torch.all(valid | (indices == -1))):
        raise AssertionError("Output contains out-of-range or dirty-tail indices")
    if not torch.equal(valid.sum(1), ends.clamp_max(k).flatten().long()):
        raise AssertionError("Output has the wrong number of valid indices")
    ordered = indices.sort(1).values
    if bool(((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)).any()):
        raise AssertionError("Output repeats a valid index")
    if not bool(torch.all(storage[:32] == -123456)) or not bool(
        torch.all(storage[-32:] == -123456)
    ):
        raise AssertionError("Kernel modified an output guard")
    # The op does not promise sorted indices or a particular choice among ties.
    # Compare the exact selected score multiset against the independent Torch op.
    columns = torch.arange(logits.shape[1], device=logits.device)
    visible = logits.masked_fill(columns[None, :] >= ends, float("-inf"))
    expected = torch.topk(visible, k, dim=1).values
    selected = logits.gather(1, indices.clamp_min(0).long())
    selected.masked_fill_(~valid, float("-inf"))
    if not torch.equal(selected.sort(1, descending=True).values, expected):
        raise AssertionError("Selected scores differ from exact torch.topk results")
    # The trivial path promises valid indices first and -1 padding afterwards.
    for row, end in enumerate(lengths.flatten().tolist()):
        count = min(k, max(0, end))
        if count < k and not bool(torch.all(indices[row, count:] == -1)):
            raise AssertionError(f"Row {row} does not contain the required -1 padding")


def run_case(torch, op, case, report):
    rows, stride, k = len(case["lengths"]), case["stride"], case["k"]
    lengths = torch.tensor(case["lengths"], dtype=torch.int32, device="cuda")
    # Exercise both (rows,) and native speculative decode (requests,next_n).
    if rows == 12:
        lengths = lengths.reshape(3, 4)
    columns = torch.arange(stride, dtype=torch.int64, device="cuda")
    offsets = torch.arange(rows, dtype=torch.int64, device="cuda")[:, None]
    # Odd multiplier permutes these power-of-two widths; float32 represents
    # every resulting integer exactly, giving an unambiguous unique-score oracle.
    logits = ((columns[None, :] * 7919 + offsets * 104729) % stride).float()
    logits -= stride // 2
    if case["distribution"] == "ties":
        logits.fill_(2.0)
        logits[0].fill_(1.0)
        logits[0, :2048] = -10.0
    ends = lengths.reshape(-1, 1).clamp_min(0)
    logits.masked_fill_(columns[None, :] >= ends, 1e20)
    storage = torch.full((rows * k + 64,), -123456,
                         dtype=torch.int32, device="cuda")
    indices = storage[32:-32].reshape(rows, k)
    workspace = torch.full((WORKSPACE_BYTES,), 0xA5,
                           dtype=torch.uint8, device="cuda")
    for _ in range(REPEATS):
        storage.fill_(-123456)
        report["kernel_launch_attempts"] += 1
        op(logits, lengths, indices, workspace, k, max(case["lengths"]))
        torch.cuda.synchronize()
        report["kernel_launches"] += 1
        assert_result(torch, logits, lengths, indices, storage, k)
        if case["distribution"] == "ties" and not bool(torch.all(indices[0] >= 2048)):
            raise AssertionError("Tie workspace overlap selected row-0 low-score entries")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="also save the JSON evidence")
    parser.add_argument("--metadata-only", action="store_true",
                        help="print planned cases; no Torch import or GPU execution")
    parser.add_argument("--require-supported", action="store_true",
                        help="return nonzero when the device is outside SM90")
    args = parser.parse_args()
    report = dict(
        status="failed", passed=False, kernel_executed=False, kernel_launches=0,
        kernel_launch_attempts=0,
        launch_count_definition="kernel_launches counts CUDA-synchronized launches",
        started_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        operator=OPERATOR, reviewed_upstream_version=UPSTREAM_VERSION,
        reviewed_upstream_commit=UPSTREAM_COMMIT, workspace_bytes=WORKSPACE_BYTES,
        workspace_derivation="rows * max(1024, k) * sizeof(Tie=8); rows<=64, k<=2048",
        required_compute_capability=[9, 0], require_supported=args.require_supported,
        source_references=[
            f"https://github.com/vllm-project/vllm/blob/{UPSTREAM_COMMIT}/csrc/libtorch_stable/cooperative_topk.cu",
            f"https://github.com/vllm-project/vllm/blob/{UPSTREAM_COMMIT}/csrc/libtorch_stable/cooperative_topk.cuh",
            f"https://github.com/vllm-project/vllm/blob/{UPSTREAM_COMMIT}/csrc/libtorch_stable/topk_histogram_4096.cuh",
            f"https://github.com/vllm-project/vllm/blob/{UPSTREAM_COMMIT}/tests/kernels/test_top_k_per_row.py",
        ],
        planned_cases=[describe_case(case) for case in cases()], cases=[],
        checks=["exact Torch score oracle", "unique valid indices", "dirty tails",
                "short/negative-length padding", "output guards", "reused workspace",
                "TMA fused/streaming paths", "cluster sizes 2/4/8", "row tie isolation"],
    )
    exit_code = 1
    try:
        if args.metadata_only:
            report["status"] = "metadata_only"
            report["reason"] = "Case metadata only; the CUDA kernel was not executed"
            exit_code = 0
        else:
            import torch
            report["torch_version"] = torch.__version__
            report["cuda"] = torch.version.cuda
            if not torch.cuda.is_available():
                raise RuntimeError("No CUDA GPU is available; kernel validation did not run")
            capability = torch.cuda.get_device_capability()
            report["gpu"] = torch.cuda.get_device_name()
            report["compute_capability"] = list(capability)
            build_info_path = Path("/opt/build-info.json")
            build_info = (json.loads(build_info_path.read_text(encoding="utf-8"))
                          if build_info_path.is_file() else {})
            report["image_gpu_architectures"] = build_info.get("gpu_architectures", [])
            report["image_source_commits"] = build_info.get("source_commits", {})
            reason = skip_reason(capability)
            if reason:
                report["status"], report["reason"] = "skipped", reason
                exit_code = 2 if args.require_supported else 0
            else:
                if torch.version.cuda != "12.8":
                    raise RuntimeError("This regression check requires a CUDA 12.8 Torch build")
                report["vllm_version"] = importlib.metadata.version("vllm")
                validate_source(report["vllm_version"], build_info)
                importlib.import_module("vllm._C_stable_libtorch")
                op = torch.ops._C.cooperative_topk
                report["operator_schema"] = str(op.default._schema)
                with torch.inference_mode():
                    for case in cases():
                        result = describe_case(case)
                        result["status"] = "running"
                        report["cases"].append(result)
                        run_case(torch, op, case, report)
                        result["status"] = "passed"
                report["status"], report["passed"] = "passed", True
                exit_code = 0
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        if report["cases"]:
            report["cases"][-1]["status"] = "failed"
    report["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    report["kernel_executed"] = report["kernel_launches"] > 0
    report["planned_case_count"] = len(report["planned_cases"])
    report["passed_case_count"] = sum(case["status"] == "passed" for case in report["cases"])
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
        except OSError as error:
            print(f"Cannot save evidence: {error}", file=sys.stderr)
            return 1
    print(text, end="")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
