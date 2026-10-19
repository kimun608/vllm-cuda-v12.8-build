"""CPU-only verification of assets restored from the official vLLM image.

Manifest: {"source_image": "...", "files": {"examples/foo": "<sha256>", ...}}.
A bare relative-path-to-SHA256 mapping is also accepted. Only examples/**,
benchmarks/** and collect_env.py belong in this manifest; CUDA version metadata
is deliberately excluded. Run inside the image, with no model or GPU required.
"""

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys


GEMMA4 = "examples/tool_chat_template_gemma4.jinja"


def check_files(workspace, manifest_path):
    document = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    expected = document.get("files", document)
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Manifest must contain a nonempty path-to-SHA256 mapping")
    for name, digest in expected.items():
        path = PurePosixPath(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in name
            or path.as_posix() != name
            or not (
                name == "collect_env.py"
                or (len(path.parts) > 1 and path.parts[0] in {"examples", "benchmarks"})
            )
        ):
            raise ValueError(f"Unexpected manifest path: {name!r}")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Invalid SHA256 for {name}")
    for required in (GEMMA4, "collect_env.py"):
        if required not in expected:
            raise ValueError(f"Manifest is missing required asset: {required}")
    if not any(name.startswith("benchmarks/") for name in expected):
        raise ValueError("Manifest is missing benchmarks assets")

    actual = {
        file.relative_to(workspace).as_posix()
        for directory in ("examples", "benchmarks")
        for file in (workspace / directory).rglob("*")
        if file.is_file()
    }
    if (workspace / "collect_env.py").is_file():
        actual.add("collect_env.py")
    modified = []
    for name in sorted(actual & expected.keys()):
        with (workspace / name).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != expected[name]:
            modified.append(name)
    missing = sorted(expected.keys() - actual)
    unexpected = sorted(actual - expected.keys())
    return {
        "passed": not (missing or modified or unexpected),
        "source_image": document.get("source_image") if "files" in document else None,
        "expected_count": len(expected),
        "matched_count": len(actual & expected.keys()) - len(modified),
        "missing": missing,
        "modified": modified,
        "unexpected": unexpected,
    }


def check_templates(workspace):
    from transformers.utils.chat_template_utils import _compile_jinja_template
    try:
        # vLLM 0.32 moved these helpers out of entrypoints.
        from vllm.renderers.chat_utils import load_chat_template, validate_chat_template
    except ModuleNotFoundError as exc:
        # Do not hide a missing dependency imported by chat_utils itself.
        if exc.name not in {"vllm.renderers", "vllm.renderers.chat_utils"}:
            raise
        from vllm.entrypoints.chat_utils import load_chat_template, validate_chat_template

    templates = sorted((workspace / "examples").glob("*.jinja"))
    if not templates or not (workspace / GEMMA4).is_file():
        raise FileNotFoundError(f"Required template is missing: {workspace / GEMMA4}")
    results = []
    # Match the official image's working directory, so the deployment's
    # --chat-template examples/... argument exercises vLLM's real path resolver.
    previous_directory = Path.cwd()
    try:
        os.chdir(workspace)
        for path in templates:
            name = path.relative_to(workspace).as_posix()
            result = {"path": name, "passed": False}
            try:
                validate_chat_template(name)
                validate_chat_template(str(path))
                relative = load_chat_template(name)
                absolute = load_chat_template(str(path))
                if not relative or relative != absolute:
                    raise ValueError("Relative and absolute template loads differ or are empty")
                if relative != path.read_text(encoding="utf-8"):
                    raise ValueError("vLLM did not load the restored workspace template")
                compiled = _compile_jinja_template(relative)
                rendered = compiled.render(
                    messages=[{"role": "user", "content": "Runtime asset smoke check."}],
                    bos_token="<bos>",
                    eos_token="<eos>",
                    add_generation_prompt=True,
                )
                if "Runtime asset smoke check." not in rendered:
                    raise ValueError("Rendered template did not preserve the user message")
                result.update(passed=True, rendered_characters=len(rendered))
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
            results.append(result)
    finally:
        os.chdir(previous_directory)
    return {
        "passed": all(result["passed"] for result in results),
        "count": len(results),
        "scope": "Root examples/*.jinja; relative/absolute validation, loading, compilation and text rendering",
        "templates": results,
    }


def check_rust_frontend():
    from vllm import envs

    # Enable auto discovery only within this verification process. Do not
    # change the image's configured/default Python frontend selection.
    overrides = {"VLLM_USE_RUST_FRONTEND": "1", "VLLM_RUST_FRONTEND_PATH": "auto"}
    original = {name: os.environ.get(name) for name in overrides}
    try:
        os.environ.update(overrides)
        resolved = envs._resolve_rust_cli_path()
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    if not resolved:
        raise FileNotFoundError("vLLM's Rust frontend resolver returned no binary")
    path = Path(resolved)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise FileNotFoundError(f"Rust frontend is missing or not executable: {path}")
    completed = subprocess.run(
        [str(path), "--version"], capture_output=True, text=True, timeout=30, check=True
    )
    return {"passed": True, "path": str(path), "version": completed.stdout.strip()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path("/vllm-workspace"))
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    report = {
        "passed": False,
        "workspace": str(workspace),
        "cpu_only": True,
        "gpu_inference_tested": False,
    }
    # Keep third-party import/runtime logs separate from the JSON result.
    with contextlib.redirect_stdout(sys.stderr):
        for name, check in (
            ("files", lambda: check_files(workspace, args.manifest.resolve())),
            ("chat_templates", lambda: check_templates(workspace)),
            ("rust_frontend", check_rust_frontend),
        ):
            try:
                report[name] = check()
            except Exception as exc:
                report[name] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
    report["passed"] = all(report[name]["passed"] for name in ("files", "chat_templates", "rust_frontend"))
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
