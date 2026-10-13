"""Resolve CUDA 12.8 release dependencies without installing or compiling packages.

Temporary Torch-family wheels contain published metadata, never executable code.
Successful resolution does not certify source, ABI, CUDA, GPU or model support.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import runpy
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile

DEFAULTS = {
    "VLLM_VERSION": "0.31.0", "TORCH_VERSION": "2.13.0",
    "TORCHVISION_VERSION": "0.28.0", "TORCHAUDIO_VERSION": "2.11.0",
    "TRITON_VERSION": "3.7.1", "TORCHCODEC_VERSION": "0.14.0",
    "RUST_VERSION": "1.95", "CUDA_BASE": "nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04",
    "VLLM_COMMIT": "", "TORCH_COMMIT": "", "TORCHVISION_COMMIT": "", "TORCHAUDIO_COMMIT": "",
    "VLLM_REQUIRED_EXTENSIONS": "_C_stable_libtorch,_moe_C_stable_libtorch",
}
REPOSITORIES = {
    "VLLM": "vllm-project/vllm", "TORCH": "pytorch/pytorch",
    "TORCHVISION": "pytorch/vision", "TORCHAUDIO": "pytorch/audio",
}
SOURCE_PACKAGES = {"torch": "TORCH", "torchvision": "TORCHVISION", "torchaudio": "TORCHAUDIO"}
POLICY_PACKAGES = {*SOURCE_PACKAGES, "triton", "torchcodec"}
CUDA_128_PACKAGES = {
    "cuda-python", "cuda-bindings", "nvidia-cuda-runtime-cu12", "nvidia-cuda-nvcc-cu12",
    "nvidia-cuda-nvrtc-cu12", "nvidia-cuda-cccl-cu12", "cuda-toolkit",
    "nvidia-cuda-runtime", "nvidia-cuda-nvcc", "nvidia-cuda-nvrtc",
    "nvidia-cuda-cccl", "nvidia-cuda-nvvm", "nvidia-cuda-cupti",
}
INCLUDE = re.compile(r"\s*(?:-r\s*|--requirement(?:\s+|=)|-c\s*|--constraint(?:\s+|=))([^\s#]+)\s*(?:#.*)?")
RELEASE_VERSION = re.compile(r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+|\.post\d+)?\Z")


def read_config(path, environ=None):
    """Parse literal KEY=VALUE data without shell evaluation or substitutions."""
    values = dict(DEFAULTS)
    if path.is_file():
        seen = set()
        for number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            match = re.fullmatch(r"([A-Z][A-Z0-9_]*)\s*=\s*(.*)", line)
            if not match:
                raise ValueError(f"{path}:{number}: expected literal KEY=VALUE")
            key, value = match.groups()
            if key in seen:
                raise ValueError(f"{path}:{number}: duplicate key {key}")
            seen.add(key)
            if value.startswith(("'", '"')):
                if len(value) < 2 or value[-1] != value[0]:
                    raise ValueError(f"{path}:{number}: unmatched quote")
                value = value[1:-1]
            if key in values:
                values[key] = value
    environ = os.environ if environ is None else environ
    values.update({key: environ[key] for key in values if key in environ})
    for key, value in values.items():
        if key.endswith("_VERSION") and key != "RUST_VERSION" and not RELEASE_VERSION.fullmatch(value):
            raise ValueError(f"{key} must be a release version without v or local suffix: {value!r}")
        if key.endswith("_COMMIT") and value and not re.fullmatch(r"[0-9a-fA-F]{40}", value):
            raise ValueError(f"{key} must be empty or a full 40-character Git commit SHA")
    if not re.fullmatch(r"nvidia/cuda:12\.8\.\d+-cudnn-devel-ubuntu24\.04(?:@sha256:[0-9a-f]{64})?", values["CUDA_BASE"]):
        raise ValueError("This preflight supports CUDA 12.8 / Ubuntu 24.04 / Python 3.12 only")
    if not all(re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", item.strip()) for item in values["VLLM_REQUIRED_EXTENSIONS"].split(",")):
        raise ValueError("VLLM_REQUIRED_EXTENSIONS must contain comma-separated Python module names")
    return values


def fetch_text(url):
    request = urllib.request.Request(url, headers={"User-Agent": "vllm-cu128-dependency-preflight"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def resolve_release(prefix, config):
    tag = f"refs/tags/v{config[prefix + '_VERSION']}"
    repo = REPOSITORIES[prefix]
    output = subprocess.check_output(
        ["git", "ls-remote", "--tags", f"https://github.com/{repo}.git", tag, tag + "^{}"],
        text=True, encoding="utf-8", errors="replace", timeout=90,
    )
    refs = {ref: sha for sha, ref in (line.split() for line in output.splitlines())}
    commit = refs.get(tag + "^{}", refs.get(tag))
    if not commit:
        raise ValueError(f"Release tag does not exist: {repo} {tag}")
    expected = config[prefix + "_COMMIT"]
    if expected and commit != expected.lower():
        raise ValueError(f"{prefix}: tag {tag} resolves to {commit}, not configured commit {expected}")
    return {"repository": repo, "tag": tag, "commit": commit, "commit_was_pinned": bool(expected)}


def copy_requirements(destination, commit, source_root=None):
    """Follow relative -r/-c includes within the repository, preserving paths."""
    pending = ["requirements/cuda.txt", "requirements/build/cuda.txt"]
    seen = set()
    while pending:
        relative = posixpath.normpath(pending.pop())
        if relative in seen:
            continue
        if relative.startswith(("../", "/")) or ":" in relative or "\\" in relative:
            raise ValueError(f"Unsupported requirements path outside source tree: {relative}")
        if source_root is None:
            content = fetch_text(f"https://raw.githubusercontent.com/vllm-project/vllm/{commit}/{relative}")
        else:
            source = (source_root / relative).resolve()
            if not source.is_relative_to(source_root.resolve()):
                raise ValueError(f"Requirements path escapes source tree: {source}")
            content = source.read_text(encoding="utf-8")
        seen.add(relative)
        for line in content.splitlines():
            include = INCLUDE.fullmatch(line)
            if include:
                pending.append(posixpath.join(str(PurePosixPath(relative).parent), include.group(1)))
            elif re.match(r"\s*(?:-r|-c|--requirement\b|--constraint\b)", line):
                raise ValueError(f"Unsupported requirement include in {relative}: {line}")
        content = content.replace("nvidia-cutlass-dsl[cu13]", "nvidia-cutlass-dsl")
        content = content.replace("humming-kernels[cu13]", "humming-kernels[cu12]")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return sorted(seen)


def dependency_name(requirement):
    match = re.match(r"\s*([A-Za-z0-9_.-]+)", requirement)
    return re.sub(r"[-_.]+", "-", match.group(1)).lower() if match else ""


def published_metadata(name, version):
    url = f"https://pypi.org/pypi/{name}/{version}/json"
    info = json.loads(fetch_text(url))["info"]
    if info["version"] != version or dependency_name(info["name"]) != name:
        raise ValueError(f"Unexpected release metadata from {url}")
    published = info.get("requires_dist") or []
    omitted, retained = [], []
    for dependency in published:
        dep_name = dependency_name(dependency)
        # Source Torch uses system CUDA/cuDNN and bundled NCCL, not wheel libraries.
        if name == "torch" and (dep_name.startswith("nvidia-") or dep_name in {"cuda-toolkit", "cuda-bindings"}):
            omitted.append(dependency)
        else:
            retained.append(dependency)
    return {
        "source": url, "published_version": version, "requires_python": info.get("requires_python"),
        "published_requires_dist": published, "source_build_requires_dist": retained,
        "omitted_binary_cuda_dependencies": omitted,
        "provides_extra": sorted({extra for dependency in published for extra in re.findall(r"extra\s*==\s*['\"]([^'\"]+)['\"]", dependency)}),
    }


class WheelLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a" and dict(attrs).get("href"):
            self.links.append(dict(attrs)["href"])


def codec_url(version):
    index = "https://download.pytorch.org/whl/cpu/torchcodec/"
    links = WheelLinks()
    links.feed(fetch_text(index))
    matches = []
    for link in links.links:
        absolute = urllib.parse.urljoin(index, link)
        parsed = urllib.parse.urlsplit(absolute)
        name = urllib.parse.unquote(parsed.path.rsplit("/", 1)[-1])
        if re.fullmatch(rf"torchcodec-{re.escape(version)}\+cpu-cp312-cp312-(?:manylinux_2_\d+|manylinux2014)_x86_64\.whl", name):
            if parsed.scheme != "https" or parsed.hostname != "download.pytorch.org" or not re.fullmatch(r"sha256=[0-9a-f]{64}", parsed.fragment):
                raise ValueError("CPU torchcodec wheel must have an official HTTPS URL and SHA256 hash")
            matches.append(absolute)
    if len(set(matches)) != 1:
        raise ValueError(f"Expected one official CPU torchcodec {version} wheel for CPython 3.12/Linux x86_64; found {len(set(matches))}")
    return matches[0]


def metadata_wheel(directory, name, version, info):
    """Write no code: this wheel is exclusively a temporary resolver fixture."""
    wheel = directory / f"{name}-{version}-py3-none-any.whl"
    dist_info = f"{name}-{version}.dist-info"
    rows = ["Metadata-Version: 2.3", f"Name: {name}", f"Version: {version}",
            "Summary: PREFLIGHT ONLY - metadata fixture, not a working package"]
    if info["requires_python"]:
        rows.append(f"Requires-Python: {info['requires_python']}")
    rows.extend(f"Provides-Extra: {extra}" for extra in info["provides_extra"])
    rows.extend(f"Requires-Dist: {dep}" for dep in info["source_build_requires_dist"])
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(f"{dist_info}/METADATA", "\n".join(rows) + "\n\n")
        archive.writestr(f"{dist_info}/WHEEL", "Wheel-Version: 1.0\nGenerator: metadata-preflight\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        archive.writestr(f"{dist_info}/RECORD", "")
    return wheel


def run_check(args, report):
    config = read_config(args.config)
    report["version_config"] = config
    with ThreadPoolExecutor(max_workers=4) as pool:
        releases = dict(zip(REPOSITORIES, pool.map(lambda prefix: resolve_release(prefix, config), REPOSITORIES)))
    report["source_releases"] = releases
    print(f"Verified release tags for vLLM {config['VLLM_VERSION']} and the selected Torch stack.", flush=True)
    source_root = None if args.download_source else args.source_root.resolve()
    if source_root is not None:
        local_commit = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True, encoding="utf-8").strip()
        if local_commit != releases["VLLM"]["commit"]:
            raise ValueError(f"Local source HEAD {local_commit} differs from selected release {releases['VLLM']['commit']}")
        dirty = subprocess.check_output(["git", "-C", str(source_root), "status", "--porcelain", "--", "requirements"], text=True, encoding="utf-8")
        if dirty.strip():
            raise ValueError("Local requirements have uncommitted changes; use a clean release or --download-source")
    with tempfile.TemporaryDirectory(prefix="vllm-cu128-preflight-") as temp:
        work = Path(temp)
        report["requirements_files"] = copy_requirements(work, releases["VLLM"]["commit"], source_root)
        print("Fetched the selected release requirements; reading package metadata.", flush=True)
        report["source_build_metadata"] = {}
        fixtures, pins = [], []
        for name, prefix in SOURCE_PACKAGES.items():
            version = config[prefix + "_VERSION"]
            info = published_metadata(name, version)
            report["source_build_metadata"][name] = info
            fixture = metadata_wheel(work, name, version + "+cu128", info)
            fixtures.append(f"{name} @ {fixture.as_uri()}")
            pins.append(f"{name}=={version}+cu128")
        cpu_codec = codec_url(config["TORCHCODEC_VERSION"])
        report["torchcodec_cpu_wheel"] = cpu_codec
        pins += [f"triton=={config['TRITON_VERSION']}", f"torchcodec=={config['TORCHCODEC_VERSION']}+cpu"]
        constraints = args.constraints.read_text(encoding="utf-8")
        # versions.env owns these pins, even when reading an older policy file.
        policy = [line for line in constraints.splitlines() if dependency_name(line) not in POLICY_PACKAGES] + pins
        constraint_file = work / "constraints.txt"
        constraint_file.write_text("\n".join(policy) + "\n", encoding="utf-8")
        report["effective_constraints"] = "\n".join(policy)
        frontend_pins = re.findall(r"^nvidia-cudnn-frontend==([^\s+]+)\+cu128$", "\n".join(policy), re.MULTILINE)
        if len(frontend_pins) != 1:
            raise ValueError("Expected one adapted nvidia-cudnn-frontend==VERSION+cu128 policy pin")
        prepare_frontend = runpy.run_path(str(Path(__file__).with_name("prepare-cudnn-frontend.py")))["prepare_frontend"]
        frontend_wheel, frontend_info = prepare_frontend(work / "frontend", version=frontend_pins[0])
        report["cudnn_frontend_wheel"] = frontend_info
        inputs = work / "preflight.in"
        inputs.write_text("\n".join(fixtures + [f"nvidia-cudnn-frontend @ {frontend_wheel.as_uri()}", f"torchcodec @ {cpu_codec}", f"triton=={config['TRITON_VERSION']}", "cmake>=3.27,<4", "setuptools>=77.0.3,<81"]) + "\n", encoding="utf-8")
        resolved = work / "RESOLVER_ONLY_DO_NOT_INSTALL.txt"
        command = [args.uv, "pip", "compile", str(inputs), str(work / "requirements/cuda.txt"),
                   str(work / "requirements/build/cuda.txt"), "--constraint", str(constraint_file),
                   "--python-version", "3.12", "--python-platform", "x86_64-manylinux_2_39",
                   "--index-strategy", "first-index", "--no-build", "--output-file", str(resolved)]
        command.append("--verbose" if args.verbose_resolver else "--quiet")
        env = os.environ.copy()
        for name in ("UV_CONSTRAINT", "PIP_CONSTRAINT"):
            env.pop(name, None)
        report["uv_version"] = subprocess.check_output([args.uv, "--version"], text=True, encoding="utf-8").strip()
        print("Resolving Linux CUDA 12.8 dependencies; external wheel metadata may need downloading.", flush=True)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        resolver_log = args.report.with_suffix(".resolver.log")
        report["resolver_log"] = str(resolver_log)
        with resolver_log.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                result_code = process.wait(timeout=args.resolver_timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                raise ValueError(f"Dependency resolution exceeded {args.resolver_timeout}s; see {resolver_log}")
        if result_code:
            output = resolver_log.read_text(encoding="utf-8", errors="replace")
            raise ValueError(output[-16000:].strip() or f"uv exited with code {result_code}")
        resolution = resolved.read_text(encoding="utf-8")
        direct_names = {dependency_name(name) for name in re.findall(r"^([A-Za-z0-9_.-]+)\s+@\s+", resolution, re.MULTILINE)}
        unknown_direct = direct_names - {*SOURCE_PACKAGES, "torchcodec", "nvidia-cudnn-frontend"}
        if unknown_direct:
            raise ValueError(f"Cannot audit versions of unexpected direct-URL dependencies: {sorted(unknown_direct)}")
        packages = dict(re.findall(r"^([A-Za-z0-9_.-]+)==([^\s;]+)", resolution, re.MULTILINE))
        packages = {dependency_name(name): version for name, version in packages.items()}
        packages.update({name: config[prefix + "_VERSION"] + "+cu128" for name, prefix in SOURCE_PACKAGES.items()})
        packages["torchcodec"] = config["TORCHCODEC_VERSION"] + "+cpu"
        packages["nvidia-cudnn-frontend"] = frontend_pins[0] + "+cu128"
        report["resolved_packages"] = packages
        report["cuda_related_packages"] = {name: version for name, version in packages.items() if name.startswith(("cuda-", "nvidia-", "flashinfer", "triton", "tokenspeed", "torch"))}
        errors = []
        for name, version in packages.items():
            if name.endswith("-cu13"):
                errors.append(f"Unexpected CUDA 13 dependency: {name}=={version}")
            if name in CUDA_128_PACKAGES and not version.startswith("12.8."):
                errors.append(f"Expected CUDA 12.8 dependency: {name}=={version}")
        report["cuda_dependency_errors"] = errors
        if errors:
            raise ValueError("\n".join(errors))
        report["success"] = True
        print(f"Resolved {len(packages)} packages using published metadata for the selected source builds.")
        print(json.dumps(report["cuda_related_packages"], indent=2))


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=project / "versions.env")
    parser.add_argument("--source-root", type=Path, default=project / "upstream-vllm")
    parser.add_argument("--download-source", action="store_true", help="Verify release tags and recursively download requirements at the resolved commit")
    parser.add_argument("--constraints", type=Path, default=project / "scripts/cu128-constraints.txt")
    parser.add_argument("--report", type=Path, default=project / "artifacts/dependency-preflight.json")
    parser.add_argument("--uv", default="uv")
    parser.add_argument("--resolver-timeout", type=int, default=1800, help="Seconds allowed for external wheel metadata resolution")
    parser.add_argument("--verbose-resolver", action="store_true", help="Save detailed resolver output for network/dependency diagnostics")
    args = parser.parse_args()
    if args.resolver_timeout <= 0:
        parser.error("--resolver-timeout must be positive")
    report = {
        "purpose": "Dependency metadata resolution only; not a build or runtime certification",
        "platform": "CPython 3.12 / Linux x86_64 / glibc 2.39", "success": False,
        "metadata_limitations": [
            "Temporary Torch-family wheels contain no code and must never be installed.",
            "Requires-Dist and Requires-Python come from selected published PyPI releases; source-built wheel metadata may differ.",
            "Torch binary CUDA dependencies (nvidia-*, cuda-toolkit, cuda-bindings) are omitted: this image uses system CUDA/cuDNN and bundled NCCL.",
            "PyPI package-level metadata does not prove source, ABI, GPU or model compatibility.",
            "Actual wheel installation, uv pip check, CUDA compilation and GPU tests remain required.",
            "cuDNN frontend uses a real adapted wheel: official code/binaries unchanged, CUTLASS dependency extra set to CUDA 12, with wheel hashes recorded.",
            "Optional compiler/disassembler packages may contain CUDA 12.9/13-era tools despite the CUDA 12.8 runtime policy.",
        ],
    }
    print("Dependency preflight only: no package installation or CUDA compilation.", flush=True)
    try:
        run_check(args, report)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        report["error"] = str(error)
        print(f"ERROR: {error}", flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Preflight report: {args.report.resolve()}")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
