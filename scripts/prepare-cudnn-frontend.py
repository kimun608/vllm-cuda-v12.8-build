"""Create a reviewed CUDA 12 dependency adaptation of the real frontend wheel.

No executable code is altered. The official CUDA-13 CUTLASS extra is replaced
with the CUDA-12 default dependency, and the wheel receives a local version.
Both preflight and production must use this same wheel and its real metadata.
"""

import argparse
import base64
import copy
import csv
from email.parser import BytesParser
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import urllib.parse
import urllib.request
import zipfile


PACKAGE = "nvidia-cudnn-frontend"
REVIEWED_VERSION = "1.29.0"
REVIEWED_SOURCE_SHA256 = "596c9b7591457b4b49227fa2e22bd5960957a198e4336ce9ada8b5d2a1e4a5a1"
ADAPTATION_ID = "cuda128-cudnn-frontend-cutlass-extra-v1"
REPORT_NAME = "nvidia-cudnn-frontend-wheel.json"
ORIGINAL_REQUIREMENT = re.compile(
    r"nvidia-cutlass-dsl\s*\[\s*cu13\s*\]\s*>=\s*4\.6\.2\s*\Z",
    re.IGNORECASE,
)
MODIFIED_REQUIREMENT = "nvidia-cutlass-dsl>=4.6.2"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def normalized_name(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "vllm-cu128-frontend-adaptation"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def official_wheel(version):
    if version != REVIEWED_VERSION:
        raise ValueError(f"Frontend {version!r} has not been reviewed; review its wheel and dependency declaration first")
    metadata_url = f"https://pypi.org/pypi/{PACKAGE}/{version}/json"
    release = json.loads(fetch(metadata_url))
    if normalized_name(release["info"]["name"]) != PACKAGE or release["info"]["version"] != version:
        raise ValueError("Unexpected PyPI release name/version")
    pattern = re.compile(
        rf"nvidia_cudnn_frontend-{re.escape(version)}-cp312-cp312-"
        r"manylinux_2_27_x86_64\.manylinux_2_28_x86_64\.whl\Z"
    )
    candidates = [item for item in release["urls"] if pattern.fullmatch(item["filename"])]
    if len(candidates) != 1:
        raise ValueError("Expected exactly one reviewed official CPython 3.12/Linux x86_64 frontend wheel")
    item = candidates[0]
    url = urllib.parse.urlsplit(item["url"])
    if url.scheme != "https" or url.hostname != "files.pythonhosted.org" or url.username or url.password:
        raise ValueError("Frontend wheel must come from the official PyPI HTTPS file host")
    digest = item["digests"]["sha256"]
    if digest != REVIEWED_SOURCE_SHA256 or item.get("yanked"):
        raise ValueError("The published frontend wheel digest/status differs from the reviewed release")
    return item, metadata_url


def rewrite_metadata(raw, version, modified_version):
    metadata = BytesParser().parsebytes(raw)
    if len(metadata.get_all("Name", [])) != 1 or normalized_name(metadata["Name"]) != PACKAGE:
        raise ValueError("Unexpected or duplicate wheel METADATA Name")
    if metadata.get_all("Version", []) != [version]:
        raise ValueError("Unexpected or duplicate wheel METADATA Version")
    before = metadata.get_all("Requires-Dist", [])
    cutlass_requirements = [item for item in before if re.match(r"nvidia[-_.]cutlass[-_.]dsl\b", item, re.IGNORECASE)]
    if len(cutlass_requirements) != 1 or not ORIGINAL_REQUIREMENT.fullmatch(cutlass_requirements[0]):
        raise ValueError("Unknown frontend CUTLASS dependency; only the reviewed unconditional [cu13]>=4.6.2 declaration may be adapted")

    # Preserve all unrelated header/body bytes, including folded headers. Only
    # the two reviewed header blocks are rewritten; WHEEL stays byte-identical.
    lines = raw.splitlines(keepends=True)
    blocks, current = [], []
    body_start = None
    for index, line in enumerate(lines):
        if line in (b"\n", b"\r\n"):
            if current:
                blocks.append(current)
            body_start = index
            break
        if line.startswith((b" ", b"\t")):
            if not current:
                raise ValueError("Invalid folded METADATA header")
            current.append(line)
        else:
            if current:
                blocks.append(current)
            current = [line]
    if body_start is None:
        raise ValueError("Wheel METADATA has no header/body separator")
    newline = b"\r\n" if b"\r\n" in raw[: sum(len(line) for line in lines[:body_start])] else b"\n"
    modified_blocks = []
    changes = {"Version": 0, "Requires-Dist": 0}
    for block in blocks:
        header = BytesParser().parsebytes(b"".join(block) + newline)
        if header.get("Version") is not None:
            changes["Version"] += 1
            modified_blocks.append(b"Version: " + modified_version.encode("ascii") + newline)
        elif header.get("Requires-Dist") is not None and ORIGINAL_REQUIREMENT.fullmatch(header["Requires-Dist"]):
            changes["Requires-Dist"] += 1
            modified_blocks.append(b"Requires-Dist: " + MODIFIED_REQUIREMENT.encode("ascii") + newline)
        else:
            modified_blocks.append(b"".join(block))
    if changes != {"Version": 1, "Requires-Dist": 1}:
        raise ValueError("Expected exactly one Version and one CUTLASS dependency header replacement")
    rewritten = b"".join(modified_blocks) + b"".join(lines[body_start:])
    after = BytesParser().parsebytes(rewritten).get_all("Requires-Dist", [])
    expected = [MODIFIED_REQUIREMENT if ORIGINAL_REQUIREMENT.fullmatch(item) else item for item in before]
    if after != expected:
        raise ValueError("An unrelated Requires-Dist entry changed")
    return rewritten, before, after


def adapt_wheel(original, version):
    modified_version = version + "+cu128"
    old_info = f"nvidia_cudnn_frontend-{version}.dist-info/"
    new_info = f"nvidia_cudnn_frontend-{modified_version}.dist-info/"
    record_path = new_info + "RECORD"
    output = io.BytesIO()
    files, removed_signatures = {}, []
    unchanged_payload = {}
    with zipfile.ZipFile(io.BytesIO(original)) as source:
        entries = source.infolist()
        if len({item.filename for item in entries}) != len(entries):
            raise ValueError("Official wheel contains duplicate entry names")
        if {item.filename.split("/", 1)[0] for item in entries if ".dist-info/" in item.filename} != {old_info.rstrip("/")}:
            raise ValueError("Unexpected wheel dist-info layout")
        if any(item.filename.startswith(f"nvidia_cudnn_frontend-{version}.data/") for item in entries):
            raise ValueError("Unexpected wheel .data layout; review before adapting")
        raw_metadata = source.read(old_info + "METADATA")
        new_metadata, before, after = rewrite_metadata(raw_metadata, version, modified_version)
        with zipfile.ZipFile(output, "w") as target:
            target.comment = source.comment
            for item in entries:
                name = item.filename
                if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts or "\\" in name:
                    raise ValueError("Unsafe wheel member path")
                if name in (old_info + "RECORD.jws", old_info + "RECORD.p7s"):
                    removed_signatures.append(name)
                    continue
                if name == old_info + "RECORD":
                    continue
                data = source.read(name)
                new_name = new_info + name[len(old_info):] if name.startswith(old_info) else name
                if name == old_info + "METADATA":
                    data = new_metadata
                elif not item.is_dir():
                    unchanged_payload[new_name] = sha256(data)
                new_item = copy.copy(item)
                new_item.filename = new_name
                target.writestr(new_item, data)
                if not item.is_dir():
                    files[new_name] = data
            record = io.StringIO(newline="")
            writer = csv.writer(record, lineterminator="\n")
            for name, data in sorted(files.items()):
                digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
                writer.writerow([name, "sha256=" + digest, str(len(data))])
            writer.writerow([record_path, "", ""])
            record_item = copy.copy(source.getinfo(old_info + "RECORD"))
            record_item.filename = record_path
            target.writestr(record_item, record.getvalue().encode("utf-8"))
        patched = output.getvalue()
        # This comparison is against the real downloaded official archive, not
        # against regenerated fixture content or a package-level metadata claim.
        with zipfile.ZipFile(io.BytesIO(patched)) as check:
            for item in entries:
                name = item.filename
                if item.is_dir() or name in (old_info + "METADATA", old_info + "RECORD", *removed_signatures):
                    continue
                mapped = new_info + name[len(old_info):] if name.startswith(old_info) else name
                if source.read(name) != check.read(mapped):
                    raise ValueError(f"Non-metadata wheel content changed: {name}")
            validate_record(check)
    payload_digest = sha256(json.dumps(unchanged_payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return patched, {
        "requires_dist_before": before,
        "requires_dist_after": after,
        "removed_signatures": removed_signatures,
        "changed_content_entries": [new_info + "METADATA", record_path],
        "code_unchanged": True,
        "unchanged_payload_files": len(unchanged_payload),
        "unchanged_payload_manifest_sha256": payload_digest,
        "wheel_file_unchanged": True,
    }


def validate_record(archive):
    record_names = [name for name in archive.namelist() if name.endswith(".dist-info/RECORD")]
    if len(record_names) != 1:
        raise ValueError("Adapted wheel must contain one RECORD")
    record_name = record_names[0]
    rows = list(csv.reader(io.StringIO(archive.read(record_name).decode("utf-8"))))
    names = set()
    for row in rows:
        if len(row) != 3 or row[0] in names:
            raise ValueError("Invalid or duplicate adapted RECORD row")
        name, digest, length = row
        names.add(name)
        if name == record_name:
            if digest or length:
                raise ValueError("RECORD cannot hash itself")
            continue
        data = archive.read(name)
        actual = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
        if digest != actual or length != str(len(data)):
            raise ValueError(f"Adapted RECORD hash/size mismatch: {name}")
    if names != {item.filename for item in archive.infolist() if not item.is_dir()}:
        raise ValueError("Adapted RECORD does not cover all wheel files")


def atomic_write(path, data):
    with tempfile.NamedTemporaryFile(prefix=path.name + ".", suffix=".tmp", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_frontend(directory, version=REVIEWED_VERSION):
    """Return (real adapted wheel Path, provenance dict); never install it."""
    official, metadata_url = official_wheel(version)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    modified_version = version + "+cu128"
    modified_name = official["filename"].replace(f"-{version}-", f"-{modified_version}-", 1)
    wheel_path = directory / modified_name
    report_path = directory / REPORT_NAME
    if wheel_path.is_file() and report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            cached = wheel_path.read_bytes()
            if (
                report["adaptation_id"] == ADAPTATION_ID
                and report["original_wheel"]["sha256"] == official["digests"]["sha256"]
                and report["original_wheel"]["url"] == official["url"]
                and report["modified_wheel"]["filename"] == modified_name
                and report["modified_wheel"]["version"] == modified_version
                and report["modified_wheel"]["sha256"] == sha256(cached)
                and report["code_unchanged"] is True
            ):
                with zipfile.ZipFile(io.BytesIO(cached)) as archive:
                    validate_record(archive)
                    metadata = BytesParser().parsebytes(archive.read(f"nvidia_cudnn_frontend-{modified_version}.dist-info/METADATA"))
                    if metadata.get_all("Version", []) != [modified_version] or metadata.get_all("Requires-Dist", []) != report["requires_dist_after"]:
                        raise ValueError("Cached wheel metadata does not match provenance")
                return wheel_path, report
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            pass
    original = fetch(official["url"])
    if len(original) != official["size"] or sha256(original) != official["digests"]["sha256"]:
        raise ValueError("Downloaded official frontend wheel failed published SHA256/size verification")
    modified, comparison = adapt_wheel(original, version)
    report = {
        "schema_version": 1,
        "adaptation_id": ADAPTATION_ID,
        "package": PACKAGE,
        "pypi_metadata_url": metadata_url,
        "original_wheel": {"filename": official["filename"], "version": version, "url": official["url"], "sha256": sha256(original), "size_bytes": len(original)},
        "modified_wheel": {"filename": modified_name, "version": modified_version, "sha256": sha256(modified), "size_bytes": len(modified)},
        "dependency_change": "Use the default CUDA 12 CUTLASS DSL flavor instead of requesting the CUDA 13 extra; preserve all executable payload bytes.",
        **comparison,
    }
    atomic_write(wheel_path, modified)
    atomic_write(report_path, (json.dumps(report, indent=2) + "\n").encode("utf-8"))
    return wheel_path, report


def version_from_constraints(path):
    declarations = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if re.match(r"\s*nvidia[-_.]cudnn[-_.]frontend\b", line, re.IGNORECASE):
            declarations.append(line)
    if len(declarations) != 1:
        raise ValueError("Constraints must contain exactly one nvidia-cudnn-frontend pin")
    match = re.fullmatch(r"\s*nvidia[-_.]cudnn[-_.]frontend\s*==\s*(\d+\.\d+\.\d+)\+cu128\s*(?:#.*)?", declarations[0], re.IGNORECASE)
    if not match:
        raise ValueError("Frontend constraints pin must be an exact reviewed <version>+cu128")
    return match.group(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--version", default=REVIEWED_VERSION)
    parser.add_argument("--constraints", type=Path, help="Read the frontend +cu128 pin; overrides --version")
    args = parser.parse_args()
    version = version_from_constraints(args.constraints) if args.constraints else args.version
    wheel, report = prepare_frontend(args.output_dir, version=version)
    print(json.dumps({"wheel": str(wheel), "provenance": str(wheel.parent / REPORT_NAME), "version": report["modified_wheel"]["version"], "code_unchanged": report["code_unchanged"]}))


if __name__ == "__main__":
    main()
