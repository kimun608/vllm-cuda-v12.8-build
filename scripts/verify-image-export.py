"""Validate a Docker-save archive against a config, OCI manifest or index ID.

Docker's classic image store reports a config digest; the containerd image
store can report a manifest/index digest. Both identities must authenticate
the selected Linux/amd64 image and its saved layer data, not merely exist.
"""

import argparse
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile


INDEX_TYPES = {
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
}
MANIFEST_TYPES = {
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
}
DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
MAX_JSON_BYTES = 16 * 1024 * 1024


def canonical_tag(value):
    if not isinstance(value, str) or not value or value != value.strip() or any(character.isspace() for character in value):
        raise ValueError("Invalid image tag")
    for prefix in ("docker.io/", "index.docker.io/", "registry-1.docker.io/"):
        if value.startswith(prefix):
            value = value[len(prefix):]
            break
    if value.startswith("library/") and value.count("/") == 1:
        value = value[len("library/"):]
    if ":" not in value.rsplit("/", 1)[-1]:
        value += ":latest"
    return value


def safe_name(value):
    if not isinstance(value, str) or "\\" in value or PurePosixPath(value).is_absolute() or ".." in PurePosixPath(value).parts:
        raise ValueError("Unsafe archive member name")
    while value.startswith("./"):
        value = value[2:]
    return value.rstrip("/")


def require_digest(value):
    if not isinstance(value, str) or not DIGEST.fullmatch(value):
        raise ValueError(f"Unsupported or invalid content digest: {value!r}")
    return value


class SavedArchive:
    def __init__(self, archive):
        self.archive = archive
        self.members = {}
        self.hashes = {}
        self.jsons = {}
        for item in archive.getmembers():
            name = safe_name(item.name)
            if name in self.members:
                raise ValueError(f"Duplicate archive member: {name}")
            self.members[name] = item

    def member(self, name):
        name = safe_name(name)
        item = self.members.get(name)
        if item is None or not item.isfile():
            raise ValueError(f"Missing regular archive file: {name}")
        return item

    def content_hash(self, name):
        name = safe_name(name)
        if name not in self.hashes:
            with self.archive.extractfile(self.member(name)) as stream:
                digest = hashlib.sha256()
                while True:
                    chunk = stream.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
            self.hashes[name] = "sha256:" + digest.hexdigest()
        return self.hashes[name]

    def json(self, name):
        name = safe_name(name)
        if name not in self.jsons:
            item = self.member(name)
            if item.size > MAX_JSON_BYTES:
                raise ValueError(f"Unexpectedly large image metadata: {name}")
            with self.archive.extractfile(item) as stream:
                self.jsons[name] = json.load(stream)
        return self.jsons[name]

    def blob(self, descriptor, json_content=False):
        if not isinstance(descriptor, dict):
            raise ValueError("Invalid OCI descriptor")
        digest = require_digest(descriptor.get("digest"))
        size = descriptor.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError(f"Invalid OCI descriptor size: {digest}")
        name = "blobs/sha256/" + digest.split(":", 1)[1]
        if name not in self.members and json_content and isinstance(descriptor.get("data"), str):
            # BuildKit OCI attestations may embed the standard empty config
            # descriptor rather than materializing its two-byte blob in tar.
            data = base64.b64decode(descriptor["data"], validate=True)
            if len(data) != size or size > MAX_JSON_BYTES or "sha256:" + hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f"Inline OCI descriptor size/SHA256 mismatch: {digest}")
            return json.loads(data)
        if self.member(name).size != size:
            raise ValueError(f"OCI descriptor size mismatch: {digest}")
        if self.content_hash(name) != digest:
            raise ValueError(f"OCI descriptor SHA256 mismatch: {digest}")
        return self.json(name) if json_content else name


def config_platform(config):
    if not isinstance(config, dict):
        raise ValueError("Invalid image config JSON")
    return config.get("os"), config.get("architecture")


def root_matches_tag(descriptor, image_tag):
    if not isinstance(descriptor, dict):
        raise ValueError("Invalid OCI root descriptor")
    annotations = descriptor.get("annotations") or {}
    if not isinstance(annotations, dict):
        raise ValueError("Invalid OCI descriptor annotations")
    full_name = annotations.get("io.containerd.image.name")
    ref_name = annotations.get("org.opencontainers.image.ref.name")
    if full_name:
        return canonical_tag(full_name) == image_tag
    if ref_name:
        if "/" in ref_name or ":" in ref_name:
            return canonical_tag(ref_name) == image_tag
        return ref_name == image_tag.rsplit(":", 1)[-1]
    # Older Docker exports omit annotations. The legacy tag/config/layers must
    # still match the authenticated graph; an unannotated root alone is no proof.
    return True


def verify_export(path, image_tag, expected_id):
    image_tag = canonical_tag(image_tag)
    expected_id = require_digest(expected_id)
    with tarfile.open(path, "r:") as archive:
        saved = SavedArchive(archive)
        legacy = saved.json("manifest.json")
        if not isinstance(legacy, list):
            raise ValueError("Docker manifest.json must contain a list")
        selected = [entry for entry in legacy if isinstance(entry, dict) and any(canonical_tag(tag) == image_tag for tag in (entry.get("RepoTags") or []))]
        if len(selected) != 1:
            raise ValueError("Export has no unique selected image tag")
        entry = selected[0]
        config_name = safe_name(entry.get("Config"))
        config = saved.json(config_name)
        config_id = saved.content_hash(config_name)
        if config_platform(config) != ("linux", "amd64"):
            raise ValueError("Selected image config is not Linux/amd64")
        rootfs = config.get("rootfs") or {}
        diff_ids = rootfs.get("diff_ids")
        if rootfs.get("type") != "layers" or not isinstance(diff_ids, list):
            raise ValueError("Selected image config has no layer diff-ID list")
        for digest in diff_ids:
            require_digest(digest)
        layer_names = entry.get("Layers")
        if not isinstance(layer_names, list) or len(layer_names) != len(diff_ids):
            raise ValueError("Docker layer list is inconsistent with image config")
        layer_names = [safe_name(name) for name in layer_names]
        legacy_layer_hashes = [saved.content_hash(name) for name in layer_names]
        identity_kind = "config" if expected_id == config_id else None
        matching_manifests = []
        verified_nodes = set()

        def walk(descriptor, ancestors=(), stack=()):
            if not isinstance(descriptor, dict):
                raise ValueError("Invalid OCI child descriptor")
            digest = require_digest(descriptor.get("digest"))
            if digest in stack or len(stack) >= 64:
                raise ValueError("Cyclic or excessively deep OCI descriptor graph")
            platform = descriptor.get("platform") or {}
            if not isinstance(platform, dict):
                raise ValueError("Invalid OCI platform descriptor")
            declared_platform = (platform.get("os"), platform.get("architecture"))
            if all(declared_platform) and declared_platform not in (("linux", "amd64"), ("unknown", "unknown")):
                # Docker can save only downloaded platforms while retaining
                # the original multi-platform index's authenticated digest.
                return
            node = saved.blob(descriptor, json_content=True)
            verified_nodes.add(digest)
            if not isinstance(node, dict) or node.get("schemaVersion") != 2:
                raise ValueError(f"Unsupported OCI metadata schema: {digest}")
            media_type = descriptor.get("mediaType") or node.get("mediaType")
            if descriptor.get("mediaType") and node.get("mediaType") and descriptor["mediaType"] != node["mediaType"]:
                raise ValueError(f"OCI media type mismatch: {digest}")
            chain = ancestors + (digest,)
            if media_type in INDEX_TYPES:
                children = node.get("manifests")
                if not isinstance(children, list):
                    raise ValueError("OCI index is missing child descriptors")
                for child in children:
                    walk(child, chain, stack + (digest,))
                return
            if media_type not in MANIFEST_TYPES:
                raise ValueError(f"Unsupported OCI graph media type: {media_type!r}")
            config_descriptor = node.get("config")
            if not isinstance(config_descriptor, dict):
                # OCI artifacts are not runnable images. Their manifest is
                # authenticated but cannot match the selected Docker image.
                if node.get("artifactType") or node.get("subject"):
                    return
                raise ValueError("OCI image manifest has no config descriptor")
            node_config = saved.blob(config_descriptor, json_content=True)
            annotations = descriptor.get("annotations") or {}
            if node.get("artifactType") or config_descriptor.get("mediaType") == "application/vnd.oci.empty.v1+json" or annotations.get("vnd.docker.reference.type") == "attestation-manifest":
                # An OCI empty config has no os/architecture. Attestations
                # authenticate auxiliary metadata, not a runnable platform.
                return
            actual_platform = config_platform(node_config)
            if platform.get("os") and platform.get("architecture") and (platform["os"], platform["architecture"]) != actual_platform:
                raise ValueError(f"OCI platform descriptor/config mismatch: {digest}")
            if actual_platform != ("linux", "amd64"):
                # BuildKit attestation manifests normally use unknown/unknown.
                return
            if config_descriptor["digest"] != config_id:
                return
            layers = node.get("layers")
            if not isinstance(layers, list) or len(layers) != len(layer_names):
                raise ValueError("OCI/Docker layer counts disagree for the selected image")
            for index, layer in enumerate(layers):
                layer_path = saved.blob(layer)
                actual_hash = saved.content_hash(layer_path)
                if legacy_layer_hashes[index] not in (actual_hash, diff_ids[index]):
                    raise ValueError("OCI/Docker layer contents disagree for the selected image")
            if identity_kind == "config" or expected_id in chain:
                matching_manifests.append((digest, chain, media_type))

        if "index.json" in saved.members:
            layout = saved.json("oci-layout")
            if not isinstance(layout, dict) or layout.get("imageLayoutVersion") != "1.0.0":
                raise ValueError("Unsupported OCI layout version")
            top_index = saved.json("index.json")
            if not isinstance(top_index, dict) or top_index.get("schemaVersion") != 2 or not isinstance(top_index.get("manifests"), list):
                raise ValueError("Invalid exported OCI root index")
            roots = [descriptor for descriptor in top_index["manifests"] if root_matches_tag(descriptor, image_tag)]
            if not roots:
                raise ValueError("OCI root index has no descriptor for the selected image tag")
            for descriptor in roots:
                walk(descriptor)
            unique_matches = {digest: (chain, media_type) for digest, chain, media_type in matching_manifests}
            if len(unique_matches) != 1:
                raise ValueError("Expected image ID has no unique authenticated Linux/amd64 manifest matching the Docker tag/config/layers")
            manifest_id, (chain, _) = next(iter(unique_matches.items()))
            if identity_kind is None:
                if expected_id not in chain:
                    raise ValueError("Expected image ID does not authenticate the selected image")
                identity_kind = "manifest" if expected_id == manifest_id else "index"
        else:
            if identity_kind != "config":
                raise ValueError("A classic Docker archive cannot authenticate a manifest/index image ID")
            if legacy_layer_hashes != diff_ids:
                raise ValueError("Classic Docker layer contents do not match config diff IDs")
            manifest_id = None
        return {
            "status": "passed",
            "image": image_tag,
            "expected_image_id": expected_id,
            "identity_kind": identity_kind,
            "config_id": config_id,
            "linux_amd64_manifest_id": manifest_id,
            "layers": len(layer_names),
            "layer_contents_verified": True,
            "metadata_nodes_verified": len(verified_nodes),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("image")
    parser.add_argument("expected_id")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"status": "failed", "image": args.image, "expected_image_id": args.expected_id}
    exit_code = 1
    try:
        report = verify_export(args.archive, args.image, args.expected_id)
        exit_code = 0
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError) as error:
        report["error"] = str(error)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
