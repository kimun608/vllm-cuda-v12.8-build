"""CPU-only patch regression tests; optionally test actual local Git release blobs.

python tests/cu128-patch-layouts.py --release-repository artifacts/research-v032/source
No network requests, CUDA compilation or production-source modifications occur.
"""
import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cu128_patch", PROJECT / "scripts/apply-cu128-patches.py")
PATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCH)
RELEASE_REPOSITORY = None

HEADER = '''#include "topk_histogram_4096.cuh"
while (!cuda::ptx::mbarrier_try_wait_parity(cuda::ptx::sem_relaxed,
    cuda::ptx::scope_cta, a, p));
cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_relaxed,
    cuda::ptx::scope_cta, cuda::ptx::space_shared, a, t);
cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
    cuda::ptx::space_global, d, s, n, m);
'''
VARIABLES = ("MACHETE_GEN_SCRIPT_HASH", "MARLIN_GEN_SCRIPT_HASH_AND_ARCH",
             "MOE_MARLIN_GEN_SCRIPT_HASH_AND_ARCH")
SENTINELS = ("quantization/machete/generated/machete_mm_dispatch.cu",
             "quantization/marlin/kernel_selector.h",
             "moe/marlin_moe_wna16/kernel_selector.h")


def write(source, relative, text):
    target = source / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def fixture(source, root):
    write(source, root / "cooperative_topk.cuh", HEADER)
    write(source, root / "quantization/fp4/nvfp4_quant_kernels.cu", "      fp4_packed_t out_val{};\n")
    cmake = "\n".join(f"if(NOT DEFINED CACHE{{{var}}}\n    OR NOT $CACHE{{{var}}} STREQUAL ${{{var}}})\nendif()" for var in VARIABLES)
    write(source, "CMakeLists.txt", cmake + "\n    include(cmake/external_projects/deepgemm.cmake)\n")
    write(source, "setup.py", '        ext_modules.append(CMakeExtension(name="vllm._deep_gemm_C", optional=True))\n')


def snapshot(source):
    return {path.relative_to(source).as_posix(): path.read_bytes()
            for path in source.rglob("*") if path.is_file()}


class PatchLayouts(unittest.TestCase):
    def apply_twice_and_check(self, source, root, report):
        with contextlib.redirect_stdout(io.StringIO()):
            PATCH.apply(source, report)
        first_manifest = report.read_bytes()
        first_files = snapshot(source)
        manifest = json.loads(first_manifest)
        self.assertEqual(manifest["cuda_source_root"], root.as_posix())
        self.assertEqual(len(manifest["patches"]), 4)
        header = (source / root / "cooperative_topk.cuh").read_text(encoding="utf-8")
        self.assertEqual(header.count(PATCH.GUARD), 3)
        self.assertEqual(header.count(PATCH.INCLUDE), 1)
        self.assertEqual(header.count("cuda::ptx::mbarrier_try_wait_parity"), 1)
        self.assertTrue((source / root / PATCH.HELPER_NAME).is_file())
        fp4 = (source / root / "quantization/fp4/nvfp4_quant_kernels.cu").read_text(encoding="utf-8")
        self.assertIn("      QuantizedType out_val{};", fp4)
        self.assertNotIn("      fp4_packed_t out_val{};", fp4)
        generator = next(item for item in manifest["patches"] if item["id"] == "regenerate-missing-cuda-sources-v1")
        self.assertEqual(generator["required_generated_files"], [(root / item).as_posix() for item in SENTINELS])
        cmake = (source / "CMakeLists.txt").read_text(encoding="utf-8")
        for sentinel in generator["required_generated_files"]:
            self.assertIn(f'OR NOT EXISTS "${{CMAKE_CURRENT_SOURCE_DIR}}/{sentinel}"', cmake)
        for item in manifest["patches"]:
            self.assertEqual(item["patched_source_sha256"], PATCH.digest((source / item["source"]).read_bytes()))
        with contextlib.redirect_stdout(io.StringIO()):
            PATCH.apply(source, report)
        self.assertEqual(first_manifest, report.read_bytes())
        self.assertEqual(first_files, snapshot(source))

    def test_both_layouts_and_idempotence(self):
        for root in (Path("csrc/libtorch_stable"), Path("csrc")):
            with self.subTest(root=root), tempfile.TemporaryDirectory() as temp:
                base = Path(temp)
                fixture(base / "source", root)
                self.apply_twice_and_check(base / "source", root, base / "report.json")

    def test_unknown_layout_fails_before_modification(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)
            write(source, "CMakeLists.txt", "unchanged")
            before = snapshot(source)
            with self.assertRaisesRegex(RuntimeError, "exactly one reviewed"):
                PATCH.apply(source, source / "report.json")
            self.assertEqual(before, snapshot(source))

    def test_ambiguous_layout_fails_before_modification(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)
            fixture(source, Path("csrc/libtorch_stable"))
            fixture(source, Path("csrc"))
            before = snapshot(source)
            with self.assertRaisesRegex(RuntimeError, "exactly one reviewed"):
                PATCH.apply(source, source / "report.json")
            self.assertEqual(before, snapshot(source))

    def test_partial_layout_fails_before_modification(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)
            write(source, "csrc/cooperative_topk.cuh", HEADER)
            before = snapshot(source)
            with self.assertRaisesRegex(RuntimeError, "Incomplete CUDA source layout"):
                PATCH.apply(source, source / "report.json")
            self.assertEqual(before, snapshot(source))

    def test_actual_release_blobs(self):
        if RELEASE_REPOSITORY is None:
            self.skipTest("Pass --release-repository with local v0.31.0/v0.32.0 Git tags")
        for tag, root in (("v0.31.0", Path("csrc/libtorch_stable")), ("v0.32.0", Path("csrc"))):
            with self.subTest(tag=tag), tempfile.TemporaryDirectory() as temp:
                base = Path(temp)
                source = base / "source"
                files = (Path("CMakeLists.txt"), Path("setup.py"), root / "cooperative_topk.cuh",
                         root / "quantization/fp4/nvfp4_quant_kernels.cu")
                for relative in files:
                    data = subprocess.check_output(["git", "-C", str(RELEASE_REPOSITORY), "show", f"{tag}:{relative.as_posix()}"])
                    target = source / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
                self.apply_twice_and_check(source, root, base / "report.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-repository", type=Path)
    args = parser.parse_args()
    RELEASE_REPOSITORY = args.release_repository
    unittest.main(argv=[__file__], verbosity=2)
