"""Apply reviewed CUDA 12.8 compatibility and build-cache fixes."""
import argparse
import hashlib
import json
from pathlib import Path
import re


PATCH_ID = "cuda128-cooperative-topk-ptx-wrappers-v1"
CUDA_SOURCE_ROOTS = (Path("csrc/libtorch_stable"), Path("csrc"))
HELPER_NAME = "cu128-ptx-compat.cuh"
INCLUDE = f'#include "{HELPER_NAME}"'
GUARD = "#if CUDART_VERSION >= 12080 && CUDART_VERSION < 12090"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def detect_cuda_source_root(source):
    """v0.32 moved stable-ABI CUDA files to csrc without changing their ABI."""
    matches = [root for root in CUDA_SOURCE_ROOTS
               if (source / root / "cooperative_topk.cuh").is_file()]
    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one reviewed CUDA source layout "
            "(csrc/libtorch_stable or csrc); "
            f"found {[root.as_posix() for root in matches]}. Review this release."
        )
    root = matches[0]
    required = (Path("CMakeLists.txt"), Path("setup.py"),
                root / "quantization/fp4/nvfp4_quant_kernels.cu")
    missing = [path.as_posix() for path in required if not (source / path).is_file()]
    if missing:
        raise RuntimeError(f"Incomplete CUDA source layout; missing {missing}. Review this release.")
    return root


def repair_fp4_dependent_type(source, manifest, source_root):
    """Keep discarded pack16 member accesses dependent on the kernel template."""
    relative = source_root / "quantization/fp4/nvfp4_quant_kernels.cu"
    target = source / relative
    if not target.exists():
        return
    original = target.read_text(encoding="utf-8")
    existing = "      fp4_packed_t out_val{};"
    replacement = """      // Keep the type dependent: CUDA 12.8 uses uint32_t, which has no hi/lo.
      using QuantizedType = decltype(
          cvt_warp_fp16_to_fp4<Type, CVT_FP4_NUM_THREADS_PER_SF, UE8M0_SF>(
              in_vec, global_scale, sf_out));
      QuantizedType out_val{};"""
    if replacement not in original:
        if existing not in original:
            return
        if original.count(existing) != 1:
            raise RuntimeError("Unexpected NVFP4 output declaration; review this release")
        target.write_text(original.replace(existing, replacement), encoding="utf-8", newline="\n")
    manifest["patches"].append({
        "id": "cuda128-nvfp4-dependent-output-type-v1",
        "source": relative.as_posix(),
        "patched_source_sha256": digest(target.read_bytes()),
        "description": "Defer pack16 member lookup until template instantiation; preserve zero initialization and both output layouts.",
    })


def repair_generator_checks(source, manifest, source_root):
    """Build cache survives failed Docker RUNs, generated source files do not."""
    cmake = source / "CMakeLists.txt"
    if not cmake.exists():
        return
    original = cmake.read_text(encoding="utf-8")
    patched = original
    generators = {
        "MACHETE_GEN_SCRIPT_HASH": "quantization/machete/generated/machete_mm_dispatch.cu",
        "MARLIN_GEN_SCRIPT_HASH_AND_ARCH": "quantization/marlin/kernel_selector.h",
        "MOE_MARLIN_GEN_SCRIPT_HASH_AND_ARCH": "moe/marlin_moe_wna16/kernel_selector.h",
    }
    guarded = []
    for variable, sentinel in generators.items():
        sentinel = (source_root / sentinel).as_posix()
        if variable not in original:
            continue
        existing = f"OR NOT $CACHE{{{variable}}} STREQUAL ${{{variable}}})"
        replacement = existing[:-1] + f'\n          OR NOT EXISTS "${{CMAKE_CURRENT_SOURCE_DIR}}/{sentinel}")'
        if replacement not in patched:
            if patched.count(existing) != 1:
                raise RuntimeError(f"Unexpected {variable} generation condition; review this release")
            patched = patched.replace(existing, replacement)
        guarded.append(sentinel)
    if guarded:
        if patched != original:
            cmake.write_text(patched, encoding="utf-8", newline="\n")
        manifest["patches"].append({
            "id": "regenerate-missing-cuda-sources-v1",
            "source": "CMakeLists.txt",
            "patched_source_sha256": digest(cmake.read_bytes()),
            "required_generated_files": guarded,
            "description": "Regenerate Machete/Marlin/MoE sources when a new Docker layer lacks cached generation outputs.",
        })


def repair_deepgemm_cuda_gate(source, manifest):
    """Omit the optional DeepGEMM backend on the CUDA 12.8 image."""
    cmake = source / "CMakeLists.txt"
    setup = source / "setup.py"
    if not cmake.exists() or not setup.exists():
        return
    cmake_text = cmake.read_text(encoding="utf-8")
    setup_text = setup.read_text(encoding="utf-8")
    include = "    include(cmake/external_projects/deepgemm.cmake)"
    extension = '        ext_modules.append(CMakeExtension(name="vllm._deep_gemm_C", optional=True))'
    cmake_guard = '''    # CUDA 12.8 image: omit optional DeepGEMM (v0.31 requires Runtime 12.9).
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.9)
        include(cmake/external_projects/deepgemm.cmake)
    else()
        message(STATUS "CUDA 12.8: omitting optional DeepGEMM backend")
    endif()'''
    setup_guard = '''        # CUDA 12.8 image: DeepGEMM is omitted; keep Qutlass enabled.
        if envs.VLLM_USE_PRECOMPILED or (
            CUDA_HOME and get_nvcc_cuda_version() >= Version("12.9")
        ):
            ext_modules.append(CMakeExtension(name="vllm._deep_gemm_C", optional=True))'''
    if "vllm._deep_gemm_C" not in setup_text and "deepgemm.cmake" not in cmake_text:
        return
    if cmake_guard not in cmake_text:
        if cmake_text.count(include) != 1:
            raise RuntimeError("Unexpected DeepGEMM CMake include; review this release")
        cmake_text = cmake_text.replace(include, cmake_guard)
    if setup_guard not in setup_text:
        if setup_text.count(extension) != 1:
            raise RuntimeError("Unexpected DeepGEMM packaging extension; review this release")
        setup_text = setup_text.replace(extension, setup_guard)
        setup_text = setup_text.replace(
            "# DeepGEMM requires CUDA 12.3+ (SM90/SM100)",
            "# Qutlass remains available on CUDA 12.3+ (SM90/SM100)",
        )
    cmake.write_text(cmake_text, encoding="utf-8", newline="\n")
    setup.write_text(setup_text, encoding="utf-8", newline="\n")
    # Report final hashes when multiple patches modify the same source file.
    cmake_hash = digest(cmake.read_bytes())
    for patch in manifest["patches"]:
        if patch["source"] == "CMakeLists.txt":
            patch["patched_source_sha256"] = cmake_hash
    manifest["patches"].append({
        "id": "cuda128-omit-deepgemm-backend-v1",
        "source": "CMakeLists.txt",
        "patched_source_sha256": cmake_hash,
        "packaging_source": "setup.py",
        "packaging_source_sha256": digest(setup.read_bytes()),
        "minimum_cuda": "12.9",
        "omitted_on_cuda128": "DeepGEMM",
        "description": "Gate both CMake and wheel packaging; preserve the dependency's CUDA 12.9 assertion and keep Qutlass enabled.",
    })


def apply(source, report):
    source_root = detect_cuda_source_root(source)
    relative_header = source_root / "cooperative_topk.cuh"
    target = source / relative_header
    manifest = {"cuda_source_root": source_root.as_posix(), "patches": []}
    if not target.exists():
        raise RuntimeError(f"Missing {relative_header}; review this release before patching")
    else:
        original = target.read_text(encoding="utf-8")
        substitutions = (
            (r"while\s*\(!cuda::ptx::mbarrier_try_wait_parity\(cuda::ptx::sem_relaxed,\s*cuda::ptx::scope_cta,\s*a,\s*p\)\);",
             "while (!vllm::cu128_ptx::mbarrier_try_wait_parity(a, p));"),
            (r"cuda::ptx::mbarrier_arrive_expect_tx\(cuda::ptx::sem_relaxed,\s*cuda::ptx::scope_cta,\s*cuda::ptx::space_shared,\s*a,\s*t\);",
             "vllm::cu128_ptx::mbarrier_arrive_expect_tx(a, t);"),
            (r"cuda::ptx::cp_async_bulk\(cuda::ptx::space_shared,\s*cuda::ptx::space_global,\s*d,\s*s,\s*n,\s*m\);",
             "vllm::cu128_ptx::cp_async_bulk(d, s, n, m);"),
        )
        counts = [len(re.findall(pattern, original)) for pattern, _ in substitutions]
        if INCLUDE not in original and counts == [0, 0, 0]:
            manifest["note"] = "Known missing-wrapper call sites are absent; no patch applied."
        else:
            if INCLUDE in original:
                if any(original.count(replacement) != 1 for _, replacement in substitutions):
                    raise RuntimeError("Incomplete existing CUDA 12.8 patch; source needs review")
                patched = original
            else:
                if counts != [1, 1, 1]:
                    raise RuntimeError(f"Unexpected cooperative_topk source ({counts}); review this release before patching")
                anchor = '#include "topk_histogram_4096.cuh"'
                if original.count(anchor) != 1:
                    raise RuntimeError("Cannot locate cooperative_topk include anchor")
                patched = original.replace(anchor, anchor + "\n" + INCLUDE)
                for pattern, replacement in substitutions:
                    patched = re.sub(
                        pattern,
                        lambda match, replacement=replacement: (
                            "\n" + GUARD + "\n  " + replacement + "\n#else\n  "
                            + match.group(0) + "\n#endif"
                        ),
                        patched,
                        count=1,
                    )
            helper = Path(__file__).with_name(HELPER_NAME).read_bytes()
            target.with_name(HELPER_NAME).write_bytes(helper)
            if patched != original:
                target.write_text(patched, encoding="utf-8", newline="\n")
            manifest["patches"].append({
                "id": PATCH_ID,
                "source": relative_header.as_posix(),
                "patched_source_sha256": digest(target.read_bytes()),
                "helper": HELPER_NAME,
                "helper_sha256": digest(helper),
                "cuda_range": ">=12.8,<12.9",
                "description": "Emit the original relaxed CTA barrier and shared::cta bulk-copy PTX instructions directly.",
            })
    repair_generator_checks(source, manifest, source_root)
    repair_fp4_dependent_type(source, manifest, source_root)
    repair_deepgemm_cuda_gate(source, manifest)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()
    apply(arguments.source, arguments.report)
