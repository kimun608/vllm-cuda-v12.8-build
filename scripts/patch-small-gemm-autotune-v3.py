"""Opt-in small-capture GEMM policy for two structurally supported vLLM files.

Default mode checks the source and records a proposed patch without modifying it.
Apply in an image build, or use --output-copy for a local review fixture. No vLLM,
Torch, CUDA, GPU, Docker or service code is imported or executed by this helper.
The opt-in runtime branch defaults vLLM's global autotune/coordinate search off
without overriding explicit environment values. These environment defaults are
process-wide and persist after that branch has run; use a fresh process to reset.
"""
import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import tempfile


PATCH_ID = "lottecard-small-capture-gemm-v3"
ENV_NAME = "LOTTECARD_GEMM_AUTOTUNE_SIZES"
SMALL_SENTINEL = "lottecard_small_cudagraph_capture_sizes_v3"
LIMIT = 16
POLICY = '''        # lottecard-small-capture-gemm-v3: explicit sizes take precedence.
        if (
            os.getenv("LOTTECARD_GEMM_AUTOTUNE_SIZES", "0") == "1"
            and self.compile_sizes is None
            and self.mode in (None, CompilationMode.VLLM_COMPILE)
            and self.backend in ("", "inductor")
        ):
            self.compile_sizes = ["lottecard_small_cudagraph_capture_sizes_v3"]
            os.environ.setdefault("VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE", "0")
            os.environ.setdefault("VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING", "0")

'''
RESOLVER_POLICY = '''                    # lottecard-small-capture-gemm-v3: actual captures only, at most16.
                    if x == "lottecard_small_cudagraph_capture_sizes_v3":
                        computed_compile_sizes.extend(
                            size for size in self.cudagraph_capture_sizes if 1 <= size <= 16
                        )
                        continue
'''
COMPILER_POLICY = '''    # lottecard-small-capture-gemm-v3: range-local options, no global Torch mutation.
    if (
        os.getenv("LOTTECARD_GEMM_AUTOTUNE_SIZES", "0") == "1"
        and compile_range.is_single_size()
        and 1 <= compile_range.start <= 16
        and os.getenv("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM", "1") == "1"
    ):
        config.setdefault("max_autotune_gemm", True)
        if config["max_autotune_gemm"]:
            config.setdefault(
                "max_autotune_gemm_backends",
                os.getenv("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS", "ATEN,TRITON"),
            )
'''
CACHE_POLICY = '''    # lottecard-small-capture-gemm-v3: segregate opt-in backend caches even with explicit sizes.
    if os.getenv("LOTTECARD_GEMM_AUTOTUNE_SIZES", "0") == "1":
        factors.append(("lottecard-small-capture-gemm-v3", 16))
'''


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def one(values, description):
    if len(values) != 1:
        raise ValueError(f"Unsupported source: expected exactly one {description}, found {len(values)}")
    return values[0]


def config_class(tree):
    return one([node for node in tree.body if isinstance(node, ast.ClassDef)
                and node.name == "CompilationConfig"], "CompilationConfig class")


def post_init(cls):
    node = one([node for node in cls.body if isinstance(node, ast.FunctionDef)
                and node.name == "__post_init__"], "CompilationConfig.__post_init__ method")
    if (node.decorator_list or node.args.posonlyargs or len(node.args.args) != 1
            or node.args.args[0].arg != "self" or node.args.vararg or node.args.kwarg
            or node.args.kwonlyargs or node.args.defaults):
        raise ValueError("Unsupported source: __post_init__ must be an ordinary self-only method")
    return node


def has_docstring(method):
    return bool(method.body and isinstance(method.body[0], ast.Expr)
                and isinstance(method.body[0].value, ast.Constant)
                and isinstance(method.body[0].value.value, str))


def resolver_string_branch(cls):
    resolver = one([node for node in cls.body if isinstance(node, ast.FunctionDef)
                    and node.name == "post_init_cudagraph_sizes"], "capture-size resolver")
    loops = [node for node in ast.walk(resolver) if isinstance(node, ast.For)
             and isinstance(node.target, ast.Name) and node.target.id == "x"
             and ast.dump(node.iter) == ast.dump(ast.parse("self.compile_sizes", mode="eval").body)]
    loop = one(loops, "compile-size x loop")
    expected = ast.dump(ast.parse("isinstance(x, str)", mode="eval").body)
    branch = one([node for node in loop.body if isinstance(node, ast.If)
                  and ast.dump(node.test) == expected], "string-size branch")
    if not branch.body or not isinstance(branch.body[0], ast.Assert):
        raise ValueError("Unsupported source: string-size branch must start with the original sentinel assertion")
    return resolver, branch


def prepare_compilation_patch(raw):
    bom = raw.startswith(b"\xef\xbb\xbf")
    source = raw.decode("utf-8-sig")
    if PATCH_ID in source or ENV_NAME in source:
        raise ValueError("Source already contains the patch marker/environment name; refusing an ambiguous or duplicate patch")
    tree = ast.parse(source)
    cls = config_class(tree)
    method = post_init(cls)
    sizes_field = one([node for node in cls.body if isinstance(node, ast.AnnAssign)
                       and isinstance(node.target, ast.Name) and node.target.id == "compile_sizes"],
                      "annotated compile_sizes field")
    if not isinstance(sizes_field.value, ast.Constant) or sizes_field.value.value is not None:
        raise ValueError("Unsupported source: compile_sizes must default to None")
    for field in ("mode", "backend"):
        one([node for node in cls.body if isinstance(node, ast.AnnAssign)
             and isinstance(node.target, ast.Name) and node.target.id == field], f"{field} field")
    mode_class = one([node for node in tree.body if isinstance(node, ast.ClassDef)
                      and node.name == "CompilationMode"], "CompilationMode enum")
    enum_modes = [node for node in mode_class.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "VLLM_COMPILE" for target in node.targets)]
    enum_value = one(enum_modes, "VLLM_COMPILE enum member").value
    if not isinstance(enum_value, ast.Constant) or enum_value.value != 3:
        raise ValueError("Unsupported source: VLLM_COMPILE must retain the reviewed enum value3")
    resolver = one([node for node in cls.body if isinstance(node, ast.FunctionDef)
                    and node.name == "post_init_cudagraph_sizes"], "capture-size resolver")
    if not any(isinstance(node, ast.Constant) and node.value == "cudagraph_capture_sizes"
               for node in ast.walk(resolver)):
        raise ValueError("Unsupported source: capture-size sentinel is not recognized")
    resolver_dump = ast.dump(resolver)
    if "attr='extend'" not in resolver_dump or "attr='cudagraph_capture_sizes'" not in resolver_dump:
        raise ValueError("Unsupported source: capture-size resolver no longer extends the capture-size list")
    _, string_branch = resolver_string_branch(cls)
    os_imports = [node for node in tree.body if isinstance(node, ast.Import)
                  and any(alias.name == "os" and alias.asname in (None, "os") for alias in node.names)]
    if any(isinstance(node, ast.Import) and any(alias.asname == "os" and alias.name != "os" for alias in node.names)
           for node in tree.body):
        raise ValueError("Unsupported source: module name os is rebound by another import")
    if len(os_imports) > 1:
        raise ValueError("Unsupported source: duplicate top-level os imports")
    lines = source.splitlines(keepends=True)
    signature = lines[method.lineno - 1].rstrip("\r\n")
    if not re.fullmatch(r"    def __post_init__\(self\)(?: -> None)?:", signature):
        raise ValueError("Unsupported source: post-init signature/layout changed")
    newline = "\r\n" if "\r\n" in source else "\n"
    insertion_line = method.body[0].end_lineno if has_docstring(method) else method.lineno
    edits = [(insertion_line, POLICY.replace("\n", newline)),
             (string_branch.lineno, RESOLVER_POLICY.replace("\n", newline))]
    added_os_import = not os_imports
    if added_os_import:
        anchor = one([node for node in tree.body if isinstance(node, ast.Import)
                      and any(alias.name == "enum" and alias.asname is None for alias in node.names)],
                     "ordinary enum import anchor")
        edits.append((anchor.end_lineno, "import os" + newline))
    for line, text in sorted(edits, reverse=True):
        lines.insert(line, text)
    patched = "".join(lines)
    modified_tree = ast.parse(patched)
    modified_method = post_init(config_class(modified_tree))
    policy_position = 1 if has_docstring(modified_method) else 0
    expected_if = ast.parse("def policy(self):\n" + POLICY).body[0].body[0]
    if ast.dump(modified_method.body[policy_position]) != ast.dump(expected_if):
        raise ValueError("Internal verification failed: inserted policy AST mismatch")
    modified_resolver = one([node for node in config_class(modified_tree).body
                             if isinstance(node, ast.FunctionDef) and node.name == "post_init_cudagraph_sizes"], "modified resolver")
    expected_resolver_if = ast.parse("def resolver(self):\n    for x in []:\n" + "\n".join(line[12:] for line in RESOLVER_POLICY.splitlines()) + "\n").body[0].body[0].body[0]
    inserted_resolver_if = one([node for node in ast.walk(modified_resolver) if isinstance(node, ast.If)
                                and ast.dump(node.test) == ast.dump(expected_resolver_if.test)], "inserted small sentinel branch")
    if ast.dump(inserted_resolver_if) != ast.dump(expected_resolver_if):
        raise ValueError("Internal verification failed: resolver policy AST mismatch")
    cleaned = copy.deepcopy(modified_tree)
    cleaned_method = post_init(config_class(cleaned))
    del cleaned_method.body[policy_position]
    clean_resolver = one([node for node in config_class(cleaned).body if isinstance(node, ast.FunctionDef)
                          and node.name == "post_init_cudagraph_sizes"], "cleaned resolver")
    for node in ast.walk(clean_resolver):
        if isinstance(node, ast.If) and inserted_resolver_if in node.body:
            node.body.remove(inserted_resolver_if)
            break
    else:
        # deepcopy changes identity; compare the unique sentinel branch by AST.
        found = False
        for node in ast.walk(clean_resolver):
            if isinstance(node, ast.If):
                matches = [child for child in node.body if ast.dump(child) == ast.dump(expected_resolver_if)]
                if matches:
                    node.body.remove(one(matches, "cleaned inserted small sentinel branch"))
                    found = True
                    break
        if not found:
            raise ValueError("Internal verification failed: inserted resolver policy not found for removal")
    if added_os_import:
        inserted_import = one([node for node in cleaned.body if isinstance(node, ast.Import)
                               and len(node.names) == 1 and node.names[0].name == "os"
                               and node.names[0].asname is None], "inserted os import")
        cleaned.body.remove(inserted_import)
    if ast.dump(cleaned) != ast.dump(tree):
        raise ValueError("Internal verification failed: AST changed outside the reviewed import/policy")
    compile(patched, "<patched-vllm-compilation>", "exec")
    output = patched.encode("utf-8-sig" if bom else "utf-8")
    return output, {"patch_id": PATCH_ID, "input_sha256": sha256(raw), "output_sha256": sha256(output),
                    "input_bytes": len(raw), "output_bytes": len(output), "added_os_import": added_os_import,
                    "newline": "CRLF" if newline == "\r\n" else "LF", "ast_only_reviewed_delta": True,
                    "source_post_init_line": method.lineno, "small_size_limit": LIMIT,
                    "internal_sentinel": SMALL_SENTINEL,
                    "runtime_conditions": {"environment": ENV_NAME + "=1", "compile_sizes": "None only",
                                           "mode": ["None (deferred automatic default)", "CompilationMode.VLLM_COMPILE"],
                                           "backend": ["", "inductor"]},
                    "branch_environment_defaults": {"VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE": "0",
                                                    "VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING": "0"},
                    "environment_default_method": "os.environ.setdefault; explicit existing values are preserved",
                    "preserved_overrides": ["explicit []", "explicit custom compile sizes", "explicit unsupported mode/backend (no-op)",
                                            "existing VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE value",
                                            "existing VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING value"],
                    "limits": ["The guard uses mode/backend as currently set at post-init; later automatic normalization is outside this patch.",
                               "The internal sentinel resolves only to actual later capture sizes1..16, possibly empty. The original full-list sentinel retains its behavior. Record final resolved compile_sizes.",
                               "The paired compiler patch opts in only small single-size local GEMM options; no global Torch GEMM ENV is introduced.",
                               "The two vLLM tuning defaults are changed only when this guard applies; explicit settings remain authoritative.",
                               "Environment defaults are process-wide once set. Later disabling the opt-in flag or choosing explicit sizes does not restore them. Use a fresh process to reset.",
                               "This bounds two autotune settings, not total startup time or the number of GEMM candidates."]}


def verify_fresh_copy_calls(tree):
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "set_inductor_config"]
    if len(calls) != 2:
        raise ValueError("Unsupported source: expected exactly two set_inductor_config call sites")
    recognized = []
    for cls in [node for node in tree.body if isinstance(node, ast.ClassDef)]:
        for method in [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "compile"]:
            for index, statement in enumerate(method.body):
                if not (isinstance(statement, ast.Expr) and statement.value in calls):
                    continue
                call = statement.value
                if (len(call.args) != 2 or call.keywords or
                        [ast.dump(arg) for arg in call.args] != [ast.dump(ast.Name(id=name, ctx=ast.Load())) for name in ("current_config", "compile_range")]):
                    raise ValueError("Unsupported source: unexpected per-range config call arguments")
                prefix = method.body[:index]
                copies = [i for i, node in enumerate(prefix) if isinstance(node, ast.Assign)
                          and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                          and node.targets[0].id == "current_config" and isinstance(node.value, ast.Dict)
                          and not node.value.keys]
                start = one(copies, "fresh current_config dictionary assignment")
                expected_update = ast.parse("if compiler_config is not None:\n    current_config.update(compiler_config)").body[0]
                if ast.dump(prefix[start + 1]) != ast.dump(expected_update):
                    raise ValueError("Unsupported source: compiler_config copy/update contract changed")
                for node in prefix[start + 2:]:
                    for nested in ast.walk(node):
                        if isinstance(nested, ast.Name) and nested.id == "current_config" and isinstance(nested.ctx, ast.Store):
                            raise ValueError("Unsupported source: current_config rebound after its fresh copy")
                recognized.append({"class": cls.name, "method": method.name,
                                   "fresh_dict_line": prefix[start].lineno,
                                   "base_copy_line": prefix[start + 1].lineno,
                                   "set_config_line": statement.lineno})
    if len(recognized) != len(calls):
        raise ValueError("Unsupported source: unreviewed set_inductor_config caller")
    return recognized


def top_function(tree, name):
    return one([node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == name], name + " top-level function")


def prepare_compiler_patch(raw):
    bom = raw.startswith(b"\xef\xbb\xbf")
    source = raw.decode("utf-8-sig")
    if PATCH_ID in source or ENV_NAME in source:
        raise ValueError("Compiler source already contains this policy; refusing duplicate/ambiguous patch")
    tree = ast.parse(source)
    function = top_function(tree, "set_inductor_config")
    expected_body = ast.parse('''def reviewed(config, compile_range):
    if compile_range.is_single_size():
        config["max_autotune"] = envs.VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE
        config["coordinate_descent_tuning"] = envs.VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING
''').body[0].body
    if [ast.dump(node) for node in function.body] != [ast.dump(node) for node in expected_body]:
        raise ValueError("Unsupported source: original set_inductor_config body changed")
    if function.decorator_list or [arg.arg for arg in function.args.args] != ["config", "compile_range"]:
        raise ValueError("Unsupported source: set_inductor_config signature/decorators changed")
    one([node for node in tree.body if isinstance(node, ast.Import)
         and len(node.names) == 1 and node.names[0].name == "os" and node.names[0].asname is None], "compiler os import")
    if any(isinstance(node, ast.Import) and any(alias.asname == "os" and alias.name != "os" for alias in node.names)
           for node in tree.body):
        raise ValueError("Unsupported source: compiler os name is rebound by another import")
    call_sites = verify_fresh_copy_calls(tree)
    factors = top_function(tree, "get_inductor_factors")
    if (factors.decorator_list or factors.args.args or not isinstance(factors.body[-1], ast.Return)
            or ast.dump(factors.body[-1].value) != ast.dump(ast.Name(id="factors", ctx=ast.Load()))
            or sum(isinstance(node, ast.Return) for node in ast.walk(factors)) != 1):
        raise ValueError("Unsupported source: factor function final-return contract changed")
    newline = "\r\n" if "\r\n" in source else "\n"
    lines = source.splitlines(keepends=True)
    edits = [(function.end_lineno, ("\n" + COMPILER_POLICY).replace("\n", newline)),
             (factors.body[-1].lineno - 1, CACHE_POLICY.replace("\n", newline))]
    for line, text in sorted(edits, reverse=True):
        lines.insert(line, text)
    patched = "".join(lines)
    modified = ast.parse(patched)
    expected_compiler = ast.parse("def policy(config, compile_range):\n" + COMPILER_POLICY).body[0].body[0]
    expected_cache = ast.parse("def policy():\n" + CACHE_POLICY).body[0].body[0]
    if ast.dump(top_function(modified, "set_inductor_config").body[-1]) != ast.dump(expected_compiler):
        raise ValueError("Internal verification failed: compiler policy AST mismatch")
    if ast.dump(top_function(modified, "get_inductor_factors").body[-2]) != ast.dump(expected_cache):
        raise ValueError("Internal verification failed: cache policy AST mismatch")
    cleaned = copy.deepcopy(modified)
    del top_function(cleaned, "set_inductor_config").body[-1]
    del top_function(cleaned, "get_inductor_factors").body[-2]
    if ast.dump(cleaned) != ast.dump(tree):
        raise ValueError("Internal verification failed: compiler AST changed outside two reviewed policy blocks")
    compile(patched, "<patched-vllm-compiler-interface>", "exec")
    output = patched.encode("utf-8-sig" if bom else "utf-8")
    return output, {"input_sha256": sha256(raw), "output_sha256": sha256(output),
                    "input_bytes": len(raw), "output_bytes": len(output),
                    "ast_only_reviewed_delta": True, "fresh_copy_call_sites": call_sites,
                    "range_condition": "flag1 + single range + start1..16 + Torch GEMM ENV absent/1",
                    "explicit_option_precedence": "Existing compiler_config keys are preserved with setdefault; Torch GEMM ENV0 disables helper opt-in, not an explicit upstream compiler_config True.",
                    "cache_key": "Append (patch_id,16) to Inductor factors only when flag1; flag0 leaves the original factors unchanged.",
                    "limits": ["Exactly two recognized source callers allocate a fresh config dictionary per range. Direct reuse of a mutated dictionary across ranges is unsupported; no private state/bookkeeping key is added.",
                               "Explicit compile-size lists/full-list sentinel remain unchanged, but flag1 still opts their small single ranges into the local policy unless compiler_config/ENV explicitly overrides it.",
                               "Large range option dictionaries retain baseline behavior; capture-size specialization still changes dispatch/range structure."]}


def atomic_write(path, raw, preserve_mode=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode if preserve_mode and path.exists() else None
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as file:
        temporary = Path(file.name)
        file.write(raw)
    try:
        if mode is not None:
            os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["check", "apply"], default="check")
    parser.add_argument("--compilation-source", required=True, type=Path)
    parser.add_argument("--compiler-source", required=True, type=Path)
    parser.add_argument("--output-directory", type=Path,
                        help="Apply to two new review copies here instead of installed source files")
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--expected-compilation-sha256", default="")
    parser.add_argument("--expected-compiler-sha256", default="")
    args = parser.parse_args()
    if args.output_directory and args.mode != "apply":
        parser.error("--output-directory requires --mode apply")
    inputs = [("compilation", args.compilation_source, args.expected_compilation_sha256, prepare_compilation_patch),
              ("compiler_interface", args.compiler_source, args.expected_compiler_sha256, prepare_compiler_patch)]
    sources = {path.resolve() for _, path, _, _ in inputs}
    if len(sources) != 2:
        raise ValueError("The two input source paths must be distinct")
    prepared = []
    for name, path, expected_sha, prepare in inputs:
        if expected_sha and not re.fullmatch("[a-fA-F0-9]{64}", expected_sha):
            parser.error("Expected source SHA256 must be64 hexadecimal characters")
        original = path.read_bytes()
        if expected_sha and sha256(original) != expected_sha.lower():
            raise ValueError(f"Installed {name} source SHA256 differs from the explicitly pinned source")
        patched, details = prepare(original)
        target = args.output_directory / (name + ".patched.py") if args.output_directory else path
        if args.output_directory and (target.resolve() in sources or target.exists()):
            raise ValueError("Output copies must be new paths distinct from both input source files")
        details.update(source_path=str(path.resolve()), output_path=str(target.resolve()),
                       expected_source_sha256=expected_sha.lower() or None)
        prepared.append((name, path, target, original, patched, details))
    targets = {item[2].resolve() for item in prepared}
    if len(targets) != 2 or args.metadata.resolve() in sources | targets or args.metadata.exists():
        raise ValueError("Metadata must be a new path distinct from source/output")
    try:
        distribution_version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        distribution_version = None
    metadata = {"patch_id": PATCH_ID, "mode": args.mode,
                "status": "checked" if args.mode == "check" else "applied",
                "checked_utc": datetime.now(timezone.utc).isoformat(),
                "distribution_version": distribution_version,
                "helper_sha256": sha256(Path(__file__).read_bytes()),
                "files": {item[0]: item[-1] for item in prepared},
                "small_size_limit": LIMIT, "internal_sentinel": SMALL_SENTINEL,
                "image_environment_requirement": ENV_NAME + "=1 only; no global Torch GEMM/backends ENV needed",
                "baseline_scope": "Non-small/dynamic local Inductor option dictionaries retain the original function behavior. Whole dispatch and high-concurrency requests can still execute small shapes.",
                "process_environment_limit": "Guarded vLLM wide/coordinate defaults persist once set; existing/cached environment values are not retroactively restored/changed."}
    if args.mode == "apply":
        written = []
        try:
            for _, path, target, original, patched, details in prepared:
                atomic_write(target, patched, preserve_mode=not args.output_directory)
                written.append((target, original))
                if sha256(target.read_bytes()) != details["output_sha256"]:
                    raise RuntimeError("Output source hash verification failed")
        except Exception:
            if not args.output_directory:
                for target, original in reversed(written):
                    atomic_write(target, original, preserve_mode=True)
            raise
    atomic_write(args.metadata, (json.dumps(metadata, indent=2) + "\n").encode())
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
