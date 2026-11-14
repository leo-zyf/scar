"""C++20 temporal execution of one shared, shape-preserving scalar body DAG.

An ordinary body ExecutionPlan is input to lowering, never a claim that this
artifact is a one-step operator. The original loop, nested callable guards and
exact positive trip count remain separate executable contracts.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

from scar.compiler.loop_model import document, loop_digest, loop_from_dict, read_loop
from scar.compiler.model import document as plan_document, plan_from_dict, read_plan
from scar.optimization.hardware import bind_cpu_placement
from scar.optimization.resource_solver import semantic_sha256
from . import cpp

SCHEMA = "scar.native-array-loop-artifact"
GuardFailure = cpp.GuardFailure
_DIRECTORY = Path(__file__).resolve().parent
_RUN = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(cpp._Array), ctypes.c_uint64,
                       ctypes.POINTER(cpp._Array))


class _Table(ctypes.Structure):
    _fields_ = [("abi_version", ctypes.c_uint32), ("struct_size", ctypes.c_uint32),
                ("trip_count", ctypes.c_uint64), ("run", _RUN),
                ("loop_sha256", ctypes.c_char_p), ("strategy_sha256", ctypes.c_char_p),
                ("execution_plan_sha256", ctypes.c_char_p), ("contract_sha256", ctypes.c_char_p)]


def _digest(raw):
    return sha256(raw).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def loop_implementation_sha256():
    files = [{"path": name, "sha256": _digest((_DIRECTORY / name).read_bytes())}
             for name in ("loop.py", "loop_v1.cpp.in", "loop_abi_v1.h", "scar_abi_v1.h", "loop_sse2_v1.cpp.in")]
    return semantic_sha256({"files": files, "boundary_implementation_sha256": cpp.runtime_implementation_sha256()})


def layout_compiler_options(plan):
    """The sourced SSE2 layout profile's exact compiler option authority."""
    return (*cpp.COMPILER_OPTIONS, "-fno-unroll-loops", "-fno-peel-loops",
            "-msse2", "-mno-avx", "-mno-fma", *(("-fopenmp",) if plan.cpu_threads > 1 else ()))


def _sources(loop):
    seen = set()
    for source in (loop.source, loop.body.source, *(call.callee.source for call in loop.body.calls)):
        key = (source.path, source.sha256, source.code_key)
        if key in seen:
            continue
        seen.add(key)
        _require(bool(source.code_key) and Path(source.path).is_absolute(), "original loop/body/leaf source versions required")
        _require(_digest(Path(source.path).read_bytes()) == source.sha256, "loop or nested callee source snapshot differs")


def _admit(loop, strategy, execution_plan):
    from scar.optimization.loop_strategy import verify_loop_strategy, loop_execution_plan
    from scar.optimization.pipeline_fusion import fuse_pipeline_region, verify_pipeline_fusion
    strategy = verify_loop_strategy(loop, strategy)
    _require(strategy["selected"]["kind"] == "TEMPORAL_FUSION" and
             strategy["capability"] == "EXACT_ELEMENTWISE_TEMPORAL_FUSION",
             "loop backend requires the selected exact elementwise temporal profile")
    raw, _ = fuse_pipeline_region(loop.body)
    _require(raw is not None, "loop body fusion has unresolved source obligations")
    verify_pipeline_fusion(loop.body, raw, strategy["body_fusion_certificate"])
    expected = loop_execution_plan(loop, strategy)
    actual = read_plan(execution_plan) if isinstance(execution_plan, (str, Path)) else plan_from_dict(plan_document(execution_plan))
    _require(actual.resource_binding is not None, "loop execution requires an explicit recorded CPU binding")
    _require(sys.platform == "linux", "CPU-bound loop execution requires Linux")
    cpp._supported(expected); cpp._supported(actual)
    _require(plan_document(actual) == plan_document(bind_cpu_placement(expected, actual.resource_binding)),
             "loop body plan differs from exact replay or admitted CPU binding")
    _require(actual.region.source == loop.body.source and actual.region.parameters == loop.body.parameters
             and len(actual.region.parameters) == 1 and
             all(node.op in ("input", "const", "add", "sub", "mul", "div", "neg") for node in actual.region.nodes),
             "loop body must be the shared one-array elementwise DAG")
    count = strategy["trip_count"]
    _require(type(count) is int and 0 < count <= 2**63 - 1, "loop count must be an exact positive int64 specialization")
    memory = strategy["selected_memory_plan"]
    shape = memory["input_shapes"][loop.state_parameter]
    _require(set(memory["input_shapes"]) == {loop.state_parameter} and len(shape) == 2 and
             all(type(n) is int and 0 <= n <= sys.maxsize // 8 for n in shape), "loop state shape exceeds host bounds")
    size = shape[0] * shape[1] * 8
    budget = actual.cost.get("memory_budget_bytes")
    _require(memory["output_shape"] == shape and memory["final_bytes"] == size <= sys.maxsize and
             memory["total_workspace_bytes"] == 0 and memory["peak_allocation_bytes"] == size and
             type(budget) is int and size <= budget <= memory["memory_budget_bytes"] <= sys.maxsize,
             "loop requires a shape-preserving fresh final within the narrower payload budget")
    return strategy, actual


def _admit_layout(loop, strategy, plan, layout):
    if layout is None:
        return None
    from scar.optimization.loop_layout import verify_loop_layout
    result = verify_loop_layout(loop, strategy, plan, layout)
    _require(result["status"] == "PLANNED_UNDER_MODEL" and isinstance(result["selected"], dict),
             "loop layout has unresolved implementation/model obligations")
    selected = result["selected"]
    _require(platform.machine().lower() in ("x86_64", "amd64") and ctypes.sizeof(ctypes.c_void_p) == 8,
             "SSE2 layout requires a 64-bit x86 execution process")
    count = selected["vector_chains"]
    _require(selected["kind"] == "SSE2_F64_TEMPORAL_BLOCK_V1" and selected["isa"] == "SSE2_X86_64"
             and selected["simd_lanes"] == 2 and type(count) is int and 1 <= count <= 16
             and selected["tile_elements"] == 2 * count
             and 2 * count <= strategy["state_shape"][1]
             and selected["lane_mapping"] == "CONTIGUOUS_ROW_CHAINS" and selected["scalar_tail"] is True,
             "unsupported or unresolved SIMD loop layout")
    return result


def _contract(loop, strategy, plan, layout=None):
    memory = strategy["selected_memory_plan"]
    result = {"schema": "scar.native-loop-contract", "schema_version": 1,
        "loop_sha256": loop_digest(loop), "strategy_sha256": semantic_sha256(strategy),
        "execution_plan_sha256": semantic_sha256(plan_document(plan)), "trip_count": strategy["trip_count"],
        "state_shape": memory["input_shapes"][loop.state_parameter], "output_shape": memory["output_shape"],
        "final_bytes": memory["final_bytes"], "memory_budget_bytes": plan.cost["memory_budget_bytes"]}
    if layout is not None:
        result.update(schema_version=2, layout_sha256=semantic_sha256(layout))
    return result


def _layout_compiler(layout, compiler, identity):
    if layout is None: return
    receipt = layout["hardware_profile_receipt"]
    profile = json.loads(Path(receipt["path"]).read_bytes()) if receipt is not None else layout["hardware_profile"]
    expected = profile["context"]["compiler"]
    _require(compiler == expected["argv"] and identity == expected["identity"],
             "layout compiler prefix/identity differs from sourced primitive context")


def _packed_rows(plan, layout, scalar_body, scalar_result):
    chains = layout["selected"]["vector_chains"]
    constants, expressions, names = [], [], {}
    intrinsic = {"add": "_mm_add_pd", "sub": "_mm_sub_pd", "mul": "_mm_mul_pd", "div": "_mm_div_pd"}
    for index, node in enumerate(plan.region.nodes):
        if node.op == "const":
            name = f"constant{index}"
            constants.append(f"    const __m128d {name} = _mm_set1_pd({float(node.attrs['value']).hex()});")
            names[node.id] = [name] * chains
            continue
        values = []
        for chain in range(chains):
            name = f"packed{index}_{chain}"
            if node.op == "input":
                expression = f"carry{chain}"
            elif node.op == "neg":
                expression = f"_mm_xor_pd({names[node.inputs[0]][chain]}, _mm_set1_pd(-0.0))"
            else:
                expression = f"{intrinsic[node.op]}({names[node.inputs[0]][chain]}, {names[node.inputs[1]][chain]})"
            expressions.append(f"            const __m128d {name} = {expression};")
            values.append(name)
        names[node.id] = values
    values = {
        "SCALAR_BODY": "\n".join(scalar_body), "SCALAR_RESULT": scalar_result,
        "VECTOR_CONSTANTS": "\n".join(constants), "TILE_ELEMENTS": str(2 * chains),
        "VECTOR_LOADS": "\n".join(f"        __m128d carry{c} = _mm_loadu_pd(input + col + {2*c});" for c in range(chains)),
        "VECTOR_BAD": " | ".join(f"nonfinite_pair(carry{c})" for c in range(chains)),
        "VECTOR_BODY": "\n".join(expressions),
        "VECTOR_UPDATES": "\n".join(f"            carry{c} = {names[plan.region.output][c]};" for c in range(chains)),
        "VECTOR_STORES": "\n".join(f"        _mm_storeu_pd(result + col + {2*c}, carry{c});" for c in range(chains)),
    }
    source = (_DIRECTORY / "loop_sse2_v1.cpp.in").read_text()
    for key, value in values.items(): source = source.replace("@" + key + "@", value)
    return source


def _source(loop, strategy, plan, layout=None):
    contract = _contract(loop, strategy, plan, layout)
    names, body = {}, []
    for index, node in enumerate(plan.region.nodes):
        name = names[node.id] = f"value{index}"
        if node.op == "input":
            expression = "carry"
        elif node.op == "const":
            expression = float(node.attrs["value"]).hex()
        elif node.op == "neg":
            expression = "-" + names[node.inputs[0]]
        else:
            expression = names[node.inputs[0]] + " " + cpp._BINARY_OPS[node.op] + " " + names[node.inputs[1]]
        body.append(f"            const double {name} = {expression};")
    binding = plan.resource_binding
    affinity_start = ["    ThreadAffinity caller_affinity;",
                      "    for (int cpu : planned_cpu_ids) if (!caller_affinity.contains(cpu)) return affinity_guard_status;"]
    if plan.cpu_threads == 1:
        affinity_start.append("    if (!caller_affinity.bind(planned_cpu_ids[0])) return affinity_guard_status;")
        loops = ["        uint32_t nonfinite = 0;",
                 "        for (int64_t row = 0; row < modeled_rows; ++row) execute_row(*state, *output, row, steps, nonfinite);",
                 "        floating_guard |= nonfinite;"]
        affinity_finish = ["    const bool kept = caller_affinity.bound();",
                           "    const bool restored = caller_affinity.restore();",
                           "    if (!kept || !restored) return affinity_guard_status;"]
    else:
        loops = [f"        #pragma omp parallel num_threads({plan.cpu_threads}) reduction(|:floating_guard,placement_guard)",
            "        {", "            ThreadAffinity affinity;",
            f"            const bool team_ready = omp_get_num_threads() == {plan.cpu_threads};",
            "            const bool affinity_ready = team_ready && affinity.bind(planned_cpu_ids[omp_get_thread_num()]);",
            "            FloatingScope environment;",
            "            const bool environment_ready = environment.begin(&execution_environment);",
            "            uint32_t nonfinite = 0;", "            #pragma omp for schedule(static)",
            "            for (int64_t row = 0; row < modeled_rows; ++row) {",
            "                if (affinity_ready && environment_ready) execute_row(*state, *output, row, steps, nonfinite);",
            "            }",
            "            const int raised = environment_ready ? std::fetestexcept(floating_flags) : 0;",
            "            const bool environment_restored = environment.restore();",
            "            floating_guard |= static_cast<uint32_t>(!environment_ready || raised || !environment_restored || nonfinite);",
            "            const bool affinity_kept = affinity.bound();",
            "            const bool affinity_restored = affinity.restore();",
            "            placement_guard |= static_cast<uint32_t>(!affinity_ready || !affinity_kept || !affinity_restored);",
            "        }"]
        affinity_finish = ["    const bool restored = caller_affinity.restore_original();",
                           "    if (placement_guard || !restored) return affinity_guard_status;"]
    replacements = {"CPU_BINDING_ENABLED": "1", "CPU_IDS": ", ".join(map(str, binding.cpu_ids)),
        "CPU_SLOTS": str(max(binding.cpu_ids) + 1), "OPENMP_INCLUDE": "#include <omp.h>" if plan.cpu_threads > 1 else "",
        "ROWS": str(contract["state_shape"][0]), "COLS": str(contract["state_shape"][1]),
        "BYTES": str(contract["final_bytes"]), "BUDGET": str(contract["memory_budget_bytes"]),
        "TRIP_COUNT": str(contract["trip_count"]), "BODY": "\n".join(body), "RESULT": names[plan.region.output],
        "LOOPS": "\n".join(loops), "AFFINITY_START": "\n".join(affinity_start),
        "AFFINITY_FINISH": "\n".join(affinity_finish), "LOOP_SHA256": contract["loop_sha256"],
        "STRATEGY_SHA256": contract["strategy_sha256"], "PLAN_SHA256": contract["execution_plan_sha256"],
        "CONTRACT_SHA256": semantic_sha256(contract)}
    source = (_DIRECTORY / "loop_v1.cpp.in").read_text()
    for key, value in replacements.items(): source = source.replace("@" + key + "@", value)
    if layout is not None:
        begin, end = source.index("void execute_row("), source.index("\nint32_t run(")
        source = source[:begin] + _packed_rows(plan, layout, body, names[plan.region.output]) + source[end:]
        guard = ('#if !defined(__x86_64__) || !defined(__SSE2__) || defined(__SSE3__) || '
                 'defined(__SSSE3__) || defined(__SSE4_1__) || defined(__SSE4_2__) || '
                 'defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__FMA__)\n'
                 '#error "loop layout requires the explicit x86_64 SSE2-only compiler profile"\n#endif\n'
                 '#include <emmintrin.h>\n#include <cpuid.h>\n')
        source = source.replace('#include "loop_abi_v1.h"', '#include "loop_abi_v1.h"\n' + guard, 1)
        source = source.replace("    if (steps == 0 || steps != modeled_count)",
                                "    if (!sse2_available()) return 7;\n    if (steps == 0 || steps != modeled_count)", 1)
        if plan.cpu_threads == 1:
            pin = "    if (!caller_affinity.bind(planned_cpu_ids[0])) return affinity_guard_status;"
            source = source.replace(pin, pin + "\n    if (!sse2_available()) return 7;", 1)
        else:
            source = source.replace("    uint32_t placement_guard = 0;", "    uint32_t placement_guard = 0;\n    uint32_t isa_guard = 0;", 1)
            source = source.replace("reduction(|:floating_guard,placement_guard)", "reduction(|:floating_guard,placement_guard,isa_guard)", 1)
            pin = "            const bool affinity_ready = team_ready && affinity.bind(planned_cpu_ids[omp_get_thread_num()]);"
            source = source.replace(pin, pin + "\n            const bool isa_ready = affinity_ready && sse2_available();\n            isa_guard |= static_cast<uint32_t>(!isa_ready);", 1)
            source = source.replace("if (affinity_ready && environment_ready) execute_row", "if (affinity_ready && environment_ready && isa_ready) execute_row", 1)
            source = source.replace("    return floating_guard ?", "    if (isa_guard) return 7;\n    return floating_guard ?", 1)
    return source


def _isa_audit(library):
    """Read compiler output; abstract register counts never prove no spills."""
    command = ["objdump", "-d", "-C", "--no-show-raw-insn", "-M", "intel", str(library)]
    process = subprocess.run(command, capture_output=True, text=True)
    _require(process.returncode == 0, "SSE2 layout requires a readable compiler-output audit")
    packed, stack, dynamic, frames = {}, [], [], []
    for line in process.stdout.splitlines():
        match = re.match(r"^\s*([0-9a-f]+):\s+([a-z0-9]+)\s*(.*)$", line)
        if match is None: continue
        address, operation, operands = match.groups()
        _require(not operation.startswith("v") and not re.match(r"f(madd|msub|nmadd|nmsub)", operation),
                 "compiled layout contains an unsupported AVX/FMA instruction")
        if operation in ("addpd", "subpd", "mulpd", "divpd"):
            packed[operation] = packed.get(operation, 0) + 1
        if "xmm" in operands and re.search(r"\[(?:rsp|rbp)[^\]]*\]", operands):
            stack.append({"address": address, "operation": operation, "operands": operands})
        if operation == "sub" and operands.startswith("rsp,"):
            amount = operands.split(",", 1)[1].strip()
            try: frames.append(int(amount, 0))
            except ValueError: dynamic.append({"address": address, "operands": operands})
    _require(not dynamic, "layout compiler output has an unbounded dynamic stack adjustment")
    return {"schema": "scar.loop-isa-audit", "schema_version": 1,
        "library_sha256": _digest(Path(library).read_bytes()), "isa_profile": "SSE2_X86_64",
        "packed_double_arithmetic_sites": packed, "instruction_counts_are_static_sites": True,
        "xmm_stack_memory_sites": len(stack), "xmm_stack_memory_examples": stack[:8],
        "xmm_stack_memory_sha256": semantic_sha256(stack), "stack_operands_are_possible_spills_not_all_proven_spills": True,
        "maximum_static_rsp_sub_bytes": max(frames, default=0), "sum_static_rsp_sub_sites_bytes": sum(frames),
        "dynamic_rsp_sub_sites": dynamic, "no_compiler_spills_proven": False,
        "scope": "THIS_ELF_STATIC_STACK_ADJUSTMENTS_EXCLUDES_CALLEE_LIBRARIES_CPU_MASK_HEAP_FULL_THREAD_STACK_AND_RSS"}


@dataclass
class LoopArtifact:
    loop: object
    execution_plan: object
    directory: Path
    _strategy: dict
    _metadata: dict
    _library: object
    _table: object
    _run: object
    _layout: dict | None = None

    @property
    def strategy(self): return json.loads(json.dumps(self._strategy))

    @property
    def layout(self): return json.loads(json.dumps(self._layout)) if self._layout is not None else None

    @property
    def report(self):
        return {**json.loads(json.dumps(self._metadata)), "artifact_directory": str(self.directory),
                "source_path": str(self.directory / self._metadata["files"]["source"]),
                "library_path": str(self.directory / self._metadata["files"]["library"]),
                "metadata_path": str(self.directory / "artifact.json")}

    def run(self, state, steps):
        import numpy as np
        if type(steps) is not int or steps != self._strategy["trip_count"] or steps <= 0:
            raise GuardFailure("loop trip count differs from exact specialization")
        state_descriptor = cpp._descriptor(state)
        shape = tuple(self._strategy["selected_memory_plan"]["input_shapes"][self.loop.state_parameter])
        if state.shape != shape or not state.flags.c_contiguous:
            raise GuardFailure("loop shape/layout differs from temporal strategy")
        if state.nbytes > self.execution_plan.cost["memory_budget_bytes"]:
            raise GuardFailure("loop final payload exceeds the narrowed budget")
        try:
            output = np.empty(shape, dtype=np.float64)
        except (MemoryError, ValueError, OverflowError) as error:
            raise GuardFailure("loop fresh final cannot be allocated") from error
        output_descriptor = cpp._descriptor(output)
        status = self._run(ctypes.byref(state_descriptor), steps, ctypes.byref(output_descriptor))
        if status:
            reason = {1: "pointer/rank/bounds/alias", 2: "shape", 3: "floating-point or non-finite carry",
                      4: "CPU affinity", 5: "trip count", 6: "payload budget", 7: "SSE2 ISA"}.get(status, str(status))
            raise GuardFailure("native loop " + reason + " guard")
        return output


def _identity(source, headers, compiler, options, fingerprint):
    return _digest(source + b"".join(headers) + _json_bytes(compiler) + _json_bytes(options) + fingerprint.encode())


def build(loop, out, *, strategy, execution_plan, layout=None):
    loop = read_loop(loop) if isinstance(loop, (str, Path)) else loop_from_dict(document(loop))
    strategy, plan = _admit(loop, strategy, execution_plan)
    layout = _admit_layout(loop, strategy, plan, layout)
    _sources(loop)
    source = _source(loop, strategy, plan, layout).encode()
    headers = [(_DIRECTORY / name).read_bytes() for name in ("loop_abi_v1.h", "scar_abi_v1.h")]
    fingerprint = loop_implementation_sha256()
    compiler = shlex.split(os.environ.get("CXX", "g++"))
    _require(bool(compiler) and shutil.which(compiler[0]), "loop C++20 compiler unavailable")
    version = subprocess.run([*compiler, "--version"], capture_output=True, text=True).stdout.splitlines()
    _layout_compiler(layout, compiler, version[0] if version else "unknown")
    options = list(layout_compiler_options(plan)) if layout is not None else [
        *cpp.COMPILER_OPTIONS, "-fno-unroll-loops", "-fno-peel-loops", *(["-fopenmp"] if plan.cpu_threads > 1 else [])]
    build_id = _identity(source, headers, compiler, options, fingerprint)
    directory = Path(out).resolve()
    _require(not directory.exists(), "fresh native loop output directory required")
    directory.mkdir(parents=True)
    files = {"loop": "loop.json", "strategy": "strategy.json", "execution_plan": "execution-plan.json",
             "source": "loop.cpp", "loop_abi_header": "loop_abi_v1.h", "array_abi_header": "scar_abi_v1.h",
             "library": "loop_" + build_id[:20] + ".so"}
    if layout is not None:
        files.update(layout="layout.json", isa_audit="isa-audit.json")
        (directory / files["layout"]).write_bytes(_json_bytes(layout))
    for key, value in (("loop", document(loop)), ("strategy", strategy), ("execution_plan", plan_document(plan))):
        (directory / files[key]).write_bytes(_json_bytes(value))
    (directory / files["source"]).write_bytes(source)
    for name, raw in zip(("loop_abi_v1.h", "scar_abi_v1.h"), headers): (directory / name).write_bytes(raw)
    fd, temporary = tempfile.mkstemp(prefix=".loop-", suffix=".so", dir=directory); os.close(fd)
    try:
        process = subprocess.run([*compiler, *options, str(directory / files["source"]), "-o", temporary, "-lm"], capture_output=True, text=True)
        if process.returncode: raise RuntimeError("C++20 loop compilation failed:\n" + process.stderr.strip())
        os.replace(temporary, directory / files["library"])
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
    if layout is not None:
        (directory / files["isa_audit"]).write_bytes(_json_bytes(_isa_audit(directory / files["library"])))
    contract = _contract(loop, strategy, plan, layout)
    metadata = {"schema": SCHEMA, "schema_version": 2 if layout is not None else 1, "loop_abi_version": 1, "backend": "cpp",
        "build_id": build_id, "loop_sha256": contract["loop_sha256"], "strategy_sha256": contract["strategy_sha256"],
        "execution_plan_sha256": contract["execution_plan_sha256"], "contract_sha256": semantic_sha256(contract),
        "loop_implementation_sha256": fingerprint, "legacy_boundary_implementation_sha256": cpp.runtime_implementation_sha256(),
        "files": files, "sha256": {key: _digest((directory / name).read_bytes()) for key, name in files.items()},
        "compiler": compiler, "compiler_options": options, "compiler_version": version[0] if version else "unknown",
        "trip_count": contract["trip_count"], "state_shape": contract["state_shape"], "output_shape": contract["output_shape"],
        "workspace_bytes": 0, "final_bytes": contract["final_bytes"], "incremental_payload_bytes": contract["final_bytes"],
        "memory_budget_bytes": contract["memory_budget_bytes"], "budget_scope": "INCREMENTAL_PAYLOAD_FRESH_FINAL_ONLY",
        "budget_excludes": ["EXTERNAL_INPUTS", "ALLOCATOR_METADATA_AND_ALIGNMENT_OVERHEAD", "THREAD_SCRATCH", "PROCESS_RSS"],
        "cpu_threads": plan.cpu_threads, "cpu_ids": list(plan.resource_binding.cpu_ids),
        "execution": "ONE_FFI_CPP_ROW_ELEMENT_TEMPORAL_LOOP_LOCAL_DOUBLE_CARRY", "body_source_unrolled": False,
        "body_kernel_count": 1, "memory_policy": "UNBOUND", "numa_page_binding": False, "asynchronous_transfers": False,
        "floating_guard": "INITIAL_AND_PER_ITERATION_FINITE_CARRY_STICKY_FENV_ALL_ITERATIONS_RESTORE_CALLER_AND_WORKERS",
        "affinity_guard": "TEMPORARY_EACH_EXECUTING_THREAD_VERIFY_AND_RESTORE",
        "floating_environment_policy": "PRESERVE_CALLER_AND_WORKER_ENVIRONMENT_NOT_NUMPY_STATUS_FLAG_EQUIVALENCE",
        "external_assumptions": ["SYNCHRONOUS_CPU_EXECUTION", "NO_FOREIGN_CONCURRENT_INPUT_WRITERS",
                                 "FP_STATUS_FLAGS_NOT_USED_AS_PROGRAM_OUTPUT"],
        "source_guard_scope": "BUILD_LOAD_SNAPSHOTS_EXTERNAL_LOOP_BODY_NESTED_LEAF_AND_BUILTIN_RANGE_GUARDS_REQUIRED",
        "cost_scope": "TEMPORAL_REPRESENTATION_PAYLOAD_ONLY_EXECUTION_TIME_UNCALIBRATED",
        "duration_predicted": False, "physical_optimality_proven": False, "deployment_authorized": False}
    if layout is not None:
        metadata.update(layout_sha256=semantic_sha256(layout),
            execution="ONE_FFI_CPP_SSE2_SPATIAL_CHAINS_TEMPORAL_LOOP_SCALAR_TAIL",
            simd_lanes=2, vector_chains=layout["selected"]["vector_chains"],
            tile_elements=layout["selected"]["tile_elements"],
            scratch_scope="FIXED_SIZE_VECTOR_LOCALS_ABSTRACT_REGISTER_MODEL_SEPARATE_FROM_AUDITED_COMPILER_STACK",
            abstract_vector_storage_bytes_per_worker=layout["abstract_vector_storage_bytes_per_worker"],
            scratch_budget_bytes_per_worker=layout["scratch_budget_bytes_per_worker"],
            scratch_budget_scope=layout["scratch_budget_scope"],
            register_model=layout["register_model"], isa_audit_sha256=semantic_sha256(
                json.loads((directory / files["isa_audit"]).read_bytes())))
    (directory / "artifact.json").write_bytes(_json_bytes(metadata))
    return load(directory)


def load(out):
    path = Path(out).resolve(); directory = path if path.is_dir() else path.parent
    metadata = json.loads((directory / "artifact.json").read_bytes())
    _require(metadata.get("schema") == SCHEMA and metadata.get("schema_version") in (1, 2) and
             metadata.get("loop_abi_version") == 1, "unsupported native loop artifact schema")
    version = metadata["schema_version"]
    expected_files = {"loop", "strategy", "execution_plan", "source", "loop_abi_header", "array_abi_header", "library"}
    if version == 2: expected_files.update(("layout", "isa_audit"))
    _require(set(metadata["files"]) == expected_files,
             "missing or unexpected native loop sidecars")
    for key, name in metadata["files"].items():
        _require(Path(name).name == name and _digest((directory / name).read_bytes()) == metadata["sha256"][key],
                 "loop artifact receipt differs: " + key)
    loop = read_loop(directory / metadata["files"]["loop"])
    strategy, plan = _admit(loop, directory / metadata["files"]["strategy"], directory / metadata["files"]["execution_plan"])
    layout = _admit_layout(loop, strategy, plan, directory / metadata["files"]["layout"]) if version == 2 else None
    _sources(loop)
    contract = _contract(loop, strategy, plan, layout)
    for key in ("loop_sha256", "strategy_sha256", "execution_plan_sha256", "trip_count", "state_shape", "output_shape", "final_bytes", "memory_budget_bytes"):
        _require(metadata[key] == contract[key], "loop manifest contract differs: " + key)
    _require(metadata["contract_sha256"] == semantic_sha256(contract) and metadata["workspace_bytes"] == 0
             and metadata["incremental_payload_bytes"] == contract["final_bytes"], "loop payload/contract receipt differs")
    _require(metadata["cpu_threads"] == plan.cpu_threads and metadata["cpu_ids"] == list(plan.resource_binding.cpu_ids), "loop CPU receipt differs")
    _require(metadata["backend"] == "cpp" and metadata["body_source_unrolled"] is False and
             metadata["body_kernel_count"] == 1 and metadata["memory_policy"] == "UNBOUND" and
             metadata["numa_page_binding"] is False and metadata["asynchronous_transfers"] is False and
             metadata["duration_predicted"] is False and metadata["physical_optimality_proven"] is False and
             metadata["deployment_authorized"] is False,
             "loop report claims differ from supported synchronous temporal execution")
    if layout is not None:
        _layout_compiler(layout, metadata["compiler"], metadata["compiler_version"])
        audit = json.loads((directory / metadata["files"]["isa_audit"]).read_bytes())
        _require(metadata["layout_sha256"] == semantic_sha256(layout)
                 and metadata["isa_audit_sha256"] == semantic_sha256(audit)
                 and metadata["compiler_options"] == list(layout_compiler_options(plan))
                 and metadata["simd_lanes"] == 2 and metadata["vector_chains"] == layout["selected"]["vector_chains"]
                 and metadata["tile_elements"] == layout["selected"]["tile_elements"]
                 and metadata["abstract_vector_storage_bytes_per_worker"] == layout["abstract_vector_storage_bytes_per_worker"]
                 and metadata["scratch_budget_bytes_per_worker"] == layout["scratch_budget_bytes_per_worker"]
                 and metadata["scratch_budget_scope"] == layout["scratch_budget_scope"]
                 and metadata["register_model"] == layout["register_model"], "loop layout/ISA report differs from admission")
        _require(audit == _isa_audit(directory / metadata["files"]["library"]), "loop compiler ISA/stack audit differs")
    source = _source(loop, strategy, plan, layout).encode()
    _require(source == (directory / metadata["files"]["source"]).read_bytes(), "loop source differs from shared body/count/strategy replay")
    headers = [(directory / metadata["files"][key]).read_bytes() for key in ("loop_abi_header", "array_abi_header")]
    _require(_identity(source, headers, metadata["compiler"], metadata["compiler_options"], metadata["loop_implementation_sha256"]) == metadata["build_id"], "loop build identity differs")
    library = ctypes.CDLL(str(directory / metadata["files"]["library"]))
    getter = library.scar_get_loop_table_v1; getter.argtypes = []; getter.restype = ctypes.POINTER(_Table)
    table = getter()
    _require(bool(table) and table.contents.abi_version == 1 and table.contents.struct_size >= ctypes.sizeof(_Table)
             and bool(table.contents.run) and table.contents.trip_count == contract["trip_count"], "native loop table ABI/count differs")
    for key in ("loop_sha256", "strategy_sha256", "execution_plan_sha256", "contract_sha256"):
        _require(getattr(table.contents, key) == metadata[key].encode(), "native loop embedded identity differs: " + key)
    return LoopArtifact(loop, plan, directory, strategy, metadata, library, table, table.contents.run, layout)


__all__ = ["LoopArtifact", "GuardFailure", "SCHEMA", "build", "load", "loop_implementation_sha256", "layout_compiler_options"]
