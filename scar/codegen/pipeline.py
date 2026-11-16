"""One Python entry and one C++ host invocation for a pure array pipeline.

The host calls existing operator ABI-1 tables in fixed source order. It owns
workspace payload buffers; only the independent fresh NumPy final can escape.
This is a synchronous CPU executor, without copies, asynchronous readiness,
NUMA page placement, or calibrated pipeline-duration claims.
"""

from __future__ import annotations

from collections.abc import Mapping
import ctypes
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

from scar.compiler.model import document as leaf_document
from scar.compiler.pipeline_model import (
    PipelinePlan, document, pipeline_digest, pipeline_from_dict, read_pipeline,
)
from scar.optimization.pipeline_memory import verify_pipeline_memory
from scar.optimization.hardware import validate_cpu_placement
from scar.optimization.resource_solver import semantic_sha256
from . import cpp

GuardFailure = cpp.GuardFailure
_DIRECTORY = Path(__file__).resolve().parent
_TablePointer = ctypes.POINTER(cpp._Table)
PARALLEL_FAMILY = 'DEPENDENCY_WAVES_CPU_SERIAL_LEAVES_V1'
READY_FAMILY = 'DEPENDENCY_READY_CPU_SERIAL_LEAVES_V1'


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def pipeline_implementation_sha256():
    files = [{"path": name, "sha256": _digest((_DIRECTORY / name).read_bytes())}
             for name in ("pipeline.py", "pipeline_v1.cpp.in", "pipeline_parallel_v1.cpp.in",
                          "pipeline_ready.py", "pipeline_ready_v1.cpp.in", "kernel_v1.cpp.in", "scar_abi_v1.h")]
    return semantic_sha256(files)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sources(pipeline):
    seen = set()
    for source in (pipeline.source, *(call.callee.source for call in pipeline.calls)):
        key = (source.path, source.sha256, source.code_key)
        if key in seen:
            continue
        seen.add(key)
        _require(bool(source.code_key) and Path(source.path).is_absolute(), "versioned pipeline/callee source required")
        _require(_digest(Path(source.path).read_bytes()) == source.sha256,
                 "pipeline or callee source snapshot differs")


def _memory(pipeline, memory_plan):
    report = verify_pipeline_memory(pipeline, memory_plan)
    _require(not pipeline.frontiers and report["status"] == "OPTIMAL_WITHIN_FIXED_POOL_MODEL" and
             report["budget_satisfied"] is True and not report["frontier"] and
             report["search_budget_exhausted"] is False and report["memory_policy"] == "UNBOUND",
             "pipeline requires a complete feasible fixed-pool memory plan")
    _require(report["pipeline_sha256"] == pipeline_digest(pipeline), "pipeline memory source receipt differs")
    for row in report["values"]:
        shape, size = row["shape"], row["nbytes"]
        _require(len(shape) == 2 and all(type(n) is int and 0 <= n <= sys.maxsize // 8 for n in shape) and
                 type(size) is int and size == shape[0] * shape[1] * 8 and size <= sys.maxsize,
                 "pipeline value exceeds representable float64 shape/byte bounds")
    _require(type(report["memory_budget_bytes"]) is int and 0 <= report["memory_budget_bytes"] <= sys.maxsize,
             "pipeline payload budget exceeds host bounds")
    _require(report["peak_allocation_bytes"] == report["total_workspace_bytes"] + report["final_bytes"] and
             report["peak_allocation_bytes"] <= report["memory_budget_bytes"], "pipeline payload budget is inconsistent")
    return report


def _compatible(call, operator, memory):
    cpp._supported(call.plan)
    if operator.plan.resource_binding is not None:
        validate_cpu_placement(operator.plan.resource_binding)
    expected, actual = leaf_document(call.plan), leaf_document(operator.plan)
    _require(expected["region"] == actual["region"] and
             all(expected[key] == actual[key] for key in
                 ("profile", "operator_abi_version", "schedule", "device", "buffer_policy")),
             "operator binding changes original region/source/DAG or loop semantics")
    old_budget, new_budget = call.plan.cost.get("memory_budget_bytes"), operator.plan.cost.get("memory_budget_bytes")
    _require(old_budget is None or new_budget is not None and new_budget <= old_budget,
             "operator binding relaxes the original leaf allocation budget")
    values = {value["id"]: value for value in memory["values"]}
    output = values[call.output]
    _require(new_budget is None or output["nbytes"] <= new_budget,
             "operator output exceeds its narrower leaf payload budget")
    if operator.timed_admission is not None:
        guards = operator.timed_admission["must_guard"]
        shapes = {parameter: values[value]["shape"] for parameter, value in
                  zip(call.plan.region.parameters, call.inputs)}
        _require(shapes == guards["input_shapes"] and output["shape"] == guards["output_shape"] and
                 output["nbytes"] == guards["output_nbytes"] and new_budget == guards["memory_budget_bytes"] and
                 guards["input_layout"] == "C_CONTIGUOUS",
                 "operator timed profile differs from pipeline call shapes/layout/budget")


def _operator_receipt(call, operator):
    metadata = operator.report
    files = {}
    for kind, name in metadata["files"].items():
        path = operator.directory / name
        files[kind] = {"path": str(path), "sha256": _digest(path.read_bytes())}
    files["manifest"] = {"path": str(operator.metadata_path), "sha256": _digest(operator.metadata_path.read_bytes())}
    return {"call_id": call.id, "artifact_directory": str(operator.directory),
            "plan_sha256": metadata["plan_sha256"], "library_sha256": metadata["sha256"]["library"],
            "cpu_threads": operator.plan.cpu_threads,
            "cpu_ids": list(operator.plan.resource_binding.cpu_ids) if operator.plan.resource_binding else None,
            "timed_guarded": operator.timed_admission is not None,
            "source_code_key": call.callee.source.code_key, "files": files}


def _admit_execution_schedule(pipeline, memory, operators, report):
    report=json.loads(json.dumps(report,allow_nan=False))
    if report.get('execution_family')==READY_FAMILY:
        from scar.optimization.pipeline_ready_schedule import verify_pipeline_ready_schedule as verify
    elif report.get('execution_family')==PARALLEL_FAMILY:
        from scar.optimization.pipeline_schedule import verify_pipeline_schedule as verify
    else:raise ValueError('unsupported explicit pipeline execution family')
    targets=set()
    for call,operator in zip(pipeline.calls,operators):
        binding=operator.plan.resource_binding
        _require(binding is not None and operator.plan.cpu_threads==1 and len(binding.cpu_ids)==1,
                 'dependency waves require explicitly bound single-CPU ABI1 operators')
        _require(binding.cpu_ids==(report['call_cpu_ids'][call.id],),
                 'operator CPU binding differs from its scheduled caller task')
        targets.add((str(Path(binding.target_path).resolve()),binding.target_sha256))
    _require(len(targets)==1,'one original hardware snapshot required for dependency waves')
    path,digest=next(iter(targets))
    _require(report['target']['uri']==str(Path(path)/'result.json')and report['target']['sha256']==digest,
             'dependency-wave target differs from operator bindings')
    replay=verify(pipeline,memory,path,report)
    _require(replay==report
             and report['requires_serial_leaf_operators']is True,
             'exact verified dependency-wave execution schedule required')
    return report


def _parallel_source(serial_source,pipeline,memory,report):
    marker='extern "C" SCAR_EXPORT_V1 const char *scar_pipeline_plan_sha256_v1()'
    _require(serial_source.count(marker)==1,'original ABI1 descriptor/plan helpers required')
    prefix=serial_source.split(marker,1)[0]
    # Reuse the existing native affinity implementation verbatim. This is a
    # control adapter around unchanged ABI1 mathematical operators.
    kernel=(_DIRECTORY/'kernel_v1.cpp.in').read_text()
    affinity='class ThreadAffinity {'+kernel.split('class ThreadAffinity {',1)[1].split('};\n#endif',1)[0]+'};'
    indexes={call.id:i for i,call in enumerate(pipeline.calls)}
    offsets=[0];flat=[]
    for wave in report['waves']:
        flat.extend(indexes[name]for name in wave);offsets.append(len(flat))
    dependencies=['    {{'+','.join('1'if other.id in report['call_predecessors'][call.id]else'0'
        for other in pipeline.calls)+'}},'for call in pipeline.calls]
    values={
        'PLAN_AND_DESCRIPTOR_HELPERS':prefix,'AFFINITY_IMPLEMENTATION':affinity,
        'CPU_SLOTS':str(max(report['cpu_ids'])+1),
        'CALL_COUNT':str(len(pipeline.calls)),'WAVE_COUNT':str(len(report['waves'])),
        'CALL_CPUS':','.join(str(report['call_cpu_ids'][call.id])for call in pipeline.calls),
        'WAVE_OFFSETS':','.join(map(str,offsets)),'WAVE_CALLS':','.join(map(str,flat)),
        'CALL_PREDECESSORS':'\n'.join(dependencies),
        'PIPELINE_SHA256':pipeline_digest(pipeline),'MEMORY_SHA256':semantic_sha256(memory),
        'EXECUTION_SCHEDULE_SHA256':semantic_sha256(report),
    }
    source=(_DIRECTORY/'pipeline_parallel_v1.cpp.in').read_text()
    for name,value in values.items():source=source.replace('@'+name+'@',value)
    return source


def _source(pipeline, memory, operators, execution_schedule=None):
    values = memory["values"]
    indices = {value["id"]: index for index, value in enumerate(values)}
    slots = {slot["id"]: index for index, slot in enumerate(memory["workspace_slots"])}
    rows = []
    for value in values:
        assignment = memory["assignment"][value["id"]]
        kind = assignment["kind"]
        number = assignment["input_index"] if kind == "EXTERNAL_INPUT" else slots[assignment["slot_id"]] if kind == "WORKSPACE" else 0
        rows.append("    {%dLL, %dLL, %dULL, %d, %dULL, %dLL, %dLL}," %
                    (*value["shape"], value["nbytes"], {"EXTERNAL_INPUT": 0, "WORKSPACE": 1, "FINAL_OUTPUT": 2}[kind],
                     number, value["producer_index"], value["last_use"]))
    calls = []
    for call, operator in zip(pipeline.calls, operators):
        inputs = ", ".join(str(indices[value]) for value in call.inputs)
        budget = operator.plan.cost.get("memory_budget_bytes")
        calls.append('    {%d, {{%s}}, %d, "%s", %s, %dULL},' %
                     (len(call.inputs), inputs, indices[call.output], operator.report["plan_sha256"],
                      "true" if operator.timed_admission is not None else "false",
                      budget if budget is not None else 2**64 - 1))
    replacements = {
        "BUDGET": str(memory["memory_budget_bytes"]), "FINAL_VALUE": str(indices[pipeline.output]),
        "INPUT_COUNT": str(len(pipeline.parameters)), "MAX_ARGUMENTS": str(max(len(c.inputs) for c in pipeline.calls)),
        "VALUE_COUNT": str(len(values)), "CALL_COUNT": str(len(pipeline.calls)),
        "SLOT_COUNT": str(len(slots)), "VALUE_PLANS": "\n".join(rows), "CALL_PLANS": "\n".join(calls),
        "SLOT_CAPACITIES": ", ".join(str(s["capacity_bytes"]) + "ULL" for s in memory["workspace_slots"]),
        "PIPELINE_SHA256": pipeline_digest(pipeline), "MEMORY_SHA256": semantic_sha256(memory),
    }
    source = (_DIRECTORY / "pipeline_v1.cpp.in").read_text()
    for key, value in replacements.items():
        source = source.replace("@" + key + "@", value)
    if execution_schedule is None:return source
    if execution_schedule['execution_family']==READY_FAMILY:
        from .pipeline_ready import render_ready_source
        return render_ready_source(source,pipeline,memory,execution_schedule)
    return _parallel_source(source,pipeline,memory,execution_schedule)


@dataclass
class PipelineArtifact:
    pipeline: PipelinePlan
    directory: Path
    _memory: dict
    _metadata: dict
    _library: object
    _run: object
    _operators: tuple
    _tables: object
    _output_shape: tuple
    _execution_schedule: dict|None = None

    @property
    def memory_plan(self):
        return json.loads(json.dumps(self._memory))

    @property
    def execution_schedule(self):
        return None if self._execution_schedule is None else json.loads(json.dumps(self._execution_schedule))

    @property
    def report(self):
        return {**json.loads(json.dumps(self._metadata)), "artifact_directory": str(self.directory),
                "source_path": str(self.directory / "pipeline.cpp"),
                "library_path": str(self.directory / self._metadata["files"]["library"]),
                "metadata_path": str(self.directory / "artifact.json")}

    def run(self, *arrays):
        import numpy as np
        if len(arrays) != len(self.pipeline.parameters):
            raise GuardFailure("pipeline input count differs")
        if any('EXACT_NUMPY_NDARRAY_COPY_METHOD_DESCRIPTOR' in call.plan.region.assumptions
               for call in self.pipeline.calls):
            from scar.runtime.array_copy_guard import guard_copy_inputs,CopyGuardFailure
            try:guard_copy_inputs(arrays)
            except CopyGuardFailure as error:raise GuardFailure(str(error))from error
        descriptors = [cpp._descriptor(array) for array in arrays]
        for parameter, array in zip(self.pipeline.parameters, arrays):
            if array.shape != tuple(self._memory["input_shapes"][parameter]):
                raise GuardFailure("pipeline input shape differs from memory plan")
            if self._execution_schedule is not None and not array.flags.c_contiguous:
                raise GuardFailure('dependency-wave input layout must match its C-contiguous resource plan')
        try:
            output = np.empty(self._output_shape, dtype=np.float64)
        except (MemoryError, ValueError, OverflowError) as error:
            raise GuardFailure("pipeline fresh final allocation failed") from error
        descriptor = cpp._descriptor(output)
        inputs = (cpp._Array * len(descriptors))(*descriptors)
        failed = ctypes.c_uint64(2**64 - 1)
        status = self._run(inputs, len(descriptors), ctypes.byref(descriptor), self._tables,
                           len(self._operators), ctypes.byref(failed))
        if status:
            reasons = {10: "pointer/alias admission", 11: "shape/layout admission", 12: "payload budget",
                       13: "workspace lifetime/alias", 14: "operator ABI/source binding", 15: "workspace allocation",
                       101: "operator pointer/rank/bounds", 102: "operator shape", 103: "operator floating-point/nonfinite",
                       104: "operator CPU affinity",16:'worker creation or floating environment'}
            where = f" at call {failed.value}" if failed.value != 2**64 - 1 else ""
            raise GuardFailure("pipeline " + reasons.get(status, f"status {status}") + where)
        return output


def build(pipeline, out, *, memory_plan, operators, execution_schedule=None):
    """Compile a host dispatcher over explicitly supplied ABI-1 leaf artifacts."""
    pipeline = read_pipeline(pipeline) if isinstance(pipeline, (str, Path)) else pipeline_from_dict(document(pipeline))
    memory = _memory(pipeline, memory_plan)
    _sources(pipeline)
    _require(isinstance(operators, Mapping) and set(operators) == {call.id for call in pipeline.calls},
             "one explicit operator artifact binding per pipeline call is required")
    loaded = []
    for call in pipeline.calls:
        value = operators[call.id]
        operator = cpp.load(value.directory if isinstance(value, cpp.CppArtifact) else value)
        _compatible(call, operator, memory)
        loaded.append(operator)
    bindings = [_operator_receipt(call, operator) for call, operator in zip(pipeline.calls, loaded)]
    if execution_schedule is not None:
        execution_schedule=_admit_execution_schedule(pipeline,memory,loaded,execution_schedule)
    source = _source(pipeline, memory, loaded,execution_schedule)
    compiler_options=(*cpp.COMPILER_OPTIONS,*(()if execution_schedule is None else('-pthread',)))
    header = (_DIRECTORY / "scar_abi_v1.h").read_bytes()
    fingerprint = pipeline_implementation_sha256()
    compiler = shlex.split(os.environ.get("CXX", "g++"))
    _require(bool(compiler) and shutil.which(compiler[0]) is not None, "pipeline C++20 compiler unavailable")
    build_id = _digest(source.encode() + header + _json_bytes(bindings) + fingerprint.encode() +
                       _json_bytes(compiler) + _json_bytes(list(compiler_options)))
    directory = Path(out).resolve(); directory.mkdir(parents=True, exist_ok=True)
    (directory / "pipeline.json").write_bytes(_json_bytes(document(pipeline)))
    (directory / "memory-plan.json").write_bytes(_json_bytes(memory))
    (directory / "pipeline.cpp").write_text(source)
    (directory / "scar_abi_v1.h").write_bytes(header)
    library = directory / ("pipeline_" + build_id[:20] + ".so")
    fd, temporary = tempfile.mkstemp(prefix=".pipeline-", suffix=".so", dir=directory); os.close(fd)
    try:
        process = subprocess.run([*compiler, *compiler_options, str(directory / "pipeline.cpp"),
                                  "-o", temporary, "-lm"], capture_output=True, text=True)
        if process.returncode:
            raise RuntimeError("pipeline C++20 compilation failed:\n" + process.stderr.strip())
        os.replace(temporary, library)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
    version = subprocess.run([*compiler, "--version"], capture_output=True, text=True).stdout.splitlines()
    files = {"pipeline": "pipeline.json", "memory_plan": "memory-plan.json", "source": "pipeline.cpp",
             "abi_header": "scar_abi_v1.h", "library": library.name}
    if execution_schedule is not None:
        files['execution_schedule']='execution-schedule.json'
        (directory/files['execution_schedule']).write_bytes(_json_bytes(execution_schedule))
    metadata = {
        "schema": "scar.native-array-pipeline", "schema_version": 1, "pipeline_abi_version": 1,
        "backend": "cpp", "build_id": build_id, "pipeline_sha256": pipeline_digest(pipeline),
        "memory_plan_sha256": semantic_sha256(memory), "pipeline_implementation_sha256": fingerprint,
        "files": files, "sha256": {key: _digest((directory / name).read_bytes()) for key, name in files.items()},
        "operators": bindings, "compiler": compiler, "compiler_options": list(compiler_options),
        "compiler_version": version[0] if version else "unknown",
        "workspace_bytes": memory["total_workspace_bytes"], "final_bytes": memory["final_bytes"],
        "incremental_payload_bytes": memory["peak_allocation_bytes"], "memory_budget_bytes": memory["memory_budget_bytes"],
        "budget_scope": memory["budget_scope"],
        "budget_excludes": [*memory["budget_excludes"], "HOST_DESCRIPTOR_AND_DISPATCH_METADATA"],
        "memory_policy": "UNBOUND", "numa_page_binding": False, "asynchronous_transfers": False,
        "execution": "ONE_PYTHON_ENTRY_ONE_FFI_CPP_FIXED_ORDER_ABI1_OPERATOR_CALLS",
        "workspace_lifetime": "ENTRY_ALLOCATION_TO_NATIVE_EXIT_RAII",
        "logical_release": "REUSE_PERMISSION_ONLY_NO_PHYSICAL_FREE",
        "final_ownership": "FRESH_NUMPY_ARRAY_CALLER_OWNED_NOT_WORKSPACE",
        "leaf_python_run_called": False, "python_operator_scheduling": False,
        "source_guard_scope": "SNAPSHOT_BUILD_LOAD_ACTUAL_GLOBAL_CALLEE_CODE_IDS_AT_EXTERNAL_ENTRY_BOUNDARY",
        "cost_scope": "LEAF_KERNEL_ABI_ONLY_PIPELINE_HOST_OVERHEAD_UNCALIBRATED",
        "duration_predicted": False, "physical_optimality_proven": False, "deployment_authorized": False,
    }
    if execution_schedule is not None:
        metadata.update(execution_family=execution_schedule['execution_family'],
            execution_schedule_sha256=semantic_sha256(execution_schedule),
            execution=('ONE_PYTHON_ENTRY_CPP_DEPENDENCY_READY_CPU_LANES_ABI1_SERIAL_OPERATORS'
                if execution_schedule['execution_family']==READY_FAMILY else
                'ONE_PYTHON_ENTRY_CPP_DEPENDENCY_WAVES_JOINED_ABI1_SERIAL_OPERATORS'),
            failure_policy='JOIN_ALL_STARTED_WORKERS_WITHHOLD_OUTPUT_THEN_ORIGINAL_CALLER_ONCE',
            floating_environment='SAVE_CALLER_COPY_FULL_ENV_TO_WORKERS_HOLD_AND_RESTORE_ALL_SCOPES')
    (directory / "artifact.json").write_bytes(_json_bytes(metadata))
    return load(directory)


def load(out):
    path = Path(out).resolve(); directory = path if path.is_dir() else path.parent
    metadata = json.loads((directory / "artifact.json").read_bytes())
    _require((metadata.get("schema"), metadata.get("schema_version"), metadata.get("pipeline_abi_version")) ==
             ("scar.native-array-pipeline", 1, 1), "unsupported native pipeline schema or ABI")
    for key, filename in metadata["files"].items():
        _require(Path(filename).name == filename and _digest((directory / filename).read_bytes()) == metadata["sha256"][key],
                 "pipeline artifact file/receipt differs: " + key)
    pipeline = read_pipeline(directory / metadata["files"]["pipeline"])
    memory = _memory(pipeline, json.loads((directory / metadata["files"]["memory_plan"]).read_bytes()))
    _sources(pipeline)
    _require(pipeline_digest(pipeline) == metadata["pipeline_sha256"] and
             semantic_sha256(memory) == metadata["memory_plan_sha256"], "pipeline source/memory digest differs")
    operators, pointers = [], []
    _require(len(metadata["operators"]) == len(pipeline.calls), "pipeline operator binding count differs")
    for call, binding in zip(pipeline.calls, metadata["operators"]):
        for receipt in binding["files"].values():
            _require(_digest(Path(receipt["path"]).read_bytes()) == receipt["sha256"], "operator artifact receipt differs")
        operator = cpp.load(binding["artifact_directory"])
        _compatible(call, operator, memory)
        _require(_operator_receipt(call, operator) == binding, "operator source/call binding differs")
        pointers.append(operator._library.scar_get_operator_table_v1())
        operators.append(operator)
    parallel_fields=('execution_family'in metadata,'execution_schedule_sha256'in metadata,
                     'execution_schedule'in metadata['files'])
    _require(all(parallel_fields)or not any(parallel_fields),'incomplete dependency-wave metadata')
    execution_schedule=None
    if all(parallel_fields):
        _require(metadata['execution_family']in(PARALLEL_FAMILY,READY_FAMILY),'unsupported pipeline control family')
        execution_schedule=_admit_execution_schedule(pipeline,memory,operators,
            json.loads((directory/metadata['files']['execution_schedule']).read_bytes()))
        _require(metadata['execution_schedule_sha256']==semantic_sha256(execution_schedule),
                 'dependency-wave schedule semantic identity differs')
    source = _source(pipeline, memory, operators,execution_schedule).encode()
    _require(source == (directory / metadata["files"]["source"]).read_bytes(),
             "pipeline host source differs from call/operator/memory bindings")
    identity = _digest(source + (directory / metadata["files"]["abi_header"]).read_bytes() +
                       _json_bytes(metadata["operators"]) + metadata["pipeline_implementation_sha256"].encode() +
                       _json_bytes(metadata["compiler"]) + _json_bytes(metadata["compiler_options"]))
    _require(identity == metadata["build_id"], "pipeline build identity differs")
    library = ctypes.CDLL(str(directory / metadata["files"]["library"]))
    for name, expected in (("scar_pipeline_plan_sha256_v1", metadata["pipeline_sha256"]),
                           ("scar_pipeline_memory_sha256_v1", metadata["memory_plan_sha256"])):
        getter = getattr(library, name); getter.argtypes = []; getter.restype = ctypes.c_char_p
        _require(getter() == expected.encode(), "native pipeline embedded source/memory receipt differs")
    if execution_schedule is not None:
        getter=library.scar_pipeline_schedule_sha256_v1;getter.argtypes=[];getter.restype=ctypes.c_char_p
        _require(getter()==metadata['execution_schedule_sha256'].encode(),
                 'native pipeline embedded execution schedule differs')
    run = library.scar_pipeline_run_v1
    run.argtypes = [ctypes.POINTER(cpp._Array), ctypes.c_uint64, ctypes.POINTER(cpp._Array),
                    ctypes.POINTER(_TablePointer), ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64)]
    run.restype = ctypes.c_int32
    final = next(value for value in memory["values"] if value["id"] == pipeline.output)
    return PipelineArtifact(pipeline, directory, memory, metadata, library, run, tuple(operators),
                            (_TablePointer * len(pointers))(*pointers), tuple(final["shape"]),execution_schedule)


__all__ = ["PipelineArtifact", "GuardFailure", "build", "load", "pipeline_implementation_sha256"]
