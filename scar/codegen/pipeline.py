"""Pipeline."""

from __future__ import annotations

from collections.abc import Mapping
import ctypes
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from scar.compiler.model import document as leaf_document
from scar.compiler.pipeline_frontend import (
    PipelinePlan,
    document,
    pipeline_digest,
    pipeline_from_dict,
    read_pipeline,
)
from scar.optimization.pipeline_memory import verify_pipeline_memory
from scar.optimization.hardware import validate_cpu_placement
from scar.optimization.resource_solver import semantic_sha256
from scar.codegen import cpp
from scar.codegen.pipeline_resource_events import (
    DIAGNOSTIC_CAPACITY_PER_EVENT,
    DIAGNOSTIC_KINDS,
    RESOURCE_EVENTS_FAMILY,
)


# One Python entry and one C++ host invocation for a pure array pipeline.
#
# The host calls existing operator ABI-1 tables in source order or under a
# replayed CPU schedule. It owns workspace payload buffers and returns the
# independent fresh NumPy final after successful execution.
GuardFailure = cpp.GuardFailure
_DIRECTORY = Path(__file__).resolve().parent
_TablePointer = ctypes.POINTER(cpp._Table)
PARALLEL_FAMILY = 'DEPENDENCY_WAVES_CPU_SERIAL_LEAVES_V1'
READY_FAMILY = 'DEPENDENCY_READY_CPU_SERIAL_LEAVES_V1'
SOLVER_ORDER_FAMILY = 'SOLVER_ORDER_CPU_SERIAL_LEAVES_V1'
_RESOURCE_EVENT_SCOPE = 'RESOURCE_EVENTS_FIXED_CPU_LANES_AND_POOL_VERSIONS'
_NO_VALUE = 2**64 - 1


class _NativeEvent(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in ('kind', 'event', 'timestamp_ns', 'value', 'slot')]
    _fields_ += [(name, ctypes.c_int64) for name in ('cpu', 'status')]


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def pipeline_implementation_sha256():
    files = [{"path": name, "sha256": _digest((_DIRECTORY / name).read_bytes())}
             for name in ("pipeline.py", "pipeline_v1.cpp.in", "pipeline_parallel_v1.cpp.in",
                          "pipeline_ready_v1.cpp.in",
                          "pipeline_resource_events.py", "pipeline_resource_events_v1.cpp.in",
                          "kernel_v1.cpp.in", "scar_abi_v1.h")]
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
    complete_profiles={('scar.pipeline-memory-plan',1,'OPTIMAL_WITHIN_FIXED_POOL_MODEL'),
        ('scar.pipeline-partial-order-memory-plan',1,'OPTIMAL_WITHIN_RAW_PARTIAL_ORDER_POOL_MODEL')}
    _require(not pipeline.frontiers and(report.get('schema'),report.get('schema_version'),report['status'])in complete_profiles and
             report["budget_satisfied"] is True and not report["frontier"] and
             report["search_budget_exhausted"] is False and report['assignment']is not None
             and report["memory_policy"] == "UNBOUND",
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
    if isinstance(report,(str,Path)):
        report=json.loads(Path(report).read_bytes())
    report=json.loads(json.dumps(report,allow_nan=False))
    _require(type(report)is dict,'explicit pipeline execution report required')
    if report.get('execution_family')==READY_FAMILY:
        from scar.optimization.pipeline_schedule import verify_pipeline_ready_schedule as verify
    elif report.get('execution_family')==PARALLEL_FAMILY:
        from scar.optimization.pipeline_schedule import verify_pipeline_schedule as verify
    elif report.get('execution_family') in (SOLVER_ORDER_FAMILY, RESOURCE_EVENTS_FAMILY):
        from scar.optimization.pipeline_execution_refinement import verify_pipeline_execution_refinement as verify
    else:raise ValueError('unsupported explicit pipeline execution family')
    targets=set()
    for call,operator in zip(pipeline.calls,operators):
        binding=operator.plan.resource_binding
        _require(binding is not None and operator.plan.cpu_threads==1 and len(binding.cpu_ids)==1,
                 'parallel pipeline control requires explicitly bound single-CPU ABI1 operators')
        _require(binding.cpu_ids==(report['call_cpu_ids'][call.id],),
                 'operator CPU binding differs from its scheduled caller task')
        targets.add((str(Path(binding.target_path).resolve()),binding.target_sha256))
    _require(len(targets)==1,'one original hardware snapshot required for parallel pipeline control')
    path,digest=next(iter(targets))
    _require(report['target']['uri']==str(Path(path)/'result.json')and report['target']['sha256']==digest,
             'pipeline execution target differs from operator bindings')
    replay=verify(pipeline,memory,path,report)
    _require(replay==report
             and report['requires_serial_leaf_operators']is True,
             'exact verified pipeline execution schedule required')
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
        return render_ready_source(source,pipeline,memory,execution_schedule)
    if execution_schedule['execution_family']==SOLVER_ORDER_FAMILY:
        return render_solver_order_source(source,pipeline,memory,execution_schedule)
    if execution_schedule['execution_family']==RESOURCE_EVENTS_FAMILY:
        from .pipeline_resource_events import render_resource_events_source
        return render_resource_events_source(source,pipeline,memory,execution_schedule)
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
    _run_diagnostic: object = None

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
        return self._invoke(arrays)

    def run_diagnostic(self, *arrays):
        """Run the separate native event observer and return output and observed rows."""
        if self._run_diagnostic is None:
            raise GuardFailure('pipeline has no dedicated native resource-event diagnostic')
        return self._invoke(arrays, diagnostic=True)

    def _invoke(self, arrays, *, diagnostic=False):
        import numpy as np
        if len(arrays) != len(self.pipeline.parameters):
            raise GuardFailure("pipeline input count differs")
        if any('EXACT_NUMPY_NDARRAY_COPY_METHOD_DESCRIPTOR' in call.plan.region.assumptions
               for call in self.pipeline.calls):
            from scar.runtime.array_copy_guard import guard_copy_inputs, CopyGuardFailure
            try:guard_copy_inputs(arrays)
            except CopyGuardFailure as error:raise GuardFailure(str(error))from error
        descriptors = [cpp._descriptor(array) for array in arrays]
        for parameter, array in zip(self.pipeline.parameters, arrays):
            if array.shape != tuple(self._memory["input_shapes"][parameter]):
                raise GuardFailure("pipeline input shape differs from memory plan")
            if self._execution_schedule is not None and not array.flags.c_contiguous:
                raise GuardFailure('scheduled pipeline input layout must match its C-contiguous resource plan')
        try:
            output = np.empty(self._output_shape, dtype=np.float64)
        except (MemoryError, ValueError, OverflowError) as error:
            raise GuardFailure("pipeline fresh final allocation failed") from error
        descriptor = cpp._descriptor(output)
        inputs = (cpp._Array * len(descriptors))(*descriptors)
        failed = ctypes.c_uint64(_NO_VALUE)
        arguments = (inputs, len(descriptors), ctypes.byref(descriptor), self._tables,
                     len(self._operators), ctypes.byref(failed))
        if diagnostic:
            events = self._execution_schedule['resource_events']
            capacity = len(events) * DIAGNOSTIC_CAPACITY_PER_EVENT
            buffer, used = (_NativeEvent * capacity)(), ctypes.c_uint64()
            status = self._run_diagnostic(*arguments, buffer, capacity, ctypes.byref(used))
            values = self._memory['values']
            slots = [row['id'] for row in self._memory['workspace_slots']] + ['final:output']
            observations = []
            for sequence, row in enumerate(buffer[:used.value]):
                event = events[row.event]
                observations.append(dict(sequence=sequence, kind=DIAGNOSTIC_KINDS[row.kind],
                    event_id=event['id'], call_id=event.get('call_id'), timestamp_ns=row.timestamp_ns,
                    cpu_id=None if row.cpu < 0 else row.cpu,
                    value_id=None if row.value == _NO_VALUE else values[row.value]['id'],
                    slot_id=None if row.slot == _NO_VALUE else slots[row.slot], status=row.status))
        else:
            status = self._run(*arguments)
        if status:
            reasons = {10: "pointer/alias admission", 11: "shape/layout admission", 12: "payload budget",
                       13: "workspace lifetime/alias", 14: "operator ABI/source binding", 15: "workspace allocation",
                       101: "operator pointer/rank/bounds", 102: "operator shape", 103: "operator floating-point/nonfinite",
                       104: "operator CPU affinity",16:'worker creation or floating environment'}
            where = f" at call {failed.value}" if failed.value != _NO_VALUE else ""
            failure = GuardFailure("pipeline " + reasons.get(status, f"status {status}") + where)
            if diagnostic:
                failure.observed_events = observations
            raise failure
        return (output, observations) if diagnostic else output


def build(pipeline, out, *, memory_plan, operators, execution_schedule=None):
    """Compile an ABI-1 host; execution_schedule may be a report or its JSON path.

    A solver-order certificate is independently replayed against its complete
    original finite domain. Only compute lane order is implemented; its model
    timestamps are never used as native timing or deployment authority.
    """
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
                'ONE_PYTHON_ENTRY_CPP_SOLVER_ORDER_FIXED_CPU_LANES_ABI1_SERIAL_OPERATORS'
                if execution_schedule['execution_family']==SOLVER_ORDER_FAMILY else
                'ONE_PYTHON_ENTRY_CPP_RESOURCE_EVENTS_FIXED_CPU_LANES_ABI1_SERIAL_OPERATORS'
                if execution_schedule['execution_family']==RESOURCE_EVENTS_FAMILY else
                'ONE_PYTHON_ENTRY_CPP_DEPENDENCY_WAVES_JOINED_ABI1_SERIAL_OPERATORS'),
            failure_policy='JOIN_ALL_STARTED_WORKERS_WITHHOLD_OUTPUT_THEN_ORIGINAL_CALLER_ONCE',
            floating_environment='SAVE_CALLER_COPY_FULL_ENV_TO_WORKERS_HOLD_AND_RESTORE_ALL_SCOPES')
        if execution_schedule['execution_family']==SOLVER_ORDER_FAMILY:
            metadata.update(implementation_scope='MODEL_SOLUTION_COMPUTE_ORDER_ONLY',
                full_model_timeline_implemented=False,
                task_model_makespan_is_native_runtime_prediction=False,
                native_host_optimality_proven=False,
                existing_host_envelope=execution_schedule['existing_host_envelope'])
        if execution_schedule['execution_family'] == RESOURCE_EVENTS_FAMILY:
            metadata.update(implementation_scope=_RESOURCE_EVENT_SCOPE,
                native_resource_events_implemented=True, full_model_timeline_implemented=False,
                task_model_makespan_is_native_runtime_prediction=False, native_host_optimality_proven=False,
                workspace_lifetime='ALLOCATE_TO_RELEASE_RESOURCE_EVENTS_WITH_RAII_ON_FAILURE',
                logical_release='LAST_SUCCESSFUL_CONSUMER_RETIRE_THEN_PHYSICAL_SLOT_RELEASE_EVENT',
                final_storage_admission='ADOPT_PRIVATE_CALLER_DESCRIPTOR_AT_FINAL_ALLOCATE_EVENT',
                native_diagnostic_symbol='scar_pipeline_run_diagnostic_v1')
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
    _require(all(parallel_fields)or not any(parallel_fields),'incomplete parallel pipeline metadata')
    execution_schedule=None
    if all(parallel_fields):
        _require(metadata['execution_family']in(PARALLEL_FAMILY,READY_FAMILY,SOLVER_ORDER_FAMILY,RESOURCE_EVENTS_FAMILY),
                 'unsupported pipeline control family')
        execution_schedule=_admit_execution_schedule(pipeline,memory,operators,
            json.loads((directory/metadata['files']['execution_schedule']).read_bytes()))
        _require(metadata['execution_family']==execution_schedule['execution_family']
                 and metadata['execution_schedule_sha256']==semantic_sha256(execution_schedule),
                 'pipeline execution family or schedule semantic identity differs')
        if execution_schedule['execution_family']==SOLVER_ORDER_FAMILY:
            _require(metadata.get('implementation_scope')=='MODEL_SOLUTION_COMPUTE_ORDER_ONLY'
                     and metadata.get('full_model_timeline_implemented')is False
                     and metadata.get('task_model_makespan_is_native_runtime_prediction')is False
                     and metadata.get('native_host_optimality_proven')is False
                     and metadata.get('existing_host_envelope')==execution_schedule['existing_host_envelope'],
                     'solver-order artifact must retain the original host envelope and compute-only scope')
        if execution_schedule['execution_family'] == RESOURCE_EVENTS_FAMILY:
            _require(metadata.get('implementation_scope') == _RESOURCE_EVENT_SCOPE
                     and metadata.get('native_resource_events_implemented') is True
                     and metadata.get('full_model_timeline_implemented') is False
                     and metadata.get('task_model_makespan_is_native_runtime_prediction') is False
                     and metadata.get('native_host_optimality_proven') is False
                     and metadata.get('native_diagnostic_symbol') == 'scar_pipeline_run_diagnostic_v1',
                     'resource-event artifact must retain its native implementation scope')
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
    diagnostic = None
    if metadata.get('execution_family') == RESOURCE_EVENTS_FAMILY:
        diagnostic = library.scar_pipeline_run_diagnostic_v1
        diagnostic.argtypes = [*run.argtypes, ctypes.POINTER(_NativeEvent), ctypes.c_uint64,
                               ctypes.POINTER(ctypes.c_uint64)]
        diagnostic.restype = ctypes.c_int32
    final = next(value for value in memory["values"] if value["id"] == pipeline.output)
    return PipelineArtifact(pipeline, directory, memory, metadata, library, run, tuple(operators),
                            (_TablePointer * len(pointers))(*pointers), tuple(final["shape"]), execution_schedule,
                            diagnostic)


# Emit dependency-ready control around unchanged, CPU-bound ABI1 operators.
_TEMPLATE_DIRECTORY=Path(__file__).resolve().parent


def render_ready_source(serial_source,pipeline,memory,report):
    marker='extern "C" SCAR_EXPORT_V1 const char *scar_pipeline_plan_sha256_v1()'
    if serial_source.count(marker)!=1:raise ValueError('original ABI1 plan/descriptor helpers required')
    prefix=serial_source.split(marker,1)[0]
    kernel=(_TEMPLATE_DIRECTORY/'kernel_v1.cpp.in').read_text()
    affinity='class ThreadAffinity {'+kernel.split('class ThreadAffinity {',1)[1].split('};\n#endif',1)[0]+'};'
    wave=(_TEMPLATE_DIRECTORY/'pipeline_parallel_v1.cpp.in').read_text()
    worker=wave[wave.index('// Save the complete environment'):wave.index('int32_t run_waves(')]
    dependencies=['    {{'+','.join('1'if other.id in report['call_predecessors'][call.id]else'0'
        for other in pipeline.calls)+'}},'for call in pipeline.calls]
    values=dict(PLAN_AND_DESCRIPTOR_HELPERS=prefix,AFFINITY_IMPLEMENTATION=affinity,
        WORKER_ENVIRONMENT_AND_ABI_CALL=worker,CPU_SLOTS=str(max(report['cpu_ids'])+1),
        CALL_COUNT=str(len(pipeline.calls)),LANE_COUNT=str(len(report['cpu_ids'])),
        LANE_CPUS=','.join(map(str,report['cpu_ids'])),
        CALL_CPUS=','.join(str(report['call_cpu_ids'][call.id])for call in pipeline.calls),
        CALL_PREDECESSORS='\n'.join(dependencies),PIPELINE_SHA256=pipeline_digest(pipeline),
        MEMORY_SHA256=semantic_sha256(memory),EXECUTION_SCHEDULE_SHA256=semantic_sha256(report))
    source=(_TEMPLATE_DIRECTORY/'pipeline_ready_v1.cpp.in').read_text()
    for name,value in values.items():source=source.replace('@'+name+'@',value)
    if '@'in source:raise ValueError('unresolved dependency-ready host template')
    return source


# Render only certified CPU-lane order over the original ready-host envelope.
#
# Full source/domain/cost replay belongs to pipeline._admit_execution_schedule.
# These structural checks also prevent a partial or inconsistent lane policy
# from being emitted if this renderer is called directly.
_READY_POLICY = '''    auto next_for_cpu = [&](int cpu) {
        for (size_t i = 0; i < call_plans.size(); ++i)
            if (state[i] == 0 && call_cpu_ids[i] == cpu && predecessors_complete(i)) return i;
        return call_plans.size();
    };
    auto any_ready = [&]() {
        for (size_t i = 0; i < call_plans.size(); ++i)
            if (state[i] == 0 && predecessors_complete(i)) return true;
        return false;
    };
'''




def render_solver_order_source(serial_source, pipeline, memory, report):
    _require(report.get('schema') == 'scar.pipeline-execution-refinement'
             and type(report.get('schema_version')) is int and report['schema_version'] == 1
             and report.get('execution_family') == SOLVER_ORDER_FAMILY
             and report.get('requires_serial_leaf_operators') is True
             and report.get('dispatch_policy') ==
             'FIXED_CPU_LANE_HEAD_ONLY_WAIT_TRUE_PREDECESSORS_NO_TIMESTAMP_WAIT'
             and report.get('full_model_timeline_implemented') is False
             and report.get('task_model_makespan_is_native_runtime_prediction') is False,
             'a compute-only solver execution refinement is required')
    indexes = {call.id: index for index, call in enumerate(pipeline.calls)}
    cpus, rows = report['cpu_ids'], report['model_compute_order']
    _require(type(cpus) is list and bool(cpus) and
             all(type(cpu) is int and cpu >= 0 for cpu in cpus) and len(set(cpus)) == len(cpus),
             'solver CPU lanes must be distinct explicit CPU IDs')
    _require(len(indexes) == len(pipeline.calls) and type(rows) is list
             and len(rows) == len(indexes) and
             len({row['call'] for row in rows}) == len(rows) and
             {row['call'] for row in rows} == set(indexes) and
             set(report['call_cpu_ids']) == set(indexes),
             'solver order must cover each original call exactly once')
    lanes = {str(cpu): [] for cpu in cpus}
    for row in rows:
        cpu = report['call_cpu_ids'][row['call']]
        _require(type(cpu) is int and cpu in cpus and row['cpu_id'] == cpu,
                 'solver call placement differs from its fixed CPU lane')
        lanes[str(cpu)].append(row['call'])
    _require(report['lane_call_order'] == lanes,
             'solver compute order and certified per-lane order differ')
    ordered = [indexes[row['call']] for row in rows]
    source = render_ready_source(serial_source, pipeline, memory, report)
    declaration = ('constexpr std::array<size_t, ' + str(len(ordered))
        + '> solver_compute_order = {{' + ','.join(map(str, ordered)) + '}};\n')
    marker = 'constexpr int32_t worker_guard = 16;\n'
    _require(source.count(marker) == 1 and source.count(_READY_POLICY) == 1,
             'original ready-host lane policy and declaration marker must be unique')
    source = source.replace(marker, marker + declaration, 1)
    policy = '''    // Resource order is a certified implementation refinement. Never
    // bypass a pending lane head merely because a later call is ready.
    auto next_for_cpu = [&](int cpu) {
        for (const size_t i : solver_compute_order) {
            if (state[i] != 0 || call_cpu_ids[i] != cpu) continue;
            return predecessors_complete(i) ? i : call_plans.size();
        }
        return call_plans.size();
    };
    auto any_ready = [&]() {
        for (const int cpu : lane_cpu_ids)
            if (next_for_cpu(cpu) != call_plans.size()) return true;
        return false;
    };
'''
    return source.replace(_READY_POLICY, policy, 1)


__all__ = [
    'PipelineArtifact',
    'GuardFailure',
    'build',
    'load',
    'pipeline_implementation_sha256',
    'READY_FAMILY',
    'render_ready_source',
    'SOLVER_ORDER_FAMILY',
    'render_solver_order_source',
    'PARALLEL_FAMILY',
]
