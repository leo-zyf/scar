"""Instrument an admitted ready host without changing its operator mathematics.

This observes the actual selected regional implementation, including overlap.
It is not a canonical no-op, a full Python entry timer, or a new executor.
"""
import ctypes
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

from scar.ir.execution_recipe import artifact_reference
from scar.optimization.resource_solver import semantic_sha256
from . import cpp

SCHEMA = 'scar.pipeline-host-probe'
CALLER_SLOTS = ('OUTER_ENTRY', 'CALLER_ENV_CONSTRUCTED', 'RUN_READY_ENTRY',
    'PREFLIGHT_END', 'POOL_ALLOC_BEGIN', 'POOL_ALLOC_END', 'CONTROL_READY',
    'RESERVE_BEGIN', 'RESERVE_END', 'LAUNCH_LOOP_END', 'JOINS_END',
    'FINAL_CHECK_BEGIN', 'RETURN_SUCCESS_BEGIN', 'POOL_DESTRUCT_BEGIN',
    'POOL_DESTRUCT_END', 'RUN_READY_RETURNED', 'CALLER_RESTORE_BEGIN',
    'CALLER_RESTORE_END', 'ORIGINAL_ABI_RETURNED', 'OUTER_RETURN')
LANE_SLOTS = ('EMPLACE_BEGIN', 'EMPLACE_END', 'LAMBDA_ENTRY', 'LAMBDA_EXIT', 'JOIN_BEGIN', 'JOIN_END')
CALL_SLOTS = ('CLAIM_BEGIN', 'CLAIM_END', 'EXECUTE_BEGIN', 'ABI_BEGIN', 'ABI_END',
    'EXECUTE_END', 'PUBLISH_LOCK_BEGIN', 'PUBLISH_LOCK_ACQUIRED', 'PUBLISH_PRE_UNLOCK', 'PUBLISH_POST_UNLOCK')
SYMBOL = 'scar_pipeline_host_probe_run_v1'


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _hash(raw):
    return sha256(raw).hexdigest()


def _json(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def _trace_type(calls, lanes):
    class Trace(ctypes.Structure):
        _fields_ = [('abi_version', ctypes.c_uint64), ('struct_size', ctypes.c_uint64),
            ('call_count', ctypes.c_uint64), ('lane_count', ctypes.c_uint64),
            ('caller_ns', ctypes.c_uint64*len(CALLER_SLOTS)),
            ('caller_valid', ctypes.c_uint8*len(CALLER_SLOTS)),
            ('lane_ns', ctypes.c_uint64*(lanes*len(LANE_SLOTS))),
            ('lane_valid', ctypes.c_uint8*(lanes*len(LANE_SLOTS))),
            ('call_ns', ctypes.c_uint64*(calls*len(CALL_SLOTS))),
            ('call_valid', ctypes.c_uint8*(calls*len(CALL_SLOTS))),
            ('call_status', ctypes.c_int32*calls)]
    return Trace


def _definitions(calls, lanes):
    source = r'''
struct HostProbeTrace {
    uint64_t abi_version, struct_size, call_count, lane_count;
    uint64_t caller_ns[@C@]; uint8_t caller_valid[@C@];
    uint64_t lane_ns[@L@ * @LN@]; uint8_t lane_valid[@L@ * @LN@];
    uint64_t call_ns[@T@ * @TN@]; uint8_t call_valid[@T@ * @TN@];
    int32_t call_status[@TN@];
};
static_assert(sizeof(HostProbeTrace) == @TRACE_BYTES@, "host probe trace ctypes/C++ layout differs");
thread_local HostProbeTrace *host_probe_active = nullptr;
void host_probe_stamp(uint64_t *value, uint8_t *valid) noexcept {
    timespec stamp{};
    if (::clock_gettime(CLOCK_MONOTONIC, &stamp) != 0 || stamp.tv_sec < 0 ||
        stamp.tv_nsec < 0 || stamp.tv_nsec >= 1000000000L ||
        uint64_t(stamp.tv_sec) > (std::numeric_limits<uint64_t>::max() - uint64_t(stamp.tv_nsec)) / 1000000000ULL) return;
    *value = uint64_t(stamp.tv_sec) * 1000000000ULL + uint64_t(stamp.tv_nsec);
    *valid = 1;
}
void host_probe_caller(size_t cut) noexcept {
    if (host_probe_active != nullptr && cut < @C@)
        host_probe_stamp(&host_probe_active->caller_ns[cut], &host_probe_active->caller_valid[cut]);
}
void host_probe_lane(HostProbeTrace *trace, size_t lane, size_t cut) noexcept {
    if (trace != nullptr && lane < @LN@ && cut < @L@)
        host_probe_stamp(&trace->lane_ns[lane * @L@ + cut], &trace->lane_valid[lane * @L@ + cut]);
}
void host_probe_call(HostProbeTrace *trace, size_t call, size_t cut) noexcept {
    if (trace != nullptr && call < @TN@ && cut < @T@)
        host_probe_stamp(&trace->call_ns[call * @T@ + cut], &trace->call_valid[call * @T@ + cut]);
}
class HostProbeLink {
    HostProbeTrace *previous_;
public:
    explicit HostProbeLink(HostProbeTrace *trace) noexcept : previous_(host_probe_active) { host_probe_active = trace; }
    ~HostProbeLink() { host_probe_active = previous_; }
};
class HostProbePoolAfter {
public:
    ~HostProbePoolAfter() { host_probe_caller(@POOL_END@); }
};
class HostProbePoolBefore {
public:
    ~HostProbePoolBefore() { host_probe_caller(@POOL_BEGIN@); }
};
class HostProbeLaneExit {
    HostProbeTrace *trace_; size_t lane_;
public:
    HostProbeLaneExit(HostProbeTrace *trace, size_t lane) noexcept : trace_(trace), lane_(lane) {}
    ~HostProbeLaneExit() { host_probe_lane(trace_, lane_, @LANE_EXIT@); }
};
class HostProbePublishExit {
    HostProbeTrace *trace_; size_t call_ = @TN@;
public:
    explicit HostProbePublishExit(HostProbeTrace *trace) noexcept : trace_(trace) {}
    void arm(size_t call) noexcept { call_ = call; }
    ~HostProbePublishExit() { host_probe_call(trace_, call_, @POST_UNLOCK@); }
};
'''
    values = dict(C=len(CALLER_SLOTS), L=len(LANE_SLOTS), T=len(CALL_SLOTS), LN=lanes, TN=calls,
        TRACE_BYTES=ctypes.sizeof(_trace_type(calls,lanes)),
        POOL_END=CALLER_SLOTS.index('POOL_DESTRUCT_END'), POOL_BEGIN=CALLER_SLOTS.index('POOL_DESTRUCT_BEGIN'),
        LANE_EXIT=LANE_SLOTS.index('LAMBDA_EXIT'), POST_UNLOCK=CALL_SLOTS.index('PUBLISH_POST_UNLOCK'))
    for name,value in values.items():
        source = source.replace('@'+name+'@', str(value))
    return source


def render_host_probe(original_source, *, call_count, lane_cpu_ids):
    """Insert unique source cuts; inverse reconstruction must recover every byte."""
    original = original_source.decode() if isinstance(original_source, bytes) else original_source
    cpus = tuple(lane_cpu_ids)
    _need(type(original) is str and type(call_count) is int and 1 <= call_count <= 4096
        and cpus and len(cpus) <= call_count and len(set(cpus)) == len(cpus)
        and all(type(cpu) is int and 0 <= cpu < 2**31 for cpu in cpus), 'bounded original call/lane geometry required')
    lane_declaration = ('constexpr std::array<int, '+str(len(cpus))+'> lane_cpu_ids = {{'
        +','.join(map(str,cpus))+'}};')
    _need(original.count(lane_declaration) == 1
        and len(re.findall(r'constexpr std::array<int, '+str(call_count)+r'> call_cpu_ids = ', original)) == 1,
        'trace geometry differs from the original fixed CPU lanes')
    source, edits = original, []
    def edit(old, new, role):
        nonlocal source
        _need(source.count(old) == 1, 'original host cut must be unique: '+role)
        edits.append((old,new,role))
        source = source.replace(old,new,1)
    def caller(name):
        return 'host_probe_caller('+str(CALLER_SLOTS.index(name))+');'
    def lane(name, index='host_probe_lane_index'):
        return 'host_probe_lane(host_probe_trace, '+index+', '+str(LANE_SLOTS.index(name))+');'
    def call(name):
        return 'host_probe_call(host_probe_active, index, '+str(CALL_SLOTS.index(name))+');'
    edit('#include <cfenv>\n', '#include <cfenv>\n#include <algorithm>\n#include <time.h>\n', 'monotonic_clock_header')
    marker = 'constexpr int32_t worker_guard = 16;\n'
    edit(marker,marker+_definitions(call_count,len(cpus)), 'fixed_trace_and_noexcept_scope_sentinels')
    marker = '                  uint64_t *failed_call, const fenv_t &caller_environment) {\n'
    edit(marker,marker+'    '+caller('RUN_READY_ENTRY')+'\n', 'run_ready_entry')
    marker = '    std::vector<std::unique_ptr<double[]>> slots(slot_capacities.size());\n'
    edit(marker, '    '+caller('PREFLIGHT_END')+'\n    HostProbePoolAfter host_probe_pool_after;\n    '
        +caller('POOL_ALLOC_BEGIN')+'\n'+marker+'    HostProbePoolBefore host_probe_pool_before;\n', 'pool_allocation_and_destructor_brackets')
    marker = '    // Descriptors never change after launch.'
    edit(marker,'    '+caller('POOL_ALLOC_END')+'\n'+marker,'pool_allocation_complete')
    marker = '    auto worker = [&](int cpu) {\n'
    edit(marker,'    HostProbeTrace *const host_probe_trace = host_probe_active;\n'+marker+
        '        const size_t host_probe_lane_index = size_t(std::find(lane_cpu_ids.begin(), lane_cpu_ids.end(), cpu) - lane_cpu_ids.begin());\n'
        '        HostProbeLink host_probe_link(host_probe_trace);\n'
        '        HostProbeLaneExit host_probe_lane_exit(host_probe_trace, host_probe_lane_index);\n'
        '        '+lane('LAMBDA_ENTRY')+'\n', 'worker_trace_link_and_exit')
    marker = '            std::unique_lock lock(mutex);\n'
    edit(marker,'            HostProbePublishExit host_probe_publish_exit(host_probe_trace);\n'+marker,
        'publish_after_original_unique_lock_destruction')
    marker = '            const int32_t admission = admit_task(index);\n'
    edit(marker,'            '+call('CLAIM_BEGIN')+'\n'+marker, 'claim_begin')
    marker = '            state[index] = 1; ++running;\n'
    edit(marker,marker+'            '+call('CLAIM_END')+'\n','claim_complete')
    marker = '            const int32_t status = execute_task(index, values, operators, caller_environment);\n'
    edit(marker,'            '+call('EXECUTE_BEGIN')+'\n'+marker+'            '+call('EXECUTE_END')+'\n'
        '            if (host_probe_trace != nullptr) host_probe_trace->call_status[index] = status;\n', 'complete_execute_task_including_helper_destructors')
    marker = '            lock.lock();\n'
    edit(marker,'            '+call('PUBLISH_LOCK_BEGIN')+'\n'+marker+'            '+call('PUBLISH_LOCK_ACQUIRED')+'\n', 'completion_mutex_reacquisition')
    marker = '            // Release this CPU lane immediately. A true successor may run\n            // while unrelated tasks on other lanes are still executing.\n            changed.notify_all();\n'
    edit(marker,marker+'            '+call('PUBLISH_PRE_UNLOCK')+'\n'
        '            host_probe_publish_exit.arm(index);\n', 'publication_brackets_not_atomic_ready_timestamps')
    marker = '        const int32_t result = operators[index]->run(arguments.data(), call.count, &output);\n'
    edit(marker,'        '+call('ABI_BEGIN')+'\n'+marker+'        '+call('ABI_END')+'\n', 'unchanged_original_operator_abi_call')
    marker = '    std::vector<std::jthread> workers;\n'
    edit(marker,'    '+caller('CONTROL_READY')+'\n'+marker,'native_control_metadata_ready')
    marker = '        workers.reserve(lane_cpu_ids.size());\n'
    edit(marker,'        '+caller('RESERVE_BEGIN')+'\n'+marker+'        '+caller('RESERVE_END')+'\n','worker_vector_reservation')
    marker = '        for (int cpu : lane_cpu_ids) workers.emplace_back(worker, cpu);\n'
    edit(marker,'        for (size_t host_probe_lane_index = 0; host_probe_lane_index < lane_cpu_ids.size(); ++host_probe_lane_index) {\n'
        '            const int cpu = lane_cpu_ids[host_probe_lane_index];\n'
        '            '+lane('EMPLACE_BEGIN')+'\n            workers.emplace_back(worker, cpu);\n'
        '            '+lane('EMPLACE_END')+'\n        }\n', 'per_lane_emplace_preserves_original_launch_order')
    marker = '    if (creation_failed) {\n'
    edit(marker,'    '+caller('LAUNCH_LOOP_END')+'\n'+marker,'launch_loop_end_can_overlap_worker_computation')
    marker = '    for (auto &thread : workers) if (thread.joinable()) thread.join();\n'
    edit(marker,'    for (size_t host_probe_lane_index = 0; host_probe_lane_index < workers.size(); ++host_probe_lane_index) {\n'
        '        auto &thread = workers[host_probe_lane_index];\n        if (thread.joinable()) {\n'
        '            '+lane('JOIN_BEGIN')+'\n            thread.join();\n'
        '            '+lane('JOIN_END')+'\n        }\n    }\n    '+caller('JOINS_END')+'\n    '
        +caller('FINAL_CHECK_BEGIN')+'\n', 'per_lane_original_join_order')
    marker = '    if (failed_call != nullptr) *failed_call = no_value;\n    return SCAR_OK_V1;\n'
    edit(marker,marker.replace('    return SCAR_OK_V1;', '    '+caller('RETURN_SUCCESS_BEGIN')+'\n    return SCAR_OK_V1;'),
        'normal_ready_return_before_raii_unwind')
    marker = '    FloatingEnvironment caller;\n'
    edit(marker,marker+'    '+caller('CALLER_ENV_CONSTRUCTED')+'\n', 'caller_floating_environment_constructed')
    marker = '        try { status = run_ready(inputs, count, output, operators, operator_count, failed_call, caller.entry()); }\n'
    edit(marker,marker.replace('caller.entry()); }', 'caller.entry()); '+caller('RUN_READY_RETURNED')+' }'),
        'run_ready_return_includes_all_local_destructors')
    marker = '    return caller.restore() ? status : worker_guard;\n'
    edit(marker,'    '+caller('CALLER_RESTORE_BEGIN')+'\n    const bool host_probe_restored = caller.restore();\n    '
        +caller('CALLER_RESTORE_END')+'\n    return host_probe_restored ? status : worker_guard;\n', 'caller_restore_keeps_original_status_rule')
    wrapper = '''
extern "C" SCAR_EXPORT_V1 int32_t scar_pipeline_host_probe_run_v1(
    const scar_array_v1 *inputs, uint64_t count, scar_array_v1 *output,
    const scar_operator_table_v1 *const *operators, uint64_t operator_count,
    uint64_t *failed_call, HostProbeTrace *trace) {
    if (trace == nullptr || trace->abi_version != 1 || trace->struct_size != sizeof(HostProbeTrace) ||
        trace->call_count != @CALLS@ || trace->lane_count != @LANES@) return 2001;
    if (host_probe_active != nullptr) return 2002;
    *trace = HostProbeTrace{};
    trace->abi_version = 1; trace->struct_size = sizeof(HostProbeTrace);
    trace->call_count = @CALLS@; trace->lane_count = @LANES@;
    HostProbeLink link(trace);
    @ENTRY@
    const int32_t status = scar_pipeline_run_v1(inputs, count, output, operators, operator_count, failed_call);
    @RETURNED@
    @EXIT@
    return status;
}
'''.replace('@CALLS@',str(call_count)).replace('@LANES@',str(len(cpus))).replace('@ENTRY@',caller('OUTER_ENTRY')).replace(
        '@RETURNED@',caller('ORIGINAL_ABI_RETURNED')).replace('@EXIT@',caller('OUTER_RETURN'))
    restored = source
    for old,new,role in reversed(edits):
        _need(restored.count(new) == 1, 'instrumentation inverse span is ambiguous: '+role)
        restored = restored.replace(new,old,1)
    _need(restored == original, 'probe source inverse failed to recover the original host')
    source += wrapper
    pins = {str(Path(__file__).resolve()):_hash(Path(__file__).read_bytes())}
    provenance = dict(schema='scar.pipeline-host-probe-source', schema_version=1,
        original_source_sha256=_hash(original.encode()), source_sha256=_hash(source.encode()),
        call_count=call_count, lane_cpu_ids=list(cpus), caller_slots=list(CALLER_SLOTS),
        lane_slots=list(LANE_SLOTS), call_slots=list(CALL_SLOTS), source_pins=pins,
        edits=[dict(role=role, original_span_sha256=_hash(old.encode()), instrumented_span_sha256=_hash(new.encode()),
            original_line=original[:original.index(old)].count('\n')+1 if original.count(old)==1 else None)
            for old,new,role in edits], original_source_inverse_equal=True,
        original_operator_call_statement_unchanged=True, original_operator_libraries_unchanged=True,
        original_pending_policy_unchanged=True, original_join_and_launch_order_unchanged=True,
        target_region_executed_when_run=True, full_target_program_executed=False,
        clock='CLOCK_MONOTONIC_INTEGER_NANOSECONDS', per_event_global_lock_added=False,
        instrumentation_cost_subtracted=False, unfilled_timestamp_is_zero_cost=False,
        publication_semantics='PRE_UNLOCK_AND_POST_UNLOCK_BRACKET_VISIBILITY_NOT_AN_ATOMIC_TIMESTAMP',
        pool_cleanup_semantics='EXACT_SLOTS_VECTOR_DESTRUCTOR_INTERVAL_NOT_PER_SLOT_RELEASE_ORDER',
        scope_excludes=['PYTHON_SOURCE_AND_INPUT_GUARDS','NUMPY_FRESH_FINAL_ALLOCATION','CTYPES_DISPATCH'],
        residuals=['CLOCK_AND_TRACE_STORES','SENTINEL_AND_TLS_BOOKKEEPING','INSTRUMENTATION_COMPILER_EFFECTS',
            'SHIM_VALIDATION_AND_EPILOGUE_OUTSIDE_RECORDED_ORIGINAL_ABI'],
        concurrent_intervals_must_not_be_added_as_serial_cost=True,
        normal_complete_trace_required_for_cost=True)
    return source, provenance


@dataclass
class Probe:
    directory: Path
    artifact: object
    metadata: dict
    _library: object
    _run: object
    _Trace: type

    def run(self, *arrays):
        import numpy as np
        from scar.runtime.array_copy_guard import guard_copy_inputs
        _need(len(arrays) == len(self.artifact.pipeline.parameters), 'probe pipeline input count differs')
        if any('EXACT_NUMPY_NDARRAY_COPY_METHOD_DESCRIPTOR' in call.plan.region.assumptions
                for call in self.artifact.pipeline.calls):
            guard_copy_inputs(arrays)
        descriptors = [cpp._descriptor(array) for array in arrays]
        for name,array in zip(self.artifact.pipeline.parameters,arrays):
            _need(array.shape == tuple(self.artifact.memory_plan['input_shapes'][name])
                and array.flags.c_contiguous, 'probe exact source shape/C layout differs')
        output = np.empty(self.artifact._output_shape, dtype=np.float64)
        descriptor = cpp._descriptor(output)
        trace = self._Trace(abi_version=1, struct_size=ctypes.sizeof(self._Trace),
            call_count=len(self.artifact.pipeline.calls), lane_count=len(self.metadata['lane_cpu_ids']))
        failed = ctypes.c_uint64(2**64-1)
        status = int(self._run((cpp._Array*len(descriptors))(*descriptors), len(descriptors), ctypes.byref(descriptor),
            self.artifact._tables, len(self.artifact._operators), ctypes.byref(failed), ctypes.byref(trace)))
        def row(values, valid, names, offset=0):
            return {name:dict(ns=int(values[offset+i]), valid=bool(valid[offset+i])) for i,name in enumerate(names)}
        caller = row(trace.caller_ns,trace.caller_valid,CALLER_SLOTS)
        lanes = [dict(cpu_id=cpu, cuts=row(trace.lane_ns,trace.lane_valid,LANE_SLOTS,i*len(LANE_SLOTS)))
            for i,cpu in enumerate(self.metadata['lane_cpu_ids'])]
        calls = [dict(call_id=call.id, status=int(trace.call_status[i]),
            cuts=row(trace.call_ns,trace.call_valid,CALL_SLOTS,i*len(CALL_SLOTS)))
            for i,call in enumerate(self.artifact.pipeline.calls)]
        complete = status == 0 and all(record['status'] == 0 for record in calls) and all(cut['valid'] for cuts in
            [caller,*[record['cuts'] for record in lanes],*[record['cuts'] for record in calls]] for cut in cuts.values())
        return dict(output=output, status=status, failed_call=int(failed.value), trace=dict(
            schema='scar.pipeline-host-probe-trace', schema_version=1, clock='CLOCK_MONOTONIC',
            caller=caller, lanes=lanes, calls=calls, normal_complete=complete,
            cost_admissible=complete,
            target_region_executed=any(record['cuts']['ABI_BEGIN']['valid'] for record in calls),
            full_target_program_executed=False,
            instrumentation_cost_subtracted=False))


def build(artifact_or_path, out):
    """Reload source-qualified delivery, then build a separately pinned probe ELF."""
    from . import pipeline, pipeline_implementation
    artifact = artifact_or_path if type(artifact_or_path) is pipeline.PipelineArtifact else None
    directory = artifact.directory if artifact is not None else Path(artifact_or_path).resolve()
    if directory.is_file():
        directory = directory.parent
    original_schema = json.loads((directory/'artifact.json').read_bytes())['schema']
    loader = pipeline_implementation.load if original_schema == pipeline_implementation.SCHEMA else pipeline.load
    admitted = loader(directory)
    if artifact is not None:
        _need(semantic_sha256(admitted.report) == semantic_sha256(artifact.report), 'qualified host object differs from original cold admission')
    metadata = admitted.report
    original_source = (admitted.directory/metadata['files']['source']).read_bytes()
    _need(_hash(original_source) == metadata['sha256']['source'], 'original host source receipt differs')
    schedule = admitted.execution_schedule
    _need(schedule is not None, 'explicit ready/lane-head execution certificate required')
    source, provenance = render_host_probe(original_source, call_count=len(admitted.pipeline.calls), lane_cpu_ids=schedule['cpu_ids'])
    directory = Path(out).resolve()
    _need(not directory.exists(), 'fresh host probe output required')
    directory.mkdir(parents=True)
    (directory/'probe.cpp').write_text(source)
    header = (admitted.directory/metadata['files']['abi_header']).read_bytes()
    (directory/'scar_abi_v1.h').write_bytes(header)
    compiler, options = metadata['compiler'], metadata['compiler_options']
    identity = _hash(source.encode()+header+_json(compiler)+_json(options))
    library_path = directory/('host_probe_'+identity[:20]+'.so')
    fd,temporary = tempfile.mkstemp(prefix='.host-probe-',suffix='.so',dir=directory); os.close(fd)
    try:
        result = subprocess.run([*compiler,*options,str(directory/'probe.cpp'),'-o',temporary,'-lm'],capture_output=True,text=True)
        if result.returncode:
            raise RuntimeError('pipeline host probe compilation failed:\n'+result.stderr.strip())
        os.replace(temporary,library_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    report = dict(schema=SCHEMA,schema_version=1, original_artifact=artifact_reference(admitted.directory/'artifact.json',original_schema),
        original_library=artifact_reference(admitted.directory/metadata['files']['library'],'scar.shared-library'),
        original_execution_schedule_sha256=metadata['execution_schedule_sha256'],
        selected_certificate=metadata.get('preparation'), source=artifact_reference(directory/'probe.cpp','scar.cpp-source'),
        header=artifact_reference(directory/'scar_abi_v1.h','scar.cpp-header'),
        library=artifact_reference(library_path,'scar.shared-library'), provenance=provenance,
        call_ids=[call.id for call in admitted.pipeline.calls],lane_cpu_ids=schedule['cpu_ids'],
        compiler=compiler,compiler_options=options,build_id=identity,symbol=SYMBOL,trace_abi_version=1,
        trace_struct_bytes=ctypes.sizeof(_trace_type(len(admitted.pipeline.calls),len(schedule['cpu_ids']))),
        target_region_executed_when_run=True,full_target_program_executed=False,
        performance_measured=False,deployment_authorized=False)
    (directory/'probe.json').write_bytes(_json(report))
    library = ctypes.CDLL(str(library_path)); run = getattr(library,SYMBOL)
    Trace = _trace_type(len(admitted.pipeline.calls),len(schedule['cpu_ids']))
    run.argtypes = [ctypes.POINTER(cpp._Array),ctypes.c_uint64,ctypes.POINTER(cpp._Array),
        ctypes.POINTER(ctypes.POINTER(cpp._Table)),ctypes.c_uint64,ctypes.POINTER(ctypes.c_uint64),ctypes.POINTER(Trace)]
    run.restype = ctypes.c_int32
    return Probe(directory,admitted,report,library,run,Trace)


__all__ = ['render_host_probe','build','Probe','CALLER_SLOTS','LANE_SLOTS','CALL_SLOTS','SYMBOL']
