"""Execute selected CPU implementations using the existing ABI1 ready host.

Only selected placement and compute lane order are implemented. All native
pool, affinity, floating-environment, join and private-output checks remain.
No model timestamps are executed, and this artifact is an explicit trial.
"""
import ctypes
from hashlib import sha256
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

from scar.compiler.model import document as leaf_document
from scar.compiler.pipeline_model import document, pipeline_digest
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.hardware import bind_cpu_placement, propose_cpu_placements
from scar.optimization.pipeline_implementation import FAMILY, CERT_SCHEMA
from scar.optimization.resource_solver import semantic_sha256
from . import cpp
from . import pipeline as original
from .pipeline_ready import render_ready_source
from .pipeline_solver_order import _READY_POLICY

SCHEMA = 'scar.native-pipeline-implementation'
_DIRECTORY = Path(__file__).resolve().parent


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _hash(raw):
    return sha256(raw).hexdigest()


def _json(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def implementation_sha256():
    from scar.optimization import pipeline_implementation
    from scar.codegen import pipeline_lifecycle
    return semantic_sha256(dict(original_host=original.pipeline_implementation_sha256(),
        renderer=_hash(Path(__file__).read_bytes()),
        selected_certificate=_hash(Path(pipeline_implementation.__file__).read_bytes()),
        lifecycle_delivery=_hash(Path(pipeline_lifecycle.__file__).read_bytes())))


def render_implementation_source(serial_source, pipeline, memory, certificate):
    """Retain ready-host safety code; replace only its pending-lane policy."""
    _require((certificate.get('schema'), certificate.get('schema_version')) == (CERT_SCHEMA, 1)
        and certificate.get('execution_family') == FAMILY
        and certificate.get('requires_serial_leaf_operators') is True
        and certificate.get('full_model_timeline_implemented') is False,
        'source-qualified implementation certificate required')
    indexes = {call.id:index for index,call in enumerate(pipeline.calls)}
    order, lanes = certificate['call_order'], certificate['lane_call_order']
    _require(len(order) == len(indexes) and set(order) == set(indexes)
        and lanes == {str(cpu):[name for name in order if certificate['call_cpu_ids'][name] == cpu]
            for cpu in certificate['cpu_ids']}, 'selected CPU lane coverage differs')
    source = render_ready_source(serial_source, pipeline, memory, certificate)
    marker = 'constexpr int32_t worker_guard = 16;\n'
    _require(source.count(marker) == 1 and source.count(_READY_POLICY) == 1,
        'original ready dispatch policy must have one exact source span')
    declaration = ('constexpr std::array<size_t, '+str(len(order))+'> implementation_call_order = {{'
        +','.join(str(indexes[name]) for name in order)+'}};\n')
    policy = '''    // Selected implementation order is a resource refinement, not a clock.
    auto next_for_cpu = [&](int cpu) {
        for (const size_t i : implementation_call_order) {
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
    return source.replace(marker, marker+declaration, 1).replace(_READY_POLICY, policy, 1)


def _selected_plans(pipeline, memory, target, problem, execution, certificate):
    selected = {row['operation']:row['option'] for row in execution['selected']}
    modes = {op.id:next(mode for mode in op.options if mode.id == selected[op.id])
        for op in problem.operations}
    result = {}
    for call in pipeline.calls:
        mode = modes[call.id]
        cpu = certificate['call_cpu_ids'][call.id]
        placements = propose_cpu_placements(call.plan, target, affinity=[cpu],
            memory_budget_bytes=memory['memory_budget_bytes'], max_threads=1)
        binding = next((row for row in placements if row.cpu_ids == (cpu,)), None)
        _require(binding is not None, 'selected original CPU option no longer available')
        plan = bind_cpu_placement(call.plan, binding)
        _require(semantic_sha256(leaf_document(plan)) == mode.parameters['bound_leaf_plan_sha256'],
            'selected lowering differs from the costed source/CPU plan')
        result[call.id] = plan
    return result


def _source(pipeline, memory, operators, certificate):
    return render_implementation_source(original._source(pipeline, memory, operators),
        pipeline, memory, certificate).encode()


def _instance(directory, metadata, pipeline, memory, operators, certificate):
    library = ctypes.CDLL(str(directory/metadata['files']['library']))
    for symbol, expected in (
        ('scar_pipeline_plan_sha256_v1', metadata['pipeline_sha256']),
        ('scar_pipeline_memory_sha256_v1', metadata['memory_plan_sha256']),
        ('scar_pipeline_schedule_sha256_v1', metadata['execution_schedule_sha256'])):
        getter = getattr(library, symbol); getter.argtypes = []; getter.restype = ctypes.c_char_p
        _require(getter() == expected.encode(), 'native embedded source/pool/selection identity differs')
    run = library.scar_pipeline_run_v1
    pointer = ctypes.POINTER(cpp._Table)
    run.argtypes = [ctypes.POINTER(cpp._Array), ctypes.c_uint64, ctypes.POINTER(cpp._Array),
        ctypes.POINTER(pointer), ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64)]
    run.restype = ctypes.c_int32
    tables = (pointer*len(operators))(*(operator._library.scar_get_operator_table_v1() for operator in operators))
    final = next(row for row in memory['values'] if row['id'] == pipeline.output)
    return original.PipelineArtifact(pipeline, directory, memory, metadata, library, run,
        tuple(operators), tables, tuple(final['shape']), certificate)


def build(preparation, out):
    """Cold replay a joint preparation, then compile exactly its chosen leaves."""
    from scar.compiler.compile_pipeline_implementation import open_pipeline_implementation, SCHEMA as PREPARATION
    from scar.optimization.schedule import build_region_schedule
    reference = preparation if isinstance(preparation, dict) else artifact_reference(Path(preparation).resolve(), PREPARATION)
    directory = Path(out).resolve()
    _require(not directory.exists(), 'fresh selected implementation native directory required')
    with open_pipeline_implementation(reference) as (_, pipeline, memory, target, problem, execution, certificate):
        _require(certificate is not None, 'UNKNOWN/incomplete implementation costs cannot compile a selected plan')
        plans = _selected_plans(pipeline, memory, target, problem, execution, certificate)
        directory.mkdir(parents=True)
        operators, shared = [], {}
        for call in pipeline.calls:
            plan = plans[call.id]; key = semantic_sha256(leaf_document(plan))
            if key not in shared:
                shared[key] = cpp.build(plan, directory/'operators'/('operator-'+str(len(shared))),
                    resource_schedule=build_region_schedule(plan, target))
            operator = shared[key]
            original._compatible(call, operator, memory)
            _require(operator.plan.cpu_threads == 1 and operator.plan.resource_binding.cpu_ids ==
                (certificate['call_cpu_ids'][call.id],), 'actual operator placement differs from solver choice')
            operators.append(operator)
        bindings = [original._operator_receipt(call, operator) for call,operator in zip(pipeline.calls, operators)]
        source = _source(pipeline, memory, operators, certificate)
        lifecycle_receipt=None
        if problem.activities:
            from scar.codegen.pipeline_lifecycle import lifecycle_delivery_proof
            lifecycle_receipt=lifecycle_delivery_proof(pipeline,memory,problem,execution,certificate,source)
        header = (_DIRECTORY/'scar_abi_v1.h').read_bytes()
        compiler = shlex.split(os.environ.get('CXX', 'g++'))
        _require(compiler and shutil.which(compiler[0]), 'original compiler unavailable')
        options = [*cpp.COMPILER_OPTIONS, '-pthread']
        fingerprint = implementation_sha256()
        build_id = _hash(source+header+_json(bindings)+fingerprint.encode()+_json(compiler)+_json(options))
        library = directory/('pipeline_'+build_id[:20]+'.so')
        files = dict(source='pipeline.cpp', pipeline='pipeline.json', memory_plan='memory-plan.json',
            execution_schedule='execution-schedule.json', abi_header='scar_abi_v1.h', library=library.name)
        for name, raw in ((files['source'],source), (files['pipeline'],_json(document(pipeline))),
            (files['memory_plan'],_json(memory)), (files['execution_schedule'],_json(certificate)),
            (files['abi_header'],header)):
            (directory/name).write_bytes(raw)
        fd, temporary = tempfile.mkstemp(prefix='.pipeline-', suffix='.so', dir=directory); os.close(fd)
        try:
            process = subprocess.run([*compiler, *options, str(directory/files['source']), '-o', temporary, '-lm'],
                capture_output=True, text=True)
            if process.returncode:
                raise RuntimeError('selected CPU pipeline compilation failed:\n'+process.stderr.strip())
            os.replace(temporary, library)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        metadata = dict(schema=SCHEMA, schema_version=1, pipeline_abi_version=1,
            preparation=reference, build_id=build_id, pipeline_sha256=pipeline_digest(pipeline),
            memory_plan_sha256=semantic_sha256(memory), execution_family=FAMILY,
            execution_schedule_sha256=semantic_sha256(certificate),
            implementation_sha256=fingerprint, files=files,
            sha256={key:_hash((directory/name).read_bytes()) for key,name in files.items()},
            compiler=compiler, compiler_options=options, operators=bindings,
            workspace_bytes=memory['total_workspace_bytes'], final_bytes=memory['final_bytes'],
            memory_budget_bytes=memory['memory_budget_bytes'], memory_policy='UNBOUND',
            unique_operator_artifacts=len(shared), cost_scope=certificate['cost_scope'],
            implementation_scope=certificate['implementation_scope'], full_model_timeline_implemented=False,
            model_makespan_is_native_runtime_prediction=False, native_host_optimality_proven=False,
            target_program_executed=False, performance_measured=False, deployment_authorized=False,
            default_backend='NO_OP', automatic_caller_installation_supported=False,
            failure_policy='JOIN_ALL_STARTED_WORKERS_WITHHOLD_OUTPUT_THEN_EXTERNAL_CALLER_FALLBACK',
            existing_host_envelope=certificate['existing_host_envelope'])
        if lifecycle_receipt is not None:metadata['host_lifecycle']=lifecycle_receipt
        (directory/'artifact.json').write_bytes(_json(metadata))
        return _instance(directory, metadata, pipeline, memory, operators, certificate)


def load(out):
    """Source/model/selection replay and exact C++ reconstruction before CDLL."""
    from scar.compiler.compile_pipeline_implementation import open_pipeline_implementation
    directory = Path(out).resolve()
    if directory.is_file():
        directory = directory.parent
    metadata = json.loads((directory/'artifact.json').read_bytes())
    _require((metadata.get('schema'),metadata.get('schema_version'),metadata.get('pipeline_abi_version')) ==
        (SCHEMA,1,1) and metadata['implementation_sha256'] == implementation_sha256()
        and metadata['execution_family'] == FAMILY and metadata['deployment_authorized'] is False
        and metadata['default_backend'] == 'NO_OP'
        and metadata['full_model_timeline_implemented'] is False
        and metadata['model_makespan_is_native_runtime_prediction'] is False
        and metadata['native_host_optimality_proven'] is False
        and metadata['target_program_executed'] is False
        and metadata['performance_measured'] is False
        and metadata['automatic_caller_installation_supported'] is False,
        'unsupported or stale implementation artifact')
    for key,name in metadata['files'].items():
        _require(Path(name).name == name and _hash((directory/name).read_bytes()) == metadata['sha256'][key],
            'selected implementation artifact bytes differ: '+key)
    with open_pipeline_implementation(metadata['preparation']) as (_, pipeline, memory, target, problem, execution, certificate):
        _require(certificate is not None, 'selected source/domain admission required')
        plans = _selected_plans(pipeline, memory, target, problem, execution, certificate)
        _require(len(metadata['operators']) == len(pipeline.calls), 'operator count differs')
        operators = []
        for call,receipt in zip(pipeline.calls, metadata['operators']):
            for file in receipt['files'].values():
                _require(_hash(Path(file['path']).read_bytes()) == file['sha256'], 'selected operator bytes differ')
            operator = cpp.load(receipt['artifact_directory'])
            original._compatible(call, operator, memory)
            _require(semantic_sha256(leaf_document(operator.plan)) == semantic_sha256(leaf_document(plans[call.id]))
                and original._operator_receipt(call, operator) == receipt,
                'actual leaf is not the chosen call/CPU implementation')
            operators.append(operator)
        source = _source(pipeline, memory, operators, certificate)
        lifecycle_receipt=None
        if problem.activities:
            from scar.codegen.pipeline_lifecycle import lifecycle_delivery_proof
            lifecycle_receipt=lifecycle_delivery_proof(pipeline,memory,problem,execution,certificate,source)
        _require(metadata.get('host_lifecycle')==lifecycle_receipt,
            'conditional worker lifecycle source receipt differs')
        _require(source == (directory/metadata['files']['source']).read_bytes()
            and _json(document(pipeline)) == (directory/metadata['files']['pipeline']).read_bytes()
            and _json(memory) == (directory/metadata['files']['memory_plan']).read_bytes()
            and _json(certificate) == (directory/metadata['files']['execution_schedule']).read_bytes()
            and (_DIRECTORY/'scar_abi_v1.h').read_bytes() == (directory/metadata['files']['abi_header']).read_bytes(),
            'source/pool/certificate/header reconstruction differs')
        _require(metadata['pipeline_sha256'] == pipeline_digest(pipeline)
            and metadata['memory_plan_sha256'] == semantic_sha256(memory)
            and metadata['execution_schedule_sha256'] == semantic_sha256(certificate)
            and metadata['compiler_options'] == [*cpp.COMPILER_OPTIONS,'-pthread'], 'native source/control identity differs')
        identity = _hash(source+(_DIRECTORY/'scar_abi_v1.h').read_bytes()+_json(metadata['operators'])
            +metadata['implementation_sha256'].encode()+_json(metadata['compiler'])+_json(metadata['compiler_options']))
        _require(identity == metadata['build_id'], 'selected implementation build identity differs')
        return _instance(directory, metadata, pipeline, memory, operators, certificate)


__all__ = ['build', 'load', 'render_implementation_source', 'implementation_sha256', 'SCHEMA']
