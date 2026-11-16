"""One private pool across fixed, individually transactional original steps.

The source plan and caller-loop authority are independently replayed. This
backend does not replace an application loop by itself or hoist its guards,
thread teams, snapshots, arithmetic, providers, restoration, or commits.
"""
from copy import deepcopy
import ctypes
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

from scar.codegen import cpp, stateful as backend
from scar.compiler.stateful_model import document, stateful_digest
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.record_codec import encode
from scar.runtime.native_stateful_repeat import RepeatGuardFailure

SCHEMA = 'scar.native-stateful-workspace-repeat'
_HERE = Path(__file__).resolve().parent
_TEMPLATE = _HERE/'stateful_workspace_v1.cpp.in'


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _hash(raw):
    return sha256(raw).hexdigest()


def _digest(value):
    return _hash(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def _json(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def implementation_sha256():
    return _digest(dict(base=backend.implementation_sha256(), files={name: _hash((_HERE/name).read_bytes())
        for name in ('stateful_workspace.py', 'stateful_workspace_v1.cpp.in', 'stateful_repeat_v1.cpp.in')}))


def _reference(path):
    path = Path(path).resolve()
    if path.is_dir():
        path /= 'result.json'
    return artifact_reference(path, 'scar.workspace-preparation')


def _replay(preparation):
    from scar.compiler.prepare_workspace import replay_workspace_preparation
    from scar.optimization.stateful_workspace import verify_workspace_reuse
    reference = _reference(preparation)
    qualified = replay_workspace_preparation(reference['uri'])
    report, region, step_plan, workspace_plan, context = qualified
    _need(read_artifact(reference) == report, 'original workspace preparation changed during replay')
    _need(verify_workspace_reuse(region, step_plan, workspace_plan,
              loop_plan=context.get('loop_plan')) == workspace_plan,
          'workspace-lifetime plan differs from independent source/initialization replay')
    region, step_plan, binding = backend._admit(region, step_plan)
    _need(workspace_plan['trip_count'] == report['trip_count']
          and type(workspace_plan['trip_count']) is int and 0 <= workspace_plan['trip_count'] < 2**64,
          'one exact planned repetition count required')
    _need(read_artifact(report['workspace_plan']) == workspace_plan,
          'retained workspace plan differs from the original preparation')
    return reference, (report, region, step_plan, workspace_plan, context), binding


def _base_metadata(reference, region, plan, binding):
    """Original loader-equivalent source/byte admission without CDLL/providers."""
    reference = encode(reference)
    _need(reference['schema'] == backend.SCHEMA, 'unmodified original stateful base manifest required')
    metadata = read_artifact(reference); directory = Path(reference['uri']).parent
    _need((metadata.get('schema'), metadata.get('schema_version'), metadata.get('stateful_abi_version'))
          == (backend.SCHEMA, 1, 1), 'original stateful ABI1 base required')
    expected_keys = {'region', 'plan', 'source', 'library', 'stateful_abi_v1.h', 'scar_abi_v1.h'}
    _need(set(metadata['files']) == expected_keys and set(metadata['sha256']) == expected_keys,
          'complete original stateful base members required')
    for key, name in metadata['files'].items():
        _need(type(name) is str and Path(name).name == name
              and _hash((directory/name).read_bytes()) == metadata['sha256'][key],
              'original stateful base checksum differs: '+key)
    _need(_digest(json.loads((directory/metadata['files']['region']).read_bytes())) == stateful_digest(region)
          and json.loads((directory/metadata['files']['plan']).read_bytes()) == plan,
          'parent native region/plan differ from the qualified workspace source')
    contract = dict(region_sha256=stateful_digest(region), plan_sha256=_digest(plan),
                    provider_runtime=metadata['provider_runtime'])
    _need(all(metadata[key] == value for key, value in contract.items())
          and metadata['contract_sha256'] == _digest(contract)
          and metadata['implementation_sha256'] == backend.implementation_sha256(),
          'original stateful base contract or current implementation differs')
    source = backend._source(region, plan, binding, contract).encode()
    _need(source == (directory/metadata['files']['source']).read_bytes(),
          'parent base C++ is not the unchanged original renderer output')
    headers = {name: (_HERE/name).read_bytes() for name in ('stateful_abi_v1.h', 'scar_abi_v1.h')}
    _need(all((directory/metadata['files'][name]).read_bytes() == raw for name, raw in headers.items()),
          'original stateful base ABI headers differ')
    compiler = metadata['compiler']
    build_id = _digest(dict(source=_hash(source), headers={name: _hash(raw) for name, raw in headers.items()},
        compiler=compiler['argv'], identity=compiler['identity'], options=compiler['options'],
        implementation_sha256=metadata['implementation_sha256'], contract=contract))
    _need(build_id == metadata['build_id']
          and compiler['options'] == [*cpp.COMPILER_OPTIONS, *(['-fopenmp'] if plan['cpu_threads'] > 1 else [])]
          and ('g++' in compiler['identity'].lower() or 'gcc' in compiler['identity'].lower()),
          'original GNU base build identity or C++20 options differ')
    provider = metadata['provider_runtime']['loaded_binary']
    raw = Path(provider['path']).read_bytes()
    _need(len(raw) == provider['bytes'] and _hash(raw) == provider['sha256'],
          'original NumPy provider binary changed')
    read_artifact(reference)
    return metadata, source, headers


def _source(original_source, metadata, plan, workspace_plan, execution_sha):
    """Preserve original source; clone only its explicit workspace lifetime."""
    text = original_source.decode(); template = (_HERE/'stateful_v1.cpp.in').read_text()
    start_marker = 'int32_t run(const scar_array_v1 *arrays, uint64_t array_count,'
    end_marker = '\nconst scar_stateful_table_v1 table ='
    _need(text.count(start_marker) == text.count(end_marker) == 1,
          'one unchanged original run/table source skeleton required')
    start, end = text.index(start_marker), text.index(end_marker)
    original_run = text[start:end]
    signature_end = original_run.index(') {')+3
    signature = original_run[:signature_end]
    declaration_marker = '    std::array<std::unique_ptr<double[]>, @SLOT_COUNT@> slots;'
    helper_marker = '    ThreadAffinity caller_affinity;'
    _need(template.count(declaration_marker) == template.count(helper_marker) == 1,
          'unique original allocation/helper source markers required')
    pool_template = template[template.index(declaration_marker):template.index(helper_marker)]
    pool = pool_template.replace('@SLOT_COUNT@', str(len(plan['workspace_slots'])))
    _need('@' not in pool and original_run.count(pool) == 1,
          'original declaration/allocation fragment is not uniquely retained')
    pool_start = original_run.index(pool)
    prefix = original_run[signature_end:pool_start]
    clone_signature = signature.replace('int32_t run(', 'int32_t run_with_workspace(', 1)
    clone_signature = (clone_signature[:-3]+',\n            std::array<std::unique_ptr<double[]>, '
        +str(len(plan['workspace_slots']))+'> &slots) {')
    body = original_run[signature_end:]
    clone_body = body.replace(pool, '', 1)
    # The original table dispatch preserves a separately optimized step body.
    # Keep that compilation boundary after lifetime hoisting: unrestricted IPA
    # can inline this large body into the repetition control and lose otherwise
    # legal loop vectorization. This changes no mathematical statement or guard.
    control_attributes = '__attribute__((noinline, noipa))\n'
    cloned_run = control_attributes+clone_signature+clone_body
    inverse = cloned_run.removeprefix(control_attributes)
    _need(inverse.startswith(clone_signature), 'workspace clone compilation-boundary source changed')
    inverse = inverse[len(clone_signature):]
    _need(signature+inverse[:len(prefix)]+pool+inverse[len(prefix):] == original_run,
          'workspace clone changed a statement outside the unique pool-lifetime block')
    repeat_path = _HERE/'stateful_repeat_v1.cpp.in'; repeat = repeat_path.read_text()
    first = '    if (!table || table->abi_version != 1 ||'
    last = '    for (uint64_t step = 0; step < repetitions; ++step) {'
    _need(repeat.count(first) == repeat.count(last) == 1,
          'original parent-table ABI check markers changed')
    table_guard = repeat[repeat.index(first):repeat.index(last)]
    for key in ('region_sha256', 'plan_sha256', 'contract_sha256'):
        table_guard = table_guard.replace('@'+key.upper()+'@', metadata[key])
    _need('@' not in table_guard, 'unresolved parent-table ABI check')
    substitutions = dict(ORIGINAL_SOURCE=text, WORKSPACE_RUN=cloned_run,
        EXECUTION_SHA256=execution_sha, TRIP_COUNT=str(workspace_plan['trip_count']),
        PARENT_TABLE_GUARD=table_guard.rstrip(), PREFLIGHT=prefix.rstrip(), ALLOCATION=pool.rstrip())
    source = _TEMPLATE.read_text()
    for key, value in substitutions.items():
        _need(source.count('@'+key+'@') == 1, 'one workspace adapter template token required')
        source = source.replace('@'+key+'@', value)
    _need('@' not in source and source.startswith(text), 'original run/getter source changed or unresolved tokens remain')
    _need(source.count(prefix) >= 3 and source.count(pool) == 2,
          'original per-step preflight or exact outer allocation block was changed')
    nonzero = sum(row['capacity_bytes'] > 0 for row in plan['workspace_slots'])
    mapping = dict(original_full_source_sha256=_hash(original_source), original_run_sha256=_hash(original_run.encode()),
        original_run_source_range=dict(char_start=start, char_end_exclusive=end),
        cloned_run_sha256=_hash(cloned_run.encode()), original_signature_sha256=_hash(signature.encode()),
        clone_signature_sha256=_hash(clone_signature.encode()),
        control_attributes=['noinline', 'noipa'], control_attributes_sha256=_hash(control_attributes.encode()),
        control_attributes_removed_for_inverse=True,
        step_compilation_boundary='PRESERVE_ORIGINAL_SEPARATE_ABI_STEP_BODY_OPTIMIZATION',
        pool_block_sha256=_hash(pool.encode()), pool_block_char_start=pool_start,
        pool_block_char_end_exclusive=pool_start+len(pool), preflight_sha256=_hash(prefix.encode()),
        original_body_without_pool_sha256=_hash(clone_body.encode()),
        parent_table_check_sha256=_hash(table_guard.encode()),
        original_run_and_getter_copied_verbatim=True,
        reinstating_pool_and_original_signature_restores_original_run=True,
        outer_preflight_copied_verbatim=True, per_step_preflight_retained=True,
        mathematical_body_changed=False, per_step_commits_retained=True,
        per_step_teams_and_restoration_retained=True, positive_capacity_slot_count=nonzero,
        zero_capacity_slots_issue_no_new_array_request=True,
        native_math_backend='UNCHANGED_ORIGINAL_STATEFUL_ABI1', target_loop_replacement_authorized=False)
    return source.encode(), mapping


def _derive(reference, qualified, binding, base_reference):
    report, region, plan, workspace_plan, context = qualified
    base_reference = encode(base_reference)
    declared = context.get('base_native_reference')
    _need(declared is None or encode(declared) == base_reference,
          'workspace parent base differs from original source preparation')
    base_metadata, original_source, headers = _base_metadata(base_reference, region, plan, binding)
    workspace_sha = _digest(workspace_plan)
    base_ids = {key: base_metadata[key] for key in ('build_id', 'region_sha256', 'plan_sha256', 'contract_sha256')}
    execution_plan = dict(schema='scar.stateful-workspace-execution-plan', schema_version=1,
        preparation=reference, workspace_plan_sha256=workspace_sha, base_native_reference=base_reference,
        base_ids=base_ids, trip_count=workspace_plan['trip_count'],
        scope='PRIVATE_POOL_LIFETIME_ONLY_EACH_ORIGINAL_STEP_REMAINS_TRANSACTIONAL',
        original_caller_loop_replacement_authorized=False)
    execution_sha = _digest(execution_plan)
    source, mapping = _source(original_source, base_metadata, plan, workspace_plan, execution_sha)
    compiler = deepcopy(base_metadata['compiler']); fingerprint = implementation_sha256()
    identity = dict(source_sha256=_hash(source), headers={name: _hash(raw) for name, raw in headers.items()},
        compiler=compiler, implementation_sha256=fingerprint, execution_plan=execution_plan, source_mapping=mapping)
    return dict(reference=reference, qualified=qualified, binding=binding, base_reference=base_reference,
        base_metadata=base_metadata, headers=headers, source=source, source_mapping=mapping,
        execution_plan=execution_plan, execution_sha=execution_sha, workspace_sha=workspace_sha,
        compiler=compiler, fingerprint=fingerprint, identity=identity, build_id=_digest(identity))


def _metadata(derived, directory):
    report, region, plan, workspace_plan, context = derived['qualified']
    files = dict(region='region.json', step_plan='step-plan.json', workspace_plan='workspace-plan.json',
        source='workspace.cpp', library='workspace_'+derived['build_id'][:20]+'.so',
        **{name: name for name in derived['headers']})
    positive = derived['source_mapping']['positive_capacity_slot_count']; count = workspace_plan['trip_count']
    return dict(schema=SCHEMA, schema_version=1, stateful_abi_version=1, repeat_abi_version=1, cpp_standard=20,
        build_id=derived['build_id'], identity=derived['identity'], implementation_sha256=derived['fingerprint'],
        preparation=derived['reference'], base_native_reference=derived['base_reference'],
        base_ids=derived['execution_plan']['base_ids'], workspace_plan_sha256=derived['workspace_sha'],
        execution_plan_sha256=derived['execution_sha'], execution_plan=derived['execution_plan'],
        source_mapping=derived['source_mapping'], files=files,
        sha256={key: _hash((directory/name).read_bytes()) for key, name in files.items()},
        compiler=derived['compiler'], cpu_threads=plan['cpu_threads'], cpu_ids=list(derived['binding'].cpu_ids),
        memory={key: plan[key] for key in ('external_array_bytes', 'workspace_payload_bytes',
            'total_region_array_payload_bytes', 'memory_budget_bytes', 'budget_scope')},
        trip_count=count, positive_capacity_slot_count=positive,
        source_derived_success_path_requests=dict(original_new_array_requests=count*positive,
            workspace_new_array_requests=positive if count else 0,
            original_pool_releases=count*positive, workspace_pool_releases=positive if count else 0),
        commit_policy='EACH_SUCCESSFUL_ORIGINAL_ABI_CALL_COMMITS_BEFORE_NEXT_STEP',
        failure_policy='STOP_BEFORE_FAILED_STEP_COMMIT_RESUME_ONLY_UNCOMMITTED_SUFFIX',
        sequence='FIXED_INPUTS_COMPOSITION_CALLER_LOOP_REPLACEMENT_REQUIRES_SEPARATE_SOURCE_GUARD',
        native_math_changed=False, allocations_hoisted=True, teams_hoisted=False,
        per_step_guards_and_snapshots_retained=True, original_run_getter_retained=True,
        external_assumptions=deepcopy(plan['external_assumptions']),
        target_loop_replacement_authorized=False, trial_only=True, cost_status='UNKNOWN',
        default_decision='NO_OP', selected_for_use=False, deployment_authorized=False,
        performance_measured=False, physical_optimality_proven=False)


@dataclass
class WorkspaceRepeatArtifact:
    base: object
    directory: Path
    _metadata: dict
    _library: object
    _repeat: object
    _last_execution: object = None

    @property
    def metadata(self):
        return deepcopy(self._metadata)

    @property
    def report(self):
        return deepcopy({**self._metadata, 'last_execution': self._last_execution})

    def run(self, arrays, scalars, repetitions):
        if type(repetitions) is not int or not 0 <= repetitions < 2**64:
            raise ValueError('exact nonnegative uint64 repetition count required')
        if repetitions == 0:
            result = dict(requested=0, completed=0, native_calls=0, status=0)
            self._last_execution = result
            return dict(result)
        if repetitions != self._metadata['trip_count']:
            self._last_execution = dict(requested=repetitions, completed=0, native_calls=0, status=None)
            raise RepeatGuardFailure(repetitions, 0, None, 'count differs from the exact workspace plan')
        progress = ctypes.c_uint64(); status = None
        def invoke(*args):
            nonlocal status
            status = self._repeat(self.base._table, *args, repetitions, ctypes.byref(progress))
            return status
        facade = replace(self.base, _table=SimpleNamespace(contents=SimpleNamespace(run=invoke)))
        try:
            facade.run(arrays, scalars)
        except backend.GuardFailure as error:
            _need(progress.value <= repetitions, 'native workspace cursor exceeds requested count')
            self._last_execution = dict(requested=repetitions, completed=progress.value,
                native_calls=progress.value, status=status)
            raise RepeatGuardFailure(repetitions, progress.value, status, str(error)) from error
        _need(status == 0 and progress.value == repetitions,
              'native workspace returned success without complete committed progress')
        result = dict(requested=repetitions, completed=progress.value, native_calls=progress.value, status=0)
        self._last_execution = result
        return dict(result)

    def run_with_fallback(self, arrays, scalars, repetitions, original_step):
        """Explicit composition helper; the source-loop guard owns caller authority."""
        try:
            result = self.run(arrays, scalars, repetitions)
            return {**result, 'fallback_calls': 0, 'native_status': 0}
        except RepeatGuardFailure as error:
            for _ in range(error.completed, repetitions):
                original_step()
            return dict(requested=repetitions, completed=repetitions, native_calls=error.completed,
                fallback_calls=repetitions-error.completed, native_status=error.status)


def build_workspace_repeat(preparation, out):
    reference, qualified, binding = _replay(preparation)
    directory = Path(out).resolve()
    _need(not directory.exists(), 'fresh workspace-repeat artifact directory required')
    report, region, plan, workspace_plan, context = qualified
    base_reference = context.get('base_native_reference')
    directory.mkdir(parents=True)
    if base_reference is None:
        base = backend.build(region, directory/'base-native', plan=plan, resource_binding=binding)
        base_reference = artifact_reference(base.directory/'artifact.json', backend.SCHEMA)
    derived = _derive(reference, qualified, binding, base_reference)
    compiler = derived['compiler']
    _need(compiler['argv'] and shutil.which(compiler['argv'][0]), 'original GNU compiler unavailable')
    identity = subprocess.run([*compiler['argv'], '--version'], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    _need(identity == compiler['identity'], 'original GNU compiler identity changed')
    for name, value in (('region.json', document(region)), ('step-plan.json', plan), ('workspace-plan.json', workspace_plan)):
        (directory/name).write_bytes(_json(value))
    (directory/'workspace.cpp').write_bytes(derived['source'])
    for name, raw in derived['headers'].items():
        (directory/name).write_bytes(raw)
    library = directory/('workspace_'+derived['build_id'][:20]+'.so')
    process = subprocess.run([*compiler['argv'], *compiler['options'], str(directory/'workspace.cpp'),
        '-o', str(library), '-lm'], capture_output=True, text=True)
    (directory/'compiler-stdout.txt').write_text(process.stdout)
    (directory/'compiler-stderr.txt').write_text(process.stderr)
    if process.returncode:
        raise RuntimeError('workspace-repeat C++ compilation failed:\n'+process.stderr)
    _need(implementation_sha256() == derived['fingerprint'], 'workspace source changed during compilation')
    read_artifact(reference)
    (directory/'artifact.json').write_bytes(_json(_metadata(derived, directory)))
    return load_workspace_repeat(directory)


def _inspect(out):
    directory = Path(out).resolve()
    if directory.is_file():
        directory = directory.parent
    metadata = json.loads((directory/'artifact.json').read_bytes())
    _need((metadata.get('schema'), metadata.get('schema_version'), metadata.get('stateful_abi_version'),
           metadata.get('repeat_abi_version')) == (SCHEMA, 1, 1, 1), 'workspace-repeat ABI1 artifact required')
    for key, name in metadata['files'].items():
        _need(type(name) is str and Path(name).name == name
              and _hash((directory/name).read_bytes()) == metadata['sha256'][key],
              'workspace artifact checksum differs: '+key)
    reference, qualified, binding = _replay(metadata['preparation']['uri'])
    _need(reference == metadata['preparation'], 'workspace preparation receipt differs')
    derived = _derive(reference, qualified, binding, metadata['base_native_reference'])
    _need(metadata == _metadata(derived, directory), 'workspace metadata differs from full source/plan replay')
    report, region, plan, workspace_plan, context = qualified
    _need((directory/'region.json').read_bytes() == _json(document(region))
          and (directory/'step-plan.json').read_bytes() == _json(plan)
          and (directory/'workspace-plan.json').read_bytes() == _json(workspace_plan)
          and (directory/'workspace.cpp').read_bytes() == derived['source']
          and all((directory/name).read_bytes() == raw for name, raw in derived['headers'].items()),
          'workspace region/plan/header/C++ bytes differ from exact source reconstruction')
    read_artifact(reference)
    return directory, metadata, qualified


def inspect_workspace_repeat(out):
    """One full source/byte admission; no CDLL or provider-capsule resolution."""
    _, metadata, qualified = _inspect(out)
    return deepcopy(metadata), qualified


def load_workspace_repeat(out):
    directory, metadata, qualified = _inspect(out)
    base = backend.load(Path(metadata['base_native_reference']['uri']).parent)
    _need({key: base.report[key] for key in metadata['base_ids']} == metadata['base_ids'],
          'loaded original parent table identities differ')
    library = ctypes.CDLL(str(directory/metadata['files']['library']))
    getter = library.scar_workspace_plan_sha256_v1; getter.argtypes = []; getter.restype = ctypes.c_char_p
    _need(getter() == metadata['execution_plan_sha256'].encode(), 'workspace embedded execution plan differs')
    original = library.scar_get_stateful_table_v1
    original.argtypes = []; original.restype = ctypes.POINTER(backend._Table); table = original()
    _need(table and table.contents.abi_version == 1 and table.contents.struct_size == ctypes.sizeof(backend._Table)
          and table.contents.run, 'retained original ABI table differs')
    for key in ('region_sha256', 'plan_sha256', 'contract_sha256'):
        _need(getattr(table.contents, key) == metadata['base_ids'][key].encode(),
              'retained original embedded table identity differs: '+key)
    _need((table.contents.array_count, table.contents.scalar_count, table.contents.provider_count)
          == (base._table.contents.array_count, base._table.contents.scalar_count, base._table.contents.provider_count),
          'retained original ABI input counts differ')
    run = library.scar_workspace_repeat_stateful_v1
    run.argtypes = [ctypes.POINTER(backend._Table), ctypes.POINTER(cpp._Array), ctypes.c_uint64,
        ctypes.POINTER(backend._Scalar), ctypes.c_uint64, ctypes.POINTER(backend._Provider), ctypes.c_uint64,
        ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64)]
    run.restype = ctypes.c_int32
    return WorkspaceRepeatArtifact(base, directory, metadata, library, run)


__all__ = ['SCHEMA', 'WorkspaceRepeatArtifact', 'RepeatGuardFailure', 'implementation_sha256',
           'build_workspace_repeat', 'load_workspace_repeat', 'inspect_workspace_repeat']
