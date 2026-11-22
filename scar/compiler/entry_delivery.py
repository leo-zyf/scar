"""Replay exact CPU delivery routes against a whole-process build scope.

A profile owns its legality and evidence reader. Qualification does not select
a route or run the target entry. Pair metrics describe retained observations,
and only an existing compact release authority can authorize execution.
"""
from hashlib import sha256
import json
from pathlib import Path
import re
from subprocess import SubprocessError

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.paired_entry_selection import (
    DEFAULT_MINIMUM_GAIN, compare_entry_pairs, replay_entry_pairs,
)


BOUNDARY = 'PROCESS_LAUNCH_THROUGH_EXIT'
QUALIFICATION = 'scar.entry-delivery-qualification'
HANDOFF = 'scar.entry-delivery-handoff'
ATEN = 'scar.aten-recurrence-delivery'
CALLER_PAYLOAD = 'scar.caller-pipeline-final-payload'
CALLER_EXPORT = 'scar.caller-pipeline-final-slot-export'
STATEFUL = 'scar.final-entry-capsule-payload'
SPARSE = 'scar.native-parameterized-sparse-executable'
NVHPC = 'scar.native-exported-nvhpc-cpu'
NVHPC_DEVICE = 'scar.native-exported-aten-cuda'
NVHPC_DEVICE_ROUTE = 'NVHPC_SOURCE_DEVICE_ENTRY'
_HEX = re.compile('[0-9a-f]{64}')
_PROFILE_ERRORS = (OSError, ValueError, KeyError, TypeError, IndexError, ImportError, SubprocessError)
_ORIGINAL_OPTIONS = (None, 'NO_OP', 'KEEP_ORIGINAL', 'original')


def _require(test, reason):
    if not test:
        raise ValueError(reason)


def _matches(reference, schema):
    _require(reference['schema'] == schema, 'unexpected evidence schema for delivery route')
    return read_artifact(reference)


def _ref_with_path(row, schema):
    return dict(uri=row['path'], sha256=row['sha256'],
                size_bytes=row['size_bytes'] if 'size_bytes' in row else row['bytes'], schema=schema)


def _entry(scope, path, digest, argv=None):
    _require((path, digest) == (scope['entry'], scope['entry_sha256'])
             and (argv is None or argv == scope['argv']), 'entry source or arguments differ from the delivery scope')


def _target(scope, reference):
    _require(scope['target_sha256'] == reference['sha256'], 'hardware target digest differs from the delivery scope')
    read_artifact(reference)


def _budget(scope, required):
    _require(type(required) is int and 0 < required <= scope['memory_budget_bytes'],
             'delivery resource bound exceeds the requested memory budget')


def _context(scope, observed):
    _entry(scope, observed['entry_path'], observed['entry_sha256'], observed['argv'])
    for name in ('working_directory', 'python_executable', 'process_affinity_cpu_ids',
                 'native_thread_environment'):
        if name in scope:
            _require(scope[name] == observed[name], 'frozen delivery context differs for ' + name)


def _frozen_environment(scope, observed_sha256):
    if 'environment_sha256' in scope:
        _require(scope['environment_sha256'] == observed_sha256, 'paired validation environment differs from the entry scope')


def _environment_digest(environment):
    return sha256(json.dumps(sorted(environment.items()), separators=(',', ':')).encode()).hexdigest()


def _aten(scope, reference, document):
    from scar.compiler.aten_entry_delivery import ABI_FILE, BINDING_FILE, LAUNCHER, _inputs
    from scar.runtime.validate_native_project import verify_native_project

    root = Path(reference['uri']).parent
    _require(root / 'artifact.json' == Path(reference['uri']), 'standalone delivery requires the exact project manifest')
    abi_ref = _ref_with_path(dict(path=str(root / ABI_FILE), **document['files'][ABI_FILE]),
                             'scar.fixed-grid-file-entry')
    binding_path = root / BINDING_FILE
    binding_schema = 'scar.aten-source-entry-binding'
    binding_ref = _ref_with_path(dict(path=str(binding_path), **document['files'][BINDING_FILE]), binding_schema)
    # The standalone project's cold checker also independently derives these files.
    abi, binding = read_artifact(abi_ref), read_artifact(binding_ref)
    source = _ref_with_path(abi['solver']['source'], 'scar.python-source')
    _entry(scope, source['uri'], source['sha256'])
    _target(scope, binding['target'])
    _budget(scope, binding['storage']['peak_bytes'])
    _require(binding['storage']['budget_bytes'] == scope['memory_budget_bytes'],
             'standalone delivery was planned for a different memory budget')
    argv = scope['argv']
    _require(len(argv) == abi['argv']['count'], 'arguments do not satisfy the standalone source ABI')
    steps = argv[abi['argv']['steps'] - 1]
    _require(re.fullmatch('[0-9]+', steps) and len(steps.lstrip('0')) <= 10
             and abi['domain']['step_count_min'] <= int(steps.lstrip('0') or '0') <= abi['domain']['step_count_max'],
             'standalone steps are outside the exact signed-int32 ABI')
    inputs = Path(argv[abi['argv']['input_root'] - 1])
    output = Path(argv[abi['argv']['output'] - 1])
    _require(inputs.is_absolute() and output.is_absolute() and output.parent.is_dir()
             and not output.exists() and not output.is_symlink(), 'standalone source ABI requires absolute inputs and fresh output')
    inputs = _inputs(abi, inputs)
    _, checked = verify_native_project(root, reference['sha256'])
    _require(checked['source_entry_cli_replayed'] is True, 'standalone source proof did not replay')
    return dict(kind='STANDALONE_CPP', target=binding['target'], plan=binding_ref, source=source, file_inputs=inputs,
                launcher=artifact_reference(root / LAUNCHER, 'scar.native-source-launcher'),
                cpu_ids=document['cpus'], memory=binding['storage'], scoped_cost_evidence=binding['costs'],
                steps=int(steps.lstrip('0') or '0'), python_runtime_required=False)


def _caller(scope, reference, document):
    from scar.runtime.caller_pipeline_capsule import TRUST, implementation_identity
    from scar.runtime.caller_pipeline_deployment import bundle_reference

    payload_ref = document['payload'] if reference['schema'] == CALLER_EXPORT else reference
    payload = _matches(payload_ref, CALLER_PAYLOAD)
    _context(scope, payload['entry_context'])
    _require(payload['schema_version'] == 1 and payload['trust_policy'] == TRUST
             and payload['policy']['frozen_before_measurement'] is True
             and payload['capsule_directory'] == str(Path(payload_ref['uri']).parent),
             'caller payload is outside the frozen release protocol')
    bundle = payload['bundle']
    rebuilt, metadata = bundle_reference(bundle['directory'], bundle['admission_sha256'])
    _require(rebuilt == bundle, 'caller guarded C++ bundle changed')
    targets = metadata['admission']['original_targets']
    _require(len(targets) == 1, 'one original hardware target required for the caller bundle')
    _target(scope, targets[0])
    memory = metadata['contract']['memory']
    _budget(scope, memory['peak_bytes'])
    _require(memory['budget_bytes'] == scope['memory_budget_bytes'],
             'caller bundle was planned for a different memory budget')
    for row in payload['source_refs'] + [payload['validator_implementation'], payload['original_regional_proof']]:
        read_artifact(row, json_document=False)
    _require(payload['runtime_identity'] == implementation_identity(), 'caller delivery runtime code changed')
    return dict(kind='PYTHON_ORCHESTRATED_CPP', target=targets[0], payload=payload_ref, bundle=bundle,
                publication=payload.get('publication'), memory=memory, python_runtime_required=True)


def _stateful(scope, reference, payload):
    from scar.runtime.compiled_stateful_capsule import verify_profile
    from scar.runtime.final_entry_capsule import STATEFUL_PROFILE, TRUST, runtime_identity

    _context(scope, payload['entry_context'])
    _require(payload['schema_version'] == 1 and payload['profile_kind'] == STATEFUL_PROFILE
             and payload['trust_policy'] == TRUST and payload['cost_status'] == 'UNMEASURED'
             and payload['deployment_authorized'] is False,
             'expected the unmeasured stateful final-entry protocol')
    profile = payload['artifact_profile']
    pipeline = _matches(profile['pipeline'], 'scar.program-compiler-pipeline')
    target = _ref_with_path(pipeline['hardware'], 'scar.target-hardware-graph')
    _target(scope, target)
    for row in profile['rows']:
        _require(row['memory_budget_bytes'] == scope['memory_budget_bytes'],
                 'stateful compiled region was planned for a different memory budget')
    for row in payload['source_refs'] + [payload['validator_implementation']]:
        read_artifact(row, json_document=False)
    _require(payload['runtime_identity'] == runtime_identity(STATEFUL_PROFILE), 'stateful delivery runtime code changed')
    verify_profile(profile)
    return dict(kind='PYTHON_ORCHESTRATED_CPP', target=target, pipeline=profile['pipeline'], rows=profile['rows'],
                memory_scope='ORIGINAL_STATEFUL_REGION_BUDGET', python_runtime_required=True)


def _sparse(scope, reference, document):
    from scar.optimization.native_parameterized_sparse_plan import SCHEMA, _load_plan, _native, work_for_count

    source = document['source']
    _entry(scope, source['uri'], source['sha256'])
    plan_ref = document['parameterized_plan']
    declared_plan = _matches(plan_ref, SCHEMA)
    target = declared_plan['resources']['placement']['hardware']
    _target(scope, target)
    memory = declared_plan['resources']['memory']
    _budget(scope, memory['minimum_launch_headroom_bytes'])
    argv = scope['argv']
    _require(len(argv) == 1 and re.fullmatch('[1-9][0-9]{0,9}', argv[0]),
             'runtime count does not satisfy the sparse source ABI')
    plan = _load_plan(plan_ref['uri'])
    work_for_count(plan, int(argv[0]))
    binary = dict(document['binary'], uri=str(Path(reference['uri']).parent / document['binary']['uri']))
    _native(dict(artifact=reference, executable=binary), plan, plan_ref)
    return dict(kind='STANDALONE_CPP', target=target, plan=plan_ref, source=source, executable=binary,
                memory=memory, python_runtime_required=False)


def _nvhpc(scope, reference, document):
    from scar.analysis.nvhpc_source_entry import SCHEMA
    from scar.compiler.compile_exported_nvhpc import check_nvhpc_project
    from scar.compiler.nvhpc_entry_variant import _cpu_cohorts, matrix_entry_inputs

    root, checked, execution, _, _ = check_nvhpc_project(Path(reference['uri']).parent, reference['sha256'])
    _require(checked == reference and 'entry_obligations_sha256' in scope,
             'standalone NVHPC source delivery requires its original F4 entry obligations')
    contract_ref = _ref_with_path(dict(path=str(root/'plan/source-entry.json'),
        **document['files']['plan/source-entry.json']), SCHEMA)
    contract = read_artifact(contract_ref)
    source = _ref_with_path(contract['source'], 'scar.python-source')
    _entry(scope, source['uri'], source['sha256'])
    problem = json.loads((root/'plan/problem.json').read_bytes())
    _target(scope, problem['target'])
    memory = execution['storage']
    _budget(scope, memory['serial_peak_bytes'])
    process_cpus = scope['process_affinity_cpu_ids']
    allowed = (_cpu_cohorts(Path(scope['target']['uri']).parent, process_cpus)
               if scope.get('search_cpu_cohorts', False) else [process_cpus])
    _require(memory['budget_bytes'] == scope['memory_budget_bytes'] and execution['cpu_ids'] in allowed,
             'NVHPC CPU cohort or memory budget differs from the entry scope')
    numerical = execution['numerical_contract']
    _require(numerical == scope.get('numerical_contract', {'kind': 'EXACT_BYTES'}),
             'NVHPC numerical contract differs from the original entry scope')
    inputs = matrix_entry_inputs(contract, scope['argv'])
    return dict(kind='STANDALONE_CPP', target=problem['target'], source=source, source_contract=contract_ref,
        operator_stack=document['stack'], file_inputs=inputs, numerical_contract=numerical,
        cpu_ids=execution['cpu_ids'], output_sizes=[row['nbytes'] for row in contract['outputs']],
        launcher=artifact_reference(root/'launch.sh', 'scar.native-source-launcher'),
        memory=memory, python_runtime_required=False)


def _nvhpc_device(scope, reference, document):
    from scar.compiler.nvhpc_source_device_validation import qualify_source_device

    return qualify_source_device(scope, reference, document)


_ADAPTERS = {
    'ATEN_SOURCE_ENTRY': ((ATEN,), _aten),
    'CALLER_CAPSULE': ((CALLER_EXPORT, CALLER_PAYLOAD), _caller),
    'STATEFUL_CAPSULE': ((STATEFUL,), _stateful),
    'PARAMETERIZED_SPARSE': ((SPARSE,), _sparse),
    'NVHPC_SOURCE_ENTRY': ((NVHPC,), _nvhpc),
    NVHPC_DEVICE_ROUTE: ((NVHPC_DEVICE,), _nvhpc_device),
}


def qualify_entry_delivery(scope, candidate):
    """Check source, requested invocation, hardware and native provenance without selecting."""
    route, reference = candidate['route'], candidate['artifact']
    result = dict(schema=QUALIFICATION, schema_version=1, status='UNRESOLVED', reason=None,
                  route=route, candidate_id=candidate['id'], scope=scope, evidence=None)
    try:
        _require(scope['boundary'] == BOUNDARY and Path(scope['entry']).is_absolute()
                 and type(scope['argv']) is list and all(type(arg) is str for arg in scope['argv'])
                 and all(type(scope[key]) is str and _HEX.fullmatch(scope[key])
                         for key in ('entry_sha256', 'target_sha256'))
                 and type(scope['memory_budget_bytes']) is int and scope['memory_budget_bytes'] > 0,
                 'exact whole-process invocation, target digest and memory budget required')
        schemas, adapter = _ADAPTERS[route]
        _require(reference['schema'] in schemas, 'unexpected native artifact schema for delivery route')
        document = read_artifact(reference)
        _require(artifact_reference(scope['entry'], 'scar.python-source')['sha256'] == scope['entry_sha256'],
                 'current entry source bytes differ from scope')
        result.update(status='QUALIFIED', evidence=dict(
            artifact=reference, **adapter(scope, reference, document)))
    except _PROFILE_ERRORS as error:
        result['reason'] = str(error)
    return result


def _paired(rows, warmup, repetitions, minimum_gain, strict_minimum_gain):
    return compare_entry_pairs(rows, incumbent='ORIGINAL', challenger='NATIVE', warmup_pairs=warmup,
                               measurement_pairs=repetitions, minimum_gain=minimum_gain,
                               strict_minimum_gain=strict_minimum_gain)


def _timing_row(pair, mode, wall_ns, returncode, stdout, stderr):
    return dict(pair=pair, mode=mode, wall_ns=wall_ns, returncode=returncode,
                stdout_sha256=sha256(stdout).hexdigest(), stderr_sha256=sha256(stderr).hexdigest())


def _aten_validation(scope, evidence, reference, minimum_gain, *, strict_minimum_gain=False):
    v = _matches(reference, 'scar.aten-source-entry-validation')
    original, native = 'ORIGINAL_PYTHON', 'NATIVE_CPP'
    artifact = evidence['artifact']
    root = str(Path(artifact['uri']).parent)
    measured = v['full_process_measured']
    original_cpus = v.get('original_cpu_ids', v['cpu_ids'])
    _require(v['schema_version'] == 1 and v['project'] == root and v['manifest_sha256'] == artifact['sha256']
             and v['argv'] == scope['argv'] and v['steps'] == evidence['steps'] and v['inputs'] == evidence['file_inputs']
             and v['cpu_ids'] == evidence['cpu_ids'] and type(measured) is bool
             and v['exact_publication_stdout_stderr_exit_verified'] is True
             and v['deployment_authorized'] is False
             and v['timing_scope'] == 'SOURCE_CLI_PROCESS_LAUNCH_TO_EXIT_INCLUDING_FILE_IO',
             'standalone validation does not bind the same C++ project, invocation and inputs')
    _require(type(original_cpus) is list and all(type(cpu) is int and cpu >= 0 for cpu in original_cpus)
             and len(set(original_cpus)) == len(original_cpus) and set(v['cpu_ids']) <= set(original_cpus),
             'standalone native placement is outside the original CPU cohort')
    if 'process_affinity_cpu_ids' in scope:
        _require(scope['process_affinity_cpu_ids'] == original_cpus, 'standalone paired process placement differs')
    if not measured:
        _require('original_cpu_ids' in v and Path(v['working_directory']).is_absolute(),
                 'unmeasured standalone validation requires its explicit launch scope')
    if 'working_directory' in v and 'working_directory' in scope:
        _require(scope['working_directory'] == v['working_directory'], 'standalone paired working directory differs')
    argv = scope['argv']
    python_command = v['commands'][original]
    _require(python_command == ['/usr/bin/taskset', '-c', ','.join(map(str, original_cpus)),
                                python_command[3], '-B', scope['entry'], *argv]
             and Path(python_command[3]).is_file() and Path(python_command[3]).is_absolute()
             and Path(python_command[3]).resolve() == Path(scope.get('python_executable', python_command[3])).resolve()
             and v['commands'][native] == [evidence['launcher']['uri'], *argv],
             'standalone validation launch differs from same-entry Python/C++ pair')
    if 'provider_environment' in scope:
        _require(v['python_provider_environment'] == scope['provider_environment'],
                 'standalone validation provider environment differs')
    _frozen_environment(scope, _environment_digest(v['python_provider_environment']))
    ordered, publication_hashes = [], set()
    _require(len(v['observations']) == len(v['runs']) == (8 if measured else 2),
             'complete standalone pair evidence required')
    output_root = Path(reference['uri']).parent
    for ordinal, (run, output) in enumerate(zip(v['runs'], v['observations'])):
        pair, slot = divmod(ordinal, 2)
        labels = (original, native) if pair % 2 == 0 else (native, original)
        _require((output['pair'], output['mode']) == (pair, labels[slot]), 'standalone pair order changed')
        stdout, stderr = output['stdout'].encode(), output['stderr'].encode()
        wall = run['wall_ns']
        _require(not stderr and run['returncode'] == 0
                 and (type(wall) is int and wall > 0 if measured else wall is None)
                 and run == _timing_row(pair, labels[slot], wall, 0, stdout, stderr),
                 'standalone exit or output digest differs from retained observation')
        publication = _ref_with_path(output['output'], 'scar.source-entry-publication')
        _require(Path(publication['uri']) == output_root / f'pair-{pair}-{labels[slot]}.f32'
                 and publication['size_bytes'] == evidence['scoped_cost_evidence']['logical_final_payload_bytes'],
                 'standalone file publication differs from the admitted scope')
        read_artifact(publication, json_document=False)
        publication_hashes.add(publication['sha256'])
        ordered.append(dict(run, mode='ORIGINAL' if labels[slot] == original else 'NATIVE'))
    _require(len(publication_hashes) == 1, 'standalone final publication differs across invocations')
    launch = dict(commands=v['commands'], cpu_ids=v['cpu_ids'], original_cpu_ids=original_cpus,
                  working_directory=v.get('working_directory'), provider_environment=v['python_provider_environment'],
                  timing_scope=v['timing_scope'])
    if not measured:
        _require(v['decision'] is None and ordered[0]['stdout_sha256'] == ordered[1]['stdout_sha256'],
                 'standalone correctness observation does not preserve stdout')
        return None, None, launch
    replay_entry_pairs(v['runs'], v['decision'], incumbent=original, challenger=native,
                       warmup_pairs=1, measurement_pairs=3)
    return _paired(ordered, 1, 3, minimum_gain, strict_minimum_gain), None, launch


def _caller_validation(scope, evidence, reference, minimum_gain, *, strict_minimum_gain=False):
    from scar.compiler.caller_pipeline_capsule import (
        DESIGN, VALIDATION, _counts, _publication_match, _publication_snapshot,
        _require_fresh_publication, _slot, _validation_design)
    from scar.runtime.caller_pipeline_capsule import NO_OP, PAYLOAD, RELEASE, SLOT, TRIAL, evaluate_slot, pack_slot
    from scar.runtime.caller_pipeline_deployment import entry_context, environment_sha256, verify_provider_snapshot

    v = _matches(reference, VALIDATION)
    payload_ref = evidence['payload']
    payload = _matches(payload_ref, PAYLOAD)
    _frozen_environment(scope, payload['environment_sha256'])
    trial = v['trial_slot']
    _require(v['schema_version'] == 1 and v['payload'] == payload_ref
             and v['complete'] is v['validated'] is True
             and trial['schema'] == SLOT and Path(trial['uri']) == Path(payload['capsule_directory']) / 'trial.slot',
             'original complete compact validation for this payload is required')
    trial_value = _matches(trial, SLOT)
    _require(trial_value == _slot(payload, payload_ref['sha256'], TRIAL)
             and read_artifact(trial, json_document=False) == pack_slot(trial_value), 'caller trial slot changed')
    design = _matches(v['design'], DESIGN)
    _require(design == _validation_design(payload, payload_ref, trial), 'frozen caller comparison design changed')
    for row in v['evidence']:
        read_artifact(row, json_document=False)
    policy, publication = payload['policy'], payload.get('publication')
    _require(publication is not None and v['declared_publications_validated'] is True,
             'whole-process caller validation requires a declared publication boundary')
    context = payload['entry_context']
    coordinates = [('WARMUP', i) for i in range(policy['warmup'])]
    coordinates += [('MEASUREMENT', i) for i in range(policy['repetitions'])]
    _require([(row['phase'], row['index']) for row in v['samples']] == coordinates,
             'complete original ordered sample coordinates required')
    timings = []
    for pair, row in enumerate(v['samples']):
        sample = _matches(row['sample_ref'], row['schema'])
        _require(sample == {k: value for k, value in row.items() if k != 'sample_ref'}, 'original caller sample changed')
        a, b = row['baseline'], row['optimized']
        folder = Path(reference['uri']).parent / 'samples' / (row['phase'].lower() + '-' + str(row['index']))
        python, entry, argv = context['python_executable'], context['entry_path'], context['argv']
        report = b['execution_report']
        _require(str(Path(a['command'][0]).resolve()) == str(Path(b['command'][0]).resolve()) == python
                 and a['command'][1:] == [entry, *argv]
                 and b['command'][1:] == ['-m', 'scar.runtime.caller_pipeline_capsule', '--capsule',
                    payload['capsule_directory'], '--slot', trial['uri'], '--payload-sha256', payload_ref['sha256'],
                    '--slot-sha256', trial['sha256'], '--exclusive-inputs', '--authority-role', str(TRIAL),
                    '--report', report['uri'], '--', entry, *argv]
                 and Path(report['uri']) == folder / 'execution.json', 'caller paired launch changed')
        streams = []
        for label, arm, mode in (('baseline', a, 'ORIGINAL'), ('optimized', b, 'NATIVE')):
            stdout, stderr = (read_artifact(arm[key], json_document=False) for key in ('stdout', 'stderr'))
            streams.append((stdout, stderr))
            timings.append(_timing_row(pair, mode, arm['wall_ns'], arm['returncode'], stdout, stderr))
            _require(_publication_snapshot(folder / (label + '.publications'), publication) == arm['publications'],
                     'retained caller publication bytes changed')
        native, fallback, native_only = _counts(read_artifact(report), payload, payload_ref, trial)
        _require(streams[0] == streams[1] and _publication_match(a['publications'], b['publications'], publication)
                 and native_only and row['native_calls'] == native and row['fallback_calls'] == fallback
                 and row['valid'] is row['frozen_context_unchanged'] is True, 'caller outputs, exits or native coverage differ')
    paired = _paired(timings, policy['warmup'], policy['repetitions'], minimum_gain, strict_minimum_gain)
    launch = dict(entry_context=context, environment_sha256=payload['environment_sha256'],
                  timing_scope=design['process_clock_scope'])
    if evidence['artifact']['schema'] != CALLER_EXPORT:
        return paired, None, launch
    exported = _matches(evidence['artifact'], CALLER_EXPORT)
    _require(exported['schema_version'] == 1 and exported['original_validation'] == reference,
             'caller release slot has different validation authority')
    expected = _slot(payload, payload_ref['sha256'], RELEASE, v['samples'], reference['sha256'])
    approved = evaluate_slot(expected, payload, payload_ref['sha256'])['approved']
    if not approved:
        expected['role'] = NO_OP
    slot_ref = exported['slot']
    _require(slot_ref['schema'] == SLOT and Path(slot_ref['uri']) == Path(payload['capsule_directory']) / 'decision.slot'
             and read_artifact(slot_ref, json_document=False) == pack_slot(expected)
             and exported['deployment_authorized'] is approved
             and exported['decision'] == ('RELEASE' if approved else 'NO_OP'),
             'caller release slot differs from original frozen observations')
    live = (scope.get('exclusive_inputs') is True and entry_context(context['entry_path'], context['argv']) == context
            and environment_sha256() == payload['environment_sha256'])
    if not approved or not live:
        return paired, None, launch
    try:
        _require_fresh_publication(Path(publication['directory']))
        verify_provider_snapshot(payload['provider_identity'], payload.get('provider_file_witness'),
                                 entry=context['entry_path'])
    except _PROFILE_ERRORS:
        return paired, None, launch
    return paired, dict(payload=payload_ref, slot=slot_ref, validation=reference), launch


def _sparse_validation(scope, evidence, reference, minimum_gain, *, strict_minimum_gain=False):
    from scar.codegen.native_parameterized_sparse_entry import check as check_native
    from scar.optimization.native_parameterized_sparse_plan import BENCHMARK, COMPARISON, compare

    stored = _matches(reference, COMPARISON)
    refs = stored['artifacts']
    _require(refs['plan'] == evidence['plan'] and stored['executable'] == {
        'artifact': evidence['artifact'], 'executable': evidence['executable']},
        'sparse paired comparison refers to different native bytes')
    run = _matches(refs['benchmark'], BENCHMARK)
    _frozen_environment(scope, _environment_digest(run['environment']))
    current = compare(refs['plan']['uri'], refs['benchmark']['uri'])
    _require(current == stored, 'sparse paired selection differs from cold observation replay')
    check_native(evidence['artifact']['uri'])
    for key, name in (('working_directory', 'cwd'), ('provider_environment', 'environment'),
                      ('process_affinity_cpu_ids', 'cpu_ids')):
        if key in scope:
            _require(scope[key] == run[name], 'sparse paired validation context differs for ' + key)
    matches = [case for case in run['cases'] if case['argv'] == scope['argv']]
    _require(len(matches) == 1, 'sparse paired validation lacks this exact runtime count')
    rows = matches[0]['rows']
    timings = [_timing_row(row['round'], 'ORIGINAL' if row['arm'] == 'original' else 'NATIVE',
                           row['wall_ns'], row['returncode'],
                           read_artifact(row['stdout'], json_document=False),
                           read_artifact(row['stderr'], json_document=False)) for row in rows]
    launch = dict(cpu_ids=run['cpu_ids'], working_directory=run['cwd'], provider_environment=run['environment'],
                  original_interpreter=run['arms']['original'], timing_scope=stored['scope'])
    return _paired(timings, 1, len(rows) // 2 - 1, minimum_gain, strict_minimum_gain), None, launch


def _nvhpc_validation(scope, evidence, reference, minimum_gain, *, strict_minimum_gain=True):
    from scar.compiler.nvhpc_source_validation import replay_source_entry_evidence

    timing, context = replay_source_entry_evidence(reference, evidence, scope, minimum_gain=minimum_gain,
                                                   strict_minimum_gain=strict_minimum_gain)
    return timing, None, context


def _nvhpc_device_validation(scope, evidence, reference, minimum_gain, *, strict_minimum_gain=True):
    from scar.compiler.nvhpc_source_device_validation import replay_source_device

    timing, status = replay_source_device(reference, evidence, scope, minimum_gain=minimum_gain,
                                         strict_minimum_gain=strict_minimum_gain)
    return timing, None, status


_VALIDATORS = {
    'ATEN_SOURCE_ENTRY': _aten_validation,
    'CALLER_CAPSULE': _caller_validation,
    'PARAMETERIZED_SPARSE': _sparse_validation,
    'NVHPC_SOURCE_ENTRY': _nvhpc_validation,
    NVHPC_DEVICE_ROUTE: _nvhpc_device_validation,
}


def replay_entry_delivery(scope, *, selected_id, candidate=None, minimum_gain=DEFAULT_MINIMUM_GAIN,
                          strict_minimum_gain=True):
    """Recheck the exact winner, retain fallback, and replay only its registered evidence."""
    result = dict(schema=HANDOFF, schema_version=1, selected_id=selected_id, scope=scope,
                  final_action='KEEP_ORIGINAL', reason='original entry remains the selected fallback',
                  qualification=None, validation=None, launch_authority=None, deployment_authorized=False)
    if selected_id in _ORIGINAL_OPTIONS:
        return result
    if candidate is None or candidate.get('id') != selected_id:
        return dict(result, reason='selected option does not identify the supplied native artifact')
    qualified = qualify_entry_delivery(scope, candidate)
    result['qualification'] = qualified
    if qualified['status'] != 'QUALIFIED':
        return dict(result, reason=qualified['reason'])

    route, validation = candidate['route'], candidate.get('validation')
    evidence = qualified['evidence']
    try:
        if route == 'CALLER_CAPSULE' and evidence['artifact']['schema'] == CALLER_EXPORT:
            frozen = read_artifact(evidence['artifact'])['original_validation']
            _require(validation is None or validation == frozen, 'supplied validation conflicts with the caller release slot')
            validation = frozen
        if validation is None or route not in _VALIDATORS:
            return dict(result, final_action='CANDIDATE_VALIDATION_REQUIRED',
                        reason='same-boundary whole-process validation is absent for this artifact')
        timing, release, launch = _VALIDATORS[route](scope, evidence, validation, minimum_gain,
                                                    strict_minimum_gain=strict_minimum_gain)
    except _PROFILE_ERRORS as error:
        return dict(result, reason='whole-process validation could not replay: ' + str(error))
    checked_effects = route != NVHPC_DEVICE_ROUTE or launch['output_contract_verified']
    result['validation'] = dict(reference=validation, timing=timing, launch=launch,
                                exact_declared_effects_replayed=checked_effects)
    if not checked_effects:
        return dict(result, reason='CUDA source comparison remains open: ' + launch['phase'])
    if timing is None:
        return dict(result, final_action='ADVANCE_TO_RELEASE_REVIEW',
                    reason='whole-process correctness replayed without performance observations')
    if timing['verdict'] != 'ADVANCE_TO_RELEASE_REVIEW':
        return dict(result, reason='whole-process paired observations did not satisfy the gain policy')
    if release:
        return dict(result, final_action='RELEASE', launch_authority=release, deployment_authorized=True,
                    reason='original caller release authority and live scope replayed')
    return dict(result, final_action='ADVANCE_TO_RELEASE_REVIEW',
                reason='whole-process evidence replayed and release authority is pending')
