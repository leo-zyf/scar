"""Collect a fresh fixed-entry source profile using a checked triangle family."""

import contextlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from scar.analysis import source_triangle_seed
from scar.compiler.call_binding import observe_call_bindings
from scar.ir.execution_recipe import artifact_reference, read_artifact

SCHEMA = 'scar.source-triangle-profile-frontend'
_DESIGN = 'scar.mesh-basis-frame-design'
_FRAMES = 'scar.mesh-basis-frame-observation'
_ACTIVATION = 'scar.mesh-basis-activation-admission'
_PLAN = 'scar.call-binding-plan'
_OBSERVATION = 'scar.study-aggregate-source-scope'
_ORACLE = 'scar.native-ordered-source-stdout'


def _require(condition, reason):
    if not condition:
        raise ValueError('triangle frontend: ' + reason)


def _write(path, record):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return artifact_reference(path, record['schema'])


def _row(document, locator, collection):
    prefix, position = locator.rsplit('/', 1)
    _require(prefix == '#/' + collection and position.isdecimal(), 'invalid retained record locator')
    return document[collection][int(position)]


def _frame_selector(receipt, code, site, phase, operands, limit=1):
    if 'frames' in receipt:
        retained = receipt['frames'][code['qualname']]
    else:
        matches = [row for row in receipt['selectors'].values() if row['code'] == code]
        _require(len(matches) == 1, 'retained frame selector is ambiguous')
        retained = matches[0]
    _require(retained['code'] == code and retained['source']['uri'] == code['filename'],
             'frame selector differs from retained exact code')
    return dict(code=code, source=retained['source'], normal_return_offsets=retained['normal_return_offsets'],
                caller_site=site, phase=phase, operands=operands, max_activations=limit)


def _local(formal, *path, kind='OBJECT'):
    return dict(origin='FORMAL', formal=formal, path=list(path), kind=kind)


def _selectors(seed, calls, activation, declaration):
    proof = seed['provenance']
    frames = read_artifact(proof['frames'])
    endpoints = {name: _row(calls, locator, 'observed_targets')['observation']
                 for name, locator in proof['observed_call_sites'].items()}
    constructor, base = (_row(frames, activation['constructor'][role], 'records')
                         for role in ('frame', 'base_init'))
    same_constructor = endpoints['basis_type']['caller_site']
    _require(same_constructor['code'] == constructor['caller_code']
             and same_constructor['offset'] == constructor['caller_offset']
             and base['caller_code'] == constructor['code']['key'],
             'retained constructor and caller relationship differs')
    own, mesh = constructor['code']['parameters'][:2]
    inherited_own = base['code']['parameters'][0]
    loop = endpoints['loop_consumer']['target_code']
    count = loop['parameters'][-1]
    basis_operands = {
        'self': _local(own), 'mesh': _local(mesh),
        'X': _local(own, 'X', kind='ARRAY'), 'W': _local(own, 'W', kind='ARRAY'),
        'element_dofs': _local(own, 'dofs', 'element_dofs', kind='ARRAY'),
    }
    base_site = {'code': base['caller_code'], 'offset': base['caller_offset']}
    selected = dict(
        constructor=_frame_selector(declaration, constructor['code'], same_constructor,
                                    'NORMAL_RETURN', basis_operands),
        base=_frame_selector(declaration, base['code'], base_site, 'NORMAL_RETURN',
                             {'self': _local(inherited_own)}),
        loop=_frame_selector(declaration, loop, endpoints['loop_consumer']['caller_site'],
                             'ENTERED', {'selected_count': _local(count, kind='COUNT')}),
    )
    for endpoint_role, role, limit, operands in (
        ('uniform_refinement', 'uniform', seed['depth'], {}),
        ('refinement', 'refinement', 1, {'return': {'origin': 'RETURN', 'kind': 'OBJECT'}}),
    ):
        observation = endpoints[endpoint_role]
        selected[role] = _frame_selector(declaration, observation['target_code'], observation['caller_site'],
                                         'NORMAL_RETURN', operands, limit)
    return selected


def _interpreter(context):
    actual = {
        'executable': sys.executable, 'magic': importlib.util.MAGIC_NUMBER.hex(),
        'name': sys.implementation.name, 'optimize': sys.flags.optimize, 'version': sys.version,
    }
    if isinstance(context, str):
        _require(Path(context).samefile(sys.executable), 'observation interpreter changed')
    else:
        _require(actual == context, 'observation interpreter ABI changed')


def _child(design_path):
    from scar.runtime.entry_runner import run_original_script
    from scar.runtime.source_profile_frames import observe_frames

    source = artifact_reference(design_path, _DESIGN)
    design = read_artifact(source)
    _interpreter(design['interpreter'])
    _require(os.getcwd() == design['cwd'] and all(os.getenv(key) == value
             for key, value in design['environment'].items()), 'original capture launch context changed')
    read_artifact(design['entry'], json_document=False)
    out = Path(design_path).parent
    aggregate = contextlib.nullcontext()
    if design['aggregate_design'] is not None:
        from scar.runtime.sparse_ordered_scope_observe import observe_scope
        aggregate = observe_scope(design['aggregate_design'], out/'original-observation.json')
    with aggregate, observe_frames(design['selectors']) as recorded:
        launch = run_original_script(design['entry']['uri'], design['argv'])
    _write(out/'frames.json', dict(schema=_FRAMES, schema_version=1, design=source,
        call_plan=artifact_reference(os.environ['SCAR_CALL_BINDING_PLAN'], _PLAN),
        process_id=recorded.pid, thread_id=recorded.thread,
        cpu_ids=sorted(os.sched_getaffinity(0)), launch=launch, records=recorded.records, roles=recorded.roles,
        dropped_due_to_cap={}, tensor_contents_recorded=False, global_hook=False,
        scope='EXACT_SELECTED_CODE_FRAME_METADATA'))


def _admit_frames(out, run, design_ref, design, original_plan, old_activation, *, resume):
    frames_ref = artifact_reference(out/'frames.json', _FRAMES)
    frames = read_artifact(frames_ref)
    raw = artifact_reference(run['raw_observations'][0], 'scar.call-binding-observations')
    attempts = read_artifact(raw)
    selectors = design['selectors']
    _require(frames['design'] == design_ref and frames['call_plan']['uri'] == run['plan']
             and read_artifact(frames['call_plan']) == original_plan
             and frames['process_id'] == attempts['process_id'] and frames['launch']['argv'] == design['argv']
             and frames['launch']['decision'] == 'NO_OP' and not attempts['errors']
             and frames['roles'].keys() == selectors.keys(), 'incomplete original frame capture')
    for role, selector in selectors.items():
        rows = [frames['records'][index] for index in frames['roles'][role]]
        _require(len(rows) == selector['max_activations']
                 and all(row['code'] == selector['code'] and row['phase'] == selector['phase'] for row in rows),
                 'selected frame does not match its exact admitted code, phase or count: ' + role)
    ctor_index, = frames['roles']['constructor']
    base_index, = frames['roles']['base']
    refined_index, = frames['roles']['refinement']
    constructor, base, refined = (frames['records'][index] for index in (ctor_index, base_index, refined_index))
    identity = constructor['fields']['self']['object_token']
    mesh = constructor['fields']['mesh']['object_token']
    _require(identity is not None and identity == base['fields']['self']['object_token']
             and mesh is not None and mesh == refined['fields']['return']['object_token'],
             'selected basis and mesh objects differ across original activations')
    ctor_matches = [index for index, attempt in enumerate(attempts['records'])
                    if attempt['index'] == constructor['raw_call_attempt']]
    _require(len(ctor_matches) == 1, 'selected constructor has no original CALL attempt')
    body_proofs = {}
    for role in ('constructor', 'base', 'uniform', 'refinement'):
        key = selectors[role]['code']['key']
        body_proofs[key] = old_activation['body_correspondences'][key]
    document = dict(schema=_ACTIVATION, schema_version=1,
        source_catalog=old_activation['source_catalog'], original_run=artifact_reference(out/'observed'/'result.json', run['schema']),
        frame_observations=frames_ref, bounded_call_observations=raw, body_correspondences=body_proofs,
        constructor=dict(original_code=constructor['code']['key'], frame=f'#/records/{ctor_index}',
                         base_init=f'#/records/{base_index}', raw_attempt=f'#/records/{ctor_matches[0]}',
                         standard_implicit_type_protocol_complete=False),
        complete_whole_program_or_eps=False, transformation_authorized=False)
    path = out/'activation-admission.json'
    if resume and path.exists():
        saved = artifact_reference(path, _ACTIVATION)
        _require(read_artifact(saved) == document, 'existing activation is not the exact original capture replay')
        return saved
    return _write(path, document)


def _project(out, source, catalog, target, max_seconds, *, resume=False):
    from scar.analysis.bounded_entry_calls import derive_bounded_entry_calls
    from scar.analysis.build import BuildSession, build

    program = out/'program'
    if resume:
        replay = 1
        program = out/f'program-replay-{replay}'
        while program.exists():
            replay += 1
            program = out/f'program-replay-{replay}'
    build(source, None, program, catalog=catalog, target=target, jobs=1,
          bounded_call_bindings=(out/'observed',), max_seconds=max_seconds)
    with BuildSession(program, seconds=max_seconds) as session:
        projection = derive_bounded_entry_calls(session)
    return _write(out/'entry-calls.json', projection)


def _bind_calls(out, prior, capture, source, catalog, target, *, resume):
    from scar.analysis.rebind_bounded_entry_calls import rebind

    try:
        rebound = rebind(prior, capture)
    except (ValueError, OSError, KeyError, TypeError) as error:
        projection = _project(out, source, catalog, target, 3600, resume=resume)
        return projection, {'decision': 'REBUILT_F4', 'reason': str(error)}
    evidence = _write(out/'entry-calls-rebound.json', rebound)
    return evidence, {'decision': 'EXACT_STATIC_PROOF_REUSE', 'key': rebound['static_proof_key']}


def observe_profile(source_seed, argv, out, *, aggregate_design, interpreter, cwd,
                    environment, expected_entry=None, max_seconds=60, cpu_ids=None, resume=False):
    """Collect one original run or resume a complete capture, then join its source graph."""
    anchor_path = Path(source_seed).resolve(strict=True)
    anchor = artifact_reference(anchor_path, source_triangle_seed.SCHEMA)
    seed = source_triangle_seed.check(anchor_path)
    _require(seed == read_artifact(anchor), 'source family changed during admission')
    entry = seed['entry']
    _require(expected_entry is None or str(Path(expected_entry).resolve(strict=True)) == entry['uri'],
             'source entry differs from the checked family')
    _require(type(argv) is list and len(argv) == 1 and type(argv[0]) is str
             and re.fullmatch('[1-9][0-9]{0,8}', argv[0]) is not None and int(argv[0]) > 1,
             'expected one finite positive repeated-loop argument')
    old_calls = read_artifact(seed['entry_calls'])
    prior_calls = old_calls.get('static_projection', seed['entry_calls'])
    old_activation = read_artifact(seed['activations'])
    old_frames = read_artifact(seed['provenance']['frames'])
    old_declaration = read_artifact(seed['provenance']['frame_design'])
    catalog = old_activation['source_catalog']
    prior_program = read_artifact(read_artifact(prior_calls)['program'])
    _require(prior_program['entry'] == entry['uri'] and prior_program['workflow_complete']
             and prior_program['f4']['catalog'] == dict(path=catalog['uri'], bytes=catalog['size_bytes'], sha256=catalog['sha256']),
             'retained catalog differs from the selected graph')
    read_artifact(catalog)
    target = prior_program['target']
    if target is not None:
        from scar.analysis.build import _check
        _check(target)
        target = Path(target['path']).parent

    plan = old_frames['call_plan']
    original_plan = read_artifact(plan)
    cwd = str(Path(cwd).resolve(strict=True))
    design = dict(schema=_DESIGN, schema_version=1, entry=entry, argv=argv, cwd=cwd, environment=environment,
        interpreter=interpreter, source_design=seed['provenance']['frame_design'], aggregate_design=aggregate_design,
        selectors=_selectors(seed, old_calls, old_activation, old_declaration))
    out = Path(out).resolve()
    for protected in (Path(anchor['uri']), Path(entry['uri']), Path(catalog['uri'])):
        _require(not protected.is_relative_to(out), 'frontend output contains admitted input material')
    if resume:
        design_ref = artifact_reference(out/'frame-design.json', _DESIGN)
        _require(read_artifact(design_ref) == design and not (out/'result.json').exists(),
                 'resume requires the same original frame design and an unfinished frontend')
    else:
        out.mkdir(parents=True, exist_ok=False)
        design_ref = _write(out/'frame-design.json', design)
    executable = interpreter if isinstance(interpreter, str) else interpreter['executable']
    command = [executable, '-B', '-m', 'scar.compiler.observe_triangle_profile', design_ref['uri']]
    if cpu_ids is not None:
        _require(cpu_ids and set(cpu_ids) <= os.sched_getaffinity(0), 'capture CPU affinity is unavailable')
        command[:0] = ['/usr/bin/taskset', '-c', ','.join(map(str, sorted(set(cpu_ids))))]
    if resume:
        run = read_artifact(artifact_reference(out/'observed'/'result.json', 'scar.call-binding-run'))
    else:
        launch_environment = {key:value for key,value in environment.items() if value is not None}
        with (out/'original.stdout').open('xb') as stdout, (out/'original.stderr').open('xb') as stderr:
            run = observe_call_bindings(command, out/'observed', plan_path=plan['uri'], max_seconds=max_seconds,
                environment=launch_environment, cwd=cwd, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
    _require(run['command'] == command and run['full_run_completed'] and run['child_returncode'] == 0
             and run['stop_reason'] is None and run['transformations_disabled']
             and run['plan'] == str(out/'observed'/'plan.json') and len(run['raw_observations']) == 1
             and Path(run['raw_observations'][0]).resolve(strict=True).parent == out/'observed',
             'bounded original source run did not complete')
    oracle = artifact_reference(out/'original.stdout', _ORACLE)
    _require(not (out/'original.stderr').stat().st_size
             and re.fullmatch(rb'[0-9a-f]{64}\n', read_artifact(oracle, json_document=False)),
             'original profile produced an unexpected output')
    activation = _admit_frames(out, run, design_ref, design, original_plan, old_activation, resume=resume)
    observation = None
    if aggregate_design is not None:
        frames = read_artifact(read_artifact(activation)['frame_observations'])
        observation = artifact_reference(out/'original-observation.json', _OBSERVATION)
        evidence = read_artifact(observation)
        _require(evidence['entry'] == entry and evidence['argv'] == argv
                 and evidence['process_id'] == frames['process_id'] and evidence['thread_id'] == frames['thread_id']
                 and evidence['normal_completion'] is True
                 and (evidence['return_digest'] + '\n').encode('ascii') == read_artifact(oracle, json_document=False),
                 'aggregate capture differs from the original entry, argv or stdout')
    capture = artifact_reference(out/'observed'/'result.json', 'scar.call-binding-run')
    calls, reuse = _bind_calls(out, prior_calls, capture, entry['uri'], Path(catalog['uri']).parent,
                               target, resume=resume)
    refined = source_triangle_seed.extract(calls['uri'], activation['uri'], seed['source_root'], argv)
    _require(refined['entry'] == entry and refined['fields'] == seed['fields'] and refined['sources'] == seed['sources']
             and refined['depth'] == seed['depth'] and refined['provenance'].get('observed_loop_count'),
             'fresh profile changed the family or lost its original scalar operand')
    seed_ref = _write(out/'source-seed.json', refined)
    record = dict(schema=SCHEMA, schema_version=1, family=anchor, argv=argv, frame_design=design_ref,
        entry_calls=calls, source_seed=seed_ref, activation=activation, oracle=oracle, original_observation=observation,
        static_projection=reuse, original_processes=1, prior_runtime_observations_transferred=False)
    return dict(record, frontend=_write(out/'result.json', record))


if __name__ == '__main__':
    _child(sys.argv[1])
