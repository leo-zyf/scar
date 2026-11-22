"""Aten export placement."""

from hashlib import sha256
from math import prod
import os
import argparse
import json
from pathlib import Path
from scar.codegen.aten_native_graph import (
    _DRIVER,
    _NUMA_HEADER,
    _READY_HEADER,
    _TEAM_DRIVER,
    _TEAM_HEADER,
    _render,
)
from scar.codegen.aten_provider import _write_binding
from scar.codegen.native_project import build_native_project, write_native_project
from scar.codegen.native_provider_bundle import _content
from scar.compiler.plan_aten_native_graph import _invoke, _receipt
from scar.compiler.plan_aten_ready import (
    _case,
    _ready_model,
    _resource_plan,
    _source_order_baseline,
)
from scar.ir.implementation_plan import problem_from_dict
from scar.ir.record_codec import encode
from scar.ir.v2.schemas import canonical_json
from scar.optimization.aten_export_rewrite import FUNCTIONAL_EXPORT_SCHEMAS
from scar.optimization.aten_native_schedule import CONTRACT_SCHEMA, SCOPE, _ATTESTATIONS
from scar.optimization.aten_team import aten_cpu_team_domain, propose_cpu_team, ready_team_available
from scar.optimization.aten_live_values import tensor_releases
from scar.optimization.joint_planner import verify_execution_plan
from tempfile import TemporaryDirectory
from scar.compiler.aten_export_hybrid_source import _crossings, qualify_hybrid_export
from scar.ir.program import codec
from scar.optimization.native_device_placement_search import search_native_device_placements


# Calibrate a fixed-shape functional export with the shared CPU ready-DAG planner.
_SCHEMA = 'scar.aten-export-cpu-placement'
_ABI = 2


def _contract(recipe):
    schemas = list(dict.fromkeys(op['schema'] for op in recipe['operations']))
    if not set(schemas).issubset(FUNCTIONAL_EXPORT_SCHEMAS.values()):
        return None
    return dict(schema=CONTRACT_SCHEMA, schema_version=1, scope=SCOPE, operator_schemas=schemas,
                **dict.fromkeys(_ATTESTATIONS, True))


def _serial(cpu):
    return dict(mode='SERIAL', cpu_ids=[cpu], lanes=None, operation_order=None)


def _selection(model):
    return dict(mode='READY_DAG', cpu_ids=model['cpu_ids'], lanes=model['lanes'],
                operation_order=model.get('operation_order', list(range(len(model['lanes'])))))


def _raw_sizes(recipe, result_bytes, raw):
    sizes = [4 * prod(row['shape']) for row in recipe['inputs']]
    if (raw['calibration']['input_storage_bytes'] != sizes
            or raw['calibration']['result_storage_bytes'] != list(result_bytes)):
        raise ValueError('ready probe geometry differs from the specialized Tensor graph')
    return sizes


def _settings(warmup, measured_pairs):
    if (type(warmup) is not int or not 3 <= warmup <= 100 or type(measured_pairs) is not int
            or not 4 <= measured_pairs <= 500 or measured_pairs % 2):
        raise ValueError('native ready placement requires 3–100 warmups and an even number of 4–500 pairs')


def _candidate_cpus(domain, serial_cpu):
    for row in domain['representatives']:
        if serial_cpu in row['cpus']:
            cpus = [serial_cpu, *(cpu for cpu in row['cpus'] if cpu != serial_cpu)]
            return cpus, [row['topology'][row['cpus'].index(cpu)] for cpu in cpus]
    return None, None


def _probe_cpp(recipe, bindings, team):
    header = _render(recipe, bindings, team, release_after=tensor_releases(recipe), ready_dag=True)
    return {'graph.hpp': header, 'entry.cpp': _DRIVER.read_text(), 'team.cpp': _TEAM_DRIVER.read_text(),
            **{path.name: path.read_text() for path in (_TEAM_HEADER, _READY_HEADER, _NUMA_HEADER)}}


def build_export_cpu_probe(root, recipe, bindings, schedule, toolchain):
    """Build the existing team ABI without changing or relinking the retained profile."""
    _, libraries, compiler, flags, link = toolchain
    source = root / 'src'
    source.mkdir(parents=True)
    _write_binding(source, bindings)
    (source / 'link-check.cpp').unlink()
    for name, body in _probe_cpp(recipe, bindings, schedule).items():
        (source / name).write_text(body)
    write_native_project(root, 'aten-ready-probe', sources=('src/team.cpp',),
        compile_options=(*flags[1:], '-O2', '-pthread'), link_options=('-pthread', link[0]),
        libraries=tuple(map(str, libraries)), runtime_dirs=(str(libraries[0].parent),))
    executable, _ = build_native_project(root, 'aten-ready-probe', compiler, expected_libraries=libraries)
    return executable


def _compare(executable, recipe, team, model, baseline, inputs, geometry, warmup, count):
    mode, prefix = '--benchmark-ready', [''.join(map(str, model['lanes']))]
    if 'operation_order' in model:
        mode = '--benchmark-ready-ordered'
        prefix.append(','.join(map(str, model['operation_order'])))
    observed = _invoke(executable, team, mode, [*prefix, baseline, warmup, count], inputs)
    return dict(observations=observed, validation=_case(observed, model, baseline, warmup, count,
                                                       geometry, recipe=recipe))


def _evidence_ref(origin, evidence, path):
    if path is None:
        path = Path(origin['path']).parent / 'ready/proposal.json'
    return dict(path=Path(path).as_posix(), evidence_sha256=sha256(canonical_json(evidence).encode()).hexdigest(),
                scope='CANONICAL_JSON_EVIDENCE_FIELD')


def _candidate_certificate(hardware, recipe, origin, evidence, evidence_path, *, recorded_resources=None):
    return _resource_plan(hardware, evidence['team'], origin, _evidence_ref(origin, evidence, evidence_path),
        recipe, evidence['calibration']['calibration'], evidence['model'], program_graph=origin,
        recorded_resources=recorded_resources)


def plan_export_cpu(recipe, *, target, input_paths, tensor_budget_bytes, bindings, toolchain, build_root,
                    origin, serial_cpu, result_bytes, warmup=3, measured_pairs=8, evidence_path=None):
    """Price one same-NUMA triple with the exact serial caller and check body profitability."""
    _settings(warmup, measured_pairs)
    result_bytes = list(result_bytes)
    contract = _contract(recipe)
    evidence = dict(schema=_SCHEMA, schema_version=1, serial_cpu=serial_cpu, contract=contract,
                    warmup_pairs=warmup, measured_pairs=measured_pairs, scope='WARM_NATIVE_BODY')
    placement = _serial(serial_cpu)
    if contract is None:
        return placement, dict(evidence, reason='READY_DAG_EFFECT_CONTRACT_INCOMPLETE'), None, None
    if not ready_team_available(recipe, contract):
        return placement, dict(evidence, reason='READY_DAG_HAS_NO_INDEPENDENT_OPERATIONS'), None, None
    domain = aten_cpu_team_domain(target, allowed_cpu_ids=os.sched_getaffinity(0))
    cpus, topology = _candidate_cpus(domain, serial_cpu)
    evidence['domain'] = domain
    if cpus is None:
        return placement, dict(evidence, reason='NO_ADMITTED_TEAM_WITH_SERIAL_CALLER'), None, None
    team = propose_cpu_team(recipe, contract=contract, target=target, cpu_ids=cpus, ready_dag=True)
    if team['lanes'] != topology:
        raise ValueError('qualified ready CPU topology differs from its nominated lanes')
    evidence['team'] = team

    receipts = [_receipt(path) for path in input_paths]
    if [row['size_bytes'] for row in receipts] != [4 * prod(row['shape']) for row in recipe['inputs']]:
        raise ValueError('pass one exact float32 file per original recipe input in ABI order')
    evidence['inputs'] = [dict(id=spec['id'], sha256=row['sha256'], size_bytes=row['size_bytes'])
                          for spec, row in zip(recipe['inputs'], receipts, strict=True)]
    executable = build_export_cpu_probe(Path(build_root), recipe, bindings, team, toolchain)
    evidence['probe'] = _content(executable)
    raw = _invoke(executable, team, '--calibrate-team', [warmup], receipts)
    _raw_sizes(recipe, result_bytes, raw)
    evidence['calibration'] = raw
    model = _ready_model(recipe, team, raw, tensor_budget_bytes, warmup, ready_dag_abi=_ABI, joint_order=True)
    original, original_baseline = _source_order_baseline(recipe, team, raw, model, tensor_budget_bytes, warmup, _ABI)
    if original is not None:
        trial = _compare(executable, recipe, team, model, original_baseline, receipts, result_bytes, warmup, measured_pairs)
        evidence['joint_order_comparison'] = trial
        if not trial['validation']['profitable']:
            model = original
    evidence['model'] = model
    reason = model['reason']
    if model['status'] == 'PLAN_READY_DAG':
        trial = _compare(executable, recipe, team, model, 'serial', receipts, result_bytes, warmup, measured_pairs)
        evidence['body_comparison'] = trial
        if trial['validation']['profitable']:
            placement = _selection(model)
        else:
            reason = 'READY_DAG_DID_NOT_BEAT_MATCHED_SERIAL_BODY'
    evidence['reason'] = reason
    if _content(executable) != evidence['probe'] or any(_receipt(row['path']) != row for row in receipts):
        raise ValueError('ready probe or input changed during measurement')
    if placement['mode'] == 'READY_DAG':
        problem, schedule = _candidate_certificate(target, recipe, origin, evidence, evidence_path)
        return placement, evidence, problem, schedule
    return placement, evidence, None, None


def _recorded_resources(evidence, document, capacity):
    team = evidence['team']
    problem = problem_from_dict(document)
    if encode(problem.target) != team['target']:
        raise ValueError('ready CPU problem target differs from its recorded team')
    resources = {resource.id: resource for resource in problem.resources}
    cores = [lane['core']['wire'] for lane in team['lanes']]
    threads = [f"target_resource:cpu-thread/{lane['cpu_id']}" for lane in team['lanes']]
    if len(set(cores + threads)) != 6:
        raise ValueError('ready CPU lanes require distinct hardware threads and cores')
    by_cpu = {}
    for lane, (core, thread) in enumerate(zip(cores, threads, strict=True)):
        witness = team['lanes'][lane]['evidence']
        entries = tuple(resources.pop(name, None) for name in (thread, core))
        refs = ((witness[key], team['constraint_fact']) for key in ('thread', 'core'))
        if any(resource is None or resource.kind != 'EXECUTION' or resource.capacity != 1 or resource.source_refs != source
               for resource, source in zip(entries, refs, strict=True)):
            raise ValueError('ready CPU lane resources differ from recorded topology')
        by_cpu[team['cpus'][lane]] = entries
    memory, = resources.values()
    if memory.id != 'host-tensor-storage-budget' or memory.kind != 'MEMORY' or memory.capacity != capacity:
        raise ValueError('ready CPU memory resource differs from the admitted capacity')
    return problem, (problem.target, {cpu: by_cpu[cpu] for cpu in team['availability_mask']}, memory)


def _replay_trial(evidence, field, recipe, model, baseline, geometry, warmup, count):
    trial = evidence[field]
    calculated = _case(trial['observations'], model, baseline, warmup, count, geometry, recipe=recipe)
    if trial['validation'] != calculated:
        raise ValueError('ready CPU comparison differs from its observed pairs')
    return calculated['profitable']


def replay_export_cpu(recipe, evidence, *, origin, tensor_budget_bytes, result_bytes, problem=None, schedule=None,
                      placement=None, evidence_path=None):
    """Recompute the native-body choice and verify its archived fixed-lane certificate."""
    warmup, count, caller = (evidence[key] for key in ('warmup_pairs', 'measured_pairs', 'serial_cpu'))
    _settings(warmup, count)
    result_bytes = list(result_bytes)
    contract = _contract(recipe)
    if (evidence['schema'] != _SCHEMA or evidence['schema_version'] != 1 or evidence['contract'] != contract
            or evidence['scope'] != 'WARM_NATIVE_BODY' or type(caller) is not int or not 0 <= caller < 1024):
        raise ValueError('ready CPU placement is outside the qualified native profile')
    result = _serial(caller)
    if contract is None:
        reason = 'READY_DAG_EFFECT_CONTRACT_INCOMPLETE'
    elif not ready_team_available(recipe, contract):
        reason = 'READY_DAG_HAS_NO_INDEPENDENT_OPERATIONS'
    elif (nomination := _candidate_cpus(evidence['domain'], caller))[0] is None:
        reason = 'NO_ADMITTED_TEAM_WITH_SERIAL_CALLER'
    else:
        cpus, topology = nomination
        team, raw = evidence['team'], evidence['calibration']
        if (team['cpus'] != cpus or team['availability_mask'] != sorted(cpus) or team['target'] != evidence['domain']['target']
                or team['lanes'] != topology or [row['cpu_id'] for row in team['lanes']] != cpus
                or len({row['numa_node'] for row in team['lanes']}) != 1):
            raise ValueError('ready CPU placement differs from the same-NUMA nomination')
        input_bytes = _raw_sizes(recipe, result_bytes, raw)
        if [(row['id'], row['size_bytes']) for row in evidence['inputs']] != list(zip(
                [item['id'] for item in recipe['inputs']], input_bytes, strict=True)):
            raise ValueError('ready CPU sampled files do not match recipe input ports')
        model = _ready_model(recipe, team, raw, tensor_budget_bytes, warmup, ready_dag_abi=_ABI, joint_order=True,
                             gain_policy=evidence['model'].get('gain_policy'))
        original, baseline = _source_order_baseline(recipe, team, raw, model, tensor_budget_bytes, warmup, _ABI)
        if original is not None:
            if not _replay_trial(evidence, 'joint_order_comparison', recipe, model, baseline, result_bytes, warmup, count):
                model = original
        elif 'joint_order_comparison' in evidence:
            raise ValueError('ready CPU placement has an unbound order comparison')
        if model != evidence['model']:
            raise ValueError('ready CPU model differs from its measured samples')
        reason = model['reason']
        if model['status'] == 'PLAN_READY_DAG':
            if _replay_trial(evidence, 'body_comparison', recipe, model, 'serial', result_bytes, warmup, count):
                result = _selection(model)
            else:
                reason = 'READY_DAG_DID_NOT_BEAT_MATCHED_SERIAL_BODY'
        elif 'body_comparison' in evidence:
            raise ValueError('ready CPU placement compared an ineligible body')
    if reason != evidence['reason'] or (placement is not None and placement != result):
        raise ValueError('ready CPU placement differs from its qualification and measured comparisons')
    if result['mode'] == 'READY_DAG':
        recorded, resources = _recorded_resources(evidence, problem, tensor_budget_bytes)
        rebuilt, expected = _candidate_certificate(None, recipe, origin, evidence, evidence_path,
                                                    recorded_resources=resources)
        verify_execution_plan(recorded, schedule)
        if rebuilt != problem or {key: item for key, item in schedule.items() if key != 'elapsed_seconds'} != {
                key: item for key, item in expected.items() if key != 'elapsed_seconds'}:
            raise ValueError('ready CPU certificate differs from its replayed lanes and services')
    elif problem is not None or schedule is not None:
        raise ValueError('serial CPU decision carried an unqualified ready execution certificate')
    return result


# Qualified device alternatives for one immutable FP32 export.
_MM = FUNCTIONAL_EXPORT_SCHEMAS['aten.mm.default']


def qualify_device_export(recipe, shapes=None, *, graph, program_ref, input_kinds):
    return qualify_hybrid_export(recipe, shapes, graph=graph,
        program_ref=program_ref, input_kinds=input_kinds, allow_cuda_only=True)


def placement_choices(projection):
    return {row['id']: ('CPU', 'CUDA') if row['schema'] == _MM else ('CPU',)
            for row in projection['operations']}


def placement_proposal(projection, choices, analysis, selected, *, actual_device_timing='UNKNOWN'):
    layout, regions, crossings = None, [], []
    if selected is not None:
        layout = selected['placement']
        placed = [dict(row, placement=layout[row['id']]) for row in projection['operations']]
        regions, crossings = _crossings(projection['inputs'], placed, projection['outputs'])
    return dict(schema='scar.native-export-placement-proposal', schema_version=1,
                domain={name: list(modes) for name, modes in choices.items()}, selected_placement=layout,
                selected_cuda_regions=regions, selected_crossings=crossings,
                analysis=analysis, actual_device_timing=actual_device_timing, default_backend='CPU',
                deployment_authorized=False)


def propose_device_placement(projection, model, *, model_ref):
    from scar.compiler.aten_export_hybrid_plan import plan_placed_hybrid_export

    choices = placement_choices(projection)

    def build(placement):
        return plan_placed_hybrid_export(projection, model, model_ref=model_ref, placement=placement)

    analysis, selected = search_native_device_placements(choices, build)
    return placement_proposal(projection, choices, analysis, selected), selected


def plan_device_export(projection, model, *, model_ref):
    report, selected = propose_device_placement(projection, model, model_ref=model_ref)
    if selected is None or 'CUDA' not in selected['placement'].values():
        raise ValueError('declared placement retains CPU or is unresolved; no CUDA candidate was nominated')
    return selected['problem'], selected['schedule'], selected['bindings'], report


def replay_device_export(projection, model, *, model_ref, problem, schedule, bindings, placement_search):
    expected_problem, expected_schedule, expected_bindings, expected = plan_device_export(
        projection, model, model_ref=model_ref)
    if (expected != placement_search or expected_problem != problem or expected_bindings != bindings
            or {key: value for key, value in expected_schedule.items() if key != 'elapsed_seconds'} !=
            {key: value for key, value in schedule.items() if key != 'elapsed_seconds'}):
        raise ValueError('CUDA placement or schedule differs from its replayed model and source choices')


def _from_cpu(cpu, cpu_sha256, profile, model_bytes, model_ref, *, empirical=None):
    from scar.compiler.native_device_pair import CPU_SCHEMA, _manifest, _member
    from scar.runtime.validate_native_project import verify_native_project

    root, _ = verify_native_project(cpu, cpu_sha256)
    manifest = _manifest(root, cpu_sha256, CPU_SCHEMA)
    family = manifest.get('profiles')
    if (family is None and profile is not None or family is not None and profile not in family):
        raise ValueError('choose exactly one published CPU shape specialization')
    base = 'plan' if profile is None else f'plan/profiles/{profile}'
    execution = json.loads(_member(root, manifest, base + '/execution.json'))
    reference = execution['program_graph']
    graph = codec.loads(_member(root, manifest, reference['path']).decode())
    recipe_file = 'source-recipe.json' if manifest.get('simplified_normal_completion') else 'recipe.json'
    recipe = json.loads(_member(root, manifest, base + '/' + recipe_file))
    projection = qualify_device_export(recipe, graph=graph, program_ref=reference,
        input_kinds={row['id']: row['kind'] for row in execution['inputs']})
    model = json.loads(model_bytes)
    if empirical is None:
        proposal, _ = propose_device_placement(projection, model, model_ref=model_ref)
    else:
        from scar.compiler.native_device_cost import propose_empirical_device_placement

        proposal = propose_empirical_device_placement(root, cpu_sha256, profile, projection, model, model_ref,
                                                       **empirical)
    return dict(proposal, cpu_manifest_sha256=cpu_sha256, cpu_profile=profile, program=reference,
                declared_model=model_ref)


def _model_reference(raw):
    return dict(path='device-model.json', sha256=sha256(raw).hexdigest(), size_bytes=len(raw))


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar plan-native-device-placement', description=__doc__)
    parser.add_argument('--cpu-project', required=True)
    parser.add_argument('--cpu-sha256')
    parser.add_argument('--cpu-profile')
    parser.add_argument('--device-model')
    parser.add_argument('--device-model-sha256')
    parser.add_argument('--empirical', action='store_true', help='use only exact measured treatments')
    parser.add_argument('--input', action='append', default=[], metavar='NAME=FILE')
    parser.add_argument('--reference', action='append', default=[])
    parser.add_argument('--device-evidence', nargs=3, action='append', default=[],
                        metavar=('CUDA_PROJECT', 'CALIBRATION_JSON', 'SHA256'))
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--out')
    action.add_argument('--check')
    parser.add_argument('--check-sha256')
    args = parser.parse_args(argv)
    if args.check:
        path = Path(args.check).resolve(strict=True)
        raw = path.read_bytes()
        if sha256(raw).hexdigest() != args.check_sha256:
            raise ValueError('device placement proposal differs from its declared digest')
        stored = json.loads(raw)
        model_path = path.with_name('device-model.json')
        if model_path.resolve(strict=True) != model_path:
            raise ValueError('device model must be local to its archived placement proposal')
        model = model_path.read_bytes()
        evidence = stored.get('empirical')
        empirical = None
        if evidence is not None:
            version = stored['analysis']['schema_version']
            if type(version) is not int or version not in (1, 2):
                raise ValueError('unsupported empirical placement analysis version')
            empirical = dict(inputs=evidence['inputs'], references=evidence['references'],
                device_evidence=[(row['project'], row['calibration'], row['sha256']) for row in evidence['device_evidence']],
                service_intervals=version == 2)
        proposal = _from_cpu(args.cpu_project, stored['cpu_manifest_sha256'], stored['cpu_profile'],
                             model, _model_reference(model), empirical=empirical)
        if proposal != stored:
            raise ValueError('device placement proposal differs from its replayed source choices')
    else:
        if not (args.cpu_sha256 and args.device_model and args.device_model_sha256):
            parser.error('--out requires the CPU manifest digest and the declared device model and digest')
        out = Path(args.out).resolve()
        if out.exists():
            raise FileExistsError(out)
        model = Path(args.device_model).read_bytes()
        model_ref = _model_reference(model)
        if model_ref['sha256'] != args.device_model_sha256:
            raise ValueError('device model differs from its declared digest')
        if not args.empirical and (args.input or args.reference or args.device_evidence):
            parser.error('sample inputs, references and device evidence require --empirical')
        empirical = dict(named_inputs=args.input, references=args.reference, device_evidence=args.device_evidence) \
            if args.empirical else None
        proposal = _from_cpu(args.cpu_project, args.cpu_sha256, args.cpu_profile, model, model_ref,
                             empirical=empirical)
        out.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix='.scar-device-placement-', dir=out.parent) as temporary:
            stage = Path(temporary)
            (stage / model_ref['path']).write_bytes(model)
            (stage / 'placement.json').write_text(json.dumps(proposal, sort_keys=True, indent=2) + '\n')
            stage.rename(out)
    print(json.dumps(dict(selected_placement=proposal['selected_placement'],
                          default_backend=proposal['default_backend'], analysis=proposal['analysis']), sort_keys=True))
    return 0
