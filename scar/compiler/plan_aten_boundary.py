"""Select a measured standalone ATen call implementation on verified CPUs."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from scar.analysis.target_graph import TargetHardwareGraph
from scar.codegen.native_provider_bundle import _content
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.implementation_plan import ImplementationOption, PlanningOperation, PlanningProblem, PlanningResource, problem_document
from scar.ir.program.target_hardware import CpuCore, HardwareThread
from scar.ir.record_codec import encode
from scar.optimization.aten_boundary_trials import EVIDENCE, SCOPE, project_aten_boundary_trial
from scar.optimization.constructive_planner import construct_implementations
from scar.optimization.entry_mandatory_bound import SCHEMA as COST_PROFILE, resolve_cost


OBSERVATIONS = 'scar.aten-native-boundary-ab.v1'
BOUNDARY = 'STANDALONE_LIBTORCH_CPU_CALL_BOUNDARY'
_ORDERS = {'SERIAL_THEN_DUAL': ['SERIAL', 'TWO_CPU'], 'DUAL_THEN_SERIAL': ['TWO_CPU', 'SERIAL']}


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _checked(receipt):
    path = Path(receipt['path'])
    _require(path.is_absolute() and path.resolve(strict=True) == path
             and receipt == dict(path=str(path), **_content(path)),
             'original native trial member changed: ' + str(path))
    return path


def _target(hardware, cpus, *, allow_cpu_subset=False):
    graph = TargetHardwareGraph(hardware)
    if allow_cpu_subset:
        from scar.optimization.aten_native_schedule import recorded_cpu_domain

        constraint, domain = recorded_cpu_domain(graph)
        _require(bool(cpus) and set(cpus).issubset(domain),
                 'native trial CPU mask is outside the target affinity and allowed CPU intersection')
    else:
        constraint, = graph.constraints()
        _require(constraint.affinity_cpu_ids == constraint.allowed_cpu_ids == cpus,
                 'native trial CPU mask differs from the target availability domain')
    inventory = graph.resources()
    threads = {r.cpu_id: r for r in inventory if type(r) is HardwareThread and r.online is True}
    cores = {cpu: r for r in inventory if type(r) is CpuCore for cpu in r.cpu_ids}
    _require(all(cpu in threads and cpu in cores for cpu in cpus)
             and len({cores[cpu].id for cpu in cpus}) == len(cpus),
             'measured lanes must name online threads on distinct target cores')
    resources = {cpu: tuple(PlanningResource(r.id.wire, 'EXECUTION', 1,
        (encode(r.evidence), encode(constraint.evidence))) for r in (threads[cpu], cores[cpu])) for cpu in cpus}
    return graph._snapshot, resources


def _admit_observation(reference, hardware, case_id):
    observations = read_artifact(reference)
    _require(observations['scope'] == BOUNDARY and observations['performance_measured'] is True
             and observations['original_python_boundary_measured'] is False
             and observations['replacement_authorized'] is False,
             'measured standalone ATen boundary required')
    boundary = observations['boundary']
    _require(boundary['clock'] == 'STEADY_CLOCK_MONOTONIC_NANOSECONDS'
             and boundary['timing_profiler_enabled'] is False
             and boundary['overlap_probe_in_measured_pairs'] is False
             and boundary['pair_validation_in_measured_region'] is False,
             'clean ATen timing and separate validation required')
    context = observations['context']
    caller, worker = context['caller_cpu'], context['worker_cpu']
    cpus = tuple(context['availability_mask'])
    _require(all(type(cpu) is int for cpu in (caller, worker)) and cpus == tuple(sorted({caller, worker}))
             and len(cpus) == 2, 'two declared original ATen CPU lanes required')
    target, resources = _target(hardware, cpus)
    _require(context['target_graph'] == dict(path=target.uri, size_bytes=target.size_bytes, sha256=target.sha256),
             'native trial was measured against a different hardware graph')

    member_names = ('source', 'executable', 'linkage', 'link_source', 'generated_header', 'raw_stdout',
                    'raw_stderr', 'dependency_listing')
    members = {name: _checked(context[name]) for name in member_names}
    _require(not members['raw_stderr'].read_bytes(), 'standalone native trial emitted diagnostics')
    linkage = json.loads(members['linkage'].read_bytes())
    link_source = json.loads(members['link_source'].read_bytes())
    _require(linkage['schema'] == 'scar.aten-provider-linkage' and linkage['cxx_standard'] == 20
             and linkage['provider'] == context['provider']
             and [row['schema'] for row in linkage['operators']] == context['provider_schemas']
             and link_source['linkage_report'] == artifact_reference(members['linkage'], linkage['schema'])
             and link_source['provider_revision'] == context['provider']['revision']
             and {row['schema'] for row in link_source['occurrences']} == set(context['provider_schemas'])
             and {key: context['generated_header'][key] for key in ('sha256', 'size_bytes')}
             == linkage['artifacts']['operators.hpp'], 'native trial linked declarations differ')
    lines = members['raw_stdout'].read_text().splitlines()
    raw = json.loads(lines[0])
    loaded = sorted(context['real_linked_images'], key=lambda row: row['path'])
    _require(loaded == sorted(context['provider']['libraries'], key=lambda row: row['path'])
             and sorted(lines[1:]) == [str(_checked(row)) for row in loaded],
             'loaded ATen provider bytes differ from the exact linkage')
    states = context['thread_observations']
    _require(all(raw.get(key) == value for key, value in states.items())
             and states['initial']['affinity'] == list(cpus), 'raw thread observations differ from native trial')
    for lane, cpu in (('caller', caller), ('worker', worker)):
        for phase in ('start', 'end'):
            state = states[lane + '_' + phase]
            _require(state['affinity'] == [cpu] and state['current_cpu'] == cpu
                     and state['intraop'] == state['interop'] == 1 and state['grad_enabled'] is False
                     and state['fp_rounding'] == states['initial']['fp_rounding'],
                     'measured ATen caller or worker context differs')

    (case_ordinal, case), = ((i, row) for i, row in enumerate(observations['cases']) if row['id'] == case_id)
    original, = (row for row in raw['cases'] if row['id'] == case_id)
    _require(all(case.get(key) == value for key, value in original.items()),
             'case sample inventory differs from original timed output')
    contract = case['input_contract']
    _require(contract['device'] == 'cpu' and contract['result_ready_at_return'] is True
             and contract['result_storage'] == 'FRESH_OWNED'
             and contract['borrowed_inputs_immutable_distinct_storage'] is True
             and all(contract[key] == case[key] for key in ('dtype', 'shape', 'strides')),
             'measured input and result contract differs from case geometry')
    modes = {'SERIAL': [caller], 'TWO_CPU': [caller, worker]}
    exact = dict(boundary=BOUNDARY, target=encode(target), provider=context['provider'],
        inputs=dict(contract=contract, values_recipe=case['input_values_recipe']),
        output=dict(ready_at_return=True, ownership='FRESH_OWNED', validation='PAIRED_BITWISE'),
        placements={mode: dict(cpu_ids=ids, aten_intraop=1, aten_interop=1) for mode, ids in modes.items()},
        clock_boundary=boundary, measurement_condition=context,
        original_observation=dict(artifact=reference, locator=f'#/cases/{case_ordinal}'))
    evidence = dict(schema=EVIDENCE, schema_version=1, boundary=BOUNDARY, context=exact,
        warmup_pairs=case['warmup_pairs'], measured_pairs=case['measured_pairs'],
        samples=[dict(row, order=_ORDERS[row['order']]) for row in case['samples']])
    return target, resources, modes, evidence


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False, indent=2) + '\n')


def _published_ref(reference, output):
    return dict(reference, uri=str(output / Path(reference['uri']).name))


def _published_selection(selection, evidence_ref, profile_ref, raw_ref, output):
    return dict(selection, original_evidence=_published_ref(evidence_ref, output),
                cost_profile=_published_ref(profile_ref, output), original_observation=raw_ref)


def _build_boundary_problem(target, cpu_resources, modes, evidence, profile_ref, selection, queries, output):
    costs = {mode: resolve_cost(query, profile_ref) for mode, query in queries.items()}
    _require(all(cost.basis == 'EMPIRICAL_MODEL' and not cost.missing_terms for cost in costs.values()),
             'resolved trial cost does not match its measured boundary')
    selected_mode = selection['selected_mode']
    eligible = modes if selected_mode == 'TWO_CPU' else ('SERIAL',)
    refs = (evidence['context']['original_observation'],)
    options = []
    for mode in eligible:
        cost = costs[mode]
        cost = replace(cost, source_refs=tuple(dict(item, artifact=_published_ref(item['artifact'], output))
                                               for item in cost.source_refs))
        demands = {resource.id: 1 for cpu in modes[mode] for resource in cpu_resources[cpu]}
        options.append(ImplementationOption(mode, 'MEASURED_STANDALONE_ATEN_CALL', refs,
            demands, {}, {}, {}, cost, dict(mode=mode, affinity_cpu_ids=modes[mode],
                executable=evidence['context']['measurement_condition']['executable'],
                source_replacement_authorized=False)))
    problem = PlanningProblem(target, refs, tuple(r for rows in cpu_resources.values() for r in rows),
        (PlanningOperation('aten-call-boundary', refs, (), tuple(options), 'SERIAL'),), (),
        assumptions=('MEASURED_ISOLATED_NATIVE_CALL_WITH_FIXED_INPUT_AND_THREAD_CONTEXT',
            'RESERVE_TESTED_CPU_LANES_FOR_WHOLE_CALL', 'NO_INDEPENDENT_MEMORY_PLACEMENT_OR_WORKSPACE_MODEL'))
    plan = construct_implementations(problem)
    _require([row['option'] for row in plan['selected']] == [selected_mode],
             'common scheduler did not reproduce the admitted trial choice')
    plan['execution_authorized'] = False
    plan['deployment_authorized'] = False
    return problem, plan


def plan_aten_boundary(observations, expected_sha256, hardware, case, out):
    """Build one common planning operation for the measured whole native call."""
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    raw_ref = artifact_reference(Path(observations).resolve(), OBSERVATIONS)
    _require(raw_ref['sha256'] == expected_sha256, 'original observation digest differs')
    target, cpu_resources, modes, evidence = _admit_observation(raw_ref, hardware, case)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-aten-boundary-', dir=out.parent) as tmp:
        stage = Path(tmp)
        _write(stage / 'evidence.json', evidence)
        evidence_ref = artifact_reference(stage / 'evidence.json', EVIDENCE)
        selection, profile, queries = project_aten_boundary_trial(
            evidence_ref, expected_context=evidence['context'])
        _write(stage / 'cost-profile.json', profile)
        profile_ref = artifact_reference(stage / 'cost-profile.json', COST_PROFILE)
        problem, plan = _build_boundary_problem(target, cpu_resources, modes, evidence,
                                                 profile_ref, selection, queries, out)
        selection = _published_selection(selection, evidence_ref, profile_ref, raw_ref, out)
        _write(stage / 'problem.json', problem_document(problem))
        _write(stage / 'plan.json', plan)
        _write(stage / 'selection.json', selection)
        stage.rename(out)
    return selection


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar plan-aten-boundary', description=__doc__)
    parser.add_argument('--observations', required=True)
    parser.add_argument('--observations-sha256', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    selection = plan_aten_boundary(args.observations, args.observations_sha256, args.target, args.case, args.out)
    print(json.dumps(dict(scope=SCOPE, selected=selection['selected_mode'], reasons=selection['reasons'],
                          output=str(Path(args.out).resolve()))))
    return 0
