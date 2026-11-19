"""Join same-run producer intervals and observed value readiness to the source graph."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

from scar.analysis.bounded_entry_calls import SCHEMA as ENTRY
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization import native_entry_boundary_cost as boundary_cost
from scar.optimization.sparse_solve_cost import _duration
from scar.runtime import source_value_observe as source_probe

SCHEMA = 'scar.native-producer-service-cost'
DESIGN = source_probe.DESIGN
OBSERVATION = source_probe.OBSERVATION
STDOUT = 'scar.source-value-route-stdout-file'
CLOCK = source_probe.CLOCK
RELATIONS = {'SAME_OBJECT_SAME_BYTES', 'SAME_REGION_SAME_BYTES', 'EQUAL_OBSERVED_BYTES'}


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _reference(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _bindings(design, graph):
    anchor = read_artifact(design['source_graph'])
    anchor_program, program = (read_artifact(row['program']) for row in (anchor, graph))
    _require(graph['schema_version'] == anchor['schema_version']
        and all(graph[key] == anchor[key] for key in ('entry', 'entry_scope', 'scope'))
        and program['entry'] == anchor_program['entry'] == design['entry']['uri']
        and program['target'] == anchor_program['target'],
        'service graph must project the same source entry and host inventory')
    original = boundary_cost._observations(anchor)
    targets = defaultdict(list)
    for locator, row in boundary_cost._observations(graph).items():
        if row['entry_scope_reachable']:
            targets[boundary_cost._identity(row)].append((locator, row))
    bindings = {}
    for key, endpoint in design['endpoints'].items():
        anchor_location = endpoint['observation']
        attempt = original[anchor_location]
        source = attempt['observation']
        matches = [(pointer, candidate) for pointer, candidate in targets[boundary_cost._identity(attempt)]
                   if candidate['operation'] == attempt['operation'] and candidate['site_id'] == attempt['site_id']
                   and all(candidate['observation'][field] == source[field]
                           for field in ('caller_site', 'source_call', 'target_code'))]
        _require(len(matches) == 1 and matches[0][1]['source_body'] is not None,
            'expected one exact expanded call with an admitted body for endpoint ' + key)
        locator, target = matches[0]
        bindings[key] = dict(observation=locator, anchor_observation=anchor_location,
            operation=target['operation'], source_port=source['source_call'])
    return bindings


def _field(field):
    shape, strides, dtype, nbytes, digest = (field[key] for key in ('shape', 'strides', 'dtype', 'nbytes', 'sha256'))
    _require(type(dtype) is str and bool(dtype) and type(nbytes) is int and nbytes >= 0
        and len(shape) == len(strides) and all(type(x) is int and x >= 0 for x in shape)
        and all(type(x) is int for x in strides) and re.fullmatch('[0-9a-f]{64}', digest) is not None,
        'observed field has no exact bounded shape, layout and byte fingerprint')
    return dtype, shape, nbytes, digest


def _activation_receipts(observation, design, endpoint_bindings, return_sites):
    activations = observation['activations']
    run_start, run_end = observation['run_ns']
    _duration(observation['run_ns'])
    _require(set(activations) == set(design['endpoints']), 'observation does not cover every selected endpoint')
    by_activation = {row['activation']: key for key, row in activations.items()}
    _require(len(by_activation) == len(activations)
        and all(type(key) is int and key >= 0 for key in by_activation),
        'selected endpoints must have distinct dynamic activations')
    bindings = {}
    parents = {}
    for key, endpoint in design['endpoints'].items():
        raw = activations[key]
        binding = endpoint_bindings[key]
        wall, cpu = raw['wall_ns'], raw['thread_cpu_ns']
        accepted_parent = by_activation.get(raw['parent_activation'])
        _require(raw['caller_site'] == endpoint['caller_site'] and raw['target_code'] == endpoint['target_code']
            and raw['thread_id'] == observation['owner_thread_id']
            and endpoint['phase'] in {'ENTRY', 'RETURN'}
            and len(wall) == len(cpu) == (2 if endpoint['phase'] == 'RETURN' else 1)
            and all(type(value) is int and value > 0 for value in (*wall, *cpu))
            and run_start <= wall[0] <= wall[-1] <= run_end
            and (endpoint.get('parent') is None or accepted_parent == endpoint['parent']),
            'activation phase, source identity, owner thread or selected parent changed')
        if len(wall) == 2:
            _require(type(raw['return_offset']) is int and raw['return_offset'] in return_sites[key]
                and _duration(cpu) <= _duration(wall), 'producer return lacks a valid body interval')
        else:
            _require(raw['return_offset'] is None, 'entry-only event cannot establish a return duration')
        if accepted_parent is not None:
            parent = activations[accepted_parent]
            _require(parent['wall_ns'][0] <= wall[0] and parent['thread_cpu_ns'][0] <= cpu[0]
                and parent['activation'] < raw['activation'],
                'selected parent was not entered before its child')
            if len(parent['wall_ns']) == 2:
                _require(wall[-1] <= parent['wall_ns'][1] and cpu[-1] <= parent['thread_cpu_ns'][1],
                    'selected parent did not contain its observed child')
        parents[key] = accepted_parent
        _require(set(raw['fields']) == set(endpoint['fields']), 'activation has a different field selection')
        for field in raw['fields'].values():
            _field(field)
        bindings[key] = dict(binding, target_code=endpoint['target_code'], caller_site=endpoint['caller_site'],
            activation=raw['activation'], parent_activation=raw['parent_activation'], phase=endpoint['phase'])
    _require(observation['captured_field_bytes'] == sum(field['nbytes']
        for row in activations.values() for field in row['fields'].values()), 'capture field bytes differ')
    return activations, bindings, parents


def _owner_service(activations, parents, bindings):
    timed = {key for key, row in activations.items() if len(row['wall_ns']) == 2}
    ordered = sorted(timed, key=lambda key: (activations[key]['wall_ns'][0], -activations[key]['wall_ns'][1]))
    children = defaultdict(list)
    stack, roots = [], []
    for key in ordered:
        wall = activations[key]['wall_ns']
        while stack and activations[stack[-1]]['wall_ns'][1] <= wall[0]:
            stack.pop()
        ancestor = parents[key]
        while ancestor is not None and ancestor not in timed:
            ancestor = parents[ancestor]
        if stack:
            _require(ancestor == stack[-1], 'overlapping intervals lack the exact selected ancestor')
            children[ancestor].append(key)
        else:
            _require(ancestor is None, 'selected measured parent does not overlap its child')
            roots.append(key)
        stack.append(key)

    result = {}
    for key in ordered:
        raw = activations[key]
        nested = children[key]
        elapsed = _duration(raw['wall_ns'])
        cpu = _duration(raw['thread_cpu_ns'])
        nested_wall = sum(_duration(activations[child]['wall_ns']) for child in nested)
        result[key] = dict(bindings[key], wall_interval_ns=raw['wall_ns'], inclusive_wall_ns=elapsed,
            inclusive_thread_cpu_ns=cpu, measured_nested_intervals=nested,
            accounted_as_separate_root=key in roots,
            parent_residual_after_measured_children_ns=elapsed-nested_wall if nested else None,
            exclusive_self_ns=None, includes_probe_overhead=True)
    return dict(intervals=result, disjoint_root_endpoints=roots,
        covered_single_owner_wall_ns=sum(result[key]['inclusive_wall_ns'] for key in roots),
        covered_single_owner_thread_cpu_ns=sum(result[key]['inclusive_thread_cpu_ns'] for key in roots),
        aggregation='SUM_DISJOINT_SELECTED_INCLUSIVE_ROOTS_ONLY',
        parent_residual_scope='UNATTRIBUTED_REMAINDER_IN_SAME_RUN')


def _value_edges(observation, design, activations, bindings):
    edges = []
    dependencies = observation['routes']
    planned = design['routes']
    selectors = []
    for raw in dependencies:
        producer, consumer = (raw[key] for key in ('producer', 'consumer'))
        source, sink = activations[producer], activations[consumer]
        _require(len(source['wall_ns']) == 2 and len(sink['wall_ns']) >= 1
            and source['wall_ns'][1] <= sink['wall_ns'][0]
            and source['thread_cpu_ns'][1] <= sink['thread_cpu_ns'][0]
            and source['activation'] < sink['activation'],
            'observed value route must run from one producer return to a later consumer entry')
        fields, selector = [], []
        for field in raw['fields']:
            selected = {key:field[key] for key in ('name', 'producer_field', 'consumer_field')}
            source_field = source['fields'][selected['producer_field']]
            target_field = sink['fields'][selected['consumer_field']]
            _require(field['relation'] in RELATIONS and _field(source_field) == _field(target_field)
                and (field['relation'] != 'SAME_OBJECT_SAME_BYTES' or source_field == target_field),
                'route bytes or observed storage relation differs from the selected operand')
            fields.append(dict(selected, relation=field['relation'], evidence=source_field))
            selector.append(selected)
        selectors.append(dict(producer=producer, consumer=consumer, fields=selector))
        edges.append(dict(producer=producer, consumer=consumer, producer_binding=bindings[producer],
            consumer_binding=bindings[consumer], fields=fields,
            producer_ready_at_return_ns=source['wall_ns'][1], consumer_entry_ns=sink['wall_ns'][0],
            observed_elapsed_between_boundaries_ns=sink['wall_ns'][0]-source['wall_ns'][1],
            transport_service_ns=None, separate_independence_proven=False))
    _require(selectors == planned and bool(edges), 'runtime value routes differ from the exact capture selectors')
    return edges


def project(boundary_path, entry_path, observation_path, *, stdout_path=None):
    boundary_ref = _reference(boundary_path, boundary_cost.SCHEMA)
    archived = boundary_cost.check(boundary_ref['uri'])
    entry_ref = _reference(entry_path, ENTRY)
    graph = read_artifact(entry_ref)
    observation_ref = _reference(observation_path, OBSERVATION)
    observation = read_artifact(observation_ref)
    design_ref = observation['design']
    design, return_sites = source_probe.prepare(design_ref)
    launch = observation['launch']
    original = read_artifact(archived['design'])
    lane = design['lane']
    _require(archived['schema_version'] == 2 and design_ref['schema'] == DESIGN
        and design['schema_version'] == observation['schema_version'] == 1
        and design['entry'] == original['source']
        and design['argv'] == original['context']['argv'] == launch['argv']
        and observation['lane'] == lane and lane['required_affinity'] == [lane['cpu_id']]
        and observation['context'] == design['context'] == {**original['context'], 'cpu_ids': [lane['cpu_id']]}
        and observation['clock'] == CLOCK and launch['entry_path'] == design['entry']['uri']
        and launch['exit_code'] == 0 and launch['installed_bindings'] == 0
        and launch['statistics']['native_calls'] == 0 and observation['diagnostic'] is True
        and observation['instrument'] == _reference(source_probe.__file__, boundary_cost.SOURCE)
        and observation['proves_future_value'] is False
        and observation['value_scope'] == 'SAME_PROCESS_SELECTED_ARRAY_ENDPOINTS',
        'source timing requires the original diagnostic probe and no deployment')
    _require(all(type(observation[key]) is int and observation[key] > 0 for key in ('pid', 'owner_thread_id')),
        'service clocks must belong to the exact original source run and physical owner lane')
    stdout = observation.get('observed_stdout')
    supplied_stdout = _reference(stdout_path, STDOUT) if stdout_path is not None else None
    if supplied_stdout is not None:
        _require(stdout is None or stdout['sha256'] == supplied_stdout['sha256'], 'stdout witnesses disagree')
        stdout = supplied_stdout
    if stdout is not None:
        read_artifact(stdout, json_document=False)
        _require(stdout['sha256'] == original['stdout_sha256'], 'same-run output changed from the original entry')
    archived_projection = boundary_cost._project_entry_graph(entry_ref['uri'], original)
    activations, bindings, parents = _activation_receipts(
        observation, design, _bindings(design, graph), return_sites)
    service = _owner_service(activations, parents, bindings)
    for row in service['intervals'].values():
        old = archived_projection['timed_from_stage_by_site'].get(row['observation'])
        row['separate_archive_site'] = old['measured_graph_observation'] if old is not None else None
    edges = _value_edges(observation, design, activations, bindings)
    return dict(schema=SCHEMA, schema_version=1, inputs=dict(archived_boundary=boundary_ref, entry_graph=entry_ref,
        observation=observation_ref, design=design_ref, supplied_stdout=supplied_stdout),
        evidence_scope='ONE_ORIGINAL_DIAGNOSTIC_RUN',
        owner=dict(lane=lane, pid=observation['pid'], thread_id=observation['owner_thread_id'], capacity=1),
        clock=CLOCK, source_graph_entry=graph['entry'], observation_anchor=design['source_graph'], original_stdout=stdout,
        external_stdout_output_bytes_verified=stdout is not None,
        owner_service=service, data_dependencies=edges,
        downstream_handoff=dict(
            latest_observed_producer_return_ns=max(edge['producer_ready_at_return_ns'] for edge in edges),
            all_outputs_simultaneously_live_proven=False, conversion_or_release_duration_ns=None,
            cross_lane_cutover_qualified=False, independent_producer_schedule_proven=False),
        archived_boundary=dict(use='SEPARATE_RUN_REFERENCE_ONLY', lane=archived['lane'],
            additive_or_subtractive_with_current_service=False),
        unresolved=dict(nonselected_nested_work_ns=None, consumer_service_ns=None,
            physical_transfer_ns=None, peak_live_bytes=None, full_entry_cost_ns=None),
        clean_benchmark=False, deployment_authorized=False)


def check(path):
    receipt = read_artifact(_reference(path, SCHEMA))
    inputs = receipt['inputs']
    supplied = inputs['supplied_stdout']
    _require(receipt == project(*(inputs[key]['uri'] for key in ('archived_boundary', 'entry_graph', 'observation')),
                                stdout_path=supplied['uri'] if supplied is not None else None),
        'native producer service cost does not replay')
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', type=Path)
    parser.add_argument('--stdout', type=Path)
    for name in ('boundary', 'entry-graph', 'observation', 'out'):
        parser.add_argument('--'+name, type=Path)
    args = parser.parse_args(argv)
    if args.check:
        print(len(check(args.check)['data_dependencies']))
        return
    if not all((args.boundary, args.entry_graph, args.observation, args.out)):
        parser.error('provide boundary, expanded entry graph, same-run observation and output')
    receipt = project(args.boundary, args.entry_graph, args.observation, stdout_path=args.stdout)
    with args.out.open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
