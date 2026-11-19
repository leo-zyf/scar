"""Join source-bound entry and exit clocks to the original native planning inputs."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from scar.analysis.bounded_entry_calls import SCHEMA as ENTRY_GRAPH
from scar.analysis.entry_call_dependence import SCHEMA as DEPENDENCE
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.resource_solver import semantic_sha256
from scar.optimization.sparse_solve_cost import _duration, _summary
from scar.runtime.csr_region_entry import launch_context
from scar.runtime.source_boundary_observe import DESIGN, OBSERVATION, SOURCE, lane_snapshot, source_cuts

SCHEMA = 'scar.native-entry-boundary-cost'
DEMANDS = 'scar.stage1-producer-return-demands-observation'
RELEASE = 'scar.sparse-parallel-entry-selection'


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _observations(graph):
    return {f'#/observed_targets/{index}': row for index, row in enumerate(graph['observed_targets'])}


def _identity(row):
    observation = row['observation']
    site = observation['caller_site']
    return (row['operation']['node']['wire'], site['code'], site['offset'], site['opname'], tuple(site['position']),
            json.dumps(json.loads(observation['metadata_json']), sort_keys=True))


def _source_bindings(graph, demands, release, scope):
    request = json.loads(read_artifact(demands['original_request'], json_document=False))
    dependence_ref = request['entry_dependence']
    dependence = read_artifact(dependence_ref)
    old_graph = read_artifact(dependence['entry_calls'])
    summary = read_artifact(demands['source_summary'])
    summary_scope = summary.get('header', summary)
    read_artifact(graph['program'])
    caller = scope['source_body']['code_key']
    _require(dependence_ref['schema'] == DEPENDENCE
        and dependence['selected_body'] == read_artifact(scope['source_scope'])['f4_loop']['source_body']
        and graph['entry'] == old_graph['entry'] and graph['entry_scope'] == old_graph['entry_scope']
        and summary['schema_version'] in (1, 2) and summary_scope['entry_projection'] ==
            dict(sha256=semantic_sha256(graph), program=graph['program'], producer=graph['producer'])
        and scope['entry']['source'] == release['entry'],
        'expanded graph and original input routes must describe the exact released source')
    old = _observations(old_graph)
    expanded = _observations(graph)
    equivalents = defaultdict(list)
    for reference, row in expanded.items():
        if row['entry_scope_reachable']:
            equivalents[_identity(row)].append(reference)

    def join(row):
        matches = equivalents[_identity(row)]
        _require(len(matches) == 1, 'original observed operation has no unique expanded graph binding')
        return matches[0]

    producers = {}
    for path, selected in demands['observed_source_targets'].items():
        for target in selected:
            attempt = expanded[target['observation']]
            code = attempt['observation']['target_code']
            _require(attempt['entry_scope_reachable'] and (code['filename'], code['key'], code['qualname'])
                == (path, target['code_key'], target['qualname']),
                'return-demand child binding changed in the expanded source graph')
    for index, producer in enumerate(demands['producers']):
        selected = expanded[producer['selected_origin']]['observation']
        code = selected['target_code']
        _require(code['key'] == producer['code'] and code['qualname'] == producer['function'],
            'retained return demand has another producer CodeID')
        producers[producer['code']] = f'#/producers/{index}'
    cuts = {}
    for live_index, binding in enumerate(dependence['live_ins']):
        for use in binding['observed_calls']:
            for reference in use['observations']:
                row = old[reference]['observation']
                if not row['caller_site']['opname'].startswith('CALL'):
                    continue
                key = join(old[reference])
                cut = cuts.setdefault(key, dict(kind='ENTRY_INPUT_PRODUCER', dependence=[]))
                cut['dependence'].append(dict(formal=binding['original_route']['formal'],
                    original_live_in=f'#/live_ins/{live_index}', original_call_fact=use['call_fact']))
    publication = scope['dependences']['ordered_publication']
    _require(publication['kind'] == 'ORDERED_HASH_ACCUMULATION'
        and publication['original_return_transport'] and publication['ordered_world'],
        'exit clocks require the declared ordered result and effect')
    digest = join(_observations(read_artifact(scope['source_graph']))[publication['original_final']])
    consumers = dependence['return_consumers']
    consumer, = consumers
    consumer_ref, = consumer['observations']
    _require(consumer['data_route'] and consumer['world_route'] and not consumer['effect_discharge_proven'],
        'original stdout must have explicit result and world routes')
    console = join(old[consumer_ref])
    loop_producers = [reference for reference, row in expanded.items()
        if row['operation'] == dependence['operation'] and row['entry_scope_reachable']
        and (row['observation']['target_code'] or {}).get('key') == caller]
    loop, = loop_producers
    cuts.update({loop: dict(kind='LOOP_CALL_INCLUSIVE'), digest: dict(kind='LOOP_TAIL_FINALIZATION'),
                 console: dict(kind='ORIGINAL_STDOUT_CALL')})

    for reference, cut in cuts.items():
        row = expanded[reference]
        observation = row['observation']
        target = json.loads(observation['metadata_json'])
        code = target['code']
        variant = code or target.get('constructed_type') or target['native_boundary']
        cut.update(original_operation=row['operation'], source_call=observation['source_call'],
            caller_site=observation['caller_site'], target=target,
            label=(variant.get('module', '') + '.' + variant['qualname']).lstrip('.'))
        if code and code['key'] in producers:
            cut['expanded_return_demand'] = producers[code['key']]
    _require({cut['expanded_return_demand'] for cut in cuts.values() if 'expanded_return_demand' in cut}
        == set(producers.values()), 'return-demand producers no longer reach the same exact loop entry')
    prefix = [key for key, cut in cuts.items() if cut['kind'] == 'ENTRY_INPUT_PRODUCER']
    prefix.sort(key=lambda key: cuts[key]['caller_site']['offset'])
    loop_site = cuts[loop]['caller_site']
    _require(all(cuts[key]['caller_site']['code'] == loop_site['code']
        and cuts[key]['caller_site']['offset'] < loop_site['offset'] for key in prefix)
        and cuts[console]['caller_site']['code'] == loop_site['code']
        and cuts[console]['caller_site']['offset'] > loop_site['offset'],
        'original entry phases no longer share the admitted ordered module')
    return dependence_ref, cuts, dict(loop=loop, finalization=digest, stdout=console,
        return_data_and_world='#/return_consumers/0', original_final_result_transport='#/dependences/ordered_publication',
        source_graph_effects_discharged=False)


def build_design(graph_path, demands_path, release_path, *, lane, repetitions):
    graph_ref, demands_ref, release_ref = (_ref(path, schema) for path, schema in (
        (graph_path, ENTRY_GRAPH), (demands_path, DEMANDS), (release_path, RELEASE)))
    graph, demands, release = (read_artifact(ref) for ref in (graph_ref, demands_ref, release_ref))
    _require(demands['entry_projection'] == graph_ref and demands['schema_version'] == 1
        and demands['data_or_effects_closed'] is False and release['count'] > 0
        and type(repetitions) is int and repetitions > 0,
        'design requires the original expanded entry and its published return demands')
    scope = read_artifact(release['source'])
    dependence, cuts, exit_routes = _source_bindings(graph, demands, release, scope)
    source = release['entry']
    _, offsets = source_cuts(source, cuts, release['context']['interpreter']['optimize'])
    for key, offset in offsets.items():
        cuts[key]['after_call_offset'] = offset
    validation = read_artifact(release['validation'])
    stdout, = {row['stdout_sha256'] for row in validation['runs']}
    stderr, = {row['stderr_sha256'] for row in validation['runs']}
    return dict(schema=DESIGN, schema_version=1, inputs=dict(graph=graph_ref, demands=demands_ref,
        release=release_ref, parallel_scope=release['source'], entry_dependence=dependence), source=source,
        context={**release['context'], 'cpu_ids': [lane['cpu_id']]}, lane=lane, cuts=cuts, exit_routes=exit_routes,
        count=release['count'], repetitions=repetitions, stdout_sha256=stdout, stderr_sha256=stderr,
        execution='ORIGINAL_PYTHON_ENTRY_WITHOUT_NATIVE_ATTACHMENT')


def _design(path):
    design_ref = _ref(path, DESIGN)
    value = read_artifact(design_ref)
    inputs = value['inputs']
    rebuilt = build_design(*(inputs[key]['uri'] for key in ('graph', 'demands', 'release')),
                           lane=value['lane'], repetitions=value['repetitions'])
    _require(value == rebuilt, 'boundary design does not replay from the original source and graph')
    return design_ref, value


def collect(path):
    reference, design = _design(path)
    context = design['context']
    _require(launch_context(context['argv']) == context and lane_snapshot(design['lane']['cpu_id']) == design['lane'],
             'run under the exact original interpreter, arguments, thread environment and requested singleton CPU')
    rows = []
    with TemporaryDirectory(prefix='scar-source-boundary-') as folder:
        for index in range(design['repetitions']):
            output = Path(folder) / f'run-{index}.json'
            command = [sys.executable, '-B', '-m', 'scar.runtime.source_boundary_observe',
                '--design', reference['uri'], '--sha256', reference['sha256'], '--report', str(output)]
            worker = subprocess.run(command, capture_output=True, check=True,
                                    timeout=120, cwd=context['cwd'])
            observation = read_artifact(_ref(output, OBSERVATION))
            observation['stdout_pipe_hex'] = worker.stdout.hex()
            observation['stderr_pipe_hex'] = worker.stderr.hex()
            rows.append(observation)
    return dict(schema=OBSERVATION, schema_version=2, design=reference, rows=rows)


def _replay_row(raw, design_ref, design, probe):
    from hashlib import sha256

    _require(raw['schema_version'] == 1 and raw['instrument'] == probe and raw['design'] == design_ref
        and raw['context'] == design['context'] and raw['lane'] == design['lane']
        and raw['diagnostic'] is True and raw['launch']['exit_code'] == 0
        and raw['launch']['installed_bindings'] == 0 and raw['launch']['statistics']['native_calls'] == 0
        and raw['launch']['entry_path'] == design['source']['uri'] and raw['launch']['argv'] == design['context']['argv'],
        'worker observation is not from the exact original entry and physical owner lane')
    output, errors = (bytes.fromhex(raw[field]) for field in ('stdout_pipe_hex', 'stderr_pipe_hex'))
    _require(sha256(output).hexdigest() == design['stdout_sha256']
        and sha256(errors).hexdigest() == design['stderr_sha256'], 'original observable output differs')
    cuts = raw['samples']
    _require(set(cuts) == set(design['cuts']), 'missing original source boundary')
    _duration(raw['run'])
    for key, row in cuts.items():
        planned = design['cuts'][key]
        _require(row['target'] == planned['target'] and row['entry_code'] == planned['caller_site']['code']
            and row['call_offset'] == planned['caller_site']['offset']
            and row['after_call_offset'] == planned['after_call_offset']
            and row['thread_id'] == raw['owner_thread_id']
            and raw['run'][0] <= row['wall'][0] < row['wall'][1] <= raw['run'][1]
            and 0 < _duration(row['thread_cpu']) <= _duration(row['wall']),
            'clock interval differs from the selected source site or owner thread')
    routes = design['exit_routes']
    loop, finish, console = (cuts[routes[key]]['wall'] for key in ('loop', 'finalization', 'stdout'))
    prefix_sites = sorted((key for key, cut in design['cuts'].items() if cut['kind'] == 'ENTRY_INPUT_PRODUCER'),
                          key=lambda key: design['cuts'][key]['caller_site']['offset'])
    prefix = [cuts[key]['wall'] for key in prefix_sites]
    serial = [*prefix, loop, console]
    _require(all(a[1] <= b[0] for a, b in zip(serial, serial[1:]))
        and loop[0] <= finish[0] < finish[1] <= loop[1],
        'source phases do not match original call order and nested finalization')
    return dict(prefix_service_ns=sum(_duration(row) for row in prefix),
        exit_service_ns=_duration(finish)+_duration(console), stdout_pipe_bytes=len(output), stderr_pipe_bytes=len(errors))


def _project_entry_graph(path, design):
    reference = _ref(path, ENTRY_GRAPH)
    measured_ref = design['inputs']['graph']
    old, expanded = (read_artifact(ref) for ref in (measured_ref, reference))
    previous_program, program = (read_artifact(row['program']) for row in (old, expanded))
    _require(expanded['schema_version'] == old['schema_version']
        and all(expanded[key] == old[key] for key in ('entry', 'entry_scope', 'scope'))
        and program['target'] == previous_program['target']
        and program['entry'] == previous_program['entry'] == design['source']['uri'],
        'entry graph projection requires the same original entry root and platform')
    original = _observations(old)
    selected = defaultdict(list)
    for locator, row in _observations(expanded).items():
        if row['entry_scope_reachable']:
            selected[_identity(row)].append((locator, row))

    def bind(locator):
        row = original[locator]
        observed = row['observation']
        matches = [(pointer, target) for pointer, target in selected[_identity(row)]
                   if target['site_id'] == row['site_id'] and target['operation'] == row['operation']
                   and all(target['observation'][key] == observed[key]
                           for key in ('caller_site', 'source_call', 'target_code'))]
        _require(row['entry_scope_reachable'] and len(matches) == 1,
            'expected exactly one matching caller site and actual target for ' + locator)
        pointer, target = matches[0]
        return pointer, dict(measured_graph_observation=locator, operation=target['operation'],
            caller_site=target['observation']['caller_site'], target_code=target['observation']['target_code'])

    timed = dict(bind(key) for key in design['cuts'])
    _require(len(timed) == len(design['cuts']), 'projected timing sites must remain distinct')
    demands = read_artifact(design['inputs']['demands'])
    untimed = {}
    for selected_targets in demands['observed_source_targets'].values():
        for source in selected_targets:
            key, target = bind(source['observation'])
            _require(key not in timed and key not in untimed,
                     'a return-demand child has an ambiguous timing assignment')
            untimed[key] = dict(target, duration_ns=None, scope='NO_SEPARATE_CHILD_TIMING')
    return dict(graph=reference, measurement_graph=measured_ref, match='EXACT_SOURCE_CALL_AND_ACTUAL_TARGET',
        timing_lane=design['lane'], timed_from_stage_by_site=timed, untimed_return_demand_sites=untimed)


def project(design_path, observation_path, *, entry_graph=None):
    from scar.runtime import source_boundary_observe as observer

    design_ref, design = _design(design_path)
    observation_ref = _ref(observation_path, OBSERVATION)
    observation = read_artifact(observation_ref)
    probe = _ref(observer.__file__, SOURCE)
    rows = observation['rows']
    _require(observation['schema_version'] == 2 and observation['design'] == design_ref
        and len(rows) == design['repetitions'] and len({row['pid'] for row in rows}) == len(rows)
        and all(a['run'][1] < b['run'][0] for a,b in zip(rows, rows[1:])),
        'expected independent fresh-process original timings')
    teams = [_replay_row(row, design_ref, design, probe) for row in rows]
    measures = {}
    for key, site in design['cuts'].items():
        measures[key] = dict(label=site['label'], kind=site['kind'],
            wall_ns=_summary([_duration(row['samples'][key]['wall']) for row in rows]),
            thread_cpu_ns=_summary([_duration(row['samples'][key]['thread_cpu']) for row in rows]))
    result = dict(schema=SCHEMA, schema_version=1, design=design_ref, observations=observation_ref,
        source_graph=design['inputs']['graph'], return_demands=design['inputs']['demands'],
        lane=design['lane'], call_count=design['count'], count_basis='SEALED_SOURCE_LOOP_AND_ARGUMENTS',
        stage_by_site=measures, exact_stdout_pipe=True,
        outside_service=dict(prefix_observed_disjoint_ns=_summary([x['prefix_service_ns'] for x in teams]),
            exit_observed_disjoint_ns=_summary([x['exit_service_ns'] for x in teams])),
        stdout_pipe_bytes=teams[0]['stdout_pipe_bytes'], instrumented=True, clean_benchmark=False,
        planning_use=dict(timing_scope='ORIGINAL_SINGLE_CORE_CONDITIONAL_ON_THIS_CONTEXT',
            assign_to_other_cpu_or_native_projection=False, reference_interval='LOOP_INCLUSIVE_CONTAINS_FINAL_DIGEST',
            use_in_whole_entry_sum_without_other_evidence=False),
        unresolved=dict(imports_and_module_gaps_ns=None, loop_without_final_digest_ns=None,
            python_stdout_buffer_flush_and_os_write_service_ns=None, post_stdout_teardown_ns=None,
            worker_lane_or_memory_traffic_ns=None, uninstrumented_whole_entry_ns=None))
    if entry_graph is not None:
        result['schema_version'] = 2
        result['entry_graph_projection'] = _project_entry_graph(entry_graph, design)
    return result


def check(path):
    receipt = read_artifact(_ref(path, SCHEMA))
    projection = receipt.get('entry_graph_projection')
    _require(receipt == project(receipt['design']['uri'], receipt['observations']['uri'],
                                entry_graph=projection['graph']['uri'] if projection is not None else None),
        'original boundary costs differ from their source and clock replay')
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='mode', required=True)
    freeze = actions.add_parser('design')
    for name in ('graph', 'demands', 'release'):
        freeze.add_argument('--'+name, type=Path, required=True)
    freeze.add_argument('--cpu', type=int, required=True)
    freeze.add_argument('--repetitions', type=int, default=3)
    capture = actions.add_parser('measure')
    replay = actions.add_parser('project')
    for action in (capture, replay):
        action.add_argument('--design', type=Path, required=True)
    replay.add_argument('--observations', type=Path, required=True)
    replay.add_argument('--entry-graph', type=Path)
    for action in (freeze, capture, replay):
        action.add_argument('--out', type=Path, required=True)
    verify = actions.add_parser('check')
    verify.add_argument('path', type=Path)
    args = parser.parse_args(argv)
    if args.mode == 'check':
        print(check(args.path)['call_count'])
        return
    _require(not args.out.exists(), 'fresh boundary observation result path required')
    if args.mode == 'design':
        result = build_design(args.graph, args.demands, args.release, lane=lane_snapshot(args.cpu),
                              repetitions=args.repetitions)
    elif args.mode == 'measure':
        result = collect(args.design)
    else:
        result = project(args.design, args.observations, entry_graph=args.entry_graph)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
