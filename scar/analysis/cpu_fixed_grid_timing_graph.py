"""Project two fixed-grid routes onto source-backed owner timing intervals."""

from hashlib import sha256
from pathlib import Path

from scar.analysis.cpu_fixed_grid_input_route_graph import verify_cpu_fixed_grid_input_route_graph
from scar.analysis.replay_transaction import replay_transaction
from scar.ir.execution_recipe import _canonical, read_artifact


SCHEMA = 'scar.cpu-fixed-grid-timing-graph'
_PHASES = ('before_recurrence', 'recurrence', 'after_recurrence')
_PYTHON, _MAPPED = 'ORIGINAL_PYTHON', 'PROVIDER_MAP'
_DOMAIN = 'STABLE_READ_ONLY_EXACT_SIZE_REGULAR_CPU_FLOAT32_NATIVE_ENDIAN_FILES'


def _need(ok, detail):
    if not ok:
        raise ValueError('CPU fixed-grid timing graph: ' + detail)


def _ref(document, pointer):
    return dict(document=document, pointer=pointer)


def _owner_order(route):
    operations = route['operations']
    outgoing, incoming = {}, {}
    for index, row in enumerate(route['edges']):
        source, target = row['source'], row['target']
        _need(source in operations and target in operations, 'retained dependency has an unknown endpoint')
        if row['kind'] == 'NORMAL_ORDER':
            _need(row['scope'] == 'SAME_ENTRY' and source not in outgoing and target not in incoming,
                  'the owner order is not one linear normal-success activation')
            outgoing[source], incoming[target] = (target, index), source
    _need(set(operations) - outgoing.keys() == {'teardown'}
          and set(operations) - incoming.keys() == {'launch'},
          'the normal-success owner path must run from launch through teardown')
    ordered, sites = ['launch'], []
    while ordered[-1] in outgoing:
        target, site = outgoing[ordered[-1]]
        _need(target not in ordered, 'owner phase partition must not expand a cyclic outer path')
        ordered.append(target)
        sites.append(site)
    _need(len(ordered) == len(operations) and 'recurrence' in ordered[1:-1],
          'the counted recurrence has no unique retained owner position')
    pivot = ordered.index('recurrence')
    groups = (ordered[:pivot], [ordered[pivot]], ordered[pivot+1:])
    owner_phase = {operation: stage for stage, group in enumerate(groups) for operation in group}
    _need(all(owner_phase[row['source']] <= owner_phase[row['target']] for row in route['edges']),
          'one retained dependency runs backwards across timing boundaries')
    return groups, (sites[pivot-1], sites[pivot])


def _boundary(site, *, mode, lifecycle_route, mapped_project=None):
    original_site = lifecycle_route['sites'][site]
    reference = original_site['source']
    if mode == _MAPPED:
        manifest = read_artifact(mapped_project)
        _need({key: reference[key] for key in ('sha256', 'size_bytes')} == manifest['files']['src/main.cpp']
              and Path(reference['uri']).name == 'main.cpp', 'mapped recurrence markers moved to a different source')
        reference = dict(reference, uri=str(Path(mapped_project['uri']).parent/'src/main.cpp'))
    raw = read_artifact(reference, json_document=False)
    start, end, offset = (original_site[key] for key in ('byte_start', 'byte_end', 'byte_offset'))
    _need(original_site['operation'] == 'recurrence' and 0 < offset < len(raw)
          and raw[offset-1:offset] == b'\n' and 0 <= start < end <= len(raw)
          and sha256(raw[start:end]).hexdigest() == original_site['selection_sha256'],
          'retained recurrence marker differs after source qualification')
    marker = 'RECURRENCE_END' if site == 'RECURRENCE_RETURN' and mode == _PYTHON else site
    return dict(kind='OWNER_SOURCE_MARKER', marker=marker, canonical_marker=site, source=reference,
                byte_start=start, byte_end=end, byte_offset=offset,
                selection_sha256=original_site['selection_sha256'], position=original_site['position'])


def _route(mode, lifecycle, input_graph):
    python = mode == _PYTHON
    name = 'original_python' if python else 'native_cpp'
    prefix = f'#/routes/{name}'
    route = lifecycle['routes'][name]
    groups, cuts = _owner_order(route)
    native = input_graph['routes']['provider_map']
    overrides = native['operation_deltas'] if not python else {}
    _need(not overrides or set(overrides) == {op for op in groups[0] if op.startswith('load:')},
          'mapped route no longer changes exactly the retained source load operations')
    begin, end = (_boundary(site, mode=mode, lifecycle_route=route, mapped_project=input_graph['mapped_project'])
                  for site in ('RECURRENCE_BEGIN', 'RECURRENCE_RETURN'))
    _need(begin['source'] == end['source'] and begin['byte_offset'] < end['byte_offset']
          and begin['position'] == 'BEFORE' and end['position'] == 'AFTER',
          'recurrence window must surround one exact original source statement')
    launch = dict(kind='PARENT_PROCESS_CLOCK', marker='launcher_start_ns', canonical_marker='LAUNCHER_START')
    reap = dict(kind='PARENT_PROCESS_CLOCK', marker='process_reaped_ns', canonical_marker='PROCESS_REAPED')
    boundaries = [(launch, begin), (begin, end), (end, reap)]
    prices = {}
    effects = read_artifact(lifecycle['effects'])
    _need(effects['normal_completion_order']['status'] == 'PROVEN_SCOPED_NORMAL_SUCCESS',
          'file/stdout roles are not connected by the original source normal order')
    for phase, operations, (start, finish) in zip(_PHASES, groups, boundaries, strict=True):
        inherited = [_ref('lifecycle', f'{prefix}/operations/{op}') for op in operations]
        deltas = [_ref('input_graph', f'#/routes/provider_map/operation_deltas/{op}')
                  for op in operations if op in overrides]
        witnesses = []
        if 'initialization' in operations and python:
            import_requests = route['operations']['initialization']['original_import_requests']
            witnesses.extend(_ref('entry_obligations', pointer) for pointer in import_requests)
        for role, key in (('file', 'file_publication'), ('stdout', 'stdout_publication')):
            if key in operations:
                witnesses.append(_ref('source_effects', '#/effects/'+role))
        if 'stdout_publication' in operations:
            witnesses.append(_ref('source_effects', '#/normal_completion_order'))
        resource_ids = sorted({identity for op in operations for identity in route['operations'][op]['resources']})
        prices[phase] = dict(operations=operations, operation_refs=inherited, operation_deltas=deltas,
            effects=witnesses, resource_ids=resource_ids, boundaries=dict(start=start, end=finish),
            resource_basis='DECLARED_OR_ELIGIBLE_TARGET_ROLES', measured_interval_ns=None,
            operation_durations_present=False, bounded_elapsed_proven=False,
            includes_unattributed_work_inside_window=True)
    read_ports = sorted(op for op in groups[0] if op.startswith('load:'))
    effect_routes = [f"#/ports/{index}" for index, port in enumerate(input_graph['ports'])
                     if port['operation'] in read_ports]
    _need(len(effect_routes) == len(read_ports), 'one scoped materialization effect is required for each graph port')
    prices['before_recurrence']['effects'].extend(_ref('input_graph', pointer) for pointer in effect_routes)
    order = [dict(source=left, target=right, kind='OWNER_SERIAL_SCOPE',
                  shared_boundary=prices[left]['boundaries']['end']['canonical_marker'],
                  retained_normal_order=_ref('lifecycle', f'{prefix}/edges/{edge}'))
             for left, right, edge in zip(_PHASES[:-1], _PHASES[1:], cuts, strict=True)]
    return dict(lifecycle_route=_ref('lifecycle', prefix), executable=route['executable'] if python else native['executable'],
        project=None if python else input_graph['mapped_project'],
        launch_environment=route['observed_launch_environment'], effective_environment=route['source_effective_environment'],
        formal_cpu_cohort=route['formal_cpu_cohort'], declared_lane_placement=route['lane_placement'],
        phases=prices, phase_order=order, physical_scheduling_complete=False,
        recurrence_scope='SOURCE_SOLVER_CALL_WITH_OPAQUE_INNER_SCHEDULE' if python else
                         'SYNC_NATIVE_REPEAT_WITH_CERTIFIED_RETAINED_LANES')


def _counted(lifecycle, input_graph):
    plan = read_artifact(input_graph['mapped_plan'])
    proof = read_artifact(input_graph['proof'])
    abi = read_artifact(proof['abi'])
    binding = read_artifact(proof['physical_binding'])
    execution = read_artifact(input_graph['recurrence']['execution'])
    effects = read_artifact(lifecycle['effects'])
    scope = lifecycle['scope']
    argv, steps = scope['argv'], scope['steps']
    index, output = (abi['argv'][key] - 1 for key in ('steps', 'output'))
    domain, control = abi['domain'], execution['control']
    _need(abi['argv'] == plan['scope']['source_argv'] == effects['source_argv']
          and len(argv) == abi['argv']['count'] and 0 <= index < len(argv) and 0 <= output < len(argv)
          and index != output and type(steps) is int and argv[index] == str(steps)
          and steps == input_graph['recurrence']['steps'] == effects['fixed_grid']['steps']
          and domain == plan['scope']['source_domain'] and domain['normal_completion_only'] is True
          and domain['inputs'] == _DOMAIN and domain['output'] == 'FRESH_FILE_WITH_EXISTING_PARENT'
          and domain['step_count_syntax'] == 'ASCII_DECIMAL_DIGITS_WITHOUT_SIGN_OR_WHITESPACE'
          and domain['step_count_min'] <= steps <= domain['step_count_max']
          and control == binding['recurrence']['control'] and control['kind'] == 'FIXED_COUNT_LOOP'
          and control['body_count'] == 1 and control['carried_state']['distance'] == 1
          and input_graph['recurrence']['retained_shared_body'] is True,
          'steps must parametrize one admitted recurrence and preserve the exact source ABI')
    return dict(coordinate='steps', argv_index=index, reference_steps=steps,
        affine_work=dict(constant=0, coefficient=1, unit='REQUESTED_FIXED_GRID_INTERVAL', duration_law_proven=False),
        native_counted_body=dict(reference=input_graph['recurrence']['execution'], control='#/control',
            body_count=1, loop_unrolled=False, native_iterations_per_step=1),
        python_solver=dict(source_call=effects['value_bindings']['recurrence']['operation_pointer'],
            requested_intervals_follow_coordinate=True, inner_operation_counts_or_schedule_proven=False),
        atlas=dict(
            work_coordinate='steps', admissible_steps=dict(minimum=domain['step_count_min'],
                maximum=domain['step_count_max'], syntax=domain['step_count_syntax']),
            step_size=abi['solver']['step_size'],
            argv=dict(variable_step_index=index, private_publication_index=output,
                fixed_arguments=[dict(index=i, value=argument) for i, argument in enumerate(argv) if i not in (index, output)]),
            file_identity_basis='EXACT_PORT_AND_SOURCE_FILE_CONTENT_WITH_STABLE_INPUTS',
            file_ports=[_ref('input_graph', f'#/ports/{i}') for i in range(len(input_graph['ports']))],
            environment_and_affinity_basis='PER_ROUTE_EFFECTIVE_ENVIRONMENT_AND_EXACT_LAUNCH_COHORT',
            same_count_per_pair_required=True, same_source_target_project_and_schedule_required=True,
            fresh_output_relocation_during_calibration_only=True, exact_stdout_and_final_tensor_per_count_required=True,
            route_admission_extends_to_unmeasured_counts=False, affine_time_or_unmeasured_bounds_proven=False,
            clean_whole_entry_measurements_separate_from_marked_occupation=True))


def derive_cpu_fixed_grid_timing_graph(lifecycle_ref, input_graph_ref):
    """Bind owner phase coverage and an affine work count without predicting elapsed time."""
    with replay_transaction():
        input_graph = verify_cpu_fixed_grid_input_route_graph(input_graph_ref)
        lifecycle = read_artifact(lifecycle_ref)
        _need(input_graph['lifecycle'] == lifecycle_ref and input_graph['selected_runtime'] == _MAPPED
              and input_graph['target'] == lifecycle['target'] and input_graph['scope'] == lifecycle['scope']
              and input_graph['source'] == lifecycle['source'] and input_graph['resources'] == lifecycle['resources'],
              'selected mapped input graph must use the same verified lifecycle and invocation')
        counted = _counted(lifecycle, input_graph)
        routes = {mode: _route(mode, lifecycle, input_graph) for mode in (_PYTHON, _MAPPED)}
        source_effects = read_artifact(lifecycle['effects'])
    return dict(schema=SCHEMA, schema_version=1, status='PROVEN_SCOPED_OWNER_TIMING_PARTITION',
        lifecycle=lifecycle_ref, input_graph=input_graph_ref, source=lifecycle['source'], target=lifecycle['target'],
        scope=lifecycle['scope'], resources=lifecycle['resources'], source_effects=lifecycle['effects'],
        entry_obligations=source_effects['entry_obligations'], routes=routes, counted=counted,
        coverage_frontier=['UNOBSERVED_INNER_PYTHON_AND_PROVIDER_SCHEDULE', 'SUBOP_DURATION_AND_RESOURCE_OCCUPANCY',
            'PHYSICAL_PAGE_FAULTS_AND_STORAGE_TRAFFIC', 'TIME_SCALING_OR_EXTRAPOLATION_BEYOND_ATLAS_CELLS'],
        execution_default='KEEP_ORIGINAL', performance_bounds_proven=False, transformation_authorized=False,
        deployment_authorized=False, target_executed=False)


def verify_cpu_fixed_grid_timing_graph(reference):
    """Cold-replay the source/operation partition and its exact work-coordinate contract."""
    _need(reference['schema'] == SCHEMA, 'unexpected fixed-grid timing graph schema')
    saved = read_artifact(reference)
    expected = derive_cpu_fixed_grid_timing_graph(saved['lifecycle'], saved['input_graph'])
    _need(_canonical(saved) == _canonical(expected), 'saved fixed-grid timing graph differs from cold replay')
    return saved
