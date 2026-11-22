"""Replay disjoint source-bound native phase observations for diagnostics."""

from scar.analysis.cpu_entry_lifecycle_graph import SCHEMA as LIFECYCLE_GRAPH
from scar.ir.execution_recipe import read_artifact
from scar.optimization.cpu_entry_resources import ADMISSIBLE
from .cpu_entry_forward_cost import verify_cpu_entry_forward_cost
from .whole_entry_choice import LIFECYCLE_PHASES


SCHEMA = 'scar.cpu-native-lifecycle-cost'
OBSERVATION = 'scar.cpu-native-array-lifecycle-observation'
_EVENTS = ('PARENT_START', 'MAIN_ENTER', 'DISPATCH_BEGIN', 'DISPATCH_RETURN',
           'PUBLICATION_BEGIN', 'PUBLICATION_END', 'PARENT_REAP')
_WINDOWS = (
    ('startup_to_main', ('startup', 'guards'), ('start', 'guard'), 'PARENT_LAUNCH_GUARD_AND_LOADER_COMPOSITE'),
    ('native_ingress_guards', ('guards',), (), 'NATIVE_ARGUMENT_AND_PROVIDER_GUARDS'),
    ('dispatch_and_join', ('worker_lifecycle', 'execution', 'synchronization'), ('team', 'work', 'join'),
     'NUMERICAL_TEAM_DISPATCH_AND_JOIN_COMPOSITE'),
    ('post_dispatch', (), (), 'SUCCESS_PATH_TRANSITION'),
    ('publication_statements', ('publication',), ('publication',), 'NATIVE_PUBLICATION_STATEMENTS'),
    ('publication_end_to_reap', ('teardown',), ('exit',), 'CLOCK_MARKER_OUTPUT_FLUSH_AND_PROCESS_REAP_COMPOSITE'),
)


def _require(condition, message):
    if not condition:
        raise ValueError('CPU native lifecycle cost: ' + message)


def _statistics(samples):
    ordered = sorted(samples)
    count = len(ordered)
    return dict(samples_ns=samples, min_ns=ordered[0],
                median_ns_twice=ordered[(count - 1)//2] + ordered[count//2], max_ns=ordered[-1])


def _collect(rows):
    _require(type(rows) is list and bool(rows), 'clock samples are required')
    windows = [[] for _ in _WINDOWS]
    total, last, measured = [], 0, False
    for index, row in enumerate(rows):
        marks = row['markers_ns']
        warmup = row['warmup']
        _require(type(row['ordinal']) is int and row['ordinal'] == index and type(warmup) is bool
                 and (not measured or not warmup) and type(marks) is dict and set(marks) == set(_EVENTS[1:-1]),
                 'clock samples must retain ordinal, initial warmups and exact source marks')
        ticks = [row['started_ns'], *(marks[name] for name in _EVENTS[1:-1]), row['reaped_ns']]
        _require(all(type(value) is int and value > 0 for value in ticks)
                 and last <= ticks[0] < ticks[-1] and ticks == sorted(ticks),
                 'clock marks must be monotonic inside one parent launch and reap')
        last = ticks[-1]
        if not warmup:
            measured = True
            for samples, begin, end in zip(windows, ticks[:-1], ticks[1:], strict=True):
                samples.append(end - begin)
            total.append(ticks[-1] - ticks[0])
    _require(measured, 'at least one non-warmup sample is required')
    return windows, total


def _context(graph_ref, graph, observation):
    from scar.analysis.cpu_native_phase_sites import SCHEMA as PHASE_SITES

    cap_ref = graph['capacity_contract']
    _require(graph_ref['schema'] == LIFECYCLE_GRAPH and cap_ref is not None and observation['graph'] == graph_ref
             and observation['capacity_contract'] == cap_ref
             and observation['scope_sha256'] == graph['scope_sha256'],
             'observed phase marks must identify the exact lifecycle graph')
    capacity = read_artifact(cap_ref)
    sites_ref = observation['phase_sites']
    sites = read_artifact(sites_ref)
    option = observation['option_id']
    matches = [(ordinal, lane) for ordinal, lane in enumerate(graph['lanes']) if lane['option_id'] == option]
    _require(capacity['status'] == ADMISSIBLE and len(matches) == 1 and option == capacity['option_id'] == sites['option_id']
             and sites_ref['schema'] == PHASE_SITES and sites['lifecycle_graph'] == graph_ref
             and sites['capacity_contract'] == cap_ref and sites['scope_sha256'] == observation['scope_sha256'],
             'phase sites must name the admitted candidate option and source graph')
    index, lane = matches[0]
    launch = capacity['native_launch']
    executable = observation['original_executable']
    diagnostic = observation['diagnostic_executable']
    _require(lane['role'] == 'CONDITIONAL_NATIVE_CANDIDATE'
             and observation['runtime'] == sites['runtime'] == capacity['evidence']['runtime']
             and executable == sites['binary'] == launch['executable'] and diagnostic != executable
             and all(observation[name] == launch[name] for name in ('cpu_ids', 'cwd', 'environment_sha256')),
             'instrumented sibling must bind the exact original native ELF and guarded launch context')
    locations = {row['id']: dict(artifact=sites_ref, locator=f'#/sites/{i}') for i, row in enumerate(sites['sites'])}
    _require(list(locations) == list(_EVENTS[1:-1]), 'ordered explicit source sites required')
    segments = {(row['begin'], row['end']): dict(artifact=sites_ref, locator=f'#/timed_segments/{i}')
                for i, row in enumerate(sites['timed_segments'])}
    _require(list(segments) == list(zip(_EVENTS[1:-2], _EVENTS[2:-1], strict=True)),
             'source windows must cover adjacent marker pairs only')
    operation_refs = {row['id']: dict(artifact=graph_ref, locator=f'#/lanes/{index}/operations/{i}')
                      for i, row in enumerate(lane['operations'])}
    return capacity, locations, segments, operation_refs


def derive_cpu_native_lifecycle_cost(lifecycle_graph_ref, observation_ref, *, forward_cost=None):
    """Return instrumented phase windows and an independently replayed optional forecast."""
    from scar.codegen.native_array_lifecycle import check

    _require(observation_ref['schema'] == OBSERVATION, 'source-bound native observation artifact required')
    observation = read_artifact(observation_ref)
    _require(observation == check(observation_ref['uri']), 'native observation differs from cold probe replay')
    # The probe replay already verifies the phase sites, lifecycle graph and capacity.
    graph = read_artifact(lifecycle_graph_ref)
    capacity, source_sites, segments, operations = _context(lifecycle_graph_ref, graph, observation)
    raw_events = ('started_ns', *_EVENTS[1:-1], 'reaped_ns')
    declared = tuple(observation['windows'].values())
    _require(len(declared) == len(_WINDOWS) and {tuple(pair) for pair in declared}
             == set(zip(raw_events[:-1], raw_events[1:], strict=True))
             and observation['clock'] == 'LINUX_CLOCK_MONOTONIC_NS', 'probe and model clock window contracts differ')
    series, total = _collect(observation['samples'])
    rows = []
    for offset, (window, samples) in enumerate(zip(_WINDOWS, series, strict=True)):
        identity, phases, operation_ids, coverage = window
        begin, end = _EVENTS[offset:offset + 2]
        rows.append(dict(id=identity, phases=list(phases), basis=coverage,
            begin=dict(clock_event=begin, source_site=source_sites.get(begin)),
            end=dict(clock_event=end, source_site=source_sites.get(end)),
            source_window=segments.get((begin, end)),
            overlapping_graph_operations=[operations[name] for name in operation_ids],
            **_statistics(samples)))
    forecast = None
    if forward_cost is not None:
        verified = verify_cpu_entry_forward_cost(forward_cost)
        _require(verified['lifecycle_graph'] == lifecycle_graph_ref
                 and verified['ledger'] == capacity['ledger'] and verified['scope_sha256'] == observation['scope_sha256']
                 and verified['option_id'] == observation['option_id'] and verified['whole_entry_ranked'] is False,
                 'conditional forward estimate belongs to a different option, ledger or lifecycle graph')
        forecast = dict(artifact=forward_cost, covered_subgraph=verified['covered_subgraph'],
                        status=verified['status'], adjusted_from_diagnostic_samples=False)
    return dict(schema=SCHEMA, schema_version=1, lifecycle_graph=lifecycle_graph_ref,
        observation=observation_ref, phase_sites=observation['phase_sites'], forward_cost=forward_cost,
        scope_sha256=observation['scope_sha256'], option_id=observation['option_id'], exact_legality=capacity['legality'],
        executable_relation=dict(kind='VERIFIED_INSTRUMENTED_SIBLING', admitted=observation['original_executable'],
                                 observed=observation['diagnostic_executable']),
        clock='LINUX_CLOCK_MONOTONIC_NS', status='SOURCE_BOUND_INSTRUMENTED_WINDOWS', windows=rows,
        diagnostic_launch_reap=_statistics(total), conditional_forecast=forecast,
        interference=dict(probe_clock_reads=True, private_marker_emission=True,
            eventual_stdout_flush_may_be_in_teardown=True, calibrated_correction_ns=None),
        missing_lifecycle_terms=list(LIFECYCLE_PHASES), whole_entry_model_bound_ns=dict(lower=None, upper=None),
        sampled_windows_are_model_bounds=False, whole_entry_ranked=False, clean_benchmark=False,
        execution_default='NO_OP', deployment_authorized=False)


def verify_cpu_native_lifecycle_cost(cost_ref):
    saved = read_artifact(cost_ref)
    _require(cost_ref['schema'] == SCHEMA and type(saved['schema_version']) is int and saved['schema_version'] == 1,
             'unsupported diagnostic cost report')
    replay = derive_cpu_native_lifecycle_cost(saved['lifecycle_graph'], saved['observation'],
                                             forward_cost=saved['forward_cost'])
    _require(saved == replay, 'native lifecycle costs differ at cold replay')
    return replay
