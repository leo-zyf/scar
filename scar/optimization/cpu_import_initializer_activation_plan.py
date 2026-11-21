"""Attach diagnostic initializer occurrences to exact F4 source projections."""

from bisect import bisect_left
from collections import Counter, defaultdict

from scar.analysis.replay_transaction import replay_transaction
from scar.analysis.source_decoded_bodies import reuse_decoded_source_bodies
from scar.ir.execution_recipe import read_artifact
from scar.optimization.cpu_import_source_effect_plan import (
    SCHEMA as SOURCE_EFFECTS, verify_cpu_import_source_effect_plan,
)

SCHEMA = 'scar.cpu-import-initializer-activation-plan'
OBSERVATION = 'scar.cpu-import-initializer-observation'
_DIRECT = 'DIRECT_WATCHED_IMPORT'
_TRANSITIVE = 'TRANSITIVE_WITHIN_WATCHED_IMPORT'
_ENCLOSED = 'ENCLOSED_WITH_REQUEST_ORIGIN_OPEN'
_OUTSIDE = 'OUTSIDE_WATCHED_IMPORT'
_UNBOUND = 'NO_COMPLETED_WATCHED_IMPORT_BOUND'


def _require(condition, detail):
    if not condition:
        raise ValueError('CPU import initializer activation plan: ' + detail)


def _module_key(name, source):
    return name, source['uri'], source['sha256'], source['size_bytes'], source['schema']


def _modules(static):
    admission = read_artifact(static['admission'])
    modules, routes = [], defaultdict(list)
    for index, (effect, admitted) in enumerate(zip(static['modules'], admission['modules'], strict=True)):
        _require(effect['admission_module'] == f'#/modules/{index}'
                 and effect['original_import'] == admitted['original_import']
                 and effect['module_name'] == admitted['module_name'], 'source effect route changed its admission')
        key = _module_key(admitted['module_name'], admitted['source'])
        route = dict(
            source_effect=f'#/modules/{index}', initializer=effect['initializer'],
            module_name=effect['module_name'], original_import=effect['original_import'],
            source_representation_status=effect['source_representation_status'],
            activation_observations=[], explicit_cache_hits=[])
        modules.append(route)
        routes[key].append(route)
    return modules, routes


def _enclosure(row, entries):
    index = row['enclosing_import']
    if index is None:
        _require(row['direct_import'] is False, 'direct activation has no enclosing watched import')
        return (_OUTSIDE if row['watched_request'] is None else _UNBOUND), None, None
    _require(type(index) is int and 0 <= index < len(entries), 'invalid enclosing import ordinal')
    enclosing = entries[index]
    _require(row['process_slot'] == enclosing['process_slot'], 'initializer enclosure crosses process slots')
    if row['direct_import']:
        kind = _DIRECT
    elif row['requester'] is not None and row['requester'] != row['watched_request']:
        kind = _TRANSITIVE
    else:
        kind = _ENCLOSED
    return kind, f'#/entries/{index}', enclosing['original_import']


def _activations(observation, routes, processes):
    results, preceding = [], defaultdict(list)
    for index, row in enumerate(observation['activations']):
        slot = row['process_slot']
        _require(slot in processes, 'initializer has no observed process slot')
        key = _module_key(row['module_name'], row['source'])
        _require(key in routes, 'initializer source was outside the original admission')
        kind, enclosure, original_site = _enclosure(row, observation['entries'])
        qualified = (row['outcome'] == 'RETURNED' and row['structural_code_match'] is True
                     and row['standard_source_loader'] is True)
        if qualified:
            start, end = row['activation_start_sequence'], row['activation_terminal_sequence']
            identity = row['module_identity']
            _require(type(start) is type(end) is type(identity) is int and 0 < start < end and identity > 0,
                     'qualified initializer is missing its process-local execution window or module identity')
            if enclosure is not None:
                entry = observation['entries'][row['enclosing_import']]
                _require(entry['import_start_sequence'] == row['enclosing_import_start_sequence']
                         and entry['import_return_sequence'] == row['enclosing_import_return_sequence']
                         and entry['import_start_sequence'] < start < end < entry['import_return_sequence'],
                         'initializer is not enclosed by its same-process watched import')
            preceding[slot, key, identity].append((end, index))
        locator = f'#/activations/{index}'
        for route in routes[key]:
            route['activation_observations'].append(locator)
        results.append(dict(
            observation=locator, process_slot=slot, module_name=row['module_name'], kind=kind, enclosing_import=enclosure,
            original_enclosing_import=original_site, outcome=row['outcome'],
            structural_code_match=row['structural_code_match'], standard_source_loader=row['standard_source_loader'],
            qualified_normal_return=qualified))
    for completions in preceding.values():
        completions.sort()
    return results, preceding


def _cache_hits(observation, routes, processes, activations, preceding):
    hits = []
    for index, entry in enumerate(observation['entries']):
        slot = entry['process_slot']
        _require(slot in processes, 'import request has no observed process slot')
        for module_index, module in enumerate(entry['modules']):
            source = module['source']
            if module['activation'] != 'CACHED' or source is None:
                continue
            key = _module_key(module['name'], source)
            explicit = [route for route in routes.get(key, ()) if route['original_import'] == entry['original_import']]
            if not explicit:
                continue
            start, identity = entry['import_start_sequence'], module['module_identity']
            completions = preceding.get((slot, key, identity), ())
            split = 0 if start is None else bisect_left(completions, (start, -1))
            previous = activations[completions[split - 1][1]] if split else None
            finish = entry['import_return_sequence']
            during = []
            if start is not None and finish is not None:
                first = bisect_left(completions, (start + 1, -1))
                limit = bisect_left(completions, (finish, -1))
                during = [activations[activation]['observation'] for _, activation in completions[first:limit]]
            pointer = f'#/explicit_cache_hits/{len(hits)}'
            for route in explicit:
                route['explicit_cache_hits'].append(pointer)
            hits.append(dict(
                observation=f'#/entries/{index}/modules/{module_index}', process_slot=slot,
                original_import=entry['original_import'], cache_state='SAME_MODULE_BEFORE_AND_AFTER_WATCHED_REQUEST',
                preceding_qualified_activation=None if previous is None else previous['observation'],
                preceding_activation_kind=None if previous is None else previous['kind'],
                same_instance_initializer_returns_in_request_window=during,
                cached_instance_lineage_proven=previous is not None,
                lineage_scope='SAME_OBJECT_BY_PROCESS_LOCAL_OBSERVER_TOKEN' if previous is not None else 'OPEN',
                cached_content_version_proven=False))
    return hits


@reuse_decoded_source_bodies()
def derive_cpu_import_activation_plan(source_effects_ref, initializer_observation_ref):
    """Cold-check diagnostics and classify only process-local observed relationships."""
    from scar.compiler.cpu_import_initializer_observation import verify_cpu_import_initializer_observation

    _require(source_effects_ref['schema'] == SOURCE_EFFECTS and initializer_observation_ref['schema'] == OBSERVATION,
             'static source effects and qualified initializer observation required')
    with replay_transaction():
        static = verify_cpu_import_source_effect_plan(source_effects_ref)
        observation = verify_cpu_import_initializer_observation(initializer_observation_ref)
    keys = ('admission', 'observation', 'program', 'entry_obligations', 'paired', 'target')
    _require(all(static[key] == observation[key] for key in keys),
             'initializer observation belongs to another source admission or original entry')
    modules, routes = _modules(static)
    processes = [dict(process_slot=row['slot'], parent_slot=row['parent_slot'], kind=row['kind'],
                      status=row['status'], observation=f'#/processes/{index}')
                 for index, row in enumerate(observation['processes'])]
    observed = {row['process_slot'] for row in processes if row['status'] == 'OBSERVED'}
    activations, preceding = _activations(observation, routes, observed)
    hits = _cache_hits(observation, routes, observed, activations, preceding)
    qualified = Counter(row['kind'] for row in activations if row['qualified_normal_return'])
    activation_counts = Counter(row['process_slot'] for row in activations if row['qualified_normal_return'])
    hit_counts = Counter(row['process_slot'] for row in hits)
    for row in processes:
        slot = row['process_slot']
        row.update(qualified_initializer_activations=activation_counts[slot], explicit_cache_hits=hit_counts[slot])
    summary = dict(
        observed_processes=len(observed), recorded_processes=len(processes),
        missing_processes=len(processes) - len(observed), admitted_module_routes=len(modules),
        initializer_observations=len(activations), qualified_initializer_activations=sum(qualified.values()),
        unqualified_initializer_observations=len(activations) - sum(qualified.values()),
        direct_watched_activations=qualified[_DIRECT], transitive_watched_activations=qualified[_TRANSITIVE],
        enclosed_activations_with_open_request_origin=qualified[_ENCLOSED],
        outside_watched_activations=qualified[_OUTSIDE],
        activations_without_completed_enclosure=qualified[_UNBOUND],
        explicit_cache_hits=len(hits),
        explicit_cache_hits_after_qualified_activation=sum(row['preceding_qualified_activation'] is not None for row in hits),
        explicit_cache_hits_after_transitive_activation=sum(row['preceding_activation_kind'] == _TRANSITIVE for row in hits),
        explicit_cache_hits_with_proven_instance_lineage=sum(row['cached_instance_lineage_proven'] for row in hits),
        constant_replacements_proven=0, import_removals_proven=0)
    return dict(
        schema=SCHEMA, schema_version=1, source_effects=source_effects_ref,
        initializer_observation=initializer_observation_ref, **{key: static[key] for key in keys},
        status='OBSERVED_SCOPED_INITIALIZER_ACTIVATIONS' if qualified else 'INITIALIZER_ACTIVATION_OPEN',
        evidence_scope='DIAGNOSTIC_PROCESS_LOCAL_NORMAL_RETURN_AND_CODE_MATCH',
        modules=modules, processes=processes, activations=activations, explicit_cache_hits=hits, summary=summary,
        coverage_frontier=[
            dict(role='INITIALIZER_REACH', status='QUALIFIED_OBSERVED_OCCURRENCES_ONLY',
                 evidence='#/activations', blockers=['OTHER_CALL_PATHS_OR_LOADER_SELECTIONS', 'COMPLETE_ACTIVATION_COVERAGE']),
            dict(role='IMPORT_CACHE', status='PROCESS_LOCAL_CACHE_OBSERVATIONS_ONLY',
                 evidence='#/explicit_cache_hits',
                 blockers=['UNMATCHED_INSTANCE_LINEAGE', 'MODULE_STATE_VERSIONS', 'CACHE_MUTATION_RELOAD_AND_FAILURE']),
            dict(role='INITIALIZER_EFFECTS', status='ORIGINAL_EFFECT_CONTRACT_OPEN',
                 blockers=['NESTED_IMPORT_AND_CALL_EFFECTS', 'EXTERNAL_EFFECTS_AND_EXCEPTIONS']),
            dict(role='VALUE_AND_REMOVAL', status='EQUIVALENCE_OPEN',
                 blockers=['REACHING_VERSION_AND_EXACT_CONSUMERS', 'PRESERVE_IMPORT_AND_INITIALIZER_EFFECTS'])],
        process_order_scope='LOCAL_OBSERVER_EVENTS', cross_process_order_proven=False, complete_execution_coverage=False,
        original_effect_contract_unchanged=True, runtime_import_effects_closed=False,
        whole_entry_effect_equivalence_proven=False, transformation_authorized=False, deployment_authorized=False,
        forward_cost_proven=False, execution_default='NO_OP', target_executed=False)


def verify_cpu_import_activation_plan(reference):
    """Replay both parent artifacts and the process-local activation join."""
    saved = read_artifact(reference)
    _require(reference['schema'] == SCHEMA and type(saved['schema_version']) is int and saved['schema_version'] == 1,
             'unsupported initializer activation plan')
    replay = derive_cpu_import_activation_plan(saved['source_effects'], saved['initializer_observation'])
    _require(saved == replay, 'saved initializer activation plan differs from replay')
    return replay
