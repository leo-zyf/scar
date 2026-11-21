"""Join observed import dispatch with source initializer effects for planning."""

from collections import Counter, defaultdict

from scar.analysis.cpu_entry_import_effects import SCHEMA as EFFECTS, verify_cpu_entry_import_effects
from scar.analysis.source_decoded_bodies import reuse_decoded_source_bodies
from scar.ir.execution_recipe import read_artifact

SCHEMA = 'scar.cpu-entry-import-frontier'
BINDINGS = 'scar.f4-cpu-entry-import-bindings'
_UNIQUE = 'UNIQUE_F4_IMPORT_REQUEST'
_BODY_MATCHED = 'SOURCE_BODY_MATCHED'


def _require(condition, detail):
    if not condition:
        raise ValueError('CPU entry import frontier: ' + detail)


def _site(index, request, occurrences, observation_coverage):
    observed = defaultdict(list)
    alternatives = {row['original']: row for row in request['alternatives']}
    rows = []
    for number, row in occurrences:
        pointer = f'#/bindings/{number}'
        _require(row['operation'] == request['operation'], 'runtime request changed original import operation')
        modules = []
        for module_index, module in enumerate(row['modules']):
            module_pointer = f'{pointer}/modules/{module_index}'
            body = module['original_body']
            alternative = module['original_alternative']
            if module['status'] == _BODY_MATCHED:
                source = alternatives.get(alternative)
                _require(source is not None and source['body'] is not None
                         and body == f"#/bodies/{source['body']}", 'runtime module changed original body alternative')
                observed[alternative].append(module_pointer)
            modules.append(dict(binding=module_pointer, status=module['status'],
                                observed_cache_transition=module['activation'], original_body=body,
                                original_alternative=alternative))
        rows.append(dict(binding=pointer, observation=row['observation'], modules=modules))
    return dict(
        original_import=request['original'], operation=request['operation'], initializer_request=f'#/requests/{index}',
        observation_coverage=observation_coverage,
        status=('OBSERVED_SOURCE_MODULE_ORIGIN' if observed else
                'OBSERVED_SITE_WITH_SOURCE_ORIGIN_OPEN' if rows else 'RUNTIME_SITE_UNOBSERVED'),
        occurrences=rows,
        alternatives=[
            dict(original=row['original'], status=row['status'],
                 original_body=None if row['body'] is None else f"#/bodies/{row['body']}",
                 origin_matches=observed[row['original']])
            for row in request['alternatives']],
        importer_effects_closed=False, initializer_execution_proven=False)


def _initializer(index, body, effect_row, observed, lookup_conditions):
    graph_pointer = f'#/initializers/{index}'
    row = dict(original_body=body['original'], source_body_status=body['status'],
               inventory=graph_pointer + '/operation_inventory',
               static_effects=graph_pointer + '/effects',
               call_sites=body['call_sites'], nested_import_sites=body['nested_import_sites'],
               observed_module_origins=observed, initializer_execution_proven=False,
               source_namespace_effects=None, literal_stores=[])
    if effect_row is None:
        return row
    effect_index, effects = effect_row
    pointer = f'#/initializers/{effect_index}'
    possible_readers = defaultdict(list)
    for read_index, read in enumerate(effects['potential_reads']):
        possible_readers[read['write']].append(dict(
            operation=read['read'], effect=f'{pointer}/potential_reads/{read_index}'))
    for access_index, access in enumerate(effects['accesses']):
        payload = access['payload_route']
        if access['kind'] != 'WRITE' or payload is None or payload['literal'] is None:
            continue
        store = f'{pointer}/accesses/{access_index}'
        readers = possible_readers[access['operation']]
        row['literal_stores'].append(dict(
            operation=access['operation'], effect=store,
            literal_origin=dict(port=payload['origin_port'], operation=payload['origin_operation'],
                                value=store + '/payload_route/literal/value'),
            symbolic_locations=store + '/alternatives',
            status='LITERAL_OPERAND_PROVEN_IN_SOURCE_GRAPH',
            possible_local_readers=readers,
            reader_status='MAY_REACH_SYMBOLIC_LOOKUP' if readers else 'LOCAL_LOOKUP_NOT_LINKED',
            lookup_conditions=lookup_conditions,
            obligations=['RUNTIME_NAMESPACE_INSTANCE_OPEN',
                         'UNIQUE_REACHING_VERSION_WITHOUT_INTERVENING_WRITE_OPEN',
                         'OUTSIDE_INITIALIZER_CONSUMERS_OPEN'],
            constant_read_replacement_proven=False))
    row['source_namespace_effects'] = dict(
        effect_body=pointer, status=effects['status'], scope=effects['scope'],
        access_counts=dict(Counter(access['kind'] for access in effects['accesses'])),
        open_address_operations=effects['open_address_operations'],
        local_reader_control_search_complete=effects['local_reader_control_search_complete'])
    return row


@reuse_decoded_source_bodies()
def derive_cpu_entry_import_frontier(effects_ref, bindings_ref):
    """Use the original F4 locators and one bounded diagnostic cohort as evidence."""
    from scar.analysis.cpu_entry_import_bindings import verify_cpu_entry_import_bindings

    _require(effects_ref['schema'] == EFFECTS and bindings_ref['schema'] == BINDINGS,
             'initializer effects and original-site import bindings required')
    effects = verify_cpu_entry_import_effects(effects_ref)
    bindings = verify_cpu_entry_import_bindings(bindings_ref)
    keys = ('initializer_graph', 'entry_obligations', 'program')
    _require(all(effects[key] == bindings[key] for key in keys),
             'the effects and runtime bindings must share the exact original F4 graph')
    initializer_graph = read_artifact(effects['initializer_graph'])
    occurrences = defaultdict(list)
    unresolved = []
    for number, row in enumerate(bindings['bindings']):
        if row['status'] == _UNIQUE:
            occurrences[row['original_import']].append((number, row))
        else:
            unresolved.append(dict(binding=f'#/bindings/{number}', status=row['status'],
                                   candidate_imports=row['candidate_imports']))
    coverage = dict.fromkeys(bindings['unobserved_imports'], 'WATCHED_WITHOUT_UNIQUE_OBSERVATION')
    coverage.update(dict.fromkeys(bindings['unwatchable_imports'], 'NO_QUALIFIED_SOURCE_WATCH_SITE'))
    sites = [_site(index, request, occurrences.pop(request['original'], ()),
                   coverage.get(request['original'], 'UNIQUE_SITE_OBSERVED'))
             for index, request in enumerate(initializer_graph['requests'])]
    _require(not occurrences, 'matched runtime request has no original F4 import site')
    observed_bodies = defaultdict(list)
    for site in sites:
        for alternative in site['alternatives']:
            if alternative['origin_matches']:
                observed_bodies[alternative['original_body']].extend(alternative['origin_matches'])
    effect_bodies = {body['original']: (index, body) for index, body in enumerate(effects['initializers'])}
    initializers = [_initializer(index, body, effect_bodies.pop(body['original'], None),
                                 observed_bodies.pop(body['original'], []),
                                 effects['access_contract']['later_lookup_conditions'])
                    for index, body in enumerate(initializer_graph['initializers'])]
    _require(not effect_bodies and not observed_bodies, 'initializer effects or observations have no original body')
    literal_stores = [store for body in initializers for store in body['literal_stores']]
    summary = dict(
        unique_observed_sites=sum(bool(site['occurrences']) for site in sites),
        source_body_matched_sites=sum(site['status'] == 'OBSERVED_SOURCE_MODULE_ORIGIN' for site in sites),
        source_body_matches=sum(len(body['observed_module_origins']) for body in initializers),
        unresolved_observations=len(unresolved), literal_store_origins=len(literal_stores),
        possible_local_literal_reads=sum(len(store['possible_local_readers']) for store in literal_stores),
        constant_replacements_proven=0, import_removals_proven=0)
    status = ('OBSERVED_SOURCE_IMPORT_ORIGINS_WITH_OPEN_REWRITE_OBLIGATIONS' if summary['source_body_matches']
              else 'OBSERVED_IMPORT_SITES_WITH_OPEN_SOURCE_ORIGINS' if summary['unique_observed_sites']
              else 'ORIGINAL_IMPORT_BINDINGS_UNRESOLVED')
    return dict(
        schema=SCHEMA, schema_version=1, status=status,
        effects=effects_ref, bindings=bindings_ref, initializer_graph=effects['initializer_graph'],
        entry_obligations=effects['entry_obligations'], program=effects['program'],
        observation=bindings['observation'], paired=bindings['paired'],
        evidence_scope='PAIRED_ORIGINAL_LAUNCH_DIAGNOSTIC',
        sites=sites, initializers=initializers, unresolved_observations=unresolved, summary=summary,
        coverage_frontier=[
            dict(role='IMPORT_DISPATCH', status='OTHER_ACTIVATIONS_AND_INPUT_PATHS_OPEN',
                 blockers=['COMPLETE_REACHABLE_SITE_COVERAGE', 'FUTURE_LOADER_AND_CACHE_SELECTION']),
            dict(role='IMPORT_PROTOCOL', status='EFFECTS_OPEN',
                 blockers=['FINDER_AND_LOADER_EFFECTS', 'CACHE_INSTALL_REUSE_RELOAD_AND_FAILURE_EFFECTS']),
            dict(role='INITIALIZER_EXECUTION', status='EFFECTS_OPEN',
                 blockers=['BODY_ACTIVATION_AND_COMPLETION', 'NESTED_CALL_IMPORT_AND_EXTERNAL_EFFECTS',
                           'EXCEPTION_PATHS']),
            dict(role='VALUE_REPLACEMENT', status='REACHING_VALUE_OPEN',
                 blockers=['EXACT_ATTRIBUTE_OR_LOOKUP_CONSUMER', 'RUNTIME_NAMESPACE_INSTANCE',
                           'UNIQUE_REACHING_BINDING_VERSION', 'LOOKUP_EFFECT_AND_VALUE_IDENTITY_EQUIVALENCE']),
            dict(role='IMPORT_REMOVAL', status='EQUIVALENCE_OPEN',
                 blockers=['RETAIN_OR_DISCHARGE_IMPORT_AND_INITIALIZER_EFFECTS',
                           'PRESERVE_ALL_LIVE_BINDINGS_AND_MODULE_IDENTITY'])],
        whole_entry_effect_equivalence_proven=False, transformation_authorized=False, deployment_authorized=False,
        runtime_import_effects_closed=False, forward_cost_proven=False, execution_default='NO_OP',
        target_executed=False)


def verify_cpu_entry_import_frontier(reference):
    """Recompute both parents and the exact locator joins without running the entry."""
    saved = read_artifact(reference)
    _require(reference['schema'] == SCHEMA and saved['schema_version'] == 1,
             'unsupported import frontier report')
    replay = derive_cpu_entry_import_frontier(saved['effects'], saved['bindings'])
    _require(replay == saved, 'saved import frontier differs at cold replay')
    return replay
