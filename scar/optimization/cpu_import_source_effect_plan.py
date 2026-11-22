"""Project admitted initializer bodies into the original CPU import effect frontier."""

from collections import Counter

from scar.analysis.source_decoded_bodies import reuse_decoded_source_bodies
from scar.ir.execution_recipe import read_artifact

SCHEMA = 'scar.cpu-import-source-effect-plan'
ADMISSION = 'scar.cpu-import-source-admission'
SOURCE_BODY = 'scar.f4-cpu-import-source-body'
_PROJECTED = 'SOURCE_INITIALIZER_PROJECTED'
_ADMISSION_OBLIGATION = 'ADMIT_SOURCE_AND_PROJECT_F4_BODY'
_PROJECTION = '#/projection'


def _require(condition, detail):
    if not condition:
        raise ValueError('CPU import source effect plan: ' + detail)


def _artifact_key(reference):
    return tuple(reference[field] for field in ('uri', 'sha256', 'size_bytes', 'schema'))


def _candidates(frontier):
    bindings = read_artifact(frontier['bindings'])
    observed = {
        f'#/bindings/{index}/modules/{module_index}': (binding, module)
        for index, binding in enumerate(bindings['bindings'])
        for module_index, module in enumerate(binding['modules'])
    }
    sites = {site['original_import']: site['operation'] for site in frontier['sites']}
    workers = {row['slot'] for row in frontier['processes']
               if row['kind'] == 'CPYTHON_MULTIPROCESSING_SPAWN' and row['status'] == 'OBSERVED'}
    result = {}
    for index, candidate in enumerate(frontier['source_admission_candidates']):
        site, name = candidate['original_import'], candidate['module_name']
        origins = candidate['observations']
        first = origins[0]['binding']
        source = observed[first][1]['source']
        _require(candidate['source'] == first + '/source' and site in sites,
                 'source admission lost its original location or evidence')
        for origin in origins:
            binding, module = observed[origin['binding']]
            _require(binding['process_slot'] == origin['process_slot'] in workers
                     and binding['original_import'] == site and binding['operation'] == sites[site]
                     and module['name'] == name and module['source'] == source,
                     'worker origin belongs to another import, image or process')
        key = site, name, _artifact_key(source)
        _require(key not in result, 'one source admission has conflicting original candidates')
        result[key] = dict(locator=f'#/source_admission_candidates/{index}', operation=sites[site],
                           worker_origins=origins)
    return result


def _initializer(reference, source):
    body = read_artifact(reference)
    _require(reference['schema'] == SOURCE_BODY and body['schema_version'] == 1 and body['source'] == source,
             'admitted body differs from the exact observed source snapshot')
    result = dict(body=reference, program=body['program'], status=body['status'], inventory=None)
    if body['status'] != _PROJECTED:
        return result
    projection = body['projection']
    root = projection['entry_body']
    _require(projection['bodies'][root]['status'] == 'PROJECTED',
             'admitted initializer is missing its projected F4 root')
    operations = projection['operations']
    result['inventory'] = dict(
        scope=body['scope'],
        root=f'{_PROJECTION}/bodies/{root}', operations=_PROJECTION + '/operations',
        imports=_PROJECTION + '/imports', calls=_PROJECTION + '/calls',
        operation_count=len(operations), edge_count=len(projection['edges']),
        effect_families=dict(Counter(row['effect_rule']['family'] for row in operations)),
        effect_resolution_kinds=dict(Counter(row['effect_rule']['resolution'] for row in operations)),
        effect_obligations=dict(Counter(name for row in operations for name in row['obligations'])),
        nested_import_obligations=dict(Counter(row['obligation'] for row in projection['imports'])),
        nested_call_obligations=dict(Counter(row['obligation'] for row in projection['calls'])),
        nested_import_count=len(projection['imports']), nested_call_count=len(projection['calls']),
        nested_imports_without_source_body=sum(
            not any(alt['body'] is not None for alt in row['alternatives']) for row in projection['imports']),
        nested_calls_with_provider_frontier=sum(bool(row['provider_frontiers']) for row in projection['calls']))
    return result


@reuse_decoded_source_bodies()
def derive_cpu_import_source_effect_plan(admission_ref):
    """Retain source projection as local evidence and preserve original execution."""
    from scar.compiler.cpu_entry_launch_scope import verify_cpu_import_source_admission

    _require(admission_ref['schema'] == ADMISSION, 'admitted source composite required')
    admission = verify_cpu_import_source_admission(admission_ref)
    frontier = read_artifact(admission['frontier'])
    _require(frontier['schema_version'] == 2 and all(admission[key] == frontier[key]
                for key in ('observation', 'entry_obligations', 'program')),
             'source composite must belong to the same original process-scoped frontier')
    candidates = _candidates(frontier)
    modules, initializers, body_indices = [], [], {}
    for index, module in enumerate(admission['modules']):
        site, name, source, body = (module[key] for key in ('original_import', 'module_name', 'source', 'body'))
        candidate_key = site, name, _artifact_key(source)
        _require(candidate_key in candidates, 'admitted source has no matching original worker import')
        candidate = candidates.pop(candidate_key)
        _require(module.get('candidate', candidate['locator']) == candidate['locator'],
                 'admitted source points at a different original candidate')
        body_key = _artifact_key(body)
        if body_key not in body_indices:
            body_indices[body_key] = len(initializers)
            initializers.append(_initializer(body, source))
        body_index = body_indices[body_key]
        projected = initializers[body_index]['inventory'] is not None
        modules.append(dict(
            admission_module=f'#/modules/{index}', frontier_candidate=candidate['locator'],
            original_import=site, original_operation=candidate['operation'], module_name=name,
            source=f'#/modules/{index}/source', worker_origins=candidate['worker_origins'],
            initializer=f'#/initializers/{body_index}',
            source_representation_status='PROVEN_IN_STATIC_PROJECTION' if projected else 'SOURCE_PROJECTION_OPEN',
            discharged=[_ADMISSION_OBLIGATION] if projected else [],
            pending=['PROVE_REACH_AND_EFFECTS'] if projected else [_ADMISSION_OBLIGATION, 'PROVE_REACH_AND_EFFECTS']))
    inventories = [row['inventory'] for row in initializers if row['inventory'] is not None]
    discharged = sum(bool(row['discharged']) for row in modules)
    summary = dict(
        admitted_modules=len(modules), source_body_projections=len(inventories),
        source_projection_obligations_discharged=discharged,
        source_projection_obligations_open=len(modules) - discharged + len(candidates),
        unadmitted_source_candidates=len(candidates),
        original_import_sites=len({row['original_import'] for row in modules}),
        worker_origin_observations=sum(len(row['worker_origins']) for row in modules),
        initializer_operations=sum(row['operation_count'] for row in inventories),
        nested_import_sites=sum(row['nested_import_count'] for row in inventories),
        nested_call_sites=sum(row['nested_call_count'] for row in inventories),
        nested_imports_without_source_body=sum(row['nested_imports_without_source_body'] for row in inventories),
        nested_calls_with_provider_frontier=sum(row['nested_calls_with_provider_frontier'] for row in inventories),
        initializer_activations_proven=0, constant_replacements_proven=0, import_removals_proven=0)
    return dict(
        schema=SCHEMA, schema_version=1, admission=admission_ref,
        frontier=admission['frontier'], bindings=frontier['bindings'], observation=admission['observation'],
        paired=frontier['paired'], entry_obligations=admission['entry_obligations'],
        program=admission['program'], target=admission['target'],
        evidence_scope='STATIC_DECLARATION_FOR_OBSERVED_CHILD_MODULE_ORIGIN',
        status='SOURCE_PROJECTION_WITH_OPEN_RUNTIME_EFFECTS' if discharged else 'SOURCE_PROJECTION_OPEN',
        modules=modules, initializers=initializers, summary=summary,
        unadmitted_frontier_candidates=[row['locator'] for row in candidates.values()],
        coverage_frontier=[
            dict(role='SOURCE_REPRESENTATION', status='SCOPED_PROJECTION_ACCOUNTED',
                 admitted='#/modules', open_candidates='#/unadmitted_frontier_candidates'),
            dict(role='LOADER_CACHE_AND_ACTIVATION', status='OBSERVED_MODULE_ORIGIN_ONLY',
                 blockers=['EXACT_LOADED_CODE_AND_LOADER', 'PROCESS_LOCAL_CACHE_INSTALL_RELOAD_AND_FAILURE',
                           'INITIALIZER_ENTRY_AND_COMPLETION']),
            dict(role='INITIALIZER_EFFECTS', status='STATIC_OBLIGATIONS_RETAINED',
                 evidence='#/initializers', blockers=['NESTED_IMPORT_SELECTION_AND_EFFECTS',
                           'CALL_TARGET_EFFECT_AND_ACTIVATION_CLOSURE', 'EXTERNAL_EFFECTS_AND_EXCEPTIONS']),
            dict(role='VALUE_AND_REMOVAL', status='ORIGINAL_EFFECT_CONTRACT_OPEN',
                 blockers=['REACHING_BINDING_VERSION_AND_EXACT_CONSUMERS',
                           'PRESERVE_LIVE_BINDINGS_AND_IMPORT_EFFECTS'])],
        original_effect_contract_unchanged=True, initializer_activation_proven=False,
        runtime_import_effects_closed=False,
        whole_entry_effect_equivalence_proven=False, transformation_authorized=False, deployment_authorized=False,
        forward_cost_proven=False, execution_default='NO_OP', target_executed=False)


def verify_cpu_import_source_effect_plan(reference):
    """Cold-check the original admission and deterministically reconstruct this view."""
    saved = read_artifact(reference)
    _require(reference['schema'] == SCHEMA and type(saved['schema_version']) is int and saved['schema_version'] == 1,
             'unsupported import source effect plan')
    replay = derive_cpu_import_source_effect_plan(saved['admission'])
    _require(replay == saved, 'saved import source plan differs from replay')
    return replay
