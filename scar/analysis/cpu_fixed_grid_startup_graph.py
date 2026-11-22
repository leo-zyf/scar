"""Relate fixed-grid source startup requests to verified import-cache observations."""

import ast
from collections import Counter

from scar.analysis.cpu_entry_import_bindings import _key, _module, _static
from scar.analysis.cpu_fixed_grid_lifecycle import verify_cpu_fixed_grid_lifecycle
from scar.analysis.replay_transaction import replay_transaction
from scar.compiler.cpu_entry_replay import _OUTPUT
from scar.compiler.cpu_import_binding_observation import verify_cpu_import_binding_observation
from scar.ir.execution_recipe import _canonical, read_artifact


SCHEMA = 'scar.cpu-fixed-grid-startup-graph'
_ROOT = '#/routes/original_python'
_STARTUP = 'lifecycle:initialization'


def _need(ok, detail):
    if not ok:
        raise ValueError('CPU fixed-grid startup graph: ' + detail)


def _position(statement):
    return [statement.lineno, statement.end_lineno, statement.col_offset, statement.end_col_offset]


def _requests(statement):
    if type(statement) is ast.Import:
        return [(alias.name, 0, ()) for alias in statement.names]
    return [(statement.module, statement.level, tuple(alias.name for alias in statement.names))]


def _declared_sites(lifecycle, initializers, sites, source):
    effects = read_artifact(lifecycle['effects'])
    obligations = read_artifact(effects['entry_obligations'])
    _need(initializers['program'] == effects['program'] == obligations['program']
          and initializers['entry_obligations'] == effects['entry_obligations']
          and lifecycle['source'] == source == effects['source'],
          'startup imports must come from the same original F4 entry and source bytes')
    module = ast.parse(read_artifact(source, json_document=False), filename=source['uri'])
    statements = {tuple(_position(node)): node for node in module.body if isinstance(node, (ast.Import, ast.ImportFrom))}
    body_index = obligations['entry_body']
    body = obligations['bodies'][body_index]
    _need(body['status'] == 'PROJECTED', 'original module root body is not projected')
    allowed = set(body['operations'])
    original = lifecycle['routes']['original_python']
    initialization = original['operations']['initialization']
    order = [(i, edge) for i, edge in enumerate(original['edges'])
             if edge['source'] == 'launch' and edge['target'] == 'initialization' and edge['kind'] == 'NORMAL_ORDER']
    _need(initialization['kind'] == 'PYTHON_IMPORTS_AND_RUNTIME' and len(order) == 1,
          'source lifecycle does not expose its original startup')

    selected = []
    allowed_imports = set(initialization['original_import_requests'])
    for site in sites:
        ordinal = site['ordinal']
        pointer = f'#/imports/{ordinal}'
        if site['file'] != source or site['operation'] not in allowed or pointer not in allowed_imports:
            continue
        position = tuple(site['span'])
        if position not in statements:
            continue
        operation = obligations['operations'][site['operation']]
        role = statements[position]
        _need((site['requested_name'], site['level'], tuple(site['fromlist'])) in _requests(role)
              and operation['effect_rule']['instruction'] == 'ImportRequest',
              'original typed request disagrees with its direct module-scope statement')
        selected.append(site)
    expected = Counter((position, *request) for position, statement in statements.items()
                       for request in _requests(statement))
    actual = Counter((tuple(site['span']), site['requested_name'], site['level'], tuple(site['fromlist']))
                     for site in selected)
    _need(bool(selected) and actual == expected,
          'not every direct startup import statement binds to an original F4 request')
    selected.sort(key=lambda site: (site['span'][0], site['span'][2], site['ordinal']))
    return dict(
        original_entry_obligations=effects['entry_obligations'], original_program=effects['program'],
        original_module_body=f'#/bodies/{body_index}', lifecycle_operation=f'{_ROOT}/operations/initialization',
        launch_edge=f'{_ROOT}/edges/{order[0][0]}',
        import_begin=f'{_ROOT}/sites/IMPORTS_BEGIN', import_return=f'{_ROOT}/sites/IMPORTS_RETURN',
    ), selected


def _verify_scope(lifecycle, initializer_ref, observation_ref):
    observation = verify_cpu_import_binding_observation(observation_ref)
    delivery = read_artifact(lifecycle['delivery'])
    pair = read_artifact(delivery['retained_catalog_pair'])
    scope = lifecycle['scope']
    process = pair['invocation']
    original = process['arms']['original']
    lifecycle_command = lifecycle['routes']['original_python']['command']
    marker = _OUTPUT + '/final.f32'
    _need(observation['initializer_graph'] == initializer_ref
          and observation['paired'] == delivery['retained_catalog_pair']
          and observation['entry'] == lifecycle['source'] == process['entry']
          and observation['interpreter'] == original['executable']
          and observation['original_argv'] == original['argv']
          and observation['original_argv'][-1] == marker and set(process['files']) == {'final.f32'}
          and lifecycle_command[3:] == [*observation['original_argv'][:-1], scope['argv'][-1]]
          and observation['cwd'] == process['cwd'] == scope['working_directory']
          and observation['cpu_ids'] == process['cpu_ids'] == scope['cpu_ids']
          and observation['environment_sha256'] == scope['environment_sha256']
          and observation['process_launches'] == 1 and observation['instrumented'] is True
          and observation['normalized_output_verified'] is True and observation['clean_benchmark'] is False,
          'startup observation does not share the exact source CLI launch and output role')
    return observation


def _nodes(sites, observation):
    requests, modules, connections, unmatched = {}, {}, [], []
    by_key, module_ids = {}, {}
    for site in sites:
        ordinal = site['ordinal']
        identity = f'import:{ordinal}'
        pointer = f'#/imports/{ordinal}'
        requests[identity] = dict(original_import=pointer, original_operation=f"#/operations/{site['operation']}",
            position=site['span'], requested_name=site['requested_name'], level=site['level'], fromlist=site['fromlist'],
            original_candidate_bodies=[row['body'] for row in site['alternatives']], observations=[],
            registry_missing_names=[], effects_closed=False)
        location = _key(site['file']['uri'], site['span'], site['requested_name'], site['level'], site['fromlist'])
        by_key.setdefault(location, []).append(site)
        connections.append(dict(source=_STARTUP, target=identity, kind='CONTAINS_ORIGINAL_MODULE_SCOPE_IMPORT'))

    for index, row in enumerate(() if observation is None else observation['entries']):
        pointer = f'#/entries/{index}'
        location = _key(row['caller'], row['position'], row['requested_name'], row['level'], row['fromlist'])
        candidates = by_key.get(location, ())
        if row.get('process_slot', 0) != 0 or len(candidates) != 1:
            unmatched.append(pointer)
            continue
        site, = candidates
        request_identity = f"import:{site['ordinal']}"
        request = requests[request_identity]
        request['observations'].append(pointer)
        for slot, module in enumerate(row['modules']):
            material = f'{pointer}/modules/{slot}'
            if module['activation'] == 'ABSENT':
                request['registry_missing_names'].append(dict(name=module['name'], observation=material))
                continue
            bound = _module(material, module, site['alternatives'])
            source = bound['source']
            key = bound['name'], bound['origin_kind'], bound['origin'], None if source is None else source['sha256']
            if key not in module_ids:
                identity = module_ids[key] = f'module:{len(modules)}'
                modules[identity] = dict(name=bound['name'], origin_kind=bound['origin_kind'], origin=bound['origin'],
                    source=source, source_bytes_basis=None if source is None else 'POST_RUN_SOURCE_PATH_READBACK',
                    loader_implementation_proven=False, initializer_execution_proven=False)
            connections.append(dict(source=request_identity, target=module_ids[key],
                kind='OBSERVED_POST_IMPORT_MODULE_CACHE_ORIGIN', observation=material,
                activation=bound['activation'], original_body=bound['original_body'],
                original_alternative=bound['original_alternative']))
    return requests, modules, connections, unmatched


def derive_cpu_fixed_grid_startup_graph(lifecycle_ref, initializer_ref, *, observation_ref=None):
    """Project exact lexical startup imports and optional observed module-cache origins."""
    with replay_transaction():
        lifecycle = verify_cpu_fixed_grid_lifecycle(lifecycle_ref)
        initializers, sites, source = _static(initializer_ref)
        startup, scope_sites = _declared_sites(lifecycle, initializers, sites, source)
        observation = None if observation_ref is None else _verify_scope(lifecycle, initializer_ref, observation_ref)
        requests, modules, edges, unmatched = _nodes(scope_sites, observation)
    complete = all(row['observations'] for row in requests.values())
    return dict(schema=SCHEMA, schema_version=1, lifecycle=lifecycle_ref, initializer_graph=initializer_ref,
        observation=observation_ref, source=source, target=lifecycle['target'], scope=lifecycle['scope'],
        initialization=startup, requests=requests, modules=modules, edges=edges, unbound_observations=unmatched,
        status='ALL_DECLARED_STARTUP_REQUESTS_OBSERVED' if complete else 'STARTUP_IMPORT_BINDINGS_OPEN',
        summary=dict(module_scope_requests=len(requests), requests_with_observations=sum(bool(row['observations'])
            for row in requests.values()), distinct_observed_origins=len(modules),
            origin_observations=sum(row['kind'] == 'OBSERVED_POST_IMPORT_MODULE_CACHE_ORIGIN' for row in edges)),
        execution_evidence_scope=None if observation is None else 'ONE_INSTRUMENTED_ORIGINAL_PROCESS_OUTPUT_RELOCATED',
        declared_request_observation_coverage_complete=complete,
        runtime_import_effects_closed=False, module_initializer_execution_proven=False,
        whole_entry_effect_equivalence_proven=False, transformation_authorized=False, target_executed=False,
        coverage_frontier=['IMPLICIT_INTERPRETER_STARTUP', 'UNWATCHED_NESTED_AND_TRANSITIVE_IMPORTS',
            'LOADER_IMPLEMENTATION_AND_LOADED_BYTECODE', 'FROM_LIST_ATTRIBUTE_BINDING_AND_REGISTRY_MUTATION',
            'PACKAGE_INITIALIZER_AND_EXTERNAL_EFFECTS', 'FAILURES_AND_ALTERNATE_IMPORT_ENVIRONMENTS'])


def verify_cpu_fixed_grid_startup_graph(reference):
    """Rebuild original source/observation identities without launching the entry."""
    _need(reference['schema'] == SCHEMA, 'unexpected fixed-grid startup graph schema')
    saved = read_artifact(reference)
    expected = derive_cpu_fixed_grid_startup_graph(saved['lifecycle'], saved['initializer_graph'],
                                                   observation_ref=saved['observation'])
    _need(_canonical(saved) == _canonical(expected), 'saved fixed-grid startup graph differs on cold replay')
    return saved
