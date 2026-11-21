"""Replay an explicit entry optimization boundary on the build main chain."""
from hashlib import sha256
import json
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference, read_artifact


REQUEST = 'scar.whole-entry-request'
DOMAIN = 'scar.whole-entry-option-domain'
SELECTION = 'scar.whole-entry-main-chain'
LEGALITIES = 'scar.whole-entry-source-legalities'
OPTION_SPACES = ('DISCOVER_ALL', 'EXACT_REGISTERED_SET')


def _write(path, value):
    with path.open('x') as output:
        json.dump(value, output, sort_keys=True, indent=2, allow_nan=False)
        output.write('\n')
    return artifact_reference(path, value['schema'])


def read_request(value):
    ref = value if isinstance(value, dict) else artifact_reference(Path(value).resolve(), REQUEST)
    document = read_artifact(ref)
    required = {'schema', 'schema_version', 'argv', 'boundary', 'registered_implementations', 'original_cost'}
    optional = {'option_space', 'audit_retained_validation'}
    if (set(document) - optional != required or document['schema'] != REQUEST or document['schema_version'] != 1
            or document['boundary'] != 'PROCESS_LAUNCH_THROUGH_EXIT' or type(document['argv']) is not list
            or any(type(arg) is not str or '\x00' in arg for arg in document['argv'])
            or type(document['registered_implementations']) is not list):
        raise ValueError('exact original entry argv and whole-process evidence request required')
    if (document.get('option_space', 'DISCOVER_ALL') not in OPTION_SPACES
            or type(document.get('audit_retained_validation', False)) is not bool):
        raise ValueError('unrecognized whole-entry option space or validation audit mode')
    candidates = {}
    for item in document['registered_implementations']:
        if type(item) is not dict or set(item) != {'route', 'artifact', 'validation', 'cost'}:
            raise ValueError('each registered implementation must cite its route, artifact and optional evidence')
        if type(item['route']) is not str or not item['route']:
            raise ValueError('registered delivery route must be named')
        artifact = item['artifact']
        read_artifact(artifact)
        key = json.dumps((item['route'], artifact['sha256']), separators=(',', ':'), allow_nan=False)
        candidate_id = 'native:'+sha256(key.encode()).hexdigest()
        if candidate_id in candidates:
            raise ValueError('duplicate registered native implementation')
        candidates[candidate_id] = item
    return ref, document, candidates


def _scope(plan, request):
    from scar.runtime.caller_pipeline_deployment import entry_context, environment_sha256

    manifest = read_artifact(plan['program'])
    context = entry_context(manifest['entry'], request['argv'])
    target = plan['hardware']
    return dict(context, boundary=request['boundary'], entry=context['entry_path'],
        target_sha256=target['sha256'],
        target=dict(uri=target['path'], sha256=target['sha256'], size_bytes=target['bytes'],
                    schema='scar.target-hardware-graph'),
        option_space=request.get('option_space', 'DISCOVER_ALL'),
        native_output_contract='STANDALONE_CPP',
        program_sha256=plan['program']['sha256'], source_admission_sha256=plan['source_admission']['sha256'],
        memory_budget_bytes=plan['memory_budget_bytes'], input_shapes=plan['input_shape_constraints'],
        environment_sha256=environment_sha256())


def _candidate(id_, row):
    return dict(id=id_, route=row['route'], artifact=row['artifact'], validation=row['validation'])


def _legalities(candidates, scope):
    from scar.optimization.resource_solver import semantic_sha256

    scope_sha = semantic_sha256(scope)
    facts = []
    if candidates:
        from scar.compiler.entry_delivery import qualify_entry_delivery

        for id_, item in sorted(candidates.items()):
            qualification = qualify_entry_delivery(scope, _candidate(id_, item))
            status, reason = 'UNRESOLVED', qualification['reason']
            if qualification['status'] == 'QUALIFIED':
                status, reason = 'LEGAL', 'STANDALONE_CPP_ARTIFACT_REPLAYED'
                if qualification['evidence']['python_runtime_required']:
                    status, reason = 'INELIGIBLE', 'NATIVE_ROUTE_REQUIRES_PYTHON_AT_RUNTIME'
            facts.append(dict(option_id=id_, scope_sha256=scope_sha,
                legal_status=status, reason=reason, qualification=qualification))
    return dict(schema=LEGALITIES, schema_version=1, scope_sha256=scope_sha, options=facts)


def _derive(plan, request, candidates, scope, legalities, legality_reference):
    options = [dict(id='original', legal_status='LEGAL', legality_evidence=[], cost=request['original_cost'])]
    for index, fact in enumerate(legalities['options']):
        id_, legal = fact['option_id'], fact['legal_status']
        item = candidates[id_]
        options.append(dict(id=id_, legal_status=legal, cost=item['cost'] if legal == 'LEGAL' else None,
            legality_evidence=[dict(artifact=legality_reference, locator=f'#/options/{index}')]))
    # A regional candidate, or even a complete native executable, cannot by
    # itself attest that all legal whole-entry compositions were generated.
    frontier = [] if scope['option_space'] == 'EXACT_REGISTERED_SET' else [dict(
        layer='ENTRY_IMPLEMENTATION_DOMAIN', reason='NO_COMPLETE_WHOLE_ENTRY_OPTION_GENERATOR',
        regional_domains=len(plan['regions']), invariant_loops=len(plan['invariant_loops']),
        composite_invariant_loops=len(plan['composite_invariant_loops']))]
    return dict(schema=DOMAIN, schema_version=1, scope=scope, baseline_id='original', options=options,
        coverage_frontier=frontier)


def prepare_whole_entry(plan, request, out):
    reference, document, candidates = read_request(request)
    scope = _scope(plan, document)
    legalities = _legalities(candidates, scope)
    legal_ref = _write(out/'whole-entry-legalities.json', legalities)
    domain = _derive(plan, document, candidates, scope, legalities, legal_ref)
    return dict(request=reference, domain=_write(out/'whole-entry-domain.json', domain),
        legalities=legal_ref, scope=scope, complete_declared_domain=not domain['coverage_frontier'])


def replay_whole_entry(plan, out, *, plan_only):
    from scar.compiler.entry_delivery import replay_entry_delivery
    from scar.optimization.whole_entry_choice import solve_whole_entry_choice

    entry = plan['whole_entry']
    _, request, candidates = read_request(entry['request'])
    live_scope = _scope(plan, request)
    if live_scope != entry['scope']:
        retained = entry['scope']
        changed = sorted(key for key in live_scope.keys() | retained.keys() if live_scope.get(key) != retained.get(key))
        raise ValueError('whole-entry scope changed in ' + ', '.join(changed))
    legalities = _legalities(candidates, live_scope)
    domain = _derive(plan, request, candidates, live_scope, legalities, entry['legalities'])
    if read_artifact(entry['domain']) != domain or read_artifact(entry['legalities']) != legalities:
        raise ValueError('retained whole-entry domain differs from original program and evidence replay')
    choice = solve_whole_entry_choice(domain, expected_option_ids=['original', *sorted(candidates)])
    selected = choice['selected_option_id']
    candidate = None if selected in (None, 'original') else _candidate(selected, candidates[selected])
    delivery = replay_entry_delivery(live_scope, selected_id=selected, candidate=candidate)
    audit = []
    if request.get('audit_retained_validation', False):
        for fact in legalities['options']:
            id_ = fact['option_id']
            if id_ != selected and fact['qualification']['status'] == 'QUALIFIED':
                observation = replay_entry_delivery(live_scope, selected_id=id_,
                    candidate=_candidate(id_, candidates[id_]))
                audit.append(dict(option_id=id_, legal_status=fact['legal_status'],
                    validation=observation['validation'], reason=observation['reason']))
    report = dict(schema=SELECTION, schema_version=1, domain=entry['domain'],
        source_admission=plan['source_admission'], selection=choice, delivery=delivery,
        retained_validation_audit=audit,
        plan_only=plan_only, native_compiled=False, target_executed=False)
    return dict(reference=_write(out/'whole-entry-selection.json', report), **report)
