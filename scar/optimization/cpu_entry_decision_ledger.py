"""Read back the evidence and open costs of a finite CPU entry domain."""

from dataclasses import asdict
from pathlib import Path

from scar.analysis.target_graph import TargetHardwareGraph
from scar.ir.execution_recipe import read_artifact
from scar.ir.program.target_hardware import CpuCore, HardwareThread, MemoryDomain, MemoryKind
from scar.ir.record_codec import encode
from .aten_native_schedule import recorded_cpu_domain
from .deployment_cost import _pointer
from .resource_solver import semantic_sha256
from .whole_entry_choice import DOMAIN, LIFECYCLE_PHASES, solve_whole_entry_choice


SCHEMA = 'scar.cpu-entry-decision-ledger'
_LEGALITIES = 'scar.whole-entry-source-legalities'
_CATALOG = 'scar.cpu-entry-provider-catalog'
_FAMILY = 'CATALOG_CPU_ENTRY'
_STREAM = 'scar.cpu-recursive-stream-source-provider'
_ARRAY = 'scar.cpu-array-task-source-provider'
_GRID = 'scar.cpu-fixed-grid-source-provider'
_SPARSE = 'scar.triangle-profile-family'
_STREAM_GRAPH = 'scar.f4-cpu-stream-capture'
_PAIRED_GRAPH = 'scar.f4-cpu-provider-boundary'
_EFFECT_FRONTIER = 'scar.f4-entry-effect-frontier'


def _require(condition, message):
    if not condition:
        raise ValueError('CPU entry decision ledger: ' + message)


def _fact(artifact, locator):
    return dict(artifact=artifact, locator=locator)


def _quantity(status='UNKNOWN', bytes_=None, source=None):
    _require(bytes_ is None or type(bytes_) is int and bytes_ >= 0, 'invalid memory quantity')
    return dict(status=status, bytes=bytes_, evidence=source)


def _memory(plan, scope, authority):
    budget = scope['memory_budget_bytes']
    _require(type(budget) is int and budget > 0, 'positive declared planning budget required')
    result = dict(request_budget_bytes=budget, owner_peak=_quantity(), graph_payload_peak=_quantity(),
        virtual_address_space_headroom=_quantity(), explicit_payload=_quantity(), whole_process_dram_peak=_quantity(),
        excluded_from_payload=[], physical_dram_reservation_proven=False, historical_memory=[])
    if plan is None:
        return result
    contract = plan['contract']
    family = plan['provider_family']
    if family == _STREAM:
        source = contract['parameterized_plan']
        resource = read_artifact(source)['resources']
        peak = resource['peak_owner_memory_bytes']
        retained = contract['memory']
        _require(authority['parameterized_plan'] == source and resource['cpu_owner_lanes'] == len(plan['context']['cpu_ids'])
                 and retained['scope'] == 'PROVIDER_OWNER_MEMORY' and retained['declared_budget_bytes'] == budget
                 and retained['peak_bytes'] == peak and (peak is None or type(peak) is int and 0 < peak <= budget)
                 and retained['status'] == ('UNKNOWN' if peak is None else 'RETAINED_PROVIDER_BOUND'),
                 'owner memory differs from the retained stream resource contract')
        result['owner_peak'] = _quantity('UNKNOWN' if peak is None else 'PROVIDER_OWNER_PEAK_BOUND', peak,
                                         _fact(source, '#/resources/peak_owner_memory_bytes'))
    elif family == _ARRAY:
        source = contract['profile']['runtime']
        decision = read_artifact(source)['decision']
        retained = contract['native_memory']
        memory = decision['memory']
        scope_kind = memory['bound_scope']
        if scope_kind == 'SINGLE_PROCESS_VAS_CAP_INCLUDING_PRIVATE_ARRAYS_AND_OPENMP_CHILD_STACKS':
            workers = 0
        else:
            _require(scope_kind == 'AGGREGATE_PROCESS_VAS_CAPS_PLUS_EXPLICIT_PAYLOAD_SUBACCOUNT',
                     'unknown array virtual address-space accounting')
            workers = decision['worker_count']
        owner, worker = memory['explicit_owner_payload_bytes'], memory['explicit_worker_payload_bytes']
        owner_cap = memory['coordinator_address_space_limit_bytes']
        worker_cap = memory['worker_address_space_limit_bytes']
        _require(all(type(value) is int and value >= 0 for value in (owner, worker, workers))
                 and all(type(value) is int and value > 0 for value in (owner_cap, worker_cap)),
                 'array memory quantities require exact nonnegative integers and positive limits')
        required, payload = owner_cap + workers * worker_cap, owner + workers * worker
        _require(retained['memory'] == memory and retained['scope'] == scope_kind
                 and retained['capacity_kind'] == 'VIRTUAL_ADDRESS_SPACE_CAPS'
                 and retained['required_address_space_headroom_bytes'] == memory['required_headroom_bytes'] == required
                 and retained['explicit_payload_bytes'] == payload and retained['declared_budget_bytes'] == budget
                 and 0 <= owner < owner_cap and 0 <= worker and (not workers or worker < worker_cap)
                 and payload <= budget and 0 < required <= budget
                 and memory['physical_reservation'] is False and retained['physical_dram_reserved'] is False,
                 'array payload or address-space headroom differs from retained provider evidence')
        result['virtual_address_space_headroom'] = _quantity('VIRTUAL_ADDRESS_SPACE_CAPS', required,
            _fact(source, '#/decision/memory/required_headroom_bytes'))
        result['explicit_payload'] = _quantity('DECLARED_PROCESS_PAYLOAD_SUBACCOUNT', payload,
                                               _fact(source, '#/decision/memory'))
    elif family == _GRID:
        project = contract['project']
        member = 'plan/execution.json'
        receipt = read_artifact(project)['files'][member]
        source = dict(uri=str(Path(project['uri']).parent / member), schema='scar.aten-recurrence-plan', **receipt)
        execution = read_artifact(source)
        storage = execution['storage']
        peak = storage['peak_tensor_storage_bytes']
        _require(execution['target'] == scope['target'] and storage == contract['storage']
                 and execution['peak_tensor_storage_bytes'] == peak and type(peak) is int and 0 < peak <= budget,
                 'fixed-grid payload differs from the original scoped target and graph')
        result['graph_payload_peak'] = _quantity('GRAPH_HELD_PAYLOAD_PEAK', peak,
                                                  _fact(source, '#/storage/peak_tensor_storage_bytes'))
        result['excluded_from_payload'] = storage['excluded_bytes']
    elif family == _SPARSE:
        source = contract['historical_placement']
        peak = read_artifact(source)['memory_floor_bytes']
        _require(peak == contract['historical_memory_floor_bytes'] and contract['historical_costs_transferred'] is False,
                 'sparse historical memory differs from the new source contract')
        result['historical_memory'].append(_quantity('HISTORICAL_NOT_CURRENT_BOUND', peak,
                                                      _fact(source, '#/memory_floor_bytes')))
    return result


def _graph(reference, scope, provider, threads, effects):
    if reference is None:
        return None
    graph = read_artifact(reference)
    _require(reference['schema'] in (_STREAM_GRAPH, _PAIRED_GRAPH)
             and graph['schema_version'] == 1 and graph['target'] == scope['target']
             and graph['program']['sha256'] == scope['program_sha256']
             and graph['source_admission']['sha256'] == scope['source_admission_sha256']
             and (graph.get('entry_obligations') or {}).get('sha256') == scope.get('entry_obligations_sha256')
             and (provider is None or graph['provider'] == provider)
             and (effects is None or (graph['program'], graph['entry_obligations'])
                  == (effects['program'], effects['entry_obligations']))
             and graph['deployment_authorized'] is False and graph['whole_program_effects_closed'] is False
             and type(graph['coverage_frontier']) is list,
             'resource and effect graph differs from the original source, target or provider')
    if reference['schema'] == _STREAM_GRAPH:
        source, cpu = graph['source'], graph['requested_cpu_scope']
        _require(graph['capture'] == scope.get('cpu_stream_capture')
                 and graph['original_observer']['argv'] == scope['argv'], 'capture or original argv differs')
    else:
        source, cpu = graph['entry']['source'], graph['observed_launch_cpu_ids']
        pinned = scope.get('cpu_paired_observation')
        _require(pinned in (None, graph['replay']) and graph['argv'] == scope['argv'],
                 'paired observation or original argv differs')
        if pinned is None:
            from scar.compiler.cpu_entry_launch_scope import verify_paired_original_scope

            paired, composition = verify_paired_original_scope(scope, graph['replay'])
            _require(paired['capture'] == scope.get('cpu_stream_capture')
                     and composition['catalog'] == scope['cpu_provider_catalog']
                     and composition['provider'] == graph['provider']
                     and [row['cpu_id'] for row in cpu] == paired['invocation']['cpu_ids']
                     and graph['compilation'] == paired['compilation']
                     and graph['working_directory'] == scope['working_directory']
                     and graph['original_environment_sha256'] == scope['environment_sha256'],
                     'late paired graph differs from the exact original launch and provider')
    _require((source['uri'], source['sha256']) == (scope['entry'], scope['entry_sha256'])
             and {row['cpu_id'] for row in cpu} <= set(scope['process_affinity_cpu_ids'])
             and all(row['target_thread'] is None or row['target_thread'] == threads.get(row['cpu_id']) for row in cpu),
             'observed launch differs from the original entry and requested CPU scope')
    return dict(source=reference, observed_cpu_endpoints=cpu, effect_frontier=graph['coverage_frontier'],
                whole_program_effects_closed=graph['whole_program_effects_closed'])


def _cost(row):
    source = row['source_cost']
    interval = row['interval_ns']
    frontier = row['cost_basis']
    lifecycle = dict.fromkeys(LIFECYCLE_PHASES, 'UNRESOLVED')
    if row['legal_status'] == 'INELIGIBLE':
        lifecycle = dict.fromkeys(LIFECYCLE_PHASES, 'NOT_RANKED')
    elif interval['upper'] is not None:
        lifecycle = _pointer(read_artifact(source['artifact']), source['locator'])['lifecycle_coverage']
    lower = interval['lower'] if interval['upper'] is not None or (
        frontier in ('RETAINED_MANDATORY_WORK_LOWER_BOUND', 'MANDATORY_SUBGRAPH_MODEL_LOWER_BOUND',
                     'BOUNDED_SCHEDULE_FEASIBLE_UPPER_UNRESOLVED')) else None
    return dict(evidence=source, model_bound_ns=dict(lower=lower, upper=interval['upper']), basis=frontier,
                lifecycle=lifecycle, missing_lifecycle_terms=[name for name in LIFECYCLE_PHASES
                        if lifecycle[name] == 'UNRESOLVED'], schedule_certificate=row['schedule_certificate'])


def _import_bindings(reference, scope, effects):
    from scar.compiler.cpu_entry_launch_scope import verify_paired_original_scope
    from scar.optimization.cpu_entry_import_frontier import verify_cpu_entry_import_frontier

    frontier = verify_cpu_entry_import_frontier(reference)
    paired, composition = verify_paired_original_scope(scope, frontier['paired'])
    _require(effects is not None and frontier['program'] == effects['program']
             and frontier['entry_obligations'] == effects['entry_obligations']
             and scope.get('cpu_paired_observation') in (None, frontier['paired'])
             and paired['capture'] == scope.get('cpu_stream_capture')
             and composition['catalog'] == scope['cpu_provider_catalog']
             and frontier['whole_entry_effect_equivalence_proven'] is False
             and frontier['transformation_authorized'] is False and frontier['deployment_authorized'] is False,
             'import bindings differ from the original effects, CPU catalog or launch')
    return dict(reference=reference, summary=frontier['summary'], coverage_frontier=frontier['coverage_frontier'])


def derive_cpu_entry_decision_ledger(domain_ref, legalities_ref, *, provider_graph=None, effect_frontier=None,
                                     import_frontier=None):
    """Project a read-only evidence ledger from exact original domain receipts."""
    domain, legalities = read_artifact(domain_ref), read_artifact(legalities_ref)
    scope = domain['scope']
    scope_sha = semantic_sha256(scope)
    _require(domain_ref['schema'] == DOMAIN and legalities_ref['schema'] == _LEGALITIES
             and type(legalities['schema_version']) is int and legalities['schema_version'] == 1
             and legalities['scope_sha256'] == scope_sha
             and scope['boundary'] == 'PROCESS_LAUNCH_THROUGH_EXIT'
             and scope.get('cpu_provider_catalog', {}).get('schema') == _CATALOG,
             'exact-scope CPU catalog option and legality documents required')
    effects = read_artifact(effect_frontier) if effect_frontier is not None else None
    if effects is not None:
        _require(effect_frontier['schema'] == _EFFECT_FRONTIER and effects['schema_version'] == 1
                 and effects['cpu_graph'] == provider_graph and effects['program']['sha256'] == scope['program_sha256']
                 and effects['entry_obligations']['sha256'] == scope['entry_obligations_sha256']
                 and effects['whole_entry_equivalence_proven'] is False and effects['transformation_authorized'] is False,
                 'effect frontier differs from the original observations and entry obligations')
    catalog = read_artifact(scope['cpu_provider_catalog'])
    baseline = domain['baseline_id']
    facts = {row['option_id']: (index, row) for index, row in enumerate(legalities['options'])}
    _require(baseline not in facts and len(facts) == len(legalities['options']), 'independent option IDs differ')
    for row in domain['options']:
        if row['id'] != baseline:
            _require(row['id'] in facts, 'option has no independent legality record')
            index, _ = facts[row['id']]
            _require(row['legality_evidence'] == [_fact(legalities_ref, f'#/options/{index}')],
                     'option legality record does not retain the independent authority')
    choice = solve_whole_entry_choice(domain, expected_option_ids=[baseline, *facts])

    target = TargetHardwareGraph(Path(scope['target']['uri']).parent)
    constraint, target_cpus = recorded_cpu_domain(target)
    original_cpus = scope['process_affinity_cpu_ids']
    inventory = target.resources()
    threads = {row.cpu_id: row.id.wire for row in inventory if type(row) is HardwareThread and row.online is True}
    cores = {cpu: row.id.wire for row in inventory if type(row) is CpuCore for cpu in row.cpu_ids}
    _require(asdict(target._snapshot) == scope['target'] and original_cpus and set(original_cpus) <= set(target_cpus),
             'original CPU cohort must belong to the exact recorded target')
    host_memory = [dict(id=row.id.wire, numa_node=row.numa_node, reported_capacity_bytes=row.capacity_bytes,
                        evidence=encode(row.evidence)) for row in inventory
                   if type(row) is MemoryDomain and row.kind is MemoryKind.HOST]
    rows, provider = [], None
    for row in choice['alternatives']:
        id_ = row['option_id']
        _, fact = facts.get(id_, (None, {}))
        generated = fact.get('generated_option', {})
        plan = generated['proof']['emission_plan'] if generated.get('implementation_family') == _FAMILY else None
        lanes = None
        authority_document = None
        timing_context = []
        if plan is not None:
            authority, context = plan['authority'], plan['context']
            cpus = context['cpu_ids']
            _require(generated['id'] == id_ and plan['catalog'] == scope['cpu_provider_catalog']
                     and plan['target'] == scope['target'] and plan['argv'] == scope['argv']
                     and (plan['entry']['uri'], plan['entry']['sha256']) == (scope['entry'], scope['entry_sha256'])
                     and plan['source_f4_program_sha256'] == scope['program_sha256']
                     and plan['effect_obligations_sha256'] == scope['entry_obligations_sha256']
                     and authority['schema'] == plan['provider_family'] and authority in catalog['authorities']
                     and (provider is None or authority == provider), 'generated provider plan differs from original scope')
            authority_document = read_artifact(authority)
            _require(cpus and set(cpus) <= set(original_cpus) and all(cpu in threads and cpu in cores for cpu in cpus)
                     and context['hardware_lanes'] == {str(cpu): [threads[cpu], cores[cpu]] for cpu in cpus},
                     'provider CPU lanes differ from the exact target resources')
            provider = authority
            lanes = dict(cpu_ids=cpus, endpoints=context['hardware_lanes'], launch_scope_matches=context['scope_matches'])
            timing_context = plan['timing_frontier']
        rows.append(dict(option_id=id_, original=id_ == baseline, legality=row['legal_status'],
            legality_reason='ORIGINAL_EXECUTION_RETAINED' if id_ == baseline else fact['reason'],
            legality_evidence=row['legality_evidence'], cpu_placement=lanes,
            memory=_memory(plan, scope, authority_document), cost=_cost(row), timing_context_only=timing_context))
    graph = _graph(provider_graph, scope, provider, threads, effects)
    report = dict(schema=SCHEMA, schema_version=1, domain=domain_ref, legalities=legalities_ref,
        provider_graph=provider_graph, effect_frontier=effect_frontier, scope_sha256=scope_sha, baseline_id=baseline,
        original_observation=graph, target=dict(graph=scope['target'], process_constraint=encode(constraint.evidence),
            allowed_target_cpu_ids=list(target_cpus), original_cpu_ids=original_cpus, reported_host_memory=host_memory),
        options=rows, domain_frontier=choice['coverage_frontier'], ranking_frontier=choice['cost_frontier'],
        ranking_status=choice['status'], execution_default='NO_OP', deployment_authorized=False)
    if import_frontier is not None:
        report['original_import_bindings'] = _import_bindings(import_frontier, scope, effects)
    return report


def verify_cpu_entry_decision_ledger(ledger_ref):
    saved = read_artifact(ledger_ref)
    _require(ledger_ref['schema'] == SCHEMA and saved['schema_version'] == 1, 'unsupported decision ledger')
    bindings = saved.get('original_import_bindings')
    replay = derive_cpu_entry_decision_ledger(saved['domain'], saved['legalities'],
        provider_graph=saved['provider_graph'], effect_frontier=saved['effect_frontier'],
        import_frontier=None if bindings is None else bindings['reference'])
    _require(saved == replay, 'retained decision ledger differs at cold replay')
    return replay
