"""Plan the resources SCAR can control without claiming full-process bounds.

The strict whole-entry problem is preserved as a separate input. A contract
changes the planning scope; excluded RSS/placement facts are not discharged or
turned into zero. One synchronous activation owns one caller lane and a scoped
payload budget. That lane is not a reservation of a physical CPU core.
"""
from copy import deepcopy
from dataclasses import replace

from scar.ir.execution_recipe import read_artifact
from scar.ir.implementation_plan import PlanningResource, validate_problem, digest
from scar.ir.record_codec import encode
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.controlled-region-planning-projection'
_OUTSIDE_SCOPE = frozenset(('whole_entry_cpu_execution_domain',
    'whole_entry_opaque_memory_and_residency'))


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _correspondence(problem, contract):
    """Check that the base problem and admitted contract name the same body."""
    _need(encode(problem.target) == contract['target'], 'controlled contract hardware differs')
    _need(len(problem.operations) == 1 and not problem.data_edges
        and not problem.activities and not problem.timing_constraints,
        'one complete synchronous region activation is supported')
    operation = problem.operations[0]
    _need(len(operation.options) == 1 and not operation.predecessors,
        'one already source-qualified implementation is required')
    mode = operation.options[0]
    _need(not mode.demands and not mode.workspace_bytes and not mode.input_locations
        and not mode.output_locations and all(row.kind == 'MEMORY' for row in problem.resources),
        'an existing concrete execution or location demand needs an explicit resource refinement map')
    _need(mode.parameters.get('logical_analysis') is False,
        'controlled qualification must consume the strict original problem, not a logical downgrade')
    source = mode.parameters['source_region']; identity = contract['identity']
    if contract['kind'] == 'STATEFUL_SPECIALIZATION':
        from scar.optimization.stateful_cpu_choices import stateful_cpu_cost_context
        domain = read_artifact(identity['domain'])
        context = stateful_cpu_cost_context(domain, identity['candidate_id'])
        _need(source['candidate_id'] == identity['candidate_id']
            and source['complete_entry_context'] == context
            and context['region_sha256'] == identity['region_sha256']
            and domain['domain_id'] == identity['domain_id'],
            'stateful resource contract differs from the original candidate, geometry or entry context')
    elif contract['kind'] == 'PIPELINE_FUSION':
        selection = source['selection']
        _need(selection['region'] == identity['selected_oir_region']
            and selection['selected'] == identity['selected_oir_alternative']
            and any(row.get('schema') == 'scar.pipeline-fusion-blueprint'
                and row.get('semantic_sha256') == identity['blueprint_semantic_sha256']
                for row in problem.source_refs),
            'fusion resource contract differs from the selected source OIR or lowering blueprint')
    else:
        raise ValueError('known controlled resource contract kind required')
    return operation, mode


def project_controlled_region(problem, contract):
    """Return a common typed problem and an explicit change-of-scope receipt.

    The enclosing controller must first re-derive the contract from its cold
    admission. Feasible here means feasible within the named controlled model,
    not a promise of available physical memory or absence of OS contention.
    """
    original = validate_problem(problem)
    _need((contract.get('schema'), contract.get('schema_version')) ==
        ('scar.controlled-region-resource-contract', 1), 'admitted controlled resource contract required')
    operation, mode = _correspondence(original, contract)
    execution, memory = contract['execution'], contract['memory']
    lane = execution['caller_lane']
    _need(lane['capacity'] == 1 and lane['kind'] == 'CONTROLLED_SYNCHRONOUS_CALL_CONCURRENCY'
        and lane['physical_cpu_id'] is None and lane['exclusive_core'] is False,
        'one unbound synchronous caller lane required')
    _need(execution['native']['cpu_threads'] == 1,
        'multiworker resource contracts need their own source-qualified scope')
    budget, included = memory['budget_bytes'], memory['included_payload_bytes']
    _need(type(budget) is int and type(included) is int and 0 <= included <= budget,
        'controlled payload must fit its explicit scoped byte budget')
    contract_sha = semantic_sha256(contract)
    reference = dict(schema=contract['schema'], semantic_sha256=contract_sha,
        parent_admission=deepcopy(contract['parent_admission']), locator='#')
    references = original.source_refs+(reference,)
    caller = 'controlled:caller-lane:'+contract_sha
    resources = [PlanningResource(caller, 'EXECUTION', 1, references+(
        dict(role='SCAR_CALL_CONCURRENCY_NOT_PHYSICAL_CPU_RESERVATION',
            allowed_cpu_ids=deepcopy(execution['allowed_cpu_ids'])),))]
    reserved = {}
    if budget:
        pool = 'controlled:payload-budget:'+contract_sha
        resources.append(PlanningResource(pool, 'MEMORY', budget, references+(
            dict(budget_scope=memory['budget_scope'], physical_residency_proven=False,
                borrowed_included_in_budget=memory['borrowed_included_in_budget']),)))
        reserved[pool] = included
    outside = [dict(field=row.field, reason=row.reason, source_refs=encode(row.source_refs),
        disposition='OUTSIDE_CONTROLLED_SCOPE_NOT_DISCHARGED')
        for row in mode.unresolved_resources if row.field in _OUTSIDE_SCOPE]
    active = tuple(row for row in mode.unresolved_resources if row.field not in _OUTSIDE_SCOPE)
    parameters = deepcopy(mode.parameters)
    parameters.update(analysis_scope='SCAR_CONTROLLED_SYNCHRONOUS_REGION_RESOURCES',
        controlled_resource_contract=reference, controlled_scope_qualified=True,
        controlled_execution=deepcopy(execution), controlled_memory=deepcopy(memory),
        strict_whole_entry_obligations=outside,
        unmodeled_context=deepcopy(contract['unmodeled_context']),
        physical_feasibility_proven=False, physical_cpu_assignment_proven=False,
        lowering_authorized=False,
        budget_accounting='ONE_ACTIVATION_SCOPED_INCLUDED_PAYLOAD_RESERVED_ONCE',
        memory_lifetime='OWNERSHIP_SCOPE_ONLY_NOT_PHYSICAL_FREE_OR_PROCESS_RSS',
        external_output_ownership_at_return='ORIGINAL_CONTRACT_RETAINED',
        parallel_entry_activation_supported=False)
    scoped_mode = replace(mode, proof_refs=mode.proof_refs+(reference,), demands={caller:1},
        workspace_bytes={}, input_locations={}, output_locations={}, parameters=parameters,
        unresolved_resources=active)
    result = validate_problem(replace(original, source_refs=references,
        resources=tuple(resources), reserved_memory=reserved,
        operations=(replace(operation, options=(scoped_mode,)),),
        assumptions=original.assumptions+('SCAR_CONTROLLED_ONE_ACTIVATION_RESOURCE_SCOPE_ONLY',
            'CALLER_LANE_IS_NOT_A_CORE_ASSIGNMENT_OR_EXCLUSIVE_RESERVATION',
            'OPAQUE_RSS_NUMA_CACHE_AND_EXTERNAL_CONTENTION_REMAIN_UNMODELED_CONTEXT',
            'CONTROLLED_FEASIBILITY_DOES_NOT_AUTHORIZE_DEPLOYMENT')))
    receipt = dict(schema=SCHEMA, schema_version=1, contract=reference,
        strict_problem_sha256=digest(original), controlled_problem_sha256=digest(result),
        planning_scope=parameters['analysis_scope'], changed_scope=True,
        original_whole_entry_obligations=outside,
        active_obligations=encode(active), resources=encode(tuple(resources)),
        reserved_payload_bytes=included, original_cost_retained=encode(mode.cost),
        resource_unknowns_replaced_by_zero=False, strict_problem_unchanged=True,
        physical_feasibility_proven=False, execution_authorized=False, deployment_authorized=False)
    return result, receipt


def verify_controlled_region(original, contract, problem, receipt):
    expected, proof = project_controlled_region(original, contract)
    _need(digest(validate_problem(problem)) == digest(expected) and receipt == proof,
        'controlled planning differs from admitted source/capacity/scope replay')
    return expected
