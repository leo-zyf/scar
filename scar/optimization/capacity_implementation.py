"""Two source-qualified capacity policies in the shared implementation IR.

N/E are explicit performance parameters, not inferred source loop counts.
The region conservatively reserves its entire private array payload for either
mode.  This formal payload constraint is not a physical RAM or whole-RSS bound.
"""
from copy import deepcopy
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.implementation_plan import (CostEstimate, ImplementationOption,
    PlanningOperation, PlanningProblem, PlanningResource, digest, problem_document,
    problem_from_dict, validate_problem)
from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.record_codec import decode, encode

COST_SCHEMA = 'scar.capacity-mode-cost-estimates'
COST_SCOPE = 'ONE_SOURCE_ACTIVATION_FOR_EXPLICIT_STEPS_AND_CAPACITY_EPOCHS'
ISOLATED = 'ISOLATED'
CAPACITY = 'CAPACITY_ONLY_DIRTY'
_MODES = (ISOLATED, CAPACITY)
_LANE = 'logical.single-owner-activation'
_MEMORY = 'formal.private-array-payload'
_RUNTIME_FILES = ('runtime/session_scope.py', 'runtime/session_scope_bundle.py',
    'runtime/native_workspace_session.py', 'runtime/native_stateful_repeat.py',
    'runtime/stateful_guard.py', 'runtime/native_stateful.py', 'runtime/stateful_contract.py',
    'runtime/native_loop.py', 'runtime/loop_contract.py', 'runtime/entry_runner.py',
    'runtime/array_copy_guard.py', 'trace/code_snapshot.py', 'entrypoint.py')


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _ref(value, schema=None):
    ref = value if type(value) is ProviderArtifact else decode(ProviderArtifact, value)
    _need(Path(ref.uri).is_absolute() and (schema is None or ref.schema == schema),
          'original absolute artifact reference with the expected schema required')
    return ref


def _fact(reference, locator):
    return encode(ProviderFactReference(reference, locator))


def _parents(capacity_plan, capacity_plan_ref, steps, epochs):
    plan = deepcopy(capacity_plan)
    ref = _ref(capacity_plan_ref, 'scar.session-capacity-plan')
    unsigned = dict(plan); identity = unsigned.pop('id', None)
    _need((plan.get('schema'), plan.get('schema_version'), plan.get('status'), plan.get('capacity_mode'))
          == ('scar.session-capacity-plan', 1, 'PROVEN_CONDITIONAL_CAPACITY_POLICY', 'CAPACITY_ONLY_DIRTY')
          and identity == 'session-capacity-plan:'+digest(unsigned),
          'verified conditional capacity plan with its complete original identity required')
    _need(type(steps) is int and type(epochs) is int and 0 <= steps < 2**60
          and (epochs == 0 if steps == 0 else 1 <= epochs <= steps),
          'explicit N=E=0 or positive steps N with 1<=capacity epochs E<=N required')
    target = _ref(plan['target'], 'scar.target-hardware-graph')
    _need(plan['resource_scope']['target'] == encode(target)
          and plan['content_policy']['full_snapshot_before_every_step'] is True
          and plan['content_policy']['retained_roi_correspondence'] is False
          and plan['content_policy']['per_step_array_and_data_owner_references_across_host_intervals'] is False,
          'capacity mode cannot acquire input owners, partial snapshots or retained content')
    slots = plan['slot_lifetimes']
    _need(type(slots) is list and slots and len({s['slot'] for s in slots}) == len(slots)
          and all(type(s['capacity_bytes']) is int and s['capacity_bytes'] > 0 for s in slots),
          'original positive private array slot capacities required')
    payload = sum(s['capacity_bytes'] for s in slots)
    _need(payload == plan['resource_scope']['known_private_array_payload_bytes'],
          'private payload differs from the original capacity plan')
    return plan, ref, target, payload


def capacity_cost_context(capacity_plan, *, capacity_plan_ref, steps, epochs):
    """Exact context for separately published mode estimates, including shapes.

    Shape/layout identity is the full original stage-plan digest and original
    slot vector, rather than guessed dimensions. Runtime source receipts pin
    the existing scoped guard/binding implementation without importing it.
    """
    plan, ref, target, payload = _parents(capacity_plan, capacity_plan_ref, steps, epochs)
    root = Path(__file__).resolve().parents[1]
    runtime = {name:artifact_reference(root/name, 'scar.python-source') for name in _RUNTIME_FILES}
    return dict(scope=COST_SCOPE, capacity_plan=encode(ref), capacity_plan_sha256=digest(plan),
        target=encode(target), source_parents=deepcopy(plan['parents']),
        original_step_plan_sha256=plan['parents']['capacity_input_sha256']['step_plan'],
        private_slot_vector=[dict(slot=s['slot'], capacity_bytes=s['capacity_bytes']) for s in plan['slot_lifetimes']],
        private_payload_bytes=payload, steps=steps, retained_capacity_epochs=epochs,
        isolated_allocation_epochs=steps, tick=dict(numerator=1, denominator=1000000000),
        runtime_policy_sources=runtime,
        policies={ISOLATED:'FRESH_ISOLATED_NATIVE', CAPACITY:'RETAIN_PRIVATE_CAPACITY_FULL_SNAPSHOT'},
        estimates_cover='COMPLETE_EXPLICIT_MODE_REGION_MODEL_NOT_A_KERNEL_OR_PARTIAL_COMPONENT',
        source_loop_count_or_replacement_authority=False)


def _costs(costs, context, plan_ref):
    authority = 'capacity-mode-context:'+digest(context)
    missing = ('NO_INDEPENDENT_COMPLETE_MODE_REGION_ESTIMATE',
               'OPAQUE_HOST_AND_RUNTIME_SERVICE_COSTS_UNMEASURED')
    unknown = lambda mode: CostEstimate(None, None, None, 'UNKNOWN', authority,
        (_fact(plan_ref, '#/resource_scope'),), (*missing, 'MODE:'+mode))
    result = {mode:unknown(mode) for mode in _MODES}
    if costs is None:
        return result
    if isinstance(costs, (str, Path)):
        ref = _ref(artifact_reference(Path(costs).resolve(), COST_SCHEMA), COST_SCHEMA)
        document = read_artifact(encode(ref))
    else:
        _need(type(costs) is dict and set(costs) == {'reference', 'document'},
              'independently published cost reference and document required')
        ref = _ref(costs['reference'], COST_SCHEMA); document = deepcopy(costs['document'])
        _need(digest(read_artifact(encode(ref))) == digest(document),
              'mode estimate source bytes differ from the supplied document')
    _need((document.get('schema'), document.get('schema_version')) == (COST_SCHEMA, 1)
          and document.get('scope') == COST_SCOPE
          and document.get('authority') == 'EXPLICIT_DECLARED_MODE_MODEL_NOT_MEASUREMENT'
          and document.get('coverage') == 'COMPLETE_EXPLICIT_MODE_REGION_MODEL'
          and digest(document.get('context')) == digest(context),
          'mode estimate scope/source/shape/target/runtime-policy/N/E context differs')
    conditions = document.get('conditions')
    _need(type(conditions) is list and conditions and all(type(s) is str and s for s in conditions),
          'explicit declared mode model conditions required')
    rows = document.get('estimates')
    _need(type(rows) is dict and set(rows) <= set(_MODES), 'only the two explicit mode estimates are supported')
    for mode, row in rows.items():
        _need(type(row) is dict and set(row) == {'estimate_ns', 'lower_ns', 'upper_ns', 'basis', 'missing_terms'},
              'explicit mode estimate fields required')
        _need(row['basis'] in ('UNKNOWN', 'DECLARED_MODEL'),
              'canonical/empirical components cannot be promoted to complete mode observations')
        _need(type(row['missing_terms']) is list and all(type(s) is str and s for s in row['missing_terms']),
              'explicit model residuals must remain recorded')
        if row['basis'] == 'UNKNOWN':
            _need(all(row[k] is None for k in ('estimate_ns', 'lower_ns', 'upper_ns')),
                  'unknown mode costs cannot acquire a zero estimate or invented bounds')
        else:
            _need(type(row['estimate_ns']) is int, 'declared mode model requires an explicit integer point estimate')
        result[mode] = CostEstimate(row['estimate_ns'], row['lower_ns'], row['upper_ns'], row['basis'], authority,
            (_fact(ref, '#/estimates/'+mode), _fact(ref, '#/conditions')),
            tuple(dict.fromkeys(row['missing_terms'])))
    read_artifact(encode(ref))
    return result


def derive_capacity_problem(capacity_plan, *, capacity_plan_ref, steps, epochs, costs=None):
    """Build one source region with both modes, preserving unknown costs.

    The caller cold-replays the capacity preparation. The adapter adds no
    source rewrite authority and models no physical CPU allocation for opaque
    host work. Only separately published, exact-context declared model costs
    are admitted in this first interface; measured mode costs need a reader.
    """
    plan, ref, target, payload = _parents(capacity_plan, capacity_plan_ref, steps, epochs)
    context = capacity_cost_context(plan, capacity_plan_ref=encode(ref), steps=steps, epochs=epochs)
    estimates = _costs(costs, context, ref)
    parent = plan['parents']; activation = _ref(parent['activation'], 'scar.workspace-session-activation')
    proof = _ref(parent['capacity_proof'], 'scar.session-capacity-proof')
    source = (_fact(ref, '#'), plan['original_source_overlay'])
    resources = (
        PlanningResource(_LANE, 'EXECUTION', 1, (_fact(ref, '#/epochs/maximum_live_epochs_in_normal_single_owner_policy'),)),
        PlanningResource(_MEMORY, 'MEMORY', payload, (_fact(ref, '#/slot_lifetimes'),)))
    choices = []
    slot_count = len(plan['slot_lifetimes'])
    for mode, lowering, allocation_epochs in ((ISOLATED, 'FRESH_ISOLATED_NATIVE', steps),
                                            (CAPACITY, 'RETAIN_PRIVATE_CAPACITY_FULL_SNAPSHOT', epochs)):
        proofs = (_fact(activation, '#/leases'), _fact(ref, '#/parents/activation')) if mode == ISOLATED else (
            _fact(proof, '#'), _fact(ref, '#/obligations'))
        parameters = dict(steps=steps, declared_retained_epochs=epochs, allocation_epochs=allocation_epochs,
            create_count=allocation_epochs, destroy_count=allocation_epochs,
            full_snapshot_steps=steps, full_snapshot_each_step=True, content_residency=False,
            work_demand=dict(scope='NORMAL_SUCCESS_FOR_EXPLICIT_STEPS_AND_EPOCHS',
                private_slot_allocations=slot_count * allocation_epochs,
                private_slot_releases=slot_count * allocation_epochs,
                private_payload_requested_bytes=payload * allocation_epochs,
                peak_live_private_payload_bytes=payload if allocation_epochs else 0,
                retained_inter_step_intervals=steps - allocation_epochs,
                private_payload_bytes_during_retained_intervals=payload if steps > allocation_epochs else 0,
                steps_requiring_full_snapshot=steps, physical_traffic_bytes=None,
                retention_byte_nanoseconds=None, relative_mode_duration_ns=None,
                source_refs=[_fact(ref, '#/slot_lifetimes'), _fact(ref, '#/epochs'),
                             _fact(ref, '#/content_policy')]),
            source_loop_count_is_not_inferred=True, performance_parameters_not_source_execution_proof=True,
            conservative_private_memory_span='ENTIRE_SOURCE_REGION_FOR_BOTH_MODES',
            formal_payload_budget_is_not_available_ram=True,
            external_state_and_session_metadata_not_bounded_by_this_resource=True,
            outer_host_physical_threads_and_memory_unknown=True,
            cost_context=context, original_budget=deepcopy(plan['resource_scope']['budget']),
            runtime_policy_source_admission_required=True, source_replacement_authorized=False,
            execution_authorized=False, deployment_authorized=False)
        choices.append(ImplementationOption(mode, lowering, proofs, {_LANE:1},
            {_MEMORY:payload if steps else 0}, {}, {}, estimates[mode], parameters))
    operation = PlanningOperation('source-activation:'+parent['outer_profile_sha256'],
        source, (), tuple(choices), ISOLATED)
    problem = PlanningProblem(target, source, resources, (operation,), (), assumptions=(
        'ONE_ORIGINAL_SHARED_SOURCE_ACTIVATION_NOT_UNROLLED_KERNELS',
        'STEPS_AND_EPOCHS_ARE_EXPLICIT_CONDITIONAL_PERFORMANCE_PARAMETERS_NOT_SOURCE_LOOP_COUNTS',
        'LOGICAL_SINGLE_OWNER_CAPACITY_IS_NOT_A_RESERVED_HARDWARE_CPU',
        'BOTH_MODES_RESERVE_PRIVATE_ARRAY_PAYLOAD_FOR_THE_ENTIRE_MODELED_REGION_CONSERVATIVELY',
        'FORMAL_PRIVATE_PAYLOAD_BUDGET_EQUALS_RETAINED_SLOT_VECTOR_NOT_INVENTORY_FREE_RAM',
        'ORIGINAL_FULL_REGION_BUDGET_REFERENCE_RETAINED_WITHOUT_REINTERPRETATION',
        'EXTERNAL_STATE_SESSION_METADATA_OPAQUE_HOST_MEMORY_AND_WHOLE_RSS_ARE_NOT_BOUNDED',
        'SOURCE_AND_RUNTIME_GUARDS_AND_ORIGINAL_PROVIDER_EXTERNAL_ASSUMPTIONS_REMAIN_REQUIRED',
        'KNOWN_NS_IF_SUPPLIED_ARE_DECLARED_MODE_MODEL_VALUES_NOT_MEASURED_PHYSICAL_LATENCY',
        'NO_SOURCE_REPLACEMENT_RUNTIME_OR_DEPLOYMENT_AUTHORITY_GRANTED'))
    return validate_problem(problem)


def verify_capacity_problem(capacity_plan, problem, **kwargs):
    """Rebuild both modes, exact cost evidence/context and resource projection."""
    retained = problem_from_dict(problem) if type(problem) is dict else validate_problem(problem)
    expected = derive_capacity_problem(capacity_plan, **kwargs)
    _need(digest(problem_document(retained)) == digest(problem_document(expected)),
          'capacity planning problem differs from original source/cost/model replay')
    return problem


__all__ = ['COST_SCHEMA', 'COST_SCOPE', 'ISOLATED', 'CAPACITY', 'capacity_cost_context',
           'derive_capacity_problem', 'verify_capacity_problem']
