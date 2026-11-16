"""One shared transactional step under an original, guarded source loop.

The recurrence is a parametric description, not a runtime content identity or
permission to cache an array. Source-F4 replay belongs to the caller frontend;
this module preserves its evidence and plans only the existing step backend.
"""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from scar.compiler.stateful_model import (
    stateful_digest, stateful_from_dict, validate_stateful_region,
)
from scar.ir.execution_recipe import read_artifact
from scar.optimization.hardware import propose_cpu_allocations
from scar.optimization.resource_solver import semantic_sha256
from scar.optimization.stateful_schedule import plan_stateful_region


SCHEMA = 'scar.stateful-loop-plan'
EXECUTION_FAMILY = 'ORDERED_TRANSACTIONAL_STATEFUL_REPEAT_ABI1_V1'
LOOP_GUARDS = (
    'FIXED_EXACT_NONNEGATIVE_UINT64_REPETITION_COUNT',
    'ORIGINAL_CALLER_CODE_GLOBAL_STEP_AND_EXACT_BUILTIN_RANGE_IDENTITIES',
    'IDENTICAL_ARRAY_IDENTITIES_DESCRIPTORS_SCALARS_AND_PROVIDERS_EACH_STEP',
    'ONLY_ARRAY_CONTENTS_ARE_LOOP_CARRIED_STATE',
    'NO_INTERSTEP_PYTHON_CODE_HOOKS_OBSERVERS_OR_CONCURRENT_STATE_ACCESS',
    'NO_INTERSTEP_PYTHON_SIGNAL_OR_CANCELLATION_RESPONSIVENESS_CONTRACT',
    'ALL_ORIGINAL_SINGLE_CALL_GUARDS_AND_EXTERNAL_ASSUMPTIONS',
)


def _json(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _loop_model(step, parameter_map, trip_count):
    """Derive loop-carried roles from writes, never from names or observations."""
    validate_stateful_region(step)
    arrays = {row.id: row for row in step.inputs if row.kind == 'array'}
    writes = {name: [] for name in arrays}
    for action in step.actions:
        if action.kind == 'write':
            writes[action.target].append(dict(action_id=action.id,
                local_epoch_before=action.epoch_before,
                local_epoch_after=action.epoch_after,
                bounds=_json(action.bounds), op=action.op))
    roots = []
    for row in step.inputs:
        outer_path = [parameter_map[row.path[0]], *row.path[1:]]
        base = dict(input_id=row.id, step_path=list(row.path), outer_path=outer_path,
            kind=row.kind, binding_stability='RUNTIME_GUARD_NOT_INFERRED_FROM_CONTENT',
            origins=deepcopy(list(row.origins)))
        if row.kind == 'scalar':
            roots.append(dict(**base, role='GUARDED_LOOP_LIVE_IN_SCALAR',
                scalar_value_fixed_only_under_required_guards=True,
                content_version=None, computation_hoisted=False))
            continue
        local_writes = writes[row.id]
        epochs = len(local_writes)
        read_epochs = sorted({node.attrs['epoch'] for node in step.nodes
            if node.op == 'input' and node.attrs['name'] == row.id})
        roots.append(dict(**base,
            role='ORDERED_ARRAY_CONTENT_CARRY' if epochs else 'GUARDED_LOOP_LIVE_IN_ARRAY',
            array_identity_and_descriptor_fixed_only_under_required_guards=True,
            content_model='PARAMETRIC_CALL_LOCAL_CONTENT_NOT_RUNTIME_LOGICAL_VALUE_ID',
            content_coordinate=dict(root=row.id, iteration='i', local_epoch='e'),
            iteration_domain=dict(lower_inclusive=0, upper_exclusive=trip_count),
            local_epoch_domain=dict(lower_inclusive=0, upper_inclusive=epochs),
            source_read_epochs=read_epochs, ordered_writes=local_writes,
            entry=dict(iteration='i', local_epoch=0),
            exit=dict(iteration='i', local_epoch=epochs),
            backedge=dict(enabled_when='i + 1 < trip_count',
                next_entry=dict(iteration='i + 1', local_epoch=0),
                previous_exit=dict(iteration='i', local_epoch=epochs),
                relation='SAME_COMMITTED_CONTENT_UNDER_LOOP_AND_STEP_GUARDS'),
            invariant_content_under_guards=not bool(epochs),
            cross_call_content_equality_observed=False, computation_hoisted=False))
    return dict(representation='SHARED_STEP_WITH_PARAMETRIC_ITERATION_AND_LOCAL_EPOCH',
        trip_count=trip_count, iteration_symbol='i', local_epoch_symbol='e',
        step_instances_stored=1, iterations_unrolled=False, inputs=roots,
        written_array_inputs=[row['input_id'] for row in roots
            if row['role'] == 'ORDERED_ARRAY_CONTENT_CARRY'],
        readonly_array_inputs=[row['input_id'] for row in roots
            if row['role'] == 'GUARDED_LOOP_LIVE_IN_ARRAY'],
        scalar_live_in_inputs=[row['input_id'] for row in roots
            if row['role'] == 'GUARDED_LOOP_LIVE_IN_SCALAR'],
        computed_values='ITERATION_LOCAL_EXCEPT_COMMITTED_ARRAY_CONTENT',
        object_identity_is_not_content_identity=True,
        runtime_logical_value_or_storage_identity_inferred=False,
        static_role_is_not_cross_call_cache_authorization=True)


def _proof_structure(proof):
    # Exact distinct proof type: this is not a pure/invariant-loop proof cast.
    from scar.compiler.stateful_loop_frontend import StatefulLoopProof
    _need(type(proof) is StatefulLoopProof, 'original StatefulLoopProof required')
    step = validate_stateful_region(proof.step)
    _need(type(proof.id) is str and bool(proof.id), 'original loop proof ID required')
    _need(type(proof.parameters) is tuple and proof.parameters
        and len(set(proof.parameters)) == len(proof.parameters)
        and all(type(name) is str and name.isidentifier() for name in proof.parameters),
        'original ordered caller parameters required')
    _need(proof.trip_count_parameter in proof.parameters,
        'loop trip count must be an original caller parameter')
    _need(type(proof.parameter_map) is dict
        and set(proof.parameter_map) == set(step.parameter_names)
        and all(value in proof.parameters for value in proof.parameter_map.values()),
        'every step parameter must map to a fixed caller parameter')
    _need(type(proof.step_global_name) is str and proof.step_global_name.isidentifier(),
        'original plain global step lookup required')
    _need(type(proof.caller_unit) is str and bool(proof.caller_unit)
        and type(proof.function_ordinal) is int and proof.function_ordinal >= 0
        and type(proof.interface_ordinal) is int and proof.interface_ordinal >= 0,
        'original caller source unit/function/interface location required')
    _need(type(proof.call_evidence) is dict and proof.call_evidence
        and type(proof.origins) is tuple and proof.origins
        and type(proof.producer_pins) is dict and proof.producer_pins,
        'original source loop/CALL evidence and producer pins required')
    positional = proof.call_evidence.get('positional_parameters')
    keywords = proof.call_evidence.get('keyword_parameters')
    _need(type(positional) in (tuple, list) and type(keywords) is dict
        and all(type(name) is str and name in proof.parameters for name in positional)
        and all(type(name) is str and name.isidentifier()
            and type(value) is str and value in proof.parameters
            for name, value in keywords.items()),
        'original positional/keyword outer-parameter call layout required')
    reference = _json(asdict(proof.step_reference)) if hasattr(proof.step_reference, '__dataclass_fields__') else _json(proof.step_reference)
    retained = stateful_from_dict(read_artifact(reference))
    _need(stateful_digest(retained) == stateful_digest(step),
        'retained original step artifact differs from shared loop step')
    for source in (proof.source, step.source):
        _need(sha256(Path(source.path).read_bytes()).hexdigest() == source.sha256,
            'original caller/step source changed')
    return step, reference


def plan_stateful_loop(proof, target, *, input_shapes, trip_count,
                       memory_budget_bytes, cpu_threads=1):
    """Return one explicit feasible CPU preview; do not choose a fastest plan.

    Full original F4 replay via validate_stateful_loop_proof(view, provider) is
    independently mandatory at C0/lowering. This structural planner cannot
    discharge those source conditions or authorize native deployment itself.
    """
    step, reference = _proof_structure(proof)
    _need(type(trip_count) is int and 0 <= trip_count < 2**64,
        'explicit exact nonnegative uint64 trip count required')
    _need(type(cpu_threads) is int and cpu_threads >= 1,
        'explicit positive CPU thread count required')
    _need(type(memory_budget_bytes) is int and memory_budget_bytes >= 0,
        'explicit nonnegative region array-payload budget required')
    _need(type(input_shapes) is dict, 'step input-ID shape declarations required')
    choices = propose_cpu_allocations(target, memory_budget_bytes=memory_budget_bytes,
        max_threads=cpu_threads)
    binding = next((row for row in choices if len(row.cpu_ids) == cpu_threads), None)
    _need(binding is not None, 'requested CPU thread count has no retained feasible placement')
    effective_budget = binding.memory_budget_bytes
    step_plan = plan_stateful_region(step, input_shapes=input_shapes,
        cpu_threads=cpu_threads, memory_budget_bytes=effective_budget,
        resource_binding=binding)
    result = dict(schema=SCHEMA, schema_version=1,
        execution_family=EXECUTION_FAMILY, proof_id=proof.id,
        proof_sha256=semantic_sha256(asdict(proof)),
        source=asdict(proof.source), step_source=asdict(step.source),
        step_reference=reference, step_sha256=stateful_digest(step),
        parameters=list(proof.parameters), parameter_map=dict(proof.parameter_map),
        trip_count_parameter=proof.trip_count_parameter,
        step_global_name=proof.step_global_name, caller_unit=proof.caller_unit,
        function_ordinal=proof.function_ordinal, interface_ordinal=proof.interface_ordinal,
        call_evidence=deepcopy(proof.call_evidence), origins=deepcopy(list(proof.origins)),
        producer_pins=deepcopy(proof.producer_pins),
        target=deepcopy(step_plan['resource_context']['target_snapshot']),
        resource_binding=asdict(binding), cpu_threads=cpu_threads,
        input_shapes=_json(input_shapes), trip_count=trip_count,
        memory_budget_bytes=memory_budget_bytes,
        effective_memory_budget_bytes=effective_budget,
        budget_scope=step_plan['budget_scope'],
        step_plan=step_plan,
        loop_model=_loop_model(step, proof.parameter_map, trip_count),
        required_live_guards=list(LOOP_GUARDS) + list(step_plan['guard_contract']),
        external_assumptions=list(step_plan['external_assumptions']),
        source_proof_authority=dict(
            complete_original_f4_replay_required_at_lowering=True,
            validator='scar.compiler.stateful_loop_frontend.validate_stateful_loop_proof',
            original_f4_validated_by_this_planner=False,
            retained_step_bytes_and_source_snapshots_checked=True),
        execution=dict(step_plan_instances=1,
            per_step_commit='UNCHANGED_ORIGINAL_TRANSACTIONAL_SINGLE_CALL_BOUNDARY',
            completed_counts_only_successfully_committed_steps=True,
            on_precommit_failure='RESUME_ORIGINAL_STEP_ONLY_FOR_UNCOMMITTED_SUFFIX',
            fallback_iteration_start='completed', fallback_iteration_stop='trip_count',
            replay_already_committed_prefix=False,
            failing_step_external_state_unchanged_required=True,
            zero_trip_count='NO_STEP_EXECUTION_NO_STEP_ALLOCATIONS_NO_COMMITS',
            allocations_or_team_setup_hoisted=False,
            per_step_allocations_and_guards_preserved=True,
            interstep_observer_support=False),
        status=step_plan['status'], frontier=deepcopy(step_plan['frontier']),
        budget_satisfied=step_plan['budget_satisfied'],
        resource_choice='FIRST_EXACT_THREAD_COUNT_ORIGINAL_UNRANKED_FEASIBLE_PREVIEW',
        cost=dict(status='UNKNOWN', duration_ns=None, timing_source=None,
            trip_count_scaling_authorized=False),
        selected=False, selection_decision='EXPLICIT_FEASIBILITY_PREVIEW_ONLY',
        time_predicted=False, time_optimality_proven=False,
        physical_optimality_proven=False, deployment_authorized=False,
        iterations_unrolled=False, target_executed=False)
    result = _json(result)
    result['id'] = 'stateful-loop-plan:' + semantic_sha256(result)
    return result


def verify_stateful_loop(proof, target, report):
    """Recompute the placement, one step plan, recurrence and all conditions."""
    value = json.loads(Path(report).read_bytes()) if isinstance(report, (str, Path)) else _json(report)
    _need(value.get('schema') == SCHEMA and type(value.get('schema_version')) is int
        and value['schema_version'] == 1, 'versioned stateful loop plan required')
    expected = plan_stateful_loop(proof, target, input_shapes=value['input_shapes'],
        trip_count=value['trip_count'], memory_budget_bytes=value['memory_budget_bytes'],
        cpu_threads=value['cpu_threads'])
    _need(semantic_sha256(value) == semantic_sha256(expected),
        'stateful loop plan differs from original step/recurrence/resource replay')
    return value


__all__ = ['SCHEMA', 'EXECUTION_FAMILY', 'LOOP_GUARDS',
    'plan_stateful_loop', 'verify_stateful_loop']
