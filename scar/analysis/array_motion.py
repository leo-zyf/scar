"""Conditional loop motion facts over the existing closed ArrayRegion DAG.

The caller owns original source/F4 admission. Parameter-origin stability is
separate from referent-content stability; this analysis grants neither and
does not construct CFG, MemorySSA, runtime values or a replacement source IR.
"""
from dataclasses import asdict
import json

from scar.compiler.model import validate_region
from scar.optimization.graph import value_kinds
from scar.optimization.resource_solver import semantic_sha256


SCHEMA = 'scar.array-motion'

_OBLIGATIONS = {
    'source': (
        ('ORIGINAL_F4_LOOP_AND_ARGUMENT_MAPPING',
         'Replay the original loop/call/carry/formal mapping and current callable environment.'),
        ('ORIGINAL_LOOP_HAS_NO_OTHER_REFERENT_WRITERS',
         'The complete original loop must preserve external input origins and contain no intervening input-content writes.'),
        ('NO_INTERMEDIATE_ESCAPE_OR_IDENTITY_OBSERVERS',
         'No intermediate identity, allocation lifetime, release callback or per-iteration result observer may distinguish motion.'),
    ),
    'runtime': (
        ('EXACT_ARRAY_AND_CALLABLE_RUNTIME_GUARDS',
         'Admit exact ndarray dtype/rank/layout/extents and the original callable; shape declarations are not live guards.'),
        ('POSITIVE_TRIP_ONLY',
         'Return the original carry alias for nonpositive counts before hoisted work, input scans or extra allocation.'),
        ('FP_AND_FAILURE_FALLBACK',
         'Retain original operand order and guarded FP/finite/failure handling; withhold private results before original-loop fallback.'),
        ('FRESH_PRIVATE_RESULTS_AND_BUDGET',
         'Results and carry destinations must be fresh private payloads within the separately verified allocation budget.'),
    ),
    'external': (
        ('NO_FOREIGN_CONCURRENT_INPUT_WRITERS',
         'Array contents remain stable throughout the loop; origin identity or exact ndarray type cannot prove this.'),
        ('NO_INTERSTEP_CALLBACKS_OR_HOOKS',
         'No callbacks, hooks or concurrent observers may modify or observe intervening contents or evaluation order.'),
        ('FP_STATUS_FLAGS_ARE_NOT_PROGRAM_OUTPUTS',
         'Moving evaluation requires the retained floating-status observation contract.'),
    ),
}


def _condition(status, reason, obligations=()):
    return dict(status=status, reason=reason, obligations=list(obligations))


def derive_array_motion(body, *, carry_parameter, loop_interface):
    """Return replayable facts and shared proposed placements, without running Q.

    ``readonly_parameters`` declares the requested interface roles. Its names,
    boolean credentials, and embedded proof documents do not discharge source
    or runtime obligations. Constant-only arithmetic is not evaluated/folded
    here; consumers depending on it cannot move ahead of that unmoved work.
    """
    validation = validate_region(body)
    if carry_parameter not in body.parameters:
        raise ValueError('motion carry must be an original body parameter')
    if type(loop_interface) is not dict or loop_interface.get('carry_parameter') != carry_parameter:
        raise ValueError('original caller carry mapping is required for array motion')
    readonly = loop_interface.get('readonly_parameters')
    if (type(readonly) not in (list, tuple) or len(set(readonly)) != len(readonly)
            or set(readonly) != set(body.parameters)-{carry_parameter}):
        raise ValueError('all non-carry original parameters require explicit requested readonly roles')
    # Canonical JSON both binds every supplied interface field and detaches the
    # result from mutable caller metadata. No field is a proof credential.
    interface = json.loads(json.dumps(loop_interface, sort_keys=True, allow_nan=False))
    document = asdict(body)
    source = dict(document['source'])
    origins = source.pop('origins')
    kinds = value_kinds(body.nodes)
    dependencies, consumers, blockers = {}, {node.id: [] for node in body.nodes}, {}
    for node in body.nodes:
        dependencies[node.id] = ({node.attrs['name']} if node.op == 'input' else
            set().union(*(dependencies[key] for key in node.inputs)))
        blockers[node.id] = set().union(*(blockers[key] for key in node.inputs))
        if node.op == 'slice' or (node.op not in ('input', 'const') and not dependencies[node.id]):
            blockers[node.id].add(node.id)
        for key in node.inputs:
            consumers[key].append(node.id)
    invariant = {node.id for node in body.nodes if node.op not in ('input', 'const')
        and dependencies[node.id] and carry_parameter not in dependencies[node.id]}
    invariant_nodes = [node.id for node in body.nodes if node.id in invariant]
    roots = [node.id for node in body.nodes if node.id in invariant and
        (node.id == body.output or any(key not in invariant for key in consumers[node.id]))]
    fully_invariant_output = body.output in invariant
    frontier = []
    if fully_invariant_output:
        frontier.append(dict(reason='FULLY_INVARIANT_OUTPUT_REQUIRES_FRESH_RESULT_LOOP_CONTRACT',
            expr=body.output))
    placements = {}
    for node in body.nodes:
        deps = dependencies[node.id]
        varying = carry_parameter in deps
        if node.op == 'input':
            classification = 'INPUT_CARRY' if varying else 'INPUT_INVARIANT_ORIGIN'
            placement = 'ITERATION_CARRY' if varying else 'PRE_TEMPORAL_LOOP'
            status = 'NOT_INVARIANT' if varying else 'CONDITIONAL'
        elif node.op == 'const':
            classification, placement, status = 'CONSTANT_LITERAL', 'PRE_TEMPORAL_LOOP', 'CONDITIONAL'
        elif not deps:
            classification, placement, status = 'CONSTANT_ONLY_COMPUTATION', 'ITERATION_BODY', 'NOT_SELECTED'
        elif node.op == 'slice':
            classification, placement, status = 'DESCRIPTOR', 'ITERATION_BODY', 'UNRESOLVED'
        elif varying:
            classification, placement, status = 'CARRY_DEPENDENT_COMPUTATION', 'ITERATION_BODY', 'NOT_INVARIANT'
        else:
            classification = 'INVARIANT_COMPUTATION'
            placement, status = ('ITERATION_BODY', 'UNRESOLVED') if (
                blockers[node.id] or fully_invariant_output) else ('PRE_TEMPORAL_LOOP', 'CONDITIONAL')
        if node.id in roots and blockers[node.id]:
            frontier.append(dict(reason='INVARIANT_DEPENDS_ON_UNMOVED_SCALAR_OR_VIEW_WORK',
                expr=node.id, blocking_exprs=sorted(blockers[node.id])))
        origin = (_condition('NOT_APPLICABLE', 'No formal input origins are read.') if not deps else
            _condition('NOT_INVARIANT' if varying else 'CONDITIONAL',
                'Carry reference rebinding makes this expression variant; it does not prove an input-content write.'
                if varying else 'Fixed formal origins require the original complete loop mapping, not their readonly labels.',
                ('ORIGINAL_F4_LOOP_AND_ARGUMENT_MAPPING',)))
        content = (_condition('NOT_APPLICABLE', 'Literal/scalar-only expression has no array referent reads.') if not deps else
            _condition('NOT_INVARIANT' if varying else 'CONDITIONAL',
                'Carry-dependent values change across iterations even when input objects are not mutated.'
                if varying else 'The same input origin may still have changing contents; stability is a separate obligation.',
                ('ORIGINAL_LOOP_HAS_NO_OTHER_REFERENT_WRITERS', 'NO_FOREIGN_CONCURRENT_INPUT_WRITERS',
                 'NO_INTERSTEP_CALLBACKS_OR_HOOKS', 'FRESH_PRIVATE_RESULTS_AND_BUDGET')))
        placements[node.id] = dict(op=node.op, value_kind=kinds[node.id],
            classification=classification, status=status, temporal_placement=placement,
            formal_dependencies=sorted(deps), carry_dependent=varying,
            blocking_exprs=sorted(blockers[node.id]),
            array_motion_candidate=node.id in invariant,
            origin_binding_stability=origin, referent_content_stability=content,
            data=_condition('NOT_SATISFIED' if varying else 'PROVEN',
                'Transitive operand closure contains the original carry formal.' if varying else
                'Transitive operand closure excludes the carry; this is a syntactic data fact, not content stability.'),
            no_clobber=_condition('CONDITIONAL' if deps else 'NOT_APPLICABLE',
                'The closed Expr profile has no input-write operation. Source-loop writes, aliases and foreign writers remain unproved.'
                if deps else 'No array referent is read by this scalar/literal closure.',
                ('ORIGINAL_LOOP_HAS_NO_OTHER_REFERENT_WRITERS', 'NO_FOREIGN_CONCURRENT_INPUT_WRITERS',
                 'FRESH_PRIVATE_RESULTS_AND_BUDGET') if deps else ()),
            execution=_condition('UNRESOLVED' if blockers[node.id] else 'CONDITIONAL',
                'Unmoved scalar-only computation or unsupported view must not be crossed by its consumers.'
                if blockers[node.id] else 'Pre-loop placement is guarded by positive count and original loop/normal-completion admission.',
                ('POSITIVE_TRIP_ONLY', 'ORIGINAL_F4_LOOP_AND_ARGUMENT_MAPPING')),
            effects=dict(status='CONDITIONAL',
                basis='VALIDATED_CLOSED_ARRAY_EXPR_PROFILE_NOT_ORIGINAL_SOURCE_EFFECT_CLOSURE',
                may_raise='UNKNOWN_UNTIL_GUARDED_EXECUTION',
                input_aliasing='MAY_ALIAS_NOT_INFERRED_DISJOINT',
                obligations=['EXACT_ARRAY_AND_CALLABLE_RUNTIME_GUARDS', 'FP_AND_FAILURE_FALLBACK',
                    'NO_INTERMEDIATE_ESCAPE_OR_IDENTITY_OBSERVERS', 'FP_STATUS_FLAGS_ARE_NOT_PROGRAM_OUTPUTS']))
    status = 'UNRESOLVED' if frontier else 'CONDITIONAL' if roots else 'NO_MOVABLE_INVARIANT'
    result = dict(schema=SCHEMA, schema_version=1, body_id=body.id,
        body_sha256=semantic_sha256(document), source=source,
        source_ref_sha256=semantic_sha256(document['source']),
        source_origins=dict(count=len(origins), sha256=semantic_sha256(origins)),
        loop_interface_sha256=semantic_sha256(interface), carry_parameter=carry_parameter,
        requested_readonly_parameters=[name for name in body.parameters if name != carry_parameter],
        dependencies={node.id: sorted(dependencies[node.id]) for node in body.nodes},
        invariant_nodes=invariant_nodes, invariant_roots=roots, placements=placements,
        status=status, conditional_legality=dict(status=status,
            authority='CLOSED_ARRAY_MODEL_WITH_UNDISCHARGED_SOURCE_RUNTIME_AND_EXTERNAL_Q',
            readonly_role_is_not_source_proof=True, general_cfg_or_memoryssa_legality_proven=False),
        conditional_obligations={scope:[dict(id=identifier, status='UNRESOLVED', description=description)
            for identifier, description in rows] for scope, rows in _OBLIGATIONS.items()},
        alias_contract='INPUTS_MAY_ALIAS_NO_CLOBBER_REQUIRES_NO_INPUT_WRITES_AND_FRESH_PRIVATE_RESULTS',
        zero_trip='RETURN_ORIGINAL_CARRY_ALIAS_BEFORE_HOIST_INPUT_GUARDS_SCANS_OR_ALLOCATION',
        constant_policy='FINITE_LITERALS_NOT_EVALUATED_CONSTANT_ONLY_OPERATIONS_NOT_FOLDED_OR_MOVED',
        preserved_assumptions=list(body.assumptions), validation=validation, frontier=frontier,
        original_f4_replayed=False, source_bytes_reopened=False, runtime_guards_discharged=False,
        memory_or_physical_storage_proven=False, cost_or_timing_inferred=False,
        source_replacement_authorized=False, selected_for_use=False)
    result['id'] = semantic_sha256(result)
    return result


def verify_array_motion(report, body, *, carry_parameter, loop_interface):
    """Recompute for this exact body/interface; stale or altered facts reject."""
    expected = derive_array_motion(body, carry_parameter=carry_parameter, loop_interface=loop_interface)
    if json.dumps(report, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError('array motion differs from the exact body/interface replay')
    return expected


__all__ = ['derive_array_motion', 'verify_array_motion']
