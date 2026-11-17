"""Source caller effect obligations on original, shared F4 operations.

This is a mechanical coverage ledger, not a purity proof or an interceptor.
Guard specifications describe checks at original ports. No check is executed
and no world token is treated as a value/content version.
"""
from collections import Counter, defaultdict
from types import SimpleNamespace

from scar.analysis.source_body_interfaces import BodyInterfaceResolver
from scar.analysis.source_call_view import source_call_site
from scar.analysis.source_effects import classify_instruction
from scar.analysis.source_reference_slice import ReferenceSlicer, source_port
from scar.analysis.source_value_flow import project_value_flow
from scar.compiler.graph_frontend import _metadata, _source_identity
from scar.compiler.model import SourceRef
from scar.ir.program import normal_graph as G, normal_links as F, source_productions as P
from scar.ir.program.control import LanguageCompletionKind
from scar.ir.program.model import ValueKind, ValueType
from scar.ir.program.normal_targets import Body
from scar.ir.program.source_value_flow import SourceBodyValueFlow
from scar.ir.program.structured import PythonProtocolKind as K
from scar.ir.record_codec import encode
from scar.optimization.resource_solver import semantic_sha256
from scar.trace.code_snapshot import code_key


SCHEMA = 'scar.session-caller-effects'
_UNKNOWN = 'UNKNOWN_EFFECT_BEFORE_SITE_INVALIDATE'
_NUMERIC = ('builtins.bool', 'builtins.int', 'builtins.float')
_NO_RELEASE_CALLBACK = ('builtins.NoneType', 'builtins.bool', 'builtins.int',
                        'builtins.float', 'builtins.str', 'builtins.bytes')


def _state(kind):
    return type(kind) is ValueType and kind.kind is ValueKind.STATE


def _key(port):
    return port.template, port.owner


def _guard(opcode, subjects=(), **parameters):
    return dict(opcode=opcode, subjects=encode(tuple(subjects)), parameters=parameters,
                checked=False, failure_action=_UNKNOWN)


def _source(view, unit, function):
    model, syntax, codes, recipe, digest = _source_identity(view.catalog, unit)
    name, span, lines = _metadata(syntax, function.template.occurrence)
    candidates = [code for code in codes if code.co_name == name
        and code.co_firstlineno in lines
        and code.co_varnames[:code.co_argcount+code.co_kwonlyargcount] == function.parameter_names]
    if len(candidates) != 1:
        raise ValueError('original caller requires one complete compile-only source CodeID')
    source = SourceRef(model['path'], digest, candidates[0].co_qualname,
        span['line'], span['column'], code_key(candidates[0]),
        (dict(method='COMPILE_ONLY_RETAINED_F0', unit=unit, recipe=recipe,
              occurrence=function.template.occurrence,
              semantic_authority='ORIGINAL_F4_NOT_COMPILER_AST'),))
    return source, syntax


def _fact_indexes(original, function, slicer):
    """Only original scope ranges; no source-name or filesystem lookup."""
    facts = original.facts
    spans, by_world, names = [], defaultdict(list), defaultdict(list)
    for group in facts.scope_groups:
        if group.owner != function.body_scope.owner:
            continue
        for span in group.tables:
            table = span.table.value
            rows = getattr(facts, table)
            if not 0 <= span.start <= span.stop <= len(rows):
                raise ValueError('caller source scope fact range is outside its original table')
            spans.append(encode(span))
            if table not in ('attributes', 'items', 'operations', 'calls', 'names'):
                continue
            for ordinal in range(span.start, span.stop):
                row = rows[ordinal]
                reference = slicer.fact(facts, table, ordinal)
                entry = table, ordinal, row, reference
                if table == 'names':
                    names[source_port(facts, row.value)].append(entry)
                else:
                    by_world[(source_port(facts, row.world),
                              source_port(facts, row.result_world))].append(entry)
    return spans, by_world, names


def _classification(instruction, subjects, matches):
    """A finite guard-spec profile; all other effects retain an invalidation."""
    role, guards, unresolved = 'UNSUPPORTED_SOURCE_EFFECT', [], [_UNKNOWN]
    if type(instruction) is P.EnterDeferredBody:
        role = 'ACTIVATION_ENTRY'
        guards = [_guard('MATCH_ORIGINAL_FUNCTION_ENVIRONMENT_AND_CALLER_FRAME', subjects,
            require_current_code_defaults_captures_and_namespace=True)]
        unresolved = []
    elif type(instruction) is P.AccessBinding:
        matched = [row for table, _, row, _ in matches if table == 'names']
        lexical = bool(matched) and all(row.namespace.kind is F.NamespaceKind.LEXICAL for row in matched)
        if instruction.action is P.BindingActionKind.READ and lexical:
            role = 'LEXICAL_BINDING_READ'
            # Scope-selected lexical reads transport references; no contents
            # or alias/value invariance follows from this source fact.
            unresolved = []
        else:
            role = 'BINDING_WRITE_OR_NONLEXICAL_REQUEST'
            guards = [_guard('MATCH_SELECTED_NAMESPACE_AND_RELEASED_OLD_REFERENCE', subjects,
                binding_action=instruction.action.value, original_binding_port=instruction.port)]
    elif type(instruction) is P.ApplyLanguageAction:
        action = instruction.action
        if action is P.LanguageActionKind.ATTRIBUTE_GET:
            attributes = [(row, ref) for table, _, row, ref in matches if table == 'attributes']
            if len(attributes) == 1 and attributes[0][0].action is F.AttributeAction.GET:
                role = 'ORDINARY_RECEIVER_ATTRIBUTE_READ'
                guards = [_guard('MATCH_ORDINARY_ATTRIBUTE_LOOKUP', subjects[:1],
                    attribute=attributes[0][0].name,
                    attribute_fact=encode(attributes[0][1]),
                    getattribute='builtins.object.__getattribute__', getattr_absent=True,
                    alternatives=['INSTANCE_DICTIONARY_VALUE_WITHOUT_DATA_DESCRIPTOR',
                        'STANDARD_PYTHON_FUNCTION_INSTANCE_DESCRIPTOR',
                        'STANDARD_STATICMETHOD_PYTHON_FUNCTION'],
                    reject_custom_descriptors_classmethod_and_class_receiver=True)]
                unresolved = []
        elif action is P.LanguageActionKind.CALL:
            role = 'EXPLICIT_CALL'
            guards = [_guard('MATCH_CURRENT_CALLABLE_ARGUMENT_ABI_AND_EFFECT_SUMMARY', subjects,
                summary_must_cover_original_selected_call_and_normal_return=True,
                attempt_is_not_entry_or_completion=True)]
        elif action in (P.LanguageActionKind.UNARY, P.LanguageActionKind.BINARY,
                        P.LanguageActionKind.COMPARE):
            operation = [row for table, _, row, _ in matches if table == 'operations']
            if len(operation) == 1:
                role = 'PRIMITIVE_SCALAR_OPERATION'
                guards = [_guard('EXACT_BUILTIN_TYPES', subjects, allowed=list(_NUMERIC),
                    original_operator=operation[0].selector.value,
                    operator_operand_order_preserved=True)]
                unresolved = []
        elif action is P.LanguageActionKind.TRUTH:
            role = 'PRIMITIVE_TRUTH_TEST'
            guards = [_guard('EXACT_BUILTIN_TYPES', subjects,
                allowed=list(_NUMERIC)+['builtins.NoneType'])]
            unresolved = []
        elif action in (P.LanguageActionKind.ATTRIBUTE_SET, P.LanguageActionKind.ATTRIBUTE_DELETE):
            role = 'ATTRIBUTE_MUTATION_REQUEST'
        elif action in (P.LanguageActionKind.ITEM_GET, P.LanguageActionKind.ITEM_SET,
                        P.LanguageActionKind.ITEM_DELETE):
            role = 'ITEM_PROTOCOL_REQUEST'
    elif type(instruction) is P.ApplyPythonProtocol:
        protocol = instruction.protocol
        if protocol is K.TRUTH:
            role = 'PRIMITIVE_TRUTH_TEST'
            guards = [_guard('EXACT_BUILTIN_TYPES', subjects,
                allowed=list(_NUMERIC)+['builtins.NoneType'])]
            unresolved = []
        elif protocol is K.FRAME_RELEASE_TEMPORARY:
            role = 'FRAME_RELEASE_TEMPORARY'
            guards = [_guard('EXACT_BUILTIN_TYPES', subjects,
                allowed=list(_NO_RELEASE_CALLBACK),
                purpose='EXCLUDE_FINALIZER_WEAKREF_CALLBACK_AND_REFERENT_RELEASE_EFFECTS')]
            unresolved = []
        elif protocol in (K.FRAME_EXIT, K.FUNCTION_FRAME_EXIT_BINDINGS):
            role = 'FRAME_RELEASE_OWNED_BINDINGS'
            guards = [_guard('ALL_RELEASED_FRAME_REFERENCES_HAVE_CALLBACK_FREE_RELEASE', subjects,
                require_complete_owned_binding_set=True,
                builtin_allowed=list(_NO_RELEASE_CALLBACK))]
            # The source instruction does not itself enumerate all live
            # objects whose last reference the interpreter may release.
            unresolved = ['COMPLETE_FRAME_RELEASE_SET_AND_PROTOCOL_NOT_YET_PROVEN', _UNKNOWN]
        elif protocol in (K.FRAME_ENTER, K.FRAME_BIND, K.FRAME_HOLD_TEMPORARY,
                          K.FRAME_FORWARD_TEMPORARY):
            role = 'FRAME_REFERENCE_OWNERSHIP_PROTOCOL'
        elif protocol is K.CALL_FUNCTION_EX:
            role = 'EXPLICIT_PACKED_CALL_PROTOCOL'
    elif type(instruction) in (P.ReferenceIsNone, P.ReferenceIdentical, P.InvertCondition):
        role, unresolved = 'REFERENCE_IDENTITY_INTRINSIC', []
    elif type(instruction) in (P.DefineLiteral, P.DefineAnnotationString, P.ReadCommonConstant):
        role, unresolved = 'LITERAL_REPRESENTATION', []
    return role, guards, unresolved


def derive_caller_effects(view, *, unit, function_ordinal, resolver=None, flow=None):
    """Cover a source body's normal reachable operations with explicit duties.

    No guard is run, unknown effect is never promoted to purity, and a static
    CALL/return route does not establish a successful dynamic invocation.
    """
    original = view.unit(unit)
    if type(function_ordinal) is not int or not 0 <= function_ordinal < len(original.facts.functions):
        raise ValueError('original function ordinal required')
    function = original.facts.functions[function_ordinal]
    if function.body_scope is None:
        raise ValueError('original function body scope required')
    source, syntax = _source(view, unit, function)
    body = Body(unit, function.template, function.body)
    source_function = source_port(original.facts, function.value)
    if flow is None:
        flow = project_value_flow(original.material, source_function, body)
    elif (type(flow) is not SourceBodyValueFlow or flow.source_function != source_function
            or flow.body != body or flow.source_templates != original.material.result_artifact):
        raise ValueError('shared caller value flow differs from its original function/body/templates')
    resolver = resolver or BodyInterfaceResolver(view)
    slicer = ReferenceSlicer(SimpleNamespace(fact_store=view.catalog.fact_store))
    spans, by_world, names = _fact_indexes(original, function, slicer)
    original_call_ordinals = {reference: ordinal for entries in by_world.values()
        for table, ordinal, _, reference in entries if table == 'calls'}
    indexes = {_key(operation.entry): i for i, operation in enumerate(flow.operations)}
    if len(indexes) != len(flow.operations):
        raise ValueError('duplicate original caller operation identity')

    calls, call_indexes = [], defaultdict(list)
    for record in flow.calls:
        call = record.call
        link = SimpleNamespace(result=source_port(original.facts, call.result),
                               callee=source_port(original.facts, call.callee))
        operation, use, callee = source_call_site(view, link)
        step = view.original(operation)
        index = indexes[(link.result.port.template, step.id)]
        ordinal = len(calls)
        calls.append(dict(fact=encode(record.fact), query_ordinal=original_call_ordinals[record.fact],
            operation=index, original_operation=encode(operation),
            callee_use=encode(use), callee_value=encode(callee), result=encode(link.result),
            call=encode(call), invocation_or_completion_proven=False))
        call_indexes[index].append(ordinal)

    operations = []
    for i, operation in enumerate(flow.operations):
        context = resolver.plan(unit, operation.entry.template)
        step = context.steps[operation.entry.owner]
        if step.instruction != operation.instruction:
            raise ValueError('caller operation differs from its original sealed production')
        before = tuple(context.value(value) for value in step.arguments
            if _state(context.emitter.value(value)[1]))
        after = tuple(context.gp(port.port) for port in operation.outputs
            if port.port.completion is LanguageCompletionKind.FALLTHROUGH and _state(port.value_type))
        matches = list(by_world.get((before[0], after[0]), ())) if len(before) == len(after) == 1 else []
        if type(step.instruction) is P.AccessBinding:
            value = (context.result(step) if step.instruction.action is P.BindingActionKind.READ
                     else context.value(step.arguments[0]) if step.arguments else None)
            matches.extend(names.get(value, ()))
        subjects = tuple(context.value(value) for value in step.arguments
            if not _state(context.emitter.value(value)[1]))
        role, guards, unresolved = _classification(step.instruction, subjects, matches)
        try:
            rule = encode(classify_instruction(step.instruction))
        except TypeError:
            rule = None
            unresolved = list(dict.fromkeys((*unresolved, 'UNSUPPORTED_ORIGINAL_INSTRUCTION_TYPE', _UNKNOWN)))
        try:
            _, span, _ = _metadata(syntax, operation.entry.template.occurrence)
        except KeyError:
            span = None  # Source template identity remains exact without a location attribute.
        operations.append(dict(index=i, original=encode(operation), source_span=span, classification=rule,
            role=role, source_facts=[dict(table=t, ordinal=n, reference=encode(ref))
                for t,n,_,ref in matches], call_sites=call_indexes[i],
            world=dict(before=encode(before), normal_after=encode(after), establishes_value_equality=False),
            guard_specs=guards, obligations=unresolved,
            policy=(_UNKNOWN if unresolved else 'REQUIRES_RUNTIME_GUARDS' if guards
                    else 'SOURCE_REFERENCE_TRANSPORT_OR_INTRINSIC'),
            runtime_guards_checked=False, dynamic_effect_set_proven=False))
    original.check()
    result = dict(schema=SCHEMA, schema_version=1, status='SOURCE_EFFECT_OBLIGATIONS_PROJECTED',
        source=encode(source), unit=unit, function_ordinal=function_ordinal,
        function_fact=encode(slicer.fact(original.facts, 'functions', function_ordinal)),
        source_function=encode(flow.source_function), body=encode(body),
        original_f4=encode(original.material.result_artifact), source_templates=encode(flow.source_templates),
        scope_fact_spans=spans, operations=operations, call_sites=calls,
        control_edges=encode(tuple(edge for edge in flow.edges if edge.kind is G.EdgeKind.CONTROL)),
        declaration_edges=encode(tuple(edge for edge in flow.edges if edge.kind is G.EdgeKind.DECLARE)),
        name_boundaries=encode(flow.name_boundaries),
        guard_spec_contract=dict(check_at='ORIGINAL_SITE_BEFORE_REQUEST',
            evaluation='BUILTIN_RAW_METADATA_OR_EXACT_TYPE_CHECKS_NO_USER_PROTOCOL_EXECUTION',
            unresolved_or_failed_check=_UNKNOWN,
            invalidation_must_precede_unknown_call_and_argument_or_protocol_effect=True,
            guard_implementation_in_this_module=False, guard_specs_are_not_satisfied_guards=True),
        coverage=dict(domain='ORIGINAL_NORMAL_REACHABLE_FUNCTION_BODY_NOT_EXCEPTIONAL_OR_RESUMED_PATHS',
            structural_operation_coverage_complete=len(operations)==len(flow.operations),
            effect_obligations_discharged=False, runtime_enforcement_installed=False,
            no_interstep_writers_proven=False, caller_replacement_authorized=False,
            call_attempt_is_not_invocation_or_completion=True,
            source_world_is_not_logical_value_version=True, source_body_shared_not_unrolled=True),
        summary=dict(operations=len(operations), call_sites=len(calls),
            control_edges=sum(edge.kind is G.EdgeKind.CONTROL for edge in flow.edges),
            roles=dict(sorted(Counter(row['role'] for row in operations).items())),
            guard_specifications=sum(len(row['guard_specs']) for row in operations),
            sites_requiring_invalidation=sum(_UNKNOWN in row['obligations'] for row in operations),
            guards_checked=0, optimization_authorized=False))
    result['id'] = 'session-caller-effects:'+semantic_sha256(result)
    return result


__all__ = ['SCHEMA', 'derive_caller_effects']
