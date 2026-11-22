"""Source effects."""

from dataclasses import fields
from types import MappingProxyType
from scar.ir.effects import ActionLabel as L, Knowledge as K
from scar.ir.program import (
    source_productions as P,
    state_ports as S,
    normal_graph as G,
    model as M,
)
from scar.ir.program.source_effects import EffectFamily as F, EffectResolution as R, FamilyRule
from scar.analysis.source_namespace_contents import NamespacePlanView
from scar.ir.program.control import LanguageCompletionKind as C
from scar.ir.program.normal_graph import GlobalPort


# Closed mechanical classification of the original F4 instruction types.
#
# Labels denote source capabilities/requests only. This catalog neither reads
# world inputs to infer purity nor resolves functions by names. Unknown types,
# including subclasses of supported records, fail instead of becoming OPAQUE.
def _build():
    rules = {}

    def add(family, labels, resolution, rows):
        for cls, dispatch in rows:
            if cls in rules:
                raise AssertionError('duplicate source effect instruction: '+cls.__name__)
            declared = {field.name for field in fields(cls)}
            if len(dispatch) != len(set(dispatch)) or not set(dispatch) <= declared:
                raise AssertionError('effect dispatch field differs from source record: '+cls.__name__)
            rules[cls] = FamilyRule(cls.__name__, family, labels, resolution, dispatch)

    add(F.REGION_CONTROL, (L.CTRL, L.STATE), R.REGION_COMPOSITION, (
        (P.EvaluateChild, ('purpose',)),
        (P.ApplyRegion, ('invocation',)),
        (P.InvokeTypeParameterFactory, ('invocation', 'environment')),
        (P.BranchControl, ('on_true', 'on_false')),
        (P.LoopControl, ('condition', 'body', 'otherwise')),
        (P.HandleCompletions, ('protected', 'routes')),
        (P.TransferCompletion, ('kind', 'target')),
    ))
    add(F.SUSPENSION, (L.CTRL, L.STATE), R.DYNAMIC_TARGET_OR_PROTOCOL, (
        (P.SuspendExecution, ('suspension', 'routes', 'delegate')),
    ))
    add(F.COMPILATION_INPUT, (L.VAL, L.REP), R.COMPILE_INPUT, (
        (P.ReadCompilationOption, ('option',)),
        (P.ReadCompilationAnnotationIndex, ('parameter',)),
    ))
    add(F.IDENTITY_BOOLEAN, (L.VAL,), R.LOCAL_INTRINSIC, (
        (P.ReferenceIsNone, ()), (P.ReferenceIdentical, ()), (P.InvertCondition, ()),
    ))
    add(F.LITERAL_REPRESENTATION, (L.VAL, L.REP), R.LOCAL_INTRINSIC, (
        (P.DefineLiteral, ('value',)), (P.DefineAnnotationString, ()),
        (P.ReadCommonConstant, ('constant',)),
    ))
    add(F.BINDING_NAMESPACE, (L.STATE,), R.ADDRESS_RULE, (
        (P.AccessBinding, ('action', 'port')),
        (P.ApplyNamespaceAction, ('kind',)),
        (P.BindTypeParameter, ('parameter',)),
        (P.DeleteExceptionBinding, ('port',)),
    ))
    add(F.ENVIRONMENT_ENTRY, (L.STATE, L.CTRL), R.INSTANCE_CONDITION, (
        (P.ReferenceEnvironment, ('source',)),
        (P.EnterRootBody, ('interface',)),
        (P.EnterDeferredBody, ('environment',)),
        (P.EnterClassBody, ('interface',)),
        (P.EnterTypeParameters, ('environment',)),
        (P.EnterEvaluationRequest, ('environment',)),
    ))
    add(F.CLASS_ACCESS, (L.STATE, L.VAL), R.ADDRESS_RULE, (
        (P.AccessClassName, ('action', 'resolution', 'interface')),
        (P.AccessClassProtocol, ('action', 'binding')),
        (P.ReadClassTypeParameters, ('interface',)),
    ))
    add(F.CODE_DECLARATION, (L.REP,), R.DEFERRED_DECLARATION, (
        (P.DeferRegion, ('environment',)), (P.DeferClassBody, ('interface',)),
    ))
    add(F.CALLABLE_CONSTRUCTION, (L.REP, L.MEM, L.STATE), R.INSTANCE_CONDITION, (
        (P.BuildFunctionAnnotations, ('annotations',)),
        (P.BuildFunction, ('environment', 'annotations')),
        (P.BuildLazyExpression, ('expression',)),
        (P.BuildClassAnnotations, ('interface', 'annotations')),
        (P.BuildClassBody, ('interface',)),
        (P.BuildScopeAnnotations, ('interface', 'annotations')),
        (P.BuildTypeAlias, ('expression', 'type_parameters')),
    ))
    add(F.VALUE_TYPE_CONSTRUCTION, (L.VAL, L.REP, L.MEM, L.STATE), R.INSTANCE_CONDITION, (
        (P.ConstructValue, ('construction', 'tuple_layout')),
        (P.CreateTypeParameter, ('parameter',)),
    ))
    add(F.VALUE_TYPE_CONSTRUCTION, (L.VAL, L.REP, L.MEM, L.STATE), R.DYNAMIC_TARGET_OR_PROTOCOL, (
        (P.BuildGenericBase, ()),
    ))
    add(F.TYPE_PARAMETER_UPDATE, (L.STATE,), R.ADDRESS_RULE, (
        (P.SetTypeParameterDefault, ('parameter',)), (P.AttachTypeParameters, ()),
    ))
    add(F.ANNOTATION_STATE, (L.STATE,), R.ADDRESS_RULE, (
        (P.ReadAnnotationRegistration, ('access', 'registration')),
        (P.ReadAnnotationNamespace, ('access',)),
        (P.AddAnnotationIndex, ('publication', 'parameter')),
    ))
    add(F.PYTHON_PROTOCOL, (L.VAL, L.MEM, L.STATE, L.CTRL), R.DYNAMIC_TARGET_OR_PROTOCOL, (
        (P.ApplyLanguageAction, ('action',)),
        (P.ApplyPythonProtocol, ('protocol', 'bindings')),
        (P.ApplyInplaceOperator, ('operator',)),
        (P.ApplyDecorator, ()),
        (P.ApplyImplicitComparison, ('operator',)),
        (P.ApplyPatternAction, ('kind',)),
    ))
    add(F.SPECIAL_METHOD_PROTOCOL, (L.VAL, L.STATE, L.CTRL), R.DYNAMIC_TARGET_OR_PROTOCOL, (
        (P.ApplyDelegationProtocol, ('protocol', 'bindings')),
        (P.ReadContextMethod, ('method',)), (P.InvokeContextMethod, ('method',)),
        (P.ContextAwaitable, ('method',)),
    ))
    add(F.IMPORT_PROTOCOL, (L.REP, L.STATE, L.CTRL, L.IO), R.DYNAMIC_TARGET_OR_PROTOCOL, (
        (P.ImportRequest, ('from_names', 'relative_level')),
        (P.ImportMember, ('component',)), (P.ImportStar, ('environment',)),
    ))
    add(F.EXCEPTION_STATE, (L.STATE, L.CTRL), R.ADDRESS_RULE, (
        (P.EnterHandledException, ('exception_abi',)),
        (P.RestoreHandledException, ()),
        (P.ContextExceptionArguments, ('exception_abi',)),
        (P.PrepareReraise, ('exception_abi',)),
    ))
    add(F.EXCEPTION_PROCESSING, (L.VAL, L.REP, L.MEM, L.STATE, L.CTRL), R.DYNAMIC_TARGET_OR_PROTOCOL, (
        (P.TranslateEscapingException, ('kind',)),
        (P.MatchExceptionType, ('exception_abi',)),
        (P.SplitExceptionGroup, ('exception_abi',)),
        (P.MergeExceptionGroup, ('exception_abi',)),
    ))
    add(F.CLASS_BUILDER_LOOKUP, (L.VAL, L.STATE), R.ADDRESS_RULE, (
        (P.ReadBuildClass, ()),
    ))
    add(F.INTERACTIVE_DISPLAY, (L.IO, L.STATE, L.CTRL), R.DYNAMIC_TARGET_OR_PROTOCOL, (
        (P.DisplayExpression, ()),
    ))
    expected = P.ProductionInstruction.__args__
    if (len(expected) != len(set(expected)) or len(rules) != len(expected)
            or set(rules) != set(expected) or {rule.family for rule in rules.values()} != set(F)):
        raise AssertionError('source effect catalog does not cover the exact original instruction/family domains')
    return MappingProxyType(rules)


_RULES = _build()


def catalog():
    """Return all rules once in the original ProductionInstruction order."""
    expected = P.ProductionInstruction.__args__
    if (len(expected) != len(set(expected)) or set(expected) != set(_RULES)
            or len({rule.instruction for rule in _RULES.values()}) != len(_RULES)):
        raise AssertionError('source instruction coverage or duplicate declarations changed')
    return tuple(_RULES[cls] for cls in expected)


def classify_instruction(ins):
    """Classify only exact supported record types, without executing values."""
    try:
        return _RULES[type(ins)]
    except KeyError:
        raise TypeError('unsupported source instruction type: '+type(ins).__name__) from None


# Project source effects without executing values or decoding other bodies.
#
# Original binding/content access tables refine individual fields when those
# adapters are already attached. Unknown callbacks, physical allocation and
# object contents remain unknown independently of known source accesses.
COLLECTION_FIELDS = ('reads', 'writes', 'allocates', 'frees', 'aliases', 'escapes')
SCALAR_FIELDS = ('rng_effect', 'may_raise', 'external_effect', 'ordering_effect')
_IDENTITY_INTRINSICS = (P.ReferenceIsNone, P.ReferenceIdentical, P.InvertCondition)


def project_effects(source, point):
    source.ensure(point)
    step = source.records[point.node]
    if type(step) is not P.ProductionStep:
        raise TypeError('source effect summaries require an original operation')
    anchor = source.nodes[point.node].anchor
    context = source.effect_contexts.get(anchor.template)
    if context is None:
        context = NamespacePlanView(source.unit, source.plans[anchor.template])
        source.effect_contexts[anchor.template] = context
    rule = classify_instruction(step.instruction)
    def state(type_): return type(type_) is M.ValueType and type_.kind is M.ValueKind.STATE
    inputs = tuple(context.value(value) for value in step.arguments if state(context.emitter.value(value)[1]))
    outputs = tuple(GlobalPort(source.unit, context.emitter.port(step.id, G.PortKind.RESULT, n, completion))
        for completion in step.completions for n, type_ in enumerate(completion.types) if state(type_))
    fields = {name:S.SourceEffectSet() for name in COLLECTION_FIELDS}
    scalar = {name:K.UNKNOWN for name in SCALAR_FIELDS}
    missing = ()
    provider = source.effect_provider
    identification_pending=False
    if provider is not None:
        provider.check(point)
        request = provider.effects(point)
        missing = request.missing_rules
        identification_pending=(point.node in provider._building or point.node in provider._failures)
        accesses = tuple((n, provider.accesses[n]) for n in request.accesses)
        if any(a.operation != point.node for _, a in accesses):
            raise ValueError('source effect access belongs to a different operation')
        read = tuple(n for n, a in accesses if a.kind is S.BindingAccessKind.READ)
        write = tuple(n for n, a in accesses if a.kind in (S.BindingAccessKind.WRITE, S.BindingAccessKind.DELETE))
        content = provider.content_provider
        contents = () if content is None else tuple((n, content.accesses[n])
            for n in content.operation_accesses.get(point.node, ()))
        if any(a.operation != point.node for _, a in contents):
            raise ValueError('source content effect belongs to a different operation')
        reads = tuple(n for n, a in contents if a.kind.value in ('GET', 'UPDATE'))
        writes = tuple(n for n, a in contents if a.kind.value in ('SET', 'DELETE', 'UPDATE'))
        fields['reads'] = S.SourceEffectSet(K.UNKNOWN, read, reads)
        fields['writes'] = S.SourceEffectSet(K.UNKNOWN, write, writes)
        # A direct lexical load has no mapping/descriptor callback. This
        # neither selects an instance nor proves referenced object contents.
        if (not identification_pending and type(step.instruction) is P.AccessBinding and step.instruction.action is P.BindingActionKind.READ
                and len(accesses) == 1 and accesses[0][1].alternatives
                and all(provider.locations[a.location].interface.kind == 'LEXICAL_BINDING'
                    for a in accesses[0][1].alternatives)):
            fields['reads'] = S.SourceEffectSet(K.KNOWN, read, reads)
            fields['writes'] = S.SourceEffectSet(K.NONE)
            for name in ('rng_effect', 'external_effect', 'ordering_effect'): scalar[name] = K.NONE
    elif rule.resolution.value == 'ADDRESS_RULE':
        missing = ('SOURCE_STATE_ADDRESS_PROVIDER_NOT_ATTACHED',)
    if not identification_pending and type(step.instruction) in _IDENTITY_INTRINSICS:
        fields['reads'] = fields['writes'] = S.SourceEffectSet(K.NONE)
        for name in SCALAR_FIELDS: scalar[name] = K.NONE
    elif any(completion.kind is C.RAISE for completion in step.completions):
        # A declared exceptional route is possible, not an observed raise.
        scalar['may_raise'] = K.KNOWN
    unresolved = tuple(name for name in COLLECTION_FIELDS if fields[name].knowledge is K.UNKNOWN) + \
        tuple(name for name in SCALAR_FIELDS if scalar[name] is K.UNKNOWN)
    return S.SourceEffectSummary(point.node, rule, inputs, outputs,
        *(fields[name] for name in COLLECTION_FIELDS), *(scalar[name] for name in SCALAR_FIELDS),
        unresolved, missing)
