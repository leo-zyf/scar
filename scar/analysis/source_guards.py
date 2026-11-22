"""Source guards."""

from __future__ import annotations

from operator import eq, ge, gt, le, lt, ne
from scar.analysis.source_productions import _digest
from scar.ir.program import source_productions as P
from scar.ir.program.control import LanguageCompletionKind as C
from scar.ir.record_codec import encode
import ast
from scar.ir.program.model import (
    ExitKind,
    StateVersionValue,
    TerminatorKind,
    ValueKind,
    ValueType,
)
from scar.ir.program.structured import PythonProtocolKind as K


# Prove when an admitted F4 source assertion may take its omitted arm.
PRESERVE = 'preserve'
OMIT_PROVEN_DEBUG = 'omit-proven-debug'
ASSERTION_POLICIES = (PRESERVE, OMIT_PROVEN_DEBUG)

_LITERAL_TYPES = (type(None), bool, int, float, complex, str, bytes, type(Ellipsis))
_ORDERED_TYPES = (bool, int, float, str, bytes)
_EQUALITY = {'Eq': eq, 'NotEq': ne}
_ORDERING = {'Lt': lt, 'LtE': le, 'Gt': gt, 'GtE': ge}
_UNKNOWN_TRUTH = object()


class AssertionEffectUnproven(ValueError):
    pass


def validate_assertion_policy(policy):
    if type(policy) is not str or policy not in ASSERTION_POLICIES:
        raise ValueError('unsupported source assertion policy')


class SourceAssertions:
    def __init__(self, syntax):
        self.syntax = syntax
        self.fields = {row.name: row.fields for row in syntax.profile.ast_schema}
        self.f2_digest = _digest(encode(syntax))

    def node(self, occurrence):
        node = self.syntax.nodes[occurrence]
        return node.constructor, dict(zip(self.fields[node.constructor], node.fields))

    def operand(self, occurrence, read_local):
        kind, fields = self.node(occurrence)
        if kind == 'Constant' and fields['value'].kind == 'SCALAR':
            value = fields['value'].scalar.to_python()
            if type(value) in _LITERAL_TYPES:
                return value
        if kind == 'Name':
            return read_local(occurrence)
        raise AssertionEffectUnproven(f'{kind} has no effect-free operand proof')

    def truth(self, occurrence, read_local):
        kind, fields = self.node(occurrence)
        if kind in ('Constant', 'Name'):
            value = self.operand(occurrence, read_local)
            if type(value) in _LITERAL_TYPES:
                return bool(value)
        if kind == 'UnaryOp' and self.node(fields['op'].reference)[0] == 'Not':
            answer = self.truth(fields['operand'].reference, read_local)
            return _UNKNOWN_TRUTH if answer is _UNKNOWN_TRUTH else not answer
        if kind == 'BoolOp':
            stop = self.node(fields['op'].reference)[0] == 'Or'
            unknown = False
            for value in fields['values'].items:
                result = self.truth(value.reference, read_local)
                if result is stop:
                    return stop
                unknown |= result is _UNKNOWN_TRUTH
            return _UNKNOWN_TRUTH if unknown else not stop
        if kind == 'Compare' and len(fields['ops'].items) == 1:
            op = self.node(fields['ops'].items[0].reference)[0]
            first = self.operand(fields['left'].reference, read_local)
            second = self.operand(fields['comparators'].items[0].reference, read_local)
            if op in ('Is', 'IsNot'):
                # Compiler interning and dynamic inputs do not identify two
                # references. Identity comparisons still cannot dispatch.
                return _UNKNOWN_TRUTH
            if type(first) is type(second):
                if op in _EQUALITY and type(first) in _LITERAL_TYPES:
                    return _EQUALITY[op](first, second)
                if op in _ORDERING and type(first) in _ORDERED_TYPES:
                    return _ORDERING[op](first, second)
        raise AssertionEffectUnproven(f'{kind} has no non-throwing truth proof')

    def prove_omission(self, context, region_id, read_local):
        plan = context.plan
        source = plan.source
        if (region_id != plan.root or plan.purpose is not P.ProductionPurpose.STATEMENT
                or type(source) is not P.ProductionOccurrence
                or self.syntax.nodes[source.occurrence].constructor != 'Assert'):
            return None
        if plan.input.path != self.syntax.input.path or plan.input.f2_sha256 != self.f2_digest:
            raise AssertionEffectUnproven('F4 assertion does not refer to the admitted F2 syntax')
        root = context.regions[region_id]
        actions = tuple(context.steps[step] for step in root.steps)
        if (len(actions) != 3 or len(root.parameters) != 1 or actions[0].source != source
                or actions[0].instruction != P.ReadCompilationOption(P.CompilationOption.ASSERTIONS_ENABLED)
                or actions[0].arguments or type(actions[1].instruction) is not P.BranchControl
                or actions[1].arguments != (P.ProductionResult(actions[0].id, 0),)
                or actions[2].instruction != P.TransferCompletion(C.FALLTHROUGH)
                or actions[2].arguments != (P.ProductionResult(actions[1].id, 0),)):
            raise AssertionEffectUnproven('assertion lacks its original compilation-option branch')
        branch = actions[1].instruction
        checked = context.regions[branch.on_true.region]
        omitted = context.regions[branch.on_false.region]
        state = (P.ProductionParameter(region_id, 0),)
        omitted_steps = tuple(context.steps[step] for step in omitted.steps)
        if (checked.id == omitted.id or checked.parent != region_id or omitted.parent != region_id
                or branch.on_true.arguments != state or branch.on_false.arguments != state
                or omitted.parameters != root.parameters or len(omitted_steps) != 1
                or omitted_steps[0].instruction != P.TransferCompletion(C.FALLTHROUGH)
                or omitted_steps[0].arguments != (P.ProductionParameter(omitted.id, 0),)
                or omitted.completions != (P.ProductionCompletion(C.FALLTHROUGH, root.parameters),)):
            raise AssertionEffectUnproven('original omission arm does not pass through its input world')

        _, fields = self.node(source.occurrence)
        decision = self.truth(fields['test'].reference, read_local)
        may_fail = decision is not True
        message = fields['msg']
        if message.kind == 'NODE' and may_fail:
            self.operand(message.reference, read_local)
        return omitted.id, dict(kind='source_assertion_omission', policy=OMIT_PROVEN_DEBUG,
            unit=context.unit, occurrence=source.occurrence, f2_sha256=self.f2_digest,
            test_occurrence=fields['test'].reference, message_occurrence=message.reference,
            raw_sha256=self.syntax.input.raw_sha256, option_step=actions[0].id.wire,
            control_step=actions[1].id.wire, checked_region=checked.id.wire,
            omitted_region=omitted.id.wire, may_assertion_fail=may_fail,
            test_effects='NO_OBSERVABLE_EFFECTS_PROVEN',
            message_effects=('ABSENT' if message.kind != 'NODE' else
                'UNREACHABLE' if not may_fail else 'NO_OBSERVABLE_EFFECTS_PROVEN'),
            allowed_difference='ASSERTION_FAILURE_OMITTED' if may_fail else 'NONE')


# Module assertions retain source branches and real propagation-pair ports.
def _need(condition, detail):
    if not condition:
        raise ValueError('module assertion: ' + detail)


def validate_module_assertion(node,expansion,*,context,code_evidence,profile,overlay,
        blocks,operations,values,protocols,invoke,expression_value,source_reference,
        exception_holds,ref_type,state_type):
    from scar.analysis.source_module_lowering import mapping_assertion_supported
    _need(mapping_assertion_supported(context,code_evidence,node,profile=profile),
        'the original compiler did not retain this supported assertion layout')
    source=source_reference(node)
    allowed=set(expansion.operations)
    links={row.operation:row for row in overlay.operation_links}
    use_links={(row.operation,row.use):row for row in overlay.use_links}

    def marked(kind,reference):
        rows=[operations[row.operation] for row in overlay.protocols
            if row.kind is kind and row.source==reference]
        _need(len(rows)==1 and rows[0].id in allowed,'missing/extra assertion primitive '+kind.value)
        _need(links[rows[0].id].semantic_operation is None and rows[0].callee is None,
            'assertion primitive acquired a guessed source callee')
        return rows[0]

    def corridor(start,stop):
        cursor,seen,path=start,set(),[]
        while cursor not in seen:
            seen.add(cursor)
            local=[operations[identity] for identity in blocks[cursor].operations if identity in allowed]
            _need(local,'assertion corridor escaped its original statement')
            for operation in local:
                path.append(operation)
                if operation.id==stop:
                    state=None
                    for action in path:
                        worlds=[use.value for use in action.uses if type(values[use.value]) is StateVersionValue]
                        if worlds:
                            state=worlds[0] if state is None else state
                            _need(all(value==state for value in worlds),'action bypassed preceding assertion world')
                        term=action.terminator
                        if term is not None and term.kind is TerminatorKind.INVOKE:
                            edge,=(edge for edge in term.successors if edge.kind is ExitKind.NORMAL)
                            state=blocks[edge.block].arguments[-1]
                        else:
                            states=[value for value in action.results if type(values[value]) is StateVersionValue]
                            if states:
                                _need(len(states)==1,'assertion action has ambiguous state results')
                                state=states[0]
                    return path
            term=local[-1].terminator
            edges=([edge for edge in term.successors if edge.kind is ExitKind.NORMAL]
                if term is not None and term.kind is TerminatorKind.INVOKE else list(term.successors)
                if term is not None and term.kind is TerminatorKind.JUMP else [])
            _need(len(edges)==1,'assertion corridor has no unique normal continuation')
            cursor=edges[0].block
        raise ValueError('module assertion: normal corridor cycles')

    truth=marked(K.TRUTH,source_reference(node.test))
    _need(len(truth.uses)==2 and truth.uses[0].value==expression_value(node.test),
        'test must feed exactly one truth conversion')
    truth_normal,_=invoke(truth,(ValueType(ValueKind.DATA,'builtins.bool'),state_type))
    corridor(expansion.entry,truth.id)
    branch=operations[truth_normal.operations[-1]]
    _need(branch.id in allowed and branch.terminator is not None
        and branch.terminator.kind is TerminatorKind.BRANCH and len(branch.uses)==3
        and branch.terminator.condition==branch.uses[0].id
        and branch.uses[0].value==truth_normal.arguments[0]
        and all(use.value==truth_normal.arguments[-1] for use in branch.uses[1:])
        and len(branch.terminator.successors)==2,
        'assertion lost its test-dependent branch and post-truth world')
    yes,no=branch.terminator.successors
    _need(yes.block==expansion.normal and yes.arguments==(branch.uses[1].id,)
        and no.arguments==(branch.uses[2].id,), 'assertion success/failure arms are swapped or disconnected')
    constructor=marked(K.MAKE_ASSERTION_ERROR,source)
    message=None if node.msg is None else expression_value(node.msg)
    _need(tuple(use.value for use in constructor.uses[:-1])==(() if node.msg is None else (message,)),
        'AssertionError must use exactly the failed-arm source message')
    constructed,construction_failure=invoke(constructor,(ref_type,state_type),memory=True)
    failed_path=corridor(no.block,constructor.id)
    normalizer=marked(K.FUNCTION_START_PROPAGATION,source)
    _need(len(normalizer.uses)==2 and normalizer.uses[0].value==constructed.arguments[0],
        'propagation must consume the newly constructed assertion exception')
    _,normalization_failure=invoke(normalizer,(ref_type,ref_type,state_type))
    normalized_path=corridor(constructed.id,normalizer.id)
    if node.msg is not None:
        held=[operation for operation in failed_path if operation.id in protocols
            and protocols[operation.id].kind is K.FRAME_HOLD_TEMPORARY
            and operation.uses[0].value==message
            and use_links[operation.id,operation.uses[0].id].role=='statement-held-reference']
        released=[operation for operation in normalized_path if operation.id in protocols
            and protocols[operation.id].kind is K.FRAME_RELEASE_TEMPORARY
            and operation.uses[0].value==message
            and use_links[operation.id,operation.uses[0].id].role=='statement-released-reference']
        _need(len(held)==len(released)==1,'message stack reference must be held then released exactly once')
        for block,expected in ((construction_failure.id,(message,)),(normalization_failure.id,())):
            pending=exception_holds(block,operations,protocols,blocks,use_links)
            _need(tuple(value for value in pending if value==message)==expected,
                'message retention differs between construction failure and post-release propagation')
    reachable,queue=set(),[no.block]
    while queue:
        block=queue.pop()
        _need(block!=expansion.normal,'failed assertion reached its success continuation')
        for identity in blocks[block].operations:
            if identity not in allowed or identity in reachable:
                continue
            reachable.add(identity)
            term=operations[identity].terminator
            if term is not None:
                queue.extend(edge.block for edge in term.successors
                    if any(child in allowed for child in blocks[edge.block].operations)
                    or edge.block==expansion.normal)
    if node.msg is not None:
        message_sources={source_reference(child) for child in ast.walk(node.msg) if isinstance(child,ast.expr)}
        _need(all(identity in reachable for identity in allowed if links[identity].source in message_sources),
            'message has operations outside the assertion failure arm')
