"""Prove when an admitted F4 source assertion may take its omitted arm."""
from operator import eq, ge, gt, le, lt, ne

from scar.analysis.source_productions import _digest
from scar.ir.program import source_productions as P
from scar.ir.program.control import LanguageCompletionKind as C
from scar.ir.record_codec import encode


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
