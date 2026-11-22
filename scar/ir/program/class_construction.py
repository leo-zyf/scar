"""Class construction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from scar.ir.program.functions import ArgumentKind
from scar.ir.program.ids import BlockID, OperationID, RegionID, UseID, ValueID
from scar.ir.program.source_subjects import SourceCompilationCodeRef
from scar.ir.v2.common import SourceReference
from scar.ir.v2.ids import OperationDefinitionID
from scar.ir.program.dependencies import (
    DependencyCondition as Condition,
    DependencyEdge as Edge,
    DependencyEvidence as Evidence,
    DependencyKind as K,
    DependencyMeaning as M,
    DependencyPoint as Point,
)
from scar.ir.program.source_references import SourceArtifactLocator


# Original parent-frame class construction ports, not observed activations.
#
# The shared suite reference establishes code provenance only. Entering that
# suite through a prepared mapping remains conditional on the selected builder;
# its cell/None payload is never the parent statement's class result.
def _need(condition, detail):
    if not condition:
        raise ValueError('class construction: ' + detail)


@dataclass(frozen=True, slots=True)
class ImplicitCallArgument:
    """An actual original operand of a call without an ast.Call occurrence."""
    use: UseID
    source: SourceReference
    kind: ArgumentKind
    index: int
    keyword: str | None = None

    def __post_init__(self):
        _need(type(self.use) is UseID and type(self.source) is SourceReference,
              'implicit actual needs its original Use and source')
        _need(type(self.kind) is ArgumentKind and self.kind in (ArgumentKind.POSITIONAL, ArgumentKind.KEYWORD)
              and type(self.index) is int and self.index >= 0,
              'this implicit call profile requires explicit ordered actuals')
        _need((self.kind is ArgumentKind.KEYWORD and type(self.keyword) is str and bool(self.keyword))
              or (self.kind is ArgumentKind.POSITIONAL and self.keyword is None),
              'keyword actual must retain its exact name')


@dataclass(frozen=True, slots=True)
class ImplicitCallPorts:
    """Reference a real INVOKE's operands and paired completions, no CallID."""
    operation: OperationID
    callee: UseID
    arguments: tuple[ImplicitCallArgument, ...]
    world: UseID
    normal: BlockID
    result: ValueID
    normal_world: ValueID
    raised: BlockID
    exception: ValueID
    traceback: ValueID
    raised_world: ValueID
    keyword_names: UseID | None = None

    def __post_init__(self):
        _need(type(self.operation) is OperationID and type(self.callee) is UseID
              and type(self.world) is UseID and type(self.normal) is BlockID
              and type(self.raised) is BlockID,
              'implicit call requires original operation/Use/completion blocks')
        _need(all(type(v) is ValueID for v in (self.result, self.normal_world,
              self.exception, self.traceback, self.raised_world)),
              'implicit completions require original Values')
        _need(type(self.arguments) is tuple and all(type(a) is ImplicitCallArgument for a in self.arguments)
              and tuple(a.index for a in self.arguments) == tuple(range(len(self.arguments))),
              'implicit actuals must retain contiguous operand order')
        _need(self.keyword_names is None or type(self.keyword_names) is UseID,
              'keyword name table must be its original optional Use')
        uses = (self.callee, *(a.use for a in self.arguments),
                *((self.keyword_names,) if self.keyword_names is not None else ()), self.world)
        _need(len(set(uses)) == len(uses), 'different operand occurrences need different Uses')


@dataclass(frozen=True, slots=True)
class ClassDecoratorApplication:
    index: int
    source: SourceReference
    call: ImplicitCallPorts

    def __post_init__(self):
        _need(type(self.index) is int and self.index >= 0
              and type(self.source) is SourceReference and type(self.call) is ImplicitCallPorts,
              'decorator application requires its original ordinal/source/ports')


@dataclass(frozen=True, slots=True)
class ClassConstruction:
    definition: OperationDefinitionID
    source: SourceReference
    region: RegionID
    entry: BlockID
    normal: BlockID
    suite: SourceCompilationCodeRef
    builder_lookup: OperationID
    factory: OperationID
    factory_code: UseID
    factory_globals: UseID
    factory_closure: tuple[UseID, ...]
    body_function: ValueID
    builder_call: ImplicitCallPorts
    decorators: tuple[ClassDecoratorApplication, ...]
    binding: OperationID
    binding_use: UseID
    binding_kind: str
    result: ValueID
    operations: tuple[OperationID, ...]
    suite_entry_condition: str = 'SELECTED_BUILDER_IS_STANDARD_CPYTHON_BUILD_CLASS'
    suite_entry_status: str = 'PARENT_FACTORY_REFERENCE_ONLY_PREPARED_NAMESPACE_UNCONNECTED'

    def __post_init__(self):
        _need(type(self.definition) is OperationDefinitionID and type(self.source) is SourceReference
              and type(self.region) is RegionID and type(self.entry) is BlockID and type(self.normal) is BlockID,
              'construction requires its original class definition and parent frame')
        _need(type(self.suite) is SourceCompilationCodeRef and self.suite.definition == self.definition
              and self.suite.source == self.source and bool(self.suite.code_path),
              'shared suite reference must retain the original class compiler provenance')
        _need(all(type(o) is OperationID for o in (self.builder_lookup, self.factory, self.binding))
              and all(type(u) is UseID for u in (self.factory_code, self.factory_globals, self.binding_use))
              and type(self.factory_closure) is tuple and all(type(u) is UseID for u in self.factory_closure)
              and type(self.body_function) is ValueID and type(self.result) is ValueID,
              'construction endpoints must reference original parent ports')
        _need(type(self.builder_call) is ImplicitCallPorts and type(self.decorators) is tuple
              and all(type(d) is ClassDecoratorApplication for d in self.decorators)
              and tuple(d.index for d in self.decorators) == tuple(reversed(range(len(self.decorators)))),
              'class decorators must retain reverse application order')
        _need(self.binding_kind in ('STORE_NAME', 'STORE_FAST', 'STORE_DEREF', 'STORE_GLOBAL'),
              'class binding needs its compiler-selected frame protocol')
        _need(type(self.operations) is tuple and bool(self.operations)
              and all(type(o) is OperationID for o in self.operations)
              and len(set(self.operations)) == len(self.operations)
              and {self.builder_lookup, self.factory, self.builder_call.operation, self.binding,
                   *(d.call.operation for d in self.decorators)} <= set(self.operations),
              'construction must retain its original operation extent')
        _need(self.suite_entry_condition == 'SELECTED_BUILDER_IS_STANDARD_CPYTHON_BUILD_CLASS'
              and self.suite_entry_status == 'PARENT_FACTORY_REFERENCE_ONLY_PREPARED_NAMESPACE_UNCONNECTED',
              'parent factory provenance cannot claim prepared namespace or suite execution')


CLASS_CONSTRUCTION_PROTOCOLS = frozenset(('LOAD_BUILD_CLASS_314', 'MAKE_CLASS_BODY_314',
    'BUILD_CLASS_CALL_314', 'APPLY_CLASS_DECORATOR_314', 'STORE_CLASS_CELL_314', 'STORE_CLASS_GLOBAL_314'))


# Original class-statement products reaching a source CALL's callable Use.
#
# The final decorator result is a source occurrence, not proof of a runtime
# class, metaclass, selected call implementation or instance allocation.
@dataclass(frozen=True, slots=True)
class ContextualClassStatementSource:
    call: Point
    callable: Point
    source: Point
    construction: ClassConstruction
    locator: SourceArtifactLocator
    path: tuple[Edge, ...]
    evidence: tuple[Evidence, ...]
    conditions: tuple[Condition, ...]
    runtime_class_verified: bool = False

    def __post_init__(self):
        for value, kind in ((self.call, Point), (self.callable, Point),
                (self.source, Point), (self.construction, ClassConstruction),
                (self.locator, SourceArtifactLocator)):
            if type(value) is not kind:
                raise TypeError('class statement source requires original ' + kind.__name__)
            value.__post_init__()
        if (type(self.call.node) is not OperationID or type(self.callable.node) is not UseID
                or type(self.source.node) is not ValueID
                or self.call.namespace != self.callable.namespace
                or self.call.context != self.callable.context
                or self.source.node != self.construction.result):
            raise ValueError('class source must retain the actual CALL, callable Use and final construction result')
        pointer = self.locator.pointer.rsplit('/', 1)
        if (len(pointer) != 2 or pointer[0] not in
                ('#/module/class_constructions', '#/language/class_constructions')
                or not pointer[1].isdigit()
                or replace(self.locator, pointer='#/graph') != self.source.namespace.graph):
            raise ValueError('class source locator must select its original construction table')
        for rows, kind in ((self.path, Edge), (self.evidence, Evidence), (self.conditions, Condition)):
            if type(rows) is not tuple or any(type(row) is not kind for row in rows):
                raise TypeError('class source evidence and paths must be immutable typed records')
        if (not self.path or not self.evidence or self.path[0].source != self.source
                or self.path[-1].target != self.callable):
            raise ValueError('class source needs a continuous final-result to callable path')
        visited = {self.source}
        cursor = self.source
        for edge in self.path:
            if (edge.source != cursor or edge.target in visited
                    or any(type(point.node) not in (UseID, ValueID)
                           for point in (edge.source, edge.target))):
                raise ValueError('class source path is disconnected, cyclic or not a reference-port path')
            if edge.kind is K.VALUE_USE:
                valid = (edge.meaning is M.VALUE and type(edge.source.node) is ValueID
                    and type(edge.target.node) is UseID
                    and edge.source.namespace == edge.target.namespace
                    and edge.source.context == edge.target.context)
            elif edge.kind in (K.BINDING_PAYLOAD, K.BINDING_READ):
                valid = edge.meaning is M.BINDING
            elif edge.kind is K.BLOCK_ARGUMENT:
                valid = edge.meaning in (M.VALUE, M.REFERENCE)
            else:
                valid = (edge.kind in (K.ARGUMENT, K.COMPLETION, K.ENVIRONMENT, K.STATE_VALUE)
                    and edge.meaning is M.REFERENCE)
            if not valid:
                raise ValueError('class source cannot treat computation dependence as callable identity')
            if any(condition not in self.conditions for condition in edge.conditions):
                raise ValueError('class source dropped an original path condition')
            if any(item not in self.evidence for item in edge.evidence):
                raise ValueError('class source dropped original reference-path evidence')
            cursor = edge.target
            visited.add(cursor)
        for ports in (self.construction.builder_call,
                *(row.call for row in self.construction.decorators)):
            subject = Point(self.source.namespace, ports.operation, self.source.context)
            if not any(condition.domain == 'canonical' and condition.name == 'SUCCESSOR_SELECTED'
                    and condition.subject == subject and condition.index is not None
                    and not condition.operands for condition in self.conditions):
                raise ValueError('class source dropped an original builder/decorator NORMAL selection')
        if type(self.runtime_class_verified) is not bool or self.runtime_class_verified:
            raise ValueError('a class statement source cannot assert a runtime class')


__all__ = [
    'ContextualClassStatementSource',
    'ImplicitCallArgument',
    'ImplicitCallPorts',
    'ClassDecoratorApplication',
    'ClassConstruction',
    'CLASS_CONSTRUCTION_PROTOCOLS',
]
