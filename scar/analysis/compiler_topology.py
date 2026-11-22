"""C3-T pinned CPython scope creation/attachment, without symbol projection."""
from __future__ import annotations
from dataclasses import replace
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

from scar.analysis.source_contexts import read_source_contexts, SourceContextView, _catalog as _context_catalog
from scar.analysis.source_role_materials import _catalog as _role_catalog
from scar.analysis.source_symbols import read_source_symbols
from scar.ir.program.source_contexts import SourceContextsMaterial
from scar.ir.program.source_syntax import SourceSyntaxMaterial
from scar.ir.program.source_roles import SourceRolesMaterial
from scar.ir.program.compiler_symbols import TopologyFieldRule, TopologyConstructor, TopologyCatalog, BlockOrigin, CompilerBlock, ProjectionStep, SourceScopeProjection, TopologyFrontier, CompilerTopologyInput, CompilerTopologyProfile, CompilerTopologyMaterial
from scar.ir.program.source_paths import SourceSymbolsMaterial
from scar.ir.record_codec import decode, encode

_VERSION = 'CPYTHON_3_14_7_COMPILER_TOPOLOGY_C3T_V1'
_PREVIOUS_IMPLEMENTATIONS = frozenset((
    'f848939ac3d9687b57cc2560634a027267509909296b4b312f827c5942e36af9',
    'f63fc221b2c68da2b41f2567a91c651e768478f8ac9a80c9ac85b8b5432cc07c',
))
_ROLES = ('ROOT', 'FUNCTION_BODY', 'LAMBDA_BODY', 'CLASS_BODY', 'TYPE_PARAMETERS',
    'HEADER_ANNOTATIONS', 'VARIABLE_ANNOTATIONS', 'TYPE_PARAMETER_BOUND',
    'TYPE_PARAMETER_DEFAULT', 'TYPE_ALIAS', 'COMPREHENSION', 'GENERATOR_EXPRESSION')
_PROJECTIONS = ('PUBLIC', 'INLINED', 'FUTURE_DETACHED', 'DETACHED_DESCENDANT')
_ATTACHMENTS = ('DIRECT', 'FUTURE_DETACHED')
_SCOPE_DISPOSITIONS = (*_PROJECTIONS, 'GRAMMAR_ONLY')
_REWRITES = ('FUTURE_DETACH', 'INLINE_COMPREHENSION')
_SPECIAL = {
    'Module': 'ROOT', 'Expression': 'ROOT', 'Interactive': 'ROOT', 'FunctionType': 'SOURCE_GRAMMAR',
    'FunctionDef': 'FUNCTION', 'AsyncFunctionDef': 'FUNCTION', 'ClassDef': 'CLASS',
    'Lambda': 'LAMBDA', 'TypeAlias': 'TYPE_ALIAS', 'AnnAssign': 'VARIABLE_ANNOTATION',
    'TypeVar': 'TYPE_PARAMETER', 'ParamSpec': 'TYPE_PARAMETER', 'TypeVarTuple': 'TYPE_PARAMETER',
    'ListComp': 'COMPREHENSION', 'SetComp': 'COMPREHENSION', 'DictComp': 'COMPREHENSION',
    'GeneratorExp': 'COMPREHENSION', 'arguments': 'ARGUMENT_DESCRIPTOR', 'arg': 'ARG_DESCRIPTOR',
    'comprehension': 'COMPREHENSION_DESCRIPTOR', 'NamedExpr': 'NAMED_EXPR',
}
_OWNER_ROLE = {
    'ROOT': 'ROOT', 'FUNCTION_BODY': 'FUNCTION_BODY', 'LAMBDA_BODY': 'LAMBDA_BODY',
    'CLASS_BODY': 'CLASS_BODY', 'TYPE_PARAMETERS': 'TYPE_PARAMETERS',
    'HEADER_ANNOTATIONS': 'HEADER_ANNOTATIONS', 'VARIABLE_ANNOTATIONS': 'VARIABLE_ANNOTATIONS',
    'TYPE_PARAMETER_BOUND': 'TYPE_PARAMETER_BOUND', 'TYPE_PARAMETER_DEFAULT': 'TYPE_PARAMETER_DEFAULT',
    'TYPE_ALIAS_VALUE': 'TYPE_ALIAS', 'COMPREHENSION': 'COMPREHENSION',
    'GENERATOR_EXPRESSION': 'GENERATOR_EXPRESSION',
}
_FUNCTIONLIKE = frozenset(set(_ROLES) - {'ROOT', 'CLASS_BODY'})
_KIND = {'ROOT': 2, 'FUNCTION_BODY': 0, 'LAMBDA_BODY': 0, 'CLASS_BODY': 1,
    'TYPE_PARAMETERS': 5, 'HEADER_ANNOTATIONS': 3, 'VARIABLE_ANNOTATIONS': 3,
    'TYPE_PARAMETER_BOUND': 6, 'TYPE_PARAMETER_DEFAULT': 6, 'TYPE_ALIAS': 4,
    'COMPREHENSION': 0, 'GENERATOR_EXPRESSION': 0}


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('ascii')).hexdigest()


@lru_cache(maxsize=1)
def _catalog():
    roles, contexts = _role_catalog(), _context_catalog()
    constructors, fields = [], []
    marker_count = child_count = 0
    names = {row.name for row in roles.constructors}
    if not set(_SPECIAL) <= names:
        raise ValueError('official AST constructors differ from the reviewed C3-T dispatch')
    for row in roles.rules:
        child = row.group == 'FIELD' and row.shape in ('NODE', 'OPTIONAL_NODE', 'LIST_NODE')
        if not child:
            dispatch = 'METADATA'
        elif contexts.rules[row.rule].content_policy == 'SYNTAX_MARKER':
            dispatch = 'SYNTAX_MARKER'; marker_count += 1
        else:
            dispatch = 'HANDLER_FIELD' if row.constructor in _SPECIAL else 'ORDERED_VISIT'
        child_count += child
        fields.append(TopologyFieldRule(row.rule, row.constructor, row.group, row.ordinal, row.name, dispatch))
    for row in roles.constructors:
        constructors.append(TopologyConstructor(row.name, _SPECIAL.get(row.name, 'ORDERED_CHILDREN'), row.fields, row.attributes))
    if len(constructors) != 113 or len(fields) != 484 or child_count != 157 or marker_count != 12:
        raise ValueError('C3-T constructor/field/marker catalog is not closed over the reviewed schema')
    payload = dict(rule_version=_VERSION, f2_schema_sha256=roles.f2_schema_sha256,
        context_catalog_sha256=contexts.content_sha256, constructors=encode(tuple(constructors)),
        fields=encode(tuple(fields)), origin_roles=list(_ROLES), projections=list(_PROJECTIONS),
        attachments=list(_ATTACHMENTS), scope_dispositions=list(_SCOPE_DISPOSITIONS), rewrite_rules=list(_REWRITES),
        schema='scar.compiler-topology-catalog', schema_version=1)
    return TopologyCatalog(_VERSION, roles.f2_schema_sha256, contexts.content_sha256,
        tuple(constructors), tuple(fields), _ROLES, _PROJECTIONS, _ATTACHMENTS,
        _SCOPE_DISPOSITIONS, _REWRITES, _digest(payload))


def compiler_topology_catalog() -> dict:
    return encode(_catalog())


class TopologyStop(ValueError):
    def __init__(self, code, origin, detail, *, budget=False):
        self.frontier = TopologyFrontier(code, origin, detail)
        self.budget = budget
        super().__init__(detail)


def _projection_children(blocks):
    """Public child sequences; original attachment edges are never mutated.

    Source construction always assigns a parent before its children. Detached
    roots are not attached; their DIRECT descendants remain in the owned graph.
    """
    result = [()] * len(blocks)
    for index in range(len(blocks) - 1, -1, -1):
        children = []
        for child in blocks[index].children:
            if type(child) is not int or not index < child < len(blocks) or blocks[child].parent != index:
                raise ValueError('compiler attachment does not follow its original construction parent')
            projection = blocks[child].projection
            if projection == 'PUBLIC':
                children.append(child)
            elif projection == 'INLINED':
                children.extend(result[child])
            elif projection not in ('FUTURE_DETACHED', 'DETACHED_DESCENDANT'):
                raise ValueError('undeclared compiler topology projection')
        result[index] = tuple(children)
    return tuple(result)


def _block_metadata(blocks, syntax, contexts, role_view=None):
    """Predict native metadata from original source anchors, never native rows."""
    if role_view is None:
        from scar.analysis.source_role_materials import SyntaxRoleView
        role_view = SyntaxRoleView(syntax)
    attributes = {row.name: { _role_catalog().rules[i].name: _role_catalog().rules[i].ordinal
        for i in row.attributes} for row in _role_catalog().constructors}

    def scalar(occurrence, field):
        value = role_view.field(occurrence, field)
        if value.kind != 'SCALAR':
            raise ValueError('compiler metadata source field is not scalar')
        return value.scalar.to_python()

    def line(occurrence):
        node = syntax.nodes[occurrence]
        value = node.attributes[attributes[node.constructor]['lineno']]
        if value.kind != 'SCALAR' or type(value.scalar.to_python()) is not int:
            raise ValueError('compiler metadata source position is absent')
        return value.scalar.to_python()

    result = []
    for index, block in enumerate(blocks):
        occurrence, role = block.origin.occurrence, block.origin.role
        node = syntax.nodes[occurrence]
        if role == 'ROOT':
            name, lineno = 'top', 0
        elif role in ('HEADER_ANNOTATIONS', 'VARIABLE_ANNOTATIONS'):
            name = '__annotate__'
            if role == 'HEADER_ANNOTATIONS':
                lineno = line(occurrence)
            else:
                scope = contexts.scopes[block.source_scope]
                value = role_view.locate(scope.anchor)[1]
                if value.kind != 'NODE':
                    raise ValueError('variable annotation has no original annotation expression')
                lineno = line(value.reference)
        elif role in ('TYPE_PARAMETER_BOUND', 'TYPE_PARAMETER_DEFAULT'):
            name = scalar(occurrence, 'name')
            value = role_view.field(occurrence, 'bound' if role == 'TYPE_PARAMETER_BOUND' else 'default_value')
            if value.kind != 'NODE':
                raise ValueError('type-parameter evaluator has no original expression')
            lineno = line(value.reference)
        elif role in ('COMPREHENSION', 'GENERATOR_EXPRESSION'):
            name = {'ListComp': 'listcomp', 'SetComp': 'setcomp', 'DictComp': 'dictcomp',
                'GeneratorExp': 'genexpr'}[node.constructor]
            lineno = line(occurrence)
        elif role == 'LAMBDA_BODY':
            name, lineno = 'lambda', line(occurrence)
        else:
            if node.constructor == 'TypeAlias':
                target = role_view.field(occurrence, 'name')
                name = scalar(target.reference, 'id')
            else:
                name = scalar(occurrence, 'name')
            lineno = line(occurrence)
        if block.parent is None:
            nested = 0
        else:
            if not 0 <= block.parent < index:
                raise ValueError('compiler metadata parent is not an original earlier construction')
            nested = int(bool(result[block.parent][3]) or blocks[block.parent].origin.role in _FUNCTIONLIKE)
        result.append((_KIND[role], name, lineno, nested))
    return tuple(result)


class _Builder:
    def __init__(self, syntax, roles, contexts, native, limits):
        self.syntax, self.roles, self.contexts, self.native, self.limits = syntax, roles, contexts, native, limits
        self.view = SourceContextView(syntax, roles, contexts)
        self.fields, self.catalog = self.view.role_view, _catalog()
        self.constructors = {row.name: row for row in self.catalog.constructors}
        self.blocks, self.rewrites, self.block_keys, self.scope_blocks = [], [], {}, {}
        self.scope_origins = {}
        for scope, row in enumerate(contexts.scopes):
            if row.owner.kind not in _OWNER_ROLE:
                raise ValueError('C1 source owner has no reviewed compiler origin role')
            origin = BlockOrigin(row.owner.occurrence, _OWNER_ROLE[row.owner.kind])
            if origin in self.scope_origins:
                raise ValueError('C1 source scopes duplicate one original compiler scope owner')
            self.scope_origins[origin] = scope
        self.current, self.visit_steps = None, 0
        self.future = self._future_annotations()
        self.variable_annotations = {}

    def field(self, occurrence, name):
        return self.fields.field(occurrence, name)

    def tick(self, occurrence, role=None):
        if self.visit_steps >= self.limits[2]:
            origin = BlockOrigin(occurrence, role or 'ROOT')
            raise TopologyStop('VISIT_BUDGET', origin, 'compiler visitor budget reached; no partial projection published', budget=True)
        self.visit_steps += 1

    def rewrite(self, rule, block):
        if len(self.rewrites) >= self.limits[1]:
            raise TopologyStop('REWRITE_BUDGET', self.blocks[block].origin,
                'compiler rewrite budget reached; no partial projection published', budget=True)
        self.rewrites.append(ProjectionStep(rule, block, self.blocks[block].parent))

    def block(self, occurrence, role, *, can_see=False, attach=True):
        self.tick(occurrence, role)
        origin = BlockOrigin(occurrence, role)
        if origin in self.block_keys:
            raise TopologyStop('DUPLICATE_COMPILER_ORIGIN', origin, 'one source origin constructed more than one compiler template')
        if len(self.blocks) >= self.limits[0]:
            raise TopologyStop('BLOCK_BUDGET', origin, 'compiler block budget reached; no partial projection published', budget=True)
        parent = self.current
        attachment = 'FUTURE_DETACHED' if self.future and role in ('HEADER_ANNOTATIONS', 'VARIABLE_ANNOTATIONS') else 'DIRECT'
        hidden_parent = parent is not None and self.blocks[parent].projection in ('FUTURE_DETACHED', 'DETACHED_DESCENDANT')
        projection = 'FUTURE_DETACHED' if attachment == 'FUTURE_DETACHED' else ('DETACHED_DESCENDANT' if hidden_parent else 'PUBLIC')
        source = self.scope_origins.get(origin)
        if source is None and role != 'HEADER_ANNOTATIONS':
            raise TopologyStop('SOURCE_SCOPE_ORIGIN_MISSING', origin, 'compiler template lacks its exact original C1 owner')
        index = len(self.blocks)
        self.block_keys[origin] = index
        self.blocks.append(CompilerBlock(origin, parent, (), source, can_see, attachment, projection, None))
        if source is not None:
            self.scope_blocks[source] = index
        if attachment == 'FUTURE_DETACHED':
            self.rewrite('FUTURE_DETACH', index)
        if attach:
            self.attach(index)
        return index

    def attach(self, block):
        row = self.blocks[block]
        if row.parent is not None and row.attachment == 'DIRECT':
            parent = self.blocks[row.parent]
            if block in parent.children:
                raise ValueError('compiler block attached to its original parent twice')
            self.blocks[row.parent] = replace(parent, children=(*parent.children, block))

    def _future_annotations(self):
        root = self.syntax.nodes[self.syntax.root]
        if root.constructor not in ('Module', 'Interactive'):
            return False
        for i, item in enumerate(self.field(self.syntax.root, 'body').items):
            node = self.syntax.nodes[item.reference]
            if i == 0 and node.constructor == 'Expr':
                value = self.field(item.reference, 'value')
                if value.kind == 'NODE' and self.syntax.nodes[value.reference].constructor == 'Constant':
                    literal = self.field(value.reference, 'value')
                    if type(literal.scalar.to_python()) is str:
                        continue
            if node.constructor != 'ImportFrom':
                break
            module = self.field(item.reference, 'module')
            if (module.kind != 'SCALAR' or module.scalar.to_python() != '__future__'
                    or self.field(item.reference, 'level').scalar.to_python() != 0):
                break
            if any(self.field(alias.reference, 'name').scalar.to_python() == 'annotations'
                    for alias in self.field(item.reference, 'names').items):
                return True
        return False

    def value(self, value):
        if value.kind == 'NODE':
            self.visit(value.reference)
        elif value.kind == 'LIST':
            for item in value.items:
                self.value(item)
        elif value.kind not in ('ABSENT', 'SCALAR'):
            raise ValueError('unreviewed source AST value kind in compiler visitor')

    def visit_field(self, occurrence, name):
        self.value(self.field(occurrence, name))

    def enter(self, block, fields):
        previous, self.current = self.current, block
        try:
            for occurrence, name in fields:
                self.visit_field(occurrence, name)
        finally:
            self.current = previous

    def type_parameters(self, occurrence):
        parameters = self.field(occurrence, 'type_params')
        if parameters.kind != 'LIST':
            raise ValueError('source type-parameter list is not its official list field')
        if not parameters.items:
            return None
        visible = self.blocks[self.current].origin.role == 'CLASS_BODY'
        block = self.block(occurrence, 'TYPE_PARAMETERS', can_see=visible)
        self.enter(block, ((occurrence, 'type_params'),))
        return block

    def defaults(self, arguments):
        self.visit_field(arguments, 'defaults')
        self.visit_field(arguments, 'kw_defaults')

    def header_annotations(self, occurrence, arguments):
        parent = self.blocks[self.current]
        visible = parent.origin.role == 'CLASS_BODY' or parent.can_see_class_scope
        block = self.block(occurrence, 'HEADER_ANNOTATIONS', can_see=visible)
        previous, self.current = self.current, block
        try:
            for name in ('posonlyargs', 'args', 'vararg', 'kwarg', 'kwonlyargs'):
                value = self.field(arguments, name)
                members = value.items if value.kind == 'LIST' else (value,)
                for member in members:
                    if member.kind == 'NODE':
                        self.visit_field(member.reference, 'annotation')
            self.visit_field(occurrence, 'returns')
        finally:
            self.current = previous

    def function(self, occurrence):
        arguments = self.field(occurrence, 'args').reference
        self.defaults(arguments)
        self.visit_field(occurrence, 'decorator_list')
        original = self.current
        parameters = self.type_parameters(occurrence)
        if parameters is not None:
            self.current = parameters
        try:
            body = self.block(occurrence, 'FUNCTION_BODY', attach=False)
            self.header_annotations(occurrence, arguments)
            self.attach(body)
            self.enter(body, ((occurrence, 'body'),))
        finally:
            self.current = original

    def class_definition(self, occurrence):
        self.visit_field(occurrence, 'decorator_list')
        original = self.current
        parameters = self.type_parameters(occurrence)
        if parameters is not None:
            self.current = parameters
        try:
            self.visit_field(occurrence, 'bases')
            self.visit_field(occurrence, 'keywords')
            body = self.block(occurrence, 'CLASS_BODY')
            self.enter(body, ((occurrence, 'body'),))
        finally:
            self.current = original

    def type_alias(self, occurrence):
        self.visit_field(occurrence, 'name')
        original = self.current
        visible = self.blocks[original].origin.role == 'CLASS_BODY'
        parameters = self.type_parameters(occurrence)
        if parameters is not None:
            self.current = parameters
        try:
            body = self.block(occurrence, 'TYPE_ALIAS', can_see=visible)
            self.enter(body, ((occurrence, 'value'),))
        finally:
            self.current = original

    def type_parameter(self, occurrence):
        constructor = self.syntax.nodes[occurrence].constructor
        fields = ('bound', 'default_value') if constructor == 'TypeVar' else ('default_value',)
        for name in fields:
            value = self.field(occurrence, name)
            if value.kind == 'NODE':
                role = 'TYPE_PARAMETER_BOUND' if name == 'bound' else 'TYPE_PARAMETER_DEFAULT'
                block = self.block(occurrence, role, can_see=self.blocks[self.current].can_see_class_scope)
                self.enter(block, ((occurrence, name),))

    def variable_annotation(self, occurrence):
        # Name targets have no nested scopes; other targets are visited first.
        self.visit_field(occurrence, 'target')
        creator = self.current
        if creator not in self.variable_annotations:
            origin = self.blocks[creator].origin.occurrence
            block = self.block(origin, 'VARIABLE_ANNOTATIONS',
                can_see=self.blocks[creator].origin.role == 'CLASS_BODY' and not self.future)
            self.variable_annotations[creator] = block
        self.enter(self.variable_annotations[creator], ((occurrence, 'annotation'),))
        self.visit_field(occurrence, 'value')

    def comprehension(self, occurrence):
        clauses = self.field(occurrence, 'generators').items
        if not clauses:
            raise TopologyStop('COMPREHENSION_WITHOUT_GENERATOR', None, 'source comprehension has no generator')
        first = clauses[0].reference
        self.visit_field(first, 'iter')
        role = 'GENERATOR_EXPRESSION' if self.syntax.nodes[occurrence].constructor == 'GeneratorExp' else 'COMPREHENSION'
        block = self.block(occurrence, role)
        original, self.current = self.current, block
        try:
            self.visit_field(first, 'target')
            self.visit_field(first, 'ifs')
            for clause in clauses[1:]:
                self.visit(clause.reference)
            if self.syntax.nodes[occurrence].constructor == 'DictComp':
                self.visit_field(occurrence, 'value')
                self.visit_field(occurrence, 'key')
            else:
                self.visit_field(occurrence, 'elt')
        finally:
            self.current = original

    def visit(self, occurrence):
        node = self.syntax.nodes[occurrence]
        self.tick(occurrence)
        if node.constructor not in self.constructors:
            raise ValueError('unknown AST constructor has no compiler visitor dispatch')
        constructor = self.constructors[node.constructor]
        handler = constructor.dispatch
        if handler == 'FUNCTION':
            self.function(occurrence)
        elif handler == 'CLASS':
            self.class_definition(occurrence)
        elif handler == 'LAMBDA':
            self.defaults(self.field(occurrence, 'args').reference)
            self.enter(self.block(occurrence, 'LAMBDA_BODY'), ((occurrence, 'body'),))
        elif handler == 'TYPE_ALIAS':
            self.type_alias(occurrence)
        elif handler == 'TYPE_PARAMETER':
            self.type_parameter(occurrence)
        elif handler == 'VARIABLE_ANNOTATION':
            self.variable_annotation(occurrence)
        elif handler == 'COMPREHENSION':
            self.comprehension(occurrence)
        elif handler == 'COMPREHENSION_DESCRIPTOR':
            for name in ('target', 'iter', 'ifs'):
                self.visit_field(occurrence, name)
        elif handler == 'NAMED_EXPR':
            self.visit_field(occurrence, 'value')
            self.visit_field(occurrence, 'target')
        elif handler in ('ARGUMENT_DESCRIPTOR', 'ARG_DESCRIPTOR', 'SOURCE_GRAMMAR'):
            raise TopologyStop('DESCRIPTOR_WITHOUT_COMPILER_OWNER', None,
                'source descriptor reached outside its reviewed enclosing compiler handler')
        elif handler in ('ROOT', 'ORDERED_CHILDREN'):
            for rule_id in constructor.fields:
                rule = self.catalog.fields[rule_id]
                if rule.dispatch in ('ORDERED_VISIT', 'HANDLER_FIELD'):
                    self.value(node.fields[rule.ordinal])
                elif rule.dispatch not in ('METADATA', 'SYNTAX_MARKER'):
                    raise ValueError('unknown compiler field dispatch')
        else:
            raise ValueError('unknown compiler constructor handler')

    def project(self):
        # Only root-attached blocks are analyzed. Future annotation subtrees
        # remain constructed but never undergo public comprehension inlining.
        stack = [(0, False)]
        while stack:
            index, exiting = stack.pop()
            if not exiting:
                stack.append((index, True))
                stack.extend((child, False) for child in reversed(self.blocks[index].children))
            else:
                row = self.blocks[index]
                if (row.origin.role == 'COMPREHENSION' and row.parent is not None
                        and not self.blocks[row.parent].can_see_class_scope):
                    self.blocks[index] = replace(row, projection='INLINED')
                    self.rewrite('INLINE_COMPREHENSION', index)
        public_children = _projection_children(self.blocks)
        metadata = _block_metadata(self.blocks, self.syntax, self.contexts, self.fields)
        mapping, reverse, queue = {0: self.native.root}, {self.native.root: 0}, [0]
        cursor = 0
        while cursor < len(queue):
            block, cursor = queue[cursor], cursor + 1
            native = self.native.scopes[mapping[block]]
            if metadata[block] != (native.type, native.name, native.lineno, native.nested):
                raise TopologyStop('PUBLIC_BLOCK_METADATA_CONTRADICTION', self.blocks[block].origin,
                    'source-derived compiler type/name/line/nested differs from the original public entry')
            children = public_children[block]
            if len(children) != len(native.children):
                raise TopologyStop('PUBLIC_CHILDREN_ARITY_CONTRADICTION', self.blocks[block].origin,
                    'source-derived public children differ in cardinality from the original compiler entry')
            for expected, actual in zip(children, native.children):
                if actual in reverse and reverse[actual] != expected:
                    raise TopologyStop('PUBLIC_BLOCK_ALIAS_CONTRADICTION', self.blocks[expected].origin,
                        'one original public entry corresponds to distinct source compiler origins')
                if expected not in mapping:
                    mapping[expected], reverse[actual] = actual, expected
                    queue.append(expected)
                elif mapping[expected] != actual:
                    raise TopologyStop('PUBLIC_BLOCK_IDENTITY_CONTRADICTION', self.blocks[expected].origin,
                        'one source compiler origin corresponds to distinct public entries')
        if len(reverse) != len(self.native.scopes) or len(mapping) != sum(b.projection == 'PUBLIC' for b in self.blocks):
            raise TopologyStop('PUBLIC_GRAPH_COVERAGE_CONTRADICTION', self.blocks[0].origin,
                'public source templates and original native entries are not a bijection')
        self.blocks = [replace(row, public_scope=mapping.get(i)) for i, row in enumerate(self.blocks)]

    def run(self):
        self.current = self.block(self.syntax.root, 'ROOT')
        self.visit(self.syntax.root)
        if len(self.scope_blocks) != len(self.contexts.scopes):
            missing = next(i for i in range(len(self.contexts.scopes)) if i not in self.scope_blocks)
            scope = self.contexts.scopes[missing]
            raise TopologyStop('SOURCE_SCOPE_COVERAGE_CONTRADICTION',
                BlockOrigin(scope.owner.occurrence, _OWNER_ROLE[scope.owner.kind]),
                'an original C1 source scope was not visited by its compiler creation rule')
        self.project()
        return tuple(SourceScopeProjection(i, self.scope_blocks[i], self.blocks[self.scope_blocks[i]].projection)
            for i in range(len(self.contexts.scopes)))


def _index(index, size, label):
    if type(index) is not int or not 0 <= index < size:
        raise ValueError('C3-T ' + label + ' is outside its original table')
    return index


_ORIGIN_CONSTRUCTORS = {
    'ROOT': ('Module', 'Interactive', 'Expression'),
    'FUNCTION_BODY': ('FunctionDef', 'AsyncFunctionDef'), 'CLASS_BODY': ('ClassDef',),
    'LAMBDA_BODY': ('Lambda',), 'TYPE_PARAMETERS': ('FunctionDef', 'AsyncFunctionDef', 'ClassDef', 'TypeAlias'),
    'HEADER_ANNOTATIONS': ('FunctionDef', 'AsyncFunctionDef'),
    'VARIABLE_ANNOTATIONS': ('Module', 'Interactive', 'FunctionDef', 'AsyncFunctionDef', 'ClassDef'),
    'TYPE_ALIAS': ('TypeAlias',), 'TYPE_PARAMETER_BOUND': ('TypeVar',),
    'TYPE_PARAMETER_DEFAULT': ('TypeVar', 'ParamSpec', 'TypeVarTuple'),
    'COMPREHENSION': ('ListComp', 'SetComp', 'DictComp'),
    'GENERATOR_EXPRESSION': ('GeneratorExp',),
}


class CompilerTopologyView:
    """Queries retain C1 scope/region IDs and original F3-A public row IDs."""

    def __init__(self, roles, c1, native, material, *, contexts=None):
        if (type(roles) is not SourceRolesMaterial or roles.status != 'CLASSIFIED'
                or type(c1) is not SourceContextsMaterial or c1.status != 'CONTEXTUALIZED'
                or type(native) is not SourceSymbolsMaterial
                or type(material) is not CompilerTopologyMaterial
                or material.status not in ('PROJECTED', 'SOURCE_ONLY_MODE')):
            raise ValueError('C3-T queries require typed complete source/topology inputs')
        self.roles, self.contexts, self.native, self.material = roles, c1, native, material
        self.context_view, self.catalog = contexts, _catalog()
        basis = material.input
        if (basis.source_scope_count != len(c1.scopes) or basis.native_scope_count != len(native.scopes)
                or basis.c1_status != c1.status or basis.f3a_status != native.status
                or basis.f3b_document_sha256 != _digest(encode(roles))
                or basis.c1_document_sha256 != _digest(encode(c1))
                or basis.f3a_document_sha256 != _digest(encode(native))
                or any(getattr(basis, field) != getattr(native.input, field)
                    or getattr(basis, field) != getattr(c1.input, field)
                    for field in ('f0_document_sha256', 'f1_document_sha256', 'f2_document_sha256', 'path'))
                or basis.mode != native.mode or basis.mode != c1.input.mode
                or material.profile.native_profile_sha256 != _digest(encode(native.profile))
                or material.profile.topology_catalog_sha256 != self.catalog.content_sha256
                or material.profile.context_catalog_sha256 != c1.profile.context_catalog_sha256
                or material.profile.f2_schema_sha256 != c1.profile.f2_schema_sha256):
            raise ValueError('C3-T source/compiler receipts or versioned catalogs differ')
        if contexts is not None and (contexts.roles != roles or contexts.material != c1):
            raise ValueError('C3-T source context view belongs to another source unit')
        self._owners = {}
        for index, scope in enumerate(c1.scopes):
            if scope.owner in self._owners:
                raise ValueError('C3-T requires a unique source environment per typed C1 owner')
            self._owners[scope.owner] = index

    def block(self, index):
        return self.material.blocks[_index(index, len(self.material.blocks), 'compiler block reference')]

    def scope(self, index):
        return self.material.scopes[_index(index, len(self.material.scopes), 'source scope reference')]

    def region(self, index):
        from scar.ir.program.source_contexts import OwnerRef
        region = self.contexts.regions[_index(index, len(self.contexts.regions), 'source region reference')]
        owner = region.owner
        if owner.kind == 'FORMAL_INTERFACE':
            # A function's parameter interface belongs to its body environment,
            # although the containing arguments slot is evaluated by its creator.
            candidates = [self._owners[key] for kind in ('FUNCTION_BODY', 'LAMBDA_BODY')
                if (key := OwnerRef(owner.occurrence, kind)) in self._owners]
            if len(candidates) != 1:
                raise ValueError('C3-T formal region lacks its unique source function/lambda body')
            scope = candidates[0]
        else:
            if owner not in self._owners:
                raise ValueError('C3-T region has no original source owner environment')
            scope = self._owners[owner]
        return self.scope(scope)

    def port(self, index):
        _index(index, len(self.roles.ports), 'source port reference')
        if self.context_view is None:
            raise ValueError('C3-T port queries require the original SourceContextView')
        return self.scope(self.context_view.context_for_port(index).state.environment)

    def _empty_header(self, occurrence):
        fields = self.context_view.role_view
        if fields.field(occurrence, 'returns').kind == 'NODE':
            return False
        arguments = fields.field(occurrence, 'args').reference
        args = []
        for name in ('posonlyargs', 'args', 'kwonlyargs'):
            args.extend(item.reference for item in fields.field(arguments, name).items)
        for name in ('vararg', 'kwarg'):
            value = fields.field(arguments, name)
            if value.kind == 'NODE':
                args.append(value.reference)
        return all(fields.field(arg, 'annotation').kind != 'NODE' for arg in args)

    def verify(self):
        from scar.analysis.compiler_topology import _projection_children
        data = self.material
        if tuple(row.scope for row in data.scopes) != tuple(range(len(self.contexts.scopes))):
            raise ValueError('C3-T source scope projection does not preserve every original ordinal')
        if data.status == 'SOURCE_ONLY_MODE':
            if data.input.mode != 'func_type' or self.native.status != 'UNSUPPORTED_SYMTABLE_MODE':
                raise ValueError('C3-T grammar-only projection has a different official compiler boundary')
            for index in range(len(self.contexts.regions)):
                self.region(index)
            return dict(blocks=0, scopes=len(data.scopes), regions=len(self.contexts.regions),
                public_blocks=0, public_edges=0, rewrites=0)
        if self.native.status != 'CAPTURED' or not data.blocks:
            raise ValueError('C3-T public projection requires a captured official scope graph')
        if data.blocks[0].parent is not None or data.blocks[0].origin.role != 'ROOT':
            raise ValueError('C3-T original compiler construction graph has a different root')
        origins, child_refs, public_refs, expected_rewrites = set(), [[] for _ in data.blocks], {}, set()
        detached = [False] * len(data.blocks)
        for index, block in enumerate(data.blocks):
            origin = block.origin
            key = origin.occurrence, origin.role
            if key in origins or origin.role not in self.catalog.origin_roles:
                raise ValueError('C3-T compiler origin is duplicated or has an undeclared role')
            origins.add(key)
            if self.context_view is not None:
                syntax = self.context_view.syntax
                _index(origin.occurrence, len(syntax.nodes), 'source occurrence')
                if syntax.nodes[origin.occurrence].constructor not in _ORIGIN_CONSTRUCTORS[origin.role]:
                    raise ValueError('C3-T compiler origin role does not match its original F2 constructor')
            if (block.attachment not in self.catalog.attachments or block.projection not in self.catalog.projections
                    or type(block.can_see_class_scope) is not bool):
                raise ValueError('C3-T compiler block has an undeclared attachment/projection policy')
            if index:
                _index(block.parent, index, 'previously created compiler parent')
            elif block.attachment != 'DIRECT' or block.projection != 'PUBLIC':
                raise ValueError('C3-T public root cannot be detached or inlined')
            if block.parent is not None:
                detached[index] = detached[block.parent]
            if block.attachment == 'FUTURE_DETACHED':
                if origin.role not in ('HEADER_ANNOTATIONS', 'VARIABLE_ANNOTATIONS'):
                    raise ValueError('C3-T future detachment requires an annotation origin')
                detached[index] = True
                expected = 'FUTURE_DETACHED'
                expected_rewrites.add(('FUTURE_DETACH', index, block.parent))
            elif detached[index]:
                expected = 'DETACHED_DESCENDANT'
            elif origin.role == 'COMPREHENSION' and not self.block(block.parent).can_see_class_scope:
                expected = 'INLINED'
                expected_rewrites.add(('INLINE_COMPREHENSION', index, block.parent))
            else:
                expected = 'PUBLIC'
            if block.projection != expected:
                raise ValueError('C3-T projection ignores the original reachable-parent/annotation rule')
            if block.projection == 'PUBLIC':
                native = _index(block.public_scope, len(self.native.scopes), 'original public compiler row')
                if native in public_refs:
                    raise ValueError('C3-T public row has multiple compiler-template identities')
                public_refs[native] = index
            elif block.public_scope is not None:
                raise ValueError('C3-T elided compiler template cannot claim an observed public row')
            if len(set(block.children)) != len(block.children):
                raise ValueError('C3-T original child attachment is duplicated')
            for child in block.children:
                _index(child, len(data.blocks), 'compiler child')
                if child <= index or self.block(child).parent != index or self.block(child).attachment != 'DIRECT':
                    raise ValueError('C3-T child was not attached to its original construction parent')
                child_refs[child].append(index)
            if block.source_scope is None:
                if origin.role != 'HEADER_ANNOTATIONS' or (self.context_view is not None and not self._empty_header(origin.occurrence)):
                    raise ValueError('C3-T source-less compiler block lacks an empty header origin')
            else:
                source = self.contexts.scopes[_index(block.source_scope, len(self.contexts.scopes), 'original source scope')]
                if (source.owner.occurrence, _OWNER_ROLE.get(source.owner.kind)) != key:
                    raise ValueError('C3-T block is not the role of its original typed source owner')
        for index, block in enumerate(data.blocks):
            expected = [] if index == 0 or block.attachment == 'FUTURE_DETACHED' else [block.parent]
            if child_refs[index] != expected:
                raise ValueError('C3-T original parent/child attachment inverse is incomplete')
        for row in data.scopes:
            block = self.block(row.block)
            if block.source_scope != row.scope or row.disposition != block.projection:
                raise ValueError('C3-T source projection does not round-trip to its compiler template')
            source = self.contexts.scopes[row.scope]
            parent = None if source.parent is None else self.scope(source.parent).block
            if block.parent != parent:
                raise ValueError('C3-T source parent does not match the original compiler environment')
        actual_rewrites = {(row.rule, row.block, row.parent) for row in data.rewrites}
        if len(actual_rewrites) != len(data.rewrites) or actual_rewrites != expected_rewrites:
            raise ValueError('C3-T rewrite witnesses are missing, duplicated or unsupported')
        if set(public_refs) != set(range(len(self.native.scopes))) or data.blocks[0].public_scope != self.native.root:
            raise ValueError('C3-T public correspondence does not cover the original compiler rows exactly once')
        children, edges = _projection_children(data.blocks), 0
        for native, index in public_refs.items():
            projected = tuple(self.block(child).public_scope for child in children[index])
            if projected != self.native.scopes[native].children:
                raise ValueError('C3-T original public child edges differ from the witnessed splice result')
            edges += len(projected)
        for index in range(len(self.contexts.regions)):
            self.region(index)
        return dict(blocks=len(data.blocks), scopes=len(data.scopes), regions=len(self.contexts.regions),
            public_blocks=len(public_refs), public_edges=edges, rewrites=len(data.rewrites))


def _profile(native):
    runtime = native.profile.runtime
    expected_types = {
        'TYPE_FUNCTION': _KIND['FUNCTION_BODY'], 'TYPE_CLASS': _KIND['CLASS_BODY'],
        'TYPE_MODULE': _KIND['ROOT'], 'TYPE_ANNOTATION': _KIND['HEADER_ANNOTATIONS'],
        'TYPE_TYPE_ALIAS': _KIND['TYPE_ALIAS'], 'TYPE_TYPE_PARAMETERS': _KIND['TYPE_PARAMETERS'],
        'TYPE_TYPE_VARIABLE': _KIND['TYPE_PARAMETER_BOUND'],
    }
    constants = {row.name: row.value for row in native.profile.constants}
    if (runtime.implementation != 'cpython' or runtime.version[:3] != (3, 14, 7)
            or runtime.implementation_version[:3] != (3, 14, 7)
            or any(constants.get(name) != value for name, value in expected_types.items())):
        raise ValueError('C3-T requires the reviewed CPython 3.14.7 compiler scope profile/type constants')
    from scar.ir.program import compiler_symbols as topology_ir
    catalog = _catalog()
    owner = sha256(Path(__file__).read_bytes()).hexdigest()
    implementation = _digest(dict(analysis=owner, view=owner,
        ir=sha256(Path(topology_ir.__file__).read_bytes()).hexdigest()))
    return CompilerTopologyProfile(_VERSION, catalog.f2_schema_sha256, catalog.context_catalog_sha256,
        catalog.content_sha256, _digest(encode(native.profile)), implementation)


def _derive(f0, f1, f2, f3b, c1, f3a, limits):
    if any(type(n) is not int or n < 0 for n in limits):
        raise ValueError('compiler topology budgets must be exact nonnegative integers')
    contexts = read_source_contexts(c1, f0, f1, f2, f3b)
    native = read_source_symbols(f3a, f0, f1, f2)
    syntax, roles = decode(SourceSyntaxMaterial, f2), decode(SourceRolesMaterial, f3b)
    return _project_topology(f0, f1, f2, f3b, c1, f3a, syntax, roles, contexts, native, limits)


def _project_topology(f0, f1, f2, f3b, c1, f3a, syntax, roles, contexts, native,
                      limits=(100_000, 200_000, 4_000_000)):
    inputs = CompilerTopologyInput(*(_digest(doc) for doc in (f0, f1, f2, f3b, c1, f3a)),
        f0['path'], syntax.mode, contexts.status, native.status, len(contexts.scopes), len(native.scopes))
    base = dict(input=inputs, profile=_profile(native), max_blocks=limits[0],
        max_rewrites=limits[1], max_visit_steps=limits[2])
    empty = dict(blocks=(), scopes=(), rewrites=(), frontiers=(), visit_steps=0)
    if contexts.status != 'CONTEXTUALIZED':
        empty['frontiers'] = (TopologyFrontier('UPSTREAM_NOT_CONTEXTUALIZED', None,
            'C1 has no complete source context templates'),)
        return CompilerTopologyMaterial(**base, status='UPSTREAM_NOT_CONTEXTUALIZED', **empty)
    if syntax.mode == 'func_type':
        empty['scopes'] = tuple(SourceScopeProjection(i, None, 'GRAMMAR_ONLY') for i in range(len(contexts.scopes)))
        material = CompilerTopologyMaterial(**base, status='SOURCE_ONLY_MODE', **empty)
        CompilerTopologyView(roles, contexts, native, material,
            contexts=SourceContextView(syntax, roles, contexts)).verify()
        return material
    if native.status != 'CAPTURED':
        empty['frontiers'] = (TopologyFrontier('UPSTREAM_NOT_CAPTURED', None,
            'F3-A has no complete public compiler scope graph: ' + native.status),)
        return CompilerTopologyMaterial(**base, status='UPSTREAM_NOT_CAPTURED', **empty)
    builder = None
    try:
        builder = _Builder(syntax, roles, contexts, native, limits)
        scopes = builder.run()
    except TopologyStop as error:
        empty['frontiers'] = (error.frontier,)
        empty['visit_steps'] = 0 if builder is None else builder.visit_steps
        return CompilerTopologyMaterial(**base,
            status='BUDGET_STOPPED' if error.budget else 'PROJECTION_CONTRADICTION', **empty)
    material = CompilerTopologyMaterial(**base, status='PROJECTED', blocks=tuple(builder.blocks),
        scopes=scopes, rewrites=tuple(builder.rewrites), frontiers=(), visit_steps=builder.visit_steps)
    CompilerTopologyView(roles, contexts, native, material, contexts=builder.view).verify()
    return material


def compiler_topology_document(f0, f1, f2, f3b, c1, f3a, *, max_blocks=100_000,
        max_rewrites=200_000, max_visit_steps=4_000_000) -> dict:
    return encode(_derive(f0, f1, f2, f3b, c1, f3a, (max_blocks, max_rewrites, max_visit_steps)))


def read_compiler_topology(document, f0, f1, f2, f3b, c1, f3a) -> CompilerTopologyMaterial:
    native = decode(SourceSymbolsMaterial, f3a)
    return _verify_topology(document, native, lambda limits: _derive(f0, f1, f2, f3b, c1, f3a, limits))


def _verify_topology(document, native, rederive):
    material = decode(CompilerTopologyMaterial, document)
    profile = _profile(native)
    if material.profile.implementation_sha256 in _PREVIOUS_IMPLEMENTATIONS:
        profile = replace(profile, implementation_sha256=material.profile.implementation_sha256)
    if material.profile != profile:
        raise ValueError('C3-T compiler topology implementation/catalog profile mismatch')
    limits = material.max_blocks, material.max_rewrites, material.max_visit_steps
    if any(type(n) is not int or n < 0 for n in limits):
        raise ValueError('compiler topology budgets must be exact nonnegative integers')
    fresh = rederive(limits)
    if material != replace(fresh, profile=profile):
        raise ValueError('C3-T topology differs from original source contexts/public compiler entries')
    return material


__all__ = ['compiler_topology_catalog', 'compiler_topology_document',
    'read_compiler_topology', 'CompilerTopologyView']
