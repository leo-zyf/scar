"""Source contexts."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
from scar.analysis.source_role_materials import (
    SyntaxRoleView,
    read_source_roles,
    _catalog as _role_catalog,
    _state,
)
from scar.ir.program.source_roles import SourceRolesMaterial, RolePort, SlotRef
from scar.ir.program.source_syntax import SourceSyntaxMaterial
from scar.ir.program.source_contexts import (
    ContextBoundary,
    ContextCatalog,
    ContextFrontier,
    ContextResolution,
    ContextRule,
    ContextState,
    EvaluationRegion,
    OwnerRef,
    PrivateContext,
    RuleContextCount,
    SourceContextsInput,
    SourceContextsMaterial,
    SourceContextsProfile,
    SourceScope,
)
from scar.ir.record_codec import decode, encode


# Sparse C1 context queries over original, unchanged F2 parent slots.
#
# This view checks record references and total query coverage. Source semantics
# are granted by the fresh C1 reader; constructing a view is not that proof.
def _index(value, size, label):
    if type(value) is not int or not 0 <= value < size:
        raise ValueError('C1 ' + label + ' reference is outside its original table')
    return value


def _acyclic(rows, root, label):
    """Validate a single-root parent forest iteratively, independent of ID order."""
    if [i for i, row in enumerate(rows) if row.parent is None] != [root]:
        raise ValueError('C1 ' + label + ' has a different or additional root')
    done = {root}
    for start in range(len(rows)):
        pending, active, current = [], set(), start
        while current not in done:
            if current in active:
                raise ValueError('cyclic C1 ' + label + ' parent relation')
            active.add(current)
            pending.append(current)
            current = _index(rows[current].parent, len(rows), label + ' parent')
        done.update(pending)


class SourceContextView:
    """One reusable parent index and sparse boundary map per source unit."""

    def __init__(self, syntax, roles, material):
        from scar.analysis.source_contexts import _catalog
        if (type(syntax) is not SourceSyntaxMaterial or syntax.status != 'PARSED'
                or type(roles) is not SourceRolesMaterial or roles.status != 'CLASSIFIED'
                or type(material) is not SourceContextsMaterial or material.status != 'CONTEXTUALIZED'):
            raise ValueError('context queries require typed PARSED/CLASSIFIED/CONTEXTUALIZED materials')
        self.syntax, self.roles, self.material = syntax, roles, material
        self.role_view, self.catalog = SyntaxRoleView(syntax), _catalog()
        if (material.profile.context_catalog_sha256 != self.catalog.content_sha256
                or material.profile.role_catalog_sha256 != self.role_view.catalog.content_sha256
                or material.profile.f2_schema_sha256 != self.role_view.catalog.f2_schema_sha256
                or material.input.f2_document_sha256 != _digest(encode(syntax))
                or material.input.f3b_document_sha256 != _digest(encode(roles))
                or roles.input.f2_document_sha256 != material.input.f2_document_sha256):
            raise ValueError('C1 references a different original syntax/role material or catalog')
        root = syntax.root
        parents = self.role_view.parents
        if root in parents or len(parents) != len(syntax.nodes) - 1:
            raise ValueError('C1 requires one original F2 occurrence root')
        # F2 normally grants this invariant; reject disconnected cycles even
        # for direct typed/synthetic callers before querying parent chains.
        seen, children = set(), {}
        for child, site in parents.items():
            children.setdefault(site.occurrence, []).append(child)
        pending = [root]
        while pending:
            occurrence = pending.pop()
            if occurrence in seen:
                raise ValueError('cyclic original F2 parent relation')
            seen.add(occurrence)
            pending.extend(children.get(occurrence, ()))
        if len(seen) != len(syntax.nodes):
            raise ValueError('disconnected original F2 occurrence graph')
        for scope in material.scopes:
            self._owner(scope.owner)
            self._anchor(scope.anchor)
        for region in material.regions:
            self._owner(region.owner)
            self._anchor(region.anchor)
            self._anchor(region.suppression_origin)
        for private in material.private_contexts:
            if private.class_owner is not None:
                self._owner(private.class_owner)
                if syntax.nodes[private.class_owner.occurrence].constructor != 'ClassDef':
                    raise ValueError('C1 private context owner is not an original class declaration')
            if private.parameter_group is not None:
                group = self.scope(private.parameter_group)
                if group.kind != 'TYPE_PARAMETERS':
                    raise ValueError('C1 selective private policy does not reference a type-parameter scope')
        self._state(material.root_context)
        _acyclic(material.scopes, material.root_context.environment, 'scope')
        _acyclic(material.regions, material.root_context.region, 'region')
        for region in material.regions:
            if region.parent is not None:
                parent = self.region(region.parent)
                if parent.eligibility == 'SUPPRESSED' and (
                        region.eligibility != 'SUPPRESSED'
                        or region.suppression_reason != parent.suppression_reason
                        or region.suppression_origin != parent.suppression_origin):
                    raise ValueError('C1 child region cancels or changes ancestor suppression')
        self._boundaries = {}
        counts = [0] * 484
        for boundary in material.boundaries:
            rule, _ = self.role_view.locate(boundary.site)
            self._state(boundary.state)
            expected = self.catalog.rules[rule.rule]
            if boundary.selector != expected.selector or boundary.content_policy != expected.content_policy:
                raise ValueError('C1 boundary differs from its shared field rule')
            key = (boundary.site.occurrence, boundary.site.group, boundary.site.ordinal)
            entries = self._boundaries.setdefault(key, {})
            if boundary.site.item_path in entries:
                raise ValueError('duplicate or conflicting C1 boundary prefix')
            entries[boundary.site.item_path] = boundary.state
            counts[rule.rule] += 1
        if counts != [row.boundaries for row in material.counts]:
            raise ValueError('C1 boundary references disagree with rule coverage')
        self._incoming = {root: material.root_context}
        self._ports = None

    def _owner(self, owner):
        if type(owner) is not OwnerRef:
            raise TypeError('C1 needs a typed static owner')
        _index(owner.occurrence, len(self.syntax.nodes), 'owner occurrence')

    def _anchor(self, site):
        if site is not None:
            self.role_view.locate(site)

    def _state(self, state):
        if type(state) is not ContextState:
            raise TypeError('C1 needs a typed context state')
        self._owner(state.owner)
        self.scope(state.environment)
        self.region(state.region)
        self.private_context(state.private_context)

    def scope(self, index):
        return self.material.scopes[_index(index, len(self.material.scopes), 'scope')]

    def region(self, index):
        return self.material.regions[_index(index, len(self.material.regions), 'region')]

    def private_context(self, index):
        return self.material.private_contexts[_index(index, len(self.material.private_contexts), 'private')]

    def _at(self, site, incoming):
        rule, _ = self.role_view.locate(site)
        transfer = self.catalog.rules[rule.rule]
        # Declaration qualifiers belong to a target edge. Receiver/index
        # expressions start a child context; scalar name payloads retain it.
        if transfer.content_policy == 'CHILD_ROUTING' and incoming.declaration_route != 'CURRENT_ENV':
            incoming = replace(incoming, declaration_route='CURRENT_ENV')
        entries = self._boundaries.get((site.occurrence, site.group, site.ordinal), {})
        path = site.item_path
        for depth in range(len(path), -1, -1):
            if path[:depth] in entries:
                return entries[path[:depth]]
        return incoming

    def entry_context(self, occurrence):
        _index(occurrence, len(self.syntax.nodes), 'occurrence')
        chain, current = [], occurrence
        while current not in self._incoming:
            edge = self.role_view.parents[current]
            chain.append((current, edge))
            current = edge.occurrence
        for child, edge in reversed(chain):
            self._incoming[child] = self._at(edge, self._incoming[edge.occurrence])
        return self._incoming[occurrence]

    def context_at_slot(self, site):
        rule, value = self.role_view.locate(site)
        transfer = self.catalog.rules[rule.rule]
        state = self._at(site, self.entry_context(site.occurrence))
        return ContextResolution(site, _state(value), transfer.selector, transfer.content_policy, state)

    resolve = context_at_slot

    def context_for_port(self, index):
        return self.context_at_slot(self.roles.ports[_index(index, len(self.roles.ports), 'role port')].site)

    def port_context(self, port):
        if type(port) is int:
            return self.context_for_port(port)
        if type(port) is not RolePort:
            raise TypeError('C1 role query needs an original RolePort or its exact ordinal')
        if self._ports is None:
            self._ports = frozenset(self.roles.ports)
        if port not in self._ports:
            raise ValueError('C1 role query is not an original port of this material')
        return self.context_at_slot(port.site)

    def verify_totality(self):
        """Stream every declared slot and original role port; retain no results."""
        counts = [[0] * 5 for _ in self.catalog.rules]
        for occurrence, node in enumerate(self.syntax.nodes):
            for group, values in (('FIELD', node.fields), ('ATTRIBUTE', node.attributes)):
                for ordinal, value in enumerate(values):
                    site = SlotRef(occurrence, group, ordinal)
                    answer = self.context_at_slot(site)
                    self._state(answer.state)
                    rule, _ = self.role_view.locate(site)
                    row = counts[rule.rule]
                    row[0] += 1
                    row[{'ABSENT': 1, 'NONE': 2, 'EMPTY_LIST': 3, 'VALUES': 4}[answer.slot_state]] += 1
        expected = [[c.slots, c.absent, c.none, c.empty_list, c.values] for c in self.material.counts]
        if counts != expected:
            raise ValueError('C1 total slot queries differ from their independent coverage counters')
        for index, port in enumerate(self.roles.ports):
            rule, _ = self.role_view.locate(port.site)
            if port.rule != rule.rule:
                raise ValueError('C1 role port does not reference its original field rule')
            self._state(self.context_for_port(index).state)
        return {'slots': sum(row[0] for row in counts), 'ports': len(self.roles.ports)}


# C1 source context templates; no binding/storage/compiler-table joins.
_VERSION = 'PYTHON_SOURCE_CONTEXTS_C1_V1'
_SPECIAL = {}
for _kind in ('FunctionDef', 'AsyncFunctionDef'):
    for _field, _selector in (('args', 'ARG_SPEC'), ('body', 'BODY'),
            ('decorator_list', 'DECORATORS'), ('returns', 'HEADER_ANNOTATION'), ('type_params', 'TYPE_PARAMS')):
        _SPECIAL[_kind, _field] = _selector
for _field, _selector in (('bases', 'CLASS_HEADER'), ('keywords', 'CLASS_HEADER'),
        ('body', 'BODY'), ('decorator_list', 'DECORATORS'), ('type_params', 'TYPE_PARAMS')):
    _SPECIAL['ClassDef', _field] = _selector
_SPECIAL.update({('Lambda', 'args'): 'ARG_SPEC', ('Lambda', 'body'): 'BODY',
    ('AnnAssign', 'target'): 'ANN_TARGET', ('AnnAssign', 'annotation'): 'ANN_ANNOTATION',
    ('AnnAssign', 'value'): 'ANN_VALUE', ('TypeAlias', 'name'): 'TYPE_ALIAS_NAME',
    ('TypeAlias', 'type_params'): 'TYPE_PARAMS', ('TypeAlias', 'value'): 'TYPE_ALIAS_VALUE',
    ('arg', 'annotation'): 'ARG_ANNOTATION', ('NamedExpr', 'target'): 'WALRUS_TARGET'})
for _field in ('posonlyargs', 'args', 'vararg', 'kwonlyargs', 'kwarg'):
    _SPECIAL['arguments', _field] = 'FORMAL_PART'
for _field in ('defaults', 'kw_defaults'):
    _SPECIAL['arguments', _field] = 'DEFAULTS'
for _kind, _field in (('TypeVar', 'bound'), ('TypeVar', 'default_value'),
        ('ParamSpec', 'default_value'), ('TypeVarTuple', 'default_value')):
    _SPECIAL[_kind, _field] = 'TYPEVAR_EVAL'
for _kind in ('ListComp', 'SetComp', 'DictComp', 'GeneratorExp'):
    _SPECIAL[_kind, 'generators'] = 'COMP_GENERATORS'
for _kind in ('ListComp', 'SetComp', 'GeneratorExp'):
    _SPECIAL[_kind, 'elt'] = 'COMP_RESULT'
for _field in ('key', 'value'):
    _SPECIAL['DictComp', _field] = 'COMP_RESULT'
for _field in ('target', 'iter', 'ifs'):
    _SPECIAL['comprehension', _field] = 'COMP_CLAUSE'
for _field in ('argtypes', 'returns'):
    _SPECIAL['FunctionType', _field] = 'TYPE_GRAMMAR'
_MARKERS = ({(kind, 'ctx') for kind in ('Attribute', 'Subscript', 'Starred', 'Name', 'List', 'Tuple')}
    | {(kind, 'op') for kind in ('AugAssign', 'BinOp', 'BoolOp', 'UnaryOp')}
    | {('Compare', 'ops'), ('Module', 'type_ignores')})


def _digest(document):
    return sha256(json.dumps(document, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('ascii')).hexdigest()


@lru_cache(maxsize=1)
def _catalog():
    roles = _role_catalog()
    rules, seen, markers, child_count = [], set(), set(), 0
    for rule in roles.rules:
        child = rule.group == 'FIELD' and rule.shape in ('NODE', 'OPTIONAL_NODE', 'LIST_NODE')
        key = rule.constructor, rule.name
        if child:
            child_count += 1
            if key in _SPECIAL:
                seen.add(key);selector, policy = _SPECIAL[key], 'CHILD_ROUTING'
            elif key in _MARKERS:
                markers.add(key);selector, policy = 'NO_EVALUATION', 'SYNTAX_MARKER'
            else:
                selector, policy = 'INHERIT', 'CHILD_ROUTING'
        else:
            # arg nodes already inherit FORMAL_INTERFACE from FORMAL_PART;
            # their scalar name is metadata with that known operation owner.
            selector = 'PAYLOAD'
            policy = 'METADATA'
        rules.append(ContextRule(rule.rule, selector, policy))
    if child_count != 157 or seen != set(_SPECIAL) or len(seen) != 50 or markers != _MARKERS or len(markers) != 12:
        raise ValueError('C1 transfer catalog does not cover its reviewed field partition')
    payload = dict(rule_version=_VERSION, f2_schema_sha256=roles.f2_schema_sha256,
        role_catalog_sha256=roles.content_sha256, rules=encode(tuple(rules)),
        schema='scar.source-context-catalog', schema_version=1)
    return ContextCatalog(_VERSION, roles.f2_schema_sha256, roles.content_sha256,
        _digest(payload), tuple(rules))


def source_context_catalog() -> dict:
    return encode(_catalog())


def _zero_counts():
    return tuple(RuleContextCount(i, 0, 0, 0, 0, 0, 0) for i in range(484))


def _limits(values):
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError('C1 budgets must be exact nonnegative integers')


class ContextBudgetExceeded(ValueError):
    def __init__(self, code, site, processed):
        self.frontier = ContextFrontier(code, site, processed, 'context budget reached; no partial templates published')
        super().__init__(self.frontier.detail)


def _child_base(state):
    # Target qualifiers belong to that edge/target, not receiver/index children.
    return state if state.declaration_route == 'CURRENT_ENV' else replace(state, declaration_route='CURRENT_ENV')


class _Router:
    def __init__(self, syntax, roles, limits):
        self.syntax, self.roles, self.limits = syntax, roles, limits
        self.view, self.catalog = SyntaxRoleView(syntax), _catalog()
        self.scopes, self.regions = [], []
        self.privates = [PrivateContext(None, 'NONE', None)]
        self.boundaries, self.counts, self.slots = [], [[0] * 6 for _ in range(484)], 0
        self.plans, self.comps, self.annotation_groups = {}, {}, {}
        self.yield_owners = set()
        self.future_annotations = self._future_annotations()

    def field(self, occurrence, name):
        return self.view.field(occurrence, name)

    def site(self, occurrence, name):
        node = self.syntax.nodes[occurrence]
        return SlotRef(occurrence, 'FIELD', self.view.fields[node.constructor][name])

    def has_child(self, value):
        return value.kind == 'NODE' or value.kind == 'LIST' and bool(value.items)

    def _future_annotations(self):
        root = self.syntax.nodes[self.syntax.root]
        if root.constructor not in ('Module', 'Interactive'):
            return False
        for i, item in enumerate(self.field(self.syntax.root, 'body').items):
            node = self.syntax.nodes[item.reference]
            if i == 0 and node.constructor == 'Expr':
                expr = self.field(item.reference, 'value')
                if expr.kind == 'NODE' and self.syntax.nodes[expr.reference].constructor == 'Constant':
                    literal = self.field(expr.reference, 'value').scalar.to_python()
                    if type(literal) is str:
                        continue
            if node.constructor != 'ImportFrom':
                break
            module = self.field(item.reference, 'module')
            level = self.field(item.reference, 'level')
            if module.kind != 'SCALAR' or module.scalar.to_python() != '__future__' or level.scalar.to_python() != 0:
                break
            for alias in self.field(item.reference, 'names').items:
                if self.field(alias.reference, 'name').scalar.to_python() == 'annotations':
                    return True
        return False

    def scope(self, kind, owner, anchor, parent, policy):
        if len(self.scopes) >= self.limits[0]:
            raise ContextBudgetExceeded('SCOPE_BUDGET', anchor, self.slots)
        self.scopes.append(SourceScope(kind, owner, anchor, parent, policy))
        return len(self.scopes) - 1

    def region(self, kind, owner, anchor, parent_state, trigger, reason=None):
        if len(self.regions) >= self.limits[1]:
            raise ContextBudgetExceeded('REGION_BUDGET', anchor, self.slots)
        parent = self.regions[parent_state.region] if parent_state is not None else None
        origin = anchor if reason is not None else None
        if parent is not None and parent.eligibility == 'SUPPRESSED':
            reason, origin = parent.suppression_reason, parent.suppression_origin
        self.regions.append(EvaluationRegion(kind, owner, anchor,
            parent_state.region if parent_state is not None else None, trigger,
            'SUPPRESSED' if reason is not None else 'ELIGIBLE', origin, reason))
        return len(self.regions) - 1

    def private(self, occurrence, mode, group=None):
        value = PrivateContext(OwnerRef(occurrence, 'CLASS_DECLARATION'), mode, group)
        if value not in self.privates:
            self.privates.append(value)
        return self.privates.index(value)

    def root_state(self):
        kind = {'exec': 'ROOT_EXEC', 'eval': 'ROOT_EVAL', 'single': 'ROOT_INTERACTIVE', 'func_type': 'TYPE_GRAMMAR'}[self.syntax.mode]
        owner = OwnerRef(self.syntax.root, 'ROOT')
        env = self.scope(kind, owner, None, None, 'EXTERNAL_ROOT_NAMESPACE_INTERFACE')
        region = self.region('TYPE_GRAMMAR' if self.syntax.mode == 'func_type' else 'ROOT', owner,
            None, None, 'NO_EVALUATION' if self.syntax.mode == 'func_type' else 'ROOT_EXECUTION',
            'TYPE_GRAMMAR' if self.syntax.mode == 'func_type' else None)
        return ContextState(owner, env, region, 0)

    def annotation_policy(self, environment):
        while environment is not None:
            scope = self.scopes[environment]
            if scope.kind == 'CLASS':
                return 'ANNOTATION_CLASS_NAMESPACE_ALLOWED'
            if scope.kind not in ('ANNOTATION', 'TYPE_PARAMETERS', 'TYPE_ALIAS', 'TYPE_PARAMETER_EVALUATOR'):
                break
            environment = scope.parent
        return 'ANNOTATION_NAMESPACE'

    def prepare(self, occurrence, state):
        node = self.syntax.nodes[occurrence]
        kind = node.constructor
        if kind in ('ListComp', 'SetComp', 'DictComp', 'GeneratorExp'):
            owner_kind = 'GENERATOR_EXPRESSION' if kind == 'GeneratorExp' else 'COMPREHENSION'
            owner = OwnerRef(occurrence, owner_kind)
            anchor = self.site(occurrence, 'generators')
            env = self.scope(owner_kind, owner, anchor, state.environment, 'COMPREHENSION_EXCLUDES_CLASS_NAMESPACE')
            region = self.region(owner_kind, owner, anchor, state,
                'GENERATOR_EXPRESSION_RESUMPTION' if kind == 'GeneratorExp' else 'COMPREHENSION_EVALUATION')
            self.comps[occurrence] = dict(creator=state, body=ContextState(owner, env, region, state.private_context))
            return
        if kind not in ('FunctionDef', 'AsyncFunctionDef', 'Lambda', 'ClassDef', 'TypeAlias'):
            return
        plan = {'creator': state, 'typeparams': state}
        self.plans[occurrence] = plan
        if kind != 'Lambda' and self.has_child(self.field(occurrence, 'type_params')):
            anchor, owner = self.site(occurrence, 'type_params'), OwnerRef(occurrence, 'TYPE_PARAMETERS')
            env = self.scope('TYPE_PARAMETERS', owner, anchor, state.environment, self.annotation_policy(state.environment))
            private = self.private(occurrence, 'SELECTIVE', env) if kind == 'ClassDef' else state.private_context
            region = self.region('TYPE_PARAMETERS', owner, anchor, state, 'DECLARATION_TIME')
            plan['typeparams'] = ContextState(owner, env, region, private)
        parent = plan['typeparams']
        if kind == 'TypeAlias':
            return
        body_kind = 'CLASS_BODY' if kind == 'ClassDef' else ('LAMBDA_BODY' if kind == 'Lambda' else 'FUNCTION_BODY')
        owner = OwnerRef(occurrence, body_kind)
        anchor = self.site(occurrence, 'body')
        env = self.scope('CLASS' if kind == 'ClassDef' else ('LAMBDA' if kind == 'Lambda' else 'FUNCTION'),
            owner, anchor, parent.environment, 'CLASS_NAMESPACE' if kind == 'ClassDef' else 'FUNCTION_EXCLUDES_CLASS_NAMESPACE')
        private = self.private(occurrence, 'ALL') if kind == 'ClassDef' else parent.private_context
        trigger = 'CLASS_CONSTRUCTION' if kind == 'ClassDef' else ('COROUTINE_RESUMPTION' if kind == 'AsyncFunctionDef' else 'FUNCTION_INVOCATION')
        region = self.region(body_kind, owner, anchor, parent, trigger)
        plan['body'] = ContextState(owner, env, region, private)
        if kind == 'ClassDef':
            return
        formal_owner = OwnerRef(occurrence, 'FORMAL_INTERFACE')
        formal_region = self.region('FORMAL_INTERFACE', formal_owner, self.site(occurrence, 'args'), state, 'FORMAL_INTERFACE')
        plan['formal'] = ContextState(formal_owner, env, formal_region, parent.private_context)

    def declaration(self, occurrence):
        current = occurrence
        while current in self.view.parents:
            edge = self.view.parents[current]
            node = self.syntax.nodes[edge.occurrence]
            rule, _ = self.view.locate(edge)
            if node.constructor in ('FunctionDef', 'AsyncFunctionDef', 'Lambda') and rule.name == 'args':
                return edge.occurrence, self.plans[edge.occurrence]
            current = edge.occurrence
        raise ValueError('argument context has no original declaration owner')

    def header_annotation(self, occurrence, site):
        plan = self.plans[occurrence]
        if 'annotation' not in plan:
            parent, owner = plan['typeparams'], OwnerRef(occurrence, 'HEADER_ANNOTATIONS')
            env = self.scope('ANNOTATION', owner, site, parent.environment, self.annotation_policy(parent.environment))
            region = self.region('FUNCTION_ANNOTATION', owner, site, parent, 'ANNOTATION_REQUEST',
                'FUTURE_ANNOTATION_STRINGIZED' if self.future_annotations else None)
            plan['annotation'] = ContextState(owner, env, region, parent.private_context)
        return plan['annotation']

    def eager_header(self, plan, site):
        if 'eager' not in plan:
            creator = plan['creator']
            region = self.region('DECLARATION_TIME', creator.owner, site, creator, 'DECLARATION_TIME')
            plan['eager'] = replace(creator, region=region)
        return plan['eager']

    def variable_annotation(self, occurrence, site, state):
        source = self.scopes[state.environment]
        if state.environment not in self.annotation_groups:
            owner = OwnerRef(source.owner.occurrence, 'VARIABLE_ANNOTATIONS')
            env = self.scope('ANNOTATION', owner, site, state.environment, self.annotation_policy(state.environment))
            self.annotation_groups[state.environment] = env
        env = self.annotation_groups[state.environment]
        owner = self.scopes[env].owner
        simple = self.field(occurrence, 'simple').scalar.to_python()
        reason = ('FUNCTION_LOCAL_ANNOTATION' if source.kind in ('FUNCTION', 'LAMBDA') else
            ('NON_SIMPLE_ANNOTATION' if simple != 1 else ('FUTURE_ANNOTATION_STRINGIZED' if self.future_annotations else None)))
        region = self.region('VARIABLE_ANNOTATION', owner, site, state, 'ANNOTATION_REQUEST', reason)
        return ContextState(owner, env, region, state.private_context)

    def comp_clause(self, occurrence):
        edge = self.view.parents.get(occurrence)
        if edge is None or edge.occurrence not in self.comps or len(edge.item_path) != 1:
            raise ValueError('comprehension clause has no original generator-list position')
        return self.comps[edge.occurrence], edge.item_path[0]

    def in_comp_iterable(self, occurrence):
        while occurrence in self.view.parents:
            edge = self.view.parents[occurrence]
            node = self.syntax.nodes[edge.occurrence]
            rule, _ = self.view.locate(edge)
            if node.constructor == 'comprehension' and rule.name == 'iter':
                return True
            occurrence = edge.occurrence
        return False

    def transfer(self, site, state, rule, value):
        if rule.content_policy == 'CHILD_ROUTING':
            state = _child_base(state)
            if not self.has_child(value):
                return state
        occurrence, selector = site.occurrence, rule.selector
        kind = self.syntax.nodes[occurrence].constructor
        if selector in ('INHERIT', 'PAYLOAD', 'NO_EVALUATION', 'TYPE_GRAMMAR'):
            return state
        if selector in ('BODY', 'TYPE_PARAMS'):
            return self.plans[occurrence]['body' if selector == 'BODY' else 'typeparams']
        if selector == 'CLASS_HEADER':
            plan = self.plans[occurrence]
            return plan['typeparams'] if plan['typeparams'] != plan['creator'] else self.eager_header(plan, site)
        if selector == 'ARG_SPEC':
            return self.plans[occurrence]['creator']
        if selector in ('DECORATORS', 'TYPE_ALIAS_NAME'):
            creator = self.eager_header(self.plans[occurrence], site)
            return replace(creator, declaration_route='TYPE_ALIAS_DECLARATION') if selector == 'TYPE_ALIAS_NAME' else creator
        if selector in ('FORMAL_PART', 'DEFAULTS'):
            _, plan = self.declaration(occurrence)
            return self.eager_header(plan, site) if selector == 'DEFAULTS' else plan['formal']
        if selector == 'HEADER_ANNOTATION':
            return self.header_annotation(occurrence, site)
        if selector == 'ARG_ANNOTATION':
            declaration, _ = self.declaration(occurrence)
            if self.syntax.nodes[declaration].constructor == 'Lambda':
                raise ValueError('lambda annotations have no reviewed C1 source context')
            return self.header_annotation(declaration, site)
        if selector == 'ANN_TARGET':
            route = 'ANNOTATION_DECLARATION' if _state(self.field(occurrence, 'value')) == 'NONE' else 'ANNOTATED_ASSIGNMENT'
            return replace(state, declaration_route=route)
        if selector == 'ANN_VALUE':
            return state
        if selector == 'ANN_ANNOTATION':
            return self.variable_annotation(occurrence, site, state)
        if selector in ('TYPE_ALIAS_VALUE', 'TYPEVAR_EVAL'):
            parent = self.plans[occurrence]['typeparams'] if selector == 'TYPE_ALIAS_VALUE' else state
            if selector == 'TYPE_ALIAS_VALUE':
                owner_kind, env_kind, trigger = 'TYPE_ALIAS_VALUE', 'TYPE_ALIAS', 'TYPE_ALIAS_VALUE_REQUEST'
            else:
                name = self.view.locate(site)[0].name
                owner_kind = 'TYPE_PARAMETER_BOUND' if name == 'bound' else 'TYPE_PARAMETER_DEFAULT'
                env_kind, trigger = 'TYPE_PARAMETER_EVALUATOR', owner_kind + '_REQUEST'
            owner = OwnerRef(occurrence, owner_kind)
            env = self.scope(env_kind, owner, site, parent.environment, self.annotation_policy(parent.environment))
            region = self.region(owner_kind, owner, site, parent, trigger)
            return ContextState(owner, env, region, parent.private_context)
        if selector in ('COMP_GENERATORS', 'COMP_RESULT'):
            return self.comps[occurrence]['body']
        if selector == 'COMP_CLAUSE':
            plan, position = self.comp_clause(occurrence)
            name = self.view.locate(site)[0].name
            if name == 'iter' and position == 0:
                return plan['creator']
            result = plan['body']
            return replace(result, declaration_route='COMPREHENSION_TARGET') if name == 'target' else result
        if selector == 'WALRUS_TARGET':
            environment_kind = self.scopes[state.environment].kind
            route = ('FORBIDDEN_COMP_ITERABLE' if self.in_comp_iterable(occurrence) else
                ('ENCLOSING_NONCOMP' if environment_kind in ('COMPREHENSION', 'GENERATOR_EXPRESSION') else
                 ('FORBIDDEN_ANNOTATION_BINDING' if environment_kind in ('ANNOTATION', 'TYPE_PARAMETERS', 'TYPE_ALIAS', 'TYPE_PARAMETER_EVALUATOR') else 'CURRENT_ENV')))
            return replace(state, declaration_route=route)
        raise ValueError('C1 selector has no implementation: ' + selector)

    def boundary(self, site, rule, state, base):
        if state == base:
            return
        if len(self.boundaries) >= self.limits[2]:
            raise ContextBudgetExceeded('BOUNDARY_BUDGET', site, self.slots)
        self.boundaries.append(ContextBoundary(site, rule.selector, state, rule.content_policy))
        self.counts[rule.rule][5] += 1

    def run(self):
        root = self.root_state()
        pending = [(self.syntax.root, root)]
        while pending:
            occurrence, state = pending.pop()
            self.prepare(occurrence, state)
            node = self.syntax.nodes[occurrence]
            if node.constructor in ('Yield', 'YieldFrom'):
                self.yield_owners.add(state.owner)
            children = []
            for group, values in (('FIELD', node.fields), ('ATTRIBUTE', node.attributes)):
                for ordinal, value in enumerate(values):
                    site = SlotRef(occurrence, group, ordinal)
                    if self.slots >= self.limits[3]:
                        raise ContextBudgetExceeded('SLOT_BUDGET', site, self.slots)
                    role_rule, _ = self.view.locate(site)
                    rule = self.catalog.rules[role_rule.rule]
                    row = self.counts[rule.rule]
                    row[0] += 1;self.slots += 1
                    row[{'ABSENT': 1, 'NONE': 2, 'EMPTY_LIST': 3, 'VALUES': 4}[_state(value)]] += 1
                    base = _child_base(state) if rule.content_policy == 'CHILD_ROUTING' else state
                    result = self.transfer(site, state, rule, value)
                    self.boundary(site, rule, result, base)
                    stack = [value]
                    while stack:
                        item = stack.pop()
                        if item.kind == 'NODE':
                            children.append((item.reference, result))
                        elif item.kind == 'LIST':
                            stack.extend(reversed(item.items))
            pending.extend(reversed(children))
        # Ownership is already assigned, so nested header yields affect their
        # creator rather than being skipped together with the nested body.
        for i, region in enumerate(self.regions):
            if region.owner in self.yield_owners and region.kind in ('FUNCTION_BODY', 'LAMBDA_BODY'):
                async_body = self.syntax.nodes[region.owner.occurrence].constructor == 'AsyncFunctionDef'
                self.regions[i] = replace(region, trigger='ASYNC_GENERATOR_RESUMPTION' if async_body else 'GENERATOR_RESUMPTION')
        counts = tuple(RuleContextCount(i, *row) for i, row in enumerate(self.counts))
        return root, counts


def _profile():
    from scar.ir.program import source_contexts as context_ir
    catalog = _catalog()
    implementation = _digest({name: sha256(Path(path).read_bytes()).hexdigest() for name, path in (
        ('analysis', __file__), ('view', __file__), ('ir', context_ir.__file__))})
    return SourceContextsProfile(_VERSION, catalog.f2_schema_sha256,
        catalog.role_catalog_sha256, catalog.content_sha256, implementation)


def _derive(f0, f1, f2, f3b, limits):
    _limits(limits)
    # B grants the source receipt by doing its one strict F0/F1/F2 replay.
    # Decode that exact, already-verified F2 rather than parsing it once more.
    roles = read_source_roles(f3b, f0, f1, f2)
    syntax = decode(SourceSyntaxMaterial, f2)
    inputs = SourceContextsInput(_digest(f0), _digest(f1), _digest(f2), _digest(f3b),
        f0['path'], syntax.mode, syntax.status, roles.status)
    profile = _profile()
    if roles.status != 'CLASSIFIED':
        material = SourceContextsMaterial(inputs, profile, *limits, 'UPSTREAM_NOT_CLASSIFIED',
            None, (), (), (), (), _zero_counts(), 0, ())
        return material, syntax, roles
    router = _Router(syntax, roles, limits)
    try:
        root, counts = router.run()
    except ContextBudgetExceeded as error:
        material = SourceContextsMaterial(inputs, profile, *limits, 'BUDGET_STOPPED',
            None, (), (), (), (), _zero_counts(), 0, (error.frontier,))
        return material, syntax, roles
    material = SourceContextsMaterial(inputs, profile, *limits, 'CONTEXTUALIZED', root,
        tuple(router.scopes), tuple(router.regions), tuple(router.privates),
        tuple(router.boundaries), counts, router.slots, ())
    del router
    from scar.analysis.source_contexts import SourceContextView
    SourceContextView(syntax, roles, material).verify_totality()
    return material, syntax, roles


def source_contexts_document(f0, f1, f2, f3b, *, max_scopes=100_000, max_regions=200_000,
        max_boundaries=1_000_000, max_slots=10_000_000) -> dict:
    limits = max_scopes, max_regions, max_boundaries, max_slots
    return encode(_derive(f0, f1, f2, f3b, limits)[0])


def read_source_contexts(document, f0, f1, f2, f3b) -> SourceContextsMaterial:
    material = decode(SourceContextsMaterial, document)
    if material.profile != _profile():
        raise ValueError('C1 context implementation/catalog profile mismatch')
    limits = material.max_scopes, material.max_regions, material.max_boundaries, material.max_slots
    fresh, _, _ = _derive(f0, f1, f2, f3b, limits)
    if material != fresh:
        raise ValueError('C1 templates differ from original F2/B slots and field transfers')
    return material


# The view is a same-layer adapter, with no compiler/table or runtime dependency.


__all__ = [
    'source_context_catalog',
    'source_contexts_document',
    'read_source_contexts',
    'SourceContextView',
    'ContextBudgetExceeded',
]
