"""C2a source-name projection and declarations; lookup/capture remain separate."""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

from scar.analysis.source_role_materials import _catalog as _role_catalog, _state
from scar.analysis.source_contexts import _catalog as _context_catalog, read_source_contexts, SourceContextView
from scar.ir.program.source_syntax import SourceSyntaxMaterial
from scar.ir.program.source_contexts import SourceContextsMaterial
from scar.ir.program.source_roles import SourceRolesMaterial
from scar.ir.program.source_declarations import DeclarationCatalog, DeclarationConflict, DeclarationContribution, DeclarationFrontier, LexicalField, NameKey, NameProjection, PortDisposition, ScopeDeclarations, SourceDeclarationsInput, SourceDeclarationsMaterial, SourceDeclarationsProfile
from scar.ir.record_codec import decode, encode


_VERSION = 'PYTHON_SOURCE_DECLARATIONS_C2A_V1'
_PHASE = 'SYMBOL_VISIT_PROJECTION'
_LEXICAL = tuple(LexicalField(*key) for key in (
    ('FunctionDef', 'name'), ('AsyncFunctionDef', 'name'), ('ClassDef', 'name'),
    ('Name', 'id'), ('Global', 'names'), ('Nonlocal', 'names'), ('ExceptHandler', 'name'),
    ('arg', 'arg'), ('alias', 'name'), ('alias', 'asname'), ('MatchMapping', 'rest'),
    ('MatchStar', 'name'), ('MatchAs', 'name'), ('TypeVar', 'name'),
    ('ParamSpec', 'name'), ('TypeVarTuple', 'name')))
_LEXICAL_KEYS = frozenset((field.constructor, field.name) for field in _LEXICAL)
_OPERATIONS = ('DECLARE', 'USE_SYNTAX', 'STORE_CANDIDATE', 'DELETE_CANDIDATE',
    'PARAMETER_INTERFACE', 'TYPE_PARAMETER_INTERFACE', 'ANNOTATION_DECLARATION',
    'GLOBAL_DIRECTIVE', 'NONLOCAL_DIRECTIVE')
_FLAGS = ('USE', 'DEF_LOCAL', 'DEF_PARAM', 'DEF_TYPE_PARAM', 'DEF_IMPORT',
    'DEF_ANNOT', 'DEF_COMP_ITER', 'DEF_GLOBAL', 'DEF_NONLOCAL')


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('ascii')).hexdigest()


@lru_cache(maxsize=1)
def _catalog():
    roles, contexts = _role_catalog(), _context_catalog()
    actual = {(rule.constructor, rule.name) for rule in roles.rules if rule.group == 'FIELD'}
    if not {(field.constructor, field.name) for field in _LEXICAL} <= actual or len(_LEXICAL) != 16:
        raise ValueError('C2a lexical carrier catalog differs from its reviewed schema')
    payload = dict(rule_version=_VERSION, projection_phase=_PHASE,
        f2_schema_sha256=roles.f2_schema_sha256, role_catalog_sha256=roles.content_sha256,
        context_catalog_sha256=contexts.content_sha256, lexical_fields=encode(_LEXICAL),
        operations=list(_OPERATIONS), visit_flags=list(_FLAGS),
        schema='scar.source-declaration-catalog', schema_version=1)
    return DeclarationCatalog(_VERSION, _PHASE, roles.f2_schema_sha256,
        roles.content_sha256, contexts.content_sha256, _LEXICAL, _OPERATIONS, _FLAGS, _digest(payload))


def source_declaration_catalog() -> dict:
    return encode(_catalog())


def _index(index, size, label):
    if type(index) is not int or not 0 <= index < size:
        raise ValueError('C2a ' + label + ' reference is outside its original table')
    return index


class SourceDeclarationView:
    """One source-unit ledger; all public integer references are material-local."""

    def __init__(self, roles, c1, material, *, contexts=None):
        if (type(roles) is not SourceRolesMaterial or roles.status != 'CLASSIFIED'
                or type(c1) is not SourceContextsMaterial or c1.status != 'CONTEXTUALIZED'
                or type(material) is not SourceDeclarationsMaterial or material.status != 'DECLARED'):
            raise ValueError('C2a queries require typed classified/contextualized/declared inputs')
        self.roles, self.contexts, self.material, self.context_view = roles, c1, material, contexts
        self.catalog = _catalog()
        if (material.input.port_count != len(roles.ports)
                or material.input.f3b_document_sha256 != _digest(encode(roles))
                or material.input.c1_document_sha256 != _digest(encode(c1))
                or c1.input.f3b_document_sha256 != material.input.f3b_document_sha256
                or material.input.f2_document_sha256 != c1.input.f2_document_sha256
                or material.profile.role_catalog_sha256 != roles.profile.catalog_sha256
                or material.profile.context_catalog_sha256 != c1.profile.context_catalog_sha256
                or material.profile.declaration_catalog_sha256 != self.catalog.content_sha256):
            raise ValueError('C2a source receipts or shared catalogs differ from B/C1')
        if contexts is not None and (contexts.roles != roles or contexts.material != c1):
            raise ValueError('C2a query context belongs to a different original source unit')
        if len(material.scopes) != len(c1.scopes) or tuple(s.scope for s in material.scopes) != tuple(range(len(c1.scopes))):
            raise ValueError('C2a scope ledger does not cover every original C1 scope exactly once')
        if len({name.value for name in material.names}) != len(material.names):
            raise ValueError('C2a effective names are not uniquely interned')

    def port(self, index):
        return self.material.ports[_index(index, len(self.material.ports), 'port')]

    def scope(self, index):
        return self.material.scopes[_index(index, len(self.material.scopes), 'scope')]

    def contribution(self, index):
        return self.material.contributions[_index(index, len(self.material.contributions), 'contribution')]

    def name(self, index):
        return self.material.names[_index(index, len(self.material.names), 'name')]

    def _source_scope(self, index):
        return self.contexts.scopes[_index(index, len(self.contexts.scopes), 'source scope')]

    def _region(self, index):
        return self.contexts.regions[_index(index, len(self.contexts.regions), 'region')]

    def verify(self):
        """Check source references, all port dispositions and inverse scope edges."""
        data = self.material
        operations, visit_flags = set(self.catalog.operations), set(self.catalog.visit_flags)
        mentioned, named_order, contribution_refs = set(), [], []
        for index, row in enumerate(data.ports):
            if type(row.port) is not int or row.port != index:
                raise ValueError('C2a port dispositions must cover the original B order exactly once')
            self._source_scope(row.source_scope)
            self._region(row.region)
            projection = row.projection
            if projection.phase != 'SYMBOL_VISIT_PROJECTION' or projection.private_mode not in ('NONE', 'ALL', 'SELECTIVE'):
                raise ValueError('C2a projection has an undeclared phase or private-name policy')
            if row.name is not None:
                self.name(row.name)
                if row.name not in mentioned:
                    mentioned.add(row.name)
                    named_order.append(row.name)
                if projection.private_mode == 'SELECTIVE':
                    self._source_scope(projection.parameter_group)
                    if type(projection.prefix_count) is not int or projection.prefix_count < 0:
                        raise ValueError('C2a selective projection requires an original parameter prefix')
            if self.context_view is not None:
                context = self.context_view.context_for_port(index).state
                if (row.source_scope, row.region) != (context.environment, context.region):
                    raise ValueError('C2a port does not reference its original C1 environment/region')
                if row.name is not None:
                    private = self.context_view.private_context(context.private_context)
                    if (projection.value_rule != self.roles.ports[index].projection
                            or (projection.private_mode, projection.class_owner, projection.parameter_group)
                            != (private.mode, private.class_owner, private.parameter_group)):
                        raise ValueError('C2a name projection does not preserve its original B/C1 inputs')
                    if projection.private_mode == 'SELECTIVE':
                        group = self._source_scope(projection.parameter_group)
                        _, parameters = self.context_view.role_view.locate(group.anchor)
                        if parameters.kind != 'LIST' or projection.prefix_count > len(parameters.items):
                            raise ValueError('C2a selective prefix lies outside its original parameter group')
            if row.contribution is not None:
                contribution = self.contribution(row.contribution)
                if (contribution.port != index or contribution.name != row.name
                        or contribution.source_scope != row.source_scope or contribution.region != row.region):
                    raise ValueError('C2a port contribution does not round-trip to its source disposition')
                contribution_refs.append(row.contribution)
        if named_order != list(range(len(data.names))):
            raise ValueError('C2a name pool differs from first-port mentions')
        if sorted(contribution_refs) != list(range(len(data.contributions))):
            raise ValueError('C2a contribution references are duplicated, missing or unowned')
        by_scope = [[] for _ in self.contexts.scopes]
        for index, contribution in enumerate(data.contributions):
            _index(contribution.port, len(self.roles.ports), 'source port')
            source = self.roles.ports[contribution.port]
            if contribution.site != source.site:
                raise ValueError('C2a contribution does not reference its original B slot')
            self._source_scope(contribution.source_scope)
            self._source_scope(contribution.home)
            self._region(contribution.region)
            self.name(contribution.name)
            if (any(op not in operations for op in contribution.operations)
                    or any(flag not in visit_flags for flag in contribution.visit_flags)
                    or len(set(contribution.operations)) != len(contribution.operations)
                    or len(set(contribution.visit_flags)) != len(contribution.visit_flags)):
                raise ValueError('C2a contribution contains undeclared or duplicate operations/flags')
            by_scope[contribution.home].append(index)
        wildcard_refs = []
        for scope in data.scopes:
            expected = tuple(sorted(by_scope[scope.scope], key=lambda i: data.contributions[i].port))
            if scope.contributions != expected:
                raise ValueError('C2a scope contribution edges are incomplete or reordered')
            names = tuple(dict.fromkeys(data.contributions[i].name for i in expected))
            if scope.names != names:
                raise ValueError('C2a scope names differ from its original contribution order')
            for index in scope.directives:
                contribution = self.contribution(index)
                if index not in expected or not any(op in ('GLOBAL_DIRECTIVE', 'NONLOCAL_DIRECTIVE') for op in contribution.operations):
                    raise ValueError('C2a directive is not an original directive contribution of its scope')
            expected_directives = tuple(i for i in expected if any(op in ('GLOBAL_DIRECTIVE', 'NONLOCAL_DIRECTIVE') for op in data.contributions[i].operations))
            if scope.directives != expected_directives:
                raise ValueError('C2a whole-scope directive summary is incomplete')
            if len(set(scope.wildcards)) != len(scope.wildcards):
                raise ValueError('C2a namespace wildcard is duplicated')
            for index in scope.wildcards:
                row = self.port(index)
                if self.roles.ports[index].role != 'IMPORT_WILDCARD_SPEC' or row.source_scope != scope.scope or row.name is not None:
                    raise ValueError('C2a wildcard is not the original star import of its scope')
                wildcard_refs.append(index)
        expected_wildcards = [i for i, port in enumerate(self.roles.ports) if port.role == 'IMPORT_WILDCARD_SPEC']
        if sorted(wildcard_refs) != expected_wildcards:
            raise ValueError('C2a namespace wildcard coverage differs from original B')
        for conflict in data.conflicts:
            self._source_scope(conflict.scope)
            if conflict.name is not None:
                self.name(conflict.name)
            if not conflict.ports:
                raise ValueError('C2a conflict requires original source-port evidence')
            for index in conflict.ports:
                self.port(index)
        return {'ports': len(data.ports), 'contributions': len(data.contributions),
            'scopes': len(data.scopes), 'names': len(data.names), 'conflicts': len(data.conflicts)}


class DeclarationBudgetExceeded(ValueError):
    def __init__(self, code, port, processed):
        self.frontier = DeclarationFrontier(code, port, processed,
            'declaration budget reached; no partial ledger published')
        super().__init__(self.frontier.detail)


class _Ledger:
    def __init__(self, syntax, roles, contexts, limits):
        self.syntax, self.roles, self.contexts, self.limits = syntax, roles, contexts, limits
        self.view = SourceContextView(syntax, roles, contexts)
        self.fields = self.view.role_view
        self.names, self.intern, self.ports, self.contributions, self.conflicts = [], {}, [], [], []
        self.scope_contributions = [[] for _ in contexts.scopes]
        self.scope_wildcards = [[] for _ in contexts.scopes]
        self.prefixes, self.future = {}, self._future_annotations()
        self.current = 0

    def field(self, occurrence, name):
        return self.fields.field(occurrence, name)

    def _future_annotations(self):
        root = self.syntax.nodes[self.syntax.root]
        if root.constructor not in ('Module', 'Interactive'):
            return False
        for i, item in enumerate(self.field(self.syntax.root, 'body').items):
            node = self.syntax.nodes[item.reference]
            if i == 0 and node.constructor == 'Expr':
                value = self.field(item.reference, 'value')
                if value.kind == 'NODE' and self.syntax.nodes[value.reference].constructor == 'Constant':
                    if type(self.field(value.reference, 'value').scalar.to_python()) is str:
                        continue
            if node.constructor != 'ImportFrom':
                break
            module = self.field(item.reference, 'module')
            if module.kind != 'SCALAR' or module.scalar.to_python() != '__future__' or self.field(item.reference, 'level').scalar.to_python() != 0:
                break
            if any(self.field(alias.reference, 'name').scalar.to_python() == 'annotations'
                    for alias in self.field(item.reference, 'names').items):
                return True
        return False

    def annotation_ancestor(self, occurrence):
        while occurrence in self.fields.parents:
            edge = self.fields.parents[occurrence]
            node = self.syntax.nodes[edge.occurrence]
            rule, _ = self.fields.locate(edge)
            if (node.constructor, rule.name) in (('AnnAssign', 'annotation'), ('arg', 'annotation'),
                    ('FunctionDef', 'returns'), ('AsyncFunctionDef', 'returns')):
                return True
            occurrence = edge.occurrence
        return False

    def visibility(self, site, state):
        if self.syntax.mode == 'func_type':
            return 'TYPE_GRAMMAR_ONLY'
        if self.future and self.annotation_ancestor(site.occurrence):
            return 'DETACHED_FUTURE_ANNOTATION'
        region = self.contexts.regions[state.region]
        if (self.contexts.scopes[state.environment].kind == 'ANNOTATION'
                and region.suppression_reason == 'FUNCTION_LOCAL_ANNOTATION'):
            return 'IGNORED_LOCAL_ANNOTATION'
        return 'SYMBOL_VISIT'

    def prefix(self, group, site):
        scope = self.contexts.scopes[group]
        if scope.kind != 'TYPE_PARAMETERS' or scope.anchor is None:
            raise ValueError('SELECTIVE private policy has no original type-parameter group')
        _, value = self.fields.locate(scope.anchor)
        members = tuple(item.reference for item in value.items)
        member_index = {member: i for i, member in enumerate(members)}
        occurrence, count = site.occurrence, len(members)
        while True:
            if occurrence in member_index:
                count = member_index[occurrence] + 1
                break
            edge = self.fields.parents.get(occurrence)
            if edge is None:
                break
            occurrence = edge.occurrence
        key = group, count
        if key not in self.prefixes:
            self.prefixes[key] = frozenset(self.field(member, 'name').scalar.to_python() for member in members[:count])
        return count, self.prefixes[key]

    def projection(self, port, state, value=None):
        private = self.contexts.private_contexts[state.private_context]
        count, allowed = self.prefix(private.parameter_group, port.site) if private.mode == 'SELECTIVE' else (None, None)
        basis = NameProjection(port.projection, private.mode, private.class_owner,
            private.parameter_group, count)
        if value is None:
            return basis, None
        if type(value) is not str:
            raise ValueError('lexical source projection requires an exact identifier string')
        if port.projection == 'IMPORT_HEAD':
            value = value.split('.', 1)[0]
        elif port.projection != 'IDENTITY':
            raise ValueError('lexical source port has no supported name projection')
        if private.mode != 'NONE' and (allowed is None or value in allowed):
            # Read the class's original F2 spelling, never its projected binder.
            owner = private.class_owner.occurrence
            if self.syntax.nodes[owner].constructor != 'ClassDef':
                raise ValueError('private policy owner is not the original ClassDef')
            class_name = self.field(owner, 'name').scalar.to_python().lstrip('_')
            if class_name and value.startswith('__') and not value.endswith('__') and '.' not in value:
                value = '_' + class_name + value
        if value not in self.intern:
            if len(self.names) >= self.limits[0]:
                raise DeclarationBudgetExceeded('NAME_BUDGET', self.current, self.current)
            self.intern[value] = len(self.names)
            self.names.append(NameKey(value))
        return basis, self.intern[value]

    def conflict(self, code, scope, name, ports, detail):
        if len(self.conflicts) >= self.limits[2]:
            raise DeclarationBudgetExceeded('CONFLICT_BUDGET', self.current, self.current)
        self.conflicts.append(DeclarationConflict(code, scope, name, tuple(ports), detail))

    def target_owner(self, occurrence):
        edge = self.fields.parents.get(occurrence)
        if edge is None:
            return None, None
        rule, _ = self.fields.locate(edge)
        return edge.occurrence, rule.name

    def comp_target(self, occurrence):
        while occurrence in self.fields.parents:
            edge = self.fields.parents[occurrence]
            node = self.syntax.nodes[edge.occurrence]
            rule, _ = self.fields.locate(edge)
            if node.constructor == 'comprehension' and rule.name == 'target':
                return True
            if node.constructor not in ('Tuple', 'List', 'Starred'):
                return False
            occurrence = edge.occurrence
        return False

    def operation(self, port):
        role, cleanup = port.role, 'NONE'
        if role == 'REFERENCE_SYNTAX':
            return ('USE_SYNTAX',), ('USE',), cleanup
        if role == 'DELETION_SYNTAX':
            return ('DECLARE', 'DELETE_CANDIDATE'), ('DEF_LOCAL',), cleanup
        if role == 'AUGMENTED_BINDING_SYNTAX':
            return ('DECLARE', 'USE_SYNTAX', 'STORE_CANDIDATE'), ('DEF_LOCAL',), cleanup
        if role in ('BINDING_SYNTAX', 'DECLARATION_ONLY'):
            parent, field = self.target_owner(port.site.occurrence)
            if parent is not None and self.syntax.nodes[parent].constructor == 'AnnAssign' and field == 'target':
                simple = self.field(parent, 'simple').scalar.to_python() == 1
                rhs = _state(self.field(parent, 'value')) != 'NONE'
                if not simple and not rhs:
                    return (), (), cleanup
                ops, flags = ['DECLARE'], ['DEF_LOCAL']
                if rhs:ops.append('STORE_CANDIDATE')
                if simple:ops.append('ANNOTATION_DECLARATION');flags.append('DEF_ANNOT')
                return tuple(ops), tuple(flags), cleanup
            flags = ('DEF_LOCAL', 'DEF_COMP_ITER') if self.comp_target(port.site.occurrence) else ('DEF_LOCAL',)
            return ('DECLARE', 'STORE_CANDIDATE'), flags, cleanup
        if role == 'PARAMETER_BINDING_SYNTAX':
            return ('DECLARE', 'PARAMETER_INTERFACE'), ('DEF_PARAM',), cleanup
        if role == 'TYPE_PARAMETER_BINDING_SYNTAX':
            return ('DECLARE', 'TYPE_PARAMETER_INTERFACE'), ('DEF_LOCAL', 'DEF_TYPE_PARAM'), cleanup
        if role in ('GLOBAL_DECLARATION_SYNTAX', 'NONLOCAL_DECLARATION_SYNTAX'):
            return (('GLOBAL_DIRECTIVE',), ('DEF_GLOBAL',), cleanup) if role.startswith('GLOBAL') else (('NONLOCAL_DIRECTIVE',), ('DEF_NONLOCAL',), cleanup)
        if role == 'IMPORT_BINDING_SYNTAX':
            return ('DECLARE', 'STORE_CANDIDATE'), ('DEF_IMPORT',), cleanup
        if role == 'EXCEPTION_BINDING_SYNTAX':
            cleanup = 'CLEAR_ON_HANDLER_EXIT'
        if role in ('FUNCTION_BINDING_SYNTAX', 'CLASS_BINDING_SYNTAX', 'TYPE_ALIAS_BINDING_SYNTAX',
                'NAMED_EXPR_BINDING_SYNTAX', 'EXCEPTION_BINDING_SYNTAX', 'PATTERN_MAPPING_BINDING_SYNTAX',
                'PATTERN_STAR_BINDING_SYNTAX', 'PATTERN_AS_BINDING_SYNTAX'):
            return ('DECLARE', 'STORE_CANDIDATE'), ('DEF_LOCAL',), cleanup
        raise ValueError('reviewed lexical carrier has no contribution rule: ' + role)

    def home(self, port, state, name):
        source = state.environment
        if state.declaration_route.startswith('FORBIDDEN_'):
            self.conflict(state.declaration_route, source, name, (self.current,), 'C1 reports a forbidden declaration route')
            return None
        if port.role != 'NAMED_EXPR_BINDING_SYNTAX' or state.declaration_route != 'ENCLOSING_NONCOMP':
            return source
        home = source
        while self.contexts.scopes[home].kind in ('COMPREHENSION', 'GENERATOR_EXPRESSION'):
            home = self.contexts.scopes[home].parent
            if home is None:
                raise ValueError('comprehension route has no enclosing source environment')
        if self.contexts.scopes[home].kind not in ('FUNCTION', 'LAMBDA', 'ROOT_EXEC', 'ROOT_INTERACTIVE', 'ROOT_EVAL'):
            self.conflict('WALRUS_FORBIDDEN_HOME', source, name, (self.current,), 'escaped walrus reaches an ineligible declaration environment')
            return None
        return home

    def add_port(self, index, port):
        self.current = index
        state = self.view.context_for_port(index).state
        field_rule, value = self.fields.locate(port.site)
        key = field_rule.constructor, field_rule.name
        source, region = state.environment, state.region
        basis, name = self.projection(port, state)
        if port.role == 'IMPORT_WILDCARD_SPEC':
            self.scope_wildcards[source].append(index)
            if self.contexts.scopes[source].kind not in ('ROOT_EXEC', 'ROOT_INTERACTIVE'):
                self.conflict('STAR_IMPORT_OUTSIDE_MODULE', source, None, (index,), 'star import has no module-level declaration interface')
            disposition = 'WILDCARD_IMPORT'
        elif port.role in ('IMPORT_MODULE_SPEC', 'IMPORT_MEMBER_SPEC', 'IMPORT_FROM_MODULE_SPEC', 'RELATIVE_MODULE_MARKER'):
            disposition = 'IMPORT_SPECIFICATION'
        elif key not in _LEXICAL_KEYS:
            disposition = 'NON_LEXICAL_LABEL'
        elif port.projection == 'NO_VALUE':
            disposition = 'NO_BINDING_MARKER'
        else:
            basis, name = self.projection(port, state, value.scalar.to_python())
            home = self.home(port, state, name)
            if home is None:
                disposition = 'CONFLICT'
            else:
                operations, flags, cleanup = self.operation(port)
                if not operations:
                    disposition = 'NON_BINDING_ANNOTATION'
                else:
                    if len(self.contributions) >= self.limits[1]:
                        raise DeclarationBudgetExceeded('CONTRIBUTION_BUDGET', index, index)
                    visibility = self.visibility(port.site, state)
                    flags = flags if visibility == 'SYMBOL_VISIT' else ()
                    contribution = len(self.contributions)
                    self.contributions.append(DeclarationContribution(index, port.site, source, home, name,
                        operations, flags, port.role, region, visibility, cleanup))
                    self.scope_contributions[home].append(contribution)
                    self.ports.append(PortDisposition(index, 'CONTRIBUTION', source, region, name, basis, contribution))
                    return
        self.ports.append(PortDisposition(index, disposition, source, region, name, basis, None))

    def position(self, port):
        node = self.syntax.nodes[self.roles.ports[port].site.occurrence]
        cls = self.fields.constructors[node.constructor]
        attrs = {self.fields.catalog.rules[rule].name: value for rule, value in zip(cls.attributes, node.attributes)}
        line = attrs.get('lineno');column = attrs.get('col_offset')
        return (line.scalar.to_python() if line is not None and line.kind == 'SCALAR' else -1,
            column.scalar.to_python() if column is not None and column.kind == 'SCALAR' else -1, port)

    def whole_scope(self):
        for scope, indices in enumerate(self.scope_contributions):
            grouped = defaultdict(list)
            for index in indices:
                grouped[self.contributions[index].name].append(index)
            for name, members in grouped.items():
                visible = [self.contributions[index] for index in members if self.contributions[index].analysis_visibility == 'SYMBOL_VISIT']
                globals_ = [row for row in visible if 'GLOBAL_DIRECTIVE' in row.operations]
                nonlocals = [row for row in visible if 'NONLOCAL_DIRECTIVE' in row.operations]
                params = [row for row in visible if 'PARAMETER_INTERFACE' in row.operations]
                typeparams = [row for row in visible if 'TYPE_PARAMETER_INTERFACE' in row.operations]
                if globals_ and nonlocals:
                    self.conflict('GLOBAL_NONLOCAL_CONFLICT', scope, name, (globals_[0].port, nonlocals[0].port), 'one source scope declares both global and nonlocal')
                for code, rows in (('DUPLICATE_PARAMETER', params), ('DUPLICATE_TYPE_PARAMETER', typeparams)):
                    if len(rows) > 1:
                        self.conflict(code, scope, name, (row.port for row in rows), 'duplicate effective parameter name')
                directives = globals_ + nonlocals
                if params and directives:
                    self.conflict('PARAMETER_DIRECTIVE_CONFLICT', scope, name, (params[0].port, directives[0].port), 'parameter and directive share an effective name')
                annotated = [row for row in visible if 'ANNOTATION_DECLARATION' in row.operations]
                if annotated and directives and self.contexts.scopes[scope].kind not in ('ROOT_EXEC', 'ROOT_INTERACTIVE'):
                    self.conflict('ANNOTATION_DIRECTIVE_CONFLICT', scope, name, (annotated[0].port, directives[0].port), 'simple annotation conflicts with a directive')
                for directive in directives:
                    prior = [row for row in visible if row is not directive and self.position(row.port) < self.position(directive.port)
                        and any(flag in row.visit_flags for flag in ('USE', 'DEF_LOCAL', 'DEF_ANNOT'))]
                    if prior:
                        self.conflict('DIRECTIVE_AFTER_USE_OR_DECLARATION', scope, name, (prior[0].port, directive.port), 'source declaration/use precedes directive')
        # A walrus cannot rebind any enclosing comprehension iterator target.
        iterator_keys = defaultdict(dict)
        for row in self.contributions:
            if 'DEF_COMP_ITER' in row.visit_flags:
                iterator_keys[row.home][row.name] = row.port
        for row in self.contributions:
            if row.reason != 'NAMED_EXPR_BINDING_SYNTAX' or row.home == row.source_scope:
                continue
            scope = row.source_scope
            while scope != row.home:
                if row.name in iterator_keys[scope]:
                    self.conflict('WALRUS_REBINDS_COMPREHENSION_TARGET', row.home, row.name,
                        (iterator_keys[scope][row.name], row.port), 'walrus and comprehension iteration target share an effective name')
                scope = self.contexts.scopes[scope].parent

    def run(self):
        for index, port in enumerate(self.roles.ports):
            self.add_port(index, port)
        self.current = len(self.roles.ports)
        self.whole_scope()
        summaries = []
        for scope, indices in enumerate(self.scope_contributions):
            names = tuple(dict.fromkeys(self.contributions[index].name for index in indices))
            directives = tuple(index for index in indices if any(op.endswith('_DIRECTIVE') for op in self.contributions[index].operations))
            summaries.append(ScopeDeclarations(scope, names, tuple(indices), directives, tuple(self.scope_wildcards[scope])))
        return tuple(summaries)


def _profile():
    from scar.ir.program import source_declarations as declarations_ir
    catalog = _catalog()
    implementation = _digest({name: sha256(Path(path).read_bytes()).hexdigest() for name, path in (
        ('analysis', __file__), ('ir', declarations_ir.__file__))})
    return SourceDeclarationsProfile(_VERSION, _PHASE, catalog.f2_schema_sha256,
        catalog.role_catalog_sha256, catalog.context_catalog_sha256, catalog.content_sha256, implementation)


def _derive(f0, f1, f2, f3b, c1, limits):
    if any(type(n) is not int or n < 0 for n in limits):
        raise ValueError('declaration budgets must be exact nonnegative integers')
    contexts = read_source_contexts(c1, f0, f1, f2, f3b)
    syntax, roles = decode(SourceSyntaxMaterial, f2), decode(SourceRolesMaterial, f3b)
    return _project_declarations(f0, f1, f2, f3b, c1, syntax, roles, contexts, limits)


def _project_declarations(f0, f1, f2, f3b, c1, syntax, roles, contexts,
                          limits=(1_000_000, 1_000_000, 100_000)):
    inputs = SourceDeclarationsInput(_digest(f0), _digest(f1), _digest(f2), _digest(f3b),
        _digest(c1), f0['path'], syntax.mode, contexts.status, len(roles.ports))
    profile = _profile()
    if contexts.status != 'CONTEXTUALIZED':
        return SourceDeclarationsMaterial(inputs, profile, *limits, 'UPSTREAM_NOT_CONTEXTUALIZED',
            (), (), (), (), (), ())
    ledger = _Ledger(syntax, roles, contexts, limits)
    try:
        scopes = ledger.run()
    except DeclarationBudgetExceeded as error:
        return SourceDeclarationsMaterial(inputs, profile, *limits, 'BUDGET_STOPPED',
            (), (), (), (), (), (error.frontier,))
    material = SourceDeclarationsMaterial(inputs, profile, *limits, 'DECLARED',
        tuple(ledger.names), tuple(ledger.ports), tuple(ledger.contributions), scopes,
        tuple(ledger.conflicts), ())
    SourceDeclarationView(roles, contexts, material, contexts=ledger.view).verify()
    return material


def source_declarations_document(f0, f1, f2, f3b, c1, *, max_names=1_000_000,
        max_contributions=1_000_000, max_conflicts=100_000) -> dict:
    return encode(_derive(f0, f1, f2, f3b, c1, (max_names, max_contributions, max_conflicts)))


def read_source_declarations(document, f0, f1, f2, f3b, c1) -> SourceDeclarationsMaterial:
    return _verify_declarations(document, lambda limits: _derive(f0, f1, f2, f3b, c1, limits))


def _verify_declarations(document, rederive):
    material = decode(SourceDeclarationsMaterial, document)
    if material.profile != _profile():
        raise ValueError('C2a declaration implementation/catalog profile mismatch')
    limits = material.max_names, material.max_contributions, material.max_conflicts
    if any(type(n) is not int or n < 0 for n in limits):
        raise ValueError('declaration budgets must be exact nonnegative integers')
    fresh = rederive(limits)
    if material != fresh:
        raise ValueError('C2a projections/contributions differ from original B ports and C1 contexts')
    return material


__all__ = ['source_declaration_catalog', 'source_declarations_document',
    'read_source_declarations', 'SourceDeclarationView']
