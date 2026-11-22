"""Statement completion and pattern interfaces over original F2 statements.

These are conservative child obligations, not completed child lowerings or
proofs of feasible paths. Deferred definitions are atomic here. Only structured
control composes suites; source loops consume their own break/continue ports.
"""
from dataclasses import dataclass

from scar.ir.program import source_productions as P
from scar.ir.program.control import LanguageCompletionKind as C


_NORMAL = (C.FALLTHROUGH, None)
_RAISE = (C.RAISE, None)
_LOOPS = frozenset(('For', 'While', 'AsyncFor'))
_IMPORT_ROLES = ('IMPORT_MODULE_SPEC', 'IMPORT_MEMBER_SPEC', 'IMPORT_FROM_MODULE_SPEC', 'RELATIVE_MODULE_MARKER')
_ATOMIC = frozenset(('Expr', 'Assign', 'AnnAssign', 'AugAssign', 'Delete', 'Import',
    'ImportFrom', 'Assert', 'FunctionDef', 'AsyncFunctionDef', 'ClassDef', 'TypeAlias'))
_SUITES = {'If': ('body', 'orelse'), 'For': ('body', 'orelse'), 'While': ('body', 'orelse'),
    'AsyncFor': ('body', 'orelse'), 'With': ('body',), 'AsyncWith': ('body',),
    'Try': ('body', 'orelse', 'finalbody'), 'TryStar': ('body', 'orelse', 'finalbody')}
_CONSTRUCTORS = _ATOMIC | frozenset(_SUITES) | frozenset(('Pass', 'Global', 'Nonlocal',
    'Return', 'Raise', 'Break', 'Continue', 'Match'))


PATTERN_FIELDS={'MatchValue':('value',),'MatchSingleton':('value',),'MatchSequence':('patterns',),
    'MatchMapping':('keys','patterns','rest'),'MatchClass':('cls','patterns','kwd_attrs','kwd_patterns'),
    'MatchStar':('name',),'MatchAs':('pattern','name'),'MatchOr':('patterns',)}


_CAPTURE_ROLES = ('PATTERN_MAPPING_BINDING_SYNTAX', 'PATTERN_STAR_BINDING_SYNTAX', 'PATTERN_AS_BINDING_SYNTAX')


def _require(condition, path, detail, *, code='STATEMENT_INTERFACE'):
    if not condition:
        from scar.analysis.source_productions import ProductionError
        raise ProductionError(code, str(path), detail)


def _node_items(roles, occurrence, name, *, code='STATEMENT_INTERFACE',
                detail='source field must contain original node references'):
    value = roles.field(occurrence, name)
    _require(value.kind == 'LIST' and all(v.kind == 'NODE' for v in value.items),
        occurrence, detail, code=code)
    return tuple(v.reference for v in value.items)


class SourceStatementInterfaceIndex:
    """Memoized postorder over source references; no AST copies or unrolling."""
    def __init__(self, source):
        self.source = source
        self.nodes = source.context.syntax.nodes
        self.roles = source.roles
        self._summaries = {}
        declared = {row.name for row in source.context.syntax.profile.ast_schema if 'stmt' in row.bases}
        _require(declared == _CONSTRUCTORS, 'constructor domain',
            'finite statement completion rules differ from the original grammar')
        self.constructors = tuple(sorted(declared))

    def active_loop(self, ref):
        # A body field enters its own loop, whereas an orelse field does not.
        if type(ref) is P.SlotRef:
            current = ref.occurrence
            if self.nodes[current].constructor in _LOOPS and self.roles.locate(ref)[0].name == 'body':
                return P.SourceLoopTarget(current)
        else:
            current = ref.occurrence
        while current in self.roles.parents:
            edge = self.roles.parents[current]
            kind = self.nodes[edge.occurrence].constructor
            field = self.roles.locate(edge)[0].name
            if kind in _LOOPS and field == 'body':
                return P.SourceLoopTarget(edge.occurrence)
            if kind in ('FunctionDef', 'AsyncFunctionDef', 'ClassDef', 'Lambda') and field == 'body':
                return None
            current = edge.occurrence
        return None

    def node_items(self, occurrence, name):
        return _node_items(self.roles, occurrence, name)

    def function_docstring(self, occurrence):
        if self.nodes[occurrence].constructor not in ('FunctionDef','AsyncFunctionDef'):
            return None
        return self.declaration_docstring(occurrence)

    def exception_binding(self,occurrence):
        _require(self.nodes[occurrence].constructor=='ExceptHandler',occurrence,'exception binding requires its original handler')
        site=P.SlotRef(occurrence,'FIELD',self.roles.fields['ExceptHandler']['name'])
        value=self.roles.locate(site)[1]
        _require(value.kind=='SCALAR',site,'handler binder requires its original optional name')
        if value.scalar.to_python() is None:return None
        port = self.source.last_role_port(site, ('EXCEPTION_BINDING_SYNTAX',))
        _require(port is not None,site,'handler name lacks its original role port')
        binding=self.source.frontend.bindings.ports[port]
        _require(binding.write_target is not None and binding.declref is not None
            and self.source.frontend.declarations.contributions[binding.declref].cleanup_policy=='CLEAR_ON_HANDLER_EXIT',
            site,'handler target requires its original write access and cleanup declaration')
        return port

    def declaration_docstring(self, occurrence):
        if self.nodes[occurrence].constructor not in ('Module','FunctionDef','AsyncFunctionDef','ClassDef'):
            return None
        body = self.node_items(occurrence,'body')
        if not body or self.nodes[body[0]].constructor != 'Expr':
            return None
        value = self.roles.field(body[0],'value')
        if value.kind != 'NODE' or self.nodes[value.reference].constructor != 'Constant':
            return None
        if type(self.roles.field(value.reference,'value').scalar.to_python()) is not str:
            return None
        return P.SlotRef(value.reference,'FIELD',self.roles.fields['Constant']['value'])

    def docstring_owner(self, statement):
        parent = self.roles.parents.get(statement)
        if parent is None or self.nodes[parent.occurrence].constructor not in ('Module','FunctionDef','AsyncFunctionDef','ClassDef'):
            return None
        return parent if self.roles.locate(parent)[0].name == 'body' and parent.item_path == (0,) and self.declaration_docstring(parent.occurrence) is not None else None

    def ignored_constant_expression(self,occurrence):
        if self.nodes[occurrence].constructor!='Expr' or self.source.roots.interactive_expression(occurrence):return False
        return self.nodes[self.roles.field(occurrence,'value').reference].constructor=='Constant'

    def two_part_slice(self, occurrence):
        if self.nodes[occurrence].constructor != 'Slice':
            return False
        parts = [self.roles.field(occurrence, name) for name in ('lower', 'upper', 'step')]
        constant = all(v.kind != 'NODE' or self.nodes[v.reference].constructor == 'Constant' for v in parts)
        return not constant and parts[2].kind != 'NODE'

    def annotation_publication(self, occurrence, exception_abi):
        return self.source.scope_annotations.publication(occurrence, exception_abi)

    def import_request(self, occurrence, exception_abi, alias=None):
        kind = self.nodes[occurrence].constructor
        _require(kind in ('Import', 'ImportFrom'), occurrence, 'expected original import statement')
        members = self.node_items(occurrence, 'names')
        _require(bool(members), occurrence, 'import requires original aliases')
        if kind == 'Import':
            _require(alias in members, occurrence, 'ordinary import requires one of its original aliases')
            module = P.SlotRef(alias, 'FIELD', self.roles.fields['alias']['name'])
            names, level = None, None
            text = self.roles.field(alias, 'name').scalar.to_python()
            from_names, relative = None, 0
            anchor = alias
        else:
            _require(alias is None, occurrence, 'from-import has one shared request')
            module = P.SlotRef(occurrence, 'FIELD', self.roles.fields[kind]['module'])
            level = P.SlotRef(occurrence, 'FIELD', self.roles.fields[kind]['level'])
            names = tuple(P.SlotRef(a, 'FIELD', self.roles.fields['alias']['name']) for a in members)
            text = self.roles.field(occurrence, 'module').scalar.to_python() or ''
            from_names = tuple(self.roles.field(a, 'name').scalar.to_python() for a in members)
            relative = self.roles.field(occurrence, 'level').scalar.to_python()
            anchor = occurrence
        return P.ImportRequest(module, names, level, self.source.context.entry_context(anchor).environment, exception_abi,
            self.import_compiler_name(module, text), from_names, relative)

    def import_compiler_name(self, site, text):
        """Apply compiler name projection using the accepted original F3 basis.

        This uses the private policy/prefix, not IMPORT_HEAD binder projection:
        a full module request and its binding head are distinct names.
        """
        port = self.source.last_role_port(site, _IMPORT_ROLES)
        basis = None if port is None else self.source.frontend.declarations.ports[port].projection
        _require(basis is not None, site, 'import name has no original F3 projection basis')
        if basis.private_mode == 'NONE':
            return text
        if basis.private_mode == 'SELECTIVE':
            scope = self.source.context.material.scopes[basis.parameter_group]
            _, value = self.roles.locate(scope.anchor)
            allowed = {self.roles.field(v.reference, 'name').scalar.to_python()
                for v in value.items[:basis.prefix_count]}
            if text not in allowed:
                return text
        owner = basis.class_owner.occurrence
        class_name = self.roles.field(owner, 'name').scalar.to_python().lstrip('_')
        return ('_' + class_name + text if class_name and text.startswith('__')
            and not text.endswith('__') and '.' not in text else text)

    def import_member(self, alias, component, exception_abi):
        _require(self.nodes[alias].constructor == 'alias', alias, 'import member requires an original alias')
        parent = self.roles.parents[alias]
        owner = self.nodes[parent.occurrence].constructor
        name = self.roles.field(alias, 'name').scalar.to_python()
        if owner == 'ImportFrom':
            _require(component is None and name != '*', alias, 'from-import requires one named member')
            text = self.import_compiler_name(P.SlotRef(alias, 'FIELD', self.roles.fields['alias']['name']), name)
        else:
            _require(owner == 'Import' and self.roles.field(alias, 'asname').scalar.to_python() is not None
                and type(component) is int and 1 <= component < len(name.split('.')),
                alias, 'dotted as-import requires an original non-head path component')
            text = name.split('.')[component]
        return P.ImportMember(P.SlotRef(alias, 'FIELD', self.roles.fields['alias']['name']), component, exception_abi, text)

    def import_star(self, alias, exception_abi):
        _require(self.nodes[alias].constructor == 'alias'
            and self.roles.field(alias, 'name').scalar.to_python() == '*', alias, 'expected original star alias')
        parent = self.roles.parents[alias]
        _require(self.nodes[parent.occurrence].constructor == 'ImportFrom'
            and self.node_items(parent.occurrence, 'names') == (alias,), alias, 'star import requires one from-import alias')
        context = self.source.context.entry_context(alias)
        _require(self.source.context.material.scopes[context.environment].kind in ('ROOT_EXEC','ROOT_INTERACTIVE'),
            alias, 'star import requires an original module namespace')
        return P.ImportStar(P.SlotRef(alias, 'FIELD', self.roles.fields['alias']['name']), context.environment, exception_abi)

    def groups(self, occurrence):
        kind = self.nodes[occurrence].constructor
        groups = {name: self.node_items(occurrence, name) for name in _SUITES.get(kind, ())}
        if kind in ('Try', 'TryStar'):
            handlers = self.roles.field(occurrence, 'handlers')
            groups.update(('handler:' + str(i), self.node_items(v.reference, 'body'))
                for i, v in enumerate(handlers.items))
        elif kind == 'Match':
            cases = self.roles.field(occurrence, 'cases')
            groups.update(('case:' + str(i), self.node_items(v.reference, 'body'))
                for i, v in enumerate(cases.items))
        return groups

    def sequence(self, occurrences):
        result = {_NORMAL}
        for occurrence in occurrences:
            if _NORMAL not in result:
                break
            result.remove(_NORMAL)
            result.update(self._summaries[occurrence])
        return result

    def completions(self, occurrence):
        _require(type(occurrence) is int and 0 <= occurrence < len(self.nodes), occurrence, 'invalid original occurrence')
        pending = [(occurrence, False)]
        while pending:
            current, ready = pending.pop()
            if current in self._summaries:
                continue
            groups = self.groups(current)
            if not ready:
                pending.append((current, True))
                pending.extend((child, False) for items in reversed(tuple(groups.values())) for child in reversed(items))
                continue
            kind = self.nodes[current].constructor
            suites = {name: self.sequence(items) for name, items in groups.items()}
            if kind in _ATOMIC:
                result = {_NORMAL, _RAISE}
            elif kind in ('Pass', 'Global', 'Nonlocal'):
                result = {_NORMAL}
            elif kind == 'Return':
                result = {(C.RETURN, None)}
                if self.roles.field(current, 'value').kind == 'NODE':
                    result.add(_RAISE)
            elif kind == 'Raise':
                result = {_RAISE}
            elif kind in ('Break', 'Continue'):
                target = self.active_loop(P.ProductionOccurrence(current))
                _require(target is not None, current, 'break/continue has no active source loop')
                result = {(C.BREAK if kind == 'Break' else C.CONTINUE, target)}
            elif kind == 'If':
                result = suites['body'] | suites['orelse'] | {_RAISE}
            elif kind in _LOOPS:
                target = P.SourceLoopTarget(current)
                body = suites['body']
                result = (body - {_NORMAL, (C.BREAK, target), (C.CONTINUE, target)}) | suites['orelse'] | {_RAISE}
                if (C.BREAK, target) in body:
                    result.add(_NORMAL)
            elif kind in ('With', 'AsyncWith'):
                result = suites['body'] | {_RAISE}
                items=self.node_items(current,'items')
                if (_RAISE in suites['body'] or len(items)>1
                        or any(self.roles.field(item,'optional_vars').kind=='NODE' for item in items)):
                    result.add(_NORMAL)  # also suppress target-store or later-enter failures
            elif kind in ('Try', 'TryStar'):
                body = suites['body']
                result = body - {_NORMAL}
                if _NORMAL in body:
                    result |= suites['orelse']
                if _RAISE in body:
                    # Unmatched exceptions and errors in handler admission remain.
                    result |= set().union(*(v for k, v in suites.items() if k.startswith('handler:')))
                    if kind=='TryStar':result.add(_NORMAL)  # native/user split and recomposition may consume the full group
                final = suites['finalbody']
                result = (result if _NORMAL in final else set()) | (final - {_NORMAL})
            elif kind == 'Match':
                # Matching/guards may raise, and no case may match. Feasibility
                # and irrefutability refinement are separate from this interface.
                result = {_NORMAL, _RAISE} | set().union(*suites.values())
            else:
                _require(False, current, 'no finite statement interface for ' + kind)
            self._summaries[current] = frozenset(result)
        rank = {kind: i for i, kind in enumerate(C)}
        return tuple(sorted(self._summaries[occurrence], key=lambda k: (rank[k[0]], -1 if k[1] is None else k[1].occurrence)))


def _require_pattern(condition,path,detail):
    _require(condition,path,detail,code='PATTERN_INTERFACE')


@dataclass(frozen=True,slots=True)
class Capture:
    source: P.SlotRef
    port: int
    name: str


class PatternInterfaceIndex:
    def __init__(self,source):
        self.source=source;self.roles=source.roles;self.nodes=source.context.syntax.nodes
        self._captures={};self._irrefutable={}
        for kind,fields in PATTERN_FIELDS.items():
            _require_pattern(tuple(self.roles.fields[kind])==fields,kind,'pattern field grammar differs')

    def site(self,owner,name,item=None):
        return P.SlotRef(owner,'FIELD',self.roles.fields[self.nodes[owner].constructor][name],() if item is None else (item,))

    def items(self,owner,name):
        return _node_items(self.roles, owner, name, code='PATTERN_INTERFACE',
            detail='pattern field requires original child nodes')

    def children(self,owner):
        kind=self.nodes[owner].constructor
        if kind in ('MatchSequence','MatchMapping','MatchOr'):return self.items(owner,'patterns')
        if kind=='MatchClass':return (*self.items(owner,'patterns'),*self.items(owner,'kwd_patterns'))
        if kind=='MatchAs':
            value=self.roles.field(owner,'pattern')
            return (value.reference,) if value.kind=='NODE' else ()
        return ()

    def capture(self,owner,name):
        site=self.site(owner,name);value=self.roles.locate(site)[1]
        _require_pattern(value.kind=='SCALAR',site,'capture name must be an optional scalar')
        raw=value.scalar.to_python()
        if raw is None:return None
        port = self.source.last_role_port(site, _CAPTURE_ROLES)
        _require_pattern(type(raw) is str and port is not None and self.source.frontend.bindings.ports[port].write_target is not None,
            site,'capture requires its original namespace write port')
        return Capture(site,port,raw)

    def wildcard(self,owner):
        return self.nodes[owner].constructor=='MatchAs' and not self.children(owner) and self.capture(owner,'name') is None

    def sequence(self,owner):
        children=self.items(owner,'patterns');stars=[i for i,c in enumerate(children) if self.nodes[c].constructor=='MatchStar']
        _require_pattern(len(stars)<=1,owner,'multiple starred sequence patterns')
        star=stars[0] if stars else None
        wildcard_star=star is not None and self.capture(children[star],'name') is None
        only=all(self.wildcard(c) or i==star and wildcard_star for i,c in enumerate(children))
        return children,star,wildcard_star,only

    def literal(self,owner):
        kind=self.nodes[owner].constructor
        if kind=='Constant':return self.roles.field(owner,'value').scalar.to_python()
        if kind=='UnaryOp':
            op=self.roles.field(owner,'op').reference
            value=self.literal(self.roles.field(owner,'operand').reference)
            _require_pattern(self.nodes[op].constructor=='USub' and type(value) in (int,float,complex),owner,'pattern unary literal must be numeric minus')
            return -value
        if kind=='BinOp':
            op=self.nodes[self.roles.field(owner,'op').reference].constructor
            left=self.literal(self.roles.field(owner,'left').reference);right=self.literal(self.roles.field(owner,'right').reference)
            _require_pattern(op in ('Add','Sub') and type(left) in (int,float) and type(right) is complex,
                owner,'pattern binary literal must be a real plus/minus imaginary literal')
            return left+right if op=='Add' else left-right
        _require_pattern(False,owner,'pattern values/keys require literals or attribute lookups')

    def captures(self,owner):
        pending=[(owner,False)]
        while pending:
            current,ready=pending.pop()
            if current in self._captures:continue
            kind=self.nodes[current].constructor
            _require_pattern(kind in PATTERN_FIELDS,current,'expected an original pattern node')
            children=self.children(current)
            if not ready:
                pending.append((current,True));pending.extend((c,False) for c in reversed(children));continue
            irrefutable=False
            if kind=='MatchOr':
                _require_pattern(len(children)>1,current,'or pattern requires at least two alternatives')
                captures=self._captures[children[0]];control={c.name:c for c in captures}
                for child in children[1:]:
                    alternative={c.name:c for c in self._captures[child]}
                    _require_pattern(control.keys()==alternative.keys(),current,'alternative patterns bind different names')
                    _require_pattern(all(self.source.frontend.bindings.ports[c.port].write_target==self.source.frontend.bindings.ports[alternative[name].port].write_target
                        for name,c in control.items()),current,'alternative captures resolve to different namespace targets')
                _require_pattern(not any(self._irrefutable[c] for c in children[:-1]),current,'irrefutable alternative must be last')
                irrefutable=self._irrefutable[children[-1]]
            else:
                captures=tuple(capture for child in children for capture in self._captures[child])
                if kind in ('MatchAs','MatchStar','MatchMapping'):
                    capture=self.capture(current,'rest' if kind=='MatchMapping' else 'name')
                    if capture is not None:captures=(*captures,capture)
                if kind=='MatchAs':irrefutable=not children or self._irrefutable[children[0]]
            _require_pattern(len({c.name for c in captures})==len(captures),current,'multiple assignments to the same name in a pattern')
            if kind=='MatchSequence':self.sequence(current)
            elif kind=='MatchStar':
                parent=self.roles.parents.get(current)
                _require_pattern(parent is not None and self.nodes[parent.occurrence].constructor=='MatchSequence',current,'star pattern requires a sequence parent')
            elif kind=='MatchSingleton':
                value=self.roles.field(current,'value').scalar.to_python()
                _require_pattern(value is None or type(value) is bool,current,'singleton must be None/True/False')
            elif kind=='MatchValue':
                value=self.roles.field(current,'value').reference
                if self.nodes[value].constructor!='Attribute':self.literal(value)
            elif kind=='MatchMapping':
                keys=self.items(current,'keys');_require_pattern(len(keys)==len(children),current,'mapping key/pattern lengths differ')
                seen=set()
                for key in keys:
                    if self.nodes[key].constructor=='Attribute':continue
                    value=self.literal(key)
                    _require_pattern(value not in seen,current,'duplicate literal mapping key');seen.add(value)
            elif kind=='MatchClass':
                attrs=self.roles.field(current,'kwd_attrs')
                _require_pattern(attrs.kind=='LIST' and len(attrs.items)==len(self.items(current,'kwd_patterns')),current,'class keyword lengths differ')
                names=tuple(v.scalar.to_python() for v in attrs.items)
                _require_pattern(all(type(n) is str for n in names) and len(set(names))==len(names),current,'repeated or invalid class keyword labels')
            self._captures[current]=captures;self._irrefutable[current]=irrefutable
        return self._captures[owner]

    def irrefutable(self,owner):
        self.captures(owner);return self._irrefutable[owner]

    def action(self,kind,source,abi):
        owner=source.occurrence;constructor=self.nodes[owner].constructor;K=P.PatternActionKind
        allowed={K.SEQUENCE_TEST:('MatchSequence',),K.MAPPING_TEST:('MatchMapping',),K.LENGTH_TEST:('MatchSequence','MatchMapping'),
            K.EQUAL:('MatchValue',),K.SEQUENCE_UNPACK:('MatchSequence',),K.CLASS_EXTRACT:('MatchClass',),
            K.KEYS_EXTRACT:('MatchMapping',),K.TUPLE_UNPACK:('MatchMapping','MatchClass')}
        if kind is K.SEQUENCE_ITEM:
            _require_pattern(type(source) is P.SlotRef and source.group=='FIELD' and len(source.item_path)==1
                and constructor=='MatchSequence' and self.roles.locate(source)[0].name=='patterns',source,'item extraction requires its original sequence item')
            children,star,wild,only=self.sequence(owner)
            _require_pattern(wild and not only and source.item_path[0]!=star and not self.wildcard(children[source.item_path[0]]),source,'item extraction requires the wildcard-star subscript path')
            count=1
        else:
            _require_pattern(type(source) is P.ProductionOccurrence and constructor in allowed.get(kind,()),source,'pattern operation has a different original constructor')
            self.captures(owner)
            count=0 if kind in (K.SEQUENCE_TEST,K.MAPPING_TEST,K.LENGTH_TEST) else 1
            if kind is K.SEQUENCE_UNPACK:
                children,star,wild,only=self.sequence(owner)
                _require_pattern(not wild and not only,source,'unpack requires a non-wildcard extraction path')
                _require_pattern(star is None or star<256 and len(children)-star-1<(((1<<31)-1)>>8),source,'star-unpack exceeds compiler operand limits')
                count=len(children)
            elif kind is K.TUPLE_UNPACK:count=len(self.children(owner))
            elif kind is K.LENGTH_TEST:
                _require_pattern(bool(self.items(owner,'keys')) if constructor=='MatchMapping' else (
                    self.sequence(owner)[1] is None or len(self.children(owner))>1),source,'compiler skips this length test')
        return P.ApplyPatternAction(kind,source,count,abi)
