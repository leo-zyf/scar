"""Source compilation contexts."""

import ast
import dis
import json
from hashlib import sha256
from types import CodeType
from scar.analysis.source_syntax import _restore_ast
from scar.ir.literals import PythonLiteral
from scar.ir.record_codec import encode
from scar.ir.program import source_productions as P
from scar.ir.program.protocol_signatures import ExceptionABI
from dataclasses import fields
from functools import lru_cache


# Static source-to-original-code contexts, without executing target code.
#
# Annotation markers retain the original expressions and their symbol effects.
# The constant-reference path proposes a code owner; exact owner instruction,
# layout, exception, line-table and non-code-constant agreement authorizes it.
# Query code fingerprints are never used as original program identities.
def _json(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)


def _require(condition,path,detail):
    if not condition:
        from scar.analysis.source_productions import ProductionError
        raise ProductionError('COMPILATION_CONTEXT',str(path),detail)


def code_paths(root):
    pending=[((),root)]
    while pending:
        path,unit=pending.pop();yield path,unit
        pending.extend(((*path,i),c) for i,c in enumerate(unit.co_consts) if type(c) is CodeType)


def _constant(value,code=None):
    if type(value) is CodeType:
        _require(code is not None,'constant','code constant needs its original digest resolver')
        return ['CODE',code(value)]
    if type(value) is tuple:return ['TUPLE',[_constant(v,code) for v in value]]
    if type(value) is frozenset:return ['FROZENSET',sorted((_constant(v,code) for v in value),key=_json)]
    if type(value) is slice:return ['SLICE',[_constant(v,code) for v in (value.start,value.stop,value.step)]]
    return ['LITERAL',encode(PythonLiteral.from_python(value,exact=True))]


_LAYOUT=('co_code','co_exceptiontable','co_linetable','co_names','co_varnames','co_cellvars','co_freevars',
    'co_argcount','co_posonlyargcount','co_kwonlyargcount','co_nlocals','co_stacksize','co_flags','co_firstlineno',
    'co_filename','co_name','co_qualname')


def code_digest(unit,memo=None):
    """Canonical compiler values, including IEEE bits; no interning/address IDs."""
    memo={} if memo is None else memo
    if id(unit) not in memo:
        fields={name:(getattr(unit,name).hex() if type(getattr(unit,name)) is bytes else getattr(unit,name)) for name in _LAYOUT}
        fields['constants']=[_constant(v,lambda c:code_digest(c,memo)) for v in unit.co_consts]
        fields['schema']='SCAR_COMPILED_CODE_V1'
        memo[id(unit)]=sha256(_json(fields).encode('ascii')).hexdigest()
    return memo[id(unit)]


def _owner_matches(original,query):
    if any(getattr(original,name)!=getattr(query,name) for name in _LAYOUT):return False
    if len(original.co_consts)!=len(query.co_consts):return False
    for left,right in zip(original.co_consts,query.co_consts):
        if type(left) is CodeType or type(right) is CodeType:
            if not type(left) is type(right) is CodeType:return False
            # Bodies/constants of annotation callbacks are deliberately
            # changed, while their scope, argument and capture layout stays.
            metadata=(name for name in _LAYOUT if name not in (
                'co_code','co_exceptiontable','co_linetable','co_stacksize'))
            if any(getattr(left,name)!=getattr(right,name) for name in metadata):return False
        elif _constant(left)!=_constant(right):return False
    return True


def _marker_values(value):
    if type(value) is bytes:yield value
    elif type(value) in (tuple,frozenset):
        for member in value:yield from _marker_values(member)
    elif type(value) is slice:
        for member in (value.start,value.stop,value.step):yield from _marker_values(member)


class CompilationContextIndex:
    def __init__(self,source):
        self.source=source;self._contexts=None

    def _collect(self):
        source=self.source;syntax=source.context.syntax;roles=source.roles
        publications={}
        for occurrence,node in enumerate(syntax.nodes):
            if node.constructor!='AnnAssign':continue
            pub=source.statements.annotation_publication(occurrence,ExceptionABI.EXCEPTION_WORLD)
            if pub is not None and pub.publication is P.AnnotationPublicationKind.DEFERRED:publications[occurrence]=pub
        if not publications:return {}
        tree=_restore_ast(syntax.root,syntax.nodes);objects={syntax.root:tree};pending=[syntax.root]
        while pending:
            current=pending.pop();node=syntax.nodes[current];obj=objects[current]
            for name,value in zip(roles.fields[node.constructor],node.fields):
                if value.kind=='NODE':objects[value.reference]=getattr(obj,name);pending.append(value.reference)
                elif value.kind=='LIST':
                    for member,child in zip(value.items,getattr(obj,name)):
                        if member.kind=='NODE':objects[member.reference]=child;pending.append(member.reference)
        original=[compile(tree,source.input.path,syntax.mode,dont_inherit=True,optimize=i) for i in (0,1,2)]
        occupied={b for root in original for _,unit in code_paths(root) for c in unit.co_consts for b in _marker_values(c)}
        markers={}
        for occurrence,pub in publications.items():
            suffix=0
            while (marker:=('SCAR-CONTEXT:'+str(occurrence)+':'+str(suffix)).encode()) in occupied:suffix+=1
            occupied.add(marker)
            scope=source.context.context_at_slot(pub.annotation).state.environment;markers[marker]=scope
            value=objects[occurrence].annotation
            mark=ast.copy_location(ast.Constant(value=marker),value)
            objects[occurrence].annotation=ast.copy_location(ast.Tuple(elts=[value,mark],ctx=ast.Load()),value)
        result={}
        for level,root in enumerate(original):
            owners=dict(code_paths(root))
            query=compile(tree,source.input.path,syntax.mode,dont_inherit=True,optimize=level)
            queries=dict(code_paths(query));groups={}
            for path,callback in queries.items():
                scopes={markers[b] for c in callback.co_consts for b in _marker_values(c) if b in markers}
                if not scopes:continue
                _require(len(scopes)==1 and bool(path),path,'marker callback must have one original scope and a constant owner')
                groups.setdefault(scopes.pop(),set()).add(path[:-1])
            memo={}
            for scope,paths in groups.items():
                request=source.scope_annotations.request(scope);variant=request.variants[level]
                registrations={e.registration.index:e.registration for e in variant.entries if e.registration is not None}
                sites=[];unverified=[]
                creator=source.context.material.scopes[scope].parent
                expected='LOAD_DEREF' if source.context.material.scopes[creator].kind=='CLASS' else 'LOAD_NAME'
                for path in sorted(paths):
                    left=owners.get(path);right=queries[path]
                    if left is None or not _owner_matches(left,right):unverified.append(path);continue
                    reference=P.CompiledCodeReference(level,path,code_digest(left,memo),len(left.co_code))
                    ops=[op for op in dis.get_instructions(left) if op.opname not in ('EXTENDED_ARG','NOT_TAKEN')]
                    for i,op in enumerate(ops):
                        if op.opname!='SET_ADD' or op.arg!=1 or i<2 or i+1>=len(ops):continue
                        lookup,index=ops[i-2:i];discard=ops[i+1]
                        if lookup.opname!=expected or lookup.argval!='__conditional_annotations__':continue
                        _require(index.opname in ('LOAD_CONST','LOAD_SMALL_INT') and type(index.argval) is int
                            and index.argval in registrations and discard.opname=='POP_TOP',path,
                            'original registry insertion differs from its source-bound compiler registration')
                        sites.append(P.AnnotationEmissionSite(registrations[index.argval],reference,
                            lookup.offset,index.offset,op.offset,discard.offset))
                status=P.CompilationContextStatus.OWNER_LAYOUT_UNVERIFIED if unverified else P.CompilationContextStatus.VERIFIED
                result[scope,level]=(status,tuple(sites),tuple(unverified))
        return result

    def sites(self,scope,level,*,emitted):
        if self._contexts is None:self._contexts=self._collect()
        row=self._contexts.get((scope,level))
        if row is not None:return row
        return (P.CompilationContextStatus.OWNER_LAYOUT_UNVERIFIED if emitted else
            P.CompilationContextStatus.NO_RETAINED_OWNER,(),())

    def bind(self,parameter,code,offset):
        _require(parameter==self.source.annotation_states.parameter(parameter.publication),'parameter',
            'compiler input differs from its original source domain/context evidence')
        matches={site.registration for variant in parameter.variants for site in variant.producer_sites
            if site.code==code and site.insert_offset==offset}
        _require(len(matches)==1,offset,'exact original code version/path and insertion offset must select one registration')
        return matches.pop()


# Static context interfaces over finite shared source-recipe templates.
#
# No code is executed and no native context is guessed. An edge denotes a
# symbolic environment qualification at a schema-addressed reference site.
# Concrete code/path/offset bindings are a separate compilation query.
def _require_compiler_route(condition, path, detail):
    if not condition:
        from scar.analysis.source_productions import ProductionError
        raise ProductionError('COMPILATION_FLOW', str(path), detail)


@lru_cache(None)
def _field(kind, name):
    return P.CompilationPathPart(P.CompilationPathKind.FIELD,
        tuple(f.name for f in fields(kind)).index(name))


def _at(record, name):
    return (_field(type(record), name),)


def _item(index):
    return P.CompilationPathPart(P.CompilationPathKind.ITEM, index)


_DECLARATIONS = (P.DeferRegion, P.DeferClassBody, P.BuildFunctionAnnotations,
    P.BuildLazyExpression, P.BuildClassAnnotations, P.BuildScopeAnnotations)


def references(instruction):
    """Enumerate entry references, excluding values and completion targets.

    A break/continue target is a control destination, not a new template entry.
    Call argument ProductionParameter.region references likewise are not calls.
    Paths address the exact field/tuple slot in the pinned instruction schema.
    """
    K = P.CompilationEdgeKind
    if type(instruction) is P.EvaluateChild:
        yield (), K.CHILD, instruction
    elif type(instruction) in _DECLARATIONS:
        yield _at(instruction, 'body'), K.CODE_DECLARATION, instruction.body
    elif type(instruction) in (P.ApplyRegion, P.InvokeTypeParameterFactory):
        kind = K.TYPE_FACTORY if type(instruction) is P.InvokeTypeParameterFactory else K.CALL
        yield _at(instruction, 'invocation') + _at(instruction.invocation, 'region'), kind, instruction.invocation.region
    elif type(instruction) is P.BranchControl:
        for name in ('on_true', 'on_false'):
            call = getattr(instruction, name)
            yield _at(instruction, name) + _at(call, 'region'), K.CALL, call.region
    elif type(instruction) is P.LoopControl:
        condition = instruction.condition; path = _at(instruction, 'condition')
        if type(condition) is P.ProductionRegionID:
            yield path, K.LOOP_CONDITION, condition
        elif type(condition) is P.IterationCondition:
            yield path + _at(condition, 'advance') + _at(condition.advance, 'region'), K.LOOP_CONDITION, condition.advance.region
            yield path + _at(condition, 'exhausted'), K.LOOP_EXIT, condition.exhausted
        yield _at(instruction, 'body'), K.LOOP_BODY, instruction.body
        if instruction.otherwise is not None:
            yield _at(instruction, 'otherwise'), K.LOOP_EXIT, instruction.otherwise
    elif type(instruction) is P.HandleCompletions:
        yield _at(instruction, 'protected') + _at(instruction.protected, 'region'), K.CALL, instruction.protected.region
        for index, route in enumerate(instruction.routes):
            yield _at(instruction, 'routes') + (_item(index),) + _at(route, 'handler'), K.COMPLETION_HANDLER, route.handler
    elif type(instruction) is P.SuspendExecution:
        for index, route in enumerate(instruction.routes):
            yield _at(instruction, 'routes') + (_item(index),) + _at(route, 'region'), K.RESUME, route.region


def _uses(plan):
    for step in plan.steps:
        instruction = step.instruction
        if type(instruction) is P.ReadCompilationOption:
            yield P.CompilationFactUse(step.id, instruction.option)
        elif type(instruction) in (P.ReadCompilationAnnotationIndex, P.AddAnnotationIndex):
            # Literal singleton indices still retain the source/compiler fact.
            yield P.CompilationFactUse(step.id, instruction.parameter.publication.annotation)


def _transition(caller, target):
    return (P.CompilationTransition.QUALIFY_SITE if caller.code_scope == target.code_scope
        else P.CompilationTransition.ENTER_CODE)


def _edges(plan, ports, child):
    for step in plan.steps:
        caller = ports[step.region]
        for path, kind, target in references(step.instruction):
            destination = child(target) if type(target) is P.EvaluateChild else target
            context = destination if type(destination) is P.CompilationChildTarget else ports[destination]
            yield P.CompilationContextEdge(step.id, path, kind, destination, _transition(caller, context))


def _entry_scope(instruction):
    if type(instruction) in (P.EnterDeferredBody, P.EnterTypeParameters, P.EnterEvaluationRequest):
        return instruction.environment.scope
    if type(instruction) in (P.EnterRootBody, P.EnterClassBody):
        return instruction.interface.scope
    return None


class CompilationFlowIndex:
    def __init__(self, source):
        self.source = source
        self._code_scopes = {}

    def code_scope(self, scope):
        """Inline scopes share their parent's code unit, not its namespace."""
        topology = self.source.frontend.topology
        chain = []; current = scope
        while current not in self._code_scopes:
            _require_compiler_route(type(current) is int and 0 <= current < len(topology.scopes), scope, 'source scope outside original topology')
            row = topology.scopes[current]
            chain.append(current)
            if row.disposition != 'INLINED':
                self._code_scopes[current] = current
                break
            _require_compiler_route(row.block is not None, scope, 'inlined scope needs its original compiler block')
            block = topology.blocks[row.block]
            parent = block.parent
            while parent is not None and topology.blocks[parent].source_scope is None:
                parent = topology.blocks[parent].parent
            _require_compiler_route(parent is not None, scope, 'inlined scope must have a source code owner')
            current = topology.blocks[parent].source_scope
            _require_compiler_route(current not in chain, scope, 'cyclic inlined code ownership')
        owner = self._code_scopes[current]
        for original in chain:
            self._code_scopes[original] = owner
        return owner

    def build(self, plan):
        regions = {r.id: r for r in plan.regions}; steps = {s.id: s for s in plan.steps}; ports = {}
        for initial in plan.regions:
            chain = []; seen = set(); current = initial.id
            while current not in ports:
                _require_compiler_route(current not in seen, current, 'cyclic region ownership')
                seen.add(current); region = regions[current]; chain.append(region)
                if region.parent is None:
                    break
                current = region.parent
            for region in reversed(chain):
                scope = _entry_scope(steps[region.steps[0]].instruction)
                if scope is None:
                    scope = (ports[region.parent].source_scope if region.parent is not None else region.context.environment)
                ports[region.id] = P.CompilationContextPort(region.id, scope, self.code_scope(scope))

        def child(instruction):
            scope = self.source.context.context_at_slot(instruction.field).state.environment
            return P.CompilationChildTarget(instruction.field, instruction.purpose, scope, self.code_scope(scope))

        return P.CompilationContextFlow(tuple(ports[r.id] for r in plan.regions),
            tuple(_edges(plan, ports, child)), tuple(_uses(plan)))


def validate_flow(plan):
    """Check exact finite reference coverage; original scope proof is separate."""
    flow = plan.compilation_flow
    if flow is None:
        return
    _require_compiler_route(tuple(p.region for p in flow.ports) == tuple(r.id for r in plan.regions), 'ports', 'one ordered context port is required per region')
    ports = {p.region: p for p in flow.ports}
    _require_compiler_route(all(p.source_scope >= 0 and p.code_scope >= 0 for p in flow.ports), 'ports', 'negative scope')
    expected = [(step, path, kind, target) for step in plan.steps for path, kind, target in references(step.instruction)]
    _require_compiler_route(len(flow.edges) == len(expected), 'edges', 'entry reference coverage differs')
    for edge, (step, path, kind, target) in zip(flow.edges, expected):
        _require_compiler_route((edge.step, edge.path, edge.kind) == (step.id, path, kind), step.id, 'reference identity/path/kind differs')
        if type(target) is P.EvaluateChild:
            _require_compiler_route(type(edge.target) is P.CompilationChildTarget and edge.target.field == target.field
                and edge.target.purpose is target.purpose and edge.target.source_scope >= 0 and edge.target.code_scope >= 0,
                step.id, 'external child interface differs')
            destination = edge.target
        else:
            _require_compiler_route(edge.target == target and target in ports, step.id, 'local target differs')
            destination = ports[target]
        _require_compiler_route(edge.transition is _transition(ports[step.region], destination), step.id, 'code environment transition differs')
        if kind not in (P.CompilationEdgeKind.CHILD, P.CompilationEdgeKind.CODE_DECLARATION, P.CompilationEdgeKind.TYPE_FACTORY):
            _require_compiler_route(edge.transition is P.CompilationTransition.QUALIFY_SITE, step.id, 'control edge cannot change its code unit')
    _require_compiler_route(flow.uses == tuple(_uses(plan)), 'uses', 'compilation fact use coverage differs')
