"""Source protocols."""

from collections import defaultdict
from dataclasses import dataclass
from types import CodeType
import dis
from scar.analysis.source_namespace_contents import NamespacePlanView
from scar.analysis.source_references import source_port
from scar.ir.program import normal_graph as T, source_productions as P, normal_graph as G
from scar.ir.program.normal_graph import SourceTemplateKey, CodeScopeWitness
from scar.ir.program.providers import ProviderFactReference
from scar.ir.program.structured import PythonProtocolKind as K
from scar.analysis.source_call_view import (
    prepare_source_ports,
    source_call_site,
    source_active_scope,
)
from scar.analysis.source_declared_calls import _owners
from scar.ir.program.dependencies import DependencyFrontier, DependencyNeighborhood
from scar.ir.program.source_protocols import ProtocolCallFrontier
from scar.ir.program.ids import OperationID, ValueID, UseID, ControlPortID, RegionID
from scar.analysis.source_annotation_code_bridge import _recipe_reference
from scar.ir.program.control import LanguageCompletionKind as C
from scar.ir.program.model import ValueKind, ValueType
from scar.ir.program.source_observations import (
    ObservedSourceRequestCodeLink,
    ObservedSourceRequestKind,
)


# Source protocols.
# Join compiler-inserted CALLs to existing F4 protocol operations.
#
# The full caller code was checked by the code provider. Source spans select
# candidate recipes; compiler markers and the original protocol ABI then select
# an operation. Ordinary Python argument binding is deliberately not involved.
@dataclass(frozen=True, slots=True)
class ProtocolSite:
    kind: str
    definition: ProviderFactReference
    operation: T.GlobalPort
    arguments: tuple[T.GlobalPort, ...]
    result: T.GlobalPort
    world: T.GlobalPort
    result_world: T.GlobalPort
    body: T.Body
    owner: CodeScopeWitness
    rule: str


def protocol_index(index, caller):
    from scar.analysis.source_observed_calls import _span
    key = caller.unit, caller.scope.owner, caller.capture.code.key
    cache = index._protocol_static
    if key in cache: return cache[key]
    instructions = tuple(dis.get_instructions(caller.code, adaptive=False, show_caches=False))
    positions = defaultdict(list)
    for ordinal, instruction in enumerate(instructions):
        if instruction.opname in ('CALL', 'CALL_KW', 'CALL_FUNCTION_EX'):
            positions[tuple(instruction.positions)].append((ordinal, instruction))
    syntax = index._source_syntax(caller.unit)
    wanted = []
    for occurrence, node in enumerate(syntax.nodes):
        if node.constructor not in ('GeneratorExp', 'ClassDef'): continue
        span = _span(syntax, occurrence)
        if span in positions:
            purpose = P.ProductionPurpose.VALUE if node.constructor == 'GeneratorExp' else P.ProductionPurpose.STATEMENT
            wanted.append((SourceTemplateKey(occurrence, purpose), node.constructor, span))
    source = index.view.unit(caller.unit); facts = source.facts
    missing = tuple(t for t, _, _ in wanted if t not in index._contexts[caller.unit])
    for template, plan in source.material.plans(missing).items():
        index._plans[caller.unit][template] = plan
        index._contexts[caller.unit][template] = NamespacePlanView(caller.unit, plan)
    result = defaultdict(list)
    for template, constructor, span in wanted:
        context = index._contexts[caller.unit][template]
        for step in context.plan.steps:
            ins = step.instruction
            if constructor == 'GeneratorExp':
                if not (type(ins) is P.ApplyPythonProtocol and ins.protocol is K.CREATE_GENEXPR): continue
                # This ABI consists of five environment components, captured
                # cells and the initial iterator, followed by the world.
                if len(step.arguments) < 7: raise ValueError('generator creation lost its protocol arguments')
                env = step.arguments[:5]
                if not all(type(v) is P.ProductionResult and v.step == env[0].step and v.index == n
                        for n, v in enumerate(env)):
                    raise ValueError('generator creation changed its original function environment')
                producer = context.steps[env[0].step]
                if not (type(producer.instruction) is P.ApplyPythonProtocol
                        and producer.instruction.protocol is K.MAKE_GENEXPR_FUNCTION):
                    raise ValueError('generator creation lacks its original function constructor')
                declarations = [(n, fn) for n, fn in enumerate(facts.functions)
                    if fn.template == template and source_port(facts, fn.value) == context.result(producer)]
                kind, table = 'CREATE_GENEXPR', 'functions'
            else:
                call = type(ins) is P.ApplyLanguageAction and ins.action is P.LanguageActionKind.CALL
                packed = type(ins) is P.ApplyPythonProtocol and ins.protocol is K.CALL_FUNCTION_EX
                if not (call or packed) or ins.selectors[0].occurrence != template.occurrence: continue
                declarations = [(n, cls) for n, cls in enumerate(facts.classes)
                    if cls.template == template and source_port(facts, cls.value) == context.result(step)]
                kind, table = 'BUILD_CLASS', 'classes'
            owner = context.owner(step.region)
            if owner.owner != caller.scope.owner: continue
            if len(declarations) != 1: raise ValueError('protocol operation needs one original definition fact')
            ordinal, declaration = declarations[0]
            world, after = context.worlds(step)
            if context.value(step.arguments[-1]) != world:
                raise ValueError('protocol changed its original last-world argument')
            definition = index.slicer.fact(facts, table, ordinal)
            site = ProtocolSite(kind, definition, context.site(step),
                tuple(context.value(v) for v in step.arguments[:-1]), context.result(step), world, after,
                T.Body(caller.unit, declaration.template, declaration.body), owner,
                'GENERATOR_CODE_MAKE_FUNCTION_ITERATOR_CALL' if kind == 'CREATE_GENEXPR'
                    else 'CLASS_BUILD_CODE_MAKE_FUNCTION_CALL')
            markers = [n for n, x in enumerate(instructions) if x.opname == 'LOAD_CONST'
                and type(x.argval) is CodeType and tuple(x.positions) == span
                and (kind != 'CREATE_GENEXPR' or x.argval.co_name == '<genexpr>')]
            for n, instruction in positions[span]:
                before = [m for m in markers if m < n
                    and any(x.opname == 'MAKE_FUNCTION' and tuple(x.positions) == span
                        for x in instructions[m + 1:n])]
                if len(before) != 1: continue
                if kind == 'CREATE_GENEXPR':
                    valid = instruction.opname == 'CALL' and instruction.arg == 0 and n > 0 \
                        and instructions[n - 1].opname in ('GET_ITER', 'GET_AITER')
                else:
                    valid = any(x.opname == 'LOAD_BUILD_CLASS' and tuple(x.positions) == span
                        for x in instructions[:before[0]])
                    if call:
                        valid = valid and instruction.opname in ('CALL', 'CALL_KW') \
                            and instruction.arg == len(step.arguments) - 2
                    else: valid = valid and instruction.opname == 'CALL_FUNCTION_EX'
                if valid: result[instruction.offset].append(site)
    cache[key] = dict(result)
    return cache[key]


# Expose retained unresolved method targets through the shared graph.
#
# No unknown operation is invented. The original call/body is already modeled;
# its missing target/class/descriptor selection is a field-level obligation.
class SourceProtocolFrontierProvider:
    def __init__(self, view, materials):
        if view.catalog is None or (view.catalog.path,view.catalog.report)!=(materials.catalog.path,materials.catalog.report):
            raise ValueError('protocol frontiers must share their original source catalog')
        self.view=view; self.materials=materials; self.records={}; self.by_unit=defaultdict(list)
        self.attached=set(); self.points=defaultdict(list); self.scopes={}
        if materials.members is not None:
            for i, _ in enumerate(materials.members.frontiers):
                context, call, evidence, reason=materials.member_frontier(i)
                row=ProtocolCallFrontier(i,context,call,evidence,reason)
                self.records[i]=row; self.by_unit[context.body.unit].append(i)
        self.available_units=frozenset(self.by_unit)
        view.register(self)

    @staticmethod
    def _key(point): return point.namespace,point.node

    def attach(self, unit):
        if unit in self.attached: return
        indices=self.by_unit.get(unit,()); source=self.view.unit(unit); facts=source.facts
        ports=[]; bodies=[]
        for i in indices:
            row=self.records[i]
            if row.call is None: bodies.append(row.context.body)
            else:
                call=row.call.call
                ports.extend(source_port(facts,p) for p in (call.callee,call.result,call.world,call.result_world))
        prepare_source_ports(self.view,ports,bodies=bodies)
        owners=_owners(facts) if any(self.records[i].call is not None for i in indices) else ()
        for i in indices:
            row=self.records[i]
            if row.call is None:
                body=row.context.body; points=(self.view.source_region(unit,body.template,body.region),)
                function=next(f for f in facts.functions if source_port(facts,f.value)==row.context.source_callable)
                self.scopes[i]=(function.body_scope.owner,)
            else:
                call=row.call.call; ordinal=facts.calls.index(call); self.scopes[i]=owners[ordinal]
                # Resolve its actual source operation/callee Use, without
                # inventing an invocation or selecting a target body.
                from types import SimpleNamespace
                site=SimpleNamespace(result=source_port(facts,call.result),callee=source_port(facts,call.callee))
                operation,use,callee=source_call_site(self.view,site)
                points=(operation,use,callee,*(self.view.source_port(source_port(facts,p))
                    for p in (call.result,call.world,call.result_world)))
            for point in points:
                key=self._key(point)
                if i not in self.points[key]: self.points[key].append(i)
        self.attached.add(unit); self.materials.check()

    def at(self, point):
        if type(point.node) not in (OperationID,ValueID,UseID,ControlPortID,RegionID): return ()
        source=self.view.namespaces.get(point.namespace); unit=getattr(source,'unit',None)
        if unit not in self.available_units: return ()
        if unit not in self.attached: self.attach(unit)
        indices=self.points.get(self._key(point),())
        if not indices: return ()
        self.materials.check(); self.view.get(point)
        active=source_active_scope(self.view,point.namespace,point.context)
        return tuple(self.records[i] for i in indices if active is None or active in self.scopes[i])

    def _neighbors(self, point):
        rows=self.at(point)
        frontiers=tuple(DependencyFrontier(point,row.reason,
            'Original member context '+str(row.ordinal)+'; target/class/descriptor selection remains unresolved at '
            +row.evidence.artifact.uri+row.evidence.locator) for row in rows)
        return DependencyNeighborhood(frontiers=frontiers)

    def incoming(self, point): return self._neighbors(point)
    def outgoing(self, point): return self._neighbors(point)

    def summary(self):
        return dict(declared=len(self.records),available_units=len(self.available_units),units=len(self.attached),
            attached_records=len({i for values in self.points.values() for i in values}),
            endpoint_bindings=sum(map(len,self.points.values())),unresolved_targets_resolved=0,
            unknown_nodes_created=0,runtime_dispatch_proven=False)

    def publish(self, out):
        from pathlib import Path
        from scar.artifacts import write_table
        from scar.artifacts import _durable_bytes, _durable_mkdir
        from scar.artifacts import _json
        from scar.analysis.source_value_flow import _artifact
        from scar.ir.record_codec import encode
        out=Path(out).absolute()
        if out.exists(): raise ValueError('fresh frontier-view output required')
        self.materials.check(); _durable_mkdir(out); shards={}
        for unit,indices in sorted(self.by_unit.items()):
            folder=out/'units'/unit; _durable_mkdir(folder)
            table=write_table(folder/'frontiers.jsonl.gz',(self.records[i] for i in indices))
            shards[unit]=dict(path='units/'+unit,table=table)
        report=dict(schema='scar.source-protocol-frontiers',schema_version=1,shards=shards,
            parent=encode(_artifact(self.materials.member_path/'result.json',self.materials.member_report['schema']))
                if self.materials.member_path is not None else None,
            summary=self.summary(),target_executed=False,first_stage_complete=False)
        self.materials.check(); _durable_bytes(out/'result.json',_json(report),replace=False)
        return report


# Read a deferred request ABI from its actual sealed declaration and region.
#
# These inputs are region parameters, not ordinary Python function formals.
# No format value, callable creator, globals or cell binding is selected here.
def request_callback_abi(view, request):
    """Return the original (env, format, world), (value, world) graph ports."""
    if type(request) is not ObservedSourceRequestCodeLink:
        raise TypeError('request callback ABI requires its independent code correspondence')
    body = request.body; source = view.unit(body.unit); material = source.material
    node = material.interfaces['nodes'][body.template]
    if node.origin.plan_sha256 != request.plan_sha256:
        raise ValueError('request callback declaration belongs to another sealed plan')
    plan = material.plans((body.template,))[body.template]
    context = NamespacePlanView(body.unit, plan)
    declarations = {
        ObservedSourceRequestKind.FUNCTION_ANNOTATIONS:P.BuildFunctionAnnotations,
        ObservedSourceRequestKind.CLASS_ANNOTATIONS:P.BuildClassAnnotations,
        ObservedSourceRequestKind.MODULE_ANNOTATIONS:P.BuildScopeAnnotations,
    }
    candidates = [(i, step) for i, step in enumerate(plan.steps) if step.id == request.declaration_step]
    if len(candidates) != 1:
        raise ValueError('request callback lost its original declaration step')
    index, declaration = candidates[0]; instruction = declaration.instruction
    if (type(instruction) is not declarations[request.kind] or instruction.body != body.region
            or instruction.annotations.scope != request.scope.source_scope
            or context.result(declaration) != request.callable
            or _recipe_reference(material, body.template, index) != request.declaration):
        raise ValueError('request callback differs from its real sealed recipe declaration')
    region = context.regions[body.region]
    if (region.kind is not P.ProductionRegionKind.DEFERRED or not region.steps
            or context.owner(region.id) != request.scope):
        raise ValueError('request callback requires its original deferred region and code scope')
    first = context.steps[region.steps[0]]
    parameters = tuple(P.ProductionParameter(region.id, i) for i in range(3))
    reference = ValueType(ValueKind.REFERENCE); world = request.environment.world
    if (type(world) is not ValueType or world.kind is not ValueKind.STATE
            or region.parameters != (reference, reference, world)
            or type(first.instruction) is not P.EnterEvaluationRequest
            or first.instruction.environment != request.environment
            or request.environment.scope != request.scope.source_scope
            or first.arguments != parameters or len(first.completions) != 1
            or first.completions[0].kind is not C.FALLTHROUGH
            or first.completions[0].types != (world,)):
        raise ValueError('request callback original environment/format/world entry ABI differs')
    returns = [completion for completion in region.completions if completion.kind is C.RETURN]
    if len(returns) != 1 or returns[0].target is not None or returns[0].types != (reference, world):
        raise ValueError('request callback requires its original value/world RETURN completion')
    inputs = tuple(context.value(parameter) for parameter in parameters)
    outputs = tuple(context.gp(context.emitter.port(region.id, G.PortKind.RESULT, i, returns[0]))
        for i in range(2))
    return inputs, outputs[0], outputs[1]
