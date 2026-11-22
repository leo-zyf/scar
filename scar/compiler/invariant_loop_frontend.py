"""Original-F4 proof of a single array carry with invariant external inputs.

The carry role comes from the original lexical write target, not its name or
argument position. The leaf's own value DAG remains the existing ArrayRegion.
This source proof does not discharge ndarray, global identity or concurrency
guards and is not an arbitrary boolean credential for loop lowering.
"""
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
import json
import re

from scar.analysis.program_view import ProgramView
from scar.analysis.source_body_interfaces import BodyInterfaceResolver
from scar.analysis.source_references import source_port
from scar.compiler.graph_frontend import (discover_graph_regions, _source_identity,
    _metadata, _origin, GraphRegionRejected, _reject, _ENV, _WORLD)
from scar.compiler.loop_frontend import _LoopLowerer, _Parameter, _Range, _Iterator, _Carry, _RANGE, _ITEM
from scar.compiler.pipeline_frontend import _global_functions, _Callee
from scar.compiler.model import ArrayRegion, SourceRef, region_from_dict
from scar.ir.program import source_productions as P, normal_graph as F, normal_graph as G
from scar.ir.program.normal_graph import Body
from scar.ir.program.control import LanguageCompletionKind as C
from scar.ir.record_codec import encode
from scar.trace.code_snapshot import code_key


@dataclass(frozen=True)
class InvariantLoopProof:
    id: str
    source: SourceRef
    parameters: tuple[str, ...]
    body: ArrayRegion
    carry_parameter: str
    trip_count_parameter: str
    parameter_map: dict[str, str]
    leaf_global_name: str
    origins: tuple[dict, ...]

    @property
    def carry_outer_parameter(self):
        return self.parameter_map[self.carry_parameter]


@dataclass(frozen=True)
class _PendingLeafResult:
    outer_parameters: tuple[int, ...]


class _InvariantLoopLowerer(_LoopLowerer):
    def binding(self, context, step, args):
        if (step.instruction.action is P.BindingActionKind.WRITE and args
                and type(args[0]) is _PendingLeafResult):
            if self.phase != 'BODY' or self.carry_writes:
                _reject('INVARIANT_LOOP_EXTRA_CARRY_WRITE', 'one body result carry write required')
            port = context.value(step.arguments[0]).port
            namespaces = tuple(dict.fromkeys(self.names.get((port, True), ())))
            if len(namespaces) != 1 or namespaces[0].kind is not F.NamespaceKind.LEXICAL:
                _reject('INVARIANT_LOOP_EXTERNAL_WRITE', 'body result must update one original lexical binding')
            target = namespaces[0]
            positions = [i for i, namespace in self.parameter_namespaces.items() if namespace == target]
            if len(positions) != 1 or positions[0] == self.roles[1]:
                _reject('INVARIANT_LOOP_CARRY_TARGET', 'body result must update an original non-trip parameter')
            carry = positions[0]; actuals = args[0].outer_parameters
            if actuals.count(carry) != 1:
                _reject('INVARIANT_LOOP_CARRY_MULTIPLICITY', 'exactly one leaf formal must receive the carried parameter')
            self.roles = carry, self.roles[1]
            self.parameter_map_indices = actuals
            self.carry_formal_index = actuals.index(carry)
            self.carry_writes += 1; self.locals[target] = _Carry(carry, True)
            self.origins.append(dict(_origin(context, step.region, step), role='lexical_carry_write',
                binding=encode(target), outer_parameter_index=carry, leaf_formal_index=self.carry_formal_index))
            return (_WORLD,)
        return super().binding(context, step, args)

    def action(self, context, step, args, origin):
        if step.instruction.action is not P.LanguageActionKind.CALL:
            _reject('INVARIANT_LOOP_EXTRA_ACTION', 'only exact range and admitted array leaf CALL supported', origin)
        fact = self.call_facts.get(context.result(step).port)
        if (fact is None or fact.keywords or fact.positional_pack is not None or fact.keyword_pack is not None
                or len(args) != len(fact.positional) + 2 or args[-1] is not _WORLD):
            _reject('INVARIANT_LOOP_CALL_ARGUMENTS', 'original fixed positional CALL arguments required', origin)
        actual = tuple(context.value(value).port for value in step.arguments)
        expected = tuple(source_port(self.facts, i).port for i in (fact.callee, *fact.positional, fact.world))
        if (actual != expected or context.result(step, 1).port != source_port(self.facts, fact.result_world).port):
            _reject('INVARIANT_LOOP_CALL_ABI', 'callee/arguments/results/world differ from original F4', origin)
        if args[0] is _RANGE:
            if self.phase != 'PRE' or self.range_calls or len(args) != 3 or type(args[1]) is not _Parameter:
                _reject('INVARIANT_LOOP_RANGE', 'one range(original trip parameter) required', origin)
            self.range_calls += 1
            self.origins.append(dict(origin, role='builtin_range', trip_parameter_index=args[1].index,
                runtime_obligation='GLOBAL_THEN_BUILTINS_LOOKUP_IS_EXACT_BUILTIN_RANGE'))
            return (_Range(args[1].index), _WORLD)
        if self.phase != 'BODY' or self.body_calls or type(args[0]) is not _Callee:
            _reject('INVARIANT_LOOP_BODY_CALL', 'one original direct global array leaf CALL per iteration required', origin)
        callee = args[0]; body = callee.plan; actuals = args[1:-1]
        if (type(body) is not ArrayRegion or len(body.parameters) < 2 or len(actuals) != len(body.parameters)
                or any(type(value) is not _Parameter or value.index == self.roles[1] for value in actuals)):
            _reject('INVARIANT_LOOP_ARRAY_ARGUMENTS', 'leaf arrays must come directly from non-trip outer parameters', origin)
        signature = callee.function.environment.parameters
        if (len(signature) != len(body.parameters) or any(parameter.kind not in
                (P.DeferredParameterKind.POSITIONAL_ONLY, P.DeferredParameterKind.POSITIONAL_OR_KEYWORD)
                or parameter.default_source is not None for parameter in signature)):
            _reject('INVARIANT_LOOP_LEAF_SIGNATURE', 'leaf has unsupported keyword/default/variadic parameters', origin)
        leaf_context = self.resolver.plan(self.source.unit, callee.function.template)
        if any(type(row.instruction) is P.ApplyDecorator for row in leaf_context.plan.steps):
            _reject('INVARIANT_LOOP_LEAF_DECORATOR', 'decorated array leaf is unsupported', origin)
        self.body_calls += 1; self.selected_body = callee
        self.origins.append(dict(origin, role='invariant_array_leaf_call', leaf_region_id=body.id,
            call_callee=encode(source_port(self.facts, fact.callee)),
            call_result=encode(source_port(self.facts, fact.result)),
            call_inputs=[encode(source_port(self.facts, i)) for i in fact.positional],
            outer_parameter_indices=[value.index for value in actuals]))
        return (_PendingLeafResult(tuple(value.index for value in actuals)), _WORLD)

    def control(self, context, step, args, origin):
        ins = step.instruction
        if (self.phase != 'PRE' or self.loops or type(ins.condition) is not P.IterationCondition
                or ins.invariant_count != 1 or ins.condition.world_parameter != 1
                or len(args) != 2 or type(args[0]) is not _Iterator or args[1] is not _WORLD):
            _reject('INVARIANT_LOOP_CONTROL', 'one original iterator/world range loop required', origin)
        self.loops += 1; self.roles = None, args[0].range.trip
        if ins.condition.advance.arguments != tuple(P.ProductionParameter(step.region, i) for i in range(2)):
            _reject('INVARIANT_LOOP_ADVANCE', 'advance must use original iterator/world parameters', origin)
        exhausted = context.regions[ins.condition.exhausted]
        steps = [context.steps[s] for s in exhausted.steps]
        if (len(steps) != 2 or type(steps[0].instruction) is not P.ApplyPythonProtocol
                or steps[0].instruction.protocol.value != 'MATCH_STOP_ITERATION'
                or steps[0].arguments != tuple(P.ProductionParameter(exhausted.id, i) for i in range(2))
                or type(steps[1].instruction) is not P.TransferCompletion
                or steps[1].instruction.kind is not C.FALLTHROUGH
                or steps[1].arguments != (P.ProductionResult(steps[0].id, 0),)):
            _reject('INVARIANT_LOOP_EXHAUSTION', 'original StopIteration classifier required', origin)
        if ins.otherwise is not None:
            otherwise = context.regions[ins.otherwise]; items = [context.steps[s] for s in otherwise.steps]
            if (len(items) != 1 or type(items[0].instruction) is not P.TransferCompletion
                    or items[0].instruction.kind is not C.FALLTHROUGH
                    or items[0].arguments != (P.ProductionParameter(otherwise.id, 1),)):
                _reject('INVARIANT_LOOP_ELSE', 'nonempty loop else is unsupported', origin)
        self.phase = 'CONDITION'; kind, condition = self.run(context, ins.condition.advance.region, args)
        if kind is not C.FALLTHROUGH or condition != (_ITEM, _WORLD) or self.next_calls != 1:
            _reject('INVARIANT_LOOP_CONDITION', 'advance must fetch only the original next range item', origin)
        self.phase = 'BODY'; kind, result = self.run(context, ins.body, (*args, _ITEM))
        carry = self.roles[0]
        if (carry is None or kind is not C.FALLTHROUGH or result != (_WORLD,)
                or self.body_calls != 1 or self.carry_writes != 1 or self.item_writes != 1
                or self.locals[self.parameter_namespaces[carry]] != _Carry(carry, True)):
            _reject('INVARIANT_LOOP_CARRY', 'one ignored loop item and one original leaf-result carry write required', origin)
        self.phase = 'POST'
        self.origins.append(dict(origin, role='single_shared_invariant_loop',
            iterator_parameter=encode(context.gp(context.emitter.port(step.region, G.PortKind.PARAMETER, 0))),
            iteration_condition=encode(ins.condition), body_region=ins.body.wire,
            lexical_carry_binding=encode(self.parameter_namespaces[carry]),
            backedge=dict(from_region=ins.body.wire, to_region=step.region.wire,
                world_parameter_index=1, explicit_array_phi=False),
            body_materializations=1, loop_unrolled=False))
        return (_WORLD,)

    def lower(self):
        function = self.function
        if (len(function.parameter_names) < 3 or function.body_scope is None
                or function.environment.kind is not P.DeferredBodyKind.FUNCTION
                or function.environment.captures or function.environment.owned_cells
                or len(function.environment.parameters) != len(function.parameter_names)
                or any(parameter.kind not in (P.DeferredParameterKind.POSITIONAL_ONLY,
                    P.DeferredParameterKind.POSITIONAL_OR_KEYWORD) or parameter.default_source is not None
                    for parameter in function.environment.parameters)):
            _reject('INVARIANT_LOOP_SIGNATURE', 'ordinary fixed explicit parameters without captures/defaults required')
        context = self.resolver.plan(self.source.unit, function.template)
        if any(type(step.instruction) is P.ApplyDecorator for step in context.plan.steps):
            _reject('INVARIANT_LOOP_DECORATOR', 'decorated outer loop function unsupported')
        for index, _ in enumerate(function.parameter_names):
            formal = source_port(self.facts, function.formals[index+1]); names = self.names.get((formal.port, True), ())
            if len(names) != 1 or names[0].kind is not F.NamespaceKind.LEXICAL:
                _reject('INVARIANT_LOOP_PARAMETER_BINDING', 'formal lacks one original lexical binding')
            self.parameter_namespaces[index] = names[0]; self.locals[names[0]] = _Parameter(index)
        kind, result = self.run(context, function.body,
            (_ENV, *(_Parameter(i) for i in range(len(function.parameter_names))), _WORLD))
        if (self.loops != 1 or self.range_calls != 1 or self.iterator_gets != 1 or self.selected_body is None
                or self.roles[0] is None or kind is not C.RETURN or result != (_Carry(self.roles[0], True), _WORLD)):
            _reject('INVARIANT_LOOP_RETURN', 'one shared range loop followed by original carry return required')
        carry, trip = self.roles
        expected = Counter({('PRE', self.parameter_namespaces[trip]): 1,
            ('POST', self.parameter_namespaces[carry]): 1})
        expected.update(('BODY', self.parameter_namespaces[i]) for i in self.parameter_map_indices)
        if self.reads != expected:
            _reject('INVARIANT_LOOP_EXTRA_READ', 'only original trip, leaf actuals and final carry reads supported')
        self.origins.append(dict(kind='invariant_loop_runtime_obligations',
            builtin_range='EXACT_BUILTIN_RANGE', trip_count='EXACT_POSITIVE_INT_OR_ORIGINAL_FALLBACK',
            leaf_lookup='OUTER_FUNCTION_GLOBALS_EXACT_SOURCE_CODE_AND_SIGNATURE',
            arrays='EXACT_FINITE_FLOAT64_NDARRAYS_NO_FOREIGN_WRITERS',
            reference_stability_does_not_prove_content_invariance=True))
        body = self.selected_body.plan
        mapping = dict(zip(body.parameters, (function.parameter_names[i] for i in self.parameter_map_indices)))
        return body, body.parameters[self.carry_formal_index], function.parameter_names[trip], mapping, self.selected_body.name, tuple(self.origins)


def discover_invariant_loops(catalog, source=None):
    """Discover original module-level range/carry/leaf bindings without execution."""
    view = catalog if isinstance(catalog, ProgramView) else ProgramView(catalog)
    leaves, _ = discover_graph_regions(view, source); by_source = {}
    for leaf in leaves:
        identity = next(row for row in leaf.source.origins if row.get('kind') == 'source_identity')
        by_source[identity['unit'], identity['occurrence']] = leaf
    units = tuple(view.catalog.fact_store.rows) if source is None else tuple(sorted(
        view.catalog.units_for_source(str(Path(source).absolute())) or ()))
    if source is not None and not units:
        return [], [dict(path=str(Path(source).absolute()), code='SOURCE_NOT_IN_CATALOG')]
    loops = []; rejected = []; resolver = BodyInterfaceResolver(view)
    for unit in units:
        original = view.unit(unit); functions = original.facts.functions
        resolver.prepare_many(Body(unit, fn.template, fn.body) for fn in functions)
        try: model, syntax, codes, recipe, digest = _source_identity(view.catalog, unit)
        except (ValueError, KeyError, SyntaxError, OSError) as error:
            rejected.append(dict(unit=unit, code='INVARIANT_LOOP_SOURCE_IDENTITY', detail=str(error))); continue
        selected = {fn.template:by_source[unit, fn.template.occurrence] for fn in functions
            if (unit, fn.template.occurrence) in by_source}
        globals = _global_functions(original.facts, selected)
        module_functions = {candidate.function.template for candidates in
            _global_functions(original.facts, {fn.template:fn for fn in functions}).values() for candidate in candidates}
        for ordinal, function in enumerate(functions):
            if function.template not in module_functions or function.template in selected: continue
            name, span, lines = _metadata(syntax, function.template.occurrence)
            candidates = [code for code in codes if code.co_name == name and code.co_firstlineno in lines
                and code.co_varnames[:code.co_argcount+code.co_kwonlyargcount] == function.parameter_names]
            try:
                if len(candidates) != 1: _reject('INVARIANT_LOOP_CODE_IDENTITY', 'unique original CodeID unavailable')
                body, carry, trip, mapping, global_name, origins = _InvariantLoopLowerer(
                    original, function, resolver, globals, syntax=syntax).lower()
                code = candidates[0]; key = code_key(code)
                identity = dict(kind='source_identity', method='COMPILE_ONLY_RETAINED_F0', unit=unit,
                    occurrence=function.template.occurrence, span=span, recipe=recipe, code_key=key,
                    semantic_authority='ORIGINAL_F4_PRODUCTION_PLANS_AND_UNIT_FACTS')
                reference = SourceRef(model['path'], digest, code.co_qualname, span['line'], span['column'], key, (identity,))
                identifier = 'invariant-array-loop:' + sha256((unit+'\0'+function.body.wire+'\0'+key).encode()).hexdigest()[:24]
                loops.append(InvariantLoopProof(identifier, reference, function.parameter_names,
                    body, carry, trip, mapping, global_name, origins))
            except GraphRegionRejected as error:
                rejected.append(dict(unit=unit, function=ordinal, path=model['path'],
                    qualname=candidates[0].co_qualname if len(candidates)==1 else name,
                    code=error.code, detail=error.detail, origin=error.origin))
    return loops, rejected


PROOF_SCHEMA = 'scar.invariant-loop-proof'


def validate_invariant_loop_proof(proof):
    """Validate the retained shape/bytes, never replace original F4 replay."""
    from scar.compiler.validation import validate_region
    if type(proof) is not InvariantLoopProof:
        raise ValueError('typed original invariant loop proof required')
    validate_region(proof.body)
    if (type(proof.id) is not str or not proof.id.startswith('invariant-array-loop:')
            or type(proof.parameters) is not tuple or len(proof.parameters) < 3
            or len(set(proof.parameters)) != len(proof.parameters)
            or any(type(name) is not str or not name.isidentifier() for name in proof.parameters)
            or proof.trip_count_parameter not in proof.parameters
            or proof.carry_parameter not in proof.body.parameters
            or type(proof.parameter_map) is not dict or set(proof.parameter_map) != set(proof.body.parameters)
            or any(value not in proof.parameters or value == proof.trip_count_parameter for value in proof.parameter_map.values())
            or type(proof.leaf_global_name) is not str or not proof.leaf_global_name.isidentifier()):
        raise ValueError('original outer/leaf parameter mapping required')
    if list(proof.parameter_map.values()).count(proof.carry_outer_parameter) != 1:
        raise ValueError('carried outer parameter must map to exactly one leaf formal')
    identities=[]
    for source in (proof.source, proof.body.source):
        if (type(source) is not SourceRef or not Path(source.path).is_absolute()
                or re.fullmatch('[0-9a-f]{64}', source.sha256) is None
                or re.fullmatch('pycode-v2:[0-9a-f]{64}', source.code_key) is None
                or sha256(Path(source.path).read_bytes()).hexdigest() != source.sha256):
            raise ValueError('original source bytes/code identity changed')
        candidates=[row for row in source.origins if row.get('kind')=='source_identity'
            and row.get('method')=='COMPILE_ONLY_RETAINED_F0'
            and row.get('semantic_authority')=='ORIGINAL_F4_PRODUCTION_PLANS_AND_UNIT_FACTS'
            and row.get('code_key')==source.code_key]
        if len(candidates)!=1:raise ValueError('original F0 identity with F4 semantic provenance required')
        identities.append(candidates[0])
    if (proof.source.path,proof.source.sha256,identities[0]['unit']) != (
            proof.body.source.path,proof.body.source.sha256,identities[1]['unit']):
        raise ValueError('loop and direct global leaf must share the original unit')
    witnesses={}
    for role in ('builtin_range','lexical_carry_write','invariant_array_leaf_call','single_shared_invariant_loop'):
        candidates=[row for row in proof.origins if row.get('role')==role
            and row.get('kind')=='original_f4' and row.get('unit')==identities[0]['unit']]
        if len(candidates)!=1:raise ValueError('missing unique original F4 loop witness: '+role)
        witnesses[role]=candidates[0]
    carry=proof.parameters.index(proof.carry_outer_parameter)
    actuals=[proof.parameters.index(proof.parameter_map[name]) for name in proof.body.parameters]
    if (witnesses['builtin_range']['trip_parameter_index']!=proof.parameters.index(proof.trip_count_parameter)
            or witnesses['lexical_carry_write']['outer_parameter_index']!=carry
            or witnesses['lexical_carry_write']['leaf_formal_index']!=proof.body.parameters.index(proof.carry_parameter)
            or witnesses['invariant_array_leaf_call']['outer_parameter_indices']!=actuals
            or witnesses['invariant_array_leaf_call']['leaf_region_id']!=proof.body.id
            or len(witnesses['invariant_array_leaf_call']['call_inputs'])!=len(proof.body.parameters)
            or witnesses['single_shared_invariant_loop']['loop_unrolled'] is not False):
        raise ValueError('retained F4 witnesses differ from the declared loop interface')
    return proof


def invariant_loop_proof_document(proof):
    return dict(schema=PROOF_SCHEMA,schema_version=1,**asdict(validate_invariant_loop_proof(proof)))


def write_invariant_loop_proof(proof,path):
    """Store the original body once; the file itself grants no permission."""
    path=Path(path).absolute();document=invariant_loop_proof_document(proof)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(document,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    from scar.ir.execution_recipe import artifact_reference
    return artifact_reference(path,PROOF_SCHEMA)


def read_invariant_loop_proof(path,*,catalog):
    """Read only after replaying the same proof through an admitted F4 catalog.

    Original IDs/provenance plus matching source bytes are not sufficient by
    themselves: a forged marker or mapping must fail original graph replay.
    The caller supplies its existing ProgramView to reuse one source session.
    """
    from scar.ir.execution_recipe import artifact_reference,read_artifact
    reference=artifact_reference(path,PROOF_SCHEMA);document=read_artifact(reference)
    fields={'schema','schema_version',*InvariantLoopProof.__dataclass_fields__}
    if set(document)!=fields or document['schema_version']!=1 or type(document['schema_version']) is not int:
        raise ValueError('unsupported original invariant loop proof fields/version')
    data={name:document[name] for name in InvariantLoopProof.__dataclass_fields__}
    source=dict(data['source']);source['origins']=tuple(source.get('origins',()))
    data.update(source=SourceRef(**source),body=region_from_dict(data['body']),
        parameters=tuple(data['parameters']),origins=tuple(data['origins']))
    proof=validate_invariant_loop_proof(InvariantLoopProof(**data))
    original,_=discover_invariant_loops(catalog,source=proof.source.path)
    canonical=lambda value:json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
    matches=[value for value in original if value.id==proof.id and
        canonical(invariant_loop_proof_document(value))==canonical(document)]
    if len(matches)!=1:raise ValueError('invariant loop proof differs from original F4 rediscovery')
    read_artifact(reference,json_document=False)
    return matches[0]


def canonical_invariant_motion_interface(proof):
    """Canonical source boundary, excluding lowering/runtime specializations.

    This is only a projection of a typed proof. Its caller still owns original
    F4 rediscovery; a serialized proof or this dictionary grants no authority.
    """
    if type(proof) is not InvariantLoopProof:
        raise TypeError('original typed invariant loop proof required')
    return dict(carry_parameter=proof.carry_parameter,
        readonly_parameters=[name for name in proof.body.parameters if name!=proof.carry_parameter],
        original_f4_proof=asdict(proof))


def _motion_origin_entry(origin):
    """Recover the original ENTER from its retained typed result coordinate."""
    from scar.ir.record_codec import decode
    from scar.ir.program.normal_graph import GlobalPort

    results=[decode(GlobalPort, value) for value in origin['results']]
    candidates=[value for value in results if value.unit==origin['unit']
        and value.port.owner.wire==origin['step']]
    if not candidates or len({(value.port.template,value.port.owner) for value in candidates})!=1:
        raise ValueError('motion origin has no unique original step coordinate')
    point=candidates[0]
    return GlobalPort(point.unit,G.Port(point.port.template,point.port.owner,G.PortKind.ENTER))


def read_invariant_loop_motion(path,*,catalog):
    """Join motion facts to one independently replayed original source loop.

    The ArrayRegion query alone has no source authority.  This adapter first
    uses the existing full F4 rediscovery, then retains the original CALL,
    lexical write and loop-control witnesses.  Array contents and live
    callable dispatch remain runtime obligations; a stable binding does not
    establish an immutable referent.
    """
    from scar.analysis.array_motion import derive_array_motion
    from scar.analysis.array_motion_legality import derive_array_motion_legality
    from scar.analysis.source_motion_window import derive_source_motion_window
    from scar.ir.execution_recipe import artifact_reference
    from scar.optimization.resource_solver import semantic_sha256

    proof=read_invariant_loop_proof(path,catalog=catalog)
    interface=canonical_invariant_motion_interface(proof)
    motion=derive_array_motion(proof.body,carry_parameter=proof.carry_parameter,loop_interface=interface)
    roles={row['role']:row for row in proof.origins if row.get('kind')=='original_f4'
        and row.get('role') in ('builtin_range','lexical_carry_write',
            'invariant_array_leaf_call','single_shared_invariant_loop')}
    call=roles['invariant_array_leaf_call']
    identity=next(row for row in proof.source.origins if row.get('kind')=='source_identity')
    view=catalog if isinstance(catalog,ProgramView) else ProgramView(catalog)
    source=view.unit(identity['unit'])
    functions=[index for index,function in enumerate(source.facts.functions)
        if function.template.occurrence==identity['occurrence']]
    if len(functions)!=1:
        raise ValueError('motion source identity lacks one original F4 function')
    window=derive_source_motion_window(view,unit=identity['unit'],function_ordinal=functions[0],
        loop_header=_motion_origin_entry(roles['single_shared_invariant_loop']),
        call_entry=_motion_origin_entry(call))
    legality=derive_array_motion_legality(proof.body,carry_parameter=proof.carry_parameter,
        loop_interface=interface,source_window=window)
    report=dict(schema='scar.source-loop-motion',schema_version=2,
        proof=artifact_reference(path,PROOF_SCHEMA),proof_id=proof.id,
        proof_sha256=semantic_sha256(asdict(proof)),body_sha256=motion['body_sha256'],
        source_identity=dict(path=proof.source.path,sha256=proof.source.sha256,
            code_key=proof.source.code_key),
        array_motion=motion,
        source_window=window,movement_legality=legality,
        canonical_interface_sha256=semantic_sha256(interface),
        parameter_routes=[dict(leaf_formal=name,outer_parameter=proof.parameter_map[name],
            original_actual=call['call_inputs'][index],
            binding_role='CARRIED' if name==proof.carry_parameter else 'OUTER_LIVE_IN',
            referent_content_invariance='CONDITIONAL_ON_EXACT_ARRAY_AND_NO_FOREIGN_WRITERS')
            for index,name in enumerate(proof.body.parameters)],
        source_witnesses=roles,
        source_facts=dict(authority='FULL_ORIGINAL_F4_REDISCOVERY',
            body_profile='ONE_RANGE_BODY_ONE_ADMITTED_ARRAY_CALL_ONE_LEXICAL_CARRY_WRITE',
            source_actions_enumerated=True,opaque_source_actions=False,
            input_writes='NONE_WITHIN_REPLAYED_ARRAY_PROFILE',
            lexical_carry_write_is_input_storage_write=False,
            positive_iteration_call='ONE_PER_ENTERED_BODY',
            zero_trip='NO_BODY_EVALUATION_AND_ORIGINAL_CARRY_ALIAS',
            bodies_shared=True,iterations_unrolled=False),
        runtime_obligations=motion['conditional_obligations'],
        runtime_obligations_discharged=False,source_replacement_authorized=False,
        cost_qualified=False,selected_for_use=False)
    report['id']='source-loop-motion:'+semantic_sha256(report)
    return proof,report


def verify_invariant_loop_motion(report,*,catalog):
    """Cold source replay; no target execution or trusted proof markers."""
    from scar.ir.execution_recipe import read_artifact
    from scar.optimization.resource_solver import semantic_sha256

    if (type(report) is not dict or report.get('schema')!='scar.source-loop-motion'
            or type(report.get('schema_version')) is not int or report['schema_version']!=2):
        raise ValueError('current source loop motion report required')
    read_artifact(report['proof'])
    proof,expected=read_invariant_loop_motion(report['proof']['uri'],catalog=catalog)
    if semantic_sha256(expected)!=semantic_sha256(report):
        raise ValueError('source loop motion differs from original F4 replay')
    return proof,expected


def validate_invariant_loop_links(proof,spec,loopsref):
    """Join the retained loop/CALL facts to an already F4-replayed proof.

    No catalog is opened and no source graph is rebuilt here. Both FactRefs
    must point into the same byte-checked loops publication and their original
    ports must equal the proof's control/CALL witnesses, including world ABI.
    """
    from scar.ir.execution_recipe import read_artifact
    if type(proof) is not InvariantLoopProof or spec.get('id')!=proof.id:
        raise ValueError('loop specification differs from its original proof ID')
    if spec.get('source_code_key')!=proof.source.code_key or spec.get('loop_link_frontier'):
        raise ValueError('original source code identity and closed loop links required')
    document=read_artifact(loopsref)
    if document.get('schema')!='scar.source-loop-dependencies' or document.get('schema_version')!=1:
        raise ValueError('original source loop dependency publication required')
    links=spec.get('loop_links')
    if type(links) is not list or len(links)!=1:
        raise ValueError('one original F4 loop/CALL relationship required')
    link=links[0]
    if (set(link)!={'kind','loop','call'} or link['kind']!='EXACT_ORIGINAL_F4_LOOP_HEADER_AND_CALL_RESULT'
            or any(type(link[name]) is not dict or set(link[name])!={'artifact','locator'}
                or link[name]['artifact']!=loopsref for name in ('loop','call'))):
        raise ValueError('loop and CALL FactRefs must use the retained parent artifact')
    location=link['loop']['locator'];call_location=link['call']['locator']
    match=re.fullmatch(r'#/functions/(0|[1-9][0-9]*)/loops/(0|[1-9][0-9]*)',location)
    call_match=re.fullmatch(re.escape(location)+r'/body_calls/(0|[1-9][0-9]*)',call_location)
    if match is None or call_match is None:raise ValueError('canonical original loop and enclosed CALL locators required')
    try:
        function=document['functions'][int(match[1])]
        loop=function['loops'][int(match[2])];call=loop['body_calls'][int(call_match[1])]
        control=next(row for row in proof.origins if row.get('role')=='single_shared_invariant_loop')
        original=next(row for row in proof.origins if row.get('role')=='invariant_array_leaf_call')
        identity=next(row for row in proof.source.origins if row.get('kind')=='source_identity')
        header=loop['header'];port=header['port']
        normal_world=[value for value in original['results'] if value['port']['completion']=='FALLTHROUGH'
            and value['port']['ordinal']==1]
        if (function['body']['unit']!=control['unit'] or function['body']['template']['occurrence']!=identity['occurrence']
                or header['unit']!=control['unit'] or port['template']!=control['template']
                or port['owner']['wire']!=control['step'] or port['kind']!='ENTER'
                or loop['body']['unit']!=control['unit'] or loop['body']['template']!=control['template']
                or loop['body']['region']['wire']!=control['body_region']
                or call['result']!=original['call_result'] or call['callee']!=original['call_callee']
                or call['positional']!=original['call_inputs'] or call['keywords']
                or call['positional_pack'] is not None or call['keyword_pack'] is not None
                or call['world_before']!=original['arguments'][-1]
                or len(normal_world)!=1 or call['world_after']!=normal_world[0]
                or len(loop['slots'])!=2 or loop['slots'][0]['formal']!=control['iterator_parameter']
                or [slot['role'] for slot in loop['slots']]!=['INVARIANT_PREFIX','CARRIED_SUFFIX']
                or not loop['control_backedges'] or not loop['slots'][1]['backedges']):
            raise ValueError('retained loop/CALL facts differ from original F4 control, operands or world ports')
    except (KeyError,IndexError,StopIteration,TypeError) as error:
        raise ValueError('original loop/CALL FactRef locator or witness is absent') from error
    return links


__all__ = ['InvariantLoopProof','discover_invariant_loops','validate_invariant_loop_proof',
    'invariant_loop_proof_document','write_invariant_loop_proof','read_invariant_loop_proof',
    'validate_invariant_loop_links','PROOF_SCHEMA']
