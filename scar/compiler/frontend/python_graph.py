"""Project the admitted Python F4 array profile into the shared ProgramGraph."""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from pathlib import Path

from scar.analysis.build_f4 import _validate_catalog_parents
from scar.analysis.program_view import ProgramView
from scar.analysis.source_decoded_bodies import reuse_decoded_source_bodies
from scar.artifacts import _durable_bytes
from scar.compiler.contracts import GraphBundle, python_build_receipts
from scar.compiler.graph_frontend import RegionDiscovery, _origin, discover_graph_regions
from scar.compiler.loop_admission import (
    document as array_loop_document, discover_graph_loops, discover_invariant_loops,
    read_invariant_loop_motion, write_invariant_loop_proof,
)
from scar.compiler.model import ArrayRegion
from scar.compiler.pipeline_frontend import PipelinePlan
from scar.compiler.pipeline_loop_frontend import (
    discover_pipeline_invariant_loops, read_composite_invariant_loop_motion,
    write_composite_invariant_loop_proof,
)
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.literals import PythonLiteral
from scar.ir.program import model as M, source_productions as P
from scar.ir.program.codec import dumps
from scar.ir.program.ids import BlockID, LocationID, OperationID, RegionID, UseID, ValueID
from scar.ir.program.source_declarations import SourceTemplateKey
from scar.ir.record_codec import decode


REFERENCE = M.ValueType(M.ValueKind.REFERENCE)
BOOLEAN = M.ValueType(M.ValueKind.DATA, 'builtins.bool')
WORLD_LOCATION = LocationID('python-f4:world')
WORLD = M.ValueType(M.ValueKind.STATE, location=WORLD_LOCATION)
RETURN = M.ExitPort('return', M.ExitKind.NORMAL, (REFERENCE, WORLD))
RAISE = M.ExitPort('raise', M.ExitKind.RAISE, (REFERENCE, WORLD))
USE_ROLES = {
    M.ValueKind.REFERENCE: M.UseRole.REFERENCE,
    M.ValueKind.DATA: M.UseRole.DATA,
    M.ValueKind.STATE: M.UseRole.STATE_CONTENT,
}


def _source_key(source):
    return source.path, source.code_key


def _write(path, document):
    raw = (json.dumps(document, sort_keys=True, allow_nan=False) + '\n').encode()
    _durable_bytes(path, raw, replace=False)
    return artifact_reference(path, document['schema'])


@dataclass
class _Block:
    id: BlockID
    arguments: tuple[ValueID, ...]
    operations: list[OperationID]


class _Function:
    """Build one original callable's SSA blocks under the F4 world contract."""

    def __init__(self, graph, source, parameters):
        self.graph, self.source = graph, source
        self.id = graph.identity(source)
        self.blocks = []
        self.entry = self.block((REFERENCE,) * len(parameters) + (WORLD,))
        self.failure = None

    def value(self, type_, origin):
        identity = ValueID(f'{self.id.value}:v{len(self.graph.values)}')
        if type_ == REFERENCE:
            value = M.ReferenceValue(identity, self.id, origin)
        elif type_ == WORLD:
            value = M.StateVersionValue(identity, self.id, origin, WORLD_LOCATION)
        else:
            value = M.DataValue(identity, self.id, origin, type_.data_type)
        self.graph.values[identity] = value
        return identity

    def block(self, types=()):
        identity = BlockID(f'{self.id.value}:b{len(self.blocks)}')
        arguments = tuple(self.value(type_, M.BlockArgument(identity, i)) for i, type_ in enumerate(types))
        block = _Block(identity, arguments, [])
        self.blocks.append(block)
        return block

    def uses(self, operation, arguments):
        return tuple(M.Use(UseID(f'{operation.value}:u{i}'), value,
            USE_ROLES[M.value_type(self.graph.values[value]).kind]) for i, value in enumerate(arguments))

    def emit(self, block, label, *, origins, arguments=(), behaviors=(), results=(),
             terminator=None, callee=None, completions=(), effects_complete=True):
        identity = OperationID(f'{block.id.value}:o{len(block.operations)}')
        uses = self.uses(identity, arguments)
        outputs = tuple(self.value(type_, M.OperationResult(identity, i)) for i, type_ in enumerate(results))
        term = terminator(tuple(use.id for use in uses)) if terminator else None
        operation = M.Operation(identity, label, behaviors, uses, outputs, effects_complete=effects_complete,
            terminator=term, callee=callee, completions=completions,
            invoke_arguments=tuple(use.id for use in uses) if completions else ())
        self.graph.operations.append(operation)
        self.graph.anchors.append(dict(operation=identity.wire, region=self.id.wire,
            source=self.source.path, code_key=self.source.code_key, origins=list(origins)))
        block.operations.append(identity)
        return identity, outputs

    def exit(self, block, kind, arguments, origins):
        name = 'raise' if kind is M.TerminatorKind.RAISE else 'return'
        self.emit(block, name, arguments=arguments, origins=origins, behaviors=(M.Behavior.CONTROL,),
            terminator=lambda uses: M.Terminator(kind, exit=name, values=uses))

    def exception(self):
        if self.failure is None:
            self.failure = self.block(RAISE.types)
            self.exit(self.failure, M.TerminatorKind.RAISE, self.failure.arguments, self.source.origins[:1])
        return self.failure

    def invoke(self, block, label, arguments, normal_types, origins, *, failure=None, callee=None,
               behaviors=(M.Behavior.READ,)):
        normal = self.block(normal_types)
        if failure is None:
            failure = self.block(RAISE.types)
            self.jump(failure, self.exception(), failure.arguments, origins)
        successors = tuple(M.CompletionSuccessor(port.kind, successor.id,
            tuple(M.CompletionResult(i) for i in range(len(types))))
            for port, successor, types in ((RETURN, normal, normal_types), (RAISE, failure, RAISE.types)))
        completions = (M.CompletionSignature(M.ExitKind.NORMAL, normal_types),
                       M.CompletionSignature(M.ExitKind.RAISE, RAISE.types))
        operation, _ = self.emit(block, label, arguments=arguments, origins=origins,
            behaviors=(*behaviors, M.Behavior.CONTROL), effects_complete=False,
            terminator=lambda _: M.Terminator(M.TerminatorKind.INVOKE, successors),
            callee=callee, completions=completions)
        if callee is not None:
            self.graph.calls.append(dict(operation=operation.wire, caller=self.id.wire,
                callee=callee.wire, original=list(origins),
                binding='CONDITIONAL_ORIGINAL_F4_SOURCE_TARGET',
                runtime_dispatch_verified=False))
        return normal, normal.arguments

    def jump(self, block, target, arguments, origins):
        operation, _ = self.emit(block, 'jump', arguments=arguments, origins=origins, behaviors=(M.Behavior.CONTROL,),
            terminator=lambda uses: M.Terminator(M.TerminatorKind.JUMP, (M.Successor(target.id, uses),)))
        return operation

    def branch(self, block, condition, yes, yes_values, no, no_values, origins):
        split = 1 + len(yes_values)
        self.emit(block, 'branch', arguments=(condition, *yes_values, *no_values), origins=origins,
            behaviors=(M.Behavior.CONTROL,), terminator=lambda uses: M.Terminator(
                M.TerminatorKind.BRANCH, (M.Successor(yes.id, uses[1:split]),
                                          M.Successor(no.id, uses[split:])), uses[0]))

    def finish(self):
        blocks = tuple(M.Block(block.id, block.arguments, tuple(block.operations)) for block in self.blocks)
        exits = (RETURN, RAISE) if self.failure is not None else (RETURN,)
        self.graph.regions[self.id] = M.Region(self.id, self.source.qualname, self.entry.id, blocks, exits)


class _Projection:
    """Share retained source functions rather than cloning an inlined loop body."""

    def __init__(self, discovery, arrays, pipelines):
        self.discovery = discovery
        self.arrays = {_source_key(region.source): region for region in arrays}
        self.pipelines = {_source_key(plan.source): plan for plan in pipelines}
        self.regions, self.values = {}, {}
        self.operations, self.anchors, self.calls, self.literals = [], [], [], []
        self.active = set()

    def identity(self, source):
        key = '\0'.join(_source_key(source))
        return RegionID('python-f4:' + sha256(key.encode()).hexdigest()[:24])

    def original_array(self, source):
        key = _source_key(source)
        if key not in self.arrays:
            found, _ = discover_graph_regions(self.discovery.view, source.path, _discovery=self.discovery)
            self.arrays.update((_source_key(region.source), region) for region in found)
        if key not in self.arrays:
            raise ValueError('retained call requires an independently admitted original array body')
        return self.arrays[key]

    def function(self, source):
        identity = self.identity(source)
        if identity in self.regions:
            return identity
        if identity in self.active:
            raise ValueError('recursive calls are outside the admitted Python F4 array profile')
        self.active.add(identity)
        try:
            if pipeline := self.pipelines.get(_source_key(source)):
                self.pipeline(pipeline)
            else:
                self.array(self.original_array(source))
        finally:
            self.active.remove(identity)
        return identity

    def array(self, region: ArrayRegion):
        function = _Function(self, region.source, region.parameters)
        block = function.entry
        arguments, world = block.arguments[:-1], block.arguments[-1]
        parameters = dict(zip(region.parameters, arguments, strict=True))
        values = {}
        witnesses = {row.get('expression'): row for row in region.source.origins if 'expression' in row}
        for node in region.nodes:
            if node.op == 'input':
                values[node.id] = parameters[node.attrs['name']]
                continue
            if node.id not in witnesses:
                raise ValueError('array expression has no original F4 operation coordinate')
            origin = (dict(witnesses[node.id], operator=node.op, attributes=node.attrs),)
            if node.op == 'const':
                literal = PythonLiteral.from_python(node.attrs['value'])
                scalar_type = M.ValueType(M.ValueKind.DATA,
                    f'builtins.{type(node.attrs["value"]).__name__}')
                operation, (values[node.id],) = function.emit(block, 'literal',
                    results=(scalar_type,), origins=origin)
                self.literals.append(M.LiteralEvaluation(M.OperationResult(operation, 0), literal))
                continue
            kinds = (M.Behavior.REPRESENT,) if node.op == 'slice' else (
                (M.Behavior.MEMORY, M.Behavior.READ) if node.op == 'copy' else
                (M.Behavior.COMPUTE, M.Behavior.READ))
            block, (values[node.id], world) = function.invoke(block, f'array.{node.op}',
                (*(values[value] for value in node.inputs), world), RETURN.types, origin, behaviors=kinds)
        function.exit(block, M.TerminatorKind.RETURN, (values[region.output], world), region.source.origins[:1])
        function.finish()

    def pipeline(self, plan: PipelinePlan):
        function = _Function(self, plan.source, plan.parameters)
        block = function.entry
        world = block.arguments[-1]
        values = dict(zip(plan.input_values, block.arguments[:-1], strict=True))
        for call in plan.calls:
            target = self.function(call.callee.source)
            if not call.origins:
                raise ValueError('pipeline call requires its original F4 coordinate')
            block, (values[call.output], world) = function.invoke(block, 'call',
                (*(values[value] for value in call.inputs), world), RETURN.types, call.origins, callee=target)
        function.exit(block, M.TerminatorKind.RETURN, (values[plan.output], world), plan.source.origins[:1])
        function.finish()

    def loop(self, proof, family):
        is_array = family == 'ARRAY_LOOP'
        roles = {row['role']: row for row in proof.origins if row.get('kind') == 'original_f4' and 'role' in row}
        control = roles['single_shared_loop' if is_array else 'single_shared_invariant_loop']
        call = roles['body_pipeline_call' if is_array else 'invariant_array_leaf_call']
        carry = proof.state_parameter if is_array else proof.carry_outer_parameter
        callee_source = proof.root_callee_source if family == 'COMPOSITE_LOOP' else proof.body.source
        callee = self.function(callee_source)

        context = self.discovery.resolver.plan(control['unit'], decode(SourceTemplateKey, control['template']))
        original_loop, = (step for step in context.plan.steps if step.id.wire == control['step'])
        iteration = original_loop.instruction.condition
        if type(iteration) is not P.IterationCondition:
            raise ValueError('retained loop body requires its original iteration condition')
        iterate = original_loop.arguments[0]
        if type(iterate) is P.ProductionParameter:
            parent = context.regions[context.regions[iterate.region].parent]
            original_call, = (context.steps[key].instruction.invocation for key in parent.steps
                if type(context.steps[key].instruction) is P.ApplyRegion
                and context.steps[key].instruction.invocation.region == iterate.region)
            iterate = original_call.arguments[iterate.index]
        if type(iterate) is not P.ProductionResult:
            raise ValueError('retained range iterator needs its original producer')
        iterator_op = context.steps[iterate.step]
        if (type(iterator_op.instruction) is not P.ApplyPythonProtocol
                or iterator_op.instruction.protocol.value != 'GET_ITERATOR' or iterate.index != 0):
            raise ValueError('retained range iterator differs from its original GET_ITERATOR')
        advance, = [context.steps[key] for key in context.regions[iteration.advance.region].steps
                    if getattr(context.steps[key].instruction, 'protocol', None)
                    and context.steps[key].instruction.protocol.value == 'COMPREHENSION_NEXT']
        stopped = context.steps[context.regions[iteration.exhausted].steps[0]]
        iterator_origin = (_origin(context, iterator_op.region, iterator_op),)
        advance_origin = (_origin(context, advance.region, advance),)
        stopped_origin = (_origin(context, stopped.region, stopped),)

        function = _Function(self, proof.source, proof.parameters)
        start = function.entry
        inputs = dict(zip(proof.parameters, start.arguments[:-1], strict=True))
        pre, (range_value, world) = function.invoke(start, 'call.builtin_range',
            (inputs[proof.trip_count_parameter], start.arguments[-1]), RETURN.types, (roles['builtin_range'],))
        pre, (iterator, world) = function.invoke(pre, 'iterator.get', (range_value, world),
            RETURN.types, iterator_origin)
        header = function.block((REFERENCE, REFERENCE, WORLD))
        entry_edge = function.jump(pre, header, (inputs[carry], iterator, world), (control,))
        current, iterator, world = header.arguments
        exhausted = function.block(RAISE.types)
        body, (_, world) = function.invoke(header, 'iterator.advance', (iterator, world),
            RETURN.types, advance_origin, failure=exhausted)

        actuals = (current,) if is_array else tuple(current if proof.parameter_map[name] == carry
            else inputs[proof.parameter_map[name]] for name in proof.body.parameters)
        body, (updated, world) = function.invoke(body, 'call', (*actuals, world),
            RETURN.types, (call,), callee=callee)
        backedge = function.jump(body, header, (updated, iterator, world), (roles['lexical_carry_write'], control))

        exception, world = exhausted.arguments
        stop, (condition, world) = function.invoke(exhausted, 'iterator.match_stop', (exception, world),
            (BOOLEAN, WORLD), stopped_origin)
        result = function.block(RETURN.types)
        failure = function.exception()
        function.branch(stop, condition, result, (current, world), failure, (exception, world), stopped_origin)
        function.exit(result, M.TerminatorKind.RETURN, result.arguments, proof.source.origins[:1])
        function.finish()
        return function.id, dict(header=header.id.wire, carry=current.wire,
            world=header.arguments[-1].wire, natural_exit=result.id.wire,
            entry_edge=entry_edge.wire, backedge=backedge.wire)

    def model(self, entries):
        root = entries[0]
        regions = tuple(region if region.id == root else replace(region, parent=root)
                        for region in self.regions.values())
        return M.ProgramGraph(root, regions, tuple(self.operations), tuple(self.values.values()),
            (M.StateLocation(WORLD_LOCATION, 'original Python F4 world'),),
            literal_evaluations=tuple(self.literals))


def _loop_evidence(directory, index, projection, family, proof):
    folder = directory / f'proof-{index}'
    folder.mkdir()
    if family == 'INVARIANT_LOOP':
        reference = write_invariant_loop_proof(proof, folder / 'proof.json')
        _, motion = read_invariant_loop_motion(reference['uri'], catalog=projection.discovery.view)
    elif family == 'COMPOSITE_LOOP':
        reference = write_composite_invariant_loop_proof(proof, folder / 'proof.json')
        _, motion = read_composite_invariant_loop_motion(reference['uri'], catalog=projection.discovery.view)
    else:
        reference = _write(folder / 'proof.json', array_loop_document(proof))
        motion = None
    identity, loop = projection.loop(proof, family)
    row = dict(id=proof.id, kind=family, region=identity.wire, loop=loop, proof=reference,
               source=asdict(proof.source))
    if motion is not None:
        row['source_motion'] = _write(folder / 'source-motion.json', motion)
    return row


def _load_source(stack, path, source):
    if (path / 'manifest.json').is_file():
        from scar.analysis.build import BuildSession

        session = stack.enter_context(BuildSession(path))
        request, hardware = python_build_receipts(session)
        entry = Path(session.report['entry']).resolve() if source is None else source
        return session.view, entry, asdict(request), asdict(hardware)
    if (path / 'catalog' / 'result.json').is_file():
        path = path / 'catalog'
    return ProgramView(path), source, None, None


@reuse_decoded_source_bodies()
def build_python_graph(catalog_dir, out, *, source=None):
    """Publish only exact admitted array functions and their structured loops."""
    path = Path(catalog_dir).resolve(strict=True)
    out = Path(out).resolve()
    source = None if source is None else Path(source).resolve(strict=True)
    with ExitStack() as stack:
        view, source, request, hardware = _load_source(stack, path, source)
        discovery = RegionDiscovery(view)
        arrays, array_gaps = discover_graph_regions(view, source, _discovery=discovery)
        direct, direct_gaps = discover_invariant_loops(view, source, _discovery=discovery)
        composite, composite_gaps = discover_pipeline_invariant_loops(view, source, _discovery=discovery)
        single, single_gaps = discover_graph_loops(view, source, _discovery=discovery)
        if not (arrays or direct or composite or single):
            raise ValueError('no admitted Python F4 array function or loop in the selected source scope')
        # Several old proof profiles can describe the same source loop. A
        # composed graph retains the actual original call chain as its one body.
        by_source = {}
        for family, proofs in (('COMPOSITE_LOOP', composite), ('INVARIANT_LOOP', direct), ('ARRAY_LOOP', single)):
            for proof in proofs:
                by_source.setdefault(_source_key(proof.source), (family, proof))
        pipelines = [composition.pipeline for proof in composite for composition in proof.compositions]
        pipelines.extend(loop.body for loop in single)
        projection = _Projection(discovery, arrays, pipelines)
        out.mkdir(parents=True, exist_ok=False)
        proofs = out / 'proofs'
        proofs.mkdir()
        rows = [_loop_evidence(proofs, index, projection, family, proof)
                for index, (family, proof) in enumerate(by_source.values())]
        for region in arrays:
            identity = projection.function(region.source)
            reference = _write(proofs / f'array-{len(rows)}.json', dict(
                schema='scar.python-f4-array-region', schema_version=1, region=asdict(region)))
            rows.append(dict(id=region.id, kind='ARRAY_REGION', region=identity.wire, proof=reference,
                             source=asdict(region.source)))
        entries = tuple(dict.fromkeys(RegionID(row['region'].split(':', 1)[1]) for row in rows))
        model = projection.model(entries)
        program_path = out / 'program.json'
        _durable_bytes(program_path, dumps(model).encode(), replace=False)
        program = artifact_reference(program_path, 'scar.program-graph')
        catalog = artifact_reference(view.catalog.path / 'result.json', view.catalog.report['schema'])
        if request is None:
            request = _write(out / 'request.json', dict(schema='scar.python-compiler-request',
                schema_version=1, entry=None if source is None else str(source), catalog=catalog))
            from scar.analysis.target_graph import build_target_graph

            paths = (str(view.catalog.path),) if source is None else (str(source),)
            build_target_graph(out / 'hardware', paths=(*paths, str(out)))
            hardware = artifact_reference(out / 'hardware/result.json', 'scar.target-hardware-graph')
        call_document = dict(schema='scar.python-call-links', schema_version=1, catalog=catalog,
            program=program, calls=projection.calls, runtime_dispatch_verified=False)
        calls = _write(out / 'call-links.json', call_document)
        report = dict(schema='scar.python-graph-evidence', schema_version=1, catalog=catalog,
            source_filter=None if source is None else str(source),
            program=program, hardware=hardware, request=request, calls=calls,
            regions=rows, anchors=projection.anchors, graph_profile='GUARDED_ORIGINAL_F4_ARRAY_PROFILE',
            rejected=dict(array_regions=array_gaps, invariant_loops=direct_gaps,
                          composite_loops=composite_gaps, array_loops=single_gaps),
            target_executed=False, loops_unrolled=False, runtime_obligations_discharged=False,
            source_replacement_authorized=False)
        discovery.validate()
        _validate_catalog_parents(view.catalog)
        for reference in (catalog, request, hardware):
            read_artifact(reference, json_document=False)
        evidence = _write(out / 'graph-evidence.json', report)
        return GraphBundle.publish(out, program=program, hardware=hardware, request=request,
            evidence=evidence, calls=calls, entry_regions=[entry.wire for entry in entries],
            model=model, report=report, call_document=call_document)
