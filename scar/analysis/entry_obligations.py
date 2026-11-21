"""Replay the original F4 entry projection used to demand replacement proofs.

Reachability is a may query over normal-success source plans. Shared regions
have one recipe, and call selections retain their separate evidence.
"""

from collections import defaultdict, deque
from dataclasses import replace
from pathlib import Path

from scar.analysis.build import BuildSession
from scar.analysis.normal_graph import normal_edges
from scar.analysis.program_dependency_export import _Tables
from scar.analysis.replay_transaction import current_transaction, replay_transaction
from scar.analysis.source_effects import classify_instruction
from scar.analysis.source_stage import _durable_bytes
from scar.analysis.source_template_links import _json
from scar.analysis.source_world_flow import _walk
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.program import normal_graph as G, source_productions as P
from scar.ir.program.control import LanguageCompletionKind
from scar.ir.program.dependencies import DependencyKind, context_frames
from scar.ir.program.ids import RegionID
from scar.ir.program.source_effects import EffectResolution
from scar.ir.record_codec import encode


SCHEMA = 'scar.f4-entry-obligations'
PROGRAM = 'scar.program-build'
LIMITS = dict(bodies=64, templates_per_body=4000, operations=8192, edges=65536)


def _check_session(session, program):
    if (session.report.get('backend') != 'F4' or not session.report.get('workflow_complete')
            or program['schema'] != PROGRAM or Path(program['uri']) != session.path/'manifest.json'
            or read_artifact(program) != session.report):
        raise ValueError('entry obligations require the original published F4 build')
    current_transaction().watch(program['uri'], program['sha256'], program['size_bytes'])


def _body(source, template, root, max_templates):
    plans = source.material.closure(template, max_templates=max_templates)
    edges = tuple(dict.fromkeys(edge for plan in plans.values()
        for edge in normal_edges(plan, source.material.interfaces)))
    successors = defaultdict(list)
    for edge in edges:
        if edge.kind is G.EdgeKind.CONTROL:
            successors[edge.source].append(edge.target)
    reachable = _walk(G.Port(template, root, G.PortKind.ENTER), successors)
    owners = {(port.template, port.owner) for port in reachable}
    selected = tuple(edge for edge in edges if (edge.source.template, edge.source.owner) in owners
        and (edge.target.template, edge.target.owner) in owners)
    steps = tuple((key, step) for key, plan in plans.items() for step in plan.steps
        if G.Port(key, step.id, G.PortKind.ENTER) in reachable)
    return steps, selected


def _call(instruction):
    return (type(instruction) is P.ApplyLanguageAction and instruction.action is P.LanguageActionKind.CALL
        or type(instruction) is P.ApplyDecorator)


def _obligations(step, rule):
    resolution = rule.resolution
    result = []
    if resolution is EffectResolution.DYNAMIC_TARGET_OR_PROTOCOL:
        result.append('PROVIDER_BEHAVIOR')
    if resolution is EffectResolution.ADDRESS_RULE:
        result.append('ADDRESS_INSTANCE_AND_EFFECT')
    if resolution is EffectResolution.INSTANCE_CONDITION:
        result.append('INSTANCE_EFFECT_CONTRACT')
    if any(completion.kind is LanguageCompletionKind.RAISE for completion in step.completions):
        result.append('EXCEPTION_BEHAVIOR')
    if type(step.instruction) is P.LoopControl:
        result.append('LOOP_TRIP_COUNT_AND_CARRIED_STATE')
    return result


class _Table:
    def __init__(self):
        self.indices = {}
        self.rows = []

    def put(self, key, make_row):
        if key not in self.indices:
            self.indices[key] = len(self.rows)
            self.rows.append(make_row())
        return self.indices[key]


class _Projection:
    def __init__(self, session, limits):
        self.view = session.view
        self.limits = limits
        self.pending = deque()
        self.bodies = []
        self.body_ids = {}
        self.units = _Table()
        self.templates = _Table()
        self.owners = _Table()
        self.ports = _Table()
        self.types = _Table()
        self.sites = _Table()
        self.operations = []
        self.operation_ids = {}
        self.edges = _Table()
        self.calls = []
        self.call_tables = _Tables()
        self.call_selections = {}
        self.imports = []
        self.modules = defaultdict(list)
        store = self.view.catalog.fact_store
        for unit, row in sorted(store.rows.items()):
            if unit not in store.original_source_units:
                self.modules[row['module']].append(unit)

    def add_body(self, point, *, entry=False):
        key = (point.namespace, point.node)
        if key not in self.body_ids:
            if len(self.bodies) >= self.limits['bodies']:
                return None
            number = self.body_ids[key] = len(self.bodies)
            # Query a shared body with unknown caller context. Preserve actual
            # guarded call selections separately on the original edges.
            selected = point if entry else replace(point, context=None)
            self.bodies.append(dict(point=encode(selected), status='QUEUED'))
            self.pending.append((selected, number))
        return self.body_ids[key]

    def unit(self, source):
        return self.units.put(source.unit, lambda: dict(unit=source.unit, namespace=encode(source.namespace)))

    def template(self, unit, template):
        return self.templates.put((unit, template), lambda: dict(unit=unit, template=encode(template)))

    def owner(self, unit, template, owner):
        return self.owners.put((unit, template, owner), lambda: dict(template=self.template(unit, template),
            original=owner.wire))

    def port(self, unit, port):
        return self.ports.put((unit, port), lambda: dict(owner=self.owner(unit, port.template, port.owner),
            kind=port.kind.value, ordinal=port.ordinal, completion=encode(port.completion),
            target=encode(port.target), path=encode(port.path)))

    def edge(self, unit, edge):
        return self.edges.put((unit, edge), lambda: dict(source=self.port(unit, edge.source),
            target=self.port(unit, edge.target), kind=edge.kind.value, transition=edge.transition.value,
            value_type=None if edge.value_type is None else self.types.put(edge.value_type, lambda: encode(edge.value_type)),
            site=None if edge.site is None else self.sites.put((unit, edge.site), lambda: dict(unit=unit,
                site=encode(edge.site))), when=edge.when,
            condition=None if edge.condition is None else self.port(unit, edge.condition)))

    def call(self, point, operation):
        neighborhood = self.view.dependencies.outgoing(point, kinds=(DependencyKind.CALL_TARGET,))
        targets = []
        for edge in neighborhood.edges:
            target = edge.target
            source_body = (type(target.node) is RegionID
                and target.namespace in self.view.namespaces)
            body = self.add_body(target) if source_body else None
            link = self.call_tables.edge(edge)
            for side in (edge.source, target):
                for frame in context_frames(side.context):
                    if frame.selection is not None:
                        self.call_selections.setdefault(self.call_tables.frame(frame), encode(frame.selection))
            targets.append(dict(edge=link, body=body,
                status=('BODY_BUDGET' if body is None else 'SOURCE_BODY') if source_body
                    else 'TARGET_METADATA_ONLY'))
        self.calls.append(dict(operation=operation, alternatives=targets,
            provider_frontiers=[self.call_tables.frontier(item) for item in neighborhood.frontiers],
            provider_enumerated=neighborhood.enumerated,
            obligation='CALL_TARGET_EFFECT_AND_ACTIVATION_CLOSURE'))

    def import_(self, instruction, operation):
        alternatives = []
        requested = instruction.requested_name
        if instruction.relative_level == 0 and requested:
            parts = requested.split('.')
            for end in range(1, len(parts) + 1):
                name = '.'.join(parts[:end])
                for unit in self.modules.get(name, ()):
                    point = self.view.entry(unit)
                    body = self.add_body(point)
                    alternatives.append(dict(module=name, body=body, point=encode(point),
                        status='BODY_BUDGET' if body is None else 'POSSIBLE_MODULE_BODY'))
        self.imports.append(dict(operation=operation, requested_name=requested,
            relative_level=instruction.relative_level, from_names=encode(instruction.from_names),
            alternatives=alternatives, obligation='IMPORT_HOOK_REGISTRY_AND_INITIALIZATION_EFFECTS'))

    def operation(self, source, unit, template, step, context):
        key = unit, template, step.id
        if key not in self.operation_ids:
            number = self.operation_ids[key] = len(self.operations)
            point = replace(source._node(template, step.id), context=context)
            rule = classify_instruction(step.instruction)
            self.operations.append(dict(unit=unit, operation=point.node.wire,
                template=encode(template), region=step.region.wire,
                reach_port=self.port(unit, G.Port(template, step.id, G.PortKind.ENTER)),
                effect_rule=encode(rule), obligations=_obligations(step, rule),
                loop=encode(step.instruction) if type(step.instruction) is P.LoopControl else None))
            if _call(step.instruction):
                self.call(point, number)
            if type(step.instruction) is P.ImportRequest:
                self.import_(step.instruction, number)
        return self.operation_ids[key]

    def run(self, entry):
        self.add_body(entry, entry=True)
        while self.pending:
            point, number = self.pending.popleft()
            source = self.view.namespaces[point.namespace]
            anchor = source.nodes[point.node].anchor
            body = self.bodies[number]
            unit = body['unit'] = self.unit(source)
            try:
                steps, edges = _body(source, anchor.template, anchor.original, self.limits['templates_per_body'])
            except ValueError as error:
                if str(error) != 'world material template closure exceeds its declared budget':
                    raise
                body['status'] = 'TEMPLATE_BUDGET'
                continue
            if len(self.operations) + sum((unit, key, step.id) not in self.operation_ids
                    for key, step in steps) > self.limits['operations']:
                body['status'] = 'OPERATION_BUDGET'
                continue
            if len(self.edges.rows) + sum((unit, edge) not in self.edges.indices for edge in edges) > self.limits['edges']:
                body['status'] = 'EDGE_BUDGET'
                continue
            body.update(status='PROJECTED', operations=[self.operation(source, unit, key, step, point.context)
                for key, step in steps], edges=[self.edge(unit, edge) for edge in edges])
        for source in self.view.units.values():
            source.material.validate_parents()
        return dict(entry_body=0, units=self.units.rows, bodies=self.bodies, operations=self.operations,
            templates=self.templates.rows, owners=self.owners.rows, ports=self.ports.rows,
            value_types=self.types.rows, sites=self.sites.rows, edges=self.edges.rows, calls=self.calls,
            call_tables=dict(**self.call_tables.tables, original_selections=[dict(frame=frame, selection=value)
                for frame, value in self.call_selections.items()]), imports=self.imports)


def _derive(session, program):
    _check_session(session, program)
    return dict(schema=SCHEMA, schema_version=1, program=program, limits=LIMITS,
        scope='NORMAL_SUCCESS_SOURCE_MAY_GRAPH', loops_unrolled=False, runtime_path_complete=False,
        effects_closed=False, transformation_authorized=False, **_Projection(session, LIMITS).run(session.entry))


def publish_entry_obligations(session, out_file):
    """Save the entry projection without invoking or rewriting its source."""
    out_file = Path(out_file).absolute()
    with replay_transaction():
        document = _derive(session, artifact_reference(session.path/'manifest.json', PROGRAM))
    _durable_bytes(out_file, _json(document), replace=False)
    return artifact_reference(out_file, SCHEMA)


def verify_entry_obligations(reference, *, program, session=None):
    """Cold-replay the entry projection from the exact original F4 graph."""
    with replay_transaction() as transaction:
        saved = read_artifact(reference)
        if reference['schema'] != SCHEMA or saved.get('program') != program:
            raise ValueError('entry obligations do not bind the original build')
        transaction.watch(reference['uri'], reference['sha256'], reference['size_bytes'])
        if session is None:
            with BuildSession(Path(program['uri']).parent) as independent:
                expected = _derive(independent, program)
        else:
            expected = _derive(session, program)
        if saved != expected:
            raise ValueError('saved entry obligations differ from original F4 replay')
    return saved
