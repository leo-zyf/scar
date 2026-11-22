"""Local MAY summaries over original shared F4 function value-flow ports.

CALL outputs and external reads remain finite original-port seeds. A separate
consumer can compose admitted actual/formal and return routes without an
unconditional all-arguments-to-result edge. No source body is copied/inlined,
no loop is unrolled, and no effect or mutation freedom is inferred here.
"""
from collections import Counter, defaultdict, deque
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

from scar.analysis.source_effects import classify_instruction
from scar.analysis.source_reference_slice import ReferenceSlicer, source_port
from scar.analysis.source_value_flow import project_value_flow
from scar.analysis.source_world_flow import _state
from scar.ir.program import normal_graph as G, normal_graph as F, normal_graph as T
from scar.ir.program import source_productions as P
from scar.ir.program.providers import ProviderArtifact
from scar.ir.record_codec import encode
from scar.optimization.resource_solver import semantic_sha256


SCHEMA = 'scar.source-function-summary'
CONDITIONS = (
    'ORIGINAL_FUNCTION_BODY_AND_ARGUMENT_CONTEXT_SELECTED',
    'FLOW_INSENSITIVE_NON_PATH_SENSITIVE_MAY_DEPENDENCY',
    'LEXICAL_WRITER_READER_JOIN_IS_NOT_SELECTED_REACHING_VERSION',
    'NONCALL_OPERATION_OUTPUT_MAY_DEPEND_ON_ANY_NONSTATE_INPUT',
    'CALL_OUTPUTS_REQUIRE_SEPARATE_ADMITTED_TARGET_AND_RETURN_COMPOSITION',
    'WORLD_STATE_AND_FUNCTION_ENVIRONMENT_ARE_NOT_DATA_VALUE_EQUALITY',
    'NO_ALIAS_CONTENT_IDENTITY_PURITY_MUTATION_OR_RUNTIME_INVOCATION_PROOF',
)
_EFFECTS = ('reads', 'writes', 'allocates', 'frees', 'aliases', 'escapes',
    'rng_effect', 'may_raise', 'external_effect', 'ordering_effect')


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _is_call(operation):
    return (type(operation.instruction) is P.ApplyLanguageAction
        and operation.instruction.action is P.LanguageActionKind.CALL)


def _propagate(flow, seeds, *, max_work):
    """One sparse integer-bitset fixed point, with factored lexical joins.

    This reuses ``data_reach``'s edge/operation/location interpretation, except
    CALL result/state transfer is deliberately left to original output seeds.
    State hubs are distinct from nonstate hubs, so world cannot taint DATA.
    """
    adjacency = defaultdict(set)
    for edge in flow.edges:
        if edge.kind in (G.EdgeKind.DATA, G.EdgeKind.STATE):
            adjacency[edge.source].add(edge.target)
    for index, operation in enumerate(flow.operations):
        if _is_call(operation):
            continue
        for state in (False, True):
            hub = ('state' if state else 'nonstate', index)
            for value in operation.inputs:
                if _state(value.value_type) == state:
                    adjacency[value.port].add(hub)
            for value in operation.outputs:
                if _state(value.value_type) == state:
                    adjacency[hub].add(value.port)
    for index, location in enumerate(flow.locations):
        hub = ('lexical', index)
        for access in location.writes:
            adjacency[access.value].add(hub)
        for access in location.reads:
            adjacency[hub].add(access.value)
    bits = defaultdict(int)
    pending = deque()
    queued = set()
    for index, port in enumerate(seeds):
        bits[port] |= 1 << index
        if port not in queued:
            pending.append(port); queued.add(port)
    work = 0
    while pending:
        point = pending.popleft(); queued.remove(point)
        # Stable order also makes bounded partial results and work receipts
        # reproducible in independent processes with different hash seeds.
        order = lambda target: ('port', _canonical(encode(target))) if type(target) is G.Port else ('hub', target)
        for target in sorted(adjacency.get(point, ()), key=order):
            if work >= max_work:
                return bits, False, work, len(adjacency)
            work += 1
            updated = bits[target] | bits[point]
            if updated != bits[target]:
                bits[target] = updated
                if target not in queued:
                    pending.append(target); queued.add(target)
    return bits, True, work, len(adjacency)


def derive_source_function_summary(view, unit, function, *, max_work=2_000_000):
    """Summarize one original facts.functions record, retaining open effects."""
    if type(max_work) is not int or max_work < 1:
        raise ValueError('positive bounded source-summary propagation work required')
    source = view.unit(unit)
    facts = source.facts
    matches = [index for index, original in enumerate(facts.functions) if original == function]
    if type(function) is not F.Function or len(matches) != 1 or function.body_scope is None:
        raise ValueError('one original source function with its exact body scope is required')
    if len(function.formals) != len(function.parameter_names)+2:
        raise ValueError('original environment/data/world formal partition is required')
    source.check()
    slicer = ReferenceSlicer(SimpleNamespace(fact_store=view.catalog.fact_store))
    definition = slicer.fact(facts, 'functions', matches[0])
    body = T.Body(unit, function.template, function.body)
    callable_port = source_port(facts, function.value)
    flow = project_value_flow(source.material, callable_port, body)
    function_id = 'source-function:'+semantic_sha256(dict(unit=unit,
        definition=encode(definition), source_function=encode(callable_port), body=encode(body)))
    model = view.catalog.fact_store.model_units[unit]
    ports, query_ports, tokens, seed_indices = {}, set(), [], {}

    def gp(port):
        return T.GlobalPort(unit, port)

    def remember(port):
        if port is not None:
            ports[port] = encode(gp(port)); query_ports.add(port)
        return None if port is None else ports[port]

    def token(port, kind, evidence=(), **extra):
        remember(port)
        # Multiple declarations of one original read/output share one finite
        # seed. Evidence is a reference ledger, not another program identity.
        key = port, kind
        if key not in seed_indices:
            seed_indices[key] = len(tokens)
            tokens.append(dict(id=semantic_sha256(dict(port=ports[port], kind=kind)),
                port=ports[port], kind=kind, evidence=[], **extra))
        row = tokens[seed_indices[key]]
        for fact in evidence:
            wire = encode(fact)
            if wire not in row['evidence']:
                row['evidence'].append(wire)
        return row['id']

    formals = []
    for index, value in enumerate(function.formals):
        port = source_port(facts, value).port
        role = 'ENVIRONMENT' if index == 0 else 'STATE' if index == len(function.formals)-1 else 'DATA'
        row = dict(index=index, role=role, port=remember(port))
        if role == 'DATA':
            row.update(ordinal=index-1, name=function.parameter_names[index-1])
        row['token'] = token(port, 'FORMAL_'+role, (definition,), formal_index=index)
        formals.append(row)
    return_port = None if function.returns is None else source_port(facts, function.returns).port
    return_world = None if function.return_world is None else source_port(facts, function.return_world).port
    remember(return_port); remember(return_world)
    call_sites = []
    call_ordinals = {slicer.fact(facts, 'calls', index): index for index in range(len(facts.calls))}
    for record in flow.calls:
        call = record.call
        query = call_ordinals[record.fact]
        def endpoint(index):
            return None if index is None else remember(source_port(facts, index).port)
        call_id = 'source-call:'+semantic_sha256(encode(record.fact))
        call_sites.append(dict(id=call_id, call_id=call_id, query=query, fact=encode(record.fact),
            callee=endpoint(call.callee),
            positional=[dict(ordinal=i, port=endpoint(value)) for i, value in enumerate(call.positional)],
            keywords=[dict(ordinal=i, name=name, port=endpoint(value))
                for i, (name, value) in enumerate(call.keywords)],
            positional_pack=endpoint(call.positional_pack), keyword_pack=endpoint(call.keyword_pack),
            result=endpoint(call.result), world=endpoint(call.world), result_world=endpoint(call.result_world),
            target_set_complete=False, target_transfer='UNRESOLVED_ORIGINAL_CALL_OUTPUT_TOKEN',
            conditions=list(CONDITIONS)))
        token(source_port(facts, call.result).port, 'CALL_RESULT', (record.fact,), call_id=call_id)
        token(source_port(facts, call.result_world).port, 'CALL_WORLD', (record.fact,), call_id=call_id)

    reads, writes = [], []
    for location in flow.locations:
        for write, accesses in ((False, location.reads), (True, location.writes)):
            for access in accesses:
                (writes if write else reads).append(dict(kind='LOCAL_BINDING', fact=encode(access.fact),
                    port=remember(access.value), namespace=encode(location.namespace),
                    conditions=['FLOW_INSENSITIVE_LEXICAL_MAY_WRITER_READER_JOIN']))
    for boundary in flow.name_boundaries:
        remember(boundary.access.value)
        row = dict(kind='LOCAL_BINDING' if boundary.namespace.kind is F.NamespaceKind.LEXICAL
            else 'NONLOCAL_BINDING', fact=encode(boundary.access.fact),
            port=ports[boundary.access.value], namespace=encode(boundary.namespace),
            reason=boundary.reason, conditions=list(CONDITIONS))
        # Missing local producers are already listed by flow.locations.
        if row['kind'] != 'LOCAL_BINDING':
            (writes if boundary.write else reads).append(row)
        if not boundary.write:
            token(boundary.access.value, 'EXTERNAL_READ', (boundary.access.fact,), source_kind=row['kind'])
    owners = {(operation.entry.template, operation.entry.owner) for operation in flow.operations}
    owners.update((edge.source.template, edge.source.owner) for edge in flow.edges)
    owners.update((edge.target.template, edge.target.owner) for edge in flow.edges if edge.kind is not G.EdgeKind.DECLARE)
    spans = []
    for group in facts.scope_groups:
        if group.owner == function.body_scope.owner:
            spans.extend(group.tables)
    seen_accesses = set()
    for span in spans:
        if span.table not in (F.FactTable.ATTRIBUTES, F.FactTable.ITEMS):
            continue
        table = span.table.value
        for ordinal in range(span.start, span.stop):
            if (table, ordinal) in seen_accesses:
                continue
            seen_accesses.add((table, ordinal))
            access = getattr(facts, table)[ordinal]
            receiver = source_port(facts, access.receiver).port
            value = None if access.value is None else source_port(facts, access.value).port
            if (receiver.template, receiver.owner) not in owners and (
                    value is None or (value.template, value.owner) not in owners):
                continue
            fact = slicer.fact(facts, table, ordinal)
            write = (access.action in (F.AttributeAction.SET, F.AttributeAction.DELETE)
                if table == 'attributes' else access.write)
            row = dict(kind='ATTRIBUTE' if table == 'attributes' else 'ITEM', fact=encode(fact),
                port=remember(value), receiver=remember(receiver),
                world=remember(source_port(facts, access.world).port),
                result_world=remember(source_port(facts, access.result_world).port),
                conditions=list(CONDITIONS))
            if table == 'attributes':
                row.update(name=access.name, action=access.action.value)
            else:
                row['key_is_port'] = type(access.key) is int
                row['key'] = (remember(source_port(facts, access.key).port)
                    if type(access.key) is int else encode(access.key))
            (writes if write else reads).append(row)
            if not write and value is not None:
                token(value, 'EXTERNAL_READ', (fact,), source_kind=row['kind'])

    operation_rules = []
    for operation in flow.operations:
        rule = classify_instruction(operation.instruction)
        operation_rules.append(dict(entry=encode(gp(operation.entry)), classification=encode(rule)))
        if _is_call(operation) or type(operation.instruction) is P.AccessBinding:
            continue
        if rule.resolution.value in ('DYNAMIC_TARGET_OR_PROTOCOL', 'ADDRESS_RULE', 'INSTANCE_CONDITION'):
            for value in operation.outputs:
                if not _state(value.value_type):
                    token(value.port, 'EXTERNAL_READ', source_kind='UNRESOLVED_OPERATION_RESULT',
                        source_operation=encode(gp(operation.entry)), classification=encode(rule))

    seeds = [key[0] for key, index in sorted(seed_indices.items(), key=lambda row: row[1])]
    bits, complete, work, vertices = _propagate(flow, seeds, max_work=max_work)
    dependencies = []
    port_index = {}
    def plain_relation(port):
        if port is None:
            return dict(port=None, data=[], environment=[], state=[], unknown=['ORIGINAL_PORT_NOT_AVAILABLE'])
        mask = bits[port]
        selected = [row for index, row in enumerate(tokens) if mask & (1 << index)]
        return dict(port=ports[port],
            data=[row['id'] for row in selected if row['kind'] not in ('FORMAL_ENVIRONMENT','FORMAL_STATE','CALL_WORLD')],
            environment=[row['id'] for row in selected if row['kind']=='FORMAL_ENVIRONMENT'],
            state=[row['id'] for row in selected if row['kind'] in ('FORMAL_STATE','CALL_WORLD')],
            unknown=[row['id'] for row in selected if row['kind'] in ('CALL_RESULT','CALL_WORLD','EXTERNAL_READ')]
                + ([] if complete else ['LOCAL_DEPENDENCY_BUDGET_EXHAUSTED']))
    selector_ports = tuple(sorted({edge.condition for edge in flow.edges
        if edge.kind is G.EdgeKind.CONTROL and edge.condition is not None}, key=lambda port:_canonical(encode(port))))
    for port in selector_ports:
        remember(port)
    channels = ('data', 'environment', 'state', 'unknown')
    def union(rows):
        return {channel:sorted({token for row in rows for token in row[channel]}) for channel in channels}
    selectors = union([plain_relation(port) for port in selector_ports])
    selectors.update(conditions=[ports[port] for port in selector_ports],
        scope='ALL_ORIGINAL_BODY_GUARDS_CONSERVATIVE_NOT_PATH_SENSITIVE')
    def relation(port):
        plain = plain_relation(port)
        return dict(port=plain['port'], **union((plain, selectors)),
            without_control={key:plain[key] for key in channels})
    for port in sorted(query_ports, key=lambda p: _canonical(ports[p])):
        port_index[_canonical(ports[port])] = len(dependencies)
        dependencies.append(relation(port))
    for row in (*reads, *writes):
        def lookup(wire):
            return dict(data=[], environment=[], state=[], unknown=[]) if wire is None else dependencies[port_index[_canonical(wire)]]
        row['value_dependencies'] = union((lookup(row['port']),))
        row['address_dependencies'] = union((lookup(row.get('receiver')),
            lookup(row.get('key') if row.get('key_is_port') else None)))
        row['state_dependencies'] = union((lookup(row.get('world')), lookup(row.get('result_world'))))
        row['dependencies'] = union((row['value_dependencies'], row['address_dependencies'],
            row['state_dependencies'], selectors))
    for row in call_sites:
        row['dispatch_environment_dependencies'] = union((dependencies[port_index[_canonical(row['callee'])]],))
    frontier = [dict(reason='CALL_TARGET_TRANSFER_NOT_COMPOSED', call_id=row['id'], fact=row['fact'])
        for row in call_sites]
    frontier.extend(dict(reason='SOURCE_EXTERNAL_READ_OR_PROTOCOL_RESULT_UNRESOLVED', token=row['id'])
        for row in tokens if row['kind']=='EXTERNAL_READ')
    if not complete:
        frontier.append(dict(reason='LOCAL_DEPENDENCY_BUDGET_EXHAUSTED', max_work=max_work))
    if function.environment.kind is not P.DeferredBodyKind.FUNCTION:
        frontier.append(dict(reason='DEFERRED_BODY_NORMAL_RETURN_IS_NOT_ORDINARY_CALL_RETURN'))
    source.check(); source.material.validate_parents()
    artifact_path = Path(view.catalog.path)/'result.json'
    raw = artifact_path.read_bytes()
    catalog = ProviderArtifact(str(artifact_path), sha256(raw).hexdigest(), len(raw), view.catalog.report['schema'])
    raw = Path(__file__).read_bytes()
    result = dict(schema=SCHEMA, schema_version=1, function_id=function_id,
        unit=unit, definition=encode(definition), function_fact=encode(definition),
        source_function=encode(callable_port), body=encode(body), body_scope=encode(function.body_scope),
        body_point=encode(view.source_region(unit, function.template, function.body)),
        source=dict(path=model['path'], sha256=model['raw_sha256']),
        source_refs=[encode(catalog), encode(flow.source_templates), encode(definition.artifact)],
        producer=encode(ProviderArtifact(str(Path(__file__).resolve()), sha256(raw).hexdigest(), len(raw), SCHEMA+'.producer')),
        ports=dict(formals=formals, environment=formals[0]['port'],
            parameters=[row for row in formals if row['role']=='DATA'], state=formals[-1]['port'],
            **{'return':remember(return_port), 'return_world':remember(return_world)}),
        normal_return=relation(return_port), normal_return_world=relation(return_world),
        tokens=tokens, port_dependencies=dependencies, port_index=port_index,
        control_selector_dependencies=selectors,
        call_sites=call_sites, reads=reads, writes=writes,
        effects={field:dict(status='UNKNOWN', reason='SOURCE_ACCESS_LEDGER_IS_NOT_DYNAMIC_EFFECT_CLOSURE') for field in _EFFECTS},
        operation_classifications=operation_rules,
        conditions=list(CONDITIONS), frontier=frontier,
        propagation=dict(algorithm='ALL_FINITE_ORIGINAL_PORT_SEEDS_INTEGER_BITSET_SPARSE_WORKLIST',
            max_work=max_work, work=work, adjacency_vertices=vertices, local_fixed_point_reached=complete,
            state_taints_data=False, call_results_cut_before_unknown_target_transfer=True),
        summary=dict(formal_data_parameters=len(function.parameter_names), calls=len(call_sites),
            tokens=len(tokens), source_reads=len(reads), source_writes=len(writes),
            read_kinds=dict(Counter(row['kind'] for row in reads)), write_kinds=dict(Counter(row['kind'] for row in writes))),
        normal_completion_only=True, source_accesses_complete_for_selected_body=False,
        source_access_fact_tables_scanned=['NAMES', 'ATTRIBUTES', 'ITEMS'],
        control_dependencies_are_conservative_body_guard_union=True,
        source_bodies_copied=False, loops_unrolled=False, recursive_calls_inlined=False,
        path_sensitive=False, mutation_freedom_proven=False, purity_proven=False,
        target_executed=False, runtime_invocations_created=False, transformation_authorized=False,
        first_stage_complete=False)
    result['id'] = semantic_sha256(result)
    return result


def verify_source_function_summary(report, view, unit, function):
    expected = derive_source_function_summary(view, unit, function,
        max_work=report['propagation']['max_work'])
    if _canonical(report) != _canonical(expected):
        raise ValueError('source function summary differs from original F4 value-flow replay')
    return expected


__all__ = ['derive_source_function_summary', 'verify_source_function_summary']
