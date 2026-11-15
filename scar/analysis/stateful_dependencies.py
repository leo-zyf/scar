"""Dependency queries over an existing ordered StatefulRegion.

This is a projection, not another program IR. Expr/action/input identities and
ordered epochs come from the original region. An alias descriptor is distinct
from the content version read through it at an action. No runtime identity,
caller-loop edge, tensor invariance or transformation permission is invented.
"""
from dataclasses import asdict
from pathlib import Path
import json

from scar.compiler.stateful_model import (SCHEMA as REGION_SCHEMA, stateful_from_dict,
    stateful_digest, validate_stateful_region)
from scar.ir.execution_recipe import artifact_reference, read_artifact


SCHEMA = 'scar.stateful-dependency-projection'
_ORDER = 'ORIGINAL_ACTION_ORDER'


def project_stateful_dependencies(region, *, out=None):
    """Expose original per-call value/state flow and a reusable transfer summary.

    A path retains one byte-checked source artifact; origin coordinates point
    into it rather than copying nested F4 provenance into each graph record.
    A caller must independently supply call/loop/effect proofs before treating
    the transfer summary as an iteration recurrence.
    """
    artifact = None
    if isinstance(region, (str, Path)):
        artifact = artifact_reference(region, REGION_SCHEMA)
        region = stateful_from_dict(read_artifact(artifact))
    region = validate_stateful_region(region)
    records = []; edges = []; reads = []; writes = []
    nodes = {node.id: node for node in region.nodes}
    expression = {value.id: '#/nodes/' + str(i) for i, value in enumerate(region.nodes)}
    actions = {value.id: '#/actions/' + str(i) for i, value in enumerate(region.actions)}
    ports = {value.id: '#/inputs/' + str(i) for i, value in enumerate(region.inputs)}
    kinds = {}; descriptors = {}; states = {}; entry = {}; current = {}

    def record(key, kind, **fields):
        records.append(dict(key=key, kind=kind, **fields)); return key

    def edge(source, target, kind, **fields):
        edges.append(dict(source=source, target=target, kind=kind, **fields))

    def origins(table, index, values):
        return [f'#/{table}/{index}/origins/{i}' for i in range(len(values))]

    def state(name, epoch):
        identity = name, epoch
        if identity not in states:
            # This key addresses the original ordered epoch pair. It is not a
            # newly established LogicalValueID or a physical allocation ID.
            key = 'epoch:' + json.dumps([name, epoch], separators=(',', ':'))
            states[identity] = record(key, 'ORDERED_ARRAY_CONTENT_VERSION', input_id=name,
                input_port=ports[name], epoch=epoch, region_entry=(epoch == 0),
                identity_scope='ONE_ORIGINAL_REGION_INVOCATION_UNDER_ALIAS_GUARDS')
        return states[identity]

    for i, value in enumerate(region.inputs):
        record(ports[value.id], 'INPUT_PORT', original_id=value.id, path=list(value.path),
            input_kind=value.kind, kind_authority=value.kind_authority,
            origins=origins('inputs', i, value.origins))
        if value.kind == 'array':
            entry[value.id] = state(value.id, 0); current[value.id] = 0

    for i, node in enumerate(region.nodes):
        kind = ('ALIAS_DESCRIPTOR' if node.op == 'slice' or node.op == 'input' and node.attrs['kind'] == 'array'
            else 'LITERAL' if node.op == 'const' else 'IMMUTABLE_EVALUATED_VALUE')
        kinds[node.id] = ('array' if node.op == 'slice' else node.attrs['kind'] if node.op == 'input'
            else 'scalar' if node.op == 'const' else
            'array' if any(kinds[value] == 'array' for value in node.inputs) else 'scalar')
        record(expression[node.id], kind, original_id=node.id, op=node.op,
            value_kind=kinds[node.id], attributes=node.attrs,
            origins=[f'#/source/origins/{j}' for j, origin in enumerate(region.source.origins)
                if origin.get('expression') == node.id])
        if node.op == 'input':
            edge(ports[node.attrs['name']], expression[node.id],
                'STORAGE_REFERENCE' if kinds[node.id] == 'array' else 'SCALAR_INPUT_VALUE')
            if kinds[node.id] == 'array':
                descriptors[node.id] = dict(input_id=node.attrs['name'], epoch=node.attrs['epoch'],
                    slice_chain=[], descriptor=expression[node.id])
        elif node.op == 'slice':
            parent = descriptors[node.inputs[0]]
            descriptors[node.id] = dict(input_id=parent['input_id'], epoch=parent['epoch'],
                slice_chain=parent['slice_chain'] + [node.attrs['bounds']], descriptor=expression[node.id])
            edge(expression[node.inputs[0]], expression[node.id], 'ALIAS_DESCRIPTOR_SOURCE')

    def operand(value, action, index):
        edge(expression[value], action, 'OPERAND_VALUE' if value not in descriptors else 'OPERAND_DESCRIPTOR',
            operand_index=index)
        if value in descriptors:
            descriptor = descriptors[value]
            if descriptor['epoch'] != current[descriptor['input_id']]:
                raise ValueError('descriptor content read differs from current original ordered epoch')
            dependency = dict(action=action, expression=expression[value], **descriptor)
            reads.append(dependency)
            edge(state(descriptor['input_id'], descriptor['epoch']), action, 'ARRAY_CONTENT_READ',
                descriptor=expression[value], slice_chain=descriptor['slice_chain'], operand_index=index)

    previous = None
    for i, action in enumerate(region.actions):
        key = actions[action.id]
        record(key, 'EVALUATE' if action.kind == 'evaluate' else 'ORDERED_ARRAY_WRITE',
            original_id=action.id, action_index=i,
            origins=origins('actions', i, action.origins),
            world_before_pointer=None if action.world_before is None else key + '/world_before',
            world_after_pointer=None if action.world_after is None else key + '/world_after')
        if previous is not None: edge(previous, key, _ORDER)
        previous = key
        if action.kind == 'evaluate':
            for index, value in enumerate(nodes[action.value].inputs): operand(value, key, index)
            edge(key, expression[action.value], 'PRODUCES_IMMUTABLE_VALUE')
        else:
            operand(action.value, key, 0)
            before = state(action.target, action.epoch_before)
            after = state(action.target, action.epoch_after)
            edge(ports[action.target], key, 'WRITE_DESTINATION_REFERENCE')
            # Full overwrite has no content dependency on its old destination;
            # partial overwrite still needs the old unmodified exterior.
            full = action.bounds is None or all(start is None and stop is None for start, stop in action.bounds)
            if action.op != 'assign': edge(before, key, 'READ_MODIFY_WRITE')
            elif not full: edge(before, key, 'PRESERVES_UNWRITTEN_REGION')
            edge(key, after, 'PRODUCES_ORDERED_CONTENT_VERSION')
            writes.append(dict(action=key, input_id=action.target, before=before, after=after,
                op=action.op, bounds=action.bounds, entire_array_overwrite=full and action.op == 'assign'))
            current[action.target] = action.epoch_after

    exits = {name: state(name, epoch) for name, epoch in current.items()}
    missing = [dict(kind='CALLER_LOOP_CONNECTION', status='UNRESOLVED',
        detail='Region artifact has no caller callsite, original loop header/backedge, actual/formal mapping '
            'or intervening field/storage effect proof. Per-call entry/exit ports cannot establish a loop recurrence.')]
    report = dict(schema=SCHEMA, schema_version=1, original_region=artifact,
        region_id=region.id, region_sha256=stateful_digest(region),
        source={key: value for key, value in asdict(region.source).items() if key != 'origins'},
        identity='ORIGINAL_REGION_RECORD_POINTERS_AND_REGION_LOCAL_ORDERED_EPOCH_PAIRS',
        nodes=records, edges=edges, entry_states=entry, exit_states=exits,
        reads=reads, writes=writes,
        callees=[dict(path=list(value.path), source={key: item for key, item in asdict(value.source).items()
            if key != 'origins'}, origins=origins('callees', i, value.origins)) for i, value in enumerate(region.callees)],
        providers=[dict(id=value.id, module=value.module, symbol=value.symbol,
            semantics=value.semantics, origins=origins('providers', i, value.origins)) for i, value in enumerate(region.providers)],
        transfer=dict(scope='ONE_ORIGINAL_METHOD_CALL', entry_states=entry, exit_states=exits,
            written_inputs=sorted({value.target for value in region.actions if value.kind == 'write'}),
            scalar_inputs=[ports[value.id] for value in region.inputs if value.kind == 'scalar'],
            same_object_across_calls_proven=False, same_storage_across_calls_proven=False,
            loop_recurrence_proven=False, iterations_unrolled=False),
        contracts=[dict(condition=value, status='UNDISCHARGED_PROFILE_REQUIREMENT') for value in region.guards],
        effects=dict(array_content_reads='EXPLICIT_UNDER_PROFILE_CONTRACT',
            array_content_writes='EXPLICIT_ORIGINAL_ORDER_AND_REGIONS',
            rng='NOT_PROVEN_BY_THIS_PROJECTION', external_effect='NOT_PROVEN_BY_THIS_PROJECTION',
            exceptions='ORIGINAL_ACTION_ORDER_RETAINED_NOT_ELIMINATED'),
        frontiers=missing, transformation_authorized=False)
    # Pin checking brackets the derivation. Projection output never upgrades
    # the original source's provenance or undischarged profile guards.
    if artifact is not None: read_artifact(artifact, json_document=False)
    if out is not None:
        out = Path(out).absolute(); out.mkdir(parents=True, exist_ok=False)
        (out/'dependencies.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n')
    return report


def trace_dependencies(projection, endpoint, *, include_ordering=False):
    """Trace a projected original endpoint to entry/literal roots.

    ``endpoint`` is an original record pointer or a key in entry/exit_states.
    Alias descriptors remain references; ARRAY_CONTENT_READ and write edges
    supply the separate content dependencies. Callee expression origins are
    retained as pointers into the one original region, never guessed calls.
    """
    if projection.get('schema') != SCHEMA or projection.get('schema_version') != 1:
        raise ValueError('original stateful dependency projection required')
    records = {row['key']: row for row in projection['nodes']}
    if len(records) != len(projection['nodes']) or endpoint not in records:
        raise ValueError('unique original dependency endpoint required')
    incoming = {}
    for index, edge in enumerate(projection['edges']):
        if edge['source'] not in records or edge['target'] not in records:
            raise ValueError('dependency edge has no original endpoint')
        if include_ordering or edge['kind'] != _ORDER:
            incoming.setdefault(edge['target'], []).append(index)
    seen = set(); pending = [endpoint]; used = set(); roots = []
    while pending:
        key = pending.pop()
        if key in seen: continue
        seen.add(key)
        indices = incoming.get(key, ())
        if not indices: roots.append(key)
        for index in indices:
            used.add(index); pending.append(projection['edges'][index]['source'])
    return dict(endpoint=endpoint, roots=[records[key] for key in sorted(roots)],
        visited_keys=sorted(seen), edge_indices=sorted(used), includes_original_action_order=include_ordering,
        content_invariance_proven=False, transformation_authorized=False)


__all__ = ['project_stateful_dependencies', 'trace_dependencies']
