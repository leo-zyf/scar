"""Project guarded F4 array summaries onto admitted storage regions."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

from scar.analysis.replay_transaction import replay_transaction
from scar.analysis.strided_content_flow import AccessKind, ContentAccess, analyze_loop_content
from scar.compiler.input_metadata import stateful_shapes_from_metadata
from scar.compiler.stateful_model import SCHEMA as REGION, document, stateful_from_dict, validate_stateful_region
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.record_codec import encode
from scar.ir.v2 import StorageAllocationID, StorageRegion, StorageRegionID


SCHEMA = 'scar.stateful-storage-accesses'


@dataclass(frozen=True, slots=True)
class StorageAccessProjection:
    """A footprint conditional on the original provider and storage guards."""

    access: ContentAccess
    source: tuple[dict, ...]
    action_ordinal: int | None = None


def observed_storage_bindings(region, metadata, *, observation=0):
    """Bind distinct owning arrays in one admitted stateful entry observation."""
    shapes, proof = stateful_shapes_from_metadata(metadata, region)
    reference = proof['artifact']
    records = [row for row in read_artifact(reference)['records']
               if (row['source']['path'], row['source']['sha256'], row['source']['code_key'])
               == (region.source.path, region.source.sha256, region.source.code_key)]
    if type(observation) is not int or not 0 <= observation < len(records):
        raise ValueError('expected one matched original boundary observation')
    selected = records[observation]
    coordinate = selected['boundary_observation']
    bindings = {}
    owners = set()
    for name, shape in shapes.items():
        value = selected['inputs'][name]
        geometry = value.get('storage_observation')
        if type(geometry) is not dict:
            raise ValueError('allocation geometry is open at the selected boundary')
        ordinal = geometry['owner_ordinal']
        offset, capacity = (geometry[key] for key in ('byte_offset', 'allocation_nbytes'))
        strides = geometry['byte_strides']
        if (type(ordinal) is not int or ordinal < 0 or ordinal in owners or type(offset) is not int
                or type(capacity) is not int or offset < 0 or offset % 8 or capacity < 0
                or geometry['itemsize'] != 8 or geometry['dtype'] != value['dtype']
                or geometry['shape'] != shape or strides != value['strides'] or any(s % 8 for s in strides)
                or geometry['writeable'] is not True):
            raise ValueError('expected distinct writable f64 roots with original geometry')
        bound = [offset, offset] if not all(shape) else [
            offset + sum(min(0, (n - 1)*s) for n, s in zip(shape, strides, strict=True)),
            offset + sum(max(0, (n - 1)*s) for n, s in zip(shape, strides, strict=True)) + 8]
        if geometry['byte_bounds'] != bound or not 0 <= bound[0] <= bound[1] <= capacity:
            raise ValueError('observed bytes exceed their original allocation')
        owners.add(ordinal)
        scope = f'{reference["sha256"]}/{observation}'
        bindings[name] = StorageRegion(StorageRegionID(f'{scope}/{name}'),
            StorageAllocationID(f'{scope}/owner/{ordinal}'), offset // 8, tuple(shape),
            tuple(s // 8 for s in strides), 'float64', 'cpu')
    next_boundary = None
    if observation + 1 < len(records):
        following = records[observation + 1]
        next_coordinate = following['boundary_observation']
        if (next_coordinate['run'], next_coordinate['process_id']) == (coordinate['run'], coordinate['process_id']):
            comparable = {}
            for name in bindings:
                item = following['inputs'][name]
                link = item.get('previous_storage_observation')
                if link is None or link['boundary_ordinal'] != coordinate['ordinal']:
                    continue
                before = selected['inputs'][name]['storage_observation']
                after = item['storage_observation']
                same_geometry = all(before[key] == after[key] for key in before if key != 'owner_ordinal')
                if type(link['same_owner']) is not bool or link['same_geometry'] is not same_geometry:
                    raise ValueError('original boundary geometry comparison differs')
                comparable[name] = dict(same_owner=link['same_owner'], same_geometry=same_geometry)
            if comparable:
                next_boundary = dict(observation=next_coordinate, arrays=comparable,
                                     allocation_continuity_between_observations_verified=False)
    evidence = dict(proof, selected_observation=coordinate,
                    storage_scope='ONE_ENTRY_OBSERVATION_CONTINUITY_NOT_ASSUMED')
    if next_boundary is not None:
        evidence['next_observed_boundary'] = next_boundary
    return bindings, evidence


def _view(region, bounds, identity):
    shape, offset = [], region.offset
    for (lo, hi), size, stride in zip(bounds, region.shape, region.strides, strict=True):
        start, stop, _ = slice(lo, hi).indices(size)
        shape.append(max(stop - start, 0))
        offset += start * stride
    return StorageRegion(StorageRegionID(identity), region.allocation, offset if all(shape) else region.offset,
                         tuple(shape), region.strides, region.dtype, region.device)


def _require_roots(bindings, expected):
    if bindings.keys() != expected:
        raise ValueError('every original array input needs one admitted root region')
    if any(type(row) is not StorageRegion or len(row.shape) != 2 for row in bindings.values()):
        raise ValueError('expected rank-two typed storage bindings for this array profile')


def project_stateful_accesses(region, bindings, *, occurrence):
    """Retain each original action's external array accesses in execution order.

    An alias descriptor does not read elements. A materialized arithmetic
    result was already evaluated at its own action and does not reread roots.
    Augmented stores read the destination before updating it.
    """
    region = validate_stateful_region(region)
    _require_roots(bindings, {row.id for row in region.inputs if row.kind == 'array'})
    views = {}
    for expr in region.nodes:
        if expr.op == 'input' and expr.attrs['kind'] == 'array':
            views[expr.id] = bindings[expr.attrs['name']]
        elif expr.op == 'slice':
            views[expr.id] = _view(views[expr.inputs[0]], expr.attrs['bounds'],
                                   f'{occurrence}/view/{expr.id}')

    nodes = {expr.id: expr for expr in region.nodes}
    result = []
    for ordinal, action in enumerate(region.actions):
        prefix = f'{occurrence}/{action.id}'
        operands = nodes[action.value].inputs if action.kind == 'evaluate' else (action.value,)
        for index, operand in enumerate(operands):
            if operand in views:
                result.append(StorageAccessProjection(
                    ContentAccess(f'{prefix}/operand/{index}', AccessKind.READ, views[operand]),
                    action.origins, ordinal))
        if action.kind == 'write':
            target = bindings[action.target]
            if action.bounds is not None:
                target = _view(target, action.bounds, f'{prefix}/view')
            kind = AccessKind.WRITE if action.op == 'assign' else AccessKind.READ_WRITE
            result.append(StorageAccessProjection(ContentAccess(f'{prefix}/target', kind, target),
                                                   action.origins, ordinal))
    return tuple(result)


def project_array_copy_accesses(summary, bindings, *, occurrence):
    """Reuse a replayed copy-only summary and its original ordered rectangles."""
    from scar.analysis.array_copy_effects import SCHEMA
    from scar.optimization.resource_solver import semantic_sha256

    body = {key: value for key, value in summary.items() if key != 'id'}
    if (summary['schema'], summary['schema_version'], summary['status']) != (SCHEMA, 1, 'CLOSED_COPY_EFFECTS_CONDITIONAL'):
        raise ValueError('expected a closed, conditional array copy summary')
    if summary['id'] != 'array-copy-effects:' + semantic_sha256(body):
        raise ValueError('array copy summary identity differs')
    _require_roots(bindings, {root['formal_index'] for root in summary['roots']})
    if any(tuple(root['shape']) != bindings[root['formal_index']].shape for root in summary['roots']):
        raise ValueError('array copy summary and live storage shapes differ')
    result = []
    for item in summary['accesses']:
        if item['kind'] == 'CREATE_ALIAS_VIEW':
            continue
        kind = {'READ_SCALAR_VALUE': AccessKind.READ, 'READ_COPY_SOURCE': AccessKind.READ,
                'WRITE_COPY_DESTINATION': AccessKind.WRITE}[item['kind']]
        identity = f'{occurrence}/{item["id"]}'
        view = _view(bindings[item['formal_index']], item['rectangle'], identity)
        result.append(StorageAccessProjection(ContentAccess(identity, kind, view), (item['source'],)))
    return tuple(result)


def _repeated_read_records(flow):
    interval_sets = {}

    def partition(parts):
        records = []
        for part in parts:
            ordinal = (None if part.intervals is None
                       else interval_sets.setdefault(part.intervals, len(interval_sets)))
            records.append(dict(kind=part.kind.value, byte_intervals=ordinal,
                                writer=part.writer, iteration_distance=part.iteration_distance))
        return records

    records = [dict(read=row.read, allocation=encode(row.allocation),
                    first_iteration=partition(row.first_iteration), later_iterations=partition(row.later_iterations))
               for row in flow]
    return records, encode(tuple(interval_sets))


@replay_transaction()
def project(catalog, boundary, metadata, *, observation=0):
    """Rebuild the source region and project one observed storage context."""
    from scar.analysis.program_view import ProgramView
    from scar.compiler.stateful_frontend import discover_stateful_regions

    view = ProgramView(catalog)
    record = artifact_reference(view.catalog.path / 'result.json', view.catalog.report['schema'])
    reference = artifact_reference(Path(boundary).resolve(strict=True), REGION)
    raw = read_artifact(reference)
    region = stateful_from_dict(raw)
    candidates, _ = discover_stateful_regions(view, region.source.path)
    matching = [item for item in candidates if item.id == region.id]
    if len(matching) != 1 or encode(document(matching[0])) != raw:
        raise ValueError('array region differs from fresh original F4 derivation')
    bindings, proof = observed_storage_bindings(region, metadata, observation=observation)
    rows = project_stateful_accesses(region, bindings, occurrence=region.id)
    recurrent, intervals = _repeated_read_records(analyze_loop_content(row.access for row in rows))
    accesses = [dict(access=encode(row.access), source_action=f'#/actions/{row.action_ordinal}') for row in rows]
    return dict(schema=SCHEMA, schema_version=1, catalog=record, boundary=reference,
                observation=observation, storage_evidence=proof, bindings=encode(bindings),
                accesses=accesses, conditional_repeated_body=recurrent, byte_interval_sets=intervals,
                repeated_body_guards=[
                    'ORIGINAL_STATEFUL_PROVIDER_AND_NORMAL_COMPLETION',
                    'IDENTICAL_CONTROL_SEQUENCE_AND_STORAGE_REGIONS_ON_BACKEDGE',
                    'NO_ADDITIONAL_MAY_ALIAS_EFFECTS_BETWEEN_ACCESS_OCCURRENCES',
                ], loop_unrolled=False, tensor_contents_recorded=False,
                caller_backedge_and_intervening_effects_verified=False, transformation_authorized=False)


def check(path):
    reference = artifact_reference(Path(path).resolve(strict=True), SCHEMA)
    saved = read_artifact(reference)
    catalog = saved['catalog']
    read_artifact(catalog)
    rebuilt = project(Path(catalog['uri']).parent, saved['boundary']['uri'],
                      saved['storage_evidence']['artifact']['uri'], observation=saved['observation'])
    if saved != rebuilt:
        raise ValueError('storage access projection differs from replay')
    return rebuilt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('catalog', 'boundary', 'metadata', 'out', 'check'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--observation', type=int, default=0)
    args = parser.parse_args(argv)
    inputs = (args.catalog, args.boundary, args.metadata, args.out)
    if args.check and not any(inputs) and args.observation == 0:
        result = check(args.check)
    elif not args.check and all(inputs):
        result = project(args.catalog, args.boundary, args.metadata, observation=args.observation)
        with args.out.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, separators=(',', ':'), sort_keys=True, allow_nan=False)
            stream.write('\n')
    else:
        parser.error('provide --check, or --catalog, --boundary, --metadata and --out')
    print(json.dumps(dict(schema=SCHEMA, roots=len(result['bindings']), accesses=len(result['accesses']),
                          reads=len(result['conditional_repeated_body']))))


if __name__ == '__main__':
    main()
