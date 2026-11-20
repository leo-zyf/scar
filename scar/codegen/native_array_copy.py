"""Lower replayed original slice-copy effects to a native ABI 1 component."""

from pathlib import Path

from scar.analysis.array_copy_effects import SCHEMA as COPY_EFFECTS
from scar.optimization.resource_solver import semantic_sha256


SCHEMA = 'scar.native-array-copy-plan'
SOURCE_SCHEMA = 'scar.source-native-update'
_TEMPLATE = Path(__file__).with_name('native_array_copy_v1.cpp.in')
_MAX_BYTES = (1 << 63) - 1
_ITEM_SIZE = 8


def _need(condition, detail):
    if not condition:
        raise ValueError('native array copy: ' + detail)


def _shape(shape):
    _need(type(shape) in (list, tuple) and len(shape) == 2
          and all(type(extent) is int and 0 <= extent <= _MAX_BYTES for extent in shape),
          'original rank-two shape required')
    rows, columns = shape
    _need(columns * _ITEM_SIZE <= _MAX_BYTES and rows * columns * _ITEM_SIZE <= _MAX_BYTES,
          'root geometry exceeds native addressable size')
    return list(shape)


def _selection(access, rectangle, shape):
    _need(type(rectangle) is list and len(rectangle) == 2 and rectangle == access['rectangle']
          and all(type(axis) is list and len(axis) == 2 and all(type(point) is int for point in axis)
                  and 0 <= axis[0] <= axis[1] <= shape[i] for i, axis in enumerate(rectangle)),
          'copy selection differs from bounded original access')
    squeezed = access['squeezed_axes']
    _need(squeezed in ([], [0], [1], [0, 1]), 'invalid original squeezed axes')
    visible_axes = [i for i in (0, 1) if i not in squeezed]
    extents = [stop - start for start, stop in rectangle]
    visible_shape = [extents[i] for i in visible_axes]
    _need(visible_shape == access['visible_shape'] and all(extents[i] == 1 for i in squeezed),
          'original squeezed and physical extents differ')
    row_stride = shape[1] * _ITEM_SIZE
    strides = (row_stride, _ITEM_SIZE)
    offset = rectangle[0][0] * row_stride + rectangle[1][0] * _ITEM_SIZE
    return visible_shape, [strides[i] for i in visible_axes], offset


def _overlap(left, right):
    return all(max(a, c) < min(b, d) for (a, b), (c, d) in zip(left, right, strict=True))


def _moves(call, ordinal, roots):
    summary = call['copy_effects']
    body = dict(summary)
    identity = body.pop('id', None)
    _need((summary['schema'], summary['schema_version']) == (COPY_EFFECTS, 1)
          and summary['status'] == 'CLOSED_COPY_EFFECTS_CONDITIONAL' and not summary['frontiers']
          and identity == 'array-copy-effects:' + semantic_sha256(body),
          'expected a closed original copy effect summary')
    formals = {row['formal_index']: row for row in summary['roots']}
    bound = {row['formal_index']: row['root_id'] for row in call['formal_to_root']}
    _need(len(formals) == len(summary['roots']) == len(bound) == len(call['formal_to_root'])
          and set(bound) == set(formals) and len(set(bound.values())) == len(bound)
          and all(root in roots and formals[index]['shape'] == roots[root]['shape']
                  for index, root in bound.items()),
          'original copy formals need distinct exact array roots and shapes')

    accesses = {row['id']: row for row in summary['accesses']}
    _need(len(accesses) == len(summary['accesses']), 'copy source accesses are ambiguous')
    epochs = dict.fromkeys(formals, 0)
    for index, row in enumerate(summary['copies']):
        source_id, destination_id = row['source_formal'], row['destination_formal']
        source, destination = accesses[row['source_access']], accesses[row['destination_access']]
        _need(row['id'] == f'copy:{index}' and source['formal_index'] == source_id
              and destination['formal_index'] == destination_id
              and (source['kind'], destination['kind']) in (
                  ('READ_COPY_SOURCE', 'WRITE_COPY_DESTINATION'),
                  ('READ_SCALAR_VALUE', 'WRITE_COPY_DESTINATION'))
              and source['local_epoch'] == row['source_epoch'] == epochs[source_id]
              and destination['local_epoch_before'] == row['destination_epoch_before'] == epochs[destination_id]
              and destination['local_epoch_after'] == row['destination_epoch_after'] == epochs[destination_id] + 1,
              'copy lowering requires live source content and ordered destination epochs')
        epochs[destination_id] += 1
        left, right = row['source_rectangle'], row['destination_rectangle']
        left_id, right_id = bound[source_id], bound[destination_id]
        left_port, right_port = roots[left_id], roots[right_id]
        source_shape, source_strides, source_offset = _selection(source, left, left_port['shape'])
        dest_shape, dest_strides, dest_offset = _selection(destination, right, right_port['shape'])
        _need(source_shape == dest_shape and
              ((left_id == right_id and row['source_destination_overlap'] is False and not _overlap(left, right))
               or (left_id != right_id and row['source_destination_overlap'] is None)),
              'overlapping source and destination requires the original copy protocol')

        # The last visible axis is unit-stride for both rank-two rectangles.
        # A squeezed lane can instead be a row or column; retain each stride.
        if not source_shape:
            repeat, width, source_step, dest_step = 1, _ITEM_SIZE, 0, 0
        elif len(source_shape) == 2:
            repeat, width = source_shape[0], source_shape[1] * _ITEM_SIZE
            source_step, dest_step = source_strides[0], dest_strides[0]
        elif source_strides[0] == dest_strides[0] == _ITEM_SIZE:
            repeat, width, source_step, dest_step = 1, source_shape[0] * _ITEM_SIZE, 0, 0
        else:
            repeat, width = source_shape[0], _ITEM_SIZE
            source_step, dest_step = source_strides[0], dest_strides[0]
        if repeat == 0 or width == 0:
            repeat, width, source_offset, dest_offset = 0, 0, 0, 0
        for offset, step, port in ((source_offset, source_step, left_port), (dest_offset, dest_step, right_port)):
            _need(repeat == 0 or offset + (repeat - 1) * step + width <= port['byte_extent'],
                  'normalized copy extends outside the ABI array')
        yield dict(call_ordinal=ordinal, summary_id=identity, source_copy=row['id'],
                   source=dict(array_index=left_port['array_index'], offset=source_offset, step=source_step),
                   destination=dict(array_index=right_port['array_index'], offset=dest_offset, step=dest_step),
                   repeat=repeat, block_bytes=width)


def _render(plan):
    arrays = ', '.join('{' + ', '.join(map(str, row['shape'])) + '}' for row in plan['arrays'])
    moves = []
    for row in plan['ordered_moves']:
        operands = ['{' + ', '.join(str(row[side][field]) for field in ('array_index', 'offset', 'step')) + '}'
                    for side in ('source', 'destination')]
        operands.extend(str(row[key]) for key in ('repeat', 'block_bytes'))
        moves.append('{' + ', '.join(operands) + '}')
    template = _TEMPLATE.read_text()
    return (template.replace('@ARRAY_COUNT@', str(len(plan['arrays'])))
            .replace('@ARRAYS@', arrays).replace('@MOVE_COUNT@', str(len(moves)))
            .replace('@MOVES@', ',\n    '.join(moves)).replace('@PLAN_SHA256@', plan['plan_sha256'])).encode()


def prepare(ordered_update, *, array_ids):
    """Consume the already replayed update in exact numeric ABI array order."""
    _need((ordered_update['schema'], ordered_update['schema_version']) == (SOURCE_SCHEMA, 1),
          'expected source-replayed ordered native update')
    declared = {row['id']: row for row in ordered_update['array_inputs']}
    array_ids = tuple(array_ids)
    _need(0 < len(array_ids) == len(declared) == len(ordered_update['array_inputs']) <= 256
          and len(set(array_ids)) == len(array_ids) and set(array_ids) == set(declared),
          'array descriptors must use exact, distinct stateful ABI port IDs')
    roots = {}
    for index, identity in enumerate(array_ids):
        row = declared[identity]
        rows, columns = shape = _shape(row['shape'])
        strides = row['strides']
        _need(row['dtype'] in ('float64', '<f8') and row['device'] == 'cpu'
              and type(strides) in (list, tuple) and len(strides) == 2
              and (rows <= 1 or strides[0] == columns) and (columns <= 1 or strides[1] == 1),
              'original CPU binary64 root must have a contiguous element layout')
        roots[identity] = dict(id=identity, array_index=index, shape=shape, byte_extent=rows * columns * _ITEM_SIZE)
    numeric, *calls = ordered_update['ordered_calls']
    _need(numeric['kind'] == 'STATEFUL_NUMERIC_REGION'
          and calls and all(row['kind'] == 'ARRAY_COPY_CALL' for row in calls),
          'source order must be one numeric region followed by original copy calls')
    moves = [row for ordinal, call in enumerate(calls, 1) for row in _moves(call, ordinal, roots)]
    _need(0 < len(moves) <= 4096, 'bounded ordered copy assignments required')
    plan = dict(schema=SCHEMA, schema_version=1, stateful_abi_version=1,
                source_update_sha256=semantic_sha256(ordered_update), arrays=list(roots.values()), ordered_moves=moves,
                operations='ORDERED_NONOVERLAPPING_LIBC_MEMCPY', standalone_application=False)
    plan['plan_sha256'] = semantic_sha256(plan)
    return plan, _render(plan)
