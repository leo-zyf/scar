"""Project witnessed dense ndarray slices into stock-ufunc loop calls."""

import argparse
import ctypes
import json
from math import prod
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization import partition_copy

SCHEMA = 'scar.numpy-partition-iteration'
_ITEM = 16
_ROLES = ('target', 'matrix', 'out')
_SELECTORS = [{'kind': 'ELLIPSIS_LAST_AXIS_INDEX', 'index': index,
               'normalized_index': index} for index in range(2)]


def _require(valid, message):
    if not valid:
        raise ValueError(message)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _address(view):
    return view.__array_interface__['data'][0]


def _view(numpy, geometry, data):
    return numpy.ndarray(tuple(geometry['shape']), dtype=geometry['dtype'], buffer=data,
                         offset=geometry['offset_bytes'], strides=tuple(geometry['strides_bytes']))


def _steps(numpy, operator, roles, arrays, addresses):
    """Verify every unbuffered K-order segment before compacting affine repeats."""
    with numpy.nditer(arrays, order='K', flags=['external_loop'],
                      op_flags=[['readonly'], ['readonly'], ['readwrite', 'no_broadcast']]) as iterator:
        locations = None
        deltas = None
        count = 0
        segments = 0
        for chunk in iterator:
            offsets = tuple(_address(view) - addresses[name] for view, name in zip(chunk, roles, strict=True))
            strides = tuple(view.strides[0] for view in chunk)
            if locations is None:
                locations = offsets
                inner_strides = strides
                count = chunk[0].size
            elif deltas is None:
                deltas = tuple(right - left for left, right in zip(locations, offsets, strict=True))
            _require(chunk[0].size == count and strides == inner_strides and
                     (deltas is None or offsets == tuple(first + segments * delta for first, delta in
                                                       zip(locations, deltas, strict=True))),
                     'the admitted slice needs nonaffine or variable-length iterator segments')
            segments += 1
    _require(count > 0 and segments * count == arrays[-1].size and locations is not None,
             'the unbuffered iterator does not cover the complete output slice')
    operands = [dict(root=name, offset=offset, stride=stride)
                for name, offset, stride in zip(roles, locations, inner_strides, strict=True)]
    step = dict(op=operator, count=count, operands=operands)
    if deltas is not None:
        step['repeat'] = dict(count=segments, offsets=list(deltas))
    return step


def _geometry(sample):
    target, matrix, output = (sample[name] for name in _ROLES)
    shape = target['shape']
    dense = [_ITEM << axis for axis in range(len(shape))]
    ids = {name: sample[name]['storage_id'] for name in _ROLES}
    _require(len(shape) >= 2 and shape == [2] * len(shape) and shape == output['shape']
             and sorted(target['strides_bytes']) == dense
             and target['strides_bytes'] == output['strides_bytes']
             and matrix['shape'] == [2, 2] and matrix['strides_bytes'] == [2 * _ITEM, _ITEM]
             and output['writeable'] is True and sample['slices'] == _SELECTORS
             and len(set(ids.values())) == len(_ROLES)
             and all(row == dict(same_object=False, same_storage=False, byte_envelopes_disjoint=True)
                     for row in sample['aliases'].values()),
             'expected a disjoint dense native complex128 two-slice partition')
    sizes = {name: sample['storages'][identity]['size_bytes'] for name, identity in ids.items()}
    for name, identity in ids.items():
        view = sample[name]
        expected = prod(view['shape']) * _ITEM
        _require(view['dtype'] == '<c16' and view['offset_bytes'] == 0
                 and sizes[name] == expected and view['byte_bounds'] == [0, expected]
                 and view['address_mod_64'] == sample['storages'][identity]['address_mod_64']
                 and view['address_mod_64'] % 8 == 0 and sample['post'][name] == view,
                 'expected aligned views covering their complete retained byte spans')
    _require(sample['normal_return'] is True and sample['returned_alias'] == 'out'
             and sample['rounding_mode_raw'] == 0 and sample['read_storage_bytes_unchanged'] is True,
             'expected successful alias-preserving completion under nearest rounding')
    return ids, sizes


def _stock_program(numpy, sample, sizes, initial, expected, *, elide_copy):
    output = bytearray(sizes['out'])
    target, matrix, out = (_view(numpy, sample[key], data) for key, data in
                           zip(_ROLES, (*initial, output), strict=True))
    addresses = {name: _address(view) for name, view in zip(_ROLES, (target, matrix, out), strict=True)}
    _require(all(view.flags.aligned for view in (target, matrix, out)), 'detached arrays must be dtype-aligned')
    steps = []
    if not elide_copy:
        numpy.copyto(out, target, casting='no')
        steps.append(dict(op='COPY_BYTES', count=sizes['out'],
                          operands=[dict(root=role, offset=0, stride=1) for role in ('target', 'out')]))
    temp_size = sizes['out'] // 2
    for i in range(2):
        dest = out[..., i]
        source = target[..., i] if elide_copy else dest
        diag = matrix[i:i + 1, i:i + 1].reshape(())
        steps.append(_steps(numpy, 'MULTIPLY', ('target' if elide_copy else 'out', 'matrix', 'out'),
                            (source, diag, dest), addresses))
        numpy.multiply(source, matrix[i, i], out=dest)
        j = 1 - i
        source = target[..., j]
        coef = matrix[i:i + 1, j:j + 1].reshape(())
        # Preserve the different packed layout of NumPy's newly allocated
        # multiply result. Using the source view's strides for it is invalid.
        temp = numpy.multiply(source, matrix[i, j])
        _require(temp.flags.aligned and temp.base is None and temp.nbytes == temp_size
                 and sorted(temp.strides) == [_ITEM << axis for axis in range(temp.ndim)],
                 'unsupported stock NumPy multiplication-result layout')
        address = {**addresses, 'temp': _address(temp)}
        steps.append(_steps(numpy, 'MULTIPLY', ('target', 'matrix', 'temp'), (source, coef, temp), address))
        steps.append(_steps(numpy, 'ADD', ('out', 'temp', 'out'), (dest, temp, dest), address))
        numpy.add(dest, temp, out=dest)
    _require(output == expected, 'detached stock NumPy output differs from witnessed physical bytes')
    return dict(buffers={**sizes, 'temp': temp_size}, steps=steps)


def _row(numpy, sample):
    ids, sizes = _geometry(sample)
    before, after = sample['storage_before'], sample['storage_after']
    initial = tuple(read_artifact(before[ids[name]], json_document=False) for name in _ROLES[:2])
    _require(all(len(data) == sizes[name] and data == read_artifact(after[ids[name]], json_document=False)
                 for name, data in zip(_ROLES[:2], initial, strict=True))
             and read_artifact(sample['matrix_packed'], json_document=False) == initial[1],
             'original source changed an input or the packed coefficients differ')
    expected = read_artifact(after[ids['out']], json_document=False)
    _require(len(expected) == sizes['out'], 'incomplete output bytes')
    # Side-effectful exceptional FP policies and warnings are outside the
    # detached success case. Reject a replay which actually raises any status.
    with numpy.errstate(all='raise'):
        original = _stock_program(numpy, sample, sizes, initial, expected, elide_copy=False)
        candidate = _stock_program(numpy, sample, sizes, initial, expected, elide_copy=True)
    return dict(id=sample['id'], case=sample['case'], program=original, copy_elided_program=candidate,
                inputs={name: before[ids[name]] for name in _ROLES[:2]}, expected_out=after[ids['out']])


def project(witness_path, partition_plan_path):
    """Replay exact source and array evidence without importing the target program."""
    import numpy
    from scar.trace import numpy_partition_witness

    witness_ref = _ref(witness_path, numpy_partition_witness.SCHEMA)
    witness = numpy_partition_witness.check(witness_ref['uri'])
    plan_ref = _ref(partition_plan_path, partition_copy.SCHEMA + '-index')
    partition_copy.check(plan_ref['uri'])
    plan = read_artifact(plan_ref)
    focus = witness['focus']
    signatures = tuple(focus['parameters'][role] for role in ('target', 'matrix', 'slices', 'out'))
    matching = [row for row in plan['candidates']
                if (row['source']['path'], row['source']['source_sha256'], row['source']['code_id'],
                    row['source']['qualname'], tuple(row['source']['inputs'])) ==
                   (focus['source']['uri'], focus['source']['sha256'], focus['code_id'], focus['qualname'], signatures)]
    _require(len(matching) == 1, 'runtime callable is not the source-proven partition operation')
    rounding = ctypes.CDLL(None).fegetround
    rounding.argtypes = ()
    rounding.restype = ctypes.c_int
    observed = witness['numpy']
    provider = numpy_partition_witness._provider(numpy)
    _require(rounding() == 0 and all(provider[key] == observed[key] for key in provider if key != 'errstate'),
             'offline NumPy or floating-point environment differs from the captured runtime')
    samples, frontiers = [], []
    for sample in witness['samples']:
        if len(sample['target']['shape']) < 2:
            frontiers.append(dict(id=sample['id'], case=sample['case'], reason='SCALAR_DISPATCH_NOT_ADMITTED'))
        else:
            samples.append(_row(numpy, sample))
    _require(samples, 'no witnessed array-dispatch partition samples')
    required = [dict(op=op, strides=list(strides)) for op, strides in sorted({
        (step['op'], tuple(operand['stride'] for operand in step['operands']))
        for sample in samples for program in ('program', 'copy_elided_program')
        for step in sample[program]['steps'] if step['op'] != 'COPY_BYTES'
    })]
    return dict(schema=SCHEMA, schema_version=1, witness=witness_ref, partition_plan=plan_ref,
                source_graphs=matching[0]['graphs'], provider=observed['extension'],
                loop_signatures=required, samples=samples, frontiers=frontiers,
                scope=dict(private_output=True, complete_output_from_all_slices=True,
                           source_multiply_then_add=True, cross_operation_fusion=False,
                           same_provider_bitwise_replay_of_witnessed_values=True,
                           iterator='UNBUFFERED_K_ORDER_EXTERNAL_LOOP',
                           normal_completion_only=True, native_loop_identity_proven=False,
                           whole_program_effect_equivalence=False, deployment_authorized=False))


def check(path):
    stored = read_artifact(_ref(path, SCHEMA))
    expected = project(stored['witness']['uri'], stored['partition_plan']['uri'])
    _require(stored == expected, 'recorded NumPy partition iteration differs from replay')
    return stored


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('witness', 'partition-plan', 'out', 'check'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args(argv)
    if args.check and not any((args.witness, args.partition_plan, args.out)):
        result = check(args.check)
    elif not args.check and all((args.witness, args.partition_plan, args.out)):
        result = project(args.witness, args.partition_plan)
        with args.out.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
    else:
        parser.error('provide --check RESULT or --witness WITNESS --partition-plan PLAN --out RESULT')
    print(json.dumps(dict(samples=len(result['samples']), loop_signatures=result['loop_signatures']), sort_keys=True))


if __name__ == '__main__':
    main()
