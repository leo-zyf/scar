"""Qualify a typed functional export for the native CUDA matrix-product ABI."""

from hashlib import sha256

from scar.compiler.aten_native_recipe import _fields, _identifier
from scar.compiler.tensor_program_graph import straight_line_graph, tensor_type
from scar.ir.program import codec
from scar.optimization.aten_export_rewrite import FUNCTIONAL_EXPORT_SCHEMAS
from scar.optimization.aten_live_values import tensor_releases


SCHEMA = 'scar.aten-export-cuda-source'
_MM = FUNCTIONAL_EXPORT_SCHEMAS['aten.mm.default']


def _matrix(shape, name):
    if (type(shape) is not list or len(shape) != 2
            or any(type(extent) is not int or not 0 < extent < 2**31 for extent in shape)):
        raise ValueError(f'CUDA export {name} requires a nonempty rank-two float32 file')
    rows, columns = shape
    if 4 * rows * columns >= 2**63:
        raise ValueError(f'CUDA export {name} exceeds the supported buffer extent')
    return list(shape)


def qualify_cuda_export(recipe, shapes=None, *, graph, program_ref, input_kinds):
    """Bind exact ATen MM operations and ordered file ports to their program IDs.

    The recipe and its concrete shapes come from the functional export adapter
    after its contiguous CPU FP32 and no-mutation checks. This boundary
    identifies row-major buffers and fresh MM results. Offline replay can
    derive all geometry from normalized recipe inputs and MM operands.
    Device placement and provider arithmetic belong to later stages.
    """
    _fields(recipe, {'schema', 'schema_version', 'inputs', 'operations', 'outputs'}, 'CUDA export recipe')
    tensor_releases(recipe)
    input_names = [item['id'] for item in recipe['inputs']]
    operation_names = [item['id'] for item in recipe['operations']]
    names = set(input_names + operation_names)
    if (input_kinds.keys() != set(input_names)
            or any(kind not in ('INPUT', 'WEIGHT') for kind in input_kinds.values())):
        raise ValueError('CUDA export file ports differ from the recipe')
    matrices = {}
    for item in recipe['inputs']:
        _fields(item, {'id', 'shape'}, 'CUDA export input')
        name = _identifier(item['id'], 'CUDA export input')
        matrices[name] = _matrix(item['shape'], name)
    operations = []
    demanded = set(recipe['outputs'])
    for node in recipe['operations']:
        _fields(node, {'id', 'schema', 'arguments'}, 'CUDA export operation')
        name = _identifier(node['id'], 'CUDA export operation')
        if node['schema'] != _MM:
            raise ValueError(f'CUDA export only admits the exact functional aten.mm overload at {name}')
        arguments = node['arguments']
        if (type(arguments) is not list or len(arguments) != 2
                or any(type(arg) is not dict or arg.keys() != {'value'} for arg in arguments)):
            raise ValueError(f'CUDA export {name} requires two ordered Tensor operands')
        left, right = (arg['value'] for arg in arguments)
        if left == right:
            raise ValueError(f'CUDA export {name} needs distinct provider operand buffers')
        rows, reduction = matrices[left]
        other_reduction, columns = matrices[right]
        if reduction != other_reduction:
            raise ValueError(f'CUDA export {name} has inconsistent matrix-product dimensions')
        matrices[name] = _matrix([rows, columns], name)
        operations.append(dict(id=name, left=left, right=right, result=name, shape=matrices[name],
                               reduction=reduction, nbytes=4 * rows * columns))
        demanded.update((left, right))
    if demanded != names:
        raise ValueError('CUDA export requires every entry value and matrix result to have a consumer')
    if shapes is not None and shapes != matrices:
        raise ValueError('CUDA export geometry differs from the functional source projection')
    canonical = straight_line_graph(recipe, {name: tensor_type(shape) for name, shape in matrices.items()})
    if canonical != graph:
        raise ValueError('CUDA export recipe differs from its original typed program graph')
    payload = (codec.dumps(graph) + '\n').encode()
    if (not isinstance(program_ref.get('path'), str) or not program_ref['path']
            or program_ref.get('sha256') != sha256(payload).hexdigest() or program_ref.get('size_bytes') != len(payload)):
        raise ValueError('CUDA export source program reference differs from its canonical bytes')

    inputs = []
    for name, value in zip(input_names, graph.regions[0].blocks[0].arguments, strict=True):
        rows, columns = matrices[name]
        inputs.append(dict(id=name, kind=input_kinds[name], shape=matrices[name], nbytes=4 * rows * columns,
                           value_id=value.wire))
    for row, operation in zip(operations, graph.operations[:-1], strict=True):
        row.update(operation=operation.id.wire, value_id=operation.results[0].wire)
    return dict(schema=SCHEMA, schema_version=1, program=dict(program_ref), inputs=inputs, operations=operations,
                outputs=list(recipe['outputs']), layout='CONTIGUOUS_ROW_MAJOR', dtype='FLOAT32',
                ownership='READ_ONLY_ENTRY_VALUES_AND_FRESH_RESULTS')
