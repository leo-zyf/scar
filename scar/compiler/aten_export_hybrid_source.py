"""Recover device-eligible regions and value crossings from a typed ATen export."""

from collections import defaultdict
from hashlib import sha256

from scar.compiler.aten_export_cuda_source import _MM, _matrix
from scar.compiler.aten_native_recipe import SCHEMA as RECIPE_SCHEMA, SCHEMA_VERSION, _arguments, _fields, _identifier
from scar.compiler.tensor_planning_inputs import straight_line_graph, tensor_type
from scar.ir.program import codec
from scar.optimization.aten_export_rewrite import FUNCTIONAL_EXPORT_SCHEMAS


SCHEMA = 'scar.aten-export-hybrid-source'
_RELU = FUNCTIONAL_EXPORT_SCHEMAS['aten.relu.default']
_ADD = FUNCTIONAL_EXPORT_SCHEMAS['aten.add.Tensor']
CPU_SCHEMAS = frozenset((_RELU, _ADD))
_SIGNATURES = {
    _MM: ('TensorType', 'TensorType'),
    _RELU: ('TensorType',),
    _ADD: ('TensorType', 'TensorType', 'NumberType'),
}


def _output_geometry(schema, tensors, matrices, name):
    left = tensors[0]
    if schema == _RELU:
        return matrices[left], {}
    right = tensors[1]
    first, second = matrices[left], matrices[right]
    if schema == _MM:
        if left == right:
            raise ValueError(f'CUDA operation {name} needs distinct provider operand buffers')
        rows, reduction = first
        other_reduction, columns = second
        if reduction != other_reduction:
            raise ValueError(f'CUDA operation {name} has inconsistent matrix-product dimensions')
        return [rows, columns], dict(left=left, right=right, reduction=reduction)
    if any(a != b and min(a, b) != 1 for a, b in zip(first, second, strict=True)):
        raise ValueError(f'CPU operation {name} has incompatible rank-two broadcast dimensions')
    return [max(a, b) for a, b in zip(first, second, strict=True)], {}


def _cuda_regions(operations, tensor_operands, consumers, outputs):
    # Connected components follow direct GPU data edges. A component can
    # also have a path through the host, so its operations keep individual
    # dependencies and the physical planner must not collapse it to a task.
    parent = {row['id']: row['id'] for row in operations if row['placement'] == 'CUDA'}

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for name in parent:
        for operand in tensor_operands[name]:
            if operand in parent:
                parent[find(name)] = find(operand)
    groups = defaultdict(list)
    for name in parent:
        groups[find(name)].append(name)
    regions = []
    for group in groups.values():
        members = set(group)
        inputs = dict.fromkeys(value for name in group for value in tensor_operands[name] if value not in members)
        returns = [name for name in group if name in outputs or any(user not in members for user in consumers[name])]
        regions.append(dict(id=f'cuda/{group[0]}', operations=group, input_values=list(inputs), output_values=returns))
    return regions


def _crossings(inputs, operations, outputs):
    definitions = {port['id']: port for port in (*inputs, *operations)}
    crossings, consumers, tensor_operands = {}, defaultdict(set), {}

    def crossing(value, destination):
        producer = definitions[value]
        source = producer.get('placement', 'CPU')
        if source == destination:
            return None
        key = (value, destination)
        return crossings.setdefault(key, dict(value=value, value_id=producer['value_id'],
            nbytes=producer['nbytes'], producer=value if 'placement' in producer else None, source=source,
            destination=destination, consumers=[], file_output_ordinals=[]))

    for row in operations:
        name = row['id']
        tensors = []
        for index, argument in enumerate(row['arguments']):
            if 'value' in argument:
                value = argument['value']
                tensors.append(value)
                consumers[value].add(name)
                boundary = crossing(value, row['placement'])
                if boundary is not None:
                    boundary['consumers'].append(dict(operation=name, operand=index))
        tensor_operands[name] = tensors
    for ordinal, value in enumerate(outputs):
        boundary = crossing(value, 'CPU')
        if boundary is not None:
            boundary['file_output_ordinals'].append(ordinal)
    return _cuda_regions(operations, tensor_operands, consumers, set(outputs)), list(crossings.values())


def qualify_hybrid_export(recipe, shapes=None, *, graph, program_ref, input_kinds, allow_cuda_only=False):
    """Replay exact contiguous rank-two FP32 MM, add, and ReLU geometry."""
    _fields(recipe, {'schema', 'schema_version', 'inputs', 'operations', 'outputs'}, 'hybrid export recipe')
    if (recipe['schema'] != RECIPE_SCHEMA or type(recipe['schema_version']) is not int
            or recipe['schema_version'] != SCHEMA_VERSION):
        raise ValueError('hybrid export requires a normalized native ATen recipe')
    for field in ('inputs', 'operations', 'outputs'):
        if type(recipe[field]) is not list or not recipe[field]:
            raise ValueError(f'hybrid export requires a nonempty {field} array')

    matrices = {}
    for item in recipe['inputs']:
        _fields(item, {'id', 'shape'}, 'hybrid export input')
        name = _identifier(item['id'], 'hybrid export input')
        if name in matrices:
            raise ValueError('hybrid export input IDs must be distinct')
        matrices[name] = _matrix(item['shape'], name)
    if (input_kinds.keys() != matrices.keys()
            or any(kind not in ('INPUT', 'WEIGHT') for kind in input_kinds.values())):
        raise ValueError('hybrid export file ports differ from the recipe')

    outputs = [_identifier(name, 'hybrid export result') for name in recipe['outputs']]
    operations, demanded = [], set(outputs)
    for node in recipe['operations']:
        _fields(node, {'id', 'schema', 'arguments'}, 'hybrid export operation')
        name = _identifier(node['id'], 'hybrid export operation')
        if name in matrices:
            raise ValueError(f'hybrid export result ID {name} is already defined')
        schema = node['schema']
        if type(schema) is not str or schema not in _SIGNATURES:
            raise ValueError(f'hybrid export has no qualified provider for operation {name}')
        arguments = _arguments(node['arguments'], _SIGNATURES[schema], matrices, name)
        tensors = [arg['value'] for arg in arguments if 'value' in arg]
        shape, geometry = _output_geometry(schema, tensors, matrices, name)
        rows, columns = matrices[name] = _matrix(shape, name)
        operations.append(dict(id=name, schema=schema, arguments=arguments, result=name, shape=matrices[name],
            nbytes=4 * rows * columns, placement='CUDA' if schema == _MM else 'CPU', **geometry))
        demanded.update(tensors)
    placements = {row['placement'] for row in operations}
    if 'CUDA' not in placements or not allow_cuda_only and 'CPU' not in placements:
        raise ValueError('hybrid export requires both CPU and CUDA operations')
    if set(outputs) - {row['id'] for row in operations} or demanded != matrices.keys():
        raise ValueError('hybrid export requires used entry values, used results and produced output values')
    if shapes is not None and shapes != matrices:
        raise ValueError('hybrid export geometry differs from the functional source projection')

    canonical = straight_line_graph(recipe, {name: tensor_type(shape) for name, shape in matrices.items()})
    if canonical != graph:
        raise ValueError('hybrid export recipe differs from its original typed program graph')
    payload = (codec.dumps(graph) + '\n').encode()
    if (not isinstance(program_ref.get('path'), str) or not program_ref['path']
            or program_ref.get('sha256') != sha256(payload).hexdigest() or program_ref.get('size_bytes') != len(payload)):
        raise ValueError('hybrid export source program reference differs from its canonical bytes')

    inputs = []
    for item, value in zip(recipe['inputs'], graph.regions[0].blocks[0].arguments, strict=True):
        name = item['id']
        rows, columns = matrices[name]
        inputs.append(dict(id=name, kind=input_kinds[name], shape=matrices[name], nbytes=4 * rows * columns,
                           value_id=value.wire))
    for row, operation in zip(operations, graph.operations[:-1], strict=True):
        row.update(operation=operation.id.wire, value_id=operation.results[0].wire)
    regions, crossings = _crossings(inputs, operations, outputs)
    return dict(schema=SCHEMA, schema_version=1, program=dict(program_ref), inputs=inputs, operations=operations,
                outputs=outputs, layout='CONTIGUOUS_ROW_MAJOR', dtype='FLOAT32',
                ownership='READ_ONLY_ENTRY_VALUES_AND_FRESH_RESULTS', cuda_regions=regions, boundaries=crossings)
