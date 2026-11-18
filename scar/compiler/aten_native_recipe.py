"""Validate ordered declarations for the standalone native ATen tensor ABI."""

from math import prod
import re


SCHEMA = 'scar.aten-native-recipe'
SCHEMA_VERSION = 1
_ID = re.compile(r'[a-zA-Z_][a-zA-Z_0-9]{0,127}')
_MAX_I64 = 2**63 - 1
_MAX_FLOAT32_ELEMENTS = _MAX_I64 // 4
_MAX_RANK = 64


def _fields(row, fields, location):
    if type(row) is not dict or row.keys() != fields:
        raise ValueError(f'{location}: expected fields {", ".join(sorted(fields))}')


def _identifier(value, location):
    if type(value) is not str or not _ID.fullmatch(value):
        raise ValueError(f'{location}: expected an ASCII value ID')
    return value


def _shape(shape, location):
    if (type(shape) is not list or len(shape) > _MAX_RANK
            or any(type(extent) is not int or not 0 <= extent <= _MAX_I64 for extent in shape)
            or prod(max(extent, 1) for extent in shape) > _MAX_FLOAT32_ELEMENTS):
        raise ValueError(f'{location}: shape exceeds the native float32 geometry')
    return list(shape)


def _signature(schema_text, linked, torch):
    if type(schema_text) is not str or schema_text not in linked:
        raise ValueError('operation requires an exact linked ATen schema')
    try:
        declared = torch._C.parse_schema(schema_text)
        registered = torch._C._dispatch_find_schema_or_throw(declared.name, declared.overload_name).schema()
    except RuntimeError as error:
        raise ValueError('linked schema is unavailable in the installed dispatcher') from error
    if not registered.name.startswith('aten::') or str(registered) != schema_text:
        raise ValueError('linked schema differs from the installed ATen declaration')

    returns = registered.returns
    if registered.is_mutable or any(arg.alias_info is not None for arg in (*registered.arguments, *returns)):
        raise ValueError('mutation and alias contracts are outside the native tensor ABI')
    if len(returns) != 1 or returns[0].real_type.kind() != 'TensorType':
        raise ValueError('the native tensor ABI requires one Tensor result per call')
    parameters = tuple(arg.real_type.kind() for arg in registered.arguments)
    if 'TensorType' not in parameters or any(kind not in {'TensorType', 'NumberType'} for kind in parameters):
        raise ValueError('the native tensor ABI requires Tensor operands and supports only Scalar literals')

    dispatcher_id = registered.name
    if registered.overload_name:
        dispatcher_id += '.' + registered.overload_name
    if not torch._C._dispatch_has_computed_kernel_for_dispatch_key(dispatcher_id, 'CPU'):
        raise ValueError('the linked ATen overload has no registered CPU dispatch route')
    return parameters


def _arguments(arguments, parameters, available, location):
    if type(arguments) is not list or len(arguments) != len(parameters):
        raise ValueError(f'{location}: supply all operands in dispatcher order')
    result = []
    for argument, parameter in zip(arguments, parameters, strict=True):
        if parameter == 'TensorType':
            _fields(argument, {'value'}, location)
            value = _identifier(argument['value'], location)
            if value not in available:
                raise ValueError(f'{location}: Tensor operand {value} is not yet defined')
            result.append({'value': value})
        else:
            _fields(argument, {'scalar'}, location)
            scalar = argument['scalar']
            if type(scalar) is not int or not -_MAX_I64 - 1 <= scalar <= _MAX_I64:
                raise ValueError(f'{location}: Scalar literal must be a signed 64-bit integer')
            result.append({'scalar': scalar})
    return result


def normalize_recipe(document, operators, torch):
    """Return a detached recipe with the same order and validated operand routes.

    Dispatcher metadata checks the callable interface of a declared CPU entry.
    The recipe retains call order, including operations whose results are unused.
    Python wrapper semantics and reordering legality require separate evidence.
    """
    _fields(document, {'schema', 'schema_version', 'inputs', 'operations', 'outputs'}, 'recipe')
    if (document['schema'] != SCHEMA or type(document['schema_version']) is not int
            or document['schema_version'] != SCHEMA_VERSION):
        raise ValueError('unsupported native ATen recipe version')
    for key in ('inputs', 'operations', 'outputs'):
        if type(document[key]) is not list or not document[key]:
            raise ValueError(f'recipe {key}: expected a nonempty array')

    available = set()
    inputs = []
    for index, row in enumerate(document['inputs']):
        location = f'input {index}'
        _fields(row, {'id', 'shape'}, location)
        value = _identifier(row['id'], location)
        if value in available:
            raise ValueError(f'{location}: duplicate value {value}')
        inputs.append({'id': value, 'shape': _shape(row['shape'], location)})
        available.add(value)

    linked = {row['schema'] for row in operators}
    signatures = {}
    operations = []
    produced = set()
    for index, row in enumerate(document['operations']):
        location = f'operation {index}'
        _fields(row, {'id', 'schema', 'arguments'}, location)
        value = _identifier(row['id'], location)
        if value in available:
            raise ValueError(f'{location}: duplicate value {value}')
        schema_text = row['schema']
        if type(schema_text) is not str:
            raise ValueError(f'{location}: expected an exact linked ATen schema')
        if schema_text not in signatures:
            signatures[schema_text] = _signature(schema_text, linked, torch)
        arguments = _arguments(row['arguments'], signatures[schema_text], available, location)
        operations.append({'id': value, 'schema': schema_text, 'arguments': arguments})
        available.add(value)
        produced.add(value)

    outputs = [_identifier(value, 'recipe output') for value in document['outputs']]
    if any(value not in produced for value in outputs):
        raise ValueError('recipe outputs must reference native operation results')
    return dict(schema=SCHEMA, schema_version=SCHEMA_VERSION, inputs=inputs,
                operations=operations, outputs=outputs)
