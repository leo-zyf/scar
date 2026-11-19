"""Capture the shared body of a qualified CPU Tensor recurrence."""

from math import isfinite, prod
from sys import float_info

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx import Node
from torch.fx.experimental.proxy_tensor import make_fx

from scar.compiler.aten_native_recipe import _shape


@torch.library.custom_op('scar_recurrence_capture::rhs', mutates_args=())
def _rhs(value: torch.Tensor) -> torch.Tensor:
    raise RuntimeError('an opaque recurrence RHS is only available during fake capture')


@_rhs.register_fake
def _fake_rhs(value):
    return torch.empty_like(value, memory_format=torch.contiguous_format)


_RHS = torch.ops.scar_recurrence_capture.rhs.default
_ADD = torch.ops.aten.add.Tensor
_SUB = torch.ops.aten.sub.Tensor
_MUL = torch.ops.aten.mul.Tensor
_MUL_SCALAR = torch.ops.aten.mul.Scalar
_OPERATORS = {_RHS: 'RHS', _ADD: 'ADD', _SUB: 'SUB', _MUL: 'MUL', _MUL_SCALAR: 'MUL_SCALAR'}
_DTYPES = {torch.float32: 'FLOAT32', torch.float64: 'FLOAT64'}
_MAX_OPERATIONS = 1024


def _describe_tensor(node, storage_ids):
    value = node.meta.get('val')
    if (not isinstance(value, FakeTensor) or value.device.type != 'cpu' or value.layout != torch.strided
            or value.dtype not in _DTYPES or value.requires_grad or not value.is_contiguous()
            or value.is_neg() or value.is_conj() or value.storage_offset() != 0):
        raise ValueError(f'{node.name} is outside the dense CPU recurrence Tensor ABI')
    shape = _shape(list(value.shape), node.name)
    storage = value.untyped_storage()
    if storage.nbytes() != prod(shape) * value.element_size() or storage._cdata in storage_ids:
        raise ValueError(f'{node.name} does not own fresh dense storage in the captured graph')
    storage_ids.add(storage._cdata)
    return dict(id=node.name, shape=shape, dtype=_DTYPES[value.dtype])


def _value(node, definitions):
    if type(node) is not Node or node not in definitions:
        raise ValueError('recurrence Tensor operands must reference earlier graph results')
    return {'value': definitions[node]['id']}


def _scalar(value):
    if not ((type(value) is int and -(1 << 63) <= value < 1 << 63) or (type(value) is float and isfinite(value))):
        raise ValueError('recurrence scalar operands must be finite float or signed int64 literals')
    return {'scalar': value}


def _operation(node, definitions, state, storage_ids):
    kind = _OPERATORS.get(node.target)
    if node.op != 'call_function' or kind is None:
        raise ValueError(f'unsupported operation in recurrence body: {node.op} {node.target}')
    operands = node.args
    expected = 1 if kind == 'RHS' else 2
    if len(operands) != expected or node.kwargs.keys() - ({'alpha'} if kind in ('ADD', 'SUB') else set()):
        raise ValueError(f'{node.name} has unsupported recurrence operands')

    record = _describe_tensor(node, storage_ids)
    native = node.target
    arguments = [_value(operands[0], definitions)]
    if kind == 'RHS':
        if any(row['shape'] != state['shape'] or row['dtype'] != state['dtype']
               for row in (definitions[operands[0]], record)):
            raise ValueError('opaque recurrence RHS must receive and return the exact state geometry')
        native = None
    elif kind == 'MUL_SCALAR' or kind == 'MUL' and type(operands[1]) is not Node:
        kind, native = 'MUL_SCALAR', _MUL_SCALAR
        arguments.append(_scalar(operands[1]))
    else:
        arguments.append(_value(operands[1], definitions))
        if kind in ('ADD', 'SUB'):
            arguments.append(_scalar(node.kwargs.get('alpha', 1)))
    return dict(record, op=kind, arguments=arguments, schema=str(node.target._schema),
                native_schema=None if native is None else str(native._schema))


def _prune_dead_time_arithmetic(inputs, operations, output):
    live = {output}
    by_id = {row['id']: row for row in (*inputs, *operations)}
    retained, discarded = [], []
    for operation in reversed(operations):
        references = [arg['value'] for arg in operation['arguments'] if 'value' in arg]
        descriptions = [operation, *(by_id[value] for value in references)]
        if (operation['id'] not in live and operation['op'] != 'RHS'
                and all(not row['shape'] and row['dtype'] == 'FLOAT64' for row in descriptions)):
            discarded.append(operation)
            continue
        retained.append(operation)
        live.update(references)
    return retained[::-1], discarded[::-1]


def capture_aten_step(step, state_shape, step_size, *, explicit_time=False):
    """Reify one step with an opaque state-to-state RHS and fixed Tensor metadata.

    The caller qualifies the original callable and effect boundary. Fake capture
    supplies operator bindings and data edges without computing Tensor contents.
    A step accepting explicit t0 and t1 must prove its retained body does not
    read either time endpoint.
    """
    if type(state_shape) not in (list, tuple):
        raise ValueError('recurrence capture requires a fixed state shape')
    shape = _shape(list(state_shape), 'state')
    if type(step_size) not in (int, float) or not 0 < step_size <= float_info.max:
        raise ValueError('recurrence step size must be positive and finite')

    def body(state, dt):
        return step(state, dt, _rhs)

    def timed_body(state, dt, t0, t1):
        return step(state, dt, _rhs, t0, t1)

    input_count = 4 if explicit_time else 2
    with torch.no_grad(), FakeTensorMode():
        state = torch.empty(shape, dtype=torch.float32, device='cpu')
        dt = torch.empty((), dtype=torch.float64, device='cpu')
        if explicit_time:
            t0 = torch.empty((), dtype=torch.float64, device='cpu')
            t1 = torch.empty((), dtype=torch.float64, device='cpu')
            graph = make_fx(timed_body, tracing_mode='fake')(state, dt, t0, t1).graph
        else:
            graph = make_fx(body, tracing_mode='fake')(state, dt).graph

    nodes = list(graph.nodes)
    if not input_count + 2 <= len(nodes) <= input_count + _MAX_OPERATIONS + 1:
        raise ValueError('recurrence body exceeds its capture node domain')
    formals = nodes[:input_count]
    if any(node.op != 'placeholder' for node in formals) or nodes[-1].op != 'output':
        raise ValueError('a recurrence requires Tensor placeholders and one state output')
    storage_ids = set()
    inputs = [_describe_tensor(node, storage_ids) for node in formals]
    expected_inputs = [(shape, 'FLOAT32'), *[([], 'FLOAT64')] * (input_count - 1)]
    if [(row['shape'], row['dtype']) for row in inputs] != expected_inputs:
        raise ValueError('recurrence inputs differ from the declared state or time geometry')
    definitions = dict(zip(formals, inputs, strict=True))
    operations = []
    for node in nodes[input_count:-1]:
        row = _operation(node, definitions, inputs[0], storage_ids)
        definitions[node] = row
        operations.append(row)

    outputs = nodes[-1].args
    produced = set(nodes[input_count:-1])
    if len(outputs) != 1 or type(outputs[0]) is not Node or outputs[0] not in produced:
        raise ValueError('recurrence must return one produced next-state Tensor')
    result = definitions[outputs[0]]
    if (result['shape'], result['dtype']) != (shape, 'FLOAT32'):
        raise ValueError('recurrence must preserve the state Tensor geometry')
    retained, discarded = _prune_dead_time_arithmetic(inputs, operations, result['id'])
    endpoints = {row['id'] for row in inputs[2:]}
    if any(arg.get('value') in endpoints for row in retained for arg in row['arguments']):
        raise ValueError('recurrence step depends on absolute time')
    return dict(inputs=inputs[:2], operations=retained, output=result['id'], discarded_scalar_ops=discarded)
