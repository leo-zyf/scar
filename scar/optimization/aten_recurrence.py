"""Validate and bound one shared native ATen recurrence body."""

from math import isfinite, prod

from scar.compiler.aten_native_recipe import SCHEMA as RHS_SCHEMA, SCHEMA_VERSION as RHS_VERSION
from scar.compiler.aten_native_recipe import _fields, _identifier, _shape
from scar.optimization.aten_live_values import plan_step_lifetime_order
from scar.optimization.aten_live_values import _bounded_sum, _bytes, tensor_releases


SCHEMA = 'scar.aten-recurrence-plan'
_STEP_FIELDS = {'inputs', 'operations', 'output', 'discarded_scalar_ops'}
_INPUT_FIELDS = {'id', 'shape', 'dtype'}
_OP_FIELDS = {*_INPUT_FIELDS, 'op', 'arguments', 'schema', 'native_schema'}
_FLOAT64_SCALAR = ((), 'FLOAT64')
_RHS_MARKER = 'scar_recurrence_capture::rhs(Tensor value) -> Tensor'
_ADD = 'aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor'
_SUB = 'aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor'
_MUL = 'aten::mul.Tensor(Tensor self, Tensor other) -> Tensor'
_MUL_SCALAR = 'aten::mul.Scalar(Tensor self, Scalar other) -> Tensor'
_SIGNATURES = {
    'ADD': ((_ADD,), _ADD, ('value', 'value', 'scalar')),
    'SUB': ((_SUB,), _SUB, ('value', 'value', 'scalar')),
    'MUL': ((_MUL,), _MUL, ('value', 'value')),
    'MUL_SCALAR': ((_MUL, _MUL_SCALAR), _MUL_SCALAR, ('value', 'scalar')),
    'RHS': ((_RHS_MARKER,), None, ('value',)),
}


def _scalar(value):
    return ((type(value) is int and -(1 << 63) <= value < 1 << 63)
            or (type(value) is float and isfinite(value)))


def _result_geometry(kind, operands, state, location):
    if kind == 'RHS':
        if operands != [state]:
            raise ValueError(f'{location}: RHS must take the carried state geometry')
        return state
    if len(operands) == 1:
        return operands[0]
    if state in operands and state[0]:
        return state
    # Float64 0-d tensors only promote a 0-d Float32 state. A dimensional
    # Float32 operand takes precedence over a same-kind scalar tensor.
    return _FLOAT64_SCALAR if _FLOAT64_SCALAR in operands else state


def _operations(rows, values, state, *, discarded=False):
    normalized = []
    for index, row in enumerate(rows):
        location = f'{"discarded scalar" if discarded else "step operation"} {index}'
        _fields(row, _OP_FIELDS, location)
        name = _identifier(row['id'], location)
        kind = row['op']
        if name in values or type(kind) is not str or kind not in _SIGNATURES:
            raise ValueError(f'{location}: duplicate value or unsupported operation')
        original, native, kinds = _SIGNATURES[kind]
        if row['schema'] not in original or row['native_schema'] != native:
            raise ValueError(f'{location}: operation schemas differ from the native recurrence contract')
        arguments = row['arguments']
        if type(arguments) is not list or len(arguments) != len(kinds):
            raise ValueError(f'{location}: supply all arguments in provider order')
        operands, copied = [], []
        for argument, key in zip(arguments, kinds, strict=True):
            _fields(argument, {key}, location)
            value = argument[key]
            if key == 'value':
                if _identifier(value, location) not in values and not discarded:
                    raise ValueError(f'{location}: every operand must follow its definition')
                # Removed time inputs exist only in independently replayed
                # capture evidence. Their references never enter lowering.
                operands.append(values.get(value, _FLOAT64_SCALAR))
            elif not _scalar(value):
                raise ValueError(f'{location}: finite double or signed 64-bit scalar required')
            copied.append({key: value})
        geometry = tuple(_shape(row['shape'], location)), row['dtype']
        if geometry != _result_geometry(kind, operands, state, location):
            raise ValueError(f'{location}: declared geometry differs from Tensor/scalar promotion')
        if discarded and (kind == 'RHS' or geometry != _FLOAT64_SCALAR
                          or any(operand != _FLOAT64_SCALAR for operand in operands)):
            raise ValueError(f'{location}: discarding state-dependent or opaque work is unsupported')
        values[name] = geometry
        normalized.append(dict(id=name, op=kind, arguments=copied, shape=list(geometry[0]), dtype=geometry[1],
                               schema=row['schema'], native_schema=native))
    return normalized


def _step(step_graph):
    _fields(step_graph, _STEP_FIELDS, 'recurrence step')
    inputs, operations, discarded = (step_graph[key] for key in ('inputs', 'operations', 'discarded_scalar_ops'))
    if (type(inputs) is not list or len(inputs) != 2 or type(operations) is not list or not operations
            or type(discarded) is not list):
        raise ValueError('recurrence requires state and step-size inputs, one shared operation body and a discard list')
    values, normalized = {}, []
    for row, dtype in zip(inputs, ('FLOAT32', 'FLOAT64'), strict=True):
        _fields(row, _INPUT_FIELDS, 'recurrence input')
        name = _identifier(row['id'], 'recurrence input')
        shape = _shape(row['shape'], 'recurrence input')
        if name in values or row['dtype'] != dtype or dtype == 'FLOAT64' and shape:
            raise ValueError('recurrence inputs must be one Float32 state and a distinct Float64 scalar step size')
        values[name] = tuple(shape), dtype
        normalized.append(dict(id=name, shape=shape, dtype=dtype))
    state = values[normalized[0]['id']]
    operations = _operations(operations, values, state)
    output = _identifier(step_graph['output'], 'recurrence output')
    if values.get(output) != state or output in (row['id'] for row in normalized):
        raise ValueError('recurrence output must be a freshly produced state with matching geometry')
    discarded = _operations(discarded, values, state, discarded=True)
    return normalized, operations, output, discarded


def _call_storage(rhs_recipe, rhs_storage, state_input_index, state_shape):
    tensor_releases(rhs_recipe)
    inputs = rhs_recipe['inputs']
    if (len(rhs_recipe['outputs']) != 1 or type(state_input_index) is not int
            or not 0 <= state_input_index < len(inputs)):
        raise ValueError('recurrence requires a valid RHS state port and exactly one RHS output')
    shapes = [_shape(row['shape'], 'RHS input') for row in inputs]
    if shapes[state_input_index] != state_shape:
        raise ValueError('RHS state port geometry differs from the recurrence state')
    input_bytes = [4 * prod(shape) for shape in shapes]
    persistent = _bounded_sum(input_bytes)
    if (type(rhs_storage) is not dict or not _bytes(rhs_storage.get('persistent_input_bytes'))
            or rhs_storage['persistent_input_bytes'] != persistent or not _bytes(rhs_storage.get('peak_tensor_storage_bytes'))
            or rhs_storage['peak_tensor_storage_bytes'] < persistent + input_bytes[state_input_index]):
        raise ValueError('RHS storage evidence must include the input payload and fresh state-shaped output')
    # The existing file ABI uses one dummy native float for an empty input.
    backing = _bounded_sum(max(4, count) for count in input_bytes)
    return backing, rhs_storage['peak_tensor_storage_bytes'] - persistent


def _storage(operations, release_after, raw_backing, carry, rhs_temporary):
    base = _bounded_sum((raw_backing, carry, 8))
    sizes = [prod(row['shape']) * (4 if row['dtype'] == 'FLOAT32' else 8) for row in operations]
    live = base
    entries = []
    for i, (op, releases) in enumerate(zip(operations, release_after, strict=True)):
        # All borrowed RHS inputs already live in the outer graph. The RHS
        # envelope includes the produced tensor, so do not add it again.
        allocation = rhs_temporary if op['op'] == 'RHS' else sizes[i]
        entries.append(_bounded_sum((live, allocation)))
        live = _bounded_sum((live, sizes[i])) - sum(sizes[index] for index in releases)
    return dict(scope='GRAPH_HELD_TENSOR_BACKING_STORAGE', raw_file_backing_bytes=raw_backing,
                previous_step_state_bytes=carry, fixed_step_size_bytes=8,
                rhs_non_input_peak_bytes=rhs_temporary, operation_peak_bytes=entries,
                peak_tensor_storage_bytes=max(entries),
                excluded_bytes=['PROVIDER_INTERNAL_WORKSPACE', 'ALLOCATOR_RESERVES', 'OTHER_PROCESS_MEMORY'])


def plan_recurrence(step_graph, rhs_recipe, rhs_storage, *, state_input_index, step_size, tensor_budget_bytes=None,
                    optimize_lifetime=False):
    """Replay a fixed-step body and bound its storage over any admitted step count.

    The caller separately qualifies the RHS provider, its output geometry, its
    effects and the source binding. The loop carries an owned state between
    invocations. Its first state also shares the retained raw input backing, so
    accounting for a distinct carry throughout is conservative.
    """
    if not _scalar(step_size) or step_size <= 0:
        raise ValueError('recurrence requires a finite positive step size')
    if tensor_budget_bytes is not None and (not _bytes(tensor_budget_bytes) or tensor_budget_bytes == 0):
        raise ValueError('recurrence requires a positive bounded Tensor storage budget')
    inputs, operations, output, discarded = _step(step_graph)
    state = inputs[0]['id']
    shape = inputs[0]['shape']
    backing, rhs_temporary = _call_storage(rhs_recipe, rhs_storage, state_input_index, shape)
    carry = 4 * prod(shape)
    lifetime = None
    if optimize_lifetime:
        lifetime = plan_step_lifetime_order(operations, output, raw_backing=backing, carry=carry,
                                            rhs_temporary=rhs_temporary)
        by_name = {row['id']: row for row in operations}
        operations = [by_name[name] for name in lifetime['order']]
    rhs_bindings = []
    for op in operations:
        if op['op'] == 'RHS':
            argument = op['arguments'][0]
            args = [dict(argument) if i == state_input_index else {'entry_input': i}
                    for i in range(len(rhs_recipe['inputs']))]
            rhs_bindings.append(dict(operation=op['id'], arguments=args))
    if not rhs_bindings:
        raise ValueError('recurrence step must invoke its qualified RHS at least once')
    release_after = tensor_releases(dict(schema=RHS_SCHEMA, schema_version=RHS_VERSION, inputs=inputs,
                                         operations=operations, outputs=[output]))
    storage = _storage(operations, release_after, backing, carry, rhs_temporary)
    peak = storage['peak_tensor_storage_bytes']
    if lifetime is not None and lifetime['peak_bytes'] != peak:
        raise ValueError('Tensor lifetime order differs from its mechanically replayed backing peak')
    if tensor_budget_bytes is not None and peak > tensor_budget_bytes:
        raise ValueError(f'recurrence Tensor storage peak {peak} exceeds budget {tensor_budget_bytes}')
    storage['budget_bytes'] = tensor_budget_bytes
    fixed_parameters = [i for i in range(len(rhs_recipe['inputs'])) if i != state_input_index]
    plan = dict(schema=SCHEMA, schema_version=1, inputs=inputs, operations=operations, output=output,
                discarded_scalar_ops=discarded, rhs_bindings=rhs_bindings, release_after=release_after,
                state_input_index=state_input_index, step_size=float(step_size), peak_tensor_storage_bytes=peak,
                storage=storage, control=dict(kind='FIXED_COUNT_LOOP', body_count=1,
                    carried_state=dict(body_input=state, body_output=output, distance=1,
                                       initial_entry_input=state_input_index),
                    invariant_entry_inputs=fixed_parameters, invariant_step_size_input=inputs[1]['id']),
                timing_basis='UNMEASURED_RECURRENCE')
    if lifetime is not None:
        plan['step_lifetime_order'] = lifetime
    return plan
