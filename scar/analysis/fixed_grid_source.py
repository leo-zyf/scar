"""Derive the fixed-grid solver boundary from a qualified Python source entry."""

import ast
from hashlib import sha256
from math import isfinite
from pathlib import Path


def _unique(rows, description):
    rows = list(rows)
    if len(rows) != 1:
        raise ValueError(f'expected one {description} in the fixed-grid source')
    return rows[0]


def _constant(node, assignment):
    if isinstance(node, ast.Name):
        return _constant(assignment(node), assignment)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        if all(isinstance(item, ast.Constant) and type(item.value) in (int, float) for item in (node.left, node.right)):
            if node.right.value != 0:
                return node.left.value / node.right.value
    raise ValueError('fixed-grid option requires a static literal or literal division')


def _attribute(node, base, field):
    return (isinstance(node, ast.Attribute) and node.attr == field
            and isinstance(node.value, ast.Name) and node.value.id == base)


def _binding(node):
    if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
        return node.id
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                         ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
        return node.name
    if isinstance(node, ast.MatchMapping):
        return node.rest
    if isinstance(node, ast.arg):
        return node.arg
    if isinstance(node, ast.alias):
        return node.asname or node.name.split('.')[0]
    return None


def _position(node):
    return node.lineno, node.col_offset


def _sequence_size(node, assignment):
    if isinstance(node, ast.Name):
        return _sequence_size(assignment(node), assignment)
    if isinstance(node, (ast.Tuple, ast.List)) and all(not isinstance(item, ast.Starred) for item in node.elts):
        return len(node.elts)
    if isinstance(node, ast.ListComp) and len(node.generators) == 1:
        generator, = node.generators
        if not generator.is_async and not generator.ifs:
            return _sequence_size(generator.iter, assignment)
    raise ValueError('starred solver arguments need a finite source sequence with a known arity')


def derive_fixed_grid_entry(source, entry, *, source_bytes=None):
    """Recover solver options for an autonomous final-state boundary.

    The enclosing Python I/O stays outside this adapter. The native file ABI
    takes its own count and Tensor operands in the admitted RHS port order.
    """
    path = Path(source['source']['path'])
    raw = path.read_bytes() if source_bytes is None else source_bytes
    if (len(raw), sha256(raw).hexdigest()) != (source['source']['size_bytes'], source['source']['sha256']):
        raise ValueError('source entry differs from the admitted RHS source')
    tree = ast.parse(raw, filename=str(path))
    fn = _unique((row for row in tree.body if type(row) is ast.FunctionDef and row.name == entry), 'entry function')
    if fn.decorator_list:
        raise ValueError('fixed-grid entry does not admit decorators')
    solver = _unique((alias for row in tree.body if isinstance(row, ast.ImportFrom)
                      and row.level == 0 and row.module == 'torchdiffeq'
                      for alias in row.names if alias.name == 'odeint'), 'torchdiffeq odeint import')
    odeint = _binding(solver)
    rhs, torch = source['function'], source['torch_global']
    protected = {odeint, rhs, torch, 'sys', 'int', entry}
    if len(protected) != 6:
        raise ValueError('fixed-grid source identities must have distinct bindings')
    admitted = {solver, fn, _unique((row for row in tree.body if type(row) is ast.FunctionDef
                                   and row.name == rhs), 'RHS definition')}
    for original, local in (('torch', torch), ('sys', 'sys')):
        admitted.add(_unique((alias for row in tree.body if type(row) is ast.Import for alias in row.names
                              if alias.name == original and _binding(alias) == local), f'{original} import'))
    for node in ast.walk(tree):
        if (_binding(node) in protected | {'*'} and node not in admitted
                or isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(node.ctx, (ast.Store, ast.Del))
                and any(isinstance(row, ast.Name) and row.id in protected for row in ast.walk(node.value))):
            raise ValueError('a fixed-grid source binding is shadowed or modified')
    solve = _unique((row for row in fn.body
        if type(row) is ast.Assign and len(row.targets) == 1 and type(row.targets[0]) is ast.Name
        and type(row.value) is ast.Call and type(row.value.func) is ast.Name and row.value.func.id == odeint),
        'unconditional solver assignment')
    result, call = solve.targets[0], solve.value
    prefix = fn.body[:fn.body.index(solve)]
    if any(type(row) not in (ast.Assign, ast.Expr) for row in prefix):
        raise ValueError('the fixed-grid solver boundary needs a straight-line entry prefix')
    if len(call.args) != 3 or len(call.keywords) != 2 or {row.arg for row in call.keywords} != {'method', 'options'}:
        raise ValueError('fixed-grid source requires explicit method and options and no event callback')
    callback, state, times = call.args
    if not isinstance(state, ast.Name) or not isinstance(times, ast.Name) or not isinstance(callback, ast.Lambda):
        raise ValueError('fixed-grid boundary requires local state, time grid and an autonomous lambda')
    callback_binders = set(ast.walk(callback.args))
    nodes = list(ast.walk(fn))
    loaded = {}
    bound = {}
    for row in nodes:
        name = _binding(row)
        if name is not None and row not in callback_binders:
            bound.setdefault(name, []).append(row)
        if isinstance(row, (ast.Global, ast.Nonlocal)):
            raise ValueError('the fixed-grid entry needs local definitions for its inputs and results')
        if isinstance(row, ast.Name) and isinstance(row.ctx, ast.Load):
            loaded.setdefault(row.id, []).append(row)

    def definition(node):
        binder = _unique(bound.get(node.id, ()), f'source definition of {node.id}')
        owner = _unique((row for row in prefix if type(row) is ast.Assign and len(row.targets) == 1
                         and binder in ast.walk(row.targets[0])
                         and (row.end_lineno, row.end_col_offset) < _position(node)),
                        f'prior definition of {node.id}')
        return binder, owner.targets[0], owner.value

    def assignment(node):
        binder, target, value = definition(node)
        if binder is not target:
            raise ValueError(f'fixed-grid scalar or sequence {node.id} requires a direct local assignment')
        return value

    def sequence_assignment(node):
        if loaded[node.id] != [node]:
            raise ValueError(f'solver sequence {node.id} has uses outside its inferred binding')
        return assignment(node)

    def check_operand(node):
        if not (len(bound.get(node.id, ())) == 1 and bound[node.id][0] in ast.walk(fn.args)):
            definition(node)

    options = {row.arg: row.value for row in call.keywords}
    mapping = options['options']
    if not isinstance(mapping, ast.Dict) or len(mapping.keys) != 1 or not isinstance(mapping.keys[0], ast.Constant) \
            or mapping.keys[0].value != 'step_size':
        raise ValueError('fixed-grid source needs an exclusive step_size option')
    method = _constant(options['method'], assignment)
    step_size = _constant(mapping.values[0], assignment)
    if type(method) is not str or type(step_size) not in (int, float) or step_size <= 0 or not isfinite(step_size):
        raise ValueError('invalid fixed-grid method or step size')

    check_operand(state)
    parameters, body = callback.args, callback.body
    if (len(parameters.args) != 2 or parameters.posonlyargs or parameters.defaults or parameters.vararg
            or parameters.kwarg or parameters.kwonlyargs or not isinstance(body, ast.Call) or body.keywords
            or not isinstance(body.func, ast.Name) or body.func.id != source['function']):
        raise ValueError('fixed-grid callback must directly forward to the admitted RHS')
    time_name, state_name = (row.arg for row in parameters.args)
    expanded = body.args[-1] if body.args and isinstance(body.args[-1], ast.Starred) else None
    explicit = body.args[:-1] if expanded is not None else body.args
    if any(not isinstance(arg, ast.Name) or arg.id == time_name for arg in explicit):
        raise ValueError('fixed-grid callback has a time-dependent or non-positional RHS binding')
    state_formal = _unique((i for i, arg in enumerate(explicit) if arg.id == state_name),
                            'carried state argument')
    for arg in explicit:
        if arg.id != state_name:
            check_operand(arg)
    arity = len(explicit)
    if expanded is not None:
        name = expanded.value
        if (not isinstance(name, ast.Name) or name.id in (state_name, time_name)
                or loaded[name.id] != [name]):
            raise ValueError('starred solver arguments need one invariant local sequence')
        binder, target, values = definition(name)
        size = _sequence_size(values, sequence_assignment)
        if binder is not target:
            if (not isinstance(target, (ast.Tuple, ast.List))
                    or not any(isinstance(item, ast.Starred) and item.value is binder for item in target.elts)
                    or any(not isinstance(item, (ast.Name, ast.Starred)) for item in target.elts)):
                raise ValueError('starred solver arguments need a known assignment or destructure')
            size -= len(target.elts) - 1
        if size < 0:
            raise ValueError('starred solver destructure exceeds the source sequence arity')
        arity += size
    if arity != len(source['parameters']):
        raise ValueError('fixed-grid callback does not bind every admitted RHS argument')
    if state_formal not in source['input_order']:
        raise ValueError('the carried state is not an admitted native RHS input')

    grid = sequence_assignment(times)
    if (not isinstance(grid, ast.Call) or not _attribute(grid.func, torch, 'tensor') or len(grid.args) != 1
            or len(grid.keywords) != 1 or grid.keywords[0].arg != 'dtype'
            or not _attribute(grid.keywords[0].value, torch, 'float64')
            or not isinstance(grid.args[0], (ast.List, ast.Tuple)) or len(grid.args[0].elts) != 2):
        raise ValueError('fixed-grid source needs a two-endpoint float64 grid')
    start, end = grid.args[0].elts
    if (_constant(start, assignment) != 0 or not isinstance(end, ast.BinOp) or not isinstance(end.left, ast.Name)
            or not isinstance(end.op, (ast.Div, ast.Mult))):
        raise ValueError('grid endpoint must be the step count times a static step size')
    scale = _constant(end.right, assignment)
    if (type(scale) not in (int, float) or scale <= 0 or not isfinite(scale)
            or (1 / scale if isinstance(end.op, ast.Div) else scale) != step_size):
        raise ValueError('grid endpoint and solver step size differ')
    count = assignment(end.left)
    if (not isinstance(count, ast.Call) or not isinstance(count.func, ast.Name) or count.func.id != 'int'
            or count.keywords or len(count.args) != 1 or not isinstance(count.args[0], ast.Subscript)
            or not _attribute(count.args[0].value, 'sys', 'argv')
            or not isinstance(count.args[0].slice, ast.Constant) or type(count.args[0].slice.value) is not int
            or count.args[0].slice.value <= 0):
        raise ValueError('fixed-grid step count requires direct integer CLI input')

    consumers = [node.value for node in nodes if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load)
                 and isinstance(node.value, ast.Name) and node.value.id == result.id and isinstance(node.slice, ast.UnaryOp)
                 and isinstance(node.slice.op, ast.USub) and isinstance(node.slice.operand, ast.Constant)
                 and type(node.slice.operand.value) is int and node.slice.operand.value == 1]
    if (bound[result.id] != [result] or not consumers or set(consumers) != set(loaded.get(result.id, ()))
            or any(_position(node) < _position(call) for node in consumers)):
        raise ValueError('fixed-grid native boundary only delivers the final Tensor')
    return dict(source=source['source'], entry=entry, rhs=source['function'], rhs_code_key=source['code_key'],
                call_span=[call.lineno, call.col_offset, call.end_lineno, call.end_col_offset],
                method=method, step_size=float(step_size), state_input_index=source['input_order'].index(state_formal),
                scope='FINAL_STATE_WITH_NATIVE_FILE_ARGUMENTS')
