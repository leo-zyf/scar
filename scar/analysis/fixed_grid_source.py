"""Derive the fixed-grid solver boundary from a qualified Python source entry."""

import ast
from hashlib import sha256
from pathlib import Path


def _unique(rows, description):
    rows = list(rows)
    if len(rows) != 1:
        raise ValueError(f'expected one {description} in the fixed-grid source')
    return rows[0]


def _constant(node, assignment):
    if isinstance(node, ast.Name):
        node = assignment(node.id)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        if all(isinstance(item, ast.Constant) and type(item.value) in (int, float) for item in (node.left, node.right)):
            return node.left.value / node.right.value
    raise ValueError('fixed-grid option requires a static literal or literal division')


def _attribute(node, base, field):
    return (isinstance(node, ast.Attribute) and node.attr == field
            and isinstance(node.value, ast.Name) and node.value.id == base)


def derive_fixed_grid_entry(source, entry):
    """Recover solver options for an autonomous final-state boundary.

    The enclosing Python I/O stays outside this adapter. The native file ABI
    takes its own count and Tensor operands in the admitted RHS port order.
    """
    path = Path(source['source']['path'])
    raw = path.read_bytes()
    if (len(raw), sha256(raw).hexdigest()) != (source['source']['size_bytes'], source['source']['sha256']):
        raise ValueError('source entry differs from the admitted RHS source')
    tree = ast.parse(raw, filename=str(path))
    fn = _unique((row for row in tree.body if type(row) is ast.FunctionDef and row.name == entry), 'entry function')
    if fn.decorator_list:
        raise ValueError('fixed-grid entry does not admit decorators')
    seen = [row for row in tree.body if isinstance(row, ast.ImportFrom) and row.level == 0 and row.module == 'torchdiffeq']
    odeint = _unique((alias.asname or alias.name for stmt in seen for alias in stmt.names
                      if alias.name == 'odeint'), 'torchdiffeq odeint import')
    protected = {odeint, source['function'], source['torch_global'], 'sys', 'int'}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                local = alias.asname or alias.name.split('.')[0]
                allowed = ((node in seen and alias.name == 'odeint' and local == odeint)
                           or (isinstance(node, ast.Import) and (alias.name, local) in
                               {('sys', 'sys'), ('torch', source['torch_global'])}))
                if (local in protected and not allowed) or alias.name == '*':
                    raise ValueError('an import changes a fixed-grid source binding')
    if (any(isinstance(row, ast.Name) and isinstance(row.ctx, (ast.Store, ast.Del)) and row.id in protected
            for row in ast.walk(tree)) or any(row.arg in protected for row in ast.walk(fn) if isinstance(row, ast.arg))):
        raise ValueError('solver or RHS is rebound in its source module')
    result, call = _unique(((row.targets[0].id, row.value) for row in fn.body
        if type(row) is ast.Assign and len(row.targets) == 1 and type(row.targets[0]) is ast.Name
        and type(row.value) is ast.Call and type(row.value.func) is ast.Name and row.value.func.id == odeint),
        'unconditional solver assignment')

    def assignment(name):
        _unique((row for row in ast.walk(fn) if isinstance(row, ast.Name) and row.id == name
                 and isinstance(row.ctx, (ast.Store, ast.Del))), f'source definition of {name}')
        return _unique((row.value for row in fn.body if type(row) is ast.Assign and row.lineno < call.lineno
                        and len(row.targets) == 1 and type(row.targets[0]) is ast.Name and row.targets[0].id == name),
                       f'prior definition of {name}')

    options = {row.arg: row.value for row in call.keywords}
    if len(call.args) != 3 or len(call.keywords) != 2 or set(options) != {'method', 'options'}:
        raise ValueError('fixed-grid source requires explicit method and options and no event callback')
    mapping = options['options']
    if not isinstance(mapping, ast.Dict) or len(mapping.keys) != 1 or not isinstance(mapping.keys[0], ast.Constant) \
            or mapping.keys[0].value != 'step_size':
        raise ValueError('fixed-grid source needs an exclusive step_size option')
    method = _constant(options['method'], assignment)
    step_size = _constant(mapping.values[0], assignment)
    if type(method) is not str or type(step_size) not in (int, float) or step_size <= 0:
        raise ValueError('invalid fixed-grid method or step size')

    callback, state, times = call.args
    if not isinstance(state, ast.Name) or not isinstance(times, ast.Name) or not isinstance(callback, ast.Lambda):
        raise ValueError('fixed-grid boundary requires local state, time grid and an autonomous lambda')
    parameters, body = callback.args, callback.body
    if (len(parameters.args) != 2 or parameters.posonlyargs or parameters.defaults or parameters.vararg
            or parameters.kwarg or parameters.kwonlyargs or not isinstance(body, ast.Call) or body.keywords
            or not isinstance(body.func, ast.Name) or body.func.id != source['function']):
        raise ValueError('fixed-grid callback must directly forward to the admitted RHS')
    time_name, state_name = (row.arg for row in parameters.args)
    state_formal = _unique((i for i, arg in enumerate(body.args) if isinstance(arg, ast.Name) and arg.id == state_name),
                            'carried state argument')
    stars = [index for index, arg in enumerate(body.args) if isinstance(arg, ast.Starred)]
    if (any(arg.id == time_name for arg in ast.walk(body) if isinstance(arg, ast.Name) and arg is not body.func)
            or any(not isinstance(arg, (ast.Name, ast.Starred)) for arg in body.args)
            or any(not isinstance(body.args[i].value, ast.Name) for i in stars)
            or stars not in ([], [len(body.args) - 1]) or (stars and state_formal >= stars[0])
            or len(body.args) > len(source['parameters'])
            or not stars and len(body.args) != len(source['parameters'])):
        raise ValueError('fixed-grid callback has a time-dependent or non-positional RHS binding')
    if state_formal not in source['input_order']:
        raise ValueError('the carried state is not an admitted native RHS input')

    grid = assignment(times.id)
    if any(node is not times for node in ast.walk(fn) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
           and node.id == times.id):
        raise ValueError('time grid must reach its solver without another use')
    torch = source['torch_global']
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
    if type(scale) not in (int, float) or (1 / scale if isinstance(end.op, ast.Div) else scale) != step_size:
        raise ValueError('grid endpoint and solver step size differ')
    count = assignment(end.left.id)
    if (not isinstance(count, ast.Call) or not isinstance(count.func, ast.Name) or count.func.id != 'int'
            or count.keywords or len(count.args) != 1 or not isinstance(count.args[0], ast.Subscript)
            or not _attribute(count.args[0].value, 'sys', 'argv')):
        raise ValueError('fixed-grid step count requires direct integer CLI input')

    consumers = [node for node in ast.walk(fn) if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                 and node.value.id == result and isinstance(node.slice, ast.UnaryOp)
                 and isinstance(node.slice.op, ast.USub) and isinstance(node.slice.operand, ast.Constant)
                 and type(node.slice.operand.value) is int and node.slice.operand.value == 1]
    reads = [node for node in ast.walk(fn) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
             and node.id == result]
    if not consumers or {id(node.value) for node in consumers} != {id(node) for node in reads}:
        raise ValueError('fixed-grid native boundary only delivers the final Tensor')
    return dict(source=source['source'], entry=entry, rhs=source['function'], rhs_code_key=source['code_key'],
                call_span=[call.lineno, call.col_offset, call.end_lineno, call.end_col_offset],
                method=method, step_size=float(step_size), state_input_index=source['input_order'].index(state_formal),
                scope='FINAL_STATE_WITH_NATIVE_FILE_ARGUMENTS')
