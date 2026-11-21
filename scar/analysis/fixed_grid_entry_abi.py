"""Recover a whole fixed-grid source entry with a closed file/publication ABI.

The solver and numerical provider are already qualified by the fixed-grid and
RHS contracts. This layer closes the surrounding source on the supported
normal-completion domain without importing or executing the entry.
"""

import ast
from dataclasses import dataclass
from math import isfinite, prod
from pathlib import Path
import re
from types import CodeType

from scar.analysis.fixed_grid_source import _attribute, _binding, _unique, derive_fixed_grid_entry
from scar.trace.code_snapshot import code_key


SCHEMA = 'scar.fixed-grid-file-entry'
_FILENAME = re.compile(r'[A-Za-z0-9_.-]+\Z')


@dataclass(frozen=True)
class _Argument:
    index: int
    kind: str
    span: tuple[int, ...]


@dataclass(frozen=True)
class _Load:
    root: _Argument
    filename: str
    shape: tuple[int, int]
    ordinal: int
    span: tuple[int, ...]


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _span(node):
    return (node.lineno, node.col_offset, node.end_lineno, node.end_col_offset)


def _name(node, name):
    return isinstance(node, ast.Name) and node.id == name


def _method_call(node, method, count):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == method and not node.keywords and len(node.args) == count)


def _literal(node, values):
    if isinstance(node, ast.Name) and node.id in values:
        value = values[node.id]
        if type(value) in (int, float, str, bool, tuple, type(None)):
            return value
    elif isinstance(node, ast.Constant) and type(node.value) in (int, float, str, bool, type(None)):
        return node.value
    elif isinstance(node, (ast.Tuple, ast.List)):
        return tuple(_literal(item, values) for item in node.elts)
    elif isinstance(node, ast.BinOp):
        left, right = _literal(node.left, values), _literal(node.right, values)
        if isinstance(node.op, ast.Add) and type(left) is type(right) and type(left) in (int, str):
            return left + right
        if type(left) in (int, float) and type(right) in (int, float):
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div) and right:
                return left / right
    raise ValueError('file entry requires a statically known literal, extent or sequence')


def _argument(node, functions):
    if (not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id not in functions
            or node.keywords or len(node.args) != 1):
        return None
    subscript, = node.args
    _need(isinstance(subscript, ast.Subscript) and _attribute(subscript.value, 'sys', 'argv')
          and isinstance(subscript.slice, ast.Constant) and type(subscript.slice.value) is int
          and subscript.slice.value > 0, 'file entry arguments must come from positive literal sys.argv indices')
    return _Argument(subscript.slice.value, functions[node.func.id], _span(node))


def _load(node, values, torch, ordinal):
    _need(_method_call(node, 'view', 2), 'file entry Tensor input needs a two-dimensional view')
    load = node.func.value
    _need(isinstance(load, ast.Call) and _attribute(load.func, torch, 'from_file') and len(load.args) == 3
          and isinstance(load.args[1], ast.Constant) and load.args[1].value is False
          and len(load.keywords) == 1 and load.keywords[0].arg == 'dtype'
          and _attribute(load.keywords[0].value, torch, 'float32'),
          'file entry inputs require private torch.from_file CPU float32 reads')
    path = load.args[0]
    _need(isinstance(path, ast.Call) and _name(path.func, 'str') and len(path.args) == 1 and not path.keywords,
          'file entry read requires str of one qualified input path')
    join, = path.args
    _need(isinstance(join, ast.BinOp) and isinstance(join.op, ast.Div) and isinstance(join.left, ast.Name),
          'file entry read requires an argv root joined to one literal filename')
    root, filename = values.get(join.left.id), _literal(join.right, values)
    _need(type(root) is _Argument and root.kind == 'PATH' and type(filename) is str
          and filename not in ('.', '..') and _FILENAME.fullmatch(filename) is not None,
          'file entry read requires a portable filename under its argv root')
    shape = tuple(_literal(dim, values) for dim in node.args)
    count = _literal(load.args[2], values)
    _need(all(type(dim) is int and dim > 0 for dim in shape) and prod(shape) < 2**61
          and type(count) is int and count == prod(shape),
          'file read extent differs from the positive static view shape')
    return _Load(root, filename, shape, ordinal, _span(node))


def _prefix_value(node, values, load_order, torch, path, protected):
    argument = _argument(node, {path: 'PATH', 'int': 'STEP_COUNT'})
    if argument is not None:
        return argument
    if _method_call(node, 'view', 2):
        result = _load(node, values, torch, len(load_order))
        load_order.append(result)
        return result
    if isinstance(node, ast.ListComp):
        _need(len(node.generators) == 1, 'file list comprehension requires one finite binding')
        generator, = node.generators
        _need(not generator.ifs and not generator.is_async and isinstance(generator.target, ast.Name)
              and generator.target.id not in protected and generator.target.id not in values,
              'file list comprehension must have one unshadowed target and no filtering')
        sequence = _literal(generator.iter, values)
        _need(type(sequence) is tuple and sequence and all(type(item) is str for item in sequence),
              'file list comprehension requires a nonempty static filename sequence')
        rows = []
        for item in sequence:
            row = _load(node.elt, {**values, generator.target.id: item}, torch, len(load_order))
            rows.append(row)
            load_order.append(row)
        return tuple(rows)
    return _literal(node, values)


def _bind(target, value, values, protected):
    if isinstance(target, ast.Name):
        _need(target.id not in values and target.id not in protected,
              'file entry local binding must be unique and cannot shadow a provider')
        values[target.id] = value
        return
    _need(isinstance(target, (ast.List, ast.Tuple)) and type(value) is tuple,
          'file entry assignment requires a simple binding or finite destructure')
    starred = [i for i, item in enumerate(target.elts) if isinstance(item, ast.Starred)]
    _need(len(starred) <= 1 and all(isinstance(item, ast.Name) or isinstance(item, ast.Starred)
                                    and isinstance(item.value, ast.Name) for item in target.elts),
          'file entry admits only direct and starred local destructure')
    if starred:
        offset, = starred
        stop = len(value) - (len(target.elts) - offset - 1)
        _need(offset <= stop, 'file entry destructure exceeds its source sequence')
        targets = [item.value if isinstance(item, ast.Starred) else item for item in target.elts]
        value = (*value[:offset], value[offset:stop], *value[stop:])
    else:
        targets = target.elts
    _need(len(targets) == len(value), 'file entry destructure differs in length from its source sequence')
    for item, element in zip(targets, value, strict=True):
        _bind(item, element, values, protected)


def _module_names(tree, fn, rhs, torch):
    """Allow only the known import providers and the standard source launch."""
    providers = {}
    required = {('json',), ('sys',), ('torch',), ('pathlib', 'Path'), ('torchdiffeq', 'odeint')}
    launch = False
    for row in tree.body:
        if row in (fn, rhs) or row is tree.body[0] and isinstance(row, ast.Expr) and isinstance(row.value, ast.Constant) and type(row.value.value) is str:
            continue
        if isinstance(row, ast.Import):
            entries = [((alias.name,), _binding(alias)) for alias in row.names]
        elif isinstance(row, ast.ImportFrom) and row.level == 0:
            entries = [((row.module, alias.name), _binding(alias)) for alias in row.names]
        elif isinstance(row, ast.If) and row is tree.body[-1]:
            eq = row.test
            _need(isinstance(eq, ast.Compare) and _name(eq.left, '__name__') and len(eq.ops) == 1
                  and type(eq.ops[0]) is ast.Eq and len(eq.comparators) == 1
                  and isinstance(eq.comparators[0], ast.Constant) and eq.comparators[0].value == '__main__'
                  and not row.orelse and len(row.body) == 1 and isinstance(row.body[0], ast.Expr)
                  and isinstance(row.body[0].value, ast.Call) and _name(row.body[0].value.func, fn.name)
                  and not row.body[0].value.args and not row.body[0].value.keywords,
                  'file entry requires the usual direct __main__ invocation')
            launch = True
            continue
        else:
            raise ValueError('file entry has unqualified module-level behavior')
        for key, local in entries:
            _need(key in required and key not in providers and local not in providers.values(),
                  'file entry references an unsupported, duplicate or shadowed provider import')
            providers[key] = local
    _need(launch and providers.keys() == required and providers[('torch',)] == torch and providers[('sys',)] == 'sys',
          'file entry is missing a uniquely bound provider or direct __main__ invocation')
    protected = {'str', 'int', 'list', 'print', '__name__', fn.name, rhs.name}
    _need(len(protected) == 7 and not protected.intersection(providers.values()),
          'file entry cannot shadow an entry function or builtin')
    protected.update(providers.values())
    return providers[('pathlib', 'Path')], providers[('json',)], protected


def _no_definition_effects(function):
    args = function.args
    return (not function.decorator_list and function.returns is None and not args.defaults
            and all(item is None for item in args.kw_defaults) and not getattr(function, 'type_params', None)
            and all(item.annotation is None for item in ast.walk(args) if isinstance(item, ast.arg)))


def _terminal(fn, solve, values, json_module, steps):
    suffix = fn.body[fn.body.index(solve)+1:]
    _need(len(suffix) == 3 and isinstance(suffix[0], ast.Assign) and len(suffix[0].targets) == 1
          and isinstance(suffix[0].targets[0], ast.Name),
          'file entry requires exactly one terminal extraction followed by file and stdout publication')
    final = suffix[0].targets[0].id
    result = suffix[0].value
    _need(_method_call(result, 'contiguous', 0) and isinstance(result.func.value, ast.Subscript),
          'file entry terminal requires contiguous extraction of the solver endpoint')
    select = result.func.value
    _need(_name(select.value, solve.targets[0].id) and isinstance(select.slice, ast.UnaryOp)
          and isinstance(select.slice.op, ast.USub) and isinstance(select.slice.operand, ast.Constant)
          and type(select.slice.operand.value) is int and select.slice.operand.value == 1,
          'file entry can publish only the last solution Tensor')
    _need(isinstance(suffix[1], ast.Expr) and _method_call(suffix[1].value, 'write_bytes', 1)
          and isinstance(suffix[1].value.func.value, ast.Name),
          'file entry terminal bytes must be written exactly once to an argv path')
    write = suffix[1].value
    output = values.get(write.func.value.id)
    data, = write.args
    _need(type(output) is _Argument and output.kind == 'PATH' and _method_call(data, 'tobytes', 0)
          and _method_call(data.func.value, 'numpy', 0) and _name(data.func.value.func.value, final),
          'file publication must write the terminal Tensor bytes to its argv path')
    _need(isinstance(suffix[2], ast.Expr) and isinstance(suffix[2].value, ast.Call),
          'file entry requires one final stdout publication')
    publish = suffix[2].value
    _need(_name(publish.func, 'print') and len(publish.args) == 1 and not publish.keywords,
          'stdout publication must be one ordinary print')
    dumps, = publish.args
    _need(isinstance(dumps, ast.Call) and _attribute(dumps.func, json_module, 'dumps') and len(dumps.args) == 1
          and isinstance(dumps.args[0], ast.Dict) and len(dumps.keywords) == 1 and dumps.keywords[0].arg == 'sort_keys'
          and isinstance(dumps.keywords[0].value, ast.Constant) and dumps.keywords[0].value.value is True,
          'stdout publication requires sorted standard JSON')
    fields = {}
    for key, arg in zip(dumps.args[0].keys, dumps.args[0].values, strict=True):
        _need(isinstance(key, ast.Constant) and type(key.value) is str and key.value not in fields,
              'stdout object requires unique literal string keys')
        if isinstance(arg, ast.Name) and values.get(arg.id) is steps:
            fields[key.value] = dict(key=key.value, kind='STEP_COUNT')
        elif (isinstance(arg, ast.Call) and _name(arg.func, 'list') and len(arg.args) == 1 and not arg.keywords
              and _attribute(arg.args[0], final, 'shape')):
            fields[key.value] = dict(key=key.value, kind='FINAL_SHAPE')
        else:
            value = _literal(arg, values)
            _need(type(value) in (int, float, str, bool, type(None)) and (type(value) is not float or isfinite(value)),
                  'stdout publication can contain only JSON scalars, final shape or the step count')
            fields[key.value] = dict(key=key.value, kind='LITERAL', value=value)
    return output, [fields[key] for key in sorted(fields)], tuple(map(_span, suffix))


def _entry_graph(root, steps, output, loads, inputs, setup, solve_span, suffix, stdout):
    nodes = [dict(id=kind, kind='ARGV', index=argument.index, source_span=list(argument.span))
             for kind, argument in (('root', root), ('steps', steps), ('output', output))]
    edges = []

    def edge(before, after, kind, **attrs):
        edges.append(dict(source=before, target=after, kind=kind, **attrs))

    nodes.extend((dict(id='thread_setup', kind='PROVIDER_THREAD_SETUP', source_spans=[list(span) for span in setup]),
                  dict(id='recurrence', kind='FIXED_GRID_RECURRENCE', body='ADMITTED_SHARED_SOLVER_PLAN', source_span=solve_span),
                  dict(id='terminal', kind='FINAL_STATE', source_span=list(suffix[0])),
                  dict(id='file', kind='FILE_PUBLICATION', source_span=list(suffix[1])),
                  dict(id='stdout', kind='STDOUT_PUBLICATION', source_span=list(suffix[2]))))
    previous = 'thread_setup'
    for row in loads:
        path, tensor = f'path:{row.ordinal}', f'tensor:{row.ordinal}'
        nodes.extend((dict(id=path, kind='JOIN_FILENAME', filename=row.filename, source_span=list(row.span)),
                      dict(id=tensor, kind='CPU_TENSOR_FILE_READ', shape=list(row.shape), dtype='FLOAT32', source_span=list(row.span))))
        edge('root', path, 'DATA')
        edge(path, tensor, 'DATA')
        edge(previous, tensor, 'ORDER')
        previous = tensor
    edge(previous, 'recurrence', 'ORDER')
    for row in inputs:
        edge(f"tensor:{row['load_order']}", 'recurrence', 'DATA', port=row['port'])
    edge('steps', 'recurrence', 'DATA')
    edge('recurrence', 'terminal', 'DATA')
    edge('terminal', 'file', 'DATA')
    edge('output', 'file', 'DATA')
    fields = {row['kind'] for row in stdout}
    if 'FINAL_SHAPE' in fields:
        edge('terminal', 'stdout', 'DATA')
    if 'STEP_COUNT' in fields:
        edge('steps', 'stdout', 'DATA')
    edge('file', 'stdout', 'ORDER')
    return dict(nodes=nodes, edges=edges)


def derive_fixed_grid_file_entry(source, entry, *, source_bytes=None):
    """Derive source argument, Tensor port and publication bindings without execution."""
    path = Path(source['source']['path'])
    raw = path.read_bytes() if source_bytes is None else source_bytes
    solver = derive_fixed_grid_entry(source, entry, source_bytes=raw)
    tree = ast.parse(raw, filename=str(path))
    fn = _unique((row for row in tree.body if type(row) is ast.FunctionDef and row.name == entry), 'entry definition')
    rhs = _unique((row for row in tree.body if type(row) is ast.FunctionDef and row.name == source['function']), 'RHS definition')
    _need(_no_definition_effects(fn) and _no_definition_effects(rhs) and not list(ast.iter_child_nodes(fn.args)),
          'file entry requires a direct zero-argument function and no definition-time effects')
    codes = compile(raw, str(path), 'exec', dont_inherit=True).co_consts
    code = _unique((code for code in codes if type(code) is CodeType and code.co_qualname == rhs.name), 'compiled RHS')
    _need(code_key(code) == source['code_key'], 'file entry RHS differs from the admitted code identity')
    torch = source['torch_global']
    path_binding, json_module, protected = _module_names(tree, fn, rhs, torch)
    solve = _unique((row for row in fn.body if isinstance(row, ast.Assign) and list(_span(row.value)) == solver['call_span']),
                    'replayed solve assignment')
    prefix = fn.body[:fn.body.index(solve)]
    _need(len(prefix) >= 2, 'file entry requires the original CPU thread setup before input loading')
    setup = prefix[:2]
    for row, method in zip(setup, ('set_num_threads', 'set_num_interop_threads'), strict=True):
        _need(isinstance(row, ast.Expr) and _method_call(row.value, method, 1)
              and _attribute(row.value.func, torch, method) and isinstance(row.value.args[0], ast.Constant)
              and type(row.value.args[0].value) is int and row.value.args[0].value == 1,
              'file entry requires one native thread and interop thread before loading')
    values, loads = {}, []
    callback, initial, grid = solve.value.args
    steps = None
    for row in prefix[2:]:
        _need(isinstance(row, ast.Assign) and len(row.targets) == 1,
              'file entry has unqualified setup outside its provider, parameter and input bindings')
        target, = row.targets
        if _name(target, grid.id):
            # The original solver proof has already resolved the same grid and count AST.
            count = row.value.args[0].elts[1].left
            steps = values.get(count.id)
            _need(type(steps) is _Argument and steps.kind == 'STEP_COUNT',
                  'file entry grid must use the admitted argv step count')
            value = object()
        else:
            value = _prefix_value(row.value, values, loads, torch, path_binding, protected)
        _bind(target, value, values, protected)
    final = fn.body[fn.body.index(solve)+1:]
    bound_names = set(values) | protected
    _need(solve.targets[0].id not in bound_names and final and isinstance(final[0], ast.Assign)
          and len(final[0].targets) == 1 and isinstance(final[0].targets[0], ast.Name)
          and final[0].targets[0].id not in bound_names | {solve.targets[0].id},
          'file entry cannot replace an earlier binding with its solution or terminal state')
    _need(steps is not None, 'file entry lacks its admitted grid')
    result = []
    callback_state = callback.args.args[1].arg
    for arg in callback.body.args:
        if isinstance(arg, ast.Starred):
            expanded = values.get(arg.value.id)
            _need(type(expanded) is tuple, 'file entry cannot resolve the starred RHS argument sequence')
            result.extend(expanded)
        else:
            result.append(values.get(initial.id if arg.id == callback_state else arg.id))
    order = source['input_order']
    _need(all(type(arg) is _Load for arg in result) and len(result) == len(source['parameters'])
          and all(type(i) is int for i in order) and sorted(order) == list(range(len(result)))
          and sorted(arg.ordinal for arg in result) == list(range(len(loads)))
          and len({arg.filename for arg in loads}) == len(loads),
          'file entry must bind each input read once to the admitted RHS ports')
    root = loads[0].root
    _need(all(arg.root is root for arg in loads), 'file entry inputs must share one argv root binding')
    output, stdout, terminal_spans = _terminal(fn, solve, values, json_module, steps)
    parsed = {value for value in values.values() if type(value) is _Argument}
    _need(parsed == {root, steps, output} and {root.index, steps.index, output.index} == {1, 2, 3},
          'file entry requires only its three distinct argument parses for root, steps and output')
    inputs = [dict(port=port, formal_index=index, path=result[index].filename, shape=list(result[index].shape),
                   dtype='FLOAT32', load_order=result[index].ordinal) for port, index in enumerate(order)]
    return dict(schema=SCHEMA, schema_version=1, solver=solver,
                argv=dict(input_root=root.index, steps=steps.index, output=output.index, count=3), inputs=inputs,
                publication=dict(file='FINAL_TENSOR_RAW_BYTES', stdout=stdout, order=['FILE', 'STDOUT']),
                graph=_entry_graph(root, steps, output, loads, inputs, [list(_span(row)) for row in setup],
                                   solver['call_span'], terminal_spans, stdout),
                domain=dict(normal_completion_only=True, step_count_min=1, step_count_max=2**31-1,
                            step_count_syntax='ASCII_DECIMAL_DIGITS_WITHOUT_SIGN_OR_WHITESPACE',
                            inputs='STABLE_READ_ONLY_EXACT_SIZE_REGULAR_CPU_FLOAT32_NATIVE_ENDIAN_FILES',
                            output='FRESH_FILE_WITH_EXISTING_PARENT',
                            file_effect_scope='LOGICAL_CONTENT_NOT_FAILURE_TIMING_OR_FILESYSTEM_METADATA'))
