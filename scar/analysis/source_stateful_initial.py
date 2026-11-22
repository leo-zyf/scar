"""Project the source contract of a two-state periodic Euler simulation entry."""

import argparse
import ast
import json
import math
import operator
from pathlib import Path
from types import CodeType

from scar.ir.execution_recipe import _canonical, artifact_reference, read_artifact
from scar.trace.code_snapshot import call_instruction, describe_code, site


SCHEMA = 'scar.source-stateful-initial'
_SOURCE = 'scar.python-source'
_CALLS = 'scar.bounded-entry-calls'
_ARITHMETIC = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
               ast.Div: operator.truediv, ast.Pow: operator.pow}


def _require(condition, reason):
    if not condition:
        raise ValueError('stateful initialization: ' + reason)


def _one(values, role):
    values = list(values)
    _require(len(values) == 1, 'expected a unique ' + role)
    return values[0]


def _span(node):
    return [node.lineno, node.end_lineno, node.col_offset, node.end_col_offset]


def _same(a, b):
    return ast.dump(a) == ast.dump(b)


def _exact(node, expression):
    return _same(node, ast.parse(expression, mode='eval').body)


def _body(owner):
    return owner.body[1:] if ast.get_docstring(owner) is not None else owner.body


def _statement(node, text):
    _require(_same(node, ast.parse(text).body[0]), 'source statement left the admitted form: ' + text)


def _name(node, context=ast.Load):
    _require(type(node) is ast.Name and type(node.ctx) is context, 'expected an exact lexical binding')
    return node.id


def _assign(node):
    _require(type(node) is ast.Assign and len(node.targets) == 1, 'expected a single assignment')
    return node.targets[0], node.value


def _attribute(node, receiver, context=ast.Load):
    _require(type(node) is ast.Attribute and type(node.ctx) is context
        and _name(node.value) == receiver, 'expected direct receiver field access')
    return node.attr


def _call(node, positionals=None, keywords=None):
    _require(type(node) is ast.Call and all(type(value) is not ast.Starred for value in node.args)
        and all(keyword.arg is not None for keyword in node.keywords)
        and (positionals is None or len(node.args) == positionals)
        and (keywords is None or len(node.keywords) == keywords), 'call uses an unsupported argument protocol')
    return node


def _method(cls, name):
    return _one((row for row in cls.body if type(row) is ast.FunctionDef and row.name == name), 'source body for ' + name)


def _parameters(owner):
    params = owner.args
    _require(not (params.posonlyargs or params.vararg or params.kwarg) and not params.defaults,
             'expected direct positional/keyword-only parameters')
    return [arg.arg for arg in params.args], {
        arg.arg: ast.literal_eval(value) for arg, value in zip(params.kwonlyargs, params.kw_defaults, strict=True)}


def _aliases(tree):
    return {alias.asname or alias.name.split('.')[0]: alias.name
        for row in tree.body if type(row) is ast.Import for alias in row.names}


def _module(aliases, expected):
    return _one((name for name, qualified in aliases.items() if qualified == expected), 'import of ' + expected)


def _imports(tree):
    return [{**dict(span=_span(row)), 'bindings': [{
        'local': item.asname or (item.name.split('.')[0] if type(row) is ast.Import else item.name),
        'provider': (('.' * row.level + (row.module or '') + '.') if type(row) is ast.ImportFrom else '') + item.name}
        for item in row.names]} for row in tree.body if type(row) in (ast.Import, ast.ImportFrom)]


def _source(path):
    ref = artifact_reference(Path(path).resolve(strict=True), _SOURCE)
    raw = read_artifact(ref, json_document=False)
    tree = ast.parse(raw, filename=ref['uri'])
    code = compile(raw, ref['uri'], 'exec', dont_inherit=True, optimize=0)
    return ref, tree, code


def _code_object(root, *names):
    for name in names:
        root = _one((row for row in root.co_consts if type(row) is CodeType and row.co_name == name), 'code for ' + name)
    return root


def _entry(tree, module_path, model, model_tree):
    guard = _one((row for row in tree.body if type(row) is ast.If and _exact(row.test, '__name__ == "__main__"')), 'entry guard')
    _require(len(guard.body) == 1 and not guard.orelse and type(guard.body[0]) is ast.Expr, 'main entry guard differs')
    invoke = _call(guard.body[0].value, 0, 0)
    main = _method(tree, _name(invoke.func))
    _require(_parameters(main) == ([], {}) and len(_body(main)) == 4, 'entry does not have the admitted direct route')
    get_options, demo, construct, call = _body(main)
    dest, read = _assign(get_options)
    _require(type(dest) is ast.Tuple and len(dest.elts) == 2, 'expected parsed namespace and unused remainder')
    args, remainder = [_name(node, ast.Store) for node in dest.elts]
    parser_call = _call(read, 0, 0)
    parser = _method(tree, _name(parser_call.func))
    _require(_parameters(parser) == ([], {}) and not parser.decorator_list, 'parser must be a direct source body')
    _require(type(demo) is ast.If and not demo.orelse and len(demo.body) == 2 and type(demo.body[0]) is ast.Expr
        and type(demo.body[1]) is ast.Return and demo.body[1].value is None, 'demo branch must terminate before the regular entry')
    demo_port = _attribute(demo.test, args)
    demo_call = _call(demo.body[0].value, 1, 0)
    _require(_name(demo_call.args[0]) == args, 'alternate branch argument differs')
    _method(tree, _name(demo_call.func))
    target, ctor = _assign(construct)
    receiver = _name(target, ast.Store)
    constructor = _call(ctor, 0)
    local = _name(constructor.func)
    unit, name = _one(((row.module, alias.name) for row in tree.body if type(row) is ast.ImportFrom and row.level == 0
        for alias in row.names if (alias.asname or alias.name) == local), 'constructor source import')
    expected = Path(module_path).parent.joinpath(*unit.split('.')).with_suffix('.py')
    _require(expected == Path(model['uri']), 'constructor import is not the selected sibling source module')
    value = _one((row for row in model_tree.body if type(row) is ast.ClassDef and row.name == name),
        'constructor class declaration')
    _require(not value.bases and not value.keywords and not value.decorator_list, 'constructor class dispatch is open')
    _require(type(call) is ast.Expr, 'expected direct integration')
    integration = _call(call.value, 2, 1)
    _require(type(integration.args[0]) is ast.Constant and type(integration.args[0].value) is int
        and integration.args[0].value == 0 and not any(type(row) is ast.Name and type(row.ctx) is ast.Load
        and row.id == remainder for after in (demo, construct, call) for row in ast.walk(after)),
        'selected entry requires literal start time and unconsumed parse remainder')
    return dict(main=main, parser=parser, receiver=receiver, args=args, demo=demo, demo_port=demo_port, invocation=invoke,
        parser_call=parser_call, constructor=constructor, class_=value, integration=integration, model_module=unit,
        integrator_name=_attribute(integration.func, receiver))


def _port_map(tree, facts):
    parser_lib = _module(_aliases(tree), 'argparse')
    assign, *options, returned = _body(facts['parser'])
    target, constructor = _assign(assign)
    parser = _name(target, ast.Store)
    _require(_exact(_call(constructor, 0, 0).func, parser_lib + '.ArgumentParser')
        and type(returned) is ast.Return and _exact(returned.value, parser + '.parse_known_args()'),
        'must select direct argparse namespace and an unused remainder')
    result = {}
    seen = set()
    for stmt in options:
        _require(type(stmt) is ast.Expr, 'parser has other effects between option declarations')
        call = _call(stmt.value)
        _require(_exact(call.func, parser + '.add_argument'), 'expected exact argparse add_argument')
        flags = [ast.literal_eval(arg) for arg in call.args]
        long = _one((flag for flag in flags if type(flag) is str and flag.startswith('--')), 'long argparse flag')
        dest = long[2:].replace('-', '_')
        _require(dest.isidentifier() and dest not in result and set(flags).isdisjoint(seen), 'duplicate or malformed flag')
        seen.update(flags)
        attributes = {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg != 'help'}
        if set(attributes) == {'action'}:
            _require(_exact(attributes['action'], "'store_true'"), 'unsupported argument action')
            kind, default = 'bool', False
        else:
            _require(set(attributes) == {'default', 'type'}, 'argument needs an exact default and builtin parser')
            kind, default = _name(attributes['type']), ast.literal_eval(attributes['default'])
            _require(kind in ('float', 'int', 'str') and type(default).__name__ in
                ({'float': ('float', 'int'), 'int': ('int',), 'str': ('str',)}[kind]), 'declared argparse scalar type differs')
        result[dest] = dict(flags=flags, parser=kind, default=default, span=_span(call))
    routes = [(facts['demo_port'], dict(target='entry', parameter='demo_control'))]
    args = facts['args']
    routes.extend((_attribute(keyword.value, args), dict(target='constructor', parameter=keyword.arg))
                  for keyword in facts['constructor'].keywords)
    call = facts['integration']
    routes.extend([(_attribute(call.args[1], args), dict(target='integrate', ordinal=1)),
        (_attribute(call.keywords[0].value, args), dict(target='integrate', parameter=call.keywords[0].arg))])
    _require(len(routes) == len(result) == len({key for key, _ in routes}) and set(result) == {key for key, _ in routes},
        'option declarations must have exact direct constructor, integrate or branch routes')
    for key, destination in routes:
        result[key]['binding'] = destination
    return result


def _reads(node):
    return {row.id for row in ast.walk(node) if type(row) is ast.Name and type(row.ctx) is ast.Load}


def _number(node, bindings):
    if type(node) is ast.Constant and type(node.value) in (int, float, bool, str):
        return node.value
    if type(node) is ast.Name and node.id in bindings:
        return bindings[node.id]
    if type(node) is ast.UnaryOp and type(node.op) is ast.USub:
        return -_number(node.operand, bindings)
    if type(node) is ast.BinOp and type(node.op) in _ARITHMETIC:
        return _ARITHMETIC[type(node.op)](_number(node.left, bindings), _number(node.right, bindings))
    if type(node) is ast.Call and _exact(node.func, 'max') and len(node.args) == 2 and not node.keywords:
        return max(_number(node.args[0], bindings), _number(node.args[1], bindings))
    raise ValueError('stateful initialization: unsupported initializer scalar operation')


def _receiver_call(node, receiver, sources):
    _require(type(node) is ast.Expr, 'expected a receiver effect call')
    call = _call(node.value, len(sources), 0)
    _require(tuple(ast.unparse(value) for value in call.args) == tuple(sources), 'receiver call operands differ')
    return _attribute(call.func, receiver), call


def _constructor(facts, ports, np):
    cls = facts['class_']
    fn = _method(cls, '__init__')
    positional, defaults = _parameters(fn)
    _require(len(positional) == 1 and not fn.decorator_list and len(_body(fn)) >= 9,
             'incomplete direct constructor body')
    receiver = positional[0]
    incoming = {item['binding']['parameter']: label for label, item in ports.items()
        if item['binding']['target'] == 'constructor'}
    _require(len(incoming) == 4 and set(incoming) <= defaults.keys(), 'constructor runtime keyword interface differs')
    scalars, scalar_nodes, runtime, bindings = {}, {}, {}, {name: val for name, val in defaults.items() if name not in incoming}
    body = _body(fn)
    *before, x_setup, y_setup, grid_setup, u_setup, v_setup, multi, init, u_ghost, v_ghost = body
    for stmt in before:
        target, value = _assign(stmt)
        deps = _reads(value).intersection(incoming)
        if type(target) is ast.Name:
            name = _name(target, ast.Store)
            _require(not deps and name not in bindings, 'constructor local geometry depends on runtime or rebinds a parameter')
            bindings[name] = _number(value, bindings)
            continue
        field = _attribute(target, receiver, ast.Store)
        _require(field not in scalar_nodes, 'receiver scalar is assigned repeatedly')
        scalar_nodes[field] = stmt
        if deps:
            argument = _name(value)
            _require(argument in incoming, 'runtime constructor value must reach a single direct field')
            runtime[field] = incoming[argument]
        else:
            scalars[field] = _number(value, bindings)
    _require(set(runtime.values()) == set(incoming.values()) and len(runtime) == len(incoming),
             'constructor runtime args must have exact isolated state destinations')
    x_target, x_expr = _assign(x_setup)
    y_target, y_expr = _assign(y_setup)
    x_name, y_name = [_name(node, ast.Store) for node in (x_target, y_target)]
    _require(x_name != y_name and x_name not in bindings and y_name not in bindings
        and _same(x_expr, y_expr), 'grid lanes require equal invariant coordinate definitions')
    linspace = _call(x_expr, 3, 0)
    _require(_exact(linspace.func, np + '.linspace') and not any(_reads(node) & incoming.keys() for node in linspace.args),
             'grid coordinates must be independent of all runtime constructor settings')
    start, stop, n = [_number(node, bindings) for node in linspace.args]
    _require(type(n) is int and 5 <= n <= 4099 and math.isfinite(start) and math.isfinite(stop) and start < stop,
             'finite ordered ghost grid domain is required')
    target, call = _assign(grid_setup)
    _require(type(target) is ast.Tuple and len(target.elts) == 2 and
        _exact(call, f'{np}.meshgrid({x_name}, {y_name})'), 'grid must retain two coordinate mesh outputs')
    x, y = [_attribute(row, receiver, ast.Store) for row in target.elts]
    targets = []
    for stmt in (u_setup, v_setup):
        node, expr = _assign(stmt)
        _require(_exact(expr, f'{np}.zeros((len({x_name}), len({y_name})))'), 'state allocation differs from selected mesh')
        targets.append(_attribute(node, receiver, ast.Store))
    u, v = targets
    _require(len({x, y, u, v}) == 4 and not {x, y, u, v}.intersection(scalar_nodes), 'grid and state roots must differ')
    _require(type(multi) is ast.If and not multi.orelse, 'optional extra storage selection differs')
    scheme = _attribute(multi.test, receiver)
    _require(scalars.get(scheme) is False and len(multi.body) == 2, 'selected constructor must leave alternate storage unallocated')
    for stmt in multi.body:
        target, value = _assign(stmt)
        _attribute(target, receiver, ast.Store)
        _require(_exact(value, f'{np}.zeros((len({x_name}), len({y_name})))'), 'alternate allocation structure differs')
    _require(type(init) is ast.If and type(init.test) is ast.Compare and len(init.test.ops) == 1
        and type(init.test.ops[0]) is ast.Eq and len(init.test.comparators) == 1 and len(init.body) == 1,
        'initial condition must have a selected literal branch')
    condition_name = _name(init.test.left)
    condition = ast.literal_eval(init.test.comparators[0])
    _require(type(condition) is str and bindings[condition_name] == condition and len(init.orelse) == 1
        and type(init.orelse[0]) is ast.If, 'default initializer must select the first deterministic branch')
    initial, init_call = _receiver_call(init.body[0], receiver, ())
    inverse, _ = _receiver_call(_one(init.orelse[0].body, 'alternate initializer'), receiver, ())
    ghost, u_call = _receiver_call(u_ghost, receiver, (f'{receiver}.{u}',))
    second_ghost, v_call = _receiver_call(v_ghost, receiver, (f'{receiver}.{v}',))
    _require(ghost == second_ghost and initial != inverse, 'state roots must use the same boundary implementation')
    fn_initial, fn_ghost = _method(cls, initial), _method(cls, ghost)
    _check_initial(fn_initial, np, x, y, u, v)
    ghost_assignments = _check_ghosts(fn_ghost)
    arrays = {role: dict(shape=[n, n], dtype='<f8', argument_independent=True, region='FULL_WITH_GHOSTS')
              for role in (x, y, u, v)}
    result = dict(scalars=scalars, runtime_fields=runtime, arrays=arrays, mesh=dict(start=start, stop=stop, size=n),
        arguments=defaults, selected_initialization=dict(parameter=condition_name, value=condition,
            branch_span=_span(init.test), bounded_paths=[_span(init_call), _span(u_call), _span(v_call)]),
        dependencies={x: ['fixed_grid'], y: ['fixed_grid'], u: [x, y, 'initial_condition', 'periodic_boundary'],
                      v: [x, y, 'initial_condition', 'periodic_boundary']},
        array_types_conditional_on=['EXACT_NUMPY_DTYPE_DEFAULTS', 'LITTLE_ENDIAN_BINARY64'],
        runtime_scalar_values_are_specialized=False)
    local = dict(fn=fn, initialization=fn_initial, ghost=fn_ghost, scheme=scheme, receiver=receiver,
        scalars=scalar_nodes, coords=(x, y), values=(u, v), init_call=init_call, ghost_calls=(u_call, v_call),
        ghost_assignments=ghost_assignments)
    return result, local


def _check_initial(fn, np, x, y, u, v):
    positional, kw = _parameters(fn)
    _require(len(positional) == 1 and not kw and not fn.decorator_list, 'initialization receiver ABI differs')
    r = positional[0]
    a, b, uu, vv = _body(fn)
    tx, rx = _assign(a)
    ty, ry = _assign(b)
    xx, yy = _name(tx, ast.Store), _name(ty, ast.Store)
    _require(xx != yy and _exact(rx, f'{r}.{x}[1:-1, 1:-1]') and _exact(ry, f'{r}.{y}[1:-1, 1:-1]'),
             'initializer may only read the two fixed coordinate interiors')
    _statement(uu, f'{r}.{u}[1:-1, 1:-1] = 1 - {np}.exp(-80 * (({xx} + 0.05)**2 + ({yy} + 0.05)**2))')
    _statement(vv, f'{r}.{v}[1:-1, 1:-1] = {np}.exp(-80 * (({xx} - 0.05)**2 + ({yy} - 0.05)**2))')


def _check_ghosts(fn):
    positional, kw = _parameters(fn)
    _require(len(positional) == 1 and not kw and len(fn.decorator_list) == 1
        and _exact(fn.decorator_list[0], 'staticmethod'), 'periodic boundary needs a direct static operation')
    data = positional[0]
    forms = (f'{data}[0, :] = {data}[-2, :]', f'{data}[:, 0] = {data}[:, -2]',
             f'{data}[-1, :] = {data}[1, :]', f'{data}[:, -1] = {data}[:, 1]')
    for stmt, expected in zip(_body(fn), forms, strict=True):
        _statement(stmt, expected)
    return [dict(axis=axis, target=target, source=source) for axis, target, source in (
        (0, 0, -2), (1, 0, -2), (0, -1, 1), (1, -1, 1))]


def _mod_guard(node, step):
    _require(type(node) is ast.If and not node.orelse and len(node.body) == 1 and type(node.test) is ast.Compare
        and type(node.test.left) is ast.BinOp and type(node.test.left.op) is ast.Mod
        and len(node.test.ops) == len(node.test.comparators) == 1 and type(node.test.ops[0]) is ast.Eq
        and _exact(node.test.comparators[0], '0') and _exact(node.test.left.left, step),
        'pre-step effect must be guarded by exact step modulo a frequency')
    return _name(node.test.left.right)


def _integration(facts, model, local, ports, np):
    cls = facts['class_']
    fn = _method(cls, facts['integrator_name'])
    args, defaults = _parameters(fn)
    _require(len(args) == 3 and not fn.decorator_list, 'integrator receiver/time interface differs')
    receiver, t0, t1 = args
    initial, initial_step, loop, movie = _body(fn)
    t_assign, t_expr = _assign(initial)
    s_assign, s_expr = _assign(initial_step)
    time, step = _name(t_assign, ast.Store), _name(s_assign, ast.Store)
    _require(time != step and set(args).isdisjoint({time, step}) and _name(t_expr) == t0 and _exact(s_expr, '0')
        and type(loop) is ast.While and not loop.orelse and _exact(loop.test, f'{time} < {t1}'),
        'expected a compact increasing pre-test time loop')
    dumps, reports, advance, fixup, increment = loop.body
    dump_freq, report_freq = [_mod_guard(node, step) for node in (dumps, reports)]
    dump_name, dump_call = _receiver_call(dumps.body[0], receiver, (step, time))
    _statement(reports.body[0], f'print(f"step={{{step}}}; time={{{time}:e}}")')
    end_port, dump_port = _one((key for key, port in ports.items() if port['binding'] == {'target':'integrate', 'ordinal':1}),
        'runtime end-time input'), _one((key for key, port in ports.items()
        if port['binding'] == {'target':'integrate','parameter':dump_freq}), 'runtime dump frequency')
    _require(ports[end_port]['parser'] == 'float' and ports[dump_port]['parser'] == 'int'
        and set(defaults) == {dump_freq, report_freq} and type(defaults[report_freq]) is int and defaults[report_freq] > 0,
        'integrator runtime end and integer effect frequencies differ')
    dest, effect = _assign(advance)
    update_call = _call(effect, 0, 1)
    update_name = _attribute(update_call.func, receiver)
    time_keyword = update_call.keywords[0].arg
    _require(_name(dest, ast.Store) == time and _name(update_call.keywords[0].value) == time,
             'update must consume and replace the same time value')
    _require(type(fixup) is ast.If and not fixup.orelse and len(fixup.body) == 1, 'unsupported corrective dt branch')
    dt_store, remaining = _assign(fixup.body[0])
    dt = _attribute(dt_store, receiver, ast.Store)
    _require(type(model['scalars'].get(dt)) is float and 0 < model['scalars'][dt] and math.isfinite(model['scalars'][dt])
        and _exact(remaining, f'{t1} - {time}') and _exact(fixup.test, f'({t1} - {time}) < {receiver}.{dt}'),
        'dt adjustment must follow actual update and use remaining time')
    _statement(increment, f'{step} += 1')
    _require(type(movie) is ast.If and not movie.orelse and len(movie.body) == 1, 'movie effect must be separately guarded')
    movie_field = _attribute(movie.test, receiver)
    _require(movie_field in model['runtime_fields'] and ports[model['runtime_fields'][movie_field]]['parser'] == 'bool'
        and ports[model['runtime_fields'][movie_field]]['default'] is False, 'normal route requires the disabled movie port')
    movie_name, _ = _receiver_call(movie.body[0], receiver, ())
    movie_fn = _method(cls, movie_name)
    fn_update, fn_dump = _method(cls, update_name), _method(cls, dump_name)
    params, update_defaults = _parameters(fn_update)
    _require(len(params) == 1 and update_defaults == {time_keyword: 0}, 'update time argument interface differs')
    update_receiver = params[0]
    choice, ug, vg, returned = _body(fn_update)
    _require(type(choice) is ast.If and _attribute(choice.test, update_receiver) == local['scheme']
        and len(choice.body) == len(choice.orelse) == 1, 'Euler selection depends on the original initial field')
    alternate, _ = _receiver_call(choice.body[0], update_receiver, ())
    euler, euler_call = _receiver_call(choice.orelse[0], update_receiver, ())
    _method(cls, alternate)
    fn_euler = _method(cls, euler)
    u, v = local['values']
    update_ghosts = []
    for stmt, field in zip((ug, vg), (u, v), strict=True):
        ghost, call = _receiver_call(stmt, update_receiver, (f'{update_receiver}.{field}',))
        _require(ghost == local['ghost'].name, 'constructor and step must preserve the same ghosts')
        update_ghosts.append(call)
    _statement(returned, f'return {time_keyword} + {update_receiver}.{dt}')
    euler_roles, boundaries = _check_euler(fn_euler, cls, np, model, ports, u, v, dt)
    result = dict(time_order=['PRE_STEP_DUMP', 'PRE_STEP_REPORT', 'EULER', 'GHOST_U', 'GHOST_V',
            'ADVANCE_TIME_USING_CURRENT_DT', 'CLIP_NEXT_DT', 'INCREMENT_STEP'], start_time=0, step_start=0,
        report_frequency=defaults[report_freq], runtime_end_port=end_port, runtime_dump_port=dump_port,
        time_parameter=t1, step_binding=step, time_binding=time, dt_field=dt, initial_dt=model['scalars'][dt],
        loop=dict(span=_span(loop), test_span=_span(loop.test), unrolled=False,
                  next_time='time + current_dt', remaining='end_time - next_time',
                  next_dt='remaining if remaining < current_dt else current_dt',
                  clip_after_step=True, first_step_can_overshoot=True, state_carries=[u, v, dt, time, step]),
        guards=dict(enter='time < end_time', dump='step % dump_freq == 0', report='step % report_frequency == 0',
                    frame_phase='PRE_UPDATE', final_dump=False, movie_enabled_port=model['runtime_fields'][movie_field]),
        euler=euler_roles, ghost_assignments=local['ghost_assignments'])
    sources = dict(fn=fn, update=fn_update, euler=fn_euler, dump=fn_dump, movie=movie_fn,
        laplacian=boundaries['laplacian'], update_call=update_call, dump_call=dump_call, report=reports.body[0].value,
        euler_call=euler_call, ghost_calls=update_ghosts, euler_ops=boundaries)
    return result, sources


def _check_euler(fn, cls, np, model, ports, u, v, dt):
    params, options = _parameters(fn)
    _require(len(params) == 1 and not options and not fn.decorator_list, 'Euler body interface differs')
    r = params[0]
    init_u, init_v, shared, write_u, write_v = _body(fn)
    u_target, u_view = _assign(init_u)
    v_target, v_view = _assign(init_v)
    a, b = [_name(target, ast.Store) for target in (u_target, v_target)]
    _require(a != b and _exact(u_view, f'{r}.{u}[1:-1,1:-1]') and _exact(v_view, f'{r}.{v}[1:-1,1:-1]'),
             'Euler must pass the exact interior views')
    temp, temp_expr = _assign(shared)
    temp = _name(temp, ast.Store)
    _require(temp not in {a, b} and _exact(temp_expr, f'{a} * {np}.power({b}, 2)'),
             'the shared cross-species term must precede either mutation')
    _require(type(write_u) is type(write_v) is ast.AugAssign and type(write_u.op) is type(write_v.op) is ast.Add,
             'Euler must commit both state arrays in source order')
    reaction = [field for field, option in model['runtime_fields'].items() if ports[option]['parser'] == 'float']
    feed = _one((field for field in reaction if any(type(node) is ast.Attribute and _exact(node, f'{r}.{field}')
                 for node in ast.walk(write_u.value))), 'feed dependency of first species')
    death = _one((field for field in reaction if field != feed), 'death dependency of second species')
    terms, loads = [], []
    for stmt in (write_u, write_v):
        extra = _one((node for node in ast.walk(stmt.value) if type(node) is ast.Call), 'one explicit stencil call per state')
        laplacian = _attribute(extra.func, r)
        coeff = _one((field for field in model['scalars'] if field != dt and any(type(node) is ast.Attribute
             and _exact(node, f'{r}.{field}') for node in ast.walk(stmt.value))), 'one fixed diffusion coefficient')
        terms.append((laplacian, coeff))
        loads.append(extra)
    (method, diffusion_u), (other, diffusion_v) = terms
    _require(method == other and diffusion_u != diffusion_v, 'diffusion calls must share a single source stencil')
    _statement(write_u, f'{a} += {r}.{dt} * ({r}.{diffusion_u} * {r}.{method}({r}.{u}) - {temp} + {r}.{feed} * (1 - {a}))')
    _statement(write_v, f'{b} += {r}.{dt} * ({r}.{diffusion_v} * {r}.{method}({r}.{v}) + {temp} - ({r}.{feed} + {r}.{death}) * {b})')
    fn_laplacian = _method(cls, method)
    pars, kw = _parameters(fn_laplacian)
    _require(len(pars) == 1 and not kw and len(fn_laplacian.decorator_list) == 1
        and _exact(fn_laplacian.decorator_list[0], 'staticmethod'), 'stencil descriptor must be the exact static method')
    axis = pars[0]
    _statement(_one(_body(fn_laplacian), 'discrete stencil return'),
        f'return {axis}[2:,1:-1] + {axis}[1:-1,2:] + {axis}[0:-2,1:-1] + {axis}[1:-1,0:-2] - 4 * {axis}[1:-1,1:-1]')
    return dict(state=[u,v], runtime_fields=[feed,death], dt=dt, diffusion=[diffusion_u,diffusion_v],
        in_place=True, operation_order='SOURCE_LEFT_ASSOCIATIVE_BINARY64; NO_REASSOCIATION',
        power_provider='NUMPY_UFUNC_IDENTITY_AND_NUMERIC_IMPLEMENTATION_OPEN'), dict(laplacian=fn_laplacian, calls=loads)


def _dump(fn, ctor, local, ports, os, plot):
    params, defaults = _parameters(fn)
    _require(len(params) == 3 and len(defaults) == 1 and tuple(defaults.values())[0] is False,
             'selected dump must have one default-disabled plotting branch')
    r, step, time = params
    exists, xc, yc, u_buffer, v_buffer, select, saved, closed, increment = _body(fn)
    outdir = _one((field for field, key in ctor['runtime_fields'].items() if ports[key]['parser'] == 'str'),
                  'live output destination field')
    destination = f'{r}.{outdir}'
    _statement(exists, f'if not {os}.path.exists({destination}):\n    {os}.makedirs({destination})')
    views = []
    for stmt, field in zip((xc, yc, u_buffer, v_buffer), (*local['coords'], *local['values']), strict=True):
        target, value = _assign(stmt)
        _require(_exact(value, f'{r}.{field}[1:-1,1:-1]'), 'plot inputs must share the admitted coordinate/state interiors')
        views.append(_name(target, ast.Store))
    xx, yy, uu, vv = views
    _require(type(select) is ast.If and _name(select.test) in defaults and len(select.orelse) == 10,
             'plot must select the exact single-species branch')
    setup, contour, title, axis_limits, *axis = select.orelse
    pair, call = _assign(setup)
    _require(type(pair) is ast.Tuple and len(pair.elts) == 2 and _exact(call, f'{plot}.subplots(1,1,figsize=(6,6))'),
             'default figure layout differs')
    fig, ax = [_name(node, ast.Store) for node in pair.elts]
    _statement(contour, f"{ax}.contourf({xx}, {yy}, {vv}, levels=50, cmap='jet')")
    _statement(title, f'{fig}.suptitle(f"time = {{{time}:e}}")')
    binding, limits = _assign(axis_limits)
    limit = _name(binding, ast.Store)
    _require(type(limits) is ast.Tuple and len(limits.elts) == 2, 'fixed axis limits must be explicit')
    bounds = [_attribute(node, r) for node in limits.elts]
    _require(all(type(ctor['scalars'].get(field)) in (int, float) for field in bounds), 'axis limits depend on runtime inputs')
    for row, method in zip(axis[:3], ('set_title','set_xlabel','set_ylabel'), strict=True):
        _require(type(row) is ast.Expr, 'axis annotations must be direct calls')
        call = _call(row.value, 1, 0)
        _require(_exact(call.func, f'{ax}.{method}') and type(call.args[0]) is ast.Constant
            and type(call.args[0].value) is str, 'axis annotation may only use a static string')
    for row, text in zip(axis[3:], (f'{ax}.set_xlim({limit})', f'{ax}.set_ylim({limit})',
            f"{ax}.set_aspect('equal')"), strict=True):
        _statement(row, text)
    _require(type(saved) is type(closed) is ast.Expr, 'expected direct output and figure close effects')
    save = _call(saved.value, 1, 1)
    join = _call(save.args[0], 2, 0)
    _require(_exact(save.func, f'{fig}.savefig') and save.keywords[0].arg == 'dpi'
        and _exact(save.keywords[0].value, '400') and _exact(join.func, f'{os}.path.join')
        and _exact(join.args[0], destination), 'renderer output path/dpi differs')
    filename = join.args[1]
    _require(type(filename) is ast.JoinedStr and len(filename.values) == 3, 'expected indexed PNG file name')
    prefix, serial, suffix = filename.values
    _require(type(prefix) is type(suffix) is ast.Constant and type(prefix.value) is str
        and prefix.value and all(char.isalnum() or char in '-_' for char in prefix.value)
        and suffix.value == '.png' and type(serial) is ast.FormattedValue and serial.conversion == -1
        and _exact(serial.format_spec, 'f"06d"'), 'invalid fixed-width decimal PNG name')
    counter = _attribute(serial.value, r)
    _require(type(ctor['scalars'].get(counter)) is int and ctor['scalars'][counter] == 0,
             'first filename must use the zero initialization of the dump counter')
    _statement(closed, f'{plot}.close({fig})')
    _statement(increment, f'{r}.{counter} += 1')
    x, y = local['coords']
    _, v = local['values']
    return dict(name=prefix.value + '000000.png', pattern=prefix.value + '{dump_count:06d}.png', counter_field=counter,
        destination_port=ctor['runtime_fields'][outdir], renderer_inputs=[x,y,v,*bounds,'time=0', 'STATIC_PLOT_STYLE'],
        content_depends_on=[x,y,v,*bounds,'time=0','STATIC_PLOT_STYLE','PINNED_RENDERER_ENVIRONMENT'],
        argument_independent=True, runtime_destination_preserved=True, provider_effects_closed=False,
        requires=['FINITE_END_TIME_GT_START', 'POSITIVE_EXACT_DUMP_FREQUENCY', 'DIRECT_NON_MOVIE_EULER_ENTRY',
            'EXACT_INITIAL_NUMPY_PROVIDER', 'EXACT_RENDERER_FONT_BACKEND_CONFIGURATION_AND_PLATFORM'],
        file_effects=['EXISTS_TEST', 'CONDITIONAL_MAKEDIRS', 'PATH_JOIN', 'SAVEFIG', 'FIGURE_CLOSE', 'INCREMENT_DUMP_COUNT'],
        phases=['BEFORE_FIRST_EULER_MUTATION', 'BEFORE_FIRST_REPORT', 'ONLY_WHEN_LOOP_ENTERS']), dict(save=save,
            exists=exists.test.operand, mkdir=exists.body[0].value, close=closed.value, selected=contour.value)


def _observed(path, role_sites):
    if path is None:
        return None, {}
    reference = artifact_reference(Path(path).resolve(strict=True), _CALLS)
    facts = read_artifact(reference)
    _require(facts['schema_version'] == 1, 'unsupported bounded source call material')
    read_artifact(facts['program'])
    out = {}
    for role, (caller, call, expected) in role_sites.items():
        candidates = [(ordinal, row) for ordinal, row in enumerate(facts['observed_targets'])
            if row['entry_scope_reachable'] and row['observation']['caller_site']['code'] == describe_code(caller)['key']
            and row['observation']['caller_site']['position'] == _span(call)]
        selected = []
        for ordinal, row in candidates:
            obs = row['observation']
            caller_site = obs['caller_site']
            _require(call_instruction(caller, caller_site['offset']) is not None and site(caller, caller_site['offset']) == caller_site,
                'archived caller site differs from the compiled source')
            if expected is not None:
                if 'key' in expected:
                    _require(obs['target_code'] == expected, 'archived application CodeID differs: ' + role)
                else:
                    meta = json.loads(obs['metadata_json'])
                    _require(meta.get('constructed_type') == expected, 'archived constructor class selection differs')
            selected.append(dict(locator=f'#/observed_targets/{ordinal}', operation=row['operation']['node'],
                source_call=obs['source_call'], target=obs['target_code'], target_kind=obs['target_kind']))
        if selected:
            out[role] = dict(position=_span(call), rows=selected, abi_frozen=expected is not None,
                             status='BOUNDED_ATTEMPT_ONLY')
    return reference, out


def _graph(ports, constructor, integration, frame, local):
    u, v = local['values']
    x, y = local['coords']
    nodes = [dict(id=role, kind=kind) for role, kind in (
        ('parse', 'ARGV_PARSE'), ('construct', 'CONSTRUCTOR'), ('initial_mesh', 'GRID_BUILD'),
        ('initial_values', 'STATE_INITIALIZE'), ('initial_ghosts', 'PERIODIC_GHOST_UPDATE'),
        ('integrate', 'SHARED_TIME_LOOP'), ('dump', 'FILE_AND_RENDERER_EFFECT'), ('report', 'STDOUT_EFFECT'),
        ('euler', 'ORDERED_STATE_MUTATION'), ('ghosts', 'PERIODIC_GHOST_UPDATE'),
        ('advance', 'TIME_ARITHMETIC'), ('next_dt', 'CONDITIONAL_STATE_UPDATE'))]
    edges = []

    def route(before, after, value, kind='DATA', dependence=0):
        edges.append(dict(source=before, target=after, port=value, kind=kind, iteration_distance=dependence))

    for key, row in ports.items():
        target = {'entry': 'integrate', 'integrate': 'integrate', 'constructor': 'construct'}[row['binding']['target']]
        route('parse', target, key, 'CONTROL' if row['parser'] == 'bool' else 'DATA')
    for field in integration['euler']['runtime_fields']:
        route('construct', 'euler', field)
    route('construct', 'dump', frame['destination_port'], 'FILE_DESTINATION')
    route('construct', 'integrate', integration['guards']['movie_enabled_port'], 'CONTROL')
    route('construct', 'initial_mesh', 'FIXED_GRID_DEFAULTS')
    for field in (x, y):
        route('initial_mesh', 'initial_values', field)
        route('initial_mesh', 'dump', field)
    for field in (u, v):
        route('initial_values', 'initial_ghosts', field)
        route('initial_ghosts', 'integrate', field)
        route('euler', 'ghosts', field)
        route('ghosts', 'integrate', field, dependence=1)
    route('integrate', 'euler', u)
    route('integrate', 'euler', v)
    route('integrate', 'dump', v)
    route('construct', 'integrate', integration['dt_field'])
    route('integrate', 'advance', integration['dt_field'])
    route('integrate', 'euler', integration['dt_field'])
    route('integrate', 'dump', integration['runtime_dump_port'], 'CONTROL')
    route('integrate', 'report', 'current_step_time', 'CONTROL')
    route('dump', 'report', 'before_report_if_dump_selected', 'ORDER')
    route('dump', 'euler', 'before_euler_if_dump_selected', 'ORDER')
    route('report', 'euler', 'before_euler_if_report_selected', 'ORDER')
    route('ghosts', 'advance', 'after_both_ghosts', 'ORDER')
    route('advance', 'next_dt', 'next_time')
    route('next_dt', 'integrate', 'next_time_next_dt_and_step', dependence=1)
    return dict(nodes=nodes, edges=edges, shared_loop_body=True, hardware_or_duration_inferred=False)


def project(entry_path, source_path, *, entry_calls_path=None):
    """Reconstruct the normal initial-state/Euler boundary from pinned source."""
    entry, driver, entry_root = _source(entry_path)
    source, model_tree, model_root = _source(source_path)
    facts = _entry(driver, entry['uri'], source, model_tree)
    cls = facts['class_']
    _require(all(type(row) in (ast.Import, ast.ImportFrom, ast.ClassDef) for row in model_tree.body)
        and sum(type(row) is ast.ClassDef for row in model_tree.body) == 1
        and all(type(row) in (ast.Expr, ast.FunctionDef) for row in cls.body)
        and len({row.name for row in cls.body if type(row) is ast.FunctionDef}) == sum(
            type(row) is ast.FunctionDef for row in cls.body), 'model module or class has complex initialization')
    _require(all(type(row) in (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.If) for row in driver.body)
        and sum(type(row) is ast.If for row in driver.body) == 1, 'entry module has additional path effects')
    aliases = _aliases(model_tree)
    np = _module(aliases, 'numpy')
    ports = _port_map(driver, facts)
    _require(ports[facts['demo_port']]['parser'] == 'bool' and ports[facts['demo_port']]['default'] is False,
             'alternate driver path needs a false boolean premise')
    constructor, local = _constructor(facts, ports, np)
    integrator, sim = _integration(facts, constructor, local, ports, np)
    initial_png, dump_calls = _dump(sim['dump'], constructor, local, ports, _module(aliases, 'os'),
        _module(aliases, 'matplotlib.pyplot'))

    def body_code(fn):
        return _code_object(model_root, cls.name, fn.name)

    fn_code = {role: body_code(fn) for role, fn in {**{key: local[key] for key in ('fn', 'initialization', 'ghost')},
        **{key: sim[key] for key in ('update','euler','dump','movie','laplacian')}}.items()}
    fn_code['integrate'] = body_code(sim['fn'])
    main, parser = [_code_object(entry_root, facts[key].name) for key in ('main', 'parser')]
    constructor['code'] = describe_code(fn_code['fn'])
    constructor['initialization_code'] = describe_code(fn_code['initialization'])
    integrator['code'] = describe_code(fn_code['integrate'])
    integrator.update({role + '_code': describe_code(fn_code[role]) for role in ('update','euler','dump','ghost','laplacian')})
    matches = dict(main=(entry_root, facts['invocation'], describe_code(main)),
        parse=(main, facts['parser_call'], describe_code(parser)),
        constructor=(main, facts['constructor'], {'module': facts['model_module'], 'qualname': cls.name}),
        integrate=(main, facts['integration'], integrator['code']),
        initialize=(fn_code['fn'], local['init_call'], constructor['initialization_code']),
        update=(fn_code['integrate'], sim['update_call'], integrator['update_code']),
        dump=(fn_code['integrate'], sim['dump_call'], integrator['dump_code']),
        report=(fn_code['integrate'], sim['report'], None),
        euler=(fn_code['update'], sim['euler_call'], integrator['euler_code']))
    for key, caller, children, expected in (
        ('initial_ghost', 'fn', local['ghost_calls'], integrator['ghost_code']),
        ('update_ghost', 'update', sim['ghost_calls'], integrator['ghost_code']),
        ('laplacian', 'euler', sim['euler_ops']['calls'], integrator['laplacian_code'])):
        matches.update({f'{key}_{ordinal}': (fn_code[caller], child, expected) for ordinal, child in enumerate(children)})
    matches.update({role: (fn_code['dump'], call, None) for role, call in dump_calls.items()})
    calls, witnesses = _observed(entry_calls_path, matches)
    frontiers = ['NUMPY_BYTE_LEVEL_PROVIDER_AND_FLOAT_ENVIRONMENT_OPEN', 'MATPLOTLIB_RENDERER_AND_FONT_ENVIRONMENT_OPEN',
        'IMPORT_AND_PROVIDER_SIDE_EFFECTS_OPEN', 'FILESYSTEM_ERRORS_AND_EXCEPTION_TIMING_OPEN',
        'UNSELECTED_DEMO_MOVIE_RANDOM_AND_OTHER_SOLVERS_OPEN', 'LATER_FRAMES_DEPEND_ON_EVOLVED_STATE']
    runtime = list(ports)
    return dict(schema=SCHEMA, schema_version=1, entry=entry, source=source, entry_calls=calls,
        scope='DIRECT_ARGPARSE_TWO_STATE_PERIODIC_EULER_WITHOUT_MOVIE_OR_DEMO', ports=ports,
        constructor=constructor, integration=integrator,
        effects=dict(initial_png=initial_png,
            stdout=dict(belongs_to='INTEGRATE_PRE_UPDATE', start='step=0; time=0.000000e+00\n',
                frequency=integrator['report_frequency'], preserved=True, no_report_if_loop_not_entered=True),
            excluded=dict(demo_port=facts['demo_port'], movie_port=integrator['guards']['movie_enabled_port'],
                demo=False, movie=False, alternate_initialization=False, alternate_time_integrator=False),
            frontiers=frontiers),
        graph=_graph(ports, constructor, integrator, initial_png, local),
        links=dict(entry_code=describe_code(entry_root), main_code=describe_code(main), parser_code=describe_code(parser),
            declared=[dict(role=role, caller=describe_code(caller)['key'], span=_span(call), expected=expected)
                for role, (caller, call, expected) in matches.items()], observed=witnesses),
        provenance=dict(imports={'entry': _imports(driver), 'source': _imports(model_tree)},
            runtime_ports=runtime, selected_call_spans={key: _span(value) for key, value in facts.items()
                if key in ('invocation','constructor','integration')}),
        target_executed=False, transformation_authorized=False, frontiers=frontiers)


def check(path):
    stored = read_artifact(artifact_reference(Path(path).resolve(strict=True), SCHEMA))
    calls = stored['entry_calls']
    for ref in (stored['entry'], stored['source'], calls):
        if ref is not None:
            read_artifact(ref, json_document=False)
    result = project(stored['entry']['uri'], stored['source']['uri'], entry_calls_path=calls['uri'] if calls else None)
    _require(_canonical(result) == _canonical(stored), 'receipt differs from cold source and call replay')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('entry','source','entry-calls','out','check'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args(argv)
    if args.check and not any((args.entry, args.source, args.entry_calls, args.out)):
        result = check(args.check)
    elif args.entry and args.source and args.out and args.check is None:
        result = project(args.entry, args.source, entry_calls_path=args.entry_calls)
        with args.out.open('x', encoding='utf-8') as output:
            output.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n')
    else:
        parser.error('provide --entry and --source with --out, or --check')
    print(json.dumps(dict(schema=SCHEMA, arrays=result['constructor']['arrays'],
        observed_roles=len(result['links']['observed']), initial_dt=result['integration']['initial_dt'])))


if __name__ == '__main__':
    main()
