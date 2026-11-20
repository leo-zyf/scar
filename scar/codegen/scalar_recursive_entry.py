"""Export a checked finite source protocol as an independent C++20 executable."""

import argparse
import ast
from collections import namedtuple
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
from tempfile import TemporaryDirectory

from scar.analysis import scalar_score_reuse_evidence as source
from scar.codegen import scalar_recursive_driver as provider, scalar_recursive_standalone as kernel
from scar.codegen.native_project import build_native_project, write_native_project
from scar.compiler.scalar_table_region import _definition
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.scalar_score_ir import _int64, _require


SCHEMA = 'scar.scalar-recursive-entry-executable'
PLAN = 'scar.scalar-recursive-entry-plan'
OBSERVATION = 'scar.scalar-recursive-entry-observation'
_PROGRAM = 'scar_recursive_entry'
_BINARY = 'bin/' + _PROGRAM
_FLAGS = ('-O3', '-Wall', '-Wextra', '-Werror', '-pedantic')
_FILE = Path(__file__).with_name('scalar_recursive_entry_v1.cpp.in')


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _local(path, role, schema):
    return dict(_ref(path / role, schema), uri=role)


def _original_tree(ref, code):
    from scar.trace.code_snapshot import describe_code

    module, codes = source._compile(ref)
    _require(describe_code(codes[code['qualname']]) == code, 'original source code identity differs')
    return module, _definition(module, code['qualname'])


def _name(node):
    return node.id if type(node) is ast.Name else None


def _literal(node):
    return ast.literal_eval(node)


def _is(node, text):
    return ast.dump(node) == ast.dump(ast.parse(text, mode='eval').body)


def _one(nodes, message):
    return source._one(nodes, message)


def _assigned(statements, name):
    return _one((row.value for row in statements if type(row) is ast.Assign
                 and len(row.targets) == 1 and _name(row.targets[0]) == name),
                f'expected one original value for {name}')


def _field_writes(rows, field_name):
    for row in rows:
        if type(row) is ast.Assign and len(row.targets) == 1:
            slot = row.targets[0]
            if type(slot) is ast.Subscript and _name(slot.value) == field_name:
                yield _literal(slot.slice), row.value


def _format_suffix(node, port):
    _require(type(node) is ast.JoinedStr and len(node.values) == 2, 'expected source scalar plus format suffix')
    expression, suffix = node.values
    _require(type(expression) is ast.FormattedValue and _name(expression.value) == port
             and expression.conversion == -1 and expression.format_spec is None,
             'score field no longer formats the returned source score')
    label = _literal(suffix)
    _require(type(label) is str and label.startswith(' ') and label.strip() and label.count(' ') == 1,
             'expected one original bound label')
    return label.strip()


def _literal_string(text):
    _require(type(text) is str and text.isascii() and '\0' not in text, 'expected ASCII source data')
    return json.dumps(text, ensure_ascii=True)


def _compare_expr(node, depth, limit, score, gamma):
    if _is(node, f'{score} >= {gamma}'):
        return 'score >= threshold'
    _require(type(node) is ast.Compare and len(node.ops) == 1 and type(node.ops[0]) is ast.GtE
             and _name(node.comparators[0]) == limit and type(node.left) is ast.BinOp
             and _name(node.left.left) == depth and type(node.left.op) is ast.Sub,
             'expected the original depth-bound consumer edge')
    offset = _literal(node.left.right)
    _require(type(offset) is int and 0 <= offset <= 10, 'original look-ahead depth leaves bounded domain')
    return f'depth - {offset} >= input.depth'


def _print_calls(rows, arity):
    return [row.value for row in rows if type(row) is ast.Expr and type(row.value) is ast.Call
            and _name(row.value.func) == 'print' and len(row.value.args) == arity and not row.value.keywords]


def _consumer(ui):
    _, fn = _original_tree(ui['ui_source'], ui['consumer'])
    loop = _one((row for row in fn.body if type(row) is ast.For), 'one original probe consumer required')
    _require(type(loop.target) is ast.Tuple and len(loop.target.elts) == 4, 'source yield binding must have four ports')
    depth, gamma, score, move = map(_name, loop.target.elts)
    limit = ui['limit_port']
    _require(all((depth, gamma, score, move)) and len(loop.body) >= 4, 'typed probe consumer roles differ')
    new_depth, depth_limit = loop.body[:2]
    _require(type(new_depth) is type(depth_limit) is ast.If and type(new_depth.test) is ast.Compare
             and len(new_depth.test.ops) == 1 and type(new_depth.test.ops[0]) is ast.Gt
             and _name(new_depth.test.left) == depth, 'original iterative-depth commit site differs')
    previous = _name(new_depth.test.comparators[0])
    initial = _literal(_assigned(fn.body, previous))
    _require(type(initial) is int and initial >= 0, 'original first depth left typed domain')
    commit = new_depth.body[0]
    _require(type(commit) is ast.Assign and len(commit.targets) == 1 and type(commit.targets[0]) is ast.Tuple
             and type(commit.value) is ast.Tuple and len(commit.targets[0].elts) == len(commit.value.elts) == 2,
             'expected original candidate and depth commit')
    best, last = map(_name, commit.targets[0].elts)
    candidate = _name(commit.value.elts[0].values[0])
    _require(last == previous and _name(commit.value.elts[1]) == depth and best and candidate
             and _is(commit.value.elts[0], f'{candidate} or {best}'), 'original candidate commit differs')
    _require(type(depth_limit.body[-1]) is ast.Break and type(depth_limit.body[0]) is ast.If
             and _is(depth_limit.body[0].test, f'{score} >= {gamma} and {move} is not None')
             and type(depth_limit.body[0].body[0]) is ast.Assign
             and _name(depth_limit.body[0].body[0].targets[0]) == best,
             'original look-ahead move publication differs')

    fields = _one((row for row in loop.body if type(row) is ast.Assign and type(row.value) is ast.Dict),
                  'one ordered source report field map required')
    output_var = _name(fields.targets[0])
    keys = list(map(_literal, fields.value.keys))
    values = fields.value.values
    _require(len(keys) == len(values) == 4 and all(type(key) is str for key in keys) and len(set(keys)) == 4
             and _name(values[0]) == depth and type(values[2]) is ast.Attribute
             and _name(values[2].value) == ui['receiver'], 'original report ports differ')
    millis, rate = values[1], values[3]
    _require(all(type(node) is ast.Call and _name(node.func) == 'round' and len(node.args) == 1 and not node.keywords
                 for node in (millis, rate)), 'original timing report must use exact rounding providers')
    timed = millis.args[0]
    ratio = rate.args[0]
    _require(type(timed) is type(ratio) is ast.BinOp and type(timed.op) is ast.Mult
             and type(ratio.op) is ast.Div and ast.dump(ratio.left) == ast.dump(values[2])
             and type(ratio.right) is ast.Call and _name(ratio.right.func) == 'max'
             and len(ratio.right.args) == 2 and not ratio.right.keywords
             and _name(ratio.right.args[0]) == _name(timed.right) is not None,
             'original elapsed and node-rate ports differ')
    scale, epsilon = _literal(timed.left), _literal(ratio.right.args[1])
    _require(type(scale) is int and scale > 0 and type(epsilon) is float and epsilon > 0
             and math.isfinite(epsilon), 'original telemetry parameters differ')

    high = _one((row for row in loop.body if type(row) is ast.If and _is(row.test, f'{score} >= {gamma}')),
                'one original score report branch required')
    _require(type(high.body[0]) is ast.If and _is(high.body[0].test, f'{move} is None')
             and type(high.body[0].body[-1]) is ast.Break, 'terminal original report differs')
    terminal = list(_field_writes(high.body[0].body, output_var))
    low, pv = list(_field_writes(high.body, output_var))
    upper = _one(_field_writes(high.orelse, output_var), 'expected one original upper bound report')
    _require(len(terminal) == 1 and terminal[0][0] == low[0] == upper[0] and _name(terminal[0][1]) == score,
             'original exact score port differs from the fail-high and fail-low paths')
    info = _one(_print_calls(loop.body, 2), 'one original info emission required')
    terminal_info = _one(_print_calls(high.body[0].body, 2), 'one original exact-score emission required')
    _require(ast.dump(info) == ast.dump(terminal_info), 'normal and exact-score publications differ')
    joined = info.args[1]
    _require(type(joined) is ast.Call and type(joined.func) is ast.Attribute and joined.func.attr == 'join'
             and _literal(joined.func.value) == ' ' and len(joined.args) == 1 and not joined.keywords
             and type(joined.args[0]) is ast.GeneratorExp and len(joined.args[0].generators) == 1,
             'original report does not join its source field order')
    loop_join = joined.args[0].generators[0]
    key, value = map(_name, loop_join.target.elts)
    _require(_is(loop_join.iter, f'{output_var}.items()') and not loop_join.ifs and not loop_join.is_async
             and _is(joined.args[0].elt, 'f"{' + key + '} {' + value + '}"'),
             'original report field rendering differs')
    rendered_keys = [*keys, low[0]]
    field_values = ('depth', 'sampled_ms', 'nodes', 'sampled_rate', 'score')
    literal = _literal_string(_literal(info.args[0]))
    report = literal + ''.join(f' << {_literal_string(" " + key + " ")} << {value}'
                               for key, value in zip(rendered_keys, field_values, strict=True))

    final = _one((row for row in fn.body if type(row) is ast.If and _print_calls(row.body, 4)
                  and _print_calls(row.orelse, 2)), 'one original move-and-reply output branch required')
    with_reply = _one(_print_calls(final.body, 4), 'one original move-and-reply emission required')
    alone = _one(_print_calls(final.orelse, 2), 'one original single-move emission required')
    played = _name(with_reply.args[1])
    _require(played and _is(_assigned(fn.body, played), f'{best} or {candidate}')
             and _literal(with_reply.args[0]) == _literal(alone.args[0]), 'final move input or marker differs')
    variation = _name(with_reply.args[3].value)
    _require(variation and _is(with_reply.args[3], f'{variation}[1]')
             and _is(final.test, f'{played} and len({variation}) > 1 and {variation}[0] == {played}')
             and type(alone.args[1]) is ast.BoolOp and type(alone.args[1].op) is ast.Or,
             'final variation consumer differs')
    otherwise = alone.args[1].values[-1]
    fallback = _literal(otherwise.orelse)
    _require(_is(alone.args[1], f'{played} or ({variation}[0] if {variation} else {fallback!r})'),
             'final single-move fallback differs')

    time_edge = _one((row for row in loop.body if type(row) is ast.If
                     and type(row.test) is ast.Compare and _name(row.test.left) == depth
                     and type(row.test.ops[0]) is ast.Gt
                     and row is not new_depth), 'expected original per-probe time-limit edge')
    time_test = time_edge.body[0].test
    _require(type(time_test) is ast.BoolOp and type(time_test.op) is ast.And
             and len(time_test.values) == 2 and type(time_test.values[0]) is ast.Compare
             and type(time_test.values[0].ops[0]) is ast.Is and _literal(time_test.values[0].comparators[0]) is None
             and type(time_test.values[1]) is ast.Compare and type(time_test.values[1].ops[0]) is ast.Gt
             and _name(time_test.values[1].left) == _name(timed.right),
             'expected bounded fixed-mode wall check')
    budget = time_test.values[1].comparators[0]
    _require(type(budget) is ast.BinOp and type(budget.op) is ast.Div
             and type(budget.left) is ast.BinOp and type(budget.left.op) is ast.Mult
             and _name(budget.left.left), 'original wall check does not use bounded budget')
    minimum_depth = _literal(time_edge.test.comparators[0])
    multiplier, divisor = _literal(budget.left.right), _literal(budget.right)
    _require(all(type(item) is int and 0 < item < 1000 for item in (minimum_depth, multiplier, divisor)),
             'original wall check constants differ')
    timing = dict(budget_port=_name(budget.left.left), soft_budget_port=_name(time_test.values[0].left))
    return dict(MS_SCALE=str(scale), NPS_FLOOR=repr(epsilon), INFO_FIELDS='<< ' + report,
                PV_FIELD=_literal_string(' ' + pv[0] + ' '), FIRST_DEPTH=str(initial),
                NEW_DEPTH='depth > last_depth', DEPTH_LIMIT=_compare_expr(depth_limit.test, depth, limit, score, gamma),
                FAIL_HIGH=_compare_expr(high.test, depth, limit, score, gamma),
                LOWER_LABEL=_literal_string(_format_suffix(low[1], score)),
                UPPER_LABEL=_literal_string(_format_suffix(upper[1], score)),
                BESTMOVE=_literal_string(_literal(with_reply.args[0]) + ' '),
                PONDER=_literal_string(' ' + _literal(with_reply.args[2]) + ' '),
                NO_MOVE=_literal_string(fallback),
                TIME_EDGE=f'depth > {minimum_depth} && seconds > input.budget * {multiplier} / {divisor}'), timing


def _clocks(owner, driver, native):
    inventory = read_artifact(owner['artifacts']['caller_inventory'])
    source_ref = inventory['ui_source'] if 'ui_source' in inventory else read_artifact(inventory['ui_contract'])['ui_source']
    descriptor = inventory['static']['scopes'][inventory['static']['launcher']['name']]
    ui, fn = _original_tree(source_ref, descriptor)
    matches = []
    for item in ast.walk(fn):
        if type(item) is not ast.Assign or len(item.targets) != 1:
            continue
        attr, value = item.targets[0], item.value
        if (type(attr) is ast.Attribute and attr.attr == native['clock_fields']['deadline']
                and _name(attr.value) == inventory['static']['launcher']['receiver']
                and type(value) is ast.BinOp and type(value.op) is ast.Add and _is(value.left, 'time.time()')
                and type(value.right) is ast.Call and _name(value.right.func) == 'min'
                and len(value.right.args) == 2 and not value.right.keywords):
            matches.append(value.right.args[1])
    clamp = _one(matches, 'expected unique original finite hard deadline cap')
    hard = _literal(_assigned(ui.body, _name(clamp)) if _name(clamp) else clamp)
    tree, _ = _original_tree(driver['source'], driver['driver']['code'])
    cls = driver['driver']['code']['qualname'].split('.')[0]
    ctor = _definition(tree, cls + '.__init__')
    soft_field = native['clock_fields']['soft']
    initializers = []
    for node in ctor.body:
        if type(node) is ast.Assign and len(node.targets) == 1 and type(node.targets[0]) is ast.Tuple and type(node.value) is ast.Tuple:
            for target, value in zip(node.targets[0].elts, node.value.elts, strict=True):
                if type(target) is ast.Attribute and target.attr == soft_field and _name(target.value) == ctor.args.args[0].arg:
                    initializers.append(value)
    initial = _one(initializers, 'expected original soft deadline initialization')
    if type(initial) is ast.BinOp and type(initial.op) is ast.LShift:
        basis, width = _literal(initial.left), _literal(initial.right)
        _require(type(basis) is int and type(width) is int and 0 <= width <= 63, 'unsupported source epoch')
        soft = basis << width
    else:
        soft = _literal(initial)
    _require(type(hard) in (int, float) and 0 < hard <= 86400 and type(soft) is int
             and float(soft) == soft and soft >= 10**12, 'source fixed mode deadline left exact domain')
    return float(hard), float(soft)


def _seed(session, native, sorted_plan):
    root = kernel._state(session['root_input'], sorted_plan)
    history = sorted({kernel._state(row, sorted_plan) for row in session['history_input']})
    _require(root in history and session['table_version'] in native['table_slots'],
             'source world does not bind its known original root and scoring table')
    fields = dict(state=root, int=0, float=0.0, set=(len(history), history), move_map=0, entry_map=0)
    world = [fields[native['world'][name]['type']] for name in native['world_order']]
    config = tuple(session['native_configuration'][port] for port in native['configuration_ports'])
    _require(all(_int64(value) for value in config), 'original globals leave the signed scalar domain')
    return ('SCAR_RECURSIVE_INPUT_V1\n' + session['table_version'] + '\n'
            + kernel._encoded((root, 0, 0, True, world, config)))


def _tape(plan, driver, native, sorted_plan, timing, limits):
    hard, soft = limits
    scope = plan['scope']
    ui = read_artifact(scope['ui_contract'])
    original = read_artifact(scope['stdin'], json_document=False).decode('ascii')
    commands = plan['commands']
    _require(original == ''.join(row['line'] + '\n' for row in commands),
             'planned commands differ from the exact original paced input')
    searches, rows = [], []
    for projected, session in zip(plan['searches'], driver['sessions'], strict=True):
        mode = session['consumer_scalars']
        depth = projected['requested_depth']
        _require(projected['history'] == session['history_input'] and projected['root'] == session['root_input']
                 and projected['configuration'] == session['native_configuration']
                 and projected['search_table_version'] == session['table_version']
                 and depth == mode[ui['request_port']] == mode[ui['limit_port']]
                 and all(type(mode[name]) is type(value) and mode[name] == value
                         for name, value in ui['constraints'].items())
                 and mode[timing['soft_budget_port']] is None and type(depth) is int and 0 < depth < 1000,
                 'standalone position and search do not bind the original exact source inputs')
        budget = mode[timing['budget_port']]
        hclock = projected['halfmove_clock']
        _require(type(budget) in (int, float) and math.isfinite(budget) and budget >= 10**5
                 and type(hclock) is int and 0 <= hclock <= 100,
                 'bounded mode would enable an unlowered time diagnostic')
        seed = _literal_string(_seed(session, native, sorted_plan))
        white = 'true' if len(session['history_input']) % 2 else 'false'
        searches.append(f'    SearchInput{{{seed}, {depth}, {hclock}, {white}, {float(budget)!r}, {hard!r}, {soft!r}}}')
    kinds = dict(STATIC_REPLY='reply', POSITION='position', SEARCH='search', CLOSE='quit')
    for i, row in enumerate(commands):
        _require(row['index'] == i and row['operation'] in kinds, 'planned command leaves supported ordered scope')
        operation = kinds[row['operation']]
        data = _literal_string(row['line'])
        response = _literal_string(''.join(line + '\n' for line in row.get('reply_lines', [])))
        binding = row.get('search_ordinal', 0)
        _require(type(binding) is int and 0 <= binding < len(searches), 'command has no original prepared search')
        rows.append(f'    Request{{{data}, Command::{operation}, {response}, {binding}}}')
    return ('constexpr std::array searches{\n' + ',\n'.join(searches) + '\n};\n'
            'constexpr std::array commands{\n' + ',\n'.join(rows) + '\n};')


def _sorted(native):
    recursive = read_artifact(native['native'])
    linked = read_artifact(read_artifact(recursive['fusion'])['sorted'])
    return linked, read_artifact(linked['region'])


def _materials(entry_plan, driver_library):
    from scar.codegen import scalar_recursive_source_pv
    from scar.optimization import scalar_recursive_entry_plan

    plan_ref = _ref(entry_plan, PLAN)
    library_ref = _ref(driver_library, provider.SCHEMA)
    plan = scalar_recursive_entry_plan.check(plan_ref['uri'])
    native = provider.check(library_ref['uri'])
    _require(plan['decision']['native_experiment_authorized'] is True
             and plan['decision']['deployment_authorized'] is False and native['schema_version'] == 2,
             'native release needs the typed fixed source scope and rebind ABI')
    scope = plan['scope']
    driver = read_artifact(scope['source_driver'])
    ui = read_artifact(scope['ui_contract'])
    owner = read_artifact(scope['owner_plan'])
    sorted_provider, sorted_plan = _sorted(native)
    _require(native['code'] == driver['driver']['code'] and native['abi']['move'] == sorted_provider['abi']['record']
             and driver['source'] == scope['source'] == sorted_plan['source'],
             'native provider does not implement the planned original source driver')
    pv = scalar_recursive_source_pv.render(scope['ui_contract']['uri'], library_ref['uri'])
    fields, timing = _consumer(ui)
    fields['MOVE_MAP'] = native['world'][driver['yield']['move_lookup']['field']]['field']
    fields['SCORE_MAP'] = native['world'][driver['initialization']['score_table']['field']]['field']
    fields['TAPE'] = _tape(plan, driver, native, sorted_plan, timing, _clocks(owner, driver, native))
    generated = _FILE.read_text()
    for key, value in fields.items():
        generated = generated.replace('@SCAR_' + key + '@', value)
    _require('@SCAR_' not in generated, 'unfilled native entry port')
    root = Path(library_ref['uri']).parent
    members = {}
    for name, ref in native['members'].items():
        if name.startswith('include/') or name == 'src/driver.cpp':
            location = (root / name).resolve(strict=True)
            _require(location.is_relative_to(root) and ref['uri'] == name, 'invalid driver source path')
            members[name] = read_artifact(dict(ref, uri=str(location)), json_document=False)
    _require('src/driver.cpp' in members, 'translated driver source missing from the complete C++ project')
    driver_source = members.pop('src/driver.cpp')
    members.update({'src/main.cpp': driver_source + b'\n' + generated.encode('ascii'),
                    'src/source_pv.inc': pv['source'].encode('ascii')})
    return plan_ref, library_ref, plan, members, pv['source_ports']


def _elf(path):
    from scar.codegen.scalar_table_standalone import _dependencies_for

    return _dependencies_for(path)


def _version(compiler):
    return subprocess.check_output([*compiler, '--version'], text=True).splitlines()[0]


def _project(root):
    write_native_project(root, _PROGRAM, include_dirs=('include',), compile_options=_FLAGS)


def export(entry_plan, driver_library, out):
    plan, library, recipe, members, ports = _materials(entry_plan, driver_library)
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    compiler = shlex.split(os.environ.get('CXX', 'c++'))
    _require(compiler, 'a C++20 compiler is required')
    with TemporaryDirectory(prefix='.scar-native-entry-', dir=out.parent) as temporary:
        root = Path(temporary) / 'package'
        root.mkdir()
        for key, data in members.items():
            member = root / key
            member.parent.mkdir(exist_ok=True)
            member.write_bytes(data)
        _project(root)
        binary, _ = build_native_project(root, _PROGRAM, compiler)
        deps = _elf(binary)
        denied = subprocess.run([str(binary)], cwd='/', env={}, input=b'\n', capture_output=True, timeout=10)
        _require(denied.returncode == 77 and not denied.stdout and not denied.stderr,
                 'executable published an effect before admitting the exact command')
        artifact = dict(schema=SCHEMA, schema_version=1, entry_plan=plan, driver_library=library,
                        source=recipe['scope']['source'], binary=_local(root, _BINARY, 'scar.elf'),
                        cxx_standard=20, python_runtime_required=False, deployment_authorized=False,
                        compiler=dict(command=compiler, version=_version(compiler)), link_dependencies=deps,
                        output_source_ports=ports, exact_command_mismatch_status=77,
                        members={key:_local(root, key, 'application/octet-stream')
                                 for key in sorted([*members, 'CMakeLists.txt'])})
        (root / 'artifact.json').write_text(json.dumps(artifact, indent=2, sort_keys=True) + '\n')
        root.rename(out)
    return artifact


def check(path):
    ref = _ref(path, SCHEMA)
    result = read_artifact(ref)
    root = Path(ref['uri']).parent
    plan, library, recipe, members, ports = _materials(result['entry_plan']['uri'], result['driver_library']['uri'])
    _require(result['schema_version'] == 1 and result['entry_plan'] == plan and result['driver_library'] == library
             and result['source'] == recipe['scope']['source'] and result['binary'] == _local(root, _BINARY, 'scar.elf')
             and result['output_source_ports'] == ports and result['exact_command_mismatch_status'] == 77
             and result['cxx_standard'] == 20 and result['python_runtime_required'] is result['deployment_authorized'] is False
             and result['link_dependencies'] == _elf(root / _BINARY)
             and result['members'] == {key:_local(root, key, 'application/octet-stream')
                                      for key in sorted([*members, 'CMakeLists.txt'])}
             and all((root / key).read_bytes() == data for key, data in members.items()),
             'compiled entry differs from the exact original checked source and physical project')
    with TemporaryDirectory(prefix='scar-native-project-replay-') as temporary:
        _project(Path(temporary))
        _require((Path(temporary) / 'CMakeLists.txt').read_bytes() == (root / 'CMakeLists.txt').read_bytes(),
                 'published C++20 project directives changed')
    return result


def _snapshot_codec(driver, native, sorted_plan):
    from scar.codegen.scalar_recursive_lowering import _state_shape
    from scar.runtime import scalar_recursive_driver_study as bridge

    owner = driver['probe']['source_recursive_owner']
    module, function = _original_tree(driver['source'], owner)
    _, entries, _ = _state_shape(module, function, owner['parameters'][0], owner['parameters'][1])
    score_type = _one(iter(entries), 'expected unique original ordered score type')
    root_type = driver['sessions'][0]['root_input']['type']
    move_type = sorted_plan['constructor']
    score = namedtuple(score_type, entries[score_type])
    move = namedtuple(move_type, sorted_plan['record_fields'])
    position = namedtuple(root_type, sorted_plan['state_field_order'])
    return bridge._Codec({score_type:score, move_type:move}, sorted_plan, native['abi']['move']['ports'][1:],
                         position, score_type, entries[score_type])


def project_observation(artifact_path, directory):
    """Recompute the original differential from the executable's typed trace."""
    from scar.runtime.scalar_recursive_driver_observe import ordered_map_digest
    from scar.runtime.scalar_recursive_observe import _pod

    artifact_ref = _ref(artifact_path, SCHEMA)
    artifact = read_artifact(artifact_ref)
    plan = read_artifact(artifact['entry_plan'])
    driver = read_artifact(plan['scope']['source_driver'])
    native = read_artifact(artifact['driver_library'])
    _, sorted_plan = _sorted(native)
    codec = _snapshot_codec(driver, native, sorted_plan)
    root = Path(directory).resolve(strict=True)
    states = {name:native['world'][name] for name in native['world_order']}
    move_field = driver['yield']['move_lookup']['field']
    score_field = driver['initialization']['score_table']['field']
    root_field = driver['initialization']['root']['field']
    history_field = driver['initialization']['history']['field']
    node_field = driver['initialization']['counter']['field']
    results, files = [], []
    for index, expected in enumerate(driver['sessions']):
        probes, snapshot = (_ref(root / f'{index}.{suffix}', 'application/octet-stream')
                            for suffix in ('probes','world'))
        _require(probes['uri'] == str(root / f'{index}.probes')
                 and snapshot['uri'] == str(root / f'{index}.world'), 'native trace escaped its observation directory')
        table = native['table_slots'].index(expected['table_version'])
        moves = expected['initialized']['move_table']['size']
        scores = expected['initialized']['score_table']['size']
        wire = 'SCAR_ENTRY_PROBES_V1\n' + kernel._encoded((table, moves, scores))
        yields = []
        for row in expected['yields']:
            depth, gamma, score, move = row['value']
            wire += kernel._encoded((depth, gamma, score, row['nodes'], move is not None, move or ()))
            yields.append(dict(value=row['value'], nodes=row['nodes']))
        _require(read_artifact(probes, json_document=False) == wire.encode('ascii'),
                 'standalone native outputs, source state or exact probe order differ')
        decoded = codec.decode_world(read_artifact(snapshot, json_document=False), states)
        published = {name:ordered_map_digest(decoded[name]) for name in (move_field, score_field)}
        history = {kernel._state(value, sorted_plan) for value in expected['history_input']}
        unit_history = {kernel._state(_pod(value), sorted_plan) for value in decoded[history_field]}
        _require(len(decoded[move_field]) == expected['final']['move_table']['size']
                 and len(decoded[score_field]) == expected['final']['score_table']['size']
                 and published == expected['final']['ordered_map_digests']
                 and _pod(decoded[root_field]) == expected['root_input']
                 and unit_history == history
                 and decoded[node_field] == expected['final']['nodes'], 'standalone ordered world differs from source')
        results.append(dict(table_version=expected['table_version'], initial_move_entries=moves,
                            initial_score_entries=scores, yields=yields, outcome='consumer_close',
                            final_nodes=decoded[node_field], final_move_entries=len(decoded[move_field]),
                            final_score_entries=len(decoded[score_field]), ordered_map_digests=published))
        files.extend((probes, snapshot))
    return dict(schema=OBSERVATION, schema_version=1, artifact=artifact_ref, observed_from_exact_launch=True,
                original_sessions_matched=True, sessions=results, trace_files=files)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ('entry-plan', 'driver-library', 'out', 'check'):
        parser.add_argument('--' + role, type=Path)
    args = parser.parse_args(argv)
    inputs = (args.entry_plan, args.driver_library, args.out)
    if args.check and not any(inputs):
        result = check(args.check)
    elif all(inputs) and not args.check:
        result = export(*inputs)
    else:
        parser.error('provide --entry-plan, --driver-library and --out, or --check')
    print(json.dumps({key:result[key] for key in ('schema','binary','python_runtime_required','link_dependencies')},
                     sort_keys=True))


if __name__ == '__main__':
    main()
