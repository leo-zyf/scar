"""Generate rotation and short-circuit leaf helpers from an admitted scalar region."""

import ast

from scar.analysis.scalar_score_reuse_evidence import _compile
from scar.codegen import scalar_sorted_standalone as native_generator
from scar.codegen.scalar_transition_standalone import _TransitionEmitter
from scar.compiler.scalar_table_region import _definition
from scar.ir.execution_recipe import read_artifact
from scar.optimization import scalar_sorted_region as generator
from scar.optimization import scalar_transition_ir as transition
from scar.trace.code_snapshot import describe_code


_require = transition._require
_node = transition._node
_ROTATION = 'evaluate_rotation'
_VISITOR = 'visit_records'
_WITNESS = 'evaluate_first_witness'


def _bindings(input_slots, state_ports):
    positions = {port['name']: (i, port['type']) for i, port in enumerate(state_ports)}
    declarations = []
    for name, (kind, symbol) in input_slots.items():
        index, declared = positions[name]
        _require(declared == kind, 'recursive leaf reads a different type at its state port')
        declarations.append(f'    [[maybe_unused]] const auto& {symbol} = std::get<{index}>(state);')
    return declarations


class _Visitor(native_generator._Emitter):
    def block(self, body, local, depth=1):
        lines = []
        for statement in body:
            if statement['op'] == 'emit':
                fields = ', '.join(self.expression(operand, local)[1] for operand in statement['values'])
                lines.append('    ' * depth + f'if (!consume(Move{{{fields}}})) return false;')
            else:
                lines.extend(super().block([statement], local, depth))
        return lines


def _visit(plan, scorer, state_ports):
    emitter = _Visitor(plan, scorer)
    emitter.maps = {name: f'integer_map_{i}' for i, name in enumerate(sorted(plan['integer_rows']))}
    setup = _bindings(emitter.inputs, state_ports)
    body = emitter.block(plan['body'], {})
    prototype = f'bool {_VISITOR}(const State& state, Consumer&& consume) {{'
    return '\n'.join(('template<class Consumer>', prototype, *setup, *body, '    return true;', '}'))


def _rotate(module, code, fields, state_ports, scorer, evidence, capture):
    function = _definition(module, code.co_qualname)
    flag = '@rotation_flag'
    normalizer = transition._Normalize(function, code, transition._input_fields(fields), (), capture['globals'], defaults=True)
    defaults = function.args.defaults
    _require(len(defaults) == 1 and type(defaults[0]) is ast.Constant and type(defaults[0].value) is bool,
             'rotation needs its exact optional bool formal')
    normalizer.aliases[normalizer.argument] = _node('input', 'bool', name=flag)
    steps, bound, returned = transition._original_body(function, normalizer)
    _require(not steps, 'recursive rotation must directly construct its immutable result')
    values = transition._constructor(returned, evidence['constructor']['name'], fields, normalizer, bound)
    materialized, output = transition._materialize(values, 'rotation')
    leaves = []
    for kind, value in zip(fields.values(), output, strict=True):
        if kind == 'bool_pair':
            leaves.extend(_node('item', 'bool', value=value, index=_node('int', 'int', value=i)) for i in (0, 1))
        else:
            leaves.append(value)
    ports = [*state_ports, {'name': flag, 'type': 'bool'}]
    emitter = _TransitionEmitter(dict(inputs=ports, outputs=state_ports), scorer)
    lowering = emitter.evaluate([*materialized, dict(op='return', values=leaves)]).split('\n', 3)[3]
    setup = _bindings({name: slot for name, slot in emitter.inputs.items() if name != flag}, state_ports)
    setup.append(f'    const bool {emitter.inputs[flag][1]} = mode;')
    prototype = f'[[maybe_unused]] State {_ROTATION}(const State& state, bool mode) {{'
    return '\n'.join((prototype, *setup, lowering)), dict(code=describe_code(code), mode=normalizer.argument,
                                                               default=defaults[0].value)


def _class(module, qualname):
    owner = module
    for part in qualname.split('.'):
        candidates = [node for node in owner.body if type(node) is ast.ClassDef and node.name == part]
        _require(len(candidates) == 1, 'recursive leaves need one original generator owner')
        owner, = candidates
    return owner


def _first_expression(function, factory):
    parameters = function.args
    args = parameters.posonlyargs + parameters.args
    if (len(args) != 1 or function.decorator_list or parameters.vararg or parameters.kwarg
            or parameters.kwonlyargs or parameters.defaults):
        return None
    statements = function.body
    if (statements and type(statements[0]) is ast.Expr and type(statements[0].value) is ast.Constant
            and type(statements[0].value.value) is str):
        statements = statements[1:]
    if len(statements) != 1 or type(statements[0]) is not ast.Return:
        return None
    call = statements[0].value
    if (type(call) is not ast.Call or type(call.func) is not ast.Name or call.func.id != 'next' or call.keywords
            or len(call.args) != 2 or type(call.args[1]) is not ast.Constant or call.args[1].value is not None):
        return None
    comp = call.args[0]
    if type(comp) is not ast.GeneratorExp or len(comp.generators) != 1:
        return None
    step, = comp.generators
    source = step.iter
    if (step.is_async or not step.ifs or type(step.target) is not ast.Name or type(comp.elt) is not ast.Name
            or comp.elt.id != step.target.id or type(source) is not ast.Call or type(source.func) is not ast.Attribute
            or source.func.attr != factory or type(source.func.value) is not ast.Name
            or source.func.value.id != args[0].arg or source.args or source.keywords):
        return None
    return step.target.id, step.ifs


class _Predicate(generator._Generator):
    def __init__(self, function, code, fields, variable, move_fields, move_types):
        super().__init__(function, code, fields, {}, move_types, {}, {})
        self.variable = variable
        self.move = {name: _node('input', kind, name=f'@witness_{i}')
                     for i, (name, kind) in enumerate(zip(move_fields, move_types, strict=True))}

    def expr(self, node, bound):
        if type(node) is ast.Attribute and type(node.value) is ast.Name and node.value.id == self.variable:
            _require(node.attr in self.move, 'first-witness predicate reads an unbound record field')
            return self.move[node.attr]
        return super().expr(node, bound)


def _witness(module, codes, sorted_plan, scorer, state_ports, fields):
    owner, _, factory = sorted_plan['factory']['qualname'].rpartition('.')
    methods = _class(module, owner).body
    matches = [(node, region) for node in methods if type(node) is ast.FunctionDef
               if (region := _first_expression(node, factory)) is not None]
    if not matches:
        return '', None
    _require(len(matches) == 1, 'the original factory has several indistinguishable first-witness methods')
    function, (variable, expressions) = matches[0]
    code = codes[owner + '.' + function.name]
    normalizer = _Predicate(function, code, fields, variable, sorted_plan['record_fields'], sorted_plan['record_types'])
    tests = [normalizer.test(node, {}) for node in expressions]
    predicate = tests[0] if len(tests) == 1 else _node('and', 'bool', args=tests)
    moves = [dict(name=f'@witness_{i}', type=kind) for i, kind in enumerate(sorted_plan['record_types'])]
    emitter = native_generator._Emitter(dict(inputs=[*state_ports, *moves]), scorer)
    kind, condition = emitter.expression(predicate, {})
    _require(kind == 'bool', 'first-witness filter is not an exact predicate')
    state_slots = {name: emitter.inputs[name] for name in (row['name'] for row in state_ports)}
    setup = _bindings(state_slots, state_ports)
    bind = [f'        [[maybe_unused]] const auto& {emitter.inputs[row["name"]][1]} = std::get<{i}>(candidate);'
            for i, row in enumerate(moves)]
    source = '\n'.join((f'[[maybe_unused]] std::optional<Move> {_WITNESS}(const State& state) {{', *setup,
                       '    std::optional<Move> match;', f'    {_VISITOR}(state, [&](const Move& candidate) {{', *bind,
                       f'        if ({condition}) {{', '            match = candidate;', '            return false;',
                       '        }', '        return true;', '    });', '    return match;', '}'))
    builtin_calls = {node.func.id for expression in expressions for node in ast.walk(expression)
                     if type(node) is ast.Call and type(node.func) is ast.Name}
    return source, dict(code=describe_code(code), predicate=predicate,
                        required_builtins=sorted({'next', *builtin_calls}), factory=sorted_plan['factory'])


def render(sorted_plan, transition_plan):
    """Append mechanically generated leaves to a same-TU checked scalar package."""
    _require(read_artifact(sorted_plan['transition']) == transition_plan
             and sorted_plan['scorer'] == transition_plan['score_plan'], 'recursive leaves need one exact ordered/state plan')
    scorer = read_artifact(transition_plan['score_plan'])
    evidence = read_artifact(transition_plan['evidence'])
    capture = read_artifact(transition_plan['capture'])
    module, codes = _compile(transition_plan['source'])
    factory, rotation = sorted_plan['factory'], evidence['rotation']['code']
    _require(describe_code(codes[factory['qualname']]) == factory
             and describe_code(codes[rotation['qualname']]) == rotation
             and factory['qualname'].rpartition('.')[0] == rotation['qualname'].rpartition('.')[0],
             'recursive leaves differ from the original generator and rotation owner')
    state_ports = transition_plan['outputs']
    fields = {name: sorted_plan['state_fields'][name] for name in sorted_plan['state_field_order']}
    _require(transition._interface(fields) == state_ports, 'recursive leaves require the exact immutable state fields')
    rotate, rotation_metadata = _rotate(module, codes[rotation['qualname']], fields, state_ports, scorer, evidence, capture)
    visit = _visit(sorted_plan, scorer, state_ports)
    witness, witness_metadata = _witness(module, codes, sorted_plan, scorer, state_ports, fields)
    fragments = [fragment for fragment in (rotate, visit, witness) if fragment]
    source = '#include <optional>\n\nnamespace {\n\n' + '\n\n'.join(fragments) + '\n\n}  // namespace\n'
    return dict(source=source, state_ports=state_ports, move_types=sorted_plan['record_types'],
                functions=dict(rotation=_ROTATION, visitor=_VISITOR, witness=_WITNESS if witness else None),
                rotation=rotation_metadata, witness=witness_metadata)
