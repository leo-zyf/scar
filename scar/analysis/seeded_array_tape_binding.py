"""Connect sealed array-tape coefficients to an original scalar RNG construction."""

import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import struct
from types import CodeType

from scar.analysis import ndarray_tape_projection, source_array_entry_contract as entry
from scar.analysis.ndarray_tape_schedule import TAPE
from scar.codegen import native_numpy_random as random, native_numpy_random_coefficients as coefficients
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.trace.code_snapshot import code_key

SCHEMA = 'scar.seeded-array-tape-binding'
SPEC = 'scar.seeded-array-tape-selectors'
_OPS = {ast.Add: 'ADD', ast.Sub: 'SUB', ast.Mult: 'MUL', ast.Div: 'DIV'}


def _need(condition, reason):
    if not condition:
        raise ValueError('seeded array tape: ' + reason)


def _one(items, reason):
    values = list(items)
    _need(len(values) == 1, reason)
    return values[0]


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _position(node):
    return [node.lineno, node.end_lineno, node.col_offset, node.end_col_offset]


def _ast(reference):
    raw = read_artifact(reference, json_document=False)
    return ast.parse(raw, filename=reference['uri']), raw


def _body(tree, raw, reference, identity):
    root = compile(raw, reference['uri'], 'exec', dont_inherit=True, optimize=0)
    pending, found = [root], []
    while pending:
        code = pending.pop()
        if code_key(code) == identity:
            found.append(code)
        pending.extend(value for value in code.co_consts if type(value) is CodeType)
    code = _one(found, 'original callable CodeID is missing or ambiguous')
    return _one((node for node in ast.walk(tree) if type(node) is ast.FunctionDef and node.name == code.co_name
                 and min([node.lineno, *(item.lineno for item in node.decorator_list)]) == code.co_firstlineno),
                'original callable does not have one source definition')


def _as_name(node):
    return node.id if type(node) is ast.Name else None


def _static_value(node):
    if type(node) is ast.Constant and type(node.value) in (int, float):
        return node.value
    if type(node) is ast.UnaryOp and type(node.op) is ast.USub:
        return -_static_value(node.operand)
    raise ValueError('expected an original numeric literal')


def _expr(node, slots, constants):
    name = entry._name(node)
    if name in slots:
        return dict(slot=slots[name])
    if name in constants:
        return dict(literal_hex=float(constants[name]).hex())
    if type(node) is ast.BinOp and type(node.op) in _OPS:
        return dict(op=_OPS[type(node.op)], args=[_expr(node.left, slots, constants), _expr(node.right, slots, constants)])
    return dict(literal_hex=float(_static_value(node)).hex())


def _eval(program, slots):
    if 'slot' in program:
        return slots[program['slot']]
    if 'literal_hex' in program:
        return float.fromhex(program['literal_hex'])
    left, right = (_eval(arg, slots) for arg in program['args'])
    return {'ADD': lambda: left + right, 'SUB': lambda: left - right,
            'MUL': lambda: left * right, 'DIV': lambda: left / right}[program['op']]()


def _formal(worker, option, namespace):
    return _one((row['formal'] for row in worker['actuals']
                 if row['source']['expression'] == entry._term(ast.parse(namespace + '.' + option, mode='eval').body)),
                'option lacks a unique source worker binding: ' + option)


def _constructor(contract, spec, rng):
    reference, worker = contract['entry'], contract['worker']
    tree, raw = _ast(reference)
    body = _body(tree, raw, reference, worker['code_id'])
    roles = spec['roles']
    namespace = contract['argv']['namespace']
    seed, depth, width = (_formal(worker, roles[role], namespace) for role in ('seed', 'layers', 'width'))
    ordinal = _one((row['formal'] for row in worker['actuals']
                   if row['source']['expression'] == entry._term(ast.Name(id=worker['ordinal'], ctx=ast.Load()))),
                   'original ordinal does not enter the worker')
    assignments = [pair for node in body.body if (pair := entry._assignment(node)) is not None]
    generator, factory = _one((pair for pair in assignments if entry._call(pair[1], spec['rng_factory'])),
                              'original generator initialization is not unique')
    wires, wire_factory = _one((pair for pair in assignments if entry._call(pair[1], spec['wire_factory'])),
                               'original positive wire range is not unique')
    _need(not factory.keywords and len(factory.args) == 1 and
          entry._term(factory.args[0]) == entry._term(ast.parse(seed + ' + ' + ordinal, mode='eval').body)
          and not wire_factory.keywords and len(wire_factory.args) == 1 and _as_name(wire_factory.args[0]) == width,
          'runtime generator/ordered wire arguments differ')
    loop = _one((node for node in body.body if type(node) is ast.For and entry._call(node.iter, 'range')
                and len(node.iter.args) == 1 and _as_name(node.iter.args[0]) == depth and not node.iter.keywords
                and type(node.target) is ast.Name and not node.orelse), 'no original ordered construction loop')
    lanes, callbacks = [], set()
    for node in loop.body:
        invocation = node.value if type(node) is ast.Expr else None
        if type(invocation) is not ast.Call or len(invocation.args) != 1 or invocation.keywords:
            continue
        generated, = invocation.args
        if type(generated) is not ast.GeneratorExp or len(generated.generators) != 1:
            continue
        item = generated.elt
        if type(item) is not ast.Call or type(item.func) is not ast.Call:
            continue
        name = entry._name(item.func.func)
        matches = [row for row in spec['lanes'] if row['factory'] == name]
        if not matches:
            continue
        selected, = matches
        comprehension, = generated.generators
        cast, = item.func.args
        draw, = cast.args
        _need(not item.keywords and not item.func.keywords and len(item.args) == 1
              and type(comprehension.target) is ast.Name and _as_name(item.args[0]) == comprehension.target.id
              and _as_name(comprehension.iter) == wires and not comprehension.ifs and not comprehension.is_async
              and entry._name(invocation.func) == spec['append'] and entry._call(cast, 'float') and not cast.keywords
              and entry._call(draw, generator + spec['uniform_suffix']) and not draw.keywords
              and [entry._term(value) for value in draw.args] == [entry._term(ast.parse(value, mode='eval').body)
                                                                 for value in spec['bound_expressions']],
              'original construction is not one ordered scalar draw per wire')
        callbacks.add(draw.func.value)
        lanes.append(dict(selected, index=len(lanes), call=entry._fact(draw, worker['code_id']),
                          append=entry._fact(invocation, worker['code_id'])))
    _need(len(lanes) == len(spec['lanes']) > 0 and len({row['type'] for row in lanes}) == len(lanes)
          and all(node in callbacks for node in ast.walk(body)
                  if type(node) is ast.Name and isinstance(node.ctx, ast.Load) and node.id == generator),
          'additional use or ordering ambiguity in the random stream')
    return dict(seed_option=roles['seed'], layer_option=roles['layers'], width_option=roles['width'],
                seed_factory=entry._fact(factory, worker['code_id']), wire_factory=entry._fact(wire_factory, worker['code_id']),
                depth=entry._fact(loop.iter, worker['code_id']), lanes=lanes,
                uniform_bounds=rng['bounds'], index_equation='((layer * lane_count) + lane) * width + wire',
                generator_definite_local_and_no_escape_in_selected_worker=True)


def _phase_spec(source, span, roles):
    reference, code = source['source'], source['code_id']
    tree, raw = _ast(reference)
    body = _body(tree, raw, reference, code)
    target = _one((node for node in ast.walk(body) if type(node) is ast.AugAssign and _position(node) == span
                   and type(node.op) is ast.Mult and type(node.value) is ast.Name), 'missing original phase update')
    name = target.value.id
    _, definition = _one((pair for node in ast.walk(body) if (pair := entry._assignment(node)) is not None
                          and pair[0] == name and node.lineno < target.lineno), 'phase has no unique original producer')
    _need(type(definition) is ast.BinOp and type(definition.op) is ast.Pow
          and type(definition.left) is ast.Constant and definition.left.value == 1j,
          'phase base differs from the stock primitive')
    conditions = [node.test for node in ast.walk(body) if type(node) is ast.If and target in node.body]
    _need(not conditions or len(conditions) == 1 and entry._term(conditions[0]) ==
          entry._term(ast.parse(name + ' != 1', mode='eval').body), 'unsupported original phase application guard')
    return dict(phase_program=_expr(definition.right, roles['phase_slots'], roles['phase_constants']),
                phase_guard='REQUIRE_NOT_UNIT' if conditions else None,
                original_exponent=entry._fact(definition.right, code))


def _decode_float(value):
    _need(value['kind'] == 'BITS' and value['dtype'] == '<f8', 'expected an observed binary64 operand')
    return struct.unpack('<d', bytes.fromhex(value['bytes_hex']))[0]


def _spectral_constants(receipt):
    turns = [float.fromhex(value) for value in receipt['half_turns']]
    rows = [[complex(real, imag) for real, imag in struct.iter_unpack('<dd', bytes.fromhex(value))]
            for value in receipt['terms']]
    _need(len(turns) == len(rows) == 2 and all(len(row) == 4 for row in rows),
          'spectral constants do not fit the admitted coefficient provider ABI')
    return turns, rows


def _spectral_bits(exponent, shift, constants):
    turns, rows = constants
    factors = [1j ** (2 * exponent * (turn + shift)) for turn in turns]
    result = bytearray()
    for elements in zip(*rows):
        value = 0j
        for element, factor in zip(elements, factors):
            value = value + element * factor
        result.extend(struct.pack('<dd', value.real, value.imag))
    return result.hex()


_TEMPLATE_DYNAMIC = {'address_mod_64', 'storage', 'coefficient', 'coefficients', 'selected', 'sha256'}


def _template_geometry(value):
    if type(value) is dict:
        return {key: _template_geometry(child) for key, child in value.items() if key not in _TEMPLATE_DYNAMIC}
    if type(value) is list:
        return [_template_geometry(child) for child in value]
    return value


def _tape(tape_ref, profile, construction, rng, coefficient_receipt, spec):
    tape = ndarray_tape_projection.check(read_artifact(tape_ref))
    observation = read_artifact(tape['observation'])
    seed = profile[construction['seed_option']]
    width, depth = (profile[construction[name + '_option']] for name in ('width', 'layer'))
    bound = _one((row for row in coefficient_receipt['samples'] if row['seed'] == seed
                  and row['tape'] == tape['observation']), 'profile has no exact real tape coefficient validation')
    random_row = _one((row for row in rng['samples'] if row['seed'] == seed), 'profile has no exact native seeded stream')
    lane_count = len(construction['lanes'])
    _need(type(width) is int and type(depth) is int and min(width, depth) > 0,
          'nonpositive construction geometry')
    count = width * depth * lane_count
    _need(bound['rng_draws'] == random_row['native'] and bound['exponents_exact'] == count,
          'seeded construction call count differs from the original coefficient validation')
    payload = read_artifact(bound['rng_draws'], json_document=False)
    draws = struct.unpack_from(f'<{count}d', payload, 12 * 8)
    lanes = {row['type']: row for row in construction['lanes']}
    spectral = _spectral_constants(coefficient_receipt)
    phase_cache, sites, signatures, constants = {}, [], [], {}
    known_bytes = {}
    for number, raw_round in enumerate(observation['rounds']):
        extent = tape['rounds'][number]
        positions = {tape['events'][i]['original_ordinal']: (i, tape['events'][i])
                     for i in range(extent['begin'], extent['end']) if tape['events'][i]['kind'] != 'CHECKPOINT'}
        visits = Counter()
        active = None
        for event in raw_round['events']:
            kind = event['kind']
            if kind == 'GATE_BEGIN':
                _need(active is None, 'nested gate scope in original tape')
                active = dict(type=event['gate_type'], axes=event['axes'], wires=event['qubits'], role=None,
                              exponent=_decode_float(event['exponent']), shift=_decode_float(event['global_shift']))
                if active['type'] in lanes:
                    lane = lanes[active['type']]
                    wire, = active['wires']
                    token = lane['index'], wire
                    layer = visits[token]
                    index = (layer * lane_count + lane['index']) * width + wire
                    _need(type(wire) is int and 0 <= wire < width and 0 <= layer < depth and
                          struct.pack('<d', draws[index] / float.fromhex(spec['pi'])) ==
                          bytes.fromhex(event['exponent']['bytes_hex']) and active['shift'] == float.fromhex(lane['shift']),
                          'selected construction draw does not equal the actual operating exponent/shift')
                    visits[token] += 1
                    active.update(role=lane['role'], layer=layer, lane=lane['index'], wire=wire, draw_index=index)
                signatures.append({key: active[key] for key in ('type', 'axes', 'wires', 'role')})
            elif kind == 'GATE_END':
                _need(active is not None, 'unmatched original gate completion')
                active = None
            elif kind in ('PARTITION', 'PHASE_MULTIPLY'):
                _need(active is not None, 'coefficient operation escaped actual gate scope')
                index, projected = positions[event['ordinal']]
                source = event['source']
                raw = projected['coefficients' if kind == 'PARTITION' else 'coefficient']
                key = kind, source['code_id'], tuple(source['position'])
                if active['role'] is None:
                    _need(kind == 'PHASE_MULTIPLY', 'unbound matrix coefficient in retained topology')
                    constants[str(index)] = raw
                    signatures.append(dict(kind=kind, source=source, constant=raw))
                    continue
                site = dict(event_index=index, original_ordinal=event['ordinal'], round=number,
                            **{name: active[name] for name in ('draw_index', 'layer', 'lane', 'wire')},
                            shift_hex=active['shift'].hex(), source=source)
                if active['role'] == 'SPECTRAL_PAIR' and kind == 'PARTITION':
                    site.update(kind='SPECTRAL_PAIR', require_exponent_not_equal=1)
                    _need(active['exponent'] != 1 and _spectral_bits(active['exponent'], active['shift'], spectral) == raw,
                          'ordered source spectral reduction differs from the observed matrix bytes')
                elif active['role'] == 'PHASE' and kind == 'PHASE_MULTIPLY':
                    if key not in phase_cache:
                        phase_cache[key] = _phase_spec(dict(code_id=source['code_id'], **observation['sources'][source['code_id']]),
                                                       source['position'], spec)
                    site.update(kind='PHASE', **phase_cache[key])
                    e = _eval(site['phase_program'], dict(EXPONENT=active['exponent'], SHIFT=active['shift']))
                    value = 1j ** e
                    _need(struct.pack('<dd', value.real, value.imag).hex() == raw and
                          (site['phase_guard'] is None or value != 1), 'phase expression differs from the original coefficient')
                else:
                    raise ValueError('seeded coefficient operation does not match its declared source lane')
                coordinate = (site['draw_index'], key)
                _need(known_bytes.setdefault(coordinate, raw) == raw, 'coefficient changes across original rounds')
                sites.append(site)
                signatures.append(dict(kind=kind, source=source, draw_index=active['draw_index']))
        _need(active is None and len(visits) == width * lane_count and set(visits.values()) == {depth},
              'retained round does not consume exactly the original seed construction coordinates')
    return dict(tape=tape_ref, seed=seed, sites=sites, fixed_coefficients=constants,
                observed_original_rounds=len(observation['rounds'])), (signatures, _template_geometry(tape['events']))


def project(config_path, contract_path, rng_path, coefficients_path, tapes):
    config, contract_ref = _ref(config_path, SPEC), _ref(contract_path, entry.SCHEMA)
    rng_ref, coefficient_ref = _ref(rng_path, random.SCHEMA), _ref(coefficients_path, coefficients.SCHEMA)
    spec = read_artifact(config)
    model = entry.check(contract_ref['uri'])
    rng = random.check(rng_ref['uri'])
    coefficient_receipt = coefficients.check(coefficient_ref['uri'])
    _need(spec['schema_version'] == 1 and spec['source_contract'] == contract_ref
          and coefficient_receipt['original_seeded_rng'] == rng_ref and len(tapes) == 2
          and len(model['observed_profiles']) == 2 and type(spec['seed_maximum']) is int
          and 0 <= spec['seed_maximum'] < 2**64, 'source-bound two-profile selection or provider identity differs')
    pi = float.fromhex(spec['pi'])
    _need(pi > 0 and [-pi, pi] == [float.fromhex(bound) for bound in rng['bounds']],
          'constructor angle and stock provider uniform bounds differ')
    construction = _constructor(model, spec, rng)
    profiles = [row['parameters'] for row in model['observed_profiles']]
    fixed = [{key: value for key, value in profile.items() if key != construction['seed_option']} for profile in profiles]
    _need(fixed[0] == fixed[1], 'observed profiles have differing nonseed options')
    task_count = _one((value for key, value in fixed[0].items()
                       if model['worker']['task_range']['expression'] == entry._term(ast.parse(
                           model['argv']['namespace'] + '.' + key, mode='eval').body)),
                      'original task range has no exact fixed option')
    _need(type(task_count) is int and task_count == 1, 'runtime seed requires the unique original ordinal zero')
    outputs, signatures = [], []
    for path in tapes:
        reference = _ref(path, TAPE)
        observed = read_artifact(reference)['observation']
        selected = _one((row['seed'] for row in coefficient_receipt['samples'] if row['tape'] == observed),
                        'projected tape has no exact native coefficient sample')
        profile = _one((row for row in profiles if row[construction['seed_option']] == selected),
                       'coefficient sample has no exact original CLI scope')
        output, signature = _tape(reference, profile, construction, rng, coefficient_receipt, spec)
        outputs.append(output)
        signatures.append(signature)
    _need(signatures[0] == signatures[1] and outputs[0]['seed'] != outputs[1]['seed'],
          'two seeds do not share the same source-ordered geometry and coefficient routes')
    return dict(schema=SCHEMA, schema_version=1, configuration=config, source_contract=contract_ref,
                random=rng_ref, coefficient_provider=coefficient_ref, construction=construction,
                seed_domain={'minimum': 0, 'maximum': spec['seed_maximum']},
                fixed_options=fixed[0], pi=spec['pi'], tapes=outputs,
                spectral=dict(half_turns=coefficient_receipt['half_turns'], components=coefficient_receipt['terms']),
                scopes=dict(topology='TWO_MATCHED_SOURCE_TAPES_EXACT_GATE_AND_COEFFICIENT_ROUTES',
                            unseen_seed_admission='REQUIRES_RUNTIME_PHASE_AND_PROVIDER_GUARDS_AND_TRAJECTORY_CHECK',
                            normalizer_selection='ORIGINAL_BRANCH_RUNTIME_LOWERING_REQUIRED',
                            no_circuit_shape_widening=True, deployment_authorized=False))


def check(path):
    stored = read_artifact(_ref(path, SCHEMA))
    actual = project(stored['configuration']['uri'], stored['source_contract']['uri'], stored['random']['uri'],
                     stored['coefficient_provider']['uri'], [row['tape']['uri'] for row in stored['tapes']])
    _need(stored == actual, 'stored runtime seed binding differs from cold replay')
    return stored


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'contract', 'rng', 'coefficients', 'out', 'check'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--tape', action='append', type=Path, default=[])
    args = parser.parse_args(argv)
    if args.check and not any((args.config, args.contract, args.rng, args.coefficients, args.out, args.tape)):
        result = check(args.check)
    elif all((args.config, args.contract, args.rng, args.coefficients, args.out, args.tape)) and not args.check:
        result = project(args.config, args.contract, args.rng, args.coefficients, args.tape)
        with args.out.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
    else:
        parser.error('provide --check RESULT or --config, --contract, --rng, --coefficients, two --tape and --out')
    print(json.dumps(dict(sites=[len(row['sites']) for row in result['tapes']], seed_domain=result['seed_domain'])))


if __name__ == '__main__':
    main()
