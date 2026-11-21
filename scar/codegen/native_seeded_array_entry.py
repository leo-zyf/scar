"""Compile runtime RNG edges into an already-qualified private array schedule."""

import argparse
import ast
import json
import math
from pathlib import Path
import struct

from scar.analysis import native_array_entry_recipe, seeded_array_tape_binding, source_array_entry_contract
from scar.codegen import native_array_entry, native_array_tape, native_numpy_complex as stock
from scar.codegen import native_numpy_observables, native_numpy_random, native_numpy_random_coefficients
from scar.ir.execution_recipe import read_artifact
from scar.optimization import seeded_array_tape_memory


SCHEMA = 'scar.native-seeded-array-entry'
LOOP_SCHEMA = 'scar.native-seeded-array-loop-entry'
_PROFILE = Path(__file__).with_name('native_seeded_array_entry_profile.cpp.in')
_LOOP_PROFILE = Path(__file__).with_name('native_seeded_array_loop_profile.cpp.in')
_ITEM = 16


def _need(condition, reason):
    if not condition:
        raise ValueError('native seeded array entry: ' + reason)


def _hex(value):
    number = float.fromhex(value)
    _need(math.isfinite(number), 'nonfinite bound binary64')
    return number.hex()


def _expr(row, *, exponent, shift):
    if 'slot' in row:
        return {'EXPONENT': exponent, 'SHIFT': shift}[row['slot']]
    if 'literal_hex' in row:
        return _hex(row['literal_hex'])
    left, right = (_expr(arg, exponent=exponent, shift=shift) for arg in row['args'])
    sign = {'ADD': '+', 'SUB': '-', 'MUL': '*', 'DIV': '/'}[row['op']]
    return f'({left} {sign} {right})'


def _program(tape, schedule, route, analysis, memory, rng):
    sites = {row['event_index']: row for row in route['sites']}
    fixed = {int(index): value for index, value in route['fixed_coefficients'].items()}
    required = {index for index, row in enumerate(tape['events'])
                if row['kind'] in ('PARTITION', 'PHASE_MULTIPLY')}
    _need(len(sites) == len(route['sites']) and not set(sites) & set(fixed)
          and set(sites) | set(fixed) == required, 'original coefficient sites are not accounted for')
    bounds = [_hex(value) for value in analysis['construction']['uniform_bounds']]
    count = rng['draws_per_seed']
    pi = _hex(analysis['pi'])
    turns = ', '.join(_hex(value) for value in analysis['spectral']['half_turns'])
    components = b''.join(bytes.fromhex(payload) for payload in analysis['spectral']['components'])
    spectral = ', '.join(value.hex() for value in struct.unpack('<16d', components))
    computed, unique, hooks, dynamic_indices, fixed_literals = [], {}, [], set(), set()
    operation_index = 0
    normal = {row['normalize_operation']: row for row in memory['rows']}
    for index, operation in enumerate(schedule['operations']):
        kind = operation['kind']
        event = operation['event_index']
        if kind == 'LOAD_LITERAL':
            literal = operation['literal']
            value = schedule['literals'][literal]
            if event in sites:
                site = sites[event]
                draw = site['draw_index']
                _need(type(draw) is int and 0 <= draw < count and event not in dynamic_indices,
                      'draw is outside the checked construction stream')
                exponent = f'exponents[{draw}]'
                shift = _hex(site['shift_hex'])
                key = json.dumps({key: site.get(key) for key in ('kind', 'draw_index', 'shift_hex',
                    'phase_program', 'phase_guard', 'require_exponent_not_equal')}, sort_keys=True)
                if key not in unique:
                    at = len(computed)
                    pointer = f'prepared[{at}].data()'
                    if site['kind'] == 'SPECTRAL_PAIR':
                        _need(operation['size'] == 4 * _ITEM, 'unexpected runtime spectral extent')
                        skip = float(site['require_exponent_not_equal']).hex()
                        expression = (f'{exponent} == {skip} || '
                            f'scar_complex_spectral_pair_v1({exponent}, {shift}, turns, components, 4, {pointer})')
                    else:
                        _need(site['kind'] == 'PHASE' and operation['size'] == _ITEM
                              and site['phase_guard'] in (None, 'REQUIRE_NOT_UNIT'), 'runtime phase scope differs')
                        angle = _expr(site['phase_program'], exponent=exponent, shift=shift)
                        expression = f'scar_pure_imaginary_phase_v1({angle}, {pointer})'
                        if site['phase_guard'] == 'REQUIRE_NOT_UNIT':
                            expression += f' || (prepared[{at}][0] == 1.0 && prepared[{at}][1] == 0.0)'
                    unique[key] = at
                    computed.append(f'    if ({expression}) return false;')
                hooks.append(f'    {{{operation_index}, HookKind::coefficient, {unique[key]}}},')
                dynamic_indices.add(event)
            else:
                _need(fixed.get(event) == value, 'fixed coefficient has no independent source witness')
                fixed_literals.add(literal)
        elif kind == 'NORMALIZE':
            _need(normal[index]['repair_canonical_legal']
                  and normal[index]['expected_status'] == int(operation['selected']),
                  'runtime branch requires a ready canonical destination')
            hooks.append(f'    {{{operation_index}, HookKind::normalization, 0}},')
        operation_index += len(operation['groups']) if kind == 'COPY' else 1
    _need(dynamic_indices == sites.keys(), 'seeded coefficient site lacks a native load')
    shape_schedule = dict(schedule, literals=[
        value if index in fixed_literals else '0' * len(value)
        for index, value in enumerate(schedule['literals'])])
    bindings = '\n'.join((
        f'constexpr std::size_t coefficient_count = {len(computed)};',
        'constexpr Hook hooks[]{', *hooks, '};',
        'bool prepare_coefficients(std::uint64_t seed, std::array<Coefficient, coefficient_count>& prepared) {',
        '    std::array<std::uint64_t, 4> state{};',
        f'    std::array<double, {count}> exponents{{}};',
        f'    constexpr double turns[]{{{turns}}};',
        f'    constexpr double components[]{{{spectral}}};',
        '    scar_numpy_pcg64_init_v1(seed, state.data(), nullptr);',
        f'    if (scar_numpy_pcg64_uniform_v1(state.data(), {bounds[0]}, {bounds[1]}, exponents.size(),',
        '                                     exponents.data())) return false;',
        f'    for (auto& exponent : exponents) exponent = exponent / {pi};',
        *computed, '    return true;', '}'))
    return bindings, native_array_tape._emit(shape_schedule), len(computed)


def _report(recipe, profile, seed_option, round_option=None):
    contract = read_artifact(recipe['source_contract'])
    worker = contract['worker']
    namespace = contract['argv']['namespace']
    term = lambda source: source_array_entry_contract._term(ast.parse(source, mode='eval').body)
    seed, = (row['formal'] for row in worker['actuals']
             if row['source']['expression'] == term(f'{namespace}.{seed_option}'))
    ordinal, = (row['formal'] for row in worker['actuals']
                if row['source']['expression'] == term(worker['ordinal']))
    report = contract['report']
    observed, = (row for row in contract['witnesses'] if row['parameters'][seed_option] == profile['parameters'][seed_option])
    _need(len(observed['worker_scopes']) == 1 and observed['worker_scopes'][0]['ordinal'] == 0,
          'the source seed-plus-ordinal route requires one serial worker')
    field, = (row['name'] for row in report['worker_fields']
              if row['source']['expression'] == term(f'{seed} + {ordinal}'))
    added = [dict(path=path, role='runtime_seed') for path in (
        [report['parameters']['field'], seed_option], [report['results_field'], 0, field])]
    if round_option is not None:
        rounds, = (row['formal'] for row in worker['actuals']
                   if row['source']['expression'] == term(f'{namespace}.{round_option}'))
        round_field, = (row['name'] for row in report['worker_fields']
                        if row['source']['expression'] == term(rounds))
        added.extend(dict(path=path, role='runtime_round_count') for path in (
            [report['parameters']['field'], round_option], [report['results_field'], 0, round_field]))
    return dict(recipe, profiles=[profile], report_bindings=[*recipe['report_bindings'], *added])


def _loop_plan(loop_path, periodic_path, binding_ref, memory_ref, tape, schedule, route, memory):
    from scar.analysis import seeded_array_loop
    from scar.optimization import seeded_array_loop_memory

    ref = native_array_entry._ref
    source_ref = ref(loop_path, seeded_array_loop.SCHEMA)
    source = seeded_array_loop.check(source_ref['uri'])
    periodic_ref = ref(periodic_path, seeded_array_loop_memory.SCHEMA)
    periodic = seeded_array_loop_memory.check(periodic_ref['uri'])
    _need(source['seed_binding'] == binding_ref and source['option']['minimum'] == 1
          and periodic['memory'] == memory_ref and periodic['arena_bytes'] == memory['arena_bytes']
          and periodic['persistent_carry_bytes'] == periodic['canonical_repair_bytes_per_round'] == 0,
          'periodic lowering requires matching original loop and private memory proof')
    template, = (row for row in source['tapes'] if row['tape'] == route['tape'])
    ranges = [periodic[name]['operations'] for name in ('first', 'steady')]
    _need(ranges == schedule['rounds'][:2]
          and [template['initial']['round'], template['continuation']['round']] == [0, 1]
          and template['continuation']['end'] == tape['rounds'][1]['end']
          and template['confirmed_rounds'] and min(template['confirmed_rounds']) > 1,
          'periodic template is not the original first/steady prefix')
    carry = periodic['carry_into_steady']
    _need(carry['operation'] == ranges[1][0] and carry['before_any_other_steady_operation']
          and carry['requires_selected_pointer_for_checkpoint_and_terminal']
          and len(carry['routes']) == 4 and all(
              row['mode'] in ('MEMCPY', 'TRANSFER_SAME_SPAN') and not row['independent_live_versions']
              for row in carry['routes']), 'periodic carry has unresolved live storage')
    stop = template['continuation']['end']
    tape = dict(tape, events=tape['events'][:stop], rounds=tape['rounds'][:2])
    schedule = dict(schedule, operations=schedule['operations'][:ranges[1][1]],
                    rounds=ranges, checkpoints=schedule['checkpoints'][:2])
    route = dict(route, sites=[row for row in route['sites'] if row['event_index'] < stop],
                 fixed_coefficients={key: value for key, value in route['fixed_coefficients'].items() if int(key) < stop})
    memory = dict(memory, rows=memory['rows'][:2])
    return source_ref, periodic_ref, source['option']['id'], periodic, tape, schedule, route, memory


def export(recipe_path, binding_path, memory_paths, out, *, loop=None, periodic_memory=None):
    ref = native_array_entry._ref
    recipe_ref = ref(recipe_path, native_array_entry.RECIPE)
    recipe = native_array_entry_recipe.check(recipe_ref['uri'])
    binding_ref = ref(binding_path, seeded_array_tape_binding.SCHEMA)
    binding = seeded_array_tape_binding.check(binding_ref['uri'])
    _need(binding['source_contract'] == recipe['source_contract'] and binding['seed_domain'] ==
          dict(minimum=0, maximum=2**32 - 1), 'source or runtime seed domain differs')
    candidates = []
    for path in memory_paths:
        memory_ref = ref(path, seeded_array_tape_memory.SCHEMA)
        memory = seeded_array_tape_memory.check(memory_ref['uri'])
        route, = (row for row in binding['tapes'] if row['tape'] == memory['tape'])
        profile, = (row for row in recipe['profiles'] if row['tape_sha256'] == route['tape']['sha256'])
        package = read_artifact(profile['package'])
        _need(package['normalize'] == memory['normalization'] == recipe['normalize']
              and package['files']['schedule.json']['sha256'] == memory['schedule']['sha256']
              and package['storage']['arena_bytes'] == memory['arena_bytes'], 'memory plan differs from original tape')
        candidates.append((memory['arena_bytes'], memory_ref, memory, route, profile, package))
    _need(len(candidates) == len(binding['tapes'])
          and len({row[3]['tape']['sha256'] for row in candidates}) == len(candidates),
          'one qualified memory plan per source topology observation required')
    _, memory_ref, memory, route, profile, package = min(candidates, key=lambda row: row[0])
    tape, schedule = read_artifact(route['tape']), read_artifact(memory['schedule'])
    _need((loop is None) == (periodic_memory is None), 'runtime loop and periodic memory proofs are required together')
    loop_data = (_loop_plan(loop, periodic_memory, binding_ref, memory_ref, tape, schedule, route, memory)
                 if loop is not None else None)
    round_option = None
    if loop_data is not None:
        loop_ref, periodic_ref, round_option, periodic, tape, schedule, route, memory = loop_data
    rng = read_artifact(binding['random'])
    coefficient = read_artifact(binding['coefficient_provider'])
    seed_option = binding['construction']['seed_option']
    sources = native_array_entry._render(_report(recipe, profile, seed_option, round_option),
                                        seeded_option=seed_option, round_option=round_option)
    program, private_tape, count = _program(tape, schedule, route, binding, memory, rng)
    sources['src/tape_0.inc'] = private_tape
    if loop_data is None:
        cpp = _PROFILE.read_text().replace('@SEEDED_ROUTES@', program)
    else:
        carry = periodic['carry_into_steady']
        cpp = native_array_entry._substitute(_LOOP_PROFILE.read_text(), dict(
            SEEDED_ROUTES=program, STATE_BYTES=carry['bytes'], STEADY_DESTINATION=carry['destination_offset']))
    sources['src/profile_0.cpp'] = cpp.encode()
    normal, libraries, library, normal_provider = native_array_tape._normalize(recipe['normalize'])
    sources.update(normal)
    sources.update(native_numpy_observables.render(normal_provider))
    _, coefficient_sources = native_numpy_random_coefficients.render(coefficient['cpython'],
                                                                       coefficient['complex_provider'])
    stock_provider = package['stock_provider']
    _need(stock_provider['provider'] == read_artifact(coefficient['complex_provider'])['provider']
          and all(coefficient_sources[key] == value for key, value in stock.render(stock_provider).items()),
          'whole entry and coefficient graph use different native loops')
    sources.update(coefficient_sources)
    sources.update(native_numpy_random.render(rng['provider']))
    report = dict(schema=SCHEMA, schema_version=1, recipe=recipe_ref, binding=binding_ref, memory=memory_ref,
        cpu_ids=recipe['cpu_ids'], independent_runtime_seeds=True, seed_option=seed_option,
        source_parameters=binding['fixed_options'], seed_range=binding['seed_domain'],
        fixed_topology=True, prepared_runtime_coefficients=count, captured_seed_coefficients_embedded=False,
        seed_validation_outside_compiler=True,
        maximum_normalization_repair_bytes=memory['uniform_repair_copy_bound_bytes'],
        arena_bytes=memory['arena_bytes'], scope='SEED_INPUT_ON_EXACT_SOURCE_TOPOLOGY')
    if loop_data is not None:
        report.update(schema=LOOP_SCHEMA, loop=loop_ref, periodic_memory=periodic_ref, round_option=round_option,
            source_parameters={key: value for key, value in binding['fixed_options'].items() if key != round_option},
            scope='RUNTIME_SEED_AND_POSITIVE_ORIGINAL_ROUND_COUNT_ON_FIXED_TOPOLOGY',
            distinct_compiled_round_bodies=2, maximum_normalization_repair_bytes=0,
            maximum_carry_copy_bytes_per_transition=periodic['maximum_carry_copy_bytes_per_transition'])
    return native_array_entry._build(out, sources, profiles=1, library=library, libraries=libraries,
        stock_provider=stock_provider, normal_provider=normal_provider, report=report, random_provider=rng['provider'],
        extra_sources=(native_numpy_random.SOURCE, native_numpy_random_coefficients.SOURCE),
        archives=(native_numpy_random.ARCHIVE,))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', required=True)
    parser.add_argument('--binding', required=True)
    parser.add_argument('--memory', action='append', required=True)
    parser.add_argument('--loop')
    parser.add_argument('--periodic-memory')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    result = export(args.recipe, args.binding, args.memory, args.out, loop=args.loop,
                    periodic_memory=args.periodic_memory)
    print(json.dumps(dict(cpu_ids=result['cpu_ids'], coefficients=result['prepared_runtime_coefficients'],
                          arena_bytes=result['arena_bytes'], seed_range=result['seed_range'])))


if __name__ == '__main__':
    main()
