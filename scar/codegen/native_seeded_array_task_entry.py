"""Lower an independent task-family schedule using the existing native array body."""

import argparse
import json
from pathlib import Path
import re

from scar.codegen import native_array_entry as entry, native_array_tape, native_seeded_array_entry as seeded
from scar.codegen import native_controller_argv as argv_backend
from scar.ir.execution_recipe import read_artifact


SCHEMA = 'scar.native-seeded-array-task-entry'
_TEMPLATE = Path(__file__).with_suffix('.cpp.in')
_DISPATCH = Path(__file__).with_name('native_ordered_dispatch_v1.h.in')
_OPENMP = Path(__file__).with_name('native_ordered_openmp_v1.h.in')
_MODE_SERIAL = 'IN_PROCESS_SERIAL'
_MODE_PROCESSES = 'ORDERED_LOCAL_PROCESSES'
_MODE_OPENMP = 'ORDERED_OPENMP'
_MODES = (_MODE_SERIAL, _MODE_PROCESSES, _MODE_OPENMP)


def _need(condition, reason):
    if not condition:
        raise ValueError('native array task entry: ' + reason)


def _reports(recipe, prior, family):
    options, fields = family['input_options'], family['output_fields']
    memory = read_artifact(prior['memory'])
    profile, = (row for row in recipe['profiles'] if row['tape_sha256'] == memory['tape']['sha256'])
    routed = seeded._report(recipe, profile, prior['seed_option'], prior['round_option'])
    document = profile['report']
    results = fields['results']
    [worker] = document[results]
    outer_rows, worker_rows = {}, {}
    for row in routed['report_bindings']:
        path, role = tuple(row['path']), row['role']
        if path[:2] == (results, 0):
            worker_rows[path[2:]] = role
        else:
            outer_rows[path] = role
    _need(worker_rows[(fields['effective_seed'],)] == 'runtime_seed'
          and worker_rows[(fields['worker_round_count'],)] == 'runtime_round_count'
          and outer_rows[(fields['pid_list'],)] == 'pid_list'
          and worker[fields['ordinal']] == 0 and worker_rows.get((fields['ordinal'],)) is None,
          'task-specific report bindings disagree with the array entry contract')
    worker_rows[(fields['ordinal'],)] = 'ordinal'
    worker_source = entry._report_writer(worker, worker_rows, initial_depth=2, extra={
        'runtime_seed': lambda _: ['    out << parent.seed + task;'],
        'runtime_round_count': lambda _: ['    out << parent.round_count;'],
        'ordinal': lambda _: ['    out << task;'],
        'tolerance': lambda _: ['    binary64(out, parent.tolerance);'],
    })
    terms = read_artifact(recipe['source_contract'])['report']
    parameter_root = terms['parameters']['field']
    outer_rows.update({(parameter_root, options['task_count']): 'task_count',
                       (parameter_root, options['worker_limit']): 'worker_limit',
                       (results,): 'task_results'})
    outer_source = entry._report_writer(document, outer_rows, extra={
        'task_count': lambda _: ['    out << result.task_count;'],
        'worker_limit': lambda _: ['    out << result.worker_limit;'],
        'pid_list': lambda depth: [f'    write_pids(out, result, pretty, {depth});'],
        'task_results': lambda depth: [f'    write_records(out, result, pretty, {depth});'],
    })
    return profile, worker_source, outer_source


def _cpu_plan(choice, scope):
    mode = choice['mode']
    cpus, workers, owner = (choice[key] for key in ('launch_cpu_ids', 'worker_cpu_ids', 'owner_cpu_id'))
    _need(cpus and len(set(cpus)) == len(cpus) and owner in cpus
          and all(type(cpu) is int and 0 <= cpu < 1024 for cpu in cpus)
          and len(set(workers)) == len(workers) and set(workers) <= set(cpus),
          'invalid owner/worker CPU placement')
    parallel_shape = (1 < choice['worker_count'] == len(workers) <= min(scope['task_count'], scope['requested_workers']))
    _need((mode == _MODE_SERIAL and choice['worker_count'] == 1 and not workers and choice['window'] == 0)
          or (mode == _MODE_PROCESSES and parallel_shape and owner not in workers and choice['window'] == len(workers))
          or (mode == _MODE_OPENMP and parallel_shape and workers[0] == owner and choice['window'] == len(workers)),
          'CPU placement does not meet the fixed service shape')
    mem = choice['memory']
    _need(mem['request_bytes'] == 16 and mem['result_bytes'] == 56
          and mem['retained_result_payload_bytes'] == 56 * scope['task_count']
          and 0 < mem['coordinator_address_space_limit_bytes'] < 2**63
          and 0 <= mem['worker_address_space_limit_bytes'] < 2**63
          and 0 <= mem['payload_budget_bytes'] < 2**63, 'invalid wire or address-space budget')
    if mode == _MODE_OPENMP:
        _need(mem['payload_budget_bytes'] == 0 and mem['thread_stack_bytes'] == 8 << 20
              and mem['retained_status_bytes'] == 4 * scope['task_count']
              and mem['required_headroom_bytes'] == mem['coordinator_address_space_limit_bytes'],
              'OpenMP requires the qualified shared address space and private thread stacks')
    elif mode == _MODE_PROCESSES:
        _need(mem['worker_address_space_limit_bytes'] > 0 and mem['payload_budget_bytes'] >= choice['window'] * 72,
              'process plan cannot bound the worker or transport memory')
    return dict(LAUNCH_CPUS=', '.join(map(str, cpus)), WORKER_CPUS=', '.join(map(str, workers)), WIDTH=len(workers),
        OWNER_CPU=owner, COORDINATOR_LIMIT=mem['coordinator_address_space_limit_bytes'],
        USE_PROCESS_TEAM=str(mode == _MODE_PROCESSES).lower(), WINDOW=choice['window'],
        CHILD_LIMIT=mem['worker_address_space_limit_bytes'], TRANSPORT_BUDGET=mem['payload_budget_bytes'])


def _openmp_values(contract):
    # The tokens only appear in the preprocessor's OpenMP branch.
    inactive = dict.fromkeys(('BLAS_PARALLEL_SYMBOL', 'BLAS_THREADS_SYMBOL', 'BLAS_CORE_SYMBOL',
                              'BLAS_PARALLEL_VALUE', 'BLAS_THREADS_VALUE', 'BLAS_CORE_VALUE'), '0')
    if contract is None:
        return inactive
    guards = contract['runtime_guards']
    return dict(BLAS_CORE_VALUE=argv_backend._quote(guards['core_value']),
                **{f'BLAS_{kind.upper()}_SYMBOL': _provider_symbol(guards[f'{kind}_symbol'])
                   for kind in ('parallel', 'threads', 'core')},
                **{f'BLAS_{kind.upper()}_VALUE': int(guards[f'{kind}_value']) for kind in ('parallel', 'threads')})


def _provider_symbol(symbol):
    _need(type(symbol) is str and re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', symbol), 'invalid provider symbol')
    return symbol


def _render(recipe, prior, family, choice, scope, thread_contract=None):
    profile, worker_report, outer_report = _reports(recipe, prior, family)
    options = family['input_options']
    _need(len(set(options.values())) == 4 and options['seed'] == prior['seed_option']
          and options['round_count'] == prior['round_option'], 'task family and loop ingress differ')
    _need(family['ordinal_range'] == dict(start=0, step=1)
          and family['effective_seed']['operation'] == 'ADD'
          and family['effective_seed']['domain'] == prior['seed_range']
          and prior['seed_range']['minimum'] == 0,
          'unqualified ordinal or effective seed domain')
    task_count, requested, rounds = (scope[name] for name in ('task_count', 'requested_workers', 'round_count'))
    _need(all(type(value) is int and 1 <= value < 2**63 for value in (task_count, requested, rounds))
          and requested in recipe['options'][options['worker_limit']]['choices']
          and task_count - 1 <= prior['seed_range']['maximum'], 'invalid original task profile')
    members = {name: read_artifact(reference, json_document=False) for name, reference in prior['files'].items()
               if name != prior['binary'] and name != 'CMakeLists.txt' and not name.startswith('provider/objects/')}
    argv, indices = entry._argv_header(recipe['options'], recipe['allow_abbrev'])
    fixed = dict(profile['parameters'])
    fixed.pop(options['seed'])
    fields = dict(task_count=task_count, round_count=rounds, worker_limit=requested)
    fixed.update({options[key]: value for key, value in fields.items()})
    guards = [f'values[{indices[key]}].text == {argv_backend._quote(entry._text(value))}'
              for key, value in fixed.items()]
    upper = prior['seed_range']['maximum'] - (task_count - 1)
    ingress = '\n'.join([entry._uint_ingress(indices[options['seed']], 'seed', f'result.seed > {upper}ULL'),
                        *(f'        result.{key} = {value}ULL;' for key, value in fields.items())])
    values = dict(_cpu_plan(choice, scope), **_openmp_values(thread_contract),
                  STATIC_SCOPE=' &&\n             '.join(guards), INGRESS=ingress,
                  DEFAULT_TOLERANCE=profile['default_tolerance'].hex(),
                  TOLERANCE_INDEX=indices[recipe['tolerance_option']], OUTPUT_INDEX=indices[recipe['output_option']],
                  WORKER_REPORT=worker_report, OUTER_REPORT=outer_report)
    members.update({'include/scar_generated/array_entry_argv.hpp': argv, entry._IO_MEMBER: entry._IO.read_bytes(),
        'include/scar_generated/native_ordered_dispatch.hpp': _DISPATCH.read_bytes(),
        'src/main.cpp': entry._substitute(_TEMPLATE.read_text(), values).encode()})
    if thread_contract is not None:
        members['include/scar_generated/native_ordered_openmp.hpp'] = _OPENMP.read_bytes()
    return profile, members


def _schedule(path):
    from scar.optimization import independent_array_task_schedule, independent_array_task_team_schedule

    schema = json.loads(Path(path).read_bytes())['schema']
    for planner in (independent_array_task_schedule, independent_array_task_team_schedule):
        if schema == planner.SCHEMA:
            ref = entry._ref(path, schema)
            return ref, planner.check(ref['uri'])
    raise ValueError('native array task entry: unsupported schedule schema')


def export(schedule_path, out, *, serial_control=False, execution_control=None, worker_count=None):
    from scar.analysis import native_array_entry_recipe
    from scar.codegen import native_numpy_random, native_numpy_random_coefficients
    ref, plan = _schedule(schedule_path)
    selected = plan['decision']['selected']
    _need(selected is not None, 'task cost evidence has no qualified choice')
    _need(not (serial_control and execution_control is not None) and (worker_count is None or execution_control in _MODES),
          'one explicit control alternative required')
    control_mode = _MODE_SERIAL if serial_control else execution_control
    if control_mode is not None:
        alternatives = [row for row in plan['alternatives'] if row['mode'] == control_mode
                        and (worker_count is None or row['worker_count'] == worker_count)
                        and not row['capacity_reasons'] and row.get('cost') is not None]
        _need(len(alternatives) == 1, 'control alternative requires one feasible priced width')
        selected, = alternatives
    # Both schedule verifiers replay source legality before returning these references.
    family = read_artifact(plan['family'])
    prior = read_artifact(plan['native'])
    recipe = native_array_entry_recipe.check(prior['recipe']['uri'])
    _need(prior['schema'] == seeded.LOOP_SCHEMA and family['loop'] == prior['loop']
          and family['source_contract'] == recipe['source_contract']
          and recipe == read_artifact(prior['recipe']), 'task plan does not close over the bound array loop')
    thread_ref = None
    thread_contract = None
    if selected['mode'] == _MODE_OPENMP:
        thread_ref = plan['thread_contract']
        thread_contract = read_artifact(thread_ref)
    profile, members = _render(recipe, prior, family, selected, plan['scope'], thread_contract)
    _, libraries, library, normal_provider = native_array_tape._normalize(recipe['normalize'])
    stock = read_artifact(profile['package'])['stock_provider']
    random = read_artifact(read_artifact(prior['binding'])['random'])['provider']
    result = dict(schema=SCHEMA, schema_version=1, native=plan['native'], family=plan['family'], schedule=ref,
                  recipe=prior['recipe'], cpu_ids=selected['launch_cpu_ids'], scope=plan['scope'], decision=selected,
                  output_fields=family['output_fields'], input_options=family['input_options'],
                  original_default_retained=True, automatic_deployment=False,
                  numerical_body_reused=prior['files']['src/profile_0.cpp'], no_worker_visible_output=True,
                  explicit_checkpoint_diagnostics=True, serial_control=serial_control,
                  execution_control=control_mode, thread_contract=thread_ref,
                  launch_environment={'OMP_STACKSIZE': '8M'} if thread_ref is not None else {})
    openmp_options = dict(compile_options=('-fopenmp', '-DSCAR_ARRAY_TASK_OPENMP'),
                         link_options=('-fopenmp',),
                         runtime_dependencies={'libgomp.so.1': plan['openmp_runtime']}) if thread_ref is not None else {}
    if 'thread_evidence' in plan:
        openmp_options['compiler_reference'] = read_artifact(plan['thread_evidence'])['compiler_binary']
    return entry._build(out, members, profiles=1, library=library, libraries=libraries, stock_provider=stock,
                        normal_provider=normal_provider, report=result, random_provider=random,
                        extra_sources=(native_numpy_random.SOURCE, native_numpy_random_coefficients.SOURCE),
                        archives=(native_numpy_random.ARCHIVE,), **openmp_options)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--schedule', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--serial-control', action='store_true', help='publish the feasible serial alternative for A/B')
    parser.add_argument('--execution-control', choices=_MODES, help='publish one feasible execution alternative for A/B')
    parser.add_argument('--worker-count', type=int, help='width of the explicit execution alternative')
    args = parser.parse_args(argv)
    result = export(args.schedule, args.out, serial_control=args.serial_control,
                    execution_control=args.execution_control, worker_count=args.worker_count)
    print(json.dumps(dict(mode=result['decision']['mode'], workers=result['decision']['worker_count'],
                          binary=result['files'][result['binary']]['uri'])))


if __name__ == '__main__':
    main()
