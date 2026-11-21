"""Project a source-bound ordered task map around a qualified numerical loop."""

import argparse
import ast
import json
from pathlib import Path

from scar.analysis import seeded_array_loop as loops
from scar.analysis import seeded_array_tape_binding as seeded
from scar.analysis import source_array_entry_contract as source
from scar.ir.execution_recipe import artifact_reference, read_artifact


SCHEMA = 'scar.seeded-array-task-family'
_PROCESS_POOL = 'concurrent.futures.ProcessPoolExecutor'
_CONTEXT = 'multiprocessing.get_context'
_CLOCKS = source._VOLATILE_PROVIDERS


def _need(condition, reason):
    if not condition:
        raise ValueError('seeded array task family: ' + reason)


def _one(rows, reason):
    values = tuple(rows)
    _need(len(values) == 1, reason)
    return values[0]


def _ref(path):
    return artifact_reference(Path(path).resolve(strict=True), SCHEMA)


def _literal(node, expected):
    return type(node) is ast.Constant and type(node.value) is type(expected) and node.value == expected


def _expr(expression):
    return ast.parse(expression, mode='eval').body


def _option(contract, expression):
    namespace = contract['argv']['namespace']
    return _one((row for row in contract['argv']['options']
                 if expression == source._term(_expr(namespace + '.' + row['id']))),
                'one direct original parser option is required')


def _origin_call(node, imports, path):
    return type(node) is ast.Call and source._origin(source._name(node.func), imports) == path


def _named_stores(node, name):
    return [row for row in ast.walk(node) if type(row) is ast.Name and row.id == name
            and isinstance(row.ctx, (ast.Store, ast.Del))]


def _receipt_positions(rows):
    return {tuple(row['position']) for row in rows}


def _before_dispatch(owner, tasks, contract, scope):
    """Check that the prefix consists of the already projected argument and resource effects."""
    argv = contract['argv']
    candidates = [argv['constructor'], argv['parse'], *(row['source'] for row in argv['options']),
                  *(row['condition'] for row in argv['guards']), *(row['condition'] for row in argv['derived']),
                  *(row['original'] for row in argv['derived']),
                  *(row['source'] for row in contract['thread_environment'])]
    known = _receipt_positions(candidates)
    for statement in owner.body[:owner.body.index(tasks)]:
        if type(statement) is ast.Assign:
            binding = source._assignment(statement)
            probe = binding[1] if binding is not None else statement
        elif type(statement) is ast.Expr:
            probe = statement.value
        elif type(statement) is ast.If:
            probe = statement.test
        else:
            probe = statement
        _need(tuple(source._fact(probe, scope)['position']) in known,
              'unqualified work before the original task list')


def _dispatch(tree, owner, contract, imports):
    scope = contract['entry_function']['code_id']
    worker = contract['worker']
    branch = _one((row for row in owner.body if type(row) is ast.If
                   and source._fact(row.test, scope) == worker['dispatch']['selector']),
                  'sequential dispatch clause is missing')
    selector = branch.test
    _need(type(selector) is ast.Compare and len(selector.ops) == len(selector.comparators) == 1
          and type(selector.ops[0]) is ast.Eq and _literal(selector.comparators[0], 1),
          'dispatch must select the serial arm exactly at one worker')
    limit = _option(contract, source._term(selector.left))
    choices = [row['value'] for row in limit['choices']]
    _need(limit['kind'] == 'int' and limit['action'] == 'store' and len(set(choices)) == len(choices)
          and all(type(value) is int and value > 0 for value in choices) and 1 in choices
          and len(choices) > 1 and limit['default']['value'] in choices,
          'original worker limit is not a finite positive integer selector')
    alternate = _one(branch.orelse, 'one standard-library pool context is required')
    _need(type(alternate) is ast.With and len(alternate.items) == len(alternate.body) == 1,
          'unsupported pool scope or extra work inside the parallel arm')
    [item] = alternate.items
    pool, handle = item.context_expr, seeded._as_name(item.optional_vars)
    _need(type(pool) is ast.Call and _origin_call(pool, imports, _PROCESS_POOL)
          and handle is not None and not pool.args and len(pool.keywords) == 2,
          'exact standard-library process pool constructor required')
    constructor = source._kwargs(pool)
    _need(set(constructor) == {'max_workers', 'mp_context'}
          and source._term(constructor['max_workers']) == source._term(selector.left),
          'original worker cap differs between selector and process pool')
    context = constructor['mp_context']
    _need(_origin_call(context, imports, _CONTEXT) and not context.keywords
          and len(context.args) == 1 and _literal(context.args[0], 'spawn'),
          'requires source-explicit fresh spawn context')
    serial = source._assignment(branch.body[0])
    _need(serial is not None, 'serial result binding is missing')
    calls = serial[1]
    [iterator] = calls.generators
    tasks = seeded._as_name(iterator.iter)
    _need(tasks is not None and seeded._as_name(calls.elt.func) == worker['method'],
          'original map worker and task list differ')
    parallel = source._assignment(alternate.body[0])
    _need(parallel is not None and parallel[0] == serial[0] == worker['result_slot']
          and source._call(parallel[1], 'list') and not parallel[1].keywords and len(parallel[1].args) == 1,
          'parallel map must realize the complete ordered result list')
    mapped = parallel[1].args[0]
    _need(source._call(mapped, handle + '.map') and [seeded._as_name(arg) for arg in mapped.args]
          == [worker['method'], tasks] and len(mapped.keywords) == 1
          and mapped.keywords[0].arg == 'chunksize' and _literal(mapped.keywords[0].value, 1),
          'parallel and serial arms must map the same function over the same task list')
    task_assignment = _one((row for row in owner.body if (assignment := source._assignment(row))
                            and assignment[0] == tasks), 'one original task list definition required')
    task_list = source._assignment(task_assignment)[1]
    _need(type(task_list) is ast.ListComp and len(task_list.generators) == 1,
          'one finite task-family range required')
    [generation] = task_list.generators
    _need(type(task_list.elt) is ast.Tuple and source._call(generation.iter, 'range')
          and len(generation.iter.args) == 1 and not generation.iter.keywords
          and seeded._as_name(generation.target) == worker['ordinal']
          and not generation.is_async and not generation.ifs,
          'task ordinal must enumerate the original zero-based range')
    count = _option(contract, source._term(generation.iter.args[0]))
    _need(count['kind'] == 'int' and count['action'] == 'store' and not count['choices'],
          'task cardinality must come from a positive integer option')
    minimum, positive = loops._source_guard(contract, tree, count)
    _need(minimum == 1 and count['default']['value'] >= minimum,
          'one or more tasks required by the original entry')
    _need(len(_named_stores(owner, tasks)) == 1 and len(_named_stores(owner, worker['result_slot'])) == 2,
          'task list or result collection is rebound')
    _before_dispatch(owner, task_assignment, contract, scope)
    return dict(task_count=count, worker_limit=limit), (task_assignment, branch), dict(
        provider=_PROCESS_POOL, context_provider=_CONTEXT, start_method='spawn', chunksize=1,
        serial_worker_value=1, serial_selector=source._fact(selector, scope), pool=source._fact(pool, scope),
        serial=source._fact(calls, scope), parallel=source._fact(mapped, scope),
        dispatch=source._fact(branch, scope), task_list=source._fact(task_list, scope),
        positive_task_count_guard=positive)


def _target_formals(contract, seed, rounds, worker_code):
    worker = contract['worker']
    namespace = contract['argv']['namespace']
    seed_formal = seeded._formal(worker, seed['id'], namespace)
    round_formal = seeded._formal(worker, rounds['id'], namespace)
    ordinal = _one((row['formal'] for row in worker['actuals']
                    if row['source']['expression'] == source._term(_expr(worker['ordinal']))),
                   'the source ordinal is not an exact input to the worker')
    ordinal_actual = source._term(_expr(worker['ordinal']))
    allowed = {row['slot'] for row in contract['argv']['derived']}
    scalar = {row['id'] for row in contract['argv']['options'] if row['kind'] in ('int', 'float64', 'str', 'bool')}
    allowed |= {namespace + '.' + name for name in scalar}
    _need(len(set(worker['formals'])) == len(worker['formals']) and len({ordinal, seed_formal, round_formal}) == 3,
          'task formal bindings collide')
    for row in worker['actuals']:
        expression = row['source']['expression']
        if expression != ordinal_actual:
            _need(any(expression == source._term(_expr(value)) for value in allowed),
                  'tasks can carry only original immutable scalar inputs plus the ordinal')
        _need(len(_named_stores(worker_code, row['formal'])) == 1, 'an immutable task input is rebound')
    return seed_formal, ordinal, round_formal


def _row(contract, kind, predicate, reason):
    return _one((row for row in contract['report'][kind] if predicate(row)), reason)


def _field_expr(rows, name):
    return _one((row['source']['expression'] for row in rows if row['name'] == name),
                'source report field is missing')


def _keyed_collection(fact, source_list, *, reducer, container):
    row = _expr_from_fact(fact)
    if not (source._call(row, reducer) and len(row.args) == 1 and not row.keywords
            and type(row.args[0]) is container):
        return None
    comp = row.args[0]
    if len(comp.generators) != 1:
        return None
    [binding] = comp.generators
    value = comp.elt
    if not (seeded._as_name(binding.iter) == source_list and seeded._as_name(binding.target) is not None
            and not binding.ifs and not binding.is_async and type(value) is ast.Subscript
            and seeded._as_name(value.value) == binding.target.id and type(value.slice) is ast.Constant
            and type(value.slice.value) is str):
        return None
    return value.slice.value


def _expr_from_fact(fact):
    """Round-trip an expression term without relying on original spelling."""
    def convert(term):
        if type(term) is list:
            return [convert(item) for item in term]
        if type(term) is not dict:
            return term
        if term['kind'] == 'Constant':
            return ast.Constant(value=term['literal']['value'])
        return getattr(ast, term['kind'])(**{name: convert(value) for name, value in term['fields'].items()})
    return convert(fact)


def _report(contract, owner, imports, dispatch_nodes, formal_names):
    scope = contract['entry_function']['code_id']
    task, branch = dispatch_nodes
    seed, ordinal, rounds = formal_names
    report = contract['report']
    root, child = report['root_fields'], report['worker_fields']

    def exact(field, expression):
        return _row(contract, field, lambda row: row['source']['expression'] == source._term(_expr(expression)),
                    'report has no unique field for ' + expression)['name']

    def metric(provider):
        return _row(contract, 'worker_fields', lambda row: row['volatile_providers'] == [provider],
                    'original worker measurement or resource field is missing')['name']

    elapsed = _row(contract, 'root_fields', lambda row: row['volatile_providers'] == ['WALL_CLOCK'],
                   'one original parent elapsed field is required')
    result_slot = contract['worker']['result_slot']

    def reduce(call, container, field=None):
        rows = []
        for row in root:
            item = _keyed_collection(row['source']['expression'], result_slot, reducer=call, container=container)
            if item is not None and (field is None or item == field):
                rows.append((row['name'], item))
        result, item = _one(rows, 'expected original ordered result reduction: ' + call)
        _field_expr(child, item)
        return result, item

    worker_cpu = metric('PROCESS_CPU_CLOCK')
    pid = metric('PROCESS_ID')
    total_cpu, _ = reduce('sum', ast.GeneratorExp, worker_cpu)
    pids, _ = reduce('sorted', ast.SetComp, pid)
    max_error, norm_error = reduce('max', ast.GeneratorExp)
    after = owner.body[owner.body.index(branch) + 1:]
    between = owner.body[owner.body.index(task) + 1:owner.body.index(branch)]
    _need(len(after) == 4 and len(between) == 1, 'extra controller work around ordered report publication')
    first, end, record = (source._assignment(statement) for statement in (between[0], *after[:2]))
    _need(first is not None and end is not None and record is not None and _origin_call(first[1], imports, 'time.perf_counter')
          and not first[1].args and not first[1].keywords and end[0] == seeded._as_name(_expr_from_fact(elapsed['source']['expression']))
          and type(end[1]) is ast.BinOp and type(end[1].op) is ast.Sub and seeded._as_name(end[1].right) == first[0]
          and _origin_call(end[1].left, imports, 'time.perf_counter') and not end[1].left.args and not end[1].left.keywords,
          'parent timing must bound the task group according to the original entry')
    _need(source._call(record[1], 'dict') and type(after[2]) is ast.If and len(after[2].body) == 1
          and type(after[2].body[0]) is ast.Expr and type(after[3]) is ast.Expr
          and source._fact(after[2].body[0].value, scope) == report['path_output']['source']
          and source._fact(after[3].value, scope) == report['stdout']['source'],
          'original file and stdout publication must follow completion of the task list')
    for row in root:
        expr = _expr_from_fact(row['source']['expression'])
        if row['name'] in (pids, max_error, total_cpu, report['results_field']):
            continue
        _need(not any(type(name) is ast.Name and name.id == result_slot for name in ast.walk(expr)),
              'a result-dependent root field has no admitted publication reduction')
    return dict(results=report['results_field'], pid_list=pids,
                ordinal=exact('worker_fields', ordinal), effective_seed=exact('worker_fields', seed + ' + ' + ordinal),
                worker_round_count=exact('worker_fields', rounds), elapsed=elapsed['name'],
                worker_cpu_total=total_cpu, max_norm_error=max_error,
                worker_cpu_seconds=worker_cpu, worker_wall_seconds=metric('WALL_CLOCK'),
                worker_peak_rss=metric('PROCESS_RESOURCE_USAGE'), worker_pid=pid, worker_norm_error=norm_error)


_BUILTIN_OPERATIONS = frozenset(('abs', 'dict', 'float', 'getattr', 'range', 'str', 'sum', 'ValueError'))


def _module_imports(tree, owner, worker):
    imported = {}
    known_defs = {owner.name, worker.name}
    _need(len(known_defs) == 2 and not owner.decorator_list and not worker.decorator_list,
          'entry and worker must be ordinary unadorned definitions')
    for index, statement in enumerate(tree.body):
        if type(statement) in (ast.Import, ast.ImportFrom):
            if type(statement) is ast.ImportFrom:
                _need(not statement.level and statement.module and all(item.name != '*' for item in statement.names),
                      'indirect or wildcard root import')
            for item in statement.names:
                name = item.asname or item.name
                _need(name not in imported and name not in known_defs, 'root import binding changes')
                imported[name] = source._imports(ast.Module(body=[statement]))[name]
        elif type(statement) is ast.FunctionDef:
            args = statement.args
            _need(statement in (owner, worker) and not statement.returns and not getattr(statement, 'type_params', ())
                  and not args.posonlyargs and not args.kwonlyargs and not args.vararg and not args.kwarg
                  and not args.defaults and all(operand.annotation is None for operand in args.args),
                  'unadmitted top-level definition or implicit definition effect')
        elif type(statement) is ast.If:
            _need(index == len(tree.body) - 1, 'entry guard must be the final module statement')
        else:
            _need(index == 0 and type(statement) is ast.Expr and type(statement.value) is ast.Constant
                  and type(statement.value.value) is str, 'unadmitted module effect at task start')
    builtins = _BUILTIN_OPERATIONS | {'list', 'min', 'max', 'sorted', 'vars', 'print', 'int', 'str'}
    _need(not (set(imported) | known_defs) & builtins, 'standard builtin binding is shadowed')
    return imported


def _provider_roots(binding):
    factory = _expr_from_fact(binding['construction']['seed_factory']['expression'])
    random_name = source._name(factory.func)
    _need(random_name is not None and '.' in random_name, 'RNG must originate in a scoped numerical provider')
    observations = []
    receivers = set()
    for member in binding['tapes']:
        tape = read_artifact(member['tape'])
        observation = read_artifact(tape['observation'])
        observations.append(dict(tape=member['tape'], observation=tape['observation']))
        endpoint = observation['endpoints']['call']
        receivers.add((endpoint['module'].split('.')[0], endpoint['class_name']))
    module, kind = _one(receivers, 'worker receiver provider changes across existing evidence')
    return random_name.split('.')[0], module, kind, observations


def _worker_scope(contract, binding, runtime, worker, imports, formals, output):
    scope = contract['worker']['code_id']
    seed, ordinal, _ = formals
    random_binding, receiver_module, receiver_type, observations = _provider_roots(binding)
    local_imports = {}
    for statement in ast.walk(worker):
        _need(type(statement) not in (ast.Global, ast.Nonlocal, ast.Try, getattr(ast, 'TryStar', ast.Try), ast.Lambda,
                                     ast.AsyncFor, ast.AsyncWith, ast.With, ast.Yield, ast.YieldFrom,
                                     ast.ImportFrom, ast.Delete, ast.NamedExpr, ast.Await)
              and (type(statement) not in (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef) or statement is worker),
              'worker crosses the private local-state envelope')
        if type(statement) is ast.Import:
            _need(statement in worker.body, 'provider imports must be direct worker initializers')
            for item in statement.names:
                name = item.asname or item.name
                _need(name not in local_imports and name not in imports and name not in _BUILTIN_OPERATIONS,
                      'qualified provider import is shadowed')
                local_imports[name] = item.name
        if isinstance(getattr(statement, 'ctx', None), (ast.Store, ast.Del)):
            _need(type(statement) in (ast.Name, ast.Tuple), 'worker writes through a potential shared alias')
    _need(random_binding in local_imports and set(local_imports.values()) == {local_imports[random_binding], receiver_module}
          and not any(_named_stores(worker, name) for name in (*local_imports, *imports, *_BUILTIN_OPERATIONS)),
          'worker providers must be stable local imports from the bound numerical route')
    private = {row[0]: row[1] for statement in worker.body if (row := source._assignment(statement))
               and type(row[1]) is ast.Call and (dotted := source._name(row[1].func))
               and dotted.split('.')[0] in local_imports}
    call = _one((node for node in ast.walk(worker) if type(node) is ast.Call
                 and seeded._position(node) == runtime['source_loop']['normal_call']['position']),
                'worker recurrence call does not exist')
    receiver = seeded._as_name(call.func.value) if type(call.func) is ast.Attribute else None
    _need(receiver in private and len(_named_stores(worker, receiver)) == 1
          and source._origin(source._name(private[receiver].func), local_imports) == receiver_module + '.' + receiver_type,
          'stateful numerical receiver is not constructed afresh inside each task')
    rng = _one((node for node in ast.walk(worker) if type(node) is ast.Call
                and source._fact(node, scope) == binding['construction']['seed_factory']),
               'the per-task random constructor differs')
    constructor_seed = _one((row.value for row in private[receiver].keywords
                             if source._term(row.value) == source._term(_expr(seed + ' + ' + ordinal))),
                            'numerical receiver lacks the same source-derived local RNG seed')
    returned = _one((node.value for node in worker.body if type(node) is ast.Return), 'one successful worker record required')
    report_terms = source._kwargs(returned)
    permitted_ports = (rng.args[0], constructor_seed, report_terms[output['effective_seed']], report_terms[output['ordinal']])
    admitted_seed_loads = {node for part in permitted_ports for node in ast.walk(part) if type(node) is ast.Name}
    _need(all(node in admitted_seed_loads for node in ast.walk(worker) if type(node) is ast.Name
              and isinstance(node.ctx, ast.Load) and node.id in (seed, ordinal)),
          'seed and ordinal reach an unqualified worker operation')
    carrier = runtime['source_loop']['carrier']['name']
    _need(not any(seeded._as_name(term) == carrier for term in report_terms.values()),
          'the task returns its mutable array state')
    provider_calls, metadata_calls = [], []
    for node in ast.walk(worker):
        if type(node) is not ast.Call:
            continue
        _need(not any(row.arg is None for row in node.keywords) and not any(type(arg) is ast.Starred for arg in node.args),
              'variable call argument expansion crosses the worker boundary')
        func = node.func.func if type(node.func) is ast.Call else node.func
        name = source._name(func)
        _need(name is not None, 'unresolved worker call target')
        root = name.split('.')[0]
        if name in _BUILTIN_OPERATIONS:
            continue
        if source._origin(name, imports) in _CLOCKS:
            metadata_calls.append(source._fact(node, scope))
            continue
        _need('.' in name and (root in local_imports or (root in private and len(_named_stores(worker, root)) == 1)),
              'unqualified external call or a method on a shared value: ' + name)
        provider_calls.append(source._fact(node, scope))

    for witness in contract['witnesses']:
        original = read_artifact(witness['observation'])
        report = json.loads(read_artifact(original['stdout'], json_document=False))
        [task] = report[output['results']]
        _need(all(type(value) in (int, float, str, bool, type(None)) for value in task.values()),
              'observed worker emits state beyond its scalar report')
    return dict(kind='PER_TASK_PRIVATE', loop_source=runtime['source_loop'],
                private_construction=source._fact(private[receiver], scope), random_factory=source._fact(rng, scope),
                numerical_evidence=observations, provider_calls=provider_calls, metadata_calls=metadata_calls,
                no_application_level_shared_writes=True, no_state_payload_published_on_observed_path=True,
                assumptions=['PINNED_NUMERICAL_PROVIDERS', 'SUCCESS_CONTRACT_ON_ADMITTED_INPUTS'],
                mandatory_validation=['ORIGINAL_MULTI_TASK_VALUE_AND_EFFECT_REPORT', 'INDEPENDENT_FULL_TASK_STATES'])


def project(loop_path):
    """Bind the qualified inner loop to the source-proven outer ordered task family."""
    loop_ref = artifact_reference(Path(loop_path).resolve(strict=True), loops.SCHEMA)
    runtime = loops.check(loop_ref['uri'])
    binding = read_artifact(runtime['seed_binding'])
    source_ref = binding['source_contract']
    contract = read_artifact(source_ref)
    reference = contract['entry']
    tree, raw = seeded._ast(reference)
    owner = source._entry_function(tree)
    worker = seeded._body(tree, raw, reference, contract['worker']['code_id'])
    _need(runtime['source_loop']['source'] == reference and runtime['source_loop']['code_id'] == contract['worker']['code_id'],
          'inner loop and outer task family use different source worker definitions')
    imports = _module_imports(tree, owner, worker)
    options, dispatch_nodes, dispatch = _dispatch(tree, owner, contract, imports)
    for role, name in (('seed', binding['construction']['seed_option']), ('round_count', runtime['option']['id'])):
        options[role] = _one((row for row in contract['argv']['options'] if row['id'] == name),
                             'capture-bound option is absent from the original parser')
    _need(len({row['id'] for row in options.values()}) == len(options)
          and all(binding['fixed_options'][options[key]['id']] == 1 for key in ('task_count', 'worker_limit')),
          'task map must extend a sealed one-task one-worker profile')
    formals = _target_formals(contract, options['seed'], options['round_count'], worker)
    fields = _report(contract, owner, imports, dispatch_nodes, formals)
    state = dict(_worker_scope(contract, binding, runtime, worker, imports, formals, fields), loop=loop_ref)
    environment = contract['thread_environment']
    _need(environment and all(row['value'] == '1' for row in environment),
          'original numerical provider parallelism lacks a separate single-thread contract')
    return dict(schema=SCHEMA, schema_version=1, loop=loop_ref, source_contract=source_ref, entry=reference,
                input_options={key: row['id'] for key, row in options.items()}, option_contracts=options,
                output_fields=fields, ordinal_range=dict(start=0, step=1),
                effective_seed=dict(operation='ADD', domain=binding['seed_domain'],
                                    guard='BASE_PLUS_LAST_ORDINAL_IN_DOMAIN'),
                worker_formals=contract['worker']['actuals'], dispatch=dispatch, state=state,
                publication=dict(result_order='ASCENDING_ORDINAL', pid_order='SORTED_UNIQUE', after_all_workers=True,
                                 path_output=contract['report']['path_output'], stdout=contract['report']['stdout']),
                thread_environment=environment, deployment_authorized=False)


def check(path):
    stored = read_artifact(_ref(path))
    _need(stored == project(stored['loop']['uri']), 'stored task family differs from original source replay')
    return stored


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_mutually_exclusive_group(required=True)
    commands.add_argument('--loop')
    commands.add_argument('--check')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if args.check and args.out is None:
        result = check(args.check)
    elif args.loop and args.out is not None:
        result = project(args.loop)
        with args.out.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
    else:
        parser.error('provide --check RECEIPT or --loop LOOP --out RECEIPT')
    print(json.dumps(dict(input_options=result['input_options'], output_fields=result['output_fields'])))


if __name__ == '__main__':
    main()
