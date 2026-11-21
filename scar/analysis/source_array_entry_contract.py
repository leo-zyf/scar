"""Project a source argparse entry and its JSON publication contract."""

import argparse
import ast
import json
import math
import operator
from pathlib import Path
import struct
from types import CodeType

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.trace.code_snapshot import code_key
from scar.trace.numpy_partition_entry_validation import normalize_report


SCHEMA = 'scar.source-array-entry-contract'
ORIGINAL = 'scar.array-entry-original-process'
_SOURCE = 'scar.python-source'
_TYPES = {'int': int, 'float64': float, 'str': str, 'path': Path}
_VOLATILE_PROVIDERS = {
    'os.getpid': 'PROCESS_ID', 'time.perf_counter': 'WALL_CLOCK',
    'time.process_time': 'PROCESS_CPU_CLOCK', 'resource.getrusage': 'PROCESS_RESOURCE_USAGE',
}


def _require(condition, text):
    if not condition:
        raise ValueError('array entry contract: ' + text)


def _one(rows, text):
    rows = tuple(rows)
    _require(len(rows) == 1, text)
    return rows[0]


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _typed(value):
    if value is None:
        return dict(kind='null', value=None)
    if type(value) is float:
        _require(math.isfinite(value), 'finite scalar required')
        return dict(kind='float64', value=value, binary64_be=struct.pack('>d', value).hex())
    kind = {bool: 'bool', int: 'int', str: 'str'}.get(type(value))
    _require(kind is not None, 'builtin literal required')
    return dict(kind=kind, value=value)


def _term(node):
    if type(node) is ast.Constant:
        return dict(kind='Constant', literal=_typed(node.value))
    if isinstance(node, ast.AST):
        return dict(kind=type(node).__name__, fields={key: _term(value) for key, value in ast.iter_fields(node)})
    if type(node) is list:
        return [_term(value) for value in node]
    return node


def _fact(node, scope):
    return dict(code_id=scope, position=[node.lineno, node.end_lineno, node.col_offset, node.end_col_offset],
                expression=_term(node))


def _name(node):
    if type(node) is ast.Name:
        return node.id
    if type(node) is ast.Attribute and (base := _name(node.value)) is not None:
        return base + '.' + node.attr
    return None


def _assignment(node):
    if type(node) is ast.Assign and len(node.targets) == 1 and type(node.targets[0]) is ast.Name:
        return node.targets[0].id, node.value
    return None


def _call(node, function):
    return type(node) is ast.Call and _name(node.func) == function


def _kwargs(node):
    return {kw.arg: kw.value for kw in node.keywords}


def _imports(tree):
    mapped = {}
    for node in tree.body:
        if type(node) is ast.Import:
            mapped.update({alias.asname or alias.name: alias.name for alias in node.names})
        elif type(node) is ast.ImportFrom and not node.level:
            mapped.update({alias.asname or alias.name: node.module + '.' + alias.name for alias in node.names})
    return mapped


def _origin(dotted, imports):
    if dotted is None:
        return None
    head, dot, tail = dotted.partition('.')
    return imports.get(head, head) + dot + tail


def _entry_function(tree):
    guards = (node for node in tree.body if type(node) is ast.If and type(node.test) is ast.Compare
              and _name(node.test.left) == '__name__' and len(node.test.ops) == 1
              and type(node.test.ops[0]) is ast.Eq and len(node.test.comparators) == 1
              and type(node.test.comparators[0]) is ast.Constant
              and node.test.comparators[0].value == '__main__')
    guard = _one(guards, 'unique module entry guard required')
    statement = _one(guard.body, 'one direct source entry call required')
    _require(not guard.orelse and type(statement) is ast.Expr and type(statement.value) is ast.Call
             and type(statement.value.func) is ast.Name and not statement.value.args and not statement.value.keywords,
             'ordinary guarded entry function required')
    return _one((node for node in tree.body if type(node) is ast.FunctionDef and node.name == statement.value.func.id),
                'unique declared entry function required')


def _argv(tree, fn, scope, imports):
    parser, constructor = _one((item for row in fn.body if (item := _assignment(row))
                                and type(item[1]) is ast.Call
                                and _origin(_name(item[1].func), imports) == 'argparse.ArgumentParser'),
                               'one standard argparse parser required')
    settings = _kwargs(constructor)
    _require(not constructor.args and set(settings) <= {'description'}, 'unsupported argparse constructor options')
    description = settings.get('description')
    description = ast.get_docstring(tree, clean=False) if _name(description) == '__doc__' else (
        ast.literal_eval(description) if description is not None else None)
    _require(description is None or type(description) is str, 'literal parser description required')
    namespace, parse = _one((item for row in fn.body if (item := _assignment(row))
                             and _call(item[1], parser + '.parse_args')), 'one strict parse_args result required')
    _require(not parse.args and not parse.keywords, 'source parser must use original process argv')
    options = []
    for row in fn.body:
        if type(row) is not ast.Expr or not _call(row.value, parser + '.add_argument'):
            continue
        declaration = row.value
        flags = [ast.literal_eval(flag) for flag in declaration.args]
        name = _one((flag for flag in flags if type(flag) is str and flag.startswith('--')),
                    'one original long option required')[2:].replace('-', '_')
        keywords = _kwargs(declaration)
        _require(set(keywords) <= {'type','default','choices','action','help'} and name.isidentifier()
                 and len({kw.arg for kw in declaration.keywords}) == len(declaration.keywords)
                 and all(type(flag) is str and flag.startswith('-') for flag in flags), 'unsupported option declaration')
        action = ast.literal_eval(keywords['action']) if 'action' in keywords else 'store'
        kind = {'int':'int', 'float':'float64', 'str':'str', 'pathlib.Path':'path'}.get(
            _origin(_name(keywords.get('type')), imports), 'str' if 'type' not in keywords else None)
        if action == 'store_true':
            _require('type' not in keywords, 'boolean parser operation differs')
            kind = 'bool'
        _require(action in ('store','store_true') and kind is not None, 'unsupported option type or action')
        default = ast.literal_eval(keywords['default']) if 'default' in keywords else (False if kind=='bool' else None)
        choices = ast.literal_eval(keywords['choices']) if 'choices' in keywords else ()
        _require(type(choices) in (tuple,list), 'literal option choices required')
        options.append(dict(id=name, flags=flags, kind=kind, action=action, default=_typed(default),
                            choices=[_typed(choice) for choice in choices], source=_fact(declaration,scope)))
    _require(options and len({row['id'] for row in options}) == len(options), 'ambiguous parser options')
    guards, derived, guard_nodes, derived_nodes = [], [], [], []
    reads = {key: value for row in fn.body if (item := _assignment(row))
             for key, value in (item,) if type(value) is ast.Attribute and _name(value.value) == namespace}
    for row in fn.body:
        if type(row) is not ast.If or row.orelse or len(row.body) != 1:
            continue
        action_stmt = row.body[0]
        if type(action_stmt) is ast.Expr and _call(action_stmt.value, parser + '.error'):
            [message] = action_stmt.value.args
            _require(not action_stmt.value.keywords and type(message) is ast.Constant and type(message.value) is str,
                     'parser guard must have original literal error message')
            guards.append(dict(condition=_fact(row.test,scope), message=message.value))
            guard_nodes.append((row, message.value))
        elif (action_assign := _assignment(action_stmt)) and action_assign[0] in reads:
            slot, value = action_assign
            derived.append(dict(slot=slot, original=_fact(reads[slot],scope),
                                condition=_fact(row.test,scope), value=_fact(value,scope)))
            derived_nodes.append((row,slot,reads[slot],value))
    provider = _ref(argparse.__file__, _SOURCE)
    return (dict(provider=provider, parse_method='parse_args', namespace=namespace, options=options,
                 constructor=_fact(constructor,scope), parse=_fact(parse,scope), description=description,
                 guards=guards, derived=derived), guard_nodes, derived_nodes)


def _worker(tree, owner, scope, keys):
    variants = []
    for row in owner.body:
        if type(row) is not ast.If or len(row.body) != 1:
            continue
        assigned = _assignment(row.body[0])
        if assigned and type(assigned[1]) is ast.ListComp and type(assigned[1].elt) is ast.Call:
            variants.append((row, *assigned))
    branch, results, calls = _one(variants, 'one direct sequential worker comprehension required')
    call = calls.elt
    [item] = call.args
    [iterator] = calls.generators
    _require(type(call.func) is ast.Name and not call.keywords and not iterator.ifs and not iterator.is_async
             and _name(item) == _name(iterator.target) is not None, 'worker call must consume the direct iteration item')
    slots_name = _name(iterator.iter)
    tasks = _one((value for row in owner.body if (data := _assignment(row)) for name, value in (data,)
                  if name == slots_name and type(value) is ast.ListComp), 'one original task list required')
    [origin] = tasks.generators
    _require(type(tasks.elt) is ast.Tuple and _call(origin.iter, 'range') and len(origin.iter.args)==1
             and not origin.iter.keywords and not origin.ifs and not origin.is_async and _name(origin.target),
             'worker tasks must be tuples over one finite range')
    worker = _one((node for node in tree.body if type(node) is ast.FunctionDef and node.name == call.func.id),
                  'worker must be a named local definition')
    formal = _one(worker.args.args, 'one tuple-bound worker formal required')
    destruct = _one((row for row in worker.body if type(row) is ast.Assign and len(row.targets)==1
                     and type(row.targets[0]) is ast.Tuple and _name(row.value)==formal.arg),
                    'worker tuple formal must have a direct source unpack')
    formals = [_name(node) for node in destruct.targets[0].elts]
    _require(len(formals)==len(tasks.elt.elts) and None not in formals, 'worker tuple arity differs')
    function = keys[worker.name]
    call_dependencies = []
    for node in ast.walk(worker):
        if type(node) is ast.Call:
            operands = [(None, expr) for expr in node.args] + [(kw.arg,kw.value) for kw in node.keywords]
            routes = [dict(keyword=name, formals=sorted(bound), source=_fact(expr,function))
                      for name, expr in operands
                      if (bound := {child.id for child in ast.walk(expr) if type(child) is ast.Name} & set(formals))]
            if routes:
                call_dependencies.append(dict(call=_fact(node,function), actuals=routes))
    data = dict(method=worker.name, code_id=function, formal=formal.arg, formals=formals, result_slot=results,
                actuals=[dict(formal=name, source=_fact(expr,scope)) for name,expr in zip(formals,tasks.elt.elts,strict=True)],
                dispatch=dict(call=_fact(call,scope), selector=_fact(branch.test,scope),
                              alternate=[_fact(row,scope) for row in branch.orelse], alternate_admitted=False),
                task_range=_fact(origin.iter.args[0],scope), ordinal=_name(origin.target),
                call_dependencies=call_dependencies)
    return data, worker, branch.test, tasks, origin


def _report_fields(expression, scope, imports, locals_, row_fields=()):
    _require(_call(expression,'dict') and not expression.args, 'original report must declare named dict fields')
    volatile = {row['name']: row['volatile_providers'] for row in row_fields}

    def kinds(node, visited):
        mapped = []
        for item in ast.walk(node):
            category = _VOLATILE_PROVIDERS.get(_origin(_name(item),imports))
            if category is not None:
                mapped.append(category)
            if type(item) is ast.Name and item.id in locals_ and item.id not in visited:
                mapped.extend(kinds(locals_[item.id], {*visited,item.id}))
            if type(item) is ast.Subscript and type(item.slice) is ast.Constant:
                mapped.extend(volatile.get(item.slice.value,()))
        return sorted(set(mapped))

    rows = [dict(name=kw.arg, source=_fact(kw.value,scope), volatile_providers=kinds(kw.value,set()))
            for kw in expression.keywords]
    _require(all(type(row['name']) is str for row in rows) and len({row['name'] for row in rows})==len(rows),
             'report fields must be separately declared')
    return rows


def _locals(fn):
    return dict(item for statement in fn.body if (item := _assignment(statement)))


def _json_effect(fn, owner, scope, imports, namespace, worker, worker_scope):
    json_alias = _one((name for name, module in imports.items() if module=='json'), 'stdlib JSON provider required')
    encoder = json_alias + '.dumps'
    printer = _one((row.value for row in owner.body if type(row) is ast.Expr and _call(row.value,'print')
                    and len(row.value.args)==1 and _call(row.value.args[0],encoder)), 'one JSON stdout publication required')
    [stdout] = printer.args
    [report] = stdout.args
    _require(not printer.keywords and not stdout.keywords and _name(report), 'unsupported stdout format')
    report = _locals(owner)[report.id]
    returned = _one((row.value for row in worker.body if type(row) is ast.Return), 'one named worker report required')
    children = _report_fields(returned,worker_scope,imports,_locals(worker))
    root = _report_fields(report,scope,imports,_locals(owner),children)
    result_list = _one((kw.arg for kw in report.keywords if _name(kw.value)==fn['result_slot']), 'worker report list must keep original shape')
    paths = [([row['name']],row['volatile_providers']) for row in root if row['volatile_providers']]
    paths += [([result_list,'*',row['name']],row['volatile_providers']) for row in children if row['volatile_providers']]
    write = _one((row for row in owner.body if type(row) is ast.If and not row.orelse
                  and type(row.test) is ast.Attribute and _name(row.test.value)==namespace
                  and len(row.body)==1 and type(row.body[0]) is ast.Expr
                  and _call(row.body[0].value, _name(row.test)+'.write_text')), 'one conditionally published report file required')
    file_call = write.body[0].value
    [payload] = file_call.args
    _require(not file_call.keywords and type(payload) is ast.BinOp and type(payload.op) is ast.Add
             and _call(payload.left,encoder) and type(payload.right) is ast.Constant and payload.right.value=='\n'
             and len(payload.left.args)==1 and _name(payload.left.args[0])==_name(stdout.args[0])
             and set(_kwargs(payload.left))=={'indent'}, 'file and stdout must publish the same original record')
    indent = ast.literal_eval(_kwargs(payload.left)['indent'])
    _require(type(indent) is int and indent>=0, 'exact JSON indentation required')
    options = _one((item for item in report.keywords if type(item.value) is ast.DictComp),
                   'one original parameter dictionary projection required')
    [projection] = options.value.generators
    key, value = options.value.key, options.value.value
    pair = projection.target
    items = projection.iter
    excluded = []
    _require(type(pair) is ast.Tuple and [_name(node) for node in pair.elts]==[_name(key),_name(value)]
             and type(items) is ast.Call and type(items.func) is ast.Attribute and items.func.attr=='items'
             and _call(items.func.value,'vars') and [_name(node) for node in items.func.value.args]==[namespace]
             and not items.args and not items.keywords and not items.func.value.keywords and not projection.is_async,
             'parameter record must project the original argparse namespace')
    for clause in projection.ifs:
        _require(type(clause) is ast.Compare and _name(clause.left)==_name(key)
                 and [type(op) for op in clause.ops]==[ast.NotEq] and len(clause.comparators)==1
                 and type(clause.comparators[0]) is ast.Constant and type(clause.comparators[0].value) is str,
                 'parameter record filters must name excluded options')
        excluded.append(clause.comparators[0].value)
    return dict(root_fields=root, worker_fields=children, results_field=result_list,
                parameters=dict(field=options.arg, excluded_options=excluded, source=_fact(options.value,scope)),
                volatile_paths=[path for path,_ in paths], volatile_sources=[dict(path=path,providers=types) for path,types in paths],
                stdout=dict(kind='JSON_TEXT_PRINT', indent=None, newline=True, source=_fact(printer,scope)),
                path_output=dict(kind='CONDITIONAL_PATH_WRITE_TEXT', option=write.test.attr, indent=indent, newline=True,
                                 default_encoding='ORIGINAL_PATHLIB_TEXT_DEFAULT', source=_fact(file_call,scope)))


def _model(source):
    raw = read_artifact(source,json_document=False)
    tree = ast.parse(raw,filename=source['uri'])
    code = compile(raw,source['uri'],'exec',dont_inherit=True,optimize=0)
    keys = {child.co_name: code_key(child) for child in code.co_consts if type(child) is CodeType}
    owner = _entry_function(tree)
    scope = keys[owner.name]
    imports = _imports(tree)
    argv, guards, derived = _argv(tree,owner,scope,imports)
    worker, worker_ast, selector, task_rows, range_rule = _worker(tree,owner,scope,keys)
    report = _json_effect(worker,owner,scope,imports,argv['namespace'],worker_ast,keys[worker_ast.name])
    env = []
    for row in owner.body:
        if type(row) is ast.For and type(row.iter) is ast.Tuple and len(row.body)==1:
            item = row.body[0]
            if type(item) is ast.Assign and len(item.targets)==1 and type(item.targets[0]) is ast.Subscript:
                target=item.targets[0]
                if _origin(_name(target.value),imports)=='os.environ' and _name(target.slice)==_name(row.target):
                    env.extend(dict(name=ast.literal_eval(key),value=ast.literal_eval(item.value),source=_fact(row,scope))
                               for key in row.iter.elts)
    return (dict(entry=source, entry_function=dict(name=owner.name, code_id=scope), argv=argv,
                 worker=worker, report=report, thread_environment=env),
            (guards,derived,selector,task_rows,range_rule))


_COMPARISONS = {ast.Eq:operator.eq, ast.NotEq:operator.ne, ast.Lt:operator.lt, ast.LtE:operator.le,
                ast.Gt:operator.gt, ast.GtE:operator.ge, ast.Is:operator.is_, ast.IsNot:operator.is_not}
_ARITHMETIC = {ast.Add:operator.add, ast.Sub:operator.sub, ast.Mult:operator.mul}


def _evaluate(node, local, namespace, options):
    def value(expr):
        return _evaluate(expr,local,namespace,options)

    if type(node) is ast.Constant:
        return _typed(node.value)['value']
    if type(node) is ast.Name:
        return local[node.id]
    if type(node) is ast.Attribute and _name(node.value)==namespace:
        return options[node.attr]
    if type(node) is ast.IfExp:
        return value(node.body if value(node.test) else node.orelse)
    if type(node) is ast.UnaryOp and type(node.op) is ast.Not:
        return not value(node.operand)
    if type(node) is ast.BoolOp and type(node.op) in (ast.And,ast.Or):
        return (all if type(node.op) is ast.And else any)(value(expr) for expr in node.values)
    if type(node) is ast.Compare and all(type(op) in _COMPARISONS for op in node.ops):
        previous = value(node.left)
        for op, expr in zip(node.ops,node.comparators,strict=True):
            next_value = value(expr)
            if type(op) in (ast.Is, ast.IsNot):
                _require(previous is None or next_value is None, 'source identity guards must compare to None')
            if not _COMPARISONS[type(op)](previous,next_value):
                return False
            previous = next_value
        return True
    if type(node) is ast.BinOp and type(node.op) in _ARITHMETIC:
        return _ARITHMETIC[type(node.op)](value(node.left),value(node.right))
    if type(node) is ast.Call and _name(node.func) in ('min','max','abs') and not node.keywords:
        return {'min':min,'max':max,'abs':abs}[_name(node.func)](*(value(expr) for expr in node.args))
    raise ValueError('array entry contract: unsupported declared option expression')


def _parsed(model, argv):
    parser = argparse.ArgumentParser(add_help=False,exit_on_error=False,description=model['argv']['description'])
    for option in model['argv']['options']:
        kind, default = option['kind'], option['default']['value']
        flags = dict(dest=option['id'],action=option['action'],default=default)
        if kind != 'bool':
            flags['type'] = _TYPES[kind]
        if option['choices']:
            flags['choices'] = [row['value'] for row in option['choices']]
        parser.add_argument(*option['flags'],**flags)
    try:
        return vars(parser.parse_args(argv))
    except (argparse.ArgumentError,SystemExit) as error:
        raise ValueError('array entry contract: unsupported parser execution path') from error


def _binding(model, internal, argv):
    guards,derived,selector,tasks,range_rule = internal
    options = _parsed(model,argv)
    namespace = model['argv']['namespace']
    local = {}
    steps = sorted([(row.lineno,False,row,message) for row,message in guards]
                   + [(row.lineno,True,row,(slot,initial,computed)) for row,slot,initial,computed in derived])
    for _,sets_local,row,detail in steps:
        if sets_local:
            slot,initial,computed = detail
            local[slot] = _evaluate(initial,local,namespace,options)
            if _evaluate(row.test,local,namespace,options):
                local[slot] = _evaluate(computed,local,namespace,options)
        else:
            _require(not _evaluate(row.test,local,namespace,options),detail)
    normal = _evaluate(selector,local,namespace,options)
    count = _evaluate(range_rule.iter.args[0],local,namespace,options)
    _require(type(count) is int and 1 <= count <= 1024,'unobserved task cardinality requires separate qualification')
    workers = []
    for ordinal in range(count):
        store = dict(local,**{model['worker']['ordinal']:ordinal})
        actuals = [_evaluate(expr,store,namespace,options) for expr in tasks.elt.elts]
        workers.append(dict(ordinal=ordinal,parameters={key:_typed(value) for key,value in zip(
            model['worker']['formals'],actuals,strict=True)}))
    params = {name:value for name,value in options.items() if name not in model['report']['parameters']['excluded_options']}
    return dict(disposition='NORMAL_SEQUENTIAL' if normal is True else 'UNSUPPORTED_ALTERNATE_DISPATCH',
                options={name:str(value) if isinstance(value,Path) else value for name,value in options.items()},
                parameters=params, derived={name:_typed(value) for name,value in local.items()}, workers=workers)


def bind_arguments(contract, argv):
    """Bind original argv and mark the matching observed profile, if any."""
    model, internal = _model(contract['entry'])
    bound = _binding(model,internal,argv)
    matched = [index for index,row in enumerate(contract['observed_profiles'])
               if _json_value(row['parameters'])==_json_value(bound['parameters'])]
    if bound['disposition']=='NORMAL_SEQUENTIAL' and not matched:
        bound['disposition']='UNSUPPORTED_UNOBSERVED_PROFILE'
    bound['observed_profile']=matched[0] if matched else None
    return bound


def _json_value(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def _report_value(expression, values):
    kind = expression['kind']
    if kind == 'Constant':
        return expression['literal']['value']
    fields = expression['fields']
    if kind == 'Name':
        return values[fields['id']]
    if kind == 'BinOp':
        operations = {'Add':operator.add,'Sub':operator.sub,'Mult':operator.mul}
        binary = operations[fields['op']['kind']]
        return binary(_report_value(fields['left'],values),_report_value(fields['right'],values))
    raise KeyError(kind)


def _verify_report_values(report, fields, values):
    for field in fields:
        try:
            value = _report_value(field['source']['expression'],values)
        except KeyError:
            continue
        _require(_json_value(report[field['name']])==_json_value(value),
                 'observed report contradicts its literal or bound source value: '+field['name'])


def _witness(model, internal, path):
    reference = _ref(path,ORIGINAL)
    observed = read_artifact(reference)
    _require(observed['schema_version']==1 and observed['entry']==model['entry']
             and observed['instrumented'] is False and observed['repetitions']==1
             and type(observed['run_returncode']) is int and observed['run_returncode']==0
             and type(observed['elapsed_ns']) is int and observed['elapsed_ns']>0,
             'original process receipt scope differs')
    read_artifact(observed['python_executable'],json_document=False)
    clean = _binding(model,internal,observed['argv'])
    _require(clean['disposition']=='NORMAL_SEQUENTIAL' and clean['parameters']==observed['parameters'],
             'normalized argv does not match the observed original options')
    stdout = read_artifact(observed['stdout'],json_document=False)
    _require(not read_artifact(observed['stderr'],json_document=False),'successful report witness has unresolved stderr')
    report = json.loads(stdout)
    shape = model['report']
    _require(list(report)==[row['name'] for row in shape['root_fields']]
             and report[shape['parameters']['field']]==clean['parameters'], 'original report fields or options differ')
    scopes = report[shape['results_field']]
    _require(type(scopes) is list and len(scopes)==len(clean['workers'])
             and all(list(row)==[field['name'] for field in shape['worker_fields']] for row in scopes),
             'original worker report shape or task cardinality differs')
    _verify_report_values(report,shape['root_fields'],{name:row['value'] for name,row in clean['derived'].items()})
    for scope,row in zip(clean['workers'],scopes,strict=True):
        _verify_report_values(row,shape['worker_fields'],{name:value['value'] for name,value in scope['parameters'].items()})
    grouped = [row['name'] for row in shape['worker_fields'] if row['volatile_providers']==['PROCESS_ID']]
    identity_field = _one(grouped,'expected one process origin in original worker result')
    pids = sorted({row[identity_field] for row in scopes})
    _require(len(pids)==1 and all(type(pid) is int and pid>0 for pid in pids),'normal worker process scope differs')
    for field in shape['root_fields']:
        if field['volatile_providers']==['PROCESS_ID']:
            _require(report[field['name']]==pids,'aggregated process scope differs')
    _require(stdout==(json.dumps(report)+'\n').encode(),'original stdout serialization differs')
    path_option = shape['path_output']['option']
    publication = observed['file_outputs'][path_option]
    _require(clean['options'][path_option]==publication['uri'], 'file publication destination differs')
    raw = read_artifact(publication,json_document=False)
    _require(raw==(json.dumps(report,indent=shape['path_output']['indent'])+'\n').encode(),
             'original file serialization differs')
    earlier = read_artifact(observed['earlier_trajectory_run'])
    _require(earlier['mode']=='original' and earlier['execution']['entry_path']==model['entry']['uri'],
             'earlier round observation names another source')
    earlier_binding = _binding(model,internal,earlier['execution']['argv'])
    _require(earlier_binding['parameters']==clean['parameters'] and earlier_binding['workers']==clean['workers'],
             'earlier round observation covers another parser/worker scope')
    public = normalize_report(report,omit_paths=shape['volatile_paths'])
    witness = dict(observation=reference,argv=observed['argv'],parameters=clean['parameters'],
                   derived=clean['derived'],worker_count=len(pids),worker_scopes=[dict(**scope,report=row)
                       for scope,row in zip(clean['workers'],scopes,strict=True)],
                   stdout=observed['stdout'],files=observed['file_outputs'],normalized_report=public,
                   cpu_ids=observed['cpu_ids'],python_executable=observed['python_executable'],
                   earlier_trajectory_run=observed['earlier_trajectory_run'],
                   elapsed_ns=observed['elapsed_ns'],instrumented=False,source_report_encoding_verified=True)
    profile = dict(parameters=clean['parameters'],derived=clean['derived'],workers=clean['workers'],worker_count=len(pids),
                   state_trajectory=observed['earlier_trajectory_run'],scope='OBSERVED_SOURCE_PROFILE')
    return witness,profile


def project(source, *, witnesses=()):
    """Project a source profile and exact publication witnesses without importing the workload."""
    entry = _ref(source,_SOURCE)
    model, internal = _model(entry)
    observed, profiles = [], []
    for path in witnesses:
        witness, profile = _witness(model,internal,path)
        _require(all(_json_value(row['parameters'])!=_json_value(profile['parameters']) for row in profiles),
                 'duplicate observed profile')
        observed.append(witness)
        profiles.append(profile)
    return dict(schema=SCHEMA,schema_version=1,**model,witnesses=observed,observed_profiles=profiles,
                unsupported_profiles=dict(rule='NO_DELIVERY_OUTSIDE_EXACT_OBSERVED_PARAMETER_ROWS',
                    varying_transport_option=model['report']['path_output']['option'], alternate_worker_dispatch='OPEN',
                    parser_help_or_error='ORIGINAL_ONLY', target_semantics_proven=False),
                target_executed_by_projection=False, deployment_authorized=False)


def check(path):
    retained = read_artifact(_ref(path,SCHEMA))
    projected = project(retained['entry']['uri'],
                        witnesses=[row['observation']['uri'] for row in retained['witnesses']])
    _require(_json_value(projected)==_json_value(retained),'cold source or witness replay differs')
    return projected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry')
    parser.add_argument('--witness',action='append',default=[])
    parser.add_argument('--out')
    parser.add_argument('--check')
    args = parser.parse_args(argv)
    if args.check and not (args.entry or args.out or args.witness):
        result = check(args.check)
    elif args.entry and args.out and not args.check:
        result = project(args.entry,witnesses=args.witness)
        with Path(args.out).open('x',encoding='utf-8') as out:
            json.dump(result,out,sort_keys=True,indent=2,allow_nan=False)
            out.write('\n')
    else:
        parser.error('use --entry SCRIPT --out FILE [--witness FILE] or --check FILE')
    print(json.dumps(dict(options=len(result['argv']['options']),profiles=len(result['observed_profiles']))))


if __name__ == '__main__':
    main()
