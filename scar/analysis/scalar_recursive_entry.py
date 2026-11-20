"""Bind a finite original command tape to typed startup and recursive search inputs."""

import argparse
import ast
import json
from pathlib import Path

from scar.analysis import scalar_recursive_owner_continuity as continuity
from scar.analysis import scalar_recursive_region as recursive
from scar.analysis import scalar_score_reuse_evidence as source
from scar.analysis.scalar_table_handoff import _ref, _write
from scar.compiler.scalar_table_region import _definition
from scar.ir.execution_recipe import read_artifact
from scar.trace.code_snapshot import describe_code

SCHEMA = 'scar.scalar-recursive-entry-source'
DESIGN = 'scar.scalar-recursive-entry-design'
CAPTURE = 'scar.scalar-recursive-entry-capture'


def _require(ok,detail):
    if not ok:
        raise ValueError('scalar recursive entry: '+detail)


def _nodes(fn):
    return [node for stmt in fn.body for node,_ in recursive._operations(stmt)]


def _literal(node,tokens,symbol):
    if type(node) is ast.Subscript and source._name(node.value)==symbol:
        part = node.slice
        if type(part) is ast.Slice:
            low,high,step = (None if item is None else ast.literal_eval(item) for item in
                             (part.lower,part.upper,part.step))
            return tokens[slice(low,high,step)]
        return tokens[ast.literal_eval(part)]
    return ast.literal_eval(node)


def _match(test,tokens,symbol):
    if type(test) is not ast.Compare or len(test.ops)!=1 or not any(
            type(item) is ast.Subscript and source._name(item.value)==symbol for item in ast.walk(test)):
        return False
    try:
        left,right = (_literal(value,tokens,symbol) for value in (test.left,test.comparators[0]))
        if type(test.ops[0]) is ast.Eq:
            return left==right
        if type(test.ops[0]) is ast.In:
            return left in right
    except (ValueError,TypeError,IndexError):
        return False
    return False


def _print_sites(nodes,code):
    return [continuity._call(code,node) for node in nodes if type(node) is ast.Call and source._name(node.func)=='print']


def _roles(launcher,code,tape,command_port,history_formal,submit):
    scopes = [node for node in _nodes(launcher) if type(node) is ast.If]
    choices = []
    for ordinal,tokens in enumerate(tape):
        candidates = []
        for branch in scopes:
            if not _match(branch.test,tokens,command_port):
                continue
            nodes = [row for stmt in branch.body for row,_ in recursive._operations(stmt)]
            printed = _print_sites(nodes,code)
            assignments = [node for node in branch.body if type(node) is ast.Assign and any(
                type(item) is ast.Name and item.id==history_formal for target in node.targets for item in ast.walk(target))]
            if any(type(node) is ast.Call and source._span(node)==submit['position'] for node in nodes):
                kind = 'SEARCH'
            elif assignments:
                kind = 'POSITION'
            elif any(type(node) is ast.Break for node in branch.body):
                kind = 'CLOSE'
            elif printed:
                kind = 'STATIC_REPLY'
            else:
                continue
            candidates.append(dict(ordinal=ordinal,tokens=tokens,role=kind,selector=source._span(branch.test),
                source_span=source._span(branch),print_sites=printed,
                history_assignments=[source._span(row) for row in assignments]))
        candidates = [row for row in candidates if not any(
            other is not row and row['selector'][0]<other['selector'][0]
            and other['selector'][0]<=row['source_span'][1] for other in candidates)]
        choices.append(source._one(candidates,'command must select one supported original controller branch'))
    return choices


def _entry(original, launcher):
    tree, codes = source._compile(original)
    rows = []
    for fn in tree.body:
        if type(fn) is not ast.FunctionDef:
            continue
        for stmt in fn.body:
            if type(stmt) is not ast.Return or type(stmt.value) is not ast.Call:
                continue
            call = stmt.value
            if type(call.func) is ast.Attribute and call.func.attr==launcher and len(call.args)==2:
                rows.append((fn,call))
    fn,call = source._one(rows,'script must transfer to one original declared command controller')
    return dict(body=describe_code(codes[fn.name]),call=continuity._call(codes[fn.name],call),
                seed_actual=source._span(call.args[1]),source=original)


def _position_helpers(commands,launcher,ui_functions,ui_codes):
    result = {}
    spans = {tuple(row['source_span']) for row in commands if row['role']=='POSITION'}
    for branch in _nodes(launcher):
        if type(branch) is not ast.If or tuple(source._span(branch)) not in spans:
            continue
        for stmt in branch.body:
            if type(stmt) is not ast.Assign or type(stmt.value) is not ast.Call:
                continue
            call = stmt.value
            name = source._name(call.func)
            if name not in ui_functions:
                continue
            body = ui_functions[name]
            effects = []
            fresh = {}
            for operation in _nodes(body):
                if type(operation) is ast.Assign and len(operation.targets)==1 and source._name(operation.targets[0]) is not None:
                    value = operation.value
                    if type(value) is ast.Call and source._name(value.func)=='list' and not value.keywords:
                        fresh[operation.targets[0].id] = operation.lineno
            candidates = sorted((node for node in _nodes(body) if type(node) in (ast.Call,ast.Subscript,ast.Attribute)),
                                key=source._span)
            for node in candidates:
                if type(node) is ast.Call:
                    effects.append(dict(kind='CALL',site=continuity._call(ui_codes[name],node),
                        function=ast.dump(node.func,include_attributes=False)))
                if (type(node) is ast.Attribute and isinstance(node.ctx,(ast.Store,ast.Del)) or
                         type(node) is ast.Subscript and isinstance(node.ctx,(ast.Store,ast.Del))):
                    owned = type(node) is ast.Subscript and type(node.ctx) is ast.Store and (
                        fresh.get(source._name(node.value),node.lineno)<node.lineno)
                    _require(owned,'position helper contains another source-visible write')
                    effects.append(dict(kind='ITEM_OR_ATTRIBUTE_WRITE',span=source._span(node),
                        expression=ast.dump(node,include_attributes=False),owner='FRESH_LOCAL_LIST'))
            result[name] = dict(code=describe_code(ui_codes[name]),
                caller=continuity._call(ui_codes[launcher.name],call),effects=effects,
                effect_scope='SELECTED_SOURCE_VALUES_CAN_BE_FROZEN_UNDER_EXACT_PROVIDER_AND_PRIVATE_SETUP_CONTRACT')
    return result


def _halfmove(launcher, positions, history, submit):
    scopes = {tuple(row['source_span']) for row in positions}
    common = []
    for row in _nodes(launcher):
        if type(row) is ast.If and tuple(source._span(row)) in scopes:
            common.append({item.id for stmt in row.body if type(stmt) is ast.Assign
                           for target in stmt.targets for item in ast.walk(target) if type(item) is ast.Name})
    bound = set.intersection(*common)-{history}
    caller = source._one((row for row in _nodes(launcher) if type(row) is ast.Call
                          and source._span(row)==submit['position']),'original controller submits no source call')
    return source._one((kw.arg for kw in caller.keywords if kw.arg in bound and source._name(kw.value)==kw.arg),
                       'position preparation must also bind the original output-context counter')


def prepare(owner_scope_path):
    owner_ref = _ref(owner_scope_path,continuity.SCHEMA)
    owner = continuity.check(owner_ref['uri'])
    _require(read_artifact(owner_ref)==owner and len(owner['windows'])==1,
             'finite entry scope must bind one exact original inter-search window')
    ui = read_artifact(owner['ui_contract'])
    driver = read_artifact(owner['source_driver'])
    stdin = read_artifact(owner['stdin'],json_document=False).decode('ascii')
    tape = [line.split() for line in stdin.splitlines()]
    _require(stdin==''.join(' '.join(row)+'\n' for row in tape), 'command tape is not the admitted ASCII line format')
    tree,codes = source._compile(ui['ui_source'])
    scopes = owner['static']; entry = scopes['launcher']; launcher = _definition(tree,entry['name'])
    commands = _roles(launcher,codes[entry['name']],tape,entry['command']['symbol'],ui['history'],entry['submit'])
    searches = [row for row in commands if row['role']=='SEARCH']
    positions = [row for row in commands if row['role']=='POSITION']
    _require(len(searches)==len(positions)==len(driver['sessions'])==2 and commands[-1]['role']=='CLOSE'
        and all(position['ordinal']+1==search['ordinal'] for position,search in zip(positions,searches,strict=True)),
        'one controlled entry expects each finite search immediately after its original position')
    consumer = ui['consumer']['qualname']
    fn = _definition(tree,consumer)
    sources = {node.name:node for node in tree.body if type(node) is ast.FunctionDef}
    return dict(schema=DESIGN,schema_version=1,owner_scope=owner_ref,ui_contract=owner['ui_contract'],
        source_driver=owner['source_driver'],source=driver['source'],ui_source=ui['ui_source'],stdin=owner['stdin'],
        owner_fields=owner['roles'],entry=_entry(driver['source'],entry['name']),controller=entry,
        controller_code=describe_code(codes[entry['name']]),halfmove_clock=_halfmove(launcher,positions,ui['history'],entry['submit']),
        commands=commands,position_helpers=_position_helpers(commands,launcher,sources,codes),
        output_protocol=dict(consumer=ui['consumer'],relay=ui['relay'],search_print_sites=_print_sites(_nodes(fn),codes[consumer]),
            caller_mode_ports=['receiver','history','request_port','limit_port'],
            table_reads=owner['static']['access_sites'],
            input_mode='EXACT_ACKNOWLEDGED_ORDERED_COMMAND_TAPE',
            search_results='EVALUATE_NATIVE_SEARCH_AND_ORIGINAL_OUTPUT_DEPENDENCIES',
            clock_and_output='ORIGINAL_PRINT_LOCALE_AND_TIMING_FIELDS_OBSERVABLE'))


def _expect_input(raw,design,driver_plan):
    searches = [row for row in design['commands'] if row['role']=='SEARCH']
    mode = read_artifact(design['ui_contract'])
    frozen = []
    for ordinal,probe in enumerate(raw['searches']):
        command = searches[ordinal]
        witness = driver_plan['sessions'][ordinal]
        _require(probe['command_ordinal']==command['ordinal'] and probe['position_ordinal']==command['ordinal']-1
            and probe['history']==witness['history_input'] and probe['history'][-1]==witness['root_input']
            and probe['configuration']==witness['native_configuration']
            and probe['move_table_size']==witness['initial']['move_table']['size']
            and probe['move_table_digest']==witness['initial_move_digest']
            and probe['consumer']['code']==mode['consumer']
            and probe['consumer']['scalars']==witness['consumer_scalars']
            and probe['halfmove_clock']==witness['consumer_scalars'][design['halfmove_clock']]
            and probe['submit']==design['controller']['submit'],
            'prepared original history or carry state differs from the pinned source run')
        _require(probe['position_table_version'] in driver_plan['initialization']['table']['versions'],
                 'position depends on an unqualified original scoring version')
        if ordinal:
            _require(probe['position_table_version']==driver_plan['sessions'][ordinal-1]['table_version'],
                     'position preparation must see the previous search table before the next reset')
        frozen.append(dict(probe,root=witness['root_input'],search_table_version=witness['table_version'],
            requested_depth=witness['consumer_scalars'][mode['request_port']],
            history_authority='EXACT_ORIGINAL_PREPARED_VALUE_ON_CLOSED_COMMAND_AND_PROVIDERS'))
    _require(len(frozen)==len(searches),'every selected original search must have its own observed prepared history')
    return frozen


def _literal_root(design,frozen):
    tree,_ = source._compile(design['source'])
    body = _definition(tree,design['entry']['body']['qualname'])
    caller = source._one((row for row in _nodes(body) if type(row) is ast.Call
        and source._span(row)==design['entry']['call']['position']),'missing original entry seed expression')
    seed = caller.args[1]
    _require(type(seed) is ast.Subscript and source._name(seed.value) and ast.literal_eval(seed.slice)==-1,
             'initial position must come from the final original source bootstrap record')
    definitions = {item.targets[0].id:item.value for item in tree.body if type(item) is ast.Assign
                   and len(item.targets)==1 and type(item.targets[0]) is ast.Name}
    holder = definitions[seed.value.id]
    _require(type(holder) is ast.List and len(holder.elts)==1,'startup position needs one source literal history')
    constructor = holder.elts[0]
    _require(type(constructor) is ast.Call and type(constructor.func) is ast.Name and not constructor.keywords,
             'initial root carrier is not a direct source record constructor')
    fields = [ast.literal_eval(definitions[value.id] if type(value) is ast.Name else value)
              for value in constructor.args]
    fields = [list(value) if type(value) is tuple else value for value in fields]
    observed = frozen[0]['root']
    _require(observed==dict(type=constructor.func.id,fields=fields),'literal initial source record differs from observed root')
    return dict(kind='ORIGINAL_LITERAL_BOOTSTRAP',source=source._span(constructor),root=observed)


def _controller_setup(design, ui_tree, engine, command, search, helper):
    fn = _definition(ui_tree,design['controller']['name'])
    branch = source._one((row for row in _nodes(fn) if type(row) is ast.If
        and source._span(row)==command['source_span']),'original controller position branch is absent')
    tokens,symbol = command['tokens'],design['controller']['command']['symbol']
    if helper is None:
        _require(len(branch.body)==2 and type(branch.body[0]) is ast.Assign and type(branch.body[1]) is ast.For,
                 'original literal position introduces unbound controller effects')
        sequence = branch.body[1].iter
        _require(type(sequence) is ast.Call and source._name(sequence.func)=='enumerate' and not sequence.keywords
            and len(sequence.args)==1 and _literal(sequence.args[0],tokens,symbol)==[],
            'literal position tape contains additional move applications')
        return
    _require(len(branch.body)==4 and all(type(item) is ast.Assign for item in branch.body[:3])
        and type(branch.body[-1]) is ast.If,'original parsed position introduces unbound controller effects')
    rooted,history,clock = branch.body[:3]
    choice = history.value
    _require(type(choice) is ast.IfExp and type(choice.test) is ast.Compare and type(choice.test.ops[0]) is ast.Eq
        and type(choice.test.left) is ast.Call and len(choice.test.left.args)==1 and type(choice.body) is ast.List
        and len(choice.body.elts)==1 and source._name(choice.body.elts[0])==source._name(rooted.targets[0])
        and source._name(choice.test.left.args[0])==source._name(rooted.targets[0]),
        'original parsed position needs its direct unrotated seed')
    color = _definition(ui_tree,source._name(choice.test.left.func))
    statements = color.body[1:] if ast.get_docstring(color) is not None else color.body
    statement, = statements
    expression = statement.value
    _require(type(statement) is ast.Return and type(expression) is ast.IfExp and type(expression.test) is ast.Call
        and type(expression.test.func) is ast.Attribute and expression.test.func.attr=='startswith'
        and len(expression.test.args)==1 and not expression.test.keywords
        and type(expression.test.func.value) is ast.Attribute and len(color.args.args)==1
        and source._name(expression.test.func.value.value)==color.args.args[0].arg,
        'original color lookup is outside exact primitive record reads')
    carrier = source._one((item for item in engine.body if type(item) is ast.ClassDef
        and item.name==search['root']['type']),'initialized root class is not the original carrier')
    base, = carrier.bases
    _require(type(base) is ast.Call and len(base.args)==2,'initialized root has no source tuple fields')
    fields = ast.literal_eval(base.args[1]).split()
    value = search['root']['fields'][fields.index(expression.test.func.value.attr)]
    branch_value = expression.body if value.startswith(ast.literal_eval(expression.test.args[0])) else expression.orelse
    _require(source._name(branch_value)==source._name(choice.test.comparators[0]),
             'original position does not take its proven unrotated controller arm')
    _require(type(clock.value) is ast.Call and source._name(clock.value.func)=='int' and len(clock.value.args)==1
        and not clock.value.keywords and int(_literal(clock.value.args[0],tokens,symbol))==search['halfmove_clock'],
        'original halfmove integer must be decoded from the pinned tape')
    tail = branch.body[-1].test
    _require(type(tail) is ast.Compare and type(tail.ops[0]) is ast.Gt and type(tail.left) is ast.Call
        and source._name(tail.left.func)=='len' and len(tail.left.args)==1 and source._name(tail.left.args[0])==symbol
        and len(tokens)<=ast.literal_eval(tail.comparators[0]),'original position suffix would execute unmodeled move actions')


def _position_effects(design,raw,frozen):
    ui_tree,_ = source._compile(design['ui_source'])
    engine,engine_codes = source._compile(design['source'])
    originals = {row['ordinal']:row for row in design['commands'] if row['role']=='POSITION'}
    admitted_calls = {'list','enumerate','sum','hasattr','int'}
    string_methods = {'split','join','upper','isupper','islower','group'}
    roles = []
    for index,position in enumerate(raw['positions']):
        command = originals[position['ordinal']]
        search = frozen[index]
        _require(position['ordinal']==search['position_ordinal'] and position['before']==position['after']
            and position['table_version']==search['position_table_version'], 'original preparation unexpectedly changed a visible holder')
        if not position['helpers']:
            _require(index==0,'additional literal positions are outside original provider scope')
            _controller_setup(design,ui_tree,engine,command,search,None)
            root = _literal_root(design,frozen)
            roles.append(dict(ordinal=position['ordinal'],normal_pure_value=root,observable_holder_writes=[],
                              source_selector=command['selector']))
            continue
        helper, = position['helpers']
        _controller_setup(design,ui_tree,engine,command,search,helper)
        binding = design['position_helpers'][helper['name']]
        _require(helper['code']==binding['code'] and helper['result']==search['root']
            and helper['command_ordinal']==position['ordinal'] and helper['arguments']==command['tokens'][2:],
            'typed prepared value does not come from its original position helper and input')
        body = _definition(ui_tree,helper['name'])
        observed_lines = set(helper['lines'])
        providers = []
        for node in ast.walk(body):
            if type(node) is not ast.Call or node.lineno not in observed_lines:
                continue
            target = node.func
            name = source._name(target)
            symbol = target.attr if isinstance(target,ast.Attribute) else None
            anchor = source._span(node)
            if name in admitted_calls:
                role = 'EXACT_BUILTIN'
            elif symbol in string_methods:
                role = 'EXACT_PRIMITIVE_METHOD'
            elif symbol=='sub' and type(node.args[1]) is ast.Lambda and not node.keywords:
                role = 'EXACT_REGEX_WITH_SOURCE_LOCAL_CALLBACK'
            elif symbol==search['root']['type']:
                role = 'EXACT_IMMUTABLE_SOURCE_RECORD_CONSTRUCTOR'
            elif symbol in engine_codes and len(node.args)==1 and not node.keywords:
                method = _definition(engine,symbol)
                _require(len(method.body)==1 and type(method.body[0]) is ast.Return
                    and all(type(value) is not ast.Attribute and (type(value) is not ast.Call or
                            source._name(value.func) in ('ord','int')) for value in ast.walk(method.body[0])),
                    'position leaf requires one original primitive value return')
                providers.append(dict(kind='EXACT_SOURCE_SCALAR_LEAF',span=anchor,code=describe_code(engine_codes[symbol])))
                continue
            else:
                # An unselected conditional-expression arm can share the executed source line.
                parents = {child:item for item in ast.walk(body) for child in ast.iter_child_nodes(item)}
                parent = parents.get(node)
                if (type(parent) is ast.IfExp and parent.orelse is node and type(parent.test) is ast.Compare
                        and type(parent.test.ops[0]) is ast.Eq and type(parent.test.left) is ast.Name
                        and len(parent.test.comparators)==1 and type(parent.test.comparators[0]) is ast.Constant):
                    parameters = binding['code']['parameters']
                    selected = helper['arguments'][parameters.index(parent.test.left.id)]
                    _require(selected==parent.test.comparators[0].value,'position conditional selected an unsupported effect arm')
                    continue
                raise ValueError('scalar recursive entry: open selected position provider '+str(symbol or name))
            providers.append(dict(kind=role,span=anchor,member=symbol or name))
        writes = [row for row in binding['effects'] if row['kind']=='ITEM_OR_ATTRIBUTE_WRITE']
        _require(all(row['owner']=='FRESH_LOCAL_LIST' and row['span'][0] in observed_lines for row in writes),
                 'source position preparation has an unaccounted write')
        roles.append(dict(ordinal=position['ordinal'],source_selector=command['selector'],normal_pure_value=dict(
            kind='ORIGINAL_BOUNDED_INPUT_PREPARATION',root=helper['result'],caller=binding['caller'],helper=helper['code'],
            selected_lines=helper['lines'],return_site=helper['return_site'],providers=providers,
            local_writes=writes,table_version=position['table_version']),observable_holder_writes=[]))
    return roles


def project(owner_path,capture_path):
    design = prepare(owner_path)
    ref = _ref(capture_path,CAPTURE)
    raw = read_artifact(ref)
    driver_plan = read_artifact(design['source_driver'])
    _require(read_artifact(raw['design'])==design and raw['source']==design['source'] and raw['stdin']==design['stdin']
        and raw['input_tokens']==[row['tokens'] for row in design['commands']] and raw['launch']['exit_code']==0
        and raw['performance_measurement'] is False and raw['original_execution'] is True and not raw['frontiers'],
        'owner entry must bind one completed, exact original input run')
    _require(raw['actual_entry']['source_call']==design['entry']['call']
        and raw['actual_entry']['entry']['code']==design['controller_code']['key']
        and raw['io']['stdin_encoding']==raw['io']['stdout_encoding']=='utf-8'
        and raw['io']['stdout_isatty'] is False and raw['io']['locale']=='C'
        and all(row['size']==0 for row in raw['initial_owner']['holder'].values())
        and raw['initial_owner']['configuration']==driver_plan['sessions'][0]['native_configuration'],
        'original startup or exact text stream scope differs')
    replies = []
    for command in design['commands']:
        if command['role']!='STATIC_REPLY':
            continue
        number = command['ordinal']
        chosen = [row for row in raw['static_prints'] if row['command_ordinal']==number]
        _require(chosen and all(row['site'] in command['print_sites'] and row['flush'] is True
            and row['normal_return'] is True and type(row['text']) is str and row['text'].isascii()
            and '\n' not in row['text'] for row in chosen),'static reply differs from original non-search output branch')
        replies.append(dict(command_ordinal=number,lines=[row['text'] for row in chosen],
                            sites=[row['site'] for row in chosen],flush=True,
                            authority='EXACT_ORIGINAL_NON_SEARCH_OUTPUT_AT_A_PINNED_CONTROLLER_SITE'))
    _require(sum(len(row['lines']) for row in replies)==len(raw['static_prints']),
             'search output cannot be specialized as a static reply')
    frozen = _expect_input(raw,design,driver_plan)
    position_effects = _position_effects(design,raw,frozen)
    return dict(schema=SCHEMA,schema_version=1,owner_scope=design['owner_scope'],ui_contract=design['ui_contract'],
        source_driver=design['source_driver'],stdin=design['stdin'],source=design['source'],capture=ref,
        commands=design['commands'],startup=dict(static_replies=replies,initial_owner=raw['initial_owner'],io=raw['io'],
            actual_entry=raw['actual_entry'],initial_table_version=raw['initial_table']),
        searches=frozen,position_preparations=position_effects,output_protocol=design['output_protocol'],
        entry=design['entry'],controller=design['controller'],position_helpers=design['position_helpers'],
        qualification=dict(no_target_execution_during_projection=True,search_outputs_not_replayed=True,
            tape_only=True,other_commands_admitted=False,unknown_provider_effects_remain=True,
            prepared_values_and_visible_table_effects_closed_under_exact_pinned_providers=True,
            whole_python_program_equivalent=False,deployment_authorized=False))


def check(path):
    result = read_artifact(_ref(path,SCHEMA))
    read_artifact(result['capture'])
    _require(result==project(result['owner_scope']['uri'],result['capture']['uri']),
             'stored standalone entry differs from independent source replay')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--owner-scope',type=Path)
    mode.add_argument('--check',type=Path)
    parser.add_argument('--capture',type=Path)
    parser.add_argument('--out',type=Path)
    args = parser.parse_args()
    if args.check:
        result = check(args.check)
        print(json.dumps(dict(commands=len(result['commands']),searches=len(result['searches']))))
    else:
        if args.out is None:
            parser.error('publication needs --out')
        result = prepare(args.owner_scope) if args.capture is None else project(args.owner_scope,args.capture)
        print(json.dumps(_write(args.out,result)))


if __name__=='__main__':
    main()
