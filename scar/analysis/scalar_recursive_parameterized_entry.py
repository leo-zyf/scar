"""Extract the bounded runtime command grammar from an admitted source controller."""

import argparse
import ast
from dis import get_instructions
import json
import operator
from pathlib import Path

from scar.analysis import scalar_recursive_entry as entry
from scar.analysis import scalar_recursive_owner_continuity as continuity
from scar.analysis import scalar_recursive_region as recursive
from scar.analysis import scalar_score_reuse_evidence as source
from scar.analysis.scalar_table_handoff import _ref, _write
from scar.compiler.scalar_table_region import _definition
from scar.ir.execution_recipe import read_artifact
from scar.trace.code_snapshot import describe_code, site

SCHEMA = 'scar.scalar-recursive-parameterized-entry'
_ARITHMETIC = {ast.Add:operator.add, ast.Sub:operator.sub, ast.Mult:operator.mul}


def _require(ok,detail):
    if not ok:
        raise ValueError('scalar parameterized entry: '+detail)


def _name(node):
    return source._name(node)


def _nodes(statements):
    return [node for stmt in statements for node,_ in recursive._operations(stmt)]


def _assignment(node):
    _require(type(node) is ast.Assign and len(node.targets)==1,'expected original single assignment')
    return node.targets[0],node.value


def _fold(node,constants):
    if type(node) is ast.Name:
        return constants[node.id]
    if type(node) is ast.Subscript:
        return _fold(node.value,constants)[ast.literal_eval(node.slice)]
    if type(node) is ast.Call and _name(node.func)=='dict' and not node.args and all(key.arg for key in node.keywords):
        return {key.arg:_fold(key.value,constants) for key in node.keywords}
    if type(node) is ast.BinOp:
        left,right = (_fold(value,constants) for value in (node.left,node.right))
        if type(left) is not int or type(right) is not int:
            raise ValueError('unsupported static arithmetic')
        if type(node.op) in _ARITHMETIC:
            return _ARITHMETIC[type(node.op)](left,right)
        if type(node.op) is ast.Pow and 0 <= right <= 63:
            return left ** right
        if type(node.op) is ast.LShift and 0 <= right <= 63:
            return left << right
    return ast.literal_eval(node)


def _constants(tree):
    values,origins = {},{}
    for node in tree.body:
        if type(node) is not ast.Assign or len(node.targets)!=1:
            continue
        target = node.targets[0]
        names = target.elts if type(target) is ast.Tuple else [target]
        if not all(type(label) is ast.Name for label in names):
            continue
        try:
            value = _fold(node.value,values)
            sequence = value if type(target) is ast.Tuple else (value,)
            entries = tuple(zip(names,sequence,strict=True))
        except (KeyError,TypeError,ValueError):
            continue
        for label,item in entries:
            values[label.id] = item
            origins[label.id] = source._span(node)
    return values,origins


def _anchor(code,node):
    return dict(code=describe_code(code),source_span=source._span(node),
        instructions=[site(code,op.offset) for op in get_instructions(code) if source._span(node)==site(code,op.offset)['position']])


def _branch(node,branches):
    return source._one((branch for branch in branches if source._span(branch.test)==node['selector']),
                       'source command branch differs from admitted controller')


def _schema(kind, tokens, branch, code, *, arguments=(), suffix=None):
    result = dict(kind=kind,tokens=tokens,arguments=list(arguments),optional_tail=suffix,
                  source=_anchor(code,branch.test),region=source._span(branch))
    if branch.body and hasattr(branch.body[0],'lineno'):
        result['effect_start'] = source._span(branch.body[0])
    return result


def _fn_record(tree,codes,name):
    fn = _definition(tree,name)
    code = codes[name]
    return fn,dict(code=describe_code(code),span=source._span(fn),formal_ports=list(code.co_varnames[:code.co_argcount]))


def _head(branch):
    compare = branch.test
    _require(type(compare) is ast.Compare and len(compare.ops)==1 and type(compare.ops[0]) is ast.Eq,
             'admitted command must select a source token or fixed prefix')
    value = ast.literal_eval(compare.comparators[0])
    return [value] if type(value) is str else value


def _empty(branches,symbol):
    return source._one((branch for branch in branches if type(branch.test) is ast.UnaryOp
        and type(branch.test.op) is ast.Not and _name(branch.test.operand)==symbol
        and len(branch.body)==1 and type(branch.body[0]) is ast.Continue), 'original blank input must only continue')


def _new_game(branches,owner_symbol,owner_class):
    matches = []
    for branch in branches:
        for stmt in branch.body:
            if type(stmt) is ast.Assign and len(stmt.targets)==1 and _name(stmt.targets[0])==owner_symbol:
                value = stmt.value
                if (type(value) is ast.Call and type(value.func) is ast.Attribute and value.func.attr==owner_class
                        and not value.args and not value.keywords):
                    matches.append((branch,stmt))
    return source._one(matches,'original owner reset branch is not selected by one command')


def _slice(node,local):
    _require(type(node) is ast.Subscript and _name(node.value)==local and type(node.slice) is ast.Slice,
             'expected original token slice')
    return [None if part is None else ast.literal_eval(part) for part in (node.slice.lower,node.slice.upper,node.slice.step)]


def _goto(scope, original, ui, code, branch, ui_constants, ui_origins):
    commands = _nodes(branch.body)
    submit = source._one((node for node in commands if type(node) is ast.Call
        and source._span(node)==scope['controller']['submit']['position']), 'missing original execution submission')
    request = source._one((value for kw in submit.keywords if kw.arg is None for value in ast.walk(kw.value)
        if type(value) is ast.Dict for key,value in zip(value.keys,value.values,strict=True)
        if type(key) is ast.Constant and key.value==ui['request_port']), 'search must read one explicit requested-depth option')
    _require(type(request) is ast.Call and type(request.func) is ast.Attribute and request.func.attr=='get'
        and len(request.args)==1 and not request.keywords,'requested depth does not resolve to explicit option store')
    key = ast.literal_eval(request.args[0]); option = _name(request.func.value)
    match = source._one((stmt for stmt in commands if type(stmt) is ast.Assign and len(stmt.targets)==1
        and type(stmt.targets[0]) is ast.Subscript and _name(stmt.targets[0].value)==option
        and type(stmt.value) is ast.Call and _name(stmt.value.func)=='int'),
        'expected one original integer option assignment')
    parsed = match.value
    _require(len(parsed.args)==1 and not parsed.keywords and type(parsed.args[0]) is ast.Call
        and _name(parsed.args[0].func)=='next','depth integer must be parsed from next original token')
    limit = source._one((stmt for stmt in commands if type(stmt) is ast.Assign and len(stmt.targets)==1
        and _name(stmt.targets[0])==ui['limit_port'] and type(stmt.value) is ast.Subscript
        and _name(stmt.value.value)==option and ast.literal_eval(stmt.value.slice)==key),
        'runtime depth must reach original accepted limit')
    _require(len(submit.args)>=6 and _name(submit.args[5])==ui['limit_port'] and _name(submit.args[2])==ui['history'],
             'original submitted depth and history no longer bind the consumer')
    budget_name = _name(submit.args[4])
    budget = source._one((stmt for stmt in branch.body[:4] if type(stmt) is ast.Assign and len(stmt.targets)==1
                         and _name(stmt.targets[0])==budget_name), 'fixed GO has no original initial budget')
    deadline = source._one((stmt for stmt in commands if type(stmt) is ast.Assign and len(stmt.targets)==1
        and type(stmt.targets[0]) is ast.Attribute and _name(stmt.targets[0].value)==scope['controller']['receiver']
        and type(stmt.value) is ast.BinOp and type(stmt.value.right) is ast.Call and _name(stmt.value.right.func)=='min'),
        'original fixed-depth hard deadline writer missing')
    value = deadline.value
    _require(type(value.op) is ast.Add and type(value.left) is ast.Call and type(value.left.func) is ast.Attribute
        and not value.left.args and not value.left.keywords and len(value.right.args)==2
        and _name(value.right.args[0])==budget_name, 'original deadline needs clock plus cap')
    cap = value.right.args[1]
    default,hard = _fold(budget.value,ui_constants),_fold(cap,ui_constants)
    _require(type(default) is type(hard) is int and 0<hard<=default,'fixed clock port values differ')
    for row in original['searches']:
        modes = row['consumer']['scalars']
        _require(modes[ui['request_port']]==modes[ui['limit_port']]==row['requested_depth']
            and sum(type(item) is int and item==default for item in modes.values())==1
            and all(modes[name]==expected and type(modes[name]) is type(expected) for name,expected in ui['constraints'].items()),
            'old source observation disagrees with computed fixed mode')
    clocks = dict(default_budget_seconds=default,budget_source=_anchor(code,budget.value),hard_cap_seconds=hard,
        hard_cap_source=ui_origins.get(_name(cap),source._span(cap)),hard_field=deadline.targets[0].attr,
        hard_write=_anchor(code,deadline.targets[0]),clock_read=continuity._call(code,value.left),
        soft_field=scope['soft_field'])
    return key,clocks,dict(parser=continuity._call(code,parsed),option_definition=_anchor(code,match.targets[0]),
        depth_binding=_anchor(code,limit.targets[0]),requested_value=continuity._call(code,request),
        observed_consumer=ui['consumer'],submit=scope['controller']['submit'],history_actual=source._span(submit.args[2]))


def _move_loop(loop,token_local, history, hclock, code):
    _require(type(loop) is ast.For and len(loop.body)==3,'original move application loop has unbound actions')
    parse,adjust,append = loop.body
    move,var = _assignment(parse)
    clock,effect = _assignment(adjust)
    _require(type(move) is ast.Name and _name(clock)==hclock and type(var) is ast.Call
        and type(effect) is ast.IfExp and type(effect.test) is ast.Call and type(append) is ast.Expr
        and type(append.value) is ast.Call and type(append.value.func) is ast.Attribute
        and append.value.func.attr=='append' and _name(append.value.func.value)==history and len(append.value.args)==1,
        'move sequence must retain source parse, clock and history ordering')
    update = append.value.args[0]
    _require(type(update) is ast.Call and type(update.func) is ast.Attribute and len(update.args)==1
        and _name(update.args[0])==move.id and type(update.func.value) is ast.Subscript
        and _name(update.func.value.value)==history and ast.literal_eval(update.func.value.slice)==-1,
        'original transition must consume prior history tail and newly parsed move')
    base = loop.iter.args[0] if type(loop.iter) is ast.Call and _name(loop.iter.func)=='enumerate' else loop.iter
    return dict(token_slice=_slice(base,token_local),parse=continuity._call(code,var),
        parser_name=_name(var.func),quiet=continuity._call(code,effect.test),quiet_name=_name(effect.test.func),
        counter_write=_anchor(code,clock),clock_expression=ast.dump(effect,include_attributes=False),
        orientation=ast.dump(var.args[1],include_attributes=False),
        transition=continuity._call(code,update),transition_member=update.func.attr,
        append=continuity._call(code,append.value),order=['PARSE_MOVE_FOR_CURRENT_SIDE','UPDATE_HALFMOVE_BEFORE_TRANSITION','APPEND_RETURNED_STATE'])


def _positions(original,driver,ui,ui_tree,ui_codes,engine,engine_codes,start,fen,launcher_code,token_local):
    history = ui['history']
    clock = entry._halfmove(_definition(ui_tree,original['controller']['name']),
        [dict(source_span=source._span(start)),dict(source_span=source._span(fen))],history,original['controller']['submit'])
    initial, start_loop = start.body
    load, select, count, suffix = fen.body
    _require(type(suffix) is ast.If and len(suffix.body)==2 and type(suffix.body[0]) is ast.Assert,
             'FEN suffix must declare its original move delimiter')
    delimiter = suffix.body[0].test
    _require(type(delimiter) is ast.Compare and type(delimiter.ops[0]) is ast.Eq and type(delimiter.left) is ast.Subscript
        and _name(delimiter.left.value)==token_local and ast.literal_eval(delimiter.left.slice)==8,
        'FEN trailing moves lack a delimiter at the source position')
    marker = ast.literal_eval(delimiter.comparators[0])
    local,prepared = _assignment(load)
    _require(type(local) is ast.Name and type(prepared) is ast.Call and len(prepared.args)==1
        and type(prepared.args[0]) is ast.Starred,'FEN parser must receive exactly the source six fields')
    helper,helper_code = _fn_record(ui_tree,ui_codes,_name(prepared.func))
    _require(_slice(prepared.args[0].value,token_local)==[2,8,None] and len(helper_code['formal_ports'])==6,
             'declared FEN fields differ from original parser ABI')
    parsed = helper.body[-1]
    _require(type(parsed) is ast.Return and type(parsed.value) is ast.IfExp and type(parsed.value.test) is ast.Compare
        and type(parsed.value.orelse) is ast.Call and type(parsed.value.orelse.func) is ast.Attribute,
        'parsed position must retain both orientation source arms')
    color = ast.literal_eval(parsed.value.test.comparators[0]); rotate = parsed.value.orelse.func.attr
    target,choose = _assignment(select)
    _require(_name(target)==history and type(choose) is ast.IfExp and type(choose.body) is type(choose.orelse) is ast.List
        and len(choose.body.elts)==1 and len(choose.orelse.elts)==2 and type(choose.orelse.elts[0]) is ast.Call
        and type(choose.orelse.elts[0].func) is ast.Attribute and choose.orelse.elts[0].func.attr==rotate
        and _name(choose.body.elts[0])==_name(choose.orelse.elts[1])==local.id,
        'controller history must retain original alternate orientation and current last record')
    clock_target,clock_parse = _assignment(count)
    _require(_name(clock_target)==clock and type(clock_parse) is ast.Call and _name(clock_parse.func)=='int'
        and type(clock_parse.args[0]) is ast.Subscript and _name(clock_parse.args[0].value)==token_local
        and ast.literal_eval(clock_parse.args[0].slice)==6,'halfmove must bind original sixth token')
    start_moves = _move_loop(start_loop,token_local,history,clock,launcher_code)
    fen_moves = _move_loop(suffix.body[1],token_local,history,clock,launcher_code)
    _require({start_moves['parser_name'],start_moves['quiet_name'],start_moves['transition_member']}
        =={fen_moves['parser_name'],fen_moves['quiet_name'],fen_moves['transition_member']},'position forms require shared move effects')
    _,parse_record = _fn_record(ui_tree,ui_codes,start_moves['parser_name'])
    _,quiet_record = _fn_record(ui_tree,ui_codes,start_moves['quiet_name'])
    state_type = original['position_preparations'][0]['normal_pure_value']['root']['type']
    _,rotate_record = _fn_record(engine,engine_codes,state_type+'.'+rotate)
    _,move_record = _fn_record(engine,engine_codes,state_type+'.'+start_moves['transition_member'])
    scores = driver['initialization']['table']
    return dict(history_port=history,halfmove_port=clock,suffix_marker=marker,root_state_type=state_type,
        startpos=dict(root=original['position_preparations'][0]['normal_pure_value']['root'],
            initial_source=_anchor(launcher_code,initial.value),moves=start_moves),
        fen=dict(parser=helper_code,call=continuity._call(launcher_code,prepared),
            read_table=dict(source_global=scores['symbol'],use='CURRENT_MODULE_VERSION_BEFORE_NEXT_SEARCH',versions=scores['versions']),
            option_presence_open=['EXTERNAL_FEATURE_OR_ALTERNATE_RECORD_PROVIDER'],
            raw_orientation=dict(direct_color=color,alternate_rotation=continuity._call(ui_codes[helper.name],parsed.value.orelse)),
            controller_orientation=dict(test=ast.dump(choose.test,include_attributes=False),direct_history_records=1,
                alternate_history_records=2,alternate_rotation=continuity._call(launcher_code,choose.orelse.elts[0])),
            halfmove_parse=continuity._call(launcher_code,clock_parse),moves=fen_moves),
        move_parser=parse_record,quiet_move=quiet_record,rotation=rotate_record,move_transition=move_record)


def _default_configuration(original,engine,selected,reset,go_body):
    values,origins = _constants(engine)
    expected = original['startup']['initial_owner']['configuration']
    _require(all(type(values.get(key)) is int and values[key]==value for key,value in expected.items()),
             'original observed option defaults did not replay from source')
    for node in [*selected,*go_body,*reset]:
        _require(not (type(node) is ast.Attribute and isinstance(node.ctx,(ast.Store,ast.Del)) and node.attr in expected)
            and not (type(node) is ast.Call and _name(node.func) in ('setattr','delattr','exec','eval')),
            'selected controller grammar contains an option mutation or unmodeled reflection')
    return dict(values=expected,source_origins={key:origins[key] for key in expected},
        invariance='SELECTED_COMMAND_GRAMMAR_HAS_NO_CONFIGURATION_WRITE; SOURCE_PROVIDER_AND_ENVIRONMENT_BOUND',
        sequence_rules=['SETOPTION_OUTSIDE_GRAMMAR','POSITION_READS_CURRENT_TABLE','SEARCH_CHANGES_SELECTED_TABLE_ROW_ONLY',
                        'NEW_GAME_REPLACES_OWNER_WITHOUT_RESETTING_MODULE_CONFIGURATION'])


def project(path,*,depth_min=1,depth_max=4):
    reference = _ref(path,entry.SCHEMA)
    original = entry.check(reference['uri'])
    _require(read_artifact(reference)==original and type(depth_min) is type(depth_max) is int
        and 1<=depth_min<=depth_max<=4,'only the declared bounded experimental depth subset is admitted')
    ui = read_artifact(original['ui_contract'])
    driver = read_artifact(original['source_driver'])
    ui_tree,codes = source._compile(ui['ui_source'])
    engine,engine_codes = source._compile(original['source'])
    controller = _definition(ui_tree,original['controller']['name'])
    code = codes[controller.name]
    branches = [node for node in _nodes(controller.body) if type(node) is ast.If]
    seed_commands = original['commands']
    replies = [row for row in seed_commands if row['role']=='STATIC_REPLY']
    positions = [row for row in seed_commands if row['role']=='POSITION']
    go, = [row for row in seed_commands if row['role']=='SEARCH'][:1]
    close, = [row for row in seed_commands if row['role']=='CLOSE']
    start,fen = (_branch(row,branches) for row in positions)
    go_branch,quit_branch = (_branch(row,branches) for row in (go,close))
    token_local = original['controller']['command']['symbol']
    owner_class = driver['driver']['code']['qualname'].split('.')[0]
    reset,allocation = _new_game(branches,original['controller']['receiver'],owner_class)
    empty = _empty(branches,token_local)
    ui_values,ui_origins = _constants(ui_tree)
    clock_source = dict(controller=original['controller'],soft_field=driver['time_and_exit']['soft']['deadline_field'])
    depth,clocks,go_ports = _goto(clock_source,original,ui,code,go_branch,ui_values,ui_origins)
    constructor,ctor_code = _fn_record(engine,engine_codes,owner_class+'.__init__')
    soft_origins = []
    for node in _nodes(constructor.body):
        if type(node) is ast.Assign and len(node.targets)==1 and type(node.targets[0]) is ast.Tuple and type(node.value) is ast.Tuple:
            soft_origins.extend(value for target,value in zip(node.targets[0].elts,node.value.elts,strict=True)
                if type(target) is ast.Attribute and target.attr==clocks['soft_field'])
    soft = source._one(soft_origins,'missing exact original soft limit initialization')
    baseline = original['startup']['initial_owner']['fields']
    clocks['soft_initial_epoch'] = _fold(soft,{})
    _require(clocks['soft_initial_epoch']==baseline[clocks['soft_field']],'soft initial epoch differs from original')
    clocks['soft_source'] = _anchor(engine_codes[owner_class+'.__init__'],soft)
    position = _positions(original,driver,ui,ui_tree,codes,engine,engine_codes,start,fen,code,token_local)
    suffix = dict(marker=position['suffix_marker'],element='ASCII_COORDINATE_MOVE',count='SOURCE_LOOP',
        squares=dict(file='a-h',rank='1-8'),promotion=['q','r','b','n'])
    dynamic_fields = [dict(name=name,type=kind) for name,kind in zip(
        ('board','color','castling','en_passant','halfmove','fullmove'),
        ('FEN_EIGHT_RANKS_OF_EIGHT_SQUARES','FEN_SIDE_TOKEN','FEN_CASTLING_TOKEN','FEN_EN_PASSANT_TOKEN',
         'ASCII_SIGNED_I64','FEN_FULLMOVE_TOKEN'),strict=True)]
    grammar = [_schema('EMPTY',[],empty,code)]
    grammar += [_schema(('UCI','ISREADY')[i],row['tokens'],_branch(row,branches),code) for i,row in enumerate(replies)]
    grammar += [_schema('POSITION_START',_head(start),start,code,suffix=suffix),
        _schema('POSITION_FEN',_head(fen),fen,code,arguments=dynamic_fields,suffix=suffix),
        _schema('GO_DEPTH',_head(go_branch)+[depth],go_branch,code,
                arguments=[dict(name='depth',type='ASCII_SIGNED_I64',minimum=depth_min,maximum=depth_max)]),
        _schema('NEW_GAME',_head(reset),reset,code),_schema('QUIT',_head(quit_branch),quit_branch,code)]
    static_replies = [_branch(row,branches) for row in replies]
    selected = _nodes([stmt for branch in (start,fen,empty,quit_branch,*static_replies) for stmt in branch.body])
    configuration = _default_configuration(original,engine,selected,_nodes(reset.body),_nodes(go_branch.body))
    configuration.update(clocks=clocks)
    return dict(schema=SCHEMA,schema_version=1,source_entry=reference,source_driver=original['source_driver'],
        ui_contract=original['ui_contract'],owner_scope=original['owner_scope'],source=original['source'],
        grammar=dict(commands=grammar,tokenization='ASCII_SUBSET_OF_ORIGINAL_WHITESPACE_SPLIT',
            dispatch='MATCH_COMPLETE_COMMAND_BEFORE_EFFECTS',protocol='ORDERED_ACKNOWLEDGED_IDLE_COMMANDS',
            depth_range=dict(minimum=depth_min,maximum=depth_max,authority='DECLARED_EXPERIMENTAL_DOMAIN'),
            malformed_or_unsupported='DECLINE_BEFORE_CURRENT_COMMAND_EFFECTS'),
        ports=dict(controller=describe_code(code),input=original['controller']['command'],go=go_ports,
            driver=driver['driver'],root=driver['initialization']['root'],bracket=driver['bracket'],
            yield_port=driver['yield'],output=original['output_protocol']),
        positions=position,configuration=configuration,
        effects=dict(acknowledgement=['STATIC_REPLY_FLUSH','SEARCH_FINAL_REPLY_FLUSH_BEFORE_NEXT_COMMAND'],
            search_state=['SET_ORIGINAL_HARD_DEADLINE','RESET_SCORE_HISTORY_AND_COUNTER','RETAIN_MOVE_MAP','SELECT_SCORING_VERSION'],
            new_game=dict(allocation=continuity._call(code,allocation.value),constructor=ctor_code,
                replacement_source=_anchor(code,allocation.targets[0]),fields=baseline,
                retains=['CONTROLLER_POSITION_HISTORY','CONTROLLER_HALFMOVE_CLOCK','MODULE_SCORING_VERSION','DEFAULT_OPTIONS']),
            close=dict(source=_anchor(code,quit_branch.test),requires_idle_owner=True,effect='ORIGINAL_IDLE_OWNER_STOP_AND_CLOSE'),
            historical_startup_replies=original['startup']['static_replies']),
        proof_frontiers=dict(held_out_input_used=False,original_new_domain_executed_by_projection=False,
            evidence='EXACT_SOURCE_CONTROL_AND_ORIGINAL_FIXED_DRIVER_HISTORY',
            unbound=['TIMED_OR_INFINITE_REQUESTS','CONCURRENT_INPUT_WHILE_SEARCHING','OPTION_MUTATION',
                     'OTHER_ORIGINAL_COMMANDS','EXTERNAL_PROVIDER_AND_SIGNAL_EFFECTS','INVALID_POSITION_LEGALITY'],
            dynamic_color_and_move_execution_still_require_validation=True,deployment_authorized=False))


def check(path):
    result = read_artifact(_ref(path,SCHEMA))
    read_artifact(result['source_entry'])
    domain = result['grammar']['depth_range']
    _require(result==project(result['source_entry']['uri'],depth_min=domain['minimum'],depth_max=domain['maximum']),
             'parameterized grammar differs from independent source replay')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--source-entry',type=Path)
    mode.add_argument('--check',type=Path)
    parser.add_argument('--out',type=Path)
    args = parser.parse_args()
    if args.check:
        result = check(args.check)
        print(json.dumps(dict(commands=len(result['grammar']['commands']),depth=result['grammar']['depth_range'])))
    else:
        if args.out is None:
            parser.error('publication requires --out')
        print(json.dumps(_write(args.out,project(args.source_entry))))


if __name__=='__main__':
    main()
