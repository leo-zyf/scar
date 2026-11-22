"""Recover a resumable scalar driver around an admitted recursive body."""

import argparse
import ast
from dis import get_instructions
import json
import operator
from pathlib import Path

from scar.analysis import scalar_loop_score_provenance as loop
from scar.analysis import scalar_recursive_region as recursive
from scar.analysis import scalar_score_reuse_evidence as source
from scar.analysis import scalar_table_handoff as tables
from scar.analysis.replay_transaction import detached_replay
from scar.analysis.retained_code_bodies import RetainedCodeBodyIndex
from scar.analysis.static.lowering.source_call_lowering import _body
from scar.ir.execution_recipe import read_artifact
from scar.ir.record_codec import encode
from scar.trace.code_snapshot import describe_code, site

SCHEMA = 'scar.scalar-recursive-driver'
CAPTURE = 'scar.scalar-recursive-driver-capture'


def _require(ok, reason):
    if not ok:
        raise ValueError('scalar recursive driver: '+reason)


def _expr(node):
    if type(node) is ast.Name:
        return dict(op='name',name=node.id)
    if type(node) is ast.Constant and type(node.value) in (int,str,bool):
        return dict(op='literal',value=node.value)
    if type(node) is ast.Attribute:
        return dict(op='field',name=node.attr,args=[_expr(node.value)])
    if type(node) is ast.UnaryOp and type(node.op) is ast.USub:
        return dict(op='neg',args=[_expr(node.operand)])
    if type(node) is ast.BoolOp and type(node.op) is ast.And:
        return dict(op='and',args=[_expr(value) for value in node.values])
    if type(node) is ast.BinOp and type(node.op) in (ast.Add,ast.Sub,ast.FloorDiv):
        return dict(op={ast.Add:'add',ast.Sub:'sub',ast.FloorDiv:'floor_div'}[type(node.op)],
                    args=[_expr(node.left),_expr(node.right)])
    if type(node) is ast.Compare and len(node.ops)==1 and type(node.ops[0]) in (ast.Lt,ast.Gt,ast.GtE,ast.In):
        return dict(op={ast.Lt:'lt',ast.Gt:'gt',ast.GtE:'ge',ast.In:'in'}[type(node.ops[0])],
                    args=[_expr(node.left),_expr(node.comparators[0])])
    raise ValueError('scalar recursive driver: unsupported bounded expression '+type(node).__name__)


def _find(tree, owner):
    class_name,member = owner['qualname'].split('.')
    actor = source._one((node for node in tree.body if type(node) is ast.ClassDef and node.name==class_name),
                        'recursive class definition absent')
    matches = []
    for node in actor.body:
        if type(node) is not ast.FunctionDef or node.name==member or node.decorator_list:
            continue
        args = node.args.posonlyargs+node.args.args
        if len(args)!=2 or not any(type(item) is ast.Yield for stmt in node.body for item,_ in recursive._operations(stmt)):
            continue
        receiver = args[0].arg
        matches.extend((node,args,item) for stmt in node.body for item,_ in recursive._operations(stmt)
            if type(item) is ast.Call and type(item.func) is ast.Attribute and item.func.attr==member
            and source._name(item.func.value)==receiver)
    driver,args,call = source._one(matches,'one direct generator driver for original recursive body expected')
    return class_name,driver,args,call


def _source_call(facts, code, node):
    row,reference = facts.coordinate('calls',node,lambda item:True)
    return row,dict(**reference,call=source._event(code,node,'CALL'),world_in=encode(facts.port(row.world)),
                    world_out=encode(facts.port(row.result_world)))


def _attribute_write(facts,code,target):
    rows = []
    for ordinal,item in enumerate(facts.facts.attributes):
        if item.name!=target.attr or item.action.name!='SET':
            continue
        point = facts.port(item.value)
        _,span,_ = loop._metadata(facts.syntax,point.port.template.occurrence)
        if [span[key] for key in ('line','end_line','column','end_column')]==source._span(target):
            rows.append((ordinal,item,point))
    ordinal,item,point = source._one(rows,'receiver slot lacks original F4 write')
    return dict(field=target.attr,store=source._event(code,target,'STORE_ATTR'),value=encode(point),
        f4=encode(facts.slicer.fact(facts.facts,'attributes',ordinal)),
        world_in=encode(facts.port(item.world)),world_out=encode(facts.port(item.result_world)))


def _assignment(node):
    _require(type(node) is ast.Assign and len(node.targets)==1,'expected one original assignment target')
    return node.targets[0],node.value


def _store(code,target):
    return source._event(code,target,'STORE_FAST')


def _init(facts,code,statements,receiver,history,tablespec,recursive_fields):
    reset,root,table,gamma = statements
    target,value = _assignment(reset)
    _require(type(target) is ast.Tuple and type(value) is ast.Tuple and len(target.elts)==len(value.elts)==3
        and all(type(item) is ast.Attribute and source._name(item.value)==receiver for item in target.elts),
        'original receiver reset must be an ordered three-slot assignment')
    first,second,third = target.elts
    zero,convert,empty = value.elts
    _require(type(zero) is ast.Constant and type(zero.value) is int and zero.value==0
        and type(convert) is ast.Call and source._name(convert.func)=='set' and len(convert.args)==1
        and source._name(convert.args[0])==history and not convert.keywords and type(empty) is ast.Dict and not empty.keys
        and set(item.attr for item in target.elts)<=recursive_fields,'receiver reset domains changed')
    _,set_call = _source_call(facts,code,convert)
    _require(type(root) is ast.Assign and len(root.targets)==2 and type(root.targets[0]) is ast.Name
        and type(root.targets[1]) is ast.Attribute and source._name(root.targets[1].value)==receiver,
        'current root requires one shared local and receiver store')
    root_local,root_field = root.targets
    index = root.value
    _require(type(index) is ast.Subscript and source._name(index.value)==history
        and _expr(index.slice)==dict(op='neg',args=[dict(op='literal',value=1)]), 'root is not the final original history element')
    _,indexed = facts.coordinate('items', index,lambda row:not row.write)
    item,choose = _assignment(table)
    _require(type(item) is ast.Subscript and source._name(item.value)==tablespec['symbol']
        and type(item.slice) is ast.Constant and type(item.slice.value) is str and type(choose) is ast.IfExp
        and type(choose.body) is ast.Name and type(choose.orelse) is ast.Name, 'scoring rows lack an explicit guarded module write')
    _,item_fact = facts.coordinate('items',item,lambda row:row.write)
    gamma_target,gamma_value = _assignment(gamma)
    _require(type(gamma_target) is ast.Name and type(gamma_value) is ast.Constant and type(gamma_value.value) is int,
             'initial bracket guess must be an exact integer')
    return dict(counter=dict(**_attribute_write(facts,code,first),initial=0),
        history=dict(**_attribute_write(facts,code,second),call=set_call,builtin='set',source_formal=history),
        score_table=dict(**_attribute_write(facts,code,third),initial='FRESH_EXACT_EMPTY_DICT'),
        root=dict(**_attribute_write(facts,code,root_field),local=root_local.id,index=-1,original_history_read=indexed),
        table=dict(symbol=tablespec['symbol'],key=item.slice.value,test=_expr(choose.test),true_row=choose.body.id,
            false_row=choose.orelse.id,store=source._event(code,item,'STORE_SUBSCR'),original_item=item_fact,
            versions=tablespec['observed_versions'],version_provider=tablespec['provider']),
        gamma=dict(symbol=gamma_target.id,value=gamma_value.value,store=_store(code,gamma_target)))


def _bracket(facts, code, outer, probe, initialization, receiver, recursive_owner):
    _require(type(outer) is ast.For and type(outer.target) is ast.Name and not outer.orelse and len(outer.body)==3,
             'driver needs a depth for-loop, bracket loop and outer exit check')
    depth = outer.target.id
    ranges = outer.iter
    _require(type(ranges) is ast.Call and source._name(ranges.func)=='range' and not ranges.keywords
        and len(ranges.args)==2 and all(type(arg) is ast.Constant and type(arg.value) is int for arg in ranges.args),
        'depth is not an exact bounded builtin range')
    _,range_ref = _source_call(facts,code,ranges)
    bracket_init,inner,soft = outer.body
    limits,values = _assignment(bracket_init)
    _require(type(limits) is ast.Tuple and type(values) is ast.Tuple and len(limits.elts)==len(values.elts)==2
        and all(type(item) is ast.Name for item in limits.elts) and type(inner) is ast.While and not inner.orelse
        and len(inner.body)==5,'expected one original structurally bounded bracket')
    lower,upper = (node.id for node in limits.elts)
    start,end = values.elts
    _require(type(end) is ast.Name,'bracket upper bound is not a versioned global')
    score_assigned,lo,hi,suspended,advance = inner.body
    score_node,original_probe = _assignment(score_assigned)
    _require(original_probe is probe and type(score_node) is ast.Name and len(probe.args)==3 and len(probe.keywords)==1
        and type(probe.keywords[0].value) is ast.Constant and probe.keywords[0].value.value is True,
        'recursive driver actuals diverge from selected direct probe')
    score, gamma, pos = score_node.id,initialization['gamma']['symbol'],initialization['root']['local']
    _require([source._name(item) for item in probe.args]==[pos,gamma,depth]
        and probe.keywords[0].arg==recursive_owner['parameters'][-1], 'declared recursive formals do not match driver argv')
    _,probe_ref = _source_call(facts,code,probe)
    branch = []
    for block,name,operator in ((lo,lower,'ge'),(hi,upper,'lt')):
        _require(type(block) is ast.If and len(block.body)==1 and not block.orelse
            and _expr(block.test)==dict(op=operator,args=[dict(op='name',name=score),dict(op='name',name=gamma)]),
            'bracket branch must compare actual score and current gamma')
        target,value = _assignment(block.body[0])
        _require(source._name(target)==name and source._name(value)==score,'bracket edge does not move the selected bound')
        branch.append(dict(field=name,test=_expr(block.test),value=score,store=_store(code,target)))
    expression = suspended.value
    _require(type(suspended) is ast.Expr and type(expression) is ast.Yield and type(expression.value) is ast.Tuple
        and len(expression.value.elts)==4,'driver record requires three scalars and one exact move lookup')
    a,b,c,move = expression.value.elts
    _require([source._name(item) for item in (a,b,c)]==[depth,gamma,score] and type(move) is ast.Call
        and type(move.func) is ast.Attribute and move.func.attr=='get' and type(move.func.value) is ast.Attribute
        and source._name(move.func.value.value)==receiver and len(move.args)==1 and source._name(move.args[0])==pos
        and not move.keywords,'yield record does not match original get on current root')
    _,lookup = _source_call(facts,code,move)
    update,value = _assignment(advance)
    _require(source._name(update)==gamma,'advance must redefine the same original gamma')
    yield_site = source._event(code,expression,'YIELD_VALUE')
    resume = source._event(code,expression,'RESUME')
    gamma_store = _store(code,update)
    _require(probe_ref['call']['offset']<branch[0]['store']['offset']<branch[1]['store']['offset']
        <lookup['call']['offset']<yield_site['offset']<resume['offset']<gamma_store['offset'],
        'control order must expose old gamma and updated bounds at yield')
    _require(type(inner.test) is ast.Compare and len(inner.test.ops)==1 and type(inner.test.ops[0]) is ast.Lt
        and source._name(inner.test.left)==lower and type(inner.test.comparators[0]) is ast.BinOp,
        'missing original bracket threshold')
    bound = inner.test.comparators[0]
    _require(type(bound.op) is ast.Sub and source._name(bound.left)==upper and type(bound.right) is ast.Name,
        'bracket test must compare lower against upper minus tolerance')
    result,record = facts.coordinate('containers',expression.value,lambda row:len(row.elements)==4)
    return dict(depth=dict(symbol=depth,from_value=ranges.args[0].value,to_exclusive=ranges.args[1].value,
            builtin='range',iterator=range_ref,advance=source._event(code,outer.iter,'FOR_ITER')),
        lower=lower,upper=upper,gamma=gamma,score=score,sentinel=end.id,roughness=bound.right.id,
        bounds_initial=[_expr(start),_expr(end)],condition=_expr(inner.test),branches=branch),dict(**probe_ref,
        source_recursive_owner=recursive_owner,root_flag=True,formals=[*recursive_owner['parameters'][1:]],
        source_actuals=[pos,gamma,depth,True],score_store=_store(code,score_node)),dict(site=yield_site,resume=resume,
        advance=dict(store=gamma_store,expression=_expr(value)),move_lookup=dict(**lookup,field=move.func.value.attr),
        record=dict(**record,ports=[encode(facts.port(port)) for port in result.elements]),
        record_names=[depth,gamma,score,'move'],bounds_updated_before_yield=True,gamma_updated_only_after_resume=True),soft


def _clock(tree, facts, code, node, receiver):
    _require(type(node) is ast.If and type(node.test) is ast.Compare and len(node.test.ops)==1
        and type(node.test.ops[0]) is ast.Gt and not node.orelse and len(node.body)==1 and type(node.body[0]) is ast.Return
        and node.body[0].value is None,'soft time gate must return before starting the next source depth')
    call,value = node.test.left,node.test.comparators[0]
    _require(type(call) is ast.Call and not call.args and not call.keywords and type(call.func) is ast.Attribute
        and type(value) is ast.Attribute and source._name(value.value)==receiver,'unsupported original soft time driver')
    alias = source._name(call.func.value)
    package = source._one((item.name for stmt in tree.body if type(stmt) is ast.Import for item in stmt.names
        if (item.asname or item.name)==alias),'driver clock must resolve named original import')
    _,anchor = _source_call(facts,code,call)
    return dict(clock=anchor,provider=[package,call.func.attr],deadline_field=value.attr,
                gate=source._span(node.test),return_span=source._span(node.body[0]),scope='BETWEEN_COMPLETED_DEPTHS')


def prepare(path):
    ref = tables._ref(path,recursive.SCHEMA)
    region = recursive.check(ref['uri'])
    tree,codes = source._compile(region['source'])
    owner = region['region']['owner']
    class_name,function,params,probe_node = _find(tree,owner)
    receiver,history = (item.arg for item in params)
    code = codes[class_name+'.'+function.name]
    evidence = read_artifact(region['source_state_observation'])
    flow = read_artifact(region['original_fusion_flow'])
    guard = read_artifact(flow['source_guard'])
    tablespec = guard['bindings']['score_table']
    facts = loop._OriginalFacts(evidence)
    index = RetainedCodeBodyIndex(_view(evidence))
    _,driver_fact,_,_ = _body(index,describe_code(code))
    statements = function.body[1:] if ast.get_docstring(function) is not None else function.body
    _require(len(statements)==5,'original generator driver has additional unmodeled top-level effects')
    initialization = _init(facts,code,statements[:-1],receiver,history,tablespec,
                           {row['field'] for row in region['graph']['receiver_state']})
    bracket,probe,yielded,soft = _bracket(facts,code,statements[-1],probe_node,initialization,receiver,owner)
    clock = _clock(tree,facts,code,soft,receiver)
    yield_instructions = [op for op in get_instructions(code) if op.opname=='YIELD_VALUE']
    _require(len(yield_instructions)==1 and yield_instructions[0].offset==yielded['site']['offset'],
             'driver contains additional suspension points')
    facts.material.check()
    index.check()
    return dict(schema=SCHEMA,schema_version=1,recursive_region=ref,source=region['source'],catalog=region['catalog'],
        driver=dict(code=describe_code(code),receiver=receiver,history_formal=history,function_fact=driver_fact),
        initialization=initialization,probe=probe,bracket=bracket,**{'yield':yielded},
        time_and_exit=dict(soft=clock,hard=dict(scope='INSIDE_ORIGINAL_RECURSIVE_PROBE',
            calls=region['graph']['external_or_unclosed_effects'],exit_sites=[row for row in region['graph']['exits'] if row['kind']=='Raise']),
            suspended_close='CALLER_MAY_CLOSE_OR_THROW_WITHOUT_RUNNING_NORMAL_GAMMA_ADVANCE',
            normal_return_sites=[site(code,op.offset) for op in get_instructions(code) if op.opname.startswith('RETURN')
                                 and op.opname!='RETURN_GENERATOR'],
            wall_clock_outcome_preserved_by_static_proof=False),capture=None,sessions=[],
        qualification=dict(original_compiled_and_f4_bodies_joined=True,external_ui_open=True,
            close_throw_and_clock_effects_complete=False,native_lowering_authorized=False,target_executed_by_projection=False))


def _view(evidence):
    from scar.analysis.program_view import ProgramView
    return ProgramView(Path(evidence['catalog']['uri']).parent)


def project(path,capture_path=None):
    plan = prepare(path)
    if capture_path is None:
        return plan
    reference = tables._ref(capture_path,CAPTURE)
    raw = read_artifact(reference)
    _require(read_artifact(raw['design'])==plan and raw['source']==plan['source']
        and raw['launch']['entry_path']==plan['source']['uri'] and raw['launch']['exit_code']==0
        and raw['original_execution'] is True and raw['performance_measurement'] is False,
        'real capture must bind the exact original driver and successful source execution')
    reuse = _replay(raw['sessions'], plan)
    return dict(plan,capture=reference,sessions=raw['sessions'],state_reuse=reuse,
                observed_frontiers=raw['frontiers'],process_id=raw['process_id'])


def _value(expression,bindings):
    op = expression['op']
    if op=='name':
        return bindings[expression['name']]
    if op=='literal':
        return expression['value']
    operate = {'add':operator.add,'sub':operator.sub,'floor_div':operator.floordiv,'neg':operator.neg}
    _require(op in operate,'unsupported arithmetic in checked original driver bracket')
    return operate[op](*(_value(value,bindings) for value in expression['args']))


def _replay(sessions, plan):
    _require(type(sessions) is list and sessions,'no selected original driver sessions')
    bracket = plan['bracket']
    previous, reuse = {},[]
    for ordinal,observation in enumerate(sessions):
        table = observation['table_version']
        _require(table in plan['initialization']['table']['versions'],'original table version was not admitted')
        leaves = observation['yields']
        initial, initialized, final = (observation[key] for key in ('initial','initialized','final'))
        _require(initialized['nodes']==0 and initialized['history']['identity']!=initial['history']['identity']
            and 0<initialized['history']['size']<=observation['history_input_size']
            and initialized['score_table']['size']==0
            and initialized['score_table']['identity']!=initial['score_table']['identity']
            and initialized['move_table']==initial['move_table']
            and initialized['root']==observation['root_input']==final['root'],
            'actual receiver initialization diverges before its original first recursive probe')
        prev_depth, gamma = None,plan['initialization']['gamma']['value']
        nodes,sequence = 0,observation['begin_sequence']
        for rank,row in enumerate(leaves):
            depth,guess,score,move = row['value']
            _require(all(type(item) is int for item in (depth,guess,score,row['lower'],row['upper'],row['nodes']))
                and (move is None or type(move) is list) and row['probe']['result']==score
                and row['probe']['actuals']==[guess,depth,True] and row['nodes']>nodes and guess==gamma
                and sequence<row['probe']['sequence']<row['probe']['return_sequence']<row['sequence']
                and row['probe']['same_receiver'] is row['probe']['same_root_position'] is True
                and row['site']==plan['yield']['site'] and prev_depth in (None,depth-1,depth),
                'original yielded record differs from its probe or carried bracket')
            if prev_depth!=depth:
                _require(prev_depth is not None or depth==bracket['depth']['from_value'],'unexpected first source depth')
                lower,upper = (_value(value,observation['module_i64']) for value in bracket['bounds_initial'])
            _require(lower < upper-observation['module_i64'][bracket['roughness']],'probe ran beyond its source bracket')
            if score>=guess:
                lower=score
            else:
                upper=score
            _require((row['lower'],row['upper'])==(lower,upper),'original yield occurs before bounds or after gamma update')
            _require(all(row['resources'][key]['identity']==initialized[key]['identity'] for key in ('score_table','move_table')),
                     'shared table holder was changed inside the original driver')
            if rank+1<len(leaves) or row['resume'] is not None:
                _require(row['resume']==plan['yield']['resume'],'yield resumed at a different original statement')
            gamma=_value(plan['yield']['advance']['expression'],{bracket['lower']:lower,bracket['upper']:upper})
            prev_depth,nodes,sequence = depth,row['nodes'],row['sequence']
        completion = observation['exit']
        _require(bool(leaves) and final['nodes']>=nodes and completion['sequence']>sequence
            and completion['kind'] in ('NORMAL_RETURN','GENERATOR_CLOSE','EXCEPTION_UNWIND'),
                 'original driver executor did not publish its termination kind')
        if completion['kind']=='GENERATOR_CLOSE':
            _require(leaves[-1]['resume'] is None and completion['thrown']['type']=='GeneratorExit'
                and completion['thrown']['site']==plan['yield']['resume'],
                'original close did not interrupt the normal post-yield resume at its source site')
        receiver = (observation['thread_id'],observation['identity'])
        if receiver in previous:
            previous_ordinal,old = previous[receiver]
            if old['final']['move_table']==initial['move_table']:
                reuse.append(dict(before_session=previous_ordinal,after_session=ordinal,
                    shared_receiver=receiver[1],move_table=initial['move_table'],
                    score_table_reset=initialized['score_table']['identity']!=old['initialized']['score_table']['identity'],
                    same_score_table_root=old['table_identity']==observation['table_identity'],
                    version_before=old['table_version'],version_after=observation['table_version']))
        previous[receiver] = ordinal,observation
    return reuse


def check(path):
    reference = tables._ref(path, SCHEMA)

    def replay():
        result = read_artifact(reference)
        capture = result['capture']
        if capture is not None:
            read_artifact(capture)
        _require(result == project(result['recursive_region']['uri'], None if capture is None else capture['uri']),
                 'stored driver differs from independent original source replay')
        return result

    return detached_replay(SCHEMA, tuple(sorted(reference.items())), replay)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recursive-region',type=Path)
    parser.add_argument('--capture',type=Path)
    parser.add_argument('--check',type=Path)
    parser.add_argument('--out',type=Path)
    args = parser.parse_args()
    if args.check:
        result = check(args.check)
        print(json.dumps(dict(sessions=len(result['sessions']),probe=result['probe']['call']['offset'])))
    else:
        if not (args.recursive_region and args.out):
            parser.error('publication requires --recursive-region and --out')
        print(json.dumps(tables._write(args.out,project(args.recursive_region,args.capture))))


if __name__ == '__main__':
    main()
