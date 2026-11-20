"""Validate a bounded source owner window across two original recursive searches."""

import argparse
import ast
from dis import get_instructions
import json
from pathlib import Path

from scar.analysis import scalar_recursive_driver as driver
from scar.analysis import scalar_recursive_region as recursive
from scar.analysis import scalar_score_reuse_evidence as source
from scar.analysis.scalar_table_handoff import _ref, _write
from scar.ir.execution_recipe import read_artifact
from scar.trace.code_snapshot import describe_code, site

SCHEMA = 'scar.scalar-recursive-owner-continuity'
DESIGN = 'scar.scalar-recursive-owner-continuity-design'
CAPTURE = 'scar.scalar-recursive-owner-continuity-capture'
UI = 'scar.scalar-recursive-ui-contract'


def _require(ok,reason):
    if not ok:
        raise ValueError('scalar recursive owner continuity: '+reason)


def _name(node):
    return source._name(node)


def _nodes(fn):
    return [item for stmt in fn.body for item,_ in recursive._operations(stmt)]


def _launcher(functions, driver_plan):
    owner_class = driver_plan['driver']['code']['qualname'].split('.')[0]
    choices = []
    for name,fn in functions.items():
        for node in _nodes(fn):
            if type(node) is not ast.Assign or len(node.targets)!=1 or _name(node.targets[0]) is None:
                continue
            call = node.value
            if type(call) is ast.Call and type(call.func) is ast.Attribute and call.func.attr==owner_class:
                choices.append((name,_name(node.targets[0])))
    name,receiver = source._one(sorted(set(choices)), 'one original controller owner allocation binding expected')
    fn = functions[name]
    launch = source._one((node for node in _nodes(fn) if type(node) is ast.Call
        and type(node.func) is ast.Attribute and node.func.attr=='submit' and len(node.args)>1
        and _name(node.args[0]) and _name(node.args[1])==receiver), 'one original owner submission expected')
    return name,receiver,launch


def _params(fn):
    return [node.arg for node in fn.args.posonlyargs+fn.args.args]


def _call(code,node):
    return source._one((site(code,op.offset) for op in get_instructions(code)
        if op.opname in ('CALL','CALL_KW','CALL_FUNCTION_EX') and site(code,op.offset)['position']==source._span(node)),
        'original controller expression has no unique compiled call')


def _after(code,boundary):
    operations = list(get_instructions(code))
    ordinal = next(i for i,op in enumerate(operations) if op.offset==boundary['offset'])
    return site(code,operations[ordinal+1].offset)


def _command_port(fn,code):
    rows = []
    for node in _nodes(fn):
        if type(node) is not ast.Assign or len(node.targets)!=1 or _name(node.targets[0]) is None:
            continue
        split = node.value
        if (type(split) is ast.Call and type(split.func) is ast.Attribute and split.func.attr=='split'
                and type(split.func.value) is ast.Call and _name(split.func.value.func)=='input'
                and not (split.args or split.keywords or split.func.value.args or split.func.value.keywords)):
            rows.append(node.targets[0])
    token = source._one(rows,'one original input-tokenization assignment required')
    store = source._event(code,token,'STORE_FAST')
    return dict(symbol=token.id,store=store,after=_after(code,store))


def _routes(functions, selectors):
    edges = []
    for _ in range(len(functions)):
        added = False
        for name,actual in tuple(selectors.items()):
            for node in _nodes(functions[name]):
                if type(node) is not ast.Call or _name(node.func) not in functions:
                    continue
                target = _name(node.func)
                for ordinal,arg in enumerate(node.args):
                    if _name(arg)!=actual:
                        continue
                    formals = _params(functions[target])
                    _require(ordinal<len(formals),'original local owner actual requires a direct formal')
                    formal = formals[ordinal]
                    _require(target not in selectors or selectors[target]==formal,'conflicting local owner formal binding')
                    if target not in selectors:
                        selectors[target] = formal
                        added = True
                    row = dict(source=name,target=target,formal=formal,span=source._span(node),actual=ordinal)
                    if row not in edges:
                        edges.append(row)
        if not added:
            return edges
    raise ValueError('owner scope propagation exceeded original module size')


def _accesses(functions,codes,selectors,source_fields,route,launcher,submit):
    table_by_name = {field:role for role,field in source_fields.items()}
    rows = []
    for name,fn in functions.items():
        owner = selectors.get(name)
        parents = {child:item for item in _nodes(fn) for child in ast.iter_child_nodes(item)}
        nodes = _nodes(fn)
        for node in nodes:
            if type(node) is ast.Attribute and node.attr in table_by_name:
                role = table_by_name[node.attr]
                direct = owner is not None and _name(node.value)==owner
                parent = parents.get(node)
                use = parents.get(parent)
                kind = 'UNRESOLVED_OWNER' if not direct else 'DIRECT_ESCAPE'
                witness = node
                if direct and type(node.ctx) is ast.Load:
                    if type(parent) is ast.Attribute and parent.value is node and parent.attr=='get' and type(use) is ast.Call and use.func is parent and len(use.args)==1 and not use.keywords:
                        kind,witness = 'KEYED_GET',use
                    if type(parent) is ast.Call and _name(parent.func)=='len' and len(parent.args)==1 and not parent.keywords:
                        kind,witness = 'LENGTH_READ',parent
                if direct and type(node.ctx) in (ast.Store,ast.Del):
                    kind = 'DIRECT_FIELD_WRITE'
                opcode = 'STORE_ATTR' if type(node.ctx) is ast.Store else 'DELETE_ATTR' if type(node.ctx) is ast.Del else 'LOAD_ATTR'
                point = _call(codes[name],witness) if witness is not node else source._event(codes[name],witness,opcode)
                rows.append(dict(owner=name,receiver=owner,role=role,kind=kind,site=point))
            if type(node) is not ast.Call or owner is None:
                continue
            actuals = [i for i,arg in enumerate(node.args) if _name(arg)==owner]
            if (actuals==[0] and _name(node.func)=='hasattr' and len(node.args)==2 and not node.keywords and
                    type(node.args[1]) is ast.Constant and node.args[1].value in table_by_name):
                rows.append(dict(owner=name,receiver=owner,role=table_by_name[node.args[1].value],
                    kind='PRESENCE_TEST',site=_call(codes[name],node)))
            elif actuals and not (name==launcher and node is submit) and not any(
                    row['source']==name and row['span']==source._span(node) for row in route):
                rows.append(dict(owner=name,receiver=owner,role=None,kind='UNKNOWN_RECEIVER_CALL',
                    site=_call(codes[name],node)))
        for node in nodes:
            if type(node) is not ast.Name or _name(node)!=owner or type(node.ctx) is not ast.Load:
                continue
            parent = parents.get(node)
            if type(parent) in (ast.Attribute,ast.Call):
                continue
            candidates = [site(codes[name],op.offset) for op in get_instructions(codes[name])
                          if op.opname.startswith(('LOAD_FAST','LOAD_DEREF'))
                          and site(codes[name],op.offset)['position']==source._span(node)]
            if candidates:
                rows.append(dict(owner=name,receiver=owner,role=None,kind='DIRECT_OWNER_ESCAPE',
                                 site=source._one(candidates,'owner alias read must select one compiled source location')))
    rows.sort(key=lambda row:(row['owner'],row['site']['offset'],row['kind']))
    _require(len({(row['owner'],row['site']['offset']) for row in rows})==len(rows),
             'one compiled access cannot borrow another access contract')
    return rows


def prepare(ui_path):
    ui_ref = _ref(ui_path,UI)
    ui = read_artifact(ui_ref)
    original = driver.check(ui['source_driver']['uri'])
    _require(read_artifact(ui['source_driver'])==original and original['capture'] is not None,
             'owner continuity needs the exact source driver observation')
    tree,codes = source._compile(ui['ui_source'])
    functions = {node.name:node for node in tree.body if type(node) is ast.FunctionDef}
    consumer = ui['consumer']['qualname']
    relay = ui['relay']['qualname']
    _require(all(describe_code(codes[ui[role]['qualname']])==ui[role] for role in ('consumer','relay')),
             'original source consumer or relay identity differs')
    launcher,receiver,submit = _launcher(functions,original)
    scopes = {consumer:ui['receiver'],launcher:receiver}
    routes = _routes(functions,scopes)
    _require(relay in scopes,'original relay did not receive the same lexical owner')
    fields = dict(move_table=original['yield']['move_lookup']['field'],
                  score_table=original['initialization']['score_table']['field'])
    access = _accesses(functions,codes,scopes,fields,routes,launcher,submit)
    descriptors = {name:describe_code(codes[name]) for name in set(scopes)|{row['owner'] for row in access}}
    reset = original['initialization']['score_table']['store']
    _, engine = source._compile(original['source'])
    code = engine[original['driver']['code']['qualname']]
    return dict(schema=DESIGN,schema_version=1,ui_contract=ui_ref,source_driver=ui['source_driver'],
        source=original['source'],ui_source=ui['ui_source'],roles=fields,
        static=dict(scopes=descriptors,owner_formals=scopes,owner_routes=routes,access_sites=access,
            launcher=dict(name=launcher,receiver=receiver,callable_local=_name(submit.args[0]),submit=_call(codes[launcher],submit),
                command=_command_port(functions[launcher],codes[launcher])),
            source_consumer=consumer,relay=relay,driver=original['driver'],probe=original['probe']['call'],
            reset=dict(site=reset,after=_after(code,reset),fact=original['initialization']['score_table']['f4']),
            proof_scope='STATIC_EXACT_SOURCE_ROUTES_AND_BOUNDED_RUNTIME_OCCURRENCES',
            unresolved=['REFLECTIVE_OR_FOREIGN_RECEIVER_ALIASES','CALLBACKS_OUTSIDE_ORIGINAL_CODE_FAMILY',
                        'OTHER_COMMAND_BRANCHES_AND_UNKNOWN_DYNAMIC_CALLS']))


def _check_window(window,design,original):
    anchors = window['anchors']
    phases = ['close','worker_return','submit','consumer_start','driver_start','fresh_score_reset','first_probe']
    seq = [anchors[event]['sequence'] for event in phases]
    thread = anchors['close']['thread_id']
    owner = window['owner_identity']
    _require(type(owner) is int and owner>0 and all(anchors[name]['actor']==owner for name in phases)
        and type(thread) is int and thread>0
        and all(anchors[name]['thread_id']==thread for name in phases if name!='submit')
        and anchors['submit']['thread_id']!=thread
        and all(type(value) is int and value>0 for value in seq) and seq==sorted(set(seq)),
             'original receiver handoff events must occupy one ordered window')
    graph = design['static']
    consumer = graph['scopes'][graph['source_consumer']]
    body = graph['driver']['code']
    _require(anchors['close']['site']['code']==body['key'] and anchors['close']['exception']=='GeneratorExit'
        and anchors['driver_start']['site']['code']==body['key']
        and all(anchors[name]['site']['code']==consumer['key'] for name in ('worker_return','consumer_start'))
        and anchors['submit']['site']==graph['launcher']['submit'] and anchors['submit']['target']==consumer
        and anchors['fresh_score_reset']['site']==graph['reset']['after'] and anchors['fresh_score_reset']['write']==graph['reset']['site']
        and anchors['first_probe']['caller']==graph['probe']
        and anchors['first_probe']['site']['code']==original['probe']['source_recursive_owner']['key'],
        'owner lifecycle is absent from the bound original sites or switched code')
    views = [anchors[name]['holders'] for name in ('close','submit','driver_start','fresh_score_reset','first_probe')]
    moves = [row['move_table']['identity'] for row in views]
    scores = [row['score_table']['identity'] for row in views]
    _require(len(set(moves))==1 and len(set(scores[:3]))==1 and scores[3]==scores[4]!=scores[0]
        and views[3]['score_table']['size']==views[4]['score_table']['size']==0
        and len({row['move_table']['ordered_sha256'] for row in views})==1
        and len({row['score_table']['ordered_sha256'] for row in views[:3]})==1,
        'move holder continuity or fresh original score reset is incomplete')
    first,second = original['sessions']
    _require(all(views[0][role]['size']==first['final'][role]['size'] for role in design['roles'])
        and all(views[0][role]['ordered_sha256']==first['final']['ordered_map_digests'][field]
                for role,field in design['roles'].items())
        and views[2]['move_table']['ordered_sha256']==second['initial_move_digest'],
        'inter-search snapshots differ from the qualified original source transcript')
    static = {(row['owner'],row['site']['offset']):row for row in design['static']['access_sites']}
    evidence,unknown = [],[]
    last = seq[0]
    for event in window['events']:
        row = static[event['owner'],event['site']['offset']]
        _require(event['sequence']>last and seq[0]<event['sequence']<seq[-1]
            and event['site']==row['site'] and event['kind']==row['kind'] and event['role']==row['role'],
            'runtime state access must be an ordered selected original source site')
        if event['kind'] in ('KEYED_GET','LENGTH_READ','PRESENCE_TEST') and event['exact_normal_builtin'] is True:
            identity = moves[0] if row['role']=='move_table' else scores[0 if event['sequence']<seq[5] else 3]
            _require(event['table_identity']==identity,'bounded runtime table read changed exact holder')
            evidence.append(event)
        else:
            unknown.append(event)
        last = event['sequence']
    return dict(anchors=anchors,owner_identity=window['owner_identity'],events=evidence,unsupported_events=unknown,
        permissions=dict(move_table_keyed_get=any(row['role']=='move_table' and row['kind']=='KEYED_GET' for row in evidence),
            directly_observed_lengths=sorted({row['role'] for row in evidence if row['kind']=='LENGTH_READ'}),
            direct_state_escape_observed=any(row['kind'] in ('DIRECT_OWNER_ESCAPE','DIRECT_ESCAPE') for row in unknown),
            selected_accesses_observed_without_unsupported_occurrences=not unknown,
            fresh_score_reset=True,move_holder_retained=True,scope='ONLY_ORIGINAL_BOUNDED_OBSERVED_WINDOW',
            live_foreign_aliases_proven_absent=False,deployment_authorized=False))


def project(ui_path,capture_path):
    plan = prepare(ui_path)
    raw_ref = _ref(capture_path,CAPTURE)
    raw = read_artifact(raw_ref)
    _require(read_artifact(raw['design'])==plan and raw['source']==plan['source']
        and raw['launch']['entry_path']==plan['source']['uri'] and raw['launch']['exit_code']==0
        and raw['original_execution'] is True and raw['performance_measurement'] is False
        and type(raw['process_id']) is int and raw['process_id']>0 and len(raw['windows'])==1,
        'completed original owner window capture is required')
    stdin = read_artifact(raw['stdin'],json_document=False).decode('ascii')
    _require(stdin==''.join(' '.join(row)+'\n' for row in raw['input_tokens']),
             'actual original parsed commands differ from the pinned input stream')
    original = read_artifact(plan['source_driver'])
    _require(len(original['sessions'])==2,'continuity case must bind two original source sessions')
    windows = [_check_window(window,plan,original) for window in raw['windows']]
    return dict(schema=SCHEMA,schema_version=1,ui_contract=plan['ui_contract'],source_driver=plan['source_driver'],
        capture=raw_ref,stdin=raw['stdin'],roles=plan['roles'],static=plan['static'],windows=windows,
        process_id=raw['process_id'],frontiers=raw['frontiers'],qualification=dict(target_executed_by_projection=False,
            absence_of_runtime_unknown_events_cannot_prove_static_alias_closure=True,whole_source_covered=False))


def check(path):
    result = read_artifact(_ref(path,SCHEMA))
    read_artifact(result['capture'])
    _require(result==project(result['ui_contract']['uri'],result['capture']['uri']),
             'owner continuity differs from independent original replay')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--ui-contract',type=Path)
    group.add_argument('--check',type=Path)
    parser.add_argument('--capture',type=Path)
    parser.add_argument('--out',type=Path)
    args = parser.parse_args()
    if args.check:
        report = check(args.check)
        print(json.dumps(dict(windows=len(report['windows']))))
    else:
        if args.out is None:
            parser.error('publication requires --out')
        result = prepare(args.ui_contract) if args.capture is None else project(args.ui_contract,args.capture)
        print(json.dumps(_write(args.out,result)))


if __name__ == '__main__':
    main()
