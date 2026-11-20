"""Bind observed filled-contour and text ports to the original controller graph."""

import argparse
import ast
from collections import defaultdict
import gzip
import json
from pathlib import Path
import re

from scar.analysis.source_agg_text import SCHEMA as TEXT, _coordinates, _format_route, _source
from scar.analysis.source_controller_effects import SCHEMA as EFFECTS
from scar.analysis.source_controller_initialization import _walk
from scar.analysis.source_syntax import _flatten_ast
from scar.compiler.plan_contour_scene import SCHEMA as CONTOUR, read_contour_scene_plan
from scar.compiler.plan_controller_startup import SCHEMA as STARTUP
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.literals import PythonLiteral
from scar.ir.record_codec import decode


SCHEMA = 'scar.source-controller-plot'


def _require(condition, message):
    if not condition:
        raise ValueError('controller plot: ' + message)


def _only(rows, message):
    rows = tuple(rows)
    _require(len(rows) == 1, message)
    return rows[0]


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _names(node):
    if type(node) is ast.Name:
        return [node.id]
    if type(node) is ast.Attribute:
        return [*_names(node.value), node.attr]
    return []


def _value(port):
    return port['port']['template']['occurrence']


def _fact(reference, table):
    namespace, ordinal = reference['locator'].rsplit('/',1)
    _require(namespace == '#/'+table and ordinal.isdecimal(), 'wrong original F4 fact table')
    unit = json.loads(gzip.decompress(read_artifact(reference['artifact'],json_document=False)))
    return unit,unit[table][int(ordinal)]


class _Syntax:
    def __init__(self, source, body):
        raw, _ = _source(source)
        tree = ast.parse(raw, filename=source['material']['uri'])
        _, self.occurrences = _flatten_ast(tree)
        self.nodes = {(type(node).__name__, _coordinates(node)): node for node in ast.walk(tree)
                      if hasattr(node, 'lineno')}
        self.parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        self.function = self.node(body['source_function'])
        self.body = body
        self.calls = {self.node(row['source_result']):row for row in body['calls']}
        _require(type(self.function) is ast.FunctionDef, 'original effects require one synchronous function')

    def node(self, port):
        row = self.occurrences[_value(port)]
        attrs = dict(zip(('lineno','col_offset','end_lineno','end_col_offset'),
                         (item.scalar.to_python() for item in row.attributes), strict=True))
        span = tuple(attrs[key] for key in ('lineno','end_lineno','col_offset','end_col_offset'))
        return self.nodes[row.constructor,span]

    def call(self, node, *, observed=True):
        row = self.calls.get(node)
        _require(row is not None and (not observed or row['observations'])
                 and row['positional_pack'] is None and row['keyword_pack'] is None,
                 'projection needs an exact original F4 CALL with direct operands')
        return row

    def field(self, node, kind):
        row = _only((row for row in self.body['field_accesses'] if row['kind'] == kind
                      and self.node(row['value']) is node), 'original ordinary field access must be unique')
        return row


def _body_and_effect(effects, source):
    body = _only((row for row in effects['bodies'] if row['source']['code_key'] == source['code_key']
                  and row['source']['sha256'] == source['material']['sha256']),
                 'geometry and title require one original source effect body')
    effect = _only((row for row in effects['effects'] if body['body'] in row['bodies']),
                   'source plot body needs one original controller CALL')
    observations = [row for row in effect['observations'] if row['body'] == body['body']
                    and row['target']['code'] == source['code_key'] and row['normal_completion_observed']]
    def signature(row):
        return [{k:v for k,v in p.items() if k!='actual_frame_reference'} for p in row['formal_routes']]

    _require(bool(observations) and all(signature(row)==signature(observations[0]) for row in observations),
             'controller snapshots require identical exact formal-interface routing')
    formals = {}
    for row in observations[0]['formal_routes']:
        route = _only(row['routes'], 'one original route per renderer formal required')
        _require(route['formal'] == row['formal'] and len(route['actuals']) == 1,
                 'renderer formal and original actual differ')
        formals[row['name']] = dict(formal=row['formal'], actual=route['actuals'][0], route=route['kind'],
                                    declared_default=row['declared_default'])
    return body,effect,observations,formals


def _binding(formals, effect, control):
    records = list(_walk(control['program']))
    call = _only((row for row in records if row['kind']=='CALL' and row.get('effect')==effect['id']
                 and row['site']==effect['site']), 'controller IR must contain the original snapshot CALL')
    producers = {row['id']:row for row in records if row['kind']=='EVAL'}
    callee = producers[call['callee']['id']]['expression']
    _require(callee['kind']=='FIELD', 'snapshot call requires original ordinary method receiver')
    receiver = producers[callee['receiver']['id']]
    slot = receiver['expression']['slot']
    _require(receiver['expression']['kind']=='SLOT' and len(call['positional']) == len(effect['original_call']['positional'])
             and not call['keywords'] and not effect['original_call']['keywords'],
             'original receiver or direct positional call contract differs')
    route = _only((row for row in control['parameters'] if row['slot']==slot),
                 'snapshot receiver requires one original controller parameter')
    unit,attribute = _fact(callee['source_fact'],'attributes')
    _require(attribute['action']=='GET' and attribute['name']==callee['field']
             and dict(unit=unit['unit'],port=unit['ports'][attribute['value']])
                 in producers[call['callee']['id']]['origin']['results'],
             'controller method lookup does not replay to its original F4 result')
    original_receiver = dict(unit=unit['unit'],port=unit['ports'][attribute['receiver']])
    original_pos = effect['original_call']['positional']
    for formal in formals.values():
        if formal['route']=='INITIAL_DEFAULT':
            continue
        _require(formal['route']=='ARGUMENT', 'unsupported original snapshot formal route')
        if formal['actual'] in original_pos:
            index = original_pos.index(formal['actual'])
            actual = producers[call['positional'][index]['id']]
            _require(actual['expression']['kind']=='SLOT', 'snapshot scalar is not an original lexical load')
            formal.update(kind='CONTROLLER_SCALAR', slot=actual['expression']['slot'],
                          argument_ordinal=index, producer=actual['origin'])
        else:
            _require(formal['actual']==original_receiver,
                     'snapshot receiver formal does not consume original method receiver')
            formal.update(kind='CONTROLLER_RECEIVER', slot=slot, producer=receiver['origin'])
    name = _only((name for name,p in formals.items() if p.get('kind')=='CONTROLLER_RECEIVER'),
                 'one ordinary snapshot receiver required')
    return name,route,call


def _node_path(node, src, dst):
    path = _names(node)
    _require(len(path)==2 and path[0]==src, 'requires an exact ordinary receiver field')
    return [dst,path[1]]


def _draw_bindings(syntax, body, lineage, source, receiver, controller_receiver, startup, update):
    _require(lineage and set(lineage['bindings']) == {'x','y','z'} and lineage['code_key']==source['code_key']
             and lineage['source']==source['material'] and lineage['sampled_alias_relation']
             and lineage['normalization_same_observed_values'], 'requires original observed field lineage')
    bindings, selectors = {}, []
    native = _only((row for row in _walk(update['controller']['program']) if row['kind']=='CALL'
                   and row.get('role')=='NATIVE_UPDATE'), 'one original numerical update required')
    update_receiver = _only((row for row in native['formal_routes']
                     if row['binding']['role']=='ATTRIBUTE_RECEIVER'),
                     'requires original numerical update receiver')
    _require(update_receiver['binding']['source_path']==[controller_receiver],
             'controller plot and update must share the original receiver root')
    active = {(controller_receiver,*row['parent_path'][1:]):row for row in update['update']['array_inputs']
              if row['parent_path'][0]==update_receiver['parameter']}
    fixed = {tuple(row['path']):row for row in startup['materialization']['outputs'] if row['kind']=='ARRAY'}
    for name in ('x','y','z'):
        anchor = lineage['bindings'][name]
        path = anchor['field_path']
        _require(path[0]==receiver and len(path)==2, 'original contour field must derive from controller receiver')
        item = _only((r for r in body['item_accesses'] if r['kind']=='READ' and r['selector'] is not None
                      and any(p['root']==update['formals'][receiver]['formal'] and
                              p['selectors']==[{'name':path[1]}] and p['frontier'] is None
                              for p in r['receiver_paths'])
                      and type(syntax.node(r['value'])) is ast.Subscript),
                     'contour input needs one original F4 ordinary source subscript')
        subscript = syntax.node(item['value'])
        parts = subscript.slice.elts if type(subscript.slice) is ast.Tuple else (subscript.slice,)
        _require(_names(subscript.value)==path and len(parts)==len(anchor['root_shape']),
                 'original contour slice root or rank differs')
        limits = []
        for part, size in zip(parts,anchor['root_shape'],strict=True):
            _require(type(part) is ast.Slice, 'original contour operand requires slice-only view')
            bounds = [None if bound is None else ast.literal_eval(bound)
                      for bound in (part.lower,part.upper,part.step)]
            _require(all(value is None or type(value) is int for value in bounds),
                     'original contour slice requires literal integer bounds')
            limits.append(slice(*bounds).indices(size))
        _require([first for first,_,_ in limits]==anchor['start']
                 and [len(range(*limit)) for limit in limits]==anchor['shape']
                 and all(stride==1 for _,_,stride in limits), 'source slice and actual contour lineage differ')
        owner = (controller_receiver,path[1])
        material = fixed[owner]
        geometry = material['geometry']
        _require(geometry['shape']==anchor['root_shape'] and geometry['byte_strides']==anchor['byte_strides']
                 and geometry['dtype']=='<f8' and geometry['itemsize']==8 and geometry['byte_offset']==0,
                 'native startup root and original contour lineage metadata differ')
        current = active.get(owner)
        if current:
            _require(current['id']==material['id'] and current['shape']==geometry['shape'],
                     'current native controller value must match original numeric materialization')
        bindings[name] = dict(**anchor, controller_path=list(owner), id=material['id'],
                kind='CONTROLLER_ARRAY' if current else 'STARTUP_ASSET', geometry=geometry,
                source_slice=item['source_fact'], original_item_value=item['value'],
                startup_material=material['source'], current_array=None if not current else current['id'],
                original_live_field_bindings_required=True)
        selectors.append(item['selector'])
    chart = _only((row for row in body['calls'] if row['observations']
             and len(row['positional_paths'])==len(selectors)
             and all(any({'key':selector} in path['selectors'] for path in paths)
                     for selector,paths in zip(selectors,row['positional_paths'],strict=True))),
             'observed contour provider needs original source view routes')
    return bindings,chart


def _field_path(row, formals, receiver, field):
    return any(path['root']==formals[receiver]['formal'] and
               path['selectors'] in ([],[{'name':field}]) and path['frontier'] is None
               for path in row['receiver_paths'])


def _increment(old,write,increment):
    artifact = write['source_fact']['artifact']
    unit,store = _fact(write['source_fact'],'attributes')
    _require(old['source_fact']['artifact']==artifact, 'integer field update crosses original source units')
    load = unit['attributes'][int(old['source_fact']['locator'].rsplit('/',1)[1])]
    ordinal,operator = _only(((i,row) for i,row in enumerate(unit['operations'])
                              if row['result']==store['value'] and row['result_world']==store['world']),
                             'counter store needs one original F4 operator result and world')
    _require(load['action']=='GET' and store['action']=='SET' and load['receiver']==store['receiver']
             and operator['action']=='INPLACE_BINARY' and operator['selector']=='Add'
             and len(operator['operands'])==2 and operator['operands'][0]==load['value'],
             'counter needs original integer read/add/rebind data edges')
    aliases = defaultdict(list)
    for start,end in unit['aliases']:
        aliases[end].append(start)
    operand = operator['operands'][1]
    reachable = set()
    while operand not in reachable:
        reachable.add(operand)
        if len(aliases[operand])!=1:
            break
        operand = aliases[operand][0]
    literal = _only(((i,row) for i,row in enumerate(unit['constants']) if row['value'] in reachable),
                    'counter increment needs one original F4 literal operand')
    observed = decode(PythonLiteral,literal[1]['literal']).to_python()
    _require(type(observed) is int and observed==increment, 'original integer increment differs')
    return dict(operator=dict(artifact=artifact,locator='#/operations/'+str(ordinal)),
                literal=dict(artifact=artifact,locator='#/constants/'+str(literal[0])))


def _output(syntax, startup, source_slice, formals, receiver, controller_receiver):
    statements = syntax.function.body
    candidates = [(index,node.value) for index,node in enumerate(statements)
                  if type(node) is ast.Expr and type(node.value) is ast.Call
                  and type(node.value.func) is ast.Attribute and node.value.func.attr=='savefig']
    index, save = _only(candidates, 'one original ordered Figure.savefig source operation required')
    _require(len(save.args)==1 and type(save.args[0]) is ast.Call,
             'original save needs one directly joined pathname')
    join = save.args[0]
    _require(_names(join.func)[-2:]==['path','join'] and len(join.args)==2 and not join.keywords
             and type(join.args[1]) is ast.JoinedStr, 'original pathname requires direct OS path + f-string')
    directory,out_expression = join.args
    directory_path = _node_path(directory,receiver,controller_receiver)
    route = _only((row for row in startup['runtime_routes'] if row['path']==directory_path and row['kind']=='str'),
                  'output directory requires the original runtime field ingress')
    parts = out_expression.values
    number = _only((part for part in parts if type(part) is ast.FormattedValue),
                   'original filename needs one exact integer formatting operation')
    counter_path = _node_path(number.value,receiver,controller_receiver)
    _require(number.conversion==-1 and type(number.format_spec) is ast.JoinedStr
             and len(number.format_spec.values)==1 and type(number.format_spec.values[0]) is ast.Constant,
             'original filename needs a plain static integer specifier')
    spec = number.format_spec.values[0].value
    _require(type(spec) is str and re.fullmatch(r'0[1-9][0-9]?d',spec),
             'native filename profile needs an original zero-padded integer specifier')
    split = parts.index(number)
    strings = [part for part in parts if part is not number]
    _require(all(type(part) is ast.Constant and type(part.value) is str and part.value.isascii()
                 and '/' not in part.value and '\x00' not in part.value for part in strings),
             'original filename needs literal ASCII basename text')
    filename = dict(prefix=''.join(part.value for part in parts[:split]),
                    suffix=''.join(part.value for part in parts[split+1:]),
                    format=spec, minimum_width=int(spec[1:-1]), field=counter_path)
    cleanup,successor = statements[index+1:index+3]
    _require(type(cleanup) is ast.Expr and type(cleanup.value) is ast.Call
             and type(successor) is ast.AugAssign and _names(successor.target)==_names(number.value)
             and type(successor.op) is ast.Add and type(successor.value) is ast.Constant
             and type(successor.value.value) is int and successor.value.value>0
             and index+3==len(statements),
             'normal original save and cleanup must precede one terminal integer field update')
    field = counter_path[-1]
    reads = syntax.field(number.value,'GET')
    old = syntax.field(successor,'GET')
    write = syntax.field(successor,'SET')
    for access in (reads,old,write):
        _require(access['selector']==field and _field_path(access,formals,receiver,field),
                 'filename read and ordered integer write must address the original field')
    others = [row for row in syntax.body['field_accesses'] if row['kind']=='SET' and row['selector']==field]
    _require(others==[write], 'counter may only be written by the original terminal update')
    initial = _only((row for row in startup['materialization']['outputs']
                     if row['path']==counter_path and row['kind']=='SCALAR'),
                    'original counter requires one source-backed startup output')
    proof = _only((row for row in source_slice['ports'] if row['controller_path']==counter_path),
                  'one exact counter constructor source slice required')
    constant = _only(proof['declaration_literals'],'original counter must establish one initial literal')
    literal = decode(PythonLiteral,constant['literal']).to_python()
    _require(type(literal) is int and literal==initial['expected']['value']
             and not any(proof[k] for k in ('supplied_inputs','declared_defaults','provider_calls',
                                             'open_value_sources','control_guards')),
             'original counter initial value must use its one source literal')
    final_calls = [syntax.call(node) for node in (join,save,cleanup.value)]
    return dict(directory=route, filename=filename, save=final_calls[1], join=final_calls[0],
                cleanup=final_calls[2]), dict(path=counter_path, initial=literal,
        increment=successor.value.value, initial_producer=constant, before_save=reads,
        increment_read=old, after_save_and_cleanup_write=write,
        source_increment=_increment(old,write,successor.value.value),
        order='ORIGINAL_ADJACENT_NORMAL_STATEMENTS', integer_range_guard='BUILTIN_INT_TO_CHECKED_I64')


def _control_guard(syntax, chart, title, formals):
    left,right = (syntax.node(row) for row in (chart['source_result'],title['call_result']))
    def chain(node):
        while node in syntax.parents:
            parent = syntax.parents[node]
            if type(parent) is ast.If and type(parent.test) is ast.Name:
                yield parent, any(node is statement for statement in parent.body)
            node = parent
    branch,selected = _only((row for row in chain(left) if row in tuple(chain(right))
                             and row[0].test.id in formals), 'one original common plot branch required')
    name = branch.test.id
    defaults = dict(zip((node.arg for node in syntax.function.args.kwonlyargs),
                        syntax.function.args.kw_defaults,strict=True))
    initial = defaults[name]
    _require(formals[name]['route']=='INITIAL_DEFAULT' and formals[name]['declared_default'] is not None
             and type(initial) is ast.Constant and type(initial.value) is bool and initial.value==selected,
             'original exact branch must agree with declared Boolean default')
    return dict(formal=name, binding=formals[name], required=selected, source_span=list(_coordinates(branch)),
                default_span=list(_coordinates(initial)), runtime_default_guard_required=True)


def _ordered_subcalls(activations, calls):
    order = []
    for activation in activations:
        matched = [_only((event for event in call['observations']
                          if event['trace_directory']==activation['trace_directory']
                          and event['caller']==activation['callee']),
                         'source plot call needs one exact attempt in each admitted activation')
                   for call in calls]
        positions = [call['attempt']['index'] for call in matched]
        _require(positions==sorted(set(positions)), 'observed source plot attempt order differs')
        order.append(dict(invocation=activation['callee'],trace_directory=activation['trace_directory'],
                          original_call_attempts=positions))
    return order


def _derive(effects_ref,startup_ref,contour_ref,text_ref):
    effects,startup,text = map(read_artifact,(effects_ref,startup_ref,text_ref))
    stored = read_artifact(contour_ref)
    contour = read_contour_scene_plan(contour_ref['uri'])
    _require(contour==stored and text['source_effects']==effects_ref
             and startup['controller_source']==effects['controller'],
             'geometry, text or startup did not resolve to the same original source controller')
    observed = read_artifact(contour['observation'])
    text_observation = read_artifact(text['observation'])
    _require(observed['agg_scene']==text_observation['agg_scene'], 'geometry and title need the same original AGG run')
    title = _format_route(text['format_route']['source'],effects)
    _require(title==text['format_route'], 'original title route differs from observed verified format')
    source = title['source']
    body,effect,observations,formals = _body_and_effect(effects,source)
    syntax = _Syntax(source,body)
    control = read_artifact(effects['controller'])
    receiver,controller_receiver,call = _binding(formals,effect,control)
    time = formals[title['formal']['name']]
    _require(time['formal']==title['formal']['formal'] and time['kind']=='CONTROLLER_SCALAR',
             'original scientific title must consume the controller clock via its exact formal')
    host = dict(controller=control, update=read_artifact(control['update_source']), formals=formals)
    path = controller_receiver['name']
    first,*rest = [row['grid']['lineage'] for row in observed['samples']]
    _require(all(row==first for row in rest), 'retained contour root geometries differ across observed frames')
    arrays,chart = _draw_bindings(syntax,body,first,source,receiver,path,startup,host)
    _require({item['frame_index'] for item in observed['samples']}==
             {item['frame_index'] for item in text['samples']},'title and geometry frames differ')
    source_slice = read_artifact(startup['source_slice'])
    output,counter = _output(syntax,startup,source_slice,formals,receiver,path)
    branch = _control_guard(syntax,chart,title,formals)
    title_call = syntax.call(syntax.node(title['call_result']))
    order = _ordered_subcalls(observations,(chart,title_call,output['join'],output['save'],output['cleanup']))
    step = _only((dict(name=name,**binding) for name,binding in formals.items()
                  if binding.get('kind')=='CONTROLLER_SCALAR' and binding!=time
                  and not any(type(node) is ast.Name and isinstance(node.ctx,ast.Load) and node.id==name
                              for node in ast.walk(syntax.function))),
                 'need one distinct original unused controller step argument')
    return dict(schema=SCHEMA,schema_version=1, source_effects=effects_ref, startup=startup_ref,
                contour=contour_ref,text=text_ref,
                effect=dict(id=effect['id'],site=effect['site'],body=body['body'],original_call=effect['original_call'],
                            formal_routes=formals, controller_call=call, observed_activations=observations,
                            controller_step=step, ordered_subcall_attempts=order),
                arrays=arrays, chart=chart, title=dict(format_route=title, controller_scalar=time),
                output=output, counter=counter, control=branch,
                open_effects=['PINNED_PLOT_AND_PATH_PROVIDER_SCOPE','ORIGINAL_DIRECTORY_CHECK_AND_CREATION',
                              'ORIGINAL_FIELD_LOOKUP_AND_INTEGER_DOMAIN','ORIGINAL_PLOT_BRANCH_AND_ERROR_EFFECTS'],
                deployment_authorized=False, whole_entry_authorized=False)


def project(effects,startup_plan,contour_plan,text_plan):
    return _derive(_ref(effects,EFFECTS),_ref(startup_plan,STARTUP),_ref(contour_plan,CONTOUR),_ref(text_plan,TEXT))


def check(path):
    result = read_artifact(_ref(path,SCHEMA))
    _require(result==_derive(result['source_effects'],result['startup'],result['contour'],result['text']),
             'original controller plot binding differs at replay')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--effects')
    parser.add_argument('--startup')
    parser.add_argument('--contour')
    parser.add_argument('--text')
    parser.add_argument('--out')
    parser.add_argument('--check')
    args=parser.parse_args(argv)
    if args.check:
        result=check(args.check)
    elif all((args.effects,args.startup,args.contour,args.text,args.out)):
        result=project(args.effects,args.startup,args.contour,args.text)
        with Path(args.out).open('x',encoding='utf-8') as stream:
            json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False)
            stream.write('\n')
    else:
        parser.error('provide --check or all projection inputs and --out')
    print(json.dumps(dict(schema=SCHEMA, effect=result['effect']['id'], arrays=list(result['arrays']))))


if __name__=='__main__':
    main()
