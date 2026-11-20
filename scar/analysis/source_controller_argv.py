"""Bind an original argparse namespace to admitted controller entry arguments."""

import argparse
import ast
import dis
import json
import math
from pathlib import Path
import struct

from scar.analysis.source_agg_text import _coordinates, _source
from scar.analysis.source_controller_plot import _fact
from scar.analysis.source_syntax import _flatten_ast
from scar.compiler.plan_controller_startup import SCHEMA as STARTUP, _scalar, _source_runtime_options
from scar.ir.execution_recipe import artifact_reference, read_artifact


SCHEMA = 'scar.source-controller-argv'
_TYPES = {'float64': float, 'int': int, 'str': str, 'bool': bool}


def _require(ok, detail):
    if not ok:
        raise ValueError('controller argv: ' + detail)


def _one(rows, detail):
    rows = tuple(rows)
    _require(len(rows) == 1, detail)
    return rows[0]


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _value(value):
    kind = _one((name for name,cls in _TYPES.items() if type(value) is cls), 'unsupported scalar type')
    if kind == 'float64':
        _require(math.isfinite(value), 'nonfinite literal or captured scalar')
        return dict(kind=kind, value=value, binary64_be=struct.pack('>d',value).hex())
    return dict(kind=kind, value=value)


class _Syntax:
    def __init__(self, material):
        self.material = material
        self.raw = read_artifact(material,json_document=False)
        self.tree = ast.parse(self.raw,filename=material['uri'])
        _, self.occurrences = _flatten_ast(self.tree)
        self.nodes = {(type(node).__name__, _coordinates(node)):node for node in ast.walk(self.tree)
                      if hasattr(node,'lineno')}

    def node(self, occurrence):
        row = self.occurrences[occurrence]
        values = dict(zip(('lineno','col_offset','end_lineno','end_col_offset'),
                          (item.scalar.to_python() for item in row.attributes),strict=True))
        span = tuple(values[k] for k in ('lineno','end_lineno','col_offset','end_col_offset'))
        return self.nodes[row.constructor,span]

    def port_node(self, port):
        return self.node(port['port']['template']['occurrence'])

    def declared_default(self, reference):
        _require(reference['group']=='FIELD','original parameter default must use its declaration field')
        origin = self.occurrences[reference['occurrence']]
        element = origin.fields[reference['ordinal']]
        for index in reference['item_path']:
            element = element.items[index]
        _require(element.kind=='NODE','expected one original literal parameter default')
        return self.node(element.reference)


def _source_option(node, namespace, options):
    if type(node) is ast.Attribute and type(node.value) is ast.Name and node.value.id == namespace:
        _require(node.attr in options, 'binding names an undeclared original option')
        return dict(kind='OPTION',id=node.attr)
    _require(type(node) is ast.Constant,'controller input needs an original namespace read or literal')
    return dict(kind='LITERAL',value=_value(node.value))


def _normal_branch(body, namespace, options):
    if not body:
        return None
    guard = _one(body,'entry needs one selected normal branch and an independent alternate')
    _require(type(guard) is ast.If and not guard.orelse
             and type(guard.body[-1]) is ast.Return and guard.body[-1].value is None,
             'alternate entry must terminate before the admitted controller')
    test = guard.test
    inverse = type(test) is ast.UnaryOp and type(test.op) is ast.Not
    source = _source_option(test.operand if inverse else test,namespace,options)
    _require(source['kind']=='OPTION' and options[source['id']]['kind']=='bool',
             'original alternate must branch on a parsed Boolean')
    return dict(option=source['id'],normal_value=inverse,other_status='UNADMITTED',
                source_span=list(_coordinates(guard)),condition_span=list(_coordinates(guard.test)),
                alternate_body_spans=[list(_coordinates(node)) for node in guard.body])


def _entry_syntax(entry, initializer, call_ast, options):
    matches = (node for node in ast.walk(entry.tree) if type(node) is ast.Call
               and list(_coordinates(node))==initializer['actual_caller']['position'])
    constructor = _one(matches,'constructor needs one original call expression')
    owner = _one((fn for fn in entry.tree.body if type(fn) is ast.FunctionDef
                 and any(node is constructor for node in ast.walk(fn))),
                 'original controller requires one constructor caller')
    _, code = _source(dict(material=entry.material,qualname=owner.name,code_key=initializer['actual_caller']['code']))
    observed = _one((i for i in dis.get_instructions(code) if i.offset==initializer['actual_caller']['offset']),
                    'observed constructor CALL offset missing in verified original entry')
    _require(observed.opname==initializer['actual_caller']['opname']
             and list(observed.positions)==list(_coordinates(constructor)),
             'constructor opcode does not match source and observed bytecode')
    body = owner.body
    _require(len(body)>=3 and type(body[-2]) is ast.Assign and body[-2].value is constructor
             and len(body[-2].targets)==1 and type(body[-2].targets[0]) is ast.Name
             and type(body[-1]) is ast.Expr and body[-1].value is call_ast
             and type(call_ast.func) is ast.Attribute and type(call_ast.func.value) is ast.Name
             and call_ast.func.value.id==body[-2].targets[0].id,
             'original normal path must call controller directly on the constructed receiver')
    names = body[0].targets[0].elts
    _require(len(names)==2 and all(type(node) is ast.Name for node in names)
             and names[0].id!=names[1].id and not any(
                 type(node) is ast.Name and type(node.ctx) is ast.Load and node.id==names[1].id
                 for node in ast.walk(owner)),
             'original parse_known_args remainder is not discarded')
    namespace = names[0].id
    branch = _normal_branch(body[1:-2],namespace,options)
    return namespace,branch


def _controller_routes(initialization, entry_calls, controller, entry, options):
    call = initialization['controller_call']
    evidence = call['actual']
    index = call['interface'].removeprefix('#/interfaces/')
    _require(index.isdecimal() and evidence['normal_completion_observed'] and evidence['parameters_complete']
             and evidence['target']['code']==controller['source']['code_key'],
             'one normally entered original controller interface required')
    link = entry_calls['interfaces'][int(index)]['interface']['link']
    _require(link['result']==evidence['source_result'],'original controller interface CALL result differs')
    unit,fact = _fact(evidence['source_fact'],'calls')
    point = lambda i: dict(unit=unit['unit'],port=unit['ports'][i])
    _require(point(fact['result'])==link['result'] and fact['positional_pack'] is None
             and fact['keyword_pack'] is None, 'original controller needs direct typed CALL arguments')
    source = controller['source']
    material = _ref(source['path'],'scar.python-source')
    _require(material['sha256']==source['sha256'],'controller source changed')
    _source(dict(material=material,qualname=source['qualname'],code_key=source['code_key']))
    syntax = _Syntax(material)
    call_ast = entry.port_node(point(fact['result']))
    _require(type(call_ast) is ast.Call and len(call_ast.args)==len(fact['positional'])
             and [node.arg for node in call_ast.keywords]==[name for name,_ in fact['keywords']],
             'syntax arguments differ from original F4 CALL slots')
    initializer = initialization['constructor']['initializer']
    namespace,branch = _entry_syntax(entry,initializer,call_ast,options)
    actuals = [(point(i), node, dict(kind='POSITIONAL',ordinal=ordinal))
               for ordinal,(i,node) in enumerate(zip(fact['positional'],call_ast.args,strict=True))]
    actuals.extend((point(i),node.value,dict(kind='KEYWORD',name=name))
                   for (name,i),node in zip(fact['keywords'],call_ast.keywords,strict=True))
    bindings = []
    receiver = call['receiver_formal']
    all_formals = controller['parameters']
    _require([row['name'] for row in all_formals]==[row['name'] for row in evidence['parameters']],
             'observed original controller formals differ')
    for param in all_formals:
        formal = param['formal']
        route = _one((r for r in link['parameters'] if r['formal']==formal),
                     'one original F4 route required for each controller parameter')
        actual, = route['actuals']
        if formal == receiver:
            _require(route == call['receiver_route'],'original constructed receiver route differs')
            continue
        if route['kind'] == 'ARGUMENT':
            _,node,argument = _one((item for item in actuals if item[0]==actual),
                                    'source actual does not correspond to original CALL operand')
            supplying = _source_option(node,namespace,options)
        else:
            _require(route['kind']=='INITIAL_DEFAULT' and param['default_source'] is not None,
                     'unsupported original parameter route')
            argument = dict(kind='INITIAL_DEFAULT',slot=param['default_source'])
            node = syntax.declared_default(param['default_source'])
            _require(node is syntax.port_node(actual),'F4 initial default and formal declaration differ')
            supplying = _source_option(node,namespace,{})
        bindings.append(dict(
            target=dict(kind='CONTROLLER_SLOT',id=param['name'],slot=param['slot']),
            source=supplying, original=dict(formal=formal,actual=actual,route=argument,
                                           source_span=list(_coordinates(node)))))
    explicit = [row for row in bindings if row['original']['route']['kind']!='INITIAL_DEFAULT']
    _require(len(explicit)==len(actuals),'controller source contains unconsumed original operands')
    return bindings,branch,dict(interface=call['interface'],call=evidence['source_result'],
                               source_fact=evidence['source_fact'],entry_binding=evidence['entry_binding'],
                               code_key=source['code_key'],receiver_formal=receiver)


def _startup_routes(report, option_names, options):
    bindings = []
    for route in report['runtime_routes']:
        name = route['option']
        _require(option_names[route['parameter']]==name and options[name]['kind']==route['kind'],
                 'startup option must feed its original constructor formal at the same type')
        path = route['path']
        if route['kind']=='str':
            target = dict(kind='EFFECT_DIRECTORY',id='.'.join(path),path=path)
        elif route['kind']=='bool':
            target = dict(kind='CONTROLLER_SLOT',id='.'.join(path),path=path)
        else:
            target = dict(kind='NUMERIC_PORT',id=route['id'],path=path)
        bindings.append(dict(target=target,source=dict(kind='OPTION',id=name),original=route))
    return bindings


def bind_arguments(specification, argv):
    """Interpret the scoped original parser contract with the Python provider."""
    parser = argparse.ArgumentParser()
    for name,option in sorted(specification['options'].items(),key=lambda row:row[1]['source_position']):
        kind = option['kind']
        action = dict(action='store_true') if kind=='bool' else dict(type=_TYPES[kind])
        parser.add_argument(*option['flags'],dest=name,default=option['default']['value'],**action)
    namespace, discarded = parser.parse_known_args(argv)
    values = vars(namespace)
    branch = specification['branch']
    if branch is not None and values[branch['option']]!=branch['normal_value']:
        return dict(disposition=branch['other_status'],values=[],discarded=discarded)
    def bind(field):
        source = field['source']
        return _value(values[source['id']]) if source['kind']=='OPTION' else source['value']
    return dict(disposition='NORMAL',values=[dict(target=row['target'],value=bind(row))
                                             for row in specification['bindings']],discarded=discarded)


def _witness(specification, source, observation):
    seen = read_artifact(observation)
    boundary = seen['one_shot_boundary']
    _require(seen['original_entry']==specification['entry'] and seen['source_controller']==source
             and boundary['code']['key']==specification['controller_call']['code_key'],
             'retained start witness names another original entry or controller')
    _require(seen['argv'][0]==specification['entry']['uri'],'retained startup command changes original entry')
    interpretation = bind_arguments(specification,seen['argv'][1:])
    _require(interpretation['disposition']=='NORMAL','retained witness needs the original normal branch')
    original = boundary['parameter_values']
    fields = {tuple(row['path']):row for row in boundary['fields'] if row['status']=='OBSERVED'}
    for result in interpretation['values']:
        target = result['target']
        if 'path' in target:
            actual = fields[tuple(target['path'])]
        else:
            actual = original[target['id']]
        _require(_scalar(actual)==result['value'],
                 'parsed argument differs from its original controller-boundary value: '+target['id'])
    return dict(observation=observation,argv=seen['argv'][1:],checked_values=interpretation['values'])


def _derive(reference):
    startup = read_artifact(reference)
    controller_ref = startup['controller_source']
    controller = read_artifact(controller_ref)
    comparison = read_artifact(startup['comparison'])
    parser = startup['materialization']['source_option_parser']
    provider = _ref(argparse.__file__,parser['stdlib_provider']['schema'])
    _require(provider==parser['stdlib_provider'],'original argparse provider differs')
    entry = _Syntax(parser['source'])
    source_slice = read_artifact(startup['source_slice'])
    initialization = read_artifact(source_slice['initialization'])
    _require(initialization['controller']==controller_ref,'startup and entry controller identities differ')
    entry_calls = read_artifact(initialization['entry_calls'])
    declarations,option_names = _source_runtime_options(entry.material,initialization['constructor']['initializer'])
    _require(declarations==parser['arguments'],'source option declarations differ from startup proof')
    options = {name:{**row,'default':_value(row['default'])} for name,row in declarations.items()}
    bindings,branch,call = _controller_routes(initialization,entry_calls,controller,entry,options)
    bindings.extend(_startup_routes(startup,option_names,options))
    ids = [row['target']['id'] for row in bindings]
    _require(len(set(ids))==len(ids),'controller arguments must have unique typed destinations')
    specification = dict(schema=SCHEMA,schema_version=1,startup=reference,controller=controller_ref,
           entry=entry.material,initialization=source_slice['initialization'],entry_calls=initialization['entry_calls'],
           options=options, bindings=bindings,branch=branch,controller_call=call,
           route=dict(provider=provider,parse_method='parse_known_args',allow_abbrev=True,unknown='DISCARD',
                      positional_arguments=False,default_authority='ORIGINAL_TYPED_LITERAL'),
           whole_entry_authorized=False,deployment_authorized=False)
    specification['witnesses'] = [_witness(specification,controller_ref,item) for item in comparison['observations']]
    _require(len(specification['witnesses'])>=2,'need distinct original controller-boundary witnesses')
    return specification


def project(startup_plan):
    return _derive(_ref(startup_plan,STARTUP))


def check(path):
    result = read_artifact(_ref(path,SCHEMA))
    _require(result==_derive(result['startup']),'stored argv plan differs from original input replay')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--startup')
    mode.add_argument('--check')
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    if args.startup and args.out:
        result = project(args.startup)
        with Path(args.out).open('x',encoding='utf-8') as stream:
            json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False)
            stream.write('\n')
    elif args.check and not args.out:
        result = check(args.check)
    else:
        parser.error('provide --check or --startup with --out')
    print(json.dumps(dict(schema=SCHEMA,bindings=len(result['bindings']),options=len(result['options']),
                          witnesses=len(result['witnesses']))))


if __name__=='__main__':
    main()
