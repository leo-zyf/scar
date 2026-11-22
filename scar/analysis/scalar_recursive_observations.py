"""Replay sampled recursive control and exact shared-state effect occurrences."""

import argparse
import ast
from collections import Counter
from dis import get_instructions
from hashlib import sha256
import json
from pathlib import Path

from scar.analysis import scalar_recursive_region as recursive
from scar.analysis import scalar_score_reuse_evidence as observed
from scar.analysis import scalar_transition_evidence as transition
from scar.analysis.scalar_score_evidence import _open, _pinned_document
from scar.analysis.scalar_table_handoff import _ref, _write
from scar.compiler.scalar_table_region import _definition
from scar.ir.execution_recipe import read_artifact
from scar.trace.code_snapshot import describe_code, site

SCHEMA = 'scar.scalar-recursive-observations'
DESIGN = 'scar.scalar-recursive-observation-design'
CAPTURE = 'scar.scalar-recursive-capture'


def _require(condition, reason):
    if not condition:
        raise ValueError('scalar recursive observations: ' + reason)


def _source(tree, classes, descriptor):
    return _definition(tree,descriptor['qualname']),classes[descriptor['qualname']]


def _next(code, selected):
    rows = tuple(get_instructions(code))
    index = next(i for i,op in enumerate(rows) if op.offset==selected['offset'])
    return site(code,rows[index+1].offset)


def _state_sites(region, function, code):
    results = []
    for field in region['graph']['receiver_state']:
        for event in field['effects']:
            effect = event['effect']
            if effect=='CALL_ON_STATE_FIELD' and event['method']=='get':
                results.append(dict(field=field['field'],action='DICT_GET',at=event['original_call']['compiled'],
                                    original=event['original_call']))
                continue
            op = {'ITEM_WRITE':'STORE_SUBSCR','ITEM_DELETE':'DELETE_SUBSCR',
                  'ATTRIBUTE_READ_WRITE':'STORE_ATTR'}.get(effect)
            if op is None:
                continue
            node = observed._one((row for row in ast.walk(function) if type(row) in (ast.Attribute,ast.Subscript)
                and observed._span(row)==event['source_span']), 'state site is absent from exact source')
            start = observed._event(code,node,op)
            results.append(dict(field=field['field'],action={'ITEM_WRITE':'DICT_STORE','ITEM_DELETE':'DICT_DELETE',
                'ATTRIBUTE_READ_WRITE':'SCALAR_STORE'}[effect],at=start,after=_next(code,start),original=event))
    return sorted(results,key=lambda row:row['at']['offset'])


def _prepare(path):
    reference = _ref(path,recursive.SCHEMA)
    region = recursive.check(reference['uri'])
    tree,codes = observed._compile(region['source'])
    owner = region['region']['owner']
    source,code = _source(tree,codes,owner)
    _require(describe_code(code)==owner, 'recursive owner version differs')
    reuse = read_artifact(region['source_state_observation'])
    reuse_capture = read_artifact(reuse['capture'])
    reuse_selection = read_artifact(reuse_capture['selection'])
    origin = read_artifact(reuse_selection['transition_selection'])
    producer = origin['producer']['code']
    rotation = origin['rotation']['code']
    score = origin['score']
    location = [node for node in ast.walk(source) if type(node) is ast.Call]
    rows = []
    for ordinal,record in enumerate(region['graph']['recursive_back_edges']):
        selected = record['original_call']['compiled']
        call = observed._one((node for node in location if observed._span(node)==selected['position']),
                             'recursive call expression missing')
        node = call.args[0]
        _require(type(node) is ast.Call and type(node.func) is ast.Attribute
                 and type(node.func.value) is ast.Name and node.func.value.id==record['actuals'][0]['formal'],
                 'first actual does not use the declared caller state binding')
        member = node.func.attr
        if member==producer['qualname'].rsplit('.',1)[-1]:
            role, target = 'DIRECT_TRANSITION', producer
        elif member==rotation['qualname'].rsplit('.',1)[-1]:
            role, target = 'ROTATED_ACTUAL', rotation
        else:
            raise ValueError('recursive first actual has no selected original producer')
        producer_site = observed._event(code,node,'CALL')
        _require(producer_site['offset']<selected['offset'], 'recursive actual must complete before child call')
        rows.append(dict(index=ordinal,site=selected,f4=record['original_call']['fact'],
                         control=record['control'],actuals=record['actuals'],defaults=record['omitted_defaults'],
                         position_formal=record['actuals'][0]['formal'],phase=role,producer=target,
                         producer_site=producer_site))
    bodies = [producer,rotation,score]
    _require(all(describe_code(codes[desc['qualname']])==desc for desc in bodies),
             'original recursive phase version differs')
    design = dict(schema=DESIGN,schema_version=1,region=reference,source=region['source'],
        owner=owner,receiver=region['region']['receiver'],formals=region['region']['formals'],sites=rows,
        state_sites=_state_sites(region,source,code),state_fields=reuse_capture['state_fields'],
        state_constructor=origin['constructor']['name'],score=score,score_call=origin['score_call'],
        internal_rotation=origin['rotation']['caller_site'],rotation=rotation,
        score_table=origin['table'])
    return design, code


def prepare(path):
    design, _ = _prepare(path)
    return design


def _integer(value):
    return type(value) is int and -(1<<63)<=value<(1<<63)


def ordered_digest(items):
    return sha256(json.dumps(items,ensure_ascii=True,sort_keys=True,separators=(',',':')).encode('ascii')).hexdigest()


def _validate_sample(row, plan):
    chosen = plan['sites'][row['site_index']]
    position = chosen['position_formal']
    expected = {field['name']:field['kind'] for field in plan['state_fields']}
    parameters = set(plan['formals'])-{plan['receiver']}
    for values in (row['parent_entry'],row['parent_actual'],row['child_actual']):
        _require(values.keys()==parameters and all(type(value) is bool or _integer(value)
            for key,value in values.items() if key!=position)
            and {key:transition.field_kind(value) for key,value in values[position].items()}==expected,
            'observed parameters differ from their original typed scalar/state protocol')
    _require(row['phase']['kind']==chosen['phase'] and row['phase']['site']==chosen['producer_site']
        and row['same_receiver'] is row['same_produced_actual_object'] is True
        and row['site']==chosen['site'] and all(type(row['child_actual'][item['formal']]) is type(item['value'])
                                               and row['child_actual'][item['formal']]==item['value']
                                               for item in chosen['defaults']),
        'recursive observed actual or original default differs')
    _require(0<row['parent_frame']<row['child_frame'] and 0<row['thread_id']
        and 0<row['produce_sequence']<row['call_sequence']<row['child_enter_sequence']<row['child_return_sequence']
        and _integer(row['child_return']), 'recursive invocation needs ordered normal return and exact result')
    _require(row['phase']['score_observed'] is (chosen['phase']=='DIRECT_TRANSITION'),
             'recursive phase lacks the matching original nested score')
    if row['phase']['score_observed']:
        _require(_integer(row['phase']['score_result']) and row['phase']['score_site']==plan['score_call'],
                 'nested score lacks its actual original result or selected site')
    else:
        _require(row['phase']['rotation_flag'] is True, 'rotated first actual does not use the original true flag')
    _require(all(_integer(row['state'][key]) for key in ('entry_counter','return_counter'))
        and row['state']['entry_counter']<row['state']['return_counter'], 'child counter effect was not observed')
    return chosen


def _state_order(rows, design):
    cuts = {(row['field'],row['action'],row['at']['offset']):row for row in design['state_sites']}
    events,latest = [],{}
    last = 0
    for row in rows:
        start,finish = row['start'],row['finish']
        key = row['field'],row['action'],row['site']['offset']
        selected = cuts[key]
        _require(row['site']==selected['at'] and row['completion'] in ('ORIGINAL_BUILTIN_RETURN','NORMAL_NEXT_INSTRUCTION')
            and all(type(row[name]) is int and row[name]>0 for name in ('thread_id','receiver_occurrence','activation'))
            and 0<start<finish and finish>last, 'state effect is missing original ordered normal completion')
        action = row['action']
        if action=='SCALAR_STORE':
            _require(_integer(row['before']) and row['after']==row['before']+1, 'original scalar update differs')
        else:
            _require(type(row['resource_occurrence']) is int and row['resource_occurrence']>0
                and type(row['size_before']) is int and type(row['size_after']) is int,
                'dictionary event lacks exact root or item counts')
            delta = row['size_after']-row['size_before']
            _require(delta in {'DICT_GET':{0},'DICT_STORE':{0,1},'DICT_DELETE':{-1}}[action],
                     'exact observed dictionary update does not match selected action')
        resource = (row['thread_id'],row['receiver_occurrence'],row['field'],row.get('resource_occurrence'))
        previous = latest.get(resource)
        if previous is not None and previous['finish']<start:
            events.append(dict(resource=list(resource[:3]),root_occurrence=resource[3],
                before=previous['finish'],after=finish,previous=previous['action'],current=action,
                scope='OBSERVED_ORDER_ONLY_UNSAMPLED_EFFECTS_MAY_INTERVENE'))
        latest[resource] = row
        last = finish
    return events


def _root_probes(raw, plan, owner):
    compiled = sorted({op.argval for op in get_instructions(owner) if op.opname=='LOAD_GLOBAL'})
    _require(compiled==raw['owner_global_reads'] and set(raw['module_i64'])<=set(compiled)
        and all(_integer(value) for value in raw['module_i64'].values()),
        'global capture must be restricted to actual owner code loads and exact integers')
    receiver = read_artifact(plan['region'])['graph']['receiver_state']
    fields = {item['field'] for item in receiver}
    state_counter = observed._one((row['field'] for row in plan['state_sites'] if row['action']=='SCALAR_STORE'),
                                  'expected one original recursive counter')
    dictionaries = {row['field'] for row in plan['state_sites'] if row['action']=='DICT_GET'}
    parameters = set(plan['formals'])-{plan['receiver']}
    expected = {row['name']:row['kind'] for row in plan['state_fields']}
    default = observed._one(plan['sites'][0]['defaults'],'original driver witness needs one root-phase formal')
    position = plan['sites'][0]['position_formal']
    identities = set()
    for row in raw['root_probes']:
        actual,before,after = (row[key] for key in ('actuals','before','after'))
        _require(row['receiver_occurrence'] not in identities and row['entry_sequence']<row['return_sequence']
            and _integer(row['result']) and before.keys()==after.keys() and fields<=before.keys()
            and actual.keys()==parameters and actual[default['formal']] is True and default['value'] is False
            and {key:transition.field_kind(value) for key,value in actual[position].items()}==expected
            and all(type(value) is bool or _integer(value) for key,value in actual.items() if key!=position)
            and _integer(before[state_counter]) and _integer(after[state_counter])
            and after[state_counter]>before[state_counter] and row['module_i64'].keys()==raw['module_i64'].keys()
            and all(_integer(value) for value in row['module_i64'].values()),
            'original empty-table driver probe does not meet its actual scalar/state protocol')
        identities.add(row['receiver_occurrence'])
        for name in dictionaries:
            _require(before[name]['items']==[] and all(type(point[name]['items']) is list
                and point[name]['ordered_sha256']==ordered_digest(point[name]['items']) for point in (before,after)),
                'original driver probe must preserve ordered dictionary input and return digests')
    return raw['root_probes']


def _project(reference, raw):
    retained = read_artifact(raw['design'])
    plan, owner = _prepare(retained['region']['uri'])
    _require(raw['schema_version']==1 and retained==plan and raw['source']==plan['source']
        and raw['launch']['entry_path']==plan['source']['uri'] and raw['launch']['exit_code']==0
        and raw['launch']['decision']=='NO_OP' and raw['original_execution'] is True
        and raw['performance_measurement'] is False and type(raw['process_id']) is int and raw['process_id']>0,
        'observations must be emitted by an actual successful original target launch')
    counts = Counter()
    ids = set()
    for row in raw['samples']:
        selected = _validate_sample(row,plan)
        _require(row['child_frame'] not in ids,'recursive child sample is repeated')
        ids.add(row['child_frame'])
        counts[str(selected['index'])] += 1
    order = _state_order(raw['state_events'],plan)
    totals = Counter(row['field']+' '+row['action'] for row in raw['state_events'])
    root_probes = _root_probes(raw, plan, owner)
    return dict(schema=SCHEMA,schema_version=1,recursive_region=plan['region'],design=raw['design'],capture=reference,
        source=plan['source'],process_id=raw['process_id'],observed_site_counts=dict(counts),
        sites=[dict(index=row['index'],phase=row['phase'],site=row['site'],original_f4=row['f4'],counts=counts[str(row['index'])])
               for row in plan['sites']],state_action_counts=dict(totals),state_order_edges=order,
        owner_global_reads=raw['owner_global_reads'],module_i64=raw['module_i64'],root_probes=root_probes,
        unresolved_sites=[row['index'] for row in plan['sites'] if not counts[str(row['index'])]],
        capture_frontiers=raw['frontiers'],qualification=dict(captured_only_bounded_occurrences=True,
            no_target_execution_during_projection=True,alias_and_nested_effects_closed=False,
            external_time_and_throw_closed=False,native_lowering_authorized=False))


def project(path):
    return _project(*_open(path, CAPTURE))


def check(path):
    _, result = _open(path, SCHEMA)
    rebuilt = _project(*_pinned_document(result['capture'], CAPTURE))
    _require(result==rebuilt,'stored observations differ from independent replay')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group(required=True)
    options.add_argument('--prepare',type=Path)
    options.add_argument('--capture',type=Path)
    options.add_argument('--check',type=Path)
    parser.add_argument('--out',type=Path)
    args = parser.parse_args()
    if args.check:
        result = check(args.check)
        print(json.dumps(dict(sites=result['observed_site_counts'],state=result['state_action_counts'])))
    else:
        if args.out is None:
            parser.error('publication requires --out')
        result = prepare(args.prepare) if args.prepare else project(args.capture)
        print(json.dumps(_write(args.out,result)))


if __name__ == '__main__':
    main()
