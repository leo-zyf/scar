"""Bind an existing two-slot generator result to its selected source consumer."""

import argparse
import ast
from collections import defaultdict, deque
from dis import Bytecode, get_instructions, hasjump
import json
from pathlib import Path

from scar.analysis import scalar_score_reuse_evidence as observed, scalar_table_handoff
from scar.analysis.program_view import ProgramView
from scar.analysis.source_references import ReferenceSlicer, source_port
from scar.compiler.graph_frontend import _metadata, _source_identity
from scar.compiler.scalar_table_region import _definition
from scar.ir.execution_recipe import read_artifact
from scar.ir.record_codec import encode
from scar.trace.code_snapshot import site

SCHEMA = 'scar.scalar-loop-score-provenance'


def _require(condition, explanation):
    if not condition:
        raise ValueError('scalar loop score provenance: ' + explanation)


def _one(rows, explanation):
    rows = tuple(rows)
    _require(len(rows) == 1, explanation)
    return rows[0]


def _span(node):
    return tuple(observed._span(node))


def _at(tree, kind, point):
    position = tuple(point['position'])
    return _one((node for node in ast.walk(tree) if isinstance(node,kind) and _span(node) == position),
                'missing original source expression for compiled site')


def _name(node):
    return node.id if isinstance(node,ast.Name) else None


class _NormalFlow:
    """Compiled normal-flow slice. Throw/close and exception edges stay external."""

    def __init__(self,code):
        self.code = code
        ops = list(get_instructions(code))
        self.ops = {op.offset:op for op in ops}
        self.next = {left.offset:right.offset for left,right in zip(ops,ops[1:])}
        self.edges = defaultdict(set)
        for op in ops:
            if op.opcode in hasjump:
                self.edges[op.offset].add(op.jump_target)
            if (op.offset in self.next and not op.opname.startswith(('RETURN','RAISE','RERAISE','JUMP'))
                    and op.opname not in ('END_ASYNC_FOR',)):
                self.edges[op.offset].add(self.next[op.offset])
        self.entry = _one((op.offset for op in ops if op.opname == 'RESUME' and op.arg == 0),
                          'compiled body needs a single initial entry')

    @staticmethod
    def _reachable(edges, start, excluded=()):
        result,pending = set(),[start]
        while pending:
            point = pending.pop()
            if point in result or point in excluded:
                continue
            result.add(point)
            pending.extend(edges.get(point,()))
        return result

    def path(self,start,end,excluded=()):
        forward = self._reachable(self.edges,start,excluded)
        _require(end in forward,'selected normal source path is unreachable')
        reverse = defaultdict(set)
        for origin in forward - {end}:
            for target in self.edges.get(origin,()):
                if target in forward:
                    reverse[target].add(origin)
        return self._reachable(reverse,end,excluded)

    def prove_definition(self,first,definition,use,names,cut=()):
        paths = self.path(first,use,cut)
        _require(definition in paths and use not in self._reachable(self.edges,first,(*cut,definition)),
                  'selected definition does not dominate use on this normal drive')
        retained = self.path(definition,use,cut)
        conflicts = [offset for offset in retained if offset != definition and names.intersection(self.writes(offset))]
        _require(not conflicts,'a selected local can be reassigned before the original use')
        return retained

    def writes(self,offset):
        op = self.ops[offset]
        if 'STORE_FAST' in op.opname or 'DELETE_FAST' in op.opname or op.opname in ('STORE_DEREF','DELETE_DEREF'):
            return op.argval if type(op.argval) is tuple else (op.argval,)
        return ()

    def site(self,offset):
        return site(self.code,offset)

    def effects(self,path):
        return [self.site(offset) for offset in sorted(path) if self.ops[offset].opname.startswith(
            ('CALL','LOAD_ATTR','LOAD_METHOD','BINARY_OP','COMPARE_OP','CONTAINS_OP','TO_BOOL','UNARY_'))]


class _OriginalFacts:
    def __init__(self,evidence):
        read_artifact(evidence['catalog'])
        view = ProgramView(Path(evidence['catalog']['uri']).parent)
        unit = evidence['original_f4']['lookup']['source_scope']['unit']
        self.material = view.unit(unit)
        self.facts = self.material.facts
        self.slicer = ReferenceSlicer(view.catalog)
        _,self.syntax,_,_,source_hash = _source_identity(view.catalog,unit)
        _require(source_hash == evidence['source']['sha256'],'F4 ports must come from identical source bytes')
        self.lookup = defaultdict(list)
        for table,field in (('names','value'),('containers','value'),('items','value'),('calls','result')):
            for ordinal,row in enumerate(getattr(self.facts,table)):
                port = self.port(getattr(row,field))
                _,span,_ = _metadata(self.syntax,port.port.template.occurrence)
                coords = tuple(span[key] for key in ('line','end_line','column','end_column'))
                self.lookup[table,coords].append((ordinal,row,port))
        self.aliases = defaultdict(list)
        for ordinal,(source,target) in enumerate(self.facts.aliases):
            self.aliases[source].append((target,ordinal))

    def port(self,index):
        return source_port(self.facts,index)

    def coordinate(self,table,node,condition,*,required=True):
        rows = [row for row in self.lookup[table,_span(node)] if condition(row[1])]
        if not rows and not required:
            return None,None
        ordinal,row,port = _one(rows,'source coordinate lacks a unique F4 '+table+' fact')
        ref = encode(self.slicer.fact(self.facts,table,ordinal))
        return row,dict(fact=ref,port=encode(port),span=list(_span(node)))

    def name(self,node,write):
        row,reference = self.coordinate('names',node,lambda value:value.namespace.name == node.id and value.write is write)
        return dict(**reference,binding=encode(row.namespace))

    def _index(self,table,reference):
        prefix = '#/'+table+'/'
        locator = reference['fact']['locator']
        _require(locator.startswith(prefix),'source references must belong to their original fact table')
        return int(locator[len(prefix):])

    def row(self,table,reference):
        return getattr(self.facts,table)[self._index(table,reference)]

    def aliases_between(self,source,target):
        pending,paths = deque([source]),{source:()}
        while pending and target not in paths:
            port = pending.popleft()
            for next_port,ordinal in self.aliases[port]:
                if next_port not in paths:
                    paths[next_port] = (*paths[port],ordinal)
                    pending.append(next_port)
        _require(target in paths,'compiled transfer lacks original F4 alias edges')
        return [encode(self.slicer.fact(self.facts,'aliases',ordinal)) for ordinal in paths[target]]

    def unpack(self,target):
        result = []
        for ordinal,binding in enumerate(target.elts):
            item,reference = self.coordinate('items',target,lambda row: not row.write
                and getattr(row.key,'payload',None) == str(ordinal))
            definition = self.name(binding,True)
            alias = self.aliases_between(item.value,self.row('names',definition).value)
            result.append(dict(slot=ordinal,symbol=binding.id,item=reference,definition=definition,
                               record=encode(self.port(item.receiver)),item_to_definition=alias))
        _require(result[0]['record'] == result[1]['record'],'unpack components require one original record port')
        return result

    def pair(self,node):
        row,reference = self.coordinate('containers',node,lambda value:len(value.elements) == 2)
        return dict(**reference,component_ports=[encode(self.port(value)) for value in row.elements])


def _source_nodes(tree,retained,design):
    caller = _definition(tree,design['caller']['qualname'])
    contract = read_artifact(design['transition_selection'])
    producer = _definition(tree,contract['producer']['code']['qualname'])
    original_call = _at(caller,ast.Call,contract['producer']['caller_site'])
    loop = _one((node for node in ast.walk(caller) if isinstance(node,ast.For) and _span(node) == tuple(design['loop_span'])),
                'exact original tuple loop unavailable')
    generator_symbol = _name(loop.iter.func)
    generator = _one((stmt for stmt in caller.body if isinstance(stmt,ast.FunctionDef) and stmt.name == generator_symbol),
                     'generator needs one explicit local definition')
    _require(not generator.decorator_list and not generator.args.args and not generator.args.kwonlyargs
        and not generator.args.posonlyargs and not generator.args.vararg and not generator.args.kwarg
        and all(name is loop.iter.func for name in ast.walk(caller)
                if isinstance(name,ast.Name) and name.id == generator_symbol)
        and not any(isinstance(node,(ast.Global,ast.Nonlocal)) and generator_symbol in node.names for node in ast.walk(caller)),
        'generator is exposed by another local reference or has an open declaration protocol')
    producer_code = retained[contract['producer']['code']['qualname']]
    return caller,producer,producer_code,original_call,loop,generator,contract


def _sentinel_test(loop, original_call, move_symbol):
    guards = []
    for branch in ast.walk(loop):
        if type(branch) is not ast.If or type(branch.test) is not ast.Compare:
            continue
        test = branch.test
        if (_name(test.left) == move_symbol and len(test.ops) == 1 and type(test.ops[0]) is ast.Is
                and type(test.comparators[0]) is ast.Constant and test.comparators[0].value is None
                and any(expression is original_call for statement in branch.orelse for expression in ast.walk(statement))):
            guards.append(test)
    return _one(guards,'null carrier must be excluded by the original guard before selected call')


def _arm(facts,retained,generator,arm,paired_f4):
    score = _at(generator,ast.Call,arm['score_call'])
    scored = _one((node for node in ast.walk(generator) if isinstance(node,ast.NamedExpr) and node.value is score),
                  'original score call needs one exact result definition')
    if arm['arm'] == 'DIRECT_YIELD':
        pair = _at(generator,ast.Yield,arm['yield_site']).value
        origin = pair
    else:
        delegate = _at(generator,ast.YieldFrom,arm['yield_site'])
        pair = _at(delegate,ast.Tuple,arm['inner_yield'])
        origin = delegate
    _require(type(pair) is ast.Tuple and len(pair.elts) == 2 and _name(pair.elts[0]) == scored.target.id
        and _name(pair.elts[1]) == _name(score.args[0]) == arm['move_name'],'score and move slots do not use their source inputs')
    flow = _NormalFlow(retained[arm['code']['qualname']])
    write = _one((offset for offset in flow.ops if scored.target.id in flow.writes(offset)
                  and tuple(flow.site(offset)['position']) == _span(scored.target)), 'missing original compiled score result store')
    end = arm.get('inner_yield',arm['yield_site'])['offset']
    first,cut = flow.entry,()
    if arm['arm'] == 'SORTED_DELEGATION':
        advance = _one((offset for offset,op in flow.ops.items() if op.opname == 'FOR_ITER'),
                       'source comprehension needs one explicit iterator advance')
        first,cut = flow.next[advance],(advance,)
    receiver = score.func.value
    held = flow.prove_definition(first,write,end,{scored.target.id,arm['move_name'],receiver.id},cut)
    invoke = arm['score_call']['offset']
    _require(write in flow.edges[flow.next[invoke]] or write in flow.edges[invoke],
              'compiled result store does not immediately receive the original successful call')
    definition, use = facts.name(scored.target,True),facts.name(pair.elts[0],False)
    _require(definition['binding'] == use['binding'],'score definition and carried use use different lexical bindings')
    move_actual,move_carried = facts.name(score.args[0],False),facts.name(pair.elts[1],False)
    _require(move_actual['binding'] == move_carried['binding'],'move actual and carried value use different lexical bindings')
    carrier = facts.pair(pair)
    _,first_call = facts.coordinate('calls',score,lambda row:True)
    _require(first_call['fact'] == paired_f4['fact'],'source carry path disagrees with original observed call fact')
    score_result_flow = facts.aliases_between(facts.row('calls',first_call).result,facts.row('names',definition).value)
    slot_flows = [facts.aliases_between(facts.row('names',value).value,target) for value,target in
        zip((use,move_carried),facts.row('containers',carrier).elements,strict=True)]
    return dict(role=arm['arm'],code=arm['code'],carrier=carrier,score_call=arm['score_call'],
        score_call_fact=first_call,score_result_to_definition=score_result_flow,carried_slots_from_uses=slot_flows,
        definition=definition,compiled_score_store=flow.site(write),carried_score_use=use,
        carried_move_use=move_carried,score_move_actual=move_actual,score_receiver=facts.name(receiver,False),
        yield_site=arm['yield_site'],source_carrier_span=list(_span(origin)),
        normal_slice_effect_sites=flow.effects(held - {write,end}),inner_yield=arm.get('inner_yield'),
        sorted_call=arm.get('sorted_call'),may_span_earlier_consumer_iterations=arm['arm'] == 'SORTED_DELEGATION',
        suspension_closure='ORIGINAL_NEXT_RESUME_THROW_CLOSE_PROTOCOL_REQUIRED')


def _table_writes(tree,facts,table):
    writes = scalar_table_handoff._write_frontiers(tree,table)
    indexed = {_span(node):node for node in ast.walk(tree) if isinstance(node,ast.Subscript)}
    for write in writes:
        if write['role'] == 'ITEM_WRITE_OR_DELETE':
            node = indexed[tuple(write['span'])]
            _,ref = facts.coordinate('items',node,lambda row:row.write,required=False)
            write['original_fact'] = ref
    mutation_methods = {'update','setdefault','clear','pop','popitem','__setitem__','__delitem__'}
    for call in ast.walk(tree):
        if (type(call) is ast.Call and type(call.func) is ast.Attribute and _name(call.func.value) == table
                and call.func.attr in mutation_methods):
            writes.append(dict(role='DIRECT_MUTATION_CALL',method=call.func.attr,span=list(_span(call))))
    return sorted(writes,key=lambda row:row['span'])


def project(path):
    ref = scalar_table_handoff._ref(path,observed.SCHEMA)
    evidence = observed.check(ref['uri'])
    _require(read_artifact(ref) == evidence,'score reuse evidence differs after original replay')
    raw = read_artifact(evidence['capture'])
    design = read_artifact(raw['selection'])
    tree,codes = observed._compile(evidence['source'])
    caller,producer,producer_code,selected,loop,generator,transition = _source_nodes(tree,codes,design)
    facts = _OriginalFacts(evidence)
    owner = _NormalFlow(codes[design['caller']['qualname']])
    next_item = observed._event(owner.code,loop.iter,'FOR_ITER')
    unpack = observed._event(owner.code,loop.target,'UNPACK_SEQUENCE')
    names = design['loop_binding']
    bindings = {names['score'],names['move']}
    offset = owner.next[unpack['offset']]
    stores = []
    while set(owner.writes(offset)) and set(owner.writes(offset)) <= bindings:
        stores.append(offset)
        offset = owner.next[offset]
        if set(value for store in stores for value in owner.writes(store)) == bindings:
            break
    _require(set(value for store in stores for value in owner.writes(store)) == bindings,
             'fixed unpack does not establish both original local slots')
    transition_site = transition['producer']['caller_site']
    advance = next_item['offset']
    path_to_transition = owner.prove_definition(owner.next[advance],stores[-1],transition_site['offset'],
        bindings|{names['receiver']},(advance,))
    path_to_transition -= set(stores)
    for row in Bytecode(owner.code).exception_entries:
        _require(not any(row.start <= point < row.end for point in path_to_transition),
                 'selected caller path has a local exception edge')
    rebinds = sorted(offset for offset in owner.ops if offset not in stores and bindings.intersection(owner.writes(offset)))
    _require(all(offset not in path_to_transition for offset in rebinds),'alternate binding reaches selected call')
    guard = _sentinel_test(loop,selected,names['move'])
    code = codes[design['generator']['qualname']]
    other_yields,unclassified = [],[]
    admitted_yields = {tuple(arm['yield_site']['position']) for arm in design['arms']}
    for node in ast.walk(generator):
        if not isinstance(node,(ast.Yield,ast.YieldFrom)) or _span(node) in admitted_yields:
            continue
        if type(node) is ast.Yield and (type(node.value) is ast.Tuple and all(
                type(cell) is ast.Constant and cell.value is None for cell in node.value.elts)):
            _require(len(node.value.elts) == 2,'null carrier does not share original record arity')
            other_yields.append(dict(site=observed._event(code,node,'YIELD_VALUE'),carrier=facts.pair(node.value),
                                     selected_transition_reachable=False))
        else:
            unclassified.append(list(_span(node)))
    exits = dict(return_spans=[list(_span(node)) for node in ast.walk(caller) if type(node) is ast.Return],
        loop_break_spans=[list(_span(node)) for node in ast.walk(loop) if type(node) is ast.Break],
        explicit_raise_spans=[list(_span(node)) for node in ast.walk(caller) if type(node) is ast.Raise])
    producer_flow = _NormalFlow(producer_code)
    second = transition['score_call']
    pre_second = producer_flow.path(producer_flow.entry,second['offset']) - {second['offset']}
    slots = facts.unpack(loop.target)
    move_at_call = facts.name(selected.args[0],False)
    receiver_at_call = facts.name(selected.func.value,False)
    _require(move_at_call['binding'] == slots[1]['definition']['binding'],'original transition actual differs from the unpacked move')
    by_arm = [_arm(facts,codes,generator,arm,evidence['original_f4']['first_scores'][arm['arm']])
              for arm in design['arms']]
    _require(all(arm['score_receiver']['binding'] == receiver_at_call['binding'] for arm in by_arm),
             'scorer and transition do not share the original lexical receiver')
    table_writes = _table_writes(tree,facts,transition['table'])
    facts.material.check()
    return dict(schema=SCHEMA,schema_version=1,evidence=ref,source=evidence['source'],catalog=evidence['catalog'],
        candidate_input=dict(record_slots=slots,move_at_transition=move_at_call,receiver_at_transition=receiver_at_call,
            transition_site=transition_site,
            original_transition_score=second,original_generator_call=design['loop_generator_call'],
            original_score_f4=evidence['original_f4']['transition_score'],owner_call=evidence['transition_evidence']),
        record_scope=dict(owner=design['caller'],generator=design['generator'],loop_span=design['loop_span'],
            declaration_span=list(_span(generator)),next_record=next_item,unpack=unpack,
            compiled_slot_definitions=[owner.site(row) for row in stores],
            generator_use='DIRECT_LOCAL_FOR_ITER_ONLY',record_shape='EXACT_TWO_POSITIONAL_SLOTS_ON_DESCRIBED_ARMS',
            null_guard_span=list(_span(guard)),alternate_binding_sites=[owner.site(row) for row in rebinds]),
        paths=dict(scored=by_arm,unscored=sorted(other_yields,key=lambda row:row['site']['offset']),exits=exits,
            unclassified_yield_spans=sorted(unclassified)),
        obligations=dict(current_iteration_consumer_uses_unpacked_move=True,successful_score_definition_reaches_carried_slot=True,
            lexical_generator_reference_unique=True,all_original_yield_shapes_classified=not unclassified,
            record_exposure_closed_under_exact_builtin_sorted_and_ordinary_source_access=True,
            normal_control_only=True,throw_and_close_effects_closed=False,
            lookup_stability_through_suspension_proven=False,external_reentry_proven_absent=False,
            runtime_version_token_implemented=False,deployment_authorized=False),
        frontiers=dict(table_symbol=transition['table'],direct_source_writes=table_writes,
            caller_window_effect_sites=owner.effects(path_to_transition - {transition_site['offset']}),
            transition_pre_score_effect_sites=producer_flow.effects(pre_second),
            sorted_carrier_possible_protocols=['OUTER_ITERATOR_CALLBACK','REPEATED_SCORER_CALL','ELEMENT_COMPARISON',
                                              'EARLIER_RECURSIVE_CONSUMER_WHILE_GENERATOR_SUSPENDED'],
            prior_recursive_consumer=transition['consumer']['caller_site'],
            state_write_f4=evidence['original_f4']['store']['fact'],
            unresolved=['ALIASED_OR_EXTERNAL_TABLE_WRITES','LIVE_GLOBAL_AND_BUILTIN_BINDINGS',
                        'MODULE_REBIND_OR_MONKEY_PATCH','CLOSE_THROW_AND_OUTSIDE_REFLECTION']),
        observed=dict(score_pairs=len(evidence['samples']),per_arm_and_version=evidence['sample_counts'],
            state_edges=len(evidence['state_edges']),scoring_frontiers=evidence['scoring_frontiers']),
        advisory_runtime_guards=['CURRENT_SUCCESSFUL_ORIGINAL_SCORE_RESULT','EXACT_RECEIVER_AND_MOVE_IDENTITY',
            'COMPARE_CURRENT_EXACT_TABLE_AND_SCORER_CONTEXT_BEFORE_USE','INVALIDATE_ON_ANY_OPEN_MUTATION_OR_REENTRY',
            'FALL_BACK_TO_ORIGINAL_CALL_ON_ANY_MISMATCH'],target_executed_by_projection=False)


def check(path):
    result = read_artifact(scalar_table_handoff._ref(path,SCHEMA))
    read_artifact(result['evidence'])
    _require(result == project(result['evidence']['uri']),'stored provenance does not match independent source replay')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--evidence', type=Path)
    inputs.add_argument('--check', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if args.check and args.out is None:
        result = check(args.check)
    elif args.evidence and args.out:
        result = project(args.evidence)
        with args.out.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
    else:
        parser.error('use --check RESULT, or --evidence EVIDENCE --out RESULT')
    print(json.dumps(dict(observed=result['observed']['score_pairs'], obligations=result['obligations'])))


if __name__ == '__main__':
    main()
