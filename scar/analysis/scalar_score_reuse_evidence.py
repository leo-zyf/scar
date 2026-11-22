"""Connect sampled generator scores to a later ordinary call in their owner."""

import argparse
import ast
from collections import Counter
from dis import Bytecode, get_instructions
import json
from pathlib import Path

from scar.analysis.source_declared_calls import _owners
from scar.analysis import scalar_table_handoff, scalar_transition_evidence
from scar.analysis.program_view import ProgramView
from scar.analysis.source_references import ReferenceSlicer, source_port
from scar.compiler.graph_frontend import _metadata, _source_identity
from scar.ir.execution_recipe import read_artifact
from scar.ir.record_codec import encode
from scar.runtime.source_value_observe import _codes
from scar.trace.code_snapshot import describe_code, site

SCHEMA = 'scar.scalar-score-reuse-evidence'
DESIGN = 'scar.scalar-score-reuse-selection'
CAPTURE = 'scar.scalar-score-reuse-capture'


def _need(ok, detail):
    if not ok:
        raise ValueError('scalar score reuse: ' + detail)


def _one(items, detail):
    items = tuple(items)
    _need(len(items) == 1, detail)
    return items[0]


def _span(node):
    return [node.lineno, node.end_lineno, node.col_offset, node.end_col_offset]


def _event(code, node, opcode):
    opcodes = ('CALL', 'CALL_KW') if opcode == 'CALL' else (opcode,)
    return _one((site(code, op.offset) for op in get_instructions(code) if op.opname in opcodes
                 and site(code, op.offset)['position'] == _span(node)), 'source expression has no exact compiled '+opcode)


def _name(node):
    return node.id if isinstance(node, ast.Name) else None


def _score_bindings(nodes, lhs, arg, receiver, method):
    for node in nodes:
        if not isinstance(node, ast.NamedExpr) or _name(node.target) != lhs or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if (isinstance(call.func, ast.Attribute) and _name(call.func.value) == receiver
                and call.func.attr == method and len(call.args) == 1 and _name(call.args[0]) == arg and not call.keywords):
            yield call


def _tuple_names(node):
    if isinstance(node, ast.Tuple) and len(node.elts) == 2:
        names = tuple(_name(value) for value in node.elts)
        if all(names):
            return names
    return None


def _compile(source):
    raw = read_artifact(source,json_document=False)
    codes = {code.co_qualname:code for code in _codes(
        compile(raw,source['uri'],'exec',dont_inherit=True,optimize=0))}
    return ast.parse(raw,filename=source['uri']), codes


def _arms(generator, code, codes, receiver, method):
    direct = []
    for block in ast.walk(generator):
        if not isinstance(block, ast.If):
            continue
        for statement in block.body:
            yielded = statement.value if isinstance(statement, ast.Expr) else None
            if not isinstance(yielded, ast.Yield):
                continue
            names = _tuple_names(yielded.value)
            if not names:
                continue
            left, move = names
            candidates = tuple(_score_bindings(ast.walk(block.test),left,move,receiver,method))
            if candidates:
                call = _one(candidates,'direct yield must have one original score binder')
                direct.append(dict(arm='DIRECT_YIELD',code=describe_code(code),score_call=_event(code,call,'CALL'),
                    yield_site=_event(code,yielded,'YIELD_VALUE'), move_name=move))
    first = _one(direct,'one original direct score yield expected')
    delegated = []
    for node in ast.walk(generator):
        if not isinstance(node, ast.YieldFrom) or not isinstance(node.value, ast.Call):
            continue
        container = node.value
        if (_name(container.func) != 'sorted' or len(container.args) != 1
                or not isinstance(container.args[0], ast.GeneratorExp)):
            continue
        expression = container.args[0]
        names = _tuple_names(expression.elt)
        if not names:
            continue
        value, move = names
        call = _one(_score_bindings(ast.walk(expression),value,move,receiver,method),
                    'sorted producer must bind its tuple element from the original score')
        actual = _one((candidate for name,candidate in codes.items()
                       if name.startswith(code.co_qualname+'.<locals>.') and candidate.co_firstlineno == expression.lineno
                       and any(site(candidate,op.offset)['position'] == _span(call)
                               for op in get_instructions(candidate) if op.opname == 'CALL')),
                      'genexpr score call lacks its exact compiled body')
        delegated.append(dict(arm='SORTED_DELEGATION',code=describe_code(actual),
            score_call=_event(actual,call,'CALL'),yield_site=_event(code,node,'YIELD_VALUE'),
            inner_yield=_event(actual,expression.elt,'YIELD_VALUE'),sorted_call=_event(code,container,'CALL'),
            move_name=move))
    return [first,_one(delegated,'one original sorted score delegation expected')]


def _state_sites(caller, code, direct, receiver):
    move = direct['move_name']
    reads = []
    for stmt in caller.body:
        if (isinstance(stmt,ast.Assign) and len(stmt.targets) == 1 and _name(stmt.targets[0]) == move
                and isinstance(stmt.value,ast.Call) and isinstance(stmt.value.func,ast.Attribute)
                and isinstance(stmt.value.func.value,ast.Attribute) and stmt.value.func.attr == 'get'
                and len(stmt.value.args) == 1 and _name(stmt.value.args[0]) == receiver and not stmt.value.keywords):
            reads.append(stmt.value)
    lookup = _one(reads,'direct-yield source does not select one original state lookup')
    root = lookup.func.value
    owner = _name(root.value)
    _need(owner is not None, 'state table requires an explicit original receiver')
    stores = [node for node in ast.walk(caller) if isinstance(node, ast.Subscript) and isinstance(node.ctx,ast.Store)
              and ast.dump(root) == ast.dump(node.value) and _name(node.slice) == receiver]
    store = _one(stores,'no one original matching state item store')
    instructions = list(get_instructions(code))
    write = _event(code,store,'STORE_SUBSCR')
    ordinal = next(i for i,op in enumerate(instructions) if op.offset == write['offset'])
    read = _event(code,lookup,'CALL')
    _need(not any(row.start <= read['offset'] < row.end for row in Bytecode(code).exception_entries),
          'original lookup may enter a local exception handler before the store')
    return dict(owner=owner,field=root.attr,key=receiver,lookup=read,
                write=write,after=site(code,instructions[ordinal+1].offset))


def _design(transition):
    old = read_artifact(transition)
    old_capture = read_artifact(old['capture'])
    old_design_ref = old_capture['selection']
    old_design = read_artifact(old_design_ref)
    source = old['source']
    tree,codes = _compile(source)
    from scar.compiler.scalar_table_region import _definition

    caller_code = codes[old['caller']['code']['qualname']]
    _need(describe_code(caller_code) == old['caller']['code'], 'parent source body differs from admitted transition')
    caller = _definition(tree,caller_code.co_qualname)
    position = old_design['producer']['caller_site']['position']
    output = _one((node for node in ast.walk(caller) if isinstance(node,ast.Call) and _span(node) == position),
                  'selected original producer does not have its caller expression')
    _need(isinstance(output.func,ast.Attribute) and _name(output.func.value) is not None
          and len(output.args) == 1 and _name(output.args[0]) is not None and not output.keywords,
          'selected transition lacks its original receiver and explicit move local')
    receiver, move = _name(output.func.value),_name(output.args[0])
    loops = [node for node in ast.walk(caller) if isinstance(node,ast.For) and _tuple_names(node.target)
             and _tuple_names(node.target)[1] == move and any(found is output for found in ast.walk(node))]
    loop = _one(loops,'expected one original two-component scoring loop around the transition')
    _need(isinstance(loop.iter,ast.Call) and _name(loop.iter.func) is not None
          and not loop.iter.args and not loop.iter.keywords,'loop must consume one original zero-argument producer')
    name = _name(loop.iter.func)
    generator = _one((stmt for stmt in caller.body if isinstance(stmt,ast.FunctionDef) and stmt.name == name),
                     'original loop producer declaration not found')
    generator_code = codes[caller_code.co_qualname+'.<locals>.'+name]
    method = old['score']['code']['qualname'].rsplit('.',1)[-1]
    arms = _arms(generator,generator_code,codes,receiver,method)
    return dict(schema=DESIGN,schema_version=1,source=source,transition_evidence=transition,
        transition_selection=old_design_ref,caller=describe_code(caller_code),generator=describe_code(generator_code),
        loop_span=_span(loop),loop_binding=dict(score=_tuple_names(loop.target)[0], move=move, receiver=receiver),
        loop_generator_call=_event(caller_code,loop.iter,'CALL'),arms=arms,state=_state_sites(caller,caller_code,arms[0],receiver))


def prepare(transition):
    return _design(scalar_table_handoff._ref(transition,scalar_transition_evidence.SCHEMA))


def _f4(prior, design):
    read_artifact(prior['catalog'])
    view = ProgramView(Path(prior['catalog']['uri']).parent)
    parent = prior['original_calls']['producer']
    material = view.unit(parent['unit'])
    facts = material.facts
    _, syntax, codes, _, source_hash = _source_identity(view.catalog,facts.unit)
    _need(source_hash == design['source']['sha256'], 'F4 unit source bytes differ from scoring scope')
    original_codes = {code['key']:code for code in map(describe_code,codes)}
    slicer, owners = ReferenceSlicer(view.catalog), _owners(facts)

    def fact(table, selected):
        code = original_codes[selected['code']]
        candidates = []
        for ordinal,row in enumerate(getattr(facts,table)):
            if table == 'items' and not row.write:
                continue
            ordinal_port = row.result if table == 'calls' else row.value
            original = source_port(facts,ordinal_port)
            _, span, _ = _metadata(syntax,original.port.template.occurrence)
            if list(span[key] for key in ('line','end_line','column','end_column')) == selected['position']:
                candidates.append((ordinal,row,original))
        ordinal,row,original = _one(candidates,'compiled site lacks one retained original source fact')
        if table == 'calls':
            owner = _one(owners[ordinal],'compiled scoring expression has ambiguous original scope owner')
            extra = dict(result=encode(original),source_scope=encode(owner))
        else:
            extra = dict(receiver=encode(source_port(facts,row.receiver)),value=encode(original),
                key=encode(source_port(facts,row.key)),world_in=encode(source_port(facts,row.world)),
                world_out=encode(source_port(facts,row.result_world)))
        return dict(site=selected,code=code,fact=encode(slicer.fact(facts,table,ordinal)),**extra)

    arms = {arm['arm']:fact('calls',arm['score_call']) for arm in design['arms']}
    _need(all(row['code'] == arm['code'] for arm,row in zip(design['arms'],arms.values(),strict=True)),
          'original nested score compiled identity differs from its retained F4 unit')
    lookup = fact('calls',design['state']['lookup'])
    _need(lookup['source_scope'] == parent['code_scope']['owner']
          and lookup['code'] == design['caller'], 'state read did not belong to original caller')
    result = dict(first_scores=arms,sorted_container=fact('calls',design['arms'][1]['sorted_call']),
                transition_score=prior['original_calls']['score'],
                lookup=lookup,store=fact('items',design['state']['write']))
    material.check()
    return result


def _increasing(*values):
    return all(type(value) is int and value > 0 for value in values) and list(values) == sorted(set(values))


def _project_edges(raw,design,old):
    _, code_map = _compile(design['source'])
    score_code = code_map[old['score']['qualname']]
    arms = {arm['arm']:arm for arm in design['arms']}
    edges = []
    counts = Counter()
    for ordinal,sample in enumerate(raw['samples']):
        row = sample.get('score_reuse')
        if row is None:
            continue
        arm = arms[row['arm']]
        returned = row['first_return_site']
        _need(row['first_score_site'] == arm['score_call'] and row['yield_site'] == arm['yield_site']
              and returned['opname'] in ('RETURN_VALUE','RETURN_CONST') and site(score_code,returned['offset']) == returned
              and row['thread_id'] == sample['thread_id'] and row['table_version'] == sample['table_version']
              and row['root_token'] == sample['root_token']
              and all(row[key] is True for key in ('same_original_loop_receiver','same_original_move_object',
                                                   'same_table_root_and_content')),
              'sample does not record matching original operation, thread, inputs and table root')
        start, finish = row['first_enter_sequence'],row['first_return_sequence']
        yielded, consumed = row['yield_sequence'],row['transition_enter_sequence']
        if row['arm'] == 'SORTED_DELEGATION':
            _need(row['inner_yield_site'] == arm['inner_yield'] and
                  _increasing(row['sorted_call_sequence'],start,finish,row['inner_yield_sequence'],yielded,consumed),
                  'sample does not pass the original sorted generator path in causal order')
        else:
            _need(_increasing(start,finish,yielded,consumed) and 'sorted_call_sequence' not in row,
                  'sample does not pass the original direct generator path in causal order')
        _need(scalar_table_handoff._int64(row['first_result']), 'first score is not an exact integer')
        equal = row['first_result'] == sample['score_delta']
        counts[(row['table_version'],row['arm'],equal)] += 1
        edges.append(dict(sample=ordinal,source_arm=row['arm'],table_version=row['table_version'],
            same_exact_score=equal,first_score=row['first_result'],second_score=sample['score_delta'],
            target_frame=sample['score_frame'],owner_frame=sample['caller_frame'],thread_id=row['thread_id']))
    _need(len({row['target_frame'] for row in edges}) == len(edges), 'duplicate scored transition occurrence')
    return edges,[dict(version=version,arm=arm,same_exact_score=equal,count=count)
        for (version,arm,equal),count in sorted(counts.items())]


def project(capture):
    reference = scalar_table_handoff._ref(capture,CAPTURE)
    raw = read_artifact(reference)
    stored = read_artifact(raw['selection'])
    prior_ref = stored['transition_evidence']
    prior = read_artifact(prior_ref)
    design = _design(prior_ref)
    old = read_artifact(design['transition_selection'])
    launch = raw['launch']
    _need(raw['schema_version'] == 1 and stored == design and raw['source'] == design['source']
          and raw['score_evidence'] == old['score_evidence']
          and type(raw['process_id']) is int and raw['process_id'] > 0
          and raw['original_execution'] is True and raw['performance_measurement'] is False
          and launch['decision'] == 'NO_OP' and launch['exit_code'] == 0
          and launch['entry_path'] == design['source']['uri'], 'expected a completed unmodified source launch')
    scalar_transition_evidence._check_capture(raw,old)
    edges,counts = _project_edges(raw,design,old)
    state = design['state']
    for row in raw['state_events']:
        _need(row['same_key_object'] is row['exact_builtin_dict'] is True and
              (row['read_site'],row['write_site'],row['normal_next']) == (state['lookup'],state['write'],state['after']) and
              _increasing(row['read_sequence'],row['write_sequence'],row['normal_sequence']) and
              all(type(row[key]) is int and row[key] > 0 for key in ('thread_id','caller_frame','table_root_token')) and
              type(row['size_at_read']) is int and row['size_at_read'] >= 0 and
              type(row['size_after_write']) is int and row['size_after_write'] > 0,
              'recorded state edge does not observe original lookup and same-key normal store')
    return dict(schema=SCHEMA,schema_version=1,source=design['source'],transition_evidence=prior_ref,
        capture=reference,catalog=prior['catalog'],process_id=raw['process_id'],
        original_f4=_f4(prior,design),samples=edges,sample_counts=counts,
        state_edges=raw['state_events'],scoring_frontiers=raw['scoring_frontiers'],
        transition_frontiers=raw['observation_frontiers'],
        qualification=dict(exact_selected_calls_observed=bool(edges),same_receiver_move_object_and_table_snapshots=bool(edges),
            continuous_no_writer_proven=False,external_effects_closed=False,
            state_reads_observed_at_builtin_call_and_write_on_normal_path=bool(raw['state_events']),
            state_content_or_no_intervening_mutation_proven=False,
            all_generator_values_covered=False,transformation_authorized=False,
            target_executed_by_projection=False))


def check(path):
    stored = read_artifact(scalar_table_handoff._ref(path,SCHEMA))
    read_artifact(stored['capture'])
    result = project(stored['capture']['uri'])
    _need(result == stored,'stored observation graph differs from independent original replay')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',type=Path)
    mode.add_argument('--capture',type=Path)
    mode.add_argument('--check',type=Path)
    parser.add_argument('--out',type=Path)
    args = parser.parse_args()
    if args.check:
        row = check(args.check)
        print(json.dumps(dict(sample_counts=row['sample_counts'],state_edges=len(row['state_edges']))))
    else:
        if args.out is None:
            parser.error('publication requires --out')
        row = prepare(args.prepare) if args.prepare else project(args.capture)
        print(json.dumps(scalar_table_handoff._write(args.out,row)))


if __name__ == '__main__':
    main()
