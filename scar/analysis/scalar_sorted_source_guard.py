"""Derive the live guard boundary of an already selected eager scalar sort."""

import argparse
import ast
from dis import get_instructions
from inspect import CO_GENERATOR
import json
from pathlib import Path

from scar.analysis import scalar_loop_score_provenance as provenance
from scar.analysis import scalar_score_reuse_evidence as observed
from scar.analysis.static.lowering.source_generator_lowering import child_code
from scar.compiler.scalar_table_region import _definition, code_tree
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.record_codec import encode
from scar.optimization import scalar_sorted_region as sorted_region
from scar.trace.code_snapshot import describe_code, site

SCHEMA = 'scar.scalar-sorted-source-guard'
_SAFE_BUILTINS = frozenset({'abs','enumerate'})


def _require(condition, explanation):
    if not condition:
        raise ValueError('scalar sorted source guard: ' + explanation)


def _reference(path, schema):
    return artifact_reference(Path(path).resolve(strict=True),schema)


def _global_names(code,*,nested=False):
    bodies = code_tree(code) if nested else (code,)
    return {op.argval for body in bodies for op in get_instructions(body) if op.opname == 'LOAD_GLOBAL'}


def _entry(code):
    ops = tuple(get_instructions(code))
    created = observed._one((op.offset for op in ops if op.opname == 'RETURN_GENERATOR'),
                            'original code has no generator creation point')
    resume = observed._one((op.offset for op in ops if op.opname == 'RESUME' and op.arg == 0),
                           'original code has no first resume point')
    _require(bool(code.co_flags & CO_GENERATOR) and created < resume, 'expected a synchronous generator before body entry')
    return dict(code=describe_code(code),creation=site(code,created),first_resume=site(code,resume),
                runtime_state='EXACT_GENERATOR_TYPE_IN_GEN_CREATED_WITH_LIVE_FRAME')


def _constructor(tree, name, fields):
    assignment = observed._one((row for row in tree.body if type(row) is ast.Assign and len(row.targets)==1
        and type(row.targets[0]) is ast.Name and row.targets[0].id==name), 'record constructor needs one original module definition')
    call = assignment.value
    _require(type(call) is ast.Call and type(call.func) is ast.Name and len(call.args)==2 and not call.keywords
        and all(type(arg) is ast.Constant and type(arg.value) is str for arg in call.args)
        and call.args[1].value.replace(',',' ').split()==fields, 'record must use the exact declared named tuple layout')
    importer = observed._one(((row, item) for row in tree.body if type(row) is ast.ImportFrom and not row.level
        and row.module=='collections' for item in row.names if item.name=='namedtuple'
        and (item.asname or item.name)==call.func.id), 'record declaration must resolve to original collections.namedtuple import')
    return dict(symbol=name,declared_type=call.args[0].value,fields=fields,assignment=observed._span(assignment),
        factory_import=dict(source=importer[0].module,symbol=importer[1].name,span=observed._span(importer[0])),
        live_carrier='EXACT_DECLARATION_RESULT_AND_STANDARD_NAMEDTUPLE_NEW_ON_TUPLE_STORAGE',
        comparison='TUPLE_COMPARISON_WITHOUT_OVERRIDE_AND_EXACT_TYPED_LEAVES')


def _bindings(plan, score, tree, codes, threshold, depth):
    factory_names = _global_names(codes['factory'],nested=True)
    score_names = _global_names(codes['score'],nested=True)
    imports = plan['builtin_calls']
    _require(all(origin == ['itertools','count'] for origin in imports.values()),'unexpected iterator import protocol')
    immutable = dict(plan['scalar_globals'])
    for name,value in score['globals'].items():
        _require(name not in immutable or immutable[name]==value,'generator and score assume different global values')
        immutable[name] = value
    _require(all(type(value) is int and -(1<<63)<=value<(1<<63) for value in immutable.values()),
             'native folded global requires exact signed-64 integer')
    factory_nonconstants = set(plan['integer_rows'])|set(imports)|{plan['constructor']}
    factory_builtins = factory_names - set(plan['scalar_globals']) - factory_nonconstants
    scorer_builtins = score_names - set(score['globals']) - {plan['table']}
    _require(factory_names == set(plan['scalar_globals'])|factory_nonconstants|factory_builtins
        and score_names == set(score['globals'])|{plan['table']}|scorer_builtins
        and factory_builtins <= _SAFE_BUILTINS and scorer_builtins <= _SAFE_BUILTINS
        and scorer_builtins == set(score['builtins']) and _global_names(codes['record_source'])=={threshold},
        'compiled generator, scorer or filter has an unmodeled global binding')
    _require(threshold not in immutable and depth in codes['record_source'].co_freevars,
             'filter requires an original live threshold and lexical capture')
    return dict(receiver=dict(symbol=plan['receiver'],state_fields=plan['state_fields'],order=plan['state_field_order'],
                  ports=plan['inputs'],condition='EXACT_ORIGINAL_TYPE_DESCRIPTOR_LAYOUT_AND_PRIMITIVE_FIELDS'),
        callable_resolution={role:dict(symbol=codes[role].co_name,code=describe_code(codes[role]),
            required='EXACT_FUNCTION_AND_GLOBALS_IN_ORIGINAL_RECEIVER_MRO_WITH_NO_LOOKUP_HOOK')
            for role in ('factory','score')},
        sort=dict(symbol='sorted',carrier=['builtins','sorted'],reverse=plan['descending']),
        predicate=dict(threshold=dict(symbol=threshold,origin='RECORD_SOURCE_GLOBALS',type='EXACT_I64_RUNTIME_VALUE'),
                       nonzero=dict(symbol=depth,origin='CURRENT_SHARED_CLOSURE_CELL',type='EXACT_I64_RUNTIME_VALUE')),
        constructor=_constructor(tree,plan['constructor'],plan['record_fields']),
        bootstrap_builtins={name:['builtins',name] for name in sorted(factory_builtins|scorer_builtins)},
        captured_imports=imports,immutable_i64=immutable,
        integer_rows=dict(expected=plan['integer_rows'],required='EXACT_DICT_OF_EXACT_STR_TO_IMMUTABLE_I64_TUPLE'),
        score_table=dict(symbol=plan['table'],provider=plan['transition'],locator='#/table_versions',
            observed_versions=sorted(score['table_versions']),required='EXACT_CONTENT_AND_SAME_PINNED_LIVE_VERSION'),
        original_reads=dict(factory=sorted(factory_names),score=sorted(score_names),record_source=[threshold]))


def _creation(plan, codes, sort_call, factory, depth, owner, source_proof):
    expression, = sort_call.args
    component, = expression.generators
    factory_code, record, outer = (codes[role] for role in ('factory','record_source','sorted_owner'))
    code_index, source = child_code(outer,expression)
    _require(source is record and record.co_varnames[:1]==('.0',) and not factory_code.co_freevars,
             'original implicit iterator and receiver-only factory calling convention changed')
    iterator = observed._event(outer,factory,'GET_ITER')
    record_call = observed._event(outer,expression,'CALL')
    _require(plan['factory_site']['offset'] < iterator['offset'] < record_call['offset'] < plan['sorted_site']['offset'],
             'factory and record create operations must execute before the original sort')
    extra = set(record.co_freevars)-{plan['receiver'],depth}
    private = observed._one(extra,'scored record requires one private assignment cell')
    _require(private in outer.co_cellvars and {plan['receiver'],depth} <= set(outer.co_freevars),
             'original scored-record capture scope changed')
    occurrences = [node for node in ast.walk(owner) if type(node) is ast.Name and node.id==private]
    expression_nodes = set(ast.walk(expression))
    _require(occurrences and all(node in expression_nodes for node in occurrences)
             and not any(type(node) in (ast.Global,ast.Nonlocal) and private in node.names for node in ast.walk(owner)),
             'suppressed scored-record cell write is visible elsewhere in the original owner')
    generator_ops = list(get_instructions(record))
    cell_writes = [site(record,op.offset) for op in generator_ops if op.opname=='STORE_DEREF']
    _require(len(cell_writes)==1 and all(op.argval==private for op in generator_ops
        if op.opname in ('STORE_DEREF','DELETE_DEREF')) and not any(op.opname=='DELETE_DEREF' for op in generator_ops),
        'record body mutates a live external capture')
    _require(type(component.target) is ast.Name and len(expression.generators)==1,
             'unexpected record-source iterator protocol')
    arm = observed._one((row for row in source_proof['paths']['scored'] if row['role']=='SORTED_DELEGATION'),
                        'original provenance did not classify the sorted arm')
    _require(arm['compiled_score_store']==cell_writes[0] and arm['sorted_call']==plan['sorted_site'],
             'original F4 proof does not bind the private cell and exact sorted call')
    return dict(sorted_owner=describe_code(outer),sort_call=plan['sorted_site'],compiled_child_constant=code_index,
        original_factory_call=plan['factory_site'],iterator_creation=iterator,record_creation=record_call,
        eager_factory_prefix_preserved=True,selected_owner_continues_original_yield_from=True,
        record_iterator=dict(**_entry(record),iterator_port='.0',freevars=list(record.co_freevars)),
        factory_iterator=dict(**_entry(factory_code),receiver_formal=factory_code.co_varnames[0]),
        same_receiver_required_in_both_frames=True,shared_live_captures=sorted({plan['receiver'],depth}),
        private_score_cell=dict(symbol=private,write=cell_writes[0],ordinary_source_observer=False,
                               excluded_observers=['FRAME_REFLECTION','TRACE_PROFILE_OR_MONITOR_HOOKS']),
        original_output_pair=arm['carrier'])


def project(plan_path, provenance_path):
    plan_ref = _reference(plan_path,sorted_region.SCHEMA)
    proof_ref = _reference(provenance_path,provenance.SCHEMA)
    plan = read_artifact(plan_ref)
    source_proof = provenance.check(proof_ref['uri'])
    _require(read_artifact(proof_ref)==source_proof and source_proof['evidence']==plan['evidence']
        and source_proof['source']==plan['source'] and
        plan==sorted_region.project(plan['evidence']['uri'],plan['transition']['uri']),
        'plan and provenance must reconstruct one original source version')
    score = read_artifact(plan['scorer'])
    score_capture = read_artifact(score['capture'])
    transition = read_artifact(plan['transition'])
    evidence = read_artifact(plan['evidence'])
    _require(score['source']==score_capture['source']==plan['source'] and transition['score_plan']==plan['scorer']
        and set(score_capture['table_versions'])==set(transition['table_versions'])
        and all(score_capture['table_versions'][version]['rows']==transition['table_versions'][version]['rows']
                for version in transition['table_versions']), 'sort needs the exact scorer and original versioned table contents')
    tree,code_map = observed._compile(plan['source'])
    descriptors = ('sorted_owner','record_source','factory','score')
    codes = {role:code_map[plan[role]['qualname']] for role in descriptors}
    _require(all(describe_code(code)==plan[role] for role,code in codes.items()),'compiled guard body differs from original plan')
    _require(not codes['score'].co_freevars,'scorer uses a closure outside the admitted parameter and global bindings')
    raw = read_artifact(evidence['capture'])
    selection = read_artifact(raw['selection'])
    arm = observed._one((row for row in selection['arms'] if row['arm']=='SORTED_DELEGATION'),'exact sorted source arm required')
    sort_call, factory, threshold, depth = sorted_region._calls(tree,code_map,selection,arm)
    _require(type(sort_call.func) is ast.Name and sort_call.func.id=='sorted' and depth is not None
        and plan['predicate']==dict(threshold=threshold,nonzero_capture=depth), 'unsupported exact runtime filter shape')
    caller = _definition(tree,selection['caller']['qualname'])
    owner = observed._one((row for row in caller.body if type(row) is ast.FunctionDef
        and row.name==codes['sorted_owner'].co_name), 'source sorting owner absent')
    facts = provenance._OriginalFacts(evidence)
    factory_row,factory_ref = facts.coordinate('calls',factory,lambda row:not row.positional and not row.keywords
                                               and row.positional_pack is None and row.keyword_pack is None)
    factory_receiver = facts.name(factory.func.value,False)
    _require(factory_receiver['binding']==source_proof['candidate_input']['receiver_at_transition']['binding'],
             'factory call does not receive the original scoring position')
    facts.material.check()
    return dict(schema=SCHEMA,schema_version=1,plan=plan_ref,provenance=proof_ref,source=plan['source'],
        catalog=evidence['catalog'],source_calls=dict(factory=dict(**factory_ref,receiver=factory_receiver,
            world_in=encode(facts.port(factory_row.world)),world_out=encode(facts.port(factory_row.result_world))),
            scorer=evidence['original_f4']['first_scores']['SORTED_DELEGATION'],sort=evidence['original_f4']['sorted_container']),
        creation=_creation(plan,codes,sort_call,factory,depth,owner,source_proof),
        bindings=_bindings(plan,score_capture,tree,codes,threshold,depth),
        replacement_boundary=dict(fallback='ORIGINAL_SORT_OF_UNTOUCHED_GENERATORS_ON_ANY_MISS',
            guard_reads='ORDINARY_TYPE_AND_DICT_INTROSPECTION_WITHOUT_USER_DISPATCH_OR_EARLY_GLOBAL_LOAD',
            substituted_effects=['GENERATOR_START_RESUMPTION_AND_FINALIZATION','INTERMEDIATE_RECORD_ALLOCATION',
                'PRIVATE_SCORE_CELL_WRITE','EXACT_SCORER_AND_FILTER','STABLE_SORT_AND_LIST_MATERIALIZATION',
                'SORT_COMPARATOR_INVOCATION_SEQUENCE'],
            preserved=['ORIGINAL_FACTORY_CALL_AND_GET_ITER','ORIGINAL_OUTER_YIELD_FROM_AND_CONSUMER',
                'EQUIVALENT_LEXICOGRAPHIC_ORDER_AND_STABLE_TIES','EXACT_RECORD_TYPE_AND_SEQUENCE_ORDER'],
            native_effect_on_miss='PRIVATE_OUTPUT_ONLY_UNTIL_COMPLETE',
            live_interval='PIN_TABLE_MAP_SCALAR_AND_TYPE_BINDINGS_THROUGH_NATIVE_COMMIT'),
        qualifications=dict(runtime_retirement_scope='TWO_SELECTED_EXACT_UNSTARTED_GENERATORS',
            factory_creation_observable_prefix_retained=True,private_cell_write_unobservable_under_declared_access=True,
            direct_original_score_table_writers=source_proof['frontiers']['direct_source_writes'],
            proof_frontiers=['UNEXCLUDED_TRACE_PROFILE_MONITOR_AND_FRAME_REFLECTION','ASYNCHRONOUS_SIGNAL_OR_REENTRANT_EFFECT',
                'UNPINNED_TABLE_CONFIG_AND_TYPE_MUTATION','GENERATOR_FINALIZATION_AND_ALLOCATION_FAILURE_EFFECTS'],
            complete_effect_closure=False,runtime_guards_verified=False,deployment_authorized=False,
            target_executed_by_projection=False))


def check(path):
    result = read_artifact(_reference(path,SCHEMA))
    read_artifact(result['plan'])
    read_artifact(result['provenance'])
    _require(result==project(result['plan']['uri'],result['provenance']['uri']),
             'stored guard does not match independent source and F4 replay')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--provenance', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check', type=Path)
    args = parser.parse_args(argv)
    inputs = (args.plan, args.provenance, args.out)
    if args.check and not any(inputs):
        result = check(args.check)
    elif not args.check and all(inputs):
        result = project(args.plan, args.provenance)
        with args.out.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
    else:
        parser.error('provide --plan, --provenance and --out, or --check')
    print(json.dumps(dict(schema=SCHEMA, qualifications=result['qualifications'])))


if __name__ == '__main__':
    main()
