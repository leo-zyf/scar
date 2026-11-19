"""Project a recursive source region with balanced calls and explicit state effects."""

import argparse
import ast
from collections import defaultdict
import json
from pathlib import Path

from scar.analysis import scalar_loop_score_provenance as loop
from scar.analysis import scalar_score_reuse_evidence as observed
from scar.analysis import scalar_sorted_transition_provenance as fusion
from scar.compiler.scalar_table_region import _definition
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.record_codec import encode
from scar.trace.code_snapshot import describe_code


SCHEMA = 'scar.scalar-recursive-source-region'
_CONTROL = ast.If, ast.IfExp, ast.For, ast.While, ast.Try, ast.With, ast.Match
_NESTED = ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.GeneratorExp, ast.ListComp, ast.SetComp, ast.DictComp
_RETIRE = ast.Return, ast.Raise, ast.Break, ast.Continue


def _require(condition, detail):
    if not condition:
        raise ValueError('recursive scalar region: ' + detail)


def _operations(node, path=()):
    yield node, path
    if isinstance(node, _NESTED):
        return
    for label, child in ast.iter_fields(node):
        segment = (*path, (tuple(observed._span(node)), label)) if isinstance(node, _CONTROL) else path
        if isinstance(child, ast.AST):
            yield from _operations(child, segment)
        elif type(child) is list:
            for item in child:
                if isinstance(item, ast.AST):
                    yield from _operations(item, segment)


def _path(path):
    return [dict(control_span=list(span), arm=arm) for span, arm in path]


def _member(node, receiver):
    return type(node) is ast.Attribute and observed._name(node.value) == receiver


def _state_root(node, receiver):
    while type(node) in (ast.Subscript, ast.Attribute):
        if _member(node, receiver):
            return node.attr
        node = node.value
    return None


def _source_call(facts, code, node):
    compiled = observed._event(code, node, 'CALL')
    row, reference = facts.coordinate('calls', node, lambda value: True)
    return row, dict(reference, compiled=compiled, world_in=encode(facts.port(row.world)),
                     world_out=encode(facts.port(row.result_world)))


def _access(node, path, receiver, augmented, facts, code, imports):
    if _member(node, receiver):
        action = 'ATTRIBUTE_READ'
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            action = 'ATTRIBUTE_READ_WRITE' if node in augmented else 'ATTRIBUTE_WRITE'
            if isinstance(node.ctx, ast.Del):
                action = 'ATTRIBUTE_DELETE'
        return dict(field=node.attr, effect=action, source_span=observed._span(node),
                    control=_path(path), lookup='RECEIVER_ATTRIBUTE_PROTOCOL')
    if type(node) is ast.Subscript and (field := _state_root(node.value, receiver)) is not None:
        read = type(node.ctx) is ast.Load
        delete = type(node.ctx) is ast.Del
        _, reference = facts.coordinate('items', node, lambda value: value.write is not read, required=False)
        return dict(field=field, effect='ITEM_READ' if read else 'ITEM_DELETE' if delete else 'ITEM_WRITE',
                    source_span=observed._span(node), original_fact=reference,
                    index_span=observed._span(node.slice), control=_path(path),
                    lookup='EXACT_KEY_VALUE_NOT_ASSUMED')
    if type(node) is ast.Call and type(node.func) is ast.Attribute:
        field = _state_root(node.func.value, receiver)
        if field is not None:
            _, source = _source_call(facts, code, node)
            return dict(field=field, effect='CALL_ON_STATE_FIELD', method=node.func.attr,
                        source_span=observed._span(node), original_call=source, control=_path(path),
                        lookup='REQUIRES_DYNAMIC_METHOD_EFFECT_CONTRACT')
        module = imports.get(observed._name(node.func.value))
        if module is not None:
            _, source = _source_call(facts, code, node)
            return dict(effect='CALL_ON_IMPORTED_MODULE', import_module=module, member=node.func.attr,
                        source_span=observed._span(node), original_call=source, control=_path(path),
                        lookup='REQUIRES_EXTERNAL_RESOURCE_OR_EFFECT_CONTRACT')
    return None


def project(flow_path):
    reference = artifact_reference(Path(flow_path).resolve(strict=True), fusion.SCHEMA)
    flow = read_artifact(reference)
    guard = read_artifact(flow['source_guard'])
    evidence_ref = read_artifact(flow['transport'])['evidence']
    evidence = read_artifact(evidence_ref)
    proof = read_artifact(flow['provenance'])
    _require(guard['catalog'] == flow['catalog'] == evidence['catalog'] == proof['catalog']
             and guard['source'] == flow['source'] == evidence['source'] == proof['source'],
             'region effects must refer to the exact original source and unit facts')
    tree, codes = observed._compile(flow['source'])
    owner_proof = proof['record_scope']['owner']
    code = codes[owner_proof['qualname']]
    _require(describe_code(code) == owner_proof, 'original recursive owner code changed')
    function = _definition(tree, code.co_qualname)
    formals = function.args.posonlyargs + function.args.args
    receiver = formals[0].arg
    method = function.name
    calls = []
    state = defaultdict(list)
    effects, control, exits, callable_reads = [], [], [], []
    operations = [(node, path) for body in function.body for node, path in _operations(body)]
    augmented = {node.target for node, _ in operations if type(node) is ast.AugAssign}
    imports = {item.asname or item.name: item.name for stmt in tree.body if type(stmt) is ast.Import for item in stmt.names}
    selectors = {node.func for node, _ in operations if type(node) is ast.Call and _member(node.func, receiver)}
    facts = loop._OriginalFacts(evidence)
    for node, path in operations:
        if type(node) is ast.Call and _member(node.func, receiver):
            if node.func.attr == method:
                row, source = _source_call(facts, code, node)
                _require(len(node.args) == len(row.positional) and all(type(arg) is not ast.Starred for arg in node.args)
                         and len(node.args) < len(formals) and not (node.keywords or row.keywords),
                         'recursive transfer uses unmapped Python argument protocol')
                implicit = formals[1 + len(node.args):]
                defaults = function.args.defaults[len(function.args.defaults) - len(implicit):]
                _require(len(defaults) == len(implicit) and all(type(value) is ast.Constant for value in defaults),
                         'recursive callee arguments are not fully declared')
                calls.append(dict(original_call=source, target_candidate=owner_proof,
                    receiver=dict(symbol=receiver, binding='SAME_LEXICAL_RECEIVER'),
                    control=_path(path),
                    actuals=[dict(formal=formal.arg, port=encode(facts.port(port)),
                                  source_span=observed._span(expr))
                             for formal, port, expr in zip(formals[1:1+len(node.args)], row.positional, node.args, strict=True)],
                    omitted_defaults=[dict(formal=formal.arg, value=expr.value)
                                      for formal,expr in zip(implicit,defaults,strict=True)],
                    continuation='RETURN_OR_EXCEPTION_TO_THIS_CALL_SITE',
                    dynamic_target='UNVERIFIED_EXCEPT_ORIGINAL_OBSERVED_HOOK',
                    world_preservation='CHILD_EFFECT_WORLD_RETURNS_BEFORE_OWNER_RESUMES'))
            else:
                _, source = _source_call(facts, code, node)
                effects.append(dict(effect='OTHER_RECEIVER_METHOD_CALL', original_call=source, member=node.func.attr,
                                    control=_path(path), lookup='MAY_READ_OR_WRITE_SHARED_RECEIVER'))
        if isinstance(node, _CONTROL) or isinstance(node, _NESTED):
            control.append(dict(kind=type(node).__name__, span=observed._span(node), parents=_path(path)))
        if isinstance(node, _RETIRE):
            exits.append(dict(kind=type(node).__name__, span=observed._span(node), control=_path(path)))
        access = _access(node, path, receiver, augmented, facts, code, imports)
        if access is not None:
            if node in selectors:
                callable_reads.append(access)
            elif 'field' in access:
                state[access.pop('field')].append(access)
            else:
                effects.append(access)
    original_call = flow['graph']['nodes']['recurse']['source']['site']
    witness_index = next((i for i,row in enumerate(calls) if row['original_call']['compiled']==original_call),None)
    _require(witness_index is not None and len(flow['samples']) > 0, 'missing original observed recursive handoff')
    calls[witness_index]['dynamic_target'] = 'ORIGINAL_SELECTED_TRANSFER_OBSERVED'
    ledger = evidence['state_edges']
    facts.material.check()
    return dict(schema=SCHEMA,schema_version=1,source=flow['source'],catalog=flow['catalog'],
        original_fusion_flow=reference,source_state_observation=evidence_ref,
        region=dict(owner=owner_proof,formals=[item.arg for item in formals],receiver=receiver,
                    activation='RECURSIVE_REGION_INSTANCE_WITH_PRIVATE_LOCALS_AND_SHARED_RECEIVER'),
        graph=dict(recursive_back_edges=calls,observed_edge_index=witness_index,structured_regions=control,
                   exits=exits,receiver_state=[dict(field=name,effects=rows) for name, rows in sorted(state.items())],
                   receiver_callable_reads=callable_reads,
                   external_or_unclosed_effects=effects),
        evidence=dict(recursive_actual_samples=len(flow['samples']),observed_original_dictionary_pairs=len(ledger),
                      original_dictionary_pairs=ledger),
        contract=dict(mode='TYPED_CANDIDATE_BODY_WITH_ORIGINAL_EFFECT_FRONTIERS',
            self_calls_unrolled=False,recursive_shared_state_versions_require_order=True,
            runtime_alias_and_dispatch_proven_for_all_branches=False,whole_recursive_effects_closed=False,
            native_lowering_authorized=False,target_executed_by_projection=False))


def check(path):
    value = read_artifact(artifact_reference(Path(path).resolve(strict=True), SCHEMA))
    _require(value==project(value['original_fusion_flow']['uri']), 'recursive region differs from cold source replay')
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--flow', type=Path)
    inputs.add_argument('--check', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if args.check and not args.out:
        result = check(args.check)
    elif args.flow and args.out:
        result = project(args.flow)
        with args.out.open('x',encoding='utf-8') as stream:
            json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False)
            stream.write('\n')
    else:
        parser.error('provide --flow with --out, or --check')
    graph = result['graph']
    print(json.dumps(dict(recursive_sites=len(graph['recursive_back_edges']),
                          receiver_fields=len(graph['receiver_state']),
                          external_or_unclosed_effects=len(graph['external_or_unclosed_effects']))))


if __name__ == '__main__':
    main()
