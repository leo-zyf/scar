"""Locate source call sites for a dead-item proof within an admitted CSR loop.

The bounded graph names the original paths and targets. The caller must still
prove the returned item's liveness and guard live providers and effects.
"""

import ast
import json

from scar.analysis.csr_topology_scope import REGION_PROOF, SCHEMA as TOPOLOGY_SCOPE
from scar.analysis.sparse_parallel_scope import SCHEMA as PARALLEL_SCOPE
from scar.ir.execution_recipe import read_artifact


SCHEMA = 'scar.csr-dead-item-sites'
_ENTRY_GRAPH = 'scar.bounded-entry-calls'
_NUMPY_DISPATCHER = {'module': 'numpy', 'qualname': '_ArrayFunctionDispatcher'}


def _document(reference, schema):
    if reference['schema'] != schema:
        raise ValueError('unexpected CSR source-scope artifact type')
    document = read_artifact(reference)
    if type(document['schema_version']) is not int or document['schema_version'] != 1:
        raise ValueError('unsupported CSR source-scope artifact version')
    return document


def _span(position):
    line, end_line, column, end_column = position
    return line, column, end_line, end_column


def _syntax_span(node):
    return node.lineno, node.col_offset, node.end_lineno, node.end_col_offset


def _calls(function):
    return {_syntax_span(node): node for node in ast.walk(function) if type(node) is ast.Call}


def _module_attribute(node, module, name):
    return (type(node) is ast.Attribute and type(node.value) is ast.Name
            and node.value.id == module and node.attr == name)


def _observations(graph):
    for index, row in enumerate(graph['observed_targets']):
        observation = row['observation']
        if row['entry_scope_reachable'] and observation['caller_site']['position'] is not None:
            yield f'#/observed_targets/{index}', row, observation


def _row_edit(graph, topology, scope, proof):
    locator = topology['copies_and_consumers']['region_attempt']
    for key, row, observation in _observations(graph):
        if key != locator:
            continue
        target = observation['target_code']
        if (row['source_body'] is not None and observation['target_kind'] == 'PYTHON_FUNCTION'
                and observation['caller_site']['code'] == scope['source_body']['code_key']
                and target['filename'] == proof['source']['uri']
                and target['key'] == proof['region']['code_key']
                and target['qualname'] == proof['region']['qualname']):
            return locator, observation
    raise ValueError('CSR row-edit proof lacks its original observed call edge')


def _producer_calls(function, alias):
    for assignment in ast.walk(function):
        if (type(assignment) is ast.Assign and len(assignment.targets) == 1
                and type(assignment.targets[0]) is ast.Name and type(assignment.value) is ast.Call):
            call = assignment.value
            if _module_attribute(call.func, alias, 'setdiff1d'):
                index = call.args[0] if len(call.args) == 2 else None
                if type(index) is ast.Call and _module_attribute(index.func, alias, 'arange'):
                    yield _syntax_span(call), _syntax_span(index)


def find_csr_dead_item_sites(selection):
    """Use immutable receipts to find candidate spans, without executing or replaying the target."""
    scope_ref = selection['source']
    scope = _document(scope_ref, PARALLEL_SCOPE)
    topology = _document(scope['topology_scope'], TOPOLOGY_SCOPE)
    proof = _document(topology['region_proof'], REGION_PROOF)
    graph_ref = scope['source_graph']
    graph = _document(graph_ref, _ENTRY_GRAPH)
    if (scope['entry']['source'] != selection['entry'] or topology['entry'] != scope['entry']
            or topology['source_scope'] != scope['source_scope'] or topology['source_graph'] != graph_ref
            or proof['entry'] != selection['entry'] or proof['source_graph'] != graph_ref):
        raise ValueError('CSR row-edit source identity differs from the admitted entry')

    row_edit, root = _row_edit(graph, topology, scope, proof)
    source = proof['source']
    filename = source['uri']
    module = ast.parse(read_artifact(source, json_document=False), filename=filename)
    bodies = {body.name: body for body in module.body if type(body) is ast.FunctionDef}
    caller = root['target_code']
    caller_calls = _calls(bodies[caller['qualname']])
    alias = proof['region']['inputs']['P']
    joined = tuple((locator, row, observation) for locator, row, observation in _observations(graph)
        if observation['run'] == root['run'] and observation['process_id'] == root['process_id'])
    candidates = []
    for call_locator, call_row, call in joined:
        call_site, callee = call['caller_site'], call['target_code']
        if (call['target_kind'] != 'PYTHON_FUNCTION' or call_site['code'] != caller['key']
                or call_row['source_body'] is None or callee['filename'] != filename
                or callee['qualname'] not in bodies):
            continue
        call_span = _span(call_site['position'])
        expression = caller_calls.get(call_span)
        if (expression is None or type(expression.func) is not ast.Name
                or expression.func.id != callee['qualname']):
            continue
        for producer_span, range_span in _producer_calls(bodies[callee['qualname']], alias):
            selected = {}
            for locator, row, observed in joined:
                position = observed['caller_site']
                if row['unit'] != call_row['unit'] or position['code'] != callee['key']:
                    continue
                span = _span(position['position'])
                metadata = json.loads(observed['metadata_json'])
                carrier = metadata['native_boundary']
                if (span == producer_span and metadata['type'] == _NUMPY_DISPATCHER
                        and observed['target_kind'] == 'OTHER_CALLABLE'):
                    role = 'producer'
                elif (span == range_span and observed['target_kind'] == 'NATIVE_CARRIER'
                        and (carrier.get('module'), carrier.get('name')) == ('numpy', 'arange')):
                    role = 'range'
                else:
                    continue
                if role in selected:
                    raise ValueError('CSR producer has multiple observed targets')
                selected[role] = dict(locator=locator, operation=row['operation'], evidence=observed['evidence'])
            if set(selected) != {'producer', 'range'}:
                continue
            candidates.append(dict(source=source, filename=filename, caller_key=caller['key'],
                callee_key=callee['key'], call_span=list(call_span), producer_span=list(producer_span),
                witnesses=dict(source_scope=scope_ref, entry_graph=graph_ref, row_edit=row_edit,
                    callee=dict(locator=call_locator, operation=call_row['operation'], evidence=call['evidence']),
                    **selected)))
    return dict(schema=SCHEMA, schema_version=1, candidates=candidates, effects_closed=False,
        frontiers=['RETURN_ITEM_LIVENESS_UNPROVEN_BY_CALL_OBSERVATION',
            'LIVE_NUMPY_DISPATCH_AND_INPUT_LAYOUT_REQUIRE_GUARDS',
            'ALLOCATION_FAILURE_AND_FRAME_OBSERVERS_REQUIRE_NORMAL_COMPLETION_SCOPE'])
