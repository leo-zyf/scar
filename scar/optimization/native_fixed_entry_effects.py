"""Audit the observed effects of a source-bound fixed native launch."""

import argparse
import ast
import json
import logging
from pathlib import Path

from scar.analysis.bounded_entry_calls import SCHEMA as ENTRY
from scar.codegen.native_sparse_publication import PROFILE, SCHEMA as PUBLICATION, _verify_arguments
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.native_region_handoff import CAPTURE
from scar.trace.code_snapshot import code_key

SCHEMA = 'scar.native-fixed-entry-effect-audit'
DELIVERY = 'scar.native-ordered-publication-launch-plan'
FILE = 'scar.fixed-entry-io-file'


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _observations(graph, module_code):
    for index, row in enumerate(graph['observed_targets']):
        observed = row['observation']
        if row['entry_scope_reachable'] and observed['caller_site']['code'] == module_code:
            yield dict(locator=f'#/observed_targets/{index}', call_site=observed['caller_site'],
                       operation=row['operation'], metadata=json.loads(observed['metadata_json']))


def _span(node):
    return [node.lineno, node.end_lineno, node.col_offset, node.end_col_offset]


def _selected_call(tree, module, site):
    calls = [node for node in ast.walk(tree) if type(node) is ast.Call]
    # Both recognized source effects have one positional argument with no packs.
    _verify_arguments(calls, module, {'caller_site':site}, 1)
    selected, = (call for call in calls if _span(call) == site['position'])
    return selected


def _argv_binding(tree, module, graph, argv, declared_count):
    candidates = [row for row in _observations(graph, code_key(module)) if row['metadata'].get('constructed_type')
                  == {'module':'builtins', 'qualname':'int'}]
    conversion, = candidates
    call = _selected_call(tree, module, conversion['call_site'])
    argument, = call.args
    _require(isinstance(argument, ast.Subscript) and isinstance(argument.value, ast.Attribute)
        and argument.value.attr == 'argv' and isinstance(argument.value.value, ast.Name)
        and isinstance(argument.slice, ast.Constant) and type(argument.slice.value) is int
        and 1 <= argument.slice.value <= len(argv), 'fixed integer conversion must read the captured argv')
    name, index = argument.value.value.id, argument.slice.value
    _require(any(alias.name == 'sys' and (alias.asname or alias.name) == name
        for node in tree.body if isinstance(node, ast.Import) for alias in node.names),
        'source argv load must name the original imported sys module')
    definition, = (node for node in ast.walk(tree) if type(node) is ast.Assign and node.value is call)
    variable, = definition.targets
    _require(isinstance(variable, ast.Name), 'fixed count must have its original single lexical binding')
    guard = _positive_guard(tree, variable.id)
    _require(int(argv[index-1]) == declared_count and declared_count > 0,
        'source argument and the original released call count differ')
    return dict(scope='OBSERVED_LITERAL_ARGV_ONLY', original_argv=argv, argument_index=index,
        original_load_span=_span(argument), binding=variable.id, conversion=conversion,
        positive_guard_span=_span(guard), invalid_argument_behavior_preserved_by_native=False,
        fixed_call_count=declared_count)


def _positive_guard(tree, variable):
    guards = []
    for node in ast.walk(tree):
        if type(node) is not ast.If or type(node.test) is not ast.Compare:
            continue
        test = node.test
        if (isinstance(test.left, ast.Name) and test.left.id == variable
                and len(test.ops) == len(test.comparators) == 1 and type(test.ops[0]) is ast.LtE
                and isinstance(test.comparators[0], ast.Constant)
                and type(test.comparators[0].value) is int and test.comparators[0].value == 0
                and len(node.body) == 1 and type(node.body[0]) is ast.Raise):
            guards.append(node)
    guard, = guards
    return guard


def _logging(graph):
    origin = _ref(logging.__file__, 'scar.python-source')
    witnesses = []
    for index, row in enumerate(graph['observed_targets']):
        observed = row['observation']
        target = observed['target_code']
        if target and target['filename'] == origin['uri']:
            witnesses.append(dict(observation=f'#/observed_targets/{index}', target=target,
                call_site=observed['caller_site'], operation=row['operation'],
                conditional_static_reachability=row['entry_scope_reachable']))
    return dict(module=origin, observed_call_attempts=witnesses,
        log_emission_or_handler_effects_proven=False, logger_level_handlers_destinations=None,
        original_logging_retained_requirement=True)


def project(graph_path, capture_path, native_plan_path, original_stdout, original_stderr,
            native_stdout, native_stderr, *, generated_seed=None):
    graph_ref, capture_ref, plan_ref = (_ref(path, schema) for path, schema in (
        (graph_path, ENTRY), (capture_path, CAPTURE), (native_plan_path, DELIVERY)))
    graph, capture, plan = map(read_artifact, (graph_ref, capture_ref, plan_ref))
    program = read_artifact(graph['program'])
    publication_ref = plan['publication']['contract']
    contract = read_artifact(publication_ref)
    release = read_artifact(contract['release'])
    recipe = read_artifact(contract['execution_recipe'])
    expected_count = recipe['resources']['case']['call_count']
    argv = capture['launch']['argv']
    _require(program['entry'] == capture['entry']['uri'] == capture['launch']['entry_path']
        and capture['entry'] == contract['entry_source'] == release['entry']
        and argv == release['context']['argv']
        and plan['topology']['source'] == capture['seed_wire']
        and plan['full_python_entry'] is False and publication_ref['schema'] == PUBLICATION
        and contract['profile'] == plan['publication']['profile'] == PROFILE,
        'effect audit requires the same source, fixed input contract and native publication')
    source = read_artifact(capture['entry'], json_document=False)
    tree = ast.parse(source, filename=program['entry'])
    module = compile(source, program['entry'], 'exec', optimize=release['context']['interpreter']['optimize'],
                     dont_inherit=True)
    config = _argv_binding(tree, module, graph, argv, expected_count)
    printed, = (row for row in _observations(graph, code_key(module))
        if row['metadata']['native_boundary'].get('module') == 'builtins'
        and row['metadata']['native_boundary'].get('qualname') == 'print')
    _selected_call(tree, module, printed['call_site'])

    config_receipt = program['tables']['configuration.json']
    config_ref = dict(uri=config_receipt['path'], size_bytes=config_receipt['bytes'],
                      sha256=config_receipt['sha256'], schema='scar.build-configuration-inputs')
    inventory = read_artifact(config_ref)
    _require(len(inventory['materials']) == program['summary']['configuration_inputs'],
        'configuration table differs from its source graph denominator')

    stdout, err, native, native_err = (_ref(path, FILE) for path in
        (original_stdout, original_stderr, native_stdout, native_stderr))
    read_artifact(capture['seed_wire'], json_document=False)
    _require(stdout['sha256'] == capture['expected_stdout_sha256'] == native['sha256']
        and stdout['size_bytes'] == native['size_bytes'] and err['sha256'] == native_err['sha256']
        and err['size_bytes'] == native_err['size_bytes'],
        'captured native and original output effects differ for the fixed invocation')
    generated = _ref(generated_seed, capture['seed_wire']['schema']) if generated_seed is not None else None
    if generated is not None:
        _require(read_artifact(generated, json_document=False) == read_artifact(capture['seed_wire'], json_document=False),
            'generated seed differs from the original captured packet')
    sealed = plan.get('source_seed')
    if sealed is not None:
        _require(generated is not None and sealed['argv'] == argv
            and sealed['sha256'] == generated['sha256'] and sealed['size_bytes'] == generated['size_bytes']
            and bytes.fromhex(sealed['wire_hex']) == read_artifact(generated, json_document=False),
            'standalone source plan embeds a different fixed argument or seed operand')
    return dict(schema=SCHEMA, schema_version=1,
        inputs=dict(graph=graph_ref, capture=capture_ref, native_plan=plan_ref, publication=publication_ref,
            original_stdout=stdout, original_stderr=err, native_stdout=native, native_stderr=native_err,
            generated_seed=generated),
        configuration=dict(config, material_inventory=config_ref,
            runtime_configuration_proven=inventory['runtime_configuration_proven']),
        stdout=dict(original_call=printed, fixed_case_exact_bytes=stdout['size_bytes'],
            original_world_retained=True, native_contract=contract['profile']),
        stderr=dict(fixed_case_exact_bytes=err['size_bytes'], evidence_scope='CAPTURED_STREAMS_ONLY'),
        logging=_logging(graph),
        imports=dict(module_statement_spans=[_span(node) for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))], transitive_import_effects_closed=False),
        generated_seed_bytes_match=True if generated is not None else None,
        decision='STANDALONE_NATIVE_FIXED_CONTRACT_ONLY', source_transformation_authorized=False,
        full_source_effect_equivalence_proven=False, source_executed_by_audit=False)


def check(path):
    receipt = read_artifact(_ref(path, SCHEMA))
    inputs = receipt['inputs']
    generated = inputs['generated_seed']
    _require(receipt == project(*(inputs[key]['uri'] for key in (
        'graph', 'capture', 'native_plan', 'original_stdout', 'original_stderr', 'native_stdout', 'native_stderr')),
        generated_seed=generated['uri'] if generated is not None else None), 'fixed-entry effect audit does not replay')
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', type=Path)
    parser.add_argument('--generated-seed', type=Path)
    for name in ('graph', 'capture', 'native-plan', 'original-stdout', 'original-stderr',
                 'native-stdout', 'native-stderr', 'out'):
        parser.add_argument('--'+name, type=Path)
    args = parser.parse_args(argv)
    if args.check:
        print(check(args.check)['decision'])
        return
    inputs = (args.graph, args.capture, args.native_plan, args.original_stdout, args.original_stderr,
              args.native_stdout, args.native_stderr)
    if not all((*inputs, args.out)):
        parser.error('provide source, capture and native receipts and their output streams')
    result = project(*inputs, generated_seed=args.generated_seed)
    with args.out.open('x') as stream:
        json.dump(result, stream, sort_keys=True, allow_nan=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
