"""Admit a source-matched synchronous child process to an explicit native trial."""

import argparse
import ast
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import posixpath
import subprocess

from scar.analysis.source_controller_children import SCHEMA as CHILDREN, check as check_children
from scar.compiler.plan_native_controller import SCHEMA as CONTROLLER
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.implementation_plan import CostEstimate
from scar.ir.record_codec import encode
from scar.trace.code_snapshot import code_key
from scar.trace.files import open_binary

SCHEMA = 'scar.controller-child-plan'
_EXECUTABLE = 'scar.native-child-executable'


def _require(condition, detail):
    if not condition:
        raise ValueError(detail)


def _reference(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _at(reference, locator):
    return dict(artifact=reference, locator=locator)


def _exact_row(anchor):
    with open_binary(anchor['path']) as stream:
        for line_number, raw in enumerate(stream, 1):
            if line_number == anchor['line_number']:
                _require(sha256(raw).hexdigest() == anchor['line_sha256'], 'original call anchor differs')
                row = json.loads(raw)
                _require(row['index'] == anchor['index'], 'original call attempt index differs')
                return row
    raise ValueError('original call attempt row unavailable')


def _original_child_site(effects, child):
    locators = child['source_observations']
    _require(len(locators) == 1, 'requires one exact original F4 child-call observation')
    prefix, bodies, body_id, calls, call_id, observations, observation_id = locators[0].split('/')
    _require((prefix, bodies, calls, observations) == ('#', 'bodies', 'calls', 'observations')
             and all(index.isdecimal() for index in (body_id, call_id, observation_id)),
             'child must originate from an exact original body call')
    body = effects['bodies'][int(body_id)]
    call = body['calls'][int(call_id)]
    observed = call['observations'][int(observation_id)]
    scope = child['call_scope']
    _require(observed['attempt'] == scope['call_attempt'] and observed['callee'] == scope['callee']
             and observed['entry'] == scope['frame_entry'] and observed['exit'] == scope['frame_exit']
             and observed['entry_binding'] == scope['entry_binding']
             and observed['completion'] == 'NORMAL' and observed['normal_completion_observed'],
             'child process must match the original normally completed call and frame')
    owners = []
    for ordinal, effect in enumerate(effects['effects']):
        for observation in effect['observations']:
            if (body['body'] in effect['bodies'] and observation['body'] == body['body']
                    and observation['callee'] == observed['caller']
                    and observation['trace_directory'] == observed['trace_directory']
                    and observation['completion'] == 'NORMAL' and observation['normal_completion_observed']):
                owners.append((ordinal, effect, observation))
    _require(len(owners) == 1, 'child caller has no unique original controller effect activation')
    return body, call, observed, owners[0]


def _argument_vector(body):
    """Extract a closed source shape. F4 and original activations bind its calls."""
    source = body['source']
    reference = _reference(source['path'], 'scar.original-python-source')
    raw = read_artifact(reference, json_document=False)
    _require(reference['sha256'] == source['sha256'], 'original body source changed')
    functions = [node for node in ast.walk(ast.parse(raw, filename=source['path']))
                 if type(node) is ast.FunctionDef and (node.lineno, node.col_offset)
                 == (source['line'], source['column'])]
    _require(len(functions) == 1, 'original child controller body is not a unique synchronous definition')
    function = functions[0]
    params = body['parameters']
    _require(len(params) == 1 and not function.decorator_list
             and params[0]['kind'] == 'POSITIONAL_OR_KEYWORD'
             and params[0]['default_source'] is None,
             'original child controller requires one ordinary explicit receiver formal')
    statements = function.body
    if (type(statements[0]) is ast.Expr and type(statements[0].value) is ast.Constant
            and type(statements[0].value.value) is str):
        statements = statements[1:]
    _require(len(statements) in (1, 2) and type(statements[-1]) is ast.Expr
             and type(statements[-1].value) is ast.Call, 'original child controller has unsupported body work')
    call = statements[-1].value
    _require(len(call.args) == 1 and not call.keywords, 'original provider must receive just the argument vector')
    vector = call.args[0]
    if len(statements) == 2:
        assignment = statements[0]
        _require(type(vector) is ast.Name and type(assignment) is ast.Assign and len(assignment.targets) == 1
                 and type(assignment.targets[0]) is ast.Name and assignment.targets[0].id == vector.id,
                 'original child vector must be a single immediately consumed assignment')
        vector = assignment.value
    _require(type(vector) in (ast.List, ast.Tuple) and vector.elts, 'original ordered argument vector unavailable')
    terms, joins, fields = [], [], set()
    formal = params[0]
    for term in vector.elts:
        if type(term) is ast.Constant:
            _require(type(term.value) is str and '\0' not in term.value, 'invalid original argument literal')
            term.value.encode('utf-8')
            terms.append(dict(kind='LITERAL', value=term.value))
            continue
        _require(type(term) is ast.Call and len(term.args) == 2 and not term.keywords,
                 'argument must be an exact literal or a qualified two-operand path join')
        directory, basename = term.args
        _require(type(directory) is ast.Attribute and type(directory.value) is ast.Name
                 and directory.value.id == formal['name'] and type(basename) is ast.Constant
                 and type(basename.value) is str and basename.value not in ('', '.', '..')
                 and '/' not in basename.value and '\0' not in basename.value,
                 'directory join must use one original receiver field and a plain relative component')
        basename.value.encode('utf-8')
        fields.add(directory.attr)
        terms.append(dict(kind='DIRECTORY_JOIN', component=basename.value))
        joins.append(term)
    _require(len(fields) == 1 and terms[0]['kind'] == 'LITERAL' and bool(terms[0]['value']),
             'requires one original directory field and an original executable token')
    field = next(iter(fields))
    field_gets = [row for row in body['field_accesses'] if row['kind'] == 'GET' and row['selector'] == field
                  and any(path['root_kind'] == 'FUNCTION_PARAMETER' and path['root'] == formal['formal']
                          and not path['selectors'] for path in row['receiver_paths'])]
    _require(len(field_gets) == len(joins), 'source receiver field reads are not all bound in original F4')
    return reference, call, joins, terms, [formal['name'], field], [row['source_fact'] for row in field_gets]


def _span(node):
    return [node.lineno, node.end_lineno, node.col_offset, node.end_col_offset]


def _observe_call(body, caller, node, provider, arity):
    key = code_key(provider.__code__)
    matches = []
    for row in body['calls']:
        for occurrence in row['observations']:
            if (occurrence['caller'] != caller['caller']
                    or occurrence['trace_directory'] != caller['trace_directory']
                    or not occurrence['normal_completion_observed']
                    or occurrence['completion'] != 'NORMAL'
                    or occurrence['target']['kind'] != 'FUNCTION' or occurrence['target']['code'] != key):
                continue
            attempt = _exact_row(occurrence['attempt'])
            site = attempt['site']
            if (site['position'] == _span(node) and site['code'] == body['source']['code_key']
                    and attempt['kind'] == 'CALL_ATTEMPT' and attempt['dispatch']['kind'] == 'EXACT_FUNCTION'
                    and attempt['dispatch']['code'] == key):
                matches.append((row, occurrence, site))
    _require(len(matches) == 1, 'source syntax call has no exact original F4 activation for the standard provider')
    row, occurrence, site = matches[0]
    _require(len(row['positional']) == arity and not row['keywords']
             and row['keyword_pack'] is None and row['positional_pack'] is None,
             'original F4 provider argument protocol differs')
    return row, occurrence, dict(source_result=row['source_result'], source_fact=row['source_fact'],
                                 attempt=occurrence['attempt'], site=site, target=key)


def _provider_defaults():
    run = subprocess.run
    popen = subprocess.Popen
    run_defaults = {'input': None, 'capture_output': False, 'timeout': None, 'check': False}
    popen_defaults = dict(executable=None, stdin=None, stdout=None, stderr=None, preexec_fn=None,
                         close_fds=True, shell=False, cwd=None, env=None, restore_signals=True,
                         start_new_session=False, pass_fds=(), user=None, group=None, extra_groups=None,
                         umask=-1, process_group=None)
    _require(run.__globals__['Popen'] is popen and run.__kwdefaults__ == run_defaults,
             'current local subprocess.run provider defaults differ')
    signature = inspect.signature(popen)
    for name, expected in popen_defaults.items():
        actual = signature.parameters[name].default
        _require(type(actual) is type(expected) and actual == expected, 'current subprocess.Popen default differs: ' + name)
    return dict(run=dict(code=code_key(run.__code__), defaults=run_defaults),
                popen=dict(code=code_key(popen.__init__.__code__), defaults=encode(popen_defaults)),
                join=dict(code=code_key(posixpath.join.__code__)),
                source=_reference(inspect.getsourcefile(run), 'scar.local-stdlib-provider-source'))


def _fixture(terms, argv):
    _require(len(terms) == len(argv), 'source argument count differs from original exec')
    directories = set()
    for term, value in zip(terms, argv, strict=True):
        if term['kind'] == 'LITERAL':
            _require(term['value'] == value, 'source literal differs from original exec')
        else:
            directory = posixpath.dirname(value)
            _require(posixpath.isabs(directory) and posixpath.join(directory, term['component']) == value,
                     'source directory join differs from original exec')
            directories.add(directory)
    _require(len(directories) == 1, 'original joined exec arguments have different directory inputs')
    return next(iter(directories))


def _derive(children_ref, controller_ref):
    children = check_children(children_ref['uri'])
    _require(children == read_artifact(children_ref) and children['schema_version'] == 1
             and len(children['children']) == 1 and not children['frontiers'],
             'requires one replayed child with complete source-scoped original spawn/exec/wait')
    child = children['children'][0]
    _require(child['same_scope_spawn_and_wait'] is True and child['exit_status'] == 0,
             'requires one successfully completed original synchronous child')
    effects_ref = children['source_effects']
    effects = read_artifact(effects_ref)
    controller = read_artifact(controller_ref)
    source_ref = controller['source']
    source = read_artifact(source_ref)
    _require(effects['controller'] == source_ref and source['effect_closure_proven'] is False
             and controller['replay']['runtime_effect_mode'] == 'ORIGINAL_SYNCHRONOUS_EFFECT_PROVIDER_REQUIRED',
             'child evidence and controller plan must name the same original source authority')
    body, original_call, occurrence, (ordinal, effect, owner) = _original_child_site(effects, child)
    site = source['effects'][ordinal]
    _require(all(site[key] == effect[key] for key in ('id', 'site', 'source_fact'))
             and site['positional_count'] == 0 and site['keyword_names'] == []
             and site['result_policy'] == 'ORIGINAL_RELEASE_ONLY' and site['effects_closed'] is False,
             'original controller child effect routing or result policy changed')
    _require(owner['target']['kind'] == 'NORMALIZED_RECEIVER_FUNCTION' and owner['parameters_complete']
             and len(owner['parameters']) == 1 and owner['parameters'][0]['name'] == body['parameters'][0]['name']
             and owner['target']['receiver']['reference'] == owner['parameters'][0]['ref'],
             'original effect receiver was not observed as the declared body formal')
    provider = _provider_defaults()
    body_ref, run, joins, terms, field_path, field_gets = _argument_vector(body)
    run_row, run_occurrence, run_evidence = _observe_call(body, occurrence, run, subprocess.run, 1)
    _require(run_row == original_call and run_occurrence == occurrence,
             'source run occurrence differs from observed original exec parent')
    join_evidence = [_observe_call(body, occurrence, node, posixpath.join, 2)[2] for node in joins]
    _require(len(body['calls']) == 1 + len(join_evidence)
             and [row['attempt']['index'] for row in [*join_evidence, run_evidence]]
             == sorted({row['attempt']['index'] for row in [*join_evidence, run_evidence]}),
             'observed source work must be exactly the original ordered joins followed by the run call')
    fixture_dir = _fixture(terms, child['argv'])
    image = Path(child['executable'])
    _require(image.is_absolute() and image.resolve(strict=True) == image and image.is_file()
             and os.access(image, os.X_OK), 'current observed executable path must be a canonical executable file')
    executable = _reference(image, _EXECUTABLE)
    child_ref = _at(children_ref, '#/children/0')
    original = _at(source_ref, f'#/effects/{ordinal}')
    cost = CostEstimate(None, None, None, 'UNKNOWN', 'ORIGINAL_SYNCHRONOUS_CHILD_EFFECT_SERVICE',
                        (original, child_ref), ('NO_MATCHED_CLEAN_CHILD_EFFECT_SERVICE_COST',))
    return dict(
        schema=SCHEMA, schema_version=1, child_observation=children_ref,
        controller_plan=controller_ref, controller_source=source_ref,
        effect_site=dict(id=site['id'], ordinal=ordinal, site=site['site'], source_fact=site['source_fact']),
        status='CONDITIONAL_EXPLICIT_TRIAL_ONLY', default_backend='NO_OP', whole_entry_authorized=False,
        child=dict(
            executable=executable, argv=terms, directory_field_path=field_path,
            runtime_directory='EXPLICIT_ABSOLUTE_ARGUMENT',
            executable_resolution='PINNED_OBSERVED_EXECUTABLE',
            shell=False, inherit_cwd=True, inherit_environment=True, inherit_stdio=True,
            close_other_fds=True, restore_default_signals=True, check_exit_status=False,
            completion='SYNCHRONOUS_WAITPID',
            nonzero_and_signal='RETURN_NORMALLY_AND_DISCARD_ORIGINAL_COMPLETED_PROCESS',
            launch_or_wait_failure='PROPAGATE_BACKEND_FAILURE',
        ),
        evidence=dict(original_effect=original, observed_child=child_ref, source_body=body_ref,
                      body_activation=owner['attempt'], run=run_evidence, path_joins=join_evidence,
                      directory_field_gets=field_gets, current_local_provider=provider),
        fixture=dict(argv=child['argv'], directory=fixture_dir, exit_status=child['exit_status'],
                     original_exec=child['exec'], original_spawn=child['spawn'], original_wait=child['wait']),
        cost=encode(cost), native_compiled=False, target_program_executed=False, performance_measured=False,
        opaque_callbacks=[dict(id=row['id'], ordinal=index, action='PRESERVE_ORIGINAL_SYNCHRONOUS_PROVIDER')
                          for index, row in enumerate(source['effects']) if index != ordinal],
        frontiers=[
            'OBSERVED_EXECUTABLE_BYTES_BEFORE_PLANNING_AND_DYNAMIC_LIBRARY_IDENTITY_OPEN',
            'CURRENT_PINNED_EXECUTABLE_REPLACES_ENVIRONMENT_DEPENDENT_PATH_LOOKUP_FOR_TRIAL',
            'EXPLICIT_FIXTURE_DIRECTORY_REPLACES_ORDINARY_STABLE_RECEIVER_FIELD_READS_FOR_TRIAL',
            'INHERITED_ENVIRONMENT_CWD_STDIO_SIGNAL_AND_FILESYSTEM_CONTEXT_REQUIRE_MATCHED_TRIAL',
            'ORIGINAL_PROVIDER_MUTABILITY_AND_BACKEND_EXCEPTION_CLASS_EQUIVALENCE_OPEN',
            'OTHER_CALLBACKS_CHILD_CONTENT_AND_WHOLE_APPLICATION_EFFECTS_OPEN',
        ],
        deployment_authorized=False,
    )


def plan_controller_child(children, controller_plan, out):
    report = _derive(_reference(children, CHILDREN), _reference(controller_plan, CONTROLLER))
    with Path(out).resolve().open('x') as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return report


def read_controller_child_plan(path):
    reference = _reference(path, SCHEMA)
    report = read_artifact(reference)
    if report != _derive(report['child_observation'], report['controller_plan']):
        raise ValueError('controller child plan differs from original source, process and current provider replay')
    read_artifact(reference)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group(required=True)
    options.add_argument('--children')
    options.add_argument('--check')
    parser.add_argument('--controller-plan')
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    if args.check:
        if args.out or args.controller_plan:
            parser.error('--check only consumes the retained plan')
        report = read_controller_child_plan(args.check)
    else:
        if not args.out or not args.controller_plan:
            parser.error('preparation requires --controller-plan and --out')
        report = plan_controller_child(args.children, args.controller_plan, args.out)
    print(json.dumps(dict(status=report['status'], site=report['effect_site']['ordinal'],
                          arguments=len(report['child']['argv']), default_backend=report['default_backend'],
                          deployment_authorized=False), sort_keys=True))


if __name__ == '__main__':
    main()
