"""Plan native startup materialization from a source-qualified constructor slice."""

import argparse
import ast
import builtins
import inspect
import json
import math
from pathlib import Path
import struct

from scar.analysis.replay_transaction import replay_transaction
from scar.compiler.plan_native_controller import SCHEMA as CONTROLLER
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.implementation_plan import CostEstimate
from scar.ir.literals import PythonLiteral
from scar.ir.record_codec import decode, encode
from scar.trace.code_snapshot import code_key

SCHEMA = 'scar.controller-startup-plan'
SLICE = 'scar.source-controller-startup-slice'
COMPARISON = 'scar.observed-controller-startup-comparison'
OBSERVATION = 'scar.observed-controller-startup-material'
_FIXED_FRONTIERS = {
    'CURRENT_PROVIDER_VALUE_AND_EFFECT_CONTRACT_OPEN',
    'CURRENT_DEFAULT_VALUES_REQUIRE_DECLARATION_GUARDS',
    'ORDINARY_RECEIVER_ALIAS_AND_INTERVENING_WORLD_EFFECTS_REQUIRE_GUARDS',
}
_PROVIDERS = {
    ('builtins', 'len'): ('EXACT_ARRAY_LENGTH', 1),
    ('builtins', 'max'): ('MAXIMUM_EXACT_BUILTIN_NUMBERS', 2),
    ('numpy', 'zeros'): ('ALLOCATE_BINARY64_ZEROS_DEFAULT', 1),
    ('numpy', 'linspace'): ('LINEAR_SPACE_BINARY64_DEFAULT', 3),
    ('numpy', 'meshgrid'): ('COPY_MESHGRID_BINARY64_DEFAULT', 2),
    ('numpy', 'exp'): ('ELEMENTWISE_EXP_BINARY64', 1),
}


def _require(condition, detail):
    if not condition:
        raise ValueError('controller startup: ' + detail)


def _reference(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _at(reference, locator):
    return dict(artifact=reference, locator=locator)


def _only(rows, detail):
    rows = list(rows)
    _require(len(rows) == 1, detail)
    return rows[0]


def _scalar(row):
    kind, value = row['kind'], row['value']
    if kind == 'float64':
        _require(type(value) is float and math.isfinite(value)
                 and struct.pack('>d', value).hex() == row['binary64_be'],
                 'scalar witness disagrees with exact binary64 data')
        return dict(kind=kind, value=value, binary64_be=row['binary64_be'])
    _require(type(value) is {'bool': bool, 'int': int, 'str': str}.get(kind),
             'unsupported scalar witness type')
    return dict(kind=kind, value=value)


def _fields(rows):
    result = {}
    for row in rows:
        path = tuple(row['path'])
        _require(path not in result and len(path) == 2 and all(part.isidentifier() for part in path),
                 'original handoff needs unique direct fields')
        result[path] = row
    return result


def _witnesses(reference, initialization_ref, controller):
    report = read_artifact(reference)
    _require(report['schema_version'] == 1 and report['source_startup'] == initialization_ref
             and report['source_controller'] == controller['source']
             and report['observations_establish_future_invariance'] is False
             and report['numerical_provider_equivalence_proven'] is False,
             'requires original same-controller witness with no inferred future invariance')
    case_refs = report['observations']
    observations = [read_artifact(item) for item in case_refs]
    _require(len(observations) >= 2 and len({row['sha256'] for row in case_refs}) == len(observations),
             'requires two distinct original observations')
    first = observations[0]
    boundary = first['one_shot_boundary']
    original_controller = read_artifact(controller['source'])
    _require(boundary['code']['key'] == original_controller['source']['code_key']
             and boundary['receiver_formal'] in original_controller['parameters'],
             'source controller boundary and observed code/formal differ')
    common = ('environment', 'interpreter', 'loaded_modules', 'cpu_affinity', 'source_file',
              'source_controller', 'source_startup', 'original_entry', 'observer', 'prior_capture')
    for reference, row in zip(case_refs, observations, strict=True):
        seen = row['one_shot_boundary']
        _require(reference['schema'] == OBSERVATION and row['schema_version'] == 1
                 and all(row[key] == first[key] for key in common)
                 and row['source_file'] == report['source_file']
                 and row['performance_measurement'] is False and row['all_future_values_proven'] is False
                 and seen['receiver_class'] == boundary['receiver_class']
                 and seen['receiver_formal'] == boundary['receiver_formal']
                 and seen['code'] == boundary['code'] == report['controller_code']
                 and seen['distinct_arrays_written'] == boundary['distinct_arrays_written']
                 and seen['parameter_values'] == boundary['parameter_values'] == report['controller_parameter_values'],
                 'witnesses must match source, constructor, controller entry and runtime context')
    _require([row['argv'] for row in observations] == report['original_case_arguments']
             and len({tuple(row['argv']) for row in observations}) == len(observations),
             'witness command provenance differs')
    for key in ('source_file', 'original_entry', 'observer', 'prior_capture'):
        read_artifact(first[key], json_document=False)
    required_modules = {name: row for name, row in first['loaded_modules'].items() if row['loaded']}
    _require(bool(required_modules) and required_modules == {
        name: row for name, row in report['loaded_modules'].items() if row['loaded']
    } and report['loaded_module_receipts_equal'] is True, 'matched loaded provider receipts required')
    for row in required_modules.values():
        read_artifact(row['artifact'], json_document=False)
    return first, required_modules, [_fields(row['one_shot_boundary']['fields']) for row in observations]


def _array_witness(rows):
    first = rows[0]['geometry']
    shape = first['shape']
    size = math.prod(shape) * 8
    stride = 8
    strides = []
    for extent in reversed(shape):
        strides.insert(0, stride)
        stride *= extent
    _require(shape and all(type(extent) is int and extent > 0 for extent in shape)
             and first == dict(shape=shape, dtype='<f8', itemsize=8, allocation_nbytes=size,
                               byte_bounds=[0, size], byte_offset=0, byte_strides=strides, writeable=True),
             'only bounded full C-contiguous little-endian binary64 arrays are qualified')
    artifacts = [row['artifact'] for row in rows]
    expected = {key: artifacts[0][key] for key in ('sha256', 'size_bytes')}
    _require(expected['size_bytes'] == size and all(row['kind'] == 'array'
             and row['status'] == 'OBSERVED' and row['geometry'] == first
             and {key: row['artifact'][key] for key in expected} == expected for row in rows),
             'original array witness geometry or bytes differs')
    for item in artifacts:
        read_artifact(item, json_document=False)
    return first, expected


def _class_signature(first, initialization):
    actual = first['one_shot_boundary']['receiver_class']
    initializer = initialization['constructor']['initializer']
    reference = first['source_file']
    raw = read_artifact(reference, json_document=False)
    source_key = actual['constructor_code']['key']
    _require(source_key == initializer['code'] and actual['constructor_code']['filename'] == reference['uri']
             and initializer['normal_completion_observed'] is True
             and initialization['constructor']['observation']['target']['kind'] == 'TYPE_OBJECT'
             and actual['module'].isidentifier() and actual['qualname'].isidentifier(),
             'witness startup class differs from original observed type/initializer')
    module = ast.parse(raw, filename=reference['uri'])
    owner = _only((node for node in module.body if type(node) is ast.ClassDef and node.name == actual['qualname']),
                  'unique original constructor class required')
    constructor = _only((node for node in owner.body if type(node) is ast.FunctionDef and node.name == '__init__'),
                        'unique original normal initializer required')
    _require(not owner.decorator_list and not owner.bases and not owner.keywords and not constructor.decorator_list
             and all((type(node) is ast.FunctionDef and all(type(decorator) is ast.Name
                      and decorator.id == 'staticmethod' for decorator in node.decorator_list))
                     or (type(node) is ast.Expr and type(node.value) is ast.Constant
                         and type(node.value.value) is str) for node in owner.body)
             and not any(type(node) is ast.FunctionDef and node.name in
                         {'__new__', '__getattribute__', '__getattr__', '__setattr__', '__del__'} for node in owner.body)
             and (constructor.lineno, f'{owner.name}.{constructor.name}')
             == (actual['constructor_code']['first_line'], actual['constructor_code']['qualname']),
             'startup materializer expects an ordinary allocator and receiver field protocol')
    args = constructor.args
    _require(not args.posonlyargs and len(args.args) == 1 and not args.defaults
             and args.kwarg is args.vararg is None and all(node is not None for node in args.kw_defaults)
             and [arg.arg for arg in (*args.args, *args.kwonlyargs)] == actual['constructor_code']['parameters']
             == [row['name'] for row in initializer['arguments']],
             'original constructor binding must consist of receiver and literal-default keyword arguments')
    defaults = {arg.arg: ast.literal_eval(default)
                for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True)}
    _require(all(type(value) in (bool, int, float, str) and
                 (type(value) is not float or math.isfinite(value)) for value in defaults.values()),
             'startup constructor defaults must be exact builtin literals')
    return actual, initializer, defaults


def _source_runtime_options(entry, initializer):
    tree = ast.parse(read_artifact(entry, json_document=False), filename=entry['uri'])
    position = initializer['actual_caller']['position']
    call = _only((node for node in ast.walk(tree) if type(node) is ast.Call
                  and [node.lineno, node.end_lineno, node.col_offset, node.end_col_offset] == position),
                 'no exact original constructor CALL syntax')
    _require(not call.args and all(kw.arg and type(kw.value) is ast.Attribute
             and type(kw.value.value) is ast.Name for kw in call.keywords),
             'constructor runtime inputs must be direct original option reads')
    receiver = _only({kw.value.value.id for kw in call.keywords}, 'constructor inputs require one parser result')
    owner = _only((fn for fn in tree.body if type(fn) is ast.FunctionDef
                   and any(node is call for node in ast.walk(fn))), 'no original constructor caller')
    assign = owner.body[0]
    _require(type(assign) is ast.Assign and len(assign.targets) == 1
             and type(assign.targets[0]) in (ast.Tuple, ast.List)
             and type(assign.targets[0].elts[0]) is ast.Name and assign.targets[0].elts[0].id == receiver
             and type(assign.value) is ast.Call and type(assign.value.func) is ast.Name
             and not assign.value.args and not assign.value.keywords,
             'original option values must originate at one local zero-argument parser call')
    fn = _only((node for node in tree.body if type(node) is ast.FunctionDef
                and node.name == assign.value.func.id), 'no unique original option parser')
    body = fn.body[1:] if ast.get_docstring(fn) is not None else fn.body
    parser = body[0]
    _require(type(parser) is ast.Assign and len(parser.targets) == 1 and type(parser.targets[0]) is ast.Name
             and type(parser.value) is ast.Call and type(parser.value.func) is ast.Attribute
             and type(parser.value.func.value) is ast.Name and parser.value.func.attr == 'ArgumentParser'
             and not parser.value.args and not parser.value.keywords,
             'original option parser must use standard argparse defaults')
    alias = parser.value.func.value.id
    _require(any(type(row) is ast.Import and any(member.name == 'argparse' and
                 (member.asname or member.name) == alias for member in row.names) for row in tree.body),
             'documented argparse provider import missing')
    parser_name = parser.targets[0].id
    last = body[-1]
    _require(type(last) is ast.Return and type(last.value) is ast.Call
             and type(last.value.func) is ast.Attribute and type(last.value.func.value) is ast.Name
             and last.value.func.value.id == parser_name and last.value.func.attr == 'parse_known_args'
             and not last.value.args and not last.value.keywords,
             'original argument parser result route differs')
    options = {}
    for row in body[1:-1]:
        _require(type(row) is ast.Expr and type(row.value) is ast.Call
                 and type(row.value.func) is ast.Attribute and type(row.value.func.value) is ast.Name
                 and row.value.func.value.id == parser_name and row.value.func.attr == 'add_argument',
                 'original option declaration has other behavior')
        flags = [ast.literal_eval(node) for node in row.value.args]
        long = _only((flag for flag in flags if type(flag) is str and flag.startswith('--')),
                     'require one original long option')
        name = long[2:].replace('-', '_')
        values = {kw.arg: kw.value for kw in row.value.keywords if kw.arg != 'help'}
        if set(values) == {'action'}:
            _require(ast.literal_eval(values['action']) == 'store_true', 'unsupported source option action')
            kind, default = 'bool', False
        else:
            _require(set(values) == {'type', 'default'} and type(values['type']) is ast.Name,
                     'original option must have a builtin type and literal default')
            kind = {'float': 'float64', 'int': 'int', 'str': 'str'}[values['type'].id]
            default = ast.literal_eval(values['default'])
        _require(name.isidentifier() and name not in options, 'duplicate or invalid option declaration')
        options[name] = dict(flags=flags, kind=kind, default=default,
                             source_position=[row.lineno, row.end_lineno, row.col_offset, row.end_col_offset])
    actuals = {kw.arg: kw.value.attr for kw in call.keywords}
    _require(set(actuals.values()) <= options.keys()
             and set(actuals) == {row['name'] for row in initializer['arguments'] if row['role'] == 'KEYWORD'},
             'original option declaration and exact F4 keyword binding differ')
    return options, actuals


def _parse_witness_arguments(entry, option_specs, args):
    _require(args and str(Path(args[0]).resolve(strict=True)) == entry['uri'], 'witness original entry differs')
    parser = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    for name, option in option_specs.items():
        kind = option['kind']
        settings = (dict(action='store_true') if kind == 'bool'
                    else dict(type={'float64': float, 'int': int, 'str': str}[kind]))
        parser.add_argument(*option['flags'], dest=name, default=option['default'], **settings)
    parsed, remaining = parser.parse_known_args(args[1:])
    _require(not remaining, 'unrecognized option in original startup witness')
    return vars(parsed)


def _route_parameter(row, demand, actuals, ingress):
    parameter = _only(row['supplied_inputs'], 'runtime route must have one supplying formal')
    _require(parameter in actuals and not row['provider_calls'] and not row['control_guards']
             and not row['declared_defaults'] and not demand['content_writers'],
             'runtime route must be direct and provider free')
    writer = _only(demand['field_writers'], 'runtime route must have one original field write')
    selected = _only(writer['payload_paths'], 'runtime route must have one original payload')
    formal = ingress[parameter]['formal']
    _require(selected['root_kind'] == 'FUNCTION_PARAMETER' and selected['root'] == formal
             and not selected['selectors'] and selected['frontier'] is None,
             'original field assignment does not directly carry its startup formal')
    return parameter


def _guard_fixed_slice(row, demand):
    _require(row['origins'] and not row['supplied_inputs']
             and not row['open_value_sources']
             and set(row['state_frontiers']) <= _FIXED_FRONTIERS
             and row['startup_value_or_replacement_qualified'] is False,
             'fixed startup slice contains unresolved value dependencies: ' + row['id']
             + ' ' + repr(row['state_frontiers']))
    if demand['kind'] == 'array' and demand.get('uses'):
        _require(demand['content_writers'] and demand['same_observed_array_arguments'],
                 'original mutable array needs observed root and content-writer connection')


def _outputs(source_ref, initialization, receipts, field_witnesses, actuals, options, cases):
    ports = receipts['ports']
    demands = {tuple(row['controller_path']): row for row in initialization['ports']}
    ingress_by_name = {row['name']: row for row in receipts['ingress']}
    explicit = {tuple(row['controller_path']): row for row in initialization['ports']}
    additional = {tuple(row['path']): row for row in initialization['initial_receiver_fields']}
    outputs, routes, decisions = [], [], []
    for index, row in enumerate(ports):
        path = tuple(row['controller_path'])
        site = _at(source_ref, f'#/ports/{index}')
        if not all(path in witness and witness[path]['status'] == 'OBSERVED' for witness in field_witnesses):
            continue
        witnesses = [witness[path] for witness in field_witnesses]
        if path in explicit:
            demand = demands.pop(path)
            _require(demand['id'] == row['id'] and demand['status'] == 'ORIGINAL_FIELD_WRITES_CONNECTED',
                     'constructor slice differs from original controller port')
        else:
            _require(path[1:] in additional, 'auxiliary startup field has no original constructor write')
            demand = dict(additional[path[1:]], kind='array' if witnesses[0]['kind'] == 'array' else 'field')
        ingress = row['supplied_inputs']
        if ingress:
            parameter = _route_parameter(row, demand, actuals, ingress_by_name)
            option_name = actuals[parameter]
            option = options[option_name]
            values = [_scalar(witness) for witness in witnesses]
            _require(all(value['kind'] == option['kind'] and type(value['value']) is type(case[option_name])
                         and value['value'] == case[option_name] for value, case in zip(values, cases, strict=True)),
                     'direct runtime ingress and original scalar handoff disagree')
            routes.append(dict(id=row['id'], path=list(path), parameter=parameter, option=option_name,
                               kind=option['kind'], witness_values=values, source=site))
            decisions.append(dict(port=row['id'], action='RETAIN_TYPED_RUNTIME_OPTION', source=site))
        else:
            _guard_fixed_slice(row, demand)
            if demand['kind'] == 'array':
                geo, expected = _array_witness(witnesses)
                descriptor = dict(kind='ARRAY', geometry=geo, expected=expected)
            else:
                values = [_scalar(witness) for witness in witnesses]
                _require(all(value == values[0] for value in values),
                         'source-independent scalar witness changed')
                descriptor = dict(kind='SCALAR', expected=values[0])
            outputs.append(dict(id=row['id'], path=list(path), source=site, **descriptor))
            decisions.append(dict(port=row['id'], action='BUILD_EXECUTE_SOURCE_AND_VERIFY_EXACT_OUTPUT', source=site))
    _require(not demands and any(row['kind'] == 'ARRAY' for row in outputs),
             'source slice is missing required controller handoff ports')
    return outputs, routes, decisions


def _provider_requirements(source_ref, receipts, output_rows, modules, defaults, constructor_source):
    import numpy
    from numpy._core import _multiarray_umath

    output_ids = {row['id'] for row in output_rows}
    needed = set()
    for row in receipts['ports']:
        if row['id'] not in output_ids:
            continue
        _require(set(row['declared_defaults']) <= defaults.keys(),
                 'startup array dependency requires an unverified declaration')
        needed.update(row['provider_calls'])
    lib = _multiarray_umath.__file__
    _require(_reference(lib, 'scar.loaded-module') == modules['numpy._core._multiarray_umath']['artifact']
             and _reference(numpy.__file__, 'scar.loaded-module') == modules['numpy']['artifact']
             and numpy.__version__ == modules['numpy']['version'],
             'live standard numeric provider does not match original witness')
    syntax = ast.parse(read_artifact(constructor_source, json_document=False), filename=constructor_source['uri'])
    aliases = {member.asname or member.name: member.name for statement in syntax.body
               if type(statement) is ast.Import for member in statement.names}
    result = []
    for index, row in enumerate(receipts['providers']):
        if row['id'] not in needed:
            continue
        paths = {tuple(path) for path in row['namespace_candidates']}
        path = _only(paths.intersection(_PROVIDERS), 'source startup provider contract unavailable: ' + repr(paths))
        _require(tuple(row['declaration_candidate']) == path, 'provider does not name its original declaration')
        fallback = paths - {path}
        _require(all(len(row) == 3 and row[0] == 'builtins' and aliases.get(row[1]) == path[0]
                     and (row[2],) == path[1:] for row in fallback),
                 'ambiguous source startup provider namespace')
        operation, arity = _PROVIDERS[path]
        _require(len(row['positional']) == arity and not row['keywords'] and len(row['observations']) == 1
                 and row['observations'][0]['target']['kind'] in ('NATIVE_BUILTIN', 'OTHER_UNCLASSIFIED'),
                 'provider invocation options are outside the supported source route')
        provider = getattr(numpy if path[0] == 'numpy' else builtins, path[1])
        wrapped = inspect.unwrap(provider)
        source = None
        if inspect.isfunction(wrapped):
            source = _reference(inspect.getsourcefile(wrapped), 'scar.loaded-module')
        result.append(dict(id=row['id'], source=_at(source_ref, f'#/providers/{index}'),
                           namespace=list(path), operation=operation, arity=arity,
                           guarded_namespace_aliases=sorted(candidate[1] for candidate in fallback),
                           implementation_source=source,
                           implementation_code=code_key(wrapped.__code__) if source else None,
                           current_provider_shape=type(provider).__qualname__,
                           exact_build_outputs_required=True, source_options=dict(keywords=[], positional_count=arity)))
    _require({row['id'] for row in result} == needed, 'source contains unbound startup provider references')
    return result


def _invocation(first, actual, initializer, defaults, routes, modules):
    by_parameter = {item['parameter']: item for item in routes}
    arguments = []
    for argument in initializer['arguments'][1:]:
        name, role = argument['name'], argument['role']
        _require(argument['parameter_kind'] == 'KEYWORD_ONLY' and name in defaults,
                 'constructor build route must bind exact original keyword layout')
        if role == 'DECLARED_DEFAULT':
            _require(argument['declared_default'] is not None and argument['actual'] is None,
                     'constructor build value requires original default provenance')
            arguments.append(dict(parameter=name, kind='LITERAL', value=defaults[name], formal=argument['formal']))
            continue
        _require(role == 'KEYWORD' and name in by_parameter, 'original supplied keyword has no field route')
        route = by_parameter[name]
        values = route['witness_values']
        if route['kind'] == 'str':
            _require(all(Path(row['value']).is_absolute() for row in values),
                     'shadowed startup effect path requires an original absolute test destination')
            kind = 'ISOLATED_OUTPUT_DIRECTORY'
        else:
            kind = 'SCALAR_WITNESS'
        arguments.append(dict(parameter=name, kind=kind, value=values[0]['value'],
                              witness_values=values, runtime_path=route['path'], formal=argument['formal']))
    interpreter = first['interpreter']
    affinity = first['cpu_affinity']
    _require(affinity and len(set(affinity)) == len(affinity) and all(type(cpu) is int and cpu >= 0 for cpu in affinity),
             'original startup context requires explicit host execution resources')
    return dict(
        source=first['source_file'], module=actual['module'], class_qualname=actual['qualname'],
        constructor_code=actual['constructor_code']['key'], keyword_arguments=arguments,
        working_directory='ISOLATED_TEMP', interpreter=_reference(interpreter['executable'], 'scar.build-python'),
        interpreter_version=interpreter['version'], environment=first['environment'], cpu_affinity=affinity,
        required_modules=modules,
    )


def _derive(source_ref, comparison_ref, controller_ref):
    from scar.analysis.source_controller_startup_slice import check as check_source

    receipts = read_artifact(source_ref)
    controller = read_artifact(controller_ref)
    _require(receipts['schema_version'] == 1
             and receipts['controller'] == controller['source']
             and receipts['startup_materialization_qualified'] is False
             and receipts['provider_effects_closed'] is False
             and controller['replay']['runtime_effect_mode'] == 'ORIGINAL_SYNCHRONOUS_EFFECT_PROVIDER_REQUIRED'
             and controller['replay']['complete_entry_authority'] is False,
             'exact original startup source/controller dependency required')
    initialization_ref = receipts['initialization']
    initialization = read_artifact(initialization_ref)
    _require(initialization['controller'] == receipts['controller']
             and initialization['exact_startup_input_materialization_qualified'] is False,
             'expected original F4 constructor connection evidence')
    first, modules, field_witnesses = _witnesses(comparison_ref, initialization_ref, controller)
    actual, initializer, defaults = _class_signature(first, initialization)
    for argument in receipts['ingress']:
        if argument['role'] == 'DECLARED_DEFAULT':
            source_value = decode(PythonLiteral, argument['original_declaration_literal']).to_python()
            parsed_value = defaults[argument['name']]
            _require(type(source_value) is type(parsed_value) and source_value == parsed_value
                     and (type(source_value) is not float
                          or struct.pack('>d', source_value) == struct.pack('>d', parsed_value)),
                     'F2 declaration and exact constructor default disagree')
    options, actuals = _source_runtime_options(first['original_entry'], initializer)
    observation = read_artifact(comparison_ref)
    cases = [_parse_witness_arguments(first['original_entry'], options, args)
             for args in observation['original_case_arguments']]
    outputs, routes, decisions = _outputs(source_ref, initialization, receipts, field_witnesses,
                                         actuals, options, cases)
    providers = _provider_requirements(source_ref, receipts, outputs, modules, defaults, first['source_file'])
    required_array_count = sum(row['kind'] == 'ARRAY' for row in outputs)
    _require(required_array_count == first['one_shot_boundary']['distinct_arrays_written'],
             'source-selected array roots do not cover the witnessed distinct startup arrays')
    _require(receipts['selected_control']['every_future_execution_selects_this_profile'] is False,
             'original selected constructor control is a conditional proof boundary')
    invocation = _invocation(first, actual, initializer, defaults, routes, modules)
    array_outputs = [row for row in outputs if row['kind'] == 'ARRAY']
    demanded_array_ids = {row['id'] for row in initialization['ports'] if row['kind'] == 'array'}
    layout = dict(asset_bytes=sum(row['expected']['size_bytes'] for row in array_outputs),
                  copied_numeric_payload_bytes=sum(row['expected']['size_bytes'] for row in array_outputs
                                                   if row['id'] in demanded_array_ids),
                  runtime_array_resource='WRITABLE_CPU_ADDRESS_SPACE',
                  runtime_image_resource='IMMUTABLE_PACKAGE_ASSET',
                  build_cpu_affinity=invocation['cpu_affinity'],
                  physical_peak_bytes=None, physical_dram_residency=None)
    _require(check_source(source_ref['uri']) == receipts == read_artifact(source_ref),
             'original source startup graph differs during cold replay')
    cost = CostEstimate(None, None, None, 'UNKNOWN', 'ORIGINAL_CONSTRUCTOR_HANDOFF',
                        (_at(source_ref, '#/ports'), _at(comparison_ref, '#/observations')),
                        ('NO_MATCHED_CLEAN_STARTUP_EFFECT_COST',))
    return dict(
        schema=SCHEMA, schema_version=1, source_slice=source_ref, comparison=comparison_ref,
        controller_plan=controller_ref, controller_source=controller['source'],
        status='CONDITIONAL_EXPLICIT_TRIAL_ONLY', whole_entry_authorized=False, deployment_authorized=False,
        default_backend='NO_OP',
        materialization=dict(
            kind='BUILD_EXECUTE_PINNED_ORIGINAL_CONSTRUCTOR', invocation=invocation, outputs=outputs,
            provider_requirements=providers, source_option_parser=dict(source=first['original_entry'],
                    stdlib_provider=_reference(argparse.__file__, 'scar.startup-argparse-provider'),
                    arguments=options),
            source_control=_at(source_ref, '#/selected_control'), original_default_values=defaults,
            captured_arrays_are_producer=False, successful_build_evaluation_required=True,
            require_all_build_values_match_exact_original_hashes=True,
            distinct_array_roots_required=required_array_count,
        ),
        runtime_routes=routes, decisions=decisions, cost=encode(cost), target_program_executed=False,
        array_resources=layout,
        original_order=['BUILD_EVALUATION_AND_BYTE_ADMISSION', 'RUNTIME_ASSET_VALIDATION',
                        'RUNTIME_TYPED_HANDOFF_TO_CONTROLLER', 'ORIGINAL_CONTROLLER_EFFECT_ORDER'],
        performance_measured=False, observations_confer_invariance=False,
        frontiers=[
            'EXACT_BUILD_SCOPE_PINNED_SOURCE_PROVIDER_AND_SELECTED_DECLARATION_DEFAULTS',
            'SOURCE_IMPORTS_AND_STARTUP_WORLD_EFFECTS_PERSIST_OUTSIDE_TRIAL',
            'PLATFORM_AND_CALLABLE_INTERPOSITION_REQUIRE_PINNED_BUILD_PROVIDER_CONTEXT',
            'SHADOW_OUTPUT_PATH_IS_DISJOINT_FROM_RUNTIME_ORIGINAL_OUTPUT_EFFECTS',
            'ORIGINAL_CALLBACKS_RENDERER_AND_WHOLE_ENTRY_EFFECTS_OPEN',
        ],
    )


@replay_transaction()
def plan_controller_startup(source_slice, comparison, controller_plan, out):
    report = _derive(_reference(source_slice, SLICE), _reference(comparison, COMPARISON),
                     _reference(controller_plan, CONTROLLER))
    with Path(out).resolve().open('x', encoding='utf-8') as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return report


@replay_transaction()
def read_controller_startup_plan(path):
    reference = _reference(path, SCHEMA)
    report = read_artifact(reference)
    _require(report == _derive(report['source_slice'], report['comparison'], report['controller_plan']),
             'startup plan differs from original source and evidence replay')
    read_artifact(reference)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--source-slice')
    group.add_argument('--check')
    parser.add_argument('--comparison')
    parser.add_argument('--controller-plan')
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    if args.check:
        if args.comparison or args.controller_plan or args.out:
            parser.error('--check only consumes the retained startup plan')
        report = read_controller_startup_plan(args.check)
    else:
        if not args.comparison or not args.controller_plan or not args.out:
            parser.error('planning requires --comparison, --controller-plan and --out')
        report = plan_controller_startup(args.source_slice, args.comparison, args.controller_plan, args.out)
    print(json.dumps(dict(status=report['status'],
                          generated_outputs=len(report['materialization']['outputs']),
                          preserved_runtime_inputs=len(report['runtime_routes']),
                          default_backend=report['default_backend']), sort_keys=True))


if __name__ == '__main__':
    main()
