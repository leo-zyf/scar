"""Extract the affine P1 triangle ingress through a finite source contract.

Only retained observations select loaded objects. Scalar values come from source
expressions with fixed Python/NumPy binary64 semantics. This reader does not
load the target, run a callback, or read the native wire used for validation.
"""

import argparse
import ast
import json
import math
import operator
import re
from pathlib import Path
from types import CodeType

from scar.compiler.call_binding import plan_digest
from scar.ir.execution_recipe import _json, artifact_reference, read_artifact
from scar.trace.code_snapshot import call_instruction, describe_code, site

SCHEMA = 'scar.source-triangle-seed'
_CALLS = 'scar.bounded-entry-calls'
_ACTIVATION = 'scar.mesh-basis-activation-admission'
_SOURCE = 'scar.python-source'
_OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}


def _require(condition, reason):
    if not condition:
        raise ValueError('triangle seed: ' + reason)


def _one(values, role):
    values = list(values)
    _require(len(values) == 1, f'{role} requires exactly one source or observation, got {len(values)}')
    return values[0]


def _form(node, expression):
    return ast.dump(node) == ast.dump(ast.parse(expression, mode='eval').body)


def _statement(node, text):
    return ast.dump(node) == ast.dump(ast.parse(text).body[0])


def _body(node):
    body = node.body
    if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
            and type(body[0].value.value) is str):
        return body[1:]
    return body


def _class(tree, name):
    return _one((node for node in tree.body if isinstance(node, ast.ClassDef)
                 and node.name == name), name)


def _method(tree, name):
    return _one((node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == name), name)


def _attribute(tree, name):
    values = []
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            values.append(node.value)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            values.append(node.value)
    return _one(values, name)


def _provenance(source, node):
    return {'source': source['uri'], 'span': [node.lineno, node.end_lineno, node.col_offset, node.end_col_offset]}


def _source(path, sources, *, expected=None):
    ref = artifact_reference(path, _SOURCE)
    _require(expected is None or ref == expected, 'source differs from captured declaration: ' + str(path))
    _require(ref['size_bytes'] <= 1024 * 1024, 'source unit exceeds bounded extraction size')
    raw = read_artifact(ref, json_document=False)
    sources[ref['uri']] = ref
    return ref, ast.parse(raw, filename=ref['uri']), raw


def _inside(root, path):
    path = Path(path).resolve(strict=True)
    _require(path.is_relative_to(root) and path.suffix == '.py', 'provider source outside selected root')
    return path


def _binding(root, type_info):
    module, qualname = type_info.get('module'), type_info.get('qualname')
    _require(type(module) is str and all(part.isidentifier() for part in module.split('.'))
             and type(qualname) is str and qualname.isidentifier(), 'provider type identity is not a source class')
    return _inside(root, root.joinpath(*module.split('.')).with_suffix('.py')), qualname


def _nested_code(root, qualname):
    pending = [root]
    while pending:
        code = pending.pop()
        if code.co_qualname == qualname:
            return code
        pending.extend(value for value in code.co_consts if isinstance(value, CodeType))
    raise ValueError('triangle seed: no source bytecode for ' + qualname)


def _observation(calls, caller, call, *, require_reachable=True):
    caller_key = describe_code(caller)['key']
    position = [call.lineno, call.end_lineno, call.col_offset, call.end_col_offset]
    found = _one(((ordinal, item['observation'])
                  for ordinal, item in enumerate(calls['observed_targets'])
                  if (item['entry_scope_reachable'] or not require_reachable)
                  and item['observation']['caller_site']['code'] == caller_key
                  and item['observation']['caller_site']['position'] == position), 'source-to-call-site binding')
    reported = found[1]['caller_site']
    _require(call_instruction(caller, reported['offset']) is not None and reported == site(caller, reported['offset']),
             'observed call offset does not match recompiled source')
    return f'#/observed_targets/{found[0]}', found[1]


def _type(calls, caller, call):
    locator, observation = _observation(calls, caller, call)
    metadata = json.loads(observation['metadata_json'])
    _require(metadata['kind'] == 'TYPE_OBJECT' and observation['target_code'] is None,
             'constructor has no matching observed type object')
    return locator, metadata['constructed_type']


def _observed_code(descriptor, source, raw, activation=None):
    code = compile(raw, source['uri'], 'exec', dont_inherit=True, optimize=0)
    selected = _nested_code(code, descriptor['qualname'])
    _require(descriptor == describe_code(selected),
             'observed code differs from retained source')
    if activation is not None:
        proof = activation['body_correspondences'][descriptor['key']]
        _require(proof['code'] == descriptor and len(proof['matches']) == 1
                 and proof['matches'][0]['source_sha256'] == source['sha256'],
                 'observed body does not pin this source version')
    return selected


def _literal(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _literal(node.operand)
        _require(type(value) in (int, float), 'numeric sign requires a scalar')
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, (ast.List, ast.Tuple)):
        return tuple(map(_literal, node.elts))
    raise ValueError('triangle seed: expected a finite numeric literal or container')


def _array(node, required_dtype=None):
    _require(isinstance(node, ast.Call) and _form(node.func, 'np.array')
             and len(node.args) == 1, 'expected literal numpy array constructor')
    dtype = [] if required_dtype is None else [('dtype', 'np.' + required_dtype)]
    _require(len(node.keywords) == len(dtype) and all(
        keyword.arg == name and _form(keyword.value, value)
        for keyword, (name, value) in zip(node.keywords, dtype, strict=True)), 'array dtype differs')
    return _literal(node.args[0])


def _factory(node, dtype):
    _require(isinstance(node, ast.Call) and _form(node.func, 'field') and not node.args
             and len(node.keywords) == 1 and node.keywords[0].arg == 'default_factory',
             'expected exact dataclass default factory')
    value = node.keywords[0].value
    _require(isinstance(value, ast.Lambda) and not value.args.posonlyargs and not value.args.args
             and not value.args.kwonlyargs and value.args.vararg is None and value.args.kwarg is None
             and isinstance(value.body, ast.Attribute) and value.body.attr == 'T',
             'array factory requires a zero-argument literal transpose')
    return tuple(zip(*_array(value.body.value, dtype), strict=True))


def _import(tree, module, name, *, alias=None):
    if name is None:
        match = [node for node in tree.body if isinstance(node, ast.Import)
                 and any(item.name == module and item.asname == alias for item in node.names)]
    else:
        match = [node for node in tree.body if isinstance(node, ast.ImportFrom)
                 and '.' * node.level + (node.module or '') == module
                 and any(item.name == name and item.asname == alias for item in node.names)]
    return _one(match, f'lexical import {module} {name or ""}')


def _read_imported_class(root, owner, tree, name, sources):
    node = _one((node for node in tree.body if isinstance(node, ast.ImportFrom)
                 and node.level > 0 and any(item.name == name and item.asname is None for item in node.names)),
                'direct relative class import: ' + name)
    base = Path(owner['uri']).parent
    for _ in range(node.level - 1):
        base = base.parent
    ref, imported, raw = _source(_inside(root, base.joinpath(*(node.module or '').split('.')).with_suffix('.py')),
                                 sources)
    return ref, imported, _class(imported, name), raw


def _expr(node, env):
    if isinstance(node, ast.Name) and node.id in env:
        return env[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left, right = _expr(node.left, env), _expr(node.right, env)
        if isinstance(left, tuple) or isinstance(right, tuple):
            lanes = len(left) if isinstance(left, tuple) else len(right)
            left = left if isinstance(left, tuple) else (left,) * lanes
            right = right if isinstance(right, tuple) else (right,) * lanes
            return tuple(_OPERATORS[type(node.op)](float(a), float(b)) for a, b in zip(left, right, strict=True))
        return _OPERATORS[type(node.op)](float(left), float(right))
    if isinstance(node, ast.Call) and _form(node.func, 'np.array') and len(node.args) == 1 and not node.keywords:
        _require(isinstance(node.args[0], (ast.List, ast.Tuple)), 'basis array requires explicit lanes')
        return tuple(_expr(value, env) for value in node.args[0].elts)
    return _literal(node)


def _local_basis(function, points, functions):
    unpack, first, returned = _body(function)
    _require(_statement(unpack, 'x, y = X') and _statement(returned, 'return phi, dphi'),
             'expected a local reference basis with explicit indexed clauses')
    branch = first
    values, derivatives = [], []
    for index in range(functions):
        _require(isinstance(branch, ast.If) and _form(branch.test, f'i == {index}') and len(branch.body) == 2,
                 'unsupported reference index selection')
        assignments = branch.body
        _require(all(isinstance(node, ast.Assign) and len(node.targets) == 1
                     and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name
                     for node, name in zip(assignments, ('phi', 'dphi'), strict=True)),
                 'reference clause must define the value and derivative once')
        phi, dphi = [_expr(node.value, dict(zip(('x', 'y'), points, strict=True))) for node in assignments]
        values.append(phi)
        derivatives.append(dphi)
        branch = _one(branch.orelse, 'reference index continuation')
    _require(_statement(branch, 'self._index_error()'), 'unsupported reference index fallback')
    return tuple(values), tuple(derivatives)


def _field(data, shape, dtype):
    def flatten(node, dimensions):
        if dimensions:
            _require(type(node) is tuple and len(node) == dimensions[0], 'primitive array shape differs')
            return [value for child in node for value in flatten(child, dimensions[1:])]
        if dtype == '<i4':
            _require(type(node) is int and -2**31 <= node < 2**31, 'signed int32 literal required')
            return [node]
        _require(type(node) is float and math.isfinite(node), 'finite binary64 source required')
        return [node.hex()]
    return {'dtype': dtype, 'shape': list(shape), 'values': flatten(data, shape)}


def _top_level_calls(tree):
    rows = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            rows.append((node.targets[0].id, node.value))
    mesh_name, refine = _one(((name, value) for name, value in rows
                             if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                             and value.func.attr == 'refined' and isinstance(value.func.value, ast.Call)),
                            'entry mesh refinement')
    ctor = refine.func.value
    _require(not ctor.args and not ctor.keywords and len(refine.args) == 1 and not refine.keywords,
             'expected the default mesh followed by a literal refinement depth')
    element_name, elem = _one(((name, value) for name, value in rows
                              if isinstance(value, ast.Call) and len(value.args) == 0 and not value.keywords
                              and any(isinstance(candidate, ast.Call) and len(candidate.args) == 2
                                      and _form(candidate.args[0], mesh_name) and _form(candidate.args[1], name)
                                      for _, candidate in rows)), 'element feeding mesh basis')
    basis = _one((value for _, value in rows if isinstance(value, ast.Call) and len(value.args) == 2
                  and not value.keywords and _form(value.args[0], mesh_name)
                  and _form(value.args[1], element_name)), 'basis constructor')
    depth = _literal(refine.args[0])
    _require(type(depth) is int and 0 < depth <= 15, 'refinement depth must be exact positive bounded integer')
    return ctor, refine, depth, elem, basis


def _iteration(tree, calls, code, argv):
    _require(type(argv) is list and len(argv) == 1 and type(argv[0]) is str
             and re.fullmatch('[1-9][0-9]{0,8}', argv[0]) is not None, 'expected one canonical positive integer argv')
    main = _one((node for node in tree.body if isinstance(node, ast.If)
                 and _form(node.test, '__name__ == "__main__"')), 'entry main guard')
    assignment = _one((node for node in main.body if isinstance(node, ast.Assign)
                       and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                       and _form(node.value, 'int(sys.argv[1])')), 'repetition count defined by argv')
    _import(tree, 'sys', None)
    name = assignment.targets[0].id
    _require(len(main.body) == 3 and main.body[0] is assignment and isinstance(main.body[1], ast.If)
             and _form(main.body[1].test, f'{name} <= 0') and not main.body[1].orelse
             and len(main.body[1].body) == 1 and isinstance(main.body[1].body[0], ast.Raise)
             and isinstance(main.body[2], ast.Expr) and isinstance(main.body[2].value, ast.Call),
             'main guard must pass the positive argv-derived count directly')
    print_call = main.body[2].value
    _require(_form(print_call.func, 'print') and len(print_call.args) == 1 and not print_call.keywords
             and isinstance(print_call.args[0], ast.Call), 'expected a measured repetition region')
    loop_call = print_call.args[0]
    _require(isinstance(loop_call.func, ast.Name) and loop_call.args
             and _form(loop_call.args[-1], name) and not loop_call.keywords,
             'iteration count must feed the observed region directly')
    int_site, int_type = _type(calls, code, assignment.value)
    loop_site, loop_obs = _observation(calls, code, loop_call)
    function = _method(tree, loop_call.func.id)
    loop_code = _nested_code(code, function.name)
    _require(int_type == {'module': 'builtins', 'qualname': 'int'}
             and loop_obs['target_code'] == describe_code(loop_code),
             'iteration count requires exact builtin int and a selected source region')
    formals = function.args
    _require(len(formals.args) == len(loop_call.args) and not formals.posonlyargs and not formals.kwonlyargs
             and formals.vararg is None and formals.kwarg is None, 'iteration count does not match the loop ABI')
    count = formals.args[-1].arg
    loop = _one((node for node in function.body if isinstance(node, ast.For)
                 and _form(node.iter, f'range({count})')), 'repetition loop')
    _, range_type = _type(calls, loop_code, loop.iter)
    _require(range_type == {'module': 'builtins', 'qualname': 'range'}, 'loop bound requires exact builtin range')
    return int(argv[0]), assignment, int_site, loop_site, count


def _quadrature(owner, dispatch, refdom_name, order):
    _import(owner, 'numpy', None, alias='np')
    refdom_gate, triangle_gate, *_ = _body(dispatch)
    _require(_statement(refdom_gate, """if hasattr(refdom_or_elem, 'refdom'):
    refdom = refdom_or_elem.refdom
else:
    refdom = refdom_or_elem""")
             and isinstance(triangle_gate, ast.If)
             and _form(triangle_gate.test, f'refdom == {refdom_name}')
             and len(triangle_gate.body) == 1
             and _statement(triangle_gate.body[0], 'return get_quadrature_tri(norder)'),
             'unsupported reference-domain dispatch')
    triangle = _method(owner, 'get_quadrature_tri')
    guard, lookup = _body(triangle)
    _require(_statement(guard, 'if norder <= 1: norder = 2')
             and isinstance(lookup, ast.Try) and len(lookup.body) == 1
             and isinstance(lookup.body[0], ast.Return), 'unsupported triangle quadrature table dispatch')
    order = max(2, order)
    selected = lookup.body[0].value
    _require(isinstance(selected, ast.Subscript) and _form(selected.slice, 'norder')
             and isinstance(selected.value, ast.Dict), 'triangle quadrature requires a literal integer-key table')
    rule = _one((value for key, value in zip(selected.value.keys, selected.value.values, strict=True)
                 if isinstance(key, ast.Constant) and type(key.value) is int and key.value == order), 'quadrature rule')
    _require(all(isinstance(key, ast.Constant) and type(key.value) is int for key in selected.value.keys)
             and isinstance(rule, ast.Tuple) and len(rule.elts) == 2, 'unsupported quadrature key or value expression')
    return _array(rule.elts[0]), _array(rule.elts[1]), rule, order


def _record(document, locator):
    _require(type(locator) is str and re.fullmatch(r'#/records/[0-9]+', locator) is not None,
             'invalid observed record locator')
    return document['records'][int(locator.rsplit('/', 1)[1])]


def _same_attempt(locator, calls, activation, raw):
    observation = calls['observed_targets'][int(locator.rsplit('/', 1)[1])]['observation']
    evidence = observation['evidence']
    row = _record(raw, evidence['locator'])
    _require(evidence['artifact'] == activation['bounded_call_observations']
             and observation['run'] == evidence['artifact']['sha256']
             and observation['process_id'] == raw['process_id']
             and observation['attempt_index'] == row['index']
             and observation['caller_site']['code'] == row['caller_code_key']
             and observation['caller_site']['offset'] == row['offset']
             and observation['thread_id'] == row['thread_id']
             and observation['target_code'] == row['target']['code']
             and json.loads(observation['metadata_json']) == row['target'],
             'projected call does not match the retained capture')
    return row


def _observed_loop_count(frames, attempt, iterations, formal):
    candidates = [(index, row) for index, row in enumerate(frames['records'])
                  if 'selected_count' in row['fields']]
    if not candidates:
        return None
    index, frame = _one(candidates, 'captured loop-count operand')
    count = frame['fields']['selected_count']
    _require(frame['process_id'] == frames['process_id'] and frame['thread_id'] == attempt['thread_id']
             and frame['raw_call_attempt'] == attempt['index'] and frame['caller_code'] == attempt['caller_code_key']
             and frame['caller_offset'] == attempt['offset'] and frame['code'] == attempt['target']['code']
             and frame['phase'] in ('ENTERED', 'NORMAL_RETURN')
             and type(count['value']) is int
             and count == {'formal': formal, 'type': ['builtins', 'int'], 'value': iterations},
             'captured loop-count operand does not match its original call and launch argv')
    return f'#/records/{index}/fields/selected_count'


def _mesh_post_init(owner, mesh_tree, mesh_symbol, nodes):
    post = _method(owner, '__post_init__')
    sorter, points, cells, count, higher, point_layout, cell_layout, boundaries, subdomains, validation = _body(post)
    _import(mesh_tree, 'numpy', None, alias='np')
    _require(_statement(sorter, 'if self.sort_t: self.t = np.sort(self.t, axis=0)')
             and _statement(points, 'self.doflocs = np.asarray(self.doflocs, dtype=np.float64, order="K")')
             and _statement(cells, 'self.t = np.asarray(self.t, dtype=np.int32, order="K")')
             and _statement(count, 'M = self.elem.refdom.nnodes')
             and _statement(_one(_body(_method(owner, 'nnodes')), 'mesh arity'), 'return self.t.shape[0]')
             and _statement(_one(_body(_method(owner, 'refdom')), 'mesh domain'), 'return self.elem.refdom')
             and isinstance(higher, ast.If) and _form(higher.test, 'self.nnodes > M and self.elem is not Element')
             and not higher.orelse and nodes == 3, 'unsupported mesh construction normalization')
    for target, layout in (('doflocs', point_layout), ('t', cell_layout)):
        _require(isinstance(layout, ast.If) and _form(layout.test, f'not self.{target}.flags["C_CONTIGUOUS"]')
                 and not layout.orelse and len(layout.body) == 2
                 and _statement(layout.body[1], f'self.{target} = np.ascontiguousarray(self.{target})'),
                 'mesh layout conversion may change logical values')
    _require(_statement(boundaries, 'object.__setattr__(self, "_boundaries", Boundaries(self.boundaries or {}))')
             and _statement(subdomains, 'object.__setattr__(self, "_subdomains", Subdomains(self.subdomains or {}))')
             and _statement(validation, 'if self.validate and logger.getEffectiveLevel() <= logging.DEBUG: self.is_valid()')
             and all(_form(_attribute(owner, name), 'None') for name in ('_boundaries', '_subdomains', '_dist'))
             and all(_form(_attribute(mesh_symbol, name), 'True') for name in ('affine', 'sort_t')),
             'seed must have the affine serial default constructor contract')
    return post


def _mesh_hierarchy(root, source, tree, symbol, base_source, base_tree, base, sources):
    _require(len(symbol.bases) == 2 and all(isinstance(node, ast.Name) for node in symbol.bases)
             and not symbol.keywords and len(symbol.decorator_list) == 1
             and _form(symbol.decorator_list[0], 'dataclass(repr=False)'),
             'expected a standard affine mesh dataclass')
    _import(tree, 'dataclasses', 'dataclass')
    _import(tree, 'dataclasses', 'field')
    _import(tree, 'numpy', None, alias='np')
    simple_ref, _, simple, _ = _read_imported_class(root, source, tree, symbol.bases[0].id, sources)
    dim_ref, dim_tree, dimension, _ = _read_imported_class(root, source, tree, symbol.bases[1].id, sources)
    _require(not simple.bases and not simple.decorator_list
             and len(dimension.bases) == 1 and _form(dimension.bases[0], base.name)
             and len(dimension.decorator_list) == 1 and _form(dimension.decorator_list[0], 'dataclass(repr=False)')
             and len(base.decorator_list) == 1 and _form(base.decorator_list[0], 'dataclass(repr=False)'),
             'mesh inheritance does not select the base dataclass')
    _import(dim_tree, 'dataclasses', 'dataclass')
    _import(base_tree, 'dataclasses', 'dataclass')
    candidate_ref, _, _, _ = _read_imported_class(root, dim_ref, dim_tree, base.name, sources)
    _require(candidate_ref == base_source, 'selected refinement and inherited normalization use different base files')
    overridden = {'__init__', '__new__', '__post_init__', '__getattribute__', '__setattr__', 'nnodes',
                  'boundaries', 'subdomains', 'refdom'}
    for cls in (symbol, simple, dimension):
        _require(all(not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name not in overridden
                     for node in cls.body), 'mesh overload changes the selected initialization protocol')
    return [_provenance(simple_ref, simple), _provenance(dim_ref, dimension)]


def _basis_order(basis_tree, basis_symbol, basis_init, base_name, basis_call):
    _require(len(basis_symbol.bases) == 1 and _form(basis_symbol.bases[0], base_name)
             and not basis_symbol.keywords and not basis_symbol.decorator_list,
             'unsupported basis dispatch hierarchy')
    arguments = basis_init.args
    defaults = dict(zip((arg.arg for arg in arguments.args[-len(arguments.defaults):]),
                        arguments.defaults, strict=True))
    _require(all(_form(defaults[name], 'None') for name in ('mapping', 'intorder', 'elements', 'quadrature', 'dofs')),
             'basis defaults do not select the complete default quadrature')
    _require(len(basis_call.args) == 2 and not basis_call.keywords,
             'basis contract does not accept overrides')
    return _one((node.value for node in basis_init.body if isinstance(node, ast.Expr)
                 and _form(node.value, f'super({basis_symbol.name}, self).__init__('
                           'mesh, elem, mapping, intorder, quadrature, mesh.refdom, dofs, disable_doflocs)')),
                'argument route into inherited basis')


def _call_projection(path):
    artifact = artifact_reference(path, _CALLS)
    calls = _json(read_artifact(artifact, json_document=False))
    if type(calls) is dict and calls.get('schema') == _CALLS:
        return artifact, calls
    from scar.analysis import rebind_bounded_entry_calls

    _require(type(calls) is dict and calls.get('schema') == rebind_bounded_entry_calls.SCHEMA,
             'requires an original or rebound bounded CALL projection')
    artifact['schema'] = rebind_bounded_entry_calls.SCHEMA
    return artifact, rebind_bounded_entry_calls.check(artifact)


def extract(entry_calls, activations, source_root, argv):
    """Build a detached, machine-readable source seed and its exact evidence."""
    calls_ref, calls = _call_projection(entry_calls)
    activation_ref = artifact_reference(activations, _ACTIVATION)
    activation = read_artifact(activation_ref)
    frame_ref = activation['frame_observations']
    raw_ref = activation['bounded_call_observations']
    frames, raw = read_artifact(frame_ref), read_artifact(raw_ref)
    design = read_artifact(frames['design'])
    original = read_artifact(activation['original_run'])
    _require(calls['schema_version'] == 1 and activation['schema_version'] == 1
             and frames['process_id'] == raw['process_id'] and not frames['dropped_due_to_cap']
             and plan_digest(read_artifact(frames['call_plan'])) == raw['plan_sha256']
             and original['child_returncode'] == 0 and original['full_run_completed']
             and original['transformations_disabled'] and raw_ref['uri'] in original['raw_observations']
             and frames['launch']['argv'] == design['argv'] == argv
             and frames['launch']['entry_path'] == design['entry']['uri'], 'not one successful original capture')

    root = Path(source_root).resolve(strict=True)
    sources = {}
    entry, tree, entry_raw = _source(design['entry']['uri'], sources, expected=design['entry'])
    caller = compile(entry_raw, entry['uri'], 'exec', dont_inherit=True, optimize=0)
    mesh_call, refine_call, depth, element_call, basis_call = _top_level_calls(tree)
    mesh_site, mesh_type = _type(calls, caller, mesh_call)
    elem_site, elem_type = _type(calls, caller, element_call)
    basis_site, basis_type = _type(calls, caller, basis_call)
    refine_site, refine_obs = _observation(calls, caller, refine_call)
    iterations, count_assignment, int_site, loop_site, count_formal = _iteration(tree, calls, caller, argv)
    selectors = {'mesh_type': mesh_site, 'element_type': elem_site, 'basis_type': basis_site,
                 'refinement': refine_site, 'int_constructor': int_site, 'loop_consumer': loop_site}

    mesh_path, mesh_name = _binding(root, mesh_type)
    mesh_ref, mesh_tree, mesh_raw = _source(mesh_path, sources)
    mesh = _class(mesh_tree, mesh_name)
    element_path, element_name = _binding(root, elem_type)
    elem_ref, elem_tree, _ = _source(element_path, sources)
    element = _class(elem_tree, element_name)
    basis_path, basis_name = _binding(root, basis_type)
    basis_ref, basis_tree, basis_raw = _source(basis_path, sources)
    basis = _class(basis_tree, basis_name)
    basis_ctor = _method(basis, '__init__')

    refine_code = refine_obs['target_code']
    _require(refine_code['qualname'].count('.') == 1 and refine_code['qualname'].endswith('.refined'),
             'refinement must select the compiled instance method')
    mesh_base_ref, mesh_base_tree, mesh_base_raw = _source(_inside(root, refine_code['filename']), sources)
    refine_python = _observed_code(refine_code, mesh_base_ref, mesh_base_raw, activation)
    mesh_base = _class(mesh_base_tree, refine_code['qualname'].split('.')[0])
    mesh_hierarchy = _mesh_hierarchy(root, mesh_ref, mesh_tree, mesh, mesh_base_ref, mesh_base_tree, mesh_base, sources)

    refined = _method(mesh_base, 'refined')
    uniform_gate = _one((node for node in refined.body if isinstance(node, ast.If)
                         and _form(node.test, 'isinstance(times_or_ix, (int, np.integer)) and '
                                              'not isinstance(times_or_ix, (bool, np.bool_))')),
                        'integer refinement branch')
    uniform_loop = _one((node for node in uniform_gate.body if isinstance(node, ast.For)
                         and _form(node.iter, 'range(times_or_ix)')), 'uniform depth loop')
    uniform_call = _one((node.value for node in uniform_loop.body if isinstance(node, ast.Assign)
                         and _form(node.value, 'm._uniform()')), 'uniform loop producer')
    uniform_site, uniform_obs = _observation(calls, refine_python, uniform_call)
    uniform_code = uniform_obs['target_code']
    _require(uniform_code['qualname'] == mesh.name + '._uniform', 'observed uniform method differs from mesh class')
    _observed_code(uniform_code, mesh_ref, mesh_raw, activation)
    _require(sum(row['code'] == uniform_code and row['phase'] == 'NORMAL_RETURN'
                 for row in frames['records']) == depth, 'uniform loop count differs from retained original')
    selectors['uniform_refinement'] = uniform_site

    _require(_form(_attribute(mesh, 'elem'), element.name), 'mesh default element differs from selected element')
    _import(elem_tree, 'numpy', None, alias='np')
    domain_expr = _attribute(element, 'refdom')
    _require(isinstance(domain_expr, ast.Name), 'reference domain must be an imported class')
    domain_ref, _, domain, _ = _read_imported_class(root, elem_ref, elem_tree, domain_expr.id, sources)
    nnodes = _literal(_attribute(domain, 'nnodes'))
    facets = _literal(_attribute(domain, 'facets'))
    _require(type(nnodes) is int and nnodes == 3 and facets == ((0, 1), (1, 2), (0, 2))
             and _form(_attribute(element, 'nodal_dofs'), '1'), 'triangle ABI requires affine nodal P1 numbering')
    normalizer = _mesh_post_init(mesh_base, mesh_base_tree, mesh, nnodes)
    coord_factory, topology_factory = _attribute(mesh, 'doflocs'), _attribute(mesh, 't')
    coordinates = _factory(coord_factory, 'float64')
    connectivity = tuple(zip(*(sorted(cell) for cell in zip(*_factory(topology_factory, 'int32'), strict=True)),
                             strict=True))
    _require(len(coordinates) == 2 and len(connectivity) == nnodes and 0 < len(coordinates[0]) <= 65536
             and 0 < len(connectivity[0]) <= 65536
             and all(type(item) is int and 0 <= item < len(coordinates[0]) for row in connectivity for item in row),
             'invalid seed coordinates or connectivity extent')

    basis_attempt = _same_attempt(basis_site, calls, activation, raw)
    constructor = activation['constructor']
    records = {name: _record(frames, constructor[name]) for name in ('frame', 'base_init')}
    ctor_frame, base_frame = records['frame'], records['base_init']
    _require(constructor['original_code'] == ctor_frame['code']['key']
             and constructor['raw_attempt'] == calls['observed_targets'][int(basis_site.rsplit('/', 1)[1])]
                                                     ['observation']['evidence']['locator']
             and ctor_frame['raw_call_attempt'] == basis_attempt['index']
             and ctor_frame['caller_code'] == describe_code(caller)['key']
             and ctor_frame['caller_offset'] == basis_attempt['offset']
             and base_frame['caller_code'] == ctor_frame['code']['key']
             and ctor_frame['thread_id'] == base_frame['thread_id'] == basis_attempt['thread_id']
             and ctor_frame['process_id'] == base_frame['process_id'] == raw['process_id']
             and ctor_frame['phase'] == base_frame['phase'] == 'NORMAL_RETURN'
             and ctor_frame['fields']['mesh']['type'] == [mesh_type['module'], mesh_name]
             and ctor_frame['fields']['self']['type'] == [basis_type['module'], basis_name]
             and ctor_frame['fields']['self']['object_token'] == base_frame['fields']['self']['object_token'],
             'basis frame does not join the selected caller, mesh and element')
    _observed_code(ctor_frame['code'], basis_ref, basis_raw, activation)
    base_name, base_method = base_frame['code']['qualname'].split('.')
    base_ref, _, base_cls, base_raw = _read_imported_class(root, basis_ref, basis_tree, base_name, sources)
    _require(base_method == '__init__', 'expected the base initializer')
    base_python = _observed_code(base_frame['code'], base_ref, base_raw, activation)
    super_call = _basis_order(basis_tree, basis, basis_ctor, base_name, basis_call)
    base_init = _method(base_cls, '__init__')
    quad_gate = _one((node for node in base_init.body if isinstance(node, ast.If)
                      and _form(node.test, 'quadrature is not None')), 'optional quadrature gate')
    quad_assignment = _one(quad_gate.orelse, 'default quadrature assignment')
    _require(_statement(_one(quad_gate.body, 'explicit quadrature assignment'), 'self.X, self.W = quadrature')
             and _statement(quad_assignment, 'self.X, self.W = get_quadrature('
                            'refdom, intorder if intorder is not None else 2 * self.elem.maxdeg)'),
             'basis does not select the supported reference rule')
    quadrature_site, quadrature_obs = _observation(calls, base_python, quad_assignment.value,
                                                   require_reachable=False)
    quadrature_code = quadrature_obs['target_code']
    _require(quadrature_code['qualname'] == 'get_quadrature', 'basis dispatch target is not the reference rule')
    quadrature_ref, quad_tree, quad_raw = _source(_inside(root, quadrature_code['filename']), sources)
    _observed_code(quadrature_code, quadrature_ref, quad_raw)
    selected_domain, _, _, _ = _read_imported_class(root, quadrature_ref, quad_tree, domain.name, sources)
    _require(selected_domain == domain_ref, 'element and rule dispatch resolve different reference domains')
    selectors['quadrature_dispatch'] = quadrature_site

    max_degree = _literal(_attribute(element, 'maxdeg'))
    _require(type(max_degree) is int and max_degree == 1, 'basis must be a first-degree polynomial')
    integration_points, weights, rule, order = _quadrature(quad_tree, _method(quad_tree, 'get_quadrature'),
                                                           domain.name, 2 * max_degree)
    qpoints = len(weights)
    _require(0 < qpoints <= 256 and len(integration_points) == 2
             and all(len(lane) == qpoints for lane in integration_points)
             and all(type(value) is float and math.isfinite(value) for lane in integration_points for value in lane),
             'quadrature rule requires bounded two-component binary64 points')
    local_basis = _method(element, 'lbasis')
    values, derivatives = _local_basis(local_basis, integration_points, nnodes)
    attempts = {role: _same_attempt(locator, calls, activation, raw) for role, locator in selectors.items()}
    observed_count = _observed_loop_count(frames, attempts['loop_consumer'], iterations, count_formal)
    _require(ctor_frame['fields']['X']['shape'] == [2, qpoints] and ctor_frame['fields']['X']['dtype'] == '<f8'
             and ctor_frame['fields']['W']['shape'] == [qpoints] and ctor_frame['fields']['W']['dtype'] == '<f8'
             and ctor_frame['fields']['element_dofs']['shape'][0] == nnodes,
             'source-derived shape differs from retained original reference rule metadata')

    fields = {
        'seed_coordinates': _field(coordinates, (2, len(coordinates[0])), '<f8'),
        'seed_connectivity': _field(connectivity, (3, len(connectivity[0])), '<i4'),
        'reference_values': _field(values, (nnodes, qpoints), '<f8'),
        'reference_derivatives': _field(derivatives, (nnodes, 2, qpoints), '<f8'),
        'quadrature_weights': _field(weights, (qpoints,), '<f8'),
    }
    return {
        'schema': SCHEMA, 'schema_version': 1, 'target_executed': False,
        'scope': 'CONDITIONAL_AFFINE_NODAL_P1_TRIANGLE_SOURCE_VALUES',
        'entry': entry, 'entry_calls': calls_ref, 'activations': activation_ref, 'source_root': str(root),
        'argv': argv, 'iterations': iterations, 'depth': depth, 'fields': fields,
        'sources': sorted(sources.values(), key=lambda ref: ref['uri']),
        'provenance': {
            'observed_call_sites': selectors,
            'observed_basis_frames': {name: constructor[name] for name in records},
            **({'observed_loop_count': observed_count} if observed_count is not None else {}),
            'frames': frame_ref, 'raw_attempts': raw_ref, 'frame_design': frames['design'],
            'original_run': activation['original_run'],
            'source_spans': {
                'iteration_count': _provenance(entry, count_assignment),
                'refinement_call': _provenance(entry, refine_call),
                'uniform_loop': _provenance(mesh_base_ref, uniform_loop),
                'uniform_body': _provenance(mesh_ref, _method(mesh, '_uniform')),
                'seed_coordinates': _provenance(mesh_ref, coord_factory),
                'seed_connectivity': _provenance(mesh_ref, topology_factory),
                'seed_constructor': _provenance(mesh_base_ref, normalizer),
                'seed_inheritance': mesh_hierarchy,
                'basis_forwarding': _provenance(basis_ref, super_call),
                'basis_rule_selection': _provenance(base_ref, quad_assignment),
                'quadrature_rule': _provenance(quadrature_ref, rule),
                'reference_basis': _provenance(elem_ref, local_basis),
                'reference_domain': _provenance(domain_ref, domain),
            },
        },
        'topology_conditions': {
            'reference_arity': nnodes, 'reference_facet_order': [list(facet) for facet in facets],
            'quadrature_order': order, 'sort_connectivity_columns': True, 'affine': True,
            'basis_values_and_derivatives': 'PYTHON_BINARY64_EXPRESSION_TREE_SCALAR_LANES_NO_FMA',
            'numpy_array_representation': 'EXACT_FLOAT64_OR_INT32_AND_VALUE_PRESERVING_TRANSPOSE',
            'boundary_and_subdomain_tags': 'DEFAULT_EMPTY',
            'runtime_namespace': 'CONDITIONAL_LEXICAL_BINDINGS_NO_MONKEY_PATCH_OR_DYNAMIC_CLASS_OVERRIDE',
            'caller_effects': 'NOT_PROVEN',
        },
    }


def check(path):
    """Read and replay an existing admission, rejecting any changed operand or evidence."""
    stored = read_artifact(artifact_reference(path, SCHEMA))
    for name in ('entry_calls', 'activations'):
        read_artifact(stored[name])
    replayed = extract(stored['entry_calls']['uri'], stored['activations']['uri'], stored['source_root'], stored['argv'])
    _require(stored == replayed, 'stored source values, source scope, or provenance do not replay')
    return replayed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry-calls', type=Path)
    parser.add_argument('--activations', type=Path)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--argv', nargs='+')
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument('--out', type=Path)
    output.add_argument('--check', type=Path)
    args = parser.parse_args()
    if args.check:
        check(args.check)
        print(args.check)
        return
    if any(value is None for value in (args.entry_calls, args.activations, args.source_root, args.argv)):
        parser.error('extraction requires --entry-calls, --activations, --source-root, and --argv')
    result = extract(args.entry_calls, args.activations, args.source_root, args.argv)
    with args.out.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(args.out)


if __name__ == '__main__':
    main()
