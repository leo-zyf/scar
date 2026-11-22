"""Project a runtime count into a checked sparse source loop without executing it."""

import argparse
import ast
from collections import defaultdict, deque
import dis
from hashlib import sha256
import json
from pathlib import Path
import re
from struct import Struct

from scar.analysis import source_triangle_seed as seed
from scar.analysis.source_declared_catalog import read_source_catalog
from scar.analysis.source_references import ReferenceSlicer, source_port
from scar.compiler.graph_frontend import _metadata, _source_identity
from scar.ir.execution_recipe import _canonical, artifact_reference, read_artifact
from scar.ir.record_codec import encode
from scar.trace.code_snapshot import describe_code, site


SCHEMA = 'scar.source-sparse-count'
_UPSTREAM = 'scar.native-parameterized-upstream'
_PROFILE = 'scar.native-upstream-wire-profile'
_CSR = Struct('<8s32s4I')
_PLACED_LIMIT = 512
_SOURCE_CONSTANTS = ('entry', 'sources', 'source_root', 'scope', 'depth', 'fields', 'topology_conditions')
_WIRE_ROLES = ('seed', 'mesh', 'basis', 'coo', 'canonical_csr')
_HEADER_SIZES = (144, 120, 100, 76, 56)


def _require(condition, message):
    if not condition:
        raise ValueError('source sparse count: ' + message)


def _one(candidates, role):
    candidates = list(candidates)
    _require(len(candidates) == 1, 'expected exactly one ' + role)
    return candidates[0]


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _span(node):
    return [node.lineno, node.end_lineno, node.col_offset, node.end_col_offset]


def _name(node, context=ast.Load):
    _require(type(node) is ast.Name and type(node.ctx) is context, 'expected a direct lexical name')
    return node.id


def _literal(node, value):
    return type(node) is ast.Constant and type(node.value) is type(value) and node.value == value


def _bind(node, width=1):
    _require(type(node) is ast.Assign and len(node.targets) == 1, 'expected a single reaching definition')
    if width == 1:
        return _name(node.targets[0], ast.Store), node.value
    target = node.targets[0]
    _require(type(target) is ast.Tuple and len(target.elts) == width, 'expected exact tuple unpack')
    return tuple(_name(value, ast.Store) for value in target.elts), node.value


def _call(node, arity, keywords=0):
    _require(type(node) is ast.Call and len(node.args) == arity and len(node.keywords) == keywords
        and all(type(arg) is not ast.Starred for arg in node.args)
        and all(keyword.arg is not None for keyword in node.keywords), 'call ABI differs')
    return node


def _binary(node, operation):
    _require(type(node) is ast.BinOp and type(node.op) is operation, 'arithmetic or evaluation order differs')
    return node.left, node.right


def _import_binding(tree, node, *, module=None, member=None):
    name = _name(node)
    matches = []
    for statement in tree.body:
        if type(statement) is ast.Import and member is None:
            matches.extend(statement for alias in statement.names
                           if (alias.asname or alias.name.split('.')[0]) == name and alias.name == module)
        if type(statement) is ast.ImportFrom and statement.level == 0 and member is not None:
            matches.extend(statement for alias in statement.names if statement.module == module
                           and alias.name == member and (alias.asname or alias.name) == name)
    return _one(matches, 'lexical import of ' + module)


def _region(tree, assignment, caller):
    main = _one((row for row in tree.body if type(row) is ast.If and row.body[0] is assignment), 'main branch')
    output = main.body[-1].value
    invocation = output.args[0]
    fn = _one((row for row in tree.body if type(row) is ast.FunctionDef and row.name == _name(invocation.func)),
              'selected loop function')
    code = seed._nested_code(caller, fn.name)
    _require(len(fn.args.args) == 4 and not fn.args.defaults and not fn.decorator_list
        and not fn.args.kw_defaults and not main.orelse and len(seed._body(fn)) == 3, 'loop ABI differs')
    matrix, load, boundary, count = [formal.arg for formal in fn.args.args]
    init, loop, finish = seed._body(fn)
    digest, make_digest = _bind(init)
    make_digest = _call(make_digest, 0)
    digest_import = _import_binding(tree, make_digest.func, module='hashlib', member='sha256')
    argv = assignment.value.args[0]
    sys_import = _import_binding(tree, argv.value.value, module='sys')
    _require(type(loop) is ast.For and not loop.orelse and len(loop.body) == 4 and type(finish) is ast.Return,
             'expected an ordinary for loop with one serial digest commit')
    index = _name(loop.target, ast.Store)
    scale_stmt, prepare_stmt, solve_stmt, appended = loop.body
    scale, expression = _bind(scale_stmt)
    one, divided = _binary(expression, ast.Add)
    numerator, denominator = _binary(divided, ast.Div)
    i, i_step = _binary(numerator, ast.Add)
    n, n_step = _binary(denominator, ast.Add)
    _require(_literal(one, 1.0) and _literal(i_step, 1) and _literal(n_step, 1)
        and (_name(i), _name(n)) == (index, count), 'scale must retain both index and count exactly')
    (edited, rhs), prepare = _bind(prepare_stmt, 2)
    prepare = _call(prepare, 2, 2)
    load_read, factor = _binary(prepare.args[1], ast.Mult)
    _require((_name(prepare.args[0]), _name(load_read), _name(factor)) == (matrix, load, scale)
        and {kw.arg: _name(kw.value) for kw in prepare.keywords} == {'D': boundary, 'diag': scale},
        'prepared matrix and load ports differ')
    solution, solve = _bind(solve_stmt)
    solve = _call(solve, 2)
    _require(tuple(_name(value) for value in solve.args) == (edited, rhs), 'solver input lineage differs')
    _require(type(appended) is ast.Expr, 'expected a digest update expression')
    append, digest_result = _call(appended.value, 1), _call(finish.value, 0)
    for call, member in ((append, 'update'), (digest_result, 'hexdigest')):
        _require(type(call.func) is ast.Attribute and (_name(call.func.value), call.func.attr) == (digest, member),
                 'digest receiver or operation differs')
    _require(_name(append.args[0]) == solution and len({matrix, load, boundary, count, digest, index, scale,
        edited, rhs, solution}) == 10, 'loop definitions must not overwrite a transported value')
    _name(prepare.func)
    _name(solve.func)
    return dict(main=main, code=code, function=fn, count=count, index=index, digest=digest,
        assignment=assignment, argv=argv, int_call=assignment.value, invocation=invocation, print_call=output,
        loop=loop, range_call=loop.iter, scale=expression, scale_store=scale_stmt.targets[0],
        prepare=prepare, solve=solve, solution_store=solve_stmt.targets[0], append=append,
        make_digest=make_digest, digest_result=digest_result, sys_import=sys_import, digest_import=digest_import)


def _geometry(tree, region):
    """Close lexical input dependencies before the selected count is assigned."""
    definitions = defaultdict(list)
    for node in tree.body[:tree.body.index(region['main'])]:
        if type(node) is ast.Assign and len(node.targets) == 1 and type(node.targets[0]) is ast.Name:
            definitions[node.targets[0].id].append(node.value)
    count_name = _name(region['assignment'].targets[0], ast.Store)

    def follow(expression, path):
        for child in ast.walk(expression):
            if type(child) is ast.Name and type(child.ctx) is ast.Load:
                _require(child.id != count_name, 'an ingress value depends on the count binding')
                if child.id in definitions:
                    value = _one(definitions[child.id], 'reaching source definition of ' + child.id)
                    _require(child.id not in path and value.lineno <= expression.lineno,
                             'ordered upstream source definitions must be acyclic')
                    yield child.id, value
                    yield from follow(value, {*path, child.id})
            if type(child) is ast.Attribute:
                _require(not (child.attr == 'argv' and type(child.value) is ast.Name and child.value.id == 'sys'),
                         'an ingress value reads argv')

    result, upstream_calls = {}, {}
    for ordinal, (role, actual) in enumerate(zip(('matrix', 'load', 'boundary'), region['invocation'].args[:-1], strict=True)):
        definitions_used = dict.fromkeys(follow(actual, set()))
        selected = next((value for _, value in definitions_used if type(value) is ast.Call), None) if type(actual) is ast.Name else actual
        upstream_calls[role] = _call(selected, 0 if role == 'boundary' else 1)
        result[role] = dict(actual=_span(actual), argument_ordinal=ordinal,
            reaching_definitions=[dict(binding=name, span=_span(value)) for name, value in definitions_used])
    mesh, refinement, depth, element, basis = seed._top_level_calls(tree)
    _require(all(any(value is basis for _, value in follow(region['invocation'].args[index], set()))
        for index in (0, 1)) and any(value is refinement for _, value in follow(region['invocation'].args[2], set())),
        'the original matrix/load must consume the same basis and boundary must consume its mesh')
    return result, {**upstream_calls, 'mesh': mesh, 'refinement': refinement, 'element': element, 'basis': basis}, depth


def _instruction(code, node, names):
    chosen = _one((row for row in dis.get_instructions(code) if row.opname in names and row.positions is not None
        and [row.positions.lineno, row.positions.end_lineno, row.positions.col_offset, row.positions.end_col_offset] == _span(node)),
        'compiled site for ' + '/'.join(names))
    return site(code, chosen.offset)


def _graph(catalog_ref, calls, region, source):
    read_artifact(catalog_ref)
    catalog = read_source_catalog(Path(catalog_ref['uri']).parent)
    unit = calls['entry_scope']['unit']
    facts = catalog.fact_store.load(unit)
    _, syntax, _, _, digest = _source_identity(catalog, unit)
    _require(digest == source['sha256'], 'source facts differ from compiled source')
    slicer = ReferenceSlicer(catalog)
    spans = {}

    def span(port):
        occurrence = source_port(facts, port).port.template.occurrence
        if occurrence not in spans:
            fields = _metadata(syntax, occurrence)[1]
            spans[occurrence] = [fields[key] for key in ('line', 'end_line', 'column', 'end_column')]
        return spans[occurrence]

    def locate(table, node, port, predicate=lambda row: True):
        return _one(((index, row) for index, row in enumerate(getattr(facts, table))
                     if predicate(row) and span(port(row)) == _span(node)), 'source graph ' + table + ' port')

    def ref(table, index):
        return encode(slicer.fact(facts, table, index))

    def reference(port):
        return dict(port=encode(source_port(facts, port)), fact=ref('ports', port))

    def match_name(node, write, kind):
        candidates = {(row.namespace, row.value): (index, row) for index, row in enumerate(facts.names)
            if row.write == write and row.namespace.kind.name == kind and row.namespace.name == node.id
            and span(row.value) == _span(node)}
        return _one(candidates.values(), 'graph reaching name ' + node.id)

    def path(start, end):
        work, visited = deque([(start, ())]), {start}
        while work:
            current, previous = work.popleft()
            if current == end:
                return dict(source=reference(start), target=reference(end), aliases=[ref('aliases', i) for i in previous])
            for following, index in aliases.get(current, ()):
                if following not in visited:
                    work.append((following, (*previous, index)))
                    visited.add(following)
        raise ValueError('source sparse count: missing exact F4 port lineage')

    aliases = defaultdict(list)
    for index, (a, b) in enumerate(facts.aliases):
        aliases[a].append((b, index))
    nodes = {role: locate('calls', region[role], lambda row: row.result) for role in (
        'int_call', 'invocation', 'range_call', 'make_digest', 'prepare', 'solve', 'append', 'digest_result', 'print_call')}
    fn_i, fn = _one(((i, row) for i, row in enumerate(facts.functions)
        if row.parameter_names == tuple(arg.arg for arg in region['function'].args.args)
        and span(row.value) == _span(region['function'])), 'F4 loop function')
    argv_i, argv = locate('items', region['argv'], lambda row: row.value, lambda row: not row.write)
    count_i, count = match_name(region['assignment'].targets[0], True, 'MODULE')
    actual_i, actual = match_name(region['invocation'].args[-1], False, 'MODULE')
    range_i, range_count = match_name(region['range_call'].args[0], False, 'LEXICAL')
    create_i, create = locate('iterations', region['loop'], lambda row: row.value, lambda row: not row.advance)
    advance_i, advance = locate('iterations', region['loop'], lambda row: row.value, lambda row: row.advance)
    index_i, index = match_name(region['loop'].target, True, 'LEXICAL')
    result_i, result = match_name(region['solution_store'], True, 'LEXICAL')
    update_i, update = match_name(region['append'].args[0], False, 'LEXICAL')
    call = {role: row for role, (_, row) in nodes.items()}
    return dict(unit=unit, function=ref('functions', fn_i), call_facts={role: ref('calls', ordinal)
        for role, (ordinal, _) in nodes.items()}, formal_port=reference(fn.formals[len(fn.parameter_names)]),
        entry_actual_port=reference(call['invocation'].positional[-1]),
        entry_call_port=reference(call['invocation'].result),
        sources=dict(argv_item=ref('items', argv_i), module_count_definition=ref('names', count_i),
            module_count_read=ref('names', actual_i), formal_count_read=ref('names', range_i),
            iterator_create=ref('iterations', create_i), iterator_advance=ref('iterations', advance_i),
            loop_index_definition=ref('names', index_i), solution_definition=ref('names', result_i), digest_input_read=ref('names', update_i)),
        local_data_paths=dict(argv_to_int=path(argv.value, call['int_call'].positional[0]),
            int_to_binding=path(call['int_call'].result, count.value),
            binding_read_to_actual=path(actual.value, call['invocation'].positional[-1]),
            count_read_to_range=path(range_count.value, call['range_call'].positional[0]),
            range_to_iterator=path(call['range_call'].result, create.source), iterator_to_advance=path(create.value, advance.source),
            advance_to_index=path(advance.value, index.value), solve_to_definition=path(call['solve'].result, result.value),
            solution_read_to_digest=path(update.value, call['append'].positional[0]),
            returned_digest_to_print=path(call['invocation'].result, call['print_call'].positional[0])))


def _member(root, ref):
    path = root / ref['uri']
    _require(not Path(ref['uri']).is_absolute() and path.is_relative_to(root) and path.resolve(strict=True) == path,
             'wire member is outside the admitted artifact')
    return dict(ref, uri=str(path))


def _upstream(path, source_ref, count):
    ref = _ref(path, _UPSTREAM)
    root, doc = Path(ref['uri']).parent, read_artifact(ref)
    profile_ref = _member(root, doc['profile'])
    _require(profile_ref['schema'] == _PROFILE, 'incorrect upstream header profile')
    profile = read_artifact(profile_ref)
    linked_source = _member(root, doc['source_seed'])
    _require(linked_source == dict(source_ref, uri=linked_source['uri']) and read_artifact(linked_source)
        and doc['schema_version'] == profile['schema_version'] == 1
        and doc['source_count_verified'] is profile['count_source_verified'] is True
        and doc['case'] == profile['case'] and doc['count'] == count == doc['case']['call_count'],
        'numerical witnesses are from a different source profile')
    legacy, target = (bytes(profile[key]) for key in ('legacy_header', 'target_header'))
    magic, _, size, nnz, _, flags = _CSR.unpack(legacy[-_CSR.size:])
    recipe = profile['recipe_id']
    _require(type(recipe) is str and re.fullmatch(r'recipe-v1:[0-9a-f]{64}', recipe) is not None
        and recipe == doc['recipe_id'] and len(legacy) == _HEADER_SIZES[0] and magic == b'SCARPSR1'
        and size == doc['case']['size'] > 0 and nnz == doc['case']['nnz'] > 0 and flags == 0,
        'incorrect canonical CSR wire envelope')
    rebound = legacy[:-_CSR.size] + _CSR.pack(magic, bytes.fromhex(recipe[len('recipe-v1:'):]), size, nnz, count, flags)
    _require(target == rebound and len(doc['wires']) == len(_WIRE_ROLES)
        and doc['seed_input'] == doc['wires'][0] and doc['canonical_csr'] == doc['wires'][-1],
        'checked wire metadata differs')
    wires, bodies = {}, {}
    for original, role, offset, historical_key in zip(doc['wires'], _WIRE_ROLES, _HEADER_SIZES,
            ('seed', 'topology', 'mesh', 'basis', 'coo'), strict=True):
        wire = _member(root, original)
        data = read_artifact(wire, json_document=False)
        body = sha256(data[offset:]).hexdigest()
        _require(len(data) > offset and data[:offset] == target[-offset:]
            and body == doc['identical_payload_sha256'][historical_key], 'header or numeric body differs for ' + role)
        wires[role], bodies[role] = dict(artifact=wire, metadata_bytes=offset), body
    return dict(artifact=ref, profile=profile_ref, wires=wires, numeric_payload_sha256=bodies, size=size, nnz=nnz)


def _observed(original, calls, caller, region, graph, ingress):
    result = {}
    sites = {name: (caller, call) for name, call in ingress.items()}
    sites.update({name: (caller, region[name]) for name in ('int_call', 'invocation', 'print_call')})
    sites.update({name: (region['code'], region[name]) for name in (
        'range_call', 'make_digest', 'prepare', 'solve', 'append', 'digest_result')})
    for name, (code, node) in sites.items():
        locator, observation = seed._observation(calls, code, node)
        candidate = calls['observed_targets'][int(locator.rsplit('/', 1)[1])]
        meta = json.loads(observation['metadata_json'])
        result[name] = dict(locator=locator, caller_site=observation['caller_site'],
            operation=candidate['operation'], source_call=observation['source_call'],
            target_code=observation['target_code'], target_kind=observation['target_kind'],
            carrier=meta.get('native_boundary'), type=meta.get('constructed_type'))
    expected_native = {
        'make_digest': ('_hashlib', 'openssl_sha256'), 'append': (None, 'HASH.update'),
        'digest_result': (None, 'HASH.hexdigest'), 'print_call': ('builtins', 'print'),
    }
    _require(all(result[role]['carrier'] is not None and tuple(result[role]['carrier'].get(name)
        for name in ('module', 'qualname')) == provider for role, provider in expected_native.items()),
        'observed digest or stdout carrier differs')
    invocation = result['invocation']
    locator, observation = seed._observation(calls, caller, region['invocation'])
    _require(locator == invocation['locator'] and invocation['source_call'] == graph['entry_call_port']['port'],
             'observed call is not the selected F4 source call port')
    interface_i, interface = _one(((i, row) for i, row in enumerate(calls['interfaces'])
        if row['caller_scope'] == calls['entry_scope'] and row['frame']['operation'] == invocation['operation']['node']
        and row['frame']['selection'] == observation['evidence']
        and row['interface']['call'] == graph['call_facts']['invocation']), 'observed source call interface')
    _one((route for route in interface['interface']['link']['parameters'] if route['kind'] == 'ARGUMENT'
          and route['formal'] == graph['formal_port']['port']
          and route['actuals'] == [graph['entry_actual_port']['port']]), 'caller count actual to callee formal')
    provenance = original['provenance']
    formal_locator = provenance.get('observed_loop_count')
    match = re.fullmatch(r'(#/records/[0-9]+)/fields/([a-z_]+)', formal_locator or '')
    _require(match is not None, 'missing typed runtime count formal')
    frames = read_artifact(provenance['frames'])
    record = seed._record(frames, match[1])
    observed = record['fields'][match[2]]
    _require(observed == dict(formal=region['count'], type=['builtins', 'int'], value=original['iterations'])
        and frames['process_id'] == observation['process_id'], 'observed formal is not the admitted call attempt')
    return dict(argv=original['argv'], count=original['iterations'], observation_kind='EXACT_TYPED_ORIGINAL_FORMAL',
        graph=original['entry_calls'], original_run=provenance['original_run'], frames=provenance['frames'],
        formal_locator=formal_locator, formal=observed, call_interface=dict(locator=f'#/interfaces/{interface_i}',
            caller_scope=interface['caller_scope'], callee_scope=interface['callee_scope'],
            call=interface['interface']['call']), sites=result)


def project(source_seed_paths, *, upstream_paths=(), max_count=_PLACED_LIMIT):
    """Replay distinct source runs and construct a bounded symbolic count interface."""
    _require(type(max_count) is int and 1 <= max_count <= _PLACED_LIMIT,
             'maximum count must be within the admitted native bound')
    source_refs = [_ref(path, seed.SCHEMA) for path in source_seed_paths]
    _require(len(source_refs) >= 2, 'two independently captured original arguments are required')
    docs = [seed.check(ref['uri']) for ref in source_refs]
    prime = docs[0]
    counts = [doc['iterations'] for doc in docs]
    _require(len(set(counts)) == len(docs) and all(1 <= count <= _PLACED_LIMIT for count in counts),
             'distinct bounded historical positive counts are required')
    _require(all(all(doc[field] == prime[field] for field in _SOURCE_CONSTANTS) for doc in docs[1:]),
             'historical source or typed geometry constants differ')
    entry = prime['entry']
    raw = read_artifact(entry, json_document=False)
    caller = compile(raw, entry['uri'], 'exec', dont_inherit=True, optimize=0)
    tree = ast.parse(raw, filename=entry['uri'])
    captures = [seed._call_projection(doc['entry_calls']['uri']) for doc in docs]
    _require(all(ref == doc['entry_calls'] for doc, (ref, _) in zip(docs, captures, strict=True)),
             'historical source call witnesses differ')
    projections = [seed._iteration(tree, calls, caller, doc['argv'])
                   for (_, calls), doc in zip(captures, docs, strict=True)]
    _require(all(row[0] == count and row[1] is projections[0][1] and row[-1] == projections[0][-1]
                 for row, count in zip(projections, counts, strict=True)), 'original count parsing path differs')
    region = _region(tree, projections[0][1], caller)
    definitions, ingress, depth = _geometry(tree, region)
    _require(depth == prime['depth'], 'mesh refinement differs from the qualified source constants')
    catalog = read_artifact(prime['activations'])['source_catalog']
    _require(all(read_artifact(doc['activations'])['source_catalog'] == catalog for doc in docs[1:]),
             'historical static source fact identity differs')
    graph = _graph(catalog, captures[0][1], region, entry)
    observations = [_observed(doc, calls, caller, region, graph, ingress)
                    for (_, calls), doc in zip(captures, docs, strict=True)]
    roles = observations[0]['sites']
    fields = ('caller_site', 'operation', 'source_call', 'target_code', 'target_kind', 'carrier', 'type')
    _require(all(all(all(row['sites'][role][key] == item[key] for key in fields) for role, item in roles.items())
                 for row in observations[1:]), 'selected call or provider differs across the two source histories')
    upstream_paths = tuple(upstream_paths)
    _require(not upstream_paths or len(upstream_paths) == len(docs), 'each source run must have its own upstream receipt')
    wires = [_upstream(path, source_ref, count)
             for path, source_ref, count in zip(upstream_paths, source_refs, counts)]
    body = wires[0]['numeric_payload_sha256'] if wires else None
    if wires:
        _require(all((row['size'], row['nnz'], row['numeric_payload_sha256']) == (wires[0]['size'], wires[0]['nnz'], body)
            for row in wires[1:]) and len({row['wires']['canonical_csr']['artifact']['sha256'] for row in wires}) == len(wires),
            'expected equal numerical bodies and distinct count-qualified wire envelopes')
    frontiers = ['IMPORTED_PROVIDER_EFFECTS_AND_AMBIENT_ARGV_READS_OPEN', 'FAILURE_PATHS_AND_STDERR_OPEN',
                 'LIVE_CALLABLE_LAYOUT_AND_NUMERICAL_PROVIDER_GUARDS_REQUIRED']
    if not wires:
        frontiers.append('CSR_NUMERICAL_PAYLOAD_NOT_ATTESTED')
    return dict(schema=SCHEMA, schema_version=1, target_executed=False, entry=entry,
        source_seeds=source_refs, source_catalog=catalog, upstream=[row['artifact'] for row in wires],
        count_port=dict(kind='ARGV_INTEGER', argv_index=1, argv_arity=1, dtype='SIGNED_INT32',
            encoding='CANONICAL_ASCII_DECIMAL', minimum=1, maximum=max_count, observed_counts=counts,
            formal=region['count'], argument_ordinal=3, parser_span=_span(region['int_call']),
            argv_span=_span(region['argv']), sys_binding_span=_span(region['sys_import']),
            positive_guard_span=_span(region['main'].body[1].test), formal_port=graph['formal_port']),
        loop=dict(code=describe_code(region['code']), caller=describe_code(caller), span=_span(region['loop']),
            iterator=dict(start=0, stop='count', step=1, index=region['index'], unrolled=False,
                next_site=_instruction(region['code'], region['range_call'], ('FOR_ITER',))),
            scale=dict(expression='1.0 + (i + 1) / (count + 1)', span=_span(region['scale']),
                dependencies=['i', 'count'], arithmetic='INTEGER_ADD_THEN_BINARY64_DIVIDE_AND_ADD_NO_FMA',
                operation_order=['INTEGER_I_PLUS_ONE', 'INTEGER_COUNT_PLUS_ONE', 'BINARY64_DIVIDE', 'BINARY64_ADD_ONE']),
            ordered_effects=['CREATE_SHA256', 'PREPARE', 'SOLVE', 'APPEND_SOLUTION', 'FINALIZE_HEXDIGEST', 'PRINT'],
            digest=dict(algorithm='SHA256', binding_span=_span(region['digest_import']),
                update_role='append', update_input='SAME_ITERATION_SOLVE_RESULT', order='ASCENDING_LOOP_INDEX',
                finalization='NORMAL_COMPLETION_OF_ALL_UPDATES', stdout='ONE_HEXDIGEST_LINE', prior_prefix_reusable=False),
            graph=graph),
        invariant_geometry=dict(scope='PINNED_SOURCE_PROVIDER_NORMAL_PATH', count_independent=True,
            refinement_depth=depth, source_fields_sha256=sha256(json.dumps(prime['fields'], sort_keys=True).encode()).hexdigest(),
            source_constants=prime['fields'], source_providers=prime['sources'], actuals=definitions,
            metadata_is_count_dependent=True, whole_wire_reusable=False, numeric_payload_sha256=body,
            upstream_shape=dict(size=wires[0]['size'], nnz=wires[0]['nnz']) if wires else None,
            historical_wires=wires), observations=observations, proof_frontiers=frontiers)


def check(path):
    stored = read_artifact(_ref(path, SCHEMA))
    for name in ('source_seeds', 'upstream'):
        for reference in stored[name]:
            read_artifact(reference)
    rebuilt = project([ref['uri'] for ref in stored['source_seeds']],
        upstream_paths=[ref['uri'] for ref in stored['upstream']], max_count=stored['count_port']['maximum'])
    _require(_canonical(stored) == _canonical(rebuilt), 'stored source sparse count does not replay')
    return rebuilt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-seed', type=Path, action='append', default=[])
    parser.add_argument('--upstream', type=Path, action='append', default=[])
    parser.add_argument('--max-count', type=int, default=_PLACED_LIMIT)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument('--out', type=Path)
    output.add_argument('--check', type=Path)
    args = parser.parse_args(argv)
    if args.check:
        result = check(args.check)
    else:
        result = project(args.source_seed, upstream_paths=args.upstream, max_count=args.max_count)
        with args.out.open('x', encoding='utf-8') as handle:
            json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')
    print(json.dumps(dict(count_port=result['count_port'], geometry=result['invariant_geometry']['upstream_shape'])))


if __name__ == '__main__':
    main()
