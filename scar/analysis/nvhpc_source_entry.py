"""Project a normal-success Python raw matrix CLI into an ordered MM recipe."""

import ast
from dataclasses import dataclass
from hashlib import sha256
from math import prod
from pathlib import Path
from types import CodeType

from scar.analysis.fixed_grid_entry_abi import (
    _literal, _method_call, _name, _need, _no_definition_effects, _span,
)
from scar.analysis.fixed_grid_source import _attribute, _binding
from scar.compiler.aten_export_cuda_source import qualify_cuda_export
from scar.compiler.aten_native_recipe import SCHEMA as RECIPE, _fields, _identifier
from scar.compiler.tensor_program_graph import straight_line_graph, tensor_type
from scar.ir.program import codec
from scar.ir.v2.schemas import canonical_json
from scar.optimization.aten_export_rewrite import FUNCTIONAL_EXPORT_SCHEMAS
from scar.trace.code_snapshot import code_key


SCHEMA = 'scar.nvhpc-source-matrix-file-entry'
_MM = FUNCTIONAL_EXPORT_SCHEMAS['aten.mm.default']
_MODULES = {('sys',), ('torch',), ('pathlib', 'Path')}


@dataclass(frozen=True, slots=True)
class _Argument:
    index: int
    binding: str
    span: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _Tensor:
    name: str
    shape: tuple[int, int]
    producer: str


def _module(tree):
    body = list(tree.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and type(body[0].value.value) is str:
        body.pop(0)
    _need(len(body) >= 3 and type(body[-2]) is ast.FunctionDef and type(body[-1]) is ast.If,
          'matrix entry requires one function and a direct __main__ launch')
    function, launch = body[-2:]
    _need(_no_definition_effects(function) and not list(ast.iter_child_nodes(function.args)),
          'matrix entry function must have no parameters or definition-time effects')
    check = launch.test
    _need(isinstance(check, ast.Compare) and _name(check.left, '__name__') and len(check.ops) == 1
          and type(check.ops[0]) is ast.Eq and isinstance(check.comparators[0], ast.Constant)
          and check.comparators[0].value == '__main__' and not launch.orelse and len(launch.body) == 1
          and isinstance(launch.body[0], ast.Expr) and isinstance(launch.body[0].value, ast.Call)
          and _name(launch.body[0].value.func, function.name) and not launch.body[0].value.args
          and not launch.body[0].value.keywords, 'matrix entry launch must call the declared function directly')
    protected = {'str', '__name__', function.name}
    _need(len(protected) == 3, 'matrix entry function cannot shadow launch or path conversion')
    providers = {}
    imports = []
    for row in body[:-2]:
        if type(row) is ast.Import:
            entries = [((alias.name,), _binding(alias)) for alias in row.names]
        elif type(row) is ast.ImportFrom and row.level == 0:
            entries = [((row.module, alias.name), _binding(alias)) for alias in row.names]
        else:
            raise ValueError('matrix entry has unqualified module-level behavior')
        for key, local in entries:
            _need(key in _MODULES and key not in providers and local not in protected,
                  'matrix entry import is unsupported, duplicate or shadowed')
            providers[key] = local
            protected.add(local)
            imports.append(dict(module=key[0], member=None if len(key) == 1 else key[1],
                binding=local, source_span=list(_span(row))))
    _need(providers.keys() == _MODULES, 'matrix entry requires uniquely bound sys, pathlib.Path and torch')
    return function, launch, providers, imports, protected


def _argument(node, binding, pathlib, sys):
    if not (isinstance(node, ast.Call) and _name(node.func, pathlib)):
        return None
    _need(len(node.args) == 1 and not node.keywords, 'matrix file path needs one literal argv selection')
    selection, = node.args
    _need(isinstance(selection, ast.Subscript) and _attribute(selection.value, sys, 'argv')
          and isinstance(selection.slice, ast.Constant) and type(selection.slice.value) is int
          and selection.slice.value > 0, 'matrix file path requires a positive literal argv index')
    return _Argument(selection.slice.value, binding, _span(node))


def _shape(dimensions):
    _need(type(dimensions) is tuple and len(dimensions) == 2
          and all(type(extent) is int and 0 < extent < 2**31 for extent in dimensions)
          and prod(dimensions) < 2**61, 'matrix file shape must be positive rank-two LP64 geometry')
    return dimensions


def _input(node, values, torch):
    _need(_method_call(node, 'view', 2) or _method_call(node, 'view', 1),
          'matrix input requires a two-dimensional view')
    load = node.func.value
    _need(isinstance(load, ast.Call) and _attribute(load.func, torch, 'from_file'),
          'matrix read must use the qualified torch.from_file provider')
    keywords = {item.arg: item.value for item in load.keywords}
    _need(len(keywords) == len(load.keywords), 'matrix read cannot rebind provider keyword arguments')
    if len(load.args) == 3 and keywords.keys() == {'dtype'}:
        path, shared, count = load.args
    elif len(load.args) == 1 and keywords.keys() == {'dtype', 'shared', 'size'}:
        path, = load.args
        shared, count = keywords['shared'], keywords['size']
    else:
        raise ValueError('matrix read requires path, shared=False, size and float32 dtype')
    _need(isinstance(shared, ast.Constant) and shared.value is False
          and _attribute(keywords['dtype'], torch, 'float32'), 'matrix read requires a private CPU float32 mapping')
    _need(isinstance(path, ast.Call) and _name(path.func, 'str') and len(path.args) == 1 and not path.keywords
          and isinstance(path.args[0], ast.Name), 'matrix read path must convert an original argv binding')
    argument = values.get(path.args[0].id)
    _need(type(argument) is _Argument, 'matrix read path has no original argv slot')
    dimensions = _shape(tuple(_literal(dim, values) for dim in node.args) if len(node.args) == 2
                        else _literal(node.args[0], values))
    extent = _literal(count, values)
    _need(type(extent) is int and extent == prod(dimensions), 'matrix mapping extent differs from its view shape')
    return argument, dimensions


def _operands(node, values, torch):
    if not (isinstance(node, ast.Call) and _attribute(node.func, torch, 'mm')):
        return None
    _need(not node.keywords and len(node.args) == 2 and all(isinstance(node_, ast.Name) for node_ in node.args),
          'matrix product requires two ordered unmodified Tensor bindings')
    operands = tuple(values.get(node_.id) for node_ in node.args)
    _need(all(type(value) is _Tensor for value in operands) and operands[0] != operands[1]
          and operands[0].shape[1] == operands[1].shape[0], 'matrix product has incompatible or aliased operand roles')
    return operands


def _output(node, values):
    _need(_method_call(node, 'write_bytes', 1) and isinstance(node.func.value, ast.Name),
          'matrix entry has an unsupported effect outside ordered file publication')
    output = values.get(node.func.value.id)
    data, = node.args
    _need(type(output) is _Argument and _method_call(data, 'tobytes', 0)
          and _method_call(data.func.value, 'numpy', 0), 'matrix publication requires raw NumPy bytes of one result')
    array = data.func.value.func.value
    if _method_call(array, 'contiguous', 0):
        array = array.func.value
    result = values.get(array.id) if isinstance(array, ast.Name) else None
    _need(type(result) is _Tensor and result.producer == 'MM', 'matrix file publication must write a product result')
    return output, result


def _parse_body(function, providers, protected):
    values, reads, products, writes = {}, [], [], []
    torch, sys, pathlib = (providers[key] for key in (('torch',), ('sys',), ('pathlib', 'Path')))
    for index, row in enumerate(function.body):
        if (index == 0 and isinstance(row, ast.Expr) and isinstance(row.value, ast.Constant)
                and type(row.value.value) is str):
            continue
        if isinstance(row, ast.Expr):
            output, result = _output(row.value, values)
            writes.append((output, result, _span(row.value)))
            continue
        _need(not writes and type(row) is ast.Assign and len(row.targets) == 1 and type(row.targets[0]) is ast.Name,
              'matrix entry admits distinct assignments followed by terminal file publications')
        name, expression = row.targets[0].id, row.value
        _need(name not in protected and name not in values, 'matrix local binding is duplicate or shadows its provider')
        argument = _argument(expression, name, pathlib, sys)
        operands = _operands(expression, values, torch)
        if argument is not None:
            value = argument
        elif operands is not None:
            left, right = operands
            value = _Tensor(name, _shape((left.shape[0], right.shape[1])), 'MM')
            products.append((value, operands, _span(expression)))
        elif isinstance(expression, ast.Call) and isinstance(expression.func, ast.Attribute) and expression.func.attr == 'view':
            _need(not products, 'matrix reads must precede all products in source order')
            path, dimensions = _input(expression, values, torch)
            value = _Tensor(name, dimensions, 'INPUT')
            reads.append((path, value, _span(expression)))
        else:
            value = _literal(expression, values)
            _need(type(value) is int, 'matrix entry additional values must be integer literals or extents')
        values[name] = value
    paths = [path for path, _, _ in (*reads, *writes)]
    parsed = {value for value in values.values() if type(value) is _Argument}
    _need(reads and products and writes and len(paths) == len(parsed) and set(paths) == parsed
          and [path.index for path in paths] == list(range(1, len(paths) + 1)),
          'matrix ABI requires all argv inputs then outputs in original effect order without duplicates')
    return reads, products, writes


def _bind_recipe(recipe, reads, products, writes):
    _fields(recipe, {'schema', 'schema_version', 'inputs', 'operations', 'outputs'}, 'matrix entry recipe')
    _need(recipe['schema'] == RECIPE and type(recipe['schema_version']) is int and recipe['schema_version'] == 1
          and all(type(recipe[key]) is list and len(recipe[key]) == len(rows)
              for key, rows in (('inputs', reads), ('operations', products), ('outputs', writes))),
          'matrix source counts differ from the ordered functional recipe')
    mapped, types = {}, {}
    for entry, (_, source, _) in zip(recipe['inputs'], reads, strict=True):
        _fields(entry, {'id', 'shape'}, 'matrix input')
        name = _identifier(entry['id'], 'matrix input')
        _need(entry['shape'] == list(source.shape) and name not in types,
              'matrix input geometry or value identity differs from the recipe')
        mapped[source.name], types[name] = name, tensor_type(source.shape)
    for entry, (source, operands, _) in zip(recipe['operations'], products, strict=True):
        _fields(entry, {'id', 'schema', 'arguments'}, 'matrix operation')
        name = _identifier(entry['id'], 'matrix operation')
        _need(entry['schema'] == _MM and name not in types
              and entry['arguments'] == [dict(value=mapped[value.name]) for value in operands],
              'matrix source uses a different ordered MM dataflow from the recipe')
        mapped[source.name], types[name] = name, tensor_type(source.shape)
    _need(recipe['outputs'] == [mapped[source.name] for _, source, _ in writes],
          'matrix output publications differ from the ordered functional recipe')
    graph = straight_line_graph(recipe, types)
    payload = (codec.dumps(graph) + '\n').encode()
    program = dict(path='plan/program.json', sha256=sha256(payload).hexdigest(), size_bytes=len(payload))
    projection = qualify_cuda_export(recipe, graph=graph, program_ref=program,
        input_kinds={item['id']: 'INPUT' for item in recipe['inputs']})
    return mapped, program, projection


def _effect_graph(imports, inputs, products, outputs, mapped):
    nodes, edges = [dict(id='launch', kind='PROCESS_LAUNCH')], []

    def ordered(row):
        edges.append(dict(source=nodes[-1]['id'], target=row['id'], kind='SOURCE_ORDER'))
        nodes.append(row)

    for index, item in enumerate(imports):
        ordered(dict(id=f'import:{index}', kind='IMPORT_INITIALIZATION', **item))
    producers = {}
    for row in inputs:
        identity = f"input:{row['argv']}"
        ordered(dict(id=identity, kind='PRIVATE_FILE_MAPPING_AND_TENSOR_VIEW', **row))
        producers[row['value']] = identity
    for value, operands, span in products:
        identity = 'mm:' + mapped[value.name]
        ordered(dict(id=identity, kind='FUNCTIONAL_MM', value=mapped[value.name], source_span=list(span)))
        for ordinal, operand in enumerate(operands):
            edges.append(dict(source=producers[mapped[operand.name]], target=identity, kind='DATA', operand=ordinal))
        producers[mapped[value.name]] = identity
    for row in outputs:
        identity = f"output:{row['argv']}"
        ordered(dict(id=identity, kind='SYNC_FILE_CONTENT_PUBLICATION', **row))
        edges.append(dict(source=producers[row['value']], target=identity, kind='DATA'))
    ordered(dict(id='release', kind='RELEASE_LOCAL_REFERENCES_ON_FUNCTION_RETURN'))
    ordered(dict(id='exit', kind='NORMAL_PROCESS_EXIT'))
    return dict(nodes=nodes, edges=edges)


def _source_parts(source):
    path = Path(source).resolve(strict=True)
    _need(path.is_file() and path.suffix == '.py', 'matrix entry source must be a regular Python file')
    raw = path.read_bytes()
    tree = ast.parse(raw, filename=str(path))
    function, launch, providers, imports, protected = _module(tree)
    reads, products, writes = _parse_body(function, providers, protected)
    return path, raw, function, launch, imports, reads, products, writes


def build_nvhpc_source_recipe(source):
    """Construct stable graph ports directly from the admitted original source."""
    _, _, _, _, _, reads, products, writes = _source_parts(source)
    names = {}
    inputs = []
    for index, (_, value, _) in enumerate(reads):
        name = names[value.name] = f'input_{index}'
        inputs.append(dict(id=name, shape=list(value.shape)))
    operations = []
    for index, (value, operands, _) in enumerate(products):
        name = names[value.name] = f'mm_{index}'
        operations.append(dict(id=name, schema=_MM, arguments=[dict(value=names[operand.name]) for operand in operands]))
    return dict(schema=RECIPE, schema_version=1, inputs=inputs, operations=operations,
        outputs=[names[value.name] for _, value, _ in writes])


def derive_nvhpc_source_entry(source, recipe):
    """Bind original Python source bytes and logical file effects to a typed recipe."""
    path, raw, function, launch, imports, reads, products, writes = _source_parts(source)
    compiled = compile(raw, str(path), 'exec', dont_inherit=True)
    body, = (item for item in compiled.co_consts if type(item) is CodeType and item.co_qualname == function.name)
    mapped, program, projection = _bind_recipe(recipe, reads, products, writes)

    def ports(rows):
        return [dict(argv=argument.index, value=mapped[value.name], shape=list(value.shape),
            nbytes=4*prod(value.shape), path_span=list(argument.span), source_span=list(span))
            for argument, value, span in rows]

    inputs, outputs = ports(reads), ports(writes)
    _need(raw == path.read_bytes(), 'matrix source changed during projection')
    return dict(schema=SCHEMA, schema_version=1, source=dict(path=str(path), sha256=sha256(raw).hexdigest(),
        size_bytes=len(raw), function=function.name, function_code_key=code_key(body), launch_span=list(_span(launch))),
        recipe_sha256=sha256(canonical_json(recipe).encode()).hexdigest(), program=program,
        projection=projection, argv_count=len(inputs)+len(outputs), inputs=inputs, outputs=outputs,
        dtype='FLOAT32', layout='CONTIGUOUS_ROW_MAJOR', byte_order='little',
        imports=imports, graph=_effect_graph(imports, inputs, products, outputs, mapped),
        domain=dict(normal_completion_only=True, inputs='STABLE_EXACT_SIZE_READ_ONLY_REGULAR_FILES',
            outputs='DISTINCT_EXCLUSIVE_FRESH_REGULAR_FILE_PATHS_WITH_EXISTING_PARENTS',
            input_output_aliases=False, stdout='EMPTY', stderr='EMPTY', exit_status=0,
            observables='ORDERED_LOGICAL_FILE_CONTENTS_AND_PROCESS_STDIO_EXIT',
            unresolved=['RESOLVED_IMPORT_PROVIDER_IDENTITY_AND_INITIALIZATION_EFFECTS',
                        'ARM_SPECIFIC_FLOAT32_NUMERICAL_CONTRACT', 'FILESYSTEM_AND_PROCESS_SCOPE_GUARDS']),
        providers_verified=False, target_executed=False, deployment_authorized=False)


def verify_nvhpc_source_entry(receipt, source, recipe):
    """Replay every source, Tensor and effect binding before use."""
    expected = derive_nvhpc_source_entry(source, recipe)
    _need(receipt == expected, 'matrix source entry differs from original typed effect replay')
    return expected
