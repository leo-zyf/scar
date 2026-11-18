"""Check an ordinary Python function against one exact native ATen recipe."""

import ast
from hashlib import sha256
from keyword import iskeyword
from pathlib import Path
import sys
from types import CodeType

from scar.compiler.aten_native_recipe import SCHEMA as RECIPE_SCHEMA, SCHEMA_VERSION as RECIPE_VERSION
from scar.trace.code_snapshot import code_key


SCHEMA = 'scar.aten-python-trial-source'
CONTRACT_SCHEMA = 'scar.aten-python-wrapper-trial-contract'
SCOPE = 'EXPLICIT_PYTHON_CPU_DIFFERENTIAL_TRIAL'
_MAX_SOURCE_BYTES = 8 * 1024 * 1024


def _name(value):
    return type(value) is str and value.isidentifier() and not iskeyword(value)


def _wrappers(contract):
    if (type(contract) is not dict or contract.keys() != {
            'schema', 'schema_version', 'scope', 'torch_global', 'wrappers', 'wrapper_effects_declared'}
            or (contract['schema'], contract['schema_version'], contract['scope']) != (CONTRACT_SCHEMA, 1, SCOPE)
            or type(contract['schema_version']) is not int or not _name(contract['torch_global'])
            or contract['wrapper_effects_declared'] is not True):
        raise ValueError('an explicit Python CPU wrapper trial contract is required')
    rows = contract['wrappers']
    if type(rows) is not list or not rows:
        raise ValueError('the wrapper trial contract has no declarations')
    by_name = {}
    for row in rows:
        if (type(row) is not dict or row.keys() != {'attribute', 'schema', 'implicit_arguments'}
                or not _name(row['attribute']) or row['attribute'] in by_name
                or type(row['schema']) is not str or not row['schema'].startswith('aten::')):
            raise ValueError('trial wrappers require unique attributes and exact ATen schemas')
        implicit = row['implicit_arguments']
        if type(implicit) is not list or any(type(item) is not dict or item.keys() != {'scalar'}
                or type(item['scalar']) is not int or not -(1 << 63) <= item['scalar'] < (1 << 63)
                for item in implicit):
            raise ValueError('implicit trial operands must be exact signed integer scalars')
        by_name[row['attribute']] = dict(row, implicit_arguments=[dict(item) for item in implicit])
    return by_name


def _function(module, name, torch_global):
    matches = [node for node in module.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
               and node.name == name]
    if len(matches) != 1 or type(matches[0]) is not ast.FunctionDef:
        raise ValueError('one direct module-level synchronous trial function is required')
    function = matches[0]
    args = function.args
    positional = args.posonlyargs + args.args
    if (function.decorator_list or function.returns is not None or function.type_comment is not None
            or getattr(function, 'type_params', ()) or args.vararg is not None or args.kwarg is not None
            or args.defaults or args.kwonlyargs or any(arg.annotation is not None or arg.type_comment is not None
                for arg in positional)):
        raise ValueError('trial functions may only declare required positional parameters')
    names = [arg.arg for arg in positional]
    if torch_global in names or len(set(names)) != len(names):
        raise ValueError('trial parameters may not overwrite the declared Torch global')
    if type(function.body[-1]) is not ast.Return or function.body[-1].value is None:
        raise ValueError('the trial function must end with a Tensor return')
    return function, names


def _recipe(recipe):
    if (type(recipe) is not dict or recipe.keys() != {'schema', 'schema_version', 'inputs', 'operations', 'outputs'}
            or (recipe['schema'], recipe['schema_version']) != (RECIPE_SCHEMA, RECIPE_VERSION)
            or type(recipe['schema_version']) is not int
            or any(type(recipe[key]) is not list or not recipe[key] for key in ('inputs', 'operations', 'outputs'))):
        raise ValueError('a nonempty normalized native ATen recipe is required')
    inputs = recipe['inputs']
    if any(type(item) is not dict or item.keys() != {'id', 'shape'} or not _name(item['id']) for item in inputs):
        raise ValueError('native input slots are incomplete')
    return [item['id'] for item in inputs]


def _body(function, names, recipe, native_inputs, torch_global, declarations):
    parameters = {name: i for i, name in enumerate(names)}
    slots, formal_order, used = {}, [], {}
    operations = []
    operation_ids = set(native_inputs)
    if len(operation_ids) != len(native_inputs):
        raise ValueError('native recipe input IDs must be distinct')

    def input_slot(node):
        if type(node) is not ast.Name:
            raise ValueError('native Tensor operands must be direct local or parameter names')
        if node.id not in slots:
            if node.id not in parameters or len(formal_order) == len(native_inputs):
                raise ValueError('source operand has no known native Tensor slot')
            slots[node.id] = native_inputs[len(formal_order)]
            formal_order.append(parameters[node.id])
        return dict(value=slots[node.id])

    def operation(node):
        if (type(node) is not ast.Call or type(node.func) is not ast.Attribute or node.keywords
                or type(node.func.value) is not ast.Name or node.func.value.id != torch_global
                or node.func.attr not in declarations):
            raise ValueError('trial operations must call a declared Torch wrapper with positional Tensor names')
        if len(operations) == len(recipe['operations']):
            raise ValueError('source contains extra calls outside the native recipe')
        declaration = declarations[node.func.attr]
        arguments = [input_slot(argument) for argument in node.args] + declaration['implicit_arguments']
        original = recipe['operations'][len(operations)]
        if (type(original) is not dict or original.keys() != {'id', 'schema', 'arguments'}
                or not _name(original['id']) or original['id'] in operation_ids
                or original['schema'] != declaration['schema'] or original['arguments'] != arguments
                or any('scalar' in item and type(item['scalar']) is not int for item in original['arguments'])):
            raise ValueError(f'source call at line {node.lineno} differs from native operand or schema order')
        operations.append(dict(id=original['id'], schema=declaration['schema'], arguments=arguments))
        operation_ids.add(original['id'])
        used[node.func.attr] = declaration
        return original['id']

    results = {}
    for statement in function.body[:-1]:
        if (type(statement) is not ast.Assign or statement.type_comment is not None or len(statement.targets) != 1
                or type(statement.targets[0]) is not ast.Name):
            raise ValueError('the trial function body must assign each intermediate to one fresh local name')
        name = statement.targets[0].id
        if name == torch_global or name in parameters or name in results:
            raise ValueError('the trial function must preserve unique locals and the declared Torch global')
        result = operation(statement.value)
        slots[name] = results[name] = result

    def returned(node):
        if type(node) is not ast.Name or node.id not in results:
            raise ValueError('every returned Tensor must be a result of a trial operation')
        return results[node.id]

    result = function.body[-1].value
    if type(result) is ast.Call:
        return_kind, outputs = 'TENSOR', [operation(result)]
    elif type(result) is ast.Tuple:
        return_kind, outputs = 'TUPLE', [returned(item) for item in result.elts]
    else:
        return_kind, outputs = 'TENSOR', [returned(result)]
    if (len(formal_order) != len(native_inputs) or len(operations) != len(recipe['operations'])
            or outputs != recipe['outputs']):
        raise ValueError('original inputs, operations or return order differ from the complete native recipe')
    return formal_order, operations, return_kind, outputs, list(used.values())


def qualify_python_trial_source(path, function_name, recipe, contract):
    """Project closed local flow using declared wrapper effects without executing source.

    The resulting receipt supports differential trials against the exact live
    function, once the installer separately checks its globals and dispatch
    context. The source closure does not authorize a replacement on its own.
    """
    if not _name(function_name):
        raise ValueError('an ordinary Python function name is required')
    declarations = _wrappers(contract)
    native_inputs = _recipe(recipe)
    path = Path(path).resolve(strict=True)
    with path.open('rb') as stream:
        raw = stream.read(_MAX_SOURCE_BYTES + 1)
    if len(raw) > _MAX_SOURCE_BYTES:
        raise ValueError('trial source exceeds the source byte budget')
    torch_global = contract['torch_global']
    module = ast.parse(raw, filename=str(path), type_comments=True)
    function, names = _function(module, function_name, torch_global)
    formal_order, operations, return_kind, outputs, used = _body(
        function, names, recipe, native_inputs, torch_global, declarations)
    compiled = compile(raw, str(path), 'exec', dont_inherit=True, optimize=sys.flags.optimize)
    entries = [item for item in compiled.co_consts if type(item) is CodeType
               and item.co_qualname == function_name and item.co_firstlineno == function.lineno]
    if len(entries) != 1 or entries[0].co_freevars or entries[0].co_cellvars:
        raise ValueError('trial source has no unique closure-free compiled entry')
    return dict(schema=SCHEMA, schema_version=1, scope=SCOPE,
        source=dict(path=str(path), sha256=sha256(raw).hexdigest(), size_bytes=len(raw)),
        function=function_name, first_lineno=function.lineno, parameters=names,
        code_key=code_key(entries[0]), input_order=formal_order, return_kind=return_kind, torch_global=torch_global,
        wrappers=used, operations=operations, outputs=outputs,
        wrapper_effect_basis='EXPLICIT_DECLARATION', source_replacement_authorized=False)
