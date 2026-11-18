"""Seal one source call whose fixed unpack discards a private computed item."""

import ast
from types import CodeType

from scar.trace.code_snapshot import code_key


HELPER = '__scar_dead_private_item_call'


def _only(rows, message):
    rows = list(rows)
    if len(rows) != 1:
        raise ValueError(message)
    return rows[0]


def _span(node):
    return [node.lineno, node.col_offset, node.end_lineno, node.end_col_offset]


def _name(node, binding):
    return type(node) is ast.Name and node.id == binding


def _attr(node, attr):
    return type(node) is ast.Attribute and node.attr == attr


def _definition(tree, compiled, key):
    code = _only((code for code in compiled.co_consts if type(code) is CodeType and code_key(code) == key),
        'expected one original module-level CodeID')
    definition = _only((node for node in tree.body if type(node) is ast.FunctionDef and node.name == code.co_qualname),
        'expected one original ordinary function')
    if code.co_freevars or code.co_cellvars or any(type(node) in (
            ast.Global, ast.Nonlocal, ast.Yield, ast.YieldFrom, ast.Await) for node in ast.walk(definition)):
        raise ValueError('closure, suspension or redirected function bindings are unsupported')
    for node in ast.walk(definition):
        if type(node) is ast.Call and ((type(node.func) is ast.Name
                and node.func.id in ('locals', 'vars', 'dir', 'eval', 'exec'))
                or (type(node.func) is ast.Attribute
                    and node.func.attr in ('_getframe', 'currentframe', 'getouterframes', 'getinnerframes'))):
            raise ValueError('direct frame reflection is unsupported')
    return definition, code


def _pre_branch(test, dead, rows):
    """Evaluate only None identity tests under the admitted input shape."""
    if type(test) is ast.BoolOp and type(test.op) is ast.And and len(test.values) == 2:
        first, second = (_pre_branch(operand, dead, rows) for operand in test.values)
        return first and second
    if (type(test) is ast.Compare and len(test.comparators) == 1
            and type(test.comparators[0]) is ast.Constant and test.comparators[0].value is None
            and type(test.ops[0]) in (ast.Is, ast.IsNot)
            and (_name(test.left, dead) or _name(test.left, rows))):
        return _name(test.left, dead) == (type(test.ops[0]) is ast.Is)
    raise ValueError('earlier branch is not an exact None identity test')


def _find_producer(callee, producer_span):
    for index, root in enumerate(callee.body):
        branch = root
        previous = []
        while type(branch) is ast.If:
            for node in branch.body:
                if type(node) is ast.Assign and _span(node.value) == producer_span:
                    return index, branch, previous, node
            previous.append(branch.test)
            branch = branch.orelse[0] if len(branch.orelse) == 1 else None
    raise ValueError('dead item producer is not in an original top-level conditional suite')


def _complement(producer, formals):
    value = producer.value
    if (len(producer.targets) != 1 or type(producer.targets[0]) is not ast.Name
            or type(value) is not ast.Call or not _attr(value.func, 'setdiff1d')
            or type(value.func.value) is not ast.Name or len(value.args) != 2 or value.keywords):
        raise ValueError('unsupported private result producer')
    numpy = value.func.value.id
    arange, rows = value.args
    if (type(rows) is not ast.Name or type(arange) is not ast.Call or not _attr(arange.func, 'arange')
            or not _name(arange.func.value, numpy) or len(arange.args) != 1 or len(arange.keywords) != 1):
        raise ValueError('expected original integer index complement')
    shape, dtype = arange.args[0], arange.keywords[0]
    if (type(shape) is not ast.Subscript or not _attr(shape.value, 'shape')
            or type(shape.value.value) is not ast.Name or type(shape.slice) is not ast.Constant
            or type(shape.slice.value) is not int or shape.slice.value != 0 or dtype.arg != 'dtype'
            or not _attr(dtype.value, 'int32') or not _name(dtype.value.value, numpy)):
        raise ValueError('expected the original leading shape and exact int32 complement')
    matrix, dead, rows = shape.value.value.id, producer.targets[0].id, rows.id
    if len({matrix, dead, rows}) != 3 or not {matrix, dead, rows} <= set(formals):
        raise ValueError('index complement operands are not distinct original formals')
    return matrix, dead, rows, numpy


def _specialization_branch(callee, index, branch, previous, producer, dead, rows):
    for test in previous:
        if _pre_branch(test, dead, rows):
            raise ValueError('dead item producer branch is unreachable under admitted inputs')
    test = branch.test
    expected = {(dead, ast.Is), (rows, ast.IsNot)}
    if (type(test) is not ast.BoolOp or type(test.op) is not ast.And or len(test.values) != 2
            or any(type(part) is not ast.Compare or type(part.left) is not ast.Name
                or len(part.comparators) != 1 or type(part.comparators[0]) is not ast.Constant
                or part.comparators[0].value is not None for part in test.values)
            or {(part.left.id, type(part.ops[0])) for part in test.values} != expected
            or branch.body[0] is not producer):
        raise ValueError('producer requires the original guarded None/present branch')

    prefix = callee.body[:index]
    if (prefix and type(prefix[0]) is ast.Expr and type(prefix[0].value) is ast.Constant
            and type(prefix[0].value.value) is str):
        prefix = prefix[1:]
    flatten = set()
    assigned = set()
    for node in prefix:
        if (type(node) is not ast.Assign or len(node.targets) != 1 or type(node.targets[0]) is not ast.Name
                or node.targets[0].id not in {dead, rows} - assigned or type(node.value) is not ast.Call
                or type(node.value.func) is not ast.Name or len(node.value.args) != 1 or node.value.keywords
                or not _name(node.value.args[0], node.targets[0].id)):
            raise ValueError('unmodeled producer prefix before the selected branch')
        assigned.add(node.targets[0].id)
        flatten.add(node.value.func.id)
    if prefix and (assigned != {dead, rows} or len(flatten) != 1):
        raise ValueError('expected the same original normalizer for both index formals')
    return next(iter(flatten), None), [*branch.body[1:], *callee.body[index + 1:]]


def _identity_normalizer(tree, original, name):
    normalizer = _only((node for node in tree.body if type(node) is ast.FunctionDef and node.name == name),
        'expected one original index normalizer definition')
    code = _only((item for item in original.co_consts if type(item) is CodeType and item.co_qualname == name),
        'expected one original index normalizer CodeID')
    formals = [arg.arg for arg in (*normalizer.args.posonlyargs, *normalizer.args.args)]
    if (len(formals) != 1 or normalizer.args.vararg or normalizer.args.kwarg or normalizer.args.kwonlyargs
            or any(type(node) in (ast.Yield, ast.YieldFrom) for node in ast.walk(normalizer))):
        raise ValueError('expected an ordinary one-argument index normalizer')
    argument, = formals
    head = normalizer.body
    if (head and type(head[0]) is ast.Expr and type(head[0].value) is ast.Constant
            and type(head[0].value.value) is str):
        head = head[1:]
    first = head[0] if head else None
    if (type(first) is not ast.If or type(first.test) is not ast.Compare
            or not _name(first.test.left, argument) or len(first.test.ops) != 1
            or type(first.test.ops[0]) is not ast.Is or type(first.test.comparators[0]) is not ast.Constant
            or first.test.comparators[0].value is not None or len(first.body) != 1
            or type(first.body[0]) is not ast.Return):
        raise ValueError('normalizer must have the original immediate None identity return')
    result = first.body[0].value
    if not (_name(result, argument) or type(result) is ast.Constant and result.value is None):
        raise ValueError('normalizer changes the admitted None input')
    remainder = first.orelse or head[1:]
    if not remainder or type(remainder[0]) is not ast.If:
        raise ValueError('normalizer lacks the original immediate array identity branch')
    second = remainder[0]
    test = second.test
    if (type(test) is not ast.Call or not _name(test.func, 'isinstance') or test.keywords
            or len(test.args) != 2 or not _name(test.args[0], argument) or type(test.args[1]) is not ast.Name
            or len(second.body) != 1 or type(second.body[0]) is not ast.Return
            or not _name(second.body[0].value, argument)):
        raise ValueError('normalizer changes the admitted exact array input')
    array = test.args[1].id
    if {'isinstance', array}.intersection((*code.co_varnames, *code.co_freevars, *code.co_cellvars)):
        raise ValueError('normalizer must select the original array test globals')
    return code_key(code), array


def _unused_return(callee, suffix, dead, arity):
    returned_items = set()
    slots = set()
    for node in ast.walk(callee):
        if type(node) is not ast.Return:
            continue
        if type(node.value) is not ast.Tuple or len(node.value.elts) != arity:
            raise ValueError('callee must return the original fixed-width tuple')
        slot = _only((i for i, item in enumerate(node.value.elts) if _name(item, dead)),
            'callee return has no unique produced item')
        slots.add(slot)
        returned_items.add(node.value.elts[slot])
    slot = _only(slots, 'computed item does not occupy one original return slot')

    array_bindings = set()
    assertions = set()
    for statement in suffix:
        for node in ast.walk(statement):
            if (type(node) is ast.Assert and node.msg is None and type(node.test) is ast.Call
                    and _name(node.test.func, 'isinstance') and len(node.test.args) == 2 and not node.test.keywords
                    and _name(node.test.args[0], dead) and type(node.test.args[1]) is ast.Name):
                assertions.add(node.test.args[0])
                array_bindings.add(node.test.args[1].id)
    if len(array_bindings) > 1 or any(_name(node, dead) and node not in returned_items | assertions
            for statement in suffix for node in ast.walk(statement)):
        raise ValueError('computed item has a callee use beyond the selected return or array assertion')
    return slot, next(iter(array_bindings), None)


def seal_dead_item(source, filename, *, caller_key, callee_key, call_span, producer_span, optimize=0):
    """Return exact source facts and two code objects for a guarded call-site specialization."""
    tree = ast.parse(source, filename=filename)
    original = compile(tree, filename, 'exec', dont_inherit=True, optimize=optimize)
    caller, original_caller = _definition(tree, original, caller_key)
    callee, original_callee = _definition(tree, original, callee_key)
    if caller is callee or any(type(node) in (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
            and node is not callee for statement in callee.body for node in ast.walk(statement)):
        raise ValueError('specialization requires a separate callee with one executable scope')

    assignment = _only((node for node in caller.body if type(node) is ast.Assign and type(node.value) is ast.Call
        and _span(node.value) == list(call_span)), 'expected an original top-level fixed-unpack call')
    call = assignment.value
    unpack = assignment.targets[0]
    formals = [arg.arg for arg in (*callee.args.posonlyargs, *callee.args.args)]
    if (len(assignment.targets) != 1 or type(unpack) is not ast.Tuple
            or any(type(item) is not ast.Name for item in unpack.elts)
            or len({item.id for item in unpack.elts}) != len(unpack.elts)
            or not _name(call.func, callee.name) or callee.name in original_caller.co_varnames
            or call.keywords or len(call.args) != len(formals) or any(type(arg) is ast.Starred for arg in call.args)
            or callee.args.vararg or callee.args.kwarg or callee.args.kwonlyargs
            or HELPER in (*original_caller.co_names, *original_caller.co_varnames)):
        raise ValueError('unsupported original fixed-unpack call ABI')

    index, branch, previous, producer = _find_producer(callee, list(producer_span))
    matrix, dead, rows, numpy = _complement(producer, formals)
    flatten, suffix = _specialization_branch(callee, index, branch, previous, producer, dead, rows)
    slot, array = _unused_return(callee, suffix, dead, len(unpack.elts))
    flatten_key = None
    if flatten is not None:
        flatten_key, normalized_array = _identity_normalizer(tree, original, flatten)
        if array is not None and array != normalized_array:
            raise ValueError('normalizer and computed item assert must use the same array binding')
        array = normalized_array
    globals_used = {name for name in (numpy, array, flatten, 'isinstance') if name is not None}
    if globals_used.intersection(original_callee.co_varnames):
        raise ValueError('selected callee providers must be ordinary global bindings')
    unused = unpack.elts[slot].id
    if any(_name(node, unused) and type(node.ctx) in (ast.Load, ast.Del)
            for statement in caller.body[caller.body.index(assignment) + 1:] for node in ast.walk(statement)):
        raise ValueError('fixed-unpack item is consumed after the selected original call')

    original_function = call.func
    call.func = ast.copy_location(ast.Name(id=HELPER, ctx=ast.Load()), original_function)
    call.args.insert(0, original_function)
    producer.value = ast.copy_location(ast.Name(id=rows, ctx=ast.Load()), producer.value)
    revised = compile(ast.fix_missing_locations(tree), filename, 'exec', dont_inherit=True, optimize=optimize)
    revised_caller, specialized_callee = (_only((code for code in revised.co_consts
        if type(code) is CodeType and code.co_qualname == name), 'missing lowered function code')
        for name in (caller.name, callee.name))
    facts = dict(caller_name=caller.name, callee_name=callee.name, caller_code_key=caller_key,
        callee_code_key=callee_key, revised_caller_code_key=code_key(revised_caller),
        specialized_callee_code_key=code_key(specialized_callee), call_span=list(call_span),
        producer_span=list(producer_span), helper=HELPER, return_slot=slot, matrix_argument=formals.index(matrix),
        rows_argument=formals.index(rows), dead_argument=formals.index(dead),
        numpy_binding=numpy, ndarray_binding=array, flatten_binding=flatten, flatten_code_key=flatten_key)
    return facts, revised_caller, specialized_callee
