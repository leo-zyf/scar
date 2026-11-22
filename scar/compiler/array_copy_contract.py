"""Array copy contract."""

from types import MappingProxyType
import ast
import dis
import inspect
import math
import types
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from scar.runtime.stateful_contract import canonical_sha256, validate_contract
from scar.runtime.stateful_guard import _field, _function, _input, _lookup, _ordinary
from scar.trace.code_snapshot import code_key


# Normal-completion contract for one exact ndarray C-order copy.
#
# The copied contents are equal at this operation under source stability. The
# result is a fresh independent materialization, never a view or an assertion
# of equal runtime allocation, object, invocation or logical-version identity.
COPY_CONTRACT='numpy.ndarray.copy.content-fresh.v1'
COPY_ATTRS=MappingProxyType({'order':'C','contract':COPY_CONTRACT})
COPY_REQUIRED_GUARDS=(
    'EXACT_NUMPY_NDARRAY_COPY_METHOD_DESCRIPTOR',
    'ARRAY_INPUT_CONTENT_STABLE_DURING_REGION',
)


def copy_attributes():return dict(COPY_ATTRS)


def validate_copy_attributes(attrs):
    """Require the canonical static contract; return a detached attribute map.

    This is format validation, not discharge of the required runtime/profile
    guards or permission to remove an allocation or expose its identity.
    """
    if type(attrs)is not dict or attrs!=COPY_ATTRS:
        raise ValueError('copy requires exact ndarray C-order content/fresh-materialization attributes')
    return copy_attributes()


def copy_semantics():
    return dict(contract=COPY_CONTRACT,relation='copy',equivalence='EXACT_CONTENT_AT_SELECTED_COPY',
        source_content='READ_UNDER_SOURCE_STABILITY',result_content='SAME_CONTENT_ON_NORMAL_COMPLETION',
        materialization='FRESH_INDEPENDENT_C_ORDER_ALLOCATION',aliases_source=False,
        writes='NEW_OUTPUT_ONLY',allocates='FRESH_OUTPUT',may_raise='NUMPY_OR_ALLOCATION_FAILURE',
        equivalence_scope='NORMAL_COMPLETION_NOT_FULL_EXCEPTION_OR_INTERNAL_ALLOCATION_SEQUENCE',
        required_guards=list(COPY_REQUIRED_GUARDS),runtime_value_or_storage_identity_proven=False,
        transformation_authorized=False)


# Bind canonical objects to retained stateful Python CodeIDs without exec.
#
# Only Python compilation and object construction occur here.  The returned
# original function and callees must never be invoked by a calibration: its
# guarded native caller must make fallback fatal.  These are synthetic ordinary
# receivers, not reconstructed application initialization or source admission.
SCHEMA = 'scar.canonical-stateful-input-binding'
_MAX_SOURCE_BYTES = 8*2**20
_MAX_CODE_OBJECTS = 10000


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _literal(node):
    """Recover immutable defaults without evaluating their defining scope."""
    value = ast.literal_eval(node)
    def closed(item, depth=0):
        return (depth <= 32 and (item is None or type(item) in (bool, int, str, bytes)
            or type(item) is float and math.isfinite(item)
            or type(item) is tuple and all(closed(v, depth+1) for v in item)))
    _need(closed(value), 'default is not a closed immutable literal')
    return value


def bind_canonical_stateful_inputs(contract, *, array_value=1.25,
        scalar_value=0.75, max_external_bytes=64*2**20):
    """Return original/args/kwargs/arrays/scalars/reset/owner/metadata.

    The supported source profile has direct ordinary-receiver staticmethod
    callees and global NumPy power providers. Unknown globals, closures,
    decorators and dynamic defaults are rejected, never treated as absent.
    No application module, constructor, method or callee body is executed.
    """
    import numpy as np
    from scar.runtime.native_stateful import _process

    checked = deepcopy(validate_contract(contract, require_selection=False))
    _process(checked)  # Existing local ABI/available-CPU guard; no mask change.
    _need(type(max_external_bytes) is int and max_external_bytes >= 0,
        'nonnegative exact external payload limit required')
    def numeric(value):
        return (type(value) in (int, float, np.float64)
            and (type(value) is not int or abs(value) <= 2**53) and math.isfinite(value))
    _need(numeric(array_value) and numeric(scalar_value),
        'finite canonical values with exact supported scalar tags required')
    array_value = float(array_value)
    scalar_value = scalar_value if type(scalar_value) is int else float(scalar_value)
    external = sum(row['nbytes'] for row in checked['inputs'] if row['kind'] == 'array')
    _need(external <= max_external_bytes, 'canonical external arrays exceed payload limit')
    _need(all(len(row['path']) == 2 and row['path'][0] in checked['parameters']
        for row in checked['callee_guards']),
        'only admitted direct receiver staticmethod paths are supported')
    _need(all(len(row['lookup_path']) == 2 and row['lookup_path'][0] not in checked['parameters']
        and row['lookup_path'][1] == 'power' for row in checked['providers']),
        'only admitted global NumPy power provider paths are supported')

    units = {}; functions = {}; function_metadata = []
    sources = [checked['source'], *(row['source'] for row in checked['callee_guards'])]
    _need(all((source['path'], source['sha256']) ==
        (checked['source']['path'], checked['source']['sha256']) for source in sources),
        'admitted stateful callees must share the original source snapshot')
    read_bytes = 0
    for source in sources:
        path = source['path']
        if path in units:
            continue
        file = Path(path)
        _need(file.is_file() and file.stat().st_size <= _MAX_SOURCE_BYTES-read_bytes,
            'bounded original Python source required')
        raw = file.read_bytes(); read_bytes += len(raw)
        _need(read_bytes <= _MAX_SOURCE_BYTES and sha256(raw).hexdigest() == source['sha256'],
            'canonical source differs from retained SHA or exceeds budget')
        tree = compile(raw, path, 'exec', flags=ast.PyCF_ONLY_AST | ast.PyCF_TYPE_COMMENTS,
            dont_inherit=True, optimize=0)
        module_code = compile(raw, path, 'exec', flags=0, dont_inherit=True, optimize=0)
        pending = [module_code]; codes = []
        while pending:
            code = pending.pop(); codes.append(code)
            _need(len(codes) <= _MAX_CODE_OBJECTS, 'source code-object budget exceeded')
            pending.extend(v for v in code.co_consts if type(v) is types.CodeType)
        namespace = dict(__name__='scar_canonical_stateful_'+source['sha256'][:16],
            __builtins__={})
        units[path] = dict(codes=codes, tree=tree, namespace=namespace,
            source_ref=dict(uri=path, sha256=source['sha256'], size_bytes=len(raw), schema='scar.python-source'))

    def bind_function(source, *, static=False):
        key = (source['path'], source['code_key'], source['qualname'])
        if key in functions:
            _need(functions[key][1] == static, 'one source function has conflicting descriptor roles')
            return functions[key][0]
        unit = units[source['path']]
        matches = [code for code in unit['codes'] if code.co_qualname == source['qualname']
            and code_key(code) == source['code_key']]
        _need(len(matches) == 1, 'original function has no unique compile-only CodeID match')
        code, = matches
        _need(not code.co_freevars and '<locals>' not in code.co_qualname
            and not code.co_flags & (inspect.CO_GENERATOR | inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR)
            and not any(type(v) is types.CodeType for v in code.co_consts),
            'closures, deferred execution and nested function bodies are outside this binding')
        nodes = [node for node in ast.walk(unit['tree']) if isinstance(node, ast.FunctionDef)
            and node.name == code.co_name and min([node.lineno,
                *(d.lineno for d in node.decorator_list)]) == code.co_firstlineno]
        _need(len(nodes) == 1, 'unique original function definition required')
        node, = nodes
        decorators = node.decorator_list
        is_static = len(decorators) == 1 and type(decorators[0]) is ast.Name and decorators[0].id == 'staticmethod'
        _need(is_static if static else not decorators or is_static,
            'unclosed function decorator or non-staticmethod callee')
        args = node.args
        _need(not args.vararg and not args.kwarg
            and not node.returns and not any(arg.annotation for arg in
                [*args.posonlyargs, *args.args, *args.kwonlyargs])
            and not getattr(node, 'type_params', ()),
            'variadic ABI, annotations and generic definitions need separate binding evidence')
        defaults = tuple(_literal(value) for value in args.defaults)
        kwdefaults = {arg.arg:_literal(value) for arg,value in zip(args.kwonlyargs, args.kw_defaults)
            if value is not None}
        fn = types.FunctionType(code, unit['namespace'], argdefs=defaults or None)
        fn.__kwdefaults__ = kwdefaults or None
        _function(source, fn)
        functions[key] = (fn, static)
        function_metadata.append(dict(source=deepcopy(source), source_ref=deepcopy(unit['source_ref']),
            code_key=code_key(code), qualname=code.co_qualname,
            descriptor='STATICMETHOD' if is_static else 'ORDINARY_FUNCTION',
            defaults='CLOSED_IMMUTABLE_LITERALS' if defaults or kwdefaults else 'NONE',
            closure='NONE', body_executed=False))
        return fn

    original = bind_function(checked['source'])
    signature = inspect.signature(original, follow_wrapped=False)
    _need(tuple(signature.parameters) == tuple(checked['parameters']),
        'canonical original signature differs from admitted parameter order')
    nodes = []
    def ordinary_node():
        cls = type('CanonicalStatefulNode'+str(len(nodes)), (), {})
        value = object.__new__(cls); _ordinary(value); nodes.append(value)
        return value
    bound = {name:ordinary_node() for name in checked['parameters']}
    arrays = {}; scalars = {}; topology = []
    # Build the object tree before inserting values. Prefix/leaf conflicts
    # cannot silently replace a receiver, an array or a scalar with another.
    leaves = {tuple(row['path']) for row in checked['inputs']}
    _need(all(not any(path[:i] in leaves for i in range(1, len(path))) for path in leaves),
        'canonical input paths contain conflicting leaf prefixes')
    for spec in checked['inputs']:
        if spec['kind'] == 'array':
            value = np.full(spec['shape'], array_value, dtype=np.float64)
            arrays[spec['id']] = value
        else:
            value = scalar_value; scalars[spec['id']] = value
        parts = spec['path']
        if len(parts) == 1:
            bound[parts[0]] = value
        else:
            current = bound[parts[0]]
            for name in parts[1:-1]:
                fields = _ordinary(current)
                if name not in fields:
                    setattr(current, name, ordinary_node())
                current = _field(current, name)
            _need(parts[-1] not in _ordinary(current), 'canonical field path conflict')
            setattr(current, parts[-1], value)
        topology.append(dict(id=spec['id'], path=list(parts), kind=spec['kind'],
            construction='FRESH_INDEPENDENT_NDARRAY' if spec['kind']=='array' else 'CANONICAL_FINITE_SCALAR'))

    callees = []
    for guard in checked['callee_guards']:
        fn = bind_function(guard['source'], static=True)
        root, name = guard['path']; receiver = bound[root]
        _need(name not in _ordinary(receiver) and name not in vars(type(receiver)),
            'callee path conflicts with canonical fields or descriptors')
        setattr(type(receiver), name, staticmethod(fn))
        _need(_lookup(guard['path'], bound, original.__globals__) is fn,
            'canonical callee path does not preserve the runtime lookup rule')
        _function(guard['source'], fn)
        callees.append(dict(path=list(guard['path']), source=deepcopy(guard['source']),
            source_ref=deepcopy(units[guard['source']['path']]['source_ref'])))
    providers = []; provider_binary = None
    if checked['providers']:
        # Reuse the already imported original NumPy object; do not resolve a
        # native loop/capsule or load any SCAR ELF in this input-only helper.
        module = vars(np._core).get('_multiarray_umath')
        _need(type(module) is types.ModuleType and np.power is vars(module).get('power'),
            'NumPy power is not the original loaded provider ufunc')
        binary = Path(module.__file__).resolve(); before = binary.stat()
        raw = binary.read_bytes(); after = binary.stat()
        stamp = lambda s:(s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        provider_binary = dict(path=str(binary), sha256=sha256(raw).hexdigest(), bytes=len(raw))
        _need(stamp(before) == stamp(after) and
            provider_binary == checked['provider_runtime']['loaded_binary'],
            'loaded NumPy provider binary differs from the admitted runtime')
    for provider in checked['providers']:
        root = provider['lookup_path'][0]
        _need(type(np.power) is np.ufunc and np.__version__ == checked['provider_runtime']['numpy_version'],
            'exact original NumPy power provider/version differs')
        for unit in units.values():
            existing = unit['namespace'].get(root, np)
            _need(existing is np, 'provider root conflicts with a canonical namespace')
            unit['namespace'][root] = np
        _need(_lookup(provider['lookup_path'], bound, original.__globals__) is np.power,
            'canonical provider path differs from the original NumPy power ufunc')
        providers.append(dict(id=provider['id'], lookup_path=list(provider['lookup_path']),
            module='numpy', symbol='power', numpy_version=np.__version__, exact_ufunc_identity=True))

    # Namespace closure is checked from bytecode, not from co_names (which
    # also includes field labels). Never populate arbitrary globals/builtins.
    for fn, _ in functions.values():
        instructions = tuple(dis.get_instructions(fn.__code__, adaptive=False))
        _need(not any(i.opname in ('STORE_GLOBAL', 'DELETE_GLOBAL', 'IMPORT_NAME', 'IMPORT_FROM')
            for i in instructions), 'global mutation or dynamic imports are outside the admitted binding')
        names = {i.argval for i in instructions if i.opname in ('LOAD_GLOBAL', 'LOAD_NAME')}
        _need(names <= set(fn.__globals__) - {'__builtins__', '__name__'},
            'unclosed original function global bindings: '+','.join(sorted(names-set(fn.__globals__))))

    positional = []; keywords = {}
    for name, parameter in signature.parameters.items():
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            keywords[name] = bound[name]
        else:
            positional.append(bound[name])
    actual = signature.bind(*positional, **keywords)
    _need(dict(actual.arguments) == bound, 'canonical positional/keyword call ABI differs')
    for spec in checked['inputs']:
        _need(_input(spec['path'], actual.arguments) is
            (arrays if spec['kind']=='array' else scalars)[spec['id']],
            'canonical input resolver differs from the runtime path')
    values = list(arrays.values())
    _need(all(not np.may_share_memory(value, other) for index,value in enumerate(values)
        for other in values[:index]), 'canonical external arrays must have independent storage')
    for source in sources:
        _function(source, functions[(source['path'], source['code_key'], source['qualname'])][0])

    def reset():
        """Refill the same buffers; never call an application function."""
        for spec in checked['inputs']:
            value = (arrays if spec['kind']=='array' else scalars)[spec['id']]
            _need(_input(spec['path'], bound) is value,
                'canonical path identity changed before reset')
            if spec['kind'] == 'array':
                value.fill(array_value)

    scalar_tag = 'EXACT_PYTHON_INT' if type(scalar_value) is int else 'EXACT_PYTHON_FLOAT'
    metadata = dict(schema=SCHEMA, schema_version=1,
        runtime_contract_sha256=canonical_sha256(checked), native=deepcopy(checked['native']['ids']),
        source_refs=[deepcopy(unit['source_ref']) for unit in units.values()],
        functions=function_metadata, path_topology=topology, callee_bindings=callees,
        provider_bindings=providers, provider_binary=provider_binary,
        compile_recipe=dict(mode='exec', flags=0, dont_inherit=True, optimize=0,
            filenames=[source['path'] for source in sources]),
        arrays={key:dict(shape=list(value.shape), dtype=value.dtype.str, strides=list(value.strides),
            nbytes=value.nbytes, c_contiguous=bool(value.flags.c_contiguous),
            writeable=bool(value.flags.writeable), independent_storage=True) for key,value in arrays.items()},
        scalars={key:dict(tag=scalar_tag, value=value,
            native_integer_flag=type(value) is int) for key,value in scalars.items()},
        canonical_array_value=array_value, max_external_bytes=max_external_bytes,
        external_array_bytes=external, reset='REFILL_SAME_CANONICAL_BUFFERS_OUTSIDE_TIMER',
        receiver='SYNTHETIC_ORDINARY_DICTIONARY_OBJECTS_NO_APPLICATION_CONSTRUCTOR',
        original_code_id_verified=True, target_module_executed=False,
        target_module_imported=False, target_python_body_executed=False,
        native_loaded=False, native_compiled=False, source_admission_performed=False,
        calibration_requires_fatal_on_fallback=True, timing_qualified=False,
        deployment_authorized=False)
    # owner is only the ordinary method receiver, not a physical resource or
    # evidence that an application class constructor was reproduced.
    owner = (type(bound[checked['parameters'][0]]) if '.' in original.__qualname__
        and function_metadata[0]['descriptor'] == 'ORDINARY_FUNCTION' else None)
    return dict(original=original, args=tuple(positional), kwargs=keywords,
        arrays=arrays, scalars=scalars, reset=reset, owner=owner, metadata=metadata)


__all__ = [
    'COPY_CONTRACT',
    'COPY_ATTRS',
    'COPY_REQUIRED_GUARDS',
    'copy_attributes',
    'validate_copy_attributes',
    'copy_semantics',
    'bind_canonical_stateful_inputs',
    'SCHEMA',
]
