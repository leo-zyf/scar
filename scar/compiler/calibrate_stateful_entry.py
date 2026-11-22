"""Matched complete canonical entry costs for one retained implementation.

The existing ELF is a macro cost primitive.  Its real guard/adapter/packing,
native body, allocation and cleanup execute together, with no internal clocks
or sums of isolated medians.  Canonical transfer remains a hypothesis until a
separate, subsequently frozen real-entry holdout validates it.
"""
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
from hashlib import sha256
import argparse
import ast
import dis
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import types
from types import CodeType, SimpleNamespace
import uuid

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.resource_solver import semantic_sha256

DESIGN = 'scar.stateful-canonical-entry-design'
OBSERVATIONS = 'scar.stateful-canonical-entry-calibration'
SCHEMA = 'scar.stateful-canonical-entry-model'
FACTS = 'scar.stateful-canonical-entry-facts'
BASIS = 'MATCHED_CANONICAL_GUARDED_ENTRY_POINT_MODEL'
_ROUNDING = 'NEAREST_INTEGER_NANOSECOND_TIES_TO_EVEN'


def _need(condition, message):
    if not condition:
        raise ValueError(message)


_MAX_SOURCE_BYTES = 8*2**20
_MAX_CODE_OBJECTS = 10000


def _literal(node):
    """Recover immutable defaults without evaluating their defining scope."""
    value = ast.literal_eval(node)
    def closed(item, depth=0):
        return (depth <= 32 and (item is None or type(item) in (bool, int, str, bytes)
            or type(item) is float and math.isfinite(item)
            or type(item) is tuple and all(closed(v, depth+1) for v in item)))
    _need(closed(value), 'default is not a closed immutable literal')
    return value


def _bind_canonical_inputs(contract, *, array_value=1.25,
        scalar_value=0.75, max_external_bytes=64*2**20):
    """Return original/args/kwargs/arrays/scalars/reset/owner/metadata.

    The supported source profile has direct ordinary-receiver staticmethod
    callees and global NumPy power providers. Unknown globals, closures,
    decorators and dynamic defaults are rejected, never treated as absent.
    No application module, constructor, method or callee body is executed.
    """
    import numpy as np
    from scar.runtime.native_stateful import _process
    from scar.runtime.stateful_contract import canonical_sha256, validate_contract
    from scar.runtime.stateful_guard import _field, _function, _input, _lookup, _ordinary
    from scar.trace.code_snapshot import code_key

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
    metadata = dict(schema='scar.canonical-stateful-input-binding', schema_version=1,
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



def _ref(value, schema=None):
    if type(value) is dict:
        return deepcopy(value)
    path = Path(value).absolute()
    return artifact_reference(path, schema or json.loads(path.read_bytes())['schema'])


def _same(left, right):
    return semantic_sha256(left) == semantic_sha256(right)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write(path, document):
    with path.open('x') as stream:
        json.dump(document, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return artifact_reference(path, document['schema'])


def _pins():
    root = Path(__file__).resolve().parents[1]
    names = ('compiler/calibrate_stateful_entry.py',
        'analysis/stateful_entry_protocol.py', 'compiler/stateful_runtime.py',
        'compiler/selected_stateful_lowering.py',
        'compiler/stateful_model.py', 'optimization/stateful_cpu_model.py',
        'runtime/stateful_guard.py', 'runtime/stateful_contract.py',
        'trace/code_snapshot.py', 'codegen/stateful.py', 'codegen/cpp.py')
    return {name:artifact_reference(root/name, 'scar.implementation-source') for name in names}


def _materials(specialization):
    from scar.compiler.selected_stateful_lowering import derive_stateful_lowering_view
    from scar.compiler.stateful_model import stateful_from_dict
    from scar.compiler.stateful_runtime import project_stateful_contract
    from scar.optimization.stateful_cpu_model import _domain, stateful_cpu_cost_context

    spec = read_artifact(specialization)
    _need((spec.get('schema'), spec.get('schema_version')) ==
        ('scar.observed-stateful-specialization', 1), 'original observed specialization required')
    candidate = read_artifact(spec['candidate'])
    domain = _domain(read_artifact(spec['domain']))
    row = next((row for row in domain['candidates'] if row['id'] == candidate['candidate_id']), None)
    _need(row is not None and row['legal_under_guards'] and row['worker_count'] == 1
        and candidate['program'] == spec['program'], 'one admitted serial source candidate required')
    native = candidate['native_artifact']
    view = derive_stateful_lowering_view(native)
    region = stateful_from_dict(read_artifact(view['region']))
    plan = read_artifact(view['plan']); metadata = read_artifact(native)
    _need(plan['region_sha256'] == domain['region_sha256']
        and view['actual_resource_binding'] == row['binding'],
        'native frame and original domain candidate differ')
    artifact = SimpleNamespace(region=region, plan=plan, report=metadata,
        directory=Path(native['uri']).parent)
    contract = project_stateful_contract(artifact, None, require_selection=False)
    return spec, candidate, domain, view, contract, stateful_cpu_cost_context(domain, row['id'])


def _derive(request):
    from scar.analysis.stateful_entry_protocol import derive_stateful_entry_protocol
    from scar.compiler.stateful_holdout import TIMER
    from scar.runtime.compiled_deployment import thread_environment

    _need(set(request) == {'specialization', 'warmup', 'repetitions', 'array_value',
        'scalar_value', 'max_external_bytes', 'capture_epoch'}, 'complete canonical request required')
    _need(type(request['warmup']) is int and request['warmup'] == 2
        and type(request['repetitions']) is int and request['repetitions'] == 7,
        'two warmups and seven fixed measurement calls required')
    _need(type(request['max_external_bytes']) is int and request['max_external_bytes'] > 0,
        'positive canonical input byte budget required')
    _need(all(type(request[name]) is float and math.isfinite(request[name])
        for name in ('array_value', 'scalar_value')), 'finite explicit canonical floats required')
    _need(type(request['capture_epoch']) is str and len(request['capture_epoch']) == 32
        and all(value in '0123456789abcdef' for value in request['capture_epoch']),
        'prepared canonical capture epoch required')
    spec, candidate, domain, view, contract, context = _materials(request['specialization'])
    protocol = derive_stateful_entry_protocol(candidate['native_artifact'],
        trial=True, exclusive_inputs=True, selection=None)
    bound = _bind_canonical_inputs(contract, **{name:request[name] for name in
        ('array_value', 'scalar_value', 'max_external_bytes')})
    return dict(context=context, domain=spec['domain'], candidate_id=candidate['candidate_id'],
        native_manifest=candidate['native_artifact'], source_native_view=view,
        contract=contract, entry_protocol=protocol, canonical_binding=bound['metadata'],
        implementation_sources=_pins(), timer=deepcopy(TIMER),
        runtime=dict(python_executable=sys.executable, python_version=sys.version,
            caller_affinity_cpu_ids=sorted(os.sched_getaffinity(0)),
            native_thread_environment=thread_environment()),
        conditions=['CANONICAL_FINITE_FLOAT_INPUTS_TRANSFER_TO_ACTUAL_SOURCE_INPUT_REGIME',
            'ONE_PERSISTENT_GUARD_AFTER_TWO_WARMUP_CALLS',
            'SAME_ENTRY_PROTOCOL_ELF_CPU_SHAPE_LAYOUT_SCALAR_TAG_AND_SOURCE',
            'RESET_CANONICAL_ARRAYS_OUTSIDE_TIMER_NO_PER_STEP_CONTENT_RECORDING',
            'CACHE_NUMA_ALLOCATOR_PREDECESSOR_DATA_AND_CONTENTION_NOT_PHYSICAL_GUARANTEES',
            'INDEPENDENT_REAL_ENTRY_HOLDOUT_REQUIRED_BEFORE_TIMING_USE'],
        original_f4_source_admission_replayed=False,
        original_f4_replay_required_by_enclosing_source_controller=True,
        preparation_target_executed=False, compiled=False, native_loaded=False,
        physical_feasibility_proven=False, deployment_authorized=False)


def prepare_canonical_entry_calibration(specialization, out, *, array_value=1.25,
        scalar_value=0.75, max_external_bytes=64*2**20):
    request = dict(specialization=_ref(specialization), warmup=2, repetitions=7,
        array_value=array_value, scalar_value=scalar_value,
        max_external_bytes=max_external_bytes, capture_epoch=uuid.uuid4().hex)
    derived = _derive(request)
    directory = Path(out).absolute(); directory.mkdir(parents=True, exist_ok=False)
    document = dict(schema=DESIGN, schema_version=1, created_at=_now(),
        request=request, **derived)
    return _write(directory/'design.json', document)


def read_canonical_entry_design(path, *, expected_sha256=None):
    reference = _ref(path, DESIGN); document = read_artifact(reference)
    _need(expected_sha256 is None or reference['sha256'] == expected_sha256,
        'frozen canonical entry design identity differs')
    _need((document.get('schema'), document.get('schema_version')) == (DESIGN, 1),
        'versioned canonical entry design required')
    expected = dict(schema=DESIGN, schema_version=1, created_at=document['created_at'],
        request=document['request'], **_derive(document['request']))
    _need(_same(document, expected), 'canonical entry source/context/protocol differs from cold replay')
    return document


def _summary(samples):
    _need(type(samples) is list and len(samples) == 7, 'seven canonical measurement calls required')
    previous = -1; durations = []
    for index, sample in enumerate(samples):
        _need(sample['ordinal'] == index and all(type(sample[name]) is int
            for name in ('ordinal', 'start_ns', 'end_ns', 'elapsed_ns', 'native_calls_after', 'fallback_calls'))
            and previous <= sample['start_ns'] < sample['end_ns']
            and sample['elapsed_ns'] == sample['end_ns']-sample['start_ns']
            and sample['native_calls_after'] == index+3 and sample['fallback_calls'] == 0,
            'original monotone clocks and exact successful-call counters required')
        previous = sample['end_ns']; durations.append(sample['elapsed_ns'])
    rational = dict(numerator=sum(durations), denominator=len(durations))
    return dict(exact_mean_ns=rational, estimate_ns=round(Fraction(**rational)),
        integer_rounding=_ROUNDING, min_ns=min(durations), max_ns=max(durations),
        samples_ns=durations, sample_range_is_guaranteed_bound=False)


def run_canonical_entry_calibration(path, *, expected_sha256):
    import numpy as np
    from scar.codegen.stateful import load
    from scar.compiler.stateful_runtime import _ArtifactAdapter
    from scar.runtime.stateful_guard import guarded_stateful_function
    from scar.runtime.compiled_deployment import thread_environment
    from scar.trace.code_snapshot import code_key

    reference = _ref(path, DESIGN)
    design = read_canonical_entry_design(reference, expected_sha256=expected_sha256)
    directory = Path(reference['uri']).parent/'run'; directory.mkdir(exist_ok=False)
    request = design['request']; bound = _bind_canonical_inputs(design['contract'],
        **{name:request[name] for name in ('array_value', 'scalar_value', 'max_external_bytes')})
    _need(_same(bound['metadata'], design['canonical_binding']), 'canonical binding changed')
    artifact = load(design['native_manifest']['uri'])
    adapter = _ArtifactAdapter(artifact, None)
    _need(_same(adapter.contract, design['contract']), 'loaded complete entry contract differs')

    def fatal(reason):
        raise RuntimeError('canonical source fallback is forbidden: '+reason)

    wrapper = guarded_stateful_function(bound['original'], adapter, owner=bound['owner'],
        exclusive_inputs=True, selection=None, trial=True, on_fallback=fatal)
    closure = inspect.getclosurevars(wrapper).nonlocals
    _need(closure['artifact'] is adapter and closure['original'] is bound['original']
        and closure['owner'] is bound['owner'] and closure['trial'] is True
        and closure['exclusive_inputs'] is True and closure['selection'] is None,
        'actual persistent guard closure differs from the frozen successful entry protocol')
    shared_call = next(code for code in guarded_stateful_function.__code__.co_consts
        if type(code) is CodeType and code.co_name == 'call')
    _need(code_key(wrapper.__code__) == code_key(shared_call), 'actual successful guard code differs')
    before = sorted(os.sched_getaffinity(0)); environment = thread_environment()
    oracle_arrays = {name:value.copy() for name,value in bound['arrays'].items()}
    artifact.run(oracle_arrays, dict(bound['scalars']))
    samples = []; observed_from = _now()
    for ordinal in range(request['warmup']+request['repetitions']):
        bound['reset']()
        start = time.perf_counter_ns()
        result = wrapper(*bound['args'], **bound['kwargs'])
        end = time.perf_counter_ns()
        _need(result is None and wrapper.scar_stats['native_calls'] == ordinal+1
            and wrapper.scar_stats['fallback_calls'] == 0, 'complete canonical native call failed')
        _need(sorted(os.sched_getaffinity(0)) == before and thread_environment() == environment,
            'native call changed caller affinity or native thread environment')
        if ordinal >= request['warmup']:
            samples.append(dict(ordinal=ordinal-request['warmup'], start_ns=start, end_ns=end,
                elapsed_ns=end-start, native_calls_after=wrapper.scar_stats['native_calls'], fallback_calls=0))
    _need(all(np.array_equal(value, oracle_arrays[name]) for name,value in bound['arrays'].items()),
        'canonical full guard path differs from direct ABI oracle with independent buffers')
    read_canonical_entry_design(reference, expected_sha256=expected_sha256)
    document = dict(schema=OBSERVATIONS, schema_version=1, design=reference,
        capture_epoch=request['capture_epoch'], observation_origin='CANONICAL_NATIVE_EXECUTION',
        observed_from=observed_from, observed_until=_now(), samples=samples, statistics=_summary(samples),
        entry_protocol=design['entry_protocol'], canonical_binding=design['canonical_binding'],
        timer=design['timer'], runtime=design['runtime'], native_manifest=design['native_manifest'],
        guard_code_key=code_key(wrapper.__code__), native_calls=wrapper.scar_stats['native_calls'],
        fallback_calls=0, independent_direct_native_oracle_calls=1, oracle_outputs_bitwise_equal=True,
        caller_affinity_before=before, caller_affinity_after=sorted(os.sched_getaffinity(0)),
        native_thread_environment=environment, original_python_body_executed=False,
        target_entry_program_executed=False, target_region_dag_executed=True,
        compiled=False, original_elf_reused=True, internal_observer_clocks_inserted=False,
        per_step_array_contents_recorded=False, clock_overhead_subtracted=False,
        variants_ranked=False, holdout_samples_fitted=False, qualified_profile=None,
        deployment_authorized=False)
    return _write(directory/'observations.json', document)


def _calibration(reference):
    from scar.runtime.stateful_guard import guarded_stateful_function
    from scar.trace.code_snapshot import code_key
    observed = read_artifact(reference)
    _need((observed.get('schema'), observed.get('schema_version')) == (OBSERVATIONS, 1),
        'complete canonical entry observations required')
    design = read_canonical_entry_design(observed['design'])
    _need(observed['observation_origin'] == 'CANONICAL_NATIVE_EXECUTION'
        and observed['capture_epoch'] == design['request']['capture_epoch']
        and _same(observed['statistics'], _summary(observed['samples']))
        and type(observed['native_calls']) is int and observed['native_calls'] == 9
        and type(observed['fallback_calls']) is int and observed['fallback_calls'] == 0
        and observed['oracle_outputs_bitwise_equal'] is True
        and type(observed['independent_direct_native_oracle_calls']) is int
        and observed['independent_direct_native_oracle_calls'] == 1,
        'canonical clocks, frozen epoch or complete native-call validation differs')
    shared_call = next(code for code in guarded_stateful_function.__code__.co_consts
        if type(code) is CodeType and code.co_name == 'call')
    _need(observed['guard_code_key'] == code_key(shared_call),
        'observed persistent successful guard code differs from the implementation')
    for key in ('entry_protocol', 'canonical_binding', 'timer', 'runtime', 'native_manifest'):
        _need(_same(observed[key], design[key]), 'canonical observation differs from design: '+key)
    _need(observed['caller_affinity_before'] == observed['caller_affinity_after'] ==
        design['runtime']['caller_affinity_cpu_ids']
        and observed['native_thread_environment'] == design['runtime']['native_thread_environment'],
        'canonical observed caller context differs')
    for key in ('compiled', 'original_python_body_executed', 'target_entry_program_executed',
        'internal_observer_clocks_inserted', 'per_step_array_contents_recorded',
        'clock_overhead_subtracted', 'variants_ranked', 'holdout_samples_fitted', 'deployment_authorized'):
        _need(observed[key] is False, 'canonical entry must not claim '+key)
    _need(observed['target_region_dag_executed'] is True and observed['original_elf_reused'] is True
        and observed['qualified_profile'] is None, 'explicit native calibration without qualification required')
    return observed, design


def _facts(reference):
    observed, design = _calibration(reference)
    return dict(schema=FACTS, schema_version=1, context=deepcopy(design['context']),
        estimate=dict(duration_ticks=observed['statistics']['estimate_ns'], tick_seconds=1e-9),
        derivation=dict(origin='DECLARED_FORWARD_MODEL', basis=BASIS, target_variant_trials=False,
            entry_protocol=deepcopy(design['entry_protocol']), calibration=reference,
            assumptions=deepcopy(design['conditions']), canonical_complete_entry_measured=True,
            one_existing_implementation_measured_on_canonical_inputs=True,
            target_region_dag_executed=True, target_entry_program_executed=False,
            implementation_search_performed=False,
            target_entry_holdout_used_to_fit=False, independent_holdout_required=True,
            component_medians_summed=False, whole_target_latency_guaranteed=False))


def _profile(facts, reference):
    fact = lambda locator:dict(artifact=reference, locator=locator)
    return dict(schema='scar.stateful-cpu-cost-profile', schema_version=1,
        context=deepcopy(facts['context']), context_evidence=[fact('#/context')],
        estimate={**deepcopy(facts['estimate']), 'evidence':[fact('#/estimate')]},
        derivation={**deepcopy(facts['derivation']), 'evidence':[fact('#/derivation')]})


def verify_canonical_entry_profile(profile):
    _need(profile.get('derivation', {}).get('basis') == BASIS, 'matched canonical entry basis required')
    reference = profile['derivation']['calibration']; facts = _facts(reference)
    fact_ref = profile['context_evidence'][0]['artifact']
    _need(_same(read_artifact(fact_ref), facts) and _same(profile, _profile(facts, fact_ref)),
        'canonical entry profile differs from complete design and raw clock replay')
    return facts


def _model(calibration, facts_ref, profile_ref, facts):
    observed, design = _calibration(calibration)
    return dict(schema=SCHEMA, schema_version=1, specialization=design['request']['specialization'],
        domain=design['domain'], candidate_id=design['candidate_id'], calibration=calibration,
        facts=facts_ref, profile=profile_ref, duration_ns=facts['estimate']['duration_ticks'],
        entry_protocol=deepcopy(design['entry_protocol']),
        estimate_origin='MATCHED_CANONICAL_COMPLETE_ENTRY_COMPONENT',
        independent_real_entry_qualified=False, qualified_profile=None,
        automatic_deployment_authorized=False, default_backend='NO_OP')


def prepare_canonical_entry_model(observations, out):
    calibration = _ref(observations, OBSERVATIONS); facts = _facts(calibration)
    directory = Path(out).absolute(); directory.mkdir(parents=True, exist_ok=False)
    facts_ref = _write(directory/'facts.json', facts)
    profile_ref = _write(directory/'profile.json', _profile(facts, facts_ref))
    result = _model(calibration, facts_ref, profile_ref, facts)
    _write(directory/'result.json', result)
    return result


def read_canonical_entry_model(path):
    reference = _ref(path, SCHEMA); model = read_artifact(reference)
    _need((model.get('schema'), model.get('schema_version')) == (SCHEMA, 1), 'canonical entry model required')
    facts = _facts(model['calibration'])
    _need(_same(read_artifact(model['facts']), facts)
        and _same(read_artifact(model['profile']), _profile(facts, model['facts']))
        and _same(model, _model(model['calibration'], model['facts'], model['profile'], facts)),
        'complete canonical model differs from source/protocol/raw clock replay')
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--specialization'); source.add_argument('--run-design')
    source.add_argument('--model-observations'); source.add_argument('--check')
    parser.add_argument('--sha256'); parser.add_argument('--out')
    args = parser.parse_args(argv)
    if args.specialization:
        if not args.out:parser.error('--out required')
        result = prepare_canonical_entry_calibration(args.specialization, args.out)
    elif args.run_design:
        if not args.sha256 or args.out:parser.error('--sha256 required; design owns its output directory')
        result = run_canonical_entry_calibration(args.run_design, expected_sha256=args.sha256)
    elif args.model_observations:
        if not args.out:parser.error('--out required')
        result = prepare_canonical_entry_model(args.model_observations, args.out)
        result = dict(model=_ref(Path(args.out)/'result.json', SCHEMA),
            duration_ns=result['duration_ns'], qualified_profile=None)
    else:
        if args.out:parser.error('--check is read-only')
        model = read_canonical_entry_model(args.check)
        result = dict(model=_ref(args.check, SCHEMA), duration_ns=model['duration_ns'],
            cold_replayed=True, qualified_profile=None)
    print(json.dumps(result, sort_keys=True)); return 0


if __name__ == '__main__':
    raise SystemExit(main())
