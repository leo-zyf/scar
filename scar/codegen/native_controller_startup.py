"""Build a detached startup image by executing the qualified source provider."""

from dataclasses import dataclass
from hashlib import sha256
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import FunctionType

from scar.ir.execution_recipe import artifact_reference, read_artifact


HEADER = 'include/scar_generated/native_controller_startup.hpp'
ASSET = 'assets/startup.bin'
RECEIPT = 'assets/startup-build.json'
BUILD_SCHEMA = 'scar.controller-startup-build'
_TEMPLATE = Path(__file__).with_name('native_controller_startup_v1.hpp.in')
_KINDS = {bool: 'bool', int: 'int', float: 'float64', str: 'str'}


def _need(condition, detail):
    if not condition:
        raise ValueError('native controller startup: ' + detail)


def _digest(data):
    return sha256(data).hexdigest()


def _json(value):
    return (json.dumps(value, allow_nan=False, sort_keys=True, indent=2) + '\n').encode()


def _scalar(value):
    kind = _KINDS.get(type(value))
    _need(kind is not None, 'exact builtin scalar field required')
    if kind == 'float64':
        _need(math.isfinite(value), 'nonfinite original scalar field')
        return dict(kind=kind, value=value, binary64_be=struct.pack('>d', value).hex())
    if kind == 'int':
        _need(-(1 << 63) <= value < 1 << 63, 'original integer exceeds native scalar ABI')
    return dict(kind=kind, value=value)


def _field(instance, path):
    _need(len(path) >= 2, 'original initializer field needs its receiver path')
    for name in path[1:]:
        namespace = object.__getattribute__(instance, '__dict__')
        _need(type(namespace) is dict and name in namespace, 'ordinary original receiver field missing: ' + name)
        instance = namespace[name]
    return instance


def _geometry(array):
    _need(array.flags.owndata and array.flags.c_contiguous, 'original array is not an owned contiguous root')
    return dict(shape=list(array.shape), dtype=array.dtype.str, itemsize=array.itemsize,
                allocation_nbytes=array.nbytes, byte_bounds=[0, array.nbytes], byte_offset=0,
                byte_strides=list(array.strides), writeable=array.flags.writeable)


def _requirements(source_module, rows):
    from scar.trace.code_snapshot import code_key

    qualified = []
    for row in rows:
        name, member = row['namespace']
        module = sys.modules[name]
        provider = getattr(module, member)
        _need(type(provider).__qualname__ == row['current_provider_shape']
              and all(vars(source_module).get(alias) is module for alias in row['guarded_namespace_aliases']),
              'original startup provider binding differs')
        if name == 'builtins':
            _need(vars(source_module).get(member, provider) is provider, 'original builtin is shadowed')
        if row['implementation_source'] is not None:
            implementation = inspect.unwrap(provider)
            reference = row['implementation_source']
            _need(type(implementation) is FunctionType
                  and code_key(implementation.__code__) == row['implementation_code']
                  and artifact_reference(Path(inspect.getsourcefile(implementation)).resolve(), reference['schema'])
                  == reference, 'original build provider implementation differs')
        qualified.append(row['id'])
    return qualified


def _worker(config_path):
    """Run the pinned constructor in the build interpreter, never in delivery."""
    from scar.trace.code_snapshot import code_key

    config_path = Path(config_path)
    request = json.loads(config_path.read_bytes())
    invocation = request['invocation']
    if invocation['cpu_affinity']:
        os.sched_setaffinity(0, invocation['cpu_affinity'])
    source = invocation['source']
    path = Path(source['uri'])
    sys.path.insert(0, str(path.parent))
    specification = importlib.util.spec_from_file_location(invocation['module'], path)
    _need(specification is not None, 'cannot bind original module')
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    exec(compile(read_artifact(source, json_document=False), str(path), 'exec'), module.__dict__)
    constructor = module
    for name in invocation['class_qualname'].split('.'):
        constructor = vars(constructor)[name]
    _need(type(constructor) is type, 'original constructor must retain its exact class')
    initializer = vars(constructor).get('__init__')
    _need(type(initializer) is FunctionType and code_key(initializer.__code__) == invocation['constructor_code'],
          'original constructor CodeID differs')
    qualified = _requirements(module, request['provider_requirements'])

    kwargs = dict(request['kwargs'])
    for name in request['isolated_directories']:
        directory = config_path.parent / 'provider-output'
        _need(not directory.exists(), 'provider output must remain isolated and fresh')
        kwargs[name] = str(directory)
    instance = constructor(**kwargs)
    modules = {}
    for name, expected in invocation['required_modules'].items():
        loaded = sys.modules.get(name)
        if loaded is None:
            modules[name] = dict(loaded=False)
        else:
            actual = Path(vars(loaded)['__file__']).resolve(strict=True)
            ref = artifact_reference(actual, expected['artifact']['schema'])
            modules[name] = dict(loaded=True, version=vars(loaded).get('__version__'), artifact=ref)
    _need(modules == invocation['required_modules'], 'original loaded provider identity differs')

    outputs, array_ranges = [], []
    np = sys.modules['numpy'] if any(item['kind'] == 'ARRAY' for item in request['outputs']) else None
    for index, item in enumerate(request['outputs']):
        value = _field(instance, item['path'])
        if item['kind'] == 'ARRAY':
            _need(type(value) is np.ndarray and _geometry(value) == item['geometry'],
                  'provider array geometry differs: ' + item['id'])
            address = value.__array_interface__['data'][0]
            bounds = address, address + value.nbytes
            _need(not any(identity == id(value) or max(bounds[0], other[0]) < min(bounds[1], other[1])
                          for identity, other in array_ranges), 'provider returned aliased startup array owners')
            array_ranges.append((id(value), bounds))
            payload = value.tobytes(order='C')
            (config_path.parent / f'array-{index}.bin').write_bytes(payload)
            observed = dict(size_bytes=len(payload), sha256=_digest(payload))
        else:
            observed = _scalar(value)
        outputs.append(dict(id=item['id'], observed=observed))
    required_roots = request.get('distinct_array_roots_required',
                                 sum(item['kind'] == 'ARRAY' for item in request['outputs']))
    _need(len(array_ranges) == required_roots, 'startup root count differs from original proof')
    routed = []
    for item in request['runtime_routes']:
        name = item['parameter']
        observed = _scalar(_field(instance, item['path']))
        if name in request['isolated_directories']:
            _need(observed == _scalar(kwargs[name]), 'provider did not receive its isolated destination')
            observed = dict(kind='ISOLATED_OUTPUT_DIRECTORY')
        routed.append(dict(parameter=name, observed=observed))
    report = dict(pid=os.getpid(), source=source, class_qualname=invocation['class_qualname'],
                  constructor_code=invocation['constructor_code'], original_provider_executed=True,
                  interpreter_version=sys.version,
                  modules=modules, qualified_provider_ids=qualified,
                  cpu_affinity=sorted(os.sched_getaffinity(0)), outputs=outputs, runtime_values=routed)
    (config_path.parent / 'worker-result.json').write_bytes(_json(report))


def _cases(invocation, runtime_routes):
    routes = {row['parameter']: row for row in runtime_routes}
    _need(len(routes) == len(runtime_routes), 'duplicate original runtime argument route')
    fixed, isolated, witnesses = {}, [], {}
    for item in invocation['keyword_arguments']:
        name, kind = item['parameter'], item['kind']
        _need(type(name) is str and name.isidentifier() and name not in fixed and name not in isolated
              and name not in witnesses, 'ambiguous original keyword')
        if kind == 'ISOLATED_OUTPUT_DIRECTORY':
            isolated.append(name)
        elif kind == 'LITERAL':
            fixed[name] = item['value']
        elif kind == 'SCALAR_WITNESS':
            values = routes[name]['witness_values']
            _need(type(values) is list and len(values) >= 2 and all(
                row == _scalar(row['value']) and row['kind'] == routes[name]['kind'] for row in values),
                'at least two typed original witnesses required for runtime route ' + name)
            witnesses[name] = values
        else:
            raise ValueError('native controller startup: unsupported original invocation argument')
    _need(set(witnesses) | set(isolated) == set(routes), 'leftover runtime input outside original constructor')
    for index in range(2 if witnesses else 1):
        values = {name: rows[index] for name, rows in witnesses.items()}
        yield dict(kwargs={**fixed, **{name: row['value'] for name, row in values.items()}},
                   isolated_directories=isolated,
                   expected_runtime={**values, **{name: dict(kind='ISOLATED_OUTPUT_DIRECTORY') for name in isolated}})


def _validate_run(report, payload, case, specification):
    invocation = specification['invocation']
    _need(type(report['pid']) is int and report['pid'] > 0
          and report['source'] == invocation['source']
          and report['class_qualname'] == invocation['class_qualname']
          and report['constructor_code'] == invocation['constructor_code']
          and report['interpreter_version'] == invocation['interpreter_version']
          and report['original_provider_executed'] is True
          and report['modules'] == invocation['required_modules']
          and report['qualified_provider_ids'] == [row['id'] for row in specification['provider_requirements']]
          and report['cpu_affinity'] == invocation['cpu_affinity'],
          'provider execution identity or launch context differs')
    expected = [dict(id=item['id'], observed=item['expected']) for item in specification['outputs']]
    _need(report['outputs'] == expected, 'fresh constructor output differs from retained witness')
    routes = specification['runtime_routes']
    _need(report['runtime_values'] == [dict(parameter=row['parameter'], observed=case['expected_runtime'][row['parameter']])
                                      for row in routes], 'constructor did not receive the exact routed runtime values')
    cursor = 0
    for item in specification['outputs']:
        if item['kind'] == 'ARRAY':
            expected = item['expected']
            section = payload[cursor:cursor + expected['size_bytes']]
            _need(len(section) == expected['size_bytes'] and _digest(section) == expected['sha256'],
                  'constructor byte materialization differs: ' + item['id'])
            cursor += len(section)
    _need(cursor == len(payload), 'unexpected startup payload bytes')


def _build(reference, specification, cases):
    invocation = specification['invocation']
    runtime = invocation['interpreter']
    read_artifact(runtime, json_document=False)
    records, payloads = [], []
    with TemporaryDirectory(prefix='scar-original-constructor-') as temporary:
        for index, case in enumerate(cases):
            root = Path(temporary) / f'run-{index}'
            paths = {name: root / name for name in ('HOME', 'TMPDIR', 'XDG_CACHE_HOME', 'MPLCONFIGDIR')}
            for directory in paths.values():
                directory.mkdir(parents=True)
            environment = {key: value for key, value in invocation['environment'].items() if value is not None}
            environment.pop('PYTHONPATH', None)
            environment.update({key: str(path) for key, path in paths.items()})
            inputs = root / 'materialization-request.json'
            inputs.write_bytes(_json(dict(**case, invocation=invocation,
                outputs=[{key: value for key, value in item.items() if key != 'expected'}
                         for item in specification['outputs']], runtime_routes=specification['runtime_routes'],
                provider_requirements=specification['provider_requirements'],
                distinct_array_roots_required=specification.get(
                    'distinct_array_roots_required', sum(item['kind'] == 'ARRAY' for item in specification['outputs'])))))
            bootstrap = ('import sys; sys.path.insert(0, sys.argv[1]); '
                         'from scar.codegen.native_controller_startup import _worker; _worker(sys.argv[2])')
            result = subprocess.run([runtime['uri'], '-B', '-P', '-s', '-c', bootstrap,
                                     str(Path(__file__).resolve().parents[2]), str(inputs)],
                                    cwd=root, env=environment, capture_output=True, timeout=60)
            _need(result.returncode == 0 and result.stdout == result.stderr == b'',
                  'isolated original constructor failed or emitted output: ' + result.stderr[-1000:].decode(errors='replace'))
            _need(not (root / 'provider-output').exists(), 'constructor exercised output-directory side effects')
            report = json.loads((root / 'worker-result.json').read_bytes())
            payload = b''.join((root / f'array-{i}.bin').read_bytes() for i, item in
                               enumerate(specification['outputs']) if item['kind'] == 'ARRAY')
            _validate_run(report, payload, case, specification)
            records.append(report)
            payloads.append(payload)
    _need(all(payload == payloads[0] for payload in payloads), 'startup image changed across routed training values')
    read_artifact(runtime, json_document=False)
    data = payloads[0]
    receipt = dict(schema=BUILD_SCHEMA, schema_version=1, planning=reference, interpreter=runtime,
                   original_constructor_executed=True, captured_values_used_as_producer=False,
                   isolated_build_directory=True, runs=records, sha256=_digest(data), size_bytes=len(data))
    return receipt, data


def _recover(package, reference, specification, cases):
    asset = package / ASSET
    receipt_path = package / RECEIPT
    _need(asset.resolve(strict=True) == asset and receipt_path.resolve(strict=True) == receipt_path,
          'startup assets must use their original local package names')
    receipt = read_artifact(artifact_reference(receipt_path, BUILD_SCHEMA))
    data = read_artifact(artifact_reference(asset, 'scar.startup-binary-image'), json_document=False)
    _need(receipt['schema_version'] == 1 and receipt['planning'] == reference
          and receipt['interpreter'] == specification['invocation']['interpreter']
          and receipt['original_constructor_executed'] is True
          and receipt['captured_values_used_as_producer'] is False
          and receipt['isolated_build_directory'] is True and receipt['sha256'] == _digest(data)
          and receipt['size_bytes'] == len(data), 'startup image or build receipt differs')
    for case, execution in zip(cases, receipt['runs'], strict=True):
        _validate_run(execution, data, case, specification)
    return receipt, data


def _literal(value):
    raw = value.encode('utf-8')
    return '"' + ''.join(f'\\x{byte:02x}' for byte in raw) + '"', len(raw)


def _integer(value):
    return f'std::bit_cast<std::int64_t>(UINT64_C({value % (1 << 64)}))'


def _floating(row):
    return f'std::bit_cast<double>(UINT64_C(0x{row["binary64_be"]}))'


def _scalar_cpp(row):
    if row['kind'] == 'float64':
        return '{' + _floating(row) + ', 0}'
    _need(row['kind'] == 'int' and abs(row['value']) <= 2**53,
          'fixed numerical scalar must fit the original stateful ABI')
    return '{static_cast<double>(' + _integer(row['value']) + '), 1}'


def _slot_cpp(row):
    if row['kind'] == 'float64':
        return '{SCAR_CONTROLLER_FLOAT64_V1, 0, 0, ' + _floating(row) + ', nullptr, 0}'
    if row['kind'] in ('bool', 'int'):
        kind, value = ('BOOL', str(int(row['value']))) if row['kind'] == 'bool' else ('INT64', _integer(row['value']))
        return f'{{SCAR_CONTROLLER_{kind}_V1, 0, {value}, 0, nullptr, 0}}'
    literal, length = _literal(row['value'])
    return f'{{SCAR_CONTROLLER_UTF8_V1, 0, 0, 0, {literal}, {length}}}'


@dataclass(frozen=True, slots=True)
class NativeControllerStartup:
    reference: dict
    specification: dict
    receipt: dict
    source: bytes
    image: bytes
    array_ordinals: tuple[int, ...]
    numeric_ordinals: tuple[int, ...]
    slot_ordinals: tuple[int, ...]
    guard_values: dict[str, bool]
    auxiliary_arrays: tuple[dict, ...]

    def members(self):
        return {HEADER: ('scar.cpp-header', self.source), ASSET: ('scar.startup-binary-image', self.image),
                RECEIPT: (BUILD_SCHEMA, _json(self.receipt))}

    def artifacts(self):
        return dict(planning=self.reference, build_receipt=RECEIPT, image=ASSET,
                    image_sha256=self.receipt['sha256'], image_bytes=self.receipt['size_bytes'],
                    array_ordinals=list(self.array_ordinals), numeric_ordinals=list(self.numeric_ordinals),
                    slot_ordinals=list(self.slot_ordinals), guard_values=self.guard_values,
                    runtime_routes=self.specification['runtime_routes'], actual_provider_runs=len(self.receipt['runs']),
                    auxiliary_arrays=list(self.auxiliary_arrays),
                    mode='ORIGINAL_CONSTRUCTOR_BUILDTIME_MATERIALIZATION',
                    captured_values_used_as_producer=False, python_runtime_required=False,
                    whole_program_translated=False, deployment_authorized=False)


def _lower(specification, ports, controller_plan, receipt, *, borrow_arrays=False):
    arrays = {row['id']: (ordinal, row) for ordinal, row in enumerate(port for port in ports if port['kind'] == 'array')}
    numbers = {row['id']: ordinal for ordinal, row in enumerate(port for port in ports if port['kind'] == 'scalar')}
    slots = {tuple(row['path']): row['ordinal'] for row in controller_plan['slots'] if row['input']}
    selected_arrays, selected_numbers, selected_slots, guards, auxiliary = {}, {}, {}, {}, []
    offset = 0
    for row in specification['outputs']:
        identity, expected, path = row['id'], row['expected'], tuple(row['path'])
        if row['kind'] == 'ARRAY':
            shape = row['geometry']['shape']
            size = expected['size_bytes']
            _need(len(shape) == 2 and all(type(n) is int and n >= 0 for n in shape)
                  and shape[0] * shape[1] * 8 == size and row['geometry']['dtype'] == '<f8',
                  'startup image array differs from numerical ABI')
            if identity in arrays:
                index, port = arrays[identity]
                _need(index not in selected_arrays and port['shape'] == shape, 'startup shape differs from ABI port')
                selected_arrays[index] = f'{{{index}, {offset}, {size}, {shape[0]}, {shape[1]}}}'
            else:
                auxiliary.append(dict(id=identity, path=list(path), offset=offset, size_bytes=size, shape=shape,
                                      bound_to_controller=False))
            offset += size
            continue
        _need(row['kind'] == 'SCALAR' and _scalar(expected['value']) == expected,
              'fixed output is outside the typed startup profile')
        if identity in numbers:
            index = numbers[identity]
            _need(index not in selected_numbers, 'duplicate fixed numerical scalar')
            selected_numbers[index] = f'{{{index}, {_scalar_cpp(expected)}}}'
        if path in slots:
            index = slots[path]
            _need(index not in selected_slots, 'duplicate fixed controller field')
            selected_slots[index] = f'{{{index}, {_slot_cpp(expected)}}}'
        if expected['kind'] == 'bool':
            key = '.'.join(path)
            _need(key not in guards, 'duplicate fixed boolean field')
            guards[key] = expected['value']
    _need(offset == receipt['size_bytes'] and selected_arrays, 'fixed arrays do not account for the entire image')
    runtime = {row['id'] for row in specification['runtime_routes']}
    _need(all(index not in selected_numbers for identity, index in numbers.items() if identity in runtime)
          and all(slots.get(tuple(row['path'])) not in selected_slots for row in specification['runtime_routes']),
          'startup tried to freeze a routed runtime value')
    source = _TEMPLATE.read_text()
    replacements = dict(ARRAY_COUNT=len(selected_arrays), ARRAYS=', '.join(selected_arrays.values()),
                        SCALAR_COUNT=len(selected_numbers), SCALARS=', '.join(selected_numbers.values()),
                        SLOT_COUNT=len(selected_slots), SLOTS=', '.join(selected_slots.values()),
                        IMAGE_SIZE=receipt['size_bytes'], IMAGE_SHA256=receipt['sha256'],
                        BORROW_ARRAY='''    scar_array_v1 borrow_array(std::size_t offset, std::int64_t rows, std::int64_t columns) {
        if (rows <= 0 || columns <= 0 || offset % sizeof(double) || offset > image_.size() ||
            static_cast<std::size_t>(rows) > (image_.size() - offset) / sizeof(double) / columns)
            throw std::runtime_error("borrowed startup array lies outside its original asset");
        const auto stride = columns * sizeof(double);
        const auto bytes = rows * stride;
        auto* data = image_.data() + offset;
        const auto address = reinterpret_cast<std::uintptr_t>(data);
        return {data, 2, {rows, columns}, {static_cast<std::int64_t>(stride), sizeof(double)},
                address, address + bytes};
    }

''' if borrow_arrays else '')
    for key, value in replacements.items():
        source = source.replace(f'@{key}@', str(value))
    return source.encode(), tuple(selected_arrays), tuple(selected_numbers), tuple(selected_slots), guards, tuple(auxiliary)


def prepare(reference, *, controller_reference, controller_plan, ports, package=None, borrow_arrays=False):
    """Materialize once for export or replay a previously published package."""
    from scar.compiler.plan_controller_startup import read_controller_startup_plan

    report = read_controller_startup_plan(reference['uri'])
    _need(report == read_artifact(reference) and report['controller_plan'] == controller_reference
          and report['status'] == 'CONDITIONAL_EXPLICIT_TRIAL_ONLY'
          and report['whole_entry_authorized'] is False and report['deployment_authorized'] is False,
          'startup requires the replayed controller and qualified explicit trial')
    specification = {**report['materialization'], 'runtime_routes': report['runtime_routes']}
    invocation = specification['invocation']
    _need(invocation['working_directory'] == 'ISOLATED_TEMP', 'original provider requires an isolated startup build')
    cases = tuple(_cases(invocation, specification['runtime_routes']))
    if package is None:
        receipt, data = _build(reference, specification, cases)
    else:
        receipt, data = _recover(Path(package).resolve(strict=True), reference, specification, cases)
    _need(len(data) == report['array_resources']['asset_bytes'], 'startup asset extent differs from the plan')
    source, arrays, numeric, slots, guards, auxiliary = _lower(
        specification, ports, controller_plan, receipt, borrow_arrays=borrow_arrays)
    return NativeControllerStartup(reference, specification, receipt, source, data, arrays, numeric, slots, guards, auxiliary)
