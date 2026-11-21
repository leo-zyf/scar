"""Lower proved repeated arithmetic delivery to shared direct C++20 calls."""

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from tempfile import TemporaryDirectory

from scar.codegen.native_cpu_arena_reuse import _linkage, _path
from scar.codegen.native_cpu_fp_guard_groups import SCHEMA as BASELINE, _ROW_END, _ROWS
from scar.codegen.native_project import build_native_project
from scar.codegen.native_seeded_array_task_entry import SCHEMA as RUNTIME
from scar.ir.execution_recipe import artifact_reference, read_artifact


SCHEMA = 'scar.native-cpu-arithmetic-dispatch'
_TAPE = 'src/tape_0.inc'
_FP_BITS = 2
_PROVIDERS = {'ADD': 'scar_numpy_complex_add', 'MULTIPLY': 'scar_numpy_complex_multiply'}
_ARITHMETIC = b'            auto loop = step.operation == Op::multiply ? scar_numpy_complex_multiply : scar_numpy_complex_add;\n'
_ANCHOR = b'constexpr std::size_t skip_fp_clear = 1;'
_ORIGINAL = b'''            std::int64_t strides[]{step.stride[0], step.stride[1], step.stride[2]};
            for (std::int64_t index = 0; index < step.repeat; ++index) {
                char* operands[3];
                for (unsigned n = 0; n < 3; ++n) {
                    operands[n] = base + step.offset[n] + step.advance[n] * index;
                }
                loop(operands, &step.count, strides, nullptr);
            }'''
_TEMPLATE = '''// All operands retain the original Step offsets and repeat order.
template<auto Provider, std::int64_t Count, std::int64_t Repeat,
         std::int64_t S0, std::int64_t S1, std::int64_t S2,
         std::int64_t A0, std::int64_t A1, std::int64_t A2>
[[gnu::noinline]] void deliver_repeated_arithmetic(char* base, const std::array<std::int64_t, 3>& offsets) {
    static constexpr std::int64_t count = Count;
    const std::int64_t strides[]{S0, S1, S2};
    for (std::int64_t index = 0; index < Repeat; ++index) {
        char* operands[]{base + offsets[0] + A0 * index,
                         base + offsets[1] + A1 * index,
                         base + offsets[2] + A2 * index};
        Provider(operands, &count, strides, nullptr);
    }
}

using RepeatedArithmetic = void (*)(char*, const std::array<std::int64_t, 3>&);
constexpr RepeatedArithmetic repeated_arithmetic[]{
'''


def _need(ok, detail):
    if not ok:
        raise ValueError('native CPU arithmetic dispatch: ' + detail)


def _replace(source, old, new):
    _need(source.count(old) == 1, 'expected one exact arithmetic tape site')
    return source.replace(old, new)


def _ints(values):
    _need(type(values) is list and len(values) == 3
          and all(type(value) is int and -(1 << 63) < value < 1 << 63 for value in values),
          'expected exact signed ABI vector')
    return ', '.join(map(str, values))


def _row(shape, binding, argument):
    fields = (_ints(binding['offsets']), _ints(shape['strides']), _ints(shape['advances']))
    vectors = ', '.join('{' + part + '}' for part in fields)
    return (f'    {{Op::{shape["operator"].lower()}, {shape["count"]}, {shape["repeat"]}, '
            f'{vectors}, {argument}}},').encode()


def _lower(tape, shapes, bindings):
    original_shapes = sorted(shapes, key=lambda row: row['id'])
    shape_ids, definitions = {}, []
    for dense_index, shape in enumerate(original_shapes, 1):
        id_ = shape['id']
        provider = _PROVIDERS.get(shape['operator'])
        count, repeat = shape['count'], shape['repeat']
        _need(provider is not None and provider == shape['provider'] and type(id_) is int and id_ >= 0
              and id_ not in shape_ids and type(count) is type(repeat) is int and 0 < count < 1 << 63
              and 1 < repeat < 1 << 63, 'shape differs from its fixed provider ABI or repeat')
        shape_ids[id_] = dense_index, shape
        definitions.append(f'    &deliver_repeated_arithmetic<&{provider}, {count}, {repeat}, '
                           f'{_ints(shape["strides"])}, {_ints(shape["advances"])}>,')

    before, marker, tail = tape.partition(_ROWS)
    body, row_end, after = tail.partition(_ROW_END)
    _need(marker and row_end and tape.count(_ROWS) == tape.count(_ROW_END) == 1,
          'expected a unique exact grouped step table')
    rows = body.split(b'\n')
    seen, used = set(), set()
    for binding in bindings:
        index, id_, guard = (binding[key] for key in ('compiled_step', 'shape', 'guard_bits'))
        _need(type(index) is int and 0 <= index < len(rows) and index not in seen
              and id_ in shape_ids and type(guard) is int and 0 <= guard < 1 << _FP_BITS,
              'compiled arithmetic slot or original guard mask differs')
        dense_index, shape = shape_ids[id_]
        _need(rows[index] == _row(shape, binding, guard), 'exact retained arithmetic row changed')
        rows[index] = _row(shape, binding, (dense_index << _FP_BITS) | guard)
        seen.add(index)
        used.add(id_)
    _need(used and used == shape_ids.keys(), 'candidate must use every selected shape')

    tape = before + marker + b'\n'.join(rows) + row_end + after
    tape = _replace(tape, _ANCHOR, (_TEMPLATE + '\n'.join(definitions) + '\n};\n\n').encode() + _ANCHOR)
    tape = _replace(tape, _ARITHMETIC, b'')
    portable = _ARITHMETIC.rstrip(b'\n') + b'\n' + _ORIGINAL
    fallback = b'\n'.join(b'    ' + line for line in portable.split(b'\n'))
    tape = _replace(tape, _ORIGINAL, b'''            const auto delivery = GroupedFp ? step.argument >> 2 : 0;
            if (delivery != 0) {
                repeated_arithmetic[delivery - 1](base, step.offset);
            } else {
''' + fallback + b'\n            }')
    return tape, list(shape_ids)


def _material(plan):
    baseline_ref, runtime_ref = plan['baseline'], plan['runtime']
    baseline, runtime = read_artifact(baseline_ref), read_artifact(runtime_ref)
    original = Path(baseline_ref['uri']).parent
    _need(baseline_ref['schema'] == BASELINE and runtime_ref['schema'] == RUNTIME
          and baseline['runtime'] == runtime_ref and baseline['scope'] == runtime['scope'] == plan['scope']
          and baseline['cxx_standard'] == 20 and baseline['python_runtime_required'] is False
          and baseline['deployment_authorized'] is False and baseline['binary'] == runtime['binary']
          and baseline['cpu_ids'] == runtime['cpu_ids'] and baseline['launch_environment'] == runtime['launch_environment'],
          'selected shape plan crosses the exact grouped runtime or launch scope')
    members = {}
    for name, item in baseline['files'].items():
        path = _path(original, name)
        _need(item['uri'] == str(path) and path.resolve(strict=True) == path, 'invalid grouped project member')
        members[name] = read_artifact(item, json_document=False)
    members.pop(baseline['binary'])
    members[_TAPE], ids = _lower(members[_TAPE], plan['shapes'], plan['bindings'])
    compiler = baseline['compiler']
    read_artifact(compiler, json_document=False)
    return baseline, runtime, original, members, compiler, ids


def _linked_specializations(executable, expected_shapecount):
    listing = subprocess.check_output(['/usr/bin/objdump', '-d', '--demangle', '--no-show-raw-insn', str(executable)],
                                      env={'LC_ALL': 'C'}, text=True)
    shapes, active = {}, None
    for line in listing.splitlines():
        heading = re.fullmatch(r'[0-9a-f]+ <(.*)>:', line)
        if heading:
            symbol = heading[1]
            active = None
            if 'deliver_repeated_arithmetic<' in symbol:
                _need(symbol not in shapes, 'duplicate linked specialization')
                active = symbol
                shapes[active] = set()
        elif active is not None:
            call = re.search(r'\bcallq?\s+[0-9a-f]+\s+<(scar_numpy_complex_(?:add|multiply))>', line)
            if call:
                shapes[active].add(call[1])
    _need(len(shapes) == expected_shapecount and all(len(targets) == 1 for targets in shapes.values()),
          'linked shape functions are missing or lack one direct stock provider target')
    return dict(shape_functions=len(shapes), direct_provider_targets=sorted({name for targets in shapes.values() for name in targets}))


def _plan(reference):
    from scar.optimization.cpu_arithmetic_dispatch_plan import verify_cpu_arithmetic_dispatch_plan

    plan = verify_cpu_arithmetic_dispatch_plan(reference)
    _need(plan['status'] == 'ELIGIBLE_FOR_EXPLICIT_VALIDATION'
          and plan['transformation'] == 'SPECIALIZE_REPEATED_NATIVE_ARITHMETIC_DISPATCH',
          'repeated arithmetic specialization is not admitted')
    return plan


def _report(reference, plan, material, root, output):
    baseline, runtime, original, members, compiler, ids = material
    files = {}
    for name, payload in members.items():
        path = _path(root, name)
        _need(path.resolve(strict=True) == path and path.read_bytes() == payload, 'project member differs: ' + name)
        files[name] = dict(uri=str(output/name), size_bytes=len(payload), sha256=sha256(payload).hexdigest(),
                           schema='scar.native-project-member')
    binary = baseline['binary']
    executable = root/binary
    mode = executable.stat().st_mode
    _need(executable.resolve(strict=True) == executable and stat.S_ISREG(mode) and os.access(executable, os.X_OK)
          and not mode & (stat.S_ISUID | stat.S_ISGID), 'ordinary local ELF executable required')
    files[binary] = dict(artifact_reference(executable, 'scar.native-project-member'), uri=str(output/binary))
    return dict(schema=SCHEMA, schema_version=1, baseline=plan['baseline'], runtime=plan['runtime'], plan=reference,
        proof=plan['proof'], baseline_executable=baseline['executable'], scope=baseline['scope'], schedule=baseline['schedule'],
        recipe=baseline['recipe'], decision=baseline['decision'], cpu_ids=baseline['cpu_ids'],
        launch_environment=baseline['launch_environment'], compiler=compiler, cxx_standard=20,
        files=files, executable=files[binary], binary=binary, linkage=_linkage(runtime, original, root),
        rewrite=dict(changed=[_TAPE], encoded_shape_ids=ids, specialized_steps=len(plan['bindings']),
            original_guard_low_bits=_FP_BITS, private_probe_uses_original_path=True, arithmetic_kernel_added=False,
            linked=_linked_specializations(executable, len(ids))),
        validation_scope='EXPLICIT_SCOPED_ARITHMETIC_DISPATCH_VALIDATION', target_executed=False,
        runtime_validated=False, python_runtime_required=False, deployment_authorized=False)


def export_native_cpu_arithmetic_dispatch(plan_ref, out):
    """Build the exact retained shape delivery plan as an independent C++20 package."""
    plan = _plan(plan_ref)
    material = _material(plan)
    baseline, _, original, members, compiler, _ = material
    out = Path(out).absolute()
    _need(out.resolve() == out and not out.exists() and not out.is_symlink()
          and not out.is_relative_to(original) and not out.is_relative_to(Path(plan['runtime']['uri']).parent),
          'fresh output outside the original and grouped native projects required')
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-arithmetic-dispatch-', dir=out.parent) as temporary:
        root = Path(temporary)/'project'
        for name, payload in members.items():
            path = _path(root, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        libraries = tuple(root/name for name in members if name.startswith('provider/objects/'))
        executable, _ = build_native_project(root, Path(baseline['binary']).name, (compiler['uri'],),
                                             expected_libraries=libraries)
        _need(executable == root/baseline['binary'], 'compiled target differs from the grouped CMake entry')
        report = _report(plan_ref, plan, material, root, out)
        (root/'result.json').write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n')
        _need(not out.exists() and not out.is_symlink(), 'fresh output became occupied during compilation')
        root.rename(out)
    return artifact_reference(out/'result.json', SCHEMA)


def verify_native_cpu_arithmetic_dispatch(reference):
    """Cold replay the exact shape package, unchanged sources and linked provider calls."""
    report = read_artifact(reference)
    path = Path(reference['uri'])
    _need(reference['schema'] == SCHEMA and path.name == 'result.json' and path.resolve(strict=True) == path,
          'expected an exact local arithmetic dispatch project receipt')
    plan = _plan(report['plan'])
    _need(report == _report(report['plan'], plan, _material(plan), path.parent, path.parent),
          'retained specialization plan or C++ package differs at replay')
    return report
