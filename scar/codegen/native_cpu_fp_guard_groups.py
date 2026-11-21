"""Lower proved sticky floating-point guard groups to an independent C++20 entry."""

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
from tempfile import TemporaryDirectory

from scar.codegen.native_cpu_arena_reuse import _linkage, _path
from scar.codegen.native_project import build_native_project
from scar.codegen.native_seeded_array_task_entry import SCHEMA as RUNTIME
from scar.ir.execution_recipe import artifact_reference, read_artifact


SCHEMA = 'scar.native-cpu-fp-guard-groups'
_TAPE = 'src/tape_0.inc'
_PROFILE = 'src/profile_0.cpp'
_ROWS = b'constexpr Step steps[]{\n'
_ROW_END = b'\n};\nconstexpr std::size_t rounds[][2]{'
_ARITHMETIC = re.compile(rb'    \{Op::(?:multiply|add), [^\n]+, 0\},')
_EXECUTE = b'bool execute_round(char* base, std::size_t begin, std::size_t end,'
_CLEAR = b'            if (std::feclearexcept(fp_errors) != 0) return false;'
_TEST = b'            if (std::fetestexcept(fp_errors)) return false;'
_CALL = b'execute_round(base, cursor, hook->step, false, unused)'


def _need(ok, reason):
    if not ok:
        raise ValueError('native CPU FP guard groups: ' + reason)


def _replace(source, original, transformed):
    _need(source.count(original) == 1, 'expected one exact generated native guard site')
    return source.replace(original, transformed)


def _lower(tape, profile, groups):
    before, marker, table = tape.partition(_ROWS)
    body, tail_marker, after = table.partition(_ROW_END)
    _need(marker and tail_marker and tape.count(_ROWS) == tape.count(_ROW_END) == 1,
          'exact generated native tape table required')
    rows = body.split(b'\n')
    previous = 0
    for group in groups:
        begin, end = group['compiled_steps']
        _need(type(begin) is type(end) is int and previous <= begin < end - 1 < len(rows),
              'compiled guard groups must be disjoint, in order, and contain at least two operations')
        for index in range(begin, end):
            row = rows[index]
            _need(_ARITHMETIC.fullmatch(row), 'guard group crosses an arithmetic row or existing argument ownership')
            # The arithmetic cases have no other role for this existing word.
            # Bits 0 and 1 suppress only the interior clear and test actions.
            skip = int(index != begin) | (int(index != end - 1) << 1)
            rows[index] = row[:-3] + str(skip).encode() + b'},'
        previous = end
    tape = before + marker + b'\n'.join(rows) + tail_marker + after
    tape = _replace(tape, _EXECUTE,
        b'constexpr std::size_t skip_fp_clear = 1;\nconstexpr std::size_t skip_fp_test = 2;\n\n'
        b'template<bool GroupedFp = false>\n' + _EXECUTE)
    tape = _replace(tape, _CLEAR, b'''            if ((!GroupedFp || !(step.argument & skip_fp_clear))
                    && std::feclearexcept(fp_errors) != 0) return false;''')
    tape = _replace(tape, _TEST, b'''            if ((!GroupedFp || !(step.argument & skip_fp_test))
                    && std::fetestexcept(fp_errors)) return false;''')
    profile = _replace(profile, _CALL, b'execute_round<true>(base, cursor, hook->step, false, unused)')
    return tape, profile


def _material(plan):
    reference = plan['runtime']
    runtime = read_artifact(reference)
    original = Path(reference['uri']).parent
    _need(reference['schema'] == RUNTIME and runtime['scope'] == plan['scope'] and plan['groups']
          and runtime['cxx_standard'] == 20 and runtime['python_runtime_required'] is False
          and runtime['decision']['mode'] == 'ORDERED_OPENMP' and runtime['deployment_authorized'] is False,
          'expected an admitted original C++20 task-private OpenMP entry')
    members = {}
    for name, item in runtime['files'].items():
        path = _path(original, name)
        _need(str(path) == item['uri'] and path.resolve(strict=True) == path, 'invalid original native project member')
        members[name] = read_artifact(item, json_document=False)
    members.pop(runtime['binary'])
    members[_TAPE], members[_PROFILE] = _lower(members[_TAPE], members[_PROFILE], plan['groups'])
    evidence = read_artifact(read_artifact(runtime['schedule'])['thread_evidence'])
    compiler = evidence['compiler_binary']
    read_artifact(compiler, json_document=False)
    return runtime, original, members, compiler


def _plan(reference):
    from scar.optimization.cpu_fp_guard_plan import verify_cpu_fp_guard_plan

    plan = verify_cpu_fp_guard_plan(reference)
    _need(plan['status'] == 'ELIGIBLE_FOR_EXPLICIT_VALIDATION'
          and plan['transformation'] == 'GROUP_CONTIGUOUS_FP_ENVIRONMENT_GUARDS', 'grouped FP guards are not admitted')
    return plan


def _report(reference, plan, material, root, output):
    runtime, original, members, compiler = material
    files = {}
    for name, payload in members.items():
        path = _path(root, name)
        _need(path.resolve(strict=True) == path and path.read_bytes() == payload, 'project member differs: ' + name)
        files[name] = dict(uri=str(output/name), size_bytes=len(payload), sha256=sha256(payload).hexdigest(),
                           schema='scar.native-project-member')
    binary = runtime['binary']
    executable = root/binary
    mode = executable.stat().st_mode
    _need(executable.resolve(strict=True) == executable and stat.S_ISREG(mode) and os.access(executable, os.X_OK)
          and not mode & (stat.S_ISUID | stat.S_ISGID), 'ordinary local ELF executable required')
    files[binary] = dict(artifact_reference(executable, 'scar.native-project-member'), uri=str(output/binary))
    return dict(schema=SCHEMA, schema_version=1, plan=reference, proof=plan['proof'], runtime=plan['runtime'],
        original_executable=runtime['files'][binary], scope=runtime['scope'], schedule=runtime['schedule'],
        recipe=runtime['recipe'], decision=runtime['decision'], cpu_ids=runtime['cpu_ids'], groups=plan['groups'],
        launch_environment=runtime['launch_environment'], compiler=compiler, cxx_standard=20,
        files=files, executable=files[binary], binary=binary, linkage=_linkage(runtime, original, root),
        rewrite=dict(changed=[_PROFILE, _TAPE], boundary_encoding='EXISTING_ARITHMETIC_STEP_ARGUMENT_BITS',
                     new_kernel=False, independent_runtime_group_lookup=False, private_probe_uses_original_guards=True),
        validation_scope='EXPLICIT_SCOPED_STICKY_GUARD_VALIDATION', target_executed=False, runtime_validated=False,
        python_runtime_required=False, deployment_authorized=False)


def export_native_cpu_fp_guard_groups(plan_ref, out):
    """Build a fresh C++20 project with the exact approved guard boundaries."""
    plan = _plan(plan_ref)
    material = _material(plan)
    runtime, original, members, compiler = material
    out = Path(out).absolute()
    _need(out.resolve() == out and not out.exists() and not out.is_symlink() and not out.is_relative_to(original),
          'fresh output outside the original project required')
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-fp-groups-', dir=out.parent) as temporary:
        root = Path(temporary)/'project'
        for name, payload in members.items():
            path = _path(root, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        libraries = tuple(root/name for name in members if name.startswith('provider/objects/'))
        executable, _ = build_native_project(root, Path(runtime['binary']).name, (compiler['uri'],),
                                             expected_libraries=libraries)
        _need(executable == root/runtime['binary'], 'compiled target differs from the original CMake entry')
        report = _report(plan_ref, plan, material, root, out)
        (root/'result.json').write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n')
        _need(not out.exists() and not out.is_symlink(), 'fresh output became occupied during compilation')
        root.rename(out)
    return artifact_reference(out/'result.json', SCHEMA)


def verify_native_cpu_fp_guard_groups(reference):
    """Cold replay the approved groups, code, unchanged members and provider linkage."""
    report = read_artifact(reference)
    path = Path(reference['uri'])
    _need(reference['schema'] == SCHEMA and path.name == 'result.json' and path.resolve(strict=True) == path,
          'expected an exact local grouped-guard project receipt')
    plan = _plan(report['plan'])
    _need(report == _report(report['plan'], plan, _material(plan), path.parent, path.parent),
          'retained guard groups or native package differs at replay')
    return report
