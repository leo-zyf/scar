"""Fuse proved coefficient hooks into the original grouped arithmetic tape pass."""

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
from tempfile import TemporaryDirectory

from scar.codegen.native_cpu_arena_reuse import _linkage, _path
from scar.codegen.native_cpu_fp_guard_groups import SCHEMA as BASELINE, _ROW_END, _ROWS
from scar.codegen.native_project import build_native_project
from scar.codegen.native_seeded_array_task_entry import SCHEMA as RUNTIME
from scar.ir.execution_recipe import artifact_reference, read_artifact


SCHEMA = 'scar.native-cpu-coefficient-dispatch'
_PROFILE = 'src/profile_0.cpp'
_TAPE = 'src/tape_0.inc'
_HOOK_DECLARATION = b'''enum class HookKind { coefficient, normalization };
struct Hook {
    std::size_t step;
    HookKind kind;
    std::size_t index;
};

'''
_HOOKS = b'constexpr Hook hooks[]{\n'
_HOOKS_END = b'\n};\nbool prepare_coefficients('
_HOOK = re.compile(rb'    \{(\d+), HookKind::(coefficient|normalization), (\d+)\},')
_BODY = b'''char* run_body(char* base, const std::array<Coefficient, coefficient_count>& prepared,
               std::size_t cursor, std::size_t end, std::vector<std::vector<char>>& unused) {
'''
_BODY_END = b'\nbool write_handoffs('
_LOOP = b'    for (auto index = begin; index < end; ++index) {\n'
_LOOP_END = b'    }\n    return true;\n}\n}\n'
_LITERAL = b'''        case Op::literal:
            std::memcpy(output, coefficients[step.argument].data(), step.count);
            break;'''
_NORMALIZE = b'''        case Op::normalize:
#ifdef SCAR_ARRAY_TAPE_NORMALIZE
            if (scar_numpy_normalize_v1(step.count, input, output) != static_cast<int>(step.argument))
                return false;
            break;
#else
            return false;
#endif'''
_CHECKPOINT = b'''        case Op::checkpoint:
            if (capture) checkpoints[step.argument].assign(input, input + step.count);
            break;'''
_NORMALIZE_HOOK = b'        if (hook->kind == HookKind::normalization) {\n'
_NORMALIZE_END = b'''        }
        std::memcpy(base + step.offset[2], prepared[hook->index].data(), step.count);'''
_LITERAL_ROW = re.compile(rb'    \{Op::literal, (\d+), 1, \{0, 0, (\d+)\}, \{0, 0, 0\}, \{0, 0, 0\}, (\d+)\},')


def _need(ok, detail):
    if not ok:
        raise ValueError('native CPU coefficient dispatch: ' + detail)


def _replace(source, old, new):
    _need(source.count(old) == 1, 'expected one exact grouped runner site')
    return source.replace(old, new)


def _literal(rows, step):
    _need(type(step) is int and 0 <= step < len(rows), 'literal lies outside the original tape')
    parsed = _LITERAL_ROW.fullmatch(rows[step])
    _need(parsed is not None, 'selected coefficient or static write is not an original literal')
    return tuple(map(int, parsed.groups()))


def _actions(tape, hook_table, hooks, elisions):
    rows_marker = tape.partition(_ROWS)
    content, end, _ = rows_marker[2].partition(_ROW_END)
    _need(rows_marker[1] and end and tape.count(_ROWS) == tape.count(_ROW_END) == 1,
          'expected the original immutable Step table')
    rows = content.split(b'\n')
    parsed_hooks = [_HOOK.fullmatch(row) for row in hook_table.split(b'\n')]
    _need(parsed_hooks and all(parsed_hooks), 'unexpected grouped coefficient hook encoding')
    observed = [(int(row[1]), row[2].decode().upper(), int(row[3])) for row in parsed_hooks]
    _need(observed == [(row['compiled_step'], row['kind'], 0 if row['kind'] == 'NORMALIZATION' else row['slot'])
                       for row in hooks],
          'plan must retain every original coefficient and normalization hook in order')

    actions = {}
    normalization = []
    for hook in hooks:
        step = hook['compiled_step']
        if hook['kind'] == 'NORMALIZATION':
            _need(0 <= step < len(rows) and rows[step].startswith(b'    {Op::normalize, '), 'normalization hook moved')
            normalization.append(step)
            continue
        slot = hook['slot']
        _need(hook['kind'] == 'COEFFICIENT' and type(slot) is int and 0 <= slot < (1 << 31) - 1
              and _literal(rows, step) == (hook['bytes'], hook['destination_offset'], hook['original_literal_index'])
              and step not in actions, 'prepared coefficient write differs from the original literal slot')
        actions[step] = slot + 1
    _need(normalization and actions, 'coefficient and normalization hooks required')

    dynamic_steps = frozenset(actions)
    for elision in elisions:
        step, producer = elision['compiled_step'], elision['producer_step']
        region = elision['bytes'], elision['destination_offset'], elision['literal_index']
        _need(step not in actions and producer not in dynamic_steps and producer < step
              and _literal(rows, step) == _literal(rows, producer) == region,
              'elided literal differs from the exact earlier static write')
        actions[step] = -1

    assignments = '\n'.join(f'    actions[{step}] = {value};' for step, value in sorted(actions.items()))
    table = (b'constexpr auto coefficient_actions = [] {\n'
             b'    std::array<std::int32_t, std::size(steps)> actions{};\n' + assignments.encode()
             + b'\n    return actions;\n}();\n\n')
    return table, len(rows)


def _prepared_copy(hooks):
    widths = {row['bytes'] for row in hooks if row['kind'] == 'COEFFICIENT'}
    remainder = b'step.count'
    source = b'                std::memcpy(output, prepared[action - 1].data(), %s);'
    if widths == {16, 64}:
        return b'''                if (step.count == 16) {
    %s
                } else {
    %s
                }''' % (source % b'16', source % b'64')
    return source % remainder


def _lower(tape, profile, hooks, elisions):
    before, hook_marker, after = profile.partition(_HOOKS)
    hook_table, hook_end, after = after.partition(_HOOKS_END)
    _need(hook_marker and hook_end and profile.count(_HOOKS) == profile.count(_HOOKS_END) == 1,
          'expected the retained profile hook table')
    actions, table_length = _actions(tape, hook_table, hooks, elisions)

    _, loop_marker, remaining = tape.partition(_LOOP)
    loop, loop_end, _ = remaining.partition(_LOOP_END)
    _need(loop_marker and loop_end and tape.count(_LOOP) == 1, 'expected one exact retained Step interpreter')
    profile = _replace(before + b'bool prepare_coefficients(' + after, _HOOK_DECLARATION, b'')
    before, body_marker, after = profile.partition(_BODY)
    original_body, write_marker, after = after.partition(_BODY_END)
    _need(body_marker and write_marker and profile.count(_BODY) == 1
          and b'execute_round<true>(base, cursor, hook->step, false, unused)' in original_body,
          'expected original grouped production body and checkpoint path')
    first, normalize_marker, last = original_body.partition(_NORMALIZE_HOOK)
    normalization, normalize_end, _ = last.partition(_NORMALIZE_END)
    _need(normalize_marker and normalize_end and not _NORMALIZE_HOOK in first,
          'expected the original production normalization success and failure paths')
    # The original production hook returns on both successful statuses and on decline.
    normalization = b'        case Op::normalize: {\n' + normalization + b'        }'
    loop = _replace(loop, _NORMALIZE, normalization)
    loop = _replace(loop, _CHECKPOINT, b'        case Op::checkpoint:\n            break;')
    loop = _replace(loop, _LITERAL, b'''        case Op::literal: {
            const auto action = coefficient_actions[index];
            if (action > 0) {
%s
            } else if (action == 0) {
                std::memcpy(output, coefficients[step.argument].data(), step.count);
            }
            break;
        }''' % _prepared_copy(hooks))
    _need(loop.count(b'!GroupedFp || ') == loop.count(b'return false;') == 2,
          'grouped floating-point clear/test boundaries changed')
    loop = loop.replace(b'!GroupedFp || ', b'').replace(b'return false;', b'return nullptr;')
    # Preserve the interpreter's address expressions, even for the private copies.
    # input is still used by Op::copy. The normalization hook keeps its own two outputs.
    body = (_BODY.replace(b'& unused)', b'&)') + _LOOP.replace(b'= begin;', b'= cursor;')
            + loop + b'    }\n    return nullptr;\n}\n')
    return before + actions + body + write_marker + after, table_length


def _plan(reference):
    from scar.optimization.cpu_coefficient_dispatch_plan import verify_cpu_coefficient_dispatch_plan

    plan = verify_cpu_coefficient_dispatch_plan(reference)
    _need(plan['status'] == 'ELIGIBLE_FOR_EXPLICIT_VALIDATION'
          and plan['transformation'] == 'FUSE_COEFFICIENT_HOOK_DISPATCH_AND_ELIDE_PROVEN_LITERAL_WRITES',
          'coefficient dispatch plan is not admitted')
    return plan


def _material(plan):
    baseline_ref, runtime_ref = plan['baseline'], plan['runtime']
    baseline, runtime = read_artifact(baseline_ref), read_artifact(runtime_ref)
    original = Path(baseline_ref['uri']).parent
    _need(baseline_ref['schema'] == BASELINE and runtime_ref['schema'] == RUNTIME
          and baseline['runtime'] == runtime_ref and baseline['scope'] == runtime['scope'] == plan['scope']
          and baseline['cxx_standard'] == 20 and baseline['python_runtime_required'] is False
          and baseline['deployment_authorized'] is False and baseline['binary'] == runtime['binary']
          and baseline['cpu_ids'] == runtime['cpu_ids'] and baseline['launch_environment'] == runtime['launch_environment'],
          'plan crosses the exact retained grouped runtime or launch')
    members = {}
    for name, item in baseline['files'].items():
        path = _path(original, name)
        _need(item['uri'] == str(path) and path.resolve(strict=True) == path, 'invalid grouped project member')
        members[name] = read_artifact(item, json_document=False)
    members.pop(baseline['binary'])
    members[_PROFILE], count = _lower(members[_TAPE], members[_PROFILE], plan['hooks'], plan['elidable_literal_writes'])
    compiler = baseline['compiler']
    read_artifact(compiler, json_document=False)
    return baseline, runtime, original, members, compiler, count


def _report(reference, plan, material, root, output):
    baseline, runtime, original, members, compiler, count = material
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
        rewrite=dict(changed=[_PROFILE], coefficient_hook_sites=sum(row['kind'] == 'COEFFICIENT' for row in plan['hooks']),
            elided_literal_sites=len(plan['elidable_literal_writes']), action_table_entries=count,
            action_table_bytes=count * 4, one_production_dispatch_loop_per_round=True,
            lookup_in_literal_case_only=True, prepared_copy_uses_only_proved_widths=True,
            private_probe_uses_original_tape=True, arithmetic_kernel_added=False),
        validation_scope='EXPLICIT_SCOPED_COEFFICIENT_DISPATCH_VALIDATION', target_executed=False,
        runtime_validated=False, python_runtime_required=False, deployment_authorized=False)


def export_native_cpu_coefficient_dispatch(plan_ref, out):
    """Build the exact ordered coefficient- and literal-dispatch plan as C++20."""
    plan = _plan(plan_ref)
    material = _material(plan)
    baseline, _, original, members, compiler, _ = material
    out = Path(out).absolute()
    _need(out.resolve() == out and not out.exists() and not out.is_symlink()
          and not out.is_relative_to(original) and not out.is_relative_to(Path(plan['runtime']['uri']).parent),
          'fresh output outside the original and grouped native projects required')
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-coefficient-dispatch-', dir=out.parent) as temporary:
        root = Path(temporary)/'project'
        for name, payload in members.items():
            path = _path(root, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        libraries = tuple(root/name for name in members if name.startswith('provider/objects/'))
        executable, _ = build_native_project(root, Path(baseline['binary']).name, (compiler['uri'],),
                                             expected_libraries=libraries)
        _need(executable == root/baseline['binary'], 'compiled target differs from the exact grouped CMake entry')
        report = _report(plan_ref, plan, material, root, out)
        (root/'result.json').write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n')
        _need(not out.exists() and not out.is_symlink(), 'fresh output became occupied during compilation')
        root.rename(out)
    return artifact_reference(out/'result.json', SCHEMA)


def verify_native_cpu_coefficient_dispatch(reference):
    """Cold replay the selected lowering, unchanged members and exact provider linkage."""
    report = read_artifact(reference)
    path = Path(reference['uri'])
    _need(reference['schema'] == SCHEMA and path.name == 'result.json' and path.resolve(strict=True) == path,
          'expected an exact local coefficient dispatch project receipt')
    plan = _plan(report['plan'])
    _need(report == _report(report['plan'], plan, _material(plan), path.parent, path.parent),
          'retained coefficient dispatch plan or C++ project differs at replay')
    return report
