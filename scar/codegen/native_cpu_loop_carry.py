"""Bind an admitted steady-loop carry directly to its early native reads."""

from hashlib import sha256
import json
import os
from pathlib import Path
import stat
from tempfile import TemporaryDirectory

from scar.codegen.native_cpu_arena_reuse import _linkage, _path
from scar.codegen.native_project import build_native_project
from scar.codegen.native_seeded_array_task_entry import SCHEMA as RUNTIME
from scar.ir.execution_recipe import artifact_reference, read_artifact


SCHEMA = 'scar.native-cpu-loop-carry'
_PROFILE = 'src/profile_0.cpp'
_CAPTURE = b'''    auto* output = std::fopen(path.c_str(), "wbx");
    if (!output) return false;
    const auto wrote = std::fprintf(output, "{\\"copies\\": %llu, \\"same_span_transfers\\": %llu}\\n",
        static_cast<unsigned long long>(counts.copies), static_cast<unsigned long long>(counts.transfers));'''
_HANDOFF = b'''        auto* destination = base + steady_destination;
        if (state == destination) {
            ++handoffs.transfers;
        } else {
            std::memcpy(destination, state, state_bytes);
            ++handoffs.copies;
        }
        state = run_body(base, prepared, rounds[1][0] + 1, rounds[1][1], captured);'''
_PREFIX_STEP = '''// Instantiate only the proved prefix. All later steps use the original tape.
constexpr const Hook* carry_hook(std::size_t index) {
    for (const auto& hook : hooks) if (hook.step == index) return &hook;
    return nullptr;
}

template<std::size_t Index, int CarryOperand = -1, std::int64_t Relative = 0>
bool execute_carry_step(char* base, char* carried,
                        const std::array<Coefficient, coefficient_count>& prepared) {
    constexpr auto step = steps[Index];
    constexpr auto* hook = carry_hook(Index);
    auto* output = base + step.offset[2];
    if constexpr (hook != nullptr) {
        static_assert(hook->kind == HookKind::coefficient && CarryOperand == -1);
        std::memcpy(output, prepared[hook->index].data(), step.count);
    } else if constexpr (step.operation == Op::basis) {
        static_assert(CarryOperand == -1);
        std::memset(output, 0, item * step.count);
        const double one = 1.0;
        std::memcpy(output + item * step.argument, &one, sizeof(one));
    } else if constexpr (step.operation == Op::literal) {
        static_assert(CarryOperand == -1);
        std::memcpy(output, coefficients[step.argument].data(), step.count);
    } else if constexpr (step.operation == Op::copy) {
        static_assert(CarryOperand == -1 || CarryOperand == 0);
        const auto* input = CarryOperand == 0 ? carried + Relative : base + step.offset[0];
        if constexpr (step.stride[0] == item && step.stride[2] == item) {
            std::memcpy(output, input, item * step.count);
        } else {
            for (std::int64_t i = 0; i < step.count; ++i) {
                std::memcpy(output + i * step.stride[2], input + i * step.stride[0], item);
            }
        }
    } else {
        static_assert(step.operation == Op::multiply || step.operation == Op::add);
        static_assert(CarryOperand >= -1 && CarryOperand <= 1);
        constexpr auto loop = step.operation == Op::multiply ? scar_numpy_complex_multiply : scar_numpy_complex_add;
        if (std::feclearexcept(fp_errors) != 0) return false;
        const std::int64_t count = step.count;
        std::int64_t strides[]{step.stride[0], step.stride[1], step.stride[2]};
        for (std::int64_t index = 0; index < step.repeat; ++index) {
            char* operands[]{
                (CarryOperand == 0 ? carried + Relative : base + step.offset[0]) + step.advance[0] * index,
                (CarryOperand == 1 ? carried + Relative : base + step.offset[1]) + step.advance[1] * index,
                output + step.advance[2] * index,
            };
            loop(operands, &count, strides, nullptr);
        }
        if (std::fetestexcept(fp_errors)) return false;
    }
    return true;
}

'''


def _need(ok, reason):
    if not ok:
        raise ValueError('native CPU loop carry: ' + reason)


def _replace(source, original, lowered):
    _need(source.count(original) == 1, 'expected one exact C++ profile rewrite site')
    return source.replace(original, lowered)


def _routes(rows):
    choices = {}
    for row in rows:
        source, action = row['source_offset'], row['action']
        _need(type(source) is int and source >= 0 and choices.setdefault(source, action) == action,
              'prior pointer does not uniquely identify the lowered route')
    chosen = [f'state == base + {offset}' for offset, action in sorted(choices.items())
              if action == 'REDIRECT_PREFIX_READS']
    _need(chosen, 'plan has no address eligible for carry redirection')
    return ' || '.join(chosen)


def _prefix(carry, anchor):
    redirects = {}
    checks = [f'static_assert(rounds[1][0] == {anchor} && steps[{anchor}].operation == Op::copy);']
    checks.append(f'static_assert(state_bytes == {carry["bytes"]} && steady_destination == {carry["destination_offset"]});')
    for row in carry['redirects']:
        index, operand, offset, count, stride, size = (row[key] for key in
            ('compiled_step', 'operand', 'relative_offset', 'count', 'stride', 'item_bytes'))
        kind = row['kind'].lower()
        _need(all(type(value) is int for value in (index, operand, offset, count, stride, size))
              and kind in ('copy', 'multiply', 'add') and operand in (0, 1) and (kind != 'copy' or operand == 0)
              and index not in redirects and index > anchor and offset >= 0 and count > 0 and size > 0,
              'unsupported or ambiguous early read redirection')
        checks.append(f'static_assert(steps[{index}].operation == Op::{kind} && steps[{index}].count == {count} && '
                      f'steps[{index}].offset[{operand}] == steady_destination + {offset} && '
                      f'steps[{index}].stride[{operand}] == {stride} && item == {size});')
        redirects[index] = f'{operand}, {offset}'
    _need(redirects and max(redirects) - anchor <= 64, 'expected a small certified prefix')
    suffix = max(redirects) + 1
    calls = [f'execute_carry_step<{index}{", " + redirects[index] if index in redirects else ""}>(base, carried, prepared)'
             for index in range(anchor + 1, suffix)]
    return ('\n'.join(checks) + f'\nconstexpr std::size_t carry_suffix_begin = {suffix};\n\n' + _PREFIX_STEP +
            'bool execute_carry_prefix(char* base, char* carried,\n'
            '                          const std::array<Coefficient, coefficient_count>& prepared) {\n'
            '    return ' + '\n        && '.join(calls) + ';\n}\n\n').encode(), suffix


def _lower(profile, carry, anchor):
    prefix, suffix = _prefix(carry, anchor)
    marker = b'bool write_handoffs(const std::string& prefix, const Handoffs& counts) {'
    profile = _replace(profile, marker, prefix + marker)
    profile = _replace(profile, b'    std::uint64_t transfers{};',
                       b'    std::uint64_t transfers{};\n    std::uint64_t redirected{};')
    profile = _replace(profile, _CAPTURE, b'''    auto* output = std::fopen(path.c_str(), "wbx");
    if (!output) return false;
    const auto wrote = std::fprintf(output,
        "{\\"copies\\": %llu, \\"same_span_transfers\\": %llu, \\"redirected\\": %llu}\\n",
        static_cast<unsigned long long>(counts.copies), static_cast<unsigned long long>(counts.transfers),
        static_cast<unsigned long long>(counts.redirected));''')
    condition = _routes(carry['routes']).encode()
    profile = _replace(profile, _HANDOFF, b'''        auto* destination = base + steady_destination;
        auto begin = rounds[1][0] + 1;
        if (state == destination) {
            ++handoffs.transfers;
        } else if (''' + condition + b''') {
            if (!execute_carry_prefix(base, state, prepared)) return 1;
            ++handoffs.redirected;
            begin = carry_suffix_begin;
        } else {
            std::memcpy(destination, state, state_bytes);
            ++handoffs.copies;
        }
        state = run_body(base, prepared, begin, rounds[1][1], captured);''')
    return profile, suffix


def _material(plan):
    reference = plan['runtime']
    runtime = read_artifact(reference)
    original = Path(reference['uri']).parent
    _need(reference['schema'] == RUNTIME and runtime['scope'] == plan['scope']
          and runtime['cxx_standard'] == 20 and runtime['python_runtime_required'] is False
          and runtime['decision']['mode'] == 'ORDERED_OPENMP' and runtime['deployment_authorized'] is False,
          'expected the admitted original C++20 task-private OpenMP project')
    members = {}
    for name, item in runtime['files'].items():
        path = _path(original, name)
        _need(str(path) == item['uri'] and path.resolve(strict=True) == path, 'invalid original native project member')
        members[name] = read_artifact(item, json_document=False)
    members.pop(runtime['binary'])
    native = read_artifact(runtime['native'])
    schedule = read_artifact(read_artifact(native['memory'])['schedule'])
    carry = plan['carry']
    operation = carry['operation']
    rows = schedule['operations']
    _need(operation == schedule['rounds'][1][0] and rows[operation]['kind'] == 'COPY'
          and len(rows[operation]['groups']) == 1, 'the lowered steady entry does not have one carry step')
    anchor = sum(len(row['groups']) if row['kind'] == 'COPY' else 1 for row in rows[:operation])
    members[_PROFILE], suffix = _lower(members[_PROFILE], carry, anchor)
    evidence = read_artifact(read_artifact(runtime['schedule'])['thread_evidence'])
    compiler = evidence['compiler_binary']
    read_artifact(compiler, json_document=False)
    return runtime, original, members, compiler, anchor, suffix


def _plan(reference):
    from scar.optimization.cpu_loop_carry_plan import verify_cpu_loop_carry_plan

    plan = verify_cpu_loop_carry_plan(reference)
    _need(plan['status'] == 'ELIGIBLE_FOR_EXPLICIT_VALIDATION'
          and plan['transformation'] == 'REDIRECT_READ_ONLY_LOOP_CARRY_PREFIX', 'carry redirection is not admitted')
    return plan


def _report(reference, plan, material, root, output):
    runtime, original, members, compiler, anchor, suffix = material
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
          and not mode & (stat.S_ISUID | stat.S_ISGID),
          'ordinary locally executable project member required')
    files[binary] = dict(artifact_reference(executable, 'scar.native-project-member'), uri=str(output/binary))
    return dict(schema=SCHEMA, schema_version=1, plan=reference, proof=plan['proof'], runtime=plan['runtime'],
        original_executable=runtime['files'][binary], scope=runtime['scope'], schedule=runtime['schedule'],
        recipe=runtime['recipe'], decision=runtime['decision'], cpu_ids=runtime['cpu_ids'],
        carry=plan['carry'], launch_environment=runtime['launch_environment'], compiler=compiler, cxx_standard=20,
        files=files, executable=files[binary], binary=binary, linkage=_linkage(runtime, original, root),
        rewrite=dict(changed=[_PROFILE], compiled_prefix=[anchor + 1, suffix], pattern='COMPILE_TIME_STEPS_THEN_ORIGINAL_TAIL',
                     runtime_per_step_dispatch_added=False, unchanged_unlisted_path='ORIGINAL_MEMCPY_AND_TAPE',
                     diagnostics_fields=['copies', 'same_span_transfers', 'redirected']),
        validation_scope='EXPLICIT_NORMAL_SUCCESS_ONLY', target_executed=False, runtime_validated=False,
        python_runtime_required=False, deployment_authorized=False)


def export_native_cpu_loop_carry(plan_ref, out):
    """Build an independent C++20 project from the retained carry rewrite plan."""
    plan = _plan(plan_ref)
    material = _material(plan)
    runtime, original, members, compiler, _, _ = material
    out = Path(out).absolute()
    _need(out.resolve() == out and not out.exists() and not out.is_symlink() and not out.is_relative_to(original),
          'fresh output outside the original project required')
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-loop-carry-', dir=out.parent) as temporary:
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


def verify_native_cpu_loop_carry(reference):
    """Cold replay the plan, compiled prefix, untouched members, and exact linkage."""
    report = read_artifact(reference)
    path = Path(reference['uri'])
    _need(reference['schema'] == SCHEMA and path.name == 'result.json' and path.resolve(strict=True) == path,
          'expected an exact local loop carry project receipt')
    plan = _plan(report['plan'])
    _need(report == _report(report['plan'], plan, _material(plan), path.parent, path.parent),
          'retained project or lowering differs at cold replay')
    return report
