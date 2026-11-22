"""Lower a proved task-private arena to one lazy owner per OpenMP lane."""

from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
from tempfile import TemporaryDirectory

from scar.codegen.native_provider_bundle import _dependencies, cpu_executable_dynamic
from scar.codegen.native_project import build_native_project
from scar.codegen.native_seeded_array_task_entry import SCHEMA as RUNTIME
from scar.ir.execution_recipe import artifact_reference, read_artifact


SCHEMA = 'scar.native-cpu-task-arena-reuse'
_MAIN = 'src/main.cpp'
_PROFILE = 'src/profile_0.cpp'
_HEADER = 'include/scar_generated/array_lane_arena.hpp'
_HEADER_SOURCE = b'''#pragma once

#include <cstdint>
#include <cstdlib>
#include <memory>

namespace scar::array_arena_v1 {
struct AlignedFree {
    void operator()(void* address) const noexcept { std::free(address); }
};

using Arena = std::unique_ptr<void, AlignedFree>;
static_assert(sizeof(Arena) == sizeof(void*));
static_assert(sizeof(void*) == 8);

int run_profile(double* norm, double* difference, const char* state_prefix,
                std::uint64_t seed, std::uint64_t round_count, Arena& arena);
}
'''
_ENTRY_DECLARATION = b'''extern "C" int scar_array_entry_profile_0(double* norm, double* difference, const char* state_prefix,
                                          std::uint64_t seed, std::uint64_t round_count)'''
_ORIGINAL_ACQUIRE = b'''    std::unique_ptr<void, decltype(&std::free)> arena(std::aligned_alloc(64, arena_bytes), &std::free);
    if (!arena) return 1;'''
_ORIGINAL_CALL = b'''    if (scar_array_entry_profile_0(&result.norm, &result.split_difference, diagnostic ? checkpoint.c_str() : nullptr,
                                  request.seed, request.rounds) != 0) return 3;'''
_ORIGINAL_DISPATCH = b'''    if (!scar::openmp_v1::dispatch(worker_cpus, result.task_count, [&](std::uint64_t task, int lane) {
        if (lane_path) lanes[task] = {lane, gettid(), sched_getcpu()};
        failures[task] = compute({result.seed + task, result.round_count}, result.tolerance,
                                 result.records[task], diagnostic, task);
    })) return 3;'''


def _need(ok, reason):
    if not ok:
        raise ValueError('native CPU arena reuse: ' + reason)


def _replace(source, old, new):
    _need(source.count(old) == 1, 'expected one exact generated C++ rewrite site')
    return source.replace(old, new)


def _lower(main, profile):
    main = _replace(main, b'#include "native_ordered_openmp.hpp"',
                    b'#include "native_ordered_openmp.hpp"\n#include "array_lane_arena.hpp"')
    main = _replace(main, b'                     const char* diagnostic, std::uint64_t ordinal) {', b'''                     const char* diagnostic, std::uint64_t ordinal
#ifdef SCAR_ARRAY_TASK_OPENMP
                     , scar::array_arena_v1::Arena& arena
#endif
) {''')
    main = _replace(main, _ORIGINAL_CALL, b'''#ifdef SCAR_ARRAY_TASK_OPENMP
    if (scar::array_arena_v1::run_profile(&result.norm, &result.split_difference,
            diagnostic ? checkpoint.c_str() : nullptr, request.seed, request.rounds, arena) != 0) return 3;
#else
''' + _ORIGINAL_CALL + b'\n#endif')
    main = _replace(main, _ORIGINAL_DISPATCH, b'''    {
        std::array<scar::array_arena_v1::Arena, worker_cpus.size()> arenas;
        static_assert(sizeof(arenas) == worker_cpus.size() * sizeof(void*));
        if (!scar::openmp_v1::dispatch(worker_cpus, result.task_count, [&](std::uint64_t task, int lane) {
            if (lane_path) lanes[task] = {lane, gettid(), sched_getcpu()};
            failures[task] = compute({result.seed + task, result.round_count}, result.tolerance,
                                     result.records[task], diagnostic, task, arenas[lane]);
        })) return 3;
    }''')
    profile = _replace(profile, b'#include "numpy_random_coefficients.hpp"',
                       b'#include "numpy_random_coefficients.hpp"\n#include "array_lane_arena.hpp"')
    profile = _replace(profile, _ENTRY_DECLARATION, b'''int scar::array_arena_v1::run_profile(double* norm, double* difference, const char* state_prefix,
                                      std::uint64_t seed, std::uint64_t round_count, Arena& arena)''')
    profile = _replace(profile, _ORIGINAL_ACQUIRE, b'''    if (!arena) arena.reset(std::aligned_alloc(64, arena_bytes));
    if (!arena) return 1;''')
    # Keep the prior entry symbol for the serial/non-OpenMP translation path.
    profile += b'\n' + _ENTRY_DECLARATION + b''' {
    scar::array_arena_v1::Arena arena;
    return scar::array_arena_v1::run_profile(norm, difference, state_prefix, seed, round_count, arena);
}
'''
    return {_MAIN: main, _PROFILE: profile, _HEADER: _HEADER_SOURCE}


def _path(root, relative):
    value = PurePosixPath(relative)
    _need(not value.is_absolute() and value.parts and '..' not in value.parts and str(value) == relative,
          'expected a package-relative member')
    return root/value


def _material(plan):
    ref = plan['runtime']
    runtime = read_artifact(ref)
    root = Path(ref['uri']).parent
    decision = runtime['decision']
    workers = decision['worker_cpu_ids']
    _need(ref['schema'] == RUNTIME and runtime['cxx_standard'] == 20 and runtime['python_runtime_required'] is False
          and runtime['deployment_authorized'] is False and decision['mode'] == 'ORDERED_OPENMP'
          and decision['worker_count'] == len(workers) > 1 and workers == plan['worker_cpu_ids']
          and plan['placement']['require_owner_slot_size_bytes'] == 8
          and runtime['scope']['task_count'] > len(workers) and plan['arena']['alignment_bytes'] == 64
          and plan['arena']['bytes'] == decision['memory']['private_arena_bytes_per_numerical_worker']
          and _HEADER not in runtime['files'], 'approved plan does not match the exact OpenMP arena shape')
    members = {}
    for name, reference in runtime['files'].items():
        path = _path(root, name)
        _need(str(path) == reference['uri'] and path.resolve(strict=True) == path, 'invalid original project member')
        members[name] = read_artifact(reference, json_document=False)
    members.pop(runtime['binary'])
    members.update(_lower(members[_MAIN], members[_PROFILE]))
    schedule = read_artifact(runtime['schedule'])
    compiler = read_artifact(schedule['thread_evidence'])['compiler_binary']
    read_artifact(compiler, json_document=False)
    return runtime, root, members, compiler


def _linkage(runtime, original, project):
    binary = project/runtime['binary']
    original_binary = original/runtime['binary']
    dynamic = cpu_executable_dynamic(binary)
    _need(dynamic == runtime['linkage']['dynamic'] == cpu_executable_dynamic(original_binary),
          'recompiled ELF dependency tags changed')

    def dependencies(path, root):
        listing = subprocess.check_output(['/usr/bin/ldd', str(path)], env={'LC_ALL': 'C'}, text=True)
        result = {}
        for name, resolved in _dependencies(listing).items():
            if resolved is not None and Path(resolved).is_relative_to(root):
                relative = Path(resolved).relative_to(root).as_posix()
                _need(relative == 'provider/objects/' + name and relative in runtime['files'],
                      'shared object no longer resolves to its exact project member')
                result[name] = relative
            else:
                result[name] = resolved
        return result

    loaded = dependencies(binary, project)
    _need(loaded == dependencies(original_binary, original), 'loader resolves a different provider or host runtime')
    locks = runtime['linkage']['qualified_runtime_dependencies']
    for name, reference in locks.items():
        _need(loaded.get(name) == reference['uri'], 'host runtime differs from its scoped link receipt')
        read_artifact(reference, json_document=False)
    _need(all(loaded.get(Path(name).name) == name for name in runtime['files']
              if name.startswith('provider/objects/')), 'missing exact numerical provider objects')
    host_libraries = {
        name: artifact_reference(Path(resolved).resolve(strict=True), 'scar.native-host-library')
        for name, resolved in loaded.items() if resolved is not None and Path(resolved).is_absolute()
    }
    return dict(dynamic=dynamic, loaded=loaded, qualified_runtime_dependencies=locks, host_libraries=host_libraries,
                numerical=_numerical_links(runtime, binary))


def _numerical_links(runtime, binary):
    from scar.codegen import native_numpy_complex, native_numpy_random, native_numpy_vector_normalize

    prior = read_artifact(runtime['native'])
    recipe = read_artifact(runtime['recipe'])
    tape = read_artifact(prior['memory'])['tape']
    profile, = (row for row in recipe['profiles'] if row['tape_sha256'] == tape['sha256'])
    stock = read_artifact(profile['package'])['stock_provider']
    random = read_artifact(read_artifact(prior['binding'])['random'])['provider']
    normal = read_artifact(recipe['normalize'])['provider']
    native_numpy_random._audit(binary, random)
    return dict(stock=native_numpy_complex.audit(binary, stock),
                normalization=native_numpy_vector_normalize._audit_divide(binary, normal),
                random='EXACT_STOCK_PCG64_AND_UNIFORM')


def _plan(reference):
    from scar.optimization.cpu_task_arena_plan import verify_cpu_task_arena_plan

    plan = verify_cpu_task_arena_plan(reference)
    _need(plan['status'] == 'ELIGIBLE_FOR_EXPLICIT_VALIDATION'
          and plan['transformation'] == 'LAZY_PER_LANE_PRIVATE_ARENA_REUSE',
          'kernel arena plan has not admitted scoped explicit validation')
    return plan


def _report(reference, plan, material, root, output):
    runtime, original, members, compiler = material
    files = {}
    for name, payload in members.items():
        path = _path(root, name)
        _need(path.resolve(strict=True) == path and path.read_bytes() == payload,
              'transformed C++ or unchanged member differs from planned lowering: ' + name)
        files[name] = dict(uri=str(output/name), size_bytes=len(payload), sha256=sha256(payload).hexdigest(),
                           schema='scar.native-project-member')
    binary = runtime['binary']
    executable = root/binary
    mode = executable.stat().st_mode
    _need(executable.resolve(strict=True) == executable and stat.S_ISREG(mode)
          and os.access(executable, os.X_OK) and not mode & (stat.S_ISUID | stat.S_ISGID),
          'result must be an ordinary executable project member')
    binary_ref = artifact_reference(executable, 'scar.native-project-member')
    files[binary] = dict(binary_ref, uri=str(output/binary))
    return dict(schema=SCHEMA, schema_version=1, plan=reference, proof=plan['proof'], runtime=plan['runtime'],
        original_executable=runtime['files'][binary], scope=runtime['scope'], schedule=runtime['schedule'],
        recipe=runtime['recipe'], decision=runtime['decision'], cpu_ids=runtime['cpu_ids'],
        worker_cpu_ids=plan['worker_cpu_ids'], arena=plan['arena'], allocation_counts=plan['allocation_counts'],
        launch_environment=runtime['launch_environment'], compiler=compiler, build_system='CMAKE', cxx_standard=20,
        files=files, binary=binary, executable=files[binary], linkage=_linkage(runtime, original, root),
        rewrite=dict(changed=[_MAIN, _PROFILE], added=[_HEADER], legacy_c_profile_abi_preserved=True,
                     per_lane_stack_pointer_words=1, diagnostics_use_transformed_path=True),
        validation_scope='EXPLICIT_NORMAL_SUCCESS_ONLY', target_executed=False, runtime_validated=False,
        allocation_failure_equivalence_proven=False, python_runtime_required=False, deployment_authorized=False)


def export_native_cpu_arena_reuse(plan_ref, out):
    """Build the exact planned source variant as an independent C++20 project."""
    plan = _plan(plan_ref)
    material = _material(plan)
    runtime, original, members, compiler = material
    out = Path(out).absolute()
    _need(out.resolve() == out and not out.exists() and not out.is_symlink() and not out.is_relative_to(original),
          'fresh output outside the original project required')
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-lane-arena-', dir=out.parent) as temporary:
        root = Path(temporary)/'project'
        for name, payload in members.items():
            path = _path(root, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        libraries = tuple(root/name for name in members if name.startswith('provider/objects/'))
        executable, _ = build_native_project(root, Path(runtime['binary']).name, (compiler['uri'],),
                                             expected_libraries=libraries)
        _need(executable == root/runtime['binary'], 'compiled target does not match the original CMake entry')
        report = _report(plan_ref, plan, material, root, out)
        (root/'result.json').write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n')
        _need(not out.exists() and not out.is_symlink(), 'fresh output became occupied during compilation')
        root.rename(out)
    return artifact_reference(out/'result.json', SCHEMA)


def verify_native_cpu_arena_reuse(reference):
    """Cold replay the plan, precise rewrite, project members and linked provider."""
    report = read_artifact(reference)
    path = Path(reference['uri'])
    _need(reference['schema'] == SCHEMA and path.name == 'result.json' and path.resolve(strict=True) == path,
          'expected an exact local native arena project manifest')
    plan = _plan(report['plan'])
    _need(report == _report(report['plan'], plan, _material(plan), path.parent, path.parent),
          'retained project or lowering differs at replay')
    return report
