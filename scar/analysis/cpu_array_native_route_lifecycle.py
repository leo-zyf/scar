"""Project the successful lifecycle of two qualified native array-task routes."""

from hashlib import sha256

from scar.analysis.cpu_array_native_route import verify_cpu_array_native_route
from scar.analysis.cpu_native_phase_sites import _source_spans
from scar.analysis.replay_transaction import replay_transaction
from scar.analysis.static_cpp_source import read_static_cpp_source
from scar.codegen import native_array_entry
from scar.codegen import native_seeded_array_task_entry as generator
from scar.ir.execution_recipe import _canonical, read_artifact


SCHEMA = 'scar.cpu-array-native-route-lifecycle'
_MAIN = 'src/main.cpp'
_PROFILE = 'src/profile_0.cpp'
_PROCESS = 'include/scar_generated/native_ordered_dispatch.hpp'
_OPENMP = 'include/scar_generated/native_ordered_openmp.hpp'
_ARGV = 'include/scar_generated/array_entry_argv.hpp'
_CMAKE = 'CMakeLists.txt'


def _need(ok, detail):
    if not ok:
        raise ValueError('CPU array native route lifecycle: ' + detail)


def _anchor(source, raw, start, end):
    return dict(member=source, byte_start=start, byte_end=end,
                selection_sha256=sha256(raw[start:end]).hexdigest())


def _span(source, raw, text, *, start_at=0):
    start = raw.find(text, start_at)
    _need(start >= 0 and raw.find(text, start + len(text)) < 0,
          f'expected unique active provider statement in {source}: {text!r}')
    return _anchor(source, raw, start, start + len(text))


def _materials(runtime, package, recipe):
    refs = package['files']
    prior = read_artifact(runtime['native'])
    family = read_artifact(runtime['family'])
    thread_ref = runtime['thread_contract']
    thread = read_artifact(thread_ref) if thread_ref else None
    _, expected = generator._render(recipe, prior, family, runtime['decision'], runtime['scope'], thread)
    _, indices = native_array_entry._argv_header(recipe['options'], recipe['allow_abbrev'])
    names = (_MAIN, _PROFILE, _ARGV, native_array_entry._IO_MEMBER, _CMAKE,
             _OPENMP if thread else _PROCESS)
    sources = {name: refs[name] for name in names}
    raw = {name: read_artifact(ref, json_document=False) for name, ref in sources.items()}
    # Both packages have already replayed their numerical lowering and ELF/provider
    # qualification. Independently pin the generator boundary used by this projection.
    _need(all(raw[name] == expected[name] for name in names if name not in (_PROFILE, _CMAKE))
          and raw[_CMAKE] == read_artifact(runtime['files'][_CMAKE], json_document=False),
          'entry sources differ from the scoped generator')
    options = [line.removesuffix(b')').split() for line in raw[_CMAKE].splitlines()
               if line.startswith(b'target_compile_options(')]
    _need(len(options) == 1 and (b'[=[-DSCAR_ARRAY_TASK_OPENMP]=]' in options[0]) == bool(thread),
          'selected preprocessor branch differs from the sealed runtime role')
    model = read_static_cpp_source(sources[_MAIN]['uri'])
    _need((model.source.path, model.source.sha256, model.source.size_bytes)
          == tuple(sources[_MAIN][field] for field in ('uri', 'sha256', 'size_bytes')),
          'parsed main differs from its exact source receipt')
    points, _ = _source_spans(raw[_MAIN], model, indices[recipe['output_option']])
    main = {identity: _anchor(_MAIN, raw[_MAIN], point.byte_start, point.byte_end)
            for identity, _, _, _, point in points}
    return sources, raw, main


def _route(runtime_ref, package_ref, placement, recipe, count):
    package = read_artifact(package_ref)
    sources, raw, main = _materials(read_artifact(runtime_ref), package, recipe)
    openmp = placement['mode'] == 'ORDERED_OPENMP'
    mode = 'OPENMP_STATIC_ORDINAL_LANES' if openmp else 'LOCAL_PROCESS_ORDERED_SOCKET_WINDOW'
    header = _OPENMP if openmp else _PROCESS
    workers = placement['worker_count']

    def cpp(name, text):
        return _span(name, raw[name], text)

    def main_cpp(text):
        return _span(_MAIN, raw[_MAIN], text, start_at=main['MAIN_ENTER']['byte_start'])

    prepare = cpp(_PROFILE, b'if (!prepare_coefficients(seed, prepared)) return 1;')
    allocate = cpp(_PROFILE, b'std::aligned_alloc(64, arena_bytes)')
    first = cpp(_PROFILE, b'auto* state = run_body(base, prepared, rounds[0][0], rounds[0][1], captured);')
    steady = cpp(_PROFILE, b'state = run_body(base, prepared, rounds[1][0] + 1, rounds[1][1], captured);')
    observable = cpp(_PROFILE, b'if (scar_numpy_split_observable_v1(state_bytes / item, state, norm, difference) != 0) return 1;')
    _need(prepare['byte_end'] < allocate['byte_start'] < first['byte_start']
          < steady['byte_start'] < observable['byte_start'], 'task lifecycle differs from the numerical source')
    if openmp:
        team = [cpp(header, b'#pragma omp parallel num_threads(width) reduction(|:lane_failed)')]
        setup = [cpp(header, b'ready = sched_setaffinity(0, sizeof(target), &target) == 0'),
                 cpp(header, b'#pragma omp barrier')]
        task_input = [cpp(header, b'ordinal = lane; ordinal < count; ordinal += width')]
        handoff = [cpp(_MAIN, b'failures[task] = compute({result.seed + task, result.round_count}, result.tolerance,')]
        join = [cpp(header, b'omp_set_dynamic(dynamic);'),
                cpp(_MAIN, b'for (const auto error : failures) if (error != 0) return error;')]
    else:
        team = [cpp(header, b'detail::OwnerAffinity affinity(config);'), cpp(header, b'const auto pid = fork();')]
        setup = [cpp(header, b'if (!pin_cpu(cpu) || setrlimit(RLIMIT_AS, &limit) != 0)'),
                 cpp(header, b'if (!detail::transfer(slot.fd, &ready, sizeof(ready), false, startup) || ready != 1)')]
        task_input = [cpp(header, b'!detail::transfer(slot.fd, request.data(), request.size(), true, until)'),
                      cpp(header, b'if (!transfer(fd, request.data(), request.size(), false))')]
        handoff = [cpp(header, b'!transfer(fd, output.data(), output.size(), true)'),
                   cpp(header, b'retire(result);'), cpp(_MAIN, b'std::memcpy(&row, retirement.result.data(), sizeof(row));')]
        join = [cpp(header, b'~Team() {'), cpp(header, b'waitpid(slot.pid, nullptr, WNOHANG)'),
                cpp(_MAIN, b'if (summary.succeeded != result.task_count) return 3;')]
    topology = ('SHARED_PROCESS_ADDRESS_SPACE' if openmp else 'DISTINCT_PROCESS_ADDRESS_SPACES')
    resources = {
        'launch_cpu_set': dict(kind='PERMITTED_PROCESSOR_SET', cpu_ids=placement['launch_cpu_ids']),
        'owner_cpu': dict(kind='DISPATCH_OWNER_PROCESSOR', cpu_ids=[placement['owner_cpu_id']]),
        'worker_cpus': dict(kind='AFFINITY_BOUND_WORKER_PROCESSORS', cpu_ids=placement['worker_cpu_ids']),
        'working_memory': dict(kind=topology, scope=placement['memory']['bound_scope']),
        'transport': dict(kind='ORDINAL_SHARED_RESULT_SLOTS' if openmp else 'UNIX_STREAM_SOCKET_FRAMES',
                          request_frame_bytes=placement['memory']['request_frame_bytes'],
                          result_frame_bytes=placement['memory']['result_frame_bytes']),
        'file': dict(kind='FILESYSTEM_WRITE_AND_CLOSE', durability='UNPROVEN'),
        'stdout': dict(kind='PROCESS_OUTPUT_STREAM', flush_timing='UNPROVEN'),
        'runtime': dict(kind='OPENMP_RUNTIME' if openmp else 'OS_PROCESS_RUNTIME'),
    }

    def op(kind, resource_ids, multiplicity, *source):
        return dict(kind=kind, resource_ids=resource_ids.split(), multiplicity=multiplicity, source=list(source))

    operations = {
        'launch': op('PROCESS_LAUNCH_AND_LOAD', 'launch_cpu_set runtime', 1),
        'ingress': op('PARSE_AND_VALIDATE', 'launch_cpu_set', 1, main['MAIN_ENTER'],
                      main_cpp(b'if (!enforce_coordinator_limit()) return 2;')),
        'team_setup': op('CREATE_OR_ACTIVATE_TEAM', 'launch_cpu_set runtime', 1, main['DISPATCH_BEGIN'], *team),
        'worker_setup': op('BIND_AND_PREPARE_WORKER', 'worker_cpus runtime', workers, *setup),
        'task_ingress': op('ORDINAL_TASK_REQUEST', 'worker_cpus transport' if openmp else
                           'owner_cpu worker_cpus transport', count, *task_input),
        'task_prepare': op('PREPARE_TASK_VALUES', 'worker_cpus working_memory', count, prepare),
        'task_allocate': op('ACQUIRE_PRIVATE_TASK_ARENA', 'worker_cpus working_memory', count, allocate),
        'numerical_body': op('EXECUTE_FIRST_AND_STEADY_ROUNDS', 'worker_cpus working_memory', count, first, steady),
        'task_result': op('FINALIZE_TASK_RESULT', 'worker_cpus working_memory', count, observable),
        'result_handoff': op('COMMIT_ORDINAL_RESULT_SLOT', 'worker_cpus transport' if openmp else
                             'owner_cpu worker_cpus transport', count, *handoff),
        'join': op('COMPLETE_AND_VALIDATE_TEAM', 'launch_cpu_set runtime', 1, *join),
        'reduction': op('REDUCE_ORDERED_RESULT_SLOTS', 'launch_cpu_set working_memory', 1,
                        cpp(_MAIN, b'for (const auto& row : result.records) {\n        result.norm_error')),
        'file_publication': op('WRITE_AND_CLOSE_OUTPUT_FILE', 'launch_cpu_set file', 1,
                               main_cpp(b'write_result(file, result, true);'), main_cpp(b'file.close();')),
        'stdout_publication': op('WRITE_STDOUT_STREAM', 'launch_cpu_set stdout', 1,
                                 main_cpp(b'write_result(std::cout, result, false);')),
        'teardown': op('RETURN_FROM_MAIN_AND_COMPLETE_PROCESS', 'launch_cpu_set runtime stdout', 1,
                       main['PUBLICATION_END']),
    }

    def edge(source, target, kind, scope='DIRECT', *, source_event='COMPLETE', target_event='BEGIN', **parameters):
        return dict(source=source, target=target, kind=kind, scope=scope,
                    source_event=source_event, target_event=target_event, **parameters)

    edges = [
        edge('launch', 'ingress', 'PROCESS_ENTRY'),
        edge('ingress', 'team_setup', 'SUCCESSFUL_ENTRY_GUARDS'),
        edge('team_setup', 'worker_setup', 'WORKER_ACTIVATED', 'SAME_WORKER', source_event='WORKER_ACTIVATED'),
        edge('worker_setup', 'task_ingress', 'ALL_WORKERS_READY', 'ALL_WORKERS'),
        edge('task_ingress', 'task_prepare', 'TASK_VALUES_AVAILABLE', 'SAME_TASK'),
        edge('task_prepare', 'task_allocate', 'PREPARED_VALUES_READY', 'SAME_TASK'),
        edge('task_allocate', 'numerical_body', 'ARENA_AVAILABLE', 'SAME_TASK'),
        edge('numerical_body', 'task_result', 'FINAL_ROUND_VALUE_AVAILABLE', 'SAME_TASK'),
        edge('task_result', 'result_handoff', 'RESULT_SLOT_VALUE_AVAILABLE', 'SAME_TASK'),
        edge('result_handoff', 'join', 'ALL_RESULTS_BEFORE_SUCCESSFUL_TEAM_RETURN', 'ALL_TASKS',
             target_event='COMPLETE'),
        edge('join', 'reduction', 'TEAM_SUCCESS'),
        edge('reduction', 'file_publication', 'REDUCED_OUTPUT_AVAILABLE'),
        edge('file_publication', 'stdout_publication', 'FILE_CLOSE_BEFORE_STDOUT_STATEMENTS'),
        edge('stdout_publication', 'teardown', 'CPP_MAIN_RETURN_AFTER_STREAM_STATEMENTS'),
    ]
    # This edge is between successive instances, never a same-task cycle. The
    # process lane has one in-flight request; the owner issues the next window
    # slot after retiring the preceding ordinal. OpenMP has no per-wave barrier.
    edges.append(edge('result_handoff', 'task_ingress', 'NEXT_TASK_ON_SAME_LANE', 'SAME_WORKER',
                      ordinal_delta=workers))
    if not openmp:
        edges.append(edge('result_handoff', 'result_handoff', 'ORDERED_OWNER_RETIREMENT', 'ORDINAL_ADJACENT',
                          source_event='OWNER_RETIRE_COMPLETE', target_event='OWNER_RETIRE_BEGIN', ordinal_delta=1))
    return dict(runtime=package_ref, origin_runtime=runtime_ref, mode=mode, sources=sources,
        executable=package['files'][package['binary']],
        resources=resources, operations=operations, edges=edges,
        task_instances=dict(ordinal_count=count, lane_by='ORDINAL_MODULO_WORKER_COUNT', worker_count=workers,
                            in_flight_window=placement['window']),
        overlap=dict(cross_lane_numeric_serialization_proven=False,
                     process_retirement_may_overlap_other_lanes=not openmp,
                     worker_setup_may_overlap_team_creation=True,
                     per_wave_barrier_present=False, launch_and_join_wall_intervals_proven=False),
        coverage_frontier=['LOADER_RUNTIME_INIT_DURATION_OPEN', 'PHYSICAL_FIRST_TOUCH_AND_NUMA_PLACEMENT_OPEN',
            'WORKER_SETUP_AND_JOIN_DURATION_OPEN', 'INTERFERENCE_AND_OVERLAP_DURATION_OPEN',
            'PUBLICATION_DURABILITY_AND_STDOUT_DRAIN_OPEN', 'PROCESS_TEARDOWN_DURATION_OPEN', 'FAILURE_PATHS_OPEN'])


def derive_cpu_array_native_route_lifecycle(proof_ref):
    """Bind mandatory normal-success operation order to native source and placement."""
    with replay_transaction():
        proof = verify_cpu_array_native_route(proof_ref)
        return _project(proof_ref, proof)


def _project(proof_ref, proof):
    _need(proof['status'] == 'PROVEN_SCOPED_CPU_ARRAY_NATIVE_ROUTE', 'qualified native routes required')
    workload, resources, scope = (proof[key] for key in ('workload', 'resources', 'scope'))
    recipe = read_artifact(workload['recipe'])
    contract_ref = recipe['source_contract']
    contract = read_artifact(contract_ref)
    report = contract['report']
    _need(contract['entry'] == workload['source'] and report['stdout']['kind'] == 'JSON_TEXT_PRINT'
          and report['path_output']['kind'] == 'CONDITIONAL_PATH_WRITE_TEXT'
          and workload['argv'][workload['output_index']], 'expected the admitted nonempty output-file route')
    local, grouped = (proof[key] for key in ('retained_runtime', 'grouped_runtime'))
    routes = {
        'retained_process': _route(local, local, resources['local_processes'], recipe, scope['task_count']),
        'grouped_openmp': _route(grouped, proof['baseline_grouped'], resources['openmp_grouped'], recipe, scope['task_count']),
    }
    return dict(schema=SCHEMA, schema_version=1, proof=proof_ref, scope=scope, hardware=resources['hardware'],
        original_source_contract=contract_ref, publication=dict(file='#/report/path_output', stdout='#/report/stdout'),
        routes=routes, status='PROVEN_SCOPED_NORMAL_SUCCESS_SOURCE_LIFECYCLE',
        shared_phase_duration_equality_proven=False, performance_bounds_proven=False,
        target_executed=False, transformation_authorized=False)


def verify_cpu_array_native_route_lifecycle(reference):
    _need(reference['schema'] == SCHEMA, 'unexpected lifecycle receipt schema')
    saved = read_artifact(reference)
    _need(_canonical(saved) == _canonical(derive_cpu_array_native_route_lifecycle(saved['proof'])),
          'saved lifecycle differs on cold replay')
    return saved
