"""Prove same-lane arena ownership from the admitted native source and storage graph."""

from collections import Counter
from hashlib import sha256

from scar.codegen import native_seeded_array_task_entry as generator
from scar.ir.execution_recipe import artifact_reference, read_artifact


SCHEMA = 'scar.cpu-task-arena-reuse'
_REVIEWED_TEMPLATES = {
    'native_seeded_array_task_entry.cpp.in': '7726c77f8915ab27033b8f72831f07756adedbd5bc7fb8ccace953f6d13fde84',
    'native_ordered_openmp_v1.h.in': 'e2f4363febe6a2eca8abd623d9d099cd33b5fae15a7a5452ba8d1f741ce49759',
}
_ALIGNMENT = 64
_ITEM = 16
_SCOPE = 'GENERATED_CPP20_FIXED_OPENMP_NORMAL_SUCCESS'
_STATUS = 'PROVEN_SCOPED_NORMAL_SUCCESS'


def _need(ok, detail):
    if not ok:
        raise ValueError('CPU task arena reuse: ' + detail)


def _span(mask, base, count, stride, width, *, write=False):
    """Check exactly the byte lanes consumed by the lowered physical operands."""
    _need(count > 0 and width > 0 and stride >= 0 and
          0 <= base <= base + (count - 1) * stride + width <= len(mask),
          'operand escapes the arena or has unsupported physical geometry')
    if stride <= width:
        start, stop = base, base + (count - 1) * stride + width
        if write:
            mask[start:stop] = b'\1' * (stop - start)
        else:
            _need(0 not in mask[start:stop], 'arena read is not initialized in this task')
    else:
        for lane in range(width):
            start, stop = base + lane, base + lane + (count - 1) * stride + 1
            if write:
                mask[start:stop:stride] = b'\1' * count
            else:
                _need(0 not in mask[start:stop:stride], 'strided arena read is not initialized in this task')


def _primitive(mask, op, offsets):
    kind = op['kind']
    if kind == 'INITIAL_BASIS':
        _span(mask, offsets[op['target']] + op['offset'], op['elements'], _ITEM, _ITEM, write=True)
    elif kind == 'LOAD_LITERAL':
        _span(mask, offsets[op['target']], 1, 0, op['size'], write=True)
    elif kind == 'COPY':
        for row in op['groups']:
            old_stride, new_stride = row['increments']
            _span(mask, offsets[op['source']] + row['read'], row['count'], old_stride, _ITEM)
            _span(mask, offsets[op['target']] + row['write'], row['count'], new_stride, _ITEM, write=True)
    elif kind in ('ADD', 'MULTIPLY'):
        group = op['group']
        repeat = group.get('repeat', dict(count=1, offsets=[0, 0, 0]))
        for ordinal in range(repeat['count']):
            for index, (operand, advance) in enumerate(zip(group['operands'], repeat['offsets'], strict=True)):
                offset = offsets[operand['root']] + operand['offset'] + advance * ordinal
                _span(mask, offset, group['count'], operand['stride'], _ITEM, write=index == 2)
    else:
        raise ValueError('CPU task arena reuse: unknown arena operand effect for ' + kind)


def _phase(phase, schedule, *, carry=None):
    start, end = phase['operations']
    steps = schedule['operations']
    offsets = schedule['storage']['offset_bytes']
    mask = bytearray(schedule['storage']['arena_bytes'])
    initial_bytes = 0
    if carry is not None:
        initial_bytes = carry['bytes']
        _need(start == carry['operation'] and steps[start]['kind'] == 'COPY',
              'steady input must be the selected prior state of this task')
        _span(mask, carry['destination_offset'], 1, 0, initial_bytes, write=True)
        start += 1
    normalize = phase['normalize_operation']
    terminal = phase['checkpoint_operation']
    _need(start < normalize and terminal == normalize + 1 and end == terminal + 1
          and steps[normalize]['kind'] == 'NORMALIZE' and steps[terminal]['kind'] == 'CHECKPOINT',
          'unexpected operations between normalization and terminal consumers')
    for op in steps[start:normalize]:
        _primitive(mask, op, offsets)
    row = steps[normalize]
    src, dst = (offsets[row[key]['storage']] + row[key]['offset_bytes'] for key in ('source', 'target'))
    size = row['elements'] * _ITEM
    _span(mask, src, 1, 0, size)
    choices = phase['choices']
    _need([(item['status'], item['offset'], item['bytes']) for item in choices] ==
          [(0, src, size), (1, dst, size)] and steps[terminal]['bytes'] == size
          and (src + size <= dst or dst + size <= src),
          'normalization success statuses do not name the exact distinct output views')
    # The reviewed normalizer writes the full destination before it can return 1.
    # On status 0 only the known input is usable; do not count scratch writes.
    _span(mask, dst, 1, 0, size, write=True)
    _span(mask, dst, 1, 0, size)
    return dict(primitive_operation_begin=start, normalize_operation=normalize,
                checkpoint_operation=terminal, initialized_entry_bytes=initial_bytes,
                operation_counts=dict(Counter(op['kind'] for op in steps[start:normalize])),
                normal_returns=[dict(status=choice['status'], offset=choice['offset'], bytes=size)
                                for choice in choices])


def _storage(native):
    periodic = read_artifact(native['periodic_memory'])
    memory = read_artifact(native['memory'])
    schedule = read_artifact(memory['schedule'])
    size = native['arena_bytes']
    _need(type(size) is int and 0 < size <= 2**28 and size % _ALIGNMENT == 0
          and periodic['arena_bytes'] == schedule['storage']['arena_bytes'] == size
          and periodic['memory'] == native['memory'] and schedule['shadow_state_is_private']
          and schedule['preserved_source_effect_order'] and periodic['first']['witnessed_live_in'] == [],
          'empty first entry and matching aligned arena are required')
    carry = periodic['carry_into_steady']
    _need(carry['before_any_other_steady_operation'] and carry['requires_selected_pointer_for_checkpoint_and_terminal']
          and periodic['all_normalizer_statuses'] == [0, 1]
          and periodic['declined_state_is_published'] is False,
          'unproven periodic predecessor or terminal route')
    first = _phase(periodic['first'], schedule)
    steady = _phase(periodic['steady'], schedule, carry=carry)
    expected = {(kind, row['status']): (row['offset'], row['bytes'])
                for kind, phase in (('FIRST', first), ('STEADY', steady))
                for row in phase['normal_returns']}
    _need(len(carry['routes']) == len(expected), 'missing status-specific carry route')
    for row in carry['routes']:
        predecessor = expected.pop((row['predecessor'], row['status']), None)
        offset = row['source_offset']
        dst, size = carry['destination_offset'], carry['bytes']
        same = offset == dst
        _need(predecessor == (offset, size) and row['destination_offset'] == dst
              and row['bytes'] == size and not row['independent_live_versions']
              and row['mode'] == ('TRANSFER_SAME_SPAN' if same else 'MEMCPY')
              and (same or offset + size <= dst or dst + size <= offset),
              'prior return does not initialize the exact steady carry span')
    return dict(first=first, steady=steady, carry_routes=carry['routes'], schedule=memory['schedule'],
                periodic_memory=native['periodic_memory'], original_tape=memory['tape'])


def _checked_runtime(runtime_ref):
    runtime = read_artifact(runtime_ref)
    decision = runtime.get('decision', {})
    _need(runtime_ref['schema'] == generator.SCHEMA and runtime.get('schema_version') == 1
          and runtime.get('cxx_standard') == 20 and decision.get('mode') == 'ORDERED_OPENMP'
          and runtime['explicit_checkpoint_diagnostics'] is True and runtime['preserves_report_schema'] is True
          and runtime['no_worker_visible_output'] is True,
          'ordered generated C++20 OpenMP provider required')
    templates = {}
    for path in (generator._TEMPLATE, generator._OPENMP):
        _need(sha256(path.read_bytes()).hexdigest() == _REVIEWED_TEMPLATES[path.name],
              'controller or fixed-lane lowering differs from reviewed semantics')
        templates[path.name] = artifact_reference(path, 'scar.cpp-audited-template')
    # This verifier also replays the source family, periodic/layout proof,
    # reviewed generated body and the exact single-thread private BLAS route.
    schedule_ref, plan = generator._schedule(runtime['schedule']['uri'])
    _need(schedule_ref == runtime['schedule'] and plan['scope'] == runtime['scope']
          and plan['native'] == runtime['native'] and plan['family'] == runtime['family']
          and plan['thread_contract'] == runtime['thread_contract'] and
          ((runtime['execution_control'] is None and decision == plan['decision']['selected'])
           or (runtime['execution_control'] == decision['mode'] and decision in plan['alternatives']
               and decision['cost'] is not None)) and not decision['capacity_reasons'],
          'runtime launch or body crosses the verified fixed team schedule')
    width, cpus = decision['worker_count'], decision['worker_cpu_ids']
    count = runtime['scope']['task_count']
    _need(1 < width == len(cpus) == len(set(cpus)) <= count and runtime['cpu_ids'] == decision['launch_cpu_ids']
          and runtime['launch_environment'] == {'OMP_STACKSIZE': '8M'}
          and all(plan['thread_domain'][key] == value for key, value in {
              'assignment': 'ORDINAL_MOD_WORKER_COUNT', 'publication': 'ORDERED_AFTER_ALL_LANES_JOIN',
              'shared_result_slots_disjoint': True}.items()),
          'no proven disjoint OpenMP lane assignment')
    native, family, recipe, thread = (read_artifact(runtime[key]) for key in ('native', 'family', 'recipe', 'thread_contract'))
    _need(runtime['recipe'] == native['recipe'] and runtime['numerical_body_reused'] == native['files']['src/profile_0.cpp']
          and family['state']['kind'] == 'PER_TASK_PRIVATE' and family['effective_seed']['operation'] == 'ADD'
          and family['publication']['after_all_workers'] is True
          and thread['effects']['rng'] == 'PER_TASK_PCG64_STATE', 'worker RNG, publication or body scope differs')
    _, generated = generator._render(recipe, native, family, decision, runtime['scope'], thread)
    files = runtime['files']
    _need(all(read_artifact(files[name], json_document=False) == source for name, source in generated.items())
          and files[runtime['binary']]['schema'] == 'scar.native-project-member',
          'generated controller, private kernel or its members differ from replay')
    _need(runtime['linkage']['qualified_runtime_dependencies'] == {'libgomp.so.1': plan['openmp_runtime']}
          and runtime['linkage']['dynamic'] == generator.entry.cpu_executable_dynamic(files[runtime['binary']]['uri']),
          'runtime OpenMP linkage differs from the qualified schedule')
    for name in files.keys() - generated.keys():
        read_artifact(files[name], json_document=False)
    read_artifact(plan['openmp_runtime'], json_document=False)
    return runtime, native, templates


def derive_cpu_task_arena_reuse(runtime_ref):
    """Prove normal-success reuse without executing the workload."""
    runtime, native, templates = _checked_runtime(runtime_ref)
    storage = _storage(native)
    files = runtime['files']
    names = ('src/main.cpp', 'src/profile_0.cpp', 'src/tape_0.inc',
             'include/scar_generated/native_ordered_openmp.hpp', 'src/numpy_vector_normalize.cpp',
             'src/numpy_observables.cpp', 'src/numpy_random.cpp', runtime['binary'])
    return dict(schema=SCHEMA, schema_version=1, runtime=runtime_ref, scope=runtime['scope'],
                status=_STATUS, decision=runtime['decision'], worker_cpu_ids=runtime['decision']['worker_cpu_ids'],
                arena=dict(bytes=native['arena_bytes'], alignment_bytes=_ALIGNMENT),
                evidence=dict(runtime_members={name: files[name] for name in names}, templates=templates,
                              schedule=runtime['schedule'], thread_contract=runtime['thread_contract'],
                              original_task_family=runtime['family'], source_loop=native['loop'], storage=storage),
                proof=dict(scope=_SCOPE, every_arena_read_current_task_initialized=True,
                    arena_pointer_does_not_escape_task=True, lane_serial_and_disjoint=True,
                    per_task_rng_report_diagnostics_replayed=True,
                    first_allocation_failure_remains_at_original_task_boundary=True,
                    first_allocation_failure_scope='REQUIRED_EMITTER_OBLIGATION',
                    removed_repeated_allocator_failure_equivalent=False,
                    measurement_metadata_semantics='SCOPED_SCHEMA_ROLE_PRESERVED_VALUES_MAY_CHANGE',
                    diagnostic_environment_variables=['SCAR_ARRAY_ENTRY_STATES', 'SCAR_ARRAY_ENTRY_LANE_REPORT'],
                    diagnostic_contract='SITES_AND_PAYLOADS_PRESERVED_ON_NORMAL_SUCCESS',
                    emitter_obligations=['LAZY_FIRST_ALLOC_AT_ORIGINAL_TASK_BOUNDARY',
                        'ONE_ALIGNED_ARENA_PER_ORIGINAL_LANE_UNTIL_LANE_EXIT',
                        'PRESERVE_PER_TASK_SEED_AND_ALL_NUMERICAL_REPORT_AND_DIAGNOSTIC_SITES',
                        'PRESERVE_PROVIDER_AND_LAUNCH_GUARDS']),
                open_effects=['REMOVED_REPEATED_ALLOCATION_FAILURES_AND_FAILURE_PREFIXES',
                              'ALLOCATOR_CALLS_ADDRESSES_FREE_TIMING_AND_INTERPOSITION',
                              'OBSERVED_WALL_CPU_AND_PEAK_RSS_METADATA_VALUES'],
                target_executed=False, transformation_authorized=False, deployment_authorized=False)


def verify_cpu_task_arena_reuse(reference):
    saved = read_artifact(reference)
    _need(reference['schema'] == SCHEMA and saved == derive_cpu_task_arena_reuse(saved['runtime']),
          'stored reuse evidence differs from its cold replay')
    return saved
