"""Replay a read-only physical loop carry prefix in the seeded-array backend."""

from dataclasses import dataclass, replace

from scar.analysis.cpu_task_arena_reuse import verify_cpu_task_arena_reuse
from scar.codegen import native_numpy_complex
from scar.ir.execution_recipe import read_artifact


SCHEMA = 'scar.cpu-loop-carry-redirect'
_SUCCESS = 'PROVEN_SCOPED_NORMAL_SUCCESS'
_ITEM = 16
_ALIGNMENT = 32
# Reviewed exact X86_64_V3 body. On disjoint (16, 0, 16) spans and a
# positive multiple of four, its overlap guards select the unaligned 4-element
# broadcast-right vector path. Address ordering cannot select scalar math.
_MULTIPLY = 'fabaecfe360d157644fde49dc5b170af2f93c37069c25f076ed96043587220cb'


def _need(ok, detail):
    if not ok:
        raise ValueError('CPU loop carry redirect: ' + detail)


@dataclass(frozen=True, slots=True)
class _Operand:
    root: str
    offset: int
    count: int
    stride: int
    width: int = _ITEM

    @property
    def end(self):
        return self.offset + (self.count - 1) * self.stride + self.width

    def overlaps(self, other):
        # Bounding spans make this a conservative alias check for strided views.
        return self.offset < other.end and other.offset < self.end


@dataclass(frozen=True, slots=True)
class _Step:
    operation: int
    compiled_step: int
    kind: str
    reads: tuple
    writes: tuple
    repeat: int = 1


def _steps(schedule, begin, stop):
    """The same operation-to-step accounting as the sealed seeded tape emitter."""
    offsets = schedule['storage']['offset_bytes']
    arena_size = schedule['storage']['arena_bytes']

    def operand(root, offset, count, stride, width=_ITEM):
        result = _Operand(root, offsets[root] + offset, count, stride, width)
        _need(count > 0 and stride >= 0 and width > 0 and 0 <= result.offset < result.end <= arena_size,
              'unsupported physical operand span')
        return result

    compiled = 0
    for index, row in enumerate(schedule['operations'][:stop]):
        kind = row['kind']
        extent = len(row['groups']) if kind == 'COPY' else 1
        if index >= begin:
            if kind == 'COPY':
                for ordinal, part in enumerate(row['groups']):
                    read = operand(row['source'], part['read'], part['count'], part['increments'][0])
                    write = operand(row['target'], part['write'], part['count'], part['increments'][1])
                    yield _Step(index, compiled + ordinal, kind, (read,), (write,))
            elif kind in ('MULTIPLY', 'ADD'):
                group = row['group']
                repeat = group.get('repeat', dict(count=1, offsets=[0, 0, 0]))
                for repetition in range(repeat['count']):
                    first, second, output = (operand(part['root'], part['offset'] + repetition * advance,
                        group['count'], part['stride'])
                        for part, advance in zip(group['operands'], repeat['offsets'], strict=True))
                    yield _Step(index, compiled, kind, (first, second), (output,), repeat['count'])
            elif kind == 'INITIAL_BASIS':
                out = operand(row['target'], row['offset'], row['elements'], _ITEM)
                yield _Step(index, compiled, kind, (), (out,))
            elif kind == 'LOAD_LITERAL':
                out = operand(row['target'], 0, 1, 0, row['size'])
                yield _Step(index, compiled, kind, (), (out,))
            else:
                raise ValueError('CPU loop carry redirect: unqualified prefix operation ' + kind)
        compiled += extent


def _prefix(schedule, periodic):
    carry = periodic['carry_into_steady']
    destination = _Operand(carry['destination_root'], carry['destination_offset'], 1, 0, carry['bytes'])
    begin, _ = periodic['steady']['operations']
    _need(begin == carry['operation'] and carry['before_any_other_steady_operation']
          and schedule['storage']['offset_bytes'][destination.root] == destination.offset
          and schedule['storage']['lifetimes'][destination.root][0] == begin,
          'steady phase has no exact carry ingress')
    handoff = next(_steps(schedule, begin, begin + 1))
    _need(handoff.kind == 'COPY' and len(schedule['operations'][begin]['groups']) == 1
          and handoff.writes[0].root == destination.root and handoff.writes[0].offset == destination.offset
          and handoff.writes[0].end == destination.end and handoff.writes[0].stride == _ITEM
          and handoff.reads[0].offset == carry['static_source_offset']
          and handoff.reads[0].stride == _ITEM and handoff.reads[0].end == carry['static_source_offset'] + carry['bytes'],
          'original carry does not initialize exactly the destination')
    # Only this prefix can consume the carried version. The first write must
    # replace the entire physical destination before any unchanged later read.
    prefix, redirects = [], []
    for step in _steps(schedule, begin + 1, periodic['steady']['normalize_operation']):
        for ordinal, read in enumerate(step.reads):
            if not destination.overlaps(read):
                continue
            _need(read.root == destination.root and destination.offset <= read.offset < read.end <= destination.end
                  and step.repeat == 1, 'overlapping logical root or partial/repeated carry operand')
            redirects.append((step, ordinal))
        writing = [out for out in step.writes if destination.overlaps(out)]
        if not writing:
            prefix.append(step)
            continue
        _need(len(writing) == 1 and writing[0].root == destination.root
              and writing[0].offset == destination.offset and writing[0].end == destination.end
              and writing[0].stride == _ITEM and writing[0].width == _ITEM
              and step.repeat == 1 and all(not destination.overlaps(read) for read in step.reads)
              and (step.kind != 'COPY' or len(schedule['operations'][step.operation]['groups']) == 1),
              'first destination write is partial or depends on its old contents')
        _need(redirects, 'prefix contains no carry consumer')
        return destination, handoff, prefix, redirects, step
    raise ValueError('CPU loop carry redirect: destination never completely replaced by a qualified prefix')


def _disjoint(operands):
    return all(not a.overlaps(b) for index, a in enumerate(operands) for b in operands[index + 1:])


def _redirect_safe(step, ordinal, read):
    source = (*step.reads[:ordinal], read, *step.reads[ordinal + 1:])
    operands = (*source, *step.writes)
    if not _disjoint(operands):
        return False
    if step.kind == 'COPY':
        return ordinal == 0 and tuple(row.stride for row in operands) == (_ITEM, _ITEM)
    return (step.kind == 'MULTIPLY' and ordinal == 0 and step.repeat == 1
            and tuple(row.stride for row in operands) == (_ITEM, 0, _ITEM)
            and len({row.count for row in operands}) == 1 and read.count % 4 == 0
            and all(row.offset % 8 == 0 for row in operands))


def _route(row, destination, prefix, redirects, arena_bytes):
    result = dict(row, redirect_eligible=False)
    offset = row['source_offset']
    if row['mode'] == 'TRANSFER_SAME_SPAN':
        return dict(result, reason='ORIGINAL_SAME_SPAN_TRANSFER')
    if (row['mode'] != 'MEMCPY' or not 0 <= offset <= arena_bytes - row['bytes']
            or offset % _ALIGNMENT != destination.offset % _ALIGNMENT):
        return dict(result, reason='UNQUALIFIED_CARRY_SOURCE_ALIGNMENT_OR_BOUNDS')
    source = replace(destination, offset=offset)
    through_last_read = [step for step in prefix if step.compiled_step <= redirects[-1][0].compiled_step]
    if any(source.overlaps(out) for step in through_last_read for out in step.writes):
        return dict(result, reason='SELECTED_SOURCE_CAN_BE_CLOBBERED_BEFORE_LAST_REDIRECT')
    for step, index in redirects:
        original = step.reads[index]
        shifted = replace(original, offset=source.offset + original.offset - destination.offset)
        if not (_redirect_safe(step, index, original) and _redirect_safe(step, index, shifted)):
            return dict(result, reason='ALIAS_OR_PROVIDER_OPERAND_CONTRACT_OPEN')
    return dict(result, redirect_eligible=True, reason='SCOPED_READ_ONLY_SAME_VALUES_AND_PROVIDER')


def _provider(runtime, tape_ref):
    recipe = read_artifact(runtime['recipe'])
    profile, = (row for row in recipe['profiles'] if row['tape_sha256'] == tape_ref['sha256'])
    package = read_artifact(profile['package'])
    binary = runtime['files'][runtime['binary']]
    stock = native_numpy_complex.audit(binary['uri'], package['stock_provider'])
    multiply = stock['functions']['multiply']
    _need(stock == runtime['linkage']['stock'] and multiply['normalized_sha256'] == _MULTIPLY
          and all(row['kind'] == 'LITERAL' for row in multiply['transfers']),
          'linked multiply is outside the reviewed disjoint contiguous broadcast-right path')
    return dict(binary=binary, variable_alignment_bytes=_ALIGNMENT, function=multiply,
        contract='DISJOINT_CONTIGUOUS_BROADCAST_RIGHT_COUNT_MULTIPLE_OF_FOUR',
        strides_bytes=[_ITEM, 0, _ITEM], preserves_fp_operations=True)


def derive_cpu_loop_carry_redirect(arena_proof_ref):
    """Identify carry reads and qualify placement using the original bytes."""
    arena_proof = verify_cpu_task_arena_reuse(arena_proof_ref)
    _need(arena_proof['status'] == _SUCCESS
          and arena_proof['proof']['every_arena_read_current_task_initialized'] is True
          and arena_proof['proof']['arena_pointer_does_not_escape_task'] is True,
          'original physical initialization and non-escape required')
    runtime = read_artifact(arena_proof['runtime'])
    storage = arena_proof['evidence']['storage']
    periodic = read_artifact(storage['periodic_memory'])
    schedule = read_artifact(storage['schedule'])
    destination, handoff, prefix, consumers, rewrite = _prefix(schedule, periodic)
    provider = _provider(runtime, storage['original_tape']) if any(
        step.kind == 'MULTIPLY' for step, _ in consumers) else None
    routes = [_route(row, destination, prefix, consumers, arena_proof['arena']['bytes'])
              for row in storage['carry_routes']]
    redirects = []
    for step, index in consumers:
        value = step.reads[index]
        redirects.append(dict(operation=step.operation, compiled_step=step.compiled_step, kind=step.kind,
            operand=0 if step.kind == 'COPY' else index, relative_offset=value.offset - destination.offset,
            count=value.count, stride=value.stride, item_bytes=value.width))
    eligible = any(row['redirect_eligible'] for row in routes)
    return dict(schema=SCHEMA, schema_version=1, status=_SUCCESS, arena_proof=arena_proof_ref,
                runtime=arena_proof['runtime'], scope=arena_proof['scope'], arena=arena_proof['arena'],
                worker_cpu_ids=arena_proof['worker_cpu_ids'],
                evidence=dict(schedule=storage['schedule'], periodic_memory=storage['periodic_memory'],
                    runtime_members=arena_proof['evidence']['runtime_members'], provider=provider),
                carry=dict(operation=handoff.operation, compiled_step=handoff.compiled_step,
                    bytes=destination.width, destination_offset=destination.offset, destination_root=destination.root,
                    last_redirect_operation=consumers[-1][0].operation,
                    first_destination_write_operation=rewrite.operation, first_destination_write_compiled_step=rewrite.compiled_step,
                    redirects=redirects, routes=routes),
                proof=dict(all_unwritten_destination_reads_accounted=True,
                    source_stable_through_last_redirect_read=eligible, write_sink_unchanged=True,
                    unchanged_publication_and_provider_guards=True,
                    unqualified_route_retains_original_handoff=True, fixed_round_scope=True),
                target_executed=False, transformation_authorized=False, deployment_authorized=False)


def verify_cpu_loop_carry_redirect(reference):
    saved = read_artifact(reference)
    _need(reference['schema'] == SCHEMA and saved == derive_cpu_loop_carry_redirect(saved['arena_proof']),
          'stored physical carry proof differs from cold replay')
    return saved
