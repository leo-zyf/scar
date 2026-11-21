"""Prove coefficient-hook delivery and phase-local literal equality in the native tape."""

from dataclasses import dataclass
import re

from scar.analysis import cpu_arithmetic_dispatch as arithmetic
from scar.analysis.cpu_fp_guard_groups import _ARITHMETIC, _HOOK, _table
from scar.analysis.replay_transaction import detached_replay
from scar.ir.execution_recipe import read_artifact


SCHEMA = 'scar.cpu-coefficient-dispatch-proof'
_ITEM = 16
_COEFFICIENT_BYTES = 64
_LITERAL_HEADER = re.compile(
    r'^constexpr std::array<std::array<unsigned char, 64>, (\d+)> coefficients\{\{\n', re.M)
_LITERAL = re.compile(r'    \{(0x[\da-f]{2}(?:, 0x[\da-f]{2})*)\},')
_COUNT = re.compile(r'^constexpr std::size_t coefficient_count = (\d+);$', re.M)
_STATE_BYTES = re.compile(r'^constexpr std::size_t state_bytes = (\d+);$', re.M)
_PRODUCTION_ANCHORS = (
    'using Coefficient = std::array<double, 8>;',
    'const std::array<Coefficient, coefficient_count>& prepared,',
    'if (!execute_round<true>(base, cursor, hook->step, false, unused)) return nullptr;',
    'std::memcpy(base + step.offset[2], prepared[hook->index].data(), step.count);',
    'if (selected == 0) return raw;',
    'if (selected == 1) return scratch;',
    'if (!prepare_coefficients(seed, prepared)) return 1;',
)


def _need(ok, detail):
    if not ok:
        raise ValueError('CPU coefficient dispatch: ' + detail)


@dataclass(frozen=True, slots=True)
class _Write:
    offset: int
    count: int
    stride: int
    width: int
    repeat: int = 1
    advance: int = 0

    @property
    def extent(self):
        return (self.count - 1) * self.stride + self.width

    @property
    def end(self):
        return self.offset + (self.repeat - 1) * self.advance + self.extent

    def intersects(self, lo, hi):
        """Intersect exact byte lanes without allocating a repeat-expanded tape."""
        if hi <= self.offset or self.end <= lo:
            return False
        if self.advance:
            first = max(0, (lo - self.offset - self.extent) // self.advance + 1)
            last = min(self.repeat - 1, (hi - 1 - self.offset) // self.advance)
        else:
            first = last = 0
        for repetition in range(first, last + 1):
            base = self.offset + repetition * self.advance
            if self.stride <= self.width:
                if base < hi and lo < base + self.extent:
                    return True
            else:
                first_item = max(0, (lo - base - self.width) // self.stride + 1)
                last_item = min(self.count - 1, (hi - 1 - base) // self.stride)
                if first_item <= last_item:
                    return True
        return False


def _write(step, arena_bytes):
    destination = step.offsets[2]
    if step.operator == 'literal':
        region = _Write(destination, 1, 0, step.count)
    elif step.operator == 'basis':
        _need(0 <= step.argument < step.count, 'basis scalar escapes its initialized output')
        region = _Write(destination, step.count, _ITEM, _ITEM)
    elif step.operator == 'copy':
        region = _Write(destination, step.count, step.strides[2], _ITEM)
    elif step.operator in _ARITHMETIC:
        region = _Write(destination, step.count, step.strides[2], _ITEM, step.repeat, step.advances[2])
    else:
        raise ValueError('CPU coefficient dispatch: unqualified arena writer ' + step.operator)
    _need(region.count > 0 and region.repeat > 0 and region.stride >= 0 and region.advance >= 0
          and 0 <= destination < region.end <= arena_bytes, 'arena write geometry is outside its proof')
    return region


def _literal_table(text):
    headers = list(_LITERAL_HEADER.finditer(text))
    _need(len(headers) == 1, 'original constant coefficient table required')
    header = headers[0]
    body, terminal, _ = text[header.end():].partition('\n}};')
    matches = [_LITERAL.fullmatch(line) for line in body.splitlines()]
    _need(bool(terminal) and len(matches) == int(header[1]) and all(matches), 'invalid generated literal table')
    output = [bytes(int(item, 16) for item in match[1].split(', ')) for match in matches]
    _need(all(0 < len(value) <= _COEFFICIENT_BYTES for value in output), 'literal exceeds its native coefficient slot')
    return output


def _profile(profile):
    count, = map(int, _COUNT.findall(profile))
    size, = map(int, _STATE_BYTES.findall(profile))
    _need(count > 0 and size > 0 and all(profile.count(anchor) == 1 for anchor in _PRODUCTION_ANCHORS)
          and profile.count('if (state_prefix) captured.emplace_back(state, state + state_bytes);') == 2,
          'profile does not preserve immutable prepared copies and selected checkpoint replay')
    hooks = [(int(step), kind, int(slot)) for step, kind, slot in _table(profile, 'constexpr Hook hooks[]{\n', _HOOK)]
    coordinates = [step for step, _, _ in hooks]
    _need(coordinates == sorted(set(coordinates)), 'runtime hooks are not in strictly increasing original order')
    return count, size, hooks


def _normalization(phase, periodic, steps, size):
    number, checkpoint = phase['normalization_compiled_step'], phase['checkpoint_compiled_step']
    normal, captured = steps[number], steps[checkpoint]
    row = periodic[phase['phase'].lower()]
    raw, scratch = normal.offsets[0], normal.offsets[2]
    _need(normal.operator == 'normalize' and normal.count * _ITEM == size and normal.repeat == 1
          and captured.operator == 'checkpoint' and captured.count == size and captured.repeat == 1
          and checkpoint == number + 1 and checkpoint + 1 == phase['end_compiled_step']
          and [(choice['status'], choice['offset'], choice['bytes']) for choice in row['choices']]
          == [(0, raw, size), (1, scratch, size)]
          and captured.offsets[0] in (raw, scratch) and periodic['all_normalizer_statuses'] == [0, 1]
          and not periodic['declined_state_is_published'], 'original selected checkpoint and normalize ABI differ')
    return dict(raw_offset=raw, scratch_offset=scratch, bytes=size,
                choices=[dict(status=0, kind='RAW', offset=raw), dict(status=1, kind='SCRATCH', offset=scratch)],
                decline_statuses=periodic['decline_statuses'])


def _phase(phase, steps, origins, schedule, periodic, hooks, literals, prepared_count, state_bytes):
    first, normalize = phase['begin_compiled_step'], phase['normalization_compiled_step']
    normal = _normalization(phase, periodic, steps, state_bytes)
    shape, static, live = [], [], {}
    repeated_dynamic = 0
    storage = schedule['storage']
    for index in range(first, normalize):
        step = steps[index]
        write = _write(step, storage['arena_bytes'])
        literal = None
        if step.operator == 'literal':
            original = schedule['operations'][origins[index]]
            _need(original['kind'] == 'LOAD_LITERAL' and original['literal'] == step.argument
                  and original['size'] == step.count and storage['offset_bytes'][original['target']] == write.offset
                  and step.repeat == 1 and 0 <= step.argument < len(literals)
                  and 0 < step.count <= _COEFFICIENT_BYTES, 'coefficient step differs from its original literal source')
            if index in hooks:
                kind, slot = hooks[index]
                _need(kind == 'coefficient' and 0 <= slot < prepared_count, 'invalid prepared coefficient route')
                shape.append(dict(compiled_step=index, kind='COEFFICIENT', slot=slot,
                    original_literal_index=step.argument, destination_offset=write.offset, bytes=step.count))
                literal = ('PREPARED', write.offset, step.count, slot)
            else:
                original_bytes = bytes.fromhex(schedule['literals'][step.argument])
                _need(len(original_bytes) == step.count and literals[step.argument] == original_bytes,
                      'static literal bytes differ from the sealed schedule')
                literal = ('STATIC', write.offset, step.count, step.argument)
            producer = live.get(literal)
            if producer is not None:
                if literal[0] == 'STATIC':
                    static.append(dict(compiled_step=index, producer_step=producer, literal_index=step.argument,
                                       bytes=step.count, destination_offset=write.offset))
                else:
                    repeated_dynamic += 1
                continue
        else:
            _need(index not in hooks, 'coefficient hook crosses another tape operation')
        live = {key: provider for key, provider in live.items()
                if not write.intersects(key[1], key[1] + key[2])}
        if literal is not None:
            live[literal] = index
    _need(hooks.get(normalize) == ('normalization', 0), 'phase lacks one original normalizer fence')
    shape.append(dict(compiled_step=normalize, kind='NORMALIZATION', slot=None, original_literal_index=None,
                      destination_offset=normal['scratch_offset'], bytes=normal['bytes']))
    return dict(phase, hooks=shape, elidable_literal_writes=static, normalization=normal,
                redundant_runtime_coefficient_writes=repeated_dynamic)


def derive_cpu_coefficient_dispatch(baseline_ref):
    """Read exact immutable source, preserve every normalizer and all runtime coefficient copies."""
    identity = tuple(baseline_ref[key] for key in ('schema', 'uri', 'sha256', 'size_bytes'))
    return detached_replay('scar.cpu-coefficient-dispatch.v1', identity, lambda: _derive(baseline_ref))


def _derive(baseline_ref):
    checked = arithmetic.derive_cpu_arithmetic_dispatch(baseline_ref)
    baseline = read_artifact(baseline_ref)
    runtime_ref = checked['runtime']
    runtime = read_artifact(runtime_ref)
    original = read_artifact(runtime['native'])
    evidence = checked['evidence']
    schedule = read_artifact(evidence['schedule'])
    periodic = read_artifact(evidence['periodic_memory'])
    steps, _ = arithmetic._steps(evidence['baseline_tape'])
    original_steps, _ = arithmetic._steps(evidence['original_tape'])
    _, origins = arithmetic._schedule(schedule, original_steps)
    profile = read_artifact(evidence['baseline_profile'], json_document=False).decode('ascii')
    literal_table = _literal_table(read_artifact(evidence['baseline_tape'], json_document=False).decode('ascii'))
    count, size, raw_hooks = _profile(profile)
    _need(count == original['prepared_runtime_coefficients'] and checked['baseline'] == baseline_ref
          and checked['scope'] == baseline['scope'] and
          all(checked['proof'][key] is True for key in ('guard_bits_preserved',
              'arithmetic_provider_calls_and_order_preserved', 'private_probe_unchanged_required')),
          'coefficient dispatch lacks the exact original grouped arithmetic scope')
    hooks = {step: (kind, slot) for step, kind, slot in raw_hooks}
    phases = [_phase(phase, steps, origins, schedule, periodic, hooks, literal_table, count, size)
              for phase in checked['phases']]
    _need([hook['compiled_step'] for phase in phases for hook in phase['hooks']] == list(hooks),
          'some original hooks have no active phase or cross its boundaries')
    summary = dict(prepared_slots=count, state_bytes=size, runtime_coefficient_hooks=sum(
        hook['kind'] == 'COEFFICIENT' for phase in phases for hook in phase['hooks']),
        normalizer_fences=len(phases), elidable_literal_writes=sum(len(phase['elidable_literal_writes'])
            for phase in phases), redundant_runtime_coefficient_writes=sum(
            phase['redundant_runtime_coefficient_writes'] for phase in phases))
    return dict(schema=SCHEMA, schema_version=1, status='PROVEN_SCOPED_COEFFICIENT_DISPATCH',
        baseline=baseline_ref, runtime=runtime_ref, scope=checked['scope'], phases=phases, summary=summary,
        evidence=evidence, proof=dict(prepared_immutable_within_task=True, runtime_coefficient_writes_preserved=True,
            normalizer_fences_preserved=True, checkpoint_publication_unchanged=True,
            static_literal_equivalence_byte_exact=True, sticky_guard_bits_unchanged=True,
            original_arithmetic_order_and_geometry_unchanged=True, private_probe_unchanged_required=True,
            literal_equivalence_scope='BYTE_EQUAL_PRIVATE_DESTINATION_WITH_EXACT_WRITES_AND_PHASE_RESET'),
        target_executed=False, transformation_authorized=False, deployment_authorized=False)


def verify_cpu_coefficient_dispatch(reference):
    saved = read_artifact(reference)
    _need(reference['schema'] == SCHEMA and saved == derive_cpu_coefficient_dispatch(saved['baseline']),
          'saved coefficient dispatch proof differs on cold replay')
    return saved
