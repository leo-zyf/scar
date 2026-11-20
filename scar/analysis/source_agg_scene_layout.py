"""Read typed AGG renderer arguments and compare aligned bounded observations."""

import argparse
from collections import Counter
import hashlib
import json
from math import prod
from pathlib import Path
import struct


SCHEMA = 'scar.source-agg-scene-layout'
_OBSERVATION = 'scar.agg-scene-observation'
_OPERATOR = ('clear', 'draw_path', 'draw_markers', 'draw_text_image', 'draw_path_collection')
_BUDGET = 64_000_000


def _span(start, end):
    return dict(start=start, end=end)


class _Tape:
    def __init__(self, data):
        if len(data) > _BUDGET:
            raise ValueError('AGG tape exceeds native byte budget')
        self.data = data
        self.offset = 0

    def take(self, size):
        start = self.offset
        if size < 0 or size > len(self.data) - start:
            raise ValueError('truncated AGG tape')
        self.offset += size
        return _span(start, self.offset)

    def values(self, fmt):
        codec = struct.Struct('<' + fmt)
        position = self.take(codec.size)
        return codec.unpack_from(self.data, position['start'])

    def scalar(self, fmt, dtype):
        position = self.take(struct.calcsize('<' + fmt))
        return dict(span=position, format=dtype)

    def array(self, rank, fmt):
        start = self.offset
        shape = self.values('I' * rank)
        if any(dimension > _BUDGET for dimension in shape):
            raise ValueError('AGG array dimension exceeds native budget')
        payload = self.take(prod(shape) * (8 if fmt == '<f8' else 1))
        return dict(span=_span(start, self.offset), payload_span=payload, shape=list(shape), format=fmt)

    def dashes(self):
        self.take(8)
        count, = self.values('I')
        if count > 128:
            raise ValueError('AGG dash pair count exceeds native budget')
        self.take(count * 16)

    def gc(self):
        start = self.offset
        self.take(struct.calcsize('<ddB4d4B4d'))
        self.dashes()
        return dict(span=_span(start, self.offset), format='graphics_state')

    def path(self):
        start = self.offset
        vertices = self.array(2, '<f8')
        codes = self.array(1, 'u1')
        simplify_start = self.offset
        simplify, threshold = self.values('Bd')
        if (prod(vertices['shape']) and vertices['shape'][1] != 2) or (
                codes['shape'][0] and codes['shape'][0] != vertices['shape'][0]):
            raise ValueError('invalid AGG path geometry')
        return dict(span=_span(start, self.offset), format='path',
                    vertices=vertices, codes=codes,
                    simplification=dict(span=_span(simplify_start, self.offset), enabled=bool(simplify),
                                        threshold=threshold))

    def collection(self, fields):
        fields['transform'] = self.scalar('6d', 'affine_2d')
        start = self.offset
        count, = self.values('I')
        if count > 16384:
            raise ValueError('AGG collection count exceeds native budget')
        paths = [self.path() for _ in range(count)]
        fields['paths'] = dict(span=_span(start, self.offset), count=count, format='path_table')
        fields['matrices'] = self.array(3, '<f8')
        fields['offsets'] = self.array(2, '<f8')
        fields['offset_transform'] = self.scalar('6d', 'affine_2d')
        for slot, rank in (('facecolors', 2), ('edgecolors', 2), ('widths', 1)):
            fields[slot] = self.array(rank, '<f8')
        start = self.offset
        count, = self.values('I')
        if count > 16384:
            raise ValueError('AGG line style count exceeds native budget')
        for _ in range(count):
            self.dashes()
        fields['line_styles'] = dict(span=_span(start, self.offset), count=count, format='dash_table')
        fields['antialias'] = self.array(1, 'u1')
        fields['hatchcolors'] = self.array(2, '<f8')
        return paths

    def command(self, ordinal):
        start = self.offset
        opcode, = self.values('B')
        if opcode >= len(_OPERATOR):
            raise ValueError('unsupported AGG tape operation')
        fields, paths = {}, []
        if opcode != 0:
            fields['graphics_state'] = self.gc()
        if opcode == 1:
            fields['path'] = self.path()
            fields['transform'] = self.scalar('6d', 'affine_2d')
            fields['facecolor'] = self.scalar('4d', 'rgba')
        elif opcode == 2:
            fields['marker'] = self.path()
            fields['marker_transform'] = self.scalar('6d', 'affine_2d')
            fields['path'] = self.path()
            fields['transform'] = self.scalar('6d', 'affine_2d')
            fields['facecolor'] = self.scalar('4d', 'rgba')
        elif opcode == 3:
            fields['bitmap'] = self.array(2, 'u1')
            fields['origin'] = self.scalar('ii', 'i32_xy')
            fields['angle'] = self.scalar('d', '<f8')
        elif opcode == 4:
            paths = self.collection(fields)
        return dict(draw_ordinal=ordinal, operator=_OPERATOR[opcode], span=_span(start, self.offset),
                    fields=fields, paths=paths)


def read_tape(data):
    """Recover exact version-1 spans and typed field shapes without copying payloads."""
    tape = _Tape(data)
    if tape.values('8s')[0] != b'SCARAGG1':
        raise ValueError('unsupported AGG tape version')
    width, height, dpi, count = tape.values('IIdI')
    if not (1 <= width <= 16384 and 1 <= height <= 16384 and 0 < dpi <= 16384
            and 4 * width * height <= _BUDGET and count <= 65536):
        raise ValueError('unsupported AGG surface or draw count')
    commands = [tape.command(ordinal) for ordinal in range(count)]
    if tape.offset != len(data):
        raise ValueError('trailing AGG tape bytes')
    return dict(surface=dict(width=width, height=height, dpi=dpi), commands=commands)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _load_pair(observation, frame_indices):
    source = Path(observation).resolve(strict=True)
    raw = source.read_bytes()
    document = json.loads(raw)
    if document['schema'] != _OBSERVATION or document['schema_version'] not in (1, 2):
        raise ValueError('expected original AGG scene observation')
    if len(frame_indices) != 2 or len(set(frame_indices)) != 2:
        raise ValueError('select two distinct original frames')
    frames = []
    for index in frame_indices:
        frame = document['frames'][index]
        if index < 0 or frame['index'] != index:
            raise ValueError('invalid original frame ordinal')
        path = (source.parent / frame['scene']).resolve(strict=True)
        if path.parent != source.parent or path.stat().st_size != frame['scene_bytes']:
            raise ValueError('original AGG tape location or size differs')
        if frame['scene_bytes'] > _BUDGET:
            raise ValueError('original AGG tape exceeds native byte budget')
        data = path.read_bytes()
        if _sha(data) != frame['scene_sha256']:
            raise ValueError('original AGG tape digest differs')
        tape = read_tape(data)
        if dict(Counter(command['operator'] for command in tape['commands'])) != frame['operations'] or (
                list(tape['surface'].values()) != frame['geometry']):
            raise ValueError('original AGG tape command metadata differs')
        frames.append((data, tape))
    return dict(uri=str(source), sha256=_sha(raw)), frames


def _field(field, data):
    start, end = field['span']['start'], field['span']['end']
    return {**{name: field[name] for name in ('span', 'shape', 'count') if name in field},
            'sha256': _sha(memoryview(data)[start:end])}


def analyze(observation, frame_indices=(0, 1)):
    """Compare one renderer tape position only when complete opcode sequences match."""
    reference, ((first_bytes, first), (second_bytes, second)) = _load_pair(observation, frame_indices)
    opcode_vectors = [[row['operator'] for row in tape['commands']] for tape in (first, second)]
    aligned = opcode_vectors[0] == opcode_vectors[1]
    commands, field_changes = [], Counter()
    if aligned:
        for left, right in zip(first['commands'], second['commands'], strict=True):
            fields = {}
            for name, original in left['fields'].items():
                lhs = _field(original, first_bytes)
                rhs = _field(right['fields'][name], second_bytes)
                changed = lhs['sha256'] != rhs['sha256']
                fields[name] = dict(format=original['format'], baseline=lhs, comparison=rhs,
                                    evidence='OBSERVED_VARYING' if changed else 'OBSERVED_EQUAL')
                if changed:
                    field_changes.update([(left['operator'], name)])
            commands.append(dict(draw_ordinal=left['draw_ordinal'], operator=left['operator'],
                                 fields=fields))
    return dict(schema=SCHEMA, schema_version=1, observation=reference, frames=list(frame_indices),
                surfaces=[first['surface'], second['surface']], command_opcode_sequences_align=aligned,
                correspondence='ORDERED_TAPE_POSITIONS' if aligned else 'NO_POSITION_ALIGNMENT',
                commands=commands, changed_fields=[dict(operator=op, field=field, count=count)
                    for (op, field), count in sorted(field_changes.items())],
                producer_bindings_proven=False, invariant_fields_proven=False,
                transformation_authorized=False)


def check(path):
    saved = json.loads(Path(path).read_bytes())
    if saved != analyze(saved['observation']['uri'], saved['frames']):
        raise ValueError('AGG scene layout report differs from original tapes')
    return saved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--observation', type=Path)
    mode.add_argument('--check', type=Path)
    parser.add_argument('--frames', nargs=2, type=int, default=(0, 1))
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if args.observation and args.out:
        result = analyze(args.observation, args.frames)
        with args.out.open('x', encoding='utf-8') as destination:
            json.dump(result, destination, allow_nan=False, sort_keys=True, separators=(',', ':'))
            destination.write('\n')
    elif args.check and not args.out:
        result = check(args.check)
    else:
        parser.error('provide --observation with --out or --check')
    print(json.dumps(dict(commands=len(result['commands']), changed_fields=result['changed_fields'])))


if __name__ == '__main__':
    main()
