"""Join a selected-site ndarray trace to a sealed original returned-state loop."""

import argparse
import json
from pathlib import Path

from scar.analysis import returned_loop_binding
from scar.analysis.ndarray_tape_schedule import TAPE
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.trace import ndarray_program_tape


_LAYOUT = ('storage', 'dtype', 'shape', 'strides_bytes', 'offset_bytes', 'address_mod_64')
_SLICES = [dict(kind='TUPLE', items=[dict(kind='ELLIPSIS'), dict(kind='INDEX', index=index)])
           for index in range(2)]
_CONTROL = {'ROUND_BEGIN', 'GATE_BEGIN', 'GATE_END', 'STATE_CREATE', 'MERGED_ROOT'}


def _need(condition, message):
    if not condition:
        raise ValueError('ndarray tape projection: ' + message)


def _layout(row):
    return {key: str(row[key]) if key == 'storage' else row[key] for key in _LAYOUT}


def _same(left, right):
    return _layout(left) == _layout(right)


def _to_events(round_):
    events = []
    pending_norm = None
    initialized_round = False
    selection = None
    gates = 0
    for ordinal, row in enumerate(round_['events'], round_['begin_ordinal']):
        kind = row['kind']
        if kind in _CONTROL:
            if kind == 'ROUND_BEGIN':
                _need(not initialized_round and ordinal == round_['begin_ordinal'],
                      'original round has no unique entry')
                initialized_round = True
            elif kind == 'GATE_BEGIN':
                _need(gates == 0, 'unexpected nested numerical dispatch')
                gates += 1
            elif kind == 'GATE_END':
                _need(gates == 1, 'missing original numerical dispatch')
                gates -= 1
            continue
        if kind == 'NORM':
            _need(pending_norm is None, 'unconsumed original norm')
            pending_norm = row
            continue
        if kind == 'INITIAL_BASIS':
            _need(row['value'] == dict(kind='EXACT_INT', value=1), 'non-unit basis coefficient')
            event = dict(output=_layout(row['output']), index=row['basis'])
        elif kind in ('COPY', 'VIEW'):
            _need(kind != 'COPY' or row['order'] == 'C', 'only original logical C-order copy is admitted')
            event = dict(input=_layout(row['input']), output=_layout(row['output']))
        elif kind == 'OUTER':
            event = {key: _layout(row[key]) for key in ('left', 'right', 'output')}
        elif kind == 'PARTITION':
            coeff = row['coefficients']
            _need(coeff['shape'] == [2, 2] and coeff['dtype'] == '<c16' and coeff['order'] == 'C'
                  and row['slices'] == _SLICES and _same(row['out'], row['output']),
                  'partition changed its original arguments or output alias')
            event = dict(target=_layout(row['target']), out=_layout(row['out']), coefficients=coeff['bytes_hex'])
        elif kind == 'PHASE_MULTIPLY':
            coeff = row['coefficient']
            _need(coeff['kind'] == 'BITS' and coeff['dtype'] == '<c16'
                  and row['view']['storage'] == row['target']['storage'], 'unsupported phase coefficient or effect')
            event = dict(view=_layout(row['view']), coefficient=coeff['bytes_hex'])
        elif kind == 'NORMALIZE':
            _need(pending_norm is not None and _same(pending_norm['input'], row['input'])
                  and pending_norm['value'] == row['norm'] and pending_norm['axes'] is None
                  and pending_norm['ord'] is None and pending_norm['keepdims'] is False
                  and row['selection'] in ('NORMALIZED', 'RAW') and selection is None,
                  'normalization needs its exact preceding norm and selected result')
            selected = row['selection'] == 'NORMALIZED'
            _need(_same(row['output'], row['normalized'] if selected else row['input']),
                  'normalization result has no selected source layout')
            event = dict(input=_layout(row['input']), normalized=_layout(row['normalized']), selected=selected)
            selection = row['output']
            pending_norm = None
        else:
            raise ValueError('unhandled observed numerical boundary ' + kind)
        events.append(dict(kind=kind, original_ordinal=ordinal, **event))
    _need(gates == 0 and pending_norm is None and initialized_round and selection is not None
          and ordinal == round_['end_ordinal'], 'round extent or normalization completion is open')
    digest = round_['output']
    _need(digest['kind'] == 'NDARRAY' and digest['shape'] == selection['shape'] and digest['order'] == 'C',
          'checkpoint shape or representation differs')
    events.append(dict(kind='CHECKPOINT', input=_layout(selection), sha256=digest['sha256'],
                       original_ordinal=ordinal))
    return events


def project(observation_path, binding_path):
    observation_ref = artifact_reference(Path(observation_path).resolve(strict=True), ndarray_program_tape.SCHEMA)
    binding_ref = artifact_reference(Path(binding_path).resolve(strict=True), returned_loop_binding.SCHEMA)
    observation = ndarray_program_tape.check(observation_ref['uri'])
    _need(observation == read_artifact(observation_ref), 'observed tape changed')
    binding = returned_loop_binding.check(binding_ref['uri'])
    _need(binding == read_artifact(binding_ref), 'source/trajectory binding changed')
    _need(observation['schema_version'] == 1 and observation['capture_is_diagnostic'] is True
          and observation['completeness'] == 'SELECTED_OPERATION_SITES_ONLY'
          and not observation['frontiers'] and not observation['entry_events'],
          'source scope must have no open retained diagnostic event')
    numpy_provider = observation['providers']['numpy_extension']
    trajectory = read_artifact(observation['trajectory'])
    _need(trajectory['endpoints'] == observation['endpoints'] and trajectory['cpu_ids'] == observation['cpu_ids'],
          'independent trajectory has different scope')
    cohorts = []
    for ref in binding['witnesses']:
        witness = read_artifact(ref)
        if witness['endpoints'] == trajectory['endpoints']:
            for activation in witness['activations']:
                rows = activation['iterations']
                states = [(row['call']['input']['value'], row['result']['output']['value']) for row in rows]
                cohorts.append((ref, states))
    states = [(row['input'], row['output']) for row in observation['rounds']]
    matched = [ref for ref, result in cohorts if result == states]
    _need(len(matched) == 1 and states == [(row['input'], row['output']) for row in trajectory['records']]
          and (states[0][0]['kind'] == 'EXACT_INT')
          and all(previous[1] == current[0] for previous, current in zip(states, states[1:])),
          'initial or loop-carried state differs from independently source-bound activation')
    events = []
    rounds = []
    for index, round_ in enumerate(observation['rounds']):
        _need(round_['index'] == index and round_['events'], 'observed loop order differs')
        start = len(events)
        events.extend(_to_events(round_))
        rounds.append(dict(index=index, begin=start, end=len(events), original_round=round_['index']))
    return dict(schema=TAPE, schema_version=1, observation=observation_ref, source_binding=binding_ref,
        witness=matched[0], numpy_provider=numpy_provider, cpu_ids=observation['cpu_ids'],
        storages=observation['storages'], events=events, rounds=rounds, frontiers=[],
        scope='FIXED_OBSERVED_COHORT_WITH_INDEPENDENT_ROUND_CHECKPOINTS',
        residual_effect_contracts='TARGET_PYTHON_ENTRY_AND_UNOBSERVED_FIELDS_REMAIN_ORIGINAL',
        captures_evolving_state_payloads=False, deploy=False)


def check(document):
    _need(document == project(document['observation']['uri'], document['source_binding']['uri']),
          'sealed tape differs from replayed source and observations')
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observation', required=True)
    parser.add_argument('--binding', required=True)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    tape = project(args.observation, args.binding)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(tape, stream, separators=(',', ':'), allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(rounds=len(tape['rounds']), events=len(tape['events']))))


if __name__ == '__main__':
    main()
