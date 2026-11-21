"""Project observed returned-state transitions onto exact caller bytecode sites."""

import argparse
import json
from math import prod
from pathlib import Path
import re
from types import CodeType

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.trace.code_snapshot import call_instruction, code_key, code_parameters, site
from scar.trace.numpy_partition_entry_validation import (
    RETURN_RELATION_OBSERVATION, SCHEMA as TRAJECTORY_SCHEMA, WEAK_RETURN_IDENTITY,
)


SCHEMA = 'scar.returned-state-recurrence-witness'
OBJECT_RELATIONS_SCHEMA = 'scar.returned-state-object-relations'
_RUN_SCHEMA = 'scar.numpy-partition-live-run'
_SOURCE = 'scar.python-source'
_CALLS = {'CALL', 'CALL_KW', 'CALL_FUNCTION_EX'}
_RESULT_READS = _CALLS | {'LOAD_ATTR', 'LOAD_SUPER_ATTR'}
_RETURNS = {'RETURN_CONST', 'RETURN_VALUE'}
_HASH = re.compile(r'[0-9a-f]{64}')
_DTYPE = re.compile(r'[<>|=][biufc]([1-9][0-9]?)')


def _document(path):
    path = Path(path).resolve(strict=True)
    doc = json.loads(path.read_bytes())
    if type(doc) is not dict or type(doc.get('schema')) is not str or doc.get('schema_version') != 1:
        raise ValueError('expected a versioned witness artifact')
    reference = artifact_reference(path, doc['schema'])
    return read_artifact(reference), reference


def _codes(reference, required):
    if reference['schema'] != _SOURCE:
        raise ValueError('endpoint source artifact type differs')
    raw = read_artifact(reference, json_document=False)
    matches = {}
    for optimize in (0, 1, 2):
        stack = [compile(raw, reference['uri'], 'exec', dont_inherit=True, optimize=optimize)]
        while stack:
            code = stack.pop()
            key = code_key(code)
            if key in required:
                matches[key] = code
            stack.extend(value for value in code.co_consts if type(value) is CodeType)
        if matches.keys() == required:
            return matches
    raise ValueError('pinned source does not reproduce every observed CodeID')


def _instruction_site(source, code, coordinate, opnames):
    offset = coordinate['offset']
    observed = site(code, offset)
    instruction = call_instruction(code, offset)
    if (observed['code'] != coordinate['code_id'] or instruction is None
            or observed['opname'] != instruction[0] or instruction[0] not in opnames
            or observed['position'] is None):
        raise ValueError('observed coordinate does not match an expected source instruction')
    return dict(source=source, code_id=observed['code'], offset=offset,
                opname=observed['opname'], oparg=instruction[1], position=observed['position'])


def _integer(value):
    return type(value) is int and value >= 0


def _state(value, *, allow_integer=False):
    if allow_integer and value.get('kind') == 'EXACT_INT' and set(value) == {'kind', 'value'}:
        if type(value['value']) is int:
            return value
    if (value.get('kind') != 'NDARRAY'
            or set(value) != {'kind', 'dtype', 'shape', 'size_bytes', 'order', 'sha256'}
            or type(value['dtype']) is not str or type(value['shape']) is not list
            or any(not _integer(size) for size in value['shape'])
            or not _integer(value['size_bytes']) or value['order'] != 'C'
            or type(value['sha256']) is not str or not _HASH.fullmatch(value['sha256'])):
        raise ValueError('invalid observed state digest or exact integer')
    dtype = _DTYPE.fullmatch(value['dtype'])
    if dtype is None or int(dtype[1]) * prod(value['shape']) != value['size_bytes']:
        raise ValueError('observed state digest geometry differs')
    return value


def _digest_comparison(previous, current):
    if previous['kind'] != 'NDARRAY' or current['kind'] != 'NDARRAY':
        return 'NOT_COMPARABLE'
    if any(previous[key] != current[key] for key in ('dtype', 'shape', 'size_bytes', 'order')):
        return 'DIGEST_GEOMETRY_DIFFERS'
    return 'EQUAL_DIGEST_OBSERVED' if previous['sha256'] == current['sha256'] else 'DIFFERENT_DIGEST_OBSERVED'


def _load_trajectory(path):
    root, reference = _document(path)
    normal_exit = 'UNKNOWN'
    locator = '#'
    if reference['schema'] == _RUN_SCHEMA:
        if root.get('diagnostic') is not True or root['execution']['exit_code'] != 0:
            raise ValueError('worker trajectory lacks successful diagnostic execution')
        normal_exit, locator = 'EXIT_CODE_ZERO_OBSERVED', '#/trajectory'
        trajectory = root['trajectory']
        if (root['affinity'] != trajectory['cpu_ids']
                or root['execution']['entry_path'] != trajectory['endpoints']['caller_source']['uri']):
            raise ValueError('worker and trajectory scopes differ')
    else:
        trajectory = root
    if (trajectory['schema'] != TRAJECTORY_SCHEMA or trajectory['schema_version'] != 1
            or trajectory['capture_is_diagnostic'] is not True or trajectory['performance_measured'] is not False
            or not _integer(trajectory['calls']) or not trajectory['records']
            or trajectory['calls'] != len(trajectory['records'])
            or trajectory['calls'] != trajectory['natural_result_property_reads']):
        raise ValueError('incomplete state trajectory')
    cpus = trajectory['cpu_ids']
    if not cpus or any(not _integer(cpu) for cpu in cpus) or cpus != sorted(set(cpus)):
        raise ValueError('invalid trajectory CPU scope')
    return trajectory, dict(artifact=reference, locator=locator), normal_exit


def _object_relations(path, trajectory):
    if path is None:
        return {}, None
    report, reference = _document(path)
    if (reference['schema'] != OBJECT_RELATIONS_SCHEMA or report['trajectory'] != trajectory
            or report['method'] != 'PYTHON_IS_WITH_RETAINED_PREVIOUS_PROPERTY_VALUE'):
        raise ValueError('object observation refers to another trajectory or comparison method')
    relations = {}
    for index, row in enumerate(report['observations']):
        key = row['activation'], row['producer_index'], row['consumer_index']
        if (any(not _integer(part) for part in key) or key in relations
                or type(row['same_object']) is not bool):
            raise ValueError('invalid or repeated object observation')
        relations[key] = dict(kind='SAME_OBJECT_OBSERVED' if row['same_object'] else 'DISTINCT_OBJECTS_OBSERVED',
                              evidence=dict(artifact=reference, locator=f'#/observations/{index}'))
    return relations, reference


def _observed_value(kind, value, record, field):
    return dict(kind=kind, value=value, evidence=f'{record}/{field}')


def _observed_return(endpoint, code, offset):
    if not _integer(offset):
        raise ValueError('invalid observed normal return offset')
    coordinate = dict(code_id=endpoint['code_id'], offset=offset)
    return dict(event='PY_RETURN', offset=offset,
                site=_instruction_site(endpoint['source'], code, coordinate, _RETURNS))


def _weak_relation(row, key, trace, locator):
    if (set(row) != {'method', 'producer_index', 'previous_alive', 'same_object'}
            or row['method'] != WEAK_RETURN_IDENTITY or row['producer_index'] != key[1]
            or type(row['previous_alive']) is not bool or type(row['same_object']) is not bool
            or (row['same_object'] and not row['previous_alive'])):
        raise ValueError('invalid weak previous property object comparison')
    return dict(kind='SAME_OBJECT_OBSERVED' if row['same_object'] else 'DISTINCT_OBJECTS_OBSERVED',
                method=WEAK_RETURN_IDENTITY, previous_alive=row['previous_alive'],
                evidence=dict(artifact=trace['artifact'], locator=f'{locator}/input_object_relation'))


def project(path, *, object_relations=None):
    """Read one trajectory, recompile its exact source bytes, and retain observation scope."""
    trajectory, trace, normal_exit = _load_trajectory(path)
    relations, relations_ref = _object_relations(object_relations, trace)
    live_relations = trajectory.get('return_relation_observation')
    if live_relations is not None and live_relations != RETURN_RELATION_OBSERVATION:
        raise ValueError('unsupported returned-state relation observation')
    endpoints = trajectory['endpoints']
    source, call, result = (endpoints[key] for key in ('caller_source', 'call', 'result'))
    parameter = endpoints['input_parameter']
    if (call['kind'] != 'METHOD' or result['kind'] not in ('PROPERTY_GETTER', 'CACHED_PROPERTY_GETTER')
            or type(parameter) is not str or not parameter.isidentifier()):
        raise ValueError('unsupported state trajectory endpoints')

    requested = [(source, {row['caller']['code_id'] for row in trajectory['records']}),
                 (call['source'], {call['code_id']}), (result['source'], {result['code_id']})]
    materials = {}
    for reference, keys in requested:
        identifier = reference['uri']
        if identifier not in materials:
            materials[identifier] = reference, set()
        saved, pending = materials[identifier]
        if saved != reference:
            raise ValueError('endpoint source has conflicting bytes')
        pending.update(keys)
    compiled = {name: _codes(reference, keys) for name, (reference, keys) in materials.items()}
    call_code = compiled[call['source']['uri']][call['code_id']]
    result_code = compiled[result['source']['uri']][result['code_id']]
    if call_code.co_name != call['method'] or result_code.co_name != result['method'] or parameter not in code_parameters(call_code):
        raise ValueError('source endpoint or formal parameter differs')

    grouped, call_sites = {}, {}
    callers = compiled[source['uri']]
    for index, row in enumerate(trajectory['records']):
        activation, ordinal = row['activation'], row['round_index']
        caller = row['caller']
        if (not _integer(row['index']) or row['index'] != index or not _integer(activation)
                or not _integer(ordinal) or not _integer(caller['offset'])):
            raise ValueError('invalid trajectory activation order')
        coordinate = caller['code_id'], caller['offset']
        if coordinate not in call_sites:
            call_sites[coordinate] = _instruction_site(source, callers[coordinate[0]], caller, _CALLS)
        observed_site = call_sites[coordinate]
        incoming = _state(row['input'], allow_integer=True)
        outgoing = _state(row['output'])
        locator = f'{trace["locator"]}/records/{index}'
        if activation not in grouped:
            if activation != len(grouped):
                raise ValueError('trajectory activation ordering differs')
            initial = _observed_value('FIRST_OBSERVED_PARAMETER_VALUE', incoming, locator, 'input')
            grouped[activation] = dict(id=activation, caller_code_id=caller['code_id'], initial=initial,
                                       iterations=[], consumptions=[])
        group = grouped[activation]
        iterations = group['iterations']
        if ordinal != len(iterations) or group['caller_code_id'] != caller['code_id']:
            raise ValueError('trajectory crossed a caller activation boundary')
        operation = dict(record_index=index, round_index=ordinal, locator=locator,
                         call=dict(site=observed_site, endpoint_code_id=call['code_id'],
                                   state_parameter=parameter,
                                   input=_observed_value('ACTUAL_PARAMETER_VALUE', incoming, locator, 'input'),
                                   normal_completion=dict(event='PY_RETURN', offset=None)),
                         result=dict(endpoint_code_id=result['code_id'],
                                     normal_completion=dict(event='PY_RETURN', offset=None),
                                     receiver_relation='SAME_OBJECT_AS_METHOD_RESULT_OBSERVED',
                                     output=_observed_value('NATURAL_PROPERTY_VALUE', outgoing, locator, 'output')),
                         arguments=row['arguments'])
        if live_relations is not None:
            observed_return = row['returned_state']
            if (set(observed_return) != {'method_return_offset', 'result_return_offset', 'result_caller'}
                    or observed_return['result_caller']['code_id'] != caller['code_id']):
                raise ValueError('returned state was read outside its source caller')
            result_site = _instruction_site(source, callers[coordinate[0]], observed_return['result_caller'], _RESULT_READS)
            operation['call']['normal_completion'] = _observed_return(call, call_code, observed_return['method_return_offset'])
            operation['result'].update(site=result_site,
                normal_completion=_observed_return(result, result_code, observed_return['result_return_offset']))
        elif 'returned_state' in row or 'input_object_relation' in row:
            raise ValueError('returned-state coordinates require their original observation contract')
        if 'input_object_relation' in row and not iterations:
            raise ValueError('first observed state has no preceding property value')
        if iterations:
            previous = iterations[-1]
            key = activation, previous['record_index'], index
            relation = relations.pop(key, None)
            if 'input_object_relation' in row:
                if relation is not None:
                    raise ValueError('two object observations for one transition')
                relation = _weak_relation(row['input_object_relation'], key, trace, locator)
            group['consumptions'].append(dict(
                kind='ADJACENT_RETURN_AND_PARAMETER_OBSERVED',
                producer=dict(record_index=key[1], value=previous['result']['output']['evidence']),
                consumer=dict(record_index=index, value=operation['call']['input']['evidence'],
                              parameter=parameter, site=observed_site),
                same_call_site=previous['call']['site'] == observed_site,
                digest_comparison=_digest_comparison(previous['result']['output']['value'], incoming),
                object_relation=relation or dict(kind='UNKNOWN', evidence=None)))
        iterations.append(operation)
    if relations:
        raise ValueError('object comparison does not match adjacent rounds in its activation')

    document = dict(schema=SCHEMA, schema_version=1, trajectory=trace, object_observations=relations_ref,
                    endpoints=endpoints, cpu_ids=trajectory['cpu_ids'], activations=list(grouped.values()),
                    application_completion=normal_exit, loop_dataflow_proven=False, future_iterations_proven=False,
                    target_executed_by_projection=False, performance_measured=False)
    if live_relations is not None:
        document['return_relation_observation'] = live_relations
    return document


def write(out, source, *, object_relations=None):
    value = project(source, object_relations=object_relations)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return value


def check(path):
    """Cold replay the receipt against exact captured and source bytes."""
    retained, reference = _document(path)
    if reference['schema'] != SCHEMA:
        raise ValueError('expected a returned-state recurrence witness')
    relations = retained['object_observations']
    actual = project(retained['trajectory']['artifact']['uri'],
                     object_relations=relations['uri'] if relations is not None else None)
    if actual != retained:
        raise ValueError('returned-state recurrence cold replay differs')
    return actual


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectory')
    parser.add_argument('--object-relations')
    parser.add_argument('--out')
    parser.add_argument('--check')
    args = parser.parse_args(argv)
    if args.check and not (args.trajectory or args.out or args.object_relations):
        value = check(args.check)
    elif args.trajectory and args.out and not args.check:
        value = write(args.out, args.trajectory, object_relations=args.object_relations)
    else:
        parser.error('use --trajectory FILE --out FILE or --check FILE')
    print(json.dumps(dict(activations=len(value['activations']),
                          calls=sum(len(row['iterations']) for row in value['activations']))))


if __name__ == '__main__':
    main()
