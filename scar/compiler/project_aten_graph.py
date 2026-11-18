"""Project same-run F4 and ATen ports to an ordered standalone CPU recipe."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from scar.compiler.aten_graph_capture_linkage import verify_graph_capture_linkage
from scar.compiler.aten_native_recipe import normalize_recipe
from scar.compiler.plan_aten_capture import derive_aten_capture_plan
from scar.ir.literals import PythonLiteral
from scar.ir.execution_recipe import read_artifact
from scar.trace.aten_native_graph_physical import recover_graph_input_metadata


def _contiguous_shape(port):
    shape = port['shape']
    strides = []
    stride = 1
    for extent in reversed(shape):
        strides.append(stride)
        stride *= max(extent, 1)
    descriptor = port['tensor_descriptor']
    if port['strides'] != strides[::-1] or descriptor[2] != 0 or descriptor[-1] != 'cpu':
        raise ValueError('observed port is outside the contiguous base CPU tensor ABI')
    return shape


def _port_key(port_source):
    return port_source['artifact']['sha256'], port_source['locator']


def _observed_tensor(port, physical_inputs):
    observed = physical_inputs.get(_port_key(port['source']))
    if (observed is None or observed['port'] != port['source'] or observed['descriptor_id'] != port['descriptor_id']
            or (observed['device'], observed['layout'], observed['dtype']) != ('cpu', 'strided', 'float32')
            or observed['shape'] != port['shape'] or observed['strides'] != port['strides']):
        raise ValueError('exact same-run CPU strided float32 input metadata required')
    _contiguous_shape(port)
    return observed


def _internal_witnesses(source):
    witnesses = {}
    for route in source['data_routes']:
        for match in route['descriptor_matches']:
            key = route['before'], route['after'], _port_key(match['output']), _port_key(match['input'])
            witnesses.setdefault(key, []).append(route)
    return witnesses


def project_recipe(source, native, physical):
    """Read provider order and keep F4 witnesses on each internal dependency.

    This produces a standalone specialization of an observed sequence. The
    raw files supply values under the new ABI. No Python wrapper is replaced.
    """
    attempts = [call['attempt'] for call in source['calls']]
    ops = native['operators']
    if [op['call_attempt'] for op in ops] != attempts or len(set(attempts)) != len(attempts):
        raise ValueError('graph projection requires one direct native operator per selected F4 call')
    if [(edge['before'], edge['after']) for edge in source['world_order']] != list(zip(attempts, attempts[1:])):
        raise ValueError('region lacks the original normal-path world order')

    original_ports = {}
    for row in physical['inputs']:
        key = _port_key(row['port'])
        if key in original_ports:
            raise ValueError('more than one physical observation for the same original input slot')
        original_ports[key] = row
    witnesses = _internal_witnesses(source)
    inputs, ops_by_attempt = {}, {}
    graph_ops, routes, exports = [], [], []
    producer, storage_owner = {}, {}
    for op in ops:
        attempt, identifier = op['call_attempt'], f'call_{op["call_attempt"]}'
        effects = op['declared_schema_effects']
        if effects['may_mutate'] or effects['alias_ports']:
            raise ValueError('observed native schema writes or aliases across the replay boundary')
        if len(op['outputs']) != 1 or op['outputs'][0]['kind'] != 'TENSOR_REFERENCE':
            raise ValueError('graph projection requires one Tensor result per operator')
        result, = op['outputs']

        arguments = []
        for argument in op['inputs']:
            if argument['kind'] == 'EXACT_SCALAR':
                scalar = PythonLiteral.from_dict(argument['literal']).to_python()
                if type(scalar) is not int:
                    raise ValueError('v1 graph replay supports exact integer Scalar operands')
                arguments.append(dict(scalar=scalar))
                routes.append(dict(kind='OBSERVED_BOXED_INTEGER', port=argument['source'], scalar=scalar))
                continue
            if argument['kind'] != 'TENSOR_REFERENCE':
                raise ValueError('unsupported or unresolved native argument')
            observed = _observed_tensor(argument, original_ports)
            descriptor = argument['descriptor_id']
            if descriptor in producer:
                prior, point = producer[descriptor]
                matched = witnesses.get((prior, attempt, _port_key(point['source']), _port_key(argument['source'])), ())
                if len(matched) != 1:
                    raise ValueError('observed internal dependency lacks a unique original F4 route')
                value = ops_by_attempt[prior]
                trace = dict(kind='F4_AND_NATIVE_REFERENCE', producer_port=point['source'], f4=matched[0])
            else:
                value, _ = inputs.setdefault(descriptor, (f'input_{len(inputs)}', argument['shape']))
                trace = dict(kind='EXTERNAL_NATIVE_INPUT', descriptor=descriptor)
            storage = argument['tensor_descriptor'][1]
            if storage_owner.setdefault(storage, descriptor) != descriptor:
                raise ValueError('distinct native values share observed storage across the replay region')
            arguments.append(dict(value=value))
            routes.append(dict(trace, value=value, port=argument['source'], physical=observed['evidence']))

        _contiguous_shape(result)
        descriptor, storage = result['descriptor_id'], result['tensor_descriptor'][1]
        if descriptor in producer or descriptor in inputs or storage in storage_owner:
            raise ValueError('observed result reuses a prior tensor or storage descriptor')
        producer[descriptor] = attempt, result
        storage_owner[storage] = descriptor
        ops_by_attempt[attempt] = identifier
        graph_ops.append(dict(id=identifier, schema=op['schema'], arguments=arguments))

    seen_return_paths = set()
    for row in sorted(source['return_routes'], key=lambda row: row['return_path']):
        path = tuple(row['return_path'])
        if path in seen_return_paths:
            raise ValueError('more than one selected producer reaches the same original return path')
        seen_return_paths.add(path)
        exports.append(dict(value=ops_by_attempt[row['before']], source=row))
    if not exports:
        raise ValueError('no selected direct operator result reaches an original caller return path')
    recipe = dict(schema='scar.aten-native-recipe', schema_version=1,
                  inputs=[dict(id=value, shape=shape) for value, shape in inputs.values()],
                  operations=graph_ops, outputs=[row['value'] for row in exports])
    return recipe, dict(routes=routes, exports=exports, formal_route_observations=source['formal_routes'],
                        original_world_order=source['world_order'])


def project_aten_graph(build, capture, attempts, *, linkage, linkage_sha256, out, binding_graph=None):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    records = derive_aten_capture_plan(build, capture, attempts, binding_graph=binding_graph)
    native, source = records['native'], records['source']
    linked = verify_graph_capture_linkage(linkage, linkage_sha256, native)
    physical = recover_graph_input_metadata(capture, native)
    if any(row['reason'] == 'NATIVE_INPUT_COMPANION_UNAVAILABLE' for row in physical['unresolved']):
        raise ValueError('native graph projection requires the TorchOp Tensor input companion from trace --native-tensor-inputs')
    recipe, graph = project_recipe(source, native, physical)
    import torch

    recipe = normalize_recipe(recipe, read_artifact(linked['linkage_report'])['operators'], torch)
    wire = (json.dumps(recipe, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    report = dict(schema='scar.aten-capture-graph-projection', schema_version=1,
                  scope='OBSERVED_OPERATOR_SEQUENCE_AS_STANDALONE_CPU_REPLAY',
                  inputs=records['inputs'], run=native['run'], process=native['process'],
                  linkage=linked, physical=physical, graph=graph, recipe_sha256=sha256(wire).hexdigest(),
                  original_source_decision=records['projection']['decision'],
                  source_replacement_authorized=False, reordered=False, performance_measured=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-project-aten-', dir=out.parent) as tmp:
        stage = Path(tmp)
        (stage / 'recipe.json').write_bytes(wire)
        (stage / 'projection.json').write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n')
        stage.rename(out)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar project-aten-graph', description=__doc__)
    parser.add_argument('--build', required=True)
    parser.add_argument('--capture', required=True)
    parser.add_argument('--call-attempt', type=int, action='append', required=True)
    parser.add_argument('--binding-graph')
    parser.add_argument('--linkage', required=True)
    parser.add_argument('--linkage-sha256', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    report = project_aten_graph(args.build, args.capture, args.call_attempt, linkage=args.linkage,
                                linkage_sha256=args.linkage_sha256, out=args.out, binding_graph=args.binding_graph)
    print(json.dumps(dict(recipe=str(Path(args.out) / 'recipe.json'), recipe_sha256=report['recipe_sha256'],
                          original_source_decision=report['original_source_decision'],
                          source_replacement_authorized=False)))
    return 0
