"""Project same-run producer lifetimes and compare private native handoff layouts."""

import argparse
from dis import get_instructions
from hashlib import sha256
import json
from math import prod
from pathlib import Path
from struct import Struct
from types import CodeType

from scar.analysis.bounded_entry_calls import SCHEMA as ENTRY
from scar.codegen.native_mesh_basis import BASIS_SCHEMA, read_package
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.native_ordered_resources import SCHEMA as RESOURCES
from scar.optimization.sparse_solve_cost import _duration
from scar.trace.code_snapshot import code_key

SCHEMA = 'scar.native-region-handoff'
CAPTURE = 'scar.mesh-basis-source-capture'
SOURCE = 'scar.python-source'
HEADER = Struct('<8s3I')
SEED = Struct('<8s4I')
MESH_FIELDS = ('mesh_coordinates', 'mesh_connectivity', 'reference_values',
               'reference_derivatives', 'quadrature_weights', 'constraints')


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _payload(raw, offset, fields):
    """Check a typed, contiguous payload and return memory views by logical port."""
    values = {}
    for key, field in fields:
        size = field.get('nbytes', field.get('size_bytes'))
        width = int(field['dtype'][-1])
        shape = field['shape']
        _require(all(type(dim) is int and dim >= 0 for dim in shape)
            and type(size) is int and size == prod(shape)*width and offset+size <= len(raw),
            'invalid typed handoff extent for ' + key)
        value = memoryview(raw)[offset:offset+size]
        _require(sha256(value).hexdigest() == field['sha256'], 'typed handoff bytes differ for ' + key)
        values[key] = value
        offset += size
    _require(offset == len(raw), 'unexpected trailing handoff payload')
    return values


def _axis_reuse(expanded, shape, compact, compact_shape, width):
    """Prove one exact repeated axis of a contiguous typed byte sequence."""
    axes = []
    for axis, extent in enumerate(shape):
        if extent <= 1 or shape[:axis] + shape[axis+1:] != compact_shape:
            continue
        block = prod(shape[axis+1:]) * width
        prefix = prod(shape[:axis])
        if all(expanded[(base*extent+i)*block:(base*extent+i+1)*block]
            == compact[base*block:(base+1)*block] for base in range(prefix) for i in range(extent)):
            axes.append(axis)
    return axes


def _codes(reference, optimize):
    source = compile(read_artifact(reference, json_document=False), reference['uri'], 'exec',
                     optimize=optimize, dont_inherit=True)
    pending, result = [source], {}
    while pending:
        code = pending.pop()
        result[code_key(code)] = {op.offset for op in get_instructions(code)
                                 if op.opname in ('RETURN_VALUE', 'RETURN_CONST')}
        pending.extend(child for child in code.co_consts if type(child) is CodeType)
    return result


def _phase_graph(capture, graph_ref, optimize):
    graph = read_artifact(graph_ref)
    _require(read_artifact(graph['program'])['entry'] == capture['entry']['uri'],
             'handoff phases belong to another original entry')
    site_rows = {}
    for index, row in enumerate(graph['observed_targets']):
        if row['entry_scope_reachable']:
            site = row['observation']['caller_site']
            site_rows.setdefault((site['code'], site['offset']), []).append((index, row))
    nodes, source_codes = {}, {}
    _duration(capture['run_ns'])
    for name, phase in capture['phases'].items():
        target, site, wall = (phase[key] for key in ('target_code', 'caller_site', 'wall_ns'))
        cpu = phase['thread_cpu_ns']
        path = target['filename']
        if path not in source_codes:
            reference = _ref(path, SOURCE)
            source_codes[path] = reference, _codes(reference, optimize)
        reference, return_sites = source_codes[path]
        _require(target['key'] in return_sites and len(wall) in (1, 2)
            and len(cpu) == len(wall) and all(type(value) is int and value > 0 for value in cpu)
            and all(type(clock) is int and clock > 0 for clock in wall)
            and capture['run_ns'][0] <= wall[0] <= wall[-1] <= capture['run_ns'][1]
            and (phase['return_offset'] in return_sites[target['key']] if len(wall) == 2
                 else phase['return_offset'] is None),
            'phase differs from the pinned source code or observed clock domain')
        matches = [(index, row) for index, row in site_rows.get((site['code'], site['offset']), ())
                   if row['observation']['caller_site'] == site
                   and (row['observation']['target_code'] or {}).get('key') == target['key']]
        _require(len(matches) <= 1, 'phase has multiple exact source graph observations')
        node = dict(source=reference, target=target, call_site=site, observed_wall_interval_ns=wall,
            inclusive_ns=_duration(wall) if len(wall) == 2 else None,
            measured_thread_cpu_ns=_duration(cpu) if len(cpu) == 2 else None,
            observation=f'#/observed_targets/{matches[0][0]}' if matches else None,
            target_binding='EXACT' if matches else 'RUNTIME_CODE_WITHOUT_EXPANDED_CALL_LINK')
        if matches:
            node['operation'] = matches[0][1]['operation']
        if len(wall) == 2:
            _require(node['measured_thread_cpu_ns'] <= node['inclusive_ns'], 'thread CPU exceeds its wall interval')
        nodes[name] = node
    return nodes


def _source_edges(capture, nodes):
    requested = (
        ('refine', 'basis', 'same_refinement_mesh_at_basis_entry'),
        ('refine', 'rows', 'same_refinement_mesh_at_boundary_query'),
        ('basis', 'matrix-input', 'same_basis_at_matrix_and_load'),
        ('basis', 'load-input', 'same_basis_at_matrix_and_load'),
        ('rows', 'solve-input', 'same_boundary_rows_at_solve'),
    )
    routes = capture['routes']
    _require(routes['same_refinement_mesh_on_basis_receiver'] is True,
             'retained producer object is no longer attached to its consumer')
    edges = []
    for producer, consumer, witness in requested:
        source, sink = (nodes[key]['observed_wall_interval_ns'] for key in (producer, consumer))
        _require(routes[witness] is True and len(source) == 2 and source[1] <= sink[0],
            'source object route lacks same-run identity or release-before-consumer order')
        edges.append(dict(producer=producer, consumer=consumer, observed_relation='SAME_OBJECT_IN_THIS_RUN',
            ready_at_ns=source[1], consumer_entered_at_ns=sink[0], witness=witness))
    ordering = ['refine', 'basis', 'matrix-input', 'load-input', 'rows', 'solve-input']
    _require(all(nodes[a]['observed_wall_interval_ns'][-1] <= nodes[b]['observed_wall_interval_ns'][0]
                 for a,b in zip(ordering, ordering[1:])), 'selected original world order changed')
    return dict(nodes=nodes, data_edges=edges, original_world_order=ordering,
        retained_aliases=[dict(owner='basis', value_from='refine', lifetime_end_ns=None)],
        no_write_or_escape_proof=False, producer_or_consumer_reordering_authorized=False)


def _placement(resource):
    selected = resource['placement']
    hardware = read_artifact(selected['hardware'])
    owner = selected['owner_cpu_id']
    node = selected['cpu_numa_node']
    processor, = (row for row in hardware['processors'] if row['cpu_id'] == owner)
    _require(node in hardware['allowed_memory_nodes'] and processor['numa_nodes'] == [node]
        and processor['permitted'] is True and owner in hardware['launching_affinity']
        and all(processor[key] == selected['owner'][key] for key in ('package_id', 'die_id', 'core_id', 'siblings')),
        'proposed host owner differs from the published NUMA and core inventory')
    return dict(physical_cpu=processor, numa_node=node, owner_cpu=owner, owner_lane_capacity=1,
        worker_cpu_ids=selected['worker_cpu_ids'],
        free_bytes_snapshot=hardware['node_free_snapshot'].get(str(node), {}).get('bytes'),
        cgroup_headroom_snapshot_bytes=hardware['finite_memory_headroom_snapshot_bytes'],
        actual_numa_page_residency_proven=False, memory_reservation=False)


def _representations(mesh_doc, compact, basis_doc, dense):
    source, sink = mesh_doc['fields']['reference_values'], basis_doc['fields']['value']
    _require(source['dtype'] == sink['dtype'], 'reference and expanded ports must retain dtype')
    width = int(source['dtype'][-1])
    axes = _axis_reuse(dense['value'], sink['shape'], compact['reference_values'], source['shape'], width)
    _require(len(axes) == 1, 'expected one exact axis reconstruction from the original reference port')
    axis, = axes
    stride = [prod(sink['shape'][i+1:]) * width for i in range(len(sink['shape']))]
    expanded_step = [0 if i == axis else prod(
        dim for j, dim in enumerate(sink['shape']) if j > i and j != axis) * width
        for i in range(len(sink['shape']))]
    return dict(bitwise_sample_discharged=True, source_payload_bytes=len(compact['reference_values']),
        dense_payload_bytes=len(dense['value']),
        serialized_payload_bytes_avoidable_with_compact_port=len(dense['value']) - len(compact['reference_values']),
        expanded_shape=sink['shape'], dtype=sink['dtype'], broadcast_axis=axis,
        contiguous_strides_bytes=stride, borrowed_strides_bytes=expanded_step,
        logical_port_dependency=True, future_values_invariant_proven=False,
        requirements=['QUALIFIED_STRIDED_CONSUMER', 'RETAIN_OWNER_GENERATION_AND_BORROW_LIFETIME',
                      'PRESERVE_EXACT_REDUCTION_ORDER_AND_OUTPUT_CONTRACT', 'CHECK_PROFIT_AT_SAME_BOUNDARY'])


def _refinement_cost(capture, graph):
    parent = graph['nodes']['refine']['observed_wall_interval_ns']
    rows = capture['refinement_steps']
    steps = [row['wall_ns'] for row in rows]
    _require(steps and parent[0] <= steps[0][0] and steps[-1][1] <= parent[1]
        and all(left[1] <= right[0] for left, right in zip(steps, steps[1:])),
        'refinement steps must be distinct children in the same original call')
    total = sum(_duration(step) for step in steps)
    return dict(child_calls=len(steps), child_wall_intervals_ns=steps, child_wall_sum_ns=total,
        parent_wall_ns=_duration(parent), unattributed_parent_remainder_ns=_duration(parent)-total,
        charge_child_and_parent_separately=False)


def _layouts(placed):
    return dict(selected='KEEP_WIRE', status='PROPOSED_GUARDED_LOCAL_HANDOFF',
        choices={
            'KEEP_WIRE': dict(edges=[['produce', 'encode'], ['encode', 'transport'],
                                     ['transport', 'decode'], ['decode', 'consume']],
                process_boundary=True, matched_native_cost_ns=None),
            'BORROW_ON_OWNER': dict(edges=[['produce', 'borrow'], ['borrow', 'consume']],
                process_boundary=False, owner_cpu=placed['owner_cpu'], minimum_execution_slots=1,
                matches_live_storage_epochs_required=True, matched_native_cost_ns=None),
        },
        removes_only=['PRIVATE_ENCODE', 'PRIVATE_TRANSPORT', 'PRIVATE_DECODE'],
        preserve_all_consumers_and_original_world_order=True, deployment_authorized=False,
        constraints=['PROVE_NO_EXTERNALLY_OBSERVABLE_TRANSPORT', 'ADMIT_TYPED_LOCAL_OPERAND_ABI',
                     'PRESERVE_ORIGINAL_MUTABLE_STORAGE_AND_ESCAPE_EFFECTS', 'MATCHED_NATIVE_COST_REQUIRED'])


def project(capture_path, basis_package, basis_witness, resources_path, *, stdout_path=None):
    capture_ref = _ref(capture_path, CAPTURE)
    basis_ref = _ref(Path(basis_package) / 'artifact.json', BASIS_SCHEMA)
    witness_ref = _ref(basis_witness, 'scar.basis-assembly-probe')
    resource_ref = _ref(resources_path, RESOURCES)
    capture, witness, resource = (read_artifact(ref) for ref in (capture_ref, witness_ref, resource_ref))
    _, basis = read_package(Path(basis_ref['uri']).parent, 'basis')
    read_artifact(capture['entry'], json_document=False)
    read_artifact(capture['instrument'], json_document=False)
    read_artifact(capture['reference_source'], json_document=False)
    stdout = _ref(stdout_path, 'scar.handoff-source-stdout') if stdout_path is not None else None
    if stdout is not None:
        _require(stdout['sha256'] == capture['expected_stdout_sha256'], 'captured output differs from the source oracle')
    _require(capture['schema_version'] == 1 and capture['entry'] == witness['entry']
        and capture['source_graph']['schema'] == ENTRY
        and capture['launch']['entry_path'] == capture['entry']['uri'] and capture['launch']['exit_code'] == 0
        and capture['launch']['argv'] == [str(resource['case']['call_count'])]
        and capture['lane']['required_affinity'] == [capture['lane']['cpu_id']]
        and capture['original_basis_bytes_equal'] is True
        and all(type(capture[key]) is int and capture[key] > 0 for key in ('pid', 'thread_id')),
        'handoff must describe the original input case and observed owner')
    optimize = capture.get('context', {}).get('interpreter', {}).get('optimize', 0)
    graph = _source_edges(capture, _phase_graph(capture, capture['source_graph'], optimize))
    mesh = read_artifact(capture['mesh_wire'], json_document=False)
    seed = read_artifact(capture['seed_wire'], json_document=False)
    dense_ref = {**witness['input'], 'schema': 'scar.native-basis-array-wire'}
    packed = read_artifact(dense_ref, json_document=False)
    header = bytes(basis['plan']['input_header'])
    magic, vertices, cells, flags = HEADER.unpack_from(mesh)
    seed_magic, _, _, depth, seed_flags = SEED.unpack_from(seed)
    body_offset = HEADER.size+len(header)
    _require((magic, flags, seed_magic, seed_flags) == (b'SCARMES1', 0, b'SCARTRI1', 0)
        and cells == basis['plan']['elements'] and capture['fields']['mesh_coordinates']['shape'] == [2, vertices]
        and mesh[HEADER.size:body_offset] == header and seed[SEED.size:SEED.size+body_offset] == mesh[:body_offset]
        and depth == len(capture['refinement_steps'])
        and packed[:len(header)] == header and basis['admitted_input'] == capture['expected_basis'] == dense_ref
        and basis['witness'] == witness_ref,
        'source and native handoffs must agree on the original typed Basis envelope')
    compact = _payload(mesh, body_offset, ((key, capture['fields'][key]) for key in MESH_FIELDS))
    seed_roles = ('seed_coordinates', 'seed_connectivity', 'reference_values', 'reference_derivatives', 'quadrature_weights')
    _payload(seed, SEED.size+body_offset, ((key, capture['fields'][key]) for key in seed_roles))
    dense_roles = ('grad', 'value', 'quadrature', 'dofs', 'constraints')
    dense = _payload(packed, len(header), ((key, witness['fields'][key]) for key in dense_roles))
    _require(compact['mesh_connectivity'] == dense['dofs'] and compact['constraints'] == dense['constraints'],
        'connectivity or constrained rows differ from the checked native input sample')
    placed = _placement(resource)
    compact_view = _representations(capture, compact, witness, dense)
    return dict(schema=SCHEMA, schema_version=1, inputs=dict(capture=capture_ref, basis_package=basis_ref,
        basis_witness=witness_ref, native_resources=resource_ref, stdout=stdout), source_graph=graph,
        observed_owner=dict(lane=capture['lane'], pid=capture['pid'], thread_id=capture['thread_id'],
                            clocks_stay_in_this_run=True),
        native_placement=placed, interval_transfer_to_native=False, refinement_cost=_refinement_cost(capture, graph),
        wire=dict(exact_mesh_record_bytes=len(mesh), exact_seed_record_bytes=len(seed),
                  materialized_basis_record_bytes=len(packed),
                  field_payload_bytes={name:len(value) for name,value in compact.items()},
                  physical_pipe_service_ns=None, peak_physical_memory_bytes=None),
        reduction=_layouts(placed),
        broadcast_view=compact_view)


def check(path):
    receipt = read_artifact(_ref(path, SCHEMA))
    source = receipt['inputs']
    stdout = source['stdout']
    _require(receipt == project(source['capture']['uri'], Path(source['basis_package']['uri']).parent,
                                source['basis_witness']['uri'], source['native_resources']['uri'],
                                stdout_path=stdout['uri'] if stdout is not None else None),
        'native handoff graph does not replay')
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', type=Path)
    parser.add_argument('--stdout', type=Path)
    for name in ('capture', 'basis-package', 'basis-witness', 'resources', 'out'):
        parser.add_argument('--'+name, type=Path)
    args = parser.parse_args(argv)
    if args.check:
        print(check(args.check)['reduction']['status'])
        return
    if not all((args.capture, args.basis_package, args.basis_witness, args.resources, args.out)):
        parser.error('provide original capture, native Basis package, Basis witness, resource plan and output')
    result = project(args.capture, args.basis_package, args.basis_witness, args.resources, stdout_path=args.stdout)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
