"""Relate owned and privately mapped file materializations to one fixed-grid graph."""

from pathlib import Path

from scar.analysis.cpu_fixed_grid_lifecycle import _anchor, verify_cpu_fixed_grid_lifecycle
from scar.analysis.replay_transaction import replay_transaction
from scar.codegen.aten_recurrence_source import _mapped_source_inputs
from scar.ir.execution_recipe import _canonical, read_artifact
from scar.optimization.cpu_fixed_grid_mapped_input_plan import verify_cpu_fixed_grid_mapped_input_plan


SCHEMA = 'scar.cpu-fixed-grid-input-route-graph'
_HEADER = 'src/runtime/file_io.hpp'
_MAIN = 'src/main.cpp'
_BASE = '#/routes/native_cpp'


def _need(ok, detail):
    if not ok:
        raise ValueError('CPU fixed-grid input route graph: ' + detail)


def _one(rows, detail):
    rows = tuple(rows)
    _need(len(rows) == 1, detail)
    return rows[0]


def _member(project, manifest, name, schema='scar.native-cpp-source'):
    path = Path(project['uri']).parent / name
    _need(path == path.resolve(strict=True), 'native route member has an indirect path')
    return dict(uri=str(path), schema=schema, **manifest['files'][name])


def _span(reference, raw, selection):
    begin = raw.find(selection)
    _need(bool(selection) and begin >= 0 and raw.find(selection, begin+len(selection)) == -1,
          'provider source anchor changed or became ambiguous')
    return _anchor(reference, raw, begin, begin+len(selection))


def _relocate_header(old, before, new, after, anchor):
    if anchor['source'] != old:
        return None
    start, end = anchor['byte_start'], anchor['byte_end']
    _need(_anchor(old, before, start, end)['selection_sha256'] == anchor['selection_sha256'],
          'owned lifecycle source anchor changed')
    placed = _span(new, after, before[start:end])
    if 'byte_offset' in anchor:
        offset = anchor['byte_offset'] + placed['byte_start'] - start
        _need(after[offset-1:offset] == b'\n', 'retained marker relocation is outside a source line')
        placed['byte_offset'] = offset
    return placed


def _cpp_routes(lifecycle, plan, proof, measured):
    owned_ref, mapped_ref = lifecycle['project'], measured['mapped_project']
    control, mapped = read_artifact(owned_ref), read_artifact(mapped_ref)
    descriptor = dict(mapped['source_entry_cli'])
    _need(descriptor.pop('mapped_input_proof') == plan['proof'] and descriptor == control['source_entry_cli'],
          'mapped ABI and publication definition differ from the owned route')
    changed = {key for key in control['files'] if control['files'][key] != mapped['files'].get(key)}
    executable = control['executable']
    _need(control['files'].keys() == mapped['files'].keys() and mapped['executable'] == executable
          and {_HEADER, executable} <= changed <= {_HEADER, executable, 'validation/build.log', 'validation/ldd.txt'},
          'mapped C++ changed unrelated code, schedule or linkage members')
    owned = _member(owned_ref, control, _HEADER)
    mapped_header = _member(mapped_ref, mapped, _HEADER)
    before, after = (read_artifact(ref, json_document=False) for ref in (owned, mapped_header))
    _need(proof['native_members'][_HEADER] == owned and _mapped_source_inputs(before.decode()) == after.decode(),
          'mapped source differs from the verified file-only rewrite')
    main = _member(mapped_ref, mapped, _MAIN)
    main_src = read_artifact(main, json_document=False)
    launch_read_order = _span(main, main_src, b'}, load_order, [&](const auto&, const auto&) { std::cout')
    mapped_sites = {
        'source_cli_dispatch': launch_read_order,
        'source_load_branch': _span(mapped_header, after, b'if (!load_order.empty()) {'),
        'exact_extent_guard': _span(mapped_header, after, b'if (bytes == 0 || fs::file_size(paths[i]) != bytes) {'),
        'private_from_file': _span(mapped_header, after, b'inputs[i] = at::from_file(paths[i], false, elements,\n'
            b'                at::TensorOptions().device(at::kCPU).dtype(at::kFloat))'),
        'shape_view': _span(mapped_header, after, b'.view(at::IntArrayRef(shape.data(), shape.size()));'),
        'runner_inputs_ready': _span(mapped_header, after,
            b'const std::array<at::Tensor, graph::output_count> outputs = runner(inputs);'),
    }
    _need([row['byte_start'] for k, row in mapped_sites.items() if k != 'source_cli_dispatch']
          == sorted(row['byte_start'] for k, row in mapped_sites.items() if k != 'source_cli_dispatch'),
          'mapped branch geometry, view and downstream readiness are unordered')
    base = lifecycle['routes']['native_cpp']
    relocs = {}
    for name, site in base['sites'].items():
        placed = _relocate_header(owned, before, mapped_header, after, site)
        if placed is not None:
            relocs[f'{_BASE}/sites/{name}'] = placed
    for name, operation in base['operations'].items():
        if name.startswith('load:'):
            continue
        for index, anchor in enumerate(operation['source']):
            placed = _relocate_header(owned, before, mapped_header, after, anchor)
            if placed is not None:
                relocs[f'{_BASE}/operations/{name}/source/{index}'] = placed
    return base, owned, mapped_header, mapped_sites, relocs


def _ports(base, resources, plan, proof):
    rows = []
    read_order = proof['input_read_order']
    inputs = {row['port']: (index, row) for index, row in enumerate(plan['inputs'])}
    pinned = {row['port']: (index, row) for index, row in enumerate(resources['file_reads'])}
    effects = {row['port']: index for index, row in enumerate(proof['inputs'])}
    _need(len(read_order) == len(inputs) == len(pinned) == len(effects) and set(read_order) == inputs.keys()
          == pinned.keys() == effects.keys(), 'route loses a qualified file port or read-order slot')
    for ordinal, port in enumerate(read_order):
        operation = f'load:{port}'
        retained = base['operations'][operation]
        plan_index, item = inputs[port]
        resource_index, physical = pinned[port]
        _need(retained['kind'] == 'EAGER_FILE_READ_AND_BORROWED_TENSOR'
              and (retained['port'], retained['ordinal']) == (port, ordinal)
              and retained['physical_port'] == f'#/file_reads/{resource_index}'
              and retained['logical_bytes'] == physical['bytes'] == item['payload_bytes']
              and physical['shape'] == item['shape']
              and Path(physical['input']['uri']).name == item['path'],
              'owned graph materialization differs from the mapped port scope')
        data = _one((i for i, edge in enumerate(base['edges'])
                    if edge['source'] == operation and edge['target'] == 'recurrence'
                    and edge['kind'] == 'TENSOR_INPUT' and edge['port'] == port),
                    'expected one original recurrence operand edge for each materialized port')
        order = _one((i for i, edge in enumerate(base['edges'])
                     if edge['source'] == operation and edge['kind'] == 'NORMAL_ORDER'),
                     'expected one forward edge in the original input read order')
        expected = f'load:{read_order[ordinal+1]}' if ordinal+1 < len(read_order) else 'recurrence'
        _need(base['edges'][order]['target'] == expected, 'the source load order differs from the verified graph')
        rows.append(dict(operation=operation, port=port, ordinal=ordinal,
            file=physical['input'], dtype=item['dtype'], shape=item['shape'], logical_bytes=item['payload_bytes'],
            owned_operation=f'{_BASE}/operations/{operation}', readiness_edge=f'{_BASE}/edges/{data}',
            input_order_edge=f'{_BASE}/edges/{order}', resources=retained['resources'],
            resource_port=retained['physical_port'], plan_input=f'#/inputs/{plan_index}',
            read_only_effect=f'#/inputs/{effects[port]}'))
    return rows


def derive_cpu_fixed_grid_input_route_graph(lifecycle_ref, mapped_plan_ref):
    """Reuse the admitted lifecycle and overlay only source-backed changes in materialization."""
    with replay_transaction():
        lifecycle = verify_cpu_fixed_grid_lifecycle(lifecycle_ref)
        plan = verify_cpu_fixed_grid_mapped_input_plan(mapped_plan_ref)
        empirical = plan['empirical']
        _need(empirical is not None and plan['observation'] == empirical['reference'],
              'a matched execution of both native backing routes is required')
        measured = read_artifact(plan['observation'])
        proof = read_artifact(plan['proof'])
        scope = lifecycle['scope']
        _need(lifecycle['delivery'] == empirical['control_delivery'] == measured['control_delivery']
              and lifecycle['project'] == plan['project'] == proof['project'] == measured['control_project']
              and plan['target'] == lifecycle['target'] == proof['target'] == measured['target']
              and plan['source'] == lifecycle['source'] == proof['source'] == measured['source']
              and scope['argv'] == empirical['argv'] == measured['argv'] and scope['steps'] == measured['steps']
              and scope['working_directory'] == empirical['working_directory'] == measured['working_directory']
              and scope['environment_sha256'] == measured['environment_sha256']
              and scope['cpu_ids'] == plan['placement']['cpus'] == measured['cpu_ids']
              and scope['source_contract_sha256'] == plan['source_contract_sha256']
              and empirical['mapped_project'] == measured['mapped_project']
              and lifecycle['validated_final_tensor'] == measured['original_output']
              and lifecycle['validated_source_stdout'] == measured['original_stdout'],
              'selected mapping differs from the original counted lifecycle or publication domain')
        resources = read_artifact(lifecycle['resources'])
        base, owned, mapped, sites, relocations = _cpp_routes(lifecycle, plan, proof, measured)
        rec = base['operations']['recurrence']
        _need(base['recurrence_binding'] == proof['execution'] and rec['count'] == scope['steps']
              and proof['physical_binding'] == resources['binding'] and proof['abi'] == resources['abi']
              and rec['loop']['control'] == resources['loop']['control'] and not rec['loop']['loops_unrolled']
              and measured['native_placement'] == base['lane_placement'] == plan['placement'],
              'native route changes the counted recurrence, input ABI or CPU lane placement')
        ports = _ports(base, resources, plan, proof)

    logical = sum(row['logical_bytes'] for row in ports)
    _need(logical == plan['movement']['logical_input_payload_bytes'] == proof['logical_copy_payload_bytes']
          and logical == plan['movement']['removed_explicit_eager_copy_payload_bytes'],
          'route geometry differs from the verified eager payload account')
    deltas = {row['operation']: dict(kind='PRIVATE_FILE_BACKED_TENSOR_VIEW',
        materialization='ATEN_FROM_FILE_PRIVATE_CPU_FLOAT32_THEN_VIEW', backing='PRIVATE_FILE_MAPPING',
        mapped_logical_bytes=row['logical_bytes'], eager_copy_payload_bytes=0,
        readiness='TENSOR_DESCRIPTOR_AVAILABLE_WITH_PHYSICAL_PAGES_OPEN',
        source_site_ids=['source_load_branch', 'exact_extent_guard', 'private_from_file', 'shape_view'],
        same_logical_input_and_readiness_edge=True, physical_read_or_first_touch_proven=False) for row in ports}
    selected = {'OWNED_COPY': measured['control_project'], 'PROVIDER_MAP': measured['mapped_project']}
    _need(plan['selected_project'] == selected[plan['selected_runtime']], 'selected project differs from the mapped plan')
    return dict(schema=SCHEMA, schema_version=1, status='PROVEN_SCOPED_FIXED_GRID_INPUT_ROUTE_GRAPH',
        lifecycle=lifecycle_ref, mapped_plan=mapped_plan_ref, proof=plan['proof'], resources=lifecycle['resources'],
        observation=plan['observation'], source=lifecycle['source'], scope=scope, target=lifecycle['target'],
        owned_project=measured['control_project'], mapped_project=measured['mapped_project'],
        selected_project=plan['selected_project'], selected_runtime=plan['selected_runtime'],
        execution_default=plan['execution_default'], ports=ports,
        routes=dict(original_python=dict(lifecycle_locator='#/routes/original_python'),
            owned_copy=dict(lifecycle_locator=_BASE, file_io=owned, executable=measured['control_executable'],
                            backing='OWNED_HEAP_BUFFER_AND_BORROWED_VIEW', eager_copy_payload_bytes=logical),
            provider_map=dict(base='owned_copy', lifecycle_locator=_BASE, file_io=mapped, executable=measured['mapped_executable'],
                operation_deltas=deltas, new_source_sites=sites, changed_header_anchors=relocations,
                unchanged_source_members_relocate_by_manifest_digest=True,
                launch_argument_tail=scope['argv'], launcher=measured['launchers']['PROVIDER_MAP'])),
        recurrence=dict(operation=f'{_BASE}/operations/recurrence', execution=proof['execution'],
            steps=scope['steps'], same_input_dependency_edges=True, retained_shared_body=True),
        backing_storage=dict(owned=plan['storage']['original_graph_held'], mapped=plan['storage']['candidate_graph_held'],
                             physical_residency_proven=False),
        logical_copy_payload_removed_bytes=logical, kernel_io_bytes_proven=False,
        first_touch_or_page_fault_schedule_proven=False, physical_dram_traffic_reduction_proven=False,
        stable_inputs_required_for_full_lifetime=True, whole_entry_equivalence_proven=False,
        target_executed=False, transformation_authorized=False, deployment_authorized=False)


def verify_cpu_fixed_grid_input_route_graph(reference):
    _need(reference['schema'] == SCHEMA, 'unexpected fixed-grid route-graph schema')
    saved = read_artifact(reference)
    expected = derive_cpu_fixed_grid_input_route_graph(saved['lifecycle'], saved['mapped_plan'])
    _need(_canonical(saved) == _canonical(expected), 'saved fixed-grid input route graph differs on cold replay')
    return saved
