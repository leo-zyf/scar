"""Bind admitted fixed-grid file ports and loop storage to the host target graph."""

from pathlib import Path
import stat

from scar.analysis.target_graph import TargetHardwareGraph
from scar.ir.execution_recipe import read_artifact
from scar.ir.program.target_hardware import CpuExecutor, HardwareThread, MemoryDomain, StorageEndpoint


SCHEMA = 'scar.cpu-fixed-grid-cli-resources'


def _need(condition, detail):
    if not condition:
        raise ValueError('fixed-grid CPU resources: ' + detail)


def _member(project, manifest, name, schema):
    path = Path(project['uri']).parent/name
    return dict(uri=str(path), schema=schema, **manifest['files'][name])


def _mount(path, endpoints, backing):
    resolved = path.resolve(strict=True)
    compatible = [row for row in endpoints if resolved.is_relative_to(Path(row.mount_point))]
    endpoint = max(compatible, key=lambda row: len(row.mount_point), default=None)
    if endpoint is None or resolved.stat().st_dev != Path(endpoint.mount_point).stat().st_dev:
        return dict(status='STORAGE_ENDPOINT_UNRESOLVED', endpoint=None, backing_devices=[])
    return dict(status='FILESYSTEM_NAMESPACE_AND_DEVICE_MATCH', endpoint=endpoint.id.wire,
                backing_devices=sorted(backing.get(endpoint.id, ())))


def derive_cpu_fixed_grid_cli_resources(project, target, argv, inputs):
    """Project logical payloads, eligible NUMA domains and known storage routes."""
    from scar.compiler.aten_entry_delivery import ABI_FILE, BINDING_FILE

    manifest = read_artifact(project)
    abi_ref = _member(project, manifest, ABI_FILE, 'scar.fixed-grid-file-entry')
    binding_ref = _member(project, manifest, BINDING_FILE, 'scar.aten-source-entry-binding')
    abi, binding = read_artifact(abi_ref), read_artifact(binding_ref)
    _need(binding['target'] == target and binding['placement']['cpus'] == manifest['cpus']
          and abi['argv']['count'] == len(argv) and len(binding['files']) == len(inputs)
          and binding['read_order'] == [row['port'] for row in binding['files']],
          'expected exact admitted source ports and native hardware placement')
    read_artifact(target)
    hardware = TargetHardwareGraph(Path(target['uri']).parent)
    resources = hardware.resources()
    links = hardware.connections()
    endpoint_ids = {row.id for row in resources if type(row) is StorageEndpoint}
    endpoints = [row for row in resources if row.id in endpoint_ids]
    backing = {endpoint: {link.target.wire for link in links
                          if link.source == endpoint and link.kind.value == 'STORAGE_BACKING'}
               for endpoint in endpoint_ids}
    threads = {row.cpu_id: row for row in resources if type(row) is HardwareThread}
    executors = [row for row in resources if type(row) is CpuExecutor]
    memory = [row for row in resources if type(row) is MemoryDomain]
    core_roles = []
    for cpu in manifest['cpus']:
        thread = threads[cpu]
        owners = [row for row in executors if cpu in row.cpu_ids]
        _need(thread.online is True and len(owners) == 1, 'native lane has no unique live target CPU executor')
        owner, = owners
        core_roles.append(dict(cpu_id=cpu, thread=thread.id.wire, cpu_executor=owner.id.wire,
            memory_candidates=sorted(row.id.wire for row in memory if row.owner == owner.id)))
    input_root = Path(argv[abi['argv']['input_root'] - 1]).resolve(strict=True)
    output = Path(argv[abi['argv']['output'] - 1])
    _need(input_root.is_dir() and output.is_absolute() and output.parent.is_dir(),
          'filesystem routes require an exact input root and existing output parent')
    ports = []
    for row, input_ref in zip(binding['files'], inputs, strict=True):
        path = input_root/row['path']
        _need(str(path) == input_ref['uri'] and input_ref['size_bytes'] == row['payload_bytes']
              and stat.S_ISREG(path.lstat().st_mode), 'input file does not implement its bound logical port')
        read_artifact(input_ref, json_document=False)
        ports.append(dict(input=input_ref, port=row['port'], value=row['value'],
                          shape=row['shape'], bytes=row['payload_bytes'],
                          storage=_mount(path, endpoints, backing)))
    costs = binding['costs']
    _need(sum(row['bytes'] for row in ports) == costs['logical_input_payload_bytes'],
          'input inventory differs from the proved file payload')
    steps = int(argv[abi['argv']['steps'] - 1])
    source_body_cost = costs['native_body_per_step_estimate_ns']
    if source_body_cost is not None:
        _need(type(source_body_cost) is int and source_body_cost > 0, 'invalid native step estimate')
    return dict(schema=SCHEMA, schema_version=1, project=project, target=target, abi=abi_ref,
        binding=binding_ref, argv=argv, inputs=inputs, file_reads=ports,
        file_publication=dict(path=str(output), bytes=costs['logical_final_payload_bytes'],
                              storage=_mount(output.parent, endpoints, backing),
                              filesystem_durability_proven=False),
        cpu_lanes=core_roles, loop=dict(steps=steps, control=binding['recurrence']['control'],
            native_step_schedule_estimate_ns=source_body_cost,
            body_only_estimate_ns=None if source_body_cost is None else steps * source_body_cost),
        logical_storage=binding['storage'], physical_dram_page_placement_proven=False,
        source_runtime_effects_closed=False, physical_storage_io_bytes_proven=False,
        missing_costs=['PROCESS_STARTUP', 'INPUT_LOAD_PAGE_STATE', 'FILE_PUBLICATION', 'STDOUT_DRAIN_AND_EXIT'],
        whole_process_interval_ns=dict(lower=None, upper=None), target_executed=False)


def verify_cpu_fixed_grid_cli_resources(reference):
    saved = read_artifact(reference)
    _need(reference['schema'] == SCHEMA and saved == derive_cpu_fixed_grid_cli_resources(
        saved['project'], saved['target'], saved['argv'], saved['inputs']), 'retained resource projection differs')
    return saved
