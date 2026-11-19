"""Select one capacity-qualified calibration seed before a parallel release exists."""

import argparse
import json
from pathlib import Path

from scar.analysis.sparse_parallel_scaling_scope import _lanes
from scar.analysis.target_inventory import _cpu_list
from scar.ir.execution_recipe import artifact_reference, read_artifact
from .sparse_parallel_cost import collect_hardware
from .sparse_parallel_recipe import CHOICE, HARDWARE, SCOPE, SERIAL_LAUNCH, _serial_evidence, _storage_requirements


BASIS = 'CAPACITY_ONLY_CALIBRATION_SEED'


def derive_trial_choice(scope_ref, serial_ref, hardware_ref, *, worker_address_space_bytes, max_workers):
    """Derive one exact trial placement without assuming any parallel service cost."""
    if ((scope_ref['schema'], serial_ref['schema'], hardware_ref['schema']) != (SCOPE, SERIAL_LAUNCH, HARDWARE)
            or type(max_workers) is not int or max_workers < 2 or type(worker_address_space_bytes) is not int
            or not 0 < worker_address_space_bytes < 2**63):
        raise ValueError('expected one sparse source/serial scope, observed hardware and bounded worker resources')
    scope = read_artifact(scope_ref)
    serial, study, _ = _serial_evidence(serial_ref, scope)
    symbolic = study['budget_bytes']
    cap = worker_address_space_bytes
    if not 0 < symbolic <= cap:
        raise ValueError('worker address space must include its existing symbolic budget')
    owner, = serial['context']['cpu_ids']
    hardware = read_artifact(hardware_ref)
    processors = {row['cpu_id']: row for row in hardware['processors']}
    host = processors.get(owner)
    if (hardware['schema_version'] != 1 or len(processors) != len(hardware['processors'])
            or not hardware['hierarchy_observation_complete'] or hardware['memory_reservation']
            or hardware['memory_placement_enforced'] or host is None or len(host['numa_nodes']) != 1):
        raise ValueError('trial requires complete live CPU and memory observations without resource reservation')
    if any(row['parsed']['memory_max']['bytes'] is not None and row['memory_current_bytes'] is None
           for row in hardware['limits']):
        raise ValueError('trial cannot assume headroom for an unobserved finite memory limit')
    numa, = host['numa_nodes']
    owner_lane, = _lanes(hardware, [owner], numa)
    cores = {tuple(owner_lane['physical_core'])}
    nearby = []
    for cpu, row in sorted(processors.items()):
        physical = tuple(row[key] for key in ('package_id', 'die_id', 'core_id'))
        if row['permitted'] and row['numa_nodes'] == [numa] and None not in physical and physical not in cores:
            nearby.append(cpu)
            cores.add(physical)
    _lanes(hardware, [owner, *nearby], numa)
    count = scope['loop']['exact_count_under_source_conditions']['repetitions']
    geometry = read_artifact(scope['source_scope'])['observed']['loop_inputs']
    size, nnz = geometry['matrix_shape'][0], geometry['nnz']
    per_worker = _storage_requirements(size, nnz, 1, cap, symbolic)
    fixed = per_worker['owner_detached_snapshot_bound_bytes'] + per_worker['owner_structure_bytes']
    slot = per_worker['owner_pending_input_bytes'] + per_worker['owner_pending_result_bound_bytes']
    available = hardware['node_free_snapshot'].get(str(numa), {})
    if not available.get('allowed') or type(available.get('bytes')) is not int:
        raise ValueError('trial requires observed NUMA memory headroom')
    headroom = available['bytes']
    if hardware['finite_memory_headroom_snapshot_bytes'] is not None:
        headroom = min(headroom, hardware['finite_memory_headroom_snapshot_bytes'])
    memory_limit = max(0, (headroom - fixed) // (cap + slot))
    quota = hardware['cpu_quota_equivalents']
    cpu_limit = len(nearby) if quota is None else max(0, quota[0] // quota[1] - 1)
    physical_limit = min(len(nearby), max_workers, count)
    width = min(physical_limit, cpu_limit, memory_limit)
    if width < 2:
        raise ValueError('host capacity cannot admit two sparse calibration workers')

    return dict(schema=CHOICE, schema_version=1, basis=BASIS, parallel_scope=scope_ref, serial_launch=serial_ref,
        hardware=hardware_ref, empirical_resource_reservation=False,
        dimensions=dict(call_count=count, max_ordered_window=max_workers),
        budget=dict(worker_total_limit_bytes=cap, total_cap_bytes=headroom, fixed_owned_bytes=fixed,
                    owner_slot_bound_bytes=slot),
        owner_service_seed_ns=None, worker_service_seed_ns=None,
        choice=dict(owner_cpu_id=owner, worker_cpu_ids=nearby[:width], worker_count=width, window=width,
                    cpu_numa_node=numa, physical_worker_limit=physical_limit, cpu_quota_worker_limit=cpu_limit,
                    memory_worker_limit=memory_limit, memory_bound_inputs_complete=True,
                    modeled_owned_memory_bytes=fixed + width * (cap + slot), uses_smt_siblings_as_extra_cores=False,
                    automatic_admission=False, resource_snapshot_is_reservation=False))


def check_trial_choice(reference):
    """Recompute the seed from its exact scope, serial launch and original hardware."""
    choice = read_artifact(reference)
    expected = derive_trial_choice(choice['parallel_scope'], choice['serial_launch'], choice['hardware'],
        worker_address_space_bytes=choice['budget']['worker_total_limit_bytes'],
        max_workers=choice['dimensions']['max_ordered_window'])
    if choice != expected:
        raise ValueError('calibration seed differs from replayed sparse resource capacity')
    return expected


def _write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return artifact_reference(path, value['schema'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check')
    parser.add_argument('--scope')
    parser.add_argument('--serial-launch')
    parser.add_argument('--cpu-ids', type=_cpu_list)
    parser.add_argument('--worker-address-space-bytes', type=int)
    parser.add_argument('--max-workers', type=int)
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    build = (args.scope, args.serial_launch, args.cpu_ids, args.worker_address_space_bytes, args.max_workers, args.out)
    if args.check and not any(build):
        check_trial_choice(artifact_reference(Path(args.check).resolve(), CHOICE))
        result = dict(verified=True)
    elif not args.check and all(build):
        out = Path(args.out).resolve()
        out.mkdir(parents=True)
        hardware = _write(out/'hardware.json', collect_hardware(args.cpu_ids))
        choice = derive_trial_choice(artifact_reference(Path(args.scope).resolve(), SCOPE),
            artifact_reference(Path(args.serial_launch).resolve(), SERIAL_LAUNCH), hardware,
            worker_address_space_bytes=args.worker_address_space_bytes, max_workers=args.max_workers)
        result = dict(choice=_write(out/'choice.json', choice), selected=choice['choice'])
    else:
        parser.error('provide --check or the source, serial launch, CPU and memory bounds, and output')
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
