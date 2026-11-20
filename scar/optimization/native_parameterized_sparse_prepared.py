"""Plan a checked immutable CSR input for a count-parameterized native entry."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from struct import Struct

from scar.analysis.replay_transaction import replay_transaction
from scar.codegen import native_parameterized_sparse_entry as entry
from scar.codegen.native_parameterized_sparse_solver import pack_header
from scar.ir.execution_recipe import artifact_reference, read_artifact


SCHEMA = 'scar.native-parameterized-sparse-preparation-plan'
_HEADER = Struct('<8s32s4I')


def _require(condition, detail):
    if not condition:
        raise ValueError('parameterized sparse preparation: ' + detail)


def _bytes(record):
    origin, offset, size = (record[key] for key in ('origin', 'offset', 'size_bytes'))
    _require(type(offset) is type(size) is int and offset == _HEADER.size and size > 0
             and origin['size_bytes'] == offset + size, 'expected one exactly framed canonical CSR source')
    raw = read_artifact(origin, json_document=False)
    _require(sha256(raw[offset:]).hexdigest() == record['sha256'], 'canonical CSR numerical body differs')
    return raw


def payload(plan):
    """Read the checked numerical body of a replayed plan for package export."""
    _require(plan['schema'] == SCHEMA and plan['schema_version'] == 1,
             'expected a checked preparation plan')
    record = plan['snapshot']
    return _bytes(record)[record['offset']:]


def _evidence(plan, native):
    scope, geometry = (plan[key] for key in ('scope', 'invariant_geometry'))
    protocol, validation = (native[key] for key in ('protocol', 'csr_boundary_validation'))
    header, framing = protocol['header'], protocol['input']
    minimum, maximum = (scope[key] for key in ('minimum_count', 'maximum_count'))
    shape = (header['size'], header['nnz'])
    output = protocol['output']
    _require(plan['decision']['native_experiment_authorized'] is True
             and scope['full_original_effect_equivalence'] is native['deployment_authorized'] is False
             and geometry['count_independent'] is geometry['metadata_is_count_dependent'] is True
             and geometry['whole_wire_reusable'] is False
             and geometry['upstream_shape'] == dict(size=shape[0], nnz=shape[1])
             and header['format'] == _HEADER.format and header['flags'] == 0
             and header['count'] == dict(minimum=minimum, maximum=maximum)
             and framing['framing'] == 'REQUIRE_EOF_BEFORE_ANY_DISPATCH'
             and output['profile']['emission'] == 'ONLY_AFTER_EXACT_COUNT_AND_NORMAL_COMPLETION'
             and output['profile']['original_python_effects_discharged'] is False
             and output['success_bytes'] == 65 and output['failure_stdout_bytes'] == 0,
             'requires invariant geometry and the same guarded count, solver and publication contracts')
    snapshot = dict(origin=validation['fixed_csr'], offset=validation['metadata_bytes'],
                    size_bytes=validation['payload_bytes'], sha256=validation['preserved_payload_sha256'])
    raw = _bytes(snapshot)
    offset = snapshot['offset']
    body = raw[offset:]
    fixed_manifest = read_artifact(plan['artifacts']['fixed_package'])
    magic, recipe, size, nnz, count, flags = _HEADER.unpack_from(raw)
    _require(magic == b'SCARPSR1' and 'recipe-v1:' + recipe.hex() == fixed_manifest['recipe_id']
             and (size, nnz, flags) == (*shape, header['flags'])
             and count == scope['fixed_profile_count']
             and snapshot['sha256'] == geometry['numeric_payload_sha256']['canonical_csr']
             and validation['checked_count'] == maximum
             and validation['emitted_wire_sha256'] == sha256(pack_header(protocol, maximum) + body).hexdigest(),
             'fixed bytes or native count-boundary replay disagrees with the source proof')

    counts, wires = scope['archived_source_counts'], geometry['historical_wires']
    _require(len(set(counts)) == len(counts) >= 2, 'needs independently observed source counts')
    comparisons = []
    for count, historical in zip(counts, wires, strict=True):
        member = historical['wires']['canonical_csr']
        record = dict(snapshot, origin=member['artifact'], offset=member['metadata_bytes'])
        previous = _bytes(record)
        old_magic, old_recipe, old_size, old_nnz, old_count, old_flags = _HEADER.unpack_from(previous)
        historical_entry = read_artifact(historical['artifact'])
        _require(old_magic == b'SCARPSR1' and 'recipe-v1:' + old_recipe.hex() == historical_entry['recipe_id']
                 and (old_size, old_nnz, old_count, old_flags) == (*shape, count, header['flags'])
                 and previous[offset:] == body, 'historical body bytes or count-specific envelope differ')
        comparisons.append(dict(count=count, **record))
    _require(len({row['origin']['sha256'] for row in comparisons}) == len(comparisons),
             'whole historical wires must remain distinct')
    return snapshot, comparisons


def _graph(plan, protocol, snapshot):
    owner = protocol['placement']['owner_cpu_id']
    workers = protocol['placement']['worker_cpu_ids']
    prepare = ('source_topology', 'source_mesh', 'source_basis', 'source_coo_csr')
    image_bytes = snapshot['size_bytes'] + snapshot['offset']
    stages = dict(count_guard=dict(kind='CONTROL', cpu_ids=protocol['placement']['launch_cpu_ids']),
                  pin_owner=dict(kind='AFFINITY', cpu_ids=[owner]),
                  embedded_payload=dict(kind='EMBEDDED_READ_ONLY_EXECUTABLE_BYTES',
                                        logical_bytes=snapshot['size_bytes']),
                  frame_csr=dict(kind='REPRESENTATION', cpu_ids=[owner],
                                 header_bytes=snapshot['offset']),
                  seal_solver_input=dict(kind='ANONYMOUS_SEALED_MEMFD', cpu_ids=[owner],
                                         logical_bytes=image_bytes),
                  spawn_solver=dict(kind='PROCESS_LAUNCH', cpu_ids=protocol['placement']['launch_cpu_ids']),
                  ordered_solver=dict(kind='REGION', cpu_ids=[owner, *workers]),
                  commit_digest=dict(kind='STDOUT', cpu_ids=[owner],
                                     logical_bytes=protocol['output']['success_bytes']))
    return dict(stages=stages, dependencies=[
        dict(before='count_guard', after='pin_owner', requirement='ADMITTED_COUNT_AND_AFFINITY'),
        dict(before='pin_owner', after='frame_csr', requirement='MATERIALIZE_ON_OWNER_CPU'),
        dict(before='count_guard', after='frame_csr', requirement='CURRENT_COUNT'),
        dict(before='embedded_payload', after='seal_solver_input', requirement='EXACT_NUMERICAL_BODY'),
        dict(before='frame_csr', after='seal_solver_input', requirement='TYPED_CURRENT_HEADER'),
        dict(before='seal_solver_input', after='spawn_solver', requirement='SEALED_REWOUND_STDIN'),
        dict(before='spawn_solver', after='ordered_solver',
             requirement='RESTORED_CPU_COHORT_AND_LIVE_ADMISSION_AND_VALIDATED_EOF'),
        dict(before='ordered_solver', after='commit_digest', requirement='EXACT_COUNT_AND_SUCCESSFUL_CHILD_EXIT'),
    ], replaced_stages=list(prepare), retained_loop=plan['control']['loop'],
        resource_admission=plan['resources']['launch'], solver_memory=protocol['memory'],
        added_artifact_payload_bytes=snapshot['size_bytes'], staged_input_logical_bytes=image_bytes,
        physical_residency=None, additional_physical_headroom=None,
        failure_stdout_bytes=protocol['output']['failure_stdout_bytes'])


@replay_transaction()
def propose(entry_path):
    """Replay the source and compiled boundary before proposing a preparation cut."""
    path = Path(entry_path).resolve(strict=True)
    manifest = path / 'artifact.json' if path.is_dir() else path
    reference = artifact_reference(manifest, entry.SCHEMA)
    native = entry.check(manifest)
    _require(native == read_artifact(reference), 'runtime entry differs from its checked artifact bytes')
    plan = read_artifact(native['parameterized_plan'])
    protocol = native['protocol']
    snapshot, historical = _evidence(plan, native)
    return dict(schema=SCHEMA, schema_version=1, entry=reference, source=native['source'],
        protocol=protocol, scope=plan['scope'], snapshot=snapshot,
        evidence=dict(source_count=plan['artifacts']['source_count'], historical=historical,
                      fixed_runtime_boundary=native['csr_boundary_validation']),
        graph=_graph(plan, protocol, snapshot),
        legality=dict(status='ADMITTED_FOR_EXACT_NATIVE_EXPERIMENT',
            replacement='IMMUTABLE_NUMERICAL_BODY_AND_FRESH_PROTOCOL_HEADER',
            build_and_replay_guards=['PINNED_SOURCE_GEOMETRY_AND_NUMERICAL_PROVIDER',
                                     'EXACT_EMBEDDED_BODY_BYTES_AND_SHA256'],
            runtime_guards=['LIVE_CPU_AND_MEMORY_ADMISSION', 'CURRENT_COUNT_AND_EXACT_SOLVER_PROTOCOL',
                            'SEALED_REWOUND_STDIN_AND_VALIDATED_EOF_BEFORE_DISPATCH'],
            snapshot_use='EMBEDDED_READ_ONLY_IMAGE_TO_SEALED_ANONYMOUS_STDIN', forward_result_cache=False,
            whole_historical_wire_reusable=False, original_source_effects_closed=False,
            original_source_frontiers=plan['source_frontiers']),
        cost=dict(status='UNKNOWN', historical_wall_times_used=False, added_storage_bytes=snapshot['size_bytes'],
            full_producer_path_ns=None, embedded_read_and_sealed_stage_ns=None, whole_entry_delta_ns=None,
            basis='FRESH_SAME_COUNT_SAME_PLACEMENT_DIRECT_PROCESS_A_B_REQUIRED'),
        decision=dict(selected='RETAIN_RUNTIME_PREPARATION', native_experiment_authorized=True,
                      deployment_authorized=False))


def check(plan_path):
    """Rebuild a preparation plan from checked inputs and compare every field."""
    reference = artifact_reference(Path(plan_path).resolve(strict=True), SCHEMA)
    stored = read_artifact(reference)
    rebuilt = propose(stored['entry']['uri'])
    _require(json.dumps(stored, sort_keys=True, allow_nan=False)
             == json.dumps(rebuilt, sort_keys=True, allow_nan=False),
             'stored preparation plan differs from cold replay')
    return rebuilt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('entry', 'out', 'check'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args(argv)
    if args.check and not args.entry and not args.out:
        result = check(args.check)
    elif args.entry and args.out and not args.check:
        result = propose(args.entry)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    else:
        parser.error('provide --entry and --out, or --check PLAN')
    print(json.dumps({key: result[key] for key in ('schema', 'snapshot', 'decision')}, sort_keys=True))


if __name__ == '__main__':
    main()
