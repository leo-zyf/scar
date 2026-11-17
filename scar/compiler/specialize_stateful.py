"""Observed input paths to a source-replayed stateful shape specialization.

The original source and scalar inputs are unchanged. The hardware allocation
is an explicit validation preview, never a cost-ranked optimum or deployment.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference, read_artifact

SCHEMA = 'scar.observed-stateful-specialization'


def _ref(value, schema):
    return value if type(value) is dict else artifact_reference(Path(value).resolve(), schema)


def _write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return artifact_reference(Path(path), value['schema'])


def specialize(program, inputs, out, *, memory_budget_bytes, max_threads=1,
               affinity=None, compile_native=False, boundary=None):
    from scar.compiler.input_metadata import stateful_shapes_from_metadata
    from scar.compiler.stateful_model import stateful_from_dict, stateful_digest
    from scar.optimization.stateful_cpu_choices import build_stateful_cpu_domain
    from scar.compiler.stateful_candidate import prepare_stateful_candidate
    directory = Path(out).resolve()
    if directory.exists():
        raise ValueError('fresh observed specialization output required')
    program_path = Path(program).resolve()
    if program_path.is_dir(): program_path /= 'manifest.json'
    parent = artifact_reference(program_path, 'scar.program-build')
    program_data = read_artifact(parent)
    if program_data.get('backend') != 'F4' or program_data.get('target') is None:
        raise ValueError('original F4 program with retained hardware required')
    observed = _ref(inputs, 'scar.region-input-metadata')
    metadata = read_artifact(observed)
    selected = metadata.get('boundary_artifact') if boundary is None else _ref(boundary, 'scar.stateful-array-region')
    if selected is None or selected.get('schema') != 'scar.stateful-array-region':
        raise ValueError('observed stateful boundary reference required; legacy metadata may supply --boundary')
    region = stateful_from_dict(read_artifact(selected))
    shapes, evidence = stateful_shapes_from_metadata(observed['uri'], region)
    directory.mkdir(parents=True)
    domain = build_stateful_cpu_domain(selected['uri'], Path(program_data['target']['path']).parent,
        input_shapes=shapes, memory_budget_bytes=memory_budget_bytes, affinity=affinity,
        max_threads=max_threads, out=directory/'domain')
    domain_ref = artifact_reference(directory/'domain/domain.json', 'scar.stateful-cpu-domain')
    candidate = None
    if compile_native:
        prepare_stateful_candidate(program_path, domain, directory/'candidate', cpu_threads=1)
        candidate = artifact_reference(directory/'candidate/candidate.json', 'scar.stateful-validation-candidate')
    read_artifact(parent); read_artifact(observed); read_artifact(selected)
    report = dict(schema=SCHEMA, schema_version=1, program=parent, inputs=observed,
        source_region=selected, region_id=region.id, region_sha256=stateful_digest(region),
        shape_evidence=evidence, input_shapes=shapes, scalar_values_specialized=False,
        domain=domain_ref, candidate=candidate, native_compiled=candidate is not None,
        source_F4_replayed=candidate is not None,
        source_authority='ORIGINAL_F4_EXACT_REDISCOVERY' if candidate is not None else 'PENDING_CANDIDATE_F4_ADMISSION',
        resource_allocation_policy='EXPLICIT_ONE_WORKER_PREVIEW_LOWEST_LEGAL_CPU_MASK_UNRANKED',
        no_workload_source_rewrite=True, no_shape_or_cost_rebase=True,
        original_scalar_state_loaded_each_call=True, input_values_retained=False,
        all_future_calls_proven=False, runtime_guards_required=True,
        target_program_executed=False, performance_measured=False,
        default_backend='NO_OP', selected_for_use=False, deployment_authorized=False,
        implementation_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    _write(directory/'result.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--program', required=True); parser.add_argument('--inputs', required=True)
    parser.add_argument('--out', required=True); parser.add_argument('--boundary')
    parser.add_argument('--memory-budget-bytes', type=int, required=True)
    parser.add_argument('--max-threads', type=int, default=1)
    parser.add_argument('--cpu-ids', help='optional allowed validation CPUs, not a ranked optimum')
    parser.add_argument('--compile', action='store_true')
    args = parser.parse_args(argv)
    result = specialize(args.program, args.inputs, args.out, memory_budget_bytes=args.memory_budget_bytes,
        max_threads=args.max_threads, affinity=None if args.cpu_ids is None else
            [int(value) for value in args.cpu_ids.split(',')],
        compile_native=args.compile, boundary=args.boundary)
    print(json.dumps({key:result[key] for key in ('region_id','input_shapes','native_compiled',
        'source_F4_replayed','scalar_values_specialized','selected_for_use')}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
