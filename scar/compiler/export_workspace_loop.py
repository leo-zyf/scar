"""Seal an original source-proven workspace loop for the shared compact runtime.

Export replays the full build-side source admission, without loading either ELF.
The compact consumer retains the original loop guard and per-step transaction.
"""
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

from .export_loop import _json_bytes, _receipt, _source, _profile_artifacts


def export_workspace_loop(native_directory, out):
    """Export current original-F4 loop authority; fixed compositions are rejected."""
    from scar.codegen.stateful_workspace import inspect_workspace_repeat
    from scar.compiler.export_stateful import project_stateful_contract
    from scar.compiler.stateful_loop_frontend import StatefulLoopProof
    from scar.compiler.stateful_model import document, stateful_digest
    from scar.ir.execution_recipe import artifact_reference, read_artifact
    from scar.optimization.resource_solver import semantic_sha256
    from scar.runtime.native_stateful_repeat import (SCHEMA, ADMISSION, TRUST,
        WORKSPACE_PROFILE, WORKSPACE_FAMILY, validate_contract, implementation_sha256,
        inspect_bundle)

    directory = Path(native_directory).resolve()
    if directory.is_file():
        directory = directory.parent
    output = Path(out).absolute()
    if output.resolve() != output or output.exists() or output.is_relative_to(directory):
        raise ValueError('fresh independent canonical workspace runtime bundle required')

    # This is the existing complete source/backend replay, not a new proof path.
    metadata, qualified = inspect_workspace_repeat(directory)
    preparation, region, step_plan, workspace_plan, context = qualified
    proof, loop_plan = context['loop_proof'], context['loop_plan']
    if (preparation['source_kind'] != 'ORIGINAL_STATEFUL_LOOP'
            or context['source_kind'] != 'ORIGINAL_STATEFUL_LOOP'
            or type(proof) is not StatefulLoopProof or type(loop_plan) is not dict):
        raise ValueError('compact caller replacement requires the original F4 loop, not fixed-input composition')
    references = {row['role']: row['artifact'] for row in context['source_refs']}
    proof_reference, plan_reference = references['loop_proof'], references['loop_plan']
    # Canonical JSON treats original typed tuples and their published lists alike.
    if (stateful_digest(region) != stateful_digest(proof.step)
            or _json_bytes(loop_plan['step_plan']) != _json_bytes(step_plan)
            or metadata['trip_count'] != loop_plan['trip_count']
            or workspace_plan['trip_count'] != loop_plan['trip_count']
            or metadata['workspace_plan_sha256'] != semantic_sha256(workspace_plan)
            or metadata['execution_plan_sha256'] != semantic_sha256(metadata['execution_plan'])):
        raise ValueError('original loop, step, workspace and execution plan identities differ')

    base_reference = metadata['base_native_reference']
    if base_reference != context['base_native_reference']:
        raise ValueError('workspace parent differs from the original source-loop base')
    base_directory = Path(base_reference['uri']).parent.resolve()
    if output.is_relative_to(base_directory):
        raise ValueError('workspace export must not write inside its original base')
    base_metadata = read_artifact(base_reference)
    if ({key: base_metadata[key] for key in metadata['base_ids']} != metadata['base_ids']
            or base_metadata['region_sha256'] != stateful_digest(region)
            or base_metadata['plan_sha256'] != semantic_sha256(step_plan)
            or _json_bytes(read_artifact(references['region'])) != _json_bytes(document(region))
            or _json_bytes(read_artifact(references['step_plan'])) != _json_bytes(step_plan)):
        raise ValueError('workspace and original source-loop base references differ')
    # inspect_workspace_repeat already admitted this exact metadata, source and
    # renderer. Project the same object protocol without constructing providers/CDLL.
    base = SimpleNamespace(directory=base_directory, region=region, plan=step_plan,
                           report=base_metadata)
    step = project_stateful_contract(base, selection=None, require_selection=False)

    evidence, retained = [], {}
    def retain(path, role, **expected):
        row = _receipt(path, role, **expected)
        prior = retained.get(row['path'])
        if prior is not None:
            if (prior['sha256'], prior['bytes']) != (row['sha256'], row['bytes']):
                raise ValueError('one workspace export evidence path has conflicting versions')
            return prior
        retained[row['path']] = row
        evidence.append(row)
        if len(evidence) > 4096:
            raise ValueError('workspace export evidence budget exceeded')
        return row

    native_receipt = retain(directory/'artifact.json', 'original-workspace-native-manifest')
    base_receipt = retain(base_reference['uri'], 'original-base-native-manifest',
        expected_sha256=base_reference['sha256'], expected_bytes=base_reference['size_bytes'])
    for parent, report, prefix in ((directory, metadata, 'workspace'),
                                    (base_directory, base_metadata, 'base')):
        for key, name in report['files'].items():
            retain(parent/name, 'original-'+prefix+'-'+key, expected_sha256=report['sha256'][key])
    for role, reference in references.items():
        retain(reference['uri'], 'original-'+role, expected_sha256=reference['sha256'],
               expected_bytes=reference['size_bytes'])
    retain(metadata['preparation']['uri'], 'original-workspace-preparation',
        expected_sha256=metadata['preparation']['sha256'], expected_bytes=metadata['preparation']['size_bytes'])
    retain(preparation['workspace_plan']['uri'], 'original-workspace-plan',
        expected_sha256=preparation['workspace_plan']['sha256'], expected_bytes=preparation['workspace_plan']['size_bytes'])
    sources = {value.path: value.sha256 for value in
        (proof.source, proof.step.source, *(row.source for row in region.callees))}
    source_rows = [retain(path, 'original-source-snapshot', expected_sha256=digest)
                   for path, digest in sources.items()]
    for reference in _profile_artifacts(step_plan['resource_binding']):
        retain(reference['uri'], 'original-target-fact', expected_sha256=reference['sha256'],
               expected_bytes=reference['size_bytes'])
    provider = base_metadata['provider_runtime']['loaded_binary']
    retain(provider['path'], 'original-NumPy-provider-binary', expected_sha256=provider['sha256'],
           expected_bytes=provider['bytes'])

    def library(parent, report):
        path = parent/report['files']['library']
        raw = path.read_bytes()
        if sha256(raw).hexdigest() != report['sha256']['library']:
            raise ValueError('native ELF changed during workspace export')
        return dict(library=path.name, sha256=sha256(raw).hexdigest(), bytes=len(raw)), raw
    base_row, base_bytes = library(base_directory, base_metadata)
    repeat_row, repeat_bytes = library(directory, metadata)
    if base_row['library'] == repeat_row['library']:
        raise ValueError('workspace and original base library names collide')
    capacities = [row['capacity_bytes'] for row in workspace_plan['slots']]
    contract = dict(schema=SCHEMA, schema_version=1, profile=WORKSPACE_PROFILE,
        execution_family=WORKSPACE_FAMILY, source=_source(proof.source), step_source=_source(proof.step.source),
        parameters=list(proof.parameters), parameter_map=dict(proof.parameter_map),
        trip_count_parameter=proof.trip_count_parameter, trip_count=loop_plan['trip_count'],
        step_global_name=proof.step_global_name,
        call_arguments=dict(positional_parameters=list(proof.call_evidence['positional_parameters']),
            keyword_parameters=[list(pair) for pair in proof.call_evidence['keyword_parameters'].items()]),
        step_contract=step, native=dict(stateful_abi_version=1, repeat_abi_version=1,
            symbol='scar_workspace_repeat_stateful_v1', base=base_row, repeat=repeat_row,
            base_ids=step['native']['ids']),
        workspace=dict(plan_sha256=metadata['workspace_plan_sha256'],
            execution_plan_sha256=metadata['execution_plan_sha256'], slot_capacities_bytes=capacities),
        original_loop_proof=proof_reference, original_loop_plan=plan_reference,
        original_step_reference=proof.step_reference,
        conditions=list(dict.fromkeys([*loop_plan['required_live_guards'], *workspace_plan['required_live_guards']])),
        external_assumptions=list(dict.fromkeys([*loop_plan['external_assumptions'], *workspace_plan['external_assumptions']])),
        commit_policy='EACH_SUCCESSFUL_ORIGINAL_ABI_CALL_COMMITS_BEFORE_NEXT_STEP',
        failure_policy='RESUME_ONLY_UNCOMMITTED_ORIGINAL_SUFFIX_WITH_DYNAMIC_CALLEE_LOOKUP',
        source_loop_replacement_authority='ORIGINAL_F4_LOOP_AND_STEP_REPLAY_AT_EXPORT',
        fixed_input_binding_is_not_invariant_array_content=True,
        workspace_allocations_hoisted=True, team_setup_hoisted=False, workspace_or_team_setup_hoisted=True,
        cost_status='UNKNOWN', trial_only=True, deployment_authorized=False)
    validate_contract(contract)
    runtime_raw = _json_bytes(contract)
    runtime_pin = implementation_sha256()
    admission = dict(schema=ADMISSION, schema_version=1, profile=WORKSPACE_PROFILE, trust_policy=TRUST,
        runtime_sha256=sha256(runtime_raw).hexdigest(), runtime_implementation_sha256=runtime_pin,
        native=contract['native'], producer=dict(name='SCAR', exporter_sha256=sha256(Path(__file__).read_bytes()).hexdigest()),
        original_loop_proof=proof_reference, original_loop_plan=plan_reference,
        original_base_manifest=base_receipt, original_repeat_manifest=native_receipt,
        original_workspace_preparation=metadata['preparation'], original_workspace_plan=preparation['workspace_plan'],
        source_refs=source_rows, evidence=evidence, original_f4_loop_and_step_replayed=True,
        native_math_unchanged=True, cost_status='UNKNOWN')
    admission_raw = _json_bytes(admission)
    pin = sha256(admission_raw).hexdigest()
    for row in evidence:
        _receipt(row['path'], row['role'], expected_sha256=row['sha256'], expected_bytes=row['bytes'])
    if implementation_sha256() != runtime_pin:
        raise ValueError('shared compact runtime changed during workspace export')
    output.mkdir(parents=True)
    (output/base_row['library']).write_bytes(base_bytes)
    (output/repeat_row['library']).write_bytes(repeat_bytes)
    (output/'runtime.json').write_bytes(runtime_raw)
    (output/'admission.json').write_bytes(admission_raw)
    inspect_bundle(output, expected_admission_sha256=pin)  # Metadata/ELF only; no CDLL.
    result = dict(schema='scar.stateful-workspace-loop-runtime-export', schema_version=1,
        bundle=str(output), expected_admission_sha256=pin, profile=WORKSPACE_PROFILE,
        runtime=artifact_reference(output/'runtime.json', SCHEMA),
        admission=artifact_reference(output/'admission.json', ADMISSION),
        proof=proof_reference, plan=plan_reference, workspace_plan=preparation['workspace_plan'],
        native_bytes_identical=True, original_f4_loop_and_step_replayed=True,
        per_step_commit_boundary_preserved=True, completed_cursor_and_original_suffix=True,
        workspace_allocations_hoisted=True, team_setup_hoisted=False, workspace_or_team_setup_hoisted=True,
        fixed_input_composition_caller_replacement=False, numerical_backend_added=False,
        native_libraries_loaded=False, target_program_executed=False, performance_measured=False,
        cost_status='UNKNOWN', trial_only=True, default_backend='NO_OP', deployment_authorized=False)
    (output/'export-report.json').write_bytes(_json_bytes(result))
    return result


def main(argv=None):
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    report = export_workspace_loop(args.native, args.out)
    print(json.dumps({key: report[key] for key in
        ('bundle', 'expected_admission_sha256', 'profile', 'default_backend')}, sort_keys=True))
    return 0


__all__ = ['export_workspace_loop', 'main']

if __name__ == '__main__':
    raise SystemExit(main())
