"""Export an existing source-replayed fusion into the shared compact C1 runtime.

The producer replays the macro source/cost decision and native body, then copies
one unchanged operator ELF. The separate compact cost scope stays UNKNOWN;
macro profitability does not authorize compact deployment or target execution.
"""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _json(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def _receipt(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return dict(path=str(path), sha256=sha256(raw).hexdigest(), bytes=len(raw))


def _source(value):
    return {name:getattr(value, name) for name in ('path','sha256','code_key','qualname')}


def export_fused_pipeline_runtime(build_result, out):
    """Cold-admit a selected macro result and export one original fused ELF."""
    from scar.compiler.macro_pipeline import check
    from scar.compiler.pipeline_fusion_delivery import _admit_artifact
    from scar.compiler.pipeline_model import pipeline_digest
    from scar.codegen.fused_pipeline import load, SCHEMA as NATIVE_SCHEMA
    from scar.ir.execution_recipe import artifact_reference, read_artifact
    from scar.runtime.caller_pipeline_guard import (SCHEMA, ADMISSION, TRUST,
        FUSED_FAMILY, FUSED_COST_SCOPE, validate_contract, implementation_sha256, inspect_bundle)

    output = Path(out).resolve()
    _need(not output.exists(), 'fresh compact fused runtime output directory required')
    reference = (deepcopy(build_result) if type(build_result) is dict else
        artifact_reference(Path(build_result).resolve(), 'scar.macro-pipeline-build'))
    published = read_artifact(reference)
    _need(published.get('schema_version') == 1 and published.get('decision') == 'REWRITE'
        and published.get('native_compiled') is True and published.get('native_artifact') is not None,
        'source-qualified selected and compiled macro fusion required')
    saved, pipeline, region, blueprint, selection = check(published['preparation'])
    _need(selection['decision'] == published['decision'] == 'REWRITE',
        'macro build and replayed source/cost decision differ')
    native_ref = published['native_artifact']
    native_manifest = read_artifact(native_ref)
    _need(native_ref['schema'] == NATIVE_SCHEMA, 'existing fused pipeline artifact required')
    artifact = load(Path(native_ref['uri']).parent)
    _need(native_ref == artifact_reference(artifact.directory/'artifact.json', NATIVE_SCHEMA),
        'macro native reference must identify the original loaded manifest')
    _admit_artifact(artifact, blueprint)
    identity = artifact.report
    _need(identity['pipeline_sha256'] == pipeline_digest(pipeline) == blueprint['pipeline_sha256']
        and native_manifest['build_id'] == identity['build_id'],
        'native fusion differs from the replayed original caller')
    operator = artifact._operator
    operator_identity = operator.report
    _need(operator.plan.resource_binding is None and operator.plan.cpu_threads == 1
        and operator.timed_admission is None and operator_identity['cxx_standard'] == 20
        and operator_identity['operator_abi_version'] == 1,
        'compact fusion requires the existing unbound single-thread ABI1 operator')
    _need(not any(str(arg).startswith(('-m','-target','--target'))
        for arg in operator_identity['compiler'][1:]),
        'compact baseline x86_64 profile cannot admit explicit machine ISA overrides')

    original_inputs = read_artifact(saved['inputs'])
    memory = artifact.memory_plan
    allowed = blueprint['execution']['allowed_cpu_ids']
    raw_library = operator.library_path.read_bytes()
    library_sha = sha256(raw_library).hexdigest()
    _need(library_sha == operator_identity['sha256']['library'], 'original operator ELF changed')
    library_name = 'operator-'+library_sha[:24]+'.so'
    row = dict(library=library_name, sha256=library_sha, bytes=len(raw_library),
        plan_sha256=operator_identity['plan_sha256'], input_count=len(pipeline.parameters),
        cpu_threads=1, resource_binding=None)
    native = dict(family=FUSED_FAMILY, operator_abi_version=1, operators=[row],
        pipeline_sha256=blueprint['pipeline_sha256'], memory_plan_sha256=blueprint['memory_plan_sha256'],
        strategy_sha256=blueprint['strategy_sha256'], optimized_plan_sha256=blueprint['optimized_plan_sha256'],
        blueprint_sha256=_digest(blueprint), region_document_sha256=_digest(region),
        pass_report_sha256=_digest(selection), build_id=identity['build_id'], native_ffi_calls_per_run=1)
    callees = {}
    for call in pipeline.calls:
        name = call.callee.lookup_path[0]
        guard = dict(global_name=name, source=_source(call.callee.source))
        _need(name not in callees or callees[name] == guard, 'original global callee identities disagree')
        callees[name] = guard
    def array(shape):
        return dict(shape=list(shape), nbytes=shape[0]*shape[1]*8,
            dtype='float64', layout='C_CONTIGUOUS')
    region_ref = saved['files']['region']
    qualification = region['qualification']
    contract = dict(schema=SCHEMA, schema_version=2, source=_source(pipeline.source),
        parameters=list(pipeline.parameters), callees=list(callees.values()),
        inputs={name:array(memory['input_shapes'][name]) for name in pipeline.parameters},
        output=array(memory['output_shape']), native=native,
        cpu=dict(ids=[], allowed_cpu_ids=list(allowed), policy='INHERIT_CALLER_CPU',
            cpu_domain_is_exclusive_reservation=False, os='linux', architecture='x86_64',
            pointer_bits=64, byteorder='little', required_isa='BASELINE_X86_64'),
        memory=dict(workspace_bytes=0, final_bytes=memory['final_bytes'],
            peak_bytes=memory['final_bytes'], budget_bytes=memory['memory_budget_bytes'],
            scope='INCREMENTAL_PAYLOAD_FRESH_FINAL_ONLY', policy='UNBOUND',
            budget_excludes=deepcopy(memory['budget_excludes'])),
        defaults_policy='NO_DEFAULT_PARAMETERS', input_writes=False, fresh_output=True,
        original_proof=original_inputs['proof'], original_conditional_call_count=len(pipeline.calls),
        original_Q=dict(artifact=region_ref, locator='#/qualification/Q'),
        original_contract=dict(artifact=region_ref, locator='#/qualification/contract'),
        source_region_artifact_sha256=region_ref['sha256'],
        original_source_refs=deepcopy(region['source_refs']),
        source_region_id=deepcopy(region['region_id']),
        source_alternative_ids=deepcopy(region['alternative_ids']),
        selected_source_alternative=deepcopy(region['alternative_ids']['FULL_FUSION']),
        runtime_guards=dict(source_and_code_ids='EACH_CALL',
            global_callee_identity_and_default_state='EACH_CALL',
            closures_and_custom_signature_metadata='EACH_CALL', global_and_local_monitoring='EACH_CALL',
            input_shape_dtype_rank_layout_alignment_and_backing_bounds='EACH_CALL',
            caller_affinity_equals_cost_domain='EACH_CALL_NO_PINNING',
            native_floating_status_and_finite_loaded_inputs='EXISTING_OPERATOR_EACH_CALL',
            native_failure='WITHHOLD_PRIVATE_OUTPUT_CALL_ENTIRE_ORIGINAL_ONCE'),
        cost_scope=FUSED_COST_SCOPE, cost_status='UNKNOWN', automatic_deployment_authorized=False,
        execution_policy='NO_OP_UNLESS_EXPLICIT_CANDIDATE_MEASUREMENT')
    if any('EXACT_NUMPY_NDARRAY_COPY_METHOD_DESCRIPTOR' in call.plan.region.assumptions
            for call in pipeline.calls):
        from scar.runtime.array_copy_guard import runtime_copy_contract
        contract['array_copy_contract'] = runtime_copy_contract()
        contract['runtime_guards']['ndarray_copy_descriptor_and_input_type'] = 'EACH_CALL'
    validate_contract(contract)
    runtime_raw = _json(contract)
    source_parents = dict(macro_build=reference, preparation=published['preparation'],
        region=region_ref, blueprint=saved['files']['blueprint'], pass_report=saved['files']['pass_report'],
        original_F4_inputs=saved['inputs'], original_fused_artifact=native_ref)
    admission = dict(schema=ADMISSION, schema_version=2, trust_policy=TRUST,
        runtime_sha256=sha256(runtime_raw).hexdigest(), native=native,
        runtime_implementation_sha256=implementation_sha256(),
        producer=dict(name='SCAR', exporter_sha256=sha256(Path(__file__).read_bytes()).hexdigest()),
        source_parents=source_parents, original_Q=contract['original_Q'],
        original_contract=contract['original_contract'],
        original_targets=[deepcopy(blueprint['target'])],
        original_libraries=[dict(original=_receipt(operator.library_path), bundle_member=library_name)],
        original_f4_and_matched_interface_replayed=True,
        original_source_effect_Q_preserved=True, additional_external_assumptions=[],
        source_qualification_sha256=_digest(qualification), native_bytes_identical=True,
        cost_scope=FUSED_COST_SCOPE, cost_status='UNKNOWN',
        macro_warm_cost_does_not_authorize_compact_execution=True,
        automatic_deployment_authorized=False)
    admission_raw = _json(admission)
    pin = sha256(admission_raw).hexdigest()
    output.mkdir(parents=True)
    (output/library_name).write_bytes(raw_library)
    (output/'runtime.json').write_bytes(runtime_raw)
    (output/'admission.json').write_bytes(admission_raw)
    inspect_bundle(output, expected_admission_sha256=pin)
    report = dict(schema='scar.caller-pipeline-runtime-export', schema_version=2,
        bundle=str(output), expected_admission_sha256=pin, admission_sha256=pin,
        admission=artifact_reference(output/'admission.json', ADMISSION),
        runtime=artifact_reference(output/'runtime.json', SCHEMA),
        execution_family=FUSED_FAMILY, source_parents=source_parents,
        pipeline_sha256=identity['pipeline_sha256'], memory_plan_sha256=identity['memory_plan_sha256'],
        original_native_manifest=_receipt(artifact.directory/'artifact.json'),
        native_bytes_identical=True, operator_abi_version=1, host_abi_present=False,
        live_guard_adapter_implemented=True, cost_scope=FUSED_COST_SCOPE, cost_status='UNKNOWN',
        original_source_effect_Q_preserved=True, additional_external_assumptions=[],
        automatic_deployment_authorized=False, default_backend='NO_OP',
        target_program_executed=False, numerical_backend_added=False)
    (output/'export-report.json').write_bytes(_json(report))
    return report


__all__ = ['export_fused_pipeline_runtime']


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    report = export_fused_pipeline_runtime(args.build, args.out)
    print(json.dumps({name:report[name] for name in ('bundle', 'expected_admission_sha256',
        'execution_family', 'cost_scope', 'cost_status', 'native_bytes_identical')}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
