"""Replay and publish the compact direct and composite temporal ABI1 bundles."""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from scar.ir.execution_recipe import read_artifact
from scar.optimization.resource_solver import semantic_sha256
from scar.runtime.native_temporal_invariant import ADMISSION, SCHEMA, TRUST, validate_contract


_SERIAL = 'CPP_SERIAL_TEMPORAL'
_OPENMP = 'CPP_OPENMP_EXPLICIT_ROW_PARTITION_TEMPORAL'


def _json(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def _receipt(path, raw=None):
    path = Path(path).resolve()
    raw = path.read_bytes() if raw is None else raw
    return dict(path=str(path), sha256=sha256(raw).hexdigest(), bytes=len(raw))


def _reference_receipt(reference):
    return dict(path=str(Path(reference['uri']).resolve()), sha256=reference['sha256'], bytes=reference['size_bytes'])


def _source(value):
    return {name: value[name] for name in ('path', 'sha256', 'code_key', 'qualname')}


def _source_receipts(sources):
    receipts = {}
    for source in sources:
        path = source['path']
        if path not in receipts:
            receipts[path] = _receipt(path)
        receipt = receipts[path]
        if receipt['sha256'] != source['sha256']:
            raise ValueError('original temporal source changed after proof replay: ' + path)
        yield receipt


def read_temporal_export_source(reference, *, catalog, composite):
    """Read the exact original F4 proof and its source-movement witness."""
    if composite:
        from scar.compiler.pipeline_loop_frontend import (
            COMPOSITE_PROOF_SCHEMA, read_composite_invariant_loop_motion as read_motion,
        )

        if reference['schema'] != COMPOSITE_PROOF_SCHEMA:
            raise ValueError('separate composite proof reference required; v1 cast is not admission')
    else:
        from scar.compiler.loop_admission import read_invariant_loop_motion as read_motion

    document = read_artifact(reference)
    proof, motion = read_motion(reference['uri'], catalog=catalog)
    return proof, motion, tuple(document['compositions']) if composite else ()


def _verify_schedule(proof, schedule, artifact, *, catalog, composite):
    from scar.optimization.invariant_loop_schedule import verify_invariant_loop_schedule

    schema = schedule['schema']
    parallel = artifact.report['schema'] == 'scar.native-parallel-temporal-invariant-loop'
    cost_schema = 'scar.cost-selected-composite-temporal-cpu-resource-plan' if composite else 'scar.cost-selected-temporal-cpu-resource-plan'
    preview_schema = 'scar.parallel-composite-temporal-resource-plan' if composite else 'scar.parallel-temporal-invariant-loop-resource-plan'
    options = dict(catalog=catalog) if composite else {}
    cost_selected = schema == cost_schema
    if parallel:
        from scar.optimization.temporal_cpu_choices import verify_selected_temporal_cpu_choice, verify_temporal_cpu_preview

        if schema not in (cost_schema, preview_schema):
            raise ValueError('parallel export requires its original replayable CPU-team resource schedule')
        verify = verify_selected_temporal_cpu_choice if cost_selected else verify_temporal_cpu_preview
        verify(proof, schedule['target_path'], schedule, **options)
        scheduled, compiled = schedule['backend'], artifact.report['capability']
        if any(semantic_sha256(scheduled[key]) != semantic_sha256(compiled[key])
               for key in ('family', 'compiler', 'generators', 'row_partition')):
            raise ValueError('parallel compiled backend differs from scheduled capability')
    else:
        if composite and schema != 'scar.composite-invariant-loop-resource-plan':
            raise ValueError('separate replayable composite resource plan required; v1 plan casts are not admission')
        verify_invariant_loop_schedule(proof, schedule['target_path'], schedule, **options)
    if schedule['selected_kind'] != 'REGISTER_TEMPORAL_FUSION' or schedule['trip_count'] <= 0:
        raise ValueError('temporal export requires an admitted positive-count resource schedule')
    return parallel, cost_selected


def publish_temporal_export(artifact, out, *, schedule, schedule_receipt, proof_reference, source,
                            catalog, composite, native_receipts=None):
    """Copy a verified temporal operator without loading or regenerating it again."""
    proof, source_motion, compositions = source
    plan = artifact.plan
    interface = plan['loop_interface']
    motion_ref = interface.get('source_motion')
    motion_hash = semantic_sha256(source_motion)
    consumed = plan.get('execution_movement_binding')
    if (motion_ref is None or source_motion['proof'] != proof_reference
            or semantic_sha256(read_artifact(motion_ref)) != motion_hash
            or consumed is None or consumed['source_motion'] != motion_ref
            or consumed['source_motion_sha256'] != motion_hash):
        raise ValueError('temporal export requires its consumed source movement certificate')
    proof_hash = semantic_sha256(asdict(proof))
    if composite:
        if (interface.get('proof_kind') != proof.profile or interface.get('synthetic_body_is_original_callable') is not False
                or semantic_sha256(interface.get('original_composite_f4_proof')) != proof_hash
                or semantic_sha256(interface.get('root_callee_source')) != semantic_sha256(asdict(proof.root_callee_source))
                or semantic_sha256(interface.get('function_global_guards')) != semantic_sha256([asdict(g) for g in proof.nested_callee_guards])):
            raise ValueError('compiled composite body/root/routes differ from original separate F4 proof')
    elif semantic_sha256(interface.get('original_f4_proof')) != proof_hash:
        raise ValueError('temporal export original F4 proof differs from compiled interface')

    parallel, cost_selected = _verify_schedule(proof, schedule, artifact, catalog=catalog, composite=composite)
    return _publish_bundle(artifact, out, schedule=schedule, schedule_receipt=schedule_receipt,
        proof_reference=proof_reference, proof=asdict(proof), compositions=compositions, composite=composite,
        parallel=parallel, cost_selected=cost_selected, native_receipts=native_receipts)


def publish_prepared_temporal_export(artifact, out, *, schedule, schedule_reference, proof,
                                     proof_reference, compositions=()):
    """Seal a selected temporal operator without reopening source analysis."""
    composite = proof_reference['schema'] == 'scar.composite-invariant-loop-proof'
    if bool(compositions) != composite:
        raise ValueError('prepared temporal proof requires its original composition receipts')
    parallel = artifact.report['schema'] == 'scar.native-parallel-temporal-invariant-loop'
    schema = schedule['schema']
    cost = 'scar.cost-selected-composite-temporal-cpu-resource-plan' if composite else 'scar.cost-selected-temporal-cpu-resource-plan'
    if parallel:
        preview = ('scar.parallel-composite-temporal-resource-plan' if composite
                   else 'scar.parallel-temporal-invariant-loop-resource-plan')
        allowed = (preview, cost)
    else:
        allowed = ('scar.composite-invariant-loop-resource-plan' if composite else 'scar.invariant-loop-resource-plan',)
    if schema not in allowed or schedule['selected_kind'] != 'REGISTER_TEMPORAL_FUSION' or schedule['trip_count'] <= 0:
        raise ValueError('prepared temporal export requires its selected native schedule profile')
    if parallel:
        scheduled, compiled = schedule['backend'], artifact.report['capability']
        if any(semantic_sha256(scheduled[key]) != semantic_sha256(compiled[key])
               for key in ('family', 'compiler', 'generators', 'row_partition')):
            raise ValueError('parallel compiled backend differs from scheduled capability')
    return _publish_bundle(artifact, out, schedule=schedule, schedule_receipt=_reference_receipt(schedule_reference),
        proof_reference=proof_reference, proof=proof, compositions=compositions, composite=composite,
        parallel=parallel, cost_selected=schema == cost, native_receipts=None)


def _publish_bundle(artifact, out, *, schedule, schedule_receipt, proof_reference, proof,
                    compositions, composite, parallel, cost_selected, native_receipts):
    plan = artifact.plan
    interface = plan['loop_interface']
    motion_ref = interface['source_motion']
    proof_hash = semantic_sha256(proof)
    binding = plan['execution_plan']['resource_binding']
    if (binding is None or semantic_sha256(binding) != semantic_sha256(schedule['resource_binding'])
            or plan['memory_budget_bytes'] != schedule['effective_incremental_budget_bytes']
            or plan['body_sha256'] != schedule['body_sha256']):
        raise ValueError('temporal export compiled resource/body boundary differs from schedule')
    specialization = interface.get('execution_specialization')
    if ((specialization is None and not composite) or specialization is not None and (
            specialization['input_shapes'] != schedule['input_shapes']
            or specialization['trip_count'] != schedule['trip_count']
            or composite and specialization.get('input_layout') != 'C_CONTIGUOUS')):
        raise ValueError('temporal export requires the scheduled shape/count specialization')

    # A freshly emitted plan/report have exactly these canonical bytes. Cold
    # export instead supplies receipts for its actual retained formatting.
    if native_receipts is None:
        native_receipts = (_receipt(artifact.directory / 'artifact.json', _json(artifact.report)),
                           _receipt(artifact.directory / 'plan.json', _json(plan)))
    evidence = [*native_receipts, schedule_receipt, _reference_receipt(proof_reference)]
    if composite:
        leaf = proof['root_callee_source']
        global_name = proof['root_global_name']
        sources = (proof['source'], leaf, *(source for guard in proof['nested_callee_guards']
                                         for source in (guard['owner_source'], guard['target_source'])))
        evidence.extend((_reference_receipt(motion_ref), *_source_receipts(sources)))
    else:
        leaf = proof['body']['source']
        global_name = proof['leaf_global_name']
        evidence.extend((*_source_receipts((proof['source'], leaf)), _reference_receipt(motion_ref)))
    evidence.extend(artifact.report['generators'])
    additional = compositions
    if cost_selected and not composite:
        additional = (*schedule['resource_facts'], schedule['no_op_profile'])
    for reference in additional:
        read_artifact(reference)
        evidence.append(_reference_receipt(reference))

    native_bytes = (artifact.directory / 'temporal.so').read_bytes()
    digest = sha256(native_bytes).hexdigest()
    if (digest, len(native_bytes)) != (artifact.report['library']['sha256'], artifact.report['library']['bytes']):
        raise ValueError('temporal native library changed after source admission')
    family = _OPENMP if parallel else _SERIAL
    threads = len(binding['cpu_ids'])
    contract = dict(schema=SCHEMA, schema_version=1,
        native=dict(library='temporal.so', library_sha256=digest, operator_descriptor_abi_version=1,
                    temporal_control_abi_version=1, control_family=family),
        source=_source(proof['source']), leaf=dict(global_name=global_name, source=_source(leaf)),
        parameters=list(proof['parameters']), input_order=list(proof['body']['parameters']), parameter_map=proof['parameter_map'],
        carry_parameter=proof['carry_parameter'], trip_count_parameter=proof['trip_count_parameter'],
        trip_count=schedule['trip_count'], arrays={name: dict(shape=shape, dtype='float64', layout='C_CONTIGUOUS')
                                                  for name, shape in schedule['input_shapes'].items()},
        memory_budget_bytes=schedule['effective_incremental_budget_bytes'], memory_scope=plan['memory_scope'],
        memory_policy='UNBOUND', cpu=dict(ids=list(binding['cpu_ids']), threads=threads, os='linux',
            architecture='x86_64', pointer_bits=64, byteorder='little', required_isa='BASELINE_X86_64'),
        floating_environment_policy='ORIGINAL_CPP_SAVE_RESTORE_FENV_STRICT_OPERAND_ORDER_PRIVATE_FAILURE_OUTPUT',
        Q=plan['Q'], source_proof=proof_reference, resource_schedule=dict(id=schedule['id'], receipt=schedule_receipt),
        profitability_status='UNPROVEN_NO_SOURCED_TIMING_MODEL', deployment_authorized=False,
        validation_execution_requires_trial=True)
    if composite:
        guards = [dict(owner_source=_source(g['owner_source']), lookup_name=g['lookup_name'],
                       target_source=_source(g['target_source']), global_path=list(g['global_path']),
                       lookup_semantics=g['lookup_semantics']) for g in proof['nested_callee_guards']]
        contract.update(proof_kind='COMPOSITE', body_identity=proof['body_identity'], derived_body_sha256=plan['body_sha256'],
            root_callee_source=_source(leaf), nested_callee_guards=guards, original_composite_f4_proof_sha256=proof_hash,
            specialization_authority='COMPILED_INTERFACE' if specialization is not None else 'EXPORT_RUNTIME_NARROWING_OF_GENERIC_NATIVE',
            selection_authority='COMPLETE_COMPOSITE_CPU_FORWARD_COST_MODEL_NOT_ENTRY_PERFORMANCE_SELECTION' if cost_selected
                else 'EXPLICIT_COMPOSITE_CPU_TEAM_PREVIEW_NOT_COST_SELECTION' if parallel
                else 'STRUCTURAL_COMPOSITE_LOGICAL_WORK_NOT_PERFORMANCE_SELECTION')
        result_fields = dict(proof_kind='COMPOSITE', original_composite_f4_replayed=True, synthetic_body_is_original_callable=False)
        admission_fields = dict(result_fields, control_family=family, cpu_threads=threads, resource_plan_schema=schedule['schema'])
        schema = 'scar.composite-temporal-invariant-runtime-export'
    else:
        contract.update(original_f4_proof_sha256=proof_hash,
            selection_authority='COMPLETE_SOURCED_FORWARD_RESOURCE_MODEL_NOT_FULL_ENTRY_PERFORMANCE' if cost_selected
                else 'EXPLICIT_CPU_TEAM_PREVIEW_NOT_COST_SELECTION' if parallel
                else 'STRUCTURAL_LOGICAL_WORK_RESOURCE_DOMINANCE_NOT_PERFORMANCE_SELECTION')
        if cost_selected:
            contract.update(resource_cost_basis=schedule['resource_cost_basis'], forward_model_selected=True,
                            full_entry_performance_qualified=False)
        admission_fields = dict(original_f4_replayed=True, native_math_changed=False)
        result_fields = dict(original_f4_replayed=True, control_family=family, cpu_threads=threads)
        schema = 'scar.temporal-invariant-runtime-export'

    raw = _json(validate_contract(contract))
    admission = _json(dict(schema=ADMISSION, schema_version=1, runtime_sha256=sha256(raw).hexdigest(),
        library_sha256=digest, trust_policy=TRUST, evidence=evidence,
        producer=dict(name='SCAR', version='0.1.0', exporter=_receipt(__file__)),
        resource_schedule_replayed=True, performance_qualified=False, deployment_authorized=False, **admission_fields))
    verified = {}
    for receipt in evidence:
        path = receipt['path']
        if path not in verified:
            verified[path] = _receipt(path)
        if verified[path] != receipt:
            raise ValueError('temporal export evidence changed during admission')
    read_artifact(proof_reference, json_document=False)
    if composite:
        for reference in (motion_ref, *compositions):
            read_artifact(reference, json_document=False)
    out = Path(out).resolve()
    if out.exists() or out.is_relative_to(artifact.directory):
        raise ValueError('fresh separate temporal bundle directory required')
    out.mkdir(parents=True)
    (out / 'temporal.so').write_bytes(native_bytes)
    (out / 'runtime.json').write_bytes(raw)
    (out / 'admission.json').write_bytes(admission)
    result = dict(schema=schema, schema_version=1, bundle=str(out), expected_admission_sha256=sha256(admission).hexdigest(),
        runtime_sha256=sha256(raw).hexdigest(), library_sha256=digest, source_directory=str(artifact.directory),
        schedule=schedule_receipt, proof=proof_reference, resource_schedule_replayed=True, native_recompiled=False,
        native_library_bytes_unchanged=True, performance_qualified=False, deployment_authorized=False,
        runtime_ir_or_codegen_required=False, bundle_bytes=len(raw) + len(admission) + len(native_bytes), **result_fields)
    (out / 'export-report.json').write_bytes(_json(result))
    return result


def _export_temporal(native, out, *, schedule_path, proof_reference, catalog, composite):
    from scar.compiler.temporal_invariant_loop import load_temporal_invariant_loop

    source = read_temporal_export_source(proof_reference, catalog=catalog, composite=composite)
    directory = Path(native).resolve()
    receipts = tuple(_receipt(directory / name) for name in ('artifact.json', 'plan.json'))
    artifact = load_temporal_invariant_loop(directory)
    path = Path(schedule_path).resolve()
    raw = path.read_bytes()
    return publish_temporal_export(artifact, out, schedule=json.loads(raw), schedule_receipt=_receipt(path, raw),
        proof_reference=proof_reference, source=source, catalog=catalog, composite=composite, native_receipts=receipts)


def export_temporal_invariant(native_directory, out, *, schedule_path, proof_reference, catalog):
    return _export_temporal(native_directory, out, schedule_path=schedule_path, proof_reference=proof_reference,
                            catalog=catalog, composite=False)


def export_composite_temporal_invariant(native, out, *, schedule_path, proof_reference, catalog):
    return _export_temporal(native, out, schedule_path=schedule_path, proof_reference=proof_reference,
                            catalog=catalog, composite=True)


__all__ = ['export_temporal_invariant', 'export_composite_temporal_invariant']
