"""Attach independently replayed costs to exact native placement treatments."""

from pathlib import Path

from scar.compiler.assess_native_device import _check_file_records, _locked_files, _paths
from scar.compiler.aten_export_placement import placement_choices, placement_proposal
from scar.compiler.aten_export_hybrid_plan import build_placed_hybrid_problem, plan_placed_hybrid_export
from scar.compiler.native_export_placement_costs import CpuPlacementCostUnavailable, admit_pageable_cpu_costs
from scar.compiler.native_export_placement_costs import load_native_hybrid_placement_costs, match_native_hybrid_placement_costs, native_hybrid_placement_cost_matches
from scar.ir.v2.schemas import canonical_json
from scar.optimization.device_cost_bounds import empirical_placement_candidate
from scar.optimization.native_device_placement_search import search_native_device_placements


def _sample_records(projection, inputs, references, named_inputs):
    user_ports = [dict(id=row['id'], size_bytes=row['nbytes'])
                  for row in projection['inputs'] if row['kind'] == 'INPUT']
    outputs = {row['id']: dict(id=row['id'], size_bytes=row['nbytes']) for row in projection['operations']}
    output_ports = [outputs[name] for name in projection['outputs']]
    if inputs is None:
        inputs = _locked_files(_paths(named_inputs, projection), user_ports)
        references = _locked_files(references, output_ports)
    return (inputs, references, _check_file_records(inputs, user_ports),
            _check_file_records(references, output_ports, reference=True))


def _evidence_by_placement(cpu, cpu_sha, profile, device_evidence, model_ref, input_ids, reference_ids):
    evidence, rows, contexts = {}, [], set()
    for project, calibration, digest in device_evidence:
        project = Path(project).resolve(strict=True)
        calibration = Path(calibration).resolve(strict=True)
        loaded = load_native_hybrid_placement_costs(calibration, calibration_sha256=digest,
            cpu_project=cpu, cuda_project=project)
        record = loaded.record
        if (record['cpu_manifest_sha256'] != cpu_sha or record['cpu_profile'] != profile
                or record['profile_id'] != model_ref['sha256']
                or record['inputs'] != input_ids or record['references'] != reference_ids):
            raise ValueError('calibration does not bind the same CPU entry, declared device profile and sample files')
        key = tuple(sorted(record['placement'].items()))
        if key in evidence:
            raise ValueError('provide at most one replayed calibration per exact device placement')
        evidence[key] = loaded
        rows.append(dict(project=str(project), calibration=str(calibration), sha256=digest, evidence=record))
        if record['cost_available']:
            contexts.add(canonical_json(record['context']))
    if len(contexts) > 1:
        raise ValueError('observed placement costs require one matched device and runtime context')
    return evidence, rows


def propose_empirical_device_placement(cpu, cpu_sha, profile, projection, model, model_ref, *,
                                      inputs=None, named_inputs=(), references, device_evidence=(),
                                      service_intervals=True):
    inputs, references, input_ids, reference_ids = _sample_records(projection, inputs, references, named_inputs)
    evidence, evidence_rows = _evidence_by_placement(cpu, cpu_sha, profile, device_evidence, model_ref,
                                                     input_ids, reference_ids)
    choices = placement_choices(projection)
    cpu_record = {'status': 'NOT_EVALUATED'}
    matches = []

    def build(placement):
        nonlocal cpu_record
        key = tuple(sorted(placement.items()))
        observed = evidence.get(key)
        # Only a supplied treatment needs its original declared schedule replayed.
        build_problem = build_placed_hybrid_problem if observed is None else plan_placed_hybrid_export
        original = build_problem(projection, model, model_ref=model_ref, placement=placement)
        if 'CUDA' not in placement.values():
            try:
                costs, treatment = admit_pageable_cpu_costs(cpu, cpu_manifest_sha256=cpu_sha,
                    cpu_profile=profile, projection=projection, problem=original[0], inputs=inputs,
                    references=references)
            except CpuPlacementCostUnavailable as error:
                cpu_record = dict(status='UNAVAILABLE', reason=str(error))
                return empirical_placement_candidate(placement, original, unavailable_reason='UNMEASURED_PAGEABLE_CPU')
            cpu_record = dict(status='ADMITTED', measured_activities=len(costs), treatment=treatment)
            return empirical_placement_candidate(placement, original, pageable_costs=costs)
        matched = None
        reason = None
        if observed is not None:
            scope = dict(profile_id=model_ref['sha256'], input_identities=input_ids, reference_identities=reference_ids)
            reason = 'CALIBRATION_TREATMENT_DIFFERS'
            available = observed.record['cost_available']
            if available:
                matched = match_native_hybrid_placement_costs(observed, placement, original,
                                                             host_dispatch=service_intervals, **scope)
            if matched is not None:
                matches.append(observed.record['calibration_sha256'])
            elif native_hybrid_placement_cost_matches(observed, placement, original, **scope):
                reason = 'FIXED_HOST_DISPATCH_SCHEDULE_UNRESOLVED' if available else observed.record['reason']
        return empirical_placement_candidate(placement, original, matched_hybrid=matched, unavailable_reason=reason)

    analysis, selected = search_native_device_placements(choices, build, cost_basis='EMPIRICAL_MODEL',
                                                         service_intervals=service_intervals)
    proposal = placement_proposal(projection, choices, analysis, selected,
                                  actual_device_timing='EXACT_TREATMENT_SERVICE_SAMPLES' if matches else 'UNKNOWN')
    proposal['empirical'] = dict(inputs=inputs, references=references, pageable_cpu=cpu_record,
                                 device_evidence=evidence_rows, matched_calibrations=matches)
    return proposal
