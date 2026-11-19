"""Build and replay one source-qualified CUDA placement for calibration."""

from scar.compiler.aten_export_device_placement import placement_choices, placement_proposal
from scar.compiler.aten_export_hybrid_plan import plan_placed_hybrid_export
from scar.ir.v2.schemas import canonical_json
from scar.optimization.native_device_placement_search import placement_assignment


PLACEMENT_TREATMENT_SCHEMA = 'scar.native-export-placement-treatment'


def plan_device_treatment(projection, model, *, model_ref, ordinal):
    """Form the physical plan for one requested non-CPU placement ordinal."""
    choices = placement_choices(projection)
    placement = placement_assignment(choices, ordinal)
    if 'CUDA' not in placement.values():
        raise ValueError('a calibration treatment requires a CUDA placement ordinal')
    problem, schedule, bindings = plan_placed_hybrid_export(projection, model, model_ref=model_ref,
                                                            placement=placement)
    schedule.pop('elapsed_seconds', None)
    report = placement_proposal(projection, choices, None, dict(placement=placement))
    del report['analysis']
    report.update(schema=PLACEMENT_TREATMENT_SCHEMA, ordinal=ordinal,
                  program=projection['program'], declared_model=model_ref,
                  problem_sha256=schedule['problem_sha256'],
                  decision='EXPLICIT_CALIBRATION_TREATMENT', target_program_executed=False,
                  execution_authorized=False, model_optimality_proven=False, physical_optimality_proven=False)
    return problem, schedule, bindings, report


def replay_device_treatment(projection, model, *, model_ref, problem, schedule, bindings, placement_treatment):
    """Rebuild the selected source assignment and verify its archived certificate."""
    expected = plan_device_treatment(projection, model, model_ref=model_ref,
                                     ordinal=placement_treatment.get('ordinal'))
    stable_schedule = {key: value for key, value in schedule.items() if key != 'elapsed_seconds'}
    if canonical_json(expected) != canonical_json((problem, stable_schedule, bindings, placement_treatment)):
        raise ValueError('CUDA calibration treatment differs from its replayed source placement or schedule')
