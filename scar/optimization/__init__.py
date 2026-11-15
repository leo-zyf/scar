"""Optimization of the guarded CPU array-region execution-plan profile."""

__all__ = ["optimize_region", "select_measured", "propose_cpu_placements",
           "bind_cpu_placement", "validate_cpu_placement", "build_region_schedule",
           "connect_materialization", "schedule_document", "schedule_from_dict",
           "validate_schedule", "solve_resource_schedule", "hardware_model_skeleton",
           "semantic_sha256", "predict_region_costs"]

_MODULES = {
    'optimize_region':'optimizer', 'select_measured':'selection',
    **{name:'hardware' for name in ('bind_cpu_placement','propose_cpu_placements','validate_cpu_placement')},
    **{name:'schedule' for name in ('build_region_schedule','connect_materialization','schedule_document',
                                  'schedule_from_dict','validate_schedule')},
    **{name:'resource_solver' for name in ('hardware_model_skeleton','semantic_sha256','solve_resource_schedule')},
    'predict_region_costs':'cost_model',
}


def __getattr__(name):
    # Metadata-only deployment cost decisions must not initialize the compiler
    # or target-graph stack. Public optimization APIs retain their real owners.
    if name not in _MODULES:raise AttributeError(name)
    from importlib import import_module
    value=getattr(import_module('.'+_MODULES[name],__name__),name)
    globals()[name]=value
    return value
