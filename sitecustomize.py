"""Auto-loaded by Python child processes launched by SCAR."""
import os

_scar_region_metadata = None

if os.environ.get('SCAR_CALL_BINDING_PLAN'):
    try:
        from scar.trace.call_binding import install as _scar_install_call_binding
        _scar_call_binding = _scar_install_call_binding()
    except Exception as _scar_binding_error:
        if os.environ.get('SCAR_TRACE_DEBUG'):
            print(f'SCAR call bindings disabled: {_scar_binding_error}', flush=True)

if os.environ.get('SCAR_REGION_METADATA_PLAN'):
    try:
        from scar.trace.region_metadata import install as _scar_install_region_metadata
        _scar_region_metadata = _scar_install_region_metadata()
    except Exception as _scar_metadata_error:
        if os.environ.get('SCAR_TRACE_DEBUG'):
            print(f'SCAR input metadata disabled: {_scar_metadata_error}', flush=True)

if os.environ.get("SCAR_TRACE_DIR"):
    try:
        if os.environ.get('SCAR_TRACE_CALL_EDGES') == '1':
            from scar.trace.call_edges import install
            _scar_lean_plan = None
            if os.environ.get('SCAR_CAPTURE_PLAN') is not None:
                from scar.trace.lean_capture import read_plan as _scar_read_plan
                _scar_lean_plan = _scar_read_plan(os.environ['SCAR_CAPTURE_PLAN'],
                    expected_sha256=os.environ['SCAR_CAPTURE_PLAN_SHA256'],
                    max_codes=int(os.environ.get('SCAR_CALL_MAX_CODES','4096')))
            _scar_recorder = install(os.environ['SCAR_TRACE_DIR'],os.environ['SCAR_CALL_PROJECT_ROOT'],
                                     lean_plan=_scar_lean_plan)
            if _scar_recorder is not None and _scar_region_metadata is not None:
                _scar_region_metadata.attach_frame_recorder(_scar_recorder)
            if _scar_recorder is not None and getattr(_scar_recorder, 'collector_witness', None) is not None:
                import sys as _scar_sys
                _scar_recorder.collector_witness.register_loader(_scar_sys._getframe())
            if _scar_recorder is not None and os.environ.get('SCAR_CALL_OPERANDS') == '1':
                from scar.trace.call_operands import install as _scar_install_operands
                _scar_install_operands(_scar_recorder,
                    python_targets=os.environ.get('SCAR_PYTHON_CALL_TARGETS') == 'PYTHON_CALL_TARGETS_V2')
            if _scar_recorder is not None and os.environ.get('SCAR_EXECUTION_EVIDENCE') == '1':
                import atexit as _scar_atexit
                from scar.trace.execution_evidence import install as _scar_install_evidence
                _scar_evidence = _scar_install_evidence(os.environ['SCAR_TRACE_DIR'], recorder=_scar_recorder)
                # Close provider artifacts before the earlier call-recorder
                # callback. It remains the sole Python profile-hook owner.
                _scar_atexit.register(_scar_evidence.close_at_exit)
            if _scar_recorder is not None and os.environ.get('SCAR_COMPILE_EVENTS') == '1':
                import atexit as _scar_atexit
                from scar.trace.compilation_audit import install as _scar_install_compilation
                _scar_compilation = _scar_install_compilation(_scar_recorder)
                _scar_atexit.register(_scar_compilation.close)
        else:
            from scar.trace.inject import install
            install()
    except Exception as _scar_error:
        # A profiler must never make the target program fail to start.
        if os.environ.get("SCAR_TRACE_DEBUG"):
            print(f"SCAR injector disabled: {_scar_error}", flush=True)
elif os.environ.get("SCAR_OPTIMIZE"):
    try:
        from scar.runtime.torch_optimizer import install
        install()
    except Exception as _scar_error:
        # The backend is opt-in; a failed optimizer must leave the original
        # target execution available for comparison.
        if os.environ.get("SCAR_TRACE_DEBUG"):
            print(f"SCAR optimizer disabled: {_scar_error}", flush=True)

if os.environ.get('SCAR_PARTITION_SPAWN_PLAN'):
    try:
        from scar.runtime.partition_workers import install_spawn_worker as _scar_install_partition_worker
        _scar_install_partition_worker()
    except (OSError, ValueError, ImportError) as _scar_error:
        if os.environ.get('SCAR_TRACE_DEBUG'):
            print(f'SCAR partition worker disabled: {_scar_error}', flush=True)
