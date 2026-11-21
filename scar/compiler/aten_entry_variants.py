"""Enumerate a source-qualified fixed-grid CPU entry domain before CLI lowering."""

from pathlib import Path
import re

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.resource_solver import semantic_sha256


SCHEMA = 'scar.aten-entry-variants'
BODY_SCHEMA = 'scar.aten-recurrence-delivery'
FAMILY = 'ATEN_FIXED_GRID_CPU_FILE_ENTRY'
_MODES = ('READY_DAG', 'SERIAL')


class _ScopeFrontier(ValueError):
    def __init__(self, reason, detail):
        super().__init__(detail)
        self.reason = reason


def _require(condition, reason, detail):
    if not condition:
        raise _ScopeFrontier(reason, detail)


def _bind_scope(scope, emission):
    from scar.compiler.aten_source_entry import _inputs

    abi = emission['abi']
    origin = abi['solver']['source']
    source = artifact_reference(origin['path'], 'scar.python-source')
    _require((source['uri'], source['sha256'], source['size_bytes'])
             == (scope['entry'], scope['entry_sha256'], origin['size_bytes'])
             and source['sha256'] == origin['sha256'],
             'ENTRY_SOURCE_MISMATCH', 'qualified recurrence source differs from the F4 entry')
    _require(emission['target']['sha256'] == scope['target_sha256'],
             'TARGET_MISMATCH', 'qualified recurrence target differs from the F4 hardware snapshot')
    _require(emission['storage']['budget_bytes'] == scope['memory_budget_bytes'],
             'TENSOR_BUDGET_MISMATCH', 'qualified recurrence requires its exact modeled Tensor budget')
    argv, slots = scope['argv'], abi['argv']
    _require(len(argv) == slots['count'], 'ABI_ARGUMENT_MISMATCH', 'entry argument count differs from the source ABI')
    value = argv[slots['steps'] - 1]
    digits = value.lstrip('0') or '0'
    _require(re.fullmatch('[0-9]+', value) is not None and len(digits) <= 10
             and abi['domain']['step_count_min'] <= int(digits) <= abi['domain']['step_count_max'],
             'ABI_STEP_DOMAIN_MISMATCH', 'step count must satisfy the original signed-int32 decimal domain')
    input_root, output = (Path(argv[slots[name] - 1]) for name in ('input_root', 'output'))
    _require(input_root.is_absolute() and output.is_absolute() and output.parent.is_dir()
             and not output.is_symlink() and not output.exists(),
             'FILE_BOUNDARY_MISMATCH', 'absolute Tensor inputs and a fresh output with an existing parent are required')
    common_shape = scope.get('input_shapes', {}).get('common_shape')
    _require(common_shape is None or all(row['shape'] == common_shape for row in abi['inputs']),
             'INPUT_SHAPE_MISMATCH', 'declared input shapes differ from the qualified source ports')
    return source, int(digits), _inputs(abi, input_root)


def _cost_demand(body, emission, observation, alternative, *, steps, file_inputs):
    placement = emission['placement']
    input_bytes = sum(row['size_bytes'] for row in file_inputs)
    output_bytes = next(row['size_bytes'] for row in file_inputs
                        if row['port'] == emission['abi']['solver']['state_input_index'])
    inputs_match = [{key: val for key, val in row.items() if key != 'port'} for row in file_inputs]
    inputs_match = inputs_match == observation['inputs']
    step_ns = alternative['body_estimate_ns']
    return dict(boundary='PROCESS_LAUNCH_THROUGH_EXIT', model='SCOPED_COMPONENT_DEMANDS',
        trip_count=steps, body=dict(estimated_step_ns=step_ns,
            estimated_total_ns=step_ns * steps if inputs_match else None,
            calibration_matches_exact_files=inputs_match, physical_schedule=emission['schedule'],
            calibration_repeats=observation['repeats'], claimed_process_bound=False),
        input_logical_bytes=input_bytes, output_logical_bytes=output_bytes,
        graph_held_peak_bytes=alternative['peak_tensor_storage_bytes'], unmodeled_memory=emission['storage']['excluded'],
        placement=placement, source_evidence=body,
        phases=dict(startup='ELF_LOADER_AND_ATEN_THREAD_CONFIGURATION', imports='INAPPLICABLE',
            guards='SOURCE_ARGV_DOMAIN_AND_INPUT_FILE_CHECKS', loading='INPUT_FILE_BACKING_AND_TENSOR_VIEWS',
            worker_lifecycle='FIXED_NATIVE_TEAM' if placement['mode'] == 'READY_DAG' else 'INAPPLICABLE',
            allocation='ATEN_TENSOR_AND_PROVIDER_WORKSPACE', file_io='READ_INPUT_AND_WRITE_OUTPUT',
            transfers='INAPPLICABLE', synchronization='CPU_TEAM_ORDER' if placement['mode'] == 'READY_DAG' else 'INAPPLICABLE',
            contention='ATEN_AND_HOST_RESOURCE_CONTENTION', execution='REPEATED_REPLAYED_CPU_STEP',
            publication='OUTPUT_FILE_THEN_STDOUT', fallback='INAPPLICABLE', teardown='RESOURCE_CLEANUP_AND_PROCESS_EXIT'),
        complete_process_ns=None)


def derive_aten_entry_variants(scope, body_ref):
    """Use original recurrence evidence to enumerate a finite source-entry family.

    The implementation IDs bind per-mode execution and the declared entry
    domain. Replay proves qualification on that domain. Estimates from the
    calibrated step are not full-entry bounds or deployment permission.
    """
    from scar.compiler.aten_source_entry_plan import plan_source_entry_emission

    scope_sha = semantic_sha256(scope)
    result = dict(schema=SCHEMA, schema_version=1, family=FAMILY, scope_sha256=scope_sha,
        body=body_ref, options=[], coverage_frontier=[], baseline_id='original',
        compiled_source_entry=False, target_executed=False)
    frontier = result['coverage_frontier']
    if body_ref is None:
        frontier.append(dict(reason='FIXED_GRID_BODY_EVIDENCE_REQUIRED'))
        return result
    try:
        _require(scope['boundary'] == 'PROCESS_LAUNCH_THROUGH_EXIT'
                 and scope['native_output_contract'] == 'STANDALONE_CPP',
                 'BOUNDARY_MISMATCH', 'fixed-grid generation requires a standalone process contract')
        _require(body_ref['schema'] == BODY_SCHEMA, 'BODY_SCHEMA_MISMATCH', 'CPU recurrence body evidence required')
        body = read_artifact(body_ref)
        _require(body.get('source_entry_replayable') is True and body.get('step_lifetime_planned') is True
                 and 'source_entry_cli' not in body and 'cpu_linkage' in body,
                 'BODY_PROOF_REQUIRED', 'source-linked CPU-only recurrence with replayable lifetime plans required')
        original_mode = body['rhs_mode']
        selected = plan_source_entry_emission(body_ref, original_mode)
        source, steps, file_inputs = _bind_scope(scope, selected)
    except _ScopeFrontier as error:
        frontier.append(dict(reason=error.reason, detail=str(error)))
        return result
    except (ValueError, OSError, KeyError, TypeError, IndexError) as error:
        frontier.append(dict(reason='BODY_EVIDENCE_REPLAY_REQUIRED', detail=str(error)))
        return result

    allowed_cpus = set(scope['process_affinity_cpu_ids'])
    alternatives = body['alternatives']
    for mode in _MODES:
        if mode not in alternatives:
            frontier.append(dict(mode=mode, reason='MODE_NOT_COVERED_BY_RETAINED_BODY_PROOF'))
            continue
        alternative = alternatives[mode]
        proof = dict(body=body_ref, mode=mode, source=source, source_contract_sha256=selected['source_contract_sha256'],
                     target=selected['target'], file_inputs=file_inputs, steps=steps,
                     memory_budget_bytes=scope['memory_budget_bytes'], peak_bytes=alternative['peak_tensor_storage_bytes'])
        demand = None
        status, reason = 'INELIGIBLE', 'GRAPH_HELD_STORAGE_EXCEEDS_BUDGET'
        if alternative['within_budget']:
            try:
                emission = selected if mode == original_mode else plan_source_entry_emission(body_ref, mode)
                if emission['source_contract_sha256'] != proof['source_contract_sha256'] or emission['target'] != proof['target']:
                    raise ValueError('alternate emission must preserve entry ABI and target')
                proof['emission_plan'] = emission
                if set(emission['placement']['cpus']) <= allowed_cpus:
                    demand = _cost_demand(body_ref, emission, read_artifact(emission['calibration']), alternative,
                                          steps=steps, file_inputs=file_inputs)
                    status, reason = 'LEGAL', 'REPLAYED_SOURCE_ABI_AND_CPU_BODY_PLAN'
                else:
                    reason = 'EXACT_CPU_PLACEMENT_OUTSIDE_ENTRY_COHORT'
            except (ValueError, OSError, KeyError, TypeError, IndexError) as error:
                status, reason = 'UNRESOLVED', 'MODE_PROOF_REPLAY_REQUIRED'
                proof['detail'] = str(error)
        identity = dict(scope_sha256=scope_sha, body_sha256=body_ref['sha256'], mode=mode)
        result['options'].append(dict(id='planned:' + semantic_sha256(identity), legal_status=status,
            reason=reason, proof=proof, cost_demand=demand))
    read_artifact(body_ref)
    return result
