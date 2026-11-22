"""Plan private file-backed input materialization for an admitted CPU recurrence."""

from math import prod
from pathlib import Path

from scar.analysis.replay_transaction import replay_transaction
from scar.ir.execution_recipe import read_artifact

SCHEMA = 'scar.cpu-fixed-grid-mapped-input-plan'
_PROOF = 'scar.cpu-fixed-grid-mapped-input-proof'
_STUDY = 'scar.cpu-fixed-grid-mapped-input-study'
_DELIVERY = 'scar.cpu-fixed-grid-cli-delivery'
_PROVEN = 'PROVEN_SCOPED_MAPPED_INPUT_MATERIALIZATION'
_OWNED, _MAPPED = 'OWNED_COPY', 'PROVIDER_MAP'
_VALIDATION = ('exact_stdout', 'exact_stderr', 'exact_exit_status', 'bitwise_final_file',
               'exact_publication_mode', 'input_files_unchanged')


def _require(condition, detail):
    if not condition:
        raise ValueError('CPU fixed-grid mapped input plan: ' + detail)


def _projection(proof):
    binding, abi = map(read_artifact, (proof['physical_binding'], proof['abi']))
    _require(proof['status'] == _PROVEN and binding['target'] == proof['target']
             and binding['placement'] == proof['placement']
             and binding['source_contract_sha256'] == proof['source_contract_sha256']
             and abi['domain']['normal_completion_only'] is True
             and abi['domain']['inputs'] == 'STABLE_READ_ONLY_EXACT_SIZE_REGULAR_CPU_FLOAT32_NATIVE_ENDIAN_FILES',
             'mapped inputs require the exact replayed project and stable read-only file scope')
    rows, seen = [], set()
    files = {row['port']: row for row in binding['files']}
    for row in proof['inputs']:
        port, path, shape, size, dtype = (row[key] for key in ('port', 'path', 'shape', 'payload_bytes', 'dtype'))
        _require(type(port) is int and port >= 0 and port in files and port not in seen and dtype == 'FLOAT32'
                 and type(shape) is list and shape
                 and all(type(axis) is int and axis > 0 for axis in shape)
                 and type(size) is int and size == 4 * prod(shape)
                 and row['element_count'] == size // 4
                 and all(row[key] == files[port][key] for key in ('path', 'shape', 'dtype', 'payload_bytes'))
                 and all(row[key] is False for key in ('scoped_declared_input_write', 'scoped_graph_result_alias',
                                                       'scoped_graph_escape')),
                 'invalid original CPU float32 input geometry or private lifetime')
        seen.add(port)
        rows.append(dict(port=port, path=path, shape=shape, dtype=dtype, payload_bytes=size,
                         element_count=size // 4))
    logical = sum(row['payload_bytes'] for row in rows)
    storage = binding['storage']
    _require(rows and seen == files.keys() and logical == proof['logical_copy_payload_bytes']
             == binding['costs']['logical_input_payload_bytes']
             and storage['scope'] == 'GRAPH_HELD_TENSOR_BACKING_STORAGE'
             and type(storage['peak_bytes']) is type(storage['budget_bytes']) is int
             and 0 < logical <= storage['peak_bytes'] <= storage['budget_bytes'],
             'mapped candidate must retain the original conservative graph-held backing budget')
    return rows, binding, abi


def _observed(reference, proof_ref, proof, inputs, binding, abi, *, archived):
    from scar.compiler.cpu_fixed_grid_mapped_input_study import (
        compare_native_input_runs, verify_cpu_fixed_grid_mapped_input_study,
    )

    _require(reference['schema'] == _STUDY, 'verified native mapped-input comparison required')
    observed = verify_cpu_fixed_grid_mapped_input_study(reference)
    delivery_ref = observed['control_delivery']
    _require(delivery_ref['schema'] == _DELIVERY, 'original owned-copy source CLI delivery required')
    control = read_artifact(delivery_ref)
    _require(observed['status'] == 'SCOPED_NATIVE_INPUT_BACKING_COMPARED'
             and observed['timing_scope'] == 'NATIVE_SOURCE_CLI_LAUNCH_THROUGH_EXIT_INCLUDES_FILE_IO'
             and observed['mapped_input_proof'] == proof_ref
             and observed['control_project'] == control['project'] == proof['project']
             and observed['mapped_project'] != proof['project']
             and observed['source'] == control['source'] == proof['source']
             and observed['target'] == control['target'] == proof['target']
             and observed['argv'] == control['argv']
             and observed['working_directory'] == control['working_directory']
             and observed['cpu_ids'] == control['cpu_ids'] == proof['placement']['cpus']
             and observed['native_placement'] == control['native_placement'] == proof['placement']
             and observed['storage'] == control['storage'] == binding['storage']
             and observed['file_inputs'] == control['file_inputs']
             and observed['deployment_authorized'] is False
             and all(observed['validation'].get(key) is True for key in _VALIDATION),
             'paired inputs, project, original launch or normal-success validation differ')
    input_root = Path(observed['argv'][abi['argv']['input_root'] - 1])
    paths = [str((input_root / row['path']).resolve(strict=True)) for row in inputs]
    _require([(row['port'], row['path'], row['size_bytes']) for row in observed['file_inputs']]
             == [(row['port'], path, row['payload_bytes']) for row, path in zip(inputs, paths, strict=True)],
             'observed files do not bind every planned mapped-input port in order')
    selection = observed['selection'] if archived else compare_native_input_runs(
        observed['runs'], observed['measurement_pairs'])
    if archived:
        _require(selection['policy'] == dict(minimum_gain=0.05, strict_observed_separation=True),
                 'legacy input plan must retain the original inclusive policy')
    diagnostic = observed.get('file_reads_evidence')
    return dict(
        reference=reference, mapped_project=observed['mapped_project'], control_delivery=delivery_ref,
        source=observed['source'], argv=observed['argv'], working_directory=observed['working_directory'],
        basis='OBSERVED_PAIRED_CXX_TO_CXX_FRESH_PROCESSES', validation=observed['validation'],
        selection=selection, measured_gain_gate_passed=selection['verdict'] == 'ADVANCE_TO_RELEASE_REVIEW',
        file_read_diagnostic=None if diagnostic is None else dict(
            locator='#/file_reads_evidence', pair_scope=diagnostic['pair_scope'],
            explicit_file_read_payload_removed_bytes=diagnostic['explicit_file_read_payload_removed_bytes'],
            bytes_mapped_at_offset_zero=diagnostic['bytes_mapped_at_offset_zero'],
            kernel_io_bytes_measured=False, page_fault_work_removed_proven=False),
        diagnostic_used_for_timing_gate=False, serves_as_analytic_bound=False)


def derive_cpu_fixed_grid_mapped_input_plan(proof_ref, *, observation=None, schema_version=2):
    """Keep source-level movement separate from the observed runtime choice."""
    from scar.analysis.cpu_fixed_grid_mapped_inputs import verify_cpu_fixed_grid_mapped_inputs

    _require(proof_ref['schema'] == _PROOF and type(schema_version) is int and schema_version in (1, 2),
             'exact mapped-input materialization proof and supported policy version required')
    with replay_transaction():
        proof = verify_cpu_fixed_grid_mapped_inputs(proof_ref)
        inputs, binding, abi = _projection(proof)
        evidence = None if observation is None else _observed(
            observation, proof_ref, proof, inputs, binding, abi, archived=schema_version == 1)
    switch = evidence is not None and evidence['measured_gain_gate_passed']
    status = ('MAPPED_INPUT_NOMINATED_FOR_SCOPED_RUNTIME' if switch else
              'KEEP_OWNED_COPY_NOT_PROFITABLE' if evidence is not None else
              'SCOPED_MAPPED_INPUT_EXPERIMENT_READY')
    logical = sum(row['payload_bytes'] for row in inputs)
    return dict(
        schema=SCHEMA, schema_version=schema_version, proof=proof_ref, observation=observation,
        project=proof['project'], target=proof['target'], source=proof['source'],
        source_contract_sha256=proof['source_contract_sha256'],
        scope=proof['scope'], placement=proof['placement'], provider=proof['provider'], inputs=inputs,
        status=status, default_runtime=_OWNED, selected_runtime=_MAPPED if switch else _OWNED,
        selected_project=evidence['mapped_project'] if switch else proof['project'],
        runtime_switch_eligible=switch, explicit_experiment_eligible=True,
        candidate=dict(materialization='ATEN_FROM_FILE_PRIVATE', operator='at::from_file', shared=False,
            same_read_only_file_inputs=True, unchanged_recurrence_and_operator_order=True),
        movement=dict(
            basis='EXPLICIT_EAGER_FILE_TO_OWNED_TENSOR_STAGING_ON_NORMAL_SUCCESS',
            baseline_eager_materializations=len(inputs), candidate_eager_materializations=0,
            removed_explicit_eager_copy_payload_bytes=logical, logical_input_payload_bytes=logical,
            storage_reads_eliminated_proven=False, physical_dram_traffic_reduction_proven=False,
            page_residency_or_physical_dram_savings_proven=False),
        storage=dict(
            original_graph_held=binding['storage'], candidate_graph_held=binding['storage'],
            counts_file_backed_tensor_payload=True, physical_memory_bound_proven=False),
        empirical=evidence,
        coverage_frontier=[
            dict(role='INITIAL_MATERIALIZATION', status='EXACT_SOURCE_LEVEL_EAGER_PAYLOAD_ACCOUNTING'),
            dict(role='PHYSICAL_COST', status='OPEN',
                 blockers=['STORAGE_IO_AND_PAGE_CACHE', 'DEMAND_PAGE_FAULTS', 'DRAM_RESIDENCY_AND_WORKING_SET']),
            dict(role='EFFECTS', status='SCOPED_NORMAL_SUCCESS',
                 blockers=['EXTERNAL_INPUT_MUTATION', 'FILE_MAPPING_IO_FAILURES',
                           'HIDDEN_PROVIDER_POINTER_RETENTION', 'WHOLE_ENTRY_EFFECTS']),
            dict(role='FORWARD_COST', status='OPEN', blockers=['COMPLETE_PROCESS_LAUNCH_TO_EXIT_COST'])],
        forward_full_entry_interval_ns=dict(lower=None, upper=None), forward_cost_proven=False,
        execution_default='NO_OP', whole_entry_equivalence_proven=False,
        transformation_authorized=False, deployment_authorized=False, target_executed=False)


def verify_cpu_fixed_grid_mapped_input_plan(reference):
    """Replay the materialization proof and the optional measured runtime decision."""
    saved = read_artifact(reference)
    _require(reference['schema'] == SCHEMA, 'unsupported fixed-grid mapped input plan')
    replay = derive_cpu_fixed_grid_mapped_input_plan(
        saved['proof'], observation=saved['observation'], schema_version=saved['schema_version'])
    _require(saved == replay, 'saved mapped input plan differs at cold replay')
    return replay
