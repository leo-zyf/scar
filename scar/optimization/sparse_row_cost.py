"""Account for a guarded CPU CSR row rewrite in its measured local scope."""

from hashlib import sha256
import json
from pathlib import Path
from statistics import median, variance

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.implementation_costs import CostQuery, query_key

SCHEMA = 'scar.sparse-csr-row-cost'
FAMILY = 'CSR_ZERO_ROWS_SET_DIAGONAL'
SCOPE = 'COPY_CSR_AND_GUARDED_ROW_UPDATE_IN_PROCESS'


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _statistics(samples):
    _need(type(samples) is list and len(samples) >= 3 and
        all(type(value) is int and value > 0 for value in samples), 'positive monotone duration samples required')
    return dict(estimate_ns=round(median(samples)), lower_ns=min(samples), upper_ns=max(samples),
        sample_variance_ns2=variance(samples), samples_ns=samples,
        bounds_are_sample_extrema=True)


def _warm_context(adapter_source, cold_entry, native, observation):
    adapter = Path(__file__).resolve().parents[1]/'runtime/sparse_csr.py'
    _need(adapter_source == artifact_reference(adapter, 'scar.python-source'),
        'measured CSR adapter differs from the current Python guard bytes')
    witness = json.loads(read_artifact(cold_entry, json_document=False))
    _need(cold_entry['schema'] == 'scar.csr-native-cold-entry-witness', 'cold entry witness identity required')
    if witness.get('schema_version') == 1:
        [call] = witness['witnesses']
        _need(witness['manifest'] == native and witness['original_driver_reached_solve'] is True
            and call['memo_hit_before_any_baseline_enforce'] is False
            and call['memo_present_after_reference'] is True, 'cold device or original provider state differs')
        read_artifact(witness['source'], json_document=False)
        entry, native_admitted, shape = witness['source']['uri'], call['native_admitted'], call['shape']
    elif 'cold_probe' in witness:
        probe = witness['cold_probe']
        [native_call] = probe['native_calls']
        _need(witness['native'] == native and witness['adapter'] == adapter_source
            and probe['provider_memo_identical'] is True and probe['csr_bits_flags_aliases'] is True
            and witness['zero_order'] == dict(rejected=True, provider_memo_unchanged=True)
            and witness['solution']['finite'] is True, 'final cold witness differs from adapter/candidate correctness')
        read_artifact(witness['entry'], json_document=False)
        entry, native_admitted = witness['entry']['uri'], True
        shape = [native_call['rows'], native_call['rows']]
        call = dict(nnz=native_call['nnz'], selected_rows=native_call['constrained'])
    else:
        [call] = witness['records']
        _need(witness['artifact'] == native['uri'] and witness['full_native_example_solve_finite'] is True
            and call['cold_provider_memo_preserved'] is True and call['equal_csr_bits_flags_and_object_identities'] is True,
            'cold candidate must preserve the original provider memo, sparse output, and solve result profile')
        entry, native_admitted, shape = witness['entry'], call['native'], call['matrix_shape']
    _need(entry == observation['source']['path']
        and sha256(Path(entry).read_bytes()).hexdigest() == observation['source']['sha256']
        and type(native_admitted) is bool and shape == observation['matrix']['shape']
        and call['nnz'] == observation['matrix']['nnz'] and call['selected_rows'] == observation['selected_rows']['count'],
        'cold entry witness must establish the exact source provider state and observed path')
    return dict(adapter_source=adapter_source, cold_entry=cold_entry,
        cold_entry_native_admitted=native_admitted,
        original_matrix_passed_to_guard=True, provider_state='MEMO_WARMED_BY_UNMODIFIED_SOURCE')


def derive_sparse_row_cost(observation, *, source_bytes, samples, setup, warmup, native_artifact,
                           adapter_source=None, cold_entry=None):
    matrix = observation['matrix']
    selection = observation['selected_rows']
    diagonal = observation['full_diagonal']
    arrays = matrix['arrays']
    rows, columns = matrix['shape']
    nnz = matrix['nnz']
    _need(matrix['format'] == 'csr' and rows == columns == diagonal['unique']
        and diagonal['missing'] == diagonal['duplicates'] == 0
        and selection['count'] == selection['distinct']
        and arrays['data']['dtype'] == '<f8' and arrays['indices']['dtype'] == arrays['indptr']['dtype'] == '<i4',
        'typed CSR rows and a complete unique diagonal required')
    _need(sha256(source_bytes).hexdigest() == observation['enforce_source']['sha256'],
        'source byte identity differs from observation')
    copied = sum(value['nbytes'] for value in arrays.values())
    rhs = observation['out_of_place']['rhs_after_copy']['nbytes']
    needed = observation['out_of_place']
    _need(needed['all_matrix_arrays_independent'] and needed['indices_same'] and needed['indptr_same']
        and needed['original_data_preserved'] and needed['copied_matrix_payload_bytes'] == copied,
        'original out-of-place matrix representation must remain exact')
    data_written = (selection['nnz']+selection['count'])*8
    context = dict(source_sha256=observation['enforce_source']['sha256'], family=FAMILY,
        matrix_shape=matrix['shape'], csr_nnz=nnz, selected_rows=selection['count'],
        selected_nnz=selection['nnz'], original_arguments=dict(diag=1.0, overwrite=False),
        input_layout={key:dict(dtype=value['dtype'], shape=value['shape'], nbytes=value['nbytes'])
            for key, value in arrays.items()}, boundary_indices_dtype=selection['array']['dtype'],
        input_layout_and_workload_hashes={key:value['sha256'] for key, value in arrays.items()}
            | {'boundary_indices':selection['array']['sha256']},
        cpu_ids=observation['environment']['cpu_ids'], native_threads=1,
        numpy=observation['environment']['numpy'], scipy=observation['environment']['scipy'],
        host_array_domain=observation['host_array_domain'])
    statistics = {key:_statistics(samples[key]) for key in ('original', 'native')}
    statistics.update(body_only={key:_statistics(samples['body_only'][key]) for key in ('original', 'native')},
        copy_only=_statistics(samples['copy_only']))
    original, native = (statistics[key]['estimate_ns'] for key in ('original','native'))
    improvement = original-native
    setup_fields = ('compile_ns', 'load_ns')
    setup_values = [setup[key] for key in setup_fields]
    _need(type(warmup) is int and warmup > 0 and
        all(value is None or type(value) is int and value >= 0 for value in setup_values), 'invalid setup or warmup')
    complete_setup = all(value is not None for value in setup_values)
    break_even = (sum(setup_values)//improvement+1) if complete_setup and improvement > 0 else None
    qualification = dict(
        status='EMPIRICAL_MATCHED_LOCAL_SCOPE',
        region_estimate_available=True, empirical_ranges_are_guaranteed=False,
        guarded_local_gain=improvement/original, guarded_local_speedup=original/native,
        local_faster_by_point_estimate=improvement > 0,
        local_sample_ranges_separated=statistics['native']['upper_ns'] < statistics['original']['lower_ns'],
        cold_one_use_profit=(sum(setup_values) < improvement) if complete_setup and improvement > 0 else False,
        break_even_calls_positive_wall_ns=break_even,
        observed_complete_entry_cost_ns=None, whole_program_selection='KEEP',
        whole_program_frontier=['IMPORT_AND_ASSEMBLY', 'SOURCE_ENFORCE_DISPATCH_AND_RHS',
            'SOLVER_AND_RETAINED_OUTPUT_EFFECTS', 'COMPLETE_PROCESS_BASELINE_AND_CANDIDATE'],
        scope_includes=['MATRIX_COPY_BEFORE_BOTH_ALTERNATIVES',
            'ORIGINAL_NUMPY_SCIPY_OR_NATIVE_ADAPTER_WITH_FULL_STRUCTURE_GUARDS'],
        named_measurements_are_disjoint_only_within_one_sample=False)
    _need((adapter_source is None) == (cold_entry is None), 'adapter and cold entry evidence must be supplied together')
    versions = {}
    if adapter_source is not None:
        versions = _warm_context(adapter_source, cold_entry, native_artifact, observation)
        context.update(versions)
        qualification.update(status='EMPIRICAL_MATCHED_WARM_SCOPE',
            exact_guard_bytes_bound=True, cold_entry_native_calls=int(versions['cold_entry_native_admitted']),
            cold_branch_cost_measured=False, warm_only=True,
            load_only_break_even_warm_calls=(setup['load_ns']//improvement+1)
                if improvement > 0 and setup['load_ns'] is not None else None)
    return dict(schema=SCHEMA, schema_version=2 if versions else 1, family=FAMILY, source_sha256=context['source_sha256'],
        observations=context, original_observation=observation,
        native_artifact=native_artifact, measurement_scope=SCOPE, warmup=warmup, timings=statistics,
        setup={**setup, 'setup_is_one_time':True}, qualification=qualification,
        resource_demands=dict(cpu_execution_threads=1, host_serial_data=True,
            source_matrix_payload_bytes=copied, independent_output_matrix_payload_bytes=copied,
            surviving_rhs_output_payload_bytes=rhs,
            copy_logical_read_bytes=copied, copy_logical_write_bytes=copied,
            structurally_validated_rows=rows, csr_columns_examined_bound=nnz,
            selected_diagonal_writes=selection['count'], selected_row_entries_zeroed=selection['nnz'],
            candidate_value_stores_upper_bytes=data_written,
            candidate_data_off_boundary_can_remain_unread=True,
            actual_dram_traffic_bytes=None, native_process_rss=None, placement_numa_residency=None),
        planning_query_key={mode:query_key(CostQuery(SCOPE, {**context, 'mode':mode}))
            for mode in ('original','native')}, **versions)


def check_sparse_row_cost(report):
    """Check byte references and recompute the cost report from retained clocks."""
    reference = report if type(report) is dict else artifact_reference(Path(report).resolve(), SCHEMA)
    saved = read_artifact(reference)
    _need(reference['schema'] == SCHEMA and saved['schema_version'] in (1, 2)
        and saved['paired_abi_returns_native_true'] is True and saved['bitwise_output_equal'] is True,
        'supported CSR measurement receipt with exact output validation required')
    read_artifact(saved['exact_measurement_harness'], json_document=False)
    if saved['schema_version'] == 2:
        read_artifact(saved['helper_measurement_harness'], json_document=False)
    native = read_artifact(saved['native_artifact'])
    for key in ('source', 'library', 'template'):
        read_artifact(native[key], json_document=False)
    observation = saved['original_observation']
    source = Path(observation['enforce_source']['path']).read_bytes()
    span = saved['source_span']
    lines = source.splitlines(keepends=True)[span['first_line']-1:span['last_line']]
    _need(sha256(b''.join(lines)).hexdigest() == span['exact_executed_body_sha256'],
        'timed original CSR block differs from the retained source span')
    timings = saved['timings']
    before, after = 'original','native'
    samples = {mode:timings[mode]['samples_ns'] for mode in (before, after)}
    samples.update(body_only={mode:timings['body_only'][mode]['samples_ns'] for mode in (before, after)},
        copy_only=timings['copy_only']['samples_ns'])
    _need(saved['warmup'] == saved['warmup_pairs']
        and all(len(timings[mode]['samples_ns']) == saved['repetitions_per_mode']
            for mode in (before, after)), 'CSR sample count differs from the published experiment')
    replayed = derive_sparse_row_cost(observation, source_bytes=source, samples=samples,
        setup=saved['setup'], warmup=saved['warmup'], native_artifact=saved['native_artifact'],
        adapter_source=saved.get('adapter_source'), cold_entry=saved.get('cold_entry'))
    _need(all(saved[key] == value for key, value in replayed.items()),
        'CSR region cost differs from retained measurements')
    return saved
