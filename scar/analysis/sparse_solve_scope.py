"""Join a bounded sparse solve to its original F4 call and installed provider."""

import ast
from collections import defaultdict, deque
import json
from pathlib import Path

from scar.analysis.bounded_entry_calls import (
    SCHEMA as ENTRY_CALLS, _admit_entry_calls, _derive_entry_source_witness_admitted,
)
from scar.analysis.build import BuildSession
from scar.analysis.csr_topology_scope import SOURCE_SCOPE
from scar.compiler.graph_frontend import _metadata, _source_identity
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.program.dependencies import DependencyKind
from scar.ir.record_codec import encode


SCHEMA = 'scar.sparse-solve-scope'
NATIVE_OBSERVATION = 'scar.sparse-solve-native-observation'
ELF_EVIDENCE = 'scar.sparse-solve-elf-evidence'


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _scope_key(scope):
    return scope['unit'], scope['ordinal']


def _entry_route(report, selected):
    """Use the scopes of observed ordinary frames, preserving guarded ABI routes."""
    records = defaultdict(set)
    for row in report['observed_targets']:
        item = row['observation']['evidence']
        if row['entry_scope_reachable']:
            records[row['provider'],row['site_id']].add((item['artifact']['sha256'],item['locator']))
    incoming = defaultdict(list)
    for index,row in enumerate(report['interfaces']):
        selection = row['frame']['selection']
        fact = selection['artifact']['sha256'],selection['locator']
        if row['ordinary_body_entry'] and fact in records[row['provider'],row['site_id']]:
            incoming[_scope_key(row['callee_scope'])].append((index,row))
    here = _scope_key(selected['caller_scope'])
    root = _scope_key(report['entry_scope'])
    pending = deque(((here, ()),))
    visited = {here}
    while pending:
        scope, chain = pending.popleft()
        if scope == root:
            return [f'#/interfaces/{index}' for index in reversed(chain)]
        for index,row in incoming[scope]:
            caller = _scope_key(row['caller_scope'])
            if caller not in visited:
                visited.add(caller)
                pending.append((caller, (*chain,index)))
    raise ValueError('sparse solve is outside the conditional observed entry call path')


def _span(node):
    return [node.lineno,node.col_offset,node.end_lineno,node.end_col_offset]


def _original_python_boundary(source, source_span):
    module = ast.parse(read_artifact(source,json_document=False),filename=source['uri'])
    # The native carrier returns two values. Its enclosing Python scope owns
    # all post-call status checks and return normalization.
    matches = [(body,index,statement) for node in ast.walk(module)
        if type(getattr(node,'body',None)) is list
        for body in (node.body,) for index,statement in enumerate(body)
        if type(statement) is ast.Assign and type(statement.value) is ast.Call and _span(statement.value) == source_span]
    _require(len(matches) == 1,'expected one original Python carrier assignment')
    block,index,assignment = matches[0]
    value = assignment.value
    target, = assignment.targets
    _require(type(target) is ast.Tuple and len(target.elts) == 2 and all(type(n) is ast.Name for n in target.elts),
        'native carrier does not return a result/status pair to the original Python scope')
    status,result_normalization = block[index+1:index+3]
    _require(type(status) is ast.If and type(result_normalization) is ast.If,
        'native carrier return does not reach the expected original status and normalization conditions')
    return dict(assignment_span=_span(assignment),arguments=[_span(arg) for arg in value.args],
        keyword_arguments=[dict(name=k.arg,span=_span(k.value)) for k in value.keywords],
        result_name=target.elts[0].id,status_name=target.elts[1].id,
        original_status_span=_span(status), original_result_normalization_span=_span(result_normalization),
        preserve_in_python=True)


def _project(session, admission, report, native, source):
    carrier = native['carrier']
    expected = native['site']
    matches = []
    for index,row in enumerate(report['observed_targets']):
        record = row['observation']
        if not row['entry_scope_reachable'] or record['caller_site'] != expected or record['target_kind'] != 'NATIVE_CARRIER':
            continue
        boundary = json.loads(record['metadata_json'])['native_boundary']
        if (boundary.get('module'),boundary.get('name')) == (carrier['module'],carrier['name']):
            matches.append((index,row))
    _require(len(matches) == 1,'no exact original bounded native carrier CALL in the entry graph')
    call_index,row = matches[0]
    provider = session._f4.providers[row['provider']]
    operation = provider.operations[row['site_id']]
    edges = [edge for edge in provider.outgoing(operation).edges
        if edge.kind is DependencyKind.CALL_TARGET and encode(provider.original(edge.target)) == row['observation']]
    _require(len(edges) == 1 and encode(operation) == row['operation'],'original native CALL_TARGET edge changed')
    witness = _derive_entry_source_witness_admitted(session,admission,path=source['uri'],sha256=source['sha256'],
        code_key=native['source']['key'])
    matches = [m for m in witness['matches'] if m['frame_selection'] == m['attempt_evidence']]
    _require(len(matches) == 1,'native CALL owner lacks its original ordinary entry frame')
    owner, = matches
    interface_index = int(owner['interface'].rsplit('/',1)[1])
    site = next(site for site in provider.plan['sites'] if site['id'] == row['site_id'])
    _,syntax,_,_,fingerprint = _source_identity(session._f4.catalog,row['unit'])
    occurrence = site['original_template']['occurrence']
    _,f2_span,_ = _metadata(syntax,occurrence)
    source_span = [f2_span[key] for key in ('line','column','end_line','end_column')]
    _require(fingerprint == source['sha256'] and syntax.nodes[occurrence].constructor == 'Call',
        'native CALL lost its exact F2 source occurrence')
    boundary = _original_python_boundary(source,source_span)
    _require(source_span == [expected['position'][i] for i in (0,2,1,3)],
        'F2 native call and observed bytecode span differ')
    return dict(source=source,code_key=native['source']['key'],source_span=source_span,
        bytecode_span=expected['position'],bytecode_offset=expected['offset'],
        bytecode_span_order=['line','end_line','column','end_column'],call_point=encode(operation),
        function_definition=owner['original_function_definition'],source_body=owner['original_body_point'],
        native_call_target_edge=encode(edges[0]), original_target_observation=f'#/observed_targets/{call_index}',
        original_entry_interface=owner['interface'],
        original_entry_route=[*_entry_route(report,report['interfaces'][interface_index]),owner['interface']],
        carrier=dict(name=carrier['name'],module=carrier['module'],library=carrier['source']),
        original_f2_occurrence=occurrence,original_python_boundary=boundary)


def _provider_contract(native, elf):
    library = native['carrier']['source']
    _require(native['carrier']['mapped'] and native['carrier']['exact_python_export'] and elf['library'] == library,
        'the native sample and installed ELF image do not match')
    read_artifact(library,json_document=False)
    for kind in ('disassembly','linkage'):
        read_artifact(elf[kind],json_document=False)
    segments = ('get_perm_c','sp_preorder','dgstrf','dgstrs')
    _require('dgssv' in {edge['target'] for edge in elf['static_direct_edges']['Py_gssv']} and
        set(segments) <= {edge['target'] for edge in elf['static_direct_edges']['dgssv']},
        'installed ELF does not support the original driver-to-phase decomposition')
    _require(all(elf['symbols'][name] and all(s['bind'] == 'LOCAL' for s in elf['symbols'][name])
        for name in ('dgssv',*segments)), 'provider phase symbol visibility differs')
    # These are conditional source/ELF relationships, never time measurements.
    phases = [
        dict(id='column-order',operation='get_perm_c',reads=['shape','indptr','indices','ordering_options'],
            writes=['initial_column_permutation'],cached_result_candidate=True),
        dict(id='symbolic-preorder',operation='sp_preorder',
            reads=['shape','indptr','indices','initial_column_permutation','ordering_options'],
            writes=['postordered_column_permutation','elimination_tree','permuted_structure'],cached_result_candidate=True),
        dict(id='numeric-factor',operation='dgstrf',
            reads=['matrix_data','elimination_tree','permuted_structure','postordered_column_permutation','pivot_options'],
            writes=['row_permutation','L','U','info'],cached_result_candidate=False),
        dict(id='triangular-solve',operation='dgstrs',
            reads=['rhs','row_permutation','postordered_column_permutation','L','U'],
            writes=['solution'],condition='NUMERICAL_FACTOR_INFO_ZERO',cached_result_candidate=False),
    ]
    for phase in phases:
        phase.update(resources=['HOST_CPU_CALLING_THREAD','HOST_PROVIDER_HEAP','HOST_DRAM'],
            timed_duration_ns=None,internal_allocation_sizes_known=False)
    return dict(phases=phases,orientation='CSR_INPUT_FACTOR_TRANSPOSE_VIEW_AND_SOLVE_TRANSPOSED',
        owned_state_candidate=['postordered_column_permutation','elimination_tree','permuted_structure'],
        refresh_each_call=['matrix_data_binding','rhs','numeric_factor','row_pivots','info','triangular_solve'],
        native_return=['ndarray_solution','integer_info'],
        options=dict(requested_colperm=None,effective_ordering_under_provider_default='COLAMD',
            provider_default_fact_mode='DOFACT', index_abi='NP_INTC_32', flag=0),
        python_effect_obligations=['ORIGINAL_SPARSE_NORMALIZATION_AND_DTYPE_PROMOTION',
            'INFO_ZERO_SUCCESS','POSITIVE_INFO_LE_N_MATRIX_RANK_WARNING_AND_FILL_NAN',
            'INFO_GT_N_MEMORY_ERROR','NEGATIVE_INFO_ORIGINAL_EXCEPTION','ORIGINAL_VECTOR_RAVEL'],
        implementation_boundary='ELF_LOCAL_SYMBOLS_WITHOUT_SUPPORTED_IN_PROCESS_SPLIT_ABI',
        provider_phases_observed_individually=False)


def _sampled_inputs(scope,native):
    entry = scope['entry']
    _require(native['entry']['source'] == entry['source'] and native['entry']['argv'] == entry['argv'],
        'solver samples came from another original entry/case')
    samples = native['native_samples']
    _require(len(samples) > 1 and len(samples) < native['observed_starts']
        and native['observed_starts'] == scope['observed']['consumer_completions'],
        'a small bounded sample and aggregate original consumer count are required')
    first = samples[0]['arrays']
    _require(all(row['structure_exactly_equals_first_sample'] and row['format'] == 'csr' and row['flag'] == 0
        and row['options'] == {'ColPerm':None} and row['use_umfpack'] is False and row['rhs_is_vector']
        and row['rhs_is_sparse'] is False and row['info'] == 0 and row['canonical_cache'] is True
        and row['sorted_cache'] is True and row['arrays']['data']['dtype'] == '<f8'
        and row['arrays']['indices']['dtype'] == row['arrays']['indptr']['dtype'] == '<i4'
        and row['arrays']['indices'] == first['indices'] and row['arrays']['indptr'] == first['indptr'] for row in samples),
        'samples do not support the selected canonical float64 CSR driver path')
    _require(all(row['matrix_data_changed_entries'] and row['changed_data_only_diagonal']
        and row['rhs_changed_entries'] and row['arrays']['data']['sha256'] != first['data']['sha256']
        and row['arrays']['rhs']['sha256'] != first['rhs']['sha256'] for row in samples[1:]),
        'samples do not establish the selected structural/numeric distinction')
    return dict(structure='EXACT_BYTES_EQUAL_ON_SELECTED_SAMPLES_ONLY',
        numeric_values='MATRIX_DIAGONAL_AND_RHS_CHANGED_ON_SELECTED_SAMPLES',
        first_structure=dict(indices=first['indices'],indptr=first['indptr']),
        sample_rows=[dict(iteration=row['index'],changed_diagonal_values=row['matrix_data_changed_entries'],
            changed_rhs_values=row['rhs_changed_entries'],info=row['info']) for row in samples],
        observation_cpu_ids=native['entry']['affinity'],observed_total_call_starts=native['observed_starts'],
        unobserved_iterations_invariant=False)


def derive_sparse_solve_scope(source_scope, source_graph, native_observation, elf_evidence):
    """Cold-bind both the Python carrier and the installed, conditional native phase graph."""
    refs = (source_scope,source_graph,native_observation,elf_evidence)
    _require(tuple(ref.get('schema') for ref in refs) == (SOURCE_SCOPE,ENTRY_CALLS,NATIVE_OBSERVATION,ELF_EVIDENCE),
        'expected the original source, bounded entry, native observation and ELF receipts')
    scope,report,native,elf = tuple(read_artifact(ref) for ref in refs)
    _require(all(doc['schema_version'] == 1 for doc in (scope,report,native,elf)), 'unsupported solver evidence version')
    _require(read_artifact(report['program'])['entry'] == scope['entry']['source']['uri'],
        'the original call graph names another entry')
    sampled = _sampled_inputs(scope,native)
    contract = _provider_contract(native,elf)
    source = artifact_reference(native['source']['filename'],'scar.python-source')
    with BuildSession(Path(report['program']['uri']).parent) as session, _admit_entry_calls(session,report) as admission:
        call = _project(session,admission,report,native,source)
    return dict(schema=SCHEMA,schema_version=1,source_scope=source_scope,source_graph=source_graph,
        native_observation=native_observation,elf_evidence=elf_evidence,entry=scope['entry'],source_call=call,
        sampled_values=sampled,provider=contract,
        source_provenance=dict(original_loop=scope['f4_loop']['operation'],original_row_edit=scope['f4_call']['operation'],
            original_loop_normal_call_results=scope['f4_loop']['normal_body_calls'],
            native_sampled_call_is_not_scope_wide_effect_proof=True),
        guard_contract=dict(symbolic_key=['native_provider_and_blas_profile','csr_flag_and_transpose',
            'shape_and_nnz','exact_indices_and_indptr_dtype_and_bytes','colperm_and_symbolic_options'],
            on_miss='ORIGINAL_PYTHON_AND_NATIVE_CALL_BEFORE_REPLACEMENT_WRITES',
            retain_original_python_status_and_warnings=True,
            ensure_original_pivoting_and_float_semantics_per_call=True),
        frontiers=['ORIGINAL_PROVIDER_EXACT_SPLIT_ABI_UNAVAILABLE_THROUGH_PUBLIC_MODULE',
            'SAMPLE_EQUALITY_DOES_NOT_PROVE_ALL_ITERATION_STRUCTURE',
            'ROW_PIVOT_PERMUTATION_AND_FACTORS_DEPEND_ON_CURRENT_MATRIX_VALUES',
            'ALTERNATE_PROVIDER_CSR_TRANSPOSE_INFO_WARNINGS_AND_BITWISE_EQUIVALENCE_UNPROVED',
            'HOST_RESOURCE_TIMINGS_AND_PEAK_OWNED_MEMORY_REQUIRE_MEASUREMENT'],
        qualification=dict(exact_original_f4_call_replayed=True,native_effects_closed=False,
            performance_proven=False,transformation_authorized=False))


def check_sparse_solve_scope(reference):
    """Independently replay the graph and compare the complete scope receipt."""
    doc = read_artifact(reference) if 'uri' in reference else reference
    _require((doc.get('schema'),doc.get('schema_version')) == (SCHEMA,1),'expected a sparse solve scope')
    expected = derive_sparse_solve_scope(*(doc[k] for k in ('source_scope','source_graph','native_observation','elf_evidence')))
    canonical = lambda value: json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
    _require(canonical(doc) == canonical(expected),'sparse solve scope differs from cold replay')
    return expected
