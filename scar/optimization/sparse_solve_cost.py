"""Sparse solve cost."""

import json
import re
from pathlib import Path
from statistics import median
from scar.ir.execution_recipe import (
    artifact_reference,
    read_artifact,
    make_recipe,
    read_recipe,
    validate_recipe,
)
from scar.analysis.sparse_solve_scope import SCHEMA as SCOPE, check_sparse_solve_scope
from scar.compiler.model import LoopAxis
from scar.ir.implementation_plan import SCHEMA as PROBLEM, ImplementationChoice, problem_from_dict
from scar.ir.program import target_hardware as H
from scar.ir.record_codec import encode
from scar.optimization.hardware import _target_materials
from scar.optimization.schedule import (
    LogicalValue,
    Materialization,
    ResourceSchedule,
    Task,
    TaskKind,
)


# Project causal timing cuts from a measured sparse direct-solver study.
SCHEMA = 'scar.sparse-direct-solve-cost'
DESIGN = SCHEMA + '-design'
OBSERVATION = SCHEMA + '-observations'

_SECTIONS = ('source_solve', 'direct_csr', 'convert_csc', 'csc_factor', 'factored_solve')


def _duration(cuts):
    start, end = cuts
    if type(start) is not int or type(end) is not int or not 0 < start < end:
        raise ValueError('ordered positive perf_counter_ns cuts required')
    return end - start


def _summary(values):
    return dict(estimate_ns=int(median(values)), lower_ns=min(values), upper_ns=max(values),
                samples=len(values), bound_kind='observed_sample_envelope')


def project(design_path, observations_path):
    """Revalidate the receipts and summarize direct cuts, without extrapolating."""
    design_ref = artifact_reference(design_path, DESIGN)
    observations_ref = artifact_reference(observations_path, OBSERVATION)
    design = read_artifact(design_ref)
    observations = read_artifact(observations_ref)
    if (design['schema_version'] != 1 or observations['schema_version'] != 1
            or observations['design'] != design_ref or observations['context'] != design['context']):
        raise ValueError('solver study version or frozen context differs')
    scope = read_artifact(design['source_scope'])
    entry = scope['entry']
    count = entry['argv_repetitions']
    if (design['entry'] != entry['source'] or design['context']['cpu_ids'] != entry['affinity']
            or design['context']['argv'] != entry['argv']
            or any(design['context']['environment'][key] != value
                   for key, value in entry['thread_environment'].items())
            or scope['observed']['scope_entry_count'] != 1
            or scope['observed']['scope_exit_count'] != 1
            or scope['observed']['region_attempts'] != count
            or scope['observed']['consumer_completions'] != count):
        raise ValueError('solver samples differ from the admitted source scope')
    for reference in [design['entry'], design['harness'], design['fixture_parser'],
                      *design['provider_files'].values()]:
        read_artifact(reference, json_document=False)
    samples = observations['samples']
    indices = [sample['iteration'] for sample in samples]
    if (indices != design['indices'] or len(indices) < 3 or len(set(indices)) != len(indices)
            or any(type(index) is not int or not 0 <= index < count for index in indices)
            or observations['data_domains']['iterations'] != count):
        raise ValueError('sparse solver sample coverage differs from the design')
    durations = {name: [] for name in (*_SECTIONS, 'csc_reference_total')}
    previous_end = 0
    for sample in samples:
        cuts = sample['clock_cuts_ns']
        if set(cuts) != set(_SECTIONS):
            raise ValueError('sparse solver cut names differ')
        ordered = sorted(cuts.values())
        actual_order = sorted(('source_solve', 'direct_csr', 'csc_reference'),
                              key=lambda name: cuts['convert_csc' if name == 'csc_reference' else name][0])
        if (cuts['convert_csc'][1] != cuts['csc_factor'][0]
                or cuts['csc_factor'][1] != cuts['factored_solve'][0]
                or sample['call_order'] != actual_order
                or ordered[0][0] < previous_end
                or any(left[1] > right[0] for left, right in zip(ordered, ordered[1:]))):
            raise ValueError('sparse solver cuts differ from the ordered study boundaries')
        previous_end = ordered[-1][1]
        for name, section in cuts.items():
            durations[name].append(_duration(section))
        durations['csc_reference_total'].append(cuts['factored_solve'][1] - cuts['convert_csc'][0])
    return dict(schema=SCHEMA, schema_version=1, design=design_ref, observations=observations_ref,
                source_scope=design['source_scope'], context=design['context'],
                provider=observations['provider'], hardware=observations['hardware'],
                data_domains=observations['data_domains'],
                timings={name: _summary(values) for name, values in durations.items()},
                correctness=dict(direct_csr_all_bitwise=all(s['direct_csr_bitwise'] for s in samples),
                                 csc_reference_all_bitwise=all(s['csc_reference_bitwise']
                                                               for s in samples)),
                attribution=dict(source_solve='Python solver selection and default one-shot solve',
                                 direct_csr='separate public SciPy CSR spsolve call',
                                 csc_reference='separate CSC public conversion/factor/solve path',
                                 gssv_internal_ordering_factor_solve_ns=None,
                                 csc_factor_symbolic_and_numeric_split_ns=None),
                decision=dict(automatic_rewrite=False, full_entry_cost_ns=None,
                              reusable_numeric_factor_proven=False,
                              compatible_symbolic_reuse_abi_proven=False))


def check(path):
    """Cold-check a published projection from pinned, raw clock evidence."""
    path = Path(path)
    result = read_artifact(artifact_reference(path, SCHEMA))
    replayed = project(result['design']['uri'], result['observations']['uri'])
    if result != replayed:
        raise ValueError('sparse solver cost projection does not replay')
    return result


# Bind a scoped sparse solve to the existing guarded C++ symbolic provider.
PROFILE = 'cpu-sparse-symbolic-solve-v1'
FAMILY = 'CPP_SYNCHRONOUS_CPU_SYMBOLIC_SPARSE_SOLVE'
DECOMPOSITION = 'scar.sparse-direct-solve-decomposition'
HARDWARE = 'scar.target-hardware-graph'
BLAS = 'scar.provider-binary'
_PROVIDER_VERSION = '1.18.1'
_SCOPE = '$SPARSE_SOLVE_ACTIVATION'
_HOST = 'host-address-space:$CALL_PROCESS'
_ONCE = 'ONCE_PER_SCOPE'
_CALL = 'PER_CALL'
_FIRST = 'ON_FIRST_SUCCESS'
_EXIT = 'PER_SCOPE_EXIT'
_BUDGET_SCOPE = 'RETAINED_SYMBOLIC_PAYLOAD_ONLY'
_LANE = 'calling-cpu-thread'
_POOL = 'owned-symbolic-metadata'
_STAGE_SCOPE = 'ONE_SCOPED_ACTIVATION_WITH_EXACT_STRUCTURE_GUARD'
_SCHEMAS = dict(source_scope=SCOPE, decomposition=DECOMPOSITION, hardware=HARDWARE, problem=PROBLEM, blas=BLAS)
_SYMBOLS = dict(abi='scar_superlu_abi_v1', create='scar_superlu_symbolic_v1',
    apply='scar_superlu_symbolic_solve_v1', destroy='scar_superlu_symbolic_destroy_v1',
    retained_bytes='scar_superlu_symbolic_bytes_v1', reference='scar_superlu_reference_v1')


def _require(condition, detail):
    if not condition:
        raise ValueError(detail)


def decomposition_stages():
    """The existing provider separates retained structure from current numeric inputs."""
    return (
        dict(id='column-order', reads=['csr-structure', 'column-policy', 'transpose-mode'],
            produces=['raw-column-permutation'], lifetime=_STAGE_SCOPE, effect='OWNED_STRUCTURAL_METADATA'),
        dict(id='preorder', reads=['csr-structure', 'raw-column-permutation'],
            produces=['final-column-permutation', 'elimination-tree', 'column-slices'],
            lifetime=_STAGE_SCOPE, effect='OWNED_STRUCTURAL_METADATA'),
        dict(id='numeric-factorization', reads=['csr-values', 'current-column-slices', 'elimination-tree',
            'final-column-permutation', 'pivot-policy'], produces=['row-pivots', 'numeric-factors'],
            lifetime=_CALL, effect='PER_CALL_WORKSPACE'),
        dict(id='triangular-solve', reads=['right-hand-side', 'row-pivots', 'numeric-factors',
            'final-column-permutation', 'transpose-mode'], produces=['solution', 'status'],
            lifetime=_CALL, effect='RETAIN_ORIGINAL_STATUS_AND_WARNING_PATH'),
    )


def _check_blas(carrier, elf, blas):
    _require(carrier.get('schema') == 'scar.native-library-file' and elf['library'] == carrier
        and elf['documentation']['matched_tag'] == f'scipy-v{_PROVIDER_VERSION}',
        'original carrier does not carry the supported sparse provider version')
    image = Path(carrier['uri']).resolve()
    _require(str(image) == carrier['uri'] and image.name.startswith('_superlu.') and image.suffix == '.so'
        and image.parent.parts[-4:] == ('scipy','sparse','linalg','_dsolve'),
        'original carrier is not the admitted SciPy native extension')
    dynamic = read_artifact(elf['linkage'],json_document=False).decode('utf-8')
    needed = set(re.findall(r'\(NEEDED\)\s+Shared library: \[([^\]]+)\]',dynamic))
    provider = Path(blas['uri']).resolve()
    _require(str(provider) == blas['uri'] and provider.parent == image.parents[4] / 'scipy.libs'
        and provider.name.startswith('libscipy_openblas') and provider.name in needed,
        'sparse recipe BLAS does not match the original carrier dependency')
    read_artifact(blas,json_document=False)


def _evidence(artifacts):
    _require(set(artifacts) == set(_SCHEMAS) and all(artifacts[key].get('schema') == schema
        for key,schema in _SCHEMAS.items()), 'sparse recipe requires the original scope, decomposition, hardware, problem and BLAS')
    scope = check_sparse_solve_scope(artifacts['source_scope'])
    entry = scope['entry']
    source = read_artifact(scope['source_scope'])
    count = source['observed']['consumer_completions']
    _require(type(count) is int and count > 0 and source['entry'] == entry and source['observed']['normal_completion']
        and source['observed']['region_attempts'] == count,
        'sparse solve recipe requires the observed exact normal-completion call count')
    cpus = entry['affinity']
    _require(type(cpus) is list and len(cpus) == 1 and type(cpus[0]) is int and cpus[0] >= 0,
        'sparse symbolic solve is supported on one original CPU lane')
    expected = dict(schema=DECOMPOSITION,schema_version=1,source=artifacts['source_scope'],
        original_call=scope['source_call'],observed_graph=scope['source_graph'],call_count=count,
        cpu_ids=cpus,stages=list(decomposition_stages()),baseline_phase_costs_ns=None,
        value_changes_keep_factorization=True,candidate_uses_owned_structural_only=True,
        full_entry_timing_qualified=False)
    decomposition = read_artifact(artifacts['decomposition'])
    _require(decomposition == expected,'sparse decomposition does not replay the exact original case and four solver stages')
    build = read_artifact(read_artifact(scope['source_graph'])['program'])
    target = build['target']
    hardware = artifacts['hardware']
    _require(hardware == dict(uri=target['path'],sha256=target['sha256'],size_bytes=target['bytes'],schema=HARDWARE),
        'sparse recipe hardware differs from its original F4 target')
    _,snapshot,inventory,_ = _target_materials(Path(hardware['uri']).parent)
    _require(encode(snapshot) == hardware,'sparse recipe hardware snapshot replay differs')
    executors = [resource for resource in inventory if isinstance(resource,H.CpuExecutor) and cpus[0] in resource.cpu_ids]
    _require(len(executors) == 1,'original solver CPU does not identify one target executor')
    cpu_ref = encode(H.TargetResourceRef(snapshot,executors[0].id))
    carrier = scope['source_call']['carrier']['library']
    _check_blas(carrier,read_artifact(scope['elf_evidence']),artifacts['blas'])
    options = scope['provider']['options']
    _require(options == dict(requested_colperm=None,effective_ordering_under_provider_default='COLAMD',
        provider_default_fact_mode='DOFACT',index_abi='NP_INTC_32',flag=0), 'unsupported original sparse solver ABI')
    return scope,count,cpus,cpu_ref,carrier,options


def _planning(artifacts,count):
    problem = problem_from_dict(read_artifact(artifacts['problem']))
    references = (dict(artifact=artifacts['source_scope'],locator='#/source_call'),
        dict(artifact=artifacts['decomposition'],locator='#'))
    _require(encode(problem.target) == artifacts['hardware'] and problem.source_refs == references
        and len(problem.operations) == 1 and not problem.data_edges and not problem.reserved_memory,
        'sparse recipe problem separates neither the original scope nor the single solver choice')
    operation, = problem.operations
    _require(operation.id == 'sparse-direct-solve:'+artifacts['source_scope']['sha256']
        and operation.source_refs == references and not operation.predecessors and operation.baseline_option == 'KEEP',
        'one original scoped sparse solver operation required')
    resources = {record.id:record for record in problem.resources}
    _require(set(resources) == {_LANE,_POOL} and resources[_LANE].kind == 'EXECUTION'
        and resources[_LANE].capacity == 1 and resources[_POOL].kind == 'MEMORY'
        and all(record.source_refs == references for record in resources.values()),
        'sparse planner must use one CPU lane and one retained-symbolic payload budget')
    budget = resources[_POOL].capacity
    options = {option.id:option for option in operation.options}
    _require(set(options) == {'KEEP','REUSE_SYMBOLIC'} and all(option.lowering == option.id
        and option.demands == {_LANE:1} and option.proof_refs == references for option in options.values()),
        'sparse problem changed the original or native alternatives')
    original,native = options['KEEP'],options['REUSE_SYMBOLIC']
    _require(original.parameters == dict(call_count=count,includes_provider_setup_each_call=True)
        and not original.workspace_bytes and native.parameters == dict(call_count=count,format='CSR',
            numeric_factorization=_CALL,transpose_mode_preserved=True,full_entry_qualified_here=False)
        and set(native.workspace_bytes) == {_POOL} and 0 < native.workspace_bytes[_POOL] <= budget,
        'sparse problem changed loop count, live numeric work or symbolic budget')
    if problem.activities or problem.timing_constraints:
        when = (ImplementationChoice(operation.id,'REUSE_SYMBOLIC'),)
        _require({a.id for a in problem.activities} == {'load_once','close_once'}
            and all(a.when == when and a.demands == {_LANE:1} and a.duration_kind == 'FIXED' for a in problem.activities),
            'qualified sparse activity interval has another lifecycle or resource binding')
        expected = {('load_once','END',operation.id,'START'),(operation.id,'END','close_once','START')}
        _require(len(problem.timing_constraints) == 2 and
            {(c.before,c.before_endpoint,c.after,c.after_endpoint) for c in problem.timing_constraints} == expected
            and all(c.when == when and c.minimum_ns == 0 and c.maximum_ns is None for c in problem.timing_constraints),
            'sparse timed plan must retain activation before calls and release after the final call')
    return budget,operation.id


def _schedule(operation, cpu, budget):
    """Represent one original call template and one activation of its retained plan."""
    values, materials, tasks = [], [], []
    cpu_refs = (dict(kind='selected_cpu', target=cpu),)
    host_refs = (dict(kind='original_target_snapshot', artifact_role='hardware'),)

    def value(name, *, producer=None, allocation=None, bytes_expr=None, escape=False,
                lifetime=_CALL, aliases=None, **lineage):
        values.append(LogicalValue(name, producer=producer, escape=escape, call_scope=_SCOPE,
            attrs=dict(lifetime=lifetime,version_coordinate='i' if lifetime == _CALL else 'activation',
                storage_alias_scope=aliases,**lineage)))
        ownership = 'EXTERNAL' if producer is None else 'SCHEDULE'
        materials.append(Materialization(name, name, _HOST, host_refs, ready_task=producer,
            external_ready=producer is None, allocated_by=allocation, ownership=ownership, bytes_expr=bytes_expr))

    def task(name,kind,lifetime,entrypoint,*,reads=(),writes=(),needs=(),**attrs):
        tasks.append(Task(name,kind,tuple(reads),tuple(writes),tuple(needs),cpu_refs,
            dict(lifetime=lifetime,entrypoint=entrypoint,duration_ns=None,**attrs)))

    for name,extent in (('csr-structure','4 * (nnz + n + 1)'),('csr-values','8 * nnz'),
            ('right-hand-side','8 * n')):
        value(name,bytes_expr=extent)
    for name in ('column-policy','transpose-mode','pivot-policy'):
        value(name,lifetime=_ONCE,guarded_by='guard-structure')
    symbolic = ('raw-column-permutation','final-column-permutation','elimination-tree','column-slices','symbolic-key')
    for name in symbolic:
        producer = 'column-order' if name == symbolic[0] else 'preorder'
        alias = 'COLUMN_PERMUTATION_PRE_AND_POST_ORDER_SHARE_BACKING' if 'permutation' in name else None
        lineage = dict(last_value_consumer='preorder',invalidated_by='preorder') if name == symbolic[0] else {}
        value(name,producer=producer,allocation='allocate-symbolic',lifetime=_ONCE,aliases=alias,**lineage)
    per_call = {'current-column-slices':('bind-current-columns',None),
        'row-pivots':('numeric-factorization','4 * n'),'numeric-factors':('numeric-factorization',None),
        'factor-status':('numeric-factorization','4'),'solution':('triangular-solve','8 * n'),
        'status':('triangular-solve','4')}
    for name,(producer,extent) in per_call.items():
        value(name,producer=producer,allocation='allocate-numeric',bytes_expr=extent,escape=name in ('solution','status'))

    create,apply,destroy = (_SYMBOLS[key] for key in ('create','apply','destroy'))
    compute = TaskKind.COMPUTE
    task('admit-boundary',compute,_ONCE,'scar.runtime.sparse_solve_entry.admit',
        source_dispatch='LIVE_ORIGINAL_F4_CARRIER_REQUIRED')
    task('load-native',compute,_ONCE,'scar.runtime.sparse_lu_native.load_symbolic_solve',needs=('admit-boundary',),
        trigger='FIRST_ELIGIBLE_ORIGINAL_CALL')
    task('allocate-symbolic',TaskKind.ALLOCATION,_ONCE,create,needs=('load-native',),writes=symbolic,
        budget_bytes=budget,budget_scope=_BUDGET_SCOPE,establishes_value_ready=False)
    stages = {row['id']:row for row in decomposition_stages()}
    task('column-order',compute,_ONCE,create,reads=stages['column-order']['reads'],
        writes=stages['column-order']['produces'],needs=('allocate-symbolic',),native_phase='get_perm_c',
        source_ref=dict(artifact_role='decomposition',pointer='#/stages/0'))
    task('preorder',compute,_ONCE,create,reads=stages['preorder']['reads'],writes=symbolic[1:],
        needs=('column-order',),native_phase='sp_preorder',source_ref=dict(artifact_role='decomposition',pointer='#/stages/1'))
    task('guard-structure',compute,_CALL,apply,reads=('symbolic-key','csr-structure','csr-values',
        'right-hand-side','transpose-mode','column-policy'),needs=('preorder',),
        guard='EXACT_CURRENT_STRUCTURE_ARGUMENT_ABI_OPTIONS_AND_FINITE_INPUTS')
    task('allocate-numeric',TaskKind.ALLOCATION,_CALL,apply,needs=('guard-structure',),writes=tuple(per_call),
        establishes_value_ready=False,native_workspace_bound_bytes=None)
    task('bind-current-columns',compute,_CALL,apply,reads=('csr-values','column-slices','symbolic-key',
        'final-column-permutation','elimination-tree'),
        writes=('current-column-slices',),needs=('allocate-numeric',),
        preserves_retained_symbolic_storage=True,rebinding='CURRENT_VALUES_AND_PRIVATE_MUTABLE_COPIES')
    task('numeric-factorization',compute,_CALL,apply,reads=stages['numeric-factorization']['reads'],
        writes=(*stages['numeric-factorization']['produces'],'factor-status'),needs=('bind-current-columns',),
        native_phase='dgstrf',source_ref=dict(artifact_role='decomposition',pointer='#/stages/2'))
    task('triangular-solve',compute,_CALL,apply,reads=(*stages['triangular-solve']['reads'],'factor-status'),
        writes=stages['triangular-solve']['produces'],needs=('numeric-factorization',),
        native_phase='dgstrs',guard='FACTOR_STATUS_ZERO',native_orientation='TRANS',
        source_ref=dict(artifact_role='decomposition',pointer='#/stages/3'))
    task('release-numeric',TaskKind.RELEASE,_CALL,apply,
        reads=tuple(name for name in per_call if name not in ('solution','status')),needs=('triangular-solve',),
        retains_publication_outputs=True)
    task('validate-first-result',compute,_FIRST,'scar.runtime.sparse_lu_native._Solver._run',
        reads=('solution','status','csr-values','csr-structure','right-hand-side'),needs=('release-numeric',),
        requires_original_carrier_bitwise_or_fallback=True)
    task('publish-result',TaskKind.EXPORT,_CALL,'ORIGINAL_SOURCE_STATUS_BRANCH',reads=('solution','status'),
        needs=('release-numeric','validate-first-result'),guard='NATIVE_NORMAL_SUCCESS_AND_VALIDATED_ACTIVATION',
        on_failure='ORIGINAL_PYTHON_CALLSITE_AND_WARNINGS')
    task('release-symbolic',TaskKind.RELEASE,_EXIT,destroy,reads=symbolic[1:],needs=('publish-result',),
        after_final_normal_publication=True,shared_storage_released_once=True)
    return ResourceSchedule(operation,tuple(values),tuple(materials),tuple(tasks),loop_axes=(LoopAxis('i','call_count'),),
        call_scope=_SCOPE,executable=True,
        proof=dict(profile=PROFILE,normal_success_path_only=True,unrolled=False,
            activation_allocations_establish_readiness=False,per_call_numeric_storage_is_fresh=True,
            raw_and_final_column_permutation_are_two_versions_not_two_accounted_buffers=True,
            scratch_and_provider_phase_times_unqualified=True))


def _assemble(artifacts):
    scope,count,cpus,cpu,carrier,options = _evidence(artifacts)
    budget,operation = _planning(artifacts,count)
    call = scope['source_call']
    boundary = dict(id=operation,original_entry=scope['entry']['source'],original_call=call['call_point'],
        original_python_status_boundary=call['original_python_boundary'],original_code_key=call['code_key'],
        original_scope=scope['source_scope'],options=options,
        input=['current-csr-structure','current-numeric-values','current-vector-rhs'],output=['solution','status'],
        Q=dict(reuse='OWNED_STRUCTURE_ONLY',value_changes_require_numeric_factorization=True,
            preserve_csr_transpose=True,publish_only_on_normal_success=True,original_status_warning_and_output_shape=True))
    control = dict(activation='LAZY_FIRST_ELIGIBLE_ORIGINAL_CALL',iteration=dict(axis='i',count=count,
        unrolled=False,shared_body_instances=1,normal_source_completion_required=True,
        publication_order='ORIGINAL_CALL_ORDER'),first_success_bitwise_comparison=True,
        scope_exit='RELEASE_OWNED_SYMBOLIC_PLAN_AFTER_FINAL_NORMAL_PUBLICATION',
        fallback='ORIGINAL_PYTHON_AND_ORIGINAL_NATIVE_CARRIER')
    resources = dict(selected_cpu=dict(cpu_ids=cpus,executor=cpu,cpu_threads=1,planning_resource=_LANE),
        inventory=dict(snapshot_role='hardware',original_resource_inventory_retained=True),
        memory=dict(explicit_budget_bytes=budget,budget_scope=_BUDGET_SCOPE,planning_resource=_POOL,
            native_scratch_bound_bytes=None,required_retained_bytes=None,
            excludes=['EXTERNAL_INPUTS','NUMERIC_FACTORS_AND_PROVIDER_SCRATCH','OUTPUT','ALLOCATOR_OVERHEAD','PROCESS_RSS'],
            memory_policy='UNBOUND',physical_reservation=False,physical_numa_residency_proven=False))
    capability = dict(schema='scar.sparse-symbolic-backend-capability',schema_version=1,family=FAMILY,
        cxx_standard=20,abi_version=1,provider_version=_PROVIDER_VERSION,original_carrier=carrier,
        blas=artifacts['blas'],symbols=dict(_SYMBOLS))
    implementation = dict(family=FAMILY,backend_inputs={key:key for key in _SCHEMAS},
        backend_capability_record=capability,implementation_option='REUSE_SYMBOLIC',
        fallback_option='KEEP',status='EXPLICIT_STUDY_ONLY',program_body_copies=1)
    obligations = [
        dict(id='original_scope',status='PROVEN',detail='Original F4 callsite, retained hardware and exact normal case count are replayed.'),
        dict(id='current_structure',status='RUNTIME_GUARD',detail='Current CSR shape and structure bytes, carrier and options must match before numeric work.'),
        dict(id='numeric_and_status',status='RUNTIME_GUARD',detail='Require normal native success and first original bitwise validation before publishing. Preserve original statuses and warnings through fallback.'),
        dict(id='symbolic_payload',status='RUNTIME_GUARD',detail='Successful allocation and retained symbolic storage within the explicit budget are mandatory.'),
        dict(id='lowering',status='LOWERING_CHECK',detail='Use the original BLAS and existing C++20 ABI. Bind every shared phase to its original Python or native entrypoint.'),
        dict(id='native_workspace_and_time',status='UNRESOLVED',detail='Original sparse native scratch, total process memory, physical placement and individual phase times remain open.'),
    ]
    return make_recipe(PROFILE,semantic_boundary=boundary,control=control,task_graph=_schedule(operation,cpu,budget),
        resources=resources,implementation=implementation,artifacts=artifacts,obligations=obligations)


def make_sparse_solve_recipe(scope_ref,decomposition_ref,hardware_ref,problem_ref,*,blas_ref):
    """Reconstitute the planned sparse scope from exact original artifacts."""
    return _assemble(dict(source_scope=scope_ref,decomposition=decomposition_ref,hardware=hardware_ref,
        problem=problem_ref,blas=blas_ref))


def verify_sparse_solve_recipe(value):
    """Rebuild every phase, resource bound and ABI mapping before accepting lowering."""
    recipe = read_recipe(value) if isinstance(value,(str,Path)) else validate_recipe(value)
    _require(recipe['profile'] == PROFILE,'unsupported sparse solve recipe profile')
    _require(recipe == _assemble(recipe['artifacts']),'sparse solve recipe differs from source, planning or lifecycle replay')
    return recipe


__all__ = [
    'PROFILE',
    'decomposition_stages',
    'make_sparse_solve_recipe',
    'verify_sparse_solve_recipe',
    'SCHEMA',
    'DESIGN',
    'OBSERVATION',
    'project',
    'check',
    'FAMILY',
    'DECOMPOSITION',
    'HARDWARE',
    'BLAS',
]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.result), sort_keys=True))
