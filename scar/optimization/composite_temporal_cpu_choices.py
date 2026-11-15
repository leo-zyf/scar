"""Original-F4 composite loops using shared body/resource scheduling mechanics.

The composite proof is never represented as an original leaf-call proof.  Its
derived DAG and each original function-global route retain separate authority.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from scar.compiler.pipeline_loop_frontend import validate_composite_invariant_loop_proof
from scar.ir.execution_recipe import read_artifact
from scar.optimization.resource_solver import semantic_sha256
from scar.optimization import temporal_cpu_choices as T


def composite_temporal_guard_profile(proof):
    """Erase source names, retaining only the shape of hot runtime checks.

    Source byte lengths are workload geometry, not source contents. Numeric
    function/global slots use a declared source-unit namespace partition. This
    preserves route geometry within that canonical model, not actual runtime
    globals-dict identity or aliasing of separately instantiated functions.
    """
    sources=[];functions=[];function_index={};source_index={};edges={};slots={}
    def function(source):
        key=(source.path,source.code_key)
        if source.path not in source_index:
            source_index[source.path]=len(sources);sources.append(Path(source.path).stat().st_size)
        if key not in function_index:
            function_index[key]=len(functions)
            functions.append(dict(source_index=source_index[source.path],namespace=source_index[source.path]))
        return function_index[key]
    original=function(proof.source);root=function(proof.root_callee_source)
    for guard in proof.nested_callee_guards:function(guard.owner_source);function(guard.target_source)
    by_path={tuple(guard.global_path):guard for guard in proof.nested_callee_guards}
    def slot(owner,name,target):
        key=(functions[owner]['namespace'],name)
        if key not in slots:slots[key]=len(slots)
        index=slots[key];prior=edges.setdefault(index,dict(slot=index,namespace=key[0],target=target))
        if prior['target']!=target:raise ValueError('inconsistent original namespace target in composite route')
        return index
    root_slot=slot(original,proof.root_global_name,root);routes=[]
    for guard in proof.nested_callee_guards:
        path=[];owner=original
        for length,name in enumerate(guard.global_path,1):
            prefix=by_path.get(tuple(guard.global_path[:length]))
            if prefix is None:raise ValueError('missing original prefix route')
            actual_owner=function(prefix.owner_source);target=function(prefix.target_source)
            if owner!=actual_owner:raise ValueError('original composite owner differs from preceding target')
            path.append(slot(owner,name,target));owner=target
        routes.append(dict(path_slots=path,owner=function(guard.owner_source),
            target=function(guard.target_source),lookup_slot=path[-1]))
    return dict(schema='scar.composite-temporal-guard-profile',schema_version=1,
        source_byte_lengths=sources,functions=functions,global_slots=list(edges.values()),
        original_function=original,root_function=root,root_slot=root_slot,routes=routes,
        namespace_partition_basis='DEFINING_SOURCE_UNIT_DECLARED_CANONICAL_MODEL',
        code_key_cache='WARM_IDENTITY_CACHE',source_reads_per_call=len(sources),
        source_contents_or_names_in_profile=False,
        source_layout_is_target_program=False)


def build_composite_temporal_cpu_domain(proof,target,*,catalog,input_shapes,trip_count,
        memory_budget_bytes,max_threads=None):
    proof=validate_composite_invariant_loop_proof(proof,catalog=catalog)
    domain=T._build_temporal_body_cpu_domain(proof.body,proof.carry_parameter,target,
        proof_id=proof.id,proof_sha256=semantic_sha256(asdict(proof)),input_shapes=input_shapes,
        trip_count=trip_count,memory_budget_bytes=memory_budget_bytes,max_threads=max_threads)
    domain.update(schema='scar.composite-temporal-cpu-domain',proof_kind='COMPOSITE',
        original_proof_profile=proof.profile,body_identity=proof.body_identity,
        original_source=asdict(proof.source),root_callee_source=asdict(proof.root_callee_source),
        root_global_name=proof.root_global_name,function_global_guards=[asdict(g) for g in proof.nested_callee_guards],
        guard_profile=composite_temporal_guard_profile(proof),
        composite_f4_and_fusion_replay_required=True)
    domain['runtime_guards'] += ['ORIGINAL_COMPOSITE_FUNCTION_GLOBAL_ROUTES_CODE_DEFAULTS_AND_SOURCE_BYTES_MATCH']
    domain.pop('id');domain['id']='composite-temporal-cpu-domain:'+semantic_sha256(domain)
    return domain


def verify_composite_temporal_cpu_domain(proof,target,domain,*,catalog):
    expected=build_composite_temporal_cpu_domain(proof,target,catalog=catalog,input_shapes=domain['input_shapes'],
        trip_count=domain['trip_count'],memory_budget_bytes=domain['memory_budget_bytes'],max_threads=domain['max_threads'])
    if semantic_sha256(expected)!=semantic_sha256(domain):raise ValueError('composite domain differs from original F4/fusion/resource replay')
    return expected


def _composite_plan(plan,domain,*,selected=False):
    plan=deepcopy(plan)
    plan.update(schema=('scar.cost-selected-composite-temporal-cpu-resource-plan' if selected else
        'scar.parallel-composite-temporal-resource-plan'),proof_kind='COMPOSITE',
        original_proof_profile=domain['original_proof_profile'],body_identity=domain['body_identity'],
        original_source=domain['original_source'],root_callee_source=domain['root_callee_source'],
        root_global_name=domain['root_global_name'],function_global_guards=domain['function_global_guards'],
        guard_profile=domain['guard_profile'],composite_f4_and_fusion_replay_required=True)
    plan.pop('id');plan['id']=('cost-selected-composite-temporal-resource-plan:' if selected else
        'parallel-composite-temporal-resource-plan:')+semantic_sha256(plan)
    return plan


def preview_composite_temporal_cpu_choice(domain,*,cpu_threads):
    return _composite_plan(T.preview_temporal_cpu_choice(domain,cpu_threads=cpu_threads),domain)


def verify_composite_temporal_cpu_preview(proof,target,plan,*,catalog):
    domain=build_composite_temporal_cpu_domain(proof,target,catalog=catalog,input_shapes=plan['input_shapes'],
        trip_count=plan['trip_count'],memory_budget_bytes=plan['memory_budget_bytes'],max_threads=plan['domain_max_threads'])
    expected=preview_composite_temporal_cpu_choice(domain,cpu_threads=plan['cpu_threads'])
    if semantic_sha256(expected)!=semantic_sha256(plan):raise ValueError('composite preview differs from original F4/fusion/resource replay')
    return expected


def plan_selected_composite_temporal_cpu_choice(domain,*,resource_facts=(),no_op_profile=None):
    result=T.plan_selected_temporal_cpu_choice(domain,resource_facts=resource_facts,no_op_profile=no_op_profile)
    if result['schedule'] is not None:result['schedule']=_composite_plan(result['schedule'],domain,selected=True)
    return result


def verify_selected_composite_temporal_cpu_choice(proof,target,plan,*,catalog):
    if (plan.get('schema'),plan.get('schema_version'))!=('scar.cost-selected-composite-temporal-cpu-resource-plan',1):
        raise ValueError('versioned composite cost-selected plan required')
    refs=plan['resource_facts'];original=plan['no_op_profile']
    for ref in (*refs,original):read_artifact(ref)
    domain=build_composite_temporal_cpu_domain(proof,target,catalog=catalog,input_shapes=plan['input_shapes'],
        trip_count=plan['trip_count'],memory_budget_bytes=plan['memory_budget_bytes'],max_threads=plan['domain_max_threads'])
    expected=plan_selected_composite_temporal_cpu_choice(domain,resource_facts=[ref['uri'] for ref in refs],
        no_op_profile=original['uri'])['schedule']
    if expected is None or semantic_sha256(expected)!=semantic_sha256(plan):
        raise ValueError('composite selection differs from original F4/fusion/full-cost replay')
    return expected
