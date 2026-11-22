"""Sourced finite CPU-team decisions for the existing temporal C++ backend.

This selects a bound team/static-row implementation, not a new program graph.
Canonical recurrence fits remain estimates. Processes/MPI, physical placement,
transfers and deployment selection are outside this implementation family.
"""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path

from scar.codegen.loop import _admit, loop_implementation_sha256
from scar.compiler.loop_admission import loop_digest
from scar.compiler.model import document
from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.record_codec import encode, loads
from .cpu_choices import _emit, _read, _receipt, _source
from .deployment_cost import _fact, _number, _profile
from .demand import analyze_region_demand
from .hardware import bind_cpu_placement
from .implementation_solver import solve_implementation_choices, task_implementation_contract
from .loop_layout import verify_loop_layout
from .loop_strategy import loop_execution_plan, verify_loop_strategy
from .resource_solver import semantic_sha256 as digest
from .schedule import ResourceSchedule, Task, TaskKind

_OPS = ('add', 'sub', 'mul', 'div', 'neg')
_UNITS = dict(rows='ticks/row', packed_tiles='ticks/tile', packed_iteration_groups='ticks/iteration_group',
    scalar_elements='ticks/element', scalar_iterations='ticks/scalar_iteration',
    packed_finite_checks='ticks/vector_check', scalar_finite_checks='ticks/scalar_check',
    logical_input_bytes='ticks/logical_byte', logical_output_bytes='ticks/logical_byte',
    packed_dependency_depth_iterations='ticks/depth_iteration_group',
    scalar_dependency_depth_iterations='ticks/depth_scalar_iteration',
    **{'packed_arithmetic.'+op:'ticks/vector_operation' for op in _OPS},
    **{'scalar_arithmetic.'+op:'ticks/scalar_operation' for op in _OPS})
BOUNDARY = 'LOOP_ARTIFACT_RUN_STEADY_SESSION_FRESH_OUTPUT'
CALIBRATION_SCOPE = 'INDEPENDENT_CANONICAL_TEMPORAL_RECURRENCES'
_INCLUDED = ('real_iteration_loop','packed_and_scalar_finite_guards','affinity_and_fenv',
             'team_launch_and_join','fresh_output_allocation')
_ERRORS = (OSError,ValueError,TypeError,KeyError,AttributeError,IndexError,OverflowError)


def loop_worker_feature_units():
    """Coefficient units; vector_operation means one two-f64 operation, not G."""
    return deepcopy(_UNITS)


def features_for_loop(loop, strategy, plan, layout):
    """Exact lowering counts; logical bytes occur once and are not DRAM bytes."""
    _,plan = _admit(loop,strategy,plan); layout = verify_loop_layout(loop,strategy,plan,layout)
    if layout['status']!='PLANNED_UNDER_MODEL' or layout['selected'] is None:
        raise ValueError('admitted SIMD layout required')
    rows,cols = strategy['state_shape']; steps = strategy['trip_count']
    chains = layout['selected']['vector_chains']; lanes = layout['selected']['simd_lanes']
    arithmetic = analyze_region_demand(plan)['per_output_element']['arithmetic']
    tiles = rows*(cols//(lanes*chains)); scalar = rows*(cols%(lanes*chains))
    groups = tiles*steps; scalar_iterations = scalar*steps
    depths = {}
    for node in plan.region.nodes:
        depths[node.id] = 0 if node.op in ('input','const') else 1+max(depths[key] for key in node.inputs)
    depth = depths[plan.region.output]
    values = dict(rows=rows,packed_tiles=tiles,packed_iteration_groups=groups,
        scalar_elements=scalar,scalar_iterations=scalar_iterations,
        packed_finite_checks=tiles*chains*(steps+1),scalar_finite_checks=scalar*(steps+1),
        logical_input_bytes=8*rows*cols,logical_output_bytes=8*rows*cols,
        packed_dependency_depth_iterations=depth*groups,scalar_dependency_depth_iterations=depth*scalar_iterations,
        **{'packed_arithmetic.'+op:groups*chains*arithmetic[op] for op in _OPS},
        **{'scalar_arithmetic.'+op:scalar_iterations*arithmetic[op] for op in _OPS})
    return dict(values=values,coefficient_units=loop_worker_feature_units(),rows=rows,columns=cols,steps=steps,
        arithmetic_per_element=arithmetic,body_dependency_depth=depth,simd_lanes=lanes,vector_chains=chains,
        scalar_tail_per_row=cols%(lanes*chains),logical_bytes_are_dram_bytes=False,worker_scaling_inferred=False)


def _row_span(rows,target):
    """Exact identifiability check; coefficients are never fitted here."""
    basis = {}
    for row in rows:
        residual = [Fraction(str(v)) for v in row]
        for pivot,vector in sorted(basis.items()):
            factor=residual[pivot]; residual=[a-factor*b for a,b in zip(residual,vector)]
        pivot=next((i for i,value in enumerate(residual) if value),None)
        if pivot is not None:
            scale=residual[pivot]; basis[pivot]=[value/scale for value in residual]
    residual=[Fraction(str(v)) for v in target]
    for pivot,vector in sorted(basis.items()):
        factor=residual[pivot]; residual=[a-factor*b for a,b in zip(residual,vector)]
    return len(basis),not any(residual)


def predict_loop_worker_costs(loop,strategy,plan,layout,profile):
    """Replay one exact-CPU effective fit, including its calibration row span."""
    features=features_for_loop(loop,strategy,plan,layout)
    profile=deepcopy(profile) if isinstance(profile,dict) else {}
    issues=[]; cache={}; sources=set()
    def issue(field,reason): issues.append(dict(field=field,reason=reason))
    def evidence(records,field,expected=None,parameter=None):
        if not isinstance(records,list) or not records:
            issue(field,'retained matching FactReferences required'); return False
        valid=True
        for index,reference in enumerate(records):
            try:
                key=digest(reference)
                if key not in cache: cache[key]=_fact(reference)
                value,artifact=cache[key]
                if expected is not None and digest(value)!=digest(expected): raise ValueError('original contract differs')
                if parameter is not None:
                    number=value.get('value') if isinstance(value,dict) else value
                    if _number(number)!=_number(parameter['value']): raise ValueError('original parameter value differs')
                    if isinstance(value,dict) and value.get('unit')!=parameter['unit']: raise ValueError('original parameter unit differs')
                if field=='context_evidence': sources.add(artifact)
                elif artifact not in sources: raise ValueError('source lacks the original matching calibration context')
            except _ERRORS as error:
                valid=False; issue(f'{field}.{index}',str(error))
        return valid
    def parameter(record,field,unit,positive=False):
        if not isinstance(record,dict) or set(record)!={'value','unit','evidence'}:
            issue(field,'explicit sourced parameter required'); return None
        value=_number(record['value'])
        if value is None or (positive and not value) or record['unit']!=unit:
            issue(field,'finite value with correct units required'); return None
        return value if evidence(record['evidence'],field+'.evidence',parameter=record) else None
    if profile.get('schema')!='scar.loop-worker-cost-profile' or type(profile.get('schema_version')) is not int or profile.get('schema_version')!=1:
        issue('schema','new temporal worker profile required; single-step costs are inapplicable')
    allowed={'schema','schema_version','context','context_evidence','applicability','applicability_evidence',
             'calibration','calibration_evidence','model'}
    if set(profile)-allowed: issue('profile','unsupported constraints cannot be silently ignored')
    binding=plan.resource_binding
    primitive=(loads(Path(layout['hardware_profile_receipt']['path']).read_bytes())
               if layout['hardware_profile_receipt'] is not None else layout['hardware_profile'])
    expected=dict(target_path=binding.target_path,target_sha256=binding.target_sha256,executor_id=binding.executor_id,
        cpu_ids=list(binding.cpu_ids),loop_implementation_sha256=loop_implementation_sha256(),
        compiler=primitive['context']['compiler'],layout={key:layout['selected'][key] for key in ('isa','simd_lanes','vector_chains')},
        input_layout='C_CONTIGUOUS_FLOAT64_2D',execution_boundary=BOUNDARY,memory_policy='UNBOUND')
    context=profile.get('context',{})
    if not isinstance(context,dict): context={}
    if (set(context)!=set(expected)|{'working_set_regime'} or
            digest({key:context.get(key) for key in expected})!=digest(expected)):
        issue('context','exact target/CPU/compiler/temporal implementation/layout/run boundary required')
    if not isinstance(context.get('working_set_regime'),str) or not context.get('working_set_regime'):
        issue('context.working_set_regime','explicit external regime assumption required')
    evidence(profile.get('context_evidence'),'context_evidence',expected=context)
    axes=dict(rows=features['rows'],columns=features['columns'],steps=features['steps'],
        body_dependency_depth=features['body_dependency_depth'],
        **{'arithmetic.'+op:features['arithmetic_per_element'][op] for op in _OPS})
    domain=profile.get('applicability',{})
    if not isinstance(domain,dict): domain={}
    if set(domain)!=set(axes): issue('applicability','complete shape/steps/arithmix/dependency calibration domain required')
    for key,value in axes.items():
        window=domain.get(key)
        if (not isinstance(window,list) or len(window)!=2 or any(type(v) is not int or v<0 for v in window)
                or not window[0]<=value<=window[1]): issue('applicability.'+key,'outside explicit inclusive calibration window')
    evidence(profile.get('applicability_evidence'),'applicability_evidence',expected=domain)
    model=profile.get('model',{}); calibration=profile.get('calibration',{})
    if not isinstance(model,dict): model={}
    if not isinstance(calibration,dict): calibration={}
    if set(model)!={'kind','feature_names','tick_seconds','intercept','coefficients'} or model.get('kind')!='EFFECTIVE_AFFINE_TEMPORAL_CALL':
        issue('model','explicit temporal effective-affine model required')
    names=model.get('feature_names',[])
    if (not isinstance(names,list) or any(not isinstance(name,str) or name not in _UNITS for name in names)
            or len(set(names))!=len(names)):
        issue('model.feature_names','unique supported feature subset required'); names=[]
    evidence(profile.get('calibration_evidence'),'calibration_evidence',expected=calibration)
    if (calibration.get('scope')!=CALIBRATION_SCOPE or calibration.get('target_program_executed') is not False
            or calibration.get('feature_names')!=['intercept',*names]
            or calibration.get('coefficient_units')!=dict(intercept='ticks',**{name:_UNITS[name] for name in names})
            or calibration.get('included_boundary')!={key:True for key in _INCLUDED}):
        issue('calibration','independent canonical recurrence fit with exact units and complete run boundary required')
    design=calibration.get('design_rows',[])
    if (not isinstance(design,list) or not design or any(not isinstance(row,list) or len(row)!=len(names)+1
            or any(type(v) is not int or v<0 for v in row) or row[0]!=1 for row in design)):
        issue('calibration.design_rows','original unscaled integer design rows required'); rank,in_span=None,False
    else:
        rank,in_span=_row_span(design,[1,*(features['values'][name] for name in names)])
        if not in_span: issue('calibration.design_rows','target row is outside the original identifiable design-row span')
    tick=parameter(model.get('tick_seconds'),'model.tick_seconds','seconds/tick',True)
    total=parameter(model.get('intercept'),'model.intercept','ticks'); terms={}
    coefficients=model.get('coefficients',{})
    if not isinstance(coefficients,dict): coefficients={}
    if set(coefficients)!=set(names): issue('model.coefficients','exact selected feature coefficients required')
    for name in names:
        coefficient=parameter(coefficients.get(name),'model.coefficients.'+name,_UNITS[name])
        if coefficient is None or total is None: total=None
        else:
            term=coefficient*features['values'][name]; total+=term
            terms[name]=dict(feature=features['values'][name],numerator=term.numerator,denominator=term.denominator)
    if layout['cost_frontier']: issue('layout.cost_frontier','primitive costs remain unresolved')
    if total is not None and total<=0: issue('model.total','positive complete-call estimate required')
    return dict(status='ESTIMATED' if not issues else 'UNRESOLVED',prediction_available=not issues,
        duration_ticks=None if issues or total is None else -(-total.numerator//total.denominator),
        tick_seconds=None if tick is None else dict(numerator=tick.numerator,denominator=tick.denominator),
        features=features,terms=terms,row_span_verified=in_span,exact_design_rank=rank,
        calibration=calibration,context=context,profile_sha256=digest(profile),unresolved=issues,
        assumptions=['EFFECTIVE_AFFINE_SUM_NOT_AN_ADDITIVITY_THEOREM','EXACT_CPU_GROUP_FIT_NOT_LINEAR_WORKER_SCALING',
                     'CALIBRATION_REGIME_NOT_PROVEN_CACHE_RESIDENCY','MODEL_OPTIMUM_NOT_PHYSICAL_OR_DEPLOYMENT_OPTIMUM'])


def _evaluate(loop,strategy,domain,out,write):
    verify_loop_strategy(loop,strategy); base=loop_execution_plan(loop,strategy)
    contract=dict(loop_sha256=loop_digest(loop),strategy_sha256=digest(strategy),body_plan_sha256=digest(document(base)))
    if (set(domain)!={'schema','schema_version','variants','search_budget','semantic_contract'} or
            domain['schema']!='scar.loop-worker-domain' or type(domain['schema_version']) is not int or domain['schema_version']!=1
            or domain['semantic_contract']!=contract): raise ValueError('original temporal worker domain/semantic contract differs')
    semantic=ResourceSchedule(base.region.id,(),(),(Task('loop-call',TaskKind.COMPUTE,
        attrs=dict(semantic_composite='SYNCHRONOUS_TEMPORAL_LOOP_CALL',semantic_contract_sha256=digest(contract))),),
        proof=dict(external_assumptions=strategy['external_assumptions']))
    task_hash=digest(task_implementation_contract(semantic,'loop-call'))
    rows=[]; options=[]; pools={}; common=None; target=None; seen=set()
    for index,item in enumerate(domain['variants']):
        if set(item)!={'id','plan','layout','profile','profile_reference','frontier_reason'} or item['id'] in seen:
            raise ValueError('complete unique temporal worker variants required')
        seen.add(item['id'])
        plan=_read(item['plan'],'plan'); layout=_read(item['layout'],'layout'); profile=_read(item['profile'],'profile')
        problems=[]; prediction=None; binding=plan.resource_binding
        try:
            _,plan=_admit(loop,strategy,plan)
            if digest(document(bind_cpu_placement(base,binding)))!=digest(document(plan)):
                raise ValueError('only original CPU binding/admissible payload budget may change')
            if target is not None and target!=(binding.target_path,binding.target_sha256): raise ValueError('family hardware snapshot differs')
            target=(binding.target_path,binding.target_sha256)
            verify_loop_layout(loop,strategy,plan,layout)
            if item['profile_reference'] is not None and digest(_fact(item['profile_reference'])[0])!=digest(profile):
                raise ValueError('original profile FactReference differs')
            prediction=predict_loop_worker_costs(loop,strategy,plan,layout,profile)
            if prediction['prediction_available']:
                context=deepcopy(prediction['context']); context.pop('cpu_ids'); context.pop('executor_id')
                compiler=context['compiler']; flags=compiler['options']
                if flags.count('-fopenmp')!=int(plan.cpu_threads>1): raise ValueError('worker-dependent compiler options differ')
                compiler['options']=[flag for flag in flags if flag!='-fopenmp']; context['tick_seconds']=prediction['tick_seconds']
                if common is not None and digest(common)!=digest(context): raise ValueError('CPU family compiler/layout/regime/tick basis differs')
                common=context
            if item['frontier_reason']: problems.append(item['frontier_reason'])
        except _ERRORS as error: problems.append(str(error))
        if binding: pools.setdefault(binding.executor_id,set()).update(binding.cpu_ids)
        demands={binding.executor_id:plan.cpu_threads} if binding else {}
        duration=prediction['duration_ticks'] if prediction and prediction['prediction_available'] and not problems else None
        conditions=['EXACT_LOOP_BODY_CALLEES_SHAPE_COUNT_AND_FRESH_FINAL_HOST_UNBOUND',
            'EXISTING_STATIC_ROWS_CPU_TEAM_AND_VALIDATED_SIMD_LAYOUT','CANONICAL_TEMPORAL_COST_APPLICABILITY_AND_RUNTIME_GUARDS']
        legality=dict(contract_sha256=task_hash,resource_demands=demands,conditions=conditions)
        evidence=_emit(out/f'variant-{index}/evidence.json',dict(schema='scar.loop-worker-evidence',schema_version=1,
            id=item['id'],prediction=prediction,issues=problems,duration_ticks=duration,legality=legality if duration is not None else None),write)
        artifact=ProviderArtifact(evidence['path'],evidence['sha256'],evidence['bytes'],'scar.loop-worker-evidence')
        fact=lambda pointer:[encode(ProviderFactReference(artifact,pointer))]
        options.append(dict(id=item['id'],duration_ticks=duration,resource_demands=demands,contract_sha256=task_hash,
            cost_evidence=fact('#/duration_ticks'),legality_conditions=conditions,legality_evidence=fact('#/legality')))
        rows.append(dict(**item,evidence=evidence,prediction_available=duration is not None,issues=problems,
                         unresolved=[] if prediction is None else prediction['unresolved']))
    solver=solve_implementation_choices(semantic,task_options={'loop-call':options},
        resource_capacities={key:len(cpus) for key,cpus in pools.items()},search_budget=domain['search_budget'])
    complete=solver['status']=='OPTIMAL_WITHIN_FINITE_MODEL' and not solver['execution_unresolved']
    return dict(schema='scar.loop-worker-choice',schema_version=1,output_directory=str(out),domain=_receipt(out/'domain.json'),
        semantic_contract=contract,variants=rows,solver=solver,complete_domain=complete,
        selected_variant_id=solver['chosen_option_ids']['loop-call'] if complete else None,
        choice_scope='FINITE_CPP_TEMPORAL_CPU_TEAM_FAMILY_FOR_LOWERING_ONLY',original_python_no_op_cost_modeled=False,
        comparison_model='AUXILIARY_SINGLE_CALL_CHOICE_NOT_PROGRAM_DATAFLOW',
        parallelism=dict(spatial='DISJOINT_STATIC_ROWS_AND_SIMD_LANES',sequential='TIME_RECURRENCE_NOT_SPLIT',
            processes='UNSUPPORTED_PARTITION_OWNERSHIP_IPC_AND_FAILURE_CONTRACT',
            mpi='UNSUPPORTED_RANK_PARTITION_TRANSFER_COLLECTIVE_AND_FRESH_GATHER_CONTRACT'),
        materialization_placement_optimized=False,implicit_copy_inserted=False,target_variants_compiled=False,
        target_variants_executed=False,deployment_authorized=False,physical_optimality_proven=False)


def plan_loop_workers(loop,strategy,variants,out,*,search_budget=100000):
    """Retain all explicit options and source receipts, then run the existing solver."""
    out=Path(out).absolute()
    if out.exists(): raise ValueError('fresh temporal worker output required')
    if type(search_budget) is not int or search_budget<1: raise ValueError('positive search budget required')
    base=loop_execution_plan(loop,strategy); items=[]; seen=set()
    for index,variant in enumerate(variants):
        if set(variant)-{'id','plan','layout','profile','frontier_reason'}: raise ValueError('unsupported temporal worker constraints')
        identifier=variant['id']
        if not isinstance(identifier,str) or not identifier or identifier in seen: raise ValueError('unique explicit variant IDs required')
        seen.add(identifier)
        if variant.get('plan') is None or variant.get('layout') is None: raise ValueError('bound plan and layout required')
        row=dict(id=identifier,frontier_reason=variant.get('frontier_reason'),profile_reference=None)
        for kind in ('plan','layout','profile'):
            value=variant.get(kind)
            if kind=='profile' and value is not None:
                data,origin=_profile(value)
                if origin['kind']=='FACT_REFERENCE': row['profile_reference']=origin['reference']; value=data
            row[kind]=_source(value,kind,out/f'sources/variant-{index}/{kind}.json')
        items.append(row)
    domain=dict(schema='scar.loop-worker-domain',schema_version=1,variants=items,search_budget=search_budget,
        semantic_contract=dict(loop_sha256=loop_digest(loop),strategy_sha256=digest(strategy),body_plan_sha256=digest(document(base))))
    _emit(out/'domain.json',domain,True)
    result=_evaluate(loop,strategy,domain,out,True); _emit(out/'choice.json',result,True)
    return result


def verify_loop_worker_choice(loop,strategy,value):
    report=loads(Path(value).read_bytes()) if isinstance(value,(str,Path)) else deepcopy(value)
    if report.get('schema')!='scar.loop-worker-choice' or type(report.get('schema_version')) is not int or report['schema_version']!=1:
        raise ValueError('versioned temporal worker choice required')
    out=Path(report['output_directory']); domain=_read(report['domain'],'domain')
    expected=_evaluate(loop,strategy,domain,out,False)
    if digest(expected)!=digest(report): raise ValueError('temporal worker choice differs from complete source replay')
    return report


def selected_loop_worker(loop,strategy,report):
    """Return the selected typed binding/layout, after complete-domain replay."""
    value=verify_loop_worker_choice(loop,strategy,report)
    if not value['complete_domain'] or value['selected_variant_id'] is None:
        raise ValueError('temporal worker domain unresolved; no selected lowering')
    row=next(row for row in value['variants'] if row['id']==value['selected_variant_id'])
    return _read(row['plan'],'plan'),_read(row['layout'],'layout')


__all__=['BOUNDARY','CALIBRATION_SCOPE','loop_worker_feature_units','features_for_loop','predict_loop_worker_costs',
         'plan_loop_workers','verify_loop_worker_choice','selected_loop_worker']
