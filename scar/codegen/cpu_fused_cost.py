"""Source-independent canonical models of strict CPU fused row stages.

The calibration never consumes a target expression DAG. Static target demand
is evaluated later with fitted coefficients and explicit geometric/context
checks. Predictions are DECLARED_COMPONENT_MODEL, not measured target costs.
"""
from copy import deepcopy
import ctypes
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

_HERE = Path(__file__).resolve().parent
_OPS = ('add', 'sub', 'mul', 'div', 'neg')
_FEATURES = ('stage', 'store_elements', 'logical_loads', 'add', 'sub', 'mul', 'div',
    'dependency_depth', 'backing_bytes', 'row_padding', 'misaligned_loads', 'llc_excess_bytes')
_ASSUMPTIONS = [
    'NONNEGATIVE_EFFECTIVE_AFFINE_MODEL_NOT_INSTRUCTION_LATENCIES_OR_PHYSICAL_OPTIMUM',
    'CANONICAL_DAGS_ARE_NOT_THE_TARGET_DAG_OR_TARGET_DATA_DISTRIBUTION',
    'LOGICAL_LOADS_AND_BACKING_EXTENTS_ARE_NOT_MEASURED_DRAM_TRAFFIC',
    'CACHE_CAPACITY_HINGE_IS_GEOMETRY_NOT_AVAILABLE_CACHE_OR_RESIDENCY_PROOF',
    'GENERIC_ALIAS_VERSIONING_AND_KERNEL_DISPATCH_ARE_PROXIES_NOT_TARGET_ASSEMBLY_IDENTITY',
    'PERSISTENT_PINNED_TEAM_STAGE_INCLUDES_OWN_IMPLICIT_COMPLETION_BARRIER',
    'WARM_REPEATED_CANONICAL_BUFFERS_NUMA_RESIDENCY_AND_CONTENTION_NOT_CONTROLLED',
    'MODEL_SELECTION_REQUIRES_SEPARATE_MEASURED_NUMERICAL_AND_PROFITABILITY_VALIDATION']


def _require(condition, message):
    if not condition: raise ValueError(message)
def _hash(raw): return sha256(raw).hexdigest()
def _json(value): return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True);path.write_bytes(_json(value))
def _receipt(path):
    raw=path.read_bytes();return dict(path=str(path),sha256=_hash(raw),bytes=len(raw))


def _canonical_cases():
    """Fixed generic dependency/fan-in families; no target is an input."""
    result=[]
    def case(loads, chains, sequence, holdout=False):
        statements=[f'const double x{i}=a.inputs[{i}][a.offsets[{i}]+row*a.strides[{i}]+col];' for i in range(loads)]
        counts={op:0 for op in _OPS};depth={f'x{i}':0 for i in range(loads)}
        def operation(op,left,right=None):
            name='v'+str(len(depth)); token=dict(add='+',sub='-',mul='*',div='/')[op]
            statements.append(f'const double {name}=({left}{token}{right});')
            counts[op]+=1;depth[name]=1+max(depth.get(left,0),depth.get(right,0));return name
        def combine(values):
            while len(values)>1:
                values=[operation('add',values[i],values[i+1]) if i+1<len(values) else values[i]
                        for i in range(0,len(values),2)]
            return values[0]
        values=[]
        for chain in range(chains):
            initial=combine([f'x{i}' for i in range(chain,loads,chains)])
            for op in sequence: initial=operation(op,initial,'a.scale')
            values.append(initial)
        output=combine(values);statements.append(f'a.output[row*a.cols+col]={output};')
        result.append(dict(id='canonical_'+str(len(result)),array_loads=loads,
            arithmetic=counts,dependent_operation_depth=depth[output],output_stores=1,
            holdout=holdout,body='\n'.join(statements),
            family='INDEPENDENT_CHAINS' if chains>1 else 'DEPENDENT_CHAIN' if sequence else 'BALANCED_LOAD_FANIN'))
    for op in ('add','sub','mul','div'):
        for length in (1,4,12):
            for chains in (1,4):case(chains,chains,[op]*length)
    for loads in (1,2,4,8):case(loads,1,[])
    case(3,1,['mul','add','sub','div'],True)
    case(4,2,['sub','mul','add'],True)
    case(7,1,[],True)
    # The rejected first design's holdouts are development data from here on.
    for previous in result:previous['holdout']=False
    for pattern in (['add','mul'],['sub','mul'],['add','sub','mul']):
        for repeats in (1,2):
            for chains in (1,2,4):case(chains,chains,pattern*repeats)
    case(5,1,['sub','mul','add','mul'],True)
    case(6,2,['mul','add','sub','mul'],True)
    case(7,1,['mul','sub','mul','add'],True)
    return result


def _layouts(loads, rows, cols, geometry):
    if geometry=='INDEPENDENT':return [dict(stride=cols,offset=0,storage_group=i) for i in range(loads)]
    if geometry=='SHARED_SHIFTED':return [dict(stride=cols+4,offset=i*(cols+4)+i%3,storage_group=0) for i in range(loads)]
    if geometry=='MIXED_ALIGNED':return [dict(stride=cols+cols%2+2*(i%2),
        offset=(i//2)*(cols+cols%2+2*(i%2)),storage_group=i%2) for i in range(loads)]
    return [dict(stride=cols+(2 if i%2 else 0),offset=(i//2)*(cols+(2 if i%2 else 0))+i%2,
                 storage_group=i%2) for i in range(loads)]


def fused_features(parameters, llc_capacity_bytes):
    """Logical operation/geometry features; source names and hashes are absent."""
    fields={'rows','cols','workers','arithmetic','array_loads','output_stores',
        'dependent_operation_depth','operand_layouts','backing_storage_groups'}
    _require(type(parameters) is dict and set(parameters)==fields, 'complete generic fused geometry required')
    p=parameters;r,c,w=(p[key] for key in ('rows','cols','workers'))
    _require(all(type(v) is int and v>0 for v in (r,c,w)), 'positive exact shape/worker counts required')
    operations=p['arithmetic'];layouts=p['operand_layouts'];loads=p['array_loads']
    _require(type(operations) is dict and set(operations)==set(_OPS)
        and all(type(n) is int and n>=0 for n in operations.values()), 'supported exact arithmetic counts required')
    _require(operations['neg']==0, 'negation cost is uncalibrated by this canonical family')
    _require(type(loads) is int and 1<=loads<=8 and type(layouts) is list and len(layouts)==loads
        and p['output_stores']==1 and type(p['dependent_operation_depth']) is int
        and 0<=p['dependent_operation_depth']<=sum(operations.values()), 'bounded load/store/dependency signature required')
    sizes={};paddings={};misaligned_rows=0;max_row_offset=0;max_col_offset=0
    for layout in layouts:
        _require(type(layout) is dict and set(layout)=={'stride','offset','storage_group'}, 'anonymous backing groups required')
        stride,offset,group=(layout[k] for k in ('stride','offset','storage_group'))
        _require(all(type(n) is int for n in (stride,offset,group)) and stride>=c and offset>=0
            and 0<=group<8, 'fixed positive unit-column row layouts required')
        sizes[group]=max(sizes.get(group,0),offset+(r-1)*stride+c)
        paddings[group]=max(paddings.get(group,0),stride-c)
        misaligned_rows+=r//2+(r%2)*(offset%2) if stride%2 else r*(offset%2)
        max_row_offset=max(max_row_offset,offset//stride);max_col_offset=max(max_col_offset,offset%stride)
    _require(p['backing_storage_groups']==len(sizes), 'backing group count differs from layouts')
    elements=r*c/w;backing_bytes=sum(sizes.values())*8
    values=dict(stage=1.,store_elements=elements,logical_loads=elements*loads,
        **{op:elements*operations[op] for op in ('add','sub','mul','div')},
        dependency_depth=elements*p['dependent_operation_depth'],backing_bytes=backing_bytes/w,
        row_padding=r*sum(paddings.values())/w,misaligned_loads=misaligned_rows*c/w,
        llc_excess_bytes=max(0,backing_bytes+r*c*8-llc_capacity_bytes)/w)
    geometry=dict(backing_bytes=backing_bytes,max_row_padding=max(paddings.values()),
        max_row_offset=max_row_offset,max_col_offset=max_col_offset,
        backing_storage_groups=len(sizes),logical_bytes_are_dram_bytes=False)
    return values,geometry


class _Layout(ctypes.Structure):
    _fields_=[('rows',ctypes.c_int64),('cols',ctypes.c_int64),('strides',ctypes.c_int64*8),
        ('offsets',ctypes.c_int64*8),('groups',ctypes.c_int32*8),('loads',ctypes.c_int32)]


def _render_source(context,cases):
    cpus=context['worker_cpu_ids']
    original=(_HERE/'loop_v1.cpp.in').read_text();start=original.index('class ThreadAffinity {')
    affinity=original[start:original.index('\n#endif',start)]
    start=original.index('class FloatingScope {');floating=original[start:original.index('\n};',start)+3]
    functions=[]
    for case in cases:
        functions.append(f'__attribute__((noinline)) void {case["id"]}(const Args &a) {{\n'
            '#pragma omp for schedule(static)\nfor(int64_t row=0;row<a.rows;++row)\n'
            'for(int64_t col=0;col<a.cols;++col){\n'+case['body']+'\n}\n}')
    source=(_HERE/'cpu_fused_cost_v1.cpp.in').read_text()
    for key,value in dict(CPU_IDS=', '.join(map(str,cpus)),CPU_SLOTS=str(max(cpus)+1),
        AFFINITY_CLASS=affinity,FLOATING_CLASS=floating,CANONICAL_FUNCTIONS='\n'.join(functions),
        CANONICAL_POINTERS=', '.join('&'+c['id'] for c in cases),
        CANONICAL_LOADS=', '.join(str(c['array_loads']) for c in cases)).items():source=source.replace('@'+key+'@',value)
    return source


def _native(context,directory,cases):
    compiler=context['backend']['compiler'];cpus=context['worker_cpu_ids']
    _require(len(cpus)>1 and set(cpus)<=os.sched_getaffinity(0)
        and sorted(os.sched_getaffinity(0))==context['caller_affinity_cpu_ids'], 'exact caller/worker mask required')
    identity=subprocess.run([*compiler['argv'],'--version'],check=True,capture_output=True,text=True).stdout.splitlines()[0]
    _require(identity==compiler['identity'] and 'g++' in identity and '-fopenmp' in compiler['options'],
             'recorded GCC/OpenMP compiler context required')
    source=_render_source(context,cases)
    directory.mkdir(parents=True);(directory/'canonical.cpp').write_text(source)
    (directory/'scar_abi_v1.h').write_bytes((_HERE/'scar_abi_v1.h').read_bytes())
    library=directory/'canonical.so';remarks=directory/'vectorization.txt'
    command=[*compiler['argv'],*compiler['options'],'-fopt-info-vec-optimized='+str(remarks),
        str(directory/'canonical.cpp'),'-o',str(library)]
    compiled=subprocess.run(command,capture_output=True,text=True)
    (directory/'compile.log').write_text(compiled.stdout+compiled.stderr)
    _require(compiled.returncode==0,'canonical fused-cost compilation failed: '+compiled.stderr[-1500:])
    assembly=subprocess.run(['objdump','-d','-C','--no-show-raw-insn',str(library)],check=True,capture_output=True,text=True).stdout
    (directory/'assembly.txt').write_text(assembly)
    instructions=re.findall(r'^\s*[0-9a-f]+:\s+([a-zA-Z][a-zA-Z0-9.]*)',assembly,re.M)
    handle=ctypes.CDLL(str(library));function=handle.scar_fused_cost_probe_v1
    function.argtypes=[ctypes.c_int32,ctypes.POINTER(_Layout),ctypes.c_uint64,ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_uint64),ctypes.POINTER(ctypes.c_int32)]
    function.restype=ctypes.c_int32
    receipt=dict(compiler=compiler,command=command,diagnostic_options=['-fopt-info-vec-optimized'],
        source=_receipt(directory/'canonical.cpp'),library=_receipt(library),
        assembly=_receipt(directory/'assembly.txt'),vectorization_remarks=_receipt(remarks),
        packed_arithmetic={op:instructions.count(op+'pd') for op in ('add','sub','mul','div')},
        scalar_arithmetic={op:instructions.count(op+'sd') for op in ('add','sub','mul','div')},
        v_prefixed_mnemonics=sorted({op for op in instructions if op.startswith('v')}),
        complete_isa_or_regalloc_proof=False,target_source_or_dag_compiled=False)
    return handle,function,receipt


def _llc(context):
    from scar.optimization.scope_accounting import _fact
    reference=context['hardware_context']['cpu_cache']['evidence'];caches,_=_fact(reference)
    cpus=set(context['worker_cpu_ids'])
    shared=[row for row in caches if row['type']=='Unified' and cpus<=set(row['shared_cpu_ids'])
        and type(row.get('capacity_bytes')) is int and row['capacity_bytes']>0]
    _require(bool(shared),'retained shared unified cache geometry required')
    last=max(shared,key=lambda row:row['level'])
    return last['capacity_bytes'],dict(reference=reference,level=last['level'],
        reported_capacity_bytes=last['capacity_bytes'],available_or_reserved_capacity_proven=False)


def _design(case):
    return ([(320,383),(512,639)] if case['holdout'] else [(128,127),(384,511),(768,1023)])


def _fit_rows(train, holdout):
    import numpy as np
    from scipy.optimize import nnls
    _require(bool(train) and bool(holdout),'independent training and holdout observations required')
    matrix=np.asarray([[row['features'][name] for name in _FEATURES] for row in train])
    targets=np.asarray([row['median_ns'] for row in train]);scales=np.max(np.abs(matrix),axis=0)
    active=scales>0;scaled=matrix[:,active]/scales[active]
    solved,_=nnls(scaled,targets,maxiter=10000);coefficients=np.zeros(len(_FEATURES));coefficients[active]=solved/scales[active]
    errors=matrix@coefficients-targets;validation=[]
    for row in holdout:
        predicted=sum(row['features'][name]*value for name,value in zip(_FEATURES,coefficients))
        validation.append(dict(primitive=row['primitive'],geometry=row['geometry'],rows=row['parameters']['rows'],
            cols=row['parameters']['cols'],observed_ns=row['median_ns'],predicted_ns=predicted,
            relative_error=abs(predicted-row['median_ns'])/max(row['median_ns'],1)))
    return dict(coefficients={name:float(value) for name,value in zip(_FEATURES,coefficients)},
        rank=int(np.linalg.matrix_rank(scaled)),active_parameters=int(sum(active)),
        unique_coefficients_identified=bool(np.linalg.matrix_rank(scaled)==sum(active)),
        nnls_zero_coefficients=[name for name,value in zip(_FEATURES,coefficients) if value==0],
        residual=dict(rmse_ns=float(np.sqrt(np.mean(errors**2))),max_relative_error=float(np.max(np.abs(errors)/np.maximum(targets,1)))),
        holdout=validation,maximum_holdout_relative_error=max(row['relative_error'] for row in validation),
        training_feature_matrix=matrix.tolist(),feature_scales=scales.tolist(),
        training_observation_count=len(train),holdout_observation_count=len(holdout))


def _regimes(observations,capacity,relative_error_limit):
    result={}
    for groups in (1,2):
        def matches(row):
            p=row['parameters'];arithmetic=p['arithmetic']
            footprint=row['geometry_properties']['backing_bytes']+p['rows']*p['cols']*8
            return (p['backing_storage_groups']==groups and sum(arithmetic.values())<=16
                and arithmetic['div']==0 and arithmetic['neg']==0 and footprint<=capacity)
        rows=[row for row in observations if matches(row)]
        fitted=_fit_rows([row for row in rows if not row['holdout']], [row for row in rows if row['holdout']])
        fitted.update(regime=dict(backing_storage_groups=groups,maximum_arithmetic_operations=16,
            division_and_negation_supported=False,working_set='TOTAL_REPORTED_EXTENTS_WITHIN_REPORTED_SHARED_LLC',
            reserved_or_effective_cache_capacity_proven=False),
            admitted=bool(fitted['maximum_holdout_relative_error']<=relative_error_limit),
            precision_policy=dict(maximum_holdout_relative_error=relative_error_limit,
                authority='EXPLICIT_ENGINEERING_ADMISSION_POLICY_NOT_ERROR_BOUND_OR_MATHEMATICAL_GUARANTEE'))
        result['backing-groups-'+str(groups)]=fitted
    return result


def calibrate_fused_cpu(resource_context,out,*,repetitions=7,warmup=2):
    """Measure one CPU mask's independent canonical design and fit NNLS."""
    context=deepcopy(resource_context);out=Path(out).resolve()
    _require(not out.exists() and type(repetitions) is int and repetitions>=3
        and type(warmup) is int and warmup>=1,'fresh output, at least 3 samples and 1 warmup required')
    capacity,cache=_llc(context);cases=_canonical_cases();started=time.time()
    handle,function,native=_native(context,out/'native',cases);observations=[]
    load_before=list(os.getloadavg());workers=len(context['worker_cpu_ids'])
    for index,case in enumerate(cases):
        for rows,cols in _design(case):
            for geometry in ('INDEPENDENT','SHARED_SHIFTED','MIXED_GROUPS','MIXED_ALIGNED'):
                layouts=_layouts(case['array_loads'],rows,cols,geometry)
                parameters=dict(rows=rows,cols=cols,workers=workers,arithmetic=case['arithmetic'],
                    array_loads=case['array_loads'],output_stores=1,dependent_operation_depth=case['dependent_operation_depth'],
                    operand_layouts=layouts,backing_storage_groups=len({d['storage_group'] for d in layouts}))
                features,properties=fused_features(parameters,capacity)
                layout=_Layout();layout.rows=rows;layout.cols=cols;layout.loads=case['array_loads']
                for i,row in enumerate(layouts):layout.strides[i]=row['stride'];layout.offsets[i]=row['offset'];layout.groups[i]=row['storage_group']
                samples=(ctypes.c_double*repetitions)();checksum=ctypes.c_uint64();cpus=(ctypes.c_int32*2)()
                status=function(index,ctypes.byref(layout),repetitions,warmup,samples,ctypes.byref(checksum),cpus)
                _require(status==0,'canonical primitive FP/affinity/allocation guard: '+str(status))
                values=list(samples);_require(all(math.isfinite(n) and n>=0 for n in values),'finite nonnegative native clocks required')
                observations.append(dict(primitive=case['id'],family=case['family'],holdout=case['holdout'],
                    geometry=geometry,parameters=parameters,features=features,geometry_properties=properties,
                    samples_ns=values,median_ns=statistics.median(values),min_ns=min(values),max_ns=max(values),
                    stdev_ns=statistics.stdev(values),caller_cpu_before=cpus[0],caller_cpu_after=cpus[1]))
    _save(out/'raw-observations.json',dict(schema='scar.cpu-fused-canonical-observations',schema_version=1,
        resource_context=context,native=native,observations=observations,repetitions=repetitions,warmup=warmup))
    global_fit=_fit_rows([row for row in observations if not row['holdout']],
        [row for row in observations if row['holdout']])
    implementation={name:_hash((_HERE/name).read_bytes()) for name in
        ('cpu_fused_cost.py','cpu_fused_cost_v1.cpp.in','loop_v1.cpp.in','scar_abi_v1.h')}
    profile=dict(schema='scar.cpu-fused-cost-model',schema_version=2,resource_context=context,
        model_kind='DECLARED_EFFECTIVE_NONNEGATIVE_AFFINE_COMPONENT_MODEL',
        features=list(_FEATURES),feature_units={name:'dimensionless stage count' if name=='stage' else
            'bytes/worker' if name.endswith('bytes') else 'logical count/worker' for name in _FEATURES},
        coefficient_unit='ns/feature-unit',global_fit=global_fit,global_fit_is_not_admitted_for_prediction=True,
        local_models=_regimes(observations,capacity,0.20),maximum_holdout_relative_error_policy=0.20,
        policy_scope='PRECISION_WITHIN_MATCHED_CACHE_GEOMETRY_AND_LOW_ARITHMETIC_REGIME',
        observations=observations,
        cache_geometry=cache,native=native,implementation_files=implementation,
        implementation_sha256=_hash(_json(implementation)),
        applicability=dict(rows=[128,768],cols=[127,1023],max_logical_loads=8,max_backing_groups=8,
            max_row_padding=4,max_row_offset=7,max_col_offset=2,supported_negation_count=0,
            arithmetic_limits={op:max(case['arithmetic'][op] for case in cases) for op in _OPS},
            dependency_depth_limit=max(case['dependent_operation_depth'] for case in cases),
            identification_methods=['CALIBRATED_LINEAR_ROW_SPAN','NNLS_OPTIMAL_FACE_MIN_MAX_LINEAR_PROGRAMS'],
            exact_resource_context=True,
            convex_hull_or_error_bound_proven=False,
            model_error_is_not_a_performance_guarantee=True),
        repetitions=repetitions,warmup=warmup,observed_started_unix_seconds=started,
        observed_finished_unix_seconds=time.time(),host_load=dict(before=load_before,after=list(os.getloadavg()),isolated=False),
        timing_scope='MAX_WORKER_LOCAL_NATIVE_STAGE_WALL_WITH_OWN_IMPLICIT_BARRIER',
        assumptions=_ASSUMPTIONS,target_workload_executed=False,target_variants_ranked=False,
        physical_optimality_proven=False,deployment_authorized=False)
    _save(out/'profile.json',profile)
    return profile


def _verify_model(model):
    """Replay features and fitted coefficients from retained canonical samples."""
    _require(model['features']==list(_FEATURES) and model['model_kind']==
        'DECLARED_EFFECTIVE_NONNEGATIVE_AFFINE_COMPONENT_MODEL', 'canonical fitted model profile differs')
    expected=[]
    for case in _canonical_cases():
        for rows,cols in _design(case):
            for geometry in ('INDEPENDENT','SHARED_SHIFTED','MIXED_GROUPS','MIXED_ALIGNED'):
                expected.append((case,rows,cols,geometry))
    observations=model['observations']
    _require(len(observations)==len(expected),'complete fixed canonical design required')
    for row,(case,rows,cols,geometry) in zip(observations,expected):
        p=row['parameters']
        _require(row['primitive']==case['id'] and row['holdout']==case['holdout'] and row['geometry']==geometry
            and p['rows']==rows and p['cols']==cols and p['arithmetic']==case['arithmetic']
            and p['dependent_operation_depth']==case['dependent_operation_depth'] and p['array_loads']==case['array_loads']
            and p['operand_layouts']==_layouts(case['array_loads'],rows,cols,geometry), 'canonical observation design differs')
        features,_=fused_features(p,model['cache_geometry']['reported_capacity_bytes'])
        _require(features==row['features'] and len(row['samples_ns'])==model['repetitions']
            and all(type(n) in (int,float) and math.isfinite(n) and n>=0 for n in row['samples_ns'])
            and statistics.median(row['samples_ns'])==row['median_ns'], 'canonical sample/feature receipt differs')
    actual=_fit_rows([row for row in observations if not row['holdout']],
        [row for row in observations if row['holdout']])
    _require(actual==model['global_fit'] and model['maximum_holdout_relative_error_policy']==0.20,
             'global diagnostics or explicit precision policy differs')
    _require(_regimes(observations,model['cache_geometry']['reported_capacity_bytes'],0.20)==model['local_models'],
             'local coefficients/holdout/precision admission differ from canonical replay')


def _identified_value(fitted,features):
    """Require a unique prediction over the calibrated nonnegative optimum."""
    import numpy as np
    from scipy.optimize import linprog
    matrix=np.asarray(fitted['training_feature_matrix']);scales=np.asarray(fitted['feature_scales']);active=scales>0
    vector=np.asarray([features[name] for name in _FEATURES])
    _require(np.all(vector[~active]==0),'fused feature has no calibrated coefficient observations')
    scaled=matrix[:,active]/scales[active];normal=vector[active]/scales[active]
    coefficients=np.asarray([fitted['coefficients'][name] for name in _FEATURES])
    value=float(vector@coefficients);weights=np.linalg.lstsq(scaled.T,normal,rcond=None)[0]
    if np.linalg.norm(scaled.T@weights-normal)<=1e-8*max(np.linalg.norm(normal),1):
        return value,dict(method='CALIBRATED_LINEAR_ROW_SPAN',identified=True)
    _,_,vectors=np.linalg.svd(scaled,full_matrices=False);rank=np.linalg.matrix_rank(scaled)
    basis=vectors[:rank];rhs=basis@(coefficients[active]*scales[active])
    minimum=linprog(normal,A_eq=basis,b_eq=rhs,bounds=(0,None),method='highs')
    maximum=linprog(-normal,A_eq=basis,b_eq=rhs,bounds=(0,None),method='highs')
    _require(minimum.success and maximum.success,'nonnegative optimal-face prediction is unresolved')
    gap=-maximum.fun-minimum.fun;tolerance=max(1e-5,abs(value)*1e-7)
    _require(gap<=tolerance,'fused prediction is not identified within the calibrated nonnegative optimal face')
    return value,dict(method='NNLS_OPTIMAL_FACE_MIN_MAX_LINEAR_PROGRAMS',identified=True,
        minimum_ns=minimum.fun,maximum_ns=-maximum.fun,gap_ns=gap,numeric_tolerance_ns=tolerance,
        scope='MODEL_IDENTIFIABILITY_NOT_PHYSICAL_ACCURACY')


def predict_fused_components(domain,candidate_id,profile,out):
    """Instantiate generic features; do not fit, compile or execute the target."""
    from scar.ir.execution_recipe import artifact_reference,read_artifact
    from scar.optimization.stateful_cpu_model import stateful_cpu_demand
    from scar.optimization.resource_solver import semantic_sha256
    from scar.optimization.stateful_cpu_model import _domain
    reference=artifact_reference(profile,'scar.cpu-fused-cost-model');model=read_artifact(reference)
    domain=_domain(domain);demand=stateful_cpu_demand(domain,candidate_id)
    _require(model['schema_version']==2 and model['resource_context']==demand['resource_context'], 'fused cost resource/compiler/runtime context differs')
    _verify_model(model)
    for row in model['native'].values():
        if type(row) is dict and {'path','sha256','bytes'}<=set(row):
            _require(_receipt(Path(row['path']))==row,'canonical native/source/remarks receipt differs')
    current={name:_hash((_HERE/name).read_bytes()) for name in model['implementation_files']}
    _require(all(current[name]==value for name,value in model['implementation_files'].items()
        if name!='cpu_fused_cost.py'),'canonical native template/header/helper implementation differs')
    _require(Path(model['native']['source']['path']).read_text()==_render_source(model['resource_context'],_canonical_cases()),
             'canonical native source differs from complete current replay')
    _require(_llc(model['resource_context'])[1]==model['cache_geometry'],'retained cache geometry differs')
    compatibility=dict(producer_implementation_sha256=model['implementation_sha256'],
        current_prediction_implementation_sha256=_hash(_json(current)),
        whole_driver_bytes_match=current==model['implementation_files'],
        canonical_native_text_replayed=True,numerical_fit_and_holdouts_replayed=True,
        scope='HISTORICAL_PRODUCER_IDENTITY_WITH_EXACT_NATIVE_TEXT_AND_NUMERICAL_MODEL_REPLAY')
    components=[];unresolved=[];proposals=[]
    for term in demand['terms']:
        if term['component']!='FUSED_SCALAR_DAG_ROW_STAGE':continue
        proposal=None
        try:
            p=term['parameters'];features,geometry=fused_features(p,model['cache_geometry']['reported_capacity_bytes'])
            a=model['applicability']
            regime_key='backing-groups-'+str(p['backing_storage_groups'])
            fitted=model['local_models'].get(regime_key)
            _require(fitted is not None,'no matched local cache/backing-group model')
            point=sum(features[name]*fitted['coefficients'][name] for name in _FEATURES)
            proposal=dict(component=term['component'],parameters=p,
                basis='NON_SELECTABLE_FITTED_COMPONENT_DIAGNOSTIC',point_estimate_ns=point,
                features=features,geometry=geometry,regime=fitted['regime'],
                source_model=dict(artifact=reference,locator='#/local_models/'+regime_key+'/coefficients'),
                source_validation=dict(artifact=reference,locator='#/local_models/'+regime_key+'/holdout'),
                source_provenance=term['source_ref'],source_graph_hash_is_not_primitive_key=True,
                rank=fitted['rank'],active_parameters=fitted['active_parameters'],
                maximum_holdout_relative_error=fitted['maximum_holdout_relative_error'],
                precision_policy=fitted['precision_policy'],precision_admitted=fitted['admitted'],
                identification=dict(identified=False,reason='NOT_YET_CHECKED'),
                geometric_applicability=False,assumptions=model['assumptions'],
                selectable=False,exportable_as_complete_cost=False,measured_target_component=False)
            _require(sum(p['arithmetic'].values())<=16 and p['arithmetic']['div']==0
                and geometry['backing_bytes']+p['rows']*p['cols']*8<=model['cache_geometry']['reported_capacity_bytes'],
                'fused working-set/arithmetic regime is outside precision-supported local scope')
            _require(a['rows'][0]<=p['rows']<=a['rows'][1] and a['cols'][0]<=p['cols']<=a['cols'][1]
                and geometry['max_row_padding']<=a['max_row_padding'] and geometry['max_row_offset']<=a['max_row_offset']
                and geometry['max_col_offset']<=a['max_col_offset']
                and p['workers']==len(model['resource_context']['worker_cpu_ids'])
                and p['dependent_operation_depth']<=a['dependency_depth_limit']
                and all(p['arithmetic'][op]<=a['arithmetic_limits'][op] for op in _OPS),
                'fused row/stride/operation/worker geometry outside canonical calibration range')
            proposal['geometric_applicability']=True
            value,identification=_identified_value(fitted,features)
            proposal['identification']=identification
            _require(math.isfinite(value) and value>=0,'invalid fitted component prediction')
            _require(fitted['admitted'], 'independent holdout maximum relative error '
                f"{fitted['maximum_holdout_relative_error']:.12g} exceeds engineering precision limit "
                f"{fitted['precision_policy']['maximum_holdout_relative_error']:.12g}")
            components.append(dict(component=term['component'],parameters=p,basis='DECLARED_COMPONENT_MODEL',median_ns=value,
                features=features,geometry=geometry,source_model=dict(artifact=reference,locator='#/local_models/'+regime_key+'/coefficients'),
                source_validation=dict(artifact=reference,locator='#/local_models/'+regime_key+'/holdout'),
                source_provenance=term['source_ref'],source_graph_hash_is_not_primitive_key=True,
                rank=fitted['rank'],active_parameters=fitted['active_parameters'],identification=identification,
                precision_policy=fitted['precision_policy'],regime=fitted['regime'],
                maximum_holdout_relative_error=fitted['maximum_holdout_relative_error'],assumptions=model['assumptions'],
                measured_target_component=False,physical_optimality_proven=False))
        except ValueError as error:
            unresolved.append(dict(component=term['component'],parameters=term['parameters'],reason=str(error)))
            if proposal is not None:
                proposal['rejection_reason']=str(error);proposals.append(proposal)
    result=dict(schema='scar.stateful-resource-calibration',schema_version=1,resource_context=demand['resource_context'],
        components=components,unsupported_components=unresolved,proposed_estimates=proposals,
        source_model=reference,implementation_compatibility=compatibility,
        all_fused_components_covered=not unresolved,complete_method_cost_measured=False,
        target_workload_executed=False,target_variants_ranked=False,physical_optimality_proven=False)
    out=Path(out).resolve();_require(not out.exists(),'fresh fused component output required');out.mkdir(parents=True)
    _save(out/'observations.json',result)
    return result


__all__=['calibrate_fused_cpu','predict_fused_components','fused_features']
