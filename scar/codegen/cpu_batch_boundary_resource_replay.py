"""Strict readback of frozen auxiliary probes and exact-mask raw samples.

No native library, timed probe, target source or full domain reconstruction is
executed. Source rendering, canonical function CodeIDs/contract and component
statistics are reconstructed from the original frozen parents.
"""
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics

from . import cpu_batch_boundary_resource as B
from scar.ir.execution_recipe import read_artifact
from scar.trace.code_snapshot import code_key


def _need(value, message): B._need(value, message)
def _equal(actual, expected, message): _need(actual == expected, message)


def _canonical(pack, folder):
    region = pack['canonical_region']; plan = pack['canonical_plan']
    lines = ['def _scar_boundary('+', '.join(region['parameter_names'])+'):',
        "    raise RuntimeError('canonical probe fallback must never execute')"]
    for index, _ in enumerate(region['callees']):
        lines += [f'def _scar_callee_{index}():', "    raise RuntimeError('canonical callee body must never execute')"]
    path = folder/'canonical_boundary.py'; raw = path.read_bytes()
    _equal(raw, ('\n'.join(lines)+'\n').encode(), 'canonical source differs from declared non-target bodies')
    compiled = compile(raw, str(path), 'exec', dont_inherit=True, optimize=0)
    codes = {c.co_name:c for c in compiled.co_consts if hasattr(c, 'co_code')}
    def source(name):
        code = codes[name]
        return dict(path=str(path),sha256=sha256(raw).hexdigest(),code_key=code_key(code),qualname=code.co_qualname)
    inputs = []
    for item in region['inputs']:
        array = item['kind'] == 'array'; shape = plan['input_shapes'][item['id']] if array else None
        inputs.append(dict(id=item['id'],path=item['path'],kind=item['kind'],shape=shape,
            nbytes=None if not array else shape[0]*shape[1]*8,written=item['id'] in plan['shadows']))
    callees = [dict(path=item['path'],kind=item['kind'],source=source('_scar_callee_'+str(index)))
        for index,item in enumerate(region['callees'])]
    providers = [dict(item) for item in region['providers']]
    structure = dict(parameters=region['parameter_names'],inputs=inputs,
        callee_paths=[item['path'] for item in region['callees']],
        provider_paths=[item['lookup_path'] for item in providers],
        scalar_types=['float']*sum(item['kind'] == 'scalar' for item in inputs),
        array_base_policy='INDEPENDENT_CONTIGUOUS_OWNING_ALLOCATIONS',
        read_only_arrays=[item['id'] for item in inputs if item['kind'] == 'array' and not item['written']],
        canonical_source=B._receipt(path),source_body='RAISE_IF_FALLBACK_NO_TARGET_DAG')
    return source('_scar_boundary'), inputs, callees, providers, structure


def _contract(pack, context, folder):
    from scar.runtime.stateful_contract import ASSUMPTIONS,FP_POLICY,MEMORY_SCOPE,validate_contract
    source,inputs,callees,providers,structure = _canonical(pack,folder)
    cpus=context['binding']['cpu_ids']; memory=pack['memory']; parameters=pack['guard_parameters']
    core=dict(region_sha256=B._semantic(parameters),plan_sha256=B._semantic(dict(parameters=parameters,cpu_ids=cpus,memory=memory)),
        provider_runtime=pack['provider_runtime'])
    ids=dict(build_id=pack['libraries']['boundary']['sha256'],region_sha256=core['region_sha256'],
        plan_sha256=core['plan_sha256'],contract_sha256=B._semantic(core))
    selection=dict(decision='TRANSFORM_UNDER_CONTRACT',selected=True,artifact=ids,
        input_shapes={item['id']:item['shape'] for item in inputs if item['kind']=='array'},external_assumptions=sorted(ASSUMPTIONS))
    expected=validate_contract(dict(schema='scar.stateful-runtime-contract',schema_version=1,profile='cpu-f64-rank2-stateful-v1',
        native=dict(library=Path(pack['libraries']['boundary']['path']).name,library_sha256=pack['libraries']['boundary']['sha256'],
            stateful_abi_version=1,ids=ids,contract=core),source=source,parameters=pack['canonical_region']['parameter_names'],
        inputs=inputs,callee_guards=callees,providers=providers,
        cpu=dict(ids=cpus,threads=len(cpus),architecture='x86_64',os='linux',pointer_bits=64,byteorder='little',required_isa='BASELINE_X86_64'),
        memory=dict(**memory,scope=MEMORY_SCOPE,policy='UNBOUND'),provider_runtime=pack['provider_runtime'],
        guards=['CANONICAL_STRUCTURAL_GUARD_AND_PACKING_MODEL'],external_assumptions=sorted(ASSUMPTIONS),
        floating_environment_policy=FP_POLICY,selection=selection))
    _equal(json.loads((folder/'canonical-contract.json').read_bytes()),expected,'canonical source/ABI/core contract differs')
    return expected, structure, ids


def _render(pack):
    domain=read_artifact(pack['domain']); region=read_artifact(pack['source_region']); plan=read_artifact(pack['base_stage_plan'])
    eligible=[r for r in domain['candidates'] if r['legal_under_guards'] and (r['worker_count']>1)==(pack['mode']=='parallel')]
    _need(eligible, 'pack mode lacks original candidates')
    canonical=dict(parameter_names=region['parameter_names'],
        inputs=[{key:r[key] for key in ('id','path','kind')} for r in region['inputs']],
        callees=[{key:r[key] for key in ('path','kind')} for r in region['callees']],
        providers=[{key:r[key] for key in ('id','module','symbol','lookup_path','semantics')} for r in region['providers']])
    _equal(pack['canonical_region'],canonical,'canonical metadata differs from original region parents')
    _equal(pack['canonical_plan'],{key:plan[key] for key in ('input_shapes','shadows')},'canonical input/layout differs from original plan')
    _equal(pack['memory'],{key:plan[key] for key in ('external_array_bytes','workspace_payload_bytes',
        'total_region_array_payload_bytes','memory_budget_bytes')},'canonical memory differs from original plan')
    resource,boundary,scalar=B._sources(eligible[0]['binding']['cpu_ids'],max(pack['caller_affinity_cpu_ids'])+1,
        pack['mode']=='parallel',pack['guard_parameters'],pack['memory'])
    _equal(pack['scalar_helpers_sha256'],scalar,'canonical tagged scalar helper differs')
    for name,text in (('resource',resource),('boundary',boundary)):
        _equal(Path(pack['sources'][name]['path']).read_bytes(),text.encode(),'rendered canonical '+name+' differs')
        expected=[*pack['compiler']['argv'],*pack['compiler']['options'],pack['sources'][name]['path'],'-o',pack['libraries'][name]['path']]
        _need(expected in pack['compiler_commands'],'canonical compiler command differs')
    for header in pack['headers']:
        _equal(Path(header['path']).read_bytes(),(B._HERE/Path(header['path']).name).read_bytes(),'canonical ABI header differs')


def _samples(row, term, basis):
    _equal(row['component'],term['component'],'component signature differs')
    _equal(row['parameters'],term['parameters'],'component parameters differ')
    _equal(row['basis'],basis,'component evidence basis differs')
    samples=row['samples_ns']; _need(type(samples)is list and len(samples)==3
        and all(type(x)in(int,float)and math.isfinite(x)and x>=0 for x in samples),'three finite raw primitive samples required')
    _equal(row['measurement']['samples_ns'],samples,'raw measurement samples differ')
    for key,value in dict(median_ns=statistics.median(samples),min_ns=min(samples),max_ns=max(samples),stdev_ns=statistics.stdev(samples)).items():
        _equal(row[key],value,'component '+key+' differs from original raw samples')


def _subset(actual, expected, label):
    _need(all(key in actual and actual[key]==value for key,value in expected.items()),label+' observations differ')


def _resource_measurement(measurement,term,pack,cpus):
    component=term['component']; parameters=term['parameters']
    serial=component in ('ROW_COPY_SERIAL','INPUT_FINITE_SCAN_SERIAL','ALLOCATE_FIXED_POOL','RELEASE_FIXED_POOL')
    team=component in ('CALLER_WORKER_AFFINITY_FENV_TEAM','SUCCESS_CHECK_AND_CALLER_RESTORATION')
    scope='SERIAL_CALLER_UNBOUND_NATIVE_CLOCK' if serial else 'PARTITION_OF_EMPTY_TEAM_GUARD_ENVELOPE' if team else \
        'MAX_WORKER_LOCAL_WALL_STAGE_WITH_OWN_IMPLICIT_COMPLETION_BARRIER'
    _subset(measurement,dict(component=component,parameters=parameters,median_ns=statistics.median(measurement['samples_ns']),
        unit='ns',basis='DIRECT_COMPONENT_TIMING',scope=scope,applicable_to_component=True,repetitions=3,warmup=1,
        caller_affinity_cpu_ids=pack['caller_affinity_cpu_ids'],parallel_cpu_ids=[] if serial else cpus,
        whole_method_boundary_measured=False,target_program_executed=False,includes_stage_completion_barrier=not serial and not team,
        provider_stage_serial=component=='NUMPY_POWER2_NATIVE_ROW_STAGE'),'canonical resource')
    for key in ('caller_cpu_before','caller_cpu_after'):
        _need(type(measurement[key])is int and measurement[key]in pack['caller_affinity_cpu_ids'],'observed caller CPU outside original mask')
    _need(type(measurement['checksum'])is int and 0<=measurement['checksum']<2**64,'native checksum coordinate invalid')
    if team:
        paired=measurement['paired_complement_samples_ns']; _need(type(paired)is list and len(paired)==3
            and all(type(x)in(int,float)and math.isfinite(x)and x>=0 for x in paired),'paired empty-team envelope missing')
        _subset(measurement,dict(partition_scope='SETUP_TO_READY_CLOCK_CUT_PLUS_RESTORATION_TEAM_EXIT_CALLER_RESTORE',
            includes_instrumentation_rendezvous=len(cpus)>1,serial_guard_uses_only_caller_scope=len(cpus)==1,
            pairing='SUM_OF_PAIRED_SEGMENTS_IS_ONE_EMPTY_ENVELOPE_NOT_TARGET_TOTAL'),'team timing partition')


def _boundary_measurement(measurement,term,cpus,affinity):
    if term['component']==B._BOUNDARY[0]:
        _subset(measurement,dict(scope='SHARED_GUARDED_METHOD_TO_NATIVE_NOOP_RETURN',
            runtime_stats=dict(native_calls=4,fallback_calls=0,reasons={}),
            callable_dispatch='POSITIONAL_PARAMETERS_NO_DEFAULTS_CANONICAL_SOURCE',clock_overhead_retained=True,
            target_program_executed=False,whole_method_boundary_measured=False),'shared live guard/FFI')
    else:
        _subset(measurement,dict(scope='MAX_WORKER_LOCAL_SINGLE_STAGE_WITH_OWN_IMPLICIT_BARRIER',persistent_team=True,
            parallel_cpu_ids=cpus,includes_stage_completion_barrier=True,setup_and_team_cost_included=False,
            separate_barrier_cost_added=False,instrumentation_rendezvous_outside_timing=True,
            canonical_operands=dict(initial=1.25,right=0.75,integer_tags=False),
            operation_topology='CANONICAL_LEFT_DEPENDENT_CHAIN_FROM_OPERATION_LIST',generic_operation_dispatch_overhead_retained=True,
            target_program_executed=False,whole_method_boundary_measured=False),'canonical scalar stage')
        workers=measurement['worker_placement']; _need(len(workers)==len(cpus),'scalar worker coverage differs')
        for cpu,row in zip(cpus,workers):
            _subset(row,dict(planned_cpu=cpu,bound_cpu=cpu,original_affinity_restored=True),'actual scalar worker binding/restoration')
            _need(row['entry_cpu']in affinity and row['restored_cpu']in affinity,'scalar worker CPU outside original caller mask')
        _need(measurement['caller_cpu_before']in affinity and measurement['caller_cpu_after']in affinity,'scalar caller CPU outside original mask')


def verify_aux_sample(pack, candidate_context, sample_dir):
    """Return checked original FactRefs without running or resampling anything."""
    from .cpu_resource_probe import _COMPONENTS
    metadata=B._load_pack(pack)
    context=read_artifact(candidate_context) if isinstance(candidate_context,dict)and'uri'in candidate_context else (
        json.loads(Path(candidate_context).read_bytes()) if isinstance(candidate_context,(str,Path)) else candidate_context)
    B._context(metadata,context); _render(metadata)
    folder=Path(sample_dir).resolve(); result=json.loads((folder/'result.json').read_bytes())
    _need((result.get('schema'),result.get('schema_version'))==('scar.cpu-batch-aux-sample-result',1),'original aux result required')
    refs={key:result[key]for key in ('resource_calibration','boundary_calibration','runtime_witness')}
    names=dict(resource_calibration='resource-calibration.json',boundary_calibration='boundary-calibration.json',runtime_witness='runtime-witness.json')
    documents={}
    for key,ref in refs.items():
        _equal(Path(ref['uri']).resolve(),folder/names[key],'original per-mask output path differs')
        documents[key]=read_artifact(ref)
    cpus=context['binding']['cpu_ids']; affinity=metadata['caller_affinity_cpu_ids']
    contract,structure,ids=_contract(metadata,context,folder)
    witness=documents['runtime_witness']
    _subset(witness,dict(schema='scar.cpu-batch-aux-runtime-witness',schema_version=1,candidate_id=context['candidate_id'],
        mode=metadata['mode'],cpu_ids=cpus,actual_caller_affinity_before=affinity,actual_caller_affinity_after=affinity,
        libraries=metadata['libraries'],table_ids=ids,table_setter_verified_before_sampling=True,
        setters_outside_all_timed_intervals=True,native_recompiled=False,original_stage_bodies_reused=True,
        max_rss_bytes=B._LIMIT,target_workload_executed=False),'actual ABI/runtime witness')
    expected_openmp=None if metadata['openmp']is None else {key:metadata['openmp'][key]for key in ('compiler_selected','actual_loader')}
    _equal(witness['openmp'],expected_openmp,'actual loaded OpenMP versus compiler-library witness differs')
    if expected_openmp is not None:
        for receipt in expected_openmp.values():B._check(receipt)
    _need(type(witness['process_id'])is int and witness['process_id']>0
        and type(witness['peak_rss_bytes'])is int and 0<witness['peak_rss_bytes']<B._LIMIT,'one bounded original process required')
    unique={}
    for term in context['demand']['terms']:
        unique.setdefault(term['key'],term)
    resource_terms=[t for t in unique.values()if t['component']not in B._BOUNDARY and t['component']in _COMPONENTS]
    unsupported=[t for t in unique.values()if t['component']not in B._BOUNDARY and t['component']not in _COMPONENTS]
    boundary_terms=[t for t in unique.values()if t['component']in B._BOUNDARY]
    sources=[]
    expected_probe=dict(cpu_ids=cpus,caller_affinity_cpu_ids=affinity,compiler=metadata['compiler'],
        provider_runtime=metadata['provider_runtime'],native_thread_environment=metadata['native_thread_environment'])
    for name,terms,basis in (('resource_calibration',resource_terms,'DIRECT_COMPONENT_TIMING'),
            ('boundary_calibration',boundary_terms,'DECLARED_COMPONENT_MODEL')):
        doc=documents[name]
        _subset(doc,dict(schema='scar.stateful-resource-calibration',schema_version=1,resource_context=context['resource_context'],
            probe_context=expected_probe,repetitions=3,warmup=1,observed_caller_affinity_before=affinity,
            observed_caller_affinity_after=affinity,observed_native_thread_environment=metadata['native_thread_environment'],
            caller_affinity_restored=True,numa_page_residency_proven=False,implicit_stage_barrier_cost_already_included=True,
            no_scope_median_subtraction=True,whole_method_boundary_measured=False,target_workload_executed=False,
            target_variants_compiled=False,target_variants_ranked=False,physical_optimality_proven=False,
            process_id=witness['process_id'],one_candidate_context_per_process=True,native_recompiled=False),'exact-mask resource source')
        _need(type(doc['peak_rss_bytes'])is int and 0<doc['peak_rss_bytes']<=witness['peak_rss_bytes'],
            'component RSS snapshot exceeds the later runtime witness')
        _need(len(doc['components'])==len(terms),'missing/duplicate component signature')
        for index,(row,term)in enumerate(zip(doc['components'],terms)):
            _samples(row,term,basis)
            if name=='resource_calibration':_resource_measurement(row['measurement'],term,metadata,cpus)
            else:_boundary_measurement(row['measurement'],term,cpus,affinity)
            sources.append(dict(component=row['component'],parameters=row['parameters'],key=term['key'],
                fact=dict(artifact=refs[name],locator='#/components/'+str(index))))
    resource_doc=documents['resource_calibration']; boundary_doc=documents['boundary_calibration']
    _equal([(r['component'],r['parameters'])for r in resource_doc['unsupported_components']],
        [(t['component'],t['parameters'])for t in unsupported],'unsupported primitive signatures differ')
    _equal(boundary_doc['unsupported_components'],[],'unexpected boundary frontier')
    _equal(boundary_doc['canonical_structure'],structure,'canonical input/reference path-sharing structure differs')
    _equal(boundary_doc['canonical_runtime_contract'],B._receipt(folder/'canonical-contract.json'),'canonical contract byte receipt differs')
    _equal(boundary_doc['extra_observations'],[],'unexpected extra boundary observation')
    extras=resource_doc['extra_observations']; _need(len(extras)==1,'one diagnostic-only barrier observation required')
    _subset(extras[0],dict(component='OMP_BARRIER_ONLY',used_in_composition=False),'barrier must not be composed twice')
    _resource_measurement(extras[0]['measurement'],dict(component='OMP_BARRIER_ONLY',parameters=dict(workers=len(cpus))),metadata,cpus)
    for source in (metadata['source_region'],metadata['base_stage_plan']):read_artifact(source,json_document=False)
    B._load_pack(pack);B._context(metadata,context)
    return dict(**refs,components=sources,runtime=witness,
        verifier_implementation=B._reference(__file__,'scar.validation-raw'),
        source_render_replayed=True,canonical_contract_reconstructed=True,
        raw_samples_statistics_recomputed=True,target_workload_executed=False,
        native_probe_executed=False,whole_method_boundary_measured=False)
