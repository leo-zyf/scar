"""Cpu instruction replay."""

from copy import deepcopy
import json
import re
import subprocess
import tempfile
import inspect
from pathlib import Path
from scar.codegen import (
    cpu_instruction_cost as C,
    cpu_parallel_instruction_cost as P,
    cpu_batch_instruction_cost as B,
    cpu_instruction_cost as S,
)
from scar.codegen.cpu_fused_cost import _HERE, _receipt, _require, _hash, _json
from scar.ir.execution_recipe import artifact_reference, read_artifact


# Read and replay an instruction-cost projection without target execution.
#
# This adapter deliberately lives outside the frozen calibration producer. It
# replays compiler source, natural-loop selection, LLVM-MCA outputs and numeric
# composition. Original physical timing samples remain retained observations;
# this replay never remeasures them and makes no stronger timing claim.
def _raw(reference):
    _require(_receipt(Path(reference['path']))==reference,'instruction receipt bytes/hash differ')
    return Path(reference['path']).read_text()


def _normalized_assembly(text):
    return '\n'.join(line.split('#',1)[0].strip() for line in text.splitlines()
        if line.split('#',1)[0].strip() and not line.lstrip().startswith('.file'))


def _replay_compiler(lowering):
    compiler=lowering['compiler'];source=_raw(lowering['source']);assembly=_raw(lowering['assembly'])
    identity=subprocess.run([*compiler['argv'],'--version'],check=True,capture_output=True,text=True).stdout.splitlines()[0]
    _require(identity==compiler['identity'],'current GCC identity differs')
    for name in ('stateful_abi_v1.h','scar_abi_v1.h'):
        path=Path(lowering['source']['path']).parent/name
        _require(path.read_bytes()==(_HERE/name).read_bytes(),'static lowering ABI header differs')
    command=lowering['command']
    expected=[*compiler['argv'],*compiler['options'],'-S','-fverbose-asm',
        '-fopt-info-vec-optimized='+lowering['vectorization']['path'],lowering['source']['path'],
        '-o',lowering['assembly']['path']]
    _require(command==expected,'original static compiler invocation differs')
    _raw(lowering['vectorization'])
    with tempfile.TemporaryDirectory(prefix='scar-static-cost-replay-') as directory:
        target=Path(directory)/'replayed.s';remarks=Path(directory)/'vectorization.txt'
        replay=[*compiler['argv'],*compiler['options'],'-S','-fverbose-asm',
            '-fopt-info-vec-optimized='+str(remarks),lowering['source']['path'],'-o',str(target)]
        p=subprocess.run(replay,capture_output=True,text=True)
        _require(p.returncode==0,'static compiler replay rejected: '+p.stderr[-500:])
        _require(_normalized_assembly(target.read_text())==_normalized_assembly(assembly),
            'static backend assembly differs from current GCC replay')
    return source,assembly


def _replay_analysis(assembly,analysis,tool,*,function=None,source_range=None,source_path=None,cache=None):
    expected=C._natural_loops(assembly,function=function,source_range=source_range,source_path=source_path)
    _require(len(expected)==len(analysis['loops']),'selected natural instruction-loop count differs')
    results=[];cache={} if cache is None else cache
    for original,recorded in zip(expected,analysis['loops']):
        text=_raw(recorded['loop_source']);raw=_raw(recorded['result'])
        # Comparing to regenerated assembly matters even if both input and its
        # newly computed hash were replaced together.
        _require(text==original['text'],'MCA instruction input differs from original compiler natural loop')
        command=[tool['executable']['path'],'-mcpu='+tool['cpu_model'],
            '-iterations='+str(tool['iterations']),'--noalias=true','--json',recorded['loop_source']['path']]
        _require(recorded['command']==command,'MCA CPU/model/alias command differs')
        key=(tuple(command),recorded['loop_source']['sha256'],recorded['result']['sha256'])
        if key not in cache:
            p=subprocess.run(command,capture_output=True,text=True)
            _require(p.returncode==0,'current LLVM-MCA replay rejected: '+p.stderr[-500:])
            actual=json.loads(p.stdout);retained=json.loads(raw)
            _require(actual==retained,'MCA output differs from exact current tool replay')
            cache[key]=actual
        result=cache[key];region=result['CodeRegions'][0];summary=region['SummaryView']
        _require(result['TargetInfo']['CPUName']==tool['cpu_model']
            and not any(i['hasUnmodeledSideEffects'] for i in region['InstructionInfoView']['InstructionList']),
            'unsupported MCA CPU or instruction effects')
        expected_record=dict(**{k:v for k,v in original.items() if k!='text'},
            loop_source=recorded['loop_source'],result=recorded['result'],command=command,
            block_throughput_cycles=summary['BlockRThroughput'],
            simulated_cycles_per_iteration=summary['TotalCycles']/summary['Iterations'],
            cycles_per_element=summary['BlockRThroughput']/original['width'],
            resource_pressure=region['ResourcePressureView'],runtime_path_observed=False)
        _require(recorded==expected_record,'retained instruction analysis differs from mechanical replay')
        results.append(recorded)
    vectors=[r for r in results if r['width']==2];scalars=[r for r in results if r['width']==1]
    value=dict(loops=results,vector=max(vectors,key=lambda r:r['cycles_per_element']) if vectors else None,
        scalar=max(scalars,key=lambda r:r['cycles_per_element']) if scalars else None,
        selection='MAX_STATIC_VECTOR_PATH_WHEN_NONALIAS_WORKSPACE_GUARD_HOLDS')
    _require(value==analysis,'instruction path selection differs')
    return value


def verify_instruction_projection(domain,candidate_id,projection):
    """Return exact admitted component rows after static source/tool replay."""
    from scar.optimization.stateful_cpu_model import _domain
    from scar.optimization.stateful_cpu_model import stateful_cpu_demand, component_key
    from scar.compiler.stateful_model import stateful_from_dict,stateful_digest
    from scar.compiler.model import CpuPlacement
    from scar.optimization.stateful_schedule import plan_stateful_region
    from scar.codegen import stateful

    domain=_domain(domain);demand=stateful_cpu_demand(domain,candidate_id)
    reference=artifact_reference(projection,'scar.stateful-resource-calibration');document=read_artifact(reference)
    _require(document['schema_version']==1 and document['resource_context']==demand['resource_context']
        and document.get('target_workload_executed') is False and document.get('target_variants_ranked') is False
        and document.get('target_dag_compiled_for_static_analysis') is True,'projection context/authority differs')
    model_ref=document['source_model'];read_artifact(model_ref)
    checked_ref,model,raw,calibrated=C._verify_model(model_ref['uri'])
    _require(checked_ref==model_ref and model['resource_context']==demand['resource_context'],'canonical model/context differs')
    tool=model['tool'];cache={};assembly=_raw(model['native']['assembly'])
    for key,analysis in calibrated.items():
        case=C._registry()[int(key)//2];group=int(key)%2+1
        _replay_analysis(assembly,analysis,tool,function=case['id']+'_g'+str(group),cache=cache)

    lowering_ref=document['static_lowering'];lowering=read_artifact(lowering_ref)
    _require(lowering['schema']=='scar.stateful-static-instruction-lowering' and lowering['schema_version']==1
        and lowering['target_dag_compiled_for_static_analysis'] is True
        and lowering['target_workload_executed'] is False and lowering['target_variants_ranked'] is False
        and lowering['native_library_built'] is False,'static lowering scope differs')
    _require(lowering['region']==domain['region'] and lowering['compiler']==demand['resource_context']['backend']['compiler']
        and lowering['backend_implementation']==stateful.implementation_sha256(),'static source/backend dependency differs')
    region=stateful_from_dict(read_artifact(domain['region']))
    candidate=next(r for r in domain['candidates'] if r['id']==candidate_id)
    binding=dict(candidate['binding']);binding['cpu_ids']=tuple(binding['cpu_ids']);binding['evidence']=tuple(binding['evidence'])
    placement=CpuPlacement(**binding)
    plan=plan_stateful_region(region,input_shapes=domain['request']['input_shapes'],cpu_threads=len(placement.cpu_ids),
        memory_budget_bytes=placement.memory_budget_bytes,resource_binding=placement)
    region,plan,placement=stateful._admit(region,plan,placement)
    _require(plan==lowering['plan'] and lowering['resource_binding']==plan['resource_binding'],'static stage plan or placement differs')
    contract=dict(region_sha256=stateful_digest(region),plan_sha256=stateful._digest(plan),
        provider_runtime=demand['resource_context']['backend']['provider_runtime'])
    source,assembly=_replay_compiler(lowering)
    _require(source==stateful._source(region,plan,placement,contract),'original backend source differs from source/plan replay')
    order={n.id:i for i,n in enumerate(region.nodes)};terms=[r for r in demand['terms'] if r['component']=='FUSED_SCALAR_DAG_ROW_STAGE']
    _require(len(document['components'])==len(terms) and not document['unsupported_components']
        and not document['proposed_estimates'] and document['all_fused_components_covered'] is True,
        'complete admitted fused projection required')
    maximum_stack=max(r['stack_memory_references'] for a in calibrated.values() for r in a['loops'])
    checked=[]
    for index,(row,term) in enumerate(zip(document['components'],terms)):
        p=term['parameters'];stage_index=int(term['source_ref']['pointer'].rsplit('/',1)[1]);stage=plan['stages'][stage_index]
        _require(stage['kind']=='COMPUTE' and p['workers']==1,'unsupported stage boundary')
        output=plan['assignment'][stage['output_payload']]['slot_id']
        inputs=[plan['assignment'][k]['slot_id'] for k in stage['input_payloads'] if k in plan['assignment']]
        _require(output not in inputs,'static output allocation aliases live input')
        expected=C.V2._layouts(p['array_loads'],p['cols'],p['backing_storage_groups']);sort=lambda r:(r['storage_group'],r['stride'],r['offset'])
        _require(sorted(p['operand_layouts'],key=sort)==sorted(expected,key=sort),'memory proxy geometry differs')
        capacity=model['cache_geometry']['reported_capacity_bytes'];_,geometry=C.V2.fused_features(p,capacity)
        _require(256<=p['rows']<=768 and 320<=p['cols']<=832
            and geometry['backing_bytes']+p['rows']*p['cols']*8<=capacity,'memory proxy applicability differs')
        span=C._stage_source_range(source,order[stage['root_expr']])
        analysis=_replay_analysis(assembly,row['instruction_analysis'],tool,source_range=span,
            source_path=lowering['source']['path'],cache=cache)
        _require(max(r['stack_memory_references'] for r in analysis['loops'])<=maximum_stack,'spill scope differs')
        qualified=model['local_models'][str(p['backing_storage_groups'])]
        _require(qualified['admitted'] is True,'fresh holdout does not authorize this component')
        cost=C._predict(dict(groups=p['backing_storage_groups'],rows=p['rows'],cols=p['cols'],workers=p['workers'],
            elements=((p['rows']+p['workers']-1)//p['workers'])*p['cols'],array_loads=p['array_loads']),analysis,model['coefficients'])
        expected_row=dict(component=term['component'],parameters=p,basis='DECLARED_COMPONENT_MODEL',median_ns=cost['duration_ns'],
            prediction=cost,source_model=dict(artifact=model_ref,locator='#/coefficients'),
            source_validation=dict(artifact=model_ref,locator='#/local_models/'+str(p['backing_storage_groups'])+'/holdout'),
            source_provenance=term['source_ref'],instruction_analysis=analysis,static_lowering=lowering_ref,
            workspace_alias_authority=dict(output_slot=output,input_slots=inputs,live_slots_distinct=True),
            precision_policy=C._POLICY,maximum_holdout_relative_error=qualified['maximum_holdout_relative_error'],
            target_dag_compiled_for_static_analysis=True,measured_target_component=False,
            physical_optimality_proven=False,assumptions=C._ASSUMPTIONS)
        _require(row==expected_row,'projected component estimate, parameters or dependencies differ')
        checked.append(dict(index=index,key=component_key(row['component'],row['parameters']),
            fact=dict(artifact=reference,locator=f'#/components/{index}'),component=deepcopy(row)))
    return dict(schema='scar.cpu-instruction-cost-projection-replay',schema_version=1,
        projection=reference,source_profile=model_ref,static_lowering=lowering_ref,
        frozen_design=model['frozen_design'],raw_observations=model['raw_observations'],instruction_analysis=model['instruction_analysis'],
        tool=tool['executable'],components=checked,compiler_reexecuted_static_only=True,mca_reexecuted_static_only=True,
        timing_observations_remeasured=False,target_workload_executed=False,target_variants_ranked=False,
        target_dag_compiled_for_static_analysis=True,model_authority='DECLARED_FORWARD_COMPONENT_COST_NOT_PHYSICAL_BOUND')


# Independent static replay of exact-mask parallel instruction costs.
#
# The frozen serial replay supplies only generic compiler/MCA receipt checking;
# this reader uses its own parallel model schema, registry, barrier composition
# and exact worker mask. No target or calibration timing is rerun.
def _replay_parallel_analysis(assembly,analysis,tool,**kwargs):
    _require(analysis['instruction_lexer']==P._lexer_evidence(assembly),
        'instruction lexical source/rule/implementation differs')
    base={k:v for k,v in analysis.items() if k!='instruction_lexer'}
    result=_replay_analysis(P._lex_assembly(assembly),base,tool,**kwargs)
    _require(result==base,'normalized natural instruction-loop replay differs')
    return analysis


def verify_parallel_instruction_projection(domain,candidate_id,projection):
    """Return exact admitted component rows after static source/tool replay."""
    from scar.optimization.stateful_cpu_model import _domain
    from scar.optimization.stateful_cpu_model import stateful_cpu_demand, component_key
    from scar.compiler.stateful_model import stateful_from_dict,stateful_digest
    from scar.compiler.model import CpuPlacement
    from scar.optimization.stateful_schedule import plan_stateful_region
    from scar.codegen import stateful

    domain=_domain(domain);demand=stateful_cpu_demand(domain,candidate_id)
    reference=artifact_reference(projection,'scar.stateful-resource-calibration');document=read_artifact(reference)
    _require(document['schema_version']==1 and document['resource_context']==demand['resource_context']
        and document.get('target_workload_executed') is False and document.get('target_variants_ranked') is False
        and document.get('target_dag_compiled_for_static_analysis') is True,'projection context/authority differs')
    model_ref=document['source_model'];read_artifact(model_ref)
    checked_ref,model,raw,calibrated=P._verify_model(model_ref['uri'])
    _require(checked_ref==model_ref and model['resource_context']==demand['resource_context'],'canonical model/context differs')
    tool=model['tool'];cache={};assembly=_raw(model['native']['assembly'])
    for key,analysis in calibrated.items():
        case=P._registry()[int(key)//2];group=int(key)%2+1
        _replay_parallel_analysis(assembly,analysis,tool,function=case['id']+'_g'+str(group),cache=cache)

    lowering_ref=document['static_lowering'];lowering=read_artifact(lowering_ref)
    _require(lowering['schema']=='scar.stateful-static-instruction-lowering' and lowering['schema_version']==1
        and lowering['target_dag_compiled_for_static_analysis'] is True
        and lowering['target_workload_executed'] is False and lowering['target_variants_ranked'] is False
        and lowering['native_library_built'] is False,'static lowering scope differs')
    _require(lowering['region']==domain['region'] and lowering['compiler']==demand['resource_context']['backend']['compiler']
        and lowering['backend_implementation']==stateful.implementation_sha256(),'static source/backend dependency differs')
    region=stateful_from_dict(read_artifact(domain['region']))
    candidate=next(r for r in domain['candidates'] if r['id']==candidate_id)
    binding=dict(candidate['binding']);binding['cpu_ids']=tuple(binding['cpu_ids']);binding['evidence']=tuple(binding['evidence'])
    placement=CpuPlacement(**binding)
    plan=plan_stateful_region(region,input_shapes=domain['request']['input_shapes'],cpu_threads=len(placement.cpu_ids),
        memory_budget_bytes=placement.memory_budget_bytes,resource_binding=placement)
    region,plan,placement=stateful._admit(region,plan,placement)
    _require(plan==lowering['plan'] and lowering['resource_binding']==plan['resource_binding'],'static stage plan or placement differs')
    contract=dict(region_sha256=stateful_digest(region),plan_sha256=stateful._digest(plan),
        provider_runtime=demand['resource_context']['backend']['provider_runtime'])
    source,assembly=_replay_compiler(lowering)
    _require(source==stateful._source(region,plan,placement,contract),'original backend source differs from source/plan replay')
    order={n.id:i for i,n in enumerate(region.nodes)};terms=[r for r in demand['terms'] if r['component']=='FUSED_SCALAR_DAG_ROW_STAGE']
    _require(len(document['components'])==len(terms) and not document['unsupported_components']
        and not document['proposed_estimates'] and document['all_fused_components_covered'] is True,
        'complete admitted fused projection required')
    maximum_stack=max(r['stack_memory_references'] for a in calibrated.values() for r in a['loops'])
    checked=[]
    for index,(row,term) in enumerate(zip(document['components'],terms)):
        p=term['parameters'];stage_index=int(term['source_ref']['pointer'].rsplit('/',1)[1]);stage=plan['stages'][stage_index]
        _require(stage['kind']=='COMPUTE' and p['workers']>1,'unsupported stage boundary')
        output=plan['assignment'][stage['output_payload']]['slot_id']
        inputs=[plan['assignment'][k]['slot_id'] for k in stage['input_payloads'] if k in plan['assignment']]
        _require(output not in inputs,'static output allocation aliases live input')
        expected=C.V2._layouts(p['array_loads'],p['cols'],p['backing_storage_groups']);sort=lambda r:(r['storage_group'],r['stride'],r['offset'])
        _require(sorted(p['operand_layouts'],key=sort)==sorted(expected,key=sort),'memory proxy geometry differs')
        capacity=model['cache_geometry']['reported_capacity_bytes'];_,geometry=C.V2.fused_features(p,capacity)
        _require(256<=p['rows']<=768 and 320<=p['cols']<=832
            and geometry['backing_bytes']+p['rows']*p['cols']*8<=capacity,'memory proxy applicability differs')
        span=C._stage_source_range(source,order[stage['root_expr']])
        analysis=_replay_parallel_analysis(assembly,row['instruction_analysis'],tool,source_range=span,
            source_path=lowering['source']['path'],cache=cache)
        _require(max(r['stack_memory_references'] for r in analysis['loops'])<=maximum_stack,'spill scope differs')
        qualified=model['local_models'][str(p['backing_storage_groups'])]
        _require(qualified['admitted'] is True,'fresh holdout does not authorize this component')
        cost=P._predict(dict(groups=p['backing_storage_groups'],rows=p['rows'],cols=p['cols'],workers=p['workers'],
            elements=((p['rows']+p['workers']-1)//p['workers'])*p['cols'],array_loads=p['array_loads']),analysis,model['coefficients'])
        expected_row=dict(component=term['component'],parameters=p,basis='DECLARED_COMPONENT_MODEL',median_ns=cost['duration_ns'],
            prediction=cost,source_model=dict(artifact=model_ref,locator='#/coefficients'),
            source_validation=dict(artifact=model_ref,locator='#/local_models/'+str(p['backing_storage_groups'])+'/holdout'),
            source_provenance=term['source_ref'],instruction_analysis=analysis,static_lowering=lowering_ref,
            workspace_alias_authority=dict(output_slot=output,input_slots=inputs,live_slots_distinct=True),
            precision_policy=P._POLICY,maximum_holdout_relative_error=qualified['maximum_holdout_relative_error'],
            target_dag_compiled_for_static_analysis=True,measured_target_component=False,
            physical_optimality_proven=False,assumptions=P._ASSUMPTIONS)
        _require(row==expected_row,'projected component estimate, parameters or dependencies differ')
        checked.append(dict(index=index,key=component_key(row['component'],row['parameters']),
            fact=dict(artifact=reference,locator=f'#/components/{index}'),component=deepcopy(row)))
    return dict(schema='scar.cpu-parallel-instruction-cost-projection-replay',schema_version=1,
        projection=reference,source_profile=model_ref,static_lowering=lowering_ref,
        frozen_design=model['frozen_design'],raw_observations=model['raw_observations'],instruction_analysis=model['instruction_analysis'],
        tool=tool['executable'],openmp_runtime=tool['openmp_runtime'],
        compiler_openmp_runtime=tool['compiler_openmp_runtime'],partition_witnesses=model['partition_witnesses'],
        qualification_design=model.get('qualification_design'),components=checked,
        compiler_reexecuted_static_only=True,mca_reexecuted_static_only=True,
        calibration_library_loaded_for_runtime_symbol_identity_only=True,
        timing_observations_remeasured=False,target_workload_executed=False,target_variants_ranked=False,
        target_dag_compiled_for_static_analysis=True,model_authority='DECLARED_FORWARD_COMPONENT_COST_NOT_PHYSICAL_BOUND')


# Operation-scoped strict replay for shared CPU calibration packs.
#
# Only compiler/MCA work is shared. Every context replays its own raw samples,
# coefficients, independent holdouts, resource binding and original source DAG.
# The explicit session is bounded to immutable source/material receipts.
class BatchReplaySession:
    """One full domain admission; repeat byte/source guards, not normalization."""
    def __init__(self,domain):
        from scar.optimization import stateful_cpu_model as choices
        from scar.codegen import stateful
        self.domain=choices._domain(domain)
        self.domain_reference=artifact_reference(Path(self.domain['base_stage_plan']['uri']).parent/'domain.json',self.domain['schema'])
        self.pins={};self.packs={};self.blocks={};self.backend_implementation=stateful.implementation_sha256()
        paths={Path(self.domain_reference['uri']),Path(self.domain['region']['uri']),Path(self.domain['base_stage_plan']['uri'])}
        hardware=Path(self.domain['target_snapshot']['uri']).parent
        paths.update(path for path in hardware.rglob('*') if path.is_file())
        from scar.compiler.stateful_model import stateful_from_dict
        region=stateful_from_dict(read_artifact(self.domain['region']))
        paths.update(Path(s.path) for s in (region.source,*(c.source for c in region.callees)))
        paths.add(Path(__file__).resolve());paths.add(Path(B.__file__).resolve())
        for value in vars(choices).values():
            if (inspect.isfunction(value) or inspect.isclass(value)) and getattr(value,'__module__','').startswith('scar.'):
                path=inspect.getsourcefile(value)
                if path:paths.add(Path(path))
        for path in paths:self.pins[str(path.resolve())]=_receipt(path.resolve())
        self.check()

    def check(self):
        from scar.codegen import stateful
        for reference in self.pins.values():
            _require(_receipt(Path(reference['path']))==reference,'batch session source/material changed: '+reference['path'])
        _require(stateful.implementation_sha256()==self.backend_implementation,'original backend implementation changed during replay')
        read_artifact(self.domain_reference)
        return True

    def admitted(self,value):
        self.check()
        if isinstance(value,(str,Path)):
            path=Path(value);path=path/'domain.json' if path.is_dir() else path
            ref=artifact_reference(path,self.domain['schema']);current=read_artifact(ref)
        else:current=value
        _require(current==self.domain,'requested source domain differs from explicit replay session')
        return self.domain

    def block(self,reference,original,tool):
        record=read_artifact(reference);normalized=B._normalized_block(original)
        identity=dict(instruction_text=normalized,tool=B._instruction_tool(tool))
        _require(record['schema']=='scar.mca-shared-instruction-block' and record['schema_version']==1
            and record['identity']==identity and record['key']==_hash(_json(identity))
            and record['timing_fact'] is False and record['registers_and_memory_operands_preserved'] is True,
            'shared block instruction/tool/alias identity differs')
        _require(_raw(record['input'])==normalized,'shared normalized instruction input differs')
        retained=json.loads(_raw(record['result']))
        expected=[tool['executable']['path'],'-mcpu='+tool['cpu_model'],'-iterations='+str(tool['iterations']),
            '--noalias=true','--json',record['input']['path']]
        _require(record['command']==expected,'shared MCA command differs')
        key=reference['sha256']
        if key not in self.blocks:
            p=subprocess.run(expected,capture_output=True,text=True)
            _require(p.returncode==0 and json.loads(p.stdout)==retained,'shared MCA output differs from current tool replay')
            self.blocks[key]=True
        summary=retained['CodeRegions'][0]['SummaryView']
        _require(retained['TargetInfo']['CPUName']==tool['cpu_model'] and summary==record['summary']
            and not any(r['hasUnmodeledSideEffects'] for r in retained['CodeRegions'][0]['InstructionInfoView']['InstructionList']),
            'shared instruction model summary/effect differs')
        return summary

    def analysis(self,assembly,analysis,tool,**kwargs):
        loops=S._natural_loops(P._lex_assembly(assembly),**kwargs)
        _require(len(loops)==len(analysis['loops']),'actual natural instruction-loop set differs')
        rows=[]
        for original,retained in zip(loops,analysis['loops']):
            summary=self.block(retained['block'],original['text'],tool)
            expected=dict(**{k:v for k,v in original.items() if k!='text'},block=retained['block'],
                original_instruction_sha256=_hash(original['text'].encode()),
                block_throughput_cycles=summary['BlockRThroughput'],cycles_per_element=summary['BlockRThroughput']/original['width'],
                runtime_path_observed=False)
            _require(expected==retained,'actual loop/source range/instruction key differs')
            rows.append(retained)
        vectors=[r for r in rows if r['width']==2];scalars=[r for r in rows if r['width']==1]
        _require(analysis['vector']==(max(vectors,key=lambda r:r['cycles_per_element']) if vectors else None)
            and analysis['scalar']==(max(scalars,key=lambda r:r['cycles_per_element']) if scalars else None)
            and analysis['original_assembly_sha256']==_hash(assembly.encode())
            and analysis['lexer_rule']=='INSTRUCTION_OPERANDS_EXCLUDE_TRAILING_COMMENTS_SOURCE_MARKS_PRESERVED'
            and type(analysis['static_cache_hits']) is int and 0<=analysis['static_cache_hits']<=len(rows),
            'static path selection or lexer provenance differs')
        return analysis

    def pack(self,reference):
        reference,pack=B._read_pack(reference);key=reference['sha256']
        if key in self.packs:return self.packs[key]
        compiler=pack['compiler'];source=_raw(pack['source']);assembly=_raw(pack['assembly'])
        _require(source==B._source(pack['available_cpu_ids']),'batch canonical source differs')
        identity=subprocess.run([*compiler['argv'],'--version'],check=True,capture_output=True,text=True).stdout.splitlines()[0]
        _require(identity==compiler['identity'],'canonical GCC version differs')
        expected=[*compiler['argv'],*compiler['options'],'-S','-fverbose-asm',
            '-fopt-info-vec-optimized='+pack['vectorization']['path'],pack['source']['path'],'-o',pack['assembly']['path']]
        _require(pack['commands'][0]==expected and pack['commands'][1]==[
            *compiler['argv'],*compiler['options'],pack['assembly']['path'],'-o',pack['library']['path']],
            'shared compiler mode/flags differ')
        _require((Path(pack['source']['path']).parent/'scar_abi_v1.h').read_bytes()==(_HERE/'scar_abi_v1.h').read_bytes(),
            'shared probe ABI header differs')
        with tempfile.TemporaryDirectory(prefix='scar-batch-pack-replay-') as directory:
            output=Path(directory)/'canonical.s';remarks=Path(directory)/'remarks.txt'
            command=[*compiler['argv'],*compiler['options'],'-S','-fverbose-asm',
                '-fopt-info-vec-optimized='+str(remarks),pack['source']['path'],'-o',str(output)]
            p=subprocess.run(command,capture_output=True,text=True)
            _require(p.returncode==0 and _normalized_assembly(output.read_text())==_normalized_assembly(assembly),
                'canonical shared assembly differs from current compiler replay')
        analyses=read_artifact(pack['instruction_analysis'])['analyses']
        for key,value in analyses.items():
            case=B._registry()[int(key)//2];group=int(key)%2+1
            self.analysis(assembly,value,pack['tool'],function=case['id']+'_g'+str(group))
        self.packs[reference['sha256']]=pack
        return pack


def verify_batch_projection(domain,candidate_id,projection_path,*,session=None):
    from scar.optimization.stateful_cpu_model import stateful_cpu_demand, component_key
    from scar.compiler.stateful_model import stateful_from_dict,stateful_digest
    from scar.compiler.model import CpuPlacement
    from scar.optimization.stateful_schedule import plan_stateful_region
    from scar.codegen import stateful
    session=session or BatchReplaySession(domain);domain=session.admitted(domain)
    reference=projection_path if isinstance(projection_path,dict) else artifact_reference(projection_path,'scar.stateful-resource-calibration')
    document=read_artifact(reference);demand=stateful_cpu_demand(domain,candidate_id)
    _require(document['schema_version']==1 and document['resource_context']==demand['resource_context']
        and document['target_workload_executed'] is False and document['target_variants_ranked'] is False
        and document['target_dag_compiled_for_static_analysis'] is True,'batch projection context/scope differs')
    model_ref=document['source_model'];read_artifact(model_ref)
    checked,model,pack_ref,pack,calibrated=B.read_context_model(model_ref['uri'])
    _require(checked==model_ref and document['shared_pack']==pack_ref and model['resource_context']==demand['resource_context'],
        'independent per-mask model/pack context differs')
    session.pack(pack_ref);tool=model['tool'];module=B._algorithm(pack['mode'])
    lowering_ref=document['static_lowering'];lowering=read_artifact(lowering_ref)
    _require(lowering['schema']=='scar.stateful-static-instruction-lowering' and lowering['schema_version']==1
        and lowering['region']==domain['region'] and lowering['compiler']==demand['resource_context']['backend']['compiler']
        and lowering['backend_implementation']==stateful.implementation_sha256()
        and lowering['target_workload_executed'] is False and lowering['native_library_built'] is False
        and lowering['target_dag_compiled_for_static_analysis'] is True,'original static backend envelope differs')
    region=stateful_from_dict(read_artifact(domain['region']));candidate=next(r for r in domain['candidates'] if r['id']==candidate_id)
    binding=dict(candidate['binding']);binding['cpu_ids']=tuple(binding['cpu_ids']);binding['evidence']=tuple(binding['evidence'])
    placement=CpuPlacement(**binding);plan=plan_stateful_region(region,input_shapes=domain['request']['input_shapes'],
        cpu_threads=len(placement.cpu_ids),memory_budget_bytes=placement.memory_budget_bytes,resource_binding=placement)
    region,plan,placement=stateful._admit(region,plan,placement)
    _require(plan==lowering['plan'] and plan['resource_binding']==lowering['resource_binding'],'original plan/placement differs')
    source,assembly=_replay_compiler(lowering)
    contract=dict(region_sha256=stateful_digest(region),plan_sha256=stateful._digest(plan),
        provider_runtime=demand['resource_context']['backend']['provider_runtime'])
    _require(source==stateful._source(region,plan,placement,contract),'source/semantic plan static replay differs')
    terms=[r for r in demand['terms'] if r['component']=='FUSED_SCALAR_DAG_ROW_STAGE']
    _require(len(document['components'])==len(terms) and not document['unsupported_components']
        and not document['proposed_estimates'] and document['all_fused_components_covered'] is True,'complete qualified fused projection required')
    order={n.id:i for i,n in enumerate(region.nodes)};maximum_stack=max(r['stack_memory_references'] for a in calibrated.values() for r in a['loops']);checked_rows=[]
    for index,(row,term) in enumerate(zip(document['components'],terms)):
        p=term['parameters'];stage_index=int(term['source_ref']['pointer'].rsplit('/',1)[1]);stage=plan['stages'][stage_index]
        output=plan['assignment'][stage['output_payload']]['slot_id'];inputs=[plan['assignment'][k]['slot_id'] for k in stage['input_payloads'] if k in plan['assignment']]
        _require(stage['kind']=='COMPUTE' and output not in inputs and p['workers']==len(model['resource_context']['worker_cpu_ids']),
            'live allocation alias or worker scope differs')
        expected=S.V2._layouts(p['array_loads'],p['cols'],p['backing_storage_groups']);sort=lambda r:(r['storage_group'],r['stride'],r['offset'])
        _require(sorted(p['operand_layouts'],key=sort)==sorted(expected,key=sort),'shared-root memory geometry differs')
        capacity=model['cache_geometry']['reported_capacity_bytes'];_,geometry=S.V2.fused_features(p,capacity)
        _require(256<=p['rows']<=768 and 320<=p['cols']<=832 and geometry['backing_bytes']+p['rows']*p['cols']*8<=capacity,
            'memory/shape applicability differs')
        span=S._stage_source_range(source,order[stage['root_expr']])
        analysis=session.analysis(assembly,row['instruction_analysis'],tool,source_range=span,source_path=lowering['source']['path'])
        _require(max(r['stack_memory_references'] for r in analysis['loops'])<=maximum_stack,'uncalibrated stack pressure')
        qualified=model['local_models'][str(p['backing_storage_groups'])];_require(qualified['admitted'] is True,'per-mask fresh holdout rejected')
        cost=module._predict(dict(groups=p['backing_storage_groups'],rows=p['rows'],cols=p['cols'],workers=p['workers'],
            elements=((p['rows']+p['workers']-1)//p['workers'])*p['cols'],array_loads=p['array_loads']),analysis,model['coefficients'])
        expected_row=dict(component=term['component'],parameters=p,basis='DECLARED_COMPONENT_MODEL',median_ns=cost['duration_ns'],prediction=cost,
            source_model=dict(artifact=model_ref,locator='#/coefficients'),source_validation=dict(artifact=model_ref,locator='#/local_models/'+str(p['backing_storage_groups'])+'/holdout'),
            source_pack=pack_ref,source_provenance=term['source_ref'],instruction_analysis=analysis,static_lowering=lowering_ref,
            workspace_alias_authority=dict(output_slot=output,input_slots=inputs,live_slots_distinct=True),precision_policy=module._POLICY,
            maximum_holdout_relative_error=qualified['maximum_holdout_relative_error'],target_dag_compiled_for_static_analysis=True,
            measured_target_component=False,physical_optimality_proven=False,assumptions=module._ASSUMPTIONS)
        _require(row==expected_row,'per-mask component estimate or original fact dependencies differ')
        checked_rows.append(dict(index=index,key=component_key(row['component'],row['parameters']),
            fact=dict(artifact=reference,locator=f'#/components/{index}'),component=deepcopy(row)))
    session.check()
    return dict(schema='scar.cpu-batch-instruction-projection-replay',schema_version=1,projection=reference,source_profile=model_ref,
        source_pack=pack_ref,static_lowering=lowering_ref,frozen_design=model['frozen_design'],raw_observations=model['raw_observations'],
        instruction_analysis=pack['instruction_analysis'],tool_contract=tool,components=checked_rows,
        compiler_reexecuted_static_only=True,mca_reexecuted_static_only=True,explicit_operation_scoped_session=True,
        timing_observations_remeasured=False,target_workload_executed=False,target_variants_ranked=False,
        target_dag_compiled_for_static_analysis=True,model_authority='DECLARED_FORWARD_COMPONENT_COST_NOT_PHYSICAL_BOUND')
