"""Operation-scoped strict replay for shared CPU calibration packs.

Only compiler/MCA work is shared. Every context replays its own raw samples,
coefficients, independent holdouts, resource binding and original source DAG.
The explicit session is bounded to immutable source/material receipts.
"""
from copy import deepcopy
import inspect
import json
from pathlib import Path
import subprocess
import tempfile

from . import cpu_batch_instruction_cost as B
from . import cpu_instruction_cost as S
from . import cpu_parallel_instruction_cost as P
from .cpu_instruction_cost_replay import _raw,_normalized_assembly,_replay_compiler
from .cpu_fused_cost import _HERE,_receipt,_require,_hash,_json
from scar.ir.execution_recipe import artifact_reference,read_artifact


class BatchReplaySession:
    """One full domain admission; repeat byte/source guards, not normalization."""
    def __init__(self,domain):
        from scar.optimization import stateful_cpu_choices as choices
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
    from scar.optimization.stateful_cpu_model import stateful_cpu_demand,component_key
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
