"""Cpu instruction replay."""

from copy import deepcopy
import json
import re
import subprocess
import tempfile
import inspect
from pathlib import Path
from scar.codegen import cpu_batch_instruction_cost as B, cpu_instruction_cost as C
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


def _replay_parallel_analysis(assembly,analysis,tool,**kwargs):
    _require(analysis['instruction_lexer']==C._lexer_evidence(assembly),
        'instruction lexical source/rule/implementation differs')
    base={k:v for k,v in analysis.items() if k!='instruction_lexer'}
    result=_replay_analysis(C._lex_assembly(assembly),base,tool,**kwargs)
    _require(result==base,'normalized natural instruction-loop replay differs')
    return analysis


def _replay_target(domain, candidate_id, context, lowering_ref):
    from scar.codegen import stateful

    lowering = read_artifact(lowering_ref)
    _require(lowering['schema'] == 'scar.stateful-static-instruction-lowering' and lowering['schema_version'] == 1
        and lowering['target_dag_compiled_for_static_analysis'] is True
        and lowering['target_workload_executed'] is False and lowering['target_variants_ranked'] is False
        and lowering['native_library_built'] is False, 'static lowering scope differs')
    _require(lowering['region'] == domain['region'] and lowering['compiler'] == context['backend']['compiler']
        and lowering['backend_implementation'] == stateful.implementation_sha256(),
        'static source/backend dependency differs')
    region, plan, placement, contract = C._static_stage_contract(domain, candidate_id, context)
    _require(plan == lowering['plan'] and lowering['resource_binding'] == plan['resource_binding'],
        'static stage plan or placement differs')
    source, assembly = _replay_compiler(lowering)
    _require(source == stateful._source(region, plan, placement, contract),
        'original backend source differs from source/plan replay')
    return region, plan, source, assembly, lowering


def _replay_components(document, reference, demand, model, profile, calibrated, target, analyze, *, source_pack=None):
    from scar.optimization.stateful_cpu_model import component_key

    region, plan, source, assembly, lowering = target
    order = {node.id: index for index, node in enumerate(region.nodes)}
    terms = [row for row in demand['terms'] if row['component'] == 'FUSED_SCALAR_DAG_ROW_STAGE']
    _require(len(document['components']) == len(terms) and not document['unsupported_components']
        and not document['proposed_estimates'] and document['all_fused_components_covered'] is True,
        'complete admitted fused projection required')
    capacity = model['cache_geometry']['reported_capacity_bytes']
    maximum_stack = max(row['stack_memory_references'] for entry in calibrated.values() for row in entry['loops'])
    checked = []
    for index, (row, term) in enumerate(zip(document['components'], terms)):
        _, span, witness = C._projection_stage(plan, term, order, source, capacity, profile)
        _require(term['parameters']['workers'] == len(model['resource_context']['worker_cpu_ids']),
            'exact target worker mask differs')
        analysis = analyze(assembly, row['instruction_analysis'], model['tool'], source_range=span,
            source_path=lowering['source']['path'])
        cost, qualification = C._projection_cost(term, analysis, model, profile, maximum_stack)
        _require(qualification['admitted'] is True, 'fresh holdout does not authorize this component')
        expected = C._projection_component(term, analysis, cost, qualification, profile, document['source_model'],
            document['static_lowering'], witness)
        if source_pack is not None:
            expected['source_pack'] = source_pack
        _require(row == expected, 'projected component estimate, parameters or dependencies differ')
        checked.append(dict(index=index, key=component_key(row['component'], row['parameters']),
            fact=dict(artifact=reference, locator=f'#/components/{index}'), component=deepcopy(row)))
    return checked


def verify_instruction_projection(domain, candidate_id, projection):
    """Replay compiler and machine-model facts against a frozen instruction profile."""
    from scar.optimization.stateful_cpu_model import _domain, stateful_cpu_demand

    domain = _domain(domain)
    demand = stateful_cpu_demand(domain, candidate_id)
    context = demand['resource_context']
    reference = artifact_reference(projection, 'scar.stateful-resource-calibration')
    document = read_artifact(reference)
    _require(document['schema_version'] == 1 and document['resource_context'] == context
        and document.get('target_workload_executed') is False and document.get('target_variants_ranked') is False
        and document.get('target_dag_compiled_for_static_analysis') is True, 'projection context/authority differs')
    model_ref = document['source_model']
    read_artifact(model_ref)
    profile = C.INSTRUCTION_COST_SCHEMAS.get(model_ref['schema'])
    _require(profile is not None, 'unsupported instruction cost projection profile')
    checked_ref, model, calibrated = C._verify_model(model_ref['uri'], profile)
    _require(checked_ref == model_ref and model['resource_context'] == context, 'canonical model/context differs')
    tool = model['tool']
    assembly = _raw(model['native']['assembly'])
    cache = {}
    recheck = _replay_parallel_analysis if profile.parallel else _replay_analysis

    def analyze(assembly, analysis, tool, **scope):
        return recheck(assembly, analysis, tool, cache=cache, **scope)

    registry = profile.registry()
    for key, analysis in calibrated.items():
        case = registry[int(key) // 2]
        group = int(key) % 2 + 1
        analyze(assembly, analysis, tool, function=case['id'] + '_g' + str(group))
    lowering_ref = document['static_lowering']
    target = _replay_target(domain, candidate_id, context, lowering_ref)
    components = _replay_components(document, reference, demand, model, profile, calibrated, target, analyze)
    result = dict(schema=profile.prefix + '-cost-projection-replay', schema_version=1,
        projection=reference, source_profile=model_ref, static_lowering=lowering_ref,
        frozen_design=model['frozen_design'], raw_observations=model['raw_observations'],
        instruction_analysis=model['instruction_analysis'], tool=tool['executable'], components=components,
        compiler_reexecuted_static_only=True, mca_reexecuted_static_only=True,
        timing_observations_remeasured=False, target_workload_executed=False, target_variants_ranked=False,
        target_dag_compiled_for_static_analysis=True, model_authority='DECLARED_FORWARD_COMPONENT_COST_NOT_PHYSICAL_BOUND')
    if profile.parallel:
        result.update(openmp_runtime=tool['openmp_runtime'], compiler_openmp_runtime=tool['compiler_openmp_runtime'],
            partition_witnesses=model['partition_witnesses'], qualification_design=model.get('qualification_design'),
            calibration_library_loaded_for_runtime_symbol_identity_only=True)
    return result


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
        loops=C._natural_loops(C._lex_assembly(assembly),**kwargs)
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
    from scar.optimization.stateful_cpu_model import stateful_cpu_demand

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
    session.pack(pack_ref)
    tool = model['tool']
    profile = C.instruction_cost_profile(pack['mode'])
    lowering_ref = document['static_lowering']
    target = _replay_target(domain, candidate_id, demand['resource_context'], lowering_ref)
    checked_rows = _replay_components(document, reference, demand, model, profile, calibrated,
        target, session.analysis, source_pack=pack_ref)
    session.check()
    return dict(schema='scar.cpu-batch-instruction-projection-replay',schema_version=1,projection=reference,source_profile=model_ref,
        source_pack=pack_ref,static_lowering=lowering_ref,frozen_design=model['frozen_design'],raw_observations=model['raw_observations'],
        instruction_analysis=pack['instruction_analysis'],tool_contract=tool,components=checked_rows,
        compiler_reexecuted_static_only=True,mca_reexecuted_static_only=True,explicit_operation_scoped_session=True,
        timing_observations_remeasured=False,target_workload_executed=False,target_variants_ranked=False,
        target_dag_compiled_for_static_analysis=True,model_authority='DECLARED_FORWARD_COMPONENT_COST_NOT_PHYSICAL_BOUND')
