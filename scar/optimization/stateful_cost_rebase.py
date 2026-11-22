"""Explicit cost-evidence equivalence for immutable stateful domain aliases.

No duration is re-estimated. Source and destination domains are independently
replayed; only original material URIs and three domain-qualified evidence
references may differ. Consumers replay this certificate before admitting the
unchanged source estimate in the destination context.
"""
import argparse
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics

from scar.ir.execution_recipe import artifact_reference,read_artifact
from scar.optimization.deployment_cost import _fact,_pointer
from scar.optimization.resource_solver import semantic_sha256

PROFILE='scar.rebased-stateful-cpu-cost-profile'
PROOF='scar.stateful-cpu-cost-rebase-proof'
_ORIGINAL='scar.stateful-cpu-cost-profile'
_DOMAIN='scar.stateful-cpu-domain'
_REFS=(('live_constraints_ref',),('hardware_context_ref',),('backend','descriptor_ref'))


def _ref(value,schema,filename=None):
    if isinstance(value,(str,Path)):
        value=Path(value).absolute()
        if value.is_dir():value/=filename or 'profile.json'
        reference=artifact_reference(value,schema)
    else:reference=deepcopy(value)
    if reference.get('schema')!=schema:raise ValueError('unsupported retained cost dependency: '+schema)
    document=read_artifact(reference)
    return reference,document


def _raw(receipt):
    if type(receipt)is not dict or set(receipt)!={'path','sha256','bytes'}:
        raise ValueError('retained raw byte receipt required')
    path=Path(receipt['path']);payload=path.read_bytes()
    if not path.is_absolute()or len(payload)!=receipt['bytes']or sha256(payload).hexdigest()!=receipt['sha256']:
        raise ValueError('original cost raw/tool/source receipt differs')


def _implementation(model_replays=()):
    root=Path(__file__).resolve().parents[1]
    names=('optimization/stateful_cost_rebase.py','optimization/stateful_cpu_model.py',
        'optimization/stateful_cpu_model.py','compiler/observe_original_stateful_cost.py')
    schemas={r['schema']for r in model_replays}
    if schemas:
        names+=('codegen/cpu_instruction_cost.py','codegen/cpu_instruction_replay.py')
    if 'scar.cpu-parallel-instruction-cost-projection-replay'in schemas:
        names+=('codegen/cpu_parallel_instruction_cost.py','codegen/cpu_instruction_replay.py')
    return {name:sha256((root/name).read_bytes()).hexdigest()for name in names}


def _domain_value(domain):
    result=deepcopy(domain);result.pop('domain_id')
    for role in ('region','base_stage_plan'):
        reference=result[role];read_artifact(reference)
        result[role]={key:value for key,value in reference.items()if key!='uri'}
    return result


def _contexts(source,target,candidate):
    from scar.optimization.stateful_cpu_model import stateful_cpu_cost_context
    left=stateful_cpu_cost_context(source,candidate);right=stateful_cpu_cost_context(target,candidate)
    normalized=deepcopy(left);changes=[]
    for path in _REFS:
        before=left;after=right;destination=normalized
        for part in path[:-1]:before=before[part];after=after[part];destination=destination[part]
        before=before[path[-1]];after=after[path[-1]]
        if (set(before)!={'domain_id','pointer'}or set(after)!={'domain_id','pointer'}
                or before['domain_id']!=source['domain_id']or after['domain_id']!=target['domain_id']
                or before['pointer']!=after['pointer']
                or semantic_sha256(_pointer(source,before['pointer']))!=semantic_sha256(_pointer(target,after['pointer']))):
            raise ValueError('cost rebase can change only a verified domain_id, never its pointer or value')
        destination[path[-1]]=deepcopy(after)
        changes.append(dict(field='/'.join(path),source=before,target=after,
            dependency_sha256=semantic_sha256(_pointer(source,before['pointer']))))
    if semantic_sha256(normalized)!=semantic_sha256(right):
        raise ValueError('stateful cost scope/source/hardware/live/budget/shape/backend/guards differ')
    return left,right,changes


def _original_dependencies(facts,domain,domain_ref,context):
    from scar.compiler.observe_original_stateful_cost import _validate_sample
    derivation=facts['derivation']
    if (set(derivation)!={'origin','basis','target_variant_trials','source_observation','assumptions'}
            or derivation['origin']!='DECLARED_FORWARD_MODEL'
            or derivation['basis']!='OBSERVED_ORIGINAL_SAME_SCOPE_MEDIAN_AS_FORWARD_EXPECTATION'
            or derivation['target_variant_trials']is not False):
        raise ValueError('unsupported original-method forward-model dependency')
    reference,observed=_ref(derivation['source_observation'],'scar.original-stateful-call-observation')
    if (observed.get('schema_version')!=1 or observed.get('context')!=context
            or observed.get('measurement_scope')!='ONE_ORIGINAL_METHOD_CALL_SITE'
            or observed.get('target_workload_executed')is not True
            or observed.get('target_variants_ranked')is not False
            or observed.get('target_variants_executed')is not False
            or observed.get('native_calls')!=0
            or observed.get('observed_caller_affinity_cpu_ids')!=domain['live_constraints']['process_affinity_cpu_ids']
            or observed.get('observed_native_thread_environment')!=domain['live_constraints']['native_thread_environment']):
        raise ValueError('original observation scope/domain/runtime dependency differs')
    design_ref,design=_ref(observed['design'],'scar.original-stateful-observation-design')
    if (design['domain']!=domain_ref or design['proof']!=domain['region']or design['context']!=context
            or design['region_sha256']!=domain['region_sha256']or design['input_shapes']!=domain['request']['input_shapes']):
        raise ValueError('original observation frozen design differs from source domain')
    read_artifact(design['program']);read_artifact(design['proof'])
    for receipt in design['tool_receipts']:_raw(receipt)
    for source in design['sources']:
        raw=Path(source['path']).read_bytes()
        if sha256(raw).hexdigest()!=source['sha256']:raise ValueError('original observed source version differs')
    entry=design['entry_context']
    if sha256(Path(entry['entry_path']).read_bytes()).hexdigest()!=entry['entry_sha256']:
        raise ValueError('original observation entry source receipt differs')
    samples=[];references=[];warmup=0;nonces=set()
    for row in observed['samples']:
        spec_ref,spec=_ref(row['spec'],'scar.original-stateful-observation-design')
        sample_ref,sample=_ref(row['sample'],'scar.original-stateful-call-sample')
        if (spec.get('frozen_design')!=design_ref or any(spec.get(key)!=value for key,value in design.items())
                or row.get('returncode')!=0 or row.get('validation_errors')
                or (row['phase'],row['index'])!=(spec['phase'],spec['index'])
                or spec['worker_nonce']in nonces):
            raise ValueError('original retained worker/design/phase evidence differs')
        nonces.add(spec['worker_nonce']);duration=_validate_sample(sample,spec)
        if duration!=row['duration_ns']or sample['process_id']!=row['process_id']:
            raise ValueError('original sample duration/process receipt differs')
        for name in ('stdout','stderr'):_raw(row[name])
        references.extend((spec_ref,sample_ref))
        if row['phase']=='MEASUREMENT':samples.append(duration)
        elif row['phase']=='WARMUP':warmup+=1
        else:raise ValueError('unsupported original observation phase')
    if (len(samples)<3 or samples!=observed['samples_ns']or len(samples)!=observed['repetitions']
            or warmup!=observed['warmup']or statistics.median(samples)!=observed['median_ns']
            or facts['estimate']!=dict(duration_ticks=math.ceil(statistics.median(samples)),tick_seconds=1e-9)):
        raise ValueError('original sample/median/forward estimate replay differs')
    return dict(kind='OBSERVED_ORIGINAL_METHOD_MEDIAN_FORWARD_EXPECTATION',observation=reference,
        frozen_design=design_ref,sample_dependencies=references,duration_ticks=facts['estimate']['duration_ticks'])


def _component_dependencies(facts,domain,candidate):
    from scar.optimization.stateful_cpu_model import stateful_cpu_demand, component_key
    derivation=facts['derivation']
    if (set(derivation)!={'origin','target_variant_trials','assumptions','component_refs'}
            or derivation['origin']not in ('DECLARED_FORWARD_MODEL','INDEPENDENT_PRIMITIVE_COMPOSITION')
            or derivation['target_variant_trials']is not False):
        raise ValueError('unsupported primitive-composition model dependency')
    demand=stateful_cpu_demand(domain,candidate);refs=derivation['component_refs'];total=0
    if len(refs)!=len(demand['terms']):raise ValueError('complete ordered primitive dependencies required')
    checked=set();instruction_replays={}
    def receipts(value,additional=frozenset()):
        if type(value)is dict:
            if set(value)=={'uri','sha256','size_bytes','schema'}:
                if (value['schema']not in ('scar.target-inventory','scar.target-hardware-graph')
                        and semantic_sha256(value)not in additional):
                    raise ValueError('unsupported additional resource-model artifact dependency')
                read_artifact(value,json_document=False);return
            if set(value)=={'path','sha256','bytes'}:_raw(value);return
            for child in value.values():receipts(child,additional)
        elif type(value)is list:
            for child in value:receipts(child,additional)
    for reference,term in zip(refs,demand['terms']):
        document=read_artifact(reference['artifact'])
        if (document.get('schema'),document.get('schema_version'))!=('scar.stateful-resource-calibration',1):
            raise ValueError('unsupported resource dependency schema')
        if (document.get('target_workload_executed')is not False or document.get('target_variants_ranked')is not False
                or document['resource_context']!=demand['resource_context']):
            raise ValueError('independent primitive context or authority differs')
        identity=semantic_sha256(reference['artifact'])
        if identity not in checked:
            additional=frozenset()
            schema=document.get('source_model',{}).get('schema')
            if schema in('scar.cpu-instruction-cost-model','scar.cpu-parallel-instruction-cost-model'):
                # The supported fitted-model adapters replay actual
                # source/plan/placement, compiler assembly, natural loops,
                # MCA outputs and every projected component. A schema or
                # hash alone cannot discharge these dependencies.
                if schema=='scar.cpu-instruction-cost-model':
                    from scar.codegen.cpu_instruction_replay import verify_instruction_projection as verify
                else:
                    from scar.codegen.cpu_instruction_replay import verify_parallel_instruction_projection as verify
                replay=verify(domain,candidate,reference['artifact']['uri'])
                if replay['projection']!=reference['artifact']:
                    raise ValueError('instruction projection differs from original component facts')
                instruction_replays[identity]=replay
                additional=frozenset(map(semantic_sha256,(replay['source_profile'],replay['static_lowering'])))
            receipts(document,additional);checked.add(identity)
        value,_=_fact(reference)
        # Fitted/external model dependencies require their own future replay
        # adapter. They cannot be admitted merely because their JSON hashes.
        if any(key in value for key in ('source_model','source_validation','measurement_sources')):
            replay=instruction_replays.get(identity)
            rows=()if replay is None else tuple(row for row in replay['components']if row['fact']==reference)
            if len(rows)!=1 or rows[0]['component']!=value:
                raise ValueError('unsupported or non-replayed additional component model dependency')
        if (component_key(value['component'],value['parameters'])!=term['key']
                or value['basis']not in ('DIRECT_COMPONENT_TIMING','DECLARED_COMPONENT_MODEL')
                or any(value.get(key)is False for key in ('selectable','precision_admitted','exportable_as_complete_cost'))
                or type(value['median_ns'])not in (int,float)or not math.isfinite(value['median_ns'])or value['median_ns']<0):
            raise ValueError('original admitted primitive estimate differs')
        total+=value['median_ns']*term['count']
    if facts['estimate']!=dict(duration_ticks=math.ceil(total),tick_seconds=1e-9):
        raise ValueError('original primitive forward estimate replay differs')
    result=dict(kind='ORDERED_INDEPENDENT_COMPONENT_SUM',component_refs=refs,duration_ticks=math.ceil(total))
    if instruction_replays:
        result['instruction_model_replays']=[{key:value for key,value in replay.items()if key!='components'}
            for replay in instruction_replays.values()]
    return result


def _model_dependencies(profile,domain,domain_ref,context,candidate):
    if set(profile)!={'schema','schema_version','context','context_evidence','estimate','derivation'}:
        raise ValueError('unknown source cost profile dependencies')
    refs=[*profile['context_evidence'],*profile['estimate']['evidence'],*profile['derivation']['evidence']]
    artifacts={semantic_sha256(reference['artifact']):reference['artifact']for reference in refs}
    if len(artifacts)!=1:raise ValueError('supported model requires one original context-bound facts artifact')
    reference=next(iter(artifacts.values()));facts=read_artifact(reference)
    if (facts.get('schema_version')!=1 or set(facts)!={'schema','schema_version','context','estimate','derivation'}
            or facts['context']!=context):raise ValueError('unsupported complete-call facts dependency')
    if facts['schema']=='scar.original-stateful-forward-cost':
        if candidate!='NO_OP':raise ValueError('original-method cost cannot describe a native implementation')
        dependencies=_original_dependencies(facts,domain,domain_ref,context)
    elif facts['schema']=='scar.stateful-cpu-model-facts':dependencies=_component_dependencies(facts,domain,candidate)
    else:raise ValueError('unsupported complete-call model dependency schema')
    return dict(facts=reference,**dependencies)


def _replay(source_profile,source_domain,target_domain,candidate):
    from scar.optimization.stateful_cpu_model import _domain, _cost
    source_ref,source=_ref(source_domain,_DOMAIN,'domain.json');source=_domain(source)
    target_ref,target=_ref(target_domain,_DOMAIN,'domain.json');target=_domain(target)
    original_ref,original=_ref(source_profile,_ORIGINAL)
    if semantic_sha256(_domain_value(source))!=semantic_sha256(_domain_value(target)):
        raise ValueError('stateful domains differ beyond immutable material URI aliases')
    source_context,target_context,changes=_contexts(source,target,candidate)
    cost,issues,_=_cost(original,source_context)
    if cost is None:raise ValueError('original cost profile rejected: '+'; '.join(issues))
    dependencies=_model_dependencies(original,source,source_ref,source_context,candidate)
    proof=dict(schema=PROOF,schema_version=1,source_profile=original_ref,source_domain=source_ref,target_domain=target_ref,
        candidate_id=candidate,source_context=source_context,target_context=target_context,
        domain_value_identity_sha256=semantic_sha256(_domain_value(source)),context_rebindings=changes,
        supported_model_dependencies=dependencies,
        reader_implementation=_implementation(dependencies.get('instruction_model_replays',())),
        new_measurements_collected=False,duration_reestimated=False,physical_optimality_proven=False)
    proof['id']='stateful-cost-rebase:'+semantic_sha256(proof)
    return proof,original,cost


def _profile(original,source_ref,target_context,proof_ref):
    return dict(schema=PROFILE,schema_version=1,context=deepcopy(target_context),
        context_evidence=deepcopy(original['context_evidence']),estimate=deepcopy(original['estimate']),
        derivation=deepcopy(original['derivation']),source_profile=source_ref,rebase_proof=proof_ref)


def rebase_stateful_cpu_cost(profile,source_domain,target_domain,*,candidate_id='NO_OP',out):
    out=Path(out).absolute()
    if out.exists():raise ValueError('fresh verified stateful cost rebase output required')
    proof,original,cost=_replay(profile,source_domain,target_domain,candidate_id)
    out.mkdir(parents=True)
    def write(path,value):path.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
    proof_path=out/'proof.json';write(proof_path,proof);proof_ref=artifact_reference(proof_path,PROOF)
    rebased=_profile(original,proof['source_profile'],proof['target_context'],proof_ref)
    path=out/'profile.json';write(path,rebased)
    result=dict(schema='scar.stateful-cpu-cost-rebase-result',schema_version=1,profile=str(path),proof=proof_ref,
        source_profile=proof['source_profile'],candidate_id=candidate_id,duration_ticks=cost['duration_ticks'],
        tick_seconds=cost['tick_seconds'],new_measurements_collected=False,duration_reestimated=False)
    write(out/'result.json',result);return result


def verify_rebased_stateful_cpu_cost(value,*,expected_context=None):
    own_path=None
    if isinstance(value,(str,Path)):
        reference,profile=_ref(value,PROFILE);own_path=reference['uri']
    else:profile=deepcopy(value)
    if (profile.get('schema'),profile.get('schema_version'))!=(PROFILE,1):raise ValueError('versioned explicit cost rebase profile required')
    proof_ref,proof=_ref(profile['rebase_proof'],PROOF)
    paths=[proof_ref['uri'],proof['source_profile']['uri']]
    if own_path is not None:paths.append(own_path)
    if len(paths)!=len(set(paths)):raise ValueError('cost proof/source-profile reference cycle')
    expected,original,cost=_replay(proof['source_profile'],proof['source_domain'],proof['target_domain'],proof['candidate_id'])
    if semantic_sha256(expected)!=semantic_sha256(proof):raise ValueError('stateful cost equivalence proof differs from full replay')
    expected_profile=_profile(original,proof['source_profile'],proof['target_context'],proof_ref)
    if (semantic_sha256(expected_profile)!=semantic_sha256(profile)
            or expected_context is not None and semantic_sha256(profile['context'])!=semantic_sha256(expected_context)):
        raise ValueError('rebased stateful cost profile differs from verified destination context')
    return dict(profile=profile,original_cost=cost,proof=proof)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',required=True);parser.add_argument('--source-domain',required=True)
    parser.add_argument('--target-domain',required=True);parser.add_argument('--candidate-id',default='NO_OP')
    parser.add_argument('--out',required=True);args=parser.parse_args(argv)
    print(json.dumps(rebase_stateful_cpu_cost(args.profile,args.source_domain,args.target_domain,
        candidate_id=args.candidate_id,out=args.out),sort_keys=True,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
