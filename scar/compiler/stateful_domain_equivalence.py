"""Exact immutable-material aliases between original stateful CPU domains.

This preserves two domain identities and their unchanged cost evidence. It
neither rewrites a profile nor republishes old costs under a destination domain.
The original rebase normalizers admit only two material URI aliases and three
domain-qualified evidence pointers, whose referenced values must be equal.
"""
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference,read_artifact
from scar.optimization.resource_solver import semantic_sha256
from scar.optimization.stateful_cost_bounds import _need,_receipt,_write
from scar.optimization.stateful_cost_rebase import _domain_value,_contexts


SCHEMA='scar.stateful-domain-equivalence'


def _reference(value):
    if isinstance(value,(str,Path)):
        path=Path(value).resolve();path=path/'domain.json' if path.is_dir() else path
        return artifact_reference(path,'scar.stateful-cpu-domain')
    reference=dict(value)
    _need(reference.get('schema')=='scar.stateful-cpu-domain','original stateful domain ArtifactRef required')
    read_artifact(reference)
    return reference


def _derive(source_reference,target_reference):
    from scar.optimization.stateful_cpu_choices import _domain
    from scar.optimization import stateful_cost_rebase,stateful_cpu_choices
    source_reference=_reference(source_reference);target_reference=_reference(target_reference)
    source=_domain(read_artifact(source_reference));target=_domain(read_artifact(target_reference))
    identity=semantic_sha256(_domain_value(source))
    _need(identity==semantic_sha256(_domain_value(target)),
        'stateful source/shape/budget/hardware/backend differs beyond two material URI aliases')
    materials=[]
    for role in ('region','base_stage_plan'):
        left,right=source[role],target[role]
        _need({k:v for k,v in left.items()if k!='uri'}=={k:v for k,v in right.items()if k!='uri'}
            and read_artifact(left,json_document=False)==read_artifact(right,json_document=False),
            'original '+role+' bytes differ beyond URI alias')
        materials.append(dict(role=role,source=left,target=right,byte_content_equal=True))
    source_ids=[r['id']for r in source['candidates']]+['NO_OP']
    target_ids=[r['id']for r in target['candidates']]+['NO_OP']
    _need(source_ids==target_ids,'original finite candidate set or order differs')
    families={};contexts=[]
    for identifier in source_ids:
        left,right,changes=_contexts(source,target,identifier)
        key=semantic_sha256(changes)
        families.setdefault(key,dict(id=key,rebindings=changes))
        contexts.append(dict(candidate_id=identifier,source_context_sha256=semantic_sha256(left),
            target_context_sha256=semantic_sha256(right),verified_rebinding_family=key))
    read_artifact(source_reference);read_artifact(target_reference)
    report=dict(schema=SCHEMA,schema_version=1,source_domain=source_reference,target_domain=target_reference,
        source_domain_id=source['domain_id'],target_domain_id=target['domain_id'],
        domain_value_identity_sha256=identity,material_aliases=materials,
        context_rebinding_families=list(families.values()),contexts=contexts,
        candidate_context_count=len(contexts),all_candidate_contexts_and_no_op_verified=True,
        allowed_aliases=['region.uri','base_stage_plan.uri','context.live_constraints_ref.domain_id',
            'context.hardware_context_ref.domain_id','context.backend.descriptor_ref.domain_id'],
        pointers_and_referenced_values_changed=False,scope_or_numeric_cost_changed=False,
        cost_profile_rewritten=False,target_domain_cost_profile_created=False,
        authority='EXACT_ORIGINAL_DOMAIN_VALUES_AND_PER_CANDIDATE_CONTEXT_EQUIVALENCE',
        implementation=_receipt(__file__),normalizer_implementations=[_receipt(stateful_cost_rebase.__file__),
            _receipt(stateful_cpu_choices.__file__)])
    report['id']='stateful-domain-equivalence:'+semantic_sha256(report)
    return report


def retain_stateful_domain_equivalence(source_domain,target_domain,out):
    return _write(Path(out).resolve(),_derive(source_domain,target_domain))


def verify_stateful_domain_equivalence(path):
    reference=artifact_reference(path,SCHEMA);report=read_artifact(reference)
    _need(report.get('schema_version')==1,'supported domain equivalence receipt required')
    _need(_derive(report['source_domain'],report['target_domain'])==report,
        'original complete domain or context equivalence replay differs')
    read_artifact(reference)
    return report
