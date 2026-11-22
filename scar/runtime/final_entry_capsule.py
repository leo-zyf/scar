"""Shared final-entry admission for registered, explicitly typed profiles.

The fixed-page cost wire format is reused unchanged; it does not impose a
caller-array contract. New profiles currently export UNMEASURED validation
slots only. Ordinary invocation remains NO-OP until final cost is admitted.
"""
from hashlib import sha256
import json
from pathlib import Path

from .caller_pipeline_capsule import (_json,_need,pack_slot,evaluate_slot,SLOT_BYTES,TRIAL)
from .caller_pipeline_deployment import entry_context,environment_sha256,provider_identity
from .compiled_deployment import _raw

PAYLOAD='scar.final-entry-capsule-payload'
EXECUTION='scar.final-entry-capsule-execution'
PROFILE='COMPILED_FIXED_COUNT_LOOPS_V1'
STATEFUL_PROFILE='COMPILED_STATEFUL_REGIONS_V1'
TRUST='EXPLICIT_EXPECTED_PAYLOAD_AND_SLOT_SHA256_FROM_BUILD_PRODUCER_NOT_SIGNATURES'


def _profile(kind):
    if kind==PROFILE:
        from . import compiled_loop_capsule
        return compiled_loop_capsule
    if kind==STATEFUL_PROFILE:
        from . import compiled_stateful_capsule
        return compiled_stateful_capsule
    raise ValueError('unregistered final-entry profile')


def runtime_identity(kind):
    profile=_profile(kind)
    runtime=Path(__file__).resolve().parent
    paths=[runtime/'final_entry_capsule.py',runtime/'caller_pipeline_capsule.py',
        runtime/'caller_pipeline_deployment.py',runtime.parent/'optimization'/'paired_entry_selection.py',
        *profile.runtime_sources()]
    rows=[(str(path),sha256(path.read_bytes()).hexdigest())for path in dict.fromkeys(paths)]
    return sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def admit(capsule,slot,*,expected_payload_sha256,expected_slot_sha256,entry,argv,report):
    capsule=Path(capsule).resolve();path=capsule/'payload.json'
    _need(path.stat().st_size<=256*1024,'bounded final payload required')
    raw=path.read_bytes();_need(sha256(raw).hexdigest()==expected_payload_sha256,'final payload trust pin differs')
    payload=_json(raw)
    _need((payload.get('schema'),payload.get('schema_version'))==(PAYLOAD,1)
        and payload['trust_policy']==TRUST and payload['capsule_directory']==str(capsule)
        and payload['slot_bytes']==SLOT_BYTES and payload['policy']['frozen_before_measurement']is True,
        'supported final-entry payload required')
    profile=_profile(payload['profile_kind']);slot=Path(slot).resolve()
    _need(slot.parent==capsule and slot.stat().st_size==SLOT_BYTES,'one fixed-page cost slot required')
    raw=slot.read_bytes();_need(sha256(raw).hexdigest()==expected_slot_sha256,'final cost-slot trust pin differs')
    decision=_json(raw);evaluation=evaluate_slot(decision,payload,expected_payload_sha256)
    _need(payload['entry_context']==entry_context(entry,argv),'entry argv/context changed')
    _need(payload['environment_sha256']==environment_sha256(),'inherited environment changed')
    _need(payload['runtime_identity']==runtime_identity(payload['profile_kind']),'final-entry implementation changed')
    report=Path(report).resolve()
    _need(not report.exists()and report.is_relative_to(Path(payload['reporting']['allowed_root']).resolve()),
        'fresh report in frozen storage scope required')
    for source in payload['source_refs']:_raw(source)
    _need(payload['provider_identity']==provider_identity(),'NumPy runtime identity changed')
    profile.verify_profile(payload['artifact_profile'])
    return payload,decision,evaluation


def run_final_entry(entry,argv,*,capsule,slot,expected_payload_sha256,expected_slot_sha256,
        report,profile_kind=PROFILE,validation_only=False,exclusive_inputs=False):
    """Use one registered existing implementation, or the entire original entry.

    Validation is unmeasured, explicit and never deployment authority. No
    selection dictionary, old timing receipt or payload Boolean enables it.
    """
    from .entry_runner import _arguments,run_original_script
    entry,argv=_arguments(entry,argv);report=Path(report).resolve()
    _need(not report.exists(),'fresh final-entry report required')
    _profile(profile_kind);payload=decision=None;selected=False
    context=entry_context(entry,argv);environment=environment_sha256();identity=runtime_identity(profile_kind)
    reason='final_cost_not_admitted_default_no_op'
    try:
        payload,decision,_=admit(capsule,slot,expected_payload_sha256=expected_payload_sha256,
            expected_slot_sha256=expected_slot_sha256,entry=entry,argv=argv,report=report)
        _need(payload['profile_kind']==profile_kind,'selected artifact profile differs')
        selected=(validation_only is True and exclusive_inputs is True and decision['role']==TRIAL)
        if selected:reason='explicit_unmeasured_protocol_validation'
    except (OSError,ValueError,KeyError,TypeError,ImportError)as error:
        reason='final_scope_or_profile_not_admitted:'+str(error)
    if selected:execution=_profile(profile_kind).execute_profile(payload['artifact_profile'],entry,argv)
    else:execution=run_original_script(entry,argv)
    stable=(entry_context(entry,argv)==context and environment_sha256()==environment and runtime_identity(profile_kind)==identity)
    try:
        if payload is not None:
            again,again_decision,_=admit(capsule,slot,expected_payload_sha256=expected_payload_sha256,
                expected_slot_sha256=expected_slot_sha256,entry=entry,argv=argv,report=report)
            stable=stable and again==payload and again_decision==decision
    except (OSError,ValueError,KeyError,TypeError,ImportError):stable=False
    result=dict(schema=EXECUTION,schema_version=1,profile_kind=profile_kind,
        payload_sha256=expected_payload_sha256,slot_sha256=expected_slot_sha256,
        selected=selected,validation_only=validation_only is True,mode='UNMEASURED_VALIDATION'if selected else'ORIGINAL_NO_OP',
        reason=reason,context_unchanged=stable,execution=execution,
        statistics=_profile(profile_kind).execution_statistics(execution),deployment_authorized=False,
        cost_status='UNMEASURED',performance_measured=False,raw_benchmark_evidence_replayed=False,
        whole_program_cpp=False,new_mathematical_backend=False)
    report.parent.mkdir(parents=True,exist_ok=True)
    with report.open('x')as stream:json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    return result


__all__=['run_final_entry','admit','runtime_identity','PAYLOAD','PROFILE','STATEFUL_PROFILE','TRUST']
