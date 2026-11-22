"""Compiled entry."""

import argparse
import importlib.abc
import json
import runpy
import sys
from hashlib import sha256
from functools import wraps
from importlib.machinery import PathFinder, SourceFileLoader
from pathlib import Path
from scar.runtime.entry_runner import _Scope, _Loader, _file, run_original_script
from scar.runtime.caller_pipeline_capsule import (
    _json,
    _need,
    evaluate_slot,
    SLOT_BYTES,
    TRIAL,
)
from scar.runtime.caller_pipeline_deployment import (
    entry_context,
    environment_sha256,
    provider_identity,
)
from scar.runtime.compiled_deployment import _raw, selected_rows


# Execute an unchanged entry with model-selected loop implementations.
#
# Original F4 proofs are replayed before execution. Python keeps compatibility
# work and external effects. Candidate validation is distinct from deployment
# profitability; this entry does not authorize an unmeasured deployment.
class _LoopScope(_Scope):
    def __init__(self,artifact,proof,spec):
        self.artifact,self.proof,self.spec=artifact,proof,spec
        self.source=Path(proof.source.path).resolve();self.name=proof.source.qualname
        self.active=True;self.bindings=[];self.seen=set();self.frontier=[]

    def attach(self,module,name,*,natural):
        if not self.active or id(module) in self.seen:return
        self.seen.add(id(module))
        if _file(module)!=self.source or not self.name.isidentifier():
            self.frontier.append('unsupported_original_function_boundary');return
        original=vars(module).get(self.name)
        try:
            if self.proof is None:
                from scar.runtime.native_temporal_invariant import guarded_temporal_invariant,SourceMismatch
                guarded=guarded_temporal_invariant(original,self.artifact,exclusive_inputs=True,
                    trial=self.trial,selection=self.selection)
            else:
                from scar.compiler.invariant_loop_runtime import guarded_invariant_loop
                from scar.compiler.runtime import SourceMismatch
                guarded=guarded_invariant_loop(original,self.artifact,self.proof,exclusive_inputs=True,trial=True)
        except (SourceMismatch,ValueError) as error:
            self.frontier.append('original_boundary_admission_rejected:'+str(error));return
        scope=self
        @wraps(original)
        def binding(*args,**kwargs):
            return guarded(*args,**kwargs) if scope.active else original(*args,**kwargs)
        binding.scar_stats=guarded.scar_stats;binding.scar_artifact=self.artifact
        setattr(module,self.name,binding)
        self.bindings.append(dict(module=module,module_name=name,original=original,owner=module,
            attribute=self.name,descriptor=original,installed=binding,binding=binding,
            stats=guarded.scar_stats,natural_import=natural))


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self,scopes):self.scopes=scopes
    def find_spec(self,fullname,path=None,target=None):
        spec=PathFinder.find_spec(fullname,path,target)
        if spec is not None and type(spec.origin) is str:
            selected=[scope for scope in self.scopes if Path(spec.origin).resolve()==scope.source]
            if selected and type(spec.loader) is SourceFileLoader:
                # Each loader completes once, then attaches all selected
                # boundaries in this original module before from-import.
                class Group:
                    frontier=[]
                    def attach(self,module,name,*,natural):
                        for scope in selected:scope.attach(module,name,natural=natural)
                spec.loader=_Loader(spec.loader,Group(),fullname)
        return spec


def _artifact_snapshot(path, schema):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return dict(uri=str(path), sha256=sha256(raw).hexdigest(), size_bytes=len(raw), schema=schema), raw


def artifact_reference(path, schema):
    return _artifact_snapshot(path, schema)[0]


def _artifact_document(raw, schema):
    value = json.loads(raw)
    if type(value) is not dict or value.get('schema') != schema:
        raise ValueError('compiled entry schema differs')
    return value


def read_artifact(reference):
    actual, raw = _artifact_snapshot(reference['uri'], reference['schema'])
    if actual != reference:
        raise ValueError('compiled entry receipt differs')
    return _artifact_document(raw, reference['schema'])


def read_pipeline(pipeline):
    path = Path(pipeline).resolve()
    if path.is_dir():
        path = path / 'result.json'
    reference, raw = _artifact_snapshot(path, 'scar.program-compiler-pipeline')
    return reference, _artifact_document(raw, reference['schema'])


def run_build(pipeline,entry,argv=(),*,exclusive_inputs=False,measure_candidate=False,deployment=None,
              expected_pipeline=None):
    reference, report = read_pipeline(pipeline)
    if expected_pipeline is not None and reference != expected_pipeline:
        raise ValueError('compiled pipeline changed after final entry admission')
    from scar.runtime.compiled_deployment import deployment_gate
    gate=(dict(decision='MEASURE_SELECTED_MODEL_IMPLEMENTATION',deployment_authorized=False)
        if measure_candidate else deployment_gate(deployment,reference,entry,argv))
    if not exclusive_inputs or gate['decision'] not in ('MEASURE_SELECTED_MODEL_IMPLEMENTATION','LOAD_MODEL_SELECTED_NATIVE'):
        return dict(mode='ORIGINAL_EXECUTION',reason='candidate_measurement_and_exclusive_inputs_required',
            execution=run_original_script(entry,argv),pipeline=reference,deployment_gate=gate)
    rows=selected_rows(report)
    if not rows:return dict(mode='ORIGINAL_EXECUTION',execution=run_original_script(entry,argv),pipeline=reference)
    if any(row.get('proof',{}).get('schema')=='scar.composite-invariant-loop-proof'
            and not row.get('bundle_export') for row in rows):
        return dict(mode='ORIGINAL_EXECUTION',reason='composite_compact_export_required',
            execution=run_original_script(entry,argv),pipeline=reference,deployment_gate=gate)
    scopes=[]
    if all(row.get('bundle_export') for row in rows):
        for row in rows:
            exported=read_artifact(row['bundle_export'])
            if exported['bundle']!=row['bundle']:raise ValueError('bundle directory differs from retained export')
            if row.get('selected_kind') == 'ARRAY_LOOP_TEMPORAL_FUSION':
                from scar.runtime.native_loop import load as load_recurrence

                if (exported['schema'] != 'scar.loop-runtime-export' or exported['schema_version'] != 2
                        or exported['source_proof'] != row['proof'] or exported['execution_plan'] != row['schedule']
                        or exported['deployment_authorized'] is not False):
                    raise ValueError('recurrence export differs from the compiled source and plan')
                artifact = load_recurrence(row['bundle'], expected_admission_sha256=exported['expected_admission_sha256'],
                                           allow_unqualified=True)
                contract = artifact.contract
                if (contract['schema_version'] != 2 or contract['source_proof'] != row['proof']
                        or contract['execution_plan'] != row['schedule'] or contract['trip_count'] != row['trip_count']
                        or contract['memory_budget_bytes'] != row['memory_budget_bytes']
                        or row['input_shapes'] != {contract['state_parameter']: contract['array']['shape']}):
                    raise ValueError('recurrence runtime contract differs from the compiled boundary')
                scopes.append(_Scope(artifact))
                continue
            if row.get('selected_kind')=='ORDERED_STATEFUL_REGION':
                from scar.runtime.native_stateful import load as stateful_load
                artifact=stateful_load(row['bundle'],expected_admission_sha256=exported['expected_admission_sha256'])
                contract=artifact.contract
                if (any(exported[key]!=contract['native']['ids'][key] for key in contract['native']['ids'])
                        or contract['selection']['input_shapes']!=row['input_shapes']
                        or contract['memory']['memory_budget_bytes']!=row['memory_budget_bytes']):
                    raise ValueError('stateful export differs from retained validated implementation')
                scopes.append(_Scope(artifact));continue
            from scar.runtime.native_temporal_invariant import load

            artifact=load(row['bundle'],expected_admission_sha256=exported['expected_admission_sha256'])
            contract=artifact.contract
            if ((contract.get('proof_kind')=='COMPOSITE') !=
                    (row['proof']['schema']=='scar.composite-invariant-loop-proof')):
                raise ValueError('composite row/runtime requires the separate original proof kind')
            receipt=contract['resource_schedule']['receipt']
            # Export replay sealed this plan hash into the compact contract.
            # Execution matches that reference without parsing the cost graph.
            if (contract['source_proof']!=row['proof'] or receipt['path']!=row['schedule']['uri']
                    or receipt['sha256']!=row['schedule']['sha256'] or receipt['bytes']!=row['schedule']['size_bytes']):
                raise ValueError('bundle proof/schedule differs from compiled pipeline')
            scope=_LoopScope.__new__(_LoopScope)
            scope.artifact,scope.proof,scope.spec=artifact,None,row
            scope.source=Path(contract['source']['path']).resolve();scope.name=contract['source']['qualname']
            scope.active=True;scope.bindings=[];scope.seen=set();scope.frontier=[]
            scope.trial=measure_candidate
            scope.selection={**gate,'native_admission_sha256':artifact.report['admission_sha256']}
            scopes.append(scope)
    else:
        from scar.analysis.build import BuildSession
        with BuildSession(Path(report['program']['uri']).parent) as session:
            scopes.extend(_legacy_scopes(rows,session,report))
    return _execute_scopes(scopes,entry,argv,reference,gate)


def _legacy_scopes(rows,session,report):
    from scar.compiler.loop_admission import read_invariant_loop_proof
    from scar.compiler.temporal_invariant_loop import load_temporal_invariant_loop
    from scar.optimization.invariant_loop_schedule import verify_invariant_loop_schedule
    scopes=[]
    for row in rows:
        read_artifact(row['proof']);read_artifact(row['native_artifact'])
        proof=read_invariant_loop_proof(row['proof']['uri'],catalog=session.view)
        if artifact_reference(Path(row['native_directory'])/'artifact.json',row['native_artifact']['schema'])!=row['native_artifact']:
            raise ValueError('native directory differs from its retained artifact')
        artifact=load_temporal_invariant_loop(row['native_directory'])
        schedule=read_artifact(row['schedule'])
        verify_invariant_loop_schedule(proof,Path(report['hardware']['path']).parent,schedule)
        specialization=artifact.plan['loop_interface']['execution_specialization']
        if (specialization['input_shapes']!=schedule['input_shapes'] or specialization['trip_count']!=schedule['trip_count']
                or specialization.get('input_layout')!='C_CONTIGUOUS'
                or artifact.plan['memory_budget_bytes']!=schedule['effective_incremental_budget_bytes']
                or artifact.plan['execution_plan']['resource_binding']!=schedule['resource_binding']):
            raise ValueError('native specialization/placement differs from selected resource schedule')
        scopes.append(_LoopScope(artifact,proof,row))
    return scopes


def _execute_scopes(scopes,entry,argv,reference,gate):
    entry=Path(entry).resolve();old_argv,old_path,old_meta=sys.argv,sys.path,sys.meta_path
    argv_values,path_values,meta_values=list(old_argv),list(old_path),list(old_meta)
    sys.argv=[str(entry),*argv];sys.path=[str(entry.parent),*old_path[1:]]
    finder=_Finder(scopes)
    sys.meta_path=[finder if item is PathFinder else item for item in old_meta]
    try:
        for name,module in tuple(sys.modules.items()):
            for scope in scopes:
                if _file(module)==scope.source:scope.attach(module,name,natural=False)
        try:runpy.run_path(str(entry),run_name='__main__')
        except SystemExit as error:
            if error.code is not None and error.code!=0:raise
    finally:
        try:
            for scope in scopes:scope.finish()
        finally:
            old_argv[:]=argv_values;old_path[:]=path_values;old_meta[:]=meta_values
            sys.argv,sys.path,sys.meta_path=old_argv,old_path,old_meta
    return dict(schema='scar.compiled-program-execution',schema_version=1,pipeline=reference,
        mode=gate['decision'],deployment_gate=gate,entry=str(entry),argv=list(argv),
        attachments=[dict(source=str(s.source),qualname=s.name,bindings=len(s.bindings),
            statistics=[r['stats'] for r in s.bindings],native_execution=s.artifact.report.get('last_execution'),
            frontier=s.frontier) for s in scopes],full_program_cpp=False,
        deployment_authorized=gate['deployment_authorized'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline',required=True);parser.add_argument('--exclusive-inputs',action='store_true')
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--measure-candidate',action='store_true');modes.add_argument('--deployment')
    parser.add_argument('--report')
    parser.add_argument('command',nargs=argparse.REMAINDER);args=parser.parse_args(argv)
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command:parser.error('original Python script required after --')
    path=None if args.report is None else Path(args.report)
    if path is not None and path.exists():parser.error('fresh report path required')
    result=run_build(args.pipeline,command[0],command[1:],exclusive_inputs=args.exclusive_inputs,
        measure_candidate=args.measure_candidate,deployment=args.deployment)
    if path is not None:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
    return 0


# Shared final-entry admission for registered, explicitly typed profiles.
#
# The fixed-page cost wire format is reused unchanged; it does not impose a
# caller-array contract. New profiles currently export UNMEASURED validation
# slots only. Ordinary invocation remains NO-OP until final cost is admitted.
PAYLOAD='scar.final-entry-capsule-payload'
EXECUTION='scar.final-entry-capsule-execution'
PROFILE='COMPILED_FIXED_COUNT_LOOPS_V1'
STATEFUL_PROFILE='COMPILED_STATEFUL_REGIONS_V1'
TRUST='EXPLICIT_EXPECTED_PAYLOAD_AND_SLOT_SHA256_FROM_BUILD_PRODUCER_NOT_SIGNATURES'


def _capsule_bundle(row):
    exported = read_artifact(row['bundle_export'])
    directory = Path(row['bundle']).resolve()
    _need(exported['bundle'] == str(directory), 'retained compact bundle path differs')
    return exported, directory


def _capsule_row(row, directory, contract, *, proof_kind, admission_schema, runtime_schema, **profile):
    from .native_loop import _file_hash_and_header

    native = directory / contract['native']['library']
    digest, header = _file_hash_and_header(native)
    _need(digest == contract['native']['library_sha256'] and header[:6] == b'\x7fELF\x02\x01',
          'original compact ELF bytes differ')
    return dict(id=row['id'], proof_kind=proof_kind, proof=row['proof'], schedule=row['schedule'],
                bundle=str(directory), bundle_export=row['bundle_export'],
                admission=artifact_reference(directory / 'admission.json', admission_schema),
                runtime=artifact_reference(directory / 'runtime.json', runtime_schema),
                native=dict(path=str(native), sha256=digest, bytes=native.stat().st_size),
                input_shapes=row['input_shapes'], memory_budget_bytes=row['memory_budget_bytes'], **profile)


def _fixed_loop_capsule_row(row):
    from .native_loop import _small_raw
    from .native_temporal_invariant import ADMISSION, SCHEMA, TRUST as EXPORT_TRUST, validate_contract

    _need(row.get('selected_kind') == 'REGISTER_TEMPORAL_FUSION' and row.get('bundle_export')
          and row['proof']['schema'] in ('scar.invariant-loop-proof', 'scar.composite-invariant-loop-proof'),
          'only compact original DIRECT/COMPOSITE fixed-loop artifacts are supported')
    exported, directory = _capsule_bundle(row)
    raw = _small_raw(directory / 'admission.json')
    admission = _json(raw)
    _need(sha256(raw).hexdigest() == exported['expected_admission_sha256']
          and admission.get('schema') == ADMISSION and admission.get('schema_version') == 1
          and admission['trust_policy'] == EXPORT_TRUST, 'original compact export admission required')
    raw = _small_raw(directory / 'runtime.json')
    contract = validate_contract(_json(raw))
    _need(sha256(raw).hexdigest() == admission['runtime_sha256'] == exported['runtime_sha256']
          and contract['native']['library_sha256'] == admission['library_sha256'],
          'compact runtime/library identities differ')
    composite = contract.get('proof_kind', 'DIRECT') == 'COMPOSITE'
    _need(composite == (row['proof']['schema'] == 'scar.composite-invariant-loop-proof')
          and contract['source_proof'] == row['proof'], 'original proof kind/source differs')
    schedule = row['schedule']
    _need(contract['resource_schedule']['receipt'] == dict(path=schedule['uri'], sha256=schedule['sha256'],
          bytes=schedule['size_bytes']), 'original selected schedule differs')
    _need(contract['trip_count'] == row['trip_count'] and contract['memory_budget_bytes'] <= row['memory_budget_bytes']
          and {name: array['shape'] for name, array in contract['arrays'].items()} == row['input_shapes'],
          'original shape/count/budget specialization differs')
    return _capsule_row(row, directory, contract, proof_kind='COMPOSITE' if composite else 'DIRECT',
                        admission_schema=ADMISSION, runtime_schema=SCHEMA, trip_count=row['trip_count'])


def _stateful_capsule_row(row):
    from .stateful_contract import ADMISSION_SCHEMA, SCHEMA, inspect_stateful_bundle

    _need(row.get('status') == 'MEASURED_STATEFUL_BUNDLE_READY' and row.get('selected_kind') == 'ORDERED_STATEFUL_REGION'
          and row.get('bundle_export') and row.get('native_artifact') and row.get('selection')
          and row.get('proof', {}).get('schema') == 'scar.stateful-validation-candidate',
          'only original exported transactional stateful ABI1 regions are supported')
    exported, directory = _capsule_bundle(row)
    _need(exported.get('schema') == 'scar.stateful-runtime-export', 'original stateful compact export schema differs')
    metadata = inspect_stateful_bundle(directory, expected_admission_sha256=exported['expected_admission_sha256'])
    contract, admission = metadata['contract'], metadata['admission']
    native = contract['native']
    _need(all(exported.get(key) == value for key, value in native['ids'].items())
          and exported['library_sha256'] == native['library_sha256']
          and exported['runtime_contract_sha256'] == admission['runtime_contract_sha256'],
          'stateful export/ABI identities differ')
    for reference, role in ((row['native_artifact'], 'original-native-manifest'),
                            (row['selection'], 'original-measured-selection')):
        receipt = dict(role=role, path=reference['uri'], sha256=reference['sha256'], bytes=reference['size_bytes'])
        _need(receipt in admission['evidence'], 'original stateful ' + role + ' receipt differs')
    shapes = {record['id']: record['shape'] for record in contract['inputs'] if record['kind'] == 'array'}
    _need(shapes == row['input_shapes'] == contract['selection']['input_shapes']
          and row['memory_budget_bytes'] == contract['memory']['memory_budget_bytes']
          and row['resource_binding']['cpu_ids'] == contract['cpu']['ids'],
          'stateful exact shape/budget/CPU specialization differs')
    return _capsule_row(row, directory, contract, proof_kind='TRANSACTIONAL_STATEFUL_REGION',
                        admission_schema=ADMISSION_SCHEMA, runtime_schema=SCHEMA,
                        native_artifact=row['native_artifact'], original_selection=row['selection'],
                        external_assumptions=contract['external_assumptions'],
                        floating_environment_policy=contract['floating_environment_policy'],
                        transaction_boundary='ONE_ORIGINAL_METHOD_PRIVATE_SHADOW_WRITES_THEN_ALL_STATE_COMMIT')


_CAPSULE_PROFILES = {PROFILE: _fixed_loop_capsule_row, STATEFUL_PROFILE: _stateful_capsule_row}
_RUNTIME_SOURCES = (
    'cli.py', '__main__.py', 'runtime/compiled_deployment.py', 'runtime/entry_runner.py',
    'runtime/native_temporal_invariant.py', 'runtime/native_loop.py', 'runtime/loop_guard.py',
    'runtime/loop_contract.py', 'runtime/array_copy_guard.py', 'trace/code_snapshot.py',
)
_STATEFUL_SOURCES = ('runtime/native_stateful.py', 'runtime/stateful_contract.py', 'runtime/stateful_guard.py')


def describe_final_profile(pipeline, kind):
    _need(kind in _CAPSULE_PROFILES, 'unregistered final-entry profile')
    reference, document = read_pipeline(pipeline)
    rows = selected_rows(document)
    _need(rows, 'at least one admitted compiled region is required')
    describe_row = _CAPSULE_PROFILES[kind]
    return dict(profile_kind=kind, pipeline=reference, rows=[describe_row(row) for row in rows])


def verify_final_profile(value, kind):
    _need(value.get('profile_kind') == kind, 'registered compact profile required')
    _need(describe_final_profile(value['pipeline']['uri'], kind) == value, 'compiled pipeline/bundle profile changed')


def _execution_statistics(execution):
    if execution.get('schema') == 'scar.entry-runner-execution':
        return dict(execution['statistics'])
    native, fallback, reasons = 0, 0, {}
    for attachment in execution.get('attachments', ()):
        for stats in attachment.get('statistics', ()):
            native += stats['native_calls']
            fallback += stats['fallback_calls']
            for reason, count in stats['reasons'].items():
                reasons[reason] = reasons.get(reason, 0) + count
    return dict(native_calls=native, fallback_calls=fallback, reasons=reasons)


def runtime_identity(kind):
    _need(kind in _CAPSULE_PROFILES, 'unregistered final-entry profile')
    runtime=Path(__file__).resolve().parent
    sources = _RUNTIME_SOURCES + (_STATEFUL_SOURCES if kind == STATEFUL_PROFILE else ())
    paths=[Path(__file__).resolve(),runtime/'caller_pipeline_capsule.py',
        runtime/'caller_pipeline_deployment.py',runtime.parent/'optimization'/'paired_entry_selection.py',
        *(runtime.parent / name for name in sources)]
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
    kind=payload['profile_kind'];slot=Path(slot).resolve()
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
    verify_final_profile(payload['artifact_profile'], kind)
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
    payload=decision=None;selected=False
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
    if selected:
        pipeline = payload['artifact_profile']['pipeline']
        execution = run_build(pipeline['uri'], entry, argv, exclusive_inputs=True, measure_candidate=True,
                              expected_pipeline=pipeline)
    else:
        execution = run_original_script(entry,argv)
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
        statistics=_execution_statistics(execution),deployment_authorized=False,
        cost_status='UNMEASURED',performance_measured=False,raw_benchmark_evidence_replayed=False,
        whole_program_cpp=False,new_mathematical_backend=False)
    report.parent.mkdir(parents=True,exist_ok=True)
    with report.open('x')as stream:json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    return result


def _capsule_main(argv):
    parser = argparse.ArgumentParser(description='Validate a pinned compiled entry capsule.')
    parser.add_argument('--profile-kind', choices=tuple(_CAPSULE_PROFILES), default=PROFILE)
    for flag in ('capsule', 'slot', 'payload-sha256', 'slot-sha256', 'report'):
        parser.add_argument('--' + flag, required=True)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--exclusive-inputs', action='store_true')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('unchanged Python entry required after --')
    run_final_entry(command[0], command[1:], capsule=args.capsule, slot=args.slot,
                    expected_payload_sha256=args.payload_sha256, expected_slot_sha256=args.slot_sha256,
                    report=args.report, profile_kind=args.profile_kind, validation_only=args.validate_only,
                    exclusive_inputs=args.exclusive_inputs)
    return 0


__all__ = [
    'run_final_entry',
    'admit',
    'runtime_identity',
    'describe_final_profile',
    'verify_final_profile',
    'PAYLOAD',
    'PROFILE',
    'STATEFUL_PROFILE',
    'TRUST',
    'artifact_reference',
    'read_artifact',
    'run_build',
    'main',
    'EXECUTION',
]


if __name__ == '__main__':
    arguments = sys.argv[1:]
    options = arguments[:arguments.index('--')] if '--' in arguments else arguments
    entry = _capsule_main if any(item == '--capsule' or item.startswith('--capsule=') for item in options) else main
    raise SystemExit(entry(arguments))
