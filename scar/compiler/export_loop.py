"""Build-time admission and small runtime projection of an existing loop ELF.

The complete loader owns graph/strategy/profile/source/ISA replay. Exported
hashes bind bytes to an explicitly trusted producer receipt, not a signature.
"""
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import platform
import struct
import sys


def _json_bytes(value):
    return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def _receipt(path,role,*,expected_sha256=None,expected_bytes=None):
    path=Path(path).absolute()
    if path.is_symlink() or not path.is_file():raise ValueError('original regular export evidence required: '+str(path))
    before=path.stat();digest=sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    after=path.stat();stamp=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    if stamp(before)!=stamp(after):raise ValueError('export evidence changed while reading: '+str(path))
    result=dict(role=role,path=str(path),sha256=digest.hexdigest(),bytes=after.st_size)
    if ((expected_sha256 is not None and result['sha256']!=expected_sha256)
            or expected_bytes is not None and result['bytes']!=expected_bytes):
        raise ValueError('export evidence differs from complete admission: '+str(path))
    return result


def _read_json(path):
    if Path(path).stat().st_size>16*1024*1024:raise ValueError('export JSON byte budget exceeded')
    def pairs(rows):
        value={}
        for key,item in rows:
            if key in value:raise ValueError('duplicate export JSON field')
            value[key]=item
        return value
    def invalid(value):raise ValueError('nonfinite export JSON value: '+value)
    return json.loads(Path(path).read_bytes(),object_pairs_hook=pairs,parse_constant=invalid)


def _source(value):
    return dict(path=value.path,sha256=value.sha256,code_key=value.code_key,qualname=value.qualname)


def _profile_artifacts(value):
    """Retain original fact artifacts; no cost or graph semantics are inferred."""
    pending=[value];seen=0
    while pending:
        row=pending.pop();seen+=1
        if seen>100000:raise ValueError('export evidence traversal budget exceeded')
        if type(row)is dict:
            if {'uri','sha256','size_bytes','schema'}<=set(row):
                if type(row['uri'])is str and Path(row['uri']).is_absolute():yield row
            pending.extend(reversed(tuple(row.values())))
        elif type(row)is list or type(row)is tuple:pending.extend(reversed(row))


def export_loop(artifact_path,out,*,selection_path):
    """Replay full admission, then copy one ELF and project its runtime guards.

    The returned expected_admission_sha256 must be trusted separately by the
    caller. Runtime consumers never need the paths in admission.evidence.
    """
    from scar.codegen.loop import load,_contract
    from scar.runtime.loop_contract import (SCHEMA,ADMISSION_SCHEMA,TRUST_POLICY,IDENTITIES,
        validate_contract,validate_admission,canonical_sha256)

    output=Path(out).absolute()
    if output.resolve()!=output or output.exists():raise ValueError('fresh canonical runtime bundle directory required')
    original=Path(artifact_path).resolve();directory=original if original.is_dir() else original.parent
    if output.is_relative_to(directory):raise ValueError('runtime export must not modify the original artifact directory')
    manifest=_receipt(directory/'artifact.json','original-native-artifact')
    artifact=load(artifact_path)  # Complete original admission, including ELF table identities.
    report=artifact.report;loop=artifact.loop;plan=artifact.execution_plan;strategy=artifact.strategy;layout=artifact.layout
    if artifact.directory.resolve()!=directory:raise ValueError('full loader returned another artifact directory')
    if (sys.platform!='linux' or platform.machine().lower() not in ('x86_64','amd64')
            or struct.calcsize('P')!=8 or sys.byteorder!='little'):
        raise ValueError('runtime export supports the admitted Linux x86_64 little-endian contract')
    evidence=[];by_path={}
    def retain(path,role,**expected):
        row=_receipt(path,role,**expected)
        old=by_path.get(row['path'])
        if old is not None:
            if (old['sha256'],old['bytes'])!=(row['sha256'],row['bytes']):
                raise ValueError('one export evidence path has conflicting content versions')
            return old
        evidence.append(row);by_path[row['path']]=row
        if len(evidence)>4096:raise ValueError('runtime export evidence receipt budget exceeded')
        return row
    retain(directory/'artifact.json',manifest['role'],expected_sha256=manifest['sha256'],expected_bytes=manifest['bytes'])
    for key,name in report['files'].items():
        retain(directory/name,'original-native-'+key,expected_sha256=report['sha256'][key])
    selection_file=Path(selection_path).absolute();selection_receipt=retain(selection_file,'selection:#')
    document=_read_json(selection_file)
    if type(document)is dict and 'selection' in document:
        selection=document['selection'];pointer='#/selection'
    else:selection=document;pointer='#'
    selection_receipt['role']='selection:'+pointer
    assumptions=list(strategy['external_assumptions'])
    if report['external_assumptions']!=assumptions:
        raise ValueError('native metadata and admitted strategy assumptions differ')
    guards=[];paths={}
    def guard(path,source):
        projected=_source(source);key=tuple(path)
        if key in paths:
            if paths[key]!=projected:raise ValueError('one runtime globals path has different code/source guards')
            return
        paths[key]=projected;guards.append(dict(environment_path=list(path),source=projected))
    body_path=tuple(loop.body_callee.lookup_path);guard(body_path,loop.body_callee.source)
    for call in loop.body.calls:guard((*body_path,*call.callee.lookup_path),call.callee.source)
    for source in (loop.source,loop.body.source,*(call.callee.source for call in loop.body.calls)):
        retain(source.path,'original-application-source',expected_sha256=source.sha256)
    # Sidecar receipts retain complete IR/profile provenance on the build side;
    # only their byte identities cross the runtime admission boundary.
    binding=plan.resource_binding
    for reference in _profile_artifacts(asdict(binding)):
        retain(reference['uri'],'original-target-fact',expected_sha256=reference['sha256'],expected_bytes=reference['size_bytes'])
    if layout is not None:
        profile=layout['hardware_profile'];profile_receipt=layout['hardware_profile_receipt']
        if profile_receipt is not None:
            retain(profile_receipt['path'],'original-hardware-profile',expected_sha256=profile_receipt['sha256'],expected_bytes=profile_receipt['bytes'])
            profile=_read_json(profile_receipt['path'])
        for reference in _profile_artifacts(profile):
            retain(reference['uri'],'original-hardware-profile-fact',expected_sha256=reference['sha256'],expected_bytes=reference['size_bytes'])
    library=report['files']['library']
    if library in ('runtime.json','admission.json','export-report.json'):raise ValueError('native library basename collides with runtime bundle metadata')
    library_bytes=(directory/library).read_bytes();library_sha256=sha256(library_bytes).hexdigest()
    if library_sha256!=report['sha256']['library']:raise ValueError('native library changed after full admission')
    ids={key:report[key] for key in sorted(IDENTITIES)}
    native_contract=_contract(loop,strategy,plan,layout)
    if canonical_sha256(native_contract)!=ids['contract_sha256']:raise ValueError('projected native contract differs from full admission')
    runtime=validate_contract(dict(schema=SCHEMA,schema_version=1,profile='cpu-f64-rank2-loop-v1',
        native=dict(library=library,library_sha256=library_sha256,loop_abi_version=report['loop_abi_version'],ids=ids,contract=native_contract),
        source=_source(loop.source),parameters=list(loop.parameters),state_parameter=loop.state_parameter,trip_count_parameter=loop.trip_count_parameter,
        callee_guards=guards,array=dict(shape=list(report['state_shape']),dtype='float64',device='cpu',layout='C',alignment_bytes=8,final_bytes=report['final_bytes']),
        trip_count=report['trip_count'],memory_budget_bytes=report['memory_budget_bytes'],
        cpu=dict(ids=list(report['cpu_ids']),threads=report['cpu_threads'],architecture='x86_64',os='linux',pointer_bits=64,byteorder='little',
            required_isa='SSE2_X86_64' if layout is not None else 'BASELINE_X86_64'),
        external_assumptions=assumptions,floating_environment_policy=report['floating_environment_policy'],selection=selection))
    runtime_bytes=_json_bytes(runtime)
    admission=validate_admission(dict(schema=ADMISSION_SCHEMA,schema_version=1,runtime_contract_sha256=sha256(runtime_bytes).hexdigest(),
        library_sha256=library_sha256,native_ids=dict(ids),producer=dict(name='SCAR',version='0.1.0',exporter_sha256=sha256(Path(__file__).read_bytes()).hexdigest()),
        evidence=evidence,trust_policy=TRUST_POLICY))
    admission_bytes=_json_bytes(admission)
    for row in evidence:_receipt(row['path'],row['role'],expected_sha256=row['sha256'],expected_bytes=row['bytes'])
    output.mkdir(parents=True)
    (output/library).write_bytes(library_bytes);(output/'runtime.json').write_bytes(runtime_bytes);(output/'admission.json').write_bytes(admission_bytes)
    result=dict(schema='scar.loop-runtime-export',schema_version=1,bundle=str(output),runtime_contract=str(output/'runtime.json'),
        admission=str(output/'admission.json'),library=str(output/library),expected_admission_sha256=sha256(admission_bytes).hexdigest(),
        runtime_contract_sha256=sha256(runtime_bytes).hexdigest(),library_sha256=library_sha256,build_id=ids['build_id'],
        admission_trust_policy=TRUST_POLICY,full_original_loader_admission=True,evidence_receipts=len(evidence),
        native_recompiled=False,target_executed=False,IR_or_profile_copied=False)
    (output/'export-report.json').write_bytes(_json_bytes(result))
    return result


__all__=['export_loop']
