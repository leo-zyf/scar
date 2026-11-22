"""Replay full stateful admission and export the unchanged kernel with guards."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import platform
import struct
import sys

from .export_loop import _json_bytes, _read_json, _receipt, _source, _profile_artifacts


def project_stateful_contract(artifact,selection=None,*,require_selection=False):
    """Build-side compact projection also used by the full-artifact guard facade."""
    from scar.runtime.stateful_contract import SCHEMA,MEMORY_SCOPE,FP_POLICY,validate_contract
    region,plan,report=artifact.region,artifact.plan,artifact.report
    binding=plan['resource_binding']; written={action.target for action in region.actions if action.kind=='write'}
    for key in ('external_array_bytes','workspace_payload_bytes','total_region_array_payload_bytes','memory_budget_bytes','cpu_threads'):
        if report[key]!=plan[key]: raise ValueError('native metadata differs from admitted plan: '+key)
    if report['cpu_ids']!=list(binding['cpu_ids']) or report['external_assumptions']!=plan['external_assumptions']:
        raise ValueError('native metadata differs from CPU/state assumptions')
    inputs=[]
    for row in region.inputs:
        shape=list(plan['input_shapes'][row.id]) if row.kind=='array' else None
        inputs.append(dict(id=row.id,path=list(row.path),kind=row.kind,shape=shape,
            nbytes=None if shape is None else shape[0]*shape[1]*8,written=row.id in written))
    ids={key:report[key] for key in ('build_id','region_sha256','plan_sha256','contract_sha256')}
    core={key:report[key] for key in ('region_sha256','plan_sha256','provider_runtime')}
    value=dict(schema=SCHEMA,schema_version=1,profile='cpu-f64-rank2-stateful-v1',
        native=dict(library=report['files']['library'],library_sha256=report['sha256']['library'],
            stateful_abi_version=report['stateful_abi_version'],ids=ids,contract=core),
        source=_source(region.source),parameters=list(region.parameter_names),inputs=inputs,
        callee_guards=[dict(path=list(row.path),kind=row.kind,source=_source(row.source)) for row in region.callees],
        providers=[dict(id=row.id,module=row.module,symbol=row.symbol,lookup_path=list(row.lookup_path),semantics=row.semantics) for row in region.providers],
        cpu=dict(ids=list(binding['cpu_ids']),threads=plan['cpu_threads'],architecture='x86_64',os='linux',pointer_bits=64,
            byteorder='little',required_isa='BASELINE_X86_64'),
        memory=dict(**{key:plan[key] for key in ('external_array_bytes','workspace_payload_bytes','total_region_array_payload_bytes','memory_budget_bytes')},
            scope=MEMORY_SCOPE,policy='UNBOUND'),provider_runtime=deepcopy(report['provider_runtime']),
        guards=list(region.guards),external_assumptions=list(plan['external_assumptions']),
        floating_environment_policy=FP_POLICY,selection=deepcopy(selection))
    return validate_contract(value,require_selection=require_selection)


def _replay_selection(artifact,selection,retain):
    """Recompute the measured selector from retained scalar validation receipts."""
    from scar.optimization.selection import select_stateful_measured
    evidence=selection.get('evidence')
    if type(evidence) is not dict or set(evidence)!={'path','sha256'}:
        raise ValueError('stateful selection requires its retained validation evidence')
    retain(evidence['path'],'original-measured-validation',expected_sha256=evidence['sha256'])
    measurement=_read_json(evidence['path'])
    if (measurement.get('native_execution_validated') is not True
            or not measurement.get('correctness') or not all(value is True for value in measurement['correctness'].values())
            or measurement.get('original_source_modified') is not False):
        raise ValueError('stateful measured correctness evidence is incomplete')
    report=artifact.report
    if any(measurement.get('artifact',{}).get(key)!=report[key] for key in ('build_id','region_sha256','plan_sha256','contract_sha256')):
        raise ValueError('stateful validation evidence belongs to another artifact')
    episodes=measurement.get('episodes')
    if type(episodes) is not list or not episodes: raise ValueError('stateful validation episodes required')
    fallbacks=0
    for row in episodes:
        if any(row.get(key) is not True for key in ('bitwise_state','stdout_equal','file_bytes_equal')):
            raise ValueError('stateful validation episode failed')
        stats=row.get('native_stats')
        if type(stats) is not list or not stats or sum(item.get('native_calls',0) for item in stats)<=0:
            raise ValueError('stateful execution was not observed')
        fallbacks+=sum(item['fallback_calls'] for item in stats)
    replay=select_stateful_measured(artifact,measurement['baseline_seconds']['samples'],
        measurement['optimized_seconds']['samples'],validated=True,native_fallbacks=fallbacks,
        minimum_gain=selection['minimum_gain'], strict_minimum_gain=selection.get('strict_minimum_gain', False))
    if any(selection.get(key)!=value for key,value in replay.items()):
        raise ValueError('stateful measured selection differs from retained sample replay')
    return selection


def export_stateful(artifact,out,*,selection_path):
    from scar.codegen.stateful import load
    from scar.runtime.stateful_contract import (ADMISSION_SCHEMA,TRUST_POLICY,validate_admission,
                                               canonical_sha256)
    output=Path(out).absolute()
    if output.resolve()!=output or output.exists(): raise ValueError('fresh canonical stateful bundle directory required')
    path=Path(artifact.directory if hasattr(artifact,'directory') else artifact).resolve()
    directory=path if path.is_dir() else path.parent
    if output.is_relative_to(directory): raise ValueError('stateful export must not modify original artifact')
    if sys.platform!='linux' or platform.machine().lower() not in ('x86_64','amd64') or struct.calcsize('P')!=8 or sys.byteorder!='little':
        raise ValueError('stateful runtime export supports Linux x86_64 little-endian only')
    evidence=[]; retained={}
    def retain(path,role,**expected):
        row=_receipt(path,role,**expected)
        if row['path'] in retained:
            old=retained[row['path']]
            if (row['sha256'],row['bytes'])!=(old['sha256'],old['bytes']): raise ValueError('conflicting stateful evidence versions')
            return old
        evidence.append(row); retained[row['path']]=row
        if len(evidence)>4096: raise ValueError('stateful export evidence budget exceeded')
        return row
    retain(directory/'artifact.json','original-native-manifest')
    full=load(directory); report=full.report
    for key,name in report['files'].items(): retain(directory/name,'original-native-'+key,expected_sha256=report['sha256'][key])
    document=_read_json(selection_path); selection=document['selection'] if type(document) is dict and 'selection' in document else document
    retain(selection_path,'original-measured-selection')
    selection=_replay_selection(full,selection,retain)
    runtime=project_stateful_contract(full,selection,require_selection=True)
    for source in (full.region.source,*(row.source for row in full.region.callees)):
        retain(source.path,'original-source-snapshot',expected_sha256=source.sha256)
    for reference in _profile_artifacts(full.plan['resource_binding']):
        retain(reference['uri'],'original-target-fact',expected_sha256=reference['sha256'],expected_bytes=reference['size_bytes'])
    provider=runtime['provider_runtime']['loaded_binary']
    retain(provider['path'],'original-NumPy-provider-binary',expected_sha256=provider['sha256'],expected_bytes=provider['bytes'])
    name=runtime['native']['library']
    if name in ('runtime.json','admission.json','export-report.json'): raise ValueError('native basename collides with bundle metadata')
    library=(directory/name).read_bytes(); _need=runtime['native']['library_sha256']
    if sha256(library).hexdigest()!=_need: raise ValueError('stateful library changed after admission')
    raw=_json_bytes(runtime)
    admission=validate_admission(dict(schema=ADMISSION_SCHEMA,schema_version=1,
        runtime_contract_sha256=sha256(raw).hexdigest(),library_sha256=_need,native_ids=runtime['native']['ids'],
        producer=dict(name='SCAR',version='0.1.0',exporter_sha256=sha256(Path(__file__).read_bytes()).hexdigest()),
        evidence=evidence,trust_policy=TRUST_POLICY))
    admission_raw=_json_bytes(admission)
    for row in evidence: _receipt(row['path'],row['role'],expected_sha256=row['sha256'],expected_bytes=row['bytes'])
    output.mkdir(parents=True); (output/name).write_bytes(library)
    (output/'runtime.json').write_bytes(raw); (output/'admission.json').write_bytes(admission_raw)
    result=dict(schema='scar.stateful-runtime-export',schema_version=1,bundle=str(output),
        expected_admission_sha256=sha256(admission_raw).hexdigest(),runtime_contract_sha256=sha256(raw).hexdigest(),
        library_sha256=_need,**runtime['native']['ids'],full_original_loader_admission=True,
        measured_selection_replayed=True,native_recompiled=False,target_executed=False,IR_or_profile_copied=False,
        evidence_receipts=len(evidence),fresh_process_cost_qualified=False,admission_trust_policy=TRUST_POLICY)
    (output/'export-report.json').write_bytes(_json_bytes(result))
    return result


def main():
    import argparse
    import json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact',required=True)
    parser.add_argument('--selection',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    print(json.dumps(export_stateful(args.artifact,args.out,selection_path=args.selection)))


__all__=['export_stateful','project_stateful_contract','main']


if __name__=='__main__': main()
