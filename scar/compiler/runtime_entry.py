"""Bounded static source ownership for an entry and a runtime loop contract.

Reuse the existing source import discovery. Candidate module names are never
runtime loader, alias, execution, initialization or callable identity proofs.
"""
from collections import Counter,defaultdict
from hashlib import sha256
import argparse
import io
import json
from pathlib import Path


def inspect_entry(entry,contract,*,max_files=256,max_bytes=8*1024*1024):
    """Project local AST/import candidates without importing application code."""
    from scar.analysis.source_discovery import discover_sources
    from scar.analysis.static_sources import _read_raw_source
    from scar.runtime.loop_contract import validate_contract,canonical_sha256

    contract=validate_contract(json.loads(json.dumps(contract,allow_nan=False)))
    entry=Path(entry).absolute()
    if entry.resolve()!=entry or entry.is_symlink() or not entry.is_file() or entry.suffix!='.py':
        raise ValueError('entry requires an original canonical regular Python file')
    # Explicit roots skip the existing discoverer's recursive root inference.
    # Contract source paths are not additional discovery seeds or sys.path.
    discovery=discover_sources(entry,project_root=entry.parent,source_roots=(entry.parent,),
        max_files=max_files,max_bytes=max_bytes,max_directories=1)
    unresolved=[];files=[];by_path={};rechecked_bytes=0
    for row in discovery.files:
        path=Path(row.snapshot.path);error=None;digest=None;size=None
        try:
            raw=_read_raw_source(path,max_bytes-rechecked_bytes);size=len(raw);rechecked_bytes+=size
            text=io.TextIOWrapper(io.BytesIO(raw),encoding=row.snapshot.encoding).read()
            if 'sha256:'+sha256(text.encode()).hexdigest()!=row.snapshot.fingerprint:
                raise ValueError('source changed after original AST/import discovery')
            digest=sha256(raw).hexdigest()
        except Exception as caught:error=type(caught).__name__+': '+str(caught)
        item=dict(path=str(path),sha256=digest,bytes=size,module_names=list(row.module_names),
            is_entry=path==entry,status='READ_STABLE' if error is None else 'SOURCE_RECHECK_FAILED')
        files.append(item);by_path[str(path)]=item
        if error is not None:unresolved.append(dict(field='files.'+str(path),reason='SOURCE_RECHECK_FAILED',detail=error))
    imports=[];incoming=defaultdict(list)
    for index,row in enumerate(discovery.imports):
        candidates=[]
        for candidate in row.candidates:
            candidates.append(dict(module=candidate.name,path=candidate.path,role=candidate.role.value,
                namespace=candidate.namespace,parent_initializers=list(candidate.parent_initializers)))
            for path in (candidate.path,*candidate.parent_initializers):
                if index not in incoming[path]:incoming[path].append(index)
        imports.append(dict(index=index,path=row.source.path,line=row.source.start_line,
            requested=row.requested,relative_level=row.relative_level,symbol=row.symbol,bound_name=row.bound_name,
            importer_module=row.importer_module,resolved_module=row.resolved_module,status=row.status.value,
            deferred=row.deferred,owner=row.owner_qualname,
            control_context=[dict(kind=c.kind.value,line=c.source.start_line,index=c.index) for c in row.control_context],
            candidates=candidates))
        if row.source.path==str(entry) and row.relative_level:
            unresolved.append(dict(field='imports.'+str(index),reason='ENTRY_RELATIVE_IMPORT_PACKAGE_CONTEXT_NOT_ESTABLISHED',
                detail='A file executed as __main__ does not establish its physical package mapping as __package__.'))
    for gap in discovery.gaps:
        unresolved.append(dict(field='imports',reason=gap.kind.value,path=gap.path,
            line=None if gap.source is None else gap.source.start_line,detail=gap.detail))
    sources={}
    def source(value,role,path):
        selected=sources.setdefault(value['path'],dict(path=value['path'],expected_sha256=value['sha256'],functions=[]))
        if selected['expected_sha256']!=value['sha256']:
            raise ValueError('one runtime source path names conflicting source versions')
        selected['functions'].append(dict(role=role,environment_path=path,qualname=value['qualname'],code_key=value['code_key']))
    source(contract['source'],'ENTRY_FUNCTION',[])
    for guard in contract['callee_guards']:source(guard['source'],'CALLEE_GUARD',list(guard['environment_path']))
    bindings=[]
    for path,row in sources.items():
        file=by_path.get(path);modules=[];status='UNRESOLVED_SOURCE_OWNER'
        if file is not None and file['sha256'] is not None:
            if file['sha256']!=row['expected_sha256']:status='SOURCE_VERSION_MISMATCH'
            else:
                if path==str(entry):modules.append(dict(name='__main__',kind='ENTRY_FILE',via_imports=[]))
                if path!=str(entry) or incoming[path]:
                    modules.extend(dict(name=name,kind='LOCAL_IMPORT_CANDIDATE',via_imports=incoming[path]) for name in file['module_names'])
                status='ENTRY_SOURCE' if path==str(entry) else 'LOCAL_IMPORT_CANDIDATE' if modules else status
        binding=dict(**row,observed_sha256=None if file is None else file['sha256'],status=status,module_candidates=modules,
            code_identity_verified=False,runtime_module_selected=False)
        bindings.append(binding)
        if status not in ('ENTRY_SOURCE','LOCAL_IMPORT_CANDIDATE'):
            unresolved.append(dict(field='source_bindings.'+path,reason=status,
                detail='Only source files reached through the bounded original entry/import candidate closure can supply ownership.'))
    unresolved.extend((
        dict(field='runtime.import_search',reason='ACTUAL_SYS_PATH_META_PATH_CUSTOM_LOADERS_AND_PRELOADED_MODULES_NOT_OBSERVED'),
        dict(field='runtime.module_initialization',reason='CONDITIONAL_DEFERRED_OR_CIRCULAR_IMPORT_COMPLETION_NOT_PROVEN'),
        dict(field='runtime.callable_aliases',reason='PRELOADED_ALIASES_AND_REBINDINGS_REQUIRE_ACTUAL_SOURCE_CODE_AND_GLOBAL_GUARDS')))
    return dict(schema='scar.runtime-entry-inspection',schema_version=1,
        status='SOURCE_OWNERS_IDENTIFIED' if all(b['status'] in ('ENTRY_SOURCE','LOCAL_IMPORT_CANDIDATE') for b in bindings) else 'UNRESOLVED_SOURCE_OWNERS',
        entry=dict(path=str(entry),sha256=by_path.get(str(entry),{}).get('sha256'),execution_name='__main__'),
        runtime_contract_sha256=canonical_sha256(contract),source_roots=list(discovery.source_roots),
        limits=dict(max_files=max_files,max_bytes=max_bytes,scope='UNIQUE_LOCAL_SOURCE_FILES',source_root_policy='ENTRY_DIRECTORY_ONLY_NO_ROOT_SCAN'),
        files=files,imports=imports,source_bindings=bindings,unresolved=unresolved,
        counts=dict(files=len(files),source_bytes=sum(f['bytes'] or 0 for f in files),imports=len(imports),
            import_statuses=dict(Counter(r['status'] for r in imports)),contract_source_files=len(bindings),
            source_binding_statuses=dict(Counter(r['status'] for r in bindings))),
        admission_trust_checked=False,target_executed=False,application_imported=False,native_library_loaded=False,
        site_packages_scanned=False,recursive_source_root_scan=False,source_bodies_copied=False,loops_unrolled=False,
        runtime_import_result_proven=False,callable_identity_proven=False,first_stage_complete=False)


def main(argv=None):
    parser=argparse.ArgumentParser(prog='scar inspect-entry',description=__doc__)
    parser.add_argument('--entry',required=True);parser.add_argument('--contract',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args(argv);output=Path(args.out).absolute()
    if output.exists():raise ValueError('fresh runtime entry inspection output required')
    contract=json.loads(Path(args.contract).read_bytes());report=inspect_entry(args.entry,contract)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],counts=report['counts'],out=str(output))))
    return 0


__all__=['inspect_entry','main']
