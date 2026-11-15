"""Compile-only module identities and conditional attachment anchors.

STORE successors are original bytecode locations, not observed predecessors,
completed initialization or proofs that a callable/global environment exists.
No AST/source splitting, target import, execution or loop expansion is used.
"""
import dis
import json
from pathlib import Path
import re
import sys
from types import CodeType


SCHEMA='scar.entry-attachment-plan'


def plan_entry_attachment(entry,contract,*,inspection=None,max_files=256,max_bytes=8*1024*1024):
    from scar.compiler.runtime_entry import inspect_entry
    from scar.analysis.static_sources import _read_raw_source,_ByteBudget,_Changed
    from scar.runtime.loop_contract import validate_contract,canonical_sha256
    from scar.trace.code_snapshot import code_key
    from hashlib import sha256

    if type(max_files)is not int or type(max_bytes)is not int or min(max_files,max_bytes)<1:
        raise ValueError('positive exact module file/byte budgets required')
    contract=validate_contract(json.loads(json.dumps(contract,allow_nan=False)))
    entry=Path(entry).absolute()
    if inspection is None:inspection=inspect_entry(entry,contract,max_files=max_files,max_bytes=max_bytes)
    if (inspection.get('schema')!='scar.runtime-entry-inspection' or inspection.get('schema_version')!=1
            or inspection['entry']['path']!=str(entry)
            or inspection['runtime_contract_sha256']!=canonical_sha256(contract)):
        raise ValueError('original entry inspection and runtime contract must agree')
    inspection_sha=canonical_sha256(inspection);bindings=inspection['source_bindings']
    by_path={row['path']:(i,row) for i,row in enumerate(bindings)}
    files={row['path']:row for row in inspection['files']}
    body=next(g for g in contract['callee_guards'] if len(g['environment_path'])==1)
    entry_name=contract['source']['qualname'];required={}
    def need(path,name):
        if name.isidentifier():required.setdefault(path,set()).add(name)
    need(contract['source']['path'],entry_name)
    need(contract['source']['path'],body['environment_path'][0])
    for guard in contract['callee_guards']:
        if len(guard['environment_path'])==2:need(body['source']['path'],guard['environment_path'][1])
    paths=list(dict.fromkeys((str(entry),*(row['path'] for row in bindings))))
    modules=[];read_bytes=0
    for path in paths:
        reference=by_path.get(path);binding=None if reference is None else reference[1];file=files.get(path)
        owner=path==contract['source']['path'];is_entry=path==str(entry)
        mode=('ENTRY_MAIN_LOCAL_POST_STORE_CANDIDATES' if owner and is_entry else
            'IMPORTED_SOURCE_POST_EXEC_CANDIDATE' if owner else
            'ENTRY_DISPATCH_ONLY' if is_entry else 'CALLEE_SOURCE_READINESS_CANDIDATE')
        expected=inspection['entry']['sha256'] if is_entry else None if binding is None else binding['expected_sha256']
        record=dict(path=path,source_sha256=expected,module_code_key=None,mode=mode,attachment_target=owner and entry_name.isidentifier(),
            compile_recipe=dict(filename=path,mode='exec',flags=0,dont_inherit=True,optimize=sys.flags.optimize),
            source_binding_ref=None if reference is None else dict(inspection_sha256=inspection_sha,pointer='#/source_bindings/'+str(reference[0])),
            module_candidates=([dict(name='__main__',kind='ENTRY_FILE',via_imports=[])] if is_entry and binding is None
                else [] if binding is None else binding['module_candidates']),
            required_global_names=sorted(required.get(path,())),anchors=[],code_matches=[],frontiers=[])
        if owner and not entry_name.isidentifier():
            record['frontiers'].append('ENTRY_QUALNAME_IS_NOT_A_DIRECT_MODULE_BINDING')
        if file is None or file.get('sha256')!=expected or binding is not None and binding['status'] not in ('ENTRY_SOURCE','LOCAL_IMPORT_CANDIDATE'):
            record['frontiers'].append('SOURCE_OWNERSHIP_OR_VERSION_UNRESOLVED');modules.append(record);continue
        if len(modules)>=max_files:
            record['frontiers'].append('MODULE_FILE_BUDGET');modules.append(record);continue
        try:
            raw=_read_raw_source(Path(path),max_bytes-read_bytes);read_bytes+=len(raw)
            if sha256(raw).hexdigest()!=expected:raise ValueError('module source changed after entry inspection')
            code=compile(raw,path,'exec',flags=0,dont_inherit=True,optimize=sys.flags.optimize)
            record['module_code_key']=code_key(code)
            instructions=tuple(dis.get_instructions(code,adaptive=False,show_caches=False))
            first_call=next((i.offset for i in instructions if i.opname.startswith('CALL')),None)
            for index,instruction in enumerate(instructions[:-1]):
                if instruction.opname not in ('STORE_NAME','STORE_GLOBAL') or instruction.argval not in record['required_global_names']:continue
                following=instructions[index+1]
                record['anchors'].append(dict(store_offset=instruction.offset,after_offset=following.offset,
                    store_opname=instruction.opname,binding=instruction.argval,after_is_jump_target=following.is_jump_target,
                    lexically_before_first_explicit_CALL=first_call is None or following.offset<=first_call,
                    predecessor_execution_proven=False,guard_readiness_proven=False))
            wanted={} if binding is None else {f['code_key']:f for f in binding['functions']}
            names={f['qualname'] for f in wanted.values()};pending=[(code,())];seen=set()
            while pending:
                child,constant_path=pending.pop()
                if id(child) in seen:continue
                seen.add(id(child))
                if len(seen)>10000:raise ValueError('module code tree budget exceeded')
                if child.co_qualname in names:
                    key=code_key(child)
                    if key in wanted:record['code_matches'].append(dict(code_key=key,qualname=child.co_qualname,constant_path=list(constant_path)))
                pending.extend((value,(*constant_path,i)) for i,value in enumerate(child.co_consts) if type(value)is CodeType)
            found={r['code_key'] for r in record['code_matches']}
            if set(wanted)-found:record['frontiers'].append('CONTRACT_CODE_CONSTANT_NOT_MATCHED_BY_THIS_COMPILE_RECIPE')
            if owner and not record['anchors']:record['frontiers'].append('NO_REQUIRED_MODULE_STORE_SUCCESSOR')
        except (OSError,ValueError,SyntaxError,OverflowError,_ByteBudget,_Changed) as error:
            record['module_code_key']=None;record['anchors']=[];record['code_matches']=[]
            record['frontiers'].append('MODULE_COMPILE_IDENTITY_UNAVAILABLE:'+type(error).__name__)
        modules.append(record)
    result=dict(schema=SCHEMA,schema_version=1,entry=str(entry),runtime_contract_sha256=canonical_sha256(contract),
        inspection_sha256=inspection_sha,interpreter=dict(implementation=sys.implementation.name,version=list(sys.version_info[:3]),cache_tag=sys.implementation.cache_tag),
        limits=dict(max_files=max_files,max_bytes=max_bytes),modules=modules,
        runtime_requirements=['EXACT_MODULE_CODE_ID_AND_SOURCE_SHA','ACTUAL_STORE_PREDECESSOR_AND_NORMAL_SUCCESSOR_WITNESS',
            'EXACT_MODULE_FRAME_AND_GLOBALS_NAMESPACE','ORIGINAL_ENTRY_AND_ALL_HIERARCHICAL_CALLEE_CODE_AND_ENVIRONMENT_GUARDS',
            'GUARD_READINESS_AND_CALL_ORDER_RECHECKED_AT_ACTUAL_EXECUTION','NO_STATIC_IMPORT_OR_INITIALIZATION_COMPLETION_INFERENCE'],
        target_executed=False,application_imported=False,source_rewritten=False,AST_split=False,source_bodies_copied=False,loops_unrolled=False,
        predecessor_execution_proven=False,guard_readiness_proven=False,attachment_before_first_invocation_proven=False,first_stage_complete=False)
    return validate_entry_attachment(result)


def validate_entry_attachment(plan):
    if (type(plan)is not dict or plan.get('schema')!=SCHEMA or type(plan.get('schema_version'))is not int or plan['schema_version']!=1
            or plan.get('predecessor_execution_proven')is not False or plan.get('guard_readiness_proven')is not False
            or plan.get('attachment_before_first_invocation_proven')is not False):
        raise ValueError('conditional original-module attachment plan required')
    for module in plan['modules']:
        key=module['module_code_key']
        if key is not None and re.fullmatch('pycode-v2:[0-9a-f]{64}',key)is None:raise ValueError('full module CodeID required')
        if key is None and module['anchors']:raise ValueError('unknown module identity cannot supply offsets')
        for anchor in module['anchors']:
            if (type(anchor['store_offset'])is not int or type(anchor['after_offset'])is not int
                    or not 0<=anchor['store_offset']<anchor['after_offset']
                    or anchor['store_opname'] not in ('STORE_NAME','STORE_GLOBAL')
                    or anchor['binding'] not in module['required_global_names']
                    or anchor['predecessor_execution_proven']is not False or anchor['guard_readiness_proven']is not False):
                raise ValueError('original STORE and conditional normal successor required')
    return plan


def verify_entry_attachment(entry,contract,plan,*,inspection=None):
    """Build-side source/CodeID/offset replay, never runtime execution proof."""
    from scar.runtime.loop_contract import canonical_sha256
    validate_entry_attachment(plan)
    expected=plan_entry_attachment(entry,contract,inspection=inspection,**plan['limits'])
    if canonical_sha256(expected)!=canonical_sha256(plan):raise ValueError('entry attachment differs from original module replay')
    return True


__all__=['plan_entry_attachment','validate_entry_attachment','verify_entry_attachment']
