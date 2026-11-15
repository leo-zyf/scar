"""Select and join bounded CALL observations through original F4 call ports.

The existing observed-callsite mapper supplies bytecode/source correspondence.
Compiled retained F0 is identity evidence only; neither a target nor a new AST
semantic interpreter runs here. A CALL attempt never proves entry/completion.
"""
from collections import defaultdict
from hashlib import sha256
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace as NS

from scar.trace.lean_capture import LeanCapturePlan


SCHEMA='scar.call-binding-plan'
OBSERVATION_SCHEMA='scar.call-binding-observations'


def plan_digest(plan):
    return sha256(json.dumps(plan,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _receipt(path):
    path=Path(path).absolute();raw=path.read_bytes()
    return dict(path=str(path),bytes=len(raw),sha256=sha256(raw).hexdigest())


def _static_call_mapping(view,unit):
    """Load the original static mapper only for explicit planning/replay."""
    from scar.analysis.source_observed_calls import ObservedSourceCallsiteIndex
    from scar.analysis.source_reference_slice import ReferenceSlicer,source_port
    from scar.analysis.source_namespace_interfaces import NamespacePlanView
    from scar.analysis.source_declared_calls import _owners

    class _StaticCallMapping:
        """Read-only mapper inputs, without invented runtime trace records."""
        _source_syntax=ObservedSourceCallsiteIndex._source_syntax
        _index=ObservedSourceCallsiteIndex._index

        def __init__(self):
            self.view=view;self._syntax={};self._roles={};self._static={}
            source=view.unit(unit);facts=source.facts
            self._owners={unit:_owners(facts)}
            self._calls={unit:tuple((i,c,source_port(facts,c.result)) for i,c in enumerate(facts.calls))}
            keys=tuple(dict.fromkeys(gp.port.template for _,_,gp in self._calls[unit]))
            self._plans={unit:source.material.plans(keys)}
            self._contexts={unit:{key:NamespacePlanView(unit,plan) for key,plan in self._plans[unit].items()}}
            self.slicer=ReferenceSlicer(NS(fact_store=view.catalog.fact_store))

    return _StaticCallMapping()


def plan_call_bindings(catalog,source,*,entry=None,call_indices=None,max_targets_per_site=2):
    """Derive local code/offset selectors for original calls in selected bodies.

    entry is an explicit declaration selector, not an optimization classifier.
    Unmapped/ambiguous operations remain per-field frontiers. Existing F2 and
    original F4 ABI checks in ObservedSourceCallsiteIndex own correspondence.
    """
    from scar.analysis.program_view import ProgramView
    from scar.analysis.source_reference_slice import source_port
    from scar.compiler.graph_frontend import _source_identity,_metadata
    from scar.ir.program.call_edges import CapturedCode
    from scar.ir.record_codec import decode,encode
    from scar.ir.semantics_v2 import SourceSnapshot
    from scar.trace.code_snapshot import describe_code,site
    from scar.trace.lean_capture import LeanBodyPlan

    if type(max_targets_per_site)is not int or max_targets_per_site not in (1,2):
        raise ValueError('each selected site needs an exact one/two observation budget')
    if call_indices is not None:
        call_indices=frozenset(call_indices)
        if any(type(i)is not int or i<0 for i in call_indices):raise ValueError('original call indices required')
    view=catalog if isinstance(catalog,ProgramView) else ProgramView(catalog)
    path=str(Path(source).absolute());units=tuple(sorted(view.catalog.units_for_source(path) or ()))
    if not units:raise ValueError('source is outside the admitted catalog')
    sites=[];frontiers=[];bodies=[];declared=set()
    for unit in units:
        original=view.unit(unit);facts=original.facts
        model,syntax,codes,_,digest=_source_identity(view.catalog,unit)
        mapper=_static_call_mapping(view,unit)
        selected=[]
        for fn in facts.functions:
            name,span,lines=_metadata(syntax,fn.template.occurrence)
            if entry is not None and entry!=name and not entry.endswith('.'+name):continue
            matches=[c for c in codes if c.co_name==name and c.co_firstlineno in lines
                and c.co_varnames[:c.co_argcount+c.co_kwonlyargcount]==fn.parameter_names]
            if len(matches)!=1:
                frontiers.append(dict(unit=unit,field='caller.code_key',reason='UNIQUE_ORIGINAL_CODE_ID_UNAVAILABLE',template=encode(fn.template)));continue
            selected.append((fn.body_scope,matches[0]))
        if entry=='<module>':selected=[(facts.root_scope,codes[0])]
        for scope,code in selected:
            if scope is None:continue
            # The mapper only reads capture.code.key. This descriptor adapter
            # is deliberately not a CodeCapture or claimed runtime event.
            descriptor=decode(CapturedCode,describe_code(code))
            caller=NS(unit=unit,scope=scope,code=code,capture=NS(code=descriptor))
            mapping=mapper._index(caller);byquery=defaultdict(list)
            wanted={i for i,owners in enumerate(mapper._owners[unit]) if scope.owner in owners
                and (call_indices is None or i in call_indices)}
            declared.update((unit,i) for i in wanted)
            for offset,rows in mapping.items():
                unique={(i,rule):(i,evidence,owner,rule) for i,evidence,owner,rule in rows}
                queries={i for i,_ in unique}
                if len(queries)!=1:
                    for i in queries&wanted:frontiers.append(dict(unit=unit,query=i,offset=offset,field='source_call',reason='CALL_OFFSET_SOURCE_AMBIGUOUS'))
                    continue
                i=next(iter(queries))
                if i not in wanted:continue
                row=next(iter(unique.values()));byquery[i].append((offset,row))
            offsets=[]
            for query in sorted(wanted):
                rows=byquery[query]
                if len(rows)!=1:
                    frontiers.append(dict(unit=unit,query=query,field='caller.offset',reason='UNIQUE_ORIGINAL_CALL_OFFSET_UNAVAILABLE'));continue
                offset,(_,evidence,owner,rule)=rows[0];fact=facts.calls[query]
                gp=source_port(facts,fact.result)
                sites.append(dict(id=unit+':call:'+str(query),code_key=descriptor.key,offset=offset,site=site(code,offset),
                    unit=unit,query=query,original_step=gp.port.owner.wire,original_template=encode(gp.port.template),
                    callee_port=fact.callee,result_port=fact.result,world_port=fact.world,result_world_port=fact.result_world,
                    fact_reference=encode(evidence),mapping_rule=rule,code_scope=encode(owner)))
                offsets.append(offset)
            if offsets:
                snapshot=SourceSnapshot(model['path'],'sha256:'+digest,'utf-8',
                    str(sys.version_info.major)+'.'+str(sys.version_info.minor))
                bodies.append(LeanBodyPlan(snapshot,descriptor,tuple(sorted(offsets)),()))
    if entry is not None and not declared:frontiers.append(dict(field='entry',reason='NO_ORIGINAL_CALLS_IN_SELECTED_DECLARATION'))
    namespace=str(Path(path).parent)
    selector=None if not bodies else LeanCapturePlan(namespace,tuple(bodies),max_codes=64).to_dict()
    plan=dict(schema=SCHEMA,schema_version=1,max_targets_per_site=max_targets_per_site,selector=selector,
        catalog=_receipt(view.catalog.path/'result.json'),sites=sites,frontiers=frontiers,
        source_compilation_role='RETAINED_F0_CODE_IDENTITY_ONLY',semantic_authority='ORIGINAL_F4_CALL_AND_WORLD_PORTS',
        target_executed=False,first_stage_complete=False)
    return validate_call_binding_plan(plan)


def validate_call_binding_plan(plan):
    if (type(plan)is not dict or plan.get('schema')!=SCHEMA or plan.get('schema_version')!=1
            or type(plan.get('max_targets_per_site'))is not int or plan['max_targets_per_site'] not in (1,2)
            or type(plan.get('sites'))is not list or len(plan['sites'])>512):
        raise ValueError('bounded versioned call-binding plan required')
    if not plan['sites']:
        if plan.get('selector')is not None:raise ValueError('empty plan must retain an empty selector')
        return plan
    selector=LeanCapturePlan.from_dict(plan['selector'],max_codes=64)
    offsets={(body.code.key,offset) for body in selector.bodies for offset in body.call_offsets}
    actual=set();ids=set()
    for row in plan['sites']:
        key=row['code_key'],row['offset']
        if (type(row['id'])is not str or row['id'] in ids or key in actual
                or type(row['offset'])is not int or row['offset']<0
                or row['site']['code']!=row['code_key'] or row['site']['offset']!=row['offset']):
            raise ValueError('one original F4 call per exact code/offset selector required')
        actual.add(key);ids.add(row['id'])
    if actual!=offsets:raise ValueError('CALL sites differ from the original lean selector')
    return plan


def plan_program_call_bindings(catalog, sources=None, *, include_modules=True,
                               max_targets_per_site=1):
    """One bounded selector across admitted source units, without running them.

    File boundaries organize immutable source material, not the execution
    graph. Each selected site retains its original unit/CALL/world identity;
    the collector disables that site after its own observation budget.
    """
    from scar.analysis.program_view import ProgramView
    from scar.trace.lean_capture import LeanBodyPlan
    from scar.ir.record_codec import decode
    view=catalog if isinstance(catalog,ProgramView) else ProgramView(catalog)
    if sources is None:
        store=view.catalog.fact_store
        paths={store.model_units.get(unit,{}).get('path',row.get('path'))
            for unit,row in store.rows.items()}
        if any(type(path)is not str or not path for path in paths):
            raise ValueError('every admitted source unit requires its original path')
    else:
        paths={str(Path(path).absolute()) for path in sources}
    if not paths:raise ValueError('at least one admitted source unit required')
    sites={};bodies={};frontiers=[];receipt=None
    for path in sorted(paths):
        for entry in ((None,'<module>') if include_modules else (None,)):
            plan=plan_call_bindings(view,path,entry=entry,max_targets_per_site=max_targets_per_site)
            if receipt is None:receipt=plan['catalog']
            if receipt!=plan['catalog']:raise ValueError('source catalog changed while selecting CALL sites')
            frontiers.extend(dict(issue,source_path=path) for issue in plan['frontiers'])
            for row in plan['sites']:
                previous=sites.setdefault(row['id'],row)
                if previous!=row:raise ValueError('one original CALL has conflicting source selectors')
            if plan['selector'] is not None:
                for raw in plan['selector']['bodies']:
                    body=decode(LeanBodyPlan,raw);previous=bodies.setdefault(body.code.key,body)
                    if previous!=body:raise ValueError('one original code has conflicting CALL selectors')
    namespace=os.path.commonpath([str(Path(path).parent) for path in paths])
    selector=None if not bodies else LeanCapturePlan(namespace,tuple(bodies.values()),max_codes=64).to_dict()
    result=dict(schema=SCHEMA,schema_version=1,max_targets_per_site=max_targets_per_site,
        selector=selector,catalog=receipt,sites=list(sites.values()),frontiers=frontiers,
        sources=sorted(paths),module_bodies_included=include_modules,
        source_selection='ORIGINAL_ADMITTED_CATALOG_UNITS',
        source_compilation_role='RETAINED_F0_CODE_IDENTITY_ONLY',
        semantic_authority='ORIGINAL_F4_CALL_AND_WORLD_PORTS',
        target_executed=False,first_stage_complete=False)
    return validate_call_binding_plan(result)


def write_call_binding_plan(plan,path):
    validate_call_binding_plan(plan);Path(path).write_text(json.dumps(plan,indent=2,allow_nan=False)+'\n')


def read_call_binding_plan(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>2_000_000:
        raise ValueError('bounded regular call-binding plan required')
    return validate_call_binding_plan(json.loads(path.read_bytes()))


def join_call_bindings(plan,observations):
    """Join exact attempt records; no entry, return, storage or effect inference."""
    plan=read_call_binding_plan(plan) if isinstance(plan,(str,Path)) else validate_call_binding_plan(plan)
    observations=[observations] if isinstance(observations,(dict,str,Path)) else list(observations)
    sites={row['id']:row for row in plan['sites']};joined=[];seen=set();counts=defaultdict(int);documents=[]
    for raw in observations:
        doc=json.loads(Path(raw).read_bytes()) if isinstance(raw,(str,Path)) else raw
        if (doc.get('schema')!=OBSERVATION_SCHEMA or doc.get('schema_version')!=1
                or doc.get('plan_sha256')!=plan_digest(plan)):
            raise ValueError('observation belongs to another versioned call-binding plan')
        pid=doc['process_id'];local=defaultdict(int)
        for row in doc['records']:
            original=sites[row['site_id']];key=pid,row['index']
            if key in seen or (row['caller_code_key'],row['offset'])!=(original['code_key'],original['offset']):
                raise ValueError('CALL attempt repeats or differs from the original code/offset')
            seen.add(key);local[row['site_id']]+=1;counts[row['site_id']]+=1
            if local[row['site_id']]>plan['max_targets_per_site']:raise ValueError('site observation budget exceeded')
            joined.append(dict(process_id=pid,attempt_index=row['index'],thread_id=row['thread_id'],site_id=row['site_id'],
                original_call=dict(unit=original['unit'],query=original['query'],step=original['original_step'],
                    template=original['original_template'],result_port=original['result_port'],fact_reference=original['fact_reference']),
                caller_code_key=row['caller_code_key'],offset=row['offset'],target=row['target'],
                entry_completion_content_and_effects_proven=False))
        if dict(local)!={key:value for key,value in doc['site_counts'].items() if value}:
            raise ValueError('raw site counts differ from retained attempt records')
        documents.append(dict(process_id=pid,status=doc['status'],errors=doc['errors']))
    return dict(schema='scar.call-binding-join',schema_version=1,plan_sha256=plan_digest(plan),observations=documents,
        planned_sites=len(sites),observed_sites=len(counts),attempts=len(joined),bindings=joined,
        unobserved_sites=[key for key in sites if not counts[key]],static_frontiers=plan['frontiers'],
        remaining_fields=['CALL_ATTEMPT_IS_NOT_ENTRY_OR_COMPLETION','ARGUMENT_LAYOUT_AND_DEFAULTS_NOT_CAPTURED',
            'NO_ARRAY_CONTENT_OR_LOGICAL_VERSION','NATIVE_SCHEMA_AND_EFFECT_FIELDS_REQUIRE_SEPARATE_EVIDENCE',
            'UNOBSERVED_PATHS_AND_FUTURE_TARGETS_OPEN'],loops_unrolled=False,first_stage_complete=False)


__all__=['plan_call_bindings','plan_program_call_bindings','plan_digest','validate_call_binding_plan','write_call_binding_plan','read_call_binding_plan','join_call_bindings']
