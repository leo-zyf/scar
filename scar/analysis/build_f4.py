"""Assemble existing F4 software, evidence and resource layers in one session.

The manifest references publications; it is not another source graph. A
missing component is a reported gap, never an empty dependency proof.
"""
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time

from scar.ir.program.dependencies import DependencyPoint as Point,DependencyNeighborhood as Neighborhood
from scar.ir.program.dependencies import NODE_TYPES
from scar.ir.record_codec import encode


def _bounded_call_binding_inputs(paths):
    """Reference original metadata publications; no events are copied or traced."""
    from scar.analysis.build import _receipt
    result=[]
    for path in dict.fromkeys(Path(p).resolve() for p in paths):
        file=path/'result.json'
        if file.is_symlink() or not file.is_file() or file.stat().st_size>2_000_000:
            raise ValueError('bounded CALL bindings require an original bounded run result')
        run=json.loads(file.read_bytes())
        if (run.get('schema')!='scar.call-binding-run' or run.get('schema_version')!=1
                or run.get('global_profile_hook')is not False or run.get('tensor_contents_recorded')is not False
                or run.get('first_stage_complete')is not False):
            raise ValueError('bounded CALL run is not the separate metadata-only capture channel')
        raw=run.get('raw_observations');plan=run.get('plan');join=run.get('join')
        if (type(plan)is not str or not Path(plan).is_absolute() or type(raw)is not list or not raw
                or len(raw)>128 or any(type(p)is not str or not Path(p).is_absolute() for p in raw)
                or len(set(raw))!=len(raw)
                or join is not None and (type(join)is not str or not Path(join).is_absolute())):
            raise ValueError('bounded CALL run lacks original plan/raw observation references')
        selected=[Path(plan),*(Path(p) for p in raw),*(() if join is None else (Path(join),))]
        if any(p.is_symlink() or not p.is_file() or p.stat().st_size>2_000_000 for p in selected):
            raise ValueError('bounded CALL inputs must remain bounded original regular files')
        result.append(dict(run=_receipt(file),plan=_receipt(Path(plan)),
            raw_observations=[_receipt(Path(p)) for p in raw],join=None if join is None else _receipt(Path(join))))
    return result


def _reports(roots):
    seen=set();rows=[]
    for root in roots:
        root=Path(root).resolve()
        for path in (root/'result.json',*root.glob('*/result.json')):
            if path in seen or not path.is_file()or path.stat().st_size>2_000_000:continue
            seen.add(path)
            try:rows.append((path.parent,json.loads(path.read_bytes())))
            except (OSError,ValueError):continue
    return sorted(rows,key=lambda p:p[0].stat().st_mtime_ns,reverse=True)


def _component_catalog_gap(name,catalog,*,path=None):
    """Keep context-only readers from coercing another typed catalog to a path."""
    if name not in ('protocol','members'):return None
    from scar.analysis.source_context_query import ContextTargetGraph
    if isinstance(catalog,ContextTargetGraph):return None
    gap=dict(layer=name,kind='UNSUPPORTED_CATALOG_KIND',
        catalog_schema=catalog.report.get('schema'),
        supported_catalog_schemas=['scar.context-source-targets'],
        detail='Constructor/member protocol materials require the original context-qualified catalog; '
            'declared and observed source catalogs do not provide that interface.')
    if path is not None:gap['path']=str(path)
    return gap


def _validate_catalog_parents(catalog):
    from scar.analysis.source_declared_catalog import DeclaredSourceCatalog
    from scar.analysis.source_context_query import ContextTargetGraph
    if isinstance(catalog,DeclaredSourceCatalog):catalog.validate_parents()
    elif isinstance(catalog,ContextTargetGraph):
        catalog.fact_store.validate_parents();catalog.interface_store.validate_parents()
    else:raise TypeError('catalog handoff requires an original typed source catalog')


class CatalogHandoff:
    """One same-process discovery admission, never a serialized verification claim.

    Only discover fills this empty handle. Consumers recheck original parent
    guards and reader bytes; independent sessions still admit their own catalog.
    """
    __slots__=('_state',)

    def __init__(self):self._state=None

    def _seal(self,catalog,spec,entry,reader_calls):
        from scar.analysis.build import _receipt,_rules
        if self._state is not None:raise ValueError('catalog handoff is already filled')
        _validate_catalog_parents(catalog)
        entry_receipt=_receipt(entry)
        if entry_receipt['sha256']!=catalog.fact_store.model_units[spec['entry_unit']]['raw_sha256']:
            raise ValueError('catalog handoff entry source differs from its admitted model')
        # The original readers checked their own producer pins. Retain their
        # implementation bytes as session guards without decoding tables again.
        readers={Path(module.__file__).resolve() for name,module in tuple(sys.modules.items())
            if name.startswith('scar.') and getattr(module,'__file__',None)}
        self._state=dict(catalog=catalog,receipt=dict(spec['catalog']),entry_unit=spec['entry_unit'],
            entry=entry_receipt,report=json.loads(json.dumps(catalog.report)),pid=os.getpid(),used=False,
            rules=_rules(),readers=tuple(_receipt(path) for path in sorted(readers)),reader_calls=reader_calls)
        self.check()

    def check(self):
        from scar.analysis.build import _check,_rules
        state=self._state
        if state is None or state['pid']!=os.getpid():raise ValueError('catalog handoff is not admitted in this process')
        catalog=state['catalog'];receipt=state['receipt']
        if (catalog.path/'result.json'!=Path(receipt['path']) or catalog.report!=state['report']
                or state['rules']!=_rules()):
            raise ValueError('catalog handoff identity, report or reader rules changed')
        for source in (receipt,state['entry'],*state['readers']):_check(source)
        if json.loads(Path(receipt['path']).read_bytes())!=state['report']:
            raise ValueError('catalog handoff report differs from its original receipt')
        _validate_catalog_parents(catalog)

    def _consume(self,spec,entry):
        state=self._state
        if state is None or state['used']:raise ValueError('catalog handoff is empty or already consumed')
        state['used']=True
        if (spec['catalog']!=state['receipt'] or spec['entry_unit']!=state['entry_unit']
                or str(entry)!=state['entry']['path']):
            raise ValueError('catalog handoff does not match this build selection')
        self.check()
        return state['catalog']


def discover(entry,trace,*,catalog=None,cache_roots=(),catalog_handoff=None):
    """Choose compatible cached materials by schema and exact catalog, not names."""
    from scar.analysis.source_declared_catalog import read_source_catalog
    from scar.analysis.build import _receipt
    roots=tuple(cache_roots) or (Path(__file__).resolve().parents[2]/'artifacts/current/queries',)
    rows=_reports(roots);gaps=[];selected=None;reader_calls=0
    schemas=('scar.context-source-targets','scar.declared-source-catalog','scar.observed-source-catalog')
    candidates=[Path(catalog)] if catalog else [p for p,r in rows if r.get('schema') in schemas]
    for path in candidates:
        try:
            reader_calls+=1
            c=read_source_catalog(path)
            units=[u for u,r in c.fact_store.rows.items() if r['path']==str(entry)
                and u not in c.fact_store.original_source_units]
            if len(units)!=1:continue
            model=c.fact_store.model_units[units[0]]
            if model['raw_sha256']!=sha256(entry.read_bytes()).hexdigest():continue
            selected=(c,units[0]);break
        except (ValueError,OSError) as error:
            gaps.append(dict(layer='inputs',kind='CACHED_CATALOG_REJECTED',path=str(path),detail=str(error)))
    if selected is None:
        if catalog:raise ValueError('explicit F4 catalog does not contain the current entry')
        return None
    c,unit=selected
    if catalog is not None:
        # Published siblings belong to this explicit catalog's experiment;
        # their normal readers still establish exact parent/receipt identity.
        local=_reports((c.path.parent,));seen={p.resolve() for p,r in local}
        rows=local+[(p,r) for p,r in rows if p.resolve() not in seen]
    components={};functions=[];component_pins={}
    priority={'scar.source-observed-function-interfaces':0,'scar.source-returned-function-interfaces':1,
        'scar.source-function-interface-collection':2,'scar.source-function-interfaces':3,
        'scar.source-compiled-function-interfaces':4}
    for path,r in sorted(rows,key=lambda row:priority.get(row[1].get('schema'),10)):
        schema=r.get('schema')
        if schema in priority:
            if schema=='scar.source-observed-function-interfaces' and (trace is None or r.get('trace')!=str(trace)):continue
            # Ordinary leaves record catalog in their declared-call parent;
            # let the normal reader establish that original relationship.
            if r.get('catalog') and Path(r['catalog']).resolve()!=c.path.resolve():continue
            functions.append(_receipt(path/'result.json'))
        key={'scar.source-function-creations':'creations','scar.source-call-context-graph':'protocol',
            'scar.constructor-member-contexts':'members','scar.source-binding-native-graph':'native',
            'scar.source-resource-graph':'resources','scar.configuration-state-bindings':'configuration',
            'scar.source-observed-boundaries':'observed_boundaries',
            'scar.observed-source-binding-states':'binding_states',
            'scar.source-state-coverage':'source_coverage',
            'scar.source-resource-scopes':'resource_scopes',
            'scar.source-observed-resumptions':'resumptions',
            'scar.source-body-activations':'body_activations'}.get(schema)
        gap=_component_catalog_gap(key,c,path=path)
        if gap is not None:
            gaps.append(gap);continue
        if key=='resumptions':
            from scar.analysis.source_observed_resumptions import _pins as resumption_pins
            if trace is None or r['trace']!=str(trace) or Path(r['catalog']).resolve()!=c.path.resolve():continue
            if r['pins']!=resumption_pins():
                gaps.append(dict(layer=key,kind='CACHED_COMPONENT_RULES_CHANGED',path=str(path)));continue
        if key=='binding_states':
            from scar.analysis.source_observed_state import compatible_pins as binding_pins_match
            plan=r['plan'];parent=json.loads(Path(plan['parent']['path']).read_bytes())
            if (trace is None or parent['trace']!=str(trace)
                    or Path(plan['catalog']['uri']).parent.resolve()!=c.path.resolve()):continue
            if not binding_pins_match(plan['pins']):
                gaps.append(dict(layer=key,kind='CACHED_COMPONENT_RULES_CHANGED',path=str(path)));continue
        if key=='source_coverage':
            from scar.analysis.program_inventory_admission import compatible_static_pins
            plan=r['plan']
            if Path(plan['catalog']).resolve()!=c.path.resolve() or not plan.get('local_graph_shards'):continue
            if not compatible_static_pins(plan):
                gaps.append(dict(layer=key,kind='CACHED_COMPONENT_RULES_CHANGED',path=str(path)));continue
        if key=='resource_scopes':
            if (trace is None or Path(r['capture'])/'trace'!=trace or r['catalog']!=str(c.path)):continue
            from scar.analysis.source_resource_scopes import _pins as resource_scope_pins
            if r['pins']!=resource_scope_pins():
                gaps.append(dict(layer=key,kind='CACHED_COMPONENT_RULES_CHANGED',path=str(path)));continue
        if key in ('observed_boundaries','body_activations'):
            basis=r['basis']
            if (trace is None or basis['trace']!=str(trace)
                    or Path(basis['catalog']['uri']).parent.resolve()!=c.path.resolve()):continue
            if key not in component_pins:
                if key=='body_activations':
                    from scar.analysis.source_body_activations import _pins
                else:
                    from scar.analysis.source_observed_boundaries import _pins
                component_pins[key]=_pins()
            if r.get('pins')!=component_pins[key]:
                gaps.append(dict(layer=key,kind='CACHED_COMPONENT_RULES_CHANGED',path=str(path)))
                continue
        if key and key not in components:
            components[key]=_receipt(path/'result.json')
    spec=dict(catalog=_receipt(c.path/'result.json'),entry_unit=unit,functions=functions,
        components=components,selection_gaps=gaps)
    if catalog_handoff is not None:
        if type(catalog_handoff)is not CatalogHandoff:raise TypeError('expected an empty CatalogHandoff')
        catalog_handoff._seal(c,spec,entry,reader_calls)
    return spec


class _LazyState:
    def __init__(self,view):self.view=view
    def _query(self,point,incoming):
        source=self.view.namespaces.get(point.namespace)
        if source is None:return Neighborhood()
        provider=self.view.state(source.unit)
        node=source.nodes.get(point.node)
        if node is not None and node.anchor.template is not None:
            from scar.ir.program import normal_graph as G
            original=node.anchor.original
            owner=original.owner if type(original)is G.Port else original
            key=node.anchor.template;source._admit((key,))
            provider.ensure_regions(key,source.region_indexes[key].touching_regions(owner),include_content=True)
        else:provider.content(complete=False)
        return (provider.incoming if incoming else provider.outgoing)(point)
    def incoming(self,point):return self._query(point,True)
    def outgoing(self,point):return self._query(point,False)


class F4BuildSession:
    def __init__(self,path,*,_publishing=False,seconds=3600,input_bytes=16*1024**3,_catalog_handoff=None):
        from types import MappingProxyType
        from scar.analysis.build import _admit_rules
        from scar.analysis.source_declared_catalog import read_source_catalog
        from scar.analysis.program_view import ProgramView
        self._started=time.monotonic();self._seconds=seconds;self._input_bytes=input_bytes
        self.timings={};self._admitted={}
        self.path=Path(path).resolve();self.report=json.loads((self.path/'manifest.json').read_bytes())
        if self.report['backend']!='F4':raise ValueError('expected an F4 build')
        self.reader_rules=MappingProxyType(_admit_rules(self.report['rules'],publishing=_publishing))
        if not _publishing and not self.report['workflow_complete']:raise ValueError('F4 build not published')
        for receipt in self.report['tables'].values():self._admit(receipt)
        spec=self.report['f4'];self._admit(spec['catalog'])
        self._catalog_handoff=_catalog_handoff
        if _catalog_handoff is None:
            self.catalog=read_source_catalog(Path(spec['catalog']['path']).parent)
            self.catalog_admission=dict(mode='INDEPENDENT_READER',session_reader_calls=1,
                global_cache_used=False)
        else:
            if not _publishing or type(_catalog_handoff)is not CatalogHandoff:
                raise ValueError('catalog handoff is only available to its publishing session')
            self.catalog=_catalog_handoff._consume(spec,Path(self.report['entry']))
            self.catalog_admission=dict(mode='SESSION_LOCAL_DISCOVERY_HANDOFF',
                discovery_reader_calls=_catalog_handoff._state['reader_calls'],session_reader_calls=0,
                consumed_once=True,original_parent_and_reader_guards_rechecked=True,global_cache_used=False)
        self._checkpoint('catalog')
        self.view=ProgramView(self.catalog);self.providers={};self.gaps=list(spec['selection_gaps'])
        self._claimed=set();self.closed=False;self.observed=None;self.source_coverage=None;self.coverage_path=None
        self.entry=self.view.entry(spec['entry_unit'],context=())
        self.entries=(self.entry,);self.session=self
        self._functions(spec['functions'])
        self._checkpoint('function_interfaces')
        self._components(spec['components'])
        self._bounded_call_bindings()
        self._invocations()
        if not _publishing:
            saved=json.loads((self.path/'observed.json').read_bytes()).get('invocations')
            if saved is not None:
                provider=self.providers.get('invocations')
                if provider is None:raise ValueError('stored invocation scope provider is not available')
                provider.restore(saved)
        self._checkpoint('cross_layer_interfaces')
        self.view.register(_LazyState(self.view))
        binding_states=self.providers.get('binding_states')
        if binding_states is not None and spec['entry_unit'] in binding_states.headers:
            binding_states.ensure(spec['entry_unit'])
        self.view.state(spec['entry_unit']).content(complete=False)
        self._checkpoint('local_state_and_content')
        from scar.analysis.build_f4_resources import LazyF4Resources
        target=self.report.get('target')
        if target is not None:self._admit(target)
        target_path=None if target is None else Path(target['path']).parent
        try:
            self.resources=LazyF4Resources(self.view,self._component_path('resources'),target=target_path)
        except (OSError,ValueError) as error:
            self.gaps.append(dict(layer='resources',kind='RESOURCE_MATERIAL_NOT_JOINED',detail=str(error)))
            self.resources=LazyF4Resources(self.view,target=target_path)
        self.resources.demand(self.entry)
        self._checkpoint('resources')

    def _checkpoint(self,layer=None):
        elapsed=time.monotonic()-self._started
        if layer is not None:self.timings[layer]=elapsed
        if elapsed>self._seconds:raise TimeoutError('F4 assembly exceeded its cooperative time budget')
        if sum(r['bytes'] for r in self._admitted.values())>self._input_bytes:
            raise OverflowError('F4 direct publication receipts exceeded the input byte budget')

    def _admit(self,receipt):
        from scar.analysis.build import _check
        _check(receipt);self._admitted[receipt['path']]=receipt
        self._checkpoint()
        return Path(receipt['path']).parent

    def _component_path(self,name):
        r=self.report['f4']['components'].get(name)
        return self._admit(r) if r is not None else None

    def _bounded_call_bindings(self):
        from scar.analysis.source_call_binding_view import SourceCallBindingProvider
        inputs=self.report.get('bounded_call_bindings',())
        if inputs and _bounded_call_binding_inputs(Path(row['run']['path']).parent for row in inputs)!=inputs:
            raise ValueError('bounded CALL publication references changed')
        for index,row in enumerate(inputs):
            for receipt in (row['run'],row['plan'],*row['raw_observations'],*(() if row['join']is None else (row['join'],))):
                self._admit(receipt)
            provider=SourceCallBindingProvider(self.view,row['plan']['path'],[r['path'] for r in row['raw_observations']])
            if row['join']is not None and json.loads(Path(row['join']['path']).read_bytes())!=provider.summary:
                raise ValueError('saved bounded CALL join differs from original raw attempt replay')
            self.providers['bounded_call_bindings:'+str(index)]=provider
            self.providers['bounded_call_interfaces:'+str(index)]=provider.attach_function_interfaces()
        if inputs:self._checkpoint('bounded_call_target_metadata')

    def _functions(self,receipts):
        from scar.analysis.source_call_view import open_function_interfaces,SourceFunctionCallProvider
        for receipt in receipts:
            try:
                store=open_function_interfaces(self._admit(receipt))
                if (store.catalog.path,store.catalog.report)!=(self.catalog.path,self.catalog.report):continue
                units=frozenset(store.report['shards'])-self._claimed
                if not units:continue
                provider=SourceFunctionCallProvider(self.view,store)
                provider.available_units=units;self._claimed.update(units)
                self.providers['functions:'+str(store.path)]=provider
            except (OSError,ValueError) as error:
                self.gaps.append(dict(layer='calls',kind='INTERFACE_MATERIAL_REJECTED',path=receipt['path'],detail=str(error)))

    def _components(self,components):
        from scar.analysis.source_declared_catalog import DeclaredSourceCatalog
        if isinstance(self.catalog,DeclaredSourceCatalog):
            self.gaps.append(dict(layer='exec',kind='COMPILED_EXEC_SELECTION_NOT_PROVIDED',
                detail='Local source declarations do not select dynamically compiled exec roots.'))
        else:
            from scar.analysis.source_exec_view import SourceExecProvider
            self.providers['exec']=SourceExecProvider(self.view,self.catalog)
        for name in ('creations','native','protocol','configuration','observed_boundaries','body_activations','resumptions','resource_scopes','binding_states','source_coverage'):
            try:
                path=self._component_path(name)
                if path is None:
                    self.gaps.append(dict(layer=name,kind='MATERIAL_NOT_AVAILABLE'));continue
                gap=_component_catalog_gap(name,self.catalog,path=path)
                if gap is not None:
                    self.gaps.append(gap);continue
                if name=='creations':
                    from scar.analysis.source_creation_view import SourceFunctionCreationProvider
                    provider=SourceFunctionCreationProvider(self.view,path)
                elif name=='native':
                    from scar.analysis.source_native_view import SourceNativeCallProvider
                    provider=SourceNativeCallProvider(self.view,path)
                elif name=='protocol':
                    from scar.analysis.source_protocol_materials import SourceProtocolMaterials
                    from scar.analysis.source_protocol_view import SourceProtocolCallProvider
                    materials=SourceProtocolMaterials(path,self.catalog,members=self._component_path('members'))
                    provider=SourceProtocolCallProvider(self.view,materials)
                elif name=='observed_boundaries':
                    from scar.analysis.source_observed_boundaries import BoundaryStore,SourceObservedBoundaryProvider
                    indices=[p.store.index for p in self.providers.values()
                        if getattr(getattr(p,'store',None),'report',{}).get('schema')=='scar.source-observed-function-interfaces']
                    provider=SourceObservedBoundaryProvider(self.view,BoundaryStore(path,index=indices[0] if indices else None))
                elif name=='body_activations':
                    from scar.analysis.source_body_activations import SourceBodyActivationStore
                    from scar.analysis.source_body_activation_view import SourceBodyActivationProvider
                    indices=[p.store.index for p in self.providers.values()
                        if getattr(getattr(p,'store',None),'report',{}).get('schema') in (
                            'scar.source-observed-function-interfaces','scar.source-observed-boundaries')]
                    store=SourceBodyActivationStore(path,index=indices[0] if indices else None)
                    provider=SourceBodyActivationProvider(self.view,store)
                    if not store.report['coverage']['entry_coverage_complete']:
                        self.gaps.append(dict(layer=name,kind='EXECUTED_ENTRIES_WITHOUT_SOURCE_BODY',
                            missing=store.report['coverage']['missing_entries']))
                elif name=='resource_scopes':
                    from scar.analysis.source_resource_scopes import ResourceScopeStore
                    from scar.analysis.source_resource_scope_view import SourceResourceScopeProvider
                    indices=[p.store.index for p in self.providers.values()
                        if getattr(getattr(p,'store',None),'index',None) is not None]
                    store=ResourceScopeStore(path,index=indices[0] if indices else None)
                    provider=SourceResourceScopeProvider(self.view,store)
                elif name=='resumptions':
                    from scar.analysis.source_resumption_view import ResumptionStore,SourceResumptionProvider
                    indices=[p.store.index for p in self.providers.values()
                        if getattr(getattr(p,'store',None),'index',None) is not None]
                    store=ResumptionStore(path,index=indices[0] if indices else None)
                    provider=SourceResumptionProvider(self.view,store)
                    if store.report['source_port_events']!=store.report['records']:
                        self.gaps.append(dict(layer=name,kind='RESUMPTION_SOURCE_PORTS_UNRESOLVED',
                            missing=store.report['records']-store.report['source_port_events']))
                elif name=='source_coverage':
                    from scar.analysis.program_inventory_admission import admit_static_inventory
                    self.source_coverage=admit_static_inventory(path,catalog=self.catalog);self.coverage_path=path
                    if not self.source_coverage['catalog_local_graph_materialized']:
                        self.gaps.append(dict(layer=name,kind='CATALOG_LOCAL_GRAPH_INCOMPLETE'))
                    continue
                elif name=='binding_states':
                    from scar.analysis.source_observed_state_view import SourceObservedStateProvider
                    indices=[p.store.index for p in self.providers.values()
                        if getattr(getattr(p,'store',None),'index',None) is not None]
                    provider=SourceObservedStateProvider(self.view,path,index=indices[0] if indices else None)
                else:
                    from scar.analysis.source_configuration_view import SourceConfigurationProvider
                    provider=SourceConfigurationProvider(self.view,path)
                self.providers[name]=provider
            except (OSError,ValueError) as error:
                self.gaps.append(dict(layer=name,kind='MATERIAL_NOT_JOINED',detail=str(error)))

    def _invocations(self):
        """Install same-run scope discovery; original group units stay lazy."""
        from scar.analysis.source_invocation_view import LazySourceInvocationProvider
        activations=self.providers.get('body_activations')
        if activations is None:
            self.gaps.append(dict(layer='invocations',kind='BODY_ACTIVATION_MATERIAL_NOT_AVAILABLE'))
            return
        functions=[provider for name,provider in self.providers.items() if name.startswith('functions:')]
        self.providers['invocations']=LazySourceInvocationProvider(self.view,activations,functions,
            gaps=self.gaps,checkpoint=self._checkpoint)

    def attach_trace(self):
        """Actual F4 selections already admitted through their matched leaf."""
        if self.report['trace']:
            from scar.analysis.build import _check
            for receipt in self.report['trace']['inputs']:_check(receipt)
        observations=[];boundaries=[];activations=[];resource_scopes=[];binding_states=[];resumptions=[]
        for p in self.providers.values():
            if getattr(p,'report',{}).get('schema')=='scar.observed-source-binding-states':
                p.check();binding_states.append(dict(path=str(p.path),counts=p.report['counts'],
                    selected_local_read_slots=p.report['selected_local_read_slots'],
                    completed_fixed_points=p.report['completed_fixed_points'],
                    budget_or_control_incomplete=p.report['budget_or_control_incomplete']))
            store=getattr(p,'store',None)
            if store is not None and store.report['schema']=='scar.source-observed-function-interfaces':
                observations.append(dict(path=str(store.path),counts=store.report['counts'],run_scope=store.report['run_scope']))
            if store is not None and store.report['schema']=='scar.source-observed-boundaries':
                store.validate()
                boundaries.append(dict(path=str(store.path),counts=store.report['counts'],run_scope=store.report['basis']['run_scope']))
            if store is not None and store.report['schema']=='scar.source-body-activations':
                store.check()
                activations.append(dict(path=str(store.path),coverage=store.report['coverage'],
                    run_scope=store.report['basis']['run_scope']))
            if store is not None and store.report['schema']=='scar.source-resource-scopes':
                store.check()
                resource_scopes.append(dict(path=str(store.path),summary=store.report['summary']))
            if store is not None and store.report['schema']=='scar.source-observed-resumptions':
                store.check()
                resumptions.append(dict(path=str(store.path),records=store.report['records'],
                    activations=store.report['activations'],source_port_events=store.report['source_port_events'],
                    source_body_interfaces=store.report['source_body_interfaces'],frontiers=store.report['frontier_count'],
                    shared_phase_interfaces=len(p.groups)))
        self.observed=dict(schema='scar.build-f4-observed-selections',publications=observations,
            boundaries=boundaries,body_activations=activations,resource_scopes=resource_scopes,binding_states=binding_states,
            resumptions=resumptions,invocations=(self.providers['invocations'].publication()
                if 'invocations' in self.providers else None),
            runtime_events_replaced_by_static_frames=False)
        self.observed['bounded_call_bindings']=[]
        for name,provider in self.providers.items():
            if not name.startswith('bounded_call_bindings:'):continue
            provider.check();index=int(name.split(':')[-1]);material=self.report['bounded_call_bindings'][index]
            summary=provider.summary
            interfaces=provider.function_interfaces
            self.observed['bounded_call_bindings'].append(dict(
                run=material['run'],plan=material['plan'],raw_observations=material['raw_observations'],
                planned_sites=summary['planned_sites'],observed_sites=summary['observed_sites'],
                attempts=summary['attempts'],target_records=len(provider.nodes),
                metadata_only=True,InvocationKey_created=False,callee_entry_or_completion_proven=False,
                conditional_source_interfaces=(None if interfaces is None else interfaces.store.coverage_summary())))
        return self.observed

    def select(self,*,file=None,entry=None,kind=None):
        file=None if file is None else str(file)
        points=[]
        for unit,row in self.catalog.fact_store.rows.items():
            if file and not (row['path']==str(Path(file).absolute()) or row['path'].endswith('/'+file)):continue
            if unit in self.catalog.fact_store.original_source_units and not file:continue
            if not file and unit!=self.report['f4']['entry_unit']:continue
            facts=self.catalog.fact_store.load(unit)
            if entry is None and kind in (None,'MODULE'):
                points.append(self.view.entry(unit,context=() if unit==self.report['f4']['entry_unit'] else None))
            if entry is not None or kind in ('FUNCTION','ASYNC_FUNCTION'):
                from scar.analysis.static_query import _Fragments
                from scar.analysis.source_roles import SyntaxRoleView
                from scar.ir.program.source_syntax import SourceSyntaxMaterial
                from scar.ir.record_codec import decode
                original=self.catalog.fact_store.original_source_units.get(unit,unit)
                model=self.catalog.fact_store.model_units[original]
                syntax=decode(SourceSyntaxMaterial,_Fragments(self.catalog.fact_store.model_for(original)).read(model['artifacts']['source_syntax']))
                roles=SyntaxRoleView(syntax)
                for fn in facts.functions:
                    try:name=roles.field(fn.template.occurrence,'name').scalar.to_python()
                    except (KeyError,AttributeError):continue
                    if entry is None or name==entry or entry.endswith('.'+name):
                        points.append(self.view.source_region(unit,fn.template,fn.body))
        return tuple(dict.fromkeys(points))

    def point(self,*,file=None,entry=None,kind=None,node):
        points=self.select(file=file,entry=entry,kind=kind)
        namespaces=tuple(dict.fromkeys(p.namespace for p in points))
        if len(namespaces)!=1:raise ValueError('select one F4 source namespace for a node query')
        prefix,raw=node.split(':',1)
        cls=next((cls for cls in NODE_TYPES if cls.prefix==prefix),None)
        if cls is None:raise ValueError('unknown typed node prefix')
        point=Point(namespaces[0],cls(raw),None);self.view.get(point);return point

    def layer_summary(self):
        return dict(unified=True,local_state=dict(bindings=True,content=True,mode='DEMAND_REGION_REQUESTS',
                analyzed_operations=sum(len(p.analyzed_operations)for p in self.view.state_providers.values()),
                binding_units_complete=sum(p.bindings_complete for p in self.view.state_providers.values())),
            cross_layer=dict(functions=any(k.startswith(('functions:','bounded_call_interfaces:')) for k in self.providers),
                **{k:k in self.providers for k in ('exec','creations','native','protocol','configuration','observed_boundaries','body_activations','invocations','resumptions','resource_scopes','binding_states')},
                bounded_call_target_metadata=any(k.startswith('bounded_call_bindings:') for k in self.providers),
                bounded_source_function_interfaces=any(k.startswith('bounded_call_interfaces:') for k in self.providers)),resources=True,
            storage=True,readback=True,providers=sorted(self.providers),
            loaded_units=len(self.view.units),catalog_units=len(self.catalog.fact_store.rows),
            catalog_local_graph_materialized=bool(self.source_coverage and self.source_coverage['catalog_local_graph_materialized']),
            body_copies=0,loops_unrolled=False)

    def coverage_summary(self):
        """Count complete shared unit domains separately from observed execution."""
        local=None
        if self.source_coverage is not None:
            report=json.loads((self.coverage_path/'result.json').read_bytes())
            local=dict(inventory=self.source_coverage['inventory'],index=str(self.coverage_path/'index.json'),
                catalog_unit_domain_complete=self.source_coverage['catalog_local_graph_materialized'],
                catalog_units=report['plan']['catalog_units'],materialized_units=report['inventoried_units'],
                counts=report['counts'],failed_units=report['failures'],
                stored_shard_bytes=self.source_coverage['stored_shard_bytes'])
            local['admission']=self.source_coverage['admission']
        observed=self.observed or self.attach_trace()
        return dict(schema='scar.program-build-coverage',schema_version=1,
            local_graph=local,observed_body_entries=[r['coverage'] for r in observed['body_activations']],
            observed_resource_scopes=[r['summary'] for r in observed['resource_scopes']],
            observed_binding_states=observed['binding_states'],
            observed_resumptions=observed['resumptions'],
            observed_invocation_scopes=observed['invocations'],
            observed_bounded_call_targets=observed['bounded_call_bindings'],
            entry_query_purpose='BOUNDED_NAVIGATION_EXAMPLE_NOT_COVERAGE_DENOMINATOR',
            runtime_path_complete=False,effects_closed=False,first_stage_complete=False)

    def readback_publication(self,report):
        """Read serialized outputs against the same admitted immutable inputs.

        Reopening a BuildSession remains the independent replay API. Publishing
        must not repeat that entire analysis: the stores already authenticated
        their source/trace relations, and retain guards for subsequent queries.
        """
        from scar.analysis.build import _check,_rules
        import gzip
        started=time.monotonic()
        if self._catalog_handoff is not None:self._catalog_handoff.check()
        if (json.loads((self.path/'manifest.json').read_bytes())!=report
                or report['rules']!=_rules()):
            raise ValueError('publication manifest or implementation changed')
        for receipt in (*self._admitted.values(),*report['tables'].values()):_check(receipt)
        spec=report['f4']
        relationships=dict(schema='scar.build-f4-relationships',catalog=spec['catalog'],
            function_leaves=spec['functions'],components=spec['components'],
            bounded_call_bindings=report.get('bounded_call_bindings',[]),copied_bodies=0)
        if json.loads((self.path/'relationships.json').read_bytes())!=relationships:
            raise ValueError('stored relationship references differ from admitted components')
        canonical=lambda value:json.loads(json.dumps(value))
        if json.loads((self.path/'observed.json').read_bytes())!=canonical(self.attach_trace()):
            raise ValueError('stored execution evidence differs from admitted observations')
        if (self.path/'coverage.json').exists():
            if self.source_coverage is not None:
                from scar.analysis.program_inventory_admission import admit_static_inventory
                if admit_static_inventory(self.coverage_path,catalog=self.catalog)!=self.source_coverage:
                    raise ValueError('admitted graph coverage changed during publication')
            if json.loads((self.path/'coverage.json').read_bytes())!=canonical(self.coverage_summary()):
                raise ValueError('stored coverage differs from original unit and execution domains')
        with gzip.open(self.path/'resources.json.gz','rt') as stream:resources=json.load(stream)
        if resources!=canonical(self.resources.payload()):
            raise ValueError('stored resource interfaces differ from admitted resource ports')
        stores={id(p.store):p.store for p in self.providers.values() if hasattr(p,'store')}
        for store in stores.values():
            guard=getattr(store,'validate',None) or getattr(store,'check',None)
            if guard is not None:guard()
        for name,provider in self.providers.items():
            if name.startswith('bounded_call_bindings:'):provider.check()
        self.view.expand(self.entry)
        for source in self.view.units.values():source.check()
        for receipt in (*self._admitted.values(),*report['tables'].values()):_check(receipt)
        if report['rules']!=_rules():raise ValueError('publication implementation changed during readback')
        if self._catalog_handoff is not None:self._catalog_handoff.check()
        self._checkpoint()
        return dict(mode='STORED_TABLES_WITH_ADMITTED_INDEX',
            output_tables=len(report['tables']),admitted_publications=len(self._admitted),
            entry_expanded=True,indices_reconstructed=0,independent_full_analysis_repeated=False,
            seconds=time.monotonic()-started)

    def close(self):
        if self.closed:return
        self.closed=True
        self.view.dependencies.providers.clear();self.view.units.clear();self.view.namespaces.clear()


def build_f4(entry,trace,out,spec,*,source_roots=(),max_points=128,max_seconds=3600,max_input_bytes=16*1024**3,target=None,
        bounded_call_bindings=(),catalog_handoff=None):
    from scar.analysis.build import _rules,_receipt,_write,_configuration_inputs,_build_target
    from scar.analysis.program_dependency_export import export_dependency_walk
    started=time.monotonic();bounded_inputs=_bounded_call_binding_inputs(bounded_call_bindings);out.mkdir(parents=True)
    target=_build_target(out,entry,source_roots,target)
    trace_ref=None
    if trace:
        from scar.analysis.program_observed_dependencies import _inputs
        trace_ref=dict(path=str(trace),inputs=[_receipt(p) for p in sorted(_inputs(trace))])
    report=dict(schema='scar.program-build',schema_version=1,backend='F4',entry=str(entry),rules=_rules(),
        f4=spec,model=None,trace=trace_ref,
        target=None if target is None else _receipt(Path(target)/'result.json'),
        bounded_call_bindings=bounded_inputs,
        tables={},workflow_complete=False,first_stage_complete=False,
        target_executed=False,original_body_bytes_copied=0,loops_unrolled=False,
        limits=dict(max_seconds=max_seconds,max_input_bytes=max_input_bytes,
            byte_scope='DIRECT_PUBLICATION_RECEIPTS',time_enforcement='COOPERATIVE_STAGE_BOUNDARIES'),
        entry_probe_limits=dict(max_points=max_points,max_edges=512,max_depth=8))
    _write(out/'manifest.json',report)
    gaps=[];configuration=_configuration_inputs((entry.parent,*(Path(p).resolve() for p in source_roots)),out,gaps)
    session=F4BuildSession(out,_publishing=True,seconds=max_seconds,input_bytes=max_input_bytes,
        _catalog_handoff=catalog_handoff)
    try:
        walk=session.view.dependencies.walk((session.entry,),direction='outgoing',include_structure=True,
            max_points=max_points,max_edges=512,max_depth=8)
        # A bounded walk may demand an original invocation-group unit. Publish
        # its receipts/counts after that demand, without copying event rows.
        observed=session.attach_trace();_write(out/'observed.json',observed)
        _write(out/'coverage.json',session.coverage_summary())
        _write(out/'entry-query.json',export_dependency_walk(walk))
        gaps.extend(session.gaps)
        gaps.extend(session.resources.gaps)
        gaps.extend(dict(layer='query',kind=f.kind,detail=f.detail,point=encode(f.point)) for f in walk.frontiers)
        # Persist references to original relation shards, not expanded bodies.
        _write(out/'relationships.json',dict(schema='scar.build-f4-relationships',catalog=spec['catalog'],
            function_leaves=spec['functions'],components=spec['components'],bounded_call_bindings=bounded_inputs,copied_bodies=0))
        payload=session.resources.payload()
        import gzip
        (out/'resources.json.gz').write_bytes(gzip.compress(json.dumps(payload,separators=(',',':')).encode(),mtime=0))
        report['layers']=session.layer_summary()
        report['catalog_admission']=session.catalog_admission
        report['assembly_elapsed_by_layer']=session.timings
        report['observed_admission']='ADMITTED' if any(observed[k] for k in ('publications','boundaries','body_activations','resumptions','resource_scopes','binding_states')) else 'NOT_JOINED'
        report['bounded_call_bindings_admission']='ADMITTED_ATTEMPT_METADATA' if observed['bounded_call_bindings'] else 'NOT_SUPPLIED'
        report['summary']=dict(source_units=len(session.catalog.fact_store.rows),loaded_units=len(session.view.units),
            configuration_inputs=len(configuration),observed_publications=len(observed['publications']),
            observed_boundary_publications=len(observed['boundaries']),
            observed_body_activation_publications=len(observed['body_activations']),
            observed_resumption_publications=len(observed['resumptions']),
            observed_resource_scope_publications=len(observed['resource_scopes']),
            observed_binding_state_publications=len(observed['binding_states']),
            bounded_call_binding_publications=len(observed['bounded_call_bindings']),
            bounded_call_target_attempts=sum(row['attempts'] for row in observed['bounded_call_bindings']),
            bounded_source_function_interfaces=sum(
                row['conditional_source_interfaces']['interfaces']
                for row in observed['bounded_call_bindings'] if row['conditional_source_interfaces'] is not None),
            observed_invocation_handles=(observed['invocations']['actual_call_handles']
                if observed['invocations'] is not None else 0),
            catalog_local_graph_materialized=bool(session.source_coverage and session.source_coverage['catalog_local_graph_materialized']),
            resource_records=len(session.resources.provider.records))
        return _finish_publication(session,report,gaps,started)
    finally:session.close()


def _finish_publication(session,report,gaps,started):
    from scar.analysis.build import _receipt,_write
    out=session.path
    gaps.append(dict(layer='coverage',kind='FULL_INFERENCE_COVERAGE_UNPROVEN',
        detail='Shared catalog graph coverage and observed entries are recorded separately in coverage.json; runtime path, content and effect closure remain unproved.'))
    _write(out/'gaps.json',dict(schema='scar.build-gaps',gaps=gaps,counts=dict(Counter(g['layer'] for g in gaps))))
    report['tables']={p.name:_receipt(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'}
    report['summary']['gaps']=len(gaps)
    _write(out/'manifest.json',report)
    session._checkpoint()
    report['publication_readback']=session.readback_publication(report)
    report['seconds']=time.monotonic()-started
    report['workflow_complete']=True;_write(out/'manifest.json',report);return report
