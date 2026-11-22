"""Original-F4 array loops through acyclic, guarded global call composition.

Composite proofs are deliberately distinct from InvariantLoopProof/v1. Their
body is a derived fused DAG, not the source body of a newly invented callable.
Every function-global hop must still be guarded at runtime before lowering is
used to replace the original loop.
"""
from dataclasses import asdict,dataclass
from hashlib import sha256
from pathlib import Path
import json

from scar.analysis.program_view import ProgramView
from scar.analysis.source_references import source_port
from scar.compiler.graph_frontend import (discover_graph_regions, GraphRegionRejected,
    _discovery_for, _metadata, _reject, _origin, _ENV, _WORLD)
from scar.compiler.loop_admission import _InvariantLoopLowerer, _PendingLeafResult
from scar.compiler.loop_admission import _Parameter
from scar.compiler.model import ArrayRegion, SourceRef,region_from_dict
from scar.compiler.pipeline_frontend import (_PipelineLowerer, _Value, _Callee, _global_functions,
    SourceFunctionComponents, _pipelines_from_leaf_plans, _prepare_leaf_plans, _source_occurrence)
from scar.compiler.pipeline_frontend import PipelinePlan, compose_pipeline, prepare_pipeline_leaf, document as pipeline_document, pipeline_from_dict
from scar.ir.program import source_productions as P, normal_graph as F
from scar.ir.program.normal_graph import Body
from scar.ir.program.control import LanguageCompletionKind as C
from scar.ir.record_codec import encode
from scar.optimization.optimizer import optimize_region
from scar.optimization.pipeline_fusion import fuse_pipeline_region, verify_pipeline_fusion
from scar.trace.code_snapshot import code_key


@dataclass(frozen=True)
class FunctionGlobalGuard:
    owner_source: SourceRef
    lookup_name: str
    target_source: SourceRef
    global_path: tuple[str, ...]
    origins: tuple[dict, ...] = ()
    lookup_semantics: str = 'FUNCTION_GLOBALS_AT_EACH_HOP'


@dataclass(frozen=True)
class PipelineComposition:
    pipeline: PipelinePlan
    fusion_certificate: dict


@dataclass(frozen=True)
class CompositeInvariantLoopProof:
    id: str
    source: SourceRef
    parameters: tuple[str, ...]
    body: ArrayRegion
    carry_parameter: str
    trip_count_parameter: str
    parameter_map: dict[str, str]
    root_callee_source: SourceRef
    root_global_name: str
    nested_callee_guards: tuple[FunctionGlobalGuard, ...]
    compositions: tuple[PipelineComposition, ...]
    origins: tuple[dict, ...]
    profile: str = 'ORIGINAL_F4_COMPOSITE_ARRAY_LOOP_V1'
    body_identity: str = 'DERIVED_FUSED_DAG_NOT_A_NEW_PYTHON_CALLABLE'
    transformation_authorized: bool = False

    @property
    def carry_outer_parameter(self):return self.parameter_map[self.carry_parameter]


@dataclass(frozen=True)
class _Component:
    execution: object
    guards: tuple[FunctionGlobalGuard, ...] = ()
    compositions: tuple[PipelineComposition, ...] = ()
    fused_body: ArrayRegion | None = None


class _OneOrMoreCallLowerer(_PipelineLowerer):
    """Reuse the original CALL/value rules; admit transparent one-call wrappers."""
    def lower(self):
        function=self.function
        if (function.body_scope is None or function.environment.kind is not P.DeferredBodyKind.FUNCTION
                or function.environment.captures or function.environment.owned_cells
                or len(function.environment.parameters)!=len(function.parameter_names)
                or any(parameter.kind not in (P.DeferredParameterKind.POSITIONAL_ONLY,
                    P.DeferredParameterKind.POSITIONAL_OR_KEYWORD) or parameter.default_source is not None
                    for parameter in function.environment.parameters)):
            _reject('COMPOSITE_PIPELINE_SIGNATURE','ordinary explicit positional array parameters required')
        context=self.resolver.plan(self.source.unit,function.template)
        if any(type(step.instruction) is P.ApplyDecorator for step in context.plan.steps):
            _reject('COMPOSITE_PIPELINE_DECORATOR','decorated wrapper is unsupported')
        arguments=[_ENV]
        for index,_ in enumerate(function.parameter_names):
            value=_Value('arg:'+str(index));arguments.append(value)
            formal=source_port(self.facts,function.formals[index+1])
            namespaces=self.names.get((formal.port,True),())
            if len(namespaces)!=1 or namespaces[0].kind is not F.NamespaceKind.LEXICAL:
                _reject('COMPOSITE_PIPELINE_PARAMETER','original lexical array formal required')
            self.locals[namespaces[0]]=value
            self.origins.append(_origin(context,function.body,value=encode(formal),pipeline_value=value.id))
        kind,result=self.run(context,function.body,(*arguments,_WORLD))
        if (kind is not C.RETURN or len(result)!=2 or type(result[0]) is not _Value
                or not self.calls or result[0].id not in {call.output for call in self.calls}):
            _reject('COMPOSITE_PIPELINE_RETURN','one or more pure calls and one fresh call result required')
        return tuple(self.calls),result[0].id,tuple(self.origins)


class _ImportedPipelineLowerer(_OneOrMoreCallLowerer):
    def __init__(self,*args,function_routes,**kwargs):
        super().__init__(*args,**kwargs);self.function_routes=function_routes

    def action(self,context,step,args,origin):
        before=len(self.calls);result=super().action(context,step,args,origin)
        if len(self.calls)>before:
            from dataclasses import replace
            call=self.calls[-1];route=self.function_routes.imported_route(self.source.unit,call.callee.lookup_path[0])
            if route is not None:self.calls[-1]=replace(call,origins=(*call.origins,route))
        return result


class _ImportedLoopLowerer(_InvariantLoopLowerer):
    def __init__(self,*args,function_routes,**kwargs):
        super().__init__(*args,**kwargs);self.function_routes=function_routes

    def action(self,context,step,args,origin):
        if not args or type(args[0]) is not _Callee:return super().action(context,step,args,origin)
        callee=args[0];identity=next(row for row in callee.plan.source.origins if row.get('kind')=='source_identity')
        target_unit=identity['unit']
        if target_unit==self.source.unit:return super().action(context,step,args,origin)
        if step.instruction.action is not P.LanguageActionKind.CALL:
            _reject('CROSS_UNIT_LOOP_EXTRA_ACTION','original fixed global CALL required',origin)
        fact=self.call_facts.get(context.result(step).port)
        if (fact is None or fact.keywords or fact.positional_pack is not None or fact.keyword_pack is not None
                or len(args)!=len(fact.positional)+2 or args[-1] is not _WORLD):
            _reject('CROSS_UNIT_LOOP_CALL_ARGUMENTS','original positional CALL ABI required',origin)
        if (tuple(context.value(value).port for value in step.arguments)!=tuple(source_port(self.facts,i).port
                for i in (fact.callee,*fact.positional,fact.world))
                or context.result(step,1).port!=source_port(self.facts,fact.result_world).port):
            _reject('CROSS_UNIT_LOOP_CALL_ABI','CALL ports differ from original F4',origin)
        body=callee.plan;actuals=args[1:-1]
        if (self.phase!='BODY' or self.body_calls or type(body) is not ArrayRegion or len(body.parameters)<2
                or len(actuals)!=len(body.parameters)
                or any(type(value) is not _Parameter or value.index==self.roles[1] for value in actuals)):
            _reject('CROSS_UNIT_LOOP_ARRAY_ARGUMENTS','one admitted composite CALL over original non-trip parameters required',origin)
        signature=callee.function.environment.parameters
        if (len(signature)!=len(body.parameters) or any(parameter.kind not in
                (P.DeferredParameterKind.POSITIONAL_ONLY,P.DeferredParameterKind.POSITIONAL_OR_KEYWORD)
                or parameter.default_source is not None for parameter in signature)):
            _reject('CROSS_UNIT_LOOP_SIGNATURE','explicit fixed callee parameters required',origin)
        # The selected callee's template is qualified by its actual source
        # unit. Equal occurrence numbers in different files are not identity.
        target_context=self.resolver.plan(target_unit,callee.function.template)
        if any(type(row.instruction) is P.ApplyDecorator for row in target_context.plan.steps):
            _reject('CROSS_UNIT_LOOP_DECORATOR','decorated imported callee unsupported',origin)
        self.body_calls+=1;self.selected_body=callee
        self.origins.append(dict(origin,role='invariant_array_leaf_call',leaf_region_id=body.id,
            call_callee=encode(source_port(self.facts,fact.callee)),call_result=encode(source_port(self.facts,fact.result)),
            call_inputs=[encode(source_port(self.facts,i)) for i in fact.positional],
            outer_parameter_indices=[value.index for value in actuals],
            declared_global_import_route=self.function_routes.imported_route(self.source.unit,callee.name)))
        return (_PendingLeafResult(tuple(value.index for value in actuals)),_WORLD)


def _source_reference(unit,function,model,syntax,codes,recipe,digest,origins=()):
    name,span,lines=_metadata(syntax,function.template.occurrence)
    candidates=[code for code in codes if code.co_name==name and code.co_firstlineno in lines
        and code.co_varnames[:code.co_argcount+code.co_kwonlyargcount]==function.parameter_names]
    if len(candidates)!=1:_reject('COMPOSITE_CODE_IDENTITY','unique original function CodeID required')
    code=candidates[0];key=code_key(code)
    identity=dict(kind='source_identity',method='COMPILE_ONLY_RETAINED_F0',unit=unit,
        occurrence=function.template.occurrence,span=span,recipe=recipe,code_key=key,
        semantic_authority='ORIGINAL_F4_PRODUCTION_PLANS_AND_UNIT_FACTS')
    return SourceRef(model['path'],digest,code.co_qualname,span['line'],span['column'],key,(identity,*origins))


def _component(pipeline,by_code):
    merged,certificate=fuse_pipeline_region(pipeline)
    if merged is None:_reject('COMPOSITE_FUSION_OPEN',str(certificate['frontiers']))
    verify_pipeline_fusion(pipeline,merged,certificate)
    guards=[];steps=[];seen=set()
    for call in pipeline.calls:
        source=call.callee.source
        child=by_code[(source.path,source.code_key)]
        for step in child.compositions:
            if step.pipeline.id not in seen:steps.append(step);seen.add(step.pipeline.id)
        name=call.callee.lookup_path[0]
        guards.append(FunctionGlobalGuard(pipeline.source,name,source,(name,),call.origins))
        guards.extend(FunctionGlobalGuard(guard.owner_source,guard.lookup_name,guard.target_source,
            (name,*guard.global_path),guard.origins) for guard in child.guards)
    # Aliases/call occurrences retain their original routes; duplicate path
    # entries are harmless and preserve distinct original CALL provenance.
    steps.append(PipelineComposition(pipeline,certificate))
    return _Component(prepare_pipeline_leaf(optimize_region(merged)),tuple(guards),tuple(steps),merged)


def discover_pipeline_invariant_loops(catalog,source=None,*,_discovery=None):
    """Resolve admitted ordinary module/from-import functions to one fixed point.

    The source argument filters returned loop owners, not their imported
    dependencies. Every dependency provider must already exist in the catalog.
    """
    discovery = _discovery_for(catalog, _discovery)
    view, resolver = discovery.view, discovery.resolver
    roots=tuple(view.catalog.fact_store.rows) if source is None else tuple(sorted(
        view.catalog.units_for_source(str(Path(source).absolute())) or ()))
    if source is not None and not roots:return [],[dict(path=str(Path(source).absolute()),code='SOURCE_NOT_IN_CATALOG')]
    routes=SourceFunctionComponents(view);units=routes.dependency_units(roots)
    store=view.catalog.fact_store
    paths=sorted({store.model_units[unit].get('path',store.rows[unit]['path']) for unit in units})
    leaves = []
    for path in paths:
        found, _ = discover_graph_regions(view, path, _discovery=discovery)
        leaves.extend(found)
    leaf_plans = _prepare_leaf_plans(leaves, units)
    pipelines, _ = _pipelines_from_leaf_plans(discovery, units, leaf_plans)
    pipeline_by_unit = {_source_occurrence(pipeline.source): pipeline for pipeline in pipelines}
    result=[];rejections=[];originals={};metadata={}
    for unit in units:
        original=view.unit(unit);originals[unit]=original
        resolver.prepare_many(Body(unit,fn.template,fn.body) for fn in original.facts.functions)
        try:metadata[unit]=discovery.identity(unit)
        except (ValueError,KeyError,SyntaxError,OSError) as error:
            rejections.append(dict(unit=unit,code='COMPOSITE_SOURCE_IDENTITY',detail=str(error)))
    components={};by_code={};last={};pending=[]
    for unit in units:
        if unit not in metadata:continue
        for function in originals[unit].facts.functions:
            plan = leaf_plans.get((unit, function.template.occurrence))
            if plan is None:
                pending.append((unit, function))
                continue
            component = _Component(plan)
            source_ref = plan.region.source
            components[unit, function.template] = component
            by_code[source_ref.path, source_ref.code_key] = component
    while pending:
        progress=False;remaining=[]
        globals_by_unit={unit:routes.globals(unit,components) for unit in metadata}
        for unit,function in pending:
            key=unit,function.template;original=originals[unit]
            model,syntax,codes,recipe,digest=metadata[unit]
            try:
                pipeline=pipeline_by_unit.get((unit,function.template.occurrence))
                if pipeline is None:
                    calls,output,origins=_ImportedPipelineLowerer(original,function,resolver,
                        globals_by_unit[unit],function_routes=routes).lower()
                    reference=_source_reference(unit,function,model,syntax,codes,recipe,digest,origins)
                    pipeline=compose_pipeline(reference,function.parameter_names,calls,output)
                component=_component(pipeline,by_code);components[key]=component
                ref=component.execution.region.source;by_code[ref.path,ref.code_key]=component;progress=True
            except GraphRegionRejected as error:
                last[key]=dict(unit=unit,occurrence=function.template.occurrence,code=error.code,
                    detail=error.detail,origin=error.origin);remaining.append((unit,function))
        pending=remaining
        if not progress:break
    for unit in roots:
        if unit not in metadata:continue
        original=originals[unit];model,syntax,codes,recipe,digest=metadata[unit]
        globals=routes.globals(unit,components,fused=True)
        module_functions={value.function.template for values in _global_functions(original.facts,
            {fn.template:fn for fn in original.facts.functions}).values() for value in values}
        for function in original.facts.functions:
            key=unit,function.template
            if function.template not in module_functions or key in components:continue
            try:
                body,carry,trip,mapping,name,origins=_ImportedLoopLowerer(original,function,resolver,globals,
                    syntax=syntax,function_routes=routes).lower()
                root=by_code[body.source.path,body.source.code_key]
                if not root.compositions:_reject('COMPOSITE_BODY_REQUIRED','loop body must expand a proven composition')
                reference=_source_reference(unit,function,model,syntax,codes,recipe,digest)
                root_guard=FunctionGlobalGuard(reference,name,body.source,(name,),
                    tuple(row for row in origins if row.get('role')=='invariant_array_leaf_call'))
                guards=(root_guard,*(FunctionGlobalGuard(guard.owner_source,guard.lookup_name,guard.target_source,
                    (name,*guard.global_path),guard.origins) for guard in root.guards))
                identifier='composite-invariant-loop:'+sha256((unit+'\0'+function.body.wire+'\0'+reference.code_key).encode()).hexdigest()[:24]
                result.append(CompositeInvariantLoopProof(identifier,reference,function.parameter_names,
                    body,carry,trip,mapping,body.source,name,guards,root.compositions,origins))
            except GraphRegionRejected as error:
                rejections.append(dict(unit=unit,occurrence=function.template.occurrence,code=error.code,
                    detail=error.detail,origin=error.origin,composition_frontier=last.get(key),
                    declared_global_frontiers=routes.frontiers.get(unit,{}),
                    unresolved_component_dependencies=routes.unresolved_dependencies(unit,function,components,last)))
    return result,rejections


def validate_composite_invariant_loop_proof(proof,*,catalog):
    """Replay the actual F4 graph and all fusion/guard routes before lowering.

    No boolean field or an InvariantLoopProof/v1 cast substitutes for this
    composite graph proof. A supplied ProgramView shares the original session.
    Runtime still has to guard every recorded function-global lookup hop.
    """
    if (type(proof) is not CompositeInvariantLoopProof or proof.profile!='ORIGINAL_F4_COMPOSITE_ARRAY_LOOP_V1'
            or proof.body_identity!='DERIVED_FUSED_DAG_NOT_A_NEW_PYTHON_CALLABLE'
            or proof.transformation_authorized is not False):
        raise ValueError('original composite loop proof with undischarged runtime guards required')
    for source in (proof.source,proof.root_callee_source,
            *(guard.target_source for guard in proof.nested_callee_guards)):
        if sha256(Path(source.path).read_bytes()).hexdigest()!=source.sha256:
            raise ValueError('composite original source bytes changed')
    original,_=discover_pipeline_invariant_loops(catalog,source=proof.source.path)
    canonical=lambda value:json.dumps(asdict(value),sort_keys=True,separators=(',',':'),allow_nan=False)
    matches=[candidate for candidate in original if candidate.id==proof.id and canonical(candidate)==canonical(proof)]
    if len(matches)!=1:raise ValueError('composite loop differs from original F4/fusion/global-route replay')
    return matches[0]


COMPOSITE_PROOF_SCHEMA='scar.composite-invariant-loop-proof'
_COMPOSITION_SCHEMA='scar.composite-loop-composition'


def write_composite_invariant_loop_proof(proof,path):
    """Store one fused body; share receipt-bound original composition shards.

    This writer checks structure/fusion/source bytes. Reading for use still
    requires original F4 rediscovery through an admitted catalog session.
    """
    from scar.ir.execution_recipe import artifact_reference
    if type(proof) is not CompositeInvariantLoopProof or proof.transformation_authorized is not False:
        raise ValueError('separate composite proof without transformation permission required')
    for source in (proof.source,proof.root_callee_source,
            *(guard.owner_source for guard in proof.nested_callee_guards),
            *(guard.target_source for guard in proof.nested_callee_guards)):
        if sha256(Path(source.path).read_bytes()).hexdigest()!=source.sha256:
            raise ValueError('composite original source bytes changed')
    path=Path(path).absolute()
    if path.exists():raise ValueError('fresh composite proof output required')
    path.parent.mkdir(parents=True,exist_ok=True);refs=[]
    def save(target,value):
        with target.open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    for i,composition in enumerate(proof.compositions):
        body,certificate=fuse_pipeline_region(composition.pipeline)
        verify_pipeline_fusion(composition.pipeline,body,composition.fusion_certificate)
        if body is None:raise ValueError('composite proof has an unclosed fusion')
        target=path.with_name(path.stem+'-composition-'+str(i)+'.json')
        if target.exists():raise ValueError('fresh composite composition output required')
        save(target,dict(schema=_COMPOSITION_SCHEMA,schema_version=1,
            pipeline=pipeline_document(composition.pipeline),fusion_certificate=certificate))
        refs.append(artifact_reference(target,_COMPOSITION_SCHEMA))
    if not refs:raise ValueError('composite loop needs original composition artifacts')
    document=dict(schema=COMPOSITE_PROOF_SCHEMA,schema_version=1,id=proof.id,
        source=asdict(proof.source),parameters=list(proof.parameters),body=asdict(proof.body),
        carry_parameter=proof.carry_parameter,trip_count_parameter=proof.trip_count_parameter,
        parameter_map=proof.parameter_map,root_callee_source=asdict(proof.root_callee_source),
        root_global_name=proof.root_global_name,nested_callee_guards=[asdict(guard) for guard in proof.nested_callee_guards],
        compositions=refs,origins=list(proof.origins),profile=proof.profile,
        body_identity=proof.body_identity,transformation_authorized=False)
    save(path,document)
    return artifact_reference(path,COMPOSITE_PROOF_SCHEMA)


def _read_composite_invariant_loop_materials(path):
    """Decode pinned source/fusion materials without granting F4 authority.

    The derived body is not a newly declared Python callable.  Reading these
    bytes never substitutes for ``validate_composite_invariant_loop_proof``.
    """
    from scar.ir.execution_recipe import artifact_reference,read_artifact
    reference=artifact_reference(path,COMPOSITE_PROOF_SCHEMA);document=read_artifact(reference)
    if (set(document)!={'schema','schema_version',*CompositeInvariantLoopProof.__dataclass_fields__}
            or type(document['schema_version']) is not int or document['schema_version']!=1):
        raise ValueError('supported composite proof fields/version required')
    def source(value):
        value=dict(value);value['origins']=tuple(value.get('origins',()))
        return SourceRef(**value)
    steps=[]
    for ref in document['compositions']:
        retained=read_artifact(ref)
        if (set(retained)!={'schema','schema_version','pipeline','fusion_certificate'}
                or retained['schema']!=_COMPOSITION_SCHEMA or type(retained['schema_version']) is not int
                or retained['schema_version']!=1):
            raise ValueError('original composite pipeline shard required')
        pipeline=pipeline_from_dict(retained['pipeline']);body,_=fuse_pipeline_region(pipeline)
        verify_pipeline_fusion(pipeline,body,retained['fusion_certificate'])
        if body is None:raise ValueError('retained composite fusion is unsupported')
        steps.append(PipelineComposition(pipeline,retained['fusion_certificate']))
    data={name:document[name] for name in CompositeInvariantLoopProof.__dataclass_fields__}
    data.update(source=source(data['source']),root_callee_source=source(data['root_callee_source']),
        parameters=tuple(data['parameters']),body=region_from_dict(data['body']),origins=tuple(data['origins']),
        compositions=tuple(steps),nested_callee_guards=tuple(FunctionGlobalGuard(
            source(guard['owner_source']),guard['lookup_name'],source(guard['target_source']),
            tuple(guard['global_path']),tuple(guard.get('origins',())),guard['lookup_semantics'])
            for guard in data['nested_callee_guards']))
    proof=CompositeInvariantLoopProof(**data)
    if (proof.profile!='ORIGINAL_F4_COMPOSITE_ARRAY_LOOP_V1'
            or proof.body_identity!='DERIVED_FUSED_DAG_NOT_A_NEW_PYTHON_CALLABLE'
            or proof.transformation_authorized is not False or not proof.compositions):
        raise ValueError('original separate composite proof without source permission required')
    from scar.compiler.model import validate_region
    validate_region(proof.body)
    canonical=lambda value:json.dumps(asdict(value),sort_keys=True,separators=(',',':'),allow_nan=False)
    roots=[row for row in proof.compositions if row.pipeline.source==proof.root_callee_source]
    if len(roots)!=1 or canonical(fuse_pipeline_region(roots[0].pipeline)[0])!=canonical(proof.body):
        raise ValueError('retained composite body differs from its original root composition fusion')
    for original in (proof.source,proof.root_callee_source,
            *(guard.owner_source for guard in proof.nested_callee_guards),
            *(guard.target_source for guard in proof.nested_callee_guards),
            *(row.pipeline.source for row in proof.compositions),
            *(call.callee.source for row in proof.compositions for call in row.pipeline.calls)):
        if sha256(Path(original.path).read_bytes()).hexdigest()!=original.sha256:
            raise ValueError('composite original source bytes changed')
    read_artifact(reference,json_document=False)
    for ref in document['compositions']:read_artifact(ref,json_document=False)
    return proof,reference


def read_composite_invariant_loop_proof(path,*,catalog):
    """Read pinned composition shards and replay every F4/fusion/global hop."""
    from scar.ir.execution_recipe import read_artifact
    proof,reference=_read_composite_invariant_loop_materials(path)
    proof=validate_composite_invariant_loop_proof(proof,catalog=catalog)
    read_artifact(reference,json_document=False)
    return proof


def canonical_composite_invariant_motion_interface(proof):
    """Keep the composite source contract distinct from a direct leaf proof."""
    if type(proof) is not CompositeInvariantLoopProof:
        raise TypeError('original typed composite loop proof required')
    return dict(carry_parameter=proof.carry_parameter,
        readonly_parameters=[name for name in proof.body.parameters if name!=proof.carry_parameter],
        original_composite_f4_proof=asdict(proof))


def read_composite_invariant_loop_motion(path,*,catalog):
    """Join the original loop window to shared, source-bound callee effects."""
    from scar.analysis.array_motion import derive_array_motion
    from scar.analysis.array_call_effects import derive_composite_call_effects
    from scar.analysis.array_motion_legality import derive_array_motion_legality
    from scar.analysis.composite_call_effect_binding import derive_composite_call_effect_binding
    from scar.analysis.source_motion_window import derive_source_motion_window
    from scar.compiler.loop_admission import _motion_origin_entry
    from scar.ir.execution_recipe import artifact_reference
    from scar.optimization.resource_solver import semantic_sha256

    view=catalog if isinstance(catalog,ProgramView) else ProgramView(catalog)
    proof=read_composite_invariant_loop_proof(path,catalog=view)
    interface=canonical_composite_invariant_motion_interface(proof)
    motion=derive_array_motion(proof.body,carry_parameter=proof.carry_parameter,loop_interface=interface)
    def role(name):
        rows=[row for row in proof.origins if row.get('kind')=='original_f4' and row.get('role')==name]
        if len(rows)!=1:raise ValueError('one original composite loop/CALL coordinate required')
        return rows[0]
    identity=next(row for row in proof.source.origins if row.get('kind')=='source_identity')
    source=view.unit(identity['unit'])
    functions=[index for index,function in enumerate(source.facts.functions)
        if function.template.occurrence==identity['occurrence']]
    if len(functions)!=1:raise ValueError('composite source identity lacks one original F4 function')
    call=role('invariant_array_leaf_call')
    window=derive_source_motion_window(view,unit=identity['unit'],function_ordinal=functions[0],
        loop_header=_motion_origin_entry(role('single_shared_invariant_loop')),
        call_entry=_motion_origin_entry(call))
    effects=derive_composite_call_effects(proof)
    binding=derive_composite_call_effect_binding(view,proof,effects)
    legality=derive_array_motion_legality(proof.body,carry_parameter=proof.carry_parameter,
        loop_interface=interface,source_window=window,callee_effects=effects,callee_binding=binding)
    report=dict(schema='scar.source-composite-loop-motion',schema_version=1,
        proof=artifact_reference(path,COMPOSITE_PROOF_SCHEMA),proof_id=proof.id,
        proof_sha256=semantic_sha256(asdict(proof)),body_sha256=motion['body_sha256'],
        canonical_interface_sha256=semantic_sha256(interface),array_motion=motion,
        source_identity=dict(path=proof.source.path,sha256=proof.source.sha256,code_key=proof.source.code_key),
        source_window=window,callee_effects=effects,callee_binding=binding,movement_legality=legality,
        parameter_routes=[dict(leaf_formal=name,outer_parameter=proof.parameter_map[name],
            original_actual=call['call_inputs'][index],
            binding_role='CARRIED' if name==proof.carry_parameter else 'OUTER_LIVE_IN',
            referent_content_invariance='CONDITIONAL_ON_EXACT_ARRAY_AND_NO_FOREIGN_WRITERS')
            for index,name in enumerate(proof.body.parameters)],
        source_facts=dict(authority='FULL_ORIGINAL_COMPOSITE_F4_REDISCOVERY',
            bodies_shared=True,iterations_unrolled=False,derived_body_is_python_callable=False),
        runtime_obligations=motion['conditional_obligations'],runtime_obligations_discharged=False,
        source_replacement_authorized=False,cost_qualified=False,selected_for_use=False)
    report['id']='source-composite-loop-motion:'+semantic_sha256(report)
    return proof,report


def verify_composite_invariant_loop_motion(report,*,catalog):
    """Cold source replay of the separate composite motion publication."""
    from scar.ir.execution_recipe import read_artifact
    from scar.optimization.resource_solver import semantic_sha256
    if (type(report) is not dict or report.get('schema')!='scar.source-composite-loop-motion'
            or type(report.get('schema_version')) is not int or report['schema_version']!=1):
        raise ValueError('current composite source loop motion report required')
    read_artifact(report['proof'])
    proof,expected=read_composite_invariant_loop_motion(report['proof']['uri'],catalog=catalog)
    if semantic_sha256(expected)!=semantic_sha256(report):
        raise ValueError('composite source loop motion differs from original F4 replay')
    return proof,expected


def validate_composite_invariant_loop_links(proof,spec,loopsref):
    """Verify composite caller ports directly, without a schema-v1 proof cast."""
    from scar.compiler.middle.python import _invariant_loop_links
    from scar.ir.execution_recipe import read_artifact
    from scar.optimization.scope_accounting import _pointer
    if (type(proof) is not CompositeInvariantLoopProof or spec.get('id')!=proof.id
            or spec.get('source_code_key')!=proof.source.code_key):
        raise ValueError('composite specification differs from its original proof/source ID')
    document=read_artifact(loopsref)
    if document.get('schema')!='scar.source-loop-dependencies' or document.get('schema_version')!=1:
        raise ValueError('original source loop dependency publication required')
    expected,frontier=_invariant_loop_links(proof,document,loopsref)
    if frontier or spec.get('loop_link_frontier') or spec.get('loop_links')!=expected or len(expected)!=1:
        raise ValueError('composite loop/CALL FactRefs differ from original F4 header/result references')
    try:
        loop=_pointer(document,expected[0]['loop']['locator'])
        call=_pointer(document,expected[0]['call']['locator'])
        control=next(row for row in proof.origins if row.get('role')=='single_shared_invariant_loop')
        original=next(row for row in proof.origins if row.get('role')=='invariant_array_leaf_call')
        normal_world=[value for value in original['results'] if value['port']['completion']=='FALLTHROUGH'
            and value['port']['ordinal']==1]
        if (loop['body']['unit']!=control['unit'] or loop['body']['template']!=control['template']
                or loop['body']['region']['wire']!=control['body_region']
                or call['callee']!=original['call_callee'] or call['positional']!=original['call_inputs']
                or call['keywords'] or call['positional_pack'] is not None or call['keyword_pack'] is not None
                or call['world_before']!=original['arguments'][-1]
                or len(normal_world)!=1 or call['world_after']!=normal_world[0]
                or len(loop['slots'])!=2 or loop['slots'][0]['formal']!=control['iterator_parameter']
                or [slot['role'] for slot in loop['slots']]!=['INVARIANT_PREFIX','CARRIED_SUFFIX']
                or not loop['control_backedges'] or not loop['slots'][1]['backedges']):
            raise ValueError('composite loop/CALL operand, world or carry ports differ from original F4')
    except (KeyError,IndexError,StopIteration,TypeError) as error:
        raise ValueError('original composite loop/CALL fact or witness is absent') from error
    return expected


__all__=['CompositeInvariantLoopProof','PipelineComposition','FunctionGlobalGuard',
    'discover_pipeline_invariant_loops','validate_composite_invariant_loop_proof',
    'write_composite_invariant_loop_proof','read_composite_invariant_loop_proof',
    'canonical_composite_invariant_motion_interface','read_composite_invariant_loop_motion',
    'verify_composite_invariant_loop_motion',
    'validate_composite_invariant_loop_links','COMPOSITE_PROOF_SCHEMA']
