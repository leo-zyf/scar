"""Conditional array and pipeline effects bound to original F4 calls.

Shared callee summaries do not merge original CALL occurrences.  Input
provenance is a value dependency, not another read of its original storage.
Original F4 admission, runtime dispatch and live contracts remain external.
"""
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

from scar.analysis.source_body_interfaces import BodyInterfaceResolver
from scar.analysis.source_field_interfaces import _InterfaceOnly
from scar.analysis.source_function_connections import Invocation, connect
from scar.analysis.source_references import source_port
from scar.compiler.graph_frontend import _origin
from scar.compiler.model import ArrayRegion, ExecutionPlan, validate_plan, validate_region
from scar.compiler.pipeline_frontend import SourceFunctionComponents, document as pipeline_document, validate_pipeline
from scar.compiler.pipeline_loop_frontend import CompositeInvariantLoopProof
from scar.ir.execution_recipe import artifact_reference
from scar.ir.program import normal_graph as T, source_productions as P
from scar.ir.record_codec import encode
from scar.optimization.graph import value_kinds
from scar.optimization.optimizer import optimize_region, _RUNTIME_GUARDS, _EXTERNAL_ASSUMPTIONS
from scar.optimization.pipeline_fusion import fusion_region_digest, replay_pipeline_fusion
from scar.optimization.resource_solver import semantic_sha256


SCHEMA = 'scar.composite-call-effects'
_BINDING_SCHEMA = 'scar.composite-call-effect-binding'
_ARITHMETIC = {'add','sub','mul','div','neg'}
_FIELDS = ('reads','writes','allocates','frees','aliases','escapes','rng_effect','external_effect','ordering_effect','may_raise')
_MOTION_Q = ('NO_INTERMEDIATE_ESCAPE_OR_IDENTITY_OBSERVERS','NO_INTERSTEP_CALLBACKS_OR_HOOKS',
    'NO_FOREIGN_CONCURRENT_INPUT_WRITERS','FP_STATUS_FLAGS_ARE_NOT_PROGRAM_OUTPUTS',
    'EXACT_ARRAY_AND_CALLABLE_RUNTIME_GUARDS','FP_AND_FAILURE_FALLBACK','FRESH_PRIVATE_RESULTS_AND_BUDGET')


def _source_key(source):
    return source.path, source.sha256, source.code_key


def _condition(status, obligations=(), blockers=(), **fields):
    return dict(status=status,obligations=list(obligations),blockers=deepcopy(list(blockers)),**fields)


def _effect_fields(known, *, reads=(), writes=(), allocations=(), output=None, order=()):
    if not known:
        return {field:dict(status='UNKNOWN',scope='UNRESOLVED_CALLEE_OR_PROFILE') for field in _FIELDS}
    return dict(
        reads=dict(status='CONDITIONAL',events=deepcopy(list(reads))),
        writes=dict(status='CONDITIONAL',input_referent_writes=[],fresh_value_writes=deepcopy(list(writes))),
        allocates=dict(status='CONDITIONAL',logical_fresh_results=deepcopy(list(allocations)),
            physical_allocator_calls='UNKNOWN',allocation_failure_sequence='UNKNOWN'),
        frees=dict(status='UNKNOWN',scope='OBJECT_LIFETIME_REFCOUNT_RELEASE_AND_ALLOCATOR_CALLBACK_ORDER'),
        aliases=dict(status='CONDITIONAL',formal_inputs='MAY_ALIAS',read_read_aliasing_allowed=True,
            result='FRESH_NOT_INPUT_STORAGE_UNDER_GUARDS',physical_storage_identities_proven=False),
        escapes=dict(status='CONDITIONAL',returned_value=output,
            explicit_intermediate_publication='NONE_IN_CLOSED_VALUE_PROFILE',outside_observers='UNKNOWN'),
        rng_effect=dict(status='CONDITIONAL_NONE',scope='EXPLICIT_CLOSED_NUMERIC_PRIMITIVE_RNG_ACTIONS'),
        external_effect=dict(status='CONDITIONAL_NONE',scope='EXPLICIT_CLOSED_PRIMITIVE_IO_OR_UNKNOWN_CALL_ACTIONS',
            allocator_and_release_callback_effects='UNKNOWN',whole_python_function_effects='UNKNOWN'),
        ordering_effect=dict(status='CONDITIONAL',original_order=list(order),operand_order_preserved=True,
            lifetime_and_failure_order='UNKNOWN',reordering_authorized=False),
        may_raise=dict(status='UNKNOWN',sources=['NUMPY_FLOATING_STATUS','SHAPE_OR_TYPE_REJECTION',
            'ALLOCATION_OR_RELEASE_FAILURE'],normal_completion_only=True))


def derive_array_effects(value):
    """Recompute primitive-profile facts independently of supplied proof flags."""
    plan = value if type(value) is ExecutionPlan else None
    region = plan.region if plan is not None else value
    if type(region) is not ArrayRegion:
        raise TypeError('existing ArrayRegion or ExecutionPlan required')
    digest = fusion_region_digest(region)
    blockers, fresh_blockers = [], []
    guards, external = [], []
    descriptor_result = False
    try:
        replay = optimize_region(region)
        result = next(node for node in region.nodes if node.id==region.output)
        descriptor_result = (result.op in ('input','slice') and
            all(node.op in ('input','const','slice') for node in region.nodes) and
            replay.proof.get('reasons')==['fresh_array_computation_output_required'])
        if descriptor_result:
            # Existing profile validation has established the descriptor DAG,
            # and explicitly rejected freshness. Preserve its read-only facts
            # without making it a callable fresh-output component.
            fresh_blockers.append(dict(reason='DESCRIPTOR_RESULT_IS_NOT_FRESH',expression=region.output,op=result.op))
            guards = [guard for guard in _RUNTIME_GUARDS if guard!='NO_INPUT_WRITES_FRESH_OUTPUT_EACH_CALL']
            external = list(_EXTERNAL_ASSUMPTIONS)
        elif replay.proof.get('supported') is not True or replay.proof.get('legal') is not True:
            raise ValueError('recomputed closed primitive profile remains unsupported: '+str(replay.proof.get('reasons')))
        else:
            if plan is not None:
                validate_plan(plan)
            validate_region(region)
            guards = list(replay.proof['runtime_guards'])
            external = list(replay.proof['legality']['external_assumptions'])
    except (ValueError,TypeError,KeyError) as error:
        blockers.append(dict(reason='CLOSED_ARRAY_PROFILE_REPLAY_FAILED',detail=str(error),
            operations=[dict(expression=node.id,op=node.op) for node in region.nodes if node.op not in
                _ARITHMETIC|{'input','const','slice','copy'}]))
    provenance, descriptors, reads, fresh = {}, {}, [], []
    if not blockers:
        kinds = value_kinds(region.nodes)
        for node in region.nodes:
            provenance[node.id] = ({node.attrs['name']} if node.op=='input' else
                set().union(*(provenance[key] for key in node.inputs)))
            if node.op=='input':
                descriptors[node.id] = dict(parameter=node.attrs['name'],slice_chain=[])
            elif node.op=='slice':
                parent = descriptors[node.inputs[0]]
                descriptors[node.id] = dict(parameter=parent['parameter'],
                    slice_chain=[*parent['slice_chain'],deepcopy(node.attrs['bounds'])])
            elif node.op in _ARITHMETIC|{'copy'}:
                for ordinal,operand in enumerate(node.inputs):
                    if kinds[operand]!='array':
                        continue
                    reads.append(dict(expression=node.id,operand_index=ordinal,value=operand,
                        kind='FORMAL_REFERENT_READ' if operand in descriptors else 'IMMUTABLE_EXPR_VALUE_READ',
                        parameter=descriptors[operand]['parameter'] if operand in descriptors else None,
                        slice_chain=deepcopy(descriptors[operand]['slice_chain']) if operand in descriptors else None,
                        input_provenance=sorted(provenance[operand]),
                        provenance_is_additional_input_read=False))
                if kinds[node.id]=='array':
                    fresh.append(node.id)
    known = not blockers
    effects = _effect_fields(known,reads=reads,writes=fresh,allocations=fresh,
        output=region.output,order=[node.id for node in region.nodes])
    if known and descriptor_result:
        effects['aliases']['result']='ALIASES_FORMAL_INPUT_DESCRIPTOR_NOT_FRESH'
        effects['escapes']['returned_input_alias']=sorted(provenance[region.output])
    result = dict(id='array-effect:'+semantic_sha256(dict(source=asdict(region.source),body_sha256=digest)),
        kind='ARRAY_REGION',source=asdict(region.source),body_sha256=digest,parameters=list(region.parameters),
        status='CONDITIONAL' if known and not fresh_blockers else 'UNRESOLVED',
        read_formals=sorted({row['parameter'] for row in reads if row['parameter'] is not None}) if known else None,
        output_provenance=sorted(provenance[region.output]) if known else None,
        input_writes=_condition('CONDITIONAL_NONE' if known else 'UNKNOWN',_MOTION_Q,blockers),
        fresh_result=_condition('CONDITIONAL_FRESH' if known and not fresh_blockers else 'UNKNOWN',guards,[*blockers,*fresh_blockers]),
        effects=effects,
        runtime_guards=guards,external_assumptions=external,blockers=[*blockers,*fresh_blockers],
        allocation_scope='LOGICAL_ARRAY_RESULT_REQUIREMENTS_NOT_PHYSICAL_ALLOCATOR_CALL_COUNTS',
        represented_body_only=True,source_f4_replayed=False,whole_python_purity_proven=False,
        source_replacement_authorized=False)
    return result


class _Composition:
    def __init__(self, proof):
        self.proof = proof
        self.proof_sha = semantic_sha256(asdict(proof))
        self.materials, self.by_source = [], defaultdict(list)
        self.callees, self.calls, self.active = {}, [], set()
        for index, composition in enumerate(proof.compositions):
            pipeline = composition.pipeline
            blockers = []
            try:
                validate_pipeline(pipeline)
                required = {'EXACT_NUMPY_NDARRAY_FLOAT64_CPU_2D','ACTUAL_GLOBAL_CALLEE_CODE_IDS_MATCH',
                    'FRESH_LEAF_OUTPUTS','NO_INPUT_WRITES'}
                if not required<=set(pipeline.assumptions):
                    raise ValueError('original closed pipeline boundary contracts are missing: '+str(sorted(required-set(pipeline.assumptions))))
                body = replay_pipeline_fusion(pipeline, composition.fusion_certificate)
                if body is None:
                    raise ValueError('composition fusion profile has explicit unresolved frontiers')
            except (ValueError,TypeError,KeyError) as error:
                body = None
                blockers.append(dict(reason='COMPOSITION_FUSION_REPLAY_FAILED',composition=index,detail=str(error)))
            material = dict(pipeline=pipeline,body=body,blockers=blockers,index=index,
                pipeline_sha256=semantic_sha256(pipeline_document(pipeline)) if not blockers else semantic_sha256(asdict(pipeline)))
            self.materials.append(material)
            self.by_source[_source_key(pipeline.source)].append(index)

    def unknown(self, source, body_sha, reason, **fields):
        identity = 'unresolved-array-effect:'+semantic_sha256(dict(source=asdict(source),body_sha256=body_sha,reason=reason,**fields))
        if identity not in self.callees:
            blocker = dict(reason=reason,**fields)
            self.callees[identity] = dict(id=identity,kind='UNRESOLVED',source=asdict(source),body_sha256=body_sha,
                parameters=[],read_formals=None,output_provenance=None,status='UNRESOLVED',blockers=[blocker],
                input_writes=_condition('UNKNOWN',_MOTION_Q,[blocker]),fresh_result=_condition('UNKNOWN',(),[blocker]),
                effects=_effect_fields(False),runtime_guards=[],external_assumptions=[])
        return self.callees[identity]

    def child(self, call):
        candidates = self.by_source.get(_source_key(call.callee.source),())
        if any(index in self.active for index in candidates):
            return self.unknown(call.callee.source,fusion_region_digest(call.plan.region),
                'RECURSIVE_COMPOSITION_NOT_EXPANDED',candidate_compositions=list(candidates))
        matches = []
        for index in candidates:
            material = self.materials[index]
            if material['body'] is None:
                continue
            replay = optimize_region(material['body'],memory_budget_bytes=call.plan.cost.get('memory_budget_bytes'),
                cpu_threads=call.plan.cpu_threads)
            if fusion_region_digest(replay.region)==fusion_region_digest(call.plan.region):
                matches.append(index)
        if len(matches)==1:
            return self.pipeline(matches[0])
        if candidates:
            return self.unknown(call.callee.source,fusion_region_digest(call.plan.region),
                'REGISTERED_COMPOSITION_BODY_NOT_UNIQUELY_MATCHED',candidate_compositions=list(candidates))
        summary = derive_array_effects(call.plan)
        self.callees.setdefault(summary['id'],summary)
        return summary

    def pipeline(self, index):
        material = self.materials[index]; pipeline = material['pipeline']; body = material['body']
        digest = fusion_region_digest(body) if body is not None else material['pipeline_sha256']
        identity = 'pipeline-effect:'+semantic_sha256(dict(source=asdict(pipeline.source),body_sha256=digest))
        if index in self.active:
            return self.unknown(pipeline.source,digest,'RECURSIVE_COMPOSITION_NOT_EXPANDED',composition=index)
        if identity in self.callees:
            return self.callees[identity]
        self.active.add(index)
        blockers = deepcopy(material['blockers'])
        values = {value:dict(kind='CALLER_INPUT',input_provenance={parameter},producer_call=None)
            for parameter,value in zip(pipeline.parameters,pipeline.input_values)}
        read_formals, direct_reads, call_refs, guards, external = set(), [], [], set(), set()
        for ordinal,call in enumerate(pipeline.calls):
            occurrence = 'array-call-effect:'+semantic_sha256(dict(pipeline_sha256=material['pipeline_sha256'],call_id=call.id,index=ordinal))
            child = self.child(call)
            call_blockers = deepcopy(child['blockers'])
            try:
                validate_plan(call.plan)
            except (ValueError,TypeError) as error:
                call_blockers.append(dict(reason='CALL_EXECUTION_PLAN_REJECTED',detail=str(error)))
            matches = [i for i,guard in enumerate(self.proof.nested_callee_guards)
                if guard.owner_source==pipeline.source and guard.target_source==call.callee.source
                and guard.lookup_name==call.callee.lookup_path[0]
                and guard.lookup_semantics=='FUNCTION_GLOBALS_AT_EACH_HOP']
            if not matches:
                call_blockers.append(dict(reason='ORIGINAL_FUNCTION_GLOBAL_GUARD_MISSING',call_id=call.id))
            bindings = []
            if len(call.inputs)!=len(call.plan.region.parameters):
                call_blockers.append(dict(reason='CALL_FORMAL_ACTUAL_ARITY_MISMATCH',call_id=call.id))
            for formal,actual in zip(call.plan.region.parameters,call.inputs):
                value = values.get(actual)
                if value is None:
                    call_blockers.append(dict(reason='CALL_ACTUAL_VALUE_NOT_READY',actual=actual))
                    value = dict(kind='UNRESOLVED',input_provenance=set(pipeline.parameters),producer_call=None)
                bindings.append(dict(formal=formal,actual_value=actual,actual_kind=value['kind'],
                    producer_call=value['producer_call'],input_provenance=sorted(value['input_provenance']),
                    provenance_is_additional_input_read=False))
            by_formal = {row['formal']:row for row in bindings}
            content_reads = child['read_formals'] if child['read_formals'] is not None else list(call.plan.region.parameters)
            reads = []
            for formal in content_reads:
                binding = by_formal.get(formal)
                if binding is None:
                    call_blockers.append(dict(reason='CALLEE_READ_FORMAL_NOT_BOUND',formal=formal))
                    continue
                read = dict(call_id=call.id,formal=formal,actual_value=binding['actual_value'],
                    kind='CALLER_INPUT_REFERENT_READ' if binding['actual_kind']=='CALLER_INPUT' else 'IMMUTABLE_CALL_RESULT_READ',
                    producer_call=binding['producer_call'],input_provenance=list(binding['input_provenance']),
                    provenance_is_additional_input_read=False,
                    footprint_scope='CALLEE_BOUNDARY_TRANSITIVE_READ_ENVELOPE',
                    primitive_load_count=None,additive_across_summary_levels=False,
                    status='CONDITIONAL' if child['read_formals'] is not None else 'UNKNOWN_CALLEE_READ_ENVELOPE')
                reads.append(read); direct_reads.append(read)
                if binding['actual_kind']=='CALLER_INPUT':
                    read_formals.update(binding['input_provenance'])
            output_formals = child['output_provenance'] if child['output_provenance'] is not None else list(call.plan.region.parameters)
            provenance = set().union(*(set(by_formal[name]['input_provenance']) for name in output_formals if name in by_formal))
            values[call.output] = dict(kind='CALL_RESULT',input_provenance=provenance,producer_call=call.id)
            row = dict(id=occurrence,owner_summary=identity,pipeline_id=pipeline.id,pipeline_sha256=material['pipeline_sha256'],
                call_id=call.id,index=ordinal,callee_summary=child['id'],callee_source=asdict(call.callee.source),
                inputs=list(call.inputs),formals=list(call.plan.region.parameters),output=call.output,
                argument_bindings=bindings,direct_reads=reads,output_input_provenance=sorted(provenance),
                output_provenance_status='CONDITIONAL' if child['output_provenance'] is not None else 'UNKNOWN',
                original_global_guard_refs=[dict(typed_proof_sha256=self.proof_sha,
                    locator='#/nested_callee_guards/'+str(i)) for i in matches],
                source_ref=dict(typed_proof_sha256=self.proof_sha,locator='#/compositions/'+str(index)+'/pipeline/calls/'+str(ordinal)),
                execution_plan_sha256=semantic_sha256(asdict(call.plan)),status='UNRESOLVED' if call_blockers else 'CONDITIONAL',
                blockers=call_blockers,effects_ref=dict(callee_summary=child['id'],field='effects'),runtime_invocation_observed=False)
            self.calls.append(row); call_refs.append(occurrence)
            guards.update(child['runtime_guards']); external.update(child['external_assumptions'])
            blockers.extend(dict(reason='CALLEE_EFFECTS_OR_BINDING_UNRESOLVED',call_id=call.id,details=call_blockers)
                for _ in (0,) if call_blockers)
        self.active.remove(index)
        known = not blockers
        output = values.get(pipeline.output)
        if output is None:
            blockers.append(dict(reason='PIPELINE_OUTPUT_VALUE_NOT_READY')); known=False
        summary = dict(id=identity,kind='PIPELINE',source=asdict(pipeline.source),body_sha256=digest,
            pipeline_id=pipeline.id,pipeline_sha256=material['pipeline_sha256'],composition_index=index,
            parameters=list(pipeline.parameters),status='CONDITIONAL' if known else 'UNRESOLVED',blockers=blockers,
            effect_scope='SUBTREE_BOUNDARY_ENVELOPE_NOT_ADDITIONAL_EXECUTION_OR_MEMORY_TRAFFIC',
            additive_with_child_effect_summaries=False,
            read_formals=sorted(read_formals) if known else None,
            output_provenance=sorted(output['input_provenance']) if output is not None and known else None,
            call_occurrences=call_refs,input_writes=_condition('CONDITIONAL_NONE' if known else 'UNKNOWN',_MOTION_Q,blockers),
            fresh_result=_condition('CONDITIONAL_FRESH' if known else 'UNKNOWN',sorted(guards),blockers),
            effects=_effect_fields(known,reads=direct_reads,writes=[call.output for call in pipeline.calls],
                allocations=[call.output for call in pipeline.calls],output=pipeline.output,order=[call.id for call in pipeline.calls]),
            runtime_guards=sorted(guards),external_assumptions=sorted(external),
            source_f4_replayed=False,whole_python_purity_proven=False,source_replacement_authorized=False)
        self.callees[identity] = summary
        return summary


def derive_composite_call_effects(proof):
    """Compose source-scoped contracts, retaining every original CALL ABI."""
    if (type(proof) is not CompositeInvariantLoopProof or proof.profile!='ORIGINAL_F4_COMPOSITE_ARRAY_LOOP_V1'
            or proof.body_identity!='DERIVED_FUSED_DAG_NOT_A_NEW_PYTHON_CALLABLE'
            or proof.transformation_authorized is not False):
        raise ValueError('typed original composite proof with no transformation permission required')
    validate_region(proof.body)
    builder = _Composition(proof)
    body_sha = fusion_region_digest(proof.body)
    candidates = [index for index,material in enumerate(builder.materials) if material['body'] is not None
        and material['pipeline'].source==proof.root_callee_source and fusion_region_digest(material['body'])==body_sha]
    blockers = []
    if len(candidates)==1:
        root = builder.pipeline(candidates[0])
    else:
        root = builder.unknown(proof.root_callee_source,body_sha,'ROOT_COMPOSITION_NOT_UNIQUELY_BOUND',candidates=candidates)
    blockers.extend(deepcopy(root['blockers']))
    root_guards = [index for index,guard in enumerate(proof.nested_callee_guards)
        if guard.owner_source==proof.source and guard.target_source==proof.root_callee_source
        and guard.lookup_name==proof.root_global_name and guard.global_path==(proof.root_global_name,)
        and guard.lookup_semantics=='FUNCTION_GLOBALS_AT_EACH_HOP']
    if not root_guards:
        blockers.append(dict(reason='ROOT_FUNCTION_GLOBAL_GUARD_MISSING'))
    edges = defaultdict(list); indegree = {identity:0 for identity in builder.callees}
    for call in builder.calls:
        edges[call['owner_summary']].append(call)
        indegree[call['callee_summary']] += 1
    counts = defaultdict(int); counts[root['id']]=1
    pending = deque(sorted(key for key,value in indegree.items() if value==0))
    visited = set()
    while pending:
        identity = pending.popleft(); visited.add(identity)
        for call in edges[identity]:
            target = call['callee_summary']; counts[target] += counts[identity]; indegree[target] -= 1
            if indegree[target]==0:pending.append(target)
    for call in builder.calls:
        call['multiplicity_per_root_call'] = counts[call['owner_summary']] if call['owner_summary'] in visited else None
        call['multiplicity_scope'] = 'STATIC_NORMAL_COMPLETION_SOURCE_CALL_TEMPLATES_NOT_OBSERVED_INVOCATIONS'
    known = root['status']=='CONDITIONAL' and not blockers
    guards = sorted(set(root['runtime_guards'])|{'ACTUAL_GLOBAL_CALLEE_CODE_IDS_MATCH'})
    obligations = [dict(id=value,status='UNRESOLVED') for value in sorted(set(_MOTION_Q)|set(guards)|set(root['external_assumptions']))]
    result = dict(schema=SCHEMA,schema_version=1,proof_sha256=builder.proof_sha,body_sha256=body_sha,
        derived_body_sha256=body_sha,root_source=asdict(proof.root_callee_source),root_summary=root['id'],
        root_global_guard_refs=[dict(typed_proof_sha256=builder.proof_sha,
            locator='#/nested_callee_guards/'+str(index)) for index in root_guards],
        status='CONDITIONAL' if known else 'UNRESOLVED',blockers=blockers,
        input_writes=_condition('CONDITIONAL_NONE' if known else 'UNKNOWN',[row['id'] for row in obligations],blockers),
        fresh_result=_condition('CONDITIONAL_FRESH' if known else 'UNKNOWN',guards,blockers),
        source_window_no_input_clobber=_condition('CONDITIONAL' if known else 'UNRESOLVED',_MOTION_Q,blockers),
        caller_exposed_effects=deepcopy(root['effects']),callees=[builder.callees[key] for key in sorted(builder.callees)],
        calls=sorted(builder.calls,key=lambda row:(row['pipeline_id'],row['index'],row['id'])),obligations=obligations,
        summary=dict(shared_callees=len(builder.callees),static_call_sites=len(builder.calls),
            source_call_templates_per_root_call=sum(call['multiplicity_per_root_call'] for call in builder.calls)
                if known and all(call['multiplicity_per_root_call'] is not None for call in builder.calls) else None),
        scope='ONE_SELECTED_COMPOSITE_CALL_NORMAL_CLOSED_ARRAY_PROFILE_UNDER_RUNTIME_AND_EXTERNAL_Q',
        producer=artifact_reference(__file__,'scar.python-producer'),source_f4_replayed=False,
        runtime_guards_discharged=False,whole_python_purity_proven=False,full_effects_closed=False,
        source_replacement_authorized=False,physical_alias_identities_proven=False,
        recursive_calls_expanded=False,cost_or_timing_inferred=False,target_executed=False)
    result['id']='composite-call-effects:'+semantic_sha256(result)
    return result


def verify_composite_call_effects(report, proof):
    expected = derive_composite_call_effects(proof)
    if semantic_sha256(expected)!=semantic_sha256(report):
        raise ValueError('composite call effects differ from complete profile/composition/ABI replay')
    return deepcopy(expected)



def _identity(source):
    rows=[row for row in source.origins if row.get('kind')=='source_identity']
    if len(rows)!=1 or rows[0].get('code_key')!=source.code_key:
        raise ValueError('one original source identity with the exact CodeID required')
    return rows[0]


def _source_label(source):
    return dict(path=source.path,sha256=source.sha256,code_key=source.code_key,
        source_ref_sha256=semantic_sha256(asdict(source)))


def derive_composite_call_effect_binding(view,proof,effects):
    """Recheck supplied effects before binding the original F4 interfaces."""
    if type(proof) is not CompositeInvariantLoopProof:
        raise TypeError('the separate original composite proof is required')
    verify_composite_call_effects(effects,proof)
    return _bind_composite_call_effects(view,proof,effects)


def derive_bound_composite_call_effects(view,proof):
    """Derive effects and bind their original CALL interfaces in one projection."""
    effects=derive_composite_call_effects(proof)
    return effects,_bind_composite_call_effects(view,proof,effects)


def _bind_composite_call_effects(view,proof,effects):
    from scar.analysis.source_motion_window import _motion_origin_entry

    routes=SourceFunctionComponents(view);slicer=routes.resolver.slicer
    resolver=BodyInterfaceResolver(view);functions={};callees=[];frontier=[]

    def function(source):
        key=_source_key(source);identity=_identity(source)
        if key in functions:
            original,record,_=functions[key]
            if original.facts.unit!=identity['unit'] or record.template.occurrence!=identity['occurrence']:
                raise ValueError('one CodeID cannot acquire a different original F4 identity')
            return functions[key]
        original=view.unit(identity['unit']);facts=original.facts
        model=view.catalog.fact_store.model_units[identity['unit']]
        if (str(Path(model['path']).absolute())!=str(Path(source.path).absolute())
                or sha256(Path(source.path).read_bytes()).hexdigest()!=source.sha256):
            raise ValueError('callee source identity differs from original source bytes/unit')
        matches=[(i,row) for i,row in enumerate(facts.functions)
            if row.template.occurrence==identity['occurrence']]
        if len(matches)!=1:raise ValueError('callee occurrence is not one original function')
        ordinal,record=matches[0]
        if record.body_scope is None:raise ValueError('original function body scope required')
        body=T.Body(facts.unit,record.template,record.body)
        resolver.prepare(body)
        index=len(callees)
        callees.append(dict(index=index,source=_source_label(source),
            identity=identity,definition=encode(slicer.fact(facts,'functions',ordinal)),
            source_function=encode(source_port(facts,record.value)),body=encode(body),
            formals=[encode(source_port(facts,value)) for value in record.formals],
            normal_return=None if record.returns is None else encode(source_port(facts,record.returns)),
            return_world=None if record.return_world is None else encode(source_port(facts,record.return_world)),
            code_identity_basis='ENCLOSING_FULL_COMPOSITE_F4_REDISCOVERY_OF_RETAINED_F0_CODE_ID',
            body_shared=True,body_copied=False))
        value=original,record,index
        functions[key]=value
        return value

    function(proof.source);function(proof.root_callee_source)
    calls=[];call_keys={};namespace_routes=[];namespace_indexes={}

    def namespace_route(owner,name,selected):
        if selected is None:return None
        key=owner.facts.unit,name,selected.unit,selected.function.template
        if key not in namespace_indexes:
            namespace_indexes[key]=len(namespace_routes)
            namespace_routes.append(selected.evidence)
        return namespace_indexes[key]

    def bind_call(owner_source,target_source,name,origin,*,pipeline_id,call_id,
                  inputs=None,output=None,parameters=None):
        owner,owner_function,owner_index=function(owner_source)
        target,target_function,target_index=function(target_source)
        entry=_motion_origin_entry(origin)
        if entry.unit!=owner.facts.unit:
            raise ValueError('CALL origin belongs to a different source unit')
        if (entry.unit,entry.port.template) not in resolver.prepared:
            raise ValueError('CALL template is outside its original source function closure')
        context=resolver.plan(entry.unit,entry.port.template)
        steps=[row for row in context.plan.steps if row.id==entry.port.owner]
        if len(steps)!=1:raise ValueError('CALL origin is not one original step')
        step=steps[0]
        if (type(step.instruction) is not P.ApplyLanguageAction
                or step.instruction.action is not P.LanguageActionKind.CALL
                or context.owner(step.region).owner!=owner_function.body_scope.owner):
            raise ValueError('original CALL must be inside its claimed function code scope')
        operands=[encode(context.value(value)) for value in step.arguments]
        expected=_origin(context,step.region,step)
        if any(origin.get(key)!=value for key,value in expected.items()):
            raise ValueError('CALL original production/source/result coordinates changed')
        if origin.get('arguments')!=operands:
            raise ValueError('CALL origin operands differ from the original production')
        records=[(i,row) for i,row in enumerate(owner.facts.calls)
            if source_port(owner.facts,row.result)==context.result(step)]
        if len(records)!=1:raise ValueError('one exact original CALL fact required')
        ordinal,record=records[0]
        port=lambda value:encode(source_port(owner.facts,value))
        call_fact=dict(result=port(record.result),callee=port(record.callee),
            positional=[port(value) for value in record.positional],
            keywords=[dict(name=key,value=port(value)) for key,value in record.keywords])
        if 'call_fact' in origin and origin['call_fact']!=call_fact:
            raise ValueError('pipeline CALL fact differs from original F4 operands')
        if 'call_callee' in origin and (origin['call_callee']!=call_fact['callee']
                or origin['call_inputs']!=call_fact['positional']
                or origin['call_result']!=call_fact['result']):
            raise ValueError('outer composite CALL differs from original F4 operands')
        expected_operands=[port(record.callee),*call_fact['positional'],
            *(row['value'] for row in call_fact['keywords']),port(record.world)]
        if (operands!=expected_operands or context.worlds(step)!=(
                source_port(owner.facts,record.world),source_port(owner.facts,record.result_world))):
            raise ValueError('original CALL operand/world ABI differs')
        if parameters is not None and tuple(parameters)!=target_function.parameter_names:
            raise ValueError('typed leaf formal order differs from its original source function')
        selected=routes.targets(owner.facts.unit).get(name)
        target_matches=(selected is not None and selected.unit==target.facts.unit
            and selected.function==target_function)
        if not target_matches:
            frontier.append(dict(reason='ORIGINAL_FUNCTION_GLOBAL_TARGET_ROUTE_OPEN',
                pipeline_id=pipeline_id,call_id=call_id,owner=owner_index,lookup_name=name))
        mini=_InterfaceOnly(slicer,target.facts,target_function)
        offset=slicer.store.offsets[owner.facts.unit]
        invocation=Invocation(offset+record.callee,tuple(offset+value for value in record.positional),
            tuple((key,offset+value) for key,value in record.keywords),offset+record.result,
            offset+record.world,offset+record.result_world,
            None if record.positional_pack is None else offset+record.positional_pack,
            None if record.keyword_pack is None else offset+record.keyword_pack)
        connection=connect(mini,mini.offsets[target.facts.unit]+target_function.value,invocation,
            conditions=('ORIGINAL_COMPOSITE_NAMESPACE_ROUTE_AND_CURRENT_FUNCTION_GLOBAL_CODE_IDS_MATCH',))
        if (connection is None or connection.activation is not T.FunctionActivation.NORMAL
                or connection.binding is not T.ArgumentBinding.DIRECT):
            frontier.append(dict(reason='ORIGINAL_CALL_SIGNATURE_NOT_DIRECT',
                pipeline_id=pipeline_id,call_id=call_id))
        row=dict(index=len(calls),pipeline_id=pipeline_id,call_id=call_id,
            owner=owner_index,callee=target_index,lookup_name=name,entry=encode(entry),
            fact=encode(slicer.fact(owner.facts,'calls',ordinal)),
            source_origin=dict(unit=origin['unit'],template=origin['template'],step=origin['step']),
            callee_port=call_fact['callee'],positional=call_fact['positional'],keywords=call_fact['keywords'],
            result=call_fact['result'],world_before=port(record.world),world_after=port(record.result_world),
            positional_pack=None if record.positional_pack is None else port(record.positional_pack),
            keyword_pack=None if record.keyword_pack is None else port(record.keyword_pack),
            connection=None if connection is None else encode(connection),
            logical_inputs=None if inputs is None else list(inputs),logical_output=output,
            conditional_target=target_matches,
            namespace_route=namespace_route(owner,name,selected),runtime_dispatch_proven=False)
        key=(_source_key(owner_source),_source_key(target_source),semantic_sha256(row['source_origin']))
        if key in call_keys:raise ValueError('one source CALL appeared twice in shared composition definitions')
        call_keys[key]=row['index'];calls.append(row)
        return row

    outer=[row for row in proof.origins if row.get('kind')=='original_f4'
        and row.get('role')=='invariant_array_leaf_call']
    if len(outer)!=1:raise ValueError('one outer original composite CALL required')
    outer_call=bind_call(proof.source,proof.root_callee_source,proof.root_global_name,outer[0],
        pipeline_id=None,call_id='OUTER_LOOP_BODY_CALL',parameters=proof.body.parameters)
    for composition in proof.compositions:
        pipeline=composition.pipeline
        for index,call in enumerate(pipeline.calls):
            origins=[row for row in call.origins if row.get('kind')=='original_f4' and row.get('call_index')==index]
            if len(origins)!=1:raise ValueError('one original static CALL occurrence required')
            bind_call(pipeline.source,call.callee.source,call.callee.lookup_path[0],origins[0],
                pipeline_id=pipeline.id,call_id=call.id,inputs=call.inputs,output=call.output,
                parameters=call.plan.region.parameters)
    guards=[];covered=set()
    for index,guard in enumerate(proof.nested_callee_guards):
        current=proof.source;path=[]
        for name in guard.global_path:
            original,_,owner_index=function(current)
            selected=routes.targets(original.facts.unit).get(name)
            if selected is None:
                frontier.append(dict(reason='FUNCTION_GLOBAL_GUARD_PATH_OPEN',guard=index,path=list(guard.global_path)))
                break
            choices=[key for key,(unit,record,_) in functions.items()
                if unit.facts.unit==selected.unit and record==selected.function]
            # Every path hop is an existing source function referenced by the proof.
            if not choices:
                candidates=[guard.owner_source,guard.target_source,
                    *(row.pipeline.source for row in proof.compositions),
                    *(call.callee.source for row in proof.compositions for call in row.pipeline.calls)]
                for candidate in candidates:function(candidate)
                choices=[key for key,(unit,record,_) in functions.items()
                    if unit.facts.unit==selected.unit and record==selected.function]
            if len(choices)!=1:raise ValueError('guard path has no unique retained source function')
            target_key=choices[0]
            target_source=next(candidate for candidate in (proof.source,proof.root_callee_source,
                guard.owner_source,guard.target_source,
                *(row.pipeline.source for row in proof.compositions),
                *(call.callee.source for row in proof.compositions for call in row.pipeline.calls))
                if _source_key(candidate)==target_key)
            path.append(dict(owner=owner_index,lookup_name=name,callee=function(target_source)[2],
                namespace_route=namespace_route(original,name,selected)))
            if len(path)==len(guard.global_path) and (_source_key(current)!=_source_key(guard.owner_source)
                    or target_key!=_source_key(guard.target_source) or name!=guard.lookup_name):
                raise ValueError('function-global path differs from its exact owner/target source identity')
            current=target_source
        if not guard.global_path or guard.lookup_semantics!='FUNCTION_GLOBALS_AT_EACH_HOP':
            raise ValueError('explicit original function-global lookup path required')
        occurrences=[]
        for origin in guard.origins:
            if origin.get('kind')!='original_f4':continue
            coordinate=dict(unit=origin['unit'],template=origin['template'],step=origin['step'])
            key=(_source_key(guard.owner_source),_source_key(guard.target_source),semantic_sha256(coordinate))
            if key not in call_keys:raise ValueError('global guard origin is not one bound source CALL')
            occurrences.append(call_keys[key]);covered.add(call_keys[key])
        if not occurrences:raise ValueError('global guard has no original CALL provenance')
        guards.append(dict(index=index,owner_source=_source_label(guard.owner_source),
            target_source=_source_label(guard.target_source),lookup_name=guard.lookup_name,
            global_path=list(guard.global_path),lookup_semantics=guard.lookup_semantics,
            source_call_indexes=occurrences,path=path,
            status='CONDITIONAL' if len(path)==len(guard.global_path) else 'UNRESOLVED'))
    if covered!=set(range(len(calls))):
        raise ValueError('every original source CALL needs a recorded function-global guard')
    for original,_,_ in functions.values():original.check()
    slicer.store.validate_parents()
    report=dict(schema=_BINDING_SCHEMA,schema_version=1,
        status='CONDITIONAL' if not frontier else 'UNRESOLVED',
        proof_sha256=semantic_sha256(asdict(proof)),body_sha256=semantic_sha256(asdict(proof.body)),
        callee_effects_sha256=semantic_sha256(effects),root_source=asdict(proof.root_callee_source),
        source=asdict(proof.source),outer_call=outer_call,callees=callees,calls=calls,guards=guards,
        namespace_routes=namespace_routes,
        source_refs=[encode(original.material.result_artifact) for original in
            {original.facts.unit:original for original,_,_ in functions.values()}.values()],
        frontier=frontier,source_f4_authority='ENCLOSING_FULL_COMPOSITE_F4_REDISCOVERY_REQUIRED',
        obligations=['ACTUAL_FUNCTION_GLOBAL_CODE_IDS_MATCH_AT_EVERY_RECORDED_HOP',
            'EXACT_ARRAY_PROFILE_AND_STABLE_REFERENT_CONTENTS',
            'EXCEPTION_FAILURE_AND_EXTERNAL_EFFECT_CONDITIONS_REMAIN_OPEN'],
        source_replacement_authorized=False,runtime_guards_discharged=False,
        effects_unconditionally_closed=False,source_bodies_copied=False,iterations_unrolled=False)
    report['id']='composite-call-effect-binding:'+semantic_sha256(report)
    return report


def verify_composite_call_effect_binding(report,view,proof,effects):
    expected=derive_composite_call_effect_binding(view,proof,effects)
    if semantic_sha256(report)!=semantic_sha256(expected):
        raise ValueError('composite effect binding differs from original F4 interfaces')
    return expected


__all__ = ['derive_array_effects', 'derive_composite_call_effects', 'verify_composite_call_effects',
           'derive_bound_composite_call_effects', 'derive_composite_call_effect_binding',
           'verify_composite_call_effect_binding']
