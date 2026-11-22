"""Recover ordered input roots through original F4 formal/field interfaces.

The input is an already source-replayed StatefulDependencyProvider. Names in
its input paths are matched against that same original function's declared
parameter slots, never against an inline callee or an external caller. This
layer emits records only; a matched activation provider owns cross-namespace
context, control and argument edges.
"""
from types import SimpleNamespace as NS

from scar.analysis.source_content_interfaces import ContentInterfaceIndex
from scar.analysis.source_namespace_interfaces import NamespacePlanView
from scar.analysis.source_reference_slice import ReferenceSlicer,source_port
from scar.ir.program import normal_graph as F, normal_graph as T, normal_graph as G, source_productions as P
from scar.ir.program import data_materials as C, ordered_regions as R
from scar.ir.program.providers import ProviderArtifact,ProviderFactReference
from scar.ir.program.data_materials import ContentAccessRef
from scar.ir.program.data_materials import FieldSelector
from scar.ir.program.source_template_links import SourceTemplateKey
from scar.ir.record_codec import decode


_BINDING_GUARDS=(
    'ORIGINAL_OUTER_FUNCTION_ACTIVATION_AND_ORDERED_ACTIVATION_MATCHED',
    'CURRENT_OUTER_FRAME_BINDINGS_SELECT_RECORDED_REFERENCE_PATH',
    'SOURCE_PARAMETER_AND_REGION_INPUT_PROFILE_CONTRACTS_HOLD',
)
_FIELD_GUARDS=(
    'ORDINARY_RECEIVER_DICTIONARY_FIELDS_NO_DESCRIPTORS',
    'SAME_SELECTED_RECEIVER_AND_FIELD_INSTANCES_AT_ORIGINAL_READ_WORLD',
    'ORIGINAL_FIELD_GET_NORMAL_COMPLETION_AND_EXCEPTION_CONTRACT_SELECTED',
    'NO_UNMODELED_FIELD_CALLBACK_OR_FOREIGN_CONCURRENT_REBINDING',
    'FIELD_REFERENCE_ROUTE_DOES_NOT_PROVE_ARRAY_CONTENT_VERSION_EQUALITY',
)


class _Boundary(ValueError):
    pass


class _InputResolver:
    def __init__(self,ordered):
        ordered.check();self.ordered=ordered;self.region=ordered.region;self.view=ordered.view
        identities=[origin for origin in self.region.source.origins if origin.get('kind')=='source_identity']
        if len(identities)!=1:raise ValueError('ordered inputs require their unique original source identity')
        identity=identities[0];self.source=self.view.unit(identity['unit']);self.source.check()
        candidates=[(i,f)for i,f in enumerate(self.source.facts.functions)
            if f.template.occurrence==identity['occurrence']]
        if len(candidates)!=1:raise ValueError('ordered inputs require the original outer function')
        self.function_index,self.function=candidates[0]
        if (self.function.parameter_names!=self.region.parameter_names or self.function.body_scope is None
                or len(self.function.formals)!=len(self.function.parameter_names)+2):
            raise ValueError('ordered input parameters differ from the original outer function ABI')
        snapshot=NS(fact_store=self.source.material.fact_store,rows=lambda *args,**kwargs:(),
            validate_parents=self.source.material.validate_parents)
        self.slicer=ReferenceSlicer(snapshot);self.interface=ContentInterfaceIndex(self.source)
        self.reads={};self.paths={};self.content=None
        reference=ordered.reference
        self.artifact=ProviderArtifact(reference['uri'],reference['sha256'],reference['size_bytes'],reference['schema'])
        self.function_fact=self.slicer.fact(self.source.facts,'functions',self.function_index)
        self.attributes={}
        for index,attribute in enumerate(self.source.facts.attributes):
            if attribute.action is F.AttributeAction.GET:
                self.attributes.setdefault(source_port(self.source.facts,attribute.value),[]).append(index)

    def path(self,endpoint):
        if endpoint not in self.paths:self.paths[endpoint]=self.slicer.slice(endpoint,max_states=20000,max_call_depth=128)
        return self.paths[endpoint]

    @staticmethod
    def leaves(path):
        return {path.states[i].port for i in set(range(len(path.states)))-{e.downstream for e in path.transitions}}

    def root_path(self,endpoint,formal):
        path=self.path(endpoint)
        if self.leaves(path)!={formal}:
            raise _Boundary('REFERENCE_PATH_DOES_NOT_SELECT_THIS_OUTER_FORMAL_OR_REQUIRES_INLINE_CALLER_BINDING')
        return path

    def read(self,result):
        if result in self.reads:return self.reads[result]
        if result.unit!=self.source.unit or type(result.port.owner)is not P.ProductionStepID:
            raise _Boundary('FIELD_RESULT_IS_NOT_AN_OUTER_SOURCE_OPERATION_RESULT')
        self.view.source_port(result)
        operation=self.source._node(result.port.template,result.port.owner)
        self.source.ensure(operation);step=self.source.records[operation.node]
        context=NamespacePlanView(self.source.unit,self.source.plans[result.port.template])
        if context.owner(step.region).owner!=self.function.body_scope.owner:
            raise _Boundary('FIELD_READ_BELONGS_TO_INLINE_OR_FOREIGN_FUNCTION_SCOPE')
        interface=self.interface.identify(context,step)
        if (interface is None or interface.kind!='GET' or type(interface.selector)is not FieldSelector
                or interface.result!=result or interface.attribute_fact is None):
            raise _Boundary('INPUT_ORIGIN_DOES_NOT_SELECT_ONE_ORIGINAL_ATTRIBUTE_GET_RESULT')
        if self.content is None:self.content=self.view.state(self.source.unit).content(complete=False)
        self.content.ensure_operation(operation)
        matches=[index for index in self.content.operation_accesses[operation.node]
            if self.content.accesses[index].kind is C.ContentRequestKind.GET
            and self.content.accesses[index].result==result
            and self.content.accesses[index].attribute_fact==interface.attribute_fact]
        if len(matches)!=1:raise _Boundary('ORIGINAL_FIELD_READ_HAS_NO_UNIQUE_CONTENT_EFFECT_REQUEST')
        access_index=matches[0];access=self.content.accesses[access_index]
        result_record=(interface,access,ContentAccessRef(self.source.unit,access_index,operation.node),
            self.slicer.fact(self.source.facts,'attributes',interface.attribute_fact))
        self.reads[result]=result_record
        return result_record

    def chain(self,result,fields,formal):
        """Reconstruct from the exact last GET, not another same-named field."""
        interface,access,access_ref,fact=self.read(result)
        if not fields or interface.selector.name!=fields[-1]:
            raise _Boundary('ORIGINAL_FIELD_SELECTOR_DIFFERS_FROM_REPLAYED_INPUT_PATH')
        path=self.path(interface.receiver)
        if len(fields)==1:
            self.root_path(interface.receiver,formal);prefix=()
        else:
            leaves=self.leaves(path)
            if len(leaves)!=1 or not leaves<=self.attributes.keys():
                raise _Boundary('NESTED_FIELD_RECEIVER_HAS_NO_UNIQUE_ORIGINAL_PRECEDING_GET')
            prefix=self.chain(next(iter(leaves)),fields[:-1],formal)
        return (*prefix,R.OrderedFieldRead(interface.selector,interface.receiver,result,
            access.world_before,access.world_after,fact,access_ref,access.operation,path))

    def origin_result(self,origin,fields):
        if origin.get('kind')!='original_f4' or origin.get('unit')!=self.source.unit:
            raise _Boundary('INPUT_ORIGIN_IS_NOT_ORIGINAL_OUTER_FUNCTION_F4')
        if not fields and origin.get('value')is not None:
            return decode(T.GlobalPort,origin['value'])
        if 'step'not in origin:raise _Boundary('INPUT_ORIGIN_HAS_NO_ORIGINAL_VALUE_OR_READ_STEP')
        template=decode(SourceTemplateKey,origin['template']);prefix=P.ProductionStepID.prefix+':'
        if not origin['step'].startswith(prefix):raise _Boundary('INPUT_ORIGIN_STEP_ID_DOMAIN_DIFFERS')
        operation=self.source._node(template,P.ProductionStepID(origin['step'][len(prefix):]))
        self.source.ensure(operation);step=self.source.records[operation.node]
        context=NamespacePlanView(self.source.unit,self.source.plans[template])
        if context.owner(step.region).owner!=self.function.body_scope.owner:
            raise _Boundary('INPUT_ORIGIN_BELONGS_TO_INLINE_OR_FOREIGN_FUNCTION_SCOPE')
        if fields:
            interface=self.interface.identify(context,step)
            if interface is None or interface.kind!='GET':raise _Boundary('FIELD_INPUT_ORIGIN_IS_NOT_A_GET_REQUEST')
            return interface.result
        return context.result(step)

    def derive(self):
        records=[];boundaries=[]
        for index,input in enumerate(self.region.inputs):
            key='#/inputs/'+str(index);record_ref=ProviderFactReference(self.artifact,key)
            try:parameter=self.function.parameter_names.index(input.path[0])
            except ValueError:
                boundaries.append(R.OrderedInputBoundary(key,'INPUT_ROOT_IS_NOT_AN_ORIGINAL_OUTER_PARAMETER'));continue
            formal=source_port(self.source.facts,self.function.formals[parameter+1]);fields=input.path[1:]
            if not input.origins:
                boundaries.append(R.OrderedInputBoundary(key,'INPUT_HAS_NO_REPLAYED_ORIGINAL_ORIGIN'));continue
            for oi,origin in enumerate(input.origins):
                evidence=ProviderFactReference(self.artifact,key+'/origins/'+str(oi))
                try:
                    result=self.origin_result(origin,fields)
                    if fields:chain=self.chain(result,fields,formal);path=None
                    else:chain=();path=self.root_path(result,formal)
                    records.append(R.OrderedInputRoute(key,record_ref,evidence,self.function_fact,
                        formal,parameter,fields,result,path,chain,(*_BINDING_GUARDS,*(_FIELD_GUARDS if fields else ()))))
                except _Boundary as reason:
                    boundaries.append(R.OrderedInputBoundary(key,str(reason),evidence))
        self.ordered.check();self.source.check()
        return tuple(records),tuple(boundaries)


def derive_ordered_inputs(ordered_provider):
    """Return typed source-side routes and specific original input boundaries."""
    return _InputResolver(ordered_provider).derive()


__all__=['derive_ordered_inputs']
