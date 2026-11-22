"""Bounded input metadata at a version-checked original function boundary.

No tensor contents, address identity, logical-version equivalence or timing
claims are inferred. Collection can run with no selected transform, so the
original function supplies the actual workload behavior.
"""
from pathlib import Path
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
import uuid
from threading import get_ident
from weakref import ref
from .model import SourceRef

@dataclass(frozen=True)
class InputBoundary:
    source: SourceRef
    parameters: tuple[str,...]
    callees: tuple=()
    nested_callees: tuple=()
    max_calls: int=2
    scalar_parameters: tuple[str,...]=()
    stateful_inputs: tuple=()
    region_id: str|None=None
    region_sha256: str|None=None

def read_input_boundary(path):
    """Read a declared code boundary; no artifact is loaded or transformed."""
    document=json.loads(Path(path).read_bytes())
    if document.get('schema')=='scar.stateful-array-region':
        from .stateful_model import stateful_from_dict,stateful_digest
        region=stateful_from_dict(document)
        return InputBoundary(region.source,region.parameter_names,
            stateful_inputs=tuple((row.id,row.path,row.kind) for row in region.inputs),
            region_id=region.id,region_sha256=stateful_digest(region))
    if document.get('schema')=='scar.array-loop-plan':
        from .loop_model import loop_from_dict
        loop=loop_from_dict(document)
        leaves=tuple({(call.callee.lookup_path,call.callee.source.code_key):call.callee
            for call in loop.body.calls}.values())
        return InputBoundary(loop.source,loop.parameters,(loop.body_callee,),
            ((loop.body_callee,leaves),),1,(loop.trip_count_parameter,))
    if document.get('schema')=='scar.array-pipeline-plan':
        from scar.compiler.pipeline_frontend import pipeline_from_dict
        pipeline=pipeline_from_dict(document)
        return InputBoundary(pipeline.source,pipeline.parameters,
            tuple({(call.callee.lookup_path,call.callee.source.code_key):call.callee
                for call in pipeline.calls}.values()))
    from .model import plan_from_dict
    plan=plan_from_dict(document)
    return InputBoundary(plan.region.source,plan.region.parameters)

def callee_metadata(guards,namespace):
    """Observe selected global function code IDs, without reading tensor data."""
    from types import FunctionType
    from scar.trace.code_snapshot import code_key
    records=[]
    for guard in guards:
        actual=namespace.get(guard.lookup_path[0])
        record=dict(lookup_path=list(guard.lookup_path),
            declared_code_key=guard.source.code_key,source_sha256_verified=False)
        if type(actual)is FunctionType:
            observed=code_key(actual.__code__)
            record.update(kind='EXACT_PYTHON_FUNCTION',observed_code_key=observed,
                observed_filename=actual.__code__.co_filename,
                declared_code_matches=observed==guard.source.code_key)
        else:record.update(kind='FUNCTION_CODE_NOT_OBSERVED',declared_code_matches=False)
        records.append(record)
    return records

def boundary_callee_metadata(boundary,namespace):
    """Follow declared callable environments rather than flattening globals."""
    from types import FunctionType
    records=callee_metadata(boundary.callees,namespace)
    for parent,guards in boundary.nested_callees:
        actual=namespace.get(parent.lookup_path[0])
        children=callee_metadata(guards,actual.__globals__) if type(actual)is FunctionType else []
        for record in records:
            if record['lookup_path']==list(parent.lookup_path):
                record['callee_environment_bindings']=children
    return records

class InputMetadataObserver:
    def __init__(self,max_calls_per_code=2,*,scalar_parameters=(),stateful_inputs=(),
                 region_id=None,region_sha256=None,allocation_epochs=None):
        if type(max_calls_per_code)is not int or max_calls_per_code<1:raise ValueError('positive metadata observation limit required')
        self.limit=max_calls_per_code;self.counts={};self.records=[]
        self.scalar_parameters=frozenset(scalar_parameters)
        self.stateful_inputs=tuple(stateful_inputs)
        self.region_id=region_id;self.region_sha256=region_sha256
        self.allocation_epochs=allocation_epochs
        self.run='metadata:'+uuid.uuid4().hex
        self._storage_observations={}

    def __call__(self,source,parameters,values):
        import numpy as np
        key=(source.path,source.sha256,source.code_key)
        count=self.counts.get(key,0)
        if count>=self.limit:return False
        self.counts[key]=count+1
        inputs={}
        bound=dict(zip(parameters,values,strict=True))
        for name,value in bound.items():
            if type(value)is np.ndarray:
                inputs[name]=dict(kind='EXACT_NUMPY_NDARRAY',shape=list(value.shape),dtype=value.dtype.str,
                    strides=list(value.strides),nbytes=value.nbytes,c_contiguous=bool(value.flags.c_contiguous),
                    aligned=bool(value.flags.aligned),device='cpu',contents_recorded=False)
            elif name in self.scalar_parameters and type(value)is int and -(1<<63)<=value<(1<<63):
                inputs[name]=dict(kind='EXACT_PYTHON_INT',value=value,contents_recorded=False,
                    scalar_value_recorded=True)
            else:inputs[name]=dict(kind='UNSUPPORTED_METADATA_TYPE',contents_recorded=False)
        if self.stateful_inputs:
            # Share delivery's exact ordinary-instance field protocol. Never
            # use getattr or evaluate a property to obtain observed inputs.
            from scar.runtime.stateful_guard import _input
            from scar.analysis.ndarray_view_contract import NdarrayView, ndarray_view_contract
            inputs={}
            owners=[]
            history_key=(key,get_ident())
            previous=self._storage_observations.get(history_key,{})
            current={}
            for identifier,path,declared_kind in self.stateful_inputs:
                item=dict(input_path=list(path),declared_kind=declared_kind,contents_recorded=False)
                try:value=_input(path,bound)
                except (TypeError,ValueError,KeyError,AttributeError,StopIteration):
                    item.update(kind='UNKNOWN',reason='UNSAFE_OR_UNAVAILABLE_DECLARED_INPUT_PATH')
                    inputs[identifier]=item;continue
                if type(value)is np.ndarray:
                    item.update(kind='EXACT_NUMPY_NDARRAY',shape=list(value.shape),dtype=value.dtype.str,
                        strides=list(value.strides),nbytes=value.nbytes,c_contiguous=bool(value.flags.c_contiguous),
                        aligned=bool(value.flags.aligned),device='cpu')
                    storage=ndarray_view_contract(value)
                    if type(storage) is NdarrayView:
                        ordinal=next((i for i, owner in enumerate(owners) if owner is storage.owner), len(owners))
                        if ordinal==len(owners):
                            owners.append(storage.owner)
                        geometry=storage.geometry()
                        item['storage_observation']=dict(geometry,owner_ordinal=ordinal)
                        if self.allocation_epochs is not None:
                            epoch=self.allocation_epochs(storage.owner)
                            if epoch is not None:item['allocator_epoch']=epoch
                        if identifier in previous:
                            index, owner, old_geometry=previous[identifier]
                            item['previous_storage_observation']=dict(boundary_ordinal=index,
                                same_owner=owner() is storage.owner,
                                same_geometry=old_geometry==geometry)
                        current[identifier]=(len(self.records)+1,ref(storage.owner),geometry)
                    else:
                        item['storage_frontier']=storage.reason
                elif type(value) in (int,float,np.float64):
                    # Scalars remain fresh runtime inputs, never compile-time
                    # constants inferred from this bounded observation.
                    finite=type(value)is int or math.isfinite(value)
                    item.update(kind='EXACT_NUMERIC_SCALAR',python_type=('numpy.float64' if type(value)is np.float64
                        else type(value).__name__),finite=finite,scalar_value_recorded=False)
                    if type(value)is int:item['within_exact_float64_integer_range']=abs(value)<=2**53
                else:item.update(kind='UNKNOWN',reason='UNSUPPORTED_METADATA_TYPE')
                inputs[identifier]=item
            if self.counts[key]<self.limit:
                self._storage_observations[history_key]=current
            else:
                self._storage_observations.pop(history_key,None)
        self.records.append(dict(boundary_observation=dict(run=self.run,process_id=os.getpid(),ordinal=len(self.records)+1),
            source=dict(path=source.path,sha256=source.sha256,code_key=source.code_key,qualname=source.qualname),inputs=inputs))
        if self.stateful_inputs:
            self.records[-1]['boundary']=dict(schema='scar.stateful-array-region',region_id=self.region_id,
                region_sha256=self.region_sha256,parameters=list(parameters))
        return self.counts[key]<self.limit

    def document(self):
        return dict(schema='scar.region-input-metadata',schema_version=1,records=self.records,
            max_calls_per_code=self.limit,tensor_contents_recorded=False,
            observation_ids_are_trace_invocation_keys=False,logical_versions_inferred=False)

    def write(self,path):Path(path).write_text(json.dumps(self.document(),indent=2,allow_nan=False)+'\n')

def shapes_from_metadata(path,source,parameters):
    """Select a single observed shape context, never an assumed grid size."""
    import numpy as np
    path=Path(path).absolute();raw=path.read_bytes();document=json.loads(raw)
    if (document.get('schema'),document.get('schema_version'))!=('scar.region-input-metadata',1):
        raise ValueError('versioned input metadata required')
    contexts=[]
    for row in document['records']:
        ref=row.get('source',{})
        if (ref.get('path'),ref.get('sha256'),ref.get('code_key'))!=(source.path,source.sha256,source.code_key):continue
        if set(row['inputs'])!=set(parameters):raise ValueError('observed parameters differ from region')
        shapes={}
        for name in parameters:
            value=row['inputs'][name];shape=value.get('shape',[])
            if (value.get('kind')!='EXACT_NUMPY_NDARRAY' or len(shape)!=2
                    or any(type(x)is not int or x<0 for x in shape)
                    or np.dtype(value['dtype'])!=np.dtype('float64')
                    or not value.get('c_contiguous') or not value.get('aligned')):
                raise ValueError('observed input is outside the contiguous float64 rank-two profile')
            strides=value.get('strides',[])
            if (len(strides)!=2 or any(type(s)is not int for s in strides)
                    or value.get('nbytes')!=shape[0]*shape[1]*8
                    or all(shape) and any(extent>1 and stride!=expected for extent,stride,expected in
                        zip(shape,strides,(shape[1]*8,8)))):
                raise ValueError('observed shape, strides and byte size are inconsistent')
            shapes[name]=shape
        if shapes not in contexts:contexts.append(shapes)
    if len(contexts)!=1:
        raise ValueError('one observed shape context required; missing or polymorphic inputs need separate guarded plans')
    return contexts[0],dict(path=str(path),sha256=sha256(raw).hexdigest(),code_key=source.code_key,
        shape_authority='OBSERVED_FUNCTION_BOUNDARY_METADATA',all_future_calls_proven=False)


def stateful_shapes_from_metadata(path,region):
    """Admit one observed state-input geometry, not source or future-call proof.

    Consumers still rediscover the original F4 region and retain every runtime
    guard. Scalar values and tensor contents are neither stored nor specialized.
    """
    import numpy as np
    from .stateful_model import read_stateful,stateful_from_dict,stateful_digest,validate_stateful_region
    if isinstance(region,(str,Path)):region=read_stateful(region)
    elif type(region)is dict:region=stateful_from_dict(region)
    else:region=validate_stateful_region(region)
    path=Path(path).resolve();raw=path.read_bytes();document=json.loads(raw)
    if (document.get('schema'),document.get('schema_version'))!=('scar.region-input-metadata',1):
        raise ValueError('versioned input metadata required')
    if document.get('tensor_contents_recorded')is not False:
        raise ValueError('metadata must explicitly exclude tensor contents')
    source=region.source
    if sha256(Path(source.path).read_bytes()).hexdigest()!=source.sha256:
        raise ValueError('original stateful source differs from observed boundary')
    identity=dict(schema='scar.stateful-array-region',region_id=region.id,
        region_sha256=stateful_digest(region),parameters=list(region.parameter_names))
    contexts=[];observations=0
    if type(document.get('records'))is not list:raise ValueError('stateful metadata records required')
    for row in document['records']:
        ref=row.get('source',{})
        if (ref.get('path'),ref.get('sha256'),ref.get('code_key'))!=(source.path,source.sha256,source.code_key):continue
        if ref.get('qualname')!=source.qualname or row.get('boundary')!=identity:
            raise ValueError('observed stateful source/region identity differs')
        if type(row.get('inputs'))is not dict or set(row['inputs'])!={item.id for item in region.inputs}:
            raise ValueError('observed stateful input IDs differ')
        shapes={}
        for spec in region.inputs:
            value=row['inputs'][spec.id]
            if (type(value)is not dict or value.get('input_path')!=list(spec.path) or value.get('declared_kind')!=spec.kind
                    or value.get('contents_recorded')is not False):
                raise ValueError('observed stateful input path/kind differs')
            if spec.kind!='array':
                if (value.get('kind')!='EXACT_NUMERIC_SCALAR' or value.get('finite')is not True
                        or value.get('python_type')not in ('int','float','numpy.float64')
                        or value.get('python_type')=='int' and value.get('within_exact_float64_integer_range')is not True
                        or value.get('scalar_value_recorded')is not False):
                    raise ValueError('observed stateful scalar is outside the finite exact profile')
                continue
            shape=value.get('shape',[]);strides=value.get('strides',[])
            if (value.get('kind')!='EXACT_NUMPY_NDARRAY' or value.get('device')!='cpu'
                    or type(shape)is not list or len(shape)!=2 or any(type(x)is not int or x<0 for x in shape)
                    or type(value.get('dtype'))is not str or np.dtype(value['dtype'])!=np.dtype('float64')
                    or value.get('c_contiguous')is not True or value.get('aligned')is not True
                    or type(strides)is not list or len(strides)!=2 or any(type(s)is not int for s in strides)
                    or value.get('nbytes')!=shape[0]*shape[1]*8
                    or all(shape) and any(extent>1 and stride!=expected for extent,stride,expected
                        in zip(shape,strides,(shape[1]*8,8)))):
                raise ValueError('observed stateful array is outside the float64 C rank-two profile')
            shapes[spec.id]=shape
        observations+=1
        if shapes not in contexts:contexts.append(shapes)
    if len(contexts)!=1:
        raise ValueError('one observed stateful shape context required; missing or polymorphic inputs need separate guarded plans')
    return contexts[0],dict(artifact=dict(uri=str(path),sha256=sha256(raw).hexdigest(),size_bytes=len(raw),
        schema='scar.region-input-metadata'),region_sha256=identity['region_sha256'],source_code_key=source.code_key,
        shape_authority='OBSERVED_SOURCE_MATCHED_STATEFUL_INPUT_PATH_METADATA',observations=observations,
        all_future_calls_proven=False,runtime_guards_required=True,original_F4_admission_required=True,
        tensor_contents_recorded=False,scalar_values_specialized=False)

def loop_context_from_metadata(path,loop):
    """Read one guarded array/scalar context; observations do not prove future calls."""
    import numpy as np
    path=Path(path).absolute();raw=path.read_bytes();document=json.loads(raw)
    if (document.get('schema'),document.get('schema_version'))!=('scar.region-input-metadata',1):
        raise ValueError('versioned input metadata required')
    contexts=[]
    for row in document['records']:
        ref=row.get('source',{})
        if (ref.get('path'),ref.get('sha256'),ref.get('code_key'))!=(loop.source.path,loop.source.sha256,loop.source.code_key):continue
        if set(row['inputs'])!=set(loop.parameters):raise ValueError('observed loop parameters differ')
        array=row['inputs'][loop.state_parameter];scalar=row['inputs'][loop.trip_count_parameter]
        shape=array.get('shape',[]);strides=array.get('strides',[])
        if (array.get('kind')!='EXACT_NUMPY_NDARRAY' or len(shape)!=2
                or any(type(x)is not int or x<0 for x in shape)
                or np.dtype(array['dtype'])!=np.dtype('float64')
                or not array.get('c_contiguous') or not array.get('aligned')
                or len(strides)!=2 or any(type(x)is not int for x in strides)
                or array.get('nbytes')!=shape[0]*shape[1]*8
                or all(shape) and any(extent>1 and stride!=expected for extent,stride,expected
                    in zip(shape,strides,(shape[1]*8,8)))):
            raise ValueError('observed loop array is outside the exact float64 C rank-two profile')
        count=scalar.get('value')
        if scalar.get('kind')!='EXACT_PYTHON_INT' or type(count)is not int or not -(1<<63)<=count<(1<<63):
            raise ValueError('one exact signed int64 loop count required')
        context=({loop.state_parameter:shape},count)
        if context not in contexts:contexts.append(context)
    if len(contexts)!=1:raise ValueError('one observed loop shape and scalar context required')
    shapes,count=contexts[0]
    return shapes,count,dict(path=str(path),sha256=sha256(raw).hexdigest(),
        code_key=loop.source.code_key,authority='OBSERVED_ARRAY_METADATA_AND_EXACT_SCALAR',
        tensor_contents_recorded=False,all_future_calls_proven=False)
