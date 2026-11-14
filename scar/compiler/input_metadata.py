"""Bounded input metadata at a version-checked original function boundary.

No tensor contents, address identity, logical-version equivalence or timing
claims are inferred. Collection can run with no selected transform, so the
original function supplies the actual workload behavior.
"""
from pathlib import Path
from dataclasses import dataclass
from hashlib import sha256
import json
import os
import uuid
from .model import SourceRef

@dataclass(frozen=True)
class InputBoundary:
    source: SourceRef
    parameters: tuple[str,...]
    callees: tuple=()
    nested_callees: tuple=()
    max_calls: int=2
    scalar_parameters: tuple[str,...]=()

def read_input_boundary(path):
    """Read a declared code boundary; no artifact is loaded or transformed."""
    document=json.loads(Path(path).read_bytes())
    if document.get('schema')=='scar.array-loop-plan':
        from .loop_model import loop_from_dict
        loop=loop_from_dict(document)
        leaves=tuple({(call.callee.lookup_path,call.callee.source.code_key):call.callee
            for call in loop.body.calls}.values())
        return InputBoundary(loop.source,loop.parameters,(loop.body_callee,),
            ((loop.body_callee,leaves),),1,(loop.trip_count_parameter,))
    if document.get('schema')=='scar.array-pipeline-plan':
        from .pipeline_model import pipeline_from_dict
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
    def __init__(self,max_calls_per_code=2,*,scalar_parameters=()):
        if type(max_calls_per_code)is not int or max_calls_per_code<1:raise ValueError('positive metadata observation limit required')
        self.limit=max_calls_per_code;self.counts={};self.records=[]
        self.scalar_parameters=frozenset(scalar_parameters)
        self.run='metadata:'+uuid.uuid4().hex

    def __call__(self,source,parameters,values):
        import numpy as np
        key=(source.path,source.sha256,source.code_key)
        count=self.counts.get(key,0)
        if count>=self.limit:return False
        self.counts[key]=count+1
        inputs={}
        for name,value in zip(parameters,values,strict=True):
            if type(value)is np.ndarray:
                inputs[name]=dict(kind='EXACT_NUMPY_NDARRAY',shape=list(value.shape),dtype=value.dtype.str,
                    strides=list(value.strides),nbytes=value.nbytes,c_contiguous=bool(value.flags.c_contiguous),
                    aligned=bool(value.flags.aligned),device='cpu',contents_recorded=False)
            elif name in self.scalar_parameters and type(value)is int and -(1<<63)<=value<(1<<63):
                inputs[name]=dict(kind='EXACT_PYTHON_INT',value=value,contents_recorded=False,
                    scalar_value_recorded=True)
            else:inputs[name]=dict(kind='UNSUPPORTED_METADATA_TYPE',contents_recorded=False)
        self.records.append(dict(boundary_observation=dict(run=self.run,process_id=os.getpid(),ordinal=len(self.records)+1),
            source=dict(path=source.path,sha256=source.sha256,code_key=source.code_key,qualname=source.qualname),inputs=inputs))
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
