"""Bounded input metadata at a version-checked original function boundary.

No tensor contents, address identity, logical-version equivalence or timing
claims are inferred. Collection can run with no selected transform, so the
original function supplies the actual workload behavior.
"""
from pathlib import Path
from hashlib import sha256
import json
import os
import uuid

class InputMetadataObserver:
    def __init__(self,max_calls_per_code=2):
        if type(max_calls_per_code)is not int or max_calls_per_code<1:raise ValueError('positive metadata observation limit required')
        self.limit=max_calls_per_code;self.counts={};self.records=[]
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
