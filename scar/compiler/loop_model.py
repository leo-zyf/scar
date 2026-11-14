"""One guarded array carry and one exact-integer trip-count boundary."""
from dataclasses import dataclass,asdict
from hashlib import sha256
import json
from pathlib import Path

from scar.compiler.model import SourceRef
from scar.compiler.pipeline_model import (PipelinePlan,CalleeGuard,validate_pipeline,
    document as pipeline_document,pipeline_from_dict)

SCHEMA='scar.array-loop-plan'
PROFILE='cpu-array-loop-v1'


@dataclass(frozen=True)
class ArrayLoopPlan:
    id: str
    source: SourceRef
    parameters: tuple[str,...]
    state_parameter: str
    trip_count_parameter: str
    body: PipelinePlan
    body_callee: CalleeGuard
    origins: tuple[dict,...]=()
    schema_version: int=1
    profile: str=PROFILE


def validate_loop(loop):
    if type(loop)is not ArrayLoopPlan or loop.schema_version!=1 or loop.profile!=PROFILE:
        raise ValueError('supported typed array loop required')
    if (type(loop.id)is not str or not loop.id or type(loop.source)is not SourceRef
            or len(loop.parameters)!=2 or len(set(loop.parameters))!=2
            or any(type(name)is not str or not name for name in loop.parameters)
            or loop.state_parameter==loop.trip_count_parameter
            or {loop.state_parameter,loop.trip_count_parameter}!=set(loop.parameters)):
        raise ValueError('loop requires exactly one named array carry and one exact-integer parameter')
    validate_pipeline(loop.body)
    if len(loop.body.parameters)!=1:
        raise ValueError('loop body must accept exactly one array carry')
    if (type(loop.body_callee)is not CalleeGuard or loop.body_callee.source!=loop.body.source
            or loop.body_callee.lookup_root!='PIPELINE_FUNCTION_GLOBALS'
            or len(loop.body_callee.lookup_path)!=1
            or any(type(name)is not str or not name for name in loop.body_callee.lookup_path)):
        raise ValueError('loop body needs its original direct global function guard')
    return loop


def document(loop):
    validate_loop(loop)
    return dict(schema=SCHEMA,schema_version=loop.schema_version,profile=loop.profile,id=loop.id,
        source=asdict(loop.source),parameters=list(loop.parameters),state_parameter=loop.state_parameter,
        trip_count_parameter=loop.trip_count_parameter,body=pipeline_document(loop.body),
        body_callee=dict(lookup_root=loop.body_callee.lookup_root,lookup_path=list(loop.body_callee.lookup_path)),
        origins=list(loop.origins))


def loop_from_dict(value):
    value=dict(value)
    if value.pop('schema',None)!=SCHEMA:raise ValueError('unsupported loop schema')
    source=dict(value['source']);source['origins']=tuple(source.get('origins',()))
    value['source']=SourceRef(**source);value['parameters']=tuple(value['parameters'])
    value['body']=pipeline_from_dict(value['body']);guard=dict(value['body_callee'])
    if set(guard)!={'lookup_root','lookup_path'}:raise ValueError('body guard must reference its shared body source')
    guard['lookup_path']=tuple(guard['lookup_path']);guard['source']=value['body'].source
    value['body_callee']=CalleeGuard(**guard);value['origins']=tuple(value.get('origins',()))
    return validate_loop(ArrayLoopPlan(**value))


def loop_digest(loop):
    return sha256(json.dumps(document(loop),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


digest=loop_digest
def read_loop(path):return loop_from_dict(json.loads(Path(path).read_text()))
def write_loop(loop,path):Path(path).write_text(json.dumps(document(loop),indent=2,allow_nan=False)+'\n')


__all__=['ArrayLoopPlan','validate_loop','document','loop_from_dict','loop_digest','digest','read_loop','write_loop']
