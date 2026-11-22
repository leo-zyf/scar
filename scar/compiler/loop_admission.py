"""Typed loop plans and replay of their body, source and placement contracts."""

from dataclasses import dataclass, asdict
from hashlib import sha256
import json
import sys
from pathlib import Path
from scar.compiler.model import SourceRef, document as plan_document, plan_from_dict, read_plan
from scar.compiler.pipeline_frontend import (
    PipelinePlan,
    CalleeGuard,
    validate_pipeline,
    document as pipeline_document,
    pipeline_from_dict,
)
from scar.compiler.validation import validate_plan


# One guarded array carry and one exact-integer trip-count boundary.
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


# Shared legality for the existing synchronous CPU temporal-loop profile.
#
# This admission owns source/body/control/shape/payload and retained placement
# checks. It does not generate C++, select instructions, compile, or execute the
# application. Live array, CPU, fenv and callable guards remain execution duties.
def _require(condition, message):
    if not condition:
        raise ValueError(message)


def check_loop_sources(loop):
    """Verify original loop/body/leaf source snapshots without importing them."""
    seen = set()
    for source in (loop.source, loop.body.source, *(call.callee.source for call in loop.body.calls)):
        key = (source.path, source.sha256, source.code_key)
        if key in seen:
            continue
        seen.add(key)
        _require(bool(source.code_key) and Path(source.path).is_absolute(),
                 "original loop/body/leaf source versions required")
        _require(sha256(Path(source.path).read_bytes()).hexdigest() == source.sha256,
                 "loop or nested callee source snapshot differs")


def _component(plan):
    # Shared structural validation covers schema/profile/ABI, effects, exact
    # schedule, attributes, constants, DAG closure, resource and payload fields.
    validate_plan(plan)
    if plan.resource_binding is not None:
        _require(sys.platform == "linux", "C++ CPU affinity binding requires Linux")
        _require(all(cpu < 2**20 for cpu in plan.resource_binding.cpu_ids),
                 "C++ CPU affinity binding supports Linux CPU IDs below 2**20")
    scalar = {}
    for node in plan.region.nodes:
        if node.op in ("input", "slice"):
            scalar[node.id] = False
        elif node.op == "const":
            scalar[node.id] = True
        else:
            scalar[node.id] = all(scalar[operand] for operand in node.inputs)
            _require(not scalar[node.id], "scalar-only operations must be folded before C++ lowering")


def admit_loop(loop, strategy, execution_plan):
    """Return the replayed strategy and exact bound body plan for lowering.

    The body is one shared one-array DAG, not an ordinary one-step substitute
    for the original loop. Source-file I/O is separate in check_loop_sources.
    """
    from scar.optimization.hardware import bind_cpu_placement
    from scar.optimization.loop_strategy import verify_loop_strategy, loop_execution_plan
    from scar.optimization.pipeline_fusion import fuse_pipeline_region, verify_pipeline_fusion

    strategy = verify_loop_strategy(loop, strategy)
    _require(strategy["selected"]["kind"] == "TEMPORAL_FUSION" and
             strategy["capability"] == "EXACT_ELEMENTWISE_TEMPORAL_FUSION",
             "loop backend requires the selected exact elementwise temporal profile")
    raw, _ = fuse_pipeline_region(loop.body)
    _require(raw is not None, "loop body fusion has unresolved source obligations")
    verify_pipeline_fusion(loop.body, raw, strategy["body_fusion_certificate"])
    expected = loop_execution_plan(loop, strategy)
    actual = (read_plan(execution_plan) if isinstance(execution_plan, (str, Path))
              else plan_from_dict(plan_document(execution_plan)))
    _require(actual.resource_binding is not None, "loop execution requires an explicit recorded CPU binding")
    _require(sys.platform == "linux", "CPU-bound loop execution requires Linux")
    _component(expected)
    _component(actual)
    _require(plan_document(actual) == plan_document(bind_cpu_placement(expected, actual.resource_binding)),
             "loop body plan differs from exact replay or admitted CPU binding")
    _require(actual.region.source == loop.body.source and actual.region.parameters == loop.body.parameters
             and len(actual.region.parameters) == 1 and
             all(node.op in ("input", "const", "add", "sub", "mul", "div", "neg")
                 for node in actual.region.nodes),
             "loop body must be the shared one-array elementwise DAG")
    count = strategy["trip_count"]
    _require(type(count) is int and 0 < count <= 2**63 - 1,
             "loop count must be an exact positive int64 specialization")
    memory = strategy["selected_memory_plan"]
    shape = memory["input_shapes"][loop.state_parameter]
    _require(set(memory["input_shapes"]) == {loop.state_parameter} and len(shape) == 2 and
             all(type(n) is int and 0 <= n <= sys.maxsize // 8 for n in shape),
             "loop state shape exceeds host bounds")
    size = shape[0] * shape[1] * 8
    budget = actual.cost.get("memory_budget_bytes")
    _require(memory["output_shape"] == shape and memory["final_bytes"] == size <= sys.maxsize and
             memory["total_workspace_bytes"] == 0 and memory["peak_allocation_bytes"] == size and
             type(budget) is int and size <= budget <= memory["memory_budget_bytes"] <= sys.maxsize,
             "loop requires a shape-preserving fresh final within the narrower payload budget")
    return strategy, actual


__all__ = [
    'ArrayLoopPlan',
    'validate_loop',
    'document',
    'loop_from_dict',
    'loop_digest',
    'digest',
    'read_loop',
    'write_loop',
    'admit_loop',
    'check_loop_sources',
    'SCHEMA',
    'PROFILE',
]
