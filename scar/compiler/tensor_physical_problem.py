"""Project one already qualified Tensor body into the shared physical planner."""

from scar.compiler.plan_aten_boundary import _target
from scar.compiler.plan_aten_native_graph import _host_tensor_budget
from scar.ir.implementation_plan import (
    CostEstimate, ImplementationOption, PlanningDataEdge, PlanningOperation, PlanningProblem, problem_document,
)


def tensor_physical_problem(hardware, cpus, *, budget, operations, inputs, outputs, sizes, reserved,
                            source, region, dispatch=None, costs=None):
    """Retain original body order and explicitly report unmeasured service costs."""
    target, lanes = _target(hardware, tuple(cpus), allow_cpu_subset=True)
    memory = _host_tensor_budget(hardware, budget, (source,))
    ids = {row['id'] for row in operations}
    operands = {row['id']: tuple(sorted({arg['value'] for arg in row['arguments'] if 'value' in arg}))
                for row in operations}
    consumers = set(outputs) | {value for uses in operands.values() for value in uses}
    tasks, data = [], []
    previous = ()
    for row in operations:
        name = row['id']
        references = (dict(source, operation=f'program_operation:{region}/operation/{name}'),)
        lowering, used, extra_workspace = dispatch(row) if dispatch else ('NATIVE_ATEN_CALL', cpus[:1], 0)
        for value in operands[name]:
            producer = value if value in ids else None
            if producer is None and value not in inputs:
                raise ValueError(f'physical input {value} has no declared source')
            data.append(PlanningDataEdge(f'{value}->{name}', value, producer, name, sizes[value],
                memory.id if producer is None else None, source_refs=references))
        unused = sizes[name] if name not in consumers else 0
        cost = costs[name] if costs is not None else CostEstimate(None, None, None, 'UNKNOWN',
            'UNMEASURED_REGION_OPERATOR', references)
        modes = (ImplementationOption('NATIVE', lowering, references,
            {resource.id: 1 for cpu in used for resource in lanes[cpu]},
            {memory.id: extra_workspace + unused} if extra_workspace or unused else {},
            dict.fromkeys(operands[name], memory.id), {name: memory.id} if name in consumers else {},
            cost,
            dict(cpu_ids=list(used), completion='SYNCHRONOUS_RETURN')),)
        tasks.append(PlanningOperation(name, references, previous, modes, 'NATIVE'))
        previous = (name,)
    endpoint = '$return'
    for value in sorted(set(outputs)):
        data.append(PlanningDataEdge(f'{value}->{endpoint}', value, value, endpoint, sizes[value], None,
                                     source_refs=(source,)))
    marker = ImplementationOption('RETURN', 'REGION_VALUE_HANDOFF', (source,), {}, {},
        dict.fromkeys(outputs, memory.id), {}, CostEstimate(0, 0, 0, 'DECLARED_MODEL', 'LOGICAL_ENDPOINT'), {})
    tasks.append(PlanningOperation(endpoint, (source,), previous, (marker,), 'RETURN'))
    return problem_document(PlanningProblem(target, (source,),
        (*[resource for entries in lanes.values() for resource in entries], memory), tuple(tasks), tuple(data),
        assumptions=('SYNCHRONOUS_CPU_BODY', 'SOURCE_ORDER', 'PROVIDER_WORKSPACE_NOT_MEASURED'),
        reserved_memory={memory.id: reserved}))
