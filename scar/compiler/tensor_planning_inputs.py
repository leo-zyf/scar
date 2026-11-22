"""Tensor planning inputs."""

from scar.compiler.plan_aten_boundary import _target
from scar.compiler.plan_aten_native_graph import _host_tensor_budget
from scar.ir.implementation_plan import (
    CostEstimate,
    ImplementationOption,
    PlanningDataEdge,
    PlanningOperation,
    PlanningProblem,
    problem_document,
)
from scar.ir.program.ids import BlockID, OperationID, RegionID, UseID, ValueID
from scar.ir.program.model import (
    Behavior,
    Block,
    BlockArgument,
    DataValue,
    ExitKind,
    ExitPort,
    LiteralOrigin,
    Operation,
    OperationResult,
    ProgramGraph,
    Region,
    Successor,
    Terminator,
    TerminatorKind,
    Use,
    value_type,
)


# Project one already qualified Tensor body into the shared physical planner.
def tensor_physical_problem(hardware, cpus, *, budget, operations, inputs, outputs, sizes, reserved,
                            source, region, dispatch=None, costs=None, order_assumption='SOURCE_ORDER'):
    """Bind an admitted body order and explicitly report unmeasured service costs."""
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
        assumptions=('SYNCHRONOUS_CPU_BODY', order_assumption, 'PROVIDER_WORKSPACE_NOT_MEASURED'),
        reserved_memory={memory.id: reserved}))


# Project closed tensor provider recipes into the canonical program graph.
def tensor_type(shape, dtype='FLOAT32'):
    return f'tensor.{dtype}[{",".join(map(str, shape))}]'


class _Graph:
    def __init__(self):
        self.operations = []
        self.values = {}

    def arguments(self, region, block, types):
        inputs = tuple(ValueID(f'{block.value}/argument/{i}') for i in range(len(types)))
        for index, (value, kind) in enumerate(zip(inputs, types, strict=True)):
            self.values[value] = DataValue(value, region, BlockArgument(block, index), kind)
        return inputs

    def operation(self, region, name, label, operands, *, dtype=None, callee=None, control=None):
        identity = OperationID(f'{region.value}/{name}')
        uses = []
        for index, operand in enumerate(operands):
            use = UseID(f'{identity.value}/operand/{index}')
            if not isinstance(operand, ValueID):
                value = ValueID(f'{identity.value}/literal/{index}')
                self.values[value] = DataValue(value, region, LiteralOrigin(operand), f'builtins.{type(operand).__name__}')
                operand = value
            uses.append(Use(use, operand))
        arguments = tuple(item.id for item in uses)
        results = (ValueID(f'{identity.value}/result'),) if dtype else ()
        for result in results:
            self.values[result] = DataValue(result, region, OperationResult(identity, 0), dtype)
        behavior = Behavior.CONTROL if control or callee else Behavior.COMPUTE
        self.operations.append(Operation(identity, label, (behavior,), tuple(uses), results,
            terminator=control(arguments) if control else None, callee=callee,
            call_arguments=arguments if callee else ()))
        return identity, results[0] if results else None

    def body(self, region, label, recipe, types, *, parent=None, calls=None):
        entry = BlockID(f'{region.value}/entry')
        names = [row['id'] for row in recipe['inputs']]
        inputs = self.arguments(region, entry, [types[name] for name in names])
        values = dict(zip(names, inputs, strict=True))
        operations = []
        for row in recipe['operations']:
            name = row['id']
            callee = None
            operands = [values[arg['value']] if 'value' in arg else arg['scalar'] for arg in row['arguments']]
            if calls and name in calls[1]:
                callee, bindings = calls
                operands = [values[arg['value']] for arg in bindings[name]]
            op, values[name] = self.operation(region, f'operation/{name}', row['schema'], operands,
                                             dtype=types[name], callee=callee)
            operations.append(op)
        outputs = [values[name] for name in recipe['outputs']]
        end, _ = self.operation(region, 'return', 'return', outputs, control=_return)
        return Region(region, label, entry, (Block(entry, inputs, (*operations, end)),),
                      (self.exit(outputs),), parent)

    def exit(self, values):
        return ExitPort('return', ExitKind.NORMAL, tuple(value_type(self.values[value]) for value in values))

    def graph(self, root, *regions):
        return ProgramGraph(root.id, (root, *regions), tuple(self.operations), tuple(self.values.values()))


def _return(uses):
    return Terminator(TerminatorKind.RETURN, exit='return', values=uses)


def straight_line_graph(recipe, types):
    graph = _Graph()
    return graph.graph(graph.body(RegionID('entry'), 'functional tensor region', recipe, types))


def recurrence_graph(step, rhs, geometry, control, rhs_bindings, *, step_size):
    """Use actual nested call and loop edges for one fixed-count recurrence."""
    graph = _Graph()
    root, body, leaf = map(RegionID, ('entry', 'step', 'rhs'))
    rhs_types = {row['id']: tensor_type(spec['shape']) for rows, specs in (
        (rhs['inputs'], geometry['inputs']), (rhs['operations'], geometry['operations']))
        for row, spec in zip(rows, specs, strict=True)}
    rhs_region = graph.body(leaf, 'native right-hand side', rhs, rhs_types, parent=body)
    step_types = {row['id']: tensor_type(row['shape'], row['dtype'])
                  for row in (*step['inputs'], *step['operations'])}
    fixed = control['invariant_entry_inputs']
    fixed_inputs = [dict(id=f'invariant_{index}') for index in fixed]
    step_types.update({arg['id']: rhs_types[rhs['inputs'][index]['id']]
                       for index, arg in zip(fixed, fixed_inputs, strict=True)})
    calls = {row['operation']: [arg if 'value' in arg else dict(value=f"invariant_{arg['entry_input']}")
                              for arg in row['arguments']] for row in rhs_bindings}
    step_region = graph.body(body, 'shared step', dict(inputs=[*step['inputs'], *fixed_inputs],
        operations=step['operations'], outputs=[step['output']]), step_types, parent=root, calls=(leaf, calls))

    entry, loop, advance, exit_block = map(BlockID, ('entry/start', 'entry/header', 'entry/advance', 'entry/exit'))
    state_type = step_types[control['carried_state']['body_input']]
    state_index = control['carried_state']['initial_entry_input']
    initial = graph.arguments(root, entry, [*(rhs_types[row['id']] for row in rhs['inputs']), 'builtins.int'])
    step_size_type = step_types[control['invariant_step_size_input']]
    make_dt, dt = graph.operation(root, 'step-size', 'materialize fixed scalar step size', [step_size],
                                  dtype=step_size_type)
    enter, _ = graph.operation(root, 'enter', 'initialize carry', (initial[state_index], initial[-1]),
        control=lambda uses: Terminator(TerminatorKind.JUMP, (Successor(loop, uses),)))
    carry, count = graph.arguments(root, loop, (state_type, 'builtins.int'))
    check, condition = graph.operation(root, 'check', 'remaining steps > 0', (count,), dtype='builtins.bool')
    branch, _ = graph.operation(root, 'branch', 'loop condition', (condition, carry),
        control=lambda uses: Terminator(TerminatorKind.BRANCH,
            (Successor(advance, ()), Successor(exit_block, (uses[1],))), condition=uses[0]))
    do_step, next_state = graph.operation(root, 'step', 'execute one step', (carry, dt, *(initial[i] for i in fixed)),
                                          dtype=state_type, callee=body)
    decrement, remaining = graph.operation(root, 'decrement', 'decrement remaining steps', (count, 1), dtype='builtins.int')
    back, _ = graph.operation(root, 'backedge', 'carry next state and remaining steps', (next_state, remaining),
        control=lambda uses: Terminator(TerminatorKind.JUMP, (Successor(loop, uses),)))
    final, = graph.arguments(root, exit_block, (state_type,))
    finish, _ = graph.operation(root, 'return', 'return final state', (final,), control=_return)
    blocks = (Block(entry, initial, (make_dt, enter)), Block(loop, (carry, count), (check, branch)),
              Block(advance, (), (do_step, decrement, back)), Block(exit_block, (final,), (finish,)))
    return graph.graph(Region(root, 'fixed-count recurrence', entry, blocks, (graph.exit((final,)),)),
                       step_region, rhs_region)
