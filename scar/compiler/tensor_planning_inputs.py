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
    problem_from_dict,
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
from hashlib import sha256
import json
import subprocess
from pathlib import Path
from statistics import median_low
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from scar.codegen.native_provider_bundle import _content
from scar.optimization.joint_planner import verify_execution_plan
from scar.optimization.paired_entry_selection import compare_entry_pairs, replay_entry_pairs


# Tensor planning inputs.
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


# Calibrate and replay native Tensor schedules using measured activity costs.
_PHASES = ('setup_ns', 'read_ns', 'loop_ns', 'publish_ns')
_MODES = ('SERIAL', 'READY_DAG')


def run_native_sample(project, arguments, scratch, *, outputs=1):
    with TemporaryDirectory(prefix='run-', dir=scratch) as directory:
        paths = [Path(directory) / str(index) for index in range(outputs)]
        started = perf_counter_ns()
        run = subprocess.run([str(project / 'launch.sh'), *arguments, *map(str, paths)],
                             env={}, capture_output=True)
        elapsed = perf_counter_ns() - started
        if run.returncode:
            raise ValueError('native calibration failed: ' + run.stderr.decode(errors='replace').strip())
        result = [_content(path) for path in paths]
        return elapsed, run.stdout, run.stderr, result


def calibrate_tensor_body(project, arguments, scratch, *, repeats, outputs=1):
    samples, expected = [], None
    for index in range(4):
        _, _, stderr, result = run_native_sample(project, ['--calibrate', *arguments], scratch, outputs=outputs)
        if expected is not None and result != expected:
            raise ValueError('native calibration changed the output across identical inputs')
        expected = result
        if index:
            samples.append(json.loads(stderr))
    return dict(schema='scar.tensor-body-observation', schema_version=1, repeats=repeats,
                samples=samples, outputs=expected, discarded_warmups=1)


def operation_costs(observation, operations, evidence):
    count = observation['repeats']
    if type(count) is not int or count <= 0 or len(observation['samples']) != 3:
        raise ValueError('native calibration requires a positive repeat count and three samples')
    vectors = []
    for row in observation['samples']:
        raw = row['operation_totals_ns']
        if (set(row) != {*_PHASES, 'operation_totals_ns'} or len(raw) != len(operations)
                or any(type(value) is not int or value <= 0 for value in raw)
                or any(type(row[key]) is not int or row[key] < 0 for key in _PHASES)
                or sum(raw) > row['loop_ns']):
            raise ValueError('native operator timers do not fit their measured loop boundary')
        vectors.append([(value + count - 1) // count for value in raw])
    return {op['id']: CostEstimate(median_low(samples), min(samples), max(samples), 'EMPIRICAL_MODEL',
            'NATIVE_SYNCHRONOUS_BODY_SERVICE_PER_STEP', (dict(evidence, operator=index),))
            for index, (op, samples) in enumerate(zip(operations, zip(*vectors, strict=True), strict=True))}


def compare_native_projects(projects, arguments, scratch, expected, *, output_count=1):
    runs, outputs = [], []
    for index in range(5):
        for mode in (_MODES if index % 2 == 0 else _MODES[::-1]):
            elapsed, stdout, stderr, actual = run_native_sample(projects[mode], arguments, scratch, outputs=output_count)
            if actual != expected or stderr:
                raise ValueError('paired native candidates must match the calibrated result and exit cleanly')
            runs.append(dict(pair=index, mode=mode, wall_ns=elapsed, returncode=0,
                             stdout_sha256=sha256(stdout).hexdigest(), stderr_sha256=sha256(stderr).hexdigest()))
            outputs.append(actual)
    decision = compare_entry_pairs(runs, incumbent=_MODES[0], challenger=_MODES[1],
                                   warmup_pairs=1, measurement_pairs=4)
    return dict(scope='COLD_STANDALONE_PROCESS_INCLUDING_FILE_IO', runs=runs, outputs=outputs, decision=decision)


def replay_native_comparison(comparison, expected):
    runs = comparison['runs']
    if len(comparison['outputs']) != len(runs) or any(actual != expected for actual in comparison['outputs']):
        raise ValueError('paired native candidates do not match the calibrated output receipts')
    decision = replay_entry_pairs(runs, comparison['decision'], incumbent=_MODES[0], challenger=_MODES[1],
                                  warmup_pairs=1, measurement_pairs=4)
    if decision != comparison['decision']:
        raise ValueError('paired native selection differs from its underlying timing rows')
    return 'READY_DAG' if decision['verdict'] == 'ADVANCE_TO_RELEASE_REVIEW' else 'SERIAL'


def scheduled_tensor_body(problem_document, schedule, body):
    """Recover the exact operation order and release positions from the certificate."""
    problem = problem_from_dict(problem_document)
    verify_execution_plan(problem, schedule)
    original = {row['id']: row for row in body['operations']}
    rank = {row.id: index for index, row in enumerate(problem.operations)}
    selected = sorted(schedule['selected'], key=lambda row: (row['start_ns'], rank[row['operation']]))
    order = [item['operation'] for item in selected]
    if not order or order[-1] != '$return' or set(order[:-1]) != original.keys():
        raise ValueError('Tensor body does not cover exactly the selected physical activities')
    if any(left['end_ns'] > right['start_ns'] for left, right in zip(selected, selected[1:])):
        raise ValueError('the synchronous Tensor body cannot emit overlapping top-level activities')
    outputs = set(body['outputs']) if 'outputs' in body else {body['output']}
    demands = {(arg['value'], row['id']) for row in body['operations']
               for arg in row['arguments'] if 'value' in arg}
    demands.update((value, '$return') for value in outputs)
    if demands != {(edge.value, edge.consumer) for edge in problem.data_edges}:
        raise ValueError('execution data edges differ from the original Tensor operands or outputs')
    position = {name: index for index, name in enumerate(order[:-1])}
    uses = {name: index for name, index in position.items()}
    for edge in problem.data_edges:
        if edge.producer in position and edge.consumer != '$return':
            uses[edge.producer] = max(uses[edge.producer], position[edge.consumer])
    releases = [[] for _ in position]
    for name, release in uses.items():
        if name not in outputs:
            releases[release].append(position[name])
    return dict(body, operations=[original[name] for name in order[:-1]], release_after=releases)


def replay_operation_costs(problem, execution, observation, evidence):
    costs = operation_costs(observation, execution['operations'], evidence)
    for task in problem.operations:
        if task.id != '$return' and any(option.cost != costs[task.id] for option in task.options):
            raise ValueError('selected operator cost differs from its native calibration')


def replay_recurrence_placement(problem, placement, rhs_recipe, rhs_problem=None, rhs_schedule=None):
    cpus = placement['cpus']
    for operation in problem.operations:
        for option in operation.options:
            if option.lowering == 'NATIVE_SHARED_RHS' and option.parameters['cpu_ids'] != cpus:
                raise ValueError('recurrence RHS launch CPUs differ from its physical activity')
            if option.lowering == 'NATIVE_ATEN_CALL' and option.parameters['cpu_ids'] != cpus[:1]:
                raise ValueError('recurrence step operator runs on a different CPU than planned')
    if placement['mode'] == 'SERIAL':
        if len(cpus) != 1 or placement['lanes'] is not None or placement['operation_order'] is not None:
            raise ValueError('serial recurrence requires only the caller CPU')
        return
    if placement['mode'] != 'READY_DAG' or rhs_problem is None or rhs_schedule is None:
        raise ValueError('parallel RHS requires its independently verified physical plan')
    rhs = problem_from_dict(rhs_problem)
    verify_execution_plan(rhs, rhs_schedule)
    names = [item['id'] for item in rhs_recipe['operations']]
    physical = [item for item in rhs.operations if item.id != '$ready-return']
    order = placement['operation_order']
    order = [names[index] for index in order] if order is not None else names
    if [item.id for item in physical] != order:
        raise ValueError('emitted RHS order differs from the archived parallel plan')
    positions = {name: i for i, name in enumerate(names)}
    chosen = {row['operation']: row['option'] for row in rhs_schedule['selected']}
    for operation in physical:
        mode, = [option for option in operation.options if option.id == chosen[operation.id]]
        lane = placement['lanes'][positions[operation.id]]
        if mode.parameters != {'cpu_id': cpus[lane], 'lane': lane}:
            raise ValueError('emitted RHS lane differs from the archived parallel plan')
