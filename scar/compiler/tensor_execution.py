"""Calibrate and replay native Tensor schedules using measured activity costs."""

from hashlib import sha256
import json
from pathlib import Path
from statistics import median_low
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter_ns

from scar.codegen.native_provider_bundle import _content
from scar.ir.implementation_plan import CostEstimate, problem_from_dict
from scar.optimization.joint_planner import verify_execution_plan
from scar.optimization.paired_entry_selection import compare_entry_pairs, replay_entry_pairs


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
