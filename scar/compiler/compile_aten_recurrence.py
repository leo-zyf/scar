"""Compile a fixed-grid Tensor recurrence and its native RHS into one C++20 program."""

import argparse
from hashlib import sha256
from importlib import import_module
import json
from math import frexp, isfinite, prod
from pathlib import Path
import shlex
from tempfile import TemporaryDirectory

from scar.analysis.aten_trial_geometry import qualify_trial_geometry
from scar.codegen.aten_native_graph import _installed_linkage, _read_declared
from scar.codegen.native_project import build_native_project, write_native_project
from scar.codegen.native_provider_bundle import _content
from scar.compiler.plan_aten_ready import read_ready_selection
from scar.compiler.tensor_physical_problem import tensor_physical_problem
from scar.compiler.tensor_program_graph import recurrence_graph
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.program import codec


SCHEMA = 'scar.aten-recurrence-delivery'
_METHODS = dict(euler='Euler', midpoint='Midpoint', heun2='Heun2', heun3='Heun3', rk4='RK4')
_GRID_SOURCE = {
    'odeint': '1fe96dc69b91d7bd53d100023b38cdb876c8b31ebbc46c12f8172c8b03b855cc',
    'solvers': '724bc51c021f84366759018ba5c61e263b58492e9e2083e6b60fda71e84e68bd',
    'fixed_grid': 'bb2915bdbbfb464cdb79593460761b8d9da73f653eb6894e178caf0ee03266b3',
    'rk_common': '7be03cff5f9ee078edf7be1200c28aa21e56640f551532991959c5aca27f7bb2',
    'misc': '49a4707d4c2bdd2ad162fa449fcad3bedec5719159cfddf1caec516bc28db5be',
}
_RHS_HEADERS = ('graph.hpp', 'operators.hpp', 'aten_native_team_v1.h', 'aten_ready_dag_v1.h',
                'aten_ready_replica_v1.h')
_EXE = 'aten-recurrence'


def _grid_step(method):
    # These source receipts qualify the provider's fixed-grid construction, its
    # null callbacks and exact endpoint selection. The step math is extracted
    # from the original callable with separate symbolic start and end times.
    modules, sources = {}, {}
    for name, expected in _GRID_SOURCE.items():
        module = import_module(f'torchdiffeq._impl.{name}')
        path = Path(module.__file__).resolve(strict=True)
        source = path.read_bytes()
        if sha256(source).hexdigest() != expected:
            raise ValueError(f'unqualified torchdiffeq fixed-grid source {path}')
        modules[name] = module
        sources[name] = dict(path=str(path), sha256=expected, size_bytes=len(source))
    solver = vars(modules['fixed_grid'])[_METHODS[method]]
    base = modules['solvers'].FixedGridODESolver
    if (modules['odeint'].SOLVERS.get(method) is not solver or solver.__bases__ != (base,)
            or solver.integrate is not base.integrate or solver.__init__ is not base.__init__):
        raise ValueError('the installed fixed-grid dispatch differs from its qualified source')

    class Options:
        perturb = False

    def step(state, dt, rhs, start, end):
        def autonomy(time, value, *, perturb=modules['misc'].Perturb.NONE):
            if perturb is not modules['misc'].Perturb.NONE:
                raise ValueError('fixed-grid lowering requires perturbation to be disabled')
            return rhs(value)

        difference, _ = solver._step_func(Options(), autonomy, start, dt, end, state)
        return state + difference

    return step, dict(api='torchdiffeq.odeint', method=method, sources=sources,
                      boundary='AUTONOMOUS_FIXED_GRID_FINAL_STATE', callbacks='NONE',
                      time_grid='ZERO_TO_EXACT_BINARY_STEP_MULTIPLE')


def _rhs_inputs(selection, selection_sha256, geometry_contract, geometry_contract_sha256, torch):
    path, report, _ = _read_declared(selection, selection_sha256)
    if report.get('delivery_scope') != 'STANDALONE_CPP20_ONLY' or report.get('selected') != 'READY_DAG':
        raise ValueError('a verified standalone native ready RHS with storage measurements is required')
    selected, artifact, recipe, replay = read_ready_selection(path, report)
    geo_path, declaration, _ = _read_declared(geometry_contract, geometry_contract_sha256)
    geometry = qualify_trial_geometry(recipe, declaration, read_artifact(replay['calibration'])['calibration'], torch)
    if len(recipe['outputs']) != 1:
        raise ValueError('the recurrence requires a single Tensor returned by its RHS')
    provider = artifact['provider']
    if provider['revision'] != geometry['provider_revision']:
        raise ValueError('RHS geometry and native graph expect different operator providers')
    return selected, artifact, recipe, geometry, dict(replay,
        geometry_contract=artifact_reference(geo_path, declaration['schema']))


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def compile_aten_recurrence(selection, *, selection_sha256, geometry_contract, geometry_contract_sha256,
                            method, step_size, out, tensor_budget_bytes, state_input_index=0, serial_rhs=False):
    import torch

    from scar.codegen.aten_recurrence_source import render_recurrence
    from scar.compiler.aten_recurrence_capture import capture_aten_step
    from scar.optimization.aten_recurrence import plan_recurrence

    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    if method not in _METHODS:
        raise ValueError('an admitted fixed-grid solver method is required')
    if (type(step_size) is not float or frexp(step_size)[0] != 0.5
            or step_size < float.fromhex('0x1p-1022') or not isfinite(step_size * (2**31 - 1))):
        raise ValueError('fixed-grid recurrence needs an exact positive binary64 power-of-two step')
    if type(tensor_budget_bytes) is not int or not 0 < tensor_budget_bytes < 2**63:
        raise ValueError('a positive bounded Tensor storage budget is required')

    selected, artifact, recipe, geometry, references = _rhs_inputs(
        selection, selection_sha256, geometry_contract, geometry_contract_sha256, torch)
    if type(state_input_index) is not int or not 0 <= state_input_index < len(recipe['inputs']):
        raise ValueError('state input index is outside the native RHS')
    shape = recipe['inputs'][state_input_index]['shape']
    result_id, = recipe['outputs']
    if next(row['shape'] for row in geometry['operations'] if row['id'] == result_id) != shape:
        raise ValueError('RHS result geometry differs from the carried state')
    step, method_provider = _grid_step(method)
    step_graph = capture_aten_step(step, shape, step_size, explicit_time=True)

    measured_storage = selected['model']['tensor_storage']
    alternatives, chosen = {}, None
    modes = ('SERIAL',) if serial_rhs else ('READY_DAG', 'SERIAL')
    for mode in modes:
        peak = measured_storage['serial_peak_bytes' if mode == 'SERIAL' else 'peak_tensor_storage_bytes']
        storage = dict(measured_storage, peak_tensor_storage_bytes=peak)
        plan = plan_recurrence(step_graph, recipe, storage, state_input_index=state_input_index, step_size=step_size)
        needed = plan['peak_tensor_storage_bytes']
        alternatives[mode] = dict(peak_tensor_storage_bytes=needed, within_budget=needed <= tensor_budget_bytes)
        if chosen is None and needed <= tensor_budget_bytes:
            chosen = mode, plan
    if chosen is None:
        raise ValueError('no native recurrence fits its Tensor storage budget')
    mode, plan = chosen
    plan['storage']['budget_bytes'] = tensor_budget_bytes
    graph = recurrence_graph(step_graph, recipe, geometry, plan['control'], plan['rhs_bindings'], step_size=step_size)

    linkage_record = artifact['linkage']
    _, linkage, _ = _read_declared(linkage_record['path'], linkage_record['sha256'])
    _, libraries, cxx, compile_flags, link_flags = _installed_linkage(linkage, torch)
    cpus = selected['cpu_ids'] if mode == 'READY_DAG' else selected['cpu_ids'][:1]
    lanes = selected['model']['lanes'] if mode == 'READY_DAG' else None
    order = selected['model'].get('operation_order') if mode == 'READY_DAG' else None
    bundle = Path(selected['bundle']['uri']).parent
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-aten-recurrence-', dir=out.parent) as temporary:
        stage = Path(temporary)
        plan_dir = stage / 'plan'
        plan_dir.mkdir()
        headers = {'graph.hpp': 'include/scar_generated', 'operators.hpp': 'src/providers'}
        for name in _RHS_HEADERS:
            member = artifact['files'].get(name)
            if member is None:
                continue
            destination = stage / headers.get(name, 'src/runtime') / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((bundle / name).read_bytes())
            if _content(destination) != member:
                raise ValueError(f'RHS header differs from the checked native bundle: {name}')
        for name, source in render_recurrence(plan, rhs_mode=mode, cpus=cpus, lanes=lanes, operation_order=order).items():
            destination = stage / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(source)
        graph_path = plan_dir / 'program.json'
        graph_path.write_text(codec.dumps(graph) + '\n')
        plan['program_graph'] = dict(path='plan/program.json', **_content(graph_path))
        plan['target'] = artifact['cpu_team']['target']
        plan['rhs_schedule'] = references['execution_plan'] if mode == 'READY_DAG' else None
        plan['placement'] = dict(mode=mode, cpus=cpus, lanes=lanes, operation_order=order)
        _write(plan_dir / 'execution.json', plan)
        physical_sizes = {row['id']: (8 if row['dtype'] == 'FLOAT64' else 4) * prod(row['shape'])
                          for row in (*plan['inputs'], *plan['operations'])}
        storage = plan['storage']

        def dispatch(row):
            if row['op'] == 'RHS':
                return 'NATIVE_SHARED_RHS', cpus, storage['rhs_non_input_peak_bytes'] - physical_sizes[row['id']]
            return 'NATIVE_ATEN_CALL', cpus[:1], 0

        physical = tensor_physical_problem(Path(plan['target']['uri']).parent, cpus, budget=tensor_budget_bytes,
            operations=plan['operations'], inputs={row['id'] for row in plan['inputs']}, outputs=[plan['output']],
            sizes=physical_sizes, reserved=storage['raw_file_backing_bytes'] + storage['previous_step_state_bytes'] + 8,
            source=plan['program_graph'], region='step', dispatch=dispatch)
        _write(plan_dir / 'problem.json', physical)
        _write(plan_dir / 'inputs.json', dict(solver=method_provider, rhs=references, source_geometry=geometry,
                                             lowered_step=step_graph))
        _write(plan_dir / 'dependencies.json', artifact['provider'])
        prefix = shlex.join(['env', '-i', 'PATH=/usr/bin:/bin', 'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
                             'taskset', '-c', ','.join(map(str, cpus))])
        launcher = stage / 'launch.sh'
        launcher.write_text(f'#!/bin/sh\nset -eu\nexec {prefix} "$(dirname -- "$0")/bin/{_EXE}" "$@"\n')
        launcher.chmod(0o755)
        write_native_project(stage, _EXE, include_dirs=('src/providers', 'src/runtime'),
            compile_options=(*compile_flags[1:], '-O2', '-pthread'),
            link_options=('-pthread', link_flags[0]), libraries=tuple(map(str, libraries)),
            runtime_dirs=(str(libraries[0].parent),), install_launcher=True)
        _, dependencies = build_native_project(stage, _EXE, cxx, expected_libraries=libraries)
        report = dict(schema=SCHEMA, schema_version=1, status='STANDALONE_CPP20_FIXED_GRID_SOLVE',
            cxx_standard=20, abi_version=1, method_provider=method_provider, provider=artifact['provider'],
            source_references=references, rhs_mode=mode, cpus=cpus, lanes=lanes, operation_order=order,
            tensor_budget_bytes=tensor_budget_bytes, alternatives=alternatives, dependencies=dependencies,
            executable='bin/' + _EXE, full_entry_performance_measured=False,
            files={path.relative_to(stage).as_posix(): _content(path)
                   for path in sorted(stage.rglob('*')) if path.is_file()})
        _write(stage / 'artifact.json', report)
        stage.rename(out)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar compile-aten-recurrence', description=__doc__)
    parser.add_argument('--selection', required=True, help='validated standalone native RHS selection.json')
    parser.add_argument('--selection-sha256', required=True)
    parser.add_argument('--geometry-contract', required=True, help='value-independent RHS operator declarations')
    parser.add_argument('--geometry-contract-sha256', required=True)
    parser.add_argument('--method', choices=_METHODS, required=True)
    parser.add_argument('--step-size', type=float, required=True)
    parser.add_argument('--state-input-index', default=0, type=int)
    parser.add_argument('--tensor-budget-bytes', required=True, type=int)
    parser.add_argument('--serial-rhs', action='store_true')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    report = compile_aten_recurrence(**vars(args))
    print(json.dumps(dict(status=report['status'], rhs_mode=report['rhs_mode'],
                          executable=str(Path(args.out).resolve() / report['executable']))))
    return 0
