"""Validate private arena retention on the unchanged entry in fresh processes."""

import argparse
from contextlib import contextmanager, nullcontext
from functools import wraps
from hashlib import file_digest
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter_ns
from types import SimpleNamespace

from scar.ir.execution_recipe import artifact_reference, read_artifact

DESIGN = 'scar.session-capacity-entry-design'
SAMPLE = 'scar.session-capacity-entry-sample'
VALIDATION = 'scar.session-capacity-entry-validation'
CASES = ('original', 'isolated', 'capacity', 'fallback')
EXPORTS = dict(isolated='scar.session-outer-export', capacity='scar.session-capacity-export')
MODULE = 'scar.compiler.validate_session_capacity_entry'


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _write(path, document):
    with path.open('x') as stream:
        json.dump(document, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return artifact_reference(path, document['schema'])


def _exports(references):
    from scar.runtime.session_scope_bundle import inspect_scope_bundle

    exports, scopes = {}, {}
    for mode, reference in references.items():
        export = read_artifact(reference)
        scope = inspect_scope_bundle(export['bundle'],
            expected_admission_sha256=export['expected_admission_sha256'])
        _need(export['default_backend'] == 'NO_OP' and export['deployment_authorized'] is False
            and scope['admission']['preparation'] == export['preparation'],
            'one explicit source-qualified workspace scope required')
        exports[mode], scopes[mode] = export, scope
    isolated, capacity = (scopes[mode]['spec'] for mode in EXPORTS)
    _need(isolated['unqualified_kernel_policy'] == 'FRESH_ISOLATED_NATIVE'
        and capacity['unqualified_kernel_policy'] == 'RETAIN_PRIVATE_CAPACITY_FULL_SNAPSHOT'
        and isolated['boundaries'] == capacity['boundaries']
        and scopes['isolated']['parent']['contract'] == scopes['capacity']['parent']['contract']
        and capacity['capacity_policy']['content_reuse'] is False
        and capacity['capacity_policy']['external_roots_retained'] is False,
        'comparison requires the same source, ABI and C++ with only the capacity policy changed')
    contract = scopes['capacity']['parent']['contract']
    source = contract['step_contract']['source']
    (kernel,) = [row for row in capacity['boundaries'] if row['role'] == 'KERNEL']
    _need(kernel['source'] == source, 'source kernel differs from compact session')
    return exports, contract, kernel


def _design(entry, exports, argv, output_token):
    _need(type(argv) is list and all(type(arg) is str for arg in argv)
        and type(output_token) is str and argv.count(output_token) == 1,
        'explicit entry arguments and one token for the fresh output directory required')
    return dict(schema=DESIGN, schema_version=1, entry=entry, exports=exports, argv=argv,
        output_token=output_token, caller_cpu_ids=sorted(os.sched_getaffinity(0)),
        python=sys.executable, implementation=artifact_reference(__file__, 'scar.python-source'),
        default_backend='NO_OP', deployment_authorized=False)


@contextmanager
def _observed(scope, kernel, contract, *, fallback, trajectory, retained):
    """Observe an installed function without using sys.profile or replacing live guards."""
    from scar.compiler.controlled_stateful_delivery import _caller_mask
    from scar.compiler.stateful_holdout import _state, _summary
    from scar.runtime.stateful_guard import _function

    step = contract['step_contract']
    region = SimpleNamespace(inputs=[SimpleNamespace(**row) for row in step['inputs']])
    path = Path(step['source']['path']).resolve()
    sys.path.insert(0, str(path.parent))
    try:
        module = importlib.import_module(path.stem)
    finally:
        del sys.path[0]
    _need(Path(vars(module)['__file__']).resolve() == path, 'ordinary import selected another kernel source')
    owner = module
    for name in kernel['owner_path']:
        owner = vars(owner)[name]
    attribute = kernel['attribute']
    descriptor = vars(owner)[attribute]
    original = descriptor.__func__ if type(descriptor) is staticmethod else descriptor
    _function(step['source'], original)
    original_mask = os.sched_getaffinity(0)
    reduced_mask = original_mask - set(step['cpu']['ids'])
    _need(not fallback or bool(reduced_mask), 'fallback experiment needs another available CPU')

    def snapshot(bound):
        return _summary(_state(region, bound, contents=True))

    def wrap(selected, signature):
        @wraps(selected)
        def observe(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            retained['arguments'] = bound.arguments
            before = snapshot(bound.arguments)
            mask = _caller_mask(reduced_mask) if fallback and len(trajectory) == 1 else nullcontext()
            with mask:
                result = selected(*args, **kwargs)
            trajectory.append(dict(before=before, after=snapshot(bound.arguments)))
            return result
        if hasattr(selected, 'scar_stats'):
            observe.scar_stats = selected.scar_stats
        return observe

    if scope is None:
        observer = wrap(original, inspect.signature(original))
        setattr(owner, attribute, staticmethod(observer) if type(descriptor) is staticmethod else observer)
    else:
        installer = scope._wrap

        def observe_binding(row, function, receiver_owner, signature):
            selected = installer(row, function, receiver_owner, signature)
            return wrap(selected, signature) if row['id'] == kernel['id'] else selected

        scope._wrap = observe_binding
    try:
        yield
        _need(bool(trajectory), 'entry did not execute its source-qualified kernel')
        retained['final'] = snapshot(retained.pop('arguments'))
        _need(os.sched_getaffinity(0) == original_mask, 'entry observation changed caller CPU affinity')
        if scope is not None:
            _need(vars(owner)[attribute] is descriptor, 'running the entry failed to restore the kernel descriptor')
    finally:
        if scope is None:
            setattr(owner, attribute, descriptor)
        else:
            scope._wrap = installer


def _worker(design_reference, case, out, *, observe):
    from scar.runtime.entry_runner import run_original_script
    from scar.runtime.session_scope_bundle import load_scope

    design = read_artifact(design_reference)
    _need(design == _design(design['entry'], design['exports'], design['argv'], design['output_token']),
        'entry design differs from running interpreter, source or caller CPU mask')
    exports, contract, kernel = _exports(design['exports'])
    entry = Path(design['entry']['uri'])
    read_artifact(design['entry'], json_document=False)
    if case == 'original':
        scope = None
    else:
        export = exports['capacity' if case == 'fallback' else case]
        scope = load_scope(export['bundle'], expected_admission_sha256=export['expected_admission_sha256'],
            trial=True, exclusive_inputs=True)
    trajectory, retained = [], {}
    observation = (_observed(scope, kernel, contract, fallback=case == 'fallback',
        trajectory=trajectory, retained=retained) if observe else nullcontext())
    install = nullcontext() if scope is None else scope.install()
    argv = [str(out/'program-output') if arg == design['output_token'] else arg for arg in design['argv']]
    with observation, install:
        run_original_script(entry, argv)
    read_artifact(design['entry'], json_document=False)
    for row in (kernel, *[item for item in contract['step_contract']['callee_guards']]):
        source = row['source']
        _need(artifact_reference(source['path'], 'scar.python-source')['sha256'] == source['sha256'],
            'entry changed its source')
    guard = None if scope is None else next(
        (item['stats'] for item in scope.bindings if item['id'] == kernel['id']), None)
    _write(out/'entry.json', dict(schema=SAMPLE, schema_version=1, design=design_reference, case=case,
        observation_enabled=observe, trajectory=trajectory if observe else None, final=retained.get('final'),
        session_statistics=None if scope is None else dict(scope.session.stats),
        scope=None if scope is None else scope.report, kernel_guard=guard, deployment_authorized=False))


def _digest(path):
    with path.open('rb') as stream:
        return dict(bytes=os.fstat(stream.fileno()).st_size, sha256=file_digest(stream, 'sha256').hexdigest())


def run_sample(design, case, out, *, observe=False):
    """Run one fresh process and collect output bytes outside the wall clock."""
    _need(case in CASES and (case != 'fallback' or observe), 'unsupported session experiment case')
    design = artifact_reference(Path(design).resolve(), DESIGN)
    out = Path(out).resolve()
    out.mkdir(parents=True)
    command = [sys.executable, '-B', '-m', MODULE, '--design', design['uri'],
        '--expected-sha256', design['sha256'], '--worker', case, '--out', str(out)]
    if observe:
        command.append('--observe')
    stdout, stderr = out/'stdout', out/'stderr'
    with stdout.open('xb') as a, stderr.open('xb') as b:
        begin = perf_counter_ns()
        subprocess.run(command, stdout=a, stderr=b, check=True)
        elapsed = perf_counter_ns() - begin
    result = read_artifact(artifact_reference(out/'entry.json', SAMPLE))
    output = out/'program-output'
    result.update(stdout=_digest(stdout), stderr=_digest(stderr),
        files={str(path.relative_to(output)):_digest(path) for path in sorted(output.rglob('*')) if path.is_file()})
    if not observe:
        result['full_process_wall_ns'] = elapsed
    return result


def validate(entry, isolated, capacity, out, *, argv, output_token):
    exports = {mode:artifact_reference(Path(path).resolve(), EXPORTS[mode])
               for mode, path in (('isolated', isolated), ('capacity', capacity))}
    _exports(exports)
    out = Path(out).resolve()
    out.mkdir(parents=True)
    design = _write(out/'design.json', _design(artifact_reference(Path(entry).resolve(), 'scar.python-source'),
        exports, argv, output_token))
    cases = {mode:run_sample(design['uri'], mode, out/mode, observe=True) for mode in CASES}
    baseline = cases['original']
    scoped = {mode:cases[mode]['scope'] for mode in CASES[1:]}
    stats = {mode:cases[mode]['session_statistics'] for mode in CASES[1:]}
    steps = len(baseline['trajectory'])
    checks = {mode:all(result[key] == baseline[key] for key in ('trajectory', 'final', 'stdout', 'stderr', 'files'))
              for mode, result in cases.items() if mode != 'original'}
    checks.update(receiver_source_bound=all(row['installed_bindings'] > 0 and not row['frontier']
            for row in scoped.values()),
        all_current_snapshots=all(row['resident_calls'] == 0 for row in stats.values()),
        isolated_releases_each_step=stats['isolated']['creates'] == stats['isolated']['releases']
            == stats['isolated']['native_calls'] == steps,
        capacity_hoisted_outside_steps=steps >= 3 and stats['capacity']['native_calls'] == steps
            and stats['capacity']['creates'] == stats['capacity']['releases'] == 1,
        single_fallback_ends_epoch=stats['fallback']['native_calls'] == steps - 1
            and scoped['fallback']['statistics']['capacity_original_fallback_calls'] == 1
            and stats['fallback']['creates'] == stats['fallback']['releases'] == 2)
    return _write(out/'validation.json', dict(schema=VALIDATION, schema_version=2, design=design, cases=cases,
        checks=checks, validated=all(checks.values()), performance_measured=False, default_backend='NO_OP',
        deployment_authorized=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry')
    parser.add_argument('--isolated-export')
    parser.add_argument('--capacity-export')
    parser.add_argument('--argv-json')
    parser.add_argument('--output-token')
    parser.add_argument('--out', required=True)
    parser.add_argument('--design')
    parser.add_argument('--worker', choices=CASES)
    parser.add_argument('--expected-sha256')
    parser.add_argument('--observe', action='store_true')
    parser.add_argument('--clean-sample', choices=CASES[:3])
    args = parser.parse_args(argv)
    if args.worker:
        reference = artifact_reference(Path(args.design).resolve(), DESIGN)
        _need(reference['sha256'] == args.expected_sha256, 'frozen design byte identity changed')
        _worker(reference, args.worker, Path(args.out), observe=args.observe)
        return 0
    if args.clean_sample:
        result = run_sample(args.design, args.clean_sample, args.out)
    else:
        result = validate(args.entry, args.isolated_export, args.capacity_export, args.out,
            argv=json.loads(args.argv_json), output_token=args.output_token)
    print(json.dumps(result, sort_keys=True))
    return int(not args.clean_sample and not read_artifact(result)['validated'])


if __name__ == '__main__':
    raise SystemExit(main())
