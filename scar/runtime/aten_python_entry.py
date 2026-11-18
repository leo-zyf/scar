"""Run an original application's named entry with an optional deferred ATen trial."""

import argparse
from contextlib import contextmanager, nullcontext
from functools import wraps
import json
import os
from pathlib import Path
import sys
from threading import get_ident
from time import perf_counter_ns
from types import FunctionType

from scar.runtime.aten_python_trial import _original, _read_manifest, install_aten_python_trial
from scar.runtime.validate_aten_python_trial import _original_module


_MODES = ('ORIGINAL', 'GUARDED', 'OBSERVE_ORIGINAL')


@contextmanager
def _observe_original(module, proof, counters, grad_enabled):
    namespace, original = _original(module, proof)
    name = proof['function']
    owner = get_ident()
    process = os.getpid()
    demand = dict(calls=0, total_callable_ns=0, min_callable_ns=0, max_callable_ns=0,
                  comparison_scope='OBSERVED_ORIGINAL_TRIAL_CALLABLE', grad_mode_calls=dict(enabled=0, disabled=0))
    active = True
    busy = False
    invalid = False

    @wraps(original)
    def observed(*args, **kwargs):
        nonlocal busy, invalid
        if not active:
            return original(*args, **kwargs)
        if os.getpid() != process or get_ident() != owner or busy:
            invalid = True
            return original(*args, **kwargs)
        busy = True
        grad = grad_enabled()
        start = perf_counter_ns()
        try:
            result = original(*args, **kwargs)
            elapsed = perf_counter_ns() - start
        except BaseException:
            invalid = True
            raise
        finally:
            busy = False
        if elapsed <= 0:
            invalid = True
            return result
        demand['total_callable_ns'] += elapsed
        if not demand['calls'] or elapsed < demand['min_callable_ns']:
            demand['min_callable_ns'] = elapsed
        demand['max_callable_ns'] = max(demand['max_callable_ns'], elapsed)
        demand['calls'] += 1
        demand['grad_mode_calls']['enabled' if grad else 'disabled'] += 1
        return result

    namespace[name] = observed
    try:
        yield dict(counters, demand=demand)
    finally:
        active = False
        if namespace.get(name) is observed:
            namespace[name] = original
    if invalid:
        raise ValueError('original callable demand requires successful non-reentrant calls from the entry owner and a positive clock')


def run_aten_python_entry(trial, *, trial_sha256, entry_function, mode, report, entry_args,
                          allow_ambient_grad=False):
    directory = Path(trial).resolve(strict=True)
    report = Path(report).resolve()
    if report.exists():
        raise FileExistsError(report)
    if mode not in _MODES:
        raise ValueError('explicit original, guarded, or observe-original entry mode required')
    setup_start = perf_counter_ns()
    manifest = _read_manifest(directory, trial_sha256)
    proof = manifest['source_proof']
    if entry_function == proof['function']:
        raise ValueError('the application entry must be distinct from the trial callable')

    entry_args = list(entry_args)
    argv = sys.argv
    mask = os.sched_getaffinity(0)
    if mask != set(manifest['cpus']):
        raise ValueError('application trial entry requires the recorded launch CPU mask')
    sys.argv = [proof['source']['path'], *entry_args]
    try:
        module = _original_module(proof)
        namespace = vars(module)
        entry = namespace.get(entry_function)
        if type(entry) is not FunctionType or entry.__globals__ is not namespace or entry.__qualname__ != entry_function:
            raise ValueError('an original top-level entry of the trial source module is required')
        original = namespace[proof['function']]
        counters = dict(native_calls=0, fallback_calls=0, fallback_reasons={})
        provider_report = dict(status='NOT_ATTACHED', libraries=[], content_rehashed_bytes=0, identity_reused_bytes=0)
        torch = sys.modules['torch']
        if mode == 'GUARDED':
            session = install_aten_python_trial(module, directory, expected_sha256=trial_sha256,
                                                exclusive_inputs=True, defer_context=True,
                                                allow_ambient_grad=allow_ambient_grad, provider_report=provider_report)
        elif mode == 'OBSERVE_ORIGINAL':
            session = _observe_original(module, proof, counters, torch.is_grad_enabled)
        else:
            session = nullcontext(counters)
        with session as accounting:
            grad_at_entry = torch.is_grad_enabled()
            entry_start = perf_counter_ns()
            result = entry()
            entry_end = perf_counter_ns()
            grad_at_return = torch.is_grad_enabled()
        finished = perf_counter_ns()
        if namespace[proof['function']] is not original:
            raise ValueError('the application trial did not restore the original function')
        if os.sched_getaffinity(0) != mask:
            raise ValueError('the application trial changed the entry CPU affinity at return')
    finally:
        sys.argv = argv

    # This profile compares ordinary scalar returns as well as the declared output files.
    if type(result) not in (type(None), bool, int, str):
        raise ValueError('the original entry return is outside the scalar validation profile')
    data = dict(schema='scar.aten-python-entry-run', schema_version=1, mode=mode,
        process_id=os.getpid(), entry_function=entry_function, cpu_ids=sorted(mask),
        trial_sha256=trial_sha256, entry_args=entry_args,
        entry_ns=entry_end-entry_start, setup_ns=entry_start-setup_start, teardown_ns=finished-entry_end,
        entry_return=dict(kind=type(result).__name__, value=result),
        grad_enabled_at_entry=grad_at_entry, grad_enabled_at_return=grad_at_return,
        affinity_restored=True, original_binding_restored=True, provider_verification=provider_report, **accounting)
    with report.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar run-aten-python-entry', description=__doc__)
    parser.add_argument('--trial', required=True)
    parser.add_argument('--trial-sha256', required=True)
    parser.add_argument('--entry-function', default='main')
    parser.add_argument('--mode', choices=_MODES, required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--allow-ambient-grad', action='store_true')
    parser.add_argument('entry_args', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.entry_args[:1] == ['--']:
        args.entry_args = args.entry_args[1:]
    run_aten_python_entry(**vars(args))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
