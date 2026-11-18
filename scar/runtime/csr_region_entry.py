"""Attach a guarded CSR region to an otherwise unchanged Python entry."""

import argparse
from functools import wraps
from importlib.util import MAGIC_NUMBER
import json
import marshal
import os
from pathlib import Path
import sys
from time import perf_counter_ns
from types import CodeType, FunctionType

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.runtime.entry_runner import _Scope, _arguments, _execute, _file
from scar.trace.code_snapshot import code_key

SELECTION = 'scar.csr-row-region-selection'
ENTRY_CODE = 'scar.python-csr-row-region-code'
DISPATCH = '__scar_dispatch_private_csr_row_region'


def interpreter_context():
    return dict(name=sys.implementation.name, executable=sys.executable,
        version=sys.version, optimize=sys.flags.optimize, magic=MAGIC_NUMBER.hex())


def launch_context(argv):
    names = ('PYTHONPATH', 'PYTHONHASHSEED', 'PYTHONOPTIMIZE', 'LD_PRELOAD', 'LD_LIBRARY_PATH',
             'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'OMP_PROC_BIND', 'OMP_PLACES', 'GLIBC_TUNABLES', 'MALLOC_ARENA_MAX')
    return dict(argv=list(argv), interpreter=interpreter_context(), cwd=os.getcwd(),
        cpu_ids=sorted(os.sched_getaffinity(0)), environment={name: os.environ.get(name) for name in names})


def _observer_active():
    monitor = getattr(sys, 'monitoring', None)
    return (sys.gettrace() is not None or sys.getprofile() is not None
            or (monitor is not None and any(monitor.get_tool(i) is not None for i in range(6))))


class RowRegionScope(_Scope):
    def __init__(self, proof, loader, *, code=None):
        self.source = Path(proof['source']['uri'])
        self.name = proof['region']['qualname']
        self.original_key = proof['region']['code_key']
        source = read_artifact(proof['source'], json_document=False)
        if code is None:
            from scar.analysis.csr_row_region import recognize

            matched, self.revised_code = recognize(source, str(self.source), self.original_key,
                optimize=sys.flags.optimize, helper=DISPATCH)
            if matched != proof['region']:
                raise ValueError('retained CSR source region differs from compiler replay')
        else:
            if code['interpreter'] != interpreter_context() or code['artifact']['schema'] != ENTRY_CODE:
                raise ValueError('sealed CSR code requires the original interpreter and profile')
            revised = marshal.loads(read_artifact(code['artifact'], json_document=False))
            if (type(revised) is not CodeType or revised.co_filename != str(self.source) or revised.co_qualname != self.name
                    or code_key(revised) != code['revised_code_key']):
                raise ValueError('sealed CSR code differs from the current interpreter or boundary')
            self.revised_code = revised
        self.loader = loader
        self.try_enforce = None
        self.activation = dict(attempts=0, load_ns=None, available=False)
        self.active, self.bindings, self.seen, self.frontier = True, [], set(), []

    def activate(self):
        if self.activation['attempts']:
            return self.try_enforce
        self.activation['attempts'] = 1
        started = perf_counter_ns()
        try:
            self.try_enforce = self.loader()
        except (OSError, ValueError, ImportError, KeyError, TypeError, AttributeError):
            self.frontier.append('CSR_NATIVE_PROVIDER_CHANGED')
        finally:
            self.activation['load_ns'] = perf_counter_ns() - started
            self.activation['available'] = self.try_enforce is not None
        return self.try_enforce

    def attach(self, module, name, *, natural):
        if not self.active or id(module) in self.seen:
            return
        self.seen.add(id(module))
        namespace = vars(module)
        original = namespace.get(self.name)
        if (not natural or _file(module) != self.source or type(original) is not FunctionType
                or original.__globals__ is not namespace or DISPATCH in namespace
                or code_key(original.__code__) != self.original_key):
            self.frontier.append('ORIGINAL_CSR_BINDING_NOT_NATURALLY_IMPORTED')
            return

        stats = dict(native_calls=0, fallback_calls=0, reasons={})

        def dispatch(original_matrix, private_matrix, rows, diag, overwrite, numpy_binding):
            # Exact original type and detached output are also checked below
            # the sparse adapter boundary. Do not bypass overwrite semantics.
            if (overwrite is False and type(original_matrix) is type(private_matrix)
                    and private_matrix is not original_matrix):
                run_native = self.try_enforce or self.activate()
                if run_native is not None and run_native(private_matrix, rows, diag, numpy_binding, original=original_matrix):
                    stats['native_calls'] += 1
                    return True
            stats['fallback_calls'] += 1
            return False

        revised = FunctionType(self.revised_code, namespace, self.name, original.__defaults__)
        revised.__kwdefaults__ = original.__kwdefaults__

        @wraps(original, assigned=('__module__', '__name__', '__qualname__', '__doc__'))
        def binding(*args, **kwargs):
            if self.active and not _observer_active() and namespace.get(DISPATCH) is dispatch:
                return revised(*args, **kwargs)
            stats['fallback_calls'] += 1
            return original(*args, **kwargs)

        if getattr(original, '__annotate__', None) is not None:
            binding.__annotate__ = original.__annotate__
        else:
            binding.__annotations__ = original.__annotations__
        namespace[DISPATCH] = dispatch
        namespace[self.name] = binding
        self.bindings.append(dict(module=module, module_name=name, owner=module,
            attribute=self.name, original=original, descriptor=original, installed=binding,
            binding=binding, natural_import=natural, stats=stats, dispatch=dispatch))

    def finish(self):
        try:
            super().finish()
        finally:
            for row in self.bindings:
                namespace = vars(row['module'])
                if namespace.get(DISPATCH) is row['dispatch']:
                    del namespace[DISPATCH]


def admit(entry, selection_path, sha256, *, study, exclusive, argv=()):
    """Return a checked scope or an original-execution reason without rerunning the graph."""
    if exclusive is not True or not selection_path or not sha256 or _observer_active():
        return None, 'ORIGINAL_EXECUTION_CONDITIONS'
    try:
        path = Path(selection_path).resolve()
        selection = read_artifact(dict(uri=str(path), sha256=sha256, size_bytes=path.stat().st_size,
            schema=SELECTION))
        version = selection['schema_version']
        if version not in (1, 2):
            return None, 'UNSUPPORTED_CSR_SELECTION'
        selected = (version == 2 and selection['decision'] == 'LOAD_NATIVE_CANDIDATE'
                    and selection['explicit_study_only'] is False)
        if not (study is True or selected):
            return None, selection['reason']
        if study is not True and selection.get('launch_context') != launch_context(argv):
            return None, 'CSR_LAUNCH_CONTEXT_CHANGED'
        if selection['entry']['uri'] != str(entry) or selection['entry']['uri'] == selection['source']['uri']:
            return None, 'ORIGINAL_ENTRY_DIFFERS'
        read_artifact(selection['entry'], json_document=False)
        proof = read_artifact(selection['proof'])
        if proof['source'] != selection['source'] or proof['entry'] != selection['entry']:
            raise ValueError('selected CSR source differs from proof')

        native = selection['native']
        read_artifact(native)
        load_path = Path(native['uri']).parent
        for reference in selection['native_members']:
            read_artifact(reference, json_document=False)

        def loader():
            from scar.runtime.sparse_csr import load_csr_enforce

            return load_csr_enforce(load_path)

        code = selection['entry_code'] if version == 2 else None
        if code is not None and code['proof'] != selection['proof']:
            raise ValueError('sealed CSR code requires its original region proof')
        scope = RowRegionScope(proof, loader, code=code)
        return scope, 'EXPLICIT_STUDY' if study is True else 'PINNED_CSR_SELECTION'
    except (OSError, ValueError, ImportError, KeyError, TypeError, StopIteration, EOFError):
        return None, 'CSR_SELECTION_OR_SOURCE_CHANGED'


def run(entry, argv, *, selection=None, sha256=None, study=False, exclusive=False):
    entry, argv = _arguments(entry, argv)
    scope, reason = admit(entry, selection, sha256, study=study, exclusive=exclusive, argv=argv)
    report = _execute(entry, argv, scope, reason)
    report['activation'] = None if scope is None else scope.activation
    report['selection_sha256'] = sha256
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection')
    parser.add_argument('--selection-sha256')
    parser.add_argument('--exclusive-inputs', action='store_true')
    parser.add_argument('--study', action='store_true')
    parser.add_argument('--report', type=Path)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('original Python script required after --')
    if args.report and args.report.exists():
        parser.error('fresh report path required')
    report = run(command[0], command[1:], selection=args.selection, sha256=args.selection_sha256,
        study=args.study, exclusive=args.exclusive_inputs)
    if args.report:
        with args.report.open('x') as stream:
            json.dump(report, stream, sort_keys=True, indent=2)
            stream.write('\n')


if __name__ == '__main__':
    main()
