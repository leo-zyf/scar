"""Run a source-sealed sparse-solver study through the original Python entry."""

import argparse
import json
import marshal
from pathlib import Path
from time import perf_counter_ns
from types import BuiltinFunctionType, CodeType, FunctionType

from scar.analysis.sparse_solve_callsite import OFFER
from scar.ir.execution_recipe import read_artifact
from scar.runtime.csr_region_entry import _observer_active, interpreter_context, launch_context
from scar.runtime.entry_runner import _arguments, _execute, _file
from scar.trace.code_snapshot import code_key

SELECTION = 'scar.sparse-solve-entry-selection'
ENTRY_CODE = 'scar.python-sparse-solve-callsite'


class SparseSolveScope:
    def __init__(self, selection, loader):
        callsite = selection['callsite']
        code = selection['entry_code']
        self.source = Path(callsite['source']['uri'])
        self.name = callsite['qualname']
        self.original_key = callsite['code_key']
        self.provider = callsite['provider_binding']
        self.carrier = callsite['carrier']['name']
        self.library = Path(callsite['carrier']['library']['uri'])
        if code['interpreter'] != interpreter_context() or code['artifact']['schema'] != ENTRY_CODE:
            raise ValueError('sealed solve callsite requires its original Python interpreter')
        self.revised = marshal.loads(read_artifact(code['artifact'], json_document=False))
        if (type(self.revised) is not CodeType or self.revised.co_filename != str(self.source)
                or self.revised.co_qualname != self.name or self.revised.co_freevars
                or code_key(self.revised) != code['code_key']):
            raise ValueError('sealed solve callsite does not match its original boundary')
        self.loader, self.candidate, self.original_carrier = loader, None, None
        self.active, self.bindings, self.frontier = True, [], []
        self.activation = dict(attempts=0, load_ns=None)
        self.kernel = None

    def activate(self, original):
        if self.activation['attempts']:
            return self.candidate if original is self.original_carrier else None
        self.activation['attempts'] = 1
        self.original_carrier = original
        started = perf_counter_ns()
        try:
            self.candidate = self.loader(original)
        except (OSError, ValueError, ImportError, KeyError, TypeError, AttributeError):
            self.frontier.append('SPARSE_NATIVE_PROVIDER_NOT_LOADED')
        finally:
            self.activation['load_ns'] = perf_counter_ns()-started
        return self.candidate

    def attach(self, module, name, *, natural):
        namespace = vars(module)
        original = namespace.get(self.name)
        provider = namespace.get(self.provider)
        if (not natural or _file(module) != self.source or type(original) is not FunctionType
                or original.__globals__ is not namespace or OFFER in namespace
                or code_key(original.__code__) != self.original_key or _file(provider) != self.library):
            self.frontier.append('SPARSE_CALLSITE_SOURCE_OR_PROVIDER_CHANGED')
            return
        expected = vars(provider).get(self.carrier)
        if (type(expected) is not BuiltinFunctionType or expected.__self__ is not provider
                or expected.__name__ != self.carrier):
            self.frontier.append('SPARSE_CALLSITE_CARRIER_CHANGED')
            return
        stats = dict(native_calls=0, fallback_calls=0, reasons={})

        def offer(carrier, args, options):
            if self.active and carrier is expected and not _observer_active():
                candidate = self.candidate if self.original_carrier is expected else self.activate(expected)
                if candidate is not None:
                    result = candidate(args, options)
                    if result is not None:
                        stats['native_calls'] += 1
                        return result
            stats['fallback_calls'] += 1
            return None

        initial_code = original.__code__
        namespace[OFFER] = offer
        original.__code__ = self.revised
        self.bindings.append(dict(module_name=name, natural_import=natural, namespace=namespace,
            function=original, initial_code=initial_code, offer=offer, stats=stats))

    def finish(self):
        self.active = False
        for row in self.bindings:
            if row['function'].__code__ is self.revised:
                row['function'].__code__ = row['initial_code']
            if row['namespace'].get(OFFER) is row['offer']:
                del row['namespace'][OFFER]
        if self.candidate is not None:
            self.kernel = dict(self.candidate.stats)
            self.candidate.close()


def admit(entry, argv, selection_path, expected_sha256, *, study, exclusive):
    if study is not True or exclusive is not True or not selection_path or not expected_sha256 or _observer_active():
        return None, 'ORIGINAL_EXECUTION_CONDITIONS'
    try:
        path = Path(selection_path).resolve()
        selection = read_artifact(dict(uri=str(path), size_bytes=path.stat().st_size,
            sha256=expected_sha256, schema=SELECTION))
        if (selection['schema_version'] != 1 or selection['explicit_study_only'] is not True
                or selection['entry']['uri'] != str(entry) or selection['launch_context'] != launch_context(argv)):
            return None, 'SPARSE_SOLVE_ENTRY_OR_CONTEXT_CHANGED'
        callsite = selection['callsite']
        for dependency in (selection['entry'], callsite['source'], callsite['carrier']['library'],
                           *selection['runtime_members']):
            read_artifact(dependency, json_document=False)
        native = selection['native']
        read_artifact(native)

        def loader(original):
            from scar.runtime.sparse_lu import load_symbolic_solve

            return load_symbolic_solve(native, budget_bytes=selection['budget_bytes'], original=original)

        return SparseSolveScope(selection, loader), 'EXPLICIT_SPARSE_SOLVE_STUDY'
    except (OSError, ValueError, ImportError, TypeError, KeyError, EOFError):
        return None, 'SPARSE_SOLVE_SELECTION_OR_SOURCE_CHANGED'


def run(entry, argv, *, selection=None, sha256=None, study=False, exclusive=False):
    entry, argv = _arguments(entry, argv)
    scope, reason = admit(entry, argv, selection, sha256, study=study, exclusive=exclusive)
    report = _execute(entry, argv, scope, reason)
    report['activation'] = None if scope is None else scope.activation
    report['symbolic'] = None if scope is None else scope.kernel
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
