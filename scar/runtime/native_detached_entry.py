"""Study C++20 compact consumption at the original sealed sparse callsite."""

import argparse
import ctypes
import json
from pathlib import Path
from threading import get_ident

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.runtime.entry_runner import _arguments, _execute
from scar.runtime.sparse_lu_native import read_native_manifest
from scar.runtime.sparse_parallel_entry import _admit, _affinity, _observe_inputs, _resources_fit
from scar.runtime.sparse_parallel_lu import _floating_control
from scar.runtime.sparse_solve_entry import admit as admit_solve
from scar.runtime.sparse_solve_prediction import DetachedCSR


_FAILURES = (ArithmeticError, EOFError, OSError, ValueError, MemoryError, TypeError, KeyError, AttributeError)


def _member(package, reference):
    return read_artifact({**reference, 'uri': str(package / reference['uri'])})


def _admit_package(package, sha256, selection, *, selection_sha256=None):
    from scar.codegen.native_ordered_sparse_export import SCHEMA, check_native_ordered_export

    root = Path(package).resolve(strict=True)
    if artifact_reference(root / 'artifact.json', SCHEMA)['sha256'] != sha256:
        raise ValueError('compact package fingerprint differs')
    manifest = check_native_ordered_export(root)
    if (manifest['schema_version'] != 3 or manifest['recipe'] != selection['execution_recipe']
            or {**manifest['detached_source'], 'uri': selection['recipe']['uri']} != selection['recipe']):
        raise ValueError('compact package and admitted source recipe differ')
    release = manifest.get('qualified_release')
    if release is not None and (
            selection_sha256 not in (None, release['sha256']) or _member(root, release) != selection):
        raise ValueError('compact package and admitted release selection differ')
    plan = _member(root, manifest['plan'])
    numeric = _member(root, manifest['numeric'])
    placement, workers, memory = plan['placement'], selection['workers'], plan['memory']
    if (numeric['build_provenance']['native'] != selection['native']
            or any(plan['case'][key] != selection['matrix'][key] for key in ('nnz', 'size'))
            or plan['case']['call_count'] != selection['count']
            or (placement['launch_cpu_ids'], placement['owner_cpu_id'], placement['worker_cpu_ids'],
                placement['cpu_numa_node'], plan['dispatch']['window'], memory['worker_address_space_limit_bytes'],
                memory['per_worker_symbolic_budget_bytes'], memory['minimum_launch_headroom_bytes'])
            != (selection['launch_cpu_ids'], selection['owner_cpu'], workers['cpu_ids'],
                selection['numa_node'], workers['window'], workers['address_space_limit_bytes'],
                selection['symbolic_budget_bytes'], selection['memory_floor_bytes'])):
        raise ValueError('compact package numeric provider, ports or resource policy differ')
    recipe = read_artifact(selection['recipe'])
    blas = read_native_manifest(selection['native'])['blas']
    read_artifact(blas, json_document=False)
    return root, manifest, recipe, blas['uri']


class _LiveBytes:
    def __init__(self):
        # Exact ndarray types, contiguous layout and exclusive input ownership are required by the caller.
        self.compare = ctypes.PyDLL(None).memcmp
        self.compare.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self.compare.restype = ctypes.c_int

    def equal_array(self, expected, live):
        return (type(live) is type(expected) and live.shape == expected.shape and live.dtype == expected.dtype
            and live.flags.c_contiguous and live.flags.aligned
            and self.compare(expected.__array_interface__['data'][0],
                live.__array_interface__['data'][0], expected.nbytes) == 0)

    def equal_request(self, expected, live, options):
        return (type(live) is tuple and len(live) == 7 and type(options) is dict and len(options) == 1
            and all(type(key) is str for key in options) and 'ColPerm' in options and options['ColPerm'] is None
            and all(type(live[slot]) is int and live[slot] == expected[slot] for slot in (0, 1, 6))
            and all(self.equal_array(expected[slot], live[slot]) for slot in (2, 3, 4, 5)))


class _CompactOffers:
    def __init__(self, selected, package, manifest, recipe, blas_path):
        self.selected, self.package, self.manifest, self.recipe = selected, package, manifest, recipe
        self.blas_path = blas_path
        self.owner = get_ident()
        self.original = self.snapshot = self.transport = None
        self.floating_control = self.blas_threads = self.live_bytes = None
        self.ticket = 0
        self.disabled = False
        self.stats = dict(inputs_captured=False, reference_calls=0, matched=0, rejected=0, first_mismatch=None)

    def activate(self, original):
        self.original = original
        blas = ctypes.CDLL(self.blas_path)
        if blas.scipy_openblas_get_parallel() != 1:
            raise ValueError('compact study requires original pthread OpenBLAS')
        self.blas_threads = blas.scipy_openblas_get_num_threads
        self.floating_control = _floating_control()
        self.live_bytes = _LiveBytes()
        return self

    def capture(self, values):
        try:
            inputs = {role: values[name] for role, name in self.recipe['roles'].items()}
            if type(inputs['count']) is not int or inputs['count'] != self.selected['count']:
                raise ValueError('original iteration count differs')
            self.snapshot = DetachedCSR(**inputs, expression=self.recipe['scale'], **self.selected['matrix'])
            self.stats['inputs_captured'] = True
        except _FAILURES:
            self._reject('DETACHED_LIVE_INPUTS_CHANGED')

    def _match(self, index, args, options):
        return (self.snapshot is not None and self.floating_control() and self.blas_threads() == 1
            and vars(self.original.__self__).get(self.original.__name__) is self.original
            and self.live_bytes.equal_request(self.snapshot.args(index), args, options))

    def _start(self):
        from scar.runtime.native_detached_stream import NativeDetachedStream

        self.transport = NativeDetachedStream(self.package, self.manifest, self.snapshot,
            self.selected['launch_cpu_ids'])

    def __call__(self, args, options):
        if self.disabled:
            return None
        index = self.ticket
        if get_ident() != self.owner or index >= self.selected['count']:
            self._reject('DETACHED_CONSUMER_SCOPE_CHANGED')
            return None
        reason = 'DETACHED_LIVE_REQUEST_CHANGED'
        try:
            if not self._match(index, args, options):
                self._reject(reason)
                return None
            reason = 'DETACHED_STREAM_FAILED'
            if self.transport is None:
                self._start()
            result = self.transport.take(index)
        except _FAILURES:
            self._reject(reason)
            return None
        self.ticket += 1
        if index == 0:
            # The first source call runs once and its result, warning or exception stays authoritative.
            self.stats['reference_calls'] += 1
            reference = self.original(*args, options=options)
            actual = reference[0]
            if (reference[1] != 0 or not self.live_bytes.equal_array(result[0], actual)
                    or actual.strides != result[0].strides or not actual.flags.owndata or not actual.flags.writeable):
                self._reject('DETACHED_FIRST_RESULT_CHANGED')
            return reference
        self.stats['matched'] += 1
        return result

    def _reject(self, reason):
        self.stats['rejected'] += 1
        self.stats['first_mismatch'] = self.stats['first_mismatch'] or reason
        self.close()

    def close(self):
        self.disabled = True
        self.snapshot = None
        if self.transport is not None:
            self.transport.close()

    def receipt(self):
        return dict(self.stats, transport=None if self.transport is None else dict(self.transport.stats))


def _execute_study(entry, argv, scope, row_scope):
    reason = 'EXPLICIT_COMPACT_SPARSE_SOLVE_STUDY'
    if row_scope is None:
        return _execute(entry, argv, scope, reason)

    from scar.runtime.sparse_pipeline_entry import _install, _Pipeline

    with _install(_Pipeline((row_scope, 'EXPLICIT_DEAD_RESULT_ITEM_STUDY'), (scope, reason))):
        report = _execute(entry, argv, None, reason)
    for binding in scope.bindings:
        stats = binding['stats']
        report['observed_boundary_calls'] += stats['native_calls'] + stats['fallback_calls']
        report['statistics']['fallback_calls'] += stats['fallback_calls']
    report['attachments'] = [dict(source=str(stage.source), module=binding['module_name'],
        natural_import=binding['natural_import']) for stage in (row_scope, scope) for binding in stage.bindings]
    report['installed_bindings'] = len(report['attachments'])
    report['decision'] = 'GUARDED_ATTACHMENT' if report['attachments'] else 'NO_OP'
    report['frontier'] = [*row_scope.frontier, *scope.frontier]
    return report


def run(entry, argv, *, selection=None, sha256=None, package=None, package_sha256=None, exclusive=False,
        dead_item=None, dead_item_sha256=None):
    entry, argv = _arguments(entry, argv)
    offers = None
    row_scope, row_reason = None, None
    reason = 'DETACHED_ORIGINAL_SELECTION_CHANGED'
    selected = _admit(entry, argv, selection, sha256, exclusive) if package and package_sha256 else None
    if selected is not None:
        reason = 'DETACHED_LIVE_RESOURCES_CHANGED'
        if _resources_fit(selected):
            try:
                root, manifest, recipe, blas = _admit_package(package, package_sha256, selected,
                    selection_sha256=sha256)
            except _FAILURES:
                reason = 'DETACHED_PACKAGE_OR_RECIPE_CHANGED'
            else:
                with _affinity({selected['owner_cpu']}):
                    study = selected['serial_study']
                    scope, reason = admit_solve(entry, argv, study['uri'], study['sha256'], study=True, exclusive=True)
                    if scope is not None:
                        offers = _CompactOffers(selected, root, manifest, recipe, blas)
                        scope.loader = offers.activate
                        if dead_item and dead_item_sha256:
                            from scar.runtime.dead_result_item import admit

                            try:
                                row_scope = admit(dead_item, dead_item_sha256, selected, selection, sha256)
                            except _FAILURES:
                                row_reason = 'DEAD_ITEM_PLAN_OR_SOURCE_CHANGED'
                        try:
                            with _observe_inputs(entry, recipe, offers.capture):
                                result = _execute_study(entry, argv, scope, row_scope)
                        finally:
                            offers.close()
    if offers is None:
        result = _execute(entry, argv, None, reason)
    else:
        result['statistics']['native_calls'] = offers.stats['matched']
        result['statistics']['reference_calls'] = offers.stats['reference_calls']
    result['schema'] = 'scar.native-detached-entry-execution'
    result['selection_sha256'], result['package_sha256'] = sha256, package_sha256
    result['compact'] = None if offers is None else offers.receipt()
    if dead_item:
        result['dead_item'] = (dict(elided=0, reason=row_reason or reason) if row_scope is None else row_scope.receipt())
    result['applied'] = 'CPP_DETACHED' if offers is not None and offers.stats['matched'] else 'KEEP'
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection')
    parser.add_argument('--selection-sha256')
    parser.add_argument('--native-detached', dest='package')
    parser.add_argument('--native-detached-sha256', dest='package_sha256')
    parser.add_argument('--dead-item', help='explicit dead-result-item study under normal completion')
    parser.add_argument('--dead-item-sha256')
    parser.add_argument('--exclusive-inputs', action='store_true')
    parser.add_argument('--report', type=Path)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('original Python script required after --')
    if args.report and args.report.exists():
        parser.error('fresh report path required')
    if bool(args.dead_item) != bool(args.dead_item_sha256):
        parser.error('pass both the dead-item plan and its fingerprint')
    result = run(command[0], command[1:], selection=args.selection, sha256=args.selection_sha256,
        package=args.package, package_sha256=args.package_sha256, exclusive=args.exclusive_inputs,
        dead_item=args.dead_item, dead_item_sha256=args.dead_item_sha256)
    if args.report:
        with args.report.open('x') as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write('\n')


if __name__ == '__main__':
    main()
