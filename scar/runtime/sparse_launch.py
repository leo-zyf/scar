"""Sparse launch."""

import argparse
import json
import math
from pathlib import Path
from scar.ir.execution_recipe import read_artifact
from scar.runtime.csr_region_entry import launch_context
from scar.runtime.entry_runner import _arguments
from scar.runtime.sparse_solve_entry import SELECTION, run as run_study
from scar.analysis.sparse_parallel_recipe import compute_scale


# Select the verified sparse-solver study for its exact launch context.
LAUNCH = 'scar.sparse-direct-solve-launch'


def _select(entry, argv, launch, expected_sha256, exclusive):
    if not launch or not expected_sha256 or exclusive is not True:
        return None, 'ORIGINAL_EXECUTION_CONDITIONS'
    try:
        path = Path(launch).resolve()
        selection = read_artifact(dict(uri=str(path), schema=LAUNCH, sha256=expected_sha256,
            size_bytes=path.stat().st_size))
        if selection['schema_version'] != 1 or selection['decision'] != 'LOAD_NATIVE_CANDIDATE':
            return None, 'KEEP_FROM_PLAN'
        if selection['entry']['uri'] != str(entry) or selection['context'] != launch_context(argv):
            return None, 'SPARSE_SOLVE_LAUNCH_CONTEXT_CHANGED'
        study = selection['study']
        if study['schema'] != SELECTION:
            raise ValueError('symbolic launch requires a sealed validated study')
        return study, 'QUALIFIED_SPARSE_SYMBOLIC_SCOPE'
    except (OSError, ValueError, KeyError, TypeError):
        return None, 'SPARSE_SOLVE_LAUNCH_ARTIFACT_CHANGED'


def run(entry, argv, *, launch=None, sha256=None, exclusive=False):
    entry, argv = _arguments(entry, argv)
    study, reason = _select(entry, argv, launch, sha256, exclusive)
    report = run_study(entry, argv, selection=None if study is None else study['uri'],
        sha256=None if study is None else study['sha256'], study=study is not None, exclusive=exclusive)
    report['launch'] = dict(reason=reason, sha256=sha256,
        applied='LOAD_NATIVE_CANDIDATE' if report['statistics']['native_calls'] else 'KEEP')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch')
    parser.add_argument('--launch-sha256')
    parser.add_argument('--exclusive-inputs', action='store_true')
    parser.add_argument('--report', type=Path)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('original Python script required after --')
    if args.report and args.report.exists():
        parser.error('fresh report path required')
    report = run(command[0], command[1:], launch=args.launch, sha256=args.launch_sha256,
        exclusive=args.exclusive_inputs)
    if args.report:
        with args.report.open('x') as stream:
            json.dump(report, stream, sort_keys=True, indent=2)
            stream.write('\n')


# Build private requests that may match the next original CSR solver calls.
class DetachedCSR:
    def __init__(self, matrix, load, boundary, count, expression, *, size, nnz):
        import numpy as np
        from scipy.sparse import csr_array, csr_matrix

        if (type(matrix) not in (csr_matrix, csr_array) or matrix.shape != (size, size)
                or type(count) is not int or count <= 0 or type(boundary) is not np.ndarray or boundary.ndim != 1):
            raise ValueError('unsupported source matrix, row selection or count')
        arrays = ((matrix.data, nnz, np.dtype('float64')), (matrix.indices, nnz, np.dtype('int32')),
                  (matrix.indptr, size+1, np.dtype('int32')), (load, size, np.dtype('float64')))
        if (any(type(arr) is not np.ndarray or arr.shape != (length,) or arr.dtype != dtype
                for arr, length, dtype in arrays) or boundary.dtype not in (np.dtype('int32'), np.dtype('int64'))
                or boundary.size > size):
            raise ValueError('expected the original CSR dtypes and vector sizes')
        self.data, self.columns, self.offsets, self.load = (arr.copy() for arr, _, _ in arrays)
        self.rows = rows = boundary.copy()
        if (np.any(rows < 0) or np.any(rows >= size) or self.offsets[0] != 0 or self.offsets[-1] != nnz
                or np.any(self.offsets[:-1] > self.offsets[1:])):
            raise ValueError('invalid detached CSR topology')
        row_for_column = np.repeat(np.arange(size), np.diff(self.offsets))
        diagonal = np.flatnonzero(self.columns == row_for_column)
        if diagonal.size != size or not np.array_equal(row_for_column[diagonal], np.arange(size)):
            raise ValueError('expected one stored diagonal in each row')
        self.data[diagonal] = 0.0 + self.data[diagonal]
        for row in rows:
            self.data[self.offsets[row]:self.offsets[row+1]] = 0.0
        self.selected_diagonal = diagonal[rows]
        for array in (self.data, self.columns, self.offsets, self.load, self.rows, self.selected_diagonal):
            array.flags.writeable = False
        self.size, self.nnz, self.count, self.expression = size, nnz, count, expression

    def args(self, index):
        scale = compute_scale(self.expression, index, self.count)
        if not math.isfinite(scale):
            raise ValueError('finite speculative row scale required')
        data = self.data.copy()
        data[self.selected_diagonal] = scale
        rhs = self.load * scale
        rhs[self.rows] = 0.0 * scale
        return self.size, self.nnz, data, self.columns, self.offsets, rhs, 0


def match_request(expected, live, options):
    """Compare one raw carrier request without evaluating application equality hooks."""
    from numpy import ndarray

    if (type(live) is not tuple or len(live) != 7 or type(options) is not dict
            or len(options) != 1 or any(type(key) is not str for key in options)
            or 'ColPerm' not in options or options['ColPerm'] is not None):
        return False
    for index in (0, 1, 6):
        if type(live[index]) is not int or live[index] != expected[index]:
            return False
    for source, candidate in zip(live[2:6], expected[2:6]):
        if (type(source) is not ndarray or source.shape != candidate.shape or source.dtype != candidate.dtype
                or not source.flags.c_contiguous or memoryview(source).cast('B') != memoryview(candidate).cast('B')):
            return False
    return True


if __name__ == '__main__':
    main()
