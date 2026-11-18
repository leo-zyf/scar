"""A scalar fold planned in nearest must not bake the caller's FP mode."""
from contextlib import contextmanager
import ctypes
from dataclasses import replace
import operator
import platform
import shutil
import sys

import numpy as np
import pytest

from scar.compiler.model import ArrayRegion, Expr, SourceRef
from scar.codegen import cpp
from scar.optimization.optimizer import optimize_region


@contextmanager
def _held_fenv():
    # This regression exercises the admitted Linux x86-64 profile. The bit
    # constants are glibc's ABI; the correctness micro also records GCC macros.
    if sys.platform != 'linux' or platform.machine().lower() not in ('x86_64', 'amd64'):
        pytest.skip('Linux x86-64 floating-environment regression')
    try:
        lib = ctypes.CDLL('libm.so.6')
    except OSError:
        pytest.skip('glibc floating environment unavailable')
    for name in ('feholdexcept', 'fesetenv'):
        fn = getattr(lib, name); fn.argtypes = [ctypes.c_void_p]; fn.restype = ctypes.c_int
    for name in ('fesetround', 'fetestexcept', 'feraiseexcept'):
        fn = getattr(lib, name); fn.argtypes = [ctypes.c_int]; fn.restype = ctypes.c_int
    lib.fegetround.argtypes = []; lib.fegetround.restype = ctypes.c_int
    before = (lib.fegetround(), lib.fetestexcept(0x3d))
    saved = ctypes.create_string_buffer(256)
    assert lib.feholdexcept(saved) == 0
    try:
        yield lib
    finally:
        assert lib.fesetenv(saved) == 0
        assert (lib.fegetround(), lib.fetestexcept(0x3d)) == before


def _evaluate(region, array):
    """Independent original operand-order Python scalar / NumPy oracle."""
    values = {}
    operators = dict(add=operator.add, mul=operator.mul)
    for node in region.nodes:
        values[node.id] = (array if node.op == 'input' else node.attrs['value']
            if node.op == 'const' else operators[node.op](*(values[i] for i in node.inputs)))
    return values[region.output]


def test_planning_rounding_does_not_change_runtime_float_constant_arithmetic(tmp_path):
    array = np.array([[1.0]], dtype=np.float64)
    region = ArrayRegion('rounding',
        SourceRef('/tmp/fp_rounding.py', '0'*64, 'rounding', 1, 0, 'rounding-test'),
        ('x',), (Expr('x', 'input', attrs=dict(name='x')),
            Expr('one', 'const', attrs=dict(value=1.0)),
            Expr('half', 'const', attrs=dict(value=float.fromhex('0x1p-53'))),
            Expr('sum', 'add', ('one', 'half')),
            Expr('output', 'mul', ('x', 'sum'))), 'output')
    bits = lambda value:value.view(np.uint64).item()
    # Deliberately make the environment being restored non-default, with a
    # pre-existing exception flag. Both nested finally blocks restore state,
    # including if the numeric regression assertion fails.
    with _held_fenv() as fenv:
        assert fenv.fesetround(0x800) == 0  # FE_UPWARD
        assert fenv.feraiseexcept(0x01) == 0  # FE_INVALID, traps held
        saved_flags = fenv.fetestexcept(0x3d)
        with _held_fenv():
            assert fenv.fesetround(0) == 0  # FE_TONEAREST during planning
            plan = optimize_region(region, memory_budget_bytes=8, cpu_threads=1)
            assert plan.proof['supported'] is True
            assert bits(_evaluate(region, array)) == 0x3ff0000000000000
            assert bits(_evaluate(plan.region, array)) == 0x3ff0000000000000
            assert fenv.fesetround(0x800) == 0
            assert bits(_evaluate(region, array)) == 0x3ff0000000000001
            assert bits(_evaluate(plan.region, array)) == 0x3ff0000000000001
            assert fenv.fegetround() == 0x800
            if shutil.which('g++'):
                artifact = cpp.build(plan, tmp_path / 'rounding')
                assert bits(artifact.run(array)) == 0x3ff0000000000001
                assert fenv.fegetround() == 0x800
                division = replace(region, nodes=tuple(
                    Expr('half', 'const', attrs={'value': 0.0}) if node.id == 'half'
                    else Expr('sum', 'div', node.inputs) if node.id == 'sum' else node
                    for node in region.nodes))
                error_plan = optimize_region(division, memory_budget_bytes=8)
                guarded = cpp.build(error_plan, tmp_path / 'empty-division')
                with pytest.raises(cpp.GuardFailure):
                    guarded.run(np.empty((0, 0), dtype=np.float64))
        assert fenv.fegetround() == 0x800
        assert fenv.fetestexcept(0x3d) == saved_flags
