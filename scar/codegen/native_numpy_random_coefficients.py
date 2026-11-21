"""Source-bound finite 1j**real and small two-term spectral plans for NumPy."""

from pathlib import Path

from scar.codegen import native_numpy_complex
from scar.codegen.stateful_numpy_provider import _Image
from scar.ir.execution_recipe import artifact_reference, read_artifact

SCHEMA = 'scar.native-complex-phase-and-spectral-pair'
SOURCE = 'src/numpy_random_coefficients.cpp'
HEADER = 'include/scar_generated/numpy_random_coefficients.hpp'
_HEADER = '''#pragma once
#include <cstddef>
extern "C" int scar_pure_imaginary_phase_v1(double exponent, double* result) noexcept;
extern "C" int scar_complex_spectral_pair_v1(double exponent, double shift,
                                             const double* half_turns, const double* components,
                                             std::size_t count, double* result) noexcept;
'''


def render(cpython, provider):
    """Check the finite branch of the original scalar ABI and emit C++ sources."""
    image = _Image(cpython)
    _, complex_power = image.instructions('complex_pow.lto_priv.0')
    _, stock_power = image.instructions('_Py_c_pow')
    dispatch = {operand for _, _, opcode, operand in complex_power if opcode == 'call'}
    libm = ('hypot', 'pow', 'atan2', 'sincos')
    math_calls = {operand for _, _, opcode, operand in stock_power if opcode == 'call'}
    if (not all(any(f'<{symbol}>' in term for term in dispatch)
                for symbol in ('c_powu', '_Py_c_quot', '_Py_c_pow'))
            or not all(any(f'<{symbol}@GLIBC_2.2.5>' in term for term in math_calls) for symbol in libm)):
        raise ValueError('unqualified original CPython complex-real exponent path')
    stock = read_artifact(provider)
    sources = native_numpy_complex.render(stock)
    if [16, 0, 16] not in stock['strides']['multiply'] or [16, 16, 16] not in stock['strides']['add']:
        raise ValueError('the source-bound stock ufunc receipt omits coefficient strides')
    sources[SOURCE] = Path(__file__).with_suffix('.cpp.in').read_bytes()
    sources[HEADER] = _HEADER.encode()
    report = dict(schema=SCHEMA, schema_version=1, cpython=cpython, complex_provider=provider,
                  base='1j', exponent_range=[-2, 2], finite=True,
                  real_noninteger='GLIBC_SINCOS_ON_ORIGINAL_ATAN2_AND_BINARY64_PRODUCT',
                  integer='ORIGINAL_CPOWU_AND_RECIPROCAL_SIGNED_ZERO_CASES',
                  limits={'spectral_terms': 2, 'contiguous_complex_elements': [1, 4]},
                  spectral_order='MULTIPLY_EACH_TERM_THEN_ADD_IN_SOURCE_ORDER_FROM_POSITIVE_ZERO',
                  source_contract='CALLER_PROVES_DIRECTION_DTYPE_REDUCTION_AND_PARAMETERS',
                  python_runtime_required=False, deployment_authorized=False)
    return report, sources


def check(path):
    """Read back the generated provider and its retained validation materials."""
    stored = read_artifact(artifact_reference(Path(path).resolve(strict=True), SCHEMA))
    original, members = render(stored['cpython'], stored['complex_provider'])
    if (any(stored[key] != value for key, value in original.items())
            or members != {name: read_artifact(ref, json_document=False) for name, ref in stored['members'].items()}):
        raise ValueError('recorded native coefficient code differs from its original ABI')
    native = stored['native_library']
    read_artifact(native, json_document=False)
    if native_numpy_complex.audit(native['uri'], read_artifact(stored['complex_provider'])) != stored['linked_stock_complex']:
        raise ValueError('linked coefficient computation differs from the original stock operators')
    read_artifact(stored['verifier'], json_document=False)
    read_artifact(stored['original_seeded_rng'])
    for sample in stored['samples']:
        read_artifact(sample['rng_draws'], json_document=False)
        if sample['tape'] is not None:
            read_artifact(sample['tape'])
    return stored
