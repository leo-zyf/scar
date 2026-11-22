"""The C++20 delivery preserves Fortran 2018's borrowed C array views."""

import json
from pathlib import Path
import sys

import pytest

from scar.artifacts import file_reference
from scar.compiler.frontend.fortran import emit_fortran
from scar.ir.execution_recipe import read_artifact

from test_native_stage_handoff import _run, _tools


FORTRAN = '''subroutine observe(a, facts, before) bind(C)
  use, intrinsic :: iso_c_binding, only: c_float, c_int
  implicit none
  integer, parameter :: first = -2
  real(c_float), intent(inout) :: a(first:, 3:)
  integer(c_int), intent(out) :: facts(*)
  real(c_float), intent(out) :: before(4)
  integer :: i
  facts(1) = lbound(a, dim=1)
  facts(2) = lbound(a, 2)
  facts(3) = size(a, 1)
  facts(4) = size(a, 2)
  facts(5) = ubound(a, 1)
  facts(6) = ubound(a, 2)
  before = 0.0_c_float
  if (facts(3) < 2 .or. facts(4) < 3) return
  i = first + 1
  before(1) = a(first, 3)
  before(2) = a(i, 3)
  before(3) = a(first, 5)
  before(4) = a(i, 5)
  a(i, 4) = a(i, 4) + 100.0_c_float
  a(first, 5) = -a(first, 5)
end subroutine
'''

CALLER = r'''#include <ISO_Fortran_binding.h>
#include <stdio.h>

extern void observe(CFI_cdesc_t *, int *, float *);

static int run(int which) {
    float store[24];
    for (int i = 0; i < 24; ++i)
        store[i] = (float)(i + 1);

    CFI_CDESC_T(2) whole_storage, section_storage;
    CFI_cdesc_t *whole = (CFI_cdesc_t *)&whole_storage;
    CFI_cdesc_t *section = (CFI_cdesc_t *)&section_storage;
    CFI_index_t dims[2] = {2, 3};
    if (which == 1) {
        dims[0] = 4;
        dims[1] = 6;
    } else if (which == 2) {
        dims[0] = 0;
    }
    if (CFI_establish(whole, store, CFI_attribute_other, CFI_type_float,
                      sizeof(float), 2, dims) != CFI_SUCCESS)
        return 1;
    CFI_cdesc_t *view = whole;
    if (which == 1) {
        const CFI_index_t first[2] = {3, 0};
        const CFI_index_t last[2] = {1, 4};
        const CFI_index_t steps[2] = {-2, 2};
        if (CFI_establish(section, NULL, CFI_attribute_other, CFI_type_float,
                          sizeof(float), 2, NULL) != CFI_SUCCESS ||
            CFI_section(section, whole, first, last, steps) != CFI_SUCCESS)
            return 2;
        view = section;
    }

    int contiguous = CFI_is_contiguous(view);
    int facts[6];
    float before[4];
    observe(view, facts, before);
    printf("%d %d:", which, contiguous);
    for (int i = 0; i < 6; ++i)
        printf(" %d", facts[i]);
    printf(" | %.0f %.0f %.0f %.0f |", before[0], before[1], before[2], before[3]);
    for (int i = 0; i < 24; ++i)
        if (store[i] != (float)(i + 1))
            printf(" %d:%.0f", i, store[i]);
    puts("");
    return 0;
}

int main(void) {
    return run(0) || run(1) || run(2);
}
'''

EXPECTED = (b"0 1: -2 3 2 3 -1 5 | 1 2 5 6 | 3:104 4:-5\n"
            b"1 0: -2 3 2 3 -1 5 | 4 2 20 18 | 9:110 19:-20\n"
            b"2 1: 1 3 0 3 0 5 | 0 0 0 0 |\n")


def test_native_fortran_cfi_shapes_and_reverse_strides(tmp_path):
    providers = _tools(tmp_path, 'free-f2008')
    inputs = tmp_path / 'source'
    inputs.mkdir()
    caller = inputs / 'caller.c'
    kernel = inputs / 'observe.f90'
    caller.write_text(CALLER, encoding='utf-8')
    kernel.write_text(FORTRAN, encoding='utf-8')
    units = [{'path': caller.name, 'language': 'c', 'flags': ['-std=c17']},
             {'path': kernel.name, 'language': 'fortran', 'flags': ['-std=f2018']}]
    _run(tmp_path, 'baseline-c', ['gcc', '-std=c17', '-fno-fast-math', '-c', caller, '-o', 'caller.o'])
    _run(tmp_path, 'baseline-fortran', ['gfortran', '-std=f2018', '-c', kernel, '-o', 'observe.o'])
    original = tmp_path / 'original'
    _run(tmp_path, 'baseline-link', ['gfortran', 'caller.o', 'observe.o', '-o', original])
    assert _run(tmp_path, 'baseline', [original]) == EXPECTED

    request = {'schema': 'scar.native-entry', 'schema_version': 1, 'language': 'mixed',
               'source_root': str(inputs), 'entry': 'main', 'units': units, 'fortran': providers,
               'source_lock': {'files': {path.name: file_reference(path, 'scar.native-source')
                                         for path in (caller, kernel)}},
               'cases': [{'id': 'view-layouts', 'comparison': 'exact'}],
               'baseline': {'argv': [str(original)], 'cwd': str(tmp_path)}}
    manifest = tmp_path / 'entry.json'
    manifest.write_text(json.dumps(request), encoding='utf-8')
    graph, plan, delivery = (tmp_path / stage for stage in ('graph', 'plan', 'delivery'))
    scar = [sys.executable, '-m', 'scar']
    _run(tmp_path, 'graph', [*scar, 'build', '--manifest', manifest, '--out', graph, '--jobs', '2'])
    artifacts = json.loads((graph / 'result.json').read_bytes())['artifacts']
    calls = read_artifact(artifacts['calls'])['calls']
    observation, = [call for call in calls if (call.get('callee') or {}).get('symbol') == 'observe']
    assert observation['status'] == 'BOUND' and observation['operation'] and observation['reachable']
    assert observation['callee']['unit'] == kernel.name
    evidence = read_artifact(artifacts['graph_evidence'])['units']
    caller_evidence, = [unit for unit in evidence if unit['unit'] == caller.name]
    typed = json.loads(read_artifact(caller_evidence['artifacts']['types'], json_document=False))
    gnu_header = _run(tmp_path, 'gnu-cfi-header', ['gfortran', '-print-file-name=include/ISO_Fortran_binding.h'])
    header = Path(gnu_header.decode().strip()).resolve(strict=True)
    assert typed['gnu_cfi'] == [{'path': str(header), 'sha256': file_reference(header, 'system-header')['sha256'],
                                'symbols': ['CFI_establish', 'CFI_is_contiguous', 'CFI_section']}]

    _run(tmp_path, 'plan', [*scar, 'plan', '--graph', graph, '--out', plan, '--repetitions', '1'])
    _run(tmp_path, 'delivery', [*scar, 'compile', '--plan', plan / 'compiler-plan.json',
                                '--out', delivery, '--repetitions', '1'])

    output = json.loads((delivery / 'result.json').read_bytes())
    executable = Path(output['executable']['path'])
    assert _run(tmp_path, 'standalone', [executable]) == EXPECTED
    assert b'libgfortran' not in _run(tmp_path, 'dependencies', ['readelf', '-d', executable])
    assert any(row['case'] == 'view-layouts' and row['matches'] for row in output['validation'])
    emission = json.loads((delivery / 'project/fortran-emission.json').read_bytes())
    descriptor = emission['c_exports'][0]['arguments'][0]['descriptor']
    assert descriptor['schema'] == 'scar.gnu-cfi-assumed-shape'
    assert descriptor['rank'] == 2 and descriptor['cfi_type'] == 'CFI_type_float'
    assert descriptor['lower_bounds'] == [-2, 3]

    for replacement, before, after, reason in (
            ('contiguous', 'intent(inout) :: a', 'intent(inout), contiguous :: a', 'CFI CONTIGUOUS'),
            ('dynamic-dimension', 'lbound(a, dim=1)', 'lbound(a, dim=size(a,2)-1)', 'default-integer constant'),
    ):
        edited = {kernel.name: FORTRAN.replace(before, after).encode()}
        with pytest.raises(ValueError, match=reason):
            emit_fortran({**request, 'units': [units[1]]}, tmp_path / replacement, edited, mixed=True)
