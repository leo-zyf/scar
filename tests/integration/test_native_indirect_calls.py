import json
from ctypes import CDLL
from hashlib import sha256
import shutil
import sys

import pytest

from test_native_stage_handoff import _run


_DISPATCH = r'''
#include <stdio.h>

extern int external_transform(int);
static int left(int x) { return x + 7; }
static int right(int x) { return x - 4; }
typedef int (*movement)(int);

int local(int n) {
    movement selected = &left;
    movement copy = selected;
    return copy(n);
}

int choice(int n, int flag) {
    movement f;
    if (flag) f = &right;
    else f = external_transform;
    return f(n);
}

int cyclic(int n) {
    movement f = &left;
    for (int i = 0; i < n; ++i)
        f = (i == 2) ? right : f;
    return f(n);
}

int unknown(int n, movement f) { return f(n); }

int main(void) {
    int n;
    if (scanf("%d", &n) != 1) return 2;
    printf("%d\n", local(n) + choice(n, n > 1) + cyclic(n) + unknown(n, external_transform));
    return 0;
}
'''


@pytest.fixture
def native_tools():
    tools = ('gcc', 'g++', 'cmake', 'ninja', 'nm', 'readelf', 'taskset')
    if any(shutil.which(tool) is None for tool in tools):
        pytest.skip('native indirect delivery requires GNU compilers and tools')
    try:
        CDLL('libclang-18.so.18')
    except OSError:
        pytest.skip('native indirect delivery requires libclang1-18')


def test_native_indirect_targets_flow_through_cfg_and_cpp20(tmp_path, native_tools):
    source = tmp_path / 'source'
    source.mkdir()
    provider = 'extern "C" int external_transform(int n) { return 3 * n; }\n'
    units, objects, locked = [], [], {}
    for name, language, standard, compiler, text in (
        ('dispatch.c', 'c', 'c17', 'gcc', _DISPATCH),
        ('provider.cpp', 'c++', 'c++17', 'g++', provider),
    ):
        path = source / name
        raw = text.encode('ascii')
        path.write_bytes(raw)
        locked[name] = dict(sha256=sha256(raw).hexdigest(), size_bytes=len(raw))
        flags = [f'-std={standard}']
        units.append(dict(path=name, language=language, flags=flags))
        obj = tmp_path / f'{name}.o'
        _run(tmp_path, f'original-{name}', [compiler, *flags, '-O2', '-c', path, '-o', obj])
        objects.append(obj)
    binary = tmp_path / 'original'
    _run(tmp_path, 'original-link', ['g++', *objects, '-o', binary])
    cases = []
    for label, argument, expected in (('small', b'1\n', b'22\n'), ('large', b'4\n', b'23\n')):
        assert _run(tmp_path, f'original-{label}', [binary], stdin=argument) == expected
        stdin = tmp_path / f'{label}.stdin'
        stdin.write_bytes(argument)
        cases.append(dict(id=label, stdin=str(stdin), comparison='exact'))

    manifest = tmp_path / 'request.json'
    manifest.write_text(json.dumps(dict(
        schema='scar.native-entry', schema_version=1, language='mixed', entry='main',
        source_root=str(source), units=units, source_lock=dict(files=locked), cases=cases,
        baseline=dict(argv=[str(binary)], cwd=str(tmp_path)),
    )))
    graph, plan, result = (tmp_path / name for name in ('graph', 'plan', 'delivery'))
    prefix = [sys.executable, '-m', 'scar']
    _run(tmp_path, 'scar-graph', [*prefix, 'build', '--manifest', manifest, '--out', graph, '--jobs', '2'])
    calls = json.loads((graph / 'call-links.json').read_bytes())
    binding = {row['caller']['symbol']: row for row in calls['calls']
               if row['caller']['symbol'] in ('local', 'choice', 'cyclic', 'unknown')}
    assert binding['local']['status'] == 'BOUND' and binding['local']['callee']['symbol'] == 'left'
    for name, expected in (('choice', {'right', 'external_transform'}), ('cyclic', {'left', 'right'})):
        call = binding[name]
        assert call['status'] == 'FINITE_TARGETS' and call['operation'] and call['reachable']
        assert {row['callee']['symbol'] for row in call['targets']} == expected
        assert any(row['kind'] == 'PHI' for row in call['target_flow']['definitions'].values())
    assert binding['unknown']['status'] == 'INDIRECT' and binding['unknown'] in calls['frontiers']
    assert 'target_flow' not in binding['unknown']

    _run(tmp_path, 'scar-plan', [*prefix, 'plan', '--graph', graph, '--out', plan, '--repetitions', '1'])
    _run(tmp_path, 'scar-delivery', [*prefix, 'compile', '--plan', plan / 'compiler-plan.json',
                                    '--out', result, '--repetitions', '1'])
    delivery = json.loads((result / 'result.json').read_bytes())
    assert delivery['delivery'] == 'COMPARED' and all(case['matches'] for case in delivery['validation'])
    project = json.loads((result / 'project/delivery.json').read_bytes())
    assert dict(caller='dispatch.c', callee='provider.cpp', symbol='external_transform') in project['mixed_linkage']


def test_native_entry_effects_close_over_bound_targets(tmp_path, native_tools):
    source = tmp_path / 'source'
    source.mkdir()
    caller = source / 'dispatch.c'
    raw = b'''
static int left(int x) { return x & 7; }
static int right(int x) { return x & 3; }
int main(int argc, char **argv) {
    int (*selected)(int) = &left;
    if (argc > 1) selected = &right;
    return selected(0);
}
'''
    caller.write_bytes(raw)
    binary = tmp_path / 'original'
    _run(tmp_path, 'original', ['gcc', '-std=c17', '-O2', caller, '-o', binary])
    manifest = tmp_path / 'request.json'
    manifest.write_text(json.dumps(dict(
        schema='scar.native-entry', schema_version=1, language='c', entry='main',
        source_root=str(source), units=[dict(path=caller.name, language='c', flags=['-std=c17'])],
        source_lock=dict(files={caller.name: dict(sha256=sha256(raw).hexdigest(), size_bytes=len(raw))}),
        cases=[dict(id='left', comparison='exact'),
               dict(id='right', argv=['right'], comparison='exact')],
        baseline=dict(argv=[str(binary)], cwd=str(tmp_path)),
    )))
    graph, plan, result = (tmp_path / label for label in ('graph', 'plan', 'delivery'))
    prefix = [sys.executable, '-m', 'scar']
    _run(tmp_path, 'scar-graph', [*prefix, 'build', '--manifest', manifest, '--out', graph])
    _run(tmp_path, 'scar-plan', [*prefix, 'plan', '--graph', graph, '--out', plan, '--repetitions', '1'])
    effects = json.loads((plan / 'rewrites.json').read_bytes())['entry_effects']
    assert effects['status'] == 'CLOSED_BODY', effects
    assert effects['normal_return_in_retained_graph'] is True and len(effects['regions']) == 3
    for region in effects['regions'].values():
        assert region['reachable_effects_closed'] is True and region['local_frontiers'] == {}
    calls = json.loads((graph / 'call-links.json').read_bytes())['calls']
    indirect, = [row for row in calls if row['status'] == 'FINITE_TARGETS']
    assert {row['callee']['symbol'] for row in indirect['targets']} == {'left', 'right'}
    _run(tmp_path, 'scar-delivery', [*prefix, 'compile', '--plan', plan / 'compiler-plan.json',
                                    '--out', result, '--repetitions', '1'])
    project = json.loads((result / 'project/delivery.json').read_bytes())
    assert project['entry_effect_scope'] == 'CLOSED_BODY'
    assert all(row['matches'] for row in json.loads((result / 'result.json').read_bytes())['validation'])
