"""Reduce exclusive native plot timing cuts without mixing clean timings."""

import json
from statistics import median

from scar.ir.execution_recipe import read_artifact


SAMPLE = 'scar.plot-phase-sample'
TRACE = 'scar.native-plot-phase-jsonl'
PHASES = ('asset_read_verify', 'font_setup', 'coordinates_pack_guard', 'glyphs',
          'z_pack', 'contour_scene', 'agg_raster', 'png_encode_publish')


def _need(ok, why):
    if not ok:
        raise ValueError('native plot phases: ' + why)


def episode(trace, backend, frames):
    """Replay one process interval. Individual phase counters are exclusive."""
    _need(backend in ('v1', 'v2') and type(frames) is int and frames > 0,
          'episode needs its exact ABI arm and frame count')
    rows = [json.loads(line) for line in read_artifact(trace, json_document=False).splitlines()]
    expected = [('render', frame) for frame in range(frames)]
    if backend == 'v2':
        expected = [('prepare', -1), *expected, ('destroy', -1)]
    _need([(row['operation'], row['frame']) for row in rows] == expected,
          'phase rows do not match one ordered controller lifetime')
    values = dict.fromkeys((*PHASES, 'residual'), 0)
    counts = dict.fromkeys(PHASES, 0)
    walls = dict.fromkeys(('prepare', 'render', 'destroy'), 0)
    render_totals = []
    for row in rows:
        _need(row['schema'] == SAMPLE and row['schema_version'] == 1
              and row['backend'] == backend and row['sequence'] == 0 and row['success'] is True,
              'wrong phase protocol, arm, or failed render')
        _need(type(row['wall_ns']) is int and row['wall_ns'] >= 0,
              'invalid monotonic duration')
        covered = 0
        for phase in PHASES:
            elapsed, calls = row[phase + '_ns'], row[phase + '_count']
            _need(type(elapsed) is int and elapsed >= 0 and type(calls) is int and calls >= 0
                  and (calls > 0 or elapsed == 0), 'invalid exclusive phase counter for ' + phase)
            covered += elapsed
            values[phase] += elapsed
            counts[phase] += calls
        _need(covered <= row['wall_ns'], 'exclusive cuts exceed their ABI call interval')
        values['residual'] += row['wall_ns'] - covered
        walls[row['operation']] += row['wall_ns']
        if row['operation'] == 'render':
            render_totals.append(row['wall_ns'])
    return dict(phases_ns=values, phase_counts=counts, operations_ns=walls,
                total_abi_ns=sum(walls.values()), per_frame_abi_ns=render_totals)


def _range(samples):
    return dict(min_ns=min(samples), median_ns=median(samples), max_ns=max(samples))


def compare(pairs, frames):
    """Report matched deltas as measured components under instrumentation."""
    metrics = {side: [episode(pair[side], side, frames) for pair in pairs] for side in ('v1', 'v2')}
    _need(len(pairs) >= 2, 'compare at least two paired episodes')
    phases = {}
    for phase in (*PHASES, 'residual'):
        old, new = [[row['phases_ns'][phase] for row in metrics[side]] for side in metrics]
        phases[phase] = dict(v1=_range(old), v2=_range(new),
                              v2_minus_v1=_range([b - a for a, b in zip(old, new, strict=True)]))
    old, new = [[row['total_abi_ns'] for row in metrics[side]] for side in metrics]
    return dict(episodes=metrics, breakdown_ns=phases, total_abi_ns=dict(v1=_range(old), v2=_range(new),
                v2_minus_v1=_range([b - a for a, b in zip(old, new, strict=True)])),
                instrumented=True, whole_process_timing=False, deployment_authorized=False)
