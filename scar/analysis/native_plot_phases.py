"""Reduce exclusive native plot timing cuts without mixing clean timings."""

import json
from statistics import median
import struct

from scar.ir.execution_recipe import read_artifact


SAMPLE = 'scar.plot-phase-sample'
TRACE = 'scar.native-plot-phase-jsonl'
PHASES = ('asset_read_verify', 'font_setup', 'coordinates_pack_guard', 'glyphs',
          'z_pack', 'contour_scene', 'agg_raster', 'png_encode_publish')
PNG_DETAIL_PHASES = ('png_filter_selection', 'png_deflate', 'png_zlib_lifecycle',
                     'png_temp_io', 'png_final_publish')


def _need(ok, why):
    if not ok:
        raise ValueError('native plot phases: ' + why)


def phases(schema_version):
    _need(type(schema_version) is int and schema_version in (1, 2), 'unsupported phase protocol')
    return PHASES if schema_version == 1 else PHASES + PNG_DETAIL_PHASES


def png_call_counts(data):
    """Read scanlines and original staged-write cadence from admitted PNG bytes."""
    _need(len(data) >= 33 and data[:8] == b'\x89PNG\r\n\x1a\n' and data[8:16] == b'\0\0\0\rIHDR',
          'invalid PNG witness header')
    height = struct.unpack_from('>I', data, 20)[0]
    _need(height > 0, 'invalid PNG height')
    # Prefix, IEND length/type/CRC, mkstemp, fdopen and fclose.
    calls, offset = 7, 8
    while offset + 12 <= len(data):
        size, kind = struct.unpack_from('>I4s', data, offset)
        offset += 12 + size
        _need(offset <= len(data), 'truncated PNG witness chunk')
        if kind == b'IDAT':
            calls += 3 + bool(size)
        elif kind == b'IEND':
            _need(size == 0 and offset == len(data) and calls > 7, 'invalid PNG witness termination')
            return height, calls
    raise ValueError('native plot phases: PNG witness has no IEND')


def episode(trace, backend, frames, *, schema_version=1, png_counts=()):
    """Replay one process interval. Individual phase counters are exclusive."""
    _need(backend in ('v1', 'v2') and type(frames) is int and frames > 0,
          'episode needs its exact ABI arm and frame count')
    rows = [json.loads(line) for line in read_artifact(trace, json_document=False).splitlines()]
    expected = [('render', frame) for frame in range(frames)]
    if backend == 'v2':
        expected = [('prepare', -1), *expected, ('destroy', -1)]
    _need([(row['operation'], row['frame']) for row in rows] == expected,
          'phase rows do not match one ordered controller lifetime')
    names = phases(schema_version)
    _need(schema_version == 1 or len(png_counts) == frames,
          'detailed PNG rows need the admitted frame bytes')
    values = dict.fromkeys((*names, 'residual'), 0)
    counts = dict.fromkeys(names, 0)
    walls = dict.fromkeys(('prepare', 'render', 'destroy'), 0)
    render_totals = []
    for row in rows:
        _need(row['schema'] == SAMPLE and row['schema_version'] == schema_version
              and row['backend'] == backend and row['sequence'] == 0 and row['success'] is True,
              'wrong phase protocol, arm, or failed render')
        _need(type(row['wall_ns']) is int and row['wall_ns'] >= 0,
              'invalid monotonic duration')
        covered = 0
        for phase in names:
            elapsed, calls = row[phase + '_ns'], row[phase + '_count']
            _need(type(elapsed) is int and elapsed >= 0 and type(calls) is int and calls >= 0
                  and (calls > 0 or elapsed == 0), 'invalid exclusive phase counter for ' + phase)
            covered += elapsed
            values[phase] += elapsed
            counts[phase] += calls
        if schema_version == 2:
            _check_png_counts(row, png_counts)
        _need(covered <= row['wall_ns'], 'exclusive cuts exceed their ABI call interval')
        values['residual'] += row['wall_ns'] - covered
        walls[row['operation']] += row['wall_ns']
        if row['operation'] == 'render':
            render_totals.append(row['wall_ns'])
    return dict(phases_ns=values, phase_counts=counts, operations_ns=walls,
                total_abi_ns=sum(walls.values()), per_frame_abi_ns=render_totals)


def _range(samples):
    return dict(min_ns=min(samples), median_ns=median(samples), max_ns=max(samples))


def _check_png_counts(row, png_counts):
    counters = {name: row[name + '_count'] for name in PNG_DETAIL_PHASES}
    if row['operation'] != 'render':
        _need(not any(counters.values()), 'PNG detail outside a render')
        return
    scanlines, temp_calls = png_counts[row['frame']]
    _need(counters['png_filter_selection'] == scanlines
          and counters['png_deflate'] > scanlines
          and counters['png_zlib_lifecycle'] == 2
          and counters['png_temp_io'] == temp_calls and counters['png_final_publish'] == 1,
          'PNG detail does not match the admitted RGBA encoder lifecycle')


def _comparison(old, new):
    return dict(v1=_range(old), v2=_range(new),
                v2_minus_v1=_range([b - a for a, b in zip(old, new, strict=True)]))


def compare(pairs, frames, *, schema_version=1, png_counts=()):
    """Report matched deltas as measured components under instrumentation."""
    _need(len(pairs) >= 2, 'compare at least two paired episodes')
    metrics = {side: [episode(pair[side], side, frames, schema_version=schema_version, png_counts=png_counts)
                      for pair in pairs] for side in ('v1', 'v2')}
    breakdown = {}
    for phase in (*phases(schema_version), 'residual'):
        old, new = [[row['phases_ns'][phase] for row in metrics[side]] for side in metrics]
        breakdown[phase] = _comparison(old, new)
    old, new = [[row['total_abi_ns'] for row in metrics[side]] for side in metrics]
    report = dict(episodes=metrics, breakdown_ns=breakdown, total_abi_ns=_comparison(old, new),
                  instrumented=True, whole_process_timing=False, deployment_authorized=False)
    if schema_version == 2:
        details = ('png_encode_publish', *PNG_DETAIL_PHASES)
        values = [[sum(row['phases_ns'][key] for key in details) for row in metrics[side]] for side in metrics]
        report['inclusive_png_ns'] = _comparison(*values)
    return report
