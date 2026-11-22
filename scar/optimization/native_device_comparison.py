"""Reduce an explicitly admitted CPU and CUDA native-file comparison."""

from hashlib import sha256
import json
from math import prod
import re

from scar.optimization.paired_entry_selection import DEFAULT_MINIMUM_GAIN, compare_entry_pairs


SCHEMA = 'scar.native-device-entry-decision'
_MODES = ('CPU', 'CUDA')
_PROBE_FIELDS = {'returncode', 'stdout', 'stderr'}
_RUN_FIELDS = {*_PROBE_FIELDS, 'pair', 'mode', 'wall_ns', 'outputs'}
_RECEIPT_FIELDS = {'sha256', 'size_bytes'}
_DIGEST = re.compile('[0-9a-f]{64}')
_EMPTY_SHA256 = sha256(b'').hexdigest()


def _stdio(row):
    return type(row['returncode']) is int and type(row['stdout']) is str and type(row['stderr']) is str


def _device_unavailable(row):
    return (row['returncode'] == 77 and row['stdout'] == ''
            and row['stderr'].startswith('CUDA device unavailable: '))


def native_device_probe_reason(probe, *, device):
    """Qualify the archived output of the candidate launcher’s device probe."""
    if (type(device) is not int or not 0 <= device < 2**31 or type(probe) is not dict
            or probe.keys() != _PROBE_FIELDS or not _stdio(probe)):
        raise ValueError('native CUDA comparison requires a device ordinal and a probe receipt')
    if _device_unavailable(probe):
        return 'CUDA_DEVICE_UNAVAILABLE'
    if (probe['returncode'] != 0 or probe['stderr'] != ''
            or probe['stdout'] != f'{{"device":{device},"available":true}}\n'):
        return 'CUDA_PROBE_FAILED'
    return None


def _receipt(row):
    return (type(row) is dict and row.keys() == _RECEIPT_FIELDS
            and type(row['sha256']) is str and _DIGEST.fullmatch(row['sha256']) is not None
            and type(row['size_bytes']) is int and 0 <= row['size_bytes'] < 2**63)


def _output_contract(references, output_shapes, byte_order):
    if (type(references) not in (list, tuple) or not references
            or type(output_shapes) not in (list, tuple) or len(references) != len(output_shapes)
            or byte_order not in ('little', 'big')):
        raise ValueError('native device comparison requires an ordered reference for each FP32 output')
    outputs = []
    for reference, shape in zip(references, output_shapes, strict=True):
        if (type(shape) not in (list, tuple) or len(shape) != 2
                or any(type(extent) is not int or not 0 < extent < 2**31 for extent in shape)
                or not _receipt(reference) or reference['size_bytes'] != 4 * prod(shape)):
            raise ValueError('native reference extent differs from the admitted FP32 output shape')
        outputs.append(dict(shape=list(shape), nbytes=reference['size_bytes']))
    return json.dumps(dict(abi_version=1, dtype='float32', byte_order=byte_order, outputs=outputs),
                      separators=(',', ':')) + '\n'


def _run_reason(row, references, cpu_stdout):
    if (type(row) is not dict or row.keys() != _RUN_FIELDS or not _stdio(row)
            or type(row['pair']) is not int or row['pair'] < 0
            or type(row['mode']) is not str or row['mode'] not in _MODES
            or type(row['wall_ns']) is not int or row['wall_ns'] <= 0
            or type(row['outputs']) is not list or len(row['outputs']) != len(references)
            or any(value is not None and not _receipt(value) for value in row['outputs'])):
        raise ValueError('invalid native device full-entry attempt')
    mode = row['mode']
    if mode == 'CUDA' and _device_unavailable(row):
        return 'CUDA_DEVICE_BECAME_UNAVAILABLE'
    if row['returncode'] != 0:
        return mode + '_EXECUTION_FAILED'
    if row['outputs'] != list(references):
        return mode + '_OUTPUT_MISMATCH'
    if row['stderr'] or row['stdout'] != (cpu_stdout if mode == 'CPU' else ''):
        return mode + '_FILE_ABI_MISMATCH'
    return None


def native_device_run_reason(row, *, references, output_shapes, byte_order):
    """Return the terminal reason, or accept a fully matched file-ABI attempt."""
    return _run_reason(row, references, _output_contract(references, output_shapes, byte_order))


def compare_native_device(probe, runs, *, references, output_shapes, device, byte_order,
                          warmup_pairs, measurement_pairs, minimum_gain=DEFAULT_MINIMUM_GAIN,
                          strict_minimum_gain=True, runtime_qualified=True):
    """Verify probe, order, streams and file receipts before comparing real samples."""
    cpu_stdout = _output_contract(references, output_shapes, byte_order)
    if (type(runs) is not list or type(warmup_pairs) is not int or warmup_pairs < 0
            or type(measurement_pairs) is not int or measurement_pairs < 3
            or type(minimum_gain) not in (int, float) or not 0 <= minimum_gain < 1
            or type(strict_minimum_gain) is not bool or type(runtime_qualified) is not bool):
        raise ValueError('invalid native device comparison settings')
    reason = native_device_probe_reason(probe, device=device)
    if reason is None and not runtime_qualified:
        reason = 'UNQUALIFIED_CUDA_CONTEXT'
    decision = dict(schema=SCHEMA, schema_version=1, verdict='KEEP_INCUMBENT', default_backend='CPU',
                    deployment_authorized=False, timing_analysis=None, performance_measured=False,
                    bitwise_outputs_verified=False, reasons=[])
    if reason is not None:
        if runs:
            raise ValueError('native device comparison contains attempts after preflight rejection')
        return dict(decision, reasons=[reason])

    expected_runs = 2 * (warmup_pairs + measurement_pairs)
    if len(runs) > expected_runs:
        raise ValueError('native device comparison exceeds its declared matched schedule')
    matched = []
    # Only the checked file ABI is shared. The CPU run emits shape metadata,
    # while the CUDA trial emits no successful stdout.
    file_abi_digest = sha256(json.dumps(list(references), sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    for index, row in enumerate(runs):
        reason = _run_reason(row, references, cpu_stdout)
        pair, position = divmod(index, 2)
        mode = _MODES[(pair + position) % 2]
        if (row['pair'], row['mode']) != (pair, mode):
            raise ValueError('native device attempts differ from the declared alternating pair order')
        if reason is not None:
            if index != len(runs) - 1:
                raise ValueError('native device comparison continued after a terminal trial failure')
            return dict(decision, reasons=[reason])
        matched.append(dict(pair=pair, mode=mode, wall_ns=row['wall_ns'], returncode=0,
                            stdout_sha256=file_abi_digest, stderr_sha256=_EMPTY_SHA256))
    if len(matched) != expected_runs:
        raise ValueError('native device comparison is missing successful matched attempts')
    result = compare_entry_pairs(matched, incumbent='CPU', challenger='CUDA', warmup_pairs=warmup_pairs,
                                 measurement_pairs=measurement_pairs, minimum_gain=minimum_gain,
                                 strict_minimum_gain=strict_minimum_gain)
    return dict(decision, verdict=result['verdict'], reasons=result['reasons'], timing_analysis=result,
                performance_measured=True, bitwise_outputs_verified=True)
