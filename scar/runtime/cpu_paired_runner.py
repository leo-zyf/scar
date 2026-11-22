"""Run admitted CPU trials in a small interpreter with no SCAR imports."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
from shutil import copyfileobj
import subprocess
import sys
from time import perf_counter_ns


_CAPACITY_FLAG = '--exec-address-space-limit'
_TIMED_FIELD = re.compile(rb'(time:) +[0-9]+\.[0-9]+')
_SCIENTIFIC = re.compile(rb'[+-]?(?:[0-9]+\.[0-9]*|\.[0-9]+)[DEde][+-]?[0-9]+')


def compare_native_outputs(case, original, generated):
    """Apply the locked case's observable output contract."""
    if original.returncode or generated.returncode or original.stderr != generated.stderr:
        return False
    if original.stdout == generated.stdout:
        return True
    if case.get('comparison') == 'timed-fields':
        return (_TIMED_FIELD.sub(rb'\1 <timing>', original.stdout)
                == _TIMED_FIELD.sub(rb'\1 <timing>', generated.stdout))
    if case.get('comparison') == 'scientific-text':
        from decimal import Decimal

        def normalize(raw):
            labels = _SCIENTIFIC.sub(b'<decimal>', raw).split()
            values = [Decimal(value.upper().replace(b'D', b'E').decode())
                      for value in _SCIENTIFIC.findall(raw)]
            return labels, values

        return normalize(original.stdout) == normalize(generated.stdout)
    return False


class _ResourceProcess(subprocess.Popen):
    usage = None

    def wait(self, timeout=None):
        if timeout is not None:
            return super().wait(timeout)
        if self.returncode is None:
            # Each synchronous launch has one owner of the exit status. GNU
            # timeout remains inside argv and still bounds the workload.
            _, status, self.usage = os.wait4(self.pid, 0)
            self.returncode = os.waitstatus_to_exitcode(status)
        return self.returncode


def _current_rss():
    # The broker's lifetime high-water mark can itself include the compiler
    # before exec. VmRSS measures the broker's new address space at this point.
    with open('/proc/self/status', 'rb') as status:
        for line in status:
            if line.startswith(b'VmRSS:'):
                _, value, unit = line.split()
                if unit == b'kB' and int(value) > 0:
                    return int(value) * 1024
    raise OSError('could not read current CPU trial runner resident size')


def _check_limit(limit):
    if type(limit) is not int or not 0 < limit < 1 << 63:
        raise ValueError('candidate address-space limit must be a positive finite byte count')


def capacity_receipt(argv, runner_argv, limit, receipt_fd, *, native_capacity_guard_executable=None):
    _check_limit(limit)
    if (type(receipt_fd) is not int or receipt_fd <= 2 or len(argv) <= 7 or not os.path.isabs(argv[7])
            or argv[:2] != ['/usr/bin/taskset', '-c'] or argv[3:6] != ['/usr/bin/timeout', '-k', '2']):
        raise ValueError('candidate address-space guard requires the exact timed entry command and private receipt fd')
    launcher = runner_argv
    if native_capacity_guard_executable is not None:
        if type(native_capacity_guard_executable) is not str or not os.path.isabs(native_capacity_guard_executable):
            raise ValueError('native address-space guard requires an absolute executable path')
        launcher = [native_capacity_guard_executable]
    launch_argv = [*argv[:7], *launcher, _CAPACITY_FLAG, str(limit), str(receipt_fd), *argv[7:]]
    receipt = dict(method='LINUX_RLIMIT_AS_BEFORE_EXEC', limit_bytes=limit, soft_bytes=limit, hard_bytes=limit,
                   scope='PER_PROCESS_VIRTUAL_ADDRESS_SPACE', process_tree_aggregate_enforced=False,
                   may_raise_inherited_limits=False, launch_argv=launch_argv, receipt_fd=receipt_fd)
    if native_capacity_guard_executable is not None:
        receipt['implementation'] = 'NATIVE_CXX20'
    return receipt


@contextmanager
def _guarded_launch(execution, limit, native_capacity_guard_executable):
    if limit is None or execution['mode'] != 'candidate':
        yield None, None, None
        return
    if sys.platform != 'linux':
        raise ValueError('candidate address-space enforcement requires Linux')
    if native_capacity_guard_executable is None:
        raise ValueError('fresh address-space enforcement requires the native C++20 guard')
    reader, writer = os.pipe()
    with os.fdopen(reader, 'rb') as readback, os.fdopen(writer, 'wb') as receipt_stream:
        receipt = capacity_receipt(execution['argv'], (), limit, writer,
                                   native_capacity_guard_executable=native_capacity_guard_executable)
        yield receipt, readback, receipt_stream


def _limit_marker(soft, hard):
    return f'{soft} {hard}\n'.encode('ascii')


def pace(process, output, transcript, *, record=False):
    events, input_offset = [], 0
    for phase in transcript:
        payload = phase['line'].encode('ascii')
        process.stdin.write(payload)
        process.stdin.flush()
        ack_offset, ack_bytes = None, None
        marker = phase['acknowledgement']
        if marker is not None:
            expected = marker.encode('ascii')
            while line := process.stdout.readline():
                matched = line.startswith(expected)
                if record and matched:
                    ack_offset, ack_bytes = output.tell(), len(line)
                output.write(line)
                if matched:
                    break
            else:
                raise ValueError('CPU entry replay: process exited before ' + marker)
        if record:
            events.append(dict(input_offset=input_offset, input_bytes=len(payload),
                               ack_stdout_offset=ack_offset, ack_stdout_bytes=ack_bytes))
            input_offset += len(payload)
    process.stdin.close()
    copyfileobj(process.stdout, output)
    process.wait()
    return events


def run_arm(execution, run, transcript, *, collect_resources=False,
            candidate_address_space_limit_bytes=None, native_capacity_guard_executable=None):
    (run / 'published').mkdir(parents=True)
    launch = _ResourceProcess if collect_resources else subprocess.Popen
    with _guarded_launch(execution, candidate_address_space_limit_bytes,
                         native_capacity_guard_executable) as (receipt, readback, receipt_stream):
        command = execution['argv'] if receipt is None else receipt['launch_argv']
        pass_fds = () if receipt is None else (receipt['receipt_fd'],)
        with (run / 'stdout.bin').open('xb') as out, (run / 'stderr.bin').open('xb') as err:
            runner_rss = _current_rss() if collect_resources else None
            started = perf_counter_ns()
            with launch(command, cwd=execution['cwd'], env=execution['environment'], pass_fds=pass_fds,
                        stdin=subprocess.PIPE if transcript else subprocess.DEVNULL,
                        stdout=subprocess.PIPE if transcript else out, stderr=err) as process:
                if receipt_stream is not None:
                    receipt_stream.close()
                if transcript:
                    pace(process, out, transcript)
                status = process.wait()
            elapsed = perf_counter_ns() - started
        if receipt is not None and readback.read(128) != _limit_marker(receipt['soft_bytes'], receipt['hard_bytes']):
            raise ValueError('candidate address-space guard did not attest exact readback before exec')
    result = dict(elapsed_ns=elapsed, status=status)
    if collect_resources:
        usage = process.usage
        if usage is None:
            raise ValueError('missing resource usage from reaped launch')
        result['resources'] = dict(peak_rss_bytes=usage.ru_maxrss * 1024,
            user_cpu_ns=round(usage.ru_utime * 1_000_000) * 1000,
            system_cpu_ns=round(usage.ru_stime * 1_000_000) * 1000,
            runner_rss_before_launch_bytes=runner_rss)
    if receipt is not None:
        result['capacity_guard'] = receipt
    return result


if __name__ == '__main__':
    request = json.load(sys.stdin)
    limit = request.get('candidate_address_space_limit_bytes')
    native_guard = request.get('native_capacity_guard_executable')
    if limit is not None:
        _check_limit(limit)
        if native_guard is None:
            raise ValueError('fresh address-space enforcement requires the native C++20 guard')
    results = [run_arm(execution, Path(directory), request['transcript'], collect_resources=True,
                       candidate_address_space_limit_bytes=limit, native_capacity_guard_executable=native_guard)
               for execution, directory in request['layout']]
    json.dump(results, sys.stdout, allow_nan=False)
