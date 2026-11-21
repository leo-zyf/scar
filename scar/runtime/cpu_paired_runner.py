"""Run admitted CPU trials in a small interpreter with no SCAR imports."""

import json
import os
from pathlib import Path
from shutil import copyfileobj
import subprocess
import sys
from time import perf_counter_ns


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


def run_arm(execution, run, transcript, *, collect_resources=False):
    (run / 'published').mkdir(parents=True)
    launch = _ResourceProcess if collect_resources else subprocess.Popen
    with (run / 'stdout.bin').open('xb') as out, (run / 'stderr.bin').open('xb') as err:
        runner_rss = _current_rss() if collect_resources else None
        started = perf_counter_ns()
        with launch(execution['argv'], cwd=execution['cwd'], env=execution['environment'],
                    stdin=subprocess.PIPE if transcript else subprocess.DEVNULL,
                    stdout=subprocess.PIPE if transcript else out, stderr=err) as process:
            if transcript:
                pace(process, out, transcript)
            status = process.wait()
        elapsed = perf_counter_ns() - started
    result = dict(elapsed_ns=elapsed, status=status)
    if collect_resources:
        usage = process.usage
        if usage is None:
            raise ValueError('missing resource usage from reaped launch')
        result['resources'] = dict(peak_rss_bytes=usage.ru_maxrss * 1024,
            user_cpu_ns=round(usage.ru_utime * 1_000_000) * 1000,
            system_cpu_ns=round(usage.ru_stime * 1_000_000) * 1000,
            runner_rss_before_launch_bytes=runner_rss)
    return result


if __name__ == '__main__':
    request = json.load(sys.stdin)
    results = [run_arm(execution, Path(directory), request['transcript'], collect_resources=True)
               for execution, directory in request['layout']]
    json.dump(results, sys.stdout, allow_nan=False)
