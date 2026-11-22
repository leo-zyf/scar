"""Collect CPU process clocks from an isolated stdlib-only interpreter."""

import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryFile
from time import CLOCK_MONOTONIC, clock_gettime_ns


def _launch(request):
    paced = request['abi'] == 'PACED_STDIN'
    with TemporaryFile() as output, TemporaryFile() as errors:
        started = clock_gettime_ns(CLOCK_MONOTONIC)
        with subprocess.Popen(request['argv'], cwd=request['cwd'], env=request['environment'],
                stdin=subprocess.PIPE if paced else subprocess.DEVNULL,
                stdout=subprocess.PIPE if paced else output, stderr=errors) as process:
            if paced:
                process.stdin.write(b'SCAR_PROBE_EXIT\n')
                process.stdin.flush()
                acknowledgement = process.stdout.readline()
                process.stdin.close()
                raw = acknowledgement + process.stdout.read()
            status = process.wait()
        reaped = clock_gettime_ns(CLOCK_MONOTONIC)
        if not paced:
            output.seek(0)
            raw = output.read()
        errors.seek(0)
        if status != 0 or errors.read(1):
            raise RuntimeError('launch proxy failed or emitted diagnostics')
    return dict(started_ns=started, reaped_ns=reaped, output=raw.decode('ascii'))


def _lifecycle(execution, directory):
    directory = Path(directory)
    (directory / 'published').mkdir(parents=True, exist_ok=False)
    with ((directory/'stdout.bin').open('xb') as stdout, (directory/'stderr.bin').open('xb') as stderr,
          (directory/'markers.bin').open('xb') as markers, (directory/'capacity.bin').open('xb') as capacity):
        os.dup2(markers.fileno(), 198)
        os.dup2(capacity.fileno(), 199)
        try:
            started = clock_gettime_ns(CLOCK_MONOTONIC)
            result = subprocess.run(execution['argv'], cwd=execution['cwd'], env=execution['environment'],
                stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, pass_fds=(198, 199))
            reaped = clock_gettime_ns(CLOCK_MONOTONIC)
        finally:
            os.close(198)
            os.close(199)
    return dict(started_ns=started, reaped_ns=reaped, status=result.returncode)


if __name__ == '__main__':
    request = json.load(sys.stdin)
    if type(request) is dict:
        samples = [_launch(request) for _ in range(request['count'])]
    else:
        samples = [_lifecycle(execution, directory) for execution, directory in request]
    json.dump(samples, sys.stdout, allow_nan=False)
