"""Measure a sealed launch proxy from a fresh stdlib-only sampler process."""

import json
import subprocess
import sys
from tempfile import TemporaryFile
from time import clock_gettime_ns, CLOCK_MONOTONIC


def _sample(request):
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


if __name__ == '__main__':
    request = json.load(sys.stdin)
    json.dump([_sample(request) for _ in range(request['count'])], sys.stdout, allow_nan=False)
