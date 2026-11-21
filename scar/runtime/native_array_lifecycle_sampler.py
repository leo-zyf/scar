"""Collect private lifecycle clocks without importing SCAR or the workload."""

import json
import os
from pathlib import Path
import subprocess
import sys
from time import clock_gettime_ns, CLOCK_MONOTONIC


def _sample(execution, directory):
    directory = Path(directory)
    (directory / 'published').mkdir(parents=True, exist_ok=False)
    with (directory/'stdout.bin').open('xb') as stdout, (directory/'stderr.bin').open('xb') as stderr, \
         (directory/'markers.bin').open('xb') as markers, (directory/'capacity.bin').open('xb') as capacity:
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
    json.dump([_sample(execution, directory) for execution, directory in request], sys.stdout, allow_nan=False)
