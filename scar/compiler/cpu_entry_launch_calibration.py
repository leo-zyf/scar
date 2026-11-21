"""Measure or replay an independent C++20 CPU process proxy."""

import argparse
import json
from pathlib import Path

from scar.codegen import cpu_launch_probe


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--out', type=Path)
    action.add_argument('--check', type=Path)
    parser.add_argument('--cpu-ids', type=lambda value: [int(cpu) for cpu in value.split(',')])
    parser.add_argument('--cwd', type=Path)
    parser.add_argument('--environment', type=Path, help='JSON object with the exact launch environment')
    parser.add_argument('--abi', choices=('ARGV', 'PACED_STDIN'), default='ARGV')
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--samples', type=int, default=5)
    args = parser.parse_args(argv)
    if args.check is not None:
        result = cpu_launch_probe.check(args.check)
    else:
        if args.cpu_ids is None or args.cwd is None or args.environment is None:
            parser.error('--out requires --cpu-ids, --cwd and --environment')
        result = cpu_launch_probe.measure(args.out, cpu_ids=args.cpu_ids, cwd=args.cwd,
            environment=json.loads(args.environment.read_bytes()), abi=args.abi,
            warmups=args.warmups, samples=args.samples)
    print(json.dumps(result, sort_keys=True))
