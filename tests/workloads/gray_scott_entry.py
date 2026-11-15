"""Deterministic validation entry for the external CPU Gray–Scott program.

This driver defines inputs/observable results only. It contains no SCAR
selection, transformation or function-name optimization logic.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default=str(Path.home()/'AAA/gray-scott/python'))
    parser.add_argument('--grid',type=int,default=512)
    parser.add_argument('--steps',type=int,default=1000)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    sys.path.insert(0,args.source)
    from gray_scott import GrayScott
    args.out.mkdir(parents=True,exist_ok=True)
    simulation=GrayScott(N=args.grid,second_order=False,movie=False,outdir=str(args.out))
    simulation.integrate(0,args.steps*simulation.dt,dump_freq=args.steps+1)
    files={str(path.relative_to(args.out)):sha256(path.read_bytes()).hexdigest()
        for path in sorted(args.out.rglob('*')) if path.is_file()}
    print(json.dumps(dict(grid=args.grid,requested_steps=args.steps,dt=simulation.dt,
        u_sha256=sha256(simulation.u.tobytes()).hexdigest(),v_sha256=sha256(simulation.v.tobytes()).hexdigest(),
        dump_count=simulation.dump_count,files=files),sort_keys=True))


if __name__=='__main__':main()
