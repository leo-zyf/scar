"""Ordinary scientific entry; SCAR attaches at its imported function boundary."""
import argparse
from hashlib import sha256
import json
import numpy as np
from licm_program import simulate


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--size',type=int,default=256)
    parser.add_argument('--steps',type=int,default=400)
    args=parser.parse_args()
    rng=np.random.default_rng(7)
    initial=rng.uniform(0,1,(args.size,args.size))
    fixed=rng.uniform(0,.01,(args.size,args.size))
    result=simulate(initial,fixed,args.steps)
    print(json.dumps(dict(shape=list(result.shape),steps=args.steps,
        result_sha256=sha256(result.tobytes()).hexdigest()),sort_keys=True))


if __name__=='__main__':
    main()
