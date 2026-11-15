"""Plain scientific entry, with no SCAR selection or instrumentation logic."""
import argparse
from hashlib import sha256
import json
import numpy as np
from simulation import simulate


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--size',type=int,default=512)
    parser.add_argument('--steps',type=int,default=1000)
    args=parser.parse_args();rng=np.random.default_rng(5)
    initial=rng.uniform(0,1,(args.size,args.size))
    fixed=rng.uniform(0,.01,(args.size,args.size))
    result=simulate(initial,fixed,args.steps)
    print(json.dumps(dict(shape=list(result.shape),steps=args.steps,
        result_sha256=sha256(result.tobytes()).hexdigest()),sort_keys=True))


if __name__=='__main__':main()
