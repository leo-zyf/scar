"""Export fully admitted loop artifacts for the compiler-independent runtime."""
import argparse
import json


def main():
    from .export_loop import export_loop
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact',required=True)
    parser.add_argument('--selection',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    result=export_loop(args.artifact,args.out,selection_path=args.selection)
    print(json.dumps(result))


if __name__=='__main__':main()
