"""Compile a proved workspace finite-property plan with the existing C++ ABI."""
from scar.compiler.reuse_workspace_guards import main as _main


def main(argv=None):
    return _main(argv, compile=True)


if __name__ == '__main__':
    raise SystemExit(main())
