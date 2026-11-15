"""Dispatch deployment before importing the graph/compiler command suite."""
import sys


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else list(argv)
    if arguments and arguments[0] == 'run':
        from scar.runtime.entry_cli import main as run
        return run(arguments[1:])
    from scar.cli import main as commands
    return commands(arguments)
