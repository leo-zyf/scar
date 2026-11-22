"""SCAR's compiler and the tools used to inspect its input and output."""

import argparse
from functools import partial
from importlib import import_module
import sys


def _invoke(module, function, arguments):
    return getattr(import_module(module), function)(arguments)


def _has_option(arguments, option):
    if '--' in arguments:
        arguments = arguments[:arguments.index('--')]
    return any(argument == option or argument.startswith(option + '=') for argument in arguments)


def _build(arguments):
    if arguments in (['--help'], ['-h']):
        parser = argparse.ArgumentParser(prog='scar build',
                                         description='Build the program and hardware graph.')
        inputs = parser.add_mutually_exclusive_group(required=True)
        inputs.add_argument('--manifest', metavar='FILE', help='build a native target')
        inputs.add_argument('--entry', metavar='FILE', help='build a Python entry with optional trace evidence')
        inputs.add_argument('--formal', action='store_true', help='publish a Python formal source catalog')
        inputs.add_argument('--static', action='store_true', help='build a Python static source graph')
        parser.add_argument('--out', required=True)
        parser.epilog = 'Pass --help after selecting an input mode to see its options.'
        parser.parse_args(arguments)
        return 0
    if _has_option(arguments, '--manifest'):
        from scar.compiler.driver import main as build
        return build(['--graph-only', *arguments])
    if '--static' in arguments:
        from scar.analysis.static_program import main as build
        return build([argument for argument in arguments if argument != '--static'])
    if '--formal' in arguments:
        from scar.compiler.frontend.python import main as build
        return build([argument for argument in arguments if argument != '--formal'])
    from scar.analysis.build import main as build
    return build(arguments)


def _compile(arguments):
    if not arguments or arguments == ['--help'] or arguments == ['-h']:
        parser = argparse.ArgumentParser(prog='scar compile',
                                         description='Build, simplify and deliver a C++20 target.')
        inputs = parser.add_mutually_exclusive_group(required=True)
        inputs.add_argument('--manifest', help='locked native compilation units')
        inputs.add_argument('--domains', help='prepared Python program domains')
        inputs.add_argument('--plan', help='replay a sealed native execution plan')
        parser.add_argument('--out', required=True)
        parser.parse_args(arguments)
        return 0
    from scar.compiler.driver import main as compile_target
    return compile_target(arguments)


def _plan(arguments):
    if _has_option(arguments, '--build'):
        from scar.compiler.middle.python import main as plan
    elif _has_option(arguments, '--graph'):
        from scar.compiler.driver import plan_main as plan
    else:
        from scar.compiler.middle.plan import main as plan
    return plan(arguments)


def _run(arguments):
    if _has_option(arguments, '--pipeline'):
        from scar.runtime.compiled_entry import main as run
    else:
        from scar.runtime.entry_cli import main as run
    return run(arguments)


COMMANDS = {
    'trace': ('record evidence from a child program', partial(_invoke, 'scar.trace.cli', 'trace_main')),
    'import-capture': ('import retained trace evidence', partial(_invoke, 'scar.trace.cli', 'import_main')),
    'build': ('build the program and hardware graph', _build),
    'build-query': ('inspect a program graph', partial(_invoke, 'scar.analysis.build', 'query_main')),
    'check-program': ('validate a saved program graph',
                      partial(_invoke, 'scar.compiler.frontend.graph', 'check_main')),
    'plan': ('plan from a resource and dependency graph', _plan),
    'compile': ('build, simplify and deliver a target', _compile),
    'run': ('run a published execution contract', _run),
    'view-graph': ('view a program graph', partial(_invoke, 'scar.visualization.cli', 'main')),
}


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments:
        command, *options = arguments
        if command == 'static':
            return _build(['--static', *options]) or 0
        if command in COMMANDS:
            return COMMANDS[command][1](options) or 0
    parser = argparse.ArgumentParser(prog='scar', description=__doc__)
    commands = parser.add_subparsers(required=True)
    for command, (help_, _) in COMMANDS.items():
        commands.add_parser(command, help=help_)
    parser.parse_args(arguments)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
