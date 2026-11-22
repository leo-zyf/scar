"""SCAR's compiler and the tools used to inspect its input and output."""

import argparse
from importlib import import_module
import sys


COMMANDS = {
    'trace': ('scar.trace.cli', 'trace_main', 'record evidence from a child program'),
    'import-capture': ('scar.trace.cli', 'import_main', 'import retained trace evidence'),
    'build': (None, None, 'build the program and hardware graph'),
    'build-query': ('scar.analysis.build', 'query_main', 'inspect a program graph'),
    'check-program': ('scar.compiler.frontend.graph', 'check_main', 'validate a saved program graph'),
    'plan': (None, None, 'plan from a resource and dependency graph'),
    'compile': (None, None, 'build, simplify and deliver a target'),
    'run': (None, None, 'run a published execution contract'),
    'view-graph': ('scar.visualization.cli', 'main', 'view a program graph'),
}


def _has_option(arguments, option):
    if '--' in arguments:
        arguments = arguments[:arguments.index('--')]
    return any(argument == option or argument.startswith(option + '=') for argument in arguments)


def _build(arguments):
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
    else:
        from scar.compiler.middle.plan import main as plan
    return plan(arguments)


def _run(arguments):
    if _has_option(arguments, '--pipeline'):
        from scar.runtime.compiled_entry import main as run
    else:
        from scar.runtime.entry_cli import main as run
    return run(arguments)


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments:
        command, *options = arguments
        if command == 'compile':
            return _compile(options) or 0
        if command == 'build':
            return _build(options) or 0
        if command == 'static':
            return _build(['--static', *options]) or 0
        if command == 'plan':
            return _plan(options) or 0
        if command == 'run':
            return _run(options) or 0
        if command in COMMANDS:
            module, function, _ = COMMANDS[command]
            return getattr(import_module(module), function)(options) or 0
    parser = argparse.ArgumentParser(prog='scar', description=__doc__)
    commands = parser.add_subparsers(required=True)
    for command, (_, _, help_) in COMMANDS.items():
        commands.add_parser(command, help=help_)
    parser.parse_args(arguments)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
