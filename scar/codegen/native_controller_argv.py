"""Generate a native argv boundary from the reviewed source option routes."""

import argparse
import json
import math
from pathlib import Path
import re
import struct

from scar.ir.execution_recipe import read_artifact

HEADER = 'include/scar_generated/native_controller_argv.hpp'
_SCHEMA = 'scar.source-controller-argv'
_KINDS = {'int': 'Int', 'float64': 'Float', 'bool': 'Bool', 'str': 'String'}
_FLAG = re.compile(r'(?:--[A-Za-z][A-Za-z0-9_-]*|-[A-Za-z])\Z')
_PLACEHOLDER = 'SCAR_ENTRY_PROGRAM'


def _require(condition, detail):
    if not condition:
        raise ValueError('native controller argv: ' + detail)


def _quote(value):
    _require(type(value) is str and '\0' not in value, 'string is not valid in alternative argv')
    return json.dumps(value, ensure_ascii=False)


def _value(row):
    kind, value = row['kind'], row['value']
    _require(kind in _KINDS, 'literal outside the typed argv profile')
    if kind == 'bool':
        _require(type(value) is bool, 'expected original boolean literal')
        text = str(value).lower()
    elif kind == 'int':
        _require(type(value) is int and -(1 << 63) <= value < (1 << 63), 'original integer exceeds int64')
        text = str(value)
    elif kind == 'float64':
        _require(type(value) is float and math.isfinite(value), 'original float is not finite binary64')
        _require('binary64_be' not in row or row['binary64_be'] == struct.pack('>d', value).hex(),
                 'literal binary64 encoding differs')
        text = repr(value)
    else:
        _require(type(value) is str, 'expected original string literal')
        text = value
    return _KINDS[kind], _quote(text)


def _help(options, abbreviate):
    parser = argparse.ArgumentParser(prog=_PLACEHOLDER, allow_abbrev=abbreviate,
        formatter_class=lambda prog: argparse.HelpFormatter(prog, width=78))
    parser.color = False
    for dest, spec in options.items():
        keywords = dict(dest=dest, help=spec.get('help'))
        if spec['kind'] == 'bool':
            keywords['action'] = 'store_true'
        parser.add_argument(*spec['flags'], **keywords)
    return _quote(parser.format_help())


def _parse_options(recipe):
    route = recipe['route']
    _require(route['parse_method'] == 'parse_known_args' and route['unknown'] == 'DISCARD'
             and type(route['allow_abbrev']) is bool, 'requires an exact discard-only parse_known_args route')
    _require(route['positional_arguments'] is False, 'original positional actions are outside this profile')
    provider = read_artifact(route['provider'], json_document=False)
    _require(b"_negative_number_matcher = _re.compile(r'-\\.?\\d')" in provider,
             'original argparse uses another option token classifier')
    declarations = recipe['options']
    _require(type(declarations) is dict and declarations, 'original option table is missing')
    options = dict(sorted(declarations.items(), key=lambda row: row[1]['source_position']))
    flags, handlers, indices = {}, [], {}
    for dest, spec in options.items():
        supplied = spec['kind']
        _require(supplied in _KINDS and type(spec['flags']) is list and spec['flags'],
                 'unsupported option action or missing literal flags')
        for flag in spec['flags']:
            _require(type(flag) is str and _FLAG.fullmatch(flag) is not None and
                     flag not in flags and flag not in ('-h', '--help'), 'invalid or duplicate flag')
            flags[flag] = len(handlers)
        declared = spec['default']['kind']
        _require(declared == supplied or (supplied == 'float64' and declared == 'int'),
                 'option default changes the original conversion contract')
        initial_kind, initial = _value(spec['default'])
        indices[dest] = len(handlers)
        handlers.append(f'{{Kind::{_KINDS[supplied]}, {{Kind::{initial_kind}, {initial}}}}}')
    help_index = len(handlers)
    handlers.append('{Kind::Help, {Kind::Help, ""}}')
    flags.update({'-h': help_index, '--help': help_index})
    return options, flags, handlers, indices, help_index


def _source_expression(source, indices, options):
    if source['kind'] == 'OPTION':
        identifier = source['id']
        _require(identifier in indices, 'arg binding uses an undeclared source option')
        return f'values[{indices[identifier]}]', options[identifier]['kind']
    _require(source['kind'] == 'LITERAL', 'arg binding has no original typed value')
    kind, value = _value(source['value'])
    return f'ParsedValue{{Kind::{kind}, {value}}}', source['value']['kind']


def render(recipe, *, slot_arguments, numeric_ports, directory_argument):
    """Lower the options into a same-process C++20 controller ingress."""
    _require(recipe['schema'] == _SCHEMA and recipe['schema_version'] == 1,
             'expected a source-bound option route')
    options, flags, handlers, indices, help_index = _parse_options(recipe)
    branch = recipe['branch']
    branch_name = branch['option']
    _require(branch_name in indices and options[branch_name]['kind'] == 'bool' and
             branch['normal_value'] is False and branch['other_status'] == 'UNADMITTED',
             'requires a single admitted branch and explicit retained alternate')
    raw = recipe['bindings']
    bindings = {(row['target']['kind'], row['target']['id']): row['source'] for row in raw}
    _require(len(raw) == len(bindings), 'duplicate native target')
    requests = [('CONTROLLER_SLOT', label) for label in slot_arguments] + [
        ('NUMERIC_PORT', label) for label in numeric_ports]
    directory = ('EFFECT_DIRECTORY', directory_argument)
    _require(len(set(requests + [directory])) == len(requests) + 1 and
             set(bindings) == set(requests) | {directory}, 'source routes do not cover the native binding ports')
    directory_value, kind = _source_expression(bindings[directory], indices, options)
    _require(kind == 'str', 'directory requires a source string')
    entries = []
    for target in requests:
        expression, kind = _source_expression(bindings[target], indices, options)
        admitted = {'int', 'float64', 'bool'} if target[0] == 'CONTROLLER_SLOT' else {'int', 'float64'}
        _require(kind in admitted, 'source value does not match native scalar target')
        entries.append(f'    append(arguments, {_quote(target[1])}, {expression});')
    branch_index = indices[branch_name]
    branch_flag = next((flag for flag in options[branch_name]['flags'] if flag.startswith('--')),
                       options[branch_name]['flags'][0])
    branch_error = _quote(f'{branch_flag} selects a branch outside the native product')
    abbreviation = recipe['route']['allow_abbrev']
    replacements = {
        'OPTION_COUNT': len(handlers), 'OPTIONS': ', '.join(handlers),
        'FLAG_COUNT': len(flags), 'FLAGS': ', '.join(f'{{{_quote(flag)}, {index}}}' for flag, index in flags.items()),
        'HELP_INDEX': help_index, 'ABBREVIATE': str(abbreviation).lower(),
        'HELP_TEXT': _help(options, abbreviation),
        'BRANCH_GUARD': f'    if (values[{branch_index}].text != "false")\n'
                        f'        throw ScarPublicArgvError({branch_error});',
        'DIRECTORY_NAME': _quote(directory_argument), 'DIRECTORY_VALUE': directory_value,
        'BINDINGS': '\n'.join(entries),
    }
    template = Path(__file__).with_name('native_controller_argv_v1.hpp.in').read_text()
    return re.sub(r'@([A-Z_]+)@', lambda match: str(replacements[match[1]]), template).encode()
