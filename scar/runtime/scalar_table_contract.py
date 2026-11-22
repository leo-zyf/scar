"""Scalar table contract."""

from types import CodeType, BuiltinFunctionType, FunctionType
from scar.trace.code_snapshot import code_key
import argparse
import builtins
import json
import marshal
import runpy
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.runtime.csr_region_entry import _observer_active, interpreter_context
from scar.runtime.entry_runner import _arguments, _execute, _file, _script_state


# Shared code artifact contract for scalar table compilation and loading.
PROOF = 'scar.scalar-table-cse'
SCHEMA = 'scar.scalar-table-region'
ENTRY_CODE = 'scar.python-scalar-table-region-code'
OFFER = '_scar_scalar_table_offer_v1'
MISS = '_scar_scalar_table_miss_v1'
RESULT = '_scar_scalar_table_result_v1'


def compile_original(raw, recipe):
    if (set(recipe) != {'filename', 'mode', 'flags', 'dont_inherit', 'optimize'}
            or recipe['mode'] != 'exec' or recipe['flags'] != 0 or recipe['dont_inherit'] is not True
            or type(recipe['optimize']) is not int or recipe['optimize'] not in (0, 1, 2)):
        raise ValueError('scalar table region requires an ordinary original Python source compilation')
    return compile(raw, recipe['filename'], 'exec', dont_inherit=True, optimize=recipe['optimize'])


def replace_leaf(root, original_key, revised):
    """Change one leaf while retaining the original module and sibling bytecode."""
    count = 0

    def splice(code):
        nonlocal count
        if code_key(code) == original_key:
            count += 1
            if code.co_qualname != revised.co_qualname or code.co_filename != revised.co_filename:
                raise ValueError('revised code does not belong to the original function')
            return revised
        updated = tuple(splice(child) if type(child) is CodeType else child for child in code.co_consts)
        return code if all(a is b for a, b in zip(code.co_consts, updated, strict=True)) else code.replace(co_consts=updated)

    combined = splice(root)
    if count != 1 or combined.co_code != root.co_code:
        raise ValueError('original main code is not a unique leaf splice')
    return combined


# Run an explicitly selected scalar table splice in its original script or import.
EXECUTION = 'scar.scalar-table-entry-execution'


def _require(condition, detail):
    if not condition:
        raise ValueError(detail)


def _pinned(path, schema, expected):
    reference = artifact_reference(Path(path).resolve(), schema)
    _require(type(expected) is str and reference['sha256'] == expected, 'selected artifact SHA-256 differs')
    return reference


def _admitted(bundle):
    report = read_artifact(bundle)
    _require(bundle['schema'] == SCHEMA and report['schema_version'] == 1
        and report['deployment_authorized'] is False and report['interpreter'] == interpreter_context()
        and report['execution_scope'] == 'EXPLICIT_EXCLUSIVE_EXPRESSION_STUDY'
        and report['proof']['schema'] == PROOF and report['code']['schema'] == ENTRY_CODE,
        'sealed scalar table study requires its original interpreter and source proof')
    source_proof = read_artifact(report['proof'])
    case = source_proof['cases'][report['proof_case']]
    region = report['region']
    _require(case['disposition'] == 'CONDITIONAL_REWRITE' and case['automatic_admission'] is False
        and case['operation'] == region['operation'] == 'Sub' and case['code_key'] == region['original_code_key']
        and case['source'] == region['source'] and case['qualname'] == region['qualname']
        and case['compile_recipe'] == region['compile_recipe'] and case['span'] == region['span'],
        'sealed scalar table original condition differs from its retained proof')
    raw = read_artifact(region['source'], json_document=False)
    leaf = marshal.loads(read_artifact(report['code'], json_document=False))
    _require(type(leaf) is CodeType and code_key(leaf) == region['revised_code_key']
        and not leaf.co_freevars and not leaf.co_cellvars and leaf.co_filename == region['source']['uri']
        and leaf.co_qualname == region['qualname'] and {OFFER, MISS} <= set(leaf.co_names),
        'sealed scalar table splice is not its original derived function')
    return region, raw, leaf


class _ImportScope:
    def __init__(self, region, leaf, offer):
        self.source = Path(region['source']['uri'])
        self.region, self.leaf, self.offer = region, leaf, offer
        self.bindings, self.frontier = [], []

    def attach(self, module, name, *, natural):
        if self.bindings:
            self.frontier.append('ALREADY_BOUND_ORIGINAL_SOURCE')
            return
        spec = module.__spec__
        if _file(module) != self.source or _observer_active() or (not natural and (
                spec is None or getattr(spec, '_initializing', False) or type(spec.loader) is not SourceFileLoader)):
            self.frontier.append('EXACT_STANDARD_MODULE_OR_OBSERVER_GUARD_REJECTED')
            return
        namespace = vars(module)
        parts = self.region['qualname'].split('.')
        owner = module
        for part in parts[:-1]:
            owner = vars(owner).get(part)
            if type(owner) is not type:
                self.frontier.append('SOURCE_CLASS_OWNER_CHANGED')
                return
        original = vars(owner).get(parts[-1])
        if (type(original) is not FunctionType or original.__globals__ is not namespace
                or original.__qualname__ != self.region['qualname']
                or code_key(original.__code__) != self.region['original_code_key']
                or not _exact_names(namespace) or not _exact_names(original.__builtins__)
                or OFFER in namespace or MISS in namespace):
            self.frontier.append('ORIGINAL_CALLABLE_OR_GLOBALS_GUARD_REJECTED')
            return
        saved = original.__code__
        namespace[OFFER], namespace[MISS] = self.offer, NotImplemented
        original.__code__ = self.leaf
        self.bindings.append(dict(namespace=namespace, original=original, saved=saved, module_name=name,
            natural_import=natural, stats=dict(native_calls=0, fallback_calls=0, reasons={})))

    def finish(self):
        for binding in self.bindings:
            original, namespace = binding['original'], binding['namespace']
            if original.__code__ is self.leaf:
                original.__code__ = binding['saved']
            if namespace.get(OFFER) is self.offer:
                del namespace[OFFER]
            if namespace.get(MISS) is NotImplemented:
                del namespace[MISS]


def _run_main(entry, argv, region, raw, leaf, offer, *, original_root=None):
    if original_root is None:
        root = compile_original(raw, region['compile_recipe'])
    else:
        _require(type(original_root) is CodeType and original_root.co_name == '<module>'
                 and original_root.co_filename == region['compile_recipe']['filename'],
                 'prepared study source must be its original module code')
        root = original_root
    code = replace_leaf(root, region['original_code_key'], leaf)
    exited = False
    with _script_state(entry, argv, None):
        try:
            # Preserve the run_path script's temporary __main__, __file__,
            # argv and package metadata while supplying the exact code tree.
            runpy._run_module_code(code, {OFFER:offer, MISS:NotImplemented}, mod_name='__main__',
                pkg_name='', script_name=str(entry))
        except SystemExit as error:
            if error.code is not None and not (isinstance(error.code, int) and int.__int__(error.code) == 0):
                raise
            exited = True
    return dict(entry_path=str(entry), argv=list(argv), installed_bindings=1, frontier=[],
        system_exit_zero=exited, exit_code=0, attachments=[dict(source=str(entry), module='__main__', natural_import=False)])


def _exact_names(namespace):
    return type(namespace) is dict and all(type(key) is str for key in namespace)


def run_trial(entry, argv, *, bundle=None, bundle_sha256=None, native=None, native_sha256=None,
              exclusive_expression=False, observe=False):
    """Use a pinned study under an explicit no-writer/no-observer contract.

    The owner keeps the original namespace lookups ordinary, reserved names
    untouched and the selected expression isolated from observers and writes.
    Native applicability is independently checked on each live input.
    """
    entry, argv = _arguments(entry, argv)
    no_op = None
    if exclusive_expression is not True:
        no_op = 'NO_EXPLICIT_EXPRESSION_EFFECT_CONTRACT'
    elif sys.implementation.name != 'cpython' or not getattr(sys, '_is_gil_enabled', lambda: True)():
        no_op = 'ORDINARY_CPYTHON_GIL_REQUIRED'
    elif _observer_active():
        no_op = 'ACTIVE_OBSERVER'
    elif not _exact_names(vars(builtins)):
        no_op = 'BUILTIN_NAME_RESOLUTION_MAY_CALL_USER_CODE'
    if no_op is not None:
        original = _execute(entry, argv, None, no_op)
        return dict(original, schema=EXECUTION, mode='ORIGINAL', deployed=False)

    from scar.codegen.scalar_table_native import SCHEMA as NATIVE, load_scalar_table

    reference = _pinned(bundle, SCHEMA, bundle_sha256)
    provider = _pinned(native, NATIVE, native_sha256)
    region, raw, leaf = _admitted(reference)
    action = load_scalar_table(provider).subtract
    _require(type(action) is BuiltinFunctionType, 'native scalar table offer must be its C callable')
    counts = dict(native_calls=0, fallback_calls=0) if observe else None

    if counts is None:
        offer = action
    else:
        def offer(table, key, left, right):
            result = action(table, key, left, right)
            counts['fallback_calls' if result is NotImplemented else 'native_calls'] += 1
            return result

    if entry == Path(region['source']['uri']):
        execution = _run_main(entry, argv, region, raw, leaf, offer)
        mode = 'SEALED_MAIN_LEAF'
    else:
        scope = _ImportScope(region, leaf, offer)
        execution = _execute(entry, argv, scope, 'explicit_scalar_table_study')
        mode = 'SEALED_STANDARD_IMPORT' if execution['installed_bindings'] else 'ORIGINAL'
    return dict(execution, schema=EXECUTION, schema_version=1, mode=mode, bundle=reference, native=provider,
        statistics=counts, observed=observe, deployed=False, performance_qualified=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle')
    parser.add_argument('--bundle-sha256')
    parser.add_argument('--native')
    parser.add_argument('--native-sha256')
    parser.add_argument('--exclusive-expression', action='store_true')
    parser.add_argument('--observe', action='store_true')
    parser.add_argument('--report', type=Path)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('original script required after --')
    if args.exclusive_expression and not all((args.bundle, args.bundle_sha256, args.native, args.native_sha256)):
        parser.error('explicit study requires source and native receipts with byte identities')
    if args.report is not None and args.report.exists():
        parser.error('fresh report path required')
    report = run_trial(command[0], command[1:], bundle=args.bundle, bundle_sha256=args.bundle_sha256,
        native=args.native, native_sha256=args.native_sha256, exclusive_expression=args.exclusive_expression,
        observe=args.observe)
    if args.report is not None:
        with args.report.open('x') as stream:
            json.dump(report, stream, sort_keys=True, indent=2)
            stream.write('\n')


if __name__ == '__main__':
    main()
