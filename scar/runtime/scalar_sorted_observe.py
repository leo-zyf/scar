"""Collect bounded source records at an exact eager sort boundary."""

import argparse
import ast
import builtins
from collections import Counter
from importlib import import_module
import json
from pathlib import Path
from threading import Lock
from types import FunctionType, GeneratorType

from scar.analysis import scalar_score_reuse_evidence as evidence
from scar.compiler.scalar_table_region import _definition, code_tree
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization import scalar_sorted_region as region
from scar.optimization.scalar_transition_ir import _flatten
from scar.runtime.scalar_table_contract import MISS, OFFER, compile_original
from scar.runtime.scalar_table_entry import _run_main
from scar.trace.code_snapshot import describe_code, site


_require = region._require


def _seal(plan, raw, recipe):
    root = compile_original(raw, recipe)
    owner = plan['sorted_owner']
    original = evidence._one((code for code in code_tree(root) if describe_code(code) == owner),
                             'source no longer contains the selected sorted owner')
    _require(all(OFFER not in (*code.co_names, *code.co_varnames, *code.co_freevars)
                 and MISS not in code.co_names for code in code_tree(root)), 'observer binding collides with source')
    selected = plan['sorted_site']
    _require(site(original, selected['offset']) == selected, 'selected sort instruction differs from source')
    tree = ast.parse(raw, filename=recipe['filename'])
    parent, _, name = owner['qualname'].rpartition('.<locals>.')
    caller = _definition(tree, parent)
    target = evidence._one((row for row in caller.body if type(row) is ast.FunctionDef and row.name == name),
                           'expected one original nested function')
    call = evidence._one((row for row in ast.walk(target) if type(row) is ast.Call
                           and evidence._span(row) == selected['position']), 'expected one exact sort expression')
    threshold = plan['predicate']['threshold']
    capture = plan['predicate']['nonzero_capture']
    call.args = [call.func, *call.args, ast.Name(id=plan['receiver'], ctx=ast.Load()),
                 ast.Name(id=threshold, ctx=ast.Load()),
                 ast.Name(id=capture, ctx=ast.Load()) if capture else ast.Constant(value=0)]
    call.func = ast.Name(id=OFFER, ctx=ast.Load())
    revised = compile(ast.fix_missing_locations(tree), recipe['filename'], 'exec',
                      dont_inherit=True, optimize=recipe['optimize'])
    code = evidence._one((code for code in code_tree(revised) if code.co_qualname == owner['qualname']
                          and code.co_firstlineno == owner['first_line']), 'observer lost the original sort owner')
    abi = ('co_flags', 'co_argcount', 'co_posonlyargcount', 'co_kwonlyargcount', 'co_varnames',
           'co_cellvars', 'co_freevars')
    _require(all(getattr(code, field) == getattr(original, field) for field in abi),
             'observer changed original function parameter or closure layout')
    return code


class _Observe:
    def __init__(self, plan, transition, limit):
        self.plan = plan
        self.ports = tuple((name, plan['state_fields'][name]) for name in plan['state_field_order'])
        self.versions = {label:{key:tuple(row) for key,row in version['rows'].items()}
                         for label,version in transition['table_versions'].items()}
        self.limit = limit
        self.samples = []
        self.counts = Counter()
        self.lock = Lock()
        self.expected_imports = {name:getattr(import_module(module), symbol)
                                 for name, (module, symbol) in plan['builtin_calls'].items()}

    def _identity(self, target, iterator, receiver):
        factory = self.plan['factory']
        scorer = self.plan['score']
        actor = type(receiver)
        _require(type(target) is type(builtins.sorted) and target is builtins.sorted
                 and type(iterator) is GeneratorType and describe_code(iterator.gi_code) == self.plan['record_source']
                 and type(actor) is type and issubclass(actor, tuple)
                 and actor._fields == tuple(name for name, _ in self.ports),
                 'unexpected original sort, iterator or receiver identity')
        source = iterator.gi_frame.f_locals.get('.0')
        _require(type(source) is GeneratorType and describe_code(source.gi_code) == factory,
                 'sort iterator did not receive the selected factory result')
        methods = [getattr(actor, proof['qualname'].rsplit('.', 1)[-1]) for proof in (factory, scorer)]
        _require(all(type(func) is FunctionType and describe_code(func.__code__) == proof
                     for func, proof in zip(methods, (factory, scorer), strict=True))
                 and methods[0].__globals__ is methods[1].__globals__, 'unexpected generator or scorer dispatch')
        namespace = methods[0].__globals__
        _require(all(namespace.get(name) is value for name, value in self.expected_imports.items())
                 and all(namespace.get(name, methods[0].__builtins__.get(name)) is value
                         for name, value in (('enumerate', builtins.enumerate), ('abs', builtins.abs))),
                 'generator standard intrinsic binding changed')
        _require(all(type(namespace.get(name)) is int and namespace[name] == value
                     for name, value in self.plan['scalar_globals'].items()), 'generator integer globals changed')
        for name, rows in self.plan['integer_rows'].items():
            actual = namespace.get(name)
            _require(type(actual) is dict and all(type(key) is str and type(row) is tuple
                     and all(type(item) is int for item in row) for key, row in actual.items())
                     and actual == {key:tuple(row) for key,row in rows.items()}, 'generator integer map changed')
        return namespace

    def _table(self, namespace):
        table = namespace[self.plan['table']]
        _require(type(table) is dict and all(type(name) is str and type(row) is tuple
                 and all(type(value) is int for value in row) for name,row in table.items()),
                 'source sort needs exact immutable scalar rows')
        return table

    def __call__(self, target, iterator, receiver, threshold, capture, *, reverse):
        if len(self.samples) >= self.limit * len(self.versions) * 2:
            return target(iterator, reverse=reverse)
        with self.lock:
            namespace = self._identity(target, iterator, receiver)
            before = self._table(namespace)
            anchored = before.copy()
            matches = [key for key, snapshot in self.versions.items() if snapshot == anchored]
            version = evidence._one(matches, 'sort activation has no exact score table version')
            _require(type(threshold) is int and type(capture) is int and reverse is self.plan['descending'],
                     'sort filter or order input differs from typed boundary')
            group = version, bool(capture)
            result = target(iterator, reverse=reverse)
            if self.counts[group] >= self.limit:
                return result
            after = self._table(namespace)
            _require(before is after and len(after) == len(anchored)
                     and all(after.get(key) is row for key, row in anchored.items()), 'table changed during observation')
            cls = namespace[self.plan['constructor']]
            _require(type(cls) is type and issubclass(cls, tuple) and list(cls._fields) == self.plan['record_fields'],
                     'sort record constructor changed')
            rows = []
            for pair in result:
                _require(type(pair) is tuple and len(pair) == 2 and type(pair[1]) is cls,
                         'sorted output contains an unexpected record type')
                rows.append([pair[0], *pair[1]])
            values = {}
            for i, (name, kind) in enumerate(self.ports):
                value = tuple.__getitem__(receiver, i)
                if kind == 'bool_pair':
                    _require(type(value) is tuple, 'boolean pair storage changed')
                    value = list(value)
                values[name] = value
            fields = dict(self.ports)
            flat = _flatten(fields, values)
            live_inputs = {port['name']:flat[port['name']] for port in self.plan['inputs']}
            self.samples.append(dict(table_version=version, threshold=threshold, nonzero_capture=capture,
                                     inputs=live_inputs, records=rows))
            self.counts[group] += 1
            return result


def collect(path, out, entries=4, argv=()):
    _require(type(entries) is int and 1 <= entries <= 16, 'capture limit must be 1–16 per version and filter branch')
    plan_ref = artifact_reference(Path(path).resolve(strict=True), region.SCHEMA)
    plan = read_artifact(plan_ref)
    _require(plan == region.project(plan['evidence']['uri'], plan['transition']['uri']),
             'capture plan differs from the original source')
    transition = read_artifact(plan['transition'])
    score_plan = read_artifact(plan['scorer'])
    score_evidence = read_artifact(score_plan['evidence'])
    recipe = read_artifact(score_evidence['proof'])['cases'][score_evidence['proof_case']]['compile_recipe']
    raw = read_artifact(plan['source'], json_document=False)
    code = _seal(plan, raw, recipe)
    observer = _Observe(plan, transition, entries)
    run = _run_main(Path(plan['source']['uri']), argv,
                    dict(compile_recipe=recipe, original_code_key=plan['sorted_owner']['key']), raw, code, observer)
    _require(observer.samples, 'original execution produced no selected sorted scopes')
    result = dict(schema=region.CAPTURE, schema_version=1, region=plan_ref, source=plan['source'],
                  sorted_site=plan['sorted_site'], factory=plan['factory'], execution=run,
                  samples=observer.samples, all_effects_closed=False)
    with Path(out).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--evidence', type=Path)
    parser.add_argument('--transition', type=Path)
    parser.add_argument('--limit', type=int, default=4)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('argv', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.plan and not (args.evidence or args.transition):
        argv = args.argv[1:] if args.argv[:1] == ['--'] else args.argv
        collect(args.plan, args.out, args.limit, argv)
    elif args.evidence and args.transition and not args.plan and not args.argv:
        plan = region.project(args.evidence, args.transition)
        with args.out.open('x', encoding='utf-8') as stream:
            json.dump(plan, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
    else:
        parser.error('provide --plan to capture, or --evidence and --transition to form the plan')


if __name__ == '__main__':
    main()
