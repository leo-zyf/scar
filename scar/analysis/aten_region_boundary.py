"""Identify closed Python boundaries for native ATen trial candidates."""

import ast
from hashlib import sha256
from pathlib import Path

from scar.analysis.aten_python_trial_source import _function, _wrappers, qualify_python_trial_source
from scar.analysis.aten_source_calls import (
    AtenSourceMatchError, original_aten_call_bindings, original_aten_call_index, original_aten_calls,
)
from scar.ir.record_codec import encode


class AtenRegionBoundaryError(ValueError):
    """A candidate cannot represent the entire original function."""

    def __init__(self, reason, detail):
        super().__init__(detail)
        self.reason = reason


def _body_calls(function, formals, torch_global, wrappers):
    """Read the complete straight-line wrapper domain before a recipe exists."""
    calls, results = [], set()
    locals_in_scope = set(formals)

    def operation(node):
        if (type(node) is not ast.Call or type(node.func) is not ast.Attribute or node.keywords
                or type(node.func.value) is not ast.Name or node.func.value.id != torch_global
                or node.func.attr not in wrappers or any(type(arg) is not ast.Name
                or arg.id not in locals_in_scope for arg in node.args)):
            raise AtenRegionBoundaryError('UNSUPPORTED_WRAPPER_CALL',
                f'line {node.lineno} does not call a declared wrapper with positional local operands')
        position = [node.lineno, node.end_lineno, node.col_offset, node.end_col_offset]
        calls.append((position, wrappers[node.func.attr]['schema']))

    for statement in function.body[:-1]:
        if (type(statement) is not ast.Assign or statement.type_comment is not None or len(statement.targets) != 1
                or type(statement.targets[0]) is not ast.Name):
            raise AtenRegionBoundaryError('OPEN_SOURCE_BODY',
                f'line {statement.lineno} is outside the declared wrapper assignment domain')
        name = statement.targets[0].id
        if name == torch_global or name in locals_in_scope:
            raise AtenRegionBoundaryError('OPEN_SOURCE_BODY',
                f'line {statement.lineno} overwrites a local or wrapper global')
        operation(statement.value)
        results.add(name)
        locals_in_scope.add(name)

    result = function.body[-1].value
    if type(result) is ast.Call:
        operation(result)
    else:
        nodes = result.elts if type(result) is ast.Tuple else (result,)
        if not nodes or any(type(node) is not ast.Name or node.id not in results for node in nodes):
            raise AtenRegionBoundaryError('OPEN_RETURN_BOUNDARY',
                'every returned value must come from a declared wrapper call')
    return calls


class AtenRegionBoundaryIndex:
    """Reuse original F4 and exact CPython CALL bindings for one capture."""

    def __init__(self, view, coordinates, contract):
        self.wrappers = _wrappers(contract)
        self.torch_global = contract['torch_global']
        self.index = original_aten_call_index(view, coordinates)
        self.bindings = original_aten_call_bindings(self.index, coordinates)
        self.functions = {link.id: link for link in self.index.codes.records.values()}
        self.modules = {}
        self.bodies = {}

    def _body(self, binding):
        link = self.functions.get(binding.code_link)
        if link is None or len(link.constant_path) != 1:
            raise AtenRegionBoundaryError('UNSUPPORTED_SOURCE_ENTRY',
                'one original direct module-level Python function is required')
        compiled = self.index.caller_for(binding).code
        if compiled.co_filename != link.source.uri or compiled.co_qualname != compiled.co_name:
            raise AtenRegionBoundaryError('UNSUPPORTED_SOURCE_ENTRY',
                'the original callable has no ordinary source module entry')
        if link.id not in self.bodies:
            if link.source not in self.modules:
                raw = Path(link.source.uri).read_bytes()
                if (len(raw), sha256(raw).hexdigest()) != (link.source.size_bytes, link.source.sha256):
                    raise ValueError('original F4 source changed during region discovery')
                self.modules[link.source] = ast.parse(raw, filename=link.source.uri, type_comments=True)
            module = self.modules[link.source]
            try:
                function, formals = _function(module, compiled.co_name, self.torch_global)
            except ValueError as error:
                raise AtenRegionBoundaryError('UNSUPPORTED_SOURCE_ENTRY', str(error)) from error
            if function.lineno != compiled.co_firstlineno or compiled.co_freevars or compiled.co_cellvars:
                raise AtenRegionBoundaryError('UNSUPPORTED_SOURCE_ENTRY',
                    'the declared function does not select a unique closure-free code entry')
            self.bodies[link.id] = _body_calls(function, formals, self.torch_global, self.wrappers)
        return link, compiled, self.bodies[link.id]

    def qualify(self, calls):
        """Establish full source coverage for the selected activation, without authorizing replacement."""
        calls = sorted(calls, key=lambda call: call['attempt'])
        if any(len(call['operators']) != 1 for call in calls):
            raise AtenRegionBoundaryError('WRAPPER_OPERATOR_CARDINALITY',
                'each source wrapper must have exactly one direct native operator')
        try:
            invocation, bindings = original_aten_calls(self.index, self.bindings, calls)
        except AtenSourceMatchError as error:
            raise AtenRegionBoundaryError('UNRESOLVED_F4_CALL_BODY', str(error)) from error
        link, compiled, expected = self._body(bindings[0])
        if any(binding.code_link != link.id for binding in bindings):
            raise AtenRegionBoundaryError('UNRESOLVED_F4_CALL_BODY',
                'the calls do not select the same original compiled source body')
        observed = [(call['site']['position'], call['operators'][0]['schema']) for call in calls]
        if len(expected) != len(observed):
            raise AtenRegionBoundaryError('INCOMPLETE_SOURCE_COVERAGE',
                f'{len(observed)} observed wrapper calls for {len(expected)} required source calls')
        if observed != expected:
            raise AtenRegionBoundaryError('SOURCE_CALL_OR_SCHEMA_MISMATCH',
                'the observed wrapper positions and exact ATen schemas differ from the complete source body')
        return dict(source=dict(path=link.source.uri, sha256=link.source.sha256, size_bytes=link.source.size_bytes),
            function=compiled.co_name, code_key=bindings[0].attempt.site.code, first_lineno=compiled.co_firstlineno,
            body=encode(link.body), body_declaration=encode(link.function_fact),
            invocation=invocation, call_attempts=[call['attempt'] for call in calls],
            source_calls=[encode(binding.call) for binding in bindings], source_replacement_authorized=False)

    def check(self):
        self.index.check()


def qualify_aten_region_recipe(boundary, recipe, contract):
    """Check the projected operands and return against the pinned whole source."""
    try:
        result = qualify_python_trial_source(boundary['source']['path'], boundary['function'], recipe, contract)
    except ValueError as error:
        raise AtenRegionBoundaryError('SOURCE_RECIPE_MISMATCH', str(error)) from error
    if any(result[name] != boundary[name] for name in ('source', 'code_key', 'first_lineno')):
        raise ValueError('the projected trial source differs from the original F4 and CPython identity')
    return result
