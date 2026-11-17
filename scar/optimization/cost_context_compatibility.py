"""Metadata-only applicability checks for canonical stateful boundary costs.

Source configuration is distinct from a recorded runtime configuration. The
legacy source check only parses retained Python. The native-group check also
replays host lowering and retained GCC records; neither path invokes a compiler
or loads native code, changes timings, or grants a qualified cost profile.
"""
import argparse
import ast
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference, read_artifact


SCHEMA = 'scar.cost-context-compatibility'
_SCOPE = 'scar.stateful-scope-boundary-calibration'
_HOLDOUT = 'scar.real-caller-cost-holdout'
_MODEL = 'scar.stateful-activity-entry-model'
_ROOT = Path(__file__).resolve().parents[1]
_TERMINAL = 'scar.runtime.stateful_guard.guarded_stateful_function'
_FACTORIES = {_TERMINAL, 'scar.runtime.stateful_guard.install_stateful_regions',
    'scar.compiler.stateful_runtime.guarded_stateful_function',
    'scar.compiler.stateful_runtime.install_stateful_regions'}
_UNKNOWN = object()


class _Symbol:
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return isinstance(other, _Symbol) and self.name == other.name


def _value(node, env):
    """Small closed literal/alias language, deliberately not Python evaluation."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return env.get(node.id, _UNKNOWN)
    if isinstance(node, ast.Attribute):
        parent = _value(node.value, env)
        return _Symbol(parent.name + '.' + node.attr) if isinstance(parent, _Symbol) else _UNKNOWN
    if isinstance(node, ast.Dict):
        result = {}
        for key, value in zip(node.keys, node.values):
            item = _value(value, env)
            if key is None:
                if type(item) is not dict:
                    return _UNKNOWN
                result.update(item)
            else:
                key = _value(key, env)
                if type(key) is not str:
                    return _UNKNOWN
                result[key] = item
        return result
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == 'dict' and 'dict' not in env and not node.args):
        return _keywords(node, env)
    return _UNKNOWN


def _keywords(call, env):
    result = {}
    for item in call.keywords:
        value = _value(item.value, env)
        if item.arg is None:
            if type(value) is not dict or any(name in result for name in value):
                return _UNKNOWN
            result.update(value)
        elif item.arg in result:
            return _UNKNOWN
        else:
            result[item.arg] = value
    return result


def _same(left, right):
    if left is _UNKNOWN or right is _UNKNOWN:
        return left is right
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(_same(left[k], right[k]) for k in left)
    return left == right


def _merge(env, alternatives):
    for name in set().union(*(set(item) for item in alternatives)):
        values = [item.get(name, _UNKNOWN) for item in alternatives]
        env[name] = values[0] if all(_same(values[0], item) for item in values[1:]) else _UNKNOWN


def _assign(target, value, env):
    if isinstance(target, ast.Name):
        env[target.id] = value
    elif isinstance(target, (ast.Tuple, ast.List)):
        for item in target.elts:
            _assign(item, _UNKNOWN, env)
    elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
        # Alias mutation is intentionally not inferred as an immutable literal.
        _invalidate(target.value.id, env)
    elif isinstance(target, ast.Attribute):
        while isinstance(target, ast.Attribute):
            target = target.value
        if isinstance(target, ast.Name):
            _invalidate(target.id, env)


def _invalidate(name, env):
    previous = env.get(name, _UNKNOWN)
    for alias, value in list(env.items()):
        if alias == name or type(previous) is dict and value is previous:
            env[alias] = _UNKNOWN
        elif isinstance(previous, _Symbol) and isinstance(value, _Symbol) and (
                value.name == previous.name or value.name.startswith(previous.name + '.')):
            env[alias] = _UNKNOWN


def _imports(stmt, env, module):
    if isinstance(stmt, ast.Import):
        for item in stmt.names:
            env[item.asname or item.name.split('.')[0]] = _Symbol(item.name if item.asname else item.name.split('.')[0])
    elif isinstance(stmt, ast.ImportFrom):
        prefix = stmt.module or ''
        if stmt.level:
            parts = module.split('.')[:-stmt.level]
            prefix = '.'.join([*parts, prefix]) if prefix else '.'.join(parts)
        for item in stmt.names:
            if item.name == '*':
                # A star import can shadow any otherwise resolved name.
                for name in env:
                    env[name] = _UNKNOWN
            else:
                env[item.asname or item.name] = _Symbol(prefix + '.' + item.name)


def _calls(expression, env, found):
    if expression is None or isinstance(expression, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return
    if isinstance(expression, ast.Call):
        symbol = _value(expression.func, env)
        if isinstance(symbol, _Symbol) and symbol.name in _FACTORIES:
            found.append((symbol.name, expression, dict(env)))
        # A call can mutate a local kwargs dictionary passed to it.  Such a
        # dictionary is no longer a closed literal at subsequent factory calls.
        for argument in expression.args:
            if isinstance(argument, ast.Name) and type(env.get(argument.id)) is dict:
                _invalidate(argument.id, env)
        for item in expression.keywords:
            if item.arg is not None and isinstance(item.value, ast.Name) and type(env.get(item.value.id)) is dict:
                _invalidate(item.value.id, env)
        if isinstance(expression.func, ast.Attribute) and isinstance(expression.func.value, ast.Name):
            name = expression.func.value.id
            if type(env.get(name)) is dict:
                _invalidate(name, env)
    # Comprehension bindings and assignment expressions require a larger
    # interpreter.  Invalidate their bound names instead of resolving aliases.
    if isinstance(expression, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        return
    if isinstance(expression, ast.NamedExpr):
        _assign(expression.target, _UNKNOWN, env)
        return
    for child in ast.iter_child_nodes(expression):
        _calls(child, env, found)


def _block(statements, env, found, module, *, functions=None):
    for stmt in statements:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            _imports(stmt, env, module)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            env[stmt.name] = _Symbol(module + '.' + stmt.name)
            if functions is not None:
                functions.append((stmt, dict(env)))
        elif isinstance(stmt, ast.ClassDef):
            env[stmt.name] = _UNKNOWN
        elif isinstance(stmt, ast.Assign):
            value = _value(stmt.value, env)
            _calls(stmt.value, env, found)
            for target in stmt.targets:
                _assign(target, value, env)
        elif isinstance(stmt, ast.AnnAssign):
            value = _value(stmt.value, env)
            _calls(stmt.value, env, found)
            _assign(stmt.target, value, env)
        elif isinstance(stmt, (ast.AugAssign, ast.Delete)):
            for target in ([stmt.target] if isinstance(stmt, ast.AugAssign) else stmt.targets):
                _assign(target, _UNKNOWN, env)
        elif isinstance(stmt, ast.If):
            _calls(stmt.test, env, found)
            test = _value(stmt.test, env)
            branches = [stmt.body] if test is True else [stmt.orelse] if test is False else [stmt.body, stmt.orelse]
            outcomes = []
            for branch in branches:
                other = dict(env)
                _block(branch, other, found, module)
                outcomes.append(other)
            _merge(env, outcomes)
        elif isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
            _calls(stmt.iter if hasattr(stmt, 'iter') else stmt.test, env, found)
            other = dict(env)
            if hasattr(stmt, 'target'):
                _assign(stmt.target, _UNKNOWN, other)
            # A loop-carried assignment is not known from just its first
            # iteration; invalidate before resolving any calls in the body.
            for item in ast.walk(ast.Module(body=stmt.body, type_ignores=[])):
                if isinstance(item, ast.Name) and isinstance(item.ctx, (ast.Store, ast.Del)):
                    other[item.id] = _UNKNOWN
            _block(stmt.body, other, found, module)
            _merge(env, [dict(env), other])
            _block(stmt.orelse, env, found, module)
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            for item in stmt.items:
                _calls(item.context_expr, env, found)
                if item.optional_vars:
                    _assign(item.optional_vars, _UNKNOWN, env)
            _block(stmt.body, env, found, module)
        elif isinstance(stmt, (ast.Try, ast.TryStar)):
            normal = dict(env)
            _block(stmt.body, normal, found, module)
            _block(stmt.orelse, normal, found, module)
            alternatives = [normal]
            for handler in stmt.handlers:
                other = dict(env)
                if handler.name:
                    other[handler.name] = _UNKNOWN
                _block(handler.body, other, found, module)
                alternatives.append(other)
            _merge(env, alternatives)
            _block(stmt.finalbody, env, found, module)
        else:
            _calls(stmt, env, found)


def _parameters(fn):
    return [*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs]


def _witness(ref, node):
    return dict(artifact=deepcopy(ref), line=node.lineno, end_line=node.end_lineno,
        ast_sha256=sha256(ast.dump(node, include_attributes=False).encode()).hexdigest())


class _Sources:
    def __init__(self, observations):
        self.modules, self.frontier, self.references = {}, [], []
        pins = observations.get('probe', {}).get('implementation_files', {})
        if type(pins) is not dict:
            pins = {}
        for name, expected in sorted(pins.items()):
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts or relative.suffix != '.py':
                continue
            path = _ROOT / relative
            module = 'scar.' + '.'.join(relative.with_suffix('').parts)
            try:
                ref = artifact_reference(path, 'scar.python-source')
                if ref['sha256'] != expected:
                    raise ValueError('source differs from retained collector pin')
                self.add(module, ref)
            except (OSError, ValueError, SyntaxError, UnicodeError) as error:
                self.frontier.append(dict(field='source_pins.' + name, reason=str(error)))

    def add(self, module, ref):
        raw = read_artifact(ref, json_document=False)
        tree = ast.parse(raw, filename=ref['uri'])
        env, definitions = {}, []
        _block(tree.body, env, [], module, functions=definitions)
        self.modules[module] = (ref, tree, env, {fn.name: fn for fn, _ in definitions})
        if ref not in self.references:
            self.references.append(deepcopy(ref))

    def root_calls(self, module):
        ref, tree, module_env, definitions = self.modules[module]
        found = []
        # Only top-level function bodies are collector entry candidates.  Their
        # parameters are unknown; their defaults are not presumed supplied.
        for fn in definitions.values():
            env = dict(module_env)
            env.update({item.arg: _UNKNOWN for item in _parameters(fn)})
            if fn.args.vararg:
                env[fn.args.vararg.arg] = _UNKNOWN
            if fn.args.kwarg:
                env[fn.args.kwarg.arg] = _UNKNOWN
            calls = []
            _block(fn.body, env, calls, module)
            found.extend((fn.name, *row) for row in calls)
        return found

    def trace(self, target, call, env):
        chain, seen = [], set()
        while target in _FACTORIES and target not in seen:
            seen.add(target)
            module, name = target.rsplit('.', 1)
            if module not in self.modules or name not in self.modules[module][3]:
                return dict(status='UNKNOWN', reason='forwarding/default source is not pinned', chain=chain)
            ref, tree, globals_env, definitions = self.modules[module]
            fn = definitions[name]
            if not _same(globals_env.get(name, _UNKNOWN), _Symbol(target)):
                return dict(status='UNKNOWN', reason='factory name was rebound after its definition', chain=chain)
            keywords = _keywords(call, env)
            trial_args = [item for item in fn.args.kwonlyargs if item.arg == 'trial']
            positional = [*fn.args.posonlyargs, *fn.args.args]
            required = positional[:len(positional)-len(fn.args.defaults)]
            bound = {item.arg for item in positional[:len(call.args)]}
            if type(keywords) is dict:
                bound.update(keywords)
            if (type(keywords) is not dict or len(trial_args) != 1
                    or any(isinstance(item, ast.Starred) for item in call.args)
                    or len(call.args) > len(positional)
                    or any(name not in {item.arg for item in _parameters(fn)} for name in keywords)
                    or any(item.arg in keywords for item in positional[:len(call.args)])
                    or any(item.arg not in bound for item in required)
                    or any(not isinstance(_value(item, globals_env), _Symbol)
                        or _value(item, globals_env).name != 'contextlib.contextmanager' for item in fn.decorator_list)):
                return dict(status='UNKNOWN', reason='call kwargs or keyword-only trial binding is not closed', chain=chain)
            supplied = 'trial' in keywords
            if supplied:
                trial = keywords['trial']
                origin = dict(kind='CALL_KEYWORD_OR_CLOSED_KWARGS', call_ast_sha256=
                    sha256(ast.dump(call, include_attributes=False).encode()).hexdigest())
            else:
                index = next(i for i, item in enumerate(fn.args.kwonlyargs) if item.arg == 'trial')
                default = fn.args.kw_defaults[index]
                trial = _value(default, globals_env)
                origin = dict(kind='PINNED_KEYWORD_ONLY_DEFAULT', evidence=_witness(ref, default)) if default is not None else dict(kind='MISSING_REQUIRED_ARGUMENT')
            step = dict(factory=target, definition=_witness(ref, fn),
                trial=trial if type(trial) is bool else None, trial_origin=origin)
            chain.append(step)
            if type(trial) is not bool:
                return dict(status='UNKNOWN', reason='trial is not a closed bool literal/forwarded parameter', chain=chain)
            if target == _TERMINAL:
                nested = [node for node in fn.body if isinstance(node, ast.FunctionDef) and node.name == 'call']
                branches = [node for node in ast.walk(nested[0]) if isinstance(node, ast.If)
                    and isinstance(node.test, ast.UnaryOp) and isinstance(node.test.op, ast.Not)
                    and isinstance(node.test.operand, ast.Name) and node.test.operand.id == 'trial'] if len(nested) == 1 else []
                writes = [node for node in ast.walk(fn) if isinstance(node, ast.Name)
                    and node.id == 'trial' and isinstance(node.ctx, (ast.Store, ast.Del))]
                if len(branches) != 1 or writes:
                    return dict(status='UNKNOWN', reason='unique unreassigned terminal trial branch not established', chain=chain)
                return dict(status='SOURCE_CONFIGURATION_PROVEN', trial=trial, chain=chain,
                    terminal_branch=_witness(ref, branches[0]),
                    nontrial_branch_selected_by_source=not trial,
                    observed_runtime_branch=False,
                    loaded_runtime_module_identity_observed=False)
            forwarded = dict(globals_env)
            forwarded.update({item.arg: _UNKNOWN for item in _parameters(fn)})
            forwarded['trial'] = trial
            calls = []
            _block(fn.body, forwarded, calls, module)
            if len(calls) != 1:
                return dict(status='UNKNOWN', reason='unique forwarding factory call not established', chain=chain)
            target, call, env = calls[0]
        return dict(status='UNKNOWN', reason='unsupported or cyclic guard forwarding', chain=chain)

    def configuration(self, module):
        calls = self.root_calls(module)
        if len(calls) != 1:
            return dict(status='UNKNOWN', reason='unique collector/harness factory call not established',
                candidate_calls=len(calls))
        function, target, call, env = calls[0]
        ref = self.modules[module][0]
        return dict(entry_function=function, call=_witness(ref, call), **self.trace(target, call, env))


def _material(value, schema):
    if isinstance(value, (str, Path)):
        value = artifact_reference(Path(value).resolve(), schema)
    document = read_artifact(value)
    if document.get('schema') != schema or document.get('schema_version') != 1:
        raise ValueError('supported retained ' + schema + '/v1 artifact required')
    return deepcopy(value), document


def _field(name, canonical, actual, *, evidence=(), qualification=None):
    relation = 'UNKNOWN' if canonical is None or actual is None else 'MATCH' if canonical == actual else 'MISMATCH'
    return dict(name=name, relation=relation, canonical=deepcopy(canonical), actual=deepcopy(actual),
        evidence=deepcopy(list(evidence)), qualification=qualification)


def _generated_source(reference, expected_sha256, expected_bytes=None):
    """Retain source bytes, without reading or loading the compiled library."""
    ref = artifact_reference(reference, 'scar.cpp-source')
    if ref['sha256'] != expected_sha256 or expected_bytes is not None and ref['size_bytes'] != expected_bytes:
        raise ValueError('generated source differs from retained byte identity')
    return ref


def check_stateful_cost_compatibility(scope_observations, holdout, *, model=None):
    """Retain/replay source facts and metadata without qualifying a cost profile.

    Inputs are ArtifactRefs (or paths normalized to ArtifactRefs).  A source
    mismatch is sufficient to reject automatic canonical transfer; source
    agreement alone does not prove an observed runtime branch or performance.
    """
    scope_ref, scope = _material(scope_observations, _SCOPE)
    holdout_ref, actual = _material(holdout, _HOLDOUT)
    references = [scope_ref, holdout_ref]
    sources = _Sources(scope)
    producers = [name for name, (_, _, env, _) in sources.modules.items()
        if env.get('SCHEMA') == scope['schema']]
    canonical_source = sources.configuration(producers[0]) if len(producers) == 1 else dict(
        status='UNKNOWN', reason='unique pinned producer with matching literal SCHEMA not established')
    harness_ref = actual.get('harness')
    if harness_ref is None:
        actual_source = dict(status='UNKNOWN', reason='retained harness source missing')
    else:
        sources.add('__retained_holdout_harness__', harness_ref)
        actual_source = sources.configuration('__retained_holdout_harness__')
    fields = [_field('recorded_guard_regime.trial',
        scope.get('guard_regime', {}).get('trial'), actual.get('guard_regime', {}).get('trial'),
        evidence=[scope_ref, holdout_ref], qualification=
        'Only a dedicated recorded guard_regime.trial bool is runtime metadata; explicit_trial is not substituted.')]
    if any(item['canonical'] is not None and type(item['canonical']) is not bool
            or item['actual'] is not None and type(item['actual']) is not bool for item in fields):
        raise ValueError('recorded trial must be a bool when present')
    source_trial = _field('source_configuration.trial', canonical_source.get('trial'), actual_source.get('trial'),
        evidence=sources.references, qualification=
        'Static configuration of retained source, not a measured branch duration or proof of loaded runtime bytes.')
    fields.append(source_trial)
    fields.append(_field('recorded_runtime_wrapper_chain', scope.get('runtime_path'), actual.get('runtime_path'),
        evidence=[scope_ref, holdout_ref], qualification='Missing actual runtime-path metadata remains UNKNOWN.'))
    canonical_terminal = canonical_source.get('terminal_branch', {}).get('artifact')
    actual_terminal = actual_source.get('terminal_branch', {}).get('artifact')
    fields.append(_field('source_configuration.guard_factory', canonical_terminal, actual_terminal,
        qualification='Shared pinned factory source; this is not a runtime loaded-module receipt.'))

    # These are independent applicability coordinates, never substitutes for
    # the guard regime or for full-call cost/physical-resource qualification.
    specialization_ref, specialization = _material(actual['specialization'], 'scar.observed-stateful-specialization')
    native_ref, native = _material(actual['native_artifact'], 'scar.native-stateful-artifact')
    references.extend([specialization_ref, native_ref])
    domain_ref, domain = _material(specialization['domain'], 'scar.stateful-cpu-domain')
    references.append(domain_ref)
    fields.append(_field('domain_id', scope.get('domain_id'), domain.get('domain_id'), evidence=[scope_ref, domain_ref]))
    fields.append(_field('candidate_id', scope.get('candidate_id'),
        actual.get('frozen_model_binding', {}).get('candidate_id'), evidence=[scope_ref, holdout_ref]))
    fields.append(_field('base_stage_plan', scope.get('domain_source', {}).get('stage_plan'),
        domain.get('base_stage_plan'), evidence=[scope_ref, domain_ref]))
    canonical_shapes = {row['id']: row.get('shape') for row in scope.get('canonical_structure', {}).get('inputs', [])
        if row.get('kind') == 'array'} or None
    fields.append(_field('input_shapes', canonical_shapes, specialization.get('input_shapes'), evidence=[scope_ref, specialization_ref]))
    fields.append(_field('source_region', scope.get('domain_source', {}).get('region'), specialization.get('source_region'),
        evidence=[scope_ref, specialization_ref]))
    fields.append(_field('compiler_descriptor', scope.get('probe', {}).get('compiler'), native.get('compiler'),
        evidence=[scope_ref, native_ref], qualification='Matching compiler/options does not prove matching generated code.'))
    fields.append(_field('provider_runtime', scope.get('resource_context', {}).get('backend', {}).get('provider_runtime'),
        native.get('provider_runtime'), evidence=[scope_ref, native_ref],
        qualification='Matching provider binary/ABI does not establish matching values, warming or cache state.'))
    fields.append(_field('caller_affinity_domain', scope.get('observed_caller_affinity_before'),
        actual.get('protocol', {}).get('caller_cpu_ids'), evidence=[scope_ref, holdout_ref],
        qualification='Allowed mask equality is not an exclusive whole-entry CPU reservation.'))
    fields.append(_field('selected_native_cpu_ids', scope.get('probe', {}).get('source_fragments', {}).get('cpus'),
        native.get('cpu_ids'), evidence=[scope_ref, native_ref],
        qualification='Internal native binding does not prove whole-entry physical binding.'))

    generated_sources = {}
    canonical_cpp = scope.get('probe', {}).get('source', {})
    actual_cpp = native.get('files', {}).get('source')
    for side, path, digest, size in (
        ('canonical', canonical_cpp.get('path'), canonical_cpp.get('sha256'), canonical_cpp.get('bytes')),
        ('actual', str(Path(native_ref['uri']).parent/actual_cpp) if type(actual_cpp) is str else None,
            native.get('sha256', {}).get('source'), None)):
        try:
            if path is None or digest is None:
                raise ValueError('retained generated source identity missing')
            generated_sources[side] = _generated_source(path, digest, size)
            references.append(generated_sources[side])
        except (OSError, ValueError) as error:
            sources.frontier.append(dict(field='compile_specialization.' + side, reason=str(error)))

    timer = dict(name='timer_scope', relation='CONDITIONAL_TRANSFER' if (
        scope.get('exclusive_partition_verified') is True and scope.get('whole_target_call_measured') is False
        and scope.get('clock') and scope.get('cuts') and actual.get('protocol', {}).get('method_scope')) else 'UNKNOWN', canonical=dict(
        clock=scope.get('clock'), cuts=scope.get('cuts'),
        whole_target_call_measured=scope.get('whole_target_call_measured'),
        exclusive_partition_verified=scope.get('exclusive_partition_verified')),
        actual=deepcopy(actual.get('protocol', {}).get('method_scope')),
        evidence=[scope_ref, holdout_ref], qualification=
        'Canonical exclusive cuts and the actual whole guarded method are different measurement scopes; no scope median subtraction or numerical equivalence asserted.')
    specialization = dict(name='compile_specialization', relation='CONDITIONAL_TRANSFER' if (
        set(generated_sources) == {'canonical', 'actual'}
        and scope.get('probe', {}).get('source_fragments', {}).get('skipped_work')
        and native.get('sha256', {}).get('source') and native.get('execution')) else 'UNKNOWN', canonical=dict(
        generated_source_sha256=scope.get('probe', {}).get('source', {}).get('sha256'),
        skipped_work=scope.get('probe', {}).get('source_fragments', {}).get('skipped_work'),
        scope_transformation=scope.get('probe', {}).get('source_fragments', {}).get('scope_transformation')),
        actual=dict(generated_source_sha256=native.get('sha256', {}).get('source'), execution=native.get('execution')),
        evidence=[scope_ref, native_ref, *generated_sources.values()], qualification=
        'A canonical empty/instrumented boundary is not the complete source-specialized native function. Transfer requires a declared assumption, not compiler-descriptor equality.')
    fields.extend([timer, specialization])
    for name in ('input_values_and_predecessor_state', 'page_cache_and_allocator_regime',
            'instrumentation_and_cross_component_interaction'):
        fields.append(_field(name, None, None, qualification=
            'No paired applicability evidence in these retained records; no residual attribution or zero cost.'))

    model_binding = None
    if model is not None:
        model_ref, model_doc = _material(model, _MODEL)
        references.append(model_ref)
        fields.append(_field('frozen_model_reference', model_ref, actual.get('frozen_model'), evidence=[holdout_ref]))
        model_binding = dict(artifact=model_ref, flags={key: deepcopy(model_doc.get(key)) for key in
            ('canonical_transfer_validated', 'complete_target_implementation_cost', 'conditional_point_estimate_available',
             'physical_resource_qualification', 'deployment_authorized')},
            qualification='Retained metadata only; this checker neither re-admits nor modifies the model/profile.')
    mismatches = [item['name'] for item in fields if item['relation'] == 'MISMATCH']
    unresolved = [item['name'] for item in fields if item['relation'] == 'UNKNOWN']
    return dict(schema=SCHEMA, schema_version=1, kind='STATEFUL_CANONICAL_SCOPE_TO_GUARDED_METHOD',
        inputs=dict(scope_observations=scope_ref, holdout=holdout_ref, model=model_binding),
        checker_source=artifact_reference(__file__, 'scar.python-source'),
        source_witnesses=dict(canonical=canonical_source, actual=actual_source,
            authority='PINNED_SOURCE_CONFIGURATION_ONLY',
            actual_loaded_guard_module_identity_observed=False), fields=fields,
        result=dict(status='INCOMPATIBLE_SOURCE_OR_RECORDED_CONTEXT' if mismatches else 'UNRESOLVED',
            mismatched_fields=mismatches, unknown_fields=unresolved,
            automatic_cost_transfer_qualified=False, qualified_profile=None),
        retained_transfer_conditions=deepcopy(scope.get('conditions', [])),
        retained_missing_terms=deepcopy(scope.get('missing_terms', [])),
        source_frontier=sources.frontier, source_refs=[*references, *sources.references],
        numerical_residual_attribution=None, cost_values_changed=False,
        frozen_model_or_holdout_qualification_changed=False,
        physical_resource_qualification='SEPARATE_NOT_EVALUATED',
        target_program_executed=False, native_loaded=False, native_compiled=False,
        deployment_authorized=False)


def verify_cost_compatibility(report, scope_observations, holdout, *, model=None):
    """Re-read retained bytes and re-run all supported static checks."""
    replay = check_stateful_cost_compatibility(scope_observations, holdout, model=model)
    if report != replay:
        raise ValueError('cost context compatibility report differs from metadata/source replay')
    return replay


def compare_native_group_compilation(native_collection, group_observations):
    """Compare GCC model fields at identical original stage-body locations.

    Model versions remain a multiset. Equality is only about the reported GCC
    scope, not machine code, physical latency, canonical data or cache state.
    """
    from collections import Counter
    from scar.compiler.collect_native_optimization import check_gcc_optimization
    from scar.compiler.calibrate_native_groups import check_native_group_observations
    from scar.analysis.gcc_cost_features import project_gcc_vector_costs

    native_ref, _ = _material(native_collection, 'scar.gcc-native-optimization-collection')
    observations_ref, _ = _material(group_observations, 'scar.native-stage-group-calibration')
    actual = check_gcc_optimization(native_ref)
    observed = check_native_group_observations(observations_ref)
    design = read_artifact(observed['design'])
    compiler = read_artifact(observed['compiler_collection'])
    subject = actual['subject']
    if type(subject) is not dict or subject.get('kind') != 'STATEFUL_NATIVE_LOWERING':
        raise ValueError('original native-view compiler collection required')
    view = subject['view']; observer = design['observer']
    if (view['native_manifest'] != observer['parent_view']['native_manifest']
            or view['identity'] != observer['parent_view']['identity']
            or view['compiler'] != compiler['compiler']):
        raise ValueError('group observations refer to a different original native/compiler context')
    left = project_gcc_vector_costs(native_ref)
    right = read_artifact(observed['compiler_features'])
    headers_match = {name:(r['sha256'], r['size_bytes']) for name,r in view['headers'].items()} == {
        name:(r['sha256'], r['size_bytes']) for name,r in design['headers'].items()}
    generator_match = left['observed_wire_header'] == right['observed_wire_header']
    original_stages = {row['id']:row for row in view['stages']}
    groups_stages = {row['id']:row for row in observer['stages']}
    if set(original_stages) != set(groups_stages):
        raise ValueError('group compilation does not preserve all original stage identities')

    def models(document, stage):
        rows = []
        spans = stage['source_line_spans']
        for index, block in enumerate(document['blocks']):
            if stage['id'] not in block['stage_ids']:
                continue
            locations = []
            for association in block['stage_associations']:
                if association['stage_id'] != stage['id']:
                    continue
                for point in association['primary_locations']:
                    matches = [s for s in spans if s['start_line'] <= point['line'] <= s['end_line']]
                    if len(matches) != 1:
                        return None, rows
                    locations.append(dict(body_line_offset=point['line']-matches[0]['start_line'],
                        column=point['column']))
            if not locations or any(row['status'] == 'UNKNOWN' for row in block['features'].values()):
                return None, rows
            signature = dict(primary_body_locations=sorted(locations, key=lambda r:(r['body_line_offset'], r['column'] or 0)),
                features={name:{key:value[key] for key in ('value','status','unit')}
                    for name,value in block['features'].items()})
            rows.append(dict(block_index=index, signature=signature, diagnostic_evidence=block['evidence']))
        return Counter(json.dumps(row['signature'],sort_keys=True,allow_nan=False) for row in rows), rows

    rows = []
    for stage_id, original in original_stages.items():
        instrumented = groups_stages[stage_id]
        if (original['body_sha256'] != instrumented['body_sha256']
                or original['source_action_ids'] != instrumented['source_action_ids']):
            raise ValueError('group observer changed original body or source-action identity')
        before, before_rows = models(left, original)
        after, after_rows = models(right, instrumented)
        status = ('UNKNOWN' if before is None or after is None or not before or not after else
            'MATCH' if before == after and headers_match and generator_match else 'MISMATCH')
        rows.append(dict(stage=stage_id, original_body_sha256=original['body_sha256'],
            source_action_ids=original['source_action_ids'], relation=status,
            actual_models=before_rows, observer_models=after_rows,
            qualification='MULTISET_OF_SUPPORTED_GCC_MODEL_FIELDS_AT_BODY_RELATIVE_PRIMARY_LOCATIONS'))
    counts = Counter(row['relation'] for row in rows)
    return dict(schema=SCHEMA, schema_version=1, kind='ORIGINAL_NATIVE_TO_ORIGINAL_FRAME_GROUP_GCC_MODEL',
        inputs=dict(native_collection=native_ref, group_observations=observations_ref),
        checker_source=artifact_reference(__file__, 'scar.python-source'),
        original_native=view['native_manifest'], original_identity=view['identity'],
        observer_identity=observer['observer_identity'], compiler_headers_match=headers_match,
        compiler_generator_match=generator_match, stages=rows,
        result=dict(status='GCC_MODEL_FIELDS_DIFFER' if counts['MISMATCH'] else
            'PARTIAL_GCC_MODEL_MATCH' if counts['MATCH'] and counts['UNKNOWN'] else
            'GCC_MODEL_FIELDS_MATCH' if counts['MATCH'] else 'UNKNOWN_GCC_MODEL_CONTEXT',
            matched_stages=counts['MATCH'], mismatched_stages=counts['MISMATCH'],
            unknown_stages=counts['UNKNOWN'], automatic_cost_transfer_qualified=False, qualified_profile=None),
        retained_missing_terms=deepcopy(observed['residuals']),
        numerical_residual_attribution=None, cost_values_changed=False,
        machine_code_identity_proven=False, target_latency_equivalence_proven=False,
        stage_costs_summed=False, target_program_executed=False,
        compiler_invoked=False, native_loaded=False, deployment_authorized=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope-observations')
    parser.add_argument('--holdout')
    parser.add_argument('--native-collection')
    parser.add_argument('--group-observations')
    parser.add_argument('--model')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    if args.native_collection or args.group_observations:
        if not args.native_collection or not args.group_observations or args.scope_observations or args.holdout or args.model:
            parser.error('native group comparison requires only --native-collection and --group-observations')
        report = compare_native_group_compilation(args.native_collection, args.group_observations)
    else:
        if not args.scope_observations or not args.holdout: parser.error('--scope-observations and --holdout required')
        report = check_stateful_cost_compatibility(args.scope_observations, args.holdout, model=args.model)
        verify_cost_compatibility(report, args.scope_observations, args.holdout, model=args.model)
    path = Path(args.out).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(report=artifact_reference(path, SCHEMA), result=report['result']), sort_keys=True))


if __name__ == '__main__':
    main()
