"""Source-bound scoped installation of explicit workspace-session boundaries.

The spec must come from the producer-pinned compact loader. This installer
checks live bindings; it cannot prove caller effects or intercept arbitrary
internal stores. Missing closure means fresh kernels, never retained content.
"""
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
import importlib.abc
from importlib.machinery import PathFinder, SourceFileLoader
import inspect
import math
from pathlib import Path
import sys
import threading
from types import FunctionType, ModuleType


SCHEMA = 'scar.workspace-session-scope'
_ROLES = {'CALLER', 'KERNEL', 'COPY', 'BARRIER'}
_ASSIGNED = ('__module__', '__name__', '__qualname__', '__doc__')


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _raw_metadata(fn):
    """Inspect only builtin function fields before any mapping traversal."""
    return (type(fn) is FunctionType and type(fn.__defaults__) in (tuple, type(None))
            and type(fn.__kwdefaults__) in (dict, type(None))
            and type(vars(fn)) is dict and type(fn.__globals__) is dict)


def _signature(fn):
    """Read ordinary call ABI without requesting deferred annotations."""
    _need(_raw_metadata(fn), 'function metadata must use exact builtin containers')
    code = fn.__code__
    _need(not code.co_flags & (0x0c | 0x20 | 0x80 | 0x200),
          'variadic or deferred execution boundary is outside the compact scope profile')
    defaults = fn.__defaults__ or ()
    kwdefaults = fn.__kwdefaults__ or {}
    params = []
    for i, name in enumerate(code.co_varnames[:code.co_argcount]):
        kind = inspect.Parameter.POSITIONAL_ONLY if i < code.co_posonlyargcount else inspect.Parameter.POSITIONAL_OR_KEYWORD
        default = inspect.Parameter.empty if i < code.co_argcount-len(defaults) else defaults[i-code.co_argcount+len(defaults)]
        params.append(inspect.Parameter(name, kind, default=default))
    for name in code.co_varnames[code.co_argcount:code.co_argcount+code.co_kwonlyargcount]:
        params.append(inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY,
            default=kwdefaults.get(name, inspect.Parameter.empty)))
    return inspect.Signature(params)


def _literal(value, depth=0):
    return (type(value) in (type(None), bool, int, float, str, bytes)
            or type(value) is tuple and depth < 8 and len(value) <= 64
            and all(_literal(item, depth+1) for item in value))


def _decorate(original, call):
    _need(_raw_metadata(original), 'function metadata must use exact builtin containers')
    call = wraps(original, assigned=_ASSIGNED)(call)
    if getattr(original, '__annotate__', None) is not None:
        def annotate(format):
            from annotationlib import Format, get_annotations
            return original.__annotations__ if format == Format.VALUE else get_annotations(original, format=format)
        call.__annotate__ = annotate
    else:
        call.__annotations__ = original.__annotations__
    return call


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, scope):
        self.scope = scope

    def find_spec(self, fullname, path=None, target=None):
        from .entry_runner import _Loader
        spec = PathFinder.find_spec(fullname, path, target)
        if spec is not None and type(spec.origin) is str and Path(spec.origin).resolve() in self.scope.sources:
            if type(spec.loader) is SourceFileLoader:
                spec.loader = _Loader(spec.loader, self.scope, fullname)
            else:
                self.scope.frontier.append('source_uses_nonstandard_loader')
        return spec


class SessionScope:
    def __init__(self, session, spec, *, trial=False, exclusive_inputs=False):
        self.session, self.spec = session, spec
        self.active = trial is True and exclusive_inputs is True and session.enabled is True
        self.owns_session = self.active
        self.owner = threading.get_ident()
        self.bindings, self.frontier, self.activations = [], [], []
        self.seen, self.suspended, self.finished = set(), 0, False
        self.stats = dict(caller_calls=0, copy_calls=0, barrier_calls=0, tainted_kernel_calls=0,
                          fresh_kernel_calls=0, retention_eligible_kernel_calls=0, guard_rejections=0,
                          isolated_kernel_attempts=0, isolated_native_calls=0,
                          isolated_original_fallback_calls=0, isolated_kernel_reasons={},
                          capacity_kernel_attempts=0, capacity_native_calls=0,
                          capacity_original_fallback_calls=0, capacity_forgets=0,
                          capacity_retained_intervals=0, capacity_created=0, capacity_freed=0,
                          capacity_cleanup_failures=0)
        self.unqualified_kernel_policy = 'ORIGINAL'
        self.capacity_policy = None
        self.capacity_baseline = None
        self.rows, self.sources = {}, set()
        if not self.active:
            return
        _need(type(spec) is dict and spec.get('schema') == SCHEMA
              and type(spec.get('schema_version')) is int and spec['schema_version'] == 1
              and type(spec.get('boundaries')) is list, 'producer-admitted compact session scope required')
        self.spec = deepcopy(spec)
        self.unqualified_kernel_policy = self.spec.get('unqualified_kernel_policy', 'ORIGINAL')
        _need(self.unqualified_kernel_policy in ('ORIGINAL', 'FRESH_ISOLATED_NATIVE',
              'RETAIN_PRIVATE_CAPACITY_FULL_SNAPSHOT'),
              'unsupported unqualified kernel policy')
        for row in self.spec['boundaries']:
            _need(type(row) is dict and type(row.get('id')) is str and row['id'] not in self.rows
                  and row.get('role') in _ROLES and row.get('descriptor') in ('FUNCTION', 'METHOD', 'STATICMETHOD')
                  and type(row.get('owner_path')) is list
                  and all(type(x) is str and x.isidentifier() for x in row['owner_path'])
                  and type(row.get('attribute')) is str and row['attribute'].isidentifier(),
                  'ordinary source boundary ID/path/role required')
            self.rows[row['id']] = row
            self.sources.add(Path(row['source']['path']).resolve())
        if self.unqualified_kernel_policy == 'RETAIN_PRIVATE_CAPACITY_FULL_SNAPSHOT':
            policy = self.spec.get('capacity_policy')
            _need(type(policy) is dict and type(policy.get('proof_id')) is str and bool(policy['proof_id'])
                  and policy.get('admission') == 'PRODUCER_PROVEN_PRIVATE_CAPACITY_ONLY'
                  and policy.get('external_roots_retained') is False and policy.get('content_reuse') is False
                  and type(policy.get('outer_boundary_ids')) is list
                  and all(type(key) is str and key in self.rows and self.rows[key]['role'] == 'CALLER'
                          for key in policy['outer_boundary_ids']),
                  'producer-pinned private-capacity-only scope admission required')
            self.capacity_policy = policy
            self.capacity_baseline = self._capacity_counts()
        self.source = next(iter(self.sources), Path('.').resolve())

    def _lose(self, reason, *, taint=False):
        if taint:
            for frame in self.activations:
                frame['tainted'] = True
        try:
            self.session.before_fallback(reason)
        except Exception as error:
            self.active = False
            self.frontier.append('session_cleanup_failed:'+type(error).__name__)
        finally:
            self._sync_capacity_counts()

    def _invalidate(self, reason):
        try:
            self.session.invalidate(reason)
            return True
        except Exception:
            self._lose('session_invalidation_failed', taint=True)
            return False

    def _forget_capacity(self, reason):
        try:
            self.session.invalidate(reason)
            self.stats['capacity_forgets'] += 1
            return True
        except Exception:
            self.stats['capacity_cleanup_failures'] += 1
            self._lose('capacity_content_invalidation_failed', taint=True)
            self.active = False
            return False
        finally:
            self.session.bindings = ()
            self._sync_capacity_counts()

    def _capacity_outer(self):
        if not self.activations or self.capacity_policy is None:
            return False
        from .stateful_guard import _ordinary
        frame = self.activations[0]; row = frame['row']; owner = frame['receiver_owner']
        try:
            if (row['id'] not in self.capacity_policy['outer_boundary_ids'] or owner is None
                    or not self._live(row) or not _raw_metadata(frame['original'])
                    or list(frame['signature'].parameters) != row['parameters']):
                return False
            bound = frame['signature'].bind(*frame['args'], **frame['kwargs']); bound.apply_defaults()
            receiver = bound.arguments[next(iter(frame['signature'].parameters))]
            return type(receiver) is owner and row['attribute'] not in _ordinary(receiver)
        except (ValueError, TypeError, KeyError, AttributeError, OSError):
            return False

    def _capacity_counts(self):
        stats = self.session.stats
        return stats['creates'], stats['releases']

    def _sync_capacity_counts(self):
        if self.capacity_baseline is not None:
            creates, frees = self._capacity_counts()
            self.stats['capacity_created'] = creates-self.capacity_baseline[0]
            self.stats['capacity_freed'] = frees-self.capacity_baseline[1]

    def _release_capacity(self, reason):
        try:
            self.session.release(reason)
        except Exception:
            self.stats['capacity_cleanup_failures'] += 1
            self._lose('capacity_release_failed', taint=True)
            self.active = False
        finally:
            self.session.bindings = ()
            self._sync_capacity_counts()

    def _live(self, row):
        from .stateful_guard import _function, _same_function_state
        record = next((r for r in self.bindings if r['id'] == row['id']), None)
        if record is None or vars(record['owner']).get(record['attribute']) is not record['installed']:
            return False
        fn = record['original']
        return (_raw_metadata(fn) and _same_function_state(fn, record['state']) and '__signature__' not in vars(fn)
                and '__wrapped__' not in vars(fn) and _function(row['source'], fn) is fn)

    def _guards(self, row, signature, args, kwargs, receiver_owner):
        from .stateful_guard import _input, _ordinary
        bound = signature.bind(*args, **kwargs); bound.apply_defaults()
        if receiver_owner is not None:
            _need(type(bound.arguments[next(iter(signature.parameters))]) is receiver_owner,
                  'caller receiver class differs')
        for guard in row.get('guards', ()):
            value = _input(guard['path'], bound.arguments); kind = guard['kind']
            if kind == 'ORDINARY_RECEIVER':
                _ordinary(value)
            elif kind == 'EXACT_BOOL':
                _need(type(value) is bool, 'caller bool guard differs')
            elif kind == 'EXACT_NUMBER':
                _need(type(value) is int or type(value) is float and math.isfinite(value),
                      'caller exact number guard differs')
                _need('min' not in guard or value >= guard['min'], 'caller number below its scope bound')
                _need('max' not in guard or value <= guard['max'], 'caller number above its scope bound')
            elif kind == 'EXACT_ARRAY_ROOT':
                from .array_copy_guard import guard_copy_inputs
                guard_copy_inputs((value,))
                _need(value.dtype.name == 'float64' and value.ndim == 2 and value.flags.c_contiguous,
                      'caller exact array root guard differs')
                _need('shape' not in guard or list(value.shape) == guard['shape'], 'caller root shape differs')
            else:
                raise ValueError('unsupported caller guard kind')
            _need('value' not in guard or type(value) in (bool, int, float, str) and value == guard['value'],
                  'caller literal guard differs')
        return bound

    def _closed(self, row, original, signature, args, kwargs, receiver_owner):
        if not _raw_metadata(original):
            return False
        from .stateful_guard import _ordinary
        proof = row.get('proof', self.spec.get('proof', {}))
        if (not all(proof.get(key) is True for key in
                ('structural_operations_complete', 'effect_profile_closed', 'unknown_calls_barriered'))
                or threading.get_ident() != self.owner or threading.active_count() != 1
                or sys.gettrace() is not None or sys.getprofile() is not None):
            return False
        monitor = getattr(sys, 'monitoring', None)
        if monitor is not None and any(monitor.get_tool(i) is not None for i in range(6)):
            return False
        if not all(_literal(v) for v in (* (original.__defaults__ or ()), *(original.__kwdefaults__ or {}).values())):
            return False
        _need(list(signature.parameters) == row['parameters'], 'caller formal ABI differs')
        bound = self._guards(row, signature, args, kwargs, receiver_owner)
        routes = row.get('lookup_routes')
        if type(routes) is not list:
            return False
        routed = set()
        for route in routes:
            path, identity = route['path'], route['boundary_id']
            if (type(path) is not list or len(path) != 2 or path[0] not in bound.arguments
                    or identity not in row.get('allowed_call_roles', ())):
                return False
            target = next((r for r in self.bindings if r['id'] == identity), None)
            receiver = bound.arguments[path[0]]
            if (target is None or type(target['owner']) is not type or type(receiver) is not target['owner']
                    or target['attribute'] != path[1] or path[1] in _ordinary(receiver)
                    or vars(target['owner']).get(path[1]) is not target['installed']):
                return False
            routed.add(identity)
        if routed != set(row.get('allowed_call_roles', ())):
            return False
        required = [row['id'], *row.get('allowed_call_roles', ())]
        return all(key in self.rows and self._live(self.rows[key]) for key in required)

    def _can_keep(self, identity):
        if self.suspended or not self.activations:
            return False
        try:
            for index, frame in enumerate(self.activations):
                target = self.activations[index+1]['row']['id'] if index+1 < len(self.activations) else identity
                if (not frame['closed'] or frame['tainted']
                        or target not in frame['row'].get('allowed_call_roles', ())
                        or not self._closed(frame['row'], frame['original'], frame['signature'],
                            frame['args'], frame['kwargs'], frame['receiver_owner'])):
                    return False
            return True
        except (ValueError, TypeError, KeyError, AttributeError, OSError):
            return False

    def _wrap(self, row, original, receiver_owner, signature):
        _need(_raw_metadata(original), 'function metadata must use exact builtin containers')
        if row['role'] == 'KERNEL':
            # The shared kernel guard still reads annotations. Reject that
            # unsupported installation rather than execute deferred user code.
            _need(getattr(original, '__annotate__', None) is None, 'deferred kernel annotations unsupported by shared guard')
            from .stateful_guard import _lookup
            callee_guards = tuple(self.session.contract['callee_guards'])
            def exact_kernel_metadata(args=None, kwargs=None):
                if not _raw_metadata(original):
                    return False
                try:
                    if args is None:
                        bound = {}
                    else:
                        values = signature.bind(*args, **kwargs); values.apply_defaults()
                        bound = values.arguments
                    for guard in callee_guards:
                        path = guard['path']
                        if args is None and path[0] in signature.parameters:
                            continue
                        if not _raw_metadata(_lookup(path, bound, original.__globals__)):
                            return False
                    return True
                except (ValueError, TypeError, KeyError, AttributeError, StopIteration):
                    return False
            _need(exact_kernel_metadata(), 'callee metadata must use exact builtin containers')
            guarded = self.session.wrap(original, owner=receiver_owner, exclusive_inputs=True)
            def original_fallback(reason, args, kwargs):
                self._lose(reason, taint=True)
                stats = guarded.scar_stats
                stats['fallback_calls'] += 1
                stats['reasons'][reason] = stats['reasons'].get(reason, 0)+1
                return original(*args, **kwargs)
            def isolated(reason, args, kwargs):
                self.stats['isolated_kernel_attempts'] += 1
                self.stats['fresh_kernel_calls'] += 1
                reasons = self.stats['isolated_kernel_reasons']
                reasons[reason] = reasons.get(reason, 0)+1
                previous_native = guarded.scar_stats['native_calls']
                previous_fallbacks = guarded.scar_stats['fallback_calls']
                try:
                    try:
                        self.session.release('isolated_kernel_entry')
                    except Exception:
                        return original_fallback('isolated_kernel_entry_cleanup_failed', args, kwargs)
                    if not exact_kernel_metadata(args, kwargs):
                        return original_fallback('function_metadata_not_exact', args, kwargs)
                    return guarded(*args, **kwargs)
                finally:
                    self.stats['isolated_native_calls'] += guarded.scar_stats['native_calls']-previous_native
                    fallbacks = guarded.scar_stats['fallback_calls']-previous_fallbacks
                    self.stats['isolated_original_fallback_calls'] += fallbacks
                    if fallbacks:
                        self._lose('isolated_kernel_original_fallback', taint=True)
                    try:
                        self.session.release('isolated_kernel_exit')
                    except Exception:
                        self._lose('isolated_kernel_exit_cleanup_failed')
            def capacity(args, kwargs):
                self.stats['capacity_kernel_attempts'] += 1
                self.stats['fresh_kernel_calls'] += 1
                previous_native = guarded.scar_stats['native_calls']
                previous_fallbacks = guarded.scar_stats['fallback_calls']
                try:
                    if not self._forget_capacity('capacity_kernel_entry'):
                        return original_fallback('capacity_kernel_entry_failed', args, kwargs)
                    if not exact_kernel_metadata(args, kwargs):
                        return original_fallback('function_metadata_not_exact', args, kwargs)
                    return guarded(*args, **kwargs)
                finally:
                    native = guarded.scar_stats['native_calls']-previous_native
                    fallbacks = guarded.scar_stats['fallback_calls']-previous_fallbacks
                    self.stats['capacity_native_calls'] += native
                    self.stats['capacity_original_fallback_calls'] += fallbacks
                    if native and not fallbacks:
                        retained = self._forget_capacity('capacity_kernel_exit') and self._capacity_outer()
                    else:
                        retained = False
                        self._lose('capacity_kernel_original_fallback', taint=True)
                        self.session.bindings = ()
                    if retained:
                        self.stats['capacity_retained_intervals'] += 1
                    else:
                        self._release_capacity('capacity_kernel_exit_release')
                    self._sync_capacity_counts()
            def call(*args, **kwargs):
                if not self.active or threading.get_ident() != self.owner:
                    return original(*args, **kwargs)
                if self.unqualified_kernel_policy == 'RETAIN_PRIVATE_CAPACITY_FULL_SNAPSHOT':
                    return capacity(args, kwargs)
                keep = None
                if self.unqualified_kernel_policy == 'FRESH_ISOLATED_NATIVE':
                    if self.suspended:
                        self.stats['tainted_kernel_calls'] += 1
                        return isolated('suspended_unknown_call', args, kwargs)
                    if any(frame['tainted'] for frame in self.activations):
                        self.stats['tainted_kernel_calls'] += 1
                        return isolated('tainted_activation', args, kwargs)
                    keep = self._can_keep(row['id'])
                    if not keep:
                        return isolated('unscoped_or_unclosed_caller', args, kwargs)
                if not exact_kernel_metadata(args, kwargs):
                    return original_fallback('function_metadata_not_exact', args, kwargs)
                if self.suspended or any(frame['tainted'] for frame in self.activations):
                    self.stats['tainted_kernel_calls'] += 1
                    return original_fallback('tainted_or_unknown_caller', args, kwargs)
                keep = self._can_keep(row['id']) if keep is None else keep
                try:
                    if keep:
                        self.stats['retention_eligible_kernel_calls'] += 1
                    else:
                        self.stats['fresh_kernel_calls'] += 1
                        if not self._invalidate('caller_retention_not_admitted'):
                            return original_fallback('session_invalidation_failed', args, kwargs)
                    previous_fallbacks = guarded.scar_stats['fallback_calls']
                    try:
                        return guarded(*args, **kwargs)
                    finally:
                        if guarded.scar_stats['fallback_calls'] != previous_fallbacks:
                            self._lose('kernel_original_fallback', taint=True)
                finally:
                    if not keep:
                        try:
                            self.session.release('unscoped_or_unclosed_kernel_exit')
                        except Exception:
                            self._lose('unscoped_or_unclosed_kernel_exit_cleanup_failed')
            call.scar_stats = guarded.scar_stats
        elif row['role'] == 'CALLER':
            def call(*args, **kwargs):
                if not self.active or threading.get_ident() != self.owner:
                    return original(*args, **kwargs)
                self.stats['caller_calls'] += 1
                outer = not self.activations
                if (not outer and self.activations[-1]['closed']
                        and row['id'] not in self.activations[-1]['row'].get('allowed_call_roles', ())):
                    self._lose('unadmitted_nested_caller', taint=True)
                if outer:
                    if self.unqualified_kernel_policy == 'RETAIN_PRIVATE_CAPACITY_FULL_SNAPSHOT':
                        self._forget_capacity('outer_caller_entry')
                    else:
                        self._invalidate('outer_caller_entry')
                try:
                    closed = (False if self.unqualified_kernel_policy == 'RETAIN_PRIVATE_CAPACITY_FULL_SNAPSHOT'
                              else self._closed(row, original, signature, args, kwargs, receiver_owner))
                except (ValueError, TypeError, KeyError, AttributeError, OSError):
                    self.stats['guard_rejections'] += 1; closed = False
                frame = dict(row=row, original=original, signature=signature, args=args, kwargs=kwargs,
                    receiver_owner=receiver_owner, closed=closed, tainted=False)
                self.activations.append(frame)
                try:
                    return original(*args, **kwargs)
                finally:
                    self.activations.pop()
                    if outer:
                        if self.unqualified_kernel_policy == 'RETAIN_PRIVATE_CAPACITY_FULL_SNAPSHOT':
                            self._release_capacity('outer_caller_exit')
                        else:
                            try:
                                self.session.release('outer_caller_exit')
                            except Exception:
                                self._lose('outer_caller_exit_cleanup_failed')
        elif row['role'] == 'COPY':
            def call(*args, **kwargs):
                if not self.active or threading.get_ident() != self.owner:
                    return original(*args, **kwargs)
                self.stats['copy_calls'] += 1
                if self.unqualified_kernel_policy == 'RETAIN_PRIVATE_CAPACITY_FULL_SNAPSHOT':
                    self._forget_capacity('capacity_copy_entry')
                    try:
                        return original(*args, **kwargs)
                    finally:
                        self._forget_capacity('capacity_copy_exit')
                        if not self._capacity_outer():
                            self._release_capacity('capacity_copy_unscoped_exit')
                try:
                    _need(self._can_keep(row['id']) and self._live(row), 'copy outside a closed live caller')
                    bound = signature.bind(*args, **kwargs); bound.apply_defaults()
                    _need(len(bound.arguments) == 1 and original.__code__.co_argcount == 1,
                          'copy boundary requires the admitted unary array ABI')
                    actual = next(iter(bound.arguments.values()))
                    from .array_copy_guard import guard_copy_inputs
                    guard_copy_inputs((actual,))
                    _need(actual.dtype.name == 'float64' and actual.ndim == 2 and actual.flags.c_contiguous,
                          'copy boundary actual array profile differs')
                    effects = [effect for effect in self.session.effects
                        if effect['source'] == row['source'] and effect['shape'] == list(actual.shape)]
                    _need(len(effects) == 1, 'copy summary is not uniquely retained by the session contract')
                    self.session.note_copy(effects[0], original, actual)
                except (ValueError, TypeError, KeyError, AttributeError, OSError):
                    self._lose('copy_boundary_not_admitted', taint=True)
                try:
                    return original(*args, **kwargs)
                except BaseException:
                    self._lose('copy_boundary_raised', taint=True)
                    raise
        else:
            def call(*args, **kwargs):
                if not self.active or threading.get_ident() != self.owner:
                    return original(*args, **kwargs)
                self.stats['barrier_calls'] += 1
                self._lose('unknown_call_before', taint=True)
                self.suspended += 1
                try:
                    return original(*args, **kwargs)
                finally:
                    self.suspended -= 1
                    self._lose('unknown_call_after', taint=True)
        wrapper = _decorate(original, call)
        if not hasattr(wrapper, 'scar_stats'):
            wrapper.scar_stats = dict(native_calls=0, fallback_calls=0, reasons={})
        return wrapper

    def attach(self, module, name, *, natural):
        if not self.active or id(module) in self.seen:
            return
        from .entry_runner import _file
        from .stateful_guard import _function, _function_state
        if type(module) is not ModuleType or _file(module) not in self.sources:
            return
        self.seen.add(id(module))
        spec = vars(module).get('__spec__')
        if not natural and (spec is None or getattr(spec, '_initializing', False)
                            or type(getattr(spec, 'loader', None)) is not SourceFileLoader):
            self.frontier.append('preloaded_source_not_completed_standard_module'); return
        for row in self.rows.values():
            if Path(row['source']['path']).resolve() != _file(module):
                continue
            try:
                owner = module
                for part in row['owner_path']:
                    _need(type(owner) in (ModuleType, type), 'custom boundary owner')
                    owner = vars(owner)[part]
                _need(type(owner) in (ModuleType, type), 'custom boundary owner')
                descriptor = vars(owner)[row['attribute']]
                if row['descriptor'] == 'STATICMETHOD':
                    _need(type(owner) is type and type(descriptor) is staticmethod, 'staticmethod descriptor differs')
                    original, receiver_owner = descriptor.__func__, None
                else:
                    _need(type(descriptor) is FunctionType
                          and (type(owner) is type) == (row['descriptor'] == 'METHOD'), 'ordinary descriptor differs')
                    original, receiver_owner = descriptor, owner if type(owner) is type else None
                _need(_raw_metadata(original), 'function metadata must use exact builtin containers')
                _function(row['source'], original)
                _need('__signature__' not in vars(original) and '__wrapped__' not in vars(original),
                      'custom callable metadata unsupported')
                signature = _signature(original)
                wrapper = self._wrap(row, original, receiver_owner, signature)
                installed = staticmethod(wrapper) if row['descriptor'] == 'STATICMETHOD' else wrapper
                setattr(owner, row['attribute'], installed)
                self.bindings.append(dict(id=row['id'], owner=owner, attribute=row['attribute'], descriptor=descriptor,
                    installed=installed, binding=wrapper, original=original, state=_function_state(original),
                    module=module, module_name=name, natural_import=natural, stats=wrapper.scar_stats))
            except (ValueError, TypeError, KeyError, AttributeError, OSError) as error:
                self.frontier.append(row['id']+':'+str(error))

    def finish(self):
        if self.finished:
            return
        self.finished = True
        self.active = False
        replacements = {id(row['binding']): row for row in self.bindings}
        try:
            for module in tuple(sys.modules.values()):
                if type(module) is not ModuleType:
                    continue
                for key, value in tuple(vars(module).items()):
                    row = replacements.get(id(value))
                    if row is not None and value is row['binding']:
                        vars(module)[key] = row['original']
            for row in reversed(self.bindings):
                if vars(row['owner']).get(row['attribute']) is row['installed']:
                    setattr(row['owner'], row['attribute'], row['descriptor'])
        finally:
            if self.owns_session:
                try:
                    self.session.close()
                except Exception as error:
                    self.frontier.append('session_close_failed:'+type(error).__name__)
                finally:
                    if self.capacity_policy is not None:
                        self.session.bindings = ()
                    self._sync_capacity_counts()

    @contextmanager
    def install(self):
        if not self.active:
            yield self
            return
        from .entry_runner import _file
        original_meta = sys.meta_path
        previous = list(original_meta)
        try:
            if sum(finder is PathFinder for finder in previous) == 1:
                proxy = _Finder(self)
                sys.meta_path = [proxy if finder is PathFinder else finder for finder in previous]
                for name, module in tuple(sys.modules.items()):
                    if type(module) is ModuleType and _file(module) in self.sources:
                        self.attach(module, name, natural=False)
            else:
                self.frontier.append('standard_pathfinder_position_unavailable')
            yield self
        finally:
            try:
                self.finish()
            finally:
                original_meta[:] = previous
                sys.meta_path = original_meta

    @property
    def report(self):
        return dict(schema='scar.workspace-session-scope-execution', schema_version=1,
            statistics=deepcopy(self.stats), installed_bindings=len(self.bindings), frontier=list(self.frontier),
            unqualified_kernel_policy=self.unqualified_kernel_policy,
            capacity_policy_proof_id=None if self.capacity_policy is None else self.capacity_policy['proof_id'],
            preservation_scope=('PRIVATE_CAPACITY_ONLY_SOURCE_BOUND_OUTER_NO_CONTENT_OR_EXTERNAL_ROOT_REFS'
                if self.capacity_policy is not None else 'SOURCE_CLOSED_LIVE_CALLER_ACTIVATIONS_ONLY'),
            source_call_coverage_proven_by_installer=False, arbitrary_internal_writes_intercepted=False,
            default_backend='NO_OP', deployment_authorized=False)


scope = SessionScope
__all__ = ['SessionScope', 'scope', 'SCHEMA']
