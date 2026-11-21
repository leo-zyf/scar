"""Attach an exact native profile at its observed Python boundary."""

from functools import wraps
from importlib.machinery import SourceFileLoader
import os
from pathlib import Path
from time import perf_counter_ns
from types import BuiltinFunctionType, FunctionType

from scar.ir.execution_recipe import read_artifact
from scar.trace.code_snapshot import code_key
from .entry_runner import _Scope, _arguments, _execute
from .native_numpy_partition import load


def _guard(original, report, extension, active):
    import numpy as np
    from numpy._core import _multiarray_umath as core

    source = report['source']
    if (type(original) is not FunctionType or original.__qualname__ != source['qualname']
            or Path(original.__code__.co_filename).resolve() != Path(source['source']['uri'])
            or code_key(original.__code__) != source['code_id']):
        raise ValueError('source identity differs')
    read_artifact(source['source'], json_document=False)

    numpy = report['numpy']
    namespace = vars(np)
    carrier = vars(core)
    read_policy = carrier.get('_get_extobj_dict')
    if (Path(np.__file__).resolve() != Path(numpy['module']['uri']).resolve()
            or Path(core.__file__).resolve() != Path(numpy['extension']['uri']).resolve()
            or np.__version__ != numpy['version']
            or type(read_policy) is not BuiltinFunctionType or read_policy.__self__ is not core):
        raise ValueError('provider binding differs')
    for key in ('module', 'extension'):
        read_artifact(numpy[key], json_document=False)
    add, multiply = carrier.get('add'), carrier.get('multiply')
    expected_type = namespace.get('ufunc')
    for symbol, ufunc in (('add', add), ('multiply', multiply)):
        expected = numpy['ufuncs'][symbol]
        if (type(ufunc) is not expected_type or namespace.get(symbol) is not ufunc
                or dict(name=ufunc.__name__, nin=ufunc.nin, nout=ufunc.nout,
                        signature=ufunc.signature, types=ufunc.types) != expected):
            raise ValueError('source ufunc binding differs')
    native = load(extension).try_apply
    keyword = source['parameters']['out']
    original_code = original.__code__
    globals_ = original.__globals__
    builtins_ = original.__builtins__
    expected_policy = dict(report['profile']['numpy_errstate'],
                           call=None, bufsize=carrier['UFUNC_BUFSIZE_DEFAULT'])
    stats = dict(native_calls=0, fallback_calls=0, reasons={})

    @wraps(original, assigned=('__module__', '__name__', '__qualname__', '__doc__'))
    def binding(*args, **kwargs):
        if not active():
            return original(*args, **kwargs)
        if len(args) != 3 or len(kwargs) != 1 or keyword not in kwargs:
            reason = 'CALL_SHAPE'
        elif (original.__code__ is not original_code or globals_.get('np') is not np
                or globals_.get('len', builtins_.get('len')) is not len
                or globals_.get('enumerate', builtins_.get('enumerate')) is not enumerate
                or namespace.get('add') is not add or namespace.get('multiply') is not multiply
                or carrier.get('add') is not add or carrier.get('multiply') is not multiply
                or carrier.get('_get_extobj_dict') is not read_policy):
            reason = 'PROVIDER_BINDING'
        elif read_policy() != expected_policy:
            reason = 'NUMPY_ERROR_POLICY'
        else:
            output = kwargs[keyword]
            result = native(args[0], args[1], args[2], output)
            if result is not NotImplemented:
                if result is not output:
                    raise RuntimeError('native publication violated the output identity contract')
                stats['native_calls'] += 1
                return result
            reason = 'NATIVE_GUARD'
        stats['fallback_calls'] += 1
        reasons = stats['reasons']
        reasons[reason] = reasons.get(reason, 0) + 1
        return original(*args, **kwargs)

    if getattr(original, '__annotate__', None) is not None:
        binding.__annotate__ = original.__annotate__
    else:
        binding.__annotations__ = original.__annotations__
    return binding, stats


class _LiveScope(_Scope):
    def __init__(self, profile, extension):
        self.profile, self.extension = profile, extension
        self.source = Path(profile['source']['source']['uri']).resolve()
        self.active, self.bindings, self.seen, self.frontier = True, [], set(), []

    def attach(self, module, name, *, natural):
        if not self.active or id(module) in self.seen or name != self.profile['source']['module']:
            return
        self.seen.add(id(module))
        spec = module.__spec__
        if not natural and (spec is None or getattr(spec, '_initializing', False)
                            or type(spec.loader) is not SourceFileLoader):
            self.frontier.append('PRELOADED_SOURCE_BOUNDARY')
            return
        symbol = self.profile['source']['symbol']
        original = vars(module).get(symbol)
        started = perf_counter_ns()
        try:
            binding, stats = _guard(original, self.profile, self.extension, lambda: self.active)
        except (ImportError, OSError, ValueError) as error:
            self.frontier.append('ATTACHMENT_DECLINED: ' + str(error))
            return
        setattr(module, symbol, binding)
        self.bindings.append(dict(owner=module, attribute=symbol, descriptor=original, installed=binding,
                                  binding=binding, original=original, stats=stats, module_name=name,
                                  natural_import=natural, binding_ns=perf_counter_ns() - started))


def run(entry, argv, profile, extension, *, exclusive_normal_completion=False):
    """Run a preverified pair with exclusive arrays and unobserved raw FP status."""
    entry, argv = _arguments(entry, argv)
    if exclusive_normal_completion is not True:
        return _execute(entry, argv, None, 'NO_EXPLICIT_LIVE_TRIAL_CONTRACT')
    if sorted(os.sched_getaffinity(0)) != profile['cpu_ids']:
        return _execute(entry, argv, None, 'NO_MATCHING_CPU_SCOPE')
    if Path(profile['entry']['uri']).resolve() != entry:
        raise ValueError('entry is outside the live profile')
    read_artifact(profile['entry'], json_document=False)
    scope = _LiveScope(profile, extension)
    report = _execute(entry, argv, scope, 'EXPLICIT_NATIVE_PARTITION_TRIAL')
    report['binding_ns'] = sum(row['binding_ns'] for row in scope.bindings)
    return report
