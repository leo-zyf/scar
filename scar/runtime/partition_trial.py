"""Bind a partition trial to the original callable and guard each invocation."""

from functools import wraps
from hashlib import sha256
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import BuiltinFunctionType, FunctionType
from time import perf_counter_ns

from scar.codegen.partition_multiply import load_partition_multiply
from scar.ir.execution_recipe import read_artifact
from scar.trace.code_snapshot import code_key
from .entry_runner import _Scope, _arguments, _execute


def _owner(array, ndarray):
    while type(array.base) is ndarray:
        array = array.base
    return array if array.base is None and array.flags.owndata else None


def _trial_material(receipt, *, replay):
    trial = read_artifact(receipt)
    (candidate,) = read_artifact(trial['plan'])['candidates']
    source = candidate['source']
    read_artifact(trial['successor'])
    if (trial.get('schema') != 'scar.partition-copy-trial' or trial.get('schema_version') != 1
            or trial.get('deployment_authorized') is not False or trial['source'] != source
            or source.get('schema_version') != 2
            or source.get('builtin_bindings') not in ({'enumerate': 'GLOBAL_OR_BUILTIN'},
                {'enumerate': 'GLOBAL_OR_BUILTIN', 'len': 'GLOBAL_OR_BUILTIN'})
            or trial['successor'] != candidate['graphs']['after'] or candidate['decision'] != 'KEEP'):
        raise ValueError('trial requires its sealed source, contract and successor')
    if replay:
        from scar.optimization.partition_copy import check

        if check(trial['plan']['uri']) != 1:
            raise ValueError('the trial must replay exactly one derived region')
        read_artifact(trial['plan'])
    return trial


def bind_trial(original, receipt, *, exclusive_normal_completion=False, parent_image=None):
    """Replay the trial and bind it under an explicit normal-completion contract."""
    if exclusive_normal_completion is not True:
        raise ValueError('explicit exclusive normal-completion trial contract required')
    return _bind_trial(original, _trial_material(receipt, replay=True), parent_image=parent_image)


def _bind_trial(original, trial, *, parent_image):
    import numpy as np
    from numpy._core import _multiarray_umath

    source = trial['source']
    if (type(original) is not FunctionType or source['qualname'] != original.__qualname__
            or source['code_id'] != code_key(original.__code__)
            or Path(original.__code__.co_filename).resolve() != Path(source['path'])
            or sha256(Path(source['path']).read_bytes()).hexdigest() != source['source_sha256']):
        raise ValueError('callable does not match the sealed trial source')

    read_policy = vars(_multiarray_umath).get('_get_extobj_dict')
    if (type(read_policy) is not BuiltinFunctionType or read_policy.__self__ is not _multiarray_umath
            or read_policy.__name__ != '_get_extobj_dict'):
        raise ValueError('the NumPy floating-point policy provider is unavailable')
    carrier = vars(_multiarray_umath)
    numpy_namespace = vars(np)
    multiply = carrier.get('multiply')
    ufunc = numpy_namespace.get('ufunc')
    if (type(ufunc) is not type or ufunc.__module__ != 'numpy' or ufunc.__name__ != 'ufunc'
            or type(multiply) is not ufunc or multiply.__name__ != 'multiply'
            or (multiply.nin, multiply.nout, multiply.signature) != (2, 1, None)
            or 'DD->D' not in multiply.types):
        raise ValueError('exact NumPy multiply provider is unavailable')
    native = load_partition_multiply(trial['native'], parent_image=parent_image).apply_without_copy
    keyword = source['inputs'][3]
    dtype = np.dtype('complex128')
    ndarray = np.ndarray
    original_code = original.__code__
    namespace = original.__globals__
    builtins = original.__builtins__
    selection = ((Ellipsis, 0), (Ellipsis, 1))
    status = dict(native_calls=0, fallback_calls=0, logical_copy_bytes_elided=0)

    def applicable(array, weights, indexing, output):
        if (type(indexing) not in (tuple, list) or len(indexing) != 2
                or any(type(key) is not tuple or len(key) != 2 or key[0] is not Ellipsis
                       or type(key[1]) is not int or key[1] != index for index, key in enumerate(indexing))
                or any(type(value) is not ndarray or value.dtype != dtype or not value.flags.aligned
                       for value in (array, weights, output))
                or array.ndim < 2 or array.shape[-1] != 2
                or any(extent < 2 for extent in array.shape)
                or array.shape != output.shape or array.strides != output.strides
                or not output.flags.writeable or weights.shape != (2, 2)
                or not weights.flags.c_contiguous):
            return False
        stride = dtype.itemsize
        for axis in sorted(range(array.ndim), key=array.strides.__getitem__):
            if array.strides[axis] != stride:
                return False
            stride *= array.shape[axis]
        output_owner = _owner(output, ndarray)
        input_owner = _owner(array, ndarray)
        weights_owner = _owner(weights, ndarray)
        if not (output_owner is not None and input_owner is not None and weights_owner is not None
                and output_owner is not input_owner and output_owner is not weights_owner
                and not output.flags.writebackifcopy and not output_owner.flags.writebackifcopy
                and original.__code__ is original_code
                and namespace.get('len', builtins.get('len')) is len
                and namespace.get('enumerate', builtins.get('enumerate')) is enumerate
                and numpy_namespace.get('multiply') is multiply):
            return False
        policy = read_policy()
        return policy['call'] is None and all(policy[name] == 'ignore'
            for name in ('divide', 'over', 'under', 'invalid'))

    @wraps(original)
    def guarded(*args, **kwargs):
        if len(args) == 3 and len(kwargs) == 1 and keyword in kwargs:
            array, weights, indexing = args
            output = kwargs[keyword]
            if applicable(array, weights, indexing, output):
                result = native(array, weights, selection, output, multiply)
                status['native_calls'] += 1
                status['logical_copy_bytes_elided'] += array.nbytes
                return result
        status['fallback_calls'] += 1
        return original(*args, **kwargs)

    return guarded, status


class PartitionScope(_Scope):
    def __init__(self, receipt, *, parent_image=None, parent_replayed=False):
        self.trial = _trial_material(receipt, replay=parent_replayed is not True)
        self.parent_image = parent_image
        self.source = Path(self.trial['source']['path'])
        self.active, self.bindings, self.frontier, self.seen = True, [], [], set()

    def attach(self, module, name, *, natural):
        if not self.active or id(module) in self.seen:
            return
        self.seen.add(id(module))
        spec = module.__spec__
        if not natural and (spec is None or getattr(spec, '_initializing', False)
                            or type(spec.loader) is not SourceFileLoader):
            self.frontier.append('preloaded_source_not_a_completed_standard_module')
            return
        attribute = self.trial['source']['qualname']
        original = vars(module).get(attribute)
        start = perf_counter_ns()
        try:
            guarded, stats = _bind_trial(original, self.trial, parent_image=self.parent_image)
        except (ValueError, OSError, ImportError) as error:
            self.frontier.append('partition_source_rejected:' + str(error))
            return
        binding_ns = perf_counter_ns() - start

        @wraps(original)
        def binding(*args, **kwargs):
            callback = guarded if self.active else original
            return callback(*args, **kwargs)

        stats['reasons'] = {}
        setattr(module, attribute, binding)
        self.bindings.append(dict(owner=module, attribute=attribute, descriptor=original, installed=binding,
            binding=binding, original=original, stats=stats, module_name=name, natural_import=natural,
            binding_ns=binding_ns))

    def worker_evidence(self):
        return dict(source=str(self.source), code_id=self.trial['source']['code_id'],
            frontier=list(self.frontier), bindings=[dict(module=row['module_name'],
                natural_import=row['natural_import'], binding_ns=row['binding_ns'], **row['stats'])
                for row in self.bindings])


def run_partition_trial(entry, argv, receipt, *, exclusive_normal_completion=False):
    """Load the source at its normal import position and run the unchanged entry."""
    entry, argv = _arguments(entry, argv)
    if not exclusive_normal_completion:
        return _execute(entry, argv, None, 'partition_trial_not_requested')
    scope = PartitionScope(receipt)
    result = _execute(entry, argv, scope, 'explicit_partition_trial')
    result['logical_copy_bytes_elided'] = sum(row['stats']['logical_copy_bytes_elided'] for row in scope.bindings)
    return result
