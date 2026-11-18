"""Activate one source-proved dead result item in its original caller."""

from base64 import b64decode
from builtins import isinstance as _isinstance
from hashlib import file_digest
import marshal
import os
from pathlib import Path
from threading import get_ident
from types import CodeType, FunctionType

from scar.compiler.plan_dead_result_item import SCHEMA, seal_site
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.runtime.csr_region_entry import _observer_active, interpreter_context
from scar.runtime.entry_runner import _file
from scar.trace.code_snapshot import code_key


def _sealed_code(plan, role, identity):
    code = marshal.loads(b64decode(plan['sealed'][role], validate=True))
    if (type(code) is not CodeType or code.co_filename != plan['site']['filename']
            or code.co_qualname != plan['facts'][role + '_name'] or code.co_freevars
            or code_key(code) != plan['facts'][identity]):
        raise ValueError('sealed dead-item code does not match its lowering identity')
    return code


def admit(path, sha256, selected, selection_path, selection_sha):
    reference = artifact_reference(path, SCHEMA)
    if reference['sha256'] != sha256:
        raise ValueError('dead-item plan fingerprint differs')
    plan = read_artifact(reference)
    source = plan['source_selection']
    if (plan['schema_version'] not in (1, 2) or plan['auto_apply'] is not False
            or source['uri'] != str(Path(selection_path).resolve()) or source['sha256'] != selection_sha
            or plan['interpreter'] != interpreter_context()):
        raise ValueError('dead-item plan has a different entry or interpreter')
    for member in (plan['provider']['source'], plan['provider']['native']):
        with Path(member['uri']).open('rb') as stream:
            if (os.fstat(stream.fileno()).st_size != member['size_bytes']
                    or file_digest(stream, 'sha256').hexdigest() != member['sha256']):
                raise ValueError('dead-item provider artifact identity differs')
    if plan['schema_version'] == 2:
        read_artifact(plan['site']['source'], json_document=False)
        caller = _sealed_code(plan, 'caller', 'revised_caller_code_key')
        callee = _sealed_code(plan, 'callee', 'specialized_callee_code_key')
    else:
        facts, caller, callee = seal_site(plan['site'], plan['interpreter']['optimize'])
        if facts != plan['facts']:
            raise ValueError('dead-item proof differs from current source lowering')
    return DeadItemScope(plan, caller, callee, selected['matrix']['size'])


class DeadItemScope:
    def __init__(self, plan, revised, specialized, size):
        self.source = Path(plan['site']['filename'])
        self.facts, self.provider = plan['facts'], plan['provider']
        self.revised, self.specialized, self.size = revised, specialized, size
        self.owner = get_ident()
        self.active, self.bindings, self.frontier = True, [], []
        self.stats = dict(elided=0, retained_first=0, fallback=0)

    def attach(self, module, name, *, natural):
        if not natural or _file(module) != self.source or self.bindings:
            return
        from numpy import ndarray
        import numpy as np
        from numpy._core import _multiarray_umath as native
        from numpy.lib import _arraysetops_impl as arraysets
        from scipy.sparse import csr_array, csr_matrix

        namespace = vars(module)
        facts = self.facts
        caller = namespace.get(facts['caller_name'])
        callee = namespace.get(facts['callee_name'])
        flatten = namespace.get(facts['flatten_binding'])
        producer, indices = np.setdiff1d, np.arange
        if (type(caller) is not FunctionType or type(callee) is not FunctionType or type(flatten) is not FunctionType
                or caller.__globals__ is not namespace or callee.__globals__ is not namespace
                or code_key(caller.__code__) != facts['caller_code_key']
                or code_key(callee.__code__) != facts['callee_code_key']
                or code_key(flatten.__code__) != facts['flatten_code_key']
                or facts['helper'] in namespace or namespace.get(facts['numpy_binding']) is not np
                or namespace.get(facts['ndarray_binding']) is not ndarray
                or self.provider['version'] != np.__version__ or producer is not arraysets.setdiff1d
                or indices is not native.arange or type(producer).__name__ != '_ArrayFunctionDispatcher'
                or code_key(producer.__wrapped__.__code__) != self.provider['producer_code_key']
                or _file(arraysets) != Path(self.provider['source']['uri'])
                or _file(native) != Path(self.provider['native']['uri'])):
            self.frontier.append('DEAD_ITEM_PROVIDER_BINDING_CHANGED')
            return

        original_code, callee_code, flatten_code = caller.__code__, callee.__code__, flatten.__code__
        specialized = FunctionType(self.specialized, namespace, callee.__name__)
        builtin_map = caller.__builtins__
        selected_types = (np.dtype('int32'), np.dtype('int64'))
        int32 = np.int32
        slots = (facts['matrix_argument'], facts['rows_argument'], facts['dead_argument'])
        # Direct NumPy globals used by the retained provider body are live dependencies.
        producer_globals = producer.__wrapped__.__globals__
        producer_bindings = tuple((key, producer_globals[key]) for key in producer.__wrapped__.__code__.co_names
            if key in producer_globals)
        primed = False

        def invoke(function, *args):
            nonlocal primed
            if (not self.active or get_ident() != self.owner or _observer_active()
                    or function is not callee or callee.__code__ is not callee_code or flatten.__code__ is not flatten_code
                    or namespace.get(facts['flatten_binding']) is not flatten
                    or namespace.get(facts['numpy_binding']) is not np or namespace.get(facts['ndarray_binding']) is not ndarray
                    or namespace.get('isinstance', builtin_map.get('isinstance')) is not _isinstance
                    or np.ndarray is not ndarray or np.setdiff1d is not producer or np.arange is not indices
                    or np.int32 is not int32
                    or any(producer_globals.get(key) is not value for key, value in producer_bindings)):
                self.stats['fallback'] += 1
                return function(*args)
            matrix, rows, dead = (args[index] for index in slots)
            if (dead is not None or type(matrix) not in (csr_matrix, csr_array)
                    or 'shape' in vars(matrix) or matrix.shape != (self.size, self.size)
                    or type(rows) is not ndarray or rows.ndim != 1 or rows.dtype not in selected_types
                    or not rows.flags.c_contiguous or not rows.flags.aligned):
                self.stats['fallback'] += 1
                return function(*args)
            if not primed:
                result = function(*args)
                self.stats['retained_first'] += 1
                primed = type(result[facts['return_slot']]) is ndarray
                return result
            result = specialized(*args)
            self.stats['elided'] += 1
            return result

        namespace[facts['helper']] = invoke
        caller.__code__ = self.revised
        self.bindings.append(dict(module_name=name, natural_import=natural, namespace=namespace,
            function=caller, initial_code=original_code, helper=invoke))

    def finish(self):
        self.active = False
        for row in self.bindings:
            if row['function'].__code__ is self.revised:
                row['function'].__code__ = row['initial_code']
            helper = self.facts['helper']
            if row['namespace'].get(helper) is row['helper']:
                del row['namespace'][helper]

    def receipt(self):
        return dict(self.stats, installed_bindings=len(self.bindings), frontier=self.frontier)
