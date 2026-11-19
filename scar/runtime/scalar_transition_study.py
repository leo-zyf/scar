"""Exercise a native transition at one source-bound call site in an isolated study."""

import argparse
import ast
import builtins
from collections import Counter, namedtuple
from _collections import _tuplegetter
import ctypes as ffi
import json
from pathlib import Path
import sys
from threading import Lock
from time import perf_counter_ns
from types import FunctionType, MethodType

from scar.compiler.scalar_table_region import _definition, code_tree
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.runtime.csr_region_entry import _observer_active
from scar.runtime.entry_runner import run_original_script
from scar.runtime.scalar_table_contract import MISS, OFFER, compile_original
from scar.runtime.scalar_table_entry import _exact_names, _run_main
from scar.trace.code_snapshot import describe_code, site


SCHEMA = 'scar.scalar-transition-source-study'
_INT64 = 1 << 63


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


class _Arg(ffi.Structure):
    _fields_ = [('data',ffi.c_char_p), ('size',ffi.c_size_t), ('scalar',ffi.c_int64), ('kind',ffi.c_uint32)]


def _seal(source, caller, selected, raw, recipe):
    root = compile_original(raw, recipe)
    choices = [node for node in code_tree(root) if describe_code(node) == caller]
    _require(len(choices) == 1 and all(OFFER not in node.co_names and OFFER not in node.co_varnames
                                     and MISS not in node.co_names for node in code_tree(root)),
             'selected call owner or reserved study name changed')
    original, = choices
    _require(site(original, selected['offset']) == selected and selected['opname'] == 'CALL',
             'native study requires an exact original CALL')
    tree = ast.parse(raw, filename=source)
    function = _definition(tree, original.co_qualname)
    choices = [node for node in ast.walk(function) if type(node) is ast.Call
               and [node.lineno,node.end_lineno,node.col_offset,node.end_col_offset] == selected['position']]
    _require(len(choices) == 1, 'selected original call does not have one source expression')
    target, = choices
    _require(type(target.func) is ast.Attribute and len(target.args) == 1 and not target.keywords,
             'study supports an original bound call with one positional argument')
    target.args = [target.func, target.args[0]]
    target.func = ast.Name(id=OFFER, ctx=ast.Load())
    revised = compile(ast.fix_missing_locations(tree), source, 'exec', dont_inherit=True, optimize=recipe['optimize'])
    replacements = [node for node in code_tree(revised) if node.co_qualname == original.co_qualname
                    and node.co_firstlineno == original.co_firstlineno]
    preserved = ('co_argcount','co_posonlyargcount','co_kwonlyargcount','co_varnames','co_freevars','co_cellvars','co_flags')
    _require(len(replacements)==1 and all(getattr(original,key)==getattr(replacements[0],key) for key in preserved),
             'selected call substitution changed its original caller ABI')
    return replacements[0]


def _decode(data, ports):
    header,*rows = data.splitlines()
    _require(header==b'SCAR_STATE_V1' and len(rows)==len(ports), 'native state output violated its declared ABI')
    result = {}
    for port,row in zip(ports,rows,strict=True):
        tag,value = row.split(b':',1)
        kind = port['type']
        if kind == 'str' and tag == b's':
            decoded = bytes.fromhex(value.decode('ascii')).decode('ascii')
        elif kind == 'int' and tag == b'i':
            decoded = int(value)
            _require(-_INT64 <= decoded < _INT64 and str(decoded).encode('ascii')==value, 'invalid native integer')
        elif kind == 'bool' and tag == b'b' and value in (b'0',b'1'):
            decoded = value == b'1'
        else:
            raise ValueError('native returned state field type changed')
        result[port['name']] = decoded
    return result


class _Offer:
    def __init__(self, executable, plan, evidence, capture, selection, *, observe):
        self.lib = ffi.PyDLL(str(executable))
        self.action = self.lib.scar_scalar_transition_v1
        self.action.argtypes = (ffi.c_char_p,ffi.c_size_t,ffi.POINTER(_Arg),ffi.c_size_t,
                                ffi.c_void_p,ffi.c_size_t,ffi.POINTER(ffi.c_size_t))
        self.action.restype = ffi.c_int
        self.arguments = (_Arg * len(plan['inputs']))()
        self.output = ffi.create_string_buffer(65536)
        self.written = ffi.c_size_t()
        self.plan = plan
        self.selection = selection
        self.evidence = evidence
        self.field_names = [row['name'] for row in capture['state_fields']]
        self.layout = [self._field_ports(row) for row in capture['state_fields']]
        fields = {port:(index,leaf if len(names)>1 else None)
                  for index,names in enumerate(self.layout) for leaf,port in enumerate(names)}
        self.ports = [(port['type'], fields[port['name']] if port['name'] in fields
                       else int(port['name'].removeprefix('arg_'))) for port in plan['inputs']]
        self.argument_count = len(self.ports) - len(fields)
        self.constants = capture['globals']
        self.observe = observe
        self.counts = Counter()
        self.versions = Counter()
        self.phase_ns = Counter()
        self.type_binding = None
        self.table_binding = None
        self.expected_new = namedtuple('_SourceTrialShape',self.field_names).__new__
        self.lock = Lock()
        self.snapshots = {key.encode('ascii'):{name:tuple(row) for name,row in snap['rows'].items()}
                          for key,snap in capture['table_versions'].items()}

    def _field_ports(self, row):
        base = 'field_'+row['name']
        return (base+'__0',base+'__1') if row['kind']=='EXACT_BOOL_TUPLE_2' else (base,)

    def _miss(self, target, argument, reason):
        if self.observe:
            self.counts[reason] += 1
        return target(argument)

    def _bind(self, actor, function):
        namespace = function.__globals__
        if (type(actor) is not type or namespace.get(self.selection['constructor']['name']) is not actor
                or not issubclass(actor,tuple) or actor._fields != tuple(self.field_names)
                or actor.__getattribute__ is not tuple.__getattribute__ or actor.__init__ is not tuple.__init__
                or not _exact_names(namespace) or not _exact_names(function.__builtins__)):
            return False
        actual = actor.__new__
        if (type(actual) is not FunctionType or actual.__code__ != self.expected_new.__code__
                or actual.__defaults__ is not None or actual.__globals__.get('_tuple_new') is not tuple.__new__):
            return False
        descriptors = []
        for i,name in enumerate(self.field_names):
            member = getattr(actor,name,None)
            if type(member) is not _tuplegetter or member.__reduce__()[1][0] != i:
                return False
            descriptors.append((name,member))
        methods = {}
        for role in ('callee','rotation','score'):
            proof = self.evidence[role]['code']
            name = proof['qualname'].rsplit('.',1)[-1]
            method = getattr(actor,name,None)
            if (type(method) is not FunctionType or method.__globals__ is not namespace
                    or describe_code(method.__code__)!=proof or method.__closure__ is not None):
                return False
            defaults = method.__defaults__
            if role == 'rotation':
                if type(defaults) is not tuple or len(defaults)!=1 or defaults[0] is not False:
                    return False
            elif defaults is not None:
                return False
            methods[name] = method, method.__code__, defaults
        self.type_binding = actor, actual, actual.__code__, namespace, methods, descriptors
        return True

    def _version(self, root):
        if type(root) is not dict or any(type(key) is not str for key in root):
            return None
        if self.table_binding is not None:
            previous, anchors, version = self.table_binding
            if root is previous and len(anchors)==len(root) and all(root.get(key) is row for key,row in anchors.items()):
                return version
        if any(type(row) is not tuple or any(type(cell) is not int for cell in row) for row in root.values()):
            return None
        for version, snapshot in self.snapshots.items():
            if root == snapshot:
                self.table_binding = root,root.copy(),version
                return version
        return None

    def _inputs(self, receiver, argument):
        if tuple.__len__(argument)!=self.argument_count:
            return None
        for i,(name,source) in enumerate(self.ports):
            if type(source) is tuple:
                field,leaf = source
                value = tuple.__getitem__(receiver,field)
                if leaf is not None:
                    if type(value) is not tuple or len(value)!=2:
                        return None
                    value = value[leaf]
            else:
                value = tuple.__getitem__(argument,source)
            if name=='str' and type(value) is str and value.isascii() and '\0' not in value:
                raw = value.encode('ascii')
                self.arguments[i] = _Arg(raw,len(raw),0,1)
            elif name=='int' and type(value) is int and -_INT64 <= value < _INT64:
                self.arguments[i] = _Arg(None,0,value,2)
            elif name=='bool' and type(value) is bool:
                self.arguments[i] = _Arg(None,0,int(value),3)
            else:
                return None
        return True

    def __call__(self, target, argument):
        if not self.lock.acquire(blocking=False):
            return self._miss(target,argument,'concurrent_study_call')
        try:
            return self._evaluate(target,argument)
        finally:
            self.lock.release()

    def _evaluate(self, target, argument):
        started = perf_counter_ns() if self.observe else 0
        if type(target) is not MethodType or not isinstance(argument,tuple) or type(argument).__iter__ is not tuple.__iter__:
            return self._miss(target,argument,'call_shape')
        receiver, function = target.__self__,target.__func__
        actor = type(receiver)
        if self.type_binding is None and not self._bind(actor,function):
            return self._miss(target,argument,'constructor_or_binding')
        type_, constructor, constructor_code, namespace, methods, descriptors = self.type_binding
        expected = self.evidence['callee']['code']['qualname'].rsplit('.',1)[-1]
        if (actor is not type_ or function is not methods[expected][0] or actor.__new__ is not constructor
                or constructor.__code__ is not constructor_code or constructor.__defaults__ is not None
                or constructor.__globals__.get('_tuple_new') is not tuple.__new__
                or actor.__getattribute__ is not tuple.__getattribute__ or actor.__init__ is not tuple.__init__
                or namespace.get(self.selection['constructor']['name']) is not actor
                or any(getattr(actor,name,None) is not method or method.__code__ is not code
                       or method.__defaults__ is not defaults for name,(method,code,defaults) in methods.items())
                or any(getattr(actor,name,None) is not member for name,member in descriptors)
                or tuple.__len__(receiver)!=len(self.layout)):
            return self._miss(target,argument,'known_class_or_method_changed')
        if (namespace.get('abs',function.__builtins__.get('abs')) is not builtins.abs
                or any(type(namespace.get(name)) is not int or namespace[name]!=value for name,value in self.constants.items())):
            return self._miss(target,argument,'source_constants')
        version = self._version(namespace.get(self.selection['table']))
        if version is None:
            return self._miss(target,argument,'unknown_table_version')
        guarded = perf_counter_ns() if self.observe else 0
        if self._inputs(receiver,argument) is None:
            return self._miss(target,argument,'input_domain')
        packed = perf_counter_ns() if self.observe else 0
        status = self.action(version,len(version),self.arguments,len(self.arguments),self.output,len(self.output),
                             ffi.byref(self.written))
        completed = perf_counter_ns() if self.observe else 0
        if status in (1,2):
            return self._miss(target,argument,'native_declined')
        _require(status==0 and 0<self.written.value<=len(self.output), 'native transition failed within its guarded input')
        if self._version(namespace.get(self.selection['table']))!=version:
            return self._miss(target,argument,'table_changed_during_call')
        state = _decode(ffi.string_at(self.output,self.written.value),self.plan['outputs'])
        fields = [tuple(state[key] for key in names) if len(names)>1 else state[names[0]] for names in self.layout]
        result = tuple.__new__(actor,fields)
        if self.observe:
            self.phase_ns['guard_and_version'] += guarded-started
            self.phase_ns['input_pack'] += packed-guarded
            self.phase_ns['ffi_and_native'] += completed-packed
            self.phase_ns['post_guard_decode_and_tuple'] += perf_counter_ns()-completed
            self.counts['native'] += 1
            self.versions[version.decode('ascii')] += 1
        return result


def run_native(entry, argv, library, digest, *, observe):
    from scar.codegen.scalar_transition_library import SCHEMA as LIBRARY

    _require(sys.implementation.name=='cpython' and getattr(sys,'_is_gil_enabled',lambda:True)()
        and not _observer_active(), 'scope study requires ordinary CPython with inactive observers')
    reference = artifact_reference(Path(library).resolve(strict=True), LIBRARY)
    _require(reference['sha256']==digest, 'study library receipt changed after admission')
    artifact = read_artifact(reference)
    root = Path(reference['uri']).parent
    executable = (root/artifact['library']['uri']).resolve(strict=True)
    _require(executable.is_relative_to(root), 'study library must be a local admitted member')
    read_artifact({**artifact['library'],'uri':str(executable)},json_document=False)
    plan = read_artifact(artifact['plan'])
    evidence = read_artifact(plan['evidence'])
    capture = read_artifact(evidence['capture'])
    selection = read_artifact(capture['selection'])
    source = plan['source']
    _require(str(entry)==source['uri']==evidence['source']['uri'], 'study launcher must be the exact original source')
    score = read_artifact(evidence['score_evidence'])
    recipe = read_artifact(score['proof'])['cases'][score['proof_case']]['compile_recipe']
    raw = read_artifact(source,json_document=False)
    caller = evidence['caller']['code']
    selected = evidence['original_calls']['producer']['site']
    leaf = _seal(str(entry),caller,selected,raw,recipe)
    offer = _Offer(executable,plan,evidence,capture,selection,observe=observe)
    execution = _run_main(entry,argv,dict(compile_recipe=recipe,original_code_key=caller['key']),raw,leaf,offer)
    return dict(schema=SCHEMA,schema_version=1,source=source,library=reference,original_site=selected,
        execution=execution,statistics=dict(offer.counts) if observe else None,
        table_calls=dict(offer.versions) if observe else None, phase_ns=dict(offer.phase_ns) if observe else None,
        admitted_for_deployment=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path)
    parser.add_argument('--sha256')
    parser.add_argument('--isolated-study',action='store_true')
    parser.add_argument('--observe',action='store_true')
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1]==['--'] else args.command
    if not command or bool(args.library)!=bool(args.sha256) or bool(args.library)!=args.isolated_study:
        parser.error('provide an original script, or a pinned library and --isolated-study')
    entry = Path(command[0]).resolve(strict=True)
    if args.library is None:
        execution = run_original_script(entry,command[1:])
        report = dict(schema=SCHEMA,schema_version=1,library=None,execution=execution,admitted_for_deployment=False)
    else:
        report = run_native(entry,command[1:],args.library,args.sha256,observe=args.observe)
    with args.report.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
