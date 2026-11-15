"""Transactional C++20 lowering of a verified ordered stateful array region.

All computation and ordered writes use private slots/shadows. Only a successful
whole-region invocation commits original write ROIs. NumPy power2 uses its
original, exactly version-pinned native ufunc loop, not a square substitution.
"""
from copy import deepcopy
import ctypes
from dataclasses import dataclass, field
from hashlib import sha256
import importlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import threading

from scar.compiler.model import CpuPlacement
from scar.compiler.stateful_model import document, stateful_from_dict, stateful_digest, read_stateful
from scar.optimization.hardware import validate_cpu_placement
from . import cpp

SCHEMA = 'scar.native-stateful-artifact'
GuardFailure = cpp.GuardFailure
_HERE = Path(__file__).resolve().parent
_CAPSULE = b'numpy_1.24_ufunc_call_info'


def _hash(raw): return sha256(raw).hexdigest()
def _json(value): return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
def _digest(value): return _hash(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
def _need(condition, message):
    if not condition: raise ValueError(message)


class _Scalar(ctypes.Structure):
    _fields_ = [('value', ctypes.c_double), ('integer', ctypes.c_int64)]


class _Provider(ctypes.Structure):
    _fields_ = [('loop', ctypes.c_void_p), ('context', ctypes.c_void_p), ('auxdata', ctypes.c_void_p)]


class _Info(ctypes.Structure):
    _fields_ = _Provider._fields_ + [('requires_pyapi', ctypes.c_ubyte), ('no_fpe', ctypes.c_ubyte)]


_RUN = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(cpp._Array), ctypes.c_uint64,
    ctypes.POINTER(_Scalar), ctypes.c_uint64, ctypes.POINTER(_Provider), ctypes.c_uint64)


class _Table(ctypes.Structure):
    _fields_ = [('abi_version', ctypes.c_uint32), ('struct_size', ctypes.c_uint32),
        ('array_count', ctypes.c_uint64), ('scalar_count', ctypes.c_uint64),
        ('provider_count', ctypes.c_uint64), ('run', _RUN), ('region_sha256', ctypes.c_char_p),
        ('plan_sha256', ctypes.c_char_p), ('contract_sha256', ctypes.c_char_p)]


def implementation_sha256():
    names = ('stateful.py', 'stateful_v1.cpp.in', 'stateful_abi_v1.h', 'scar_abi_v1.h', 'loop_v1.cpp.in')
    return _digest(dict(files={name:_hash((_HERE/name).read_bytes()) for name in names},
                        boundary=cpp.runtime_implementation_sha256()))


def _providers(region):
    import numpy as np
    module = importlib.import_module('numpy._core._multiarray_umath')
    _need(np.__version__ == '2.5.2' and sys.maxsize == 2**63-1,
          'stateful provider profile requires exact NumPy 2.5.2 and 64-bit process')
    binary = Path(module.__file__).resolve(); raw = binary.read_bytes()
    runtime = dict(numpy_version=np.__version__, ndarray_c_abi=int(module._get_ndarray_c_version()),
        capsule_abi=_CAPSULE.decode(), loaded_binary=dict(path=str(binary), sha256=_hash(raw), bytes=len(raw)),
        dtype_signature=['float64','float64','float64'], fixed_strides=[8,0,8], requires_pyapi=False,
        API_stability='EXPERIMENTAL_EXACT_NUMPY_VERSION_AND_BINARY_PIN',
        execution='ORIGINAL_NUMPY_UFUNC_STRIDED_LOOP_SERIAL_PROVIDER_STAGE')
    rows, owners = [], []
    original = np.power
    for provider in region.providers:
        _need(provider.module == 'numpy' and provider.symbol == 'power', 'unsupported semantic provider')
        _need(type(original) is np.ufunc and original is module.power and original.__name__ == 'power',
              'original NumPy power ufunc required')
        dtypes, capsule = original._resolve_dtypes_and_context((np.dtype('float64'),)*3)
        _need(dtypes == (np.dtype('float64'),)*3, 'power dtype resolution differs')
        original._get_strided_loop(capsule, fixed_strides=(8,0,8))
        getter = ctypes.pythonapi.PyCapsule_GetPointer
        getter.argtypes = [ctypes.py_object, ctypes.c_char_p]; getter.restype = ctypes.c_void_p
        info = _Info.from_address(getter(capsule, _CAPSULE))
        _need(info.loop and info.context and not info.requires_pyapi, 'power native loop requires unsupported Python API')
        rows.append(_Provider(info.loop, info.context, info.auxdata)); owners.append(capsule)
    return runtime, (_Provider*len(rows))(*rows), owners, original


def _admit(region, plan, binding=None):
    from scar.optimization.stateful_schedule import verify_stateful_plan
    region = read_stateful(region) if isinstance(region, (str,Path)) else stateful_from_dict(
        region if isinstance(region,dict) else document(region))
    plan = verify_stateful_plan(region, plan)
    _need(sys.platform == 'linux', 'stateful CPU placement requires Linux')
    value = dict(plan['resource_binding']); value['cpu_ids'] = tuple(value['cpu_ids'])
    value['evidence'] = tuple(value.get('evidence', ())); actual = CpuPlacement(**value)
    _need(binding is None or binding == actual, 'explicit CPU placement differs from plan')
    validate_cpu_placement(actual)
    _need(len(actual.cpu_ids) == plan['cpu_threads'] and max(actual.cpu_ids) < 2**20,
          'stateful CPU worker allocation differs')
    _need(plan.get('status') == 'READY' and plan.get('allocation_search_complete') is True
          and not plan.get('frontier') and plan.get('budget_satisfied') is True,
          'stateful plan has unresolved obligations')
    for source in (region.source, *(callee.source for callee in region.callees)):
        _need(_hash(Path(source.path).read_bytes()) == source.sha256, 'stateful original source snapshot differs')
    return region, plan, actual


def _source(region, plan, binding, contract):
    nodes = {node.id:node for node in region.nodes}; order = {node.id:i for i,node in enumerate(region.nodes)}
    arrays = [row.id for row in region.inputs if row.kind == 'array']
    scalars = [row.id for row in region.inputs if row.kind == 'scalar']
    shapes, views = {}, {}
    for node in region.nodes:
        if node.op == 'input':
            shapes[node.id] = tuple(plan['input_shapes'][node.attrs['name']]) if node.attrs['kind']=='array' else None
            if shapes[node.id] is not None: views[node.id] = (node.attrs['name'],0,0,*shapes[node.id])
        elif node.op == 'const': shapes[node.id] = None
        elif node.op == 'slice':
            root,r,c,rows,cols = views[node.inputs[0]]
            a,b = node.attrs['bounds']; rs,re,_ = slice(*a).indices(rows); cs,ce,_ = slice(*b).indices(cols)
            shapes[node.id] = (max(0,re-rs),max(0,ce-cs)); views[node.id] = (root,r+rs,c+cs,*shapes[node.id])
        else:
            operands = [shapes[key] for key in node.inputs if shapes[key] is not None]
            _need(not operands or all(shape==operands[0] for shape in operands), 'stateful binary array shapes differ')
            shapes[node.id] = operands[0] if operands else None
    slots = {row['id']:i for i,row in enumerate(plan['workspace_slots'])}
    def payload(value): return f"slots[{slots[plan['assignment'][value]['slot_id']]}].get()"
    def root_pointer(value):
        return payload(plan['shadows'][value]) if value in plan['shadows'] else f'static_cast<const double *>(arrays[{arrays.index(value)}].data)'
    available, scalar_ready, current_exprs = {}, set(), set()
    def scalar_value(key, statements):
        node = nodes[key]
        if key in scalar_ready: return f'scalar{order[key]}'
        if node.op=='const': return 'Scalar{'+float(node.attrs['value']).hex()+', '+str(type(node.attrs['value']) is int).lower()+'}'
        if node.op=='input':
            i = scalars.index(node.attrs['name']); return f'Scalar{{scalars[{i}].value, scalars[{i}].integer != 0}}'
        _need(key in current_exprs, 'scalar expression is outside the verified stage closure')
        args = [scalar_value(v, statements) for v in node.inputs]
        result = f'scalar{order[key]}'
        expression = f'scalar_neg({args[0]})' if node.op=='neg' else f"scalar_op({args[0]}, {args[1]}, '{dict(add='+',sub='-',mul='*',div='/')[node.op]}', numeric_guard)"
        statements.append(f'{result} = {expression};'); scalar_ready.add(key); return result
    def pointer(key):
        if key in available: return payload(available[key]), shapes[key][1]
        _need(key in views, 'provider input must be an alias or ready materialized value')
        root,r,c,rows,cols = views[key]; stride = plan['input_shapes'][root][1]
        return f'({root_pointer(root)} + {r*stride+c})', stride
    def expression(key, statements, cache):
        if shapes[key] is None: return '('+scalar_value(key, statements)+').value'
        if key in cache: return cache[key]
        if key in available or key in views:
            ptr,stride = pointer(key); return f'{ptr}[row * {stride} + col]'
        node = nodes[key]; _need(node.op!='power2', 'power2 must retain its original provider materialization')
        _need(key in current_exprs, 'array expression is outside the verified stage closure')
        operands = [expression(v,statements,cache) for v in node.inputs]
        value = '-'+operands[0] if node.op=='neg' else f'({operands[0]} {dict(add="+",sub="-",mul="*",div="/")[node.op]} {operands[1]})'
        name = f'value{order[key]}'; statements.append(f'const double {name} = {value};'); cache[key]=name; return name
    stage_text = []
    parallel = plan['cpu_threads'] > 1
    for stage in plan['stages']:
        kind = stage['kind']
        if kind=='COMMIT': continue
        current_exprs = set(stage['expr_ids'])
        body=[]; root=stage.get('root_expr'); output=stage.get('output_payload')
        if kind=='SCALAR':
            scalar_value(root,body)
            if parallel: body=['#pragma omp single','{',*['    '+line for line in body],'}']
        else:
            shape = stage['shape']; rows,cols = shape
            if kind=='SNAPSHOT_ROOT':
                target=stage['target']; source=f'static_cast<const double *>(arrays[{arrays.index(target)}].data)'
                inner=[f'{payload(output)}[row * {cols} + col] = {source}[row * {cols} + col];']
            elif kind=='COMPUTE':
                inner=[]; value=expression(root,inner,{})
                inner.append(f'{payload(output)}[row * {cols} + col] = {value};')
            elif kind=='PROVIDER':
                node=nodes[root]; _need(node.op=='power2','unsupported native provider stage')
                source,stride=pointer(node.inputs[0]); index=[p.id for p in region.providers].index(node.attrs['provider'])
                body=[f'for (int64_t row = 0; row < {rows if cols else 0}; ++row) {{',
                    '    double exponent = 2.0;',f'    char *data[] = {{reinterpret_cast<char *>(const_cast<double *>({source}) + row * {stride}), reinterpret_cast<char *>(&exponent), reinterpret_cast<char *>({payload(output)} + row * {cols})}};',
                    f'    const intptr_t dimensions[] = {{{cols}}}, strides[] = {{8,0,8}};',
                    f'    if (providers[{index}].loop(providers[{index}].context, data, dimensions, strides, providers[{index}].auxdata) != 0) {{',
                    '        numeric_guard |= 1;',f'        for (int64_t col = 0; col < {cols}; ++col) {payload(output)}[row * {cols} + col] = 0.0;', '    }','}']
                if parallel: body=['#pragma omp single','{',*['    '+line for line in body],'}']
            elif kind=='WRITE_SHADOW':
                _need(shapes[root] is None or root in available, 'write RHS must be materialized before modifying state')
                target=stage['target']; r,c = [axis[0] for axis in stage['normalized_bounds']]
                stride=plan['input_shapes'][target][1]; destination=f'{root_pointer(target)}[({r} + row) * {stride} + {c} + col]'
                inner=[]; value=expression(root,inner,{})
                token=dict(assign='=',add='+=',sub='-=',mul='*=',div='/=')[stage['op']]
                inner.append(f'{destination} {token} {value};')
            else: raise ValueError('unsupported stateful stage kind: '+kind)
            if kind!='PROVIDER':
                body=(["#pragma omp for schedule(static)"] if parallel else [])+[f'for (int64_t row = 0; row < {rows}; ++row) {{',f'    for (int64_t col = 0; col < {cols}; ++col) {{',*['        '+line for line in inner],'    }','}']
            if output and root: available[root]=output
        stage_text.extend(body)
        released = set(stage.get('release_values_after', ()))
        available = {key:value for key,value in available.items() if value not in released}
    program='\n'.join('            '+line for line in stage_text)
    if parallel:
        execution=f'''    uint32_t participant_guard = 0;
    #pragma omp parallel num_threads({plan['cpu_threads']}) reduction(|:numeric_guard,placement_guard)
    {{
        ThreadAffinity affinity;
        const bool team_ready = omp_get_num_threads() == {plan['cpu_threads']};
        const bool ready = team_ready && affinity.bind(planned_cpu_ids[omp_get_thread_num()]);
        FloatingScope environment;
        const bool environment_ready = environment.begin(&execution_environment);
        #pragma omp atomic update
        participant_guard |= static_cast<uint32_t>(!ready || !environment_ready);
        #pragma omp barrier
        if (!participant_guard) {{
{program}
        }}
        numeric_guard |= static_cast<uint32_t>(!environment_ready || std::fetestexcept(floating_flags));
        numeric_guard |= static_cast<uint32_t>(!environment.restore());
        const bool kept = affinity.bound();
        const bool restored = affinity.restore();
        placement_guard |= static_cast<uint32_t>(!ready || !kept || !restored);
    }}'''
    else:
        execution='    if (!caller_affinity.bind(planned_cpu_ids[0])) return 4;\n'+program+'\n    placement_guard |= static_cast<uint32_t>(!caller_affinity.bound());'
    commits=[]
    for row in plan['commit_rows']:
        target=row['target']; (r,re),(c,ce)=row['normalized_bounds']; stride=plan['input_shapes'][target][1]
        commits += [f'    for (int64_t row = {r}; row < {re}; ++row) if ({ce-c})',f'        std::memcpy(static_cast<double *>(arrays[{arrays.index(target)}].data) + row * {stride} + {c}, {root_pointer(target)} + row * {stride} + {c}, {ce-c} * sizeof(double));']
    original=(_HERE/'loop_v1.cpp.in').read_text()
    affinity=original[original.index('class ThreadAffinity {'):original.index('\n#endif',original.index('class ThreadAffinity {'))]
    floating=original[original.index('class FloatingScope {'):original.index('\nbool valid_array',original.index('class FloatingScope {'))]
    guards=[f'    if (!valid_array(arrays[{i}], {plan["input_shapes"][name][0]}, {plan["input_shapes"][name][1]})) return 2;' for i,name in enumerate(arrays)]
    finite=[f'    for (int64_t i = 0; i < {math.prod(plan["input_shapes"][name])}; ++i) if (!finite(static_cast<const double *>(arrays[{i}].data)[i])) return 3;' for i,name in enumerate(arrays)]
    replacements=dict(OPENMP_INCLUDE='#include <omp.h>' if parallel else '',CPU_IDS=', '.join(map(str,binding.cpu_ids)),CPU_SLOTS=str(max(binding.cpu_ids)+1),AFFINITY_CLASS=affinity,FLOATING_CLASS=floating,
        ARRAY_COUNT=str(len(arrays)),SCALAR_COUNT=str(len(scalars)),PROVIDER_COUNT=str(len(region.providers)),ARRAY_GUARDS='\n'.join(guards),INPUT_FINITE_GUARDS='\n'.join(finite),
        SLOT_COUNT=str(len(slots)),SLOT_CAPACITIES=', '.join(str(row['capacity_bytes'])+'ULL' for row in plan['workspace_slots']),
        PAYLOAD_BYTES=str(plan['workspace_payload_bytes']),EXTERNAL_BYTES=str(plan['external_array_bytes']),TOTAL_BYTES=str(plan['total_region_array_payload_bytes']),BUDGET=str(plan['memory_budget_bytes']),
        SCALAR_DECLARATIONS='\n'.join(f'    Scalar scalar{i};' for i,node in enumerate(region.nodes) if shapes[node.id] is None and node.op not in ('input','const')),
        EXECUTION=execution,COMMITS='\n'.join(commits),REGION_SHA256=stateful_digest(region),PLAN_SHA256=_digest(plan),CONTRACT_SHA256=_digest(contract))
    source=(_HERE/'stateful_v1.cpp.in').read_text()
    for key,value in replacements.items(): source=source.replace('@'+key+'@',value)
    _need('@' not in source,'unresolved native source template token')
    return source


@dataclass
class StatefulArtifact:
    region: object
    plan: dict
    directory: Path
    _metadata: dict
    _library: object
    _table: object
    _providers: object
    _owners: list
    _power: object
    _lock: object = field(default_factory=threading.Lock)

    @property
    def report(self): return deepcopy({**self._metadata,'artifact_directory':str(self.directory)})

    def run(self, arrays, scalars):
        import numpy as np
        if np.power is not self._power: raise GuardFailure('stateful original NumPy power provider changed')
        array_ids=[row.id for row in self.region.inputs if row.kind=='array']; scalar_ids=[row.id for row in self.region.inputs if row.kind=='scalar']
        if type(arrays) is not dict or set(arrays)!=set(array_ids) or type(scalars) is not dict or set(scalars)!=set(scalar_ids):
            raise GuardFailure('stateful input IDs differ')
        descriptors=[]; written=set(self.plan['shadows'])
        for name in array_ids:
            value=arrays[name]; descriptor=cpp._descriptor(value)
            if list(value.shape)!=self.plan['input_shapes'][name] or not value.flags.c_contiguous or (name in written and not value.flags.writeable):
                raise GuardFailure('stateful array shape/layout/writeability differs')
            descriptors.append(descriptor)
        for i,left in enumerate(descriptors):
            for right in descriptors[:i]:
                if left.lower_bound<right.upper_bound and right.lower_bound<left.upper_bound:
                    raise GuardFailure('stateful external arrays alias')
        values=[]
        for name in scalar_ids:
            value=scalars[name]
            if type(value) not in (float,int) or not math.isfinite(value) or type(value) is int and abs(value)>2**53:
                raise GuardFailure('stateful scalar requires finite exact Python float or bounded int')
            values.append(_Scalar(float(value),int(type(value) is int)))
        if not self._lock.acquire(blocking=False): raise GuardFailure('stateful provider capsule already in use')
        try:
            status=self._table.contents.run((cpp._Array*len(descriptors))(*descriptors),len(descriptors),(_Scalar*len(values))(*values),len(values),self._providers,len(self.region.providers))
        finally: self._lock.release()
        if status: raise GuardFailure('stateful precommit guard: '+{1:'arguments/alias',2:'shape/layout',3:'floating-point/nonfinite',4:'CPU affinity',5:'allocation',6:'provider',7:'payload budget',8:'scalar exactness'}.get(status,str(status)))
        return None


def build(region, out, *, plan, resource_binding=None):
    region,plan,binding=_admit(region,plan,resource_binding)
    runtime,_,_,_=_providers(region)
    compiler=shlex.split(os.environ.get('CXX','g++')); _need(compiler and shutil.which(compiler[0]),'C++20 compiler unavailable')
    identity=subprocess.run([*compiler,'--version'],capture_output=True,text=True,check=True).stdout.splitlines()[0]
    options=[*cpp.COMPILER_OPTIONS,*(['-fopenmp'] if plan['cpu_threads']>1 else [])]
    contract=dict(region_sha256=stateful_digest(region),plan_sha256=_digest(plan),provider_runtime=runtime)
    source=_source(region,plan,binding,contract).encode()
    headers={name:(_HERE/name).read_bytes() for name in ('stateful_abi_v1.h','scar_abi_v1.h')}
    fingerprint=implementation_sha256(); build_id=_digest(dict(source=_hash(source),headers={k:_hash(v) for k,v in headers.items()},compiler=compiler,identity=identity,options=options,implementation_sha256=fingerprint,contract=contract))
    directory=Path(out).resolve(); _need(not directory.exists(),'fresh stateful output directory required'); directory.mkdir(parents=True)
    files=dict(region='region.json',plan='plan.json',source='stateful.cpp',library='stateful_'+build_id[:20]+'.so',**{name:name for name in headers})
    (directory/files['region']).write_bytes(_json(document(region))); (directory/files['plan']).write_bytes(_json(plan)); (directory/files['source']).write_bytes(source)
    for name,raw in headers.items(): (directory/name).write_bytes(raw)
    process=subprocess.run([*compiler,*options,str(directory/files['source']),'-o',str(directory/files['library']),'-lm'],capture_output=True,text=True)
    if process.returncode: raise RuntimeError('C++20 stateful compilation failed:\n'+process.stderr)
    metadata=dict(schema=SCHEMA,schema_version=1,stateful_abi_version=1,build_id=build_id,**contract,contract_sha256=_digest(contract),
        compiler=dict(argv=compiler,identity=identity,options=options),implementation_sha256=fingerprint,files=files,sha256={key:_hash((directory/name).read_bytes()) for key,name in files.items()},
        cpu_threads=plan['cpu_threads'],cpu_ids=list(binding.cpu_ids),memory_budget_bytes=plan['memory_budget_bytes'],workspace_payload_bytes=plan['workspace_payload_bytes'],external_array_bytes=plan['external_array_bytes'],total_region_array_payload_bytes=plan['total_region_array_payload_bytes'],
        execution='ONE_NATIVE_CALL_ORDERED_PRIVATE_SHADOW_STAGES_THEN_ORIGINAL_WRITE_ROI_COMMITS',
        failure_policy='GUARD_FAILURE_ONLY_BEFORE_EXTERNAL_COMMIT_ORIGINAL_ONCE',memory_policy='UNBOUND',time_fusion=False,
        cost_status='UNCALIBRATED',deployment_authorized=False,physical_optimality_proven=False,
        external_assumptions=['SYNCHRONOUS_CPU_EXECUTION','NO_FOREIGN_CONCURRENT_STATE_READERS_OR_WRITERS','FP_STATUS_FLAGS_NOT_USED_AS_PROGRAM_OUTPUT'])
    (directory/'artifact.json').write_bytes(_json(metadata)); return load(directory)


def load(out):
    directory=Path(out).resolve(); directory=directory if directory.is_dir() else directory.parent
    metadata=json.loads((directory/'artifact.json').read_bytes())
    _need((metadata.get('schema'),metadata.get('schema_version'),metadata.get('stateful_abi_version'))==(SCHEMA,1,1),'unsupported stateful artifact')
    for key,name in metadata['files'].items():
        _need(Path(name).name==name and _hash((directory/name).read_bytes())==metadata['sha256'][key],'stateful artifact checksum differs: '+key)
    region,plan,binding=_admit(directory/metadata['files']['region'],directory/metadata['files']['plan'])
    runtime,providers,owners,power=_providers(region)
    _need(runtime==metadata['provider_runtime'],'stateful pinned NumPy provider runtime differs')
    contract=dict(region_sha256=stateful_digest(region),plan_sha256=_digest(plan),provider_runtime=runtime)
    _need(all(metadata[key]==value for key,value in contract.items()) and _digest(contract)==metadata['contract_sha256'],'stateful contract differs')
    source=_source(region,plan,binding,contract).encode(); _need(source==(directory/metadata['files']['source']).read_bytes(),'stateful source differs from plan replay')
    _need(_digest(dict(source=_hash(source),headers={name:_hash((directory/name).read_bytes()) for name in ('stateful_abi_v1.h','scar_abi_v1.h')},compiler=metadata['compiler']['argv'],identity=metadata['compiler']['identity'],options=metadata['compiler']['options'],implementation_sha256=metadata['implementation_sha256'],contract=contract))==metadata['build_id'],'stateful build identity differs')
    library=ctypes.CDLL(str(directory/metadata['files']['library'])); getter=library.scar_get_stateful_table_v1; getter.argtypes=[]; getter.restype=ctypes.POINTER(_Table); table=getter()
    _need(table and table.contents.abi_version==1 and table.contents.struct_size==ctypes.sizeof(_Table) and table.contents.run,'stateful native ABI differs')
    _need((table.contents.array_count,table.contents.scalar_count,table.contents.provider_count)==(sum(i.kind=='array' for i in region.inputs),sum(i.kind=='scalar' for i in region.inputs),len(region.providers)),'stateful ABI input counts differ')
    for key in ('region_sha256','plan_sha256','contract_sha256'): _need(getattr(table.contents,key)==metadata[key].encode(),'stateful embedded identity differs')
    return StatefulArtifact(region,plan,directory,metadata,library,table,providers,owners,power)


__all__=['SCHEMA','GuardFailure','StatefulArtifact','build','load','implementation_sha256']
