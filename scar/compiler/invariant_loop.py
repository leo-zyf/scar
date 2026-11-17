"""Dependency-derived LICM using existing pure array regions and operator ABI v1.

The caller interface is retained evidence, not a boolean proof token. This
module supplies conditional graph rewriting/native composition; original
caller replacement requires its separate F4 and runtime callable admission.
"""
from dataclasses import asdict,dataclass,replace
import ctypes
from hashlib import sha256
import json
from pathlib import Path
import subprocess

from scar.compiler.model import ArrayRegion,ExecutionPlan,Expr,region_from_dict
from scar.compiler.pipeline_model import prepare_pipeline_leaf
from scar.compiler.validation import validate_region
from scar.analysis.array_motion import derive_array_motion
from scar.codegen import cpp
from scar.optimization.optimizer import optimize_region
from scar.optimization.resource_solver import semantic_sha256


def _json(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def _receipt(path):
    path=Path(path).resolve();raw=path.read_bytes()
    return dict(path=str(path),sha256=sha256(raw).hexdigest(),bytes=len(raw))
def _body(value):
    value=value.region if isinstance(value,ExecutionPlan) else value
    validate_region(value)
    if any(n.op=='slice' for n in value.nodes):raise ValueError('LICM native profile requires equal-shape elementwise inputs without slices')
    return value


def plan_invariant_loop(body,*,carry_parameter,loop_interface,memory_budget_bytes,hoist_invariants=True):
    """Infer maximal invariant subgraphs; input roles come from caller graph."""
    body=_body(body)
    if carry_parameter not in body.parameters:raise ValueError('carry must be an original body parameter')
    if type(loop_interface) is not dict or loop_interface.get('carry_parameter')!=carry_parameter:
        raise ValueError('explicit caller graph carry mapping required')
    readonly=[p for p in body.parameters if p!=carry_parameter]
    if set(loop_interface.get('readonly_parameters',()))!=set(readonly):
        raise ValueError('all external body inputs require readonly loop-boundary mapping')
    if type(memory_budget_bytes) is not int or memory_budget_bytes<8:
        raise ValueError('explicit incremental array payload budget required')
    if type(hoist_invariants) is not bool:
        raise ValueError('hoist_invariants requires an exact boolean diagnostic placement policy')
    motion=derive_array_motion(body,carry_parameter=carry_parameter,loop_interface=loop_interface)
    nodes={n.id:n for n in body.nodes}
    dependencies=motion['dependencies']
    invariant=set(motion['invariant_nodes'])
    roots=list(motion['invariant_roots'])
    if not roots:raise ValueError('no nonconstant loop-invariant array computation reaches varying work')
    if any(motion['placements'][key]['status']!='CONDITIONAL' for key in roots):
        raise ValueError('loop-invariant roots have unresolved movement dependencies')
    if body.output in invariant:raise ValueError('fully invariant body output requires separate fresh-result loop contract')
    inferred_roots=list(roots)
    if not hoist_invariants:roots=[]
    replacements={}
    for i,root in enumerate(roots):
        name='__scar_hoisted_'+str(i)
        while name in body.parameters:
            name = f'{name}_'
        replacements[root]=name
    def subregion(root,cut):
        live=set();pending=[root]
        while pending:
            key=pending.pop()
            if key in live:continue
            live.add(key)
            if key not in cut:pending.extend(nodes[key].inputs)
        inputs={nodes[key].attrs['name'] for key in live if nodes[key].op=='input'}
        parameters=tuple(p for p in body.parameters if p in inputs)
        parameters+=tuple(name for key,name in cut.items() if key in live)
        selected=[]
        for node in body.nodes:
            if node.id not in live:continue
            if node.id in cut:selected.append(Expr(node.id,'input',attrs=dict(name=cut[node.id])))
            elif node.op=='input':selected.append(replace(node,attrs=dict(name=node.attrs['name'])))
            else:selected.append(node)
        return ArrayRegion(body.id+':licm:'+root+('Res' if cut else 'Pre'),body.source,
            parameters,tuple(selected),root,body.assumptions,())
    hoisted=[subregion(root,{}) for root in roots]
    residual=subregion(body.output,replacements)
    result=dict(schema='scar.invariant-loop-plan',schema_version=1,body=asdict(body),
        body_sha256=semantic_sha256(asdict(body)),carry_parameter=carry_parameter,
        loop_interface=loop_interface,source_replacement_authorized=False,
        authority='CONDITIONAL_ON_ORIGINAL_CALLER_F4_AND_RUNTIME_ADMISSION',
        dependencies=dependencies,motion_analysis=motion,
        hoist_invariants=hoist_invariants,inferred_invariant_roots=inferred_roots,
        placement_policy='AUTOMATIC_DEPENDENCY_DERIVED_LICM' if hoist_invariants else
            'DIAGNOSTIC_ONLY_SAME_NATIVE_HOST_WITH_ALL_BODY_WORK_INSIDE_LOOP',
        hoisted_roots=roots,hoisted_values=replacements,
        hoisted_regions=[asdict(r) for r in hoisted],residual_region=asdict(residual),
        original_arithmetic_nodes=sum(n.op not in ('input','const') for n in body.nodes),
        invariant_arithmetic_nodes=len(invariant),
        residual_arithmetic_nodes=sum(n.op not in ('input','const') for n in residual.nodes),
        memory_budget_bytes=memory_budget_bytes,
        memory_scope='INCREMENTAL_ARRAY_PAYLOAD_HOISTED_OUTPUTS_AND_TWO_PRIVATE_CARRY_BUFFERS',
        zero_trip='RETURN_ORIGINAL_CARRY_ALIAS_BEFORE_HOIST_OR_INPUT_GUARDS',
        failure_policy='WITHHOLD_ALL_PRIVATE_OUTPUTS_THEN_ORIGINAL_LOOP_ONCE',
        Q=['NO_INPUT_WRITES','NO_INTERMEDIATE_ESCAPE_OR_IDENTITY_OBSERVERS',
            'NO_FOREIGN_CONCURRENT_INPUT_WRITERS','NO_INTERSTEP_CALLBACKS_OR_HOOKS',
            'FP_STATUS_FLAGS_ARE_NOT_PROGRAM_OUTPUTS','EXACT_ARRAY_AND_CALLABLE_RUNTIME_GUARDS'],
        iterations_unrolled=False,performance_optimality_proven=False,selected_for_use=False)
    result['id']='invariant-loop:'+semantic_sha256(result)
    return result


def _source(plan,kernels):
    names=plan['body']['parameters'];hoisted=plan['hoisted_regions'];residual=plan['residual_region']
    values={parameter:f'inputs[{i}]' for i,parameter in enumerate(names)}
    lines=[]
    for i,(root,region) in enumerate(zip(plan['hoisted_roots'],hoisted)):
        args=', '.join(values[p] for p in region['parameters'])
        lines.extend([f'    const scar_array_v1 args{i}[] = {{{args}}};',
            f'    int32_t status{i} = tables[{i}]->run(args{i}, {len(region["parameters"])}, &outputs[{i}]);',
            f'    if (status{i}) return status{i};','    ++counters[0];'])
        values[plan['hoisted_values'][root]]=f'outputs[{i}]'
    base=len(hoisted);carry=plan['carry_parameter']
    values[carry]='current'
    args=', '.join(values[p] for p in residual['parameters'])
    lines.extend([f'    scar_array_v1 current = inputs[{names.index(carry)}];',
        '    for (uint64_t step = 0; step < repetitions; ++step) {',
        f'        const uint64_t slot = {base} + step % 2;',
        f'        const scar_array_v1 args[] = {{{args}}};',
        f'        const int32_t status = tables[{base}]->run(args, {len(residual["parameters"])}, &outputs[slot]);',
        '        if (status) return status;',
        '        current = outputs[slot]; *result_slot = slot; ++counters[1];','    }','    return 0;'])
    checks=[]
    for i,kernel in enumerate(kernels):
        digest=kernel.report['plan_sha256']
        checks.append(f'    if (!tables[{i}] || tables[{i}]->abi_version != 1 || tables[{i}]->struct_size < sizeof(scar_operator_table_v1) || !tables[{i}]->run || !tables[{i}]->plan_sha256 || std::strcmp(tables[{i}]->plan_sha256,"{digest}")) return 1;')
    return ('#include "scar_abi_v1.h"\n#include <cstring>\nextern "C" SCAR_EXPORT_V1 int32_t scar_invariant_loop_v1(\n'
        '    const scar_operator_table_v1 *const *tables, const scar_array_v1 *inputs,\n'
        '    scar_array_v1 *outputs, uint64_t repetitions, uint64_t *result_slot, uint64_t *counters) {\n'
        '    if (!tables || !inputs || !outputs || !result_slot || !counters || !repetitions) return 1;\n'
        '    counters[0]=0; counters[1]=0;\n'+'\n'.join(checks+lines)+'\n}\n').encode()


@dataclass
class InvariantLoopArtifact:
    plan: dict
    directory: Path
    kernels: tuple
    library: object
    run_native: object
    report: dict

    def run(self,inputs,trip_count,original_loop=None):
        """Run a graph-admitted sequence; caller replacement stays external."""
        import numpy as np
        if type(trip_count) is not int or not -(2**63)<=trip_count<2**63:
            raise ValueError('exact signed int64 loop count required')
        # Python range never evaluates its body on a nonpositive count.
        if trip_count<=0:
            self.report['last_execution']=dict(hoisted_calls=0,residual_calls=0,status=0,
                output_alias='ORIGINAL_CARRY')
            return inputs[self.plan['carry_parameter']]
        counters=(ctypes.c_uint64*2)();slot=ctypes.c_uint64()
        try:
            parameters=self.plan['body']['parameters']
            if type(inputs) is not dict or set(inputs)!=set(parameters):raise cpp.GuardFailure('original body input mapping differs')
            arrays=[inputs[p] for p in parameters];descriptors=[cpp._descriptor(a) for a in arrays]
            shape=arrays[0].shape
            if any(a.shape!=shape for a in arrays):raise cpp.GuardFailure('LICM requires equal original array shapes')
            count=len(self.plan['hoisted_roots'])+2
            if shape[0]*shape[1]*8*count>self.plan['memory_budget_bytes']:
                raise cpp.GuardFailure('LICM private payload exceeds explicit budget')
            private=[np.empty(shape,dtype=np.float64) for _ in range(count)]
            outputs=(cpp._Array*count)(*(cpp._descriptor(a) for a in private))
            tables=[]
            for kernel in self.kernels:
                getter=kernel._library.scar_get_operator_table_v1
                tables.append(getter())
            table_array=(ctypes.POINTER(cpp._Table)*len(tables))(*tables)
            status=self.run_native(table_array,(cpp._Array*len(descriptors))(*descriptors),outputs,
                trip_count,ctypes.byref(slot),counters)
            self.report['last_execution']=dict(hoisted_calls=counters[0],residual_calls=counters[1],status=status)
            if status:raise cpp.GuardFailure('LICM private native work rejected: '+str(status))
            return private[slot.value]
        except (cpp.GuardFailure,MemoryError) as error:
            self.report['last_execution']=dict(hoisted_calls=counters[0],residual_calls=counters[1],
                fallback=original_loop is not None,reason=str(error))
            if original_loop is None:raise
            return original_loop()


def build_invariant_loop(body,out,*,carry_parameter,loop_interface,memory_budget_bytes,hoist_invariants=True):
    """Emit host scheduling only; existing C++ backend emits every math leaf."""
    plan=plan_invariant_loop(body,carry_parameter=carry_parameter,loop_interface=loop_interface,
        memory_budget_bytes=memory_budget_bytes,hoist_invariants=hoist_invariants)
    source=plan['body']['source']
    if _receipt(source['path'])['sha256']!=source['sha256']:raise ValueError('original body source changed')
    out=Path(out).resolve()
    if out.exists():raise ValueError('fresh invariant loop output required')
    out.mkdir(parents=True);(out/'plan.json').write_bytes(_json(plan))
    kernels=[]
    for i,region in enumerate([*plan['hoisted_regions'],plan['residual_region']]):
        execution=prepare_pipeline_leaf(optimize_region(region_from_dict(region),memory_budget_bytes=memory_budget_bytes))
        kernels.append(cpp.build(execution,out/('kernel-'+str(i))))
    raw=_source(plan,kernels);(out/'host.cpp').write_bytes(raw)
    header=Path(cpp.__file__).parent/'scar_abi_v1.h';(out/header.name).write_bytes(header.read_bytes())
    compiler=kernels[0].report['compiler'];options=kernels[0].report['compiler_options']
    library_path=out/'host.so';command=[*compiler,*options,str(out/'host.cpp'),'-o',str(library_path)]
    process=subprocess.run(command,capture_output=True,text=True)
    if process.returncode:raise RuntimeError('LICM host compilation failed:\n'+process.stderr)
    report=dict(schema='scar.native-invariant-loop',schema_version=1,plan_id=plan['id'],
        cxx_standard=20,operator_abi_version=1,host=_receipt(out/'host.cpp'),library=_receipt(library_path),
        kernels=[_receipt(k.directory/'artifact.json') for k in kernels],generator=_receipt(__file__),
        loop_interface=loop_interface,source_replacement_authorized=False,
        native_math_backend='EXISTING_SCAR_CPP_OPERATOR_ABI_V1',
        invariant_selection='TRANSITIVE_GRAPH_DEPENDENCIES_NOT_MANUAL_NODE_LIST',
        hoist_invariants=hoist_invariants,placement_policy=plan['placement_policy'],
        last_execution=None,physical_optimality_proven=False,selected_for_use=False)
    (out/'artifact.json').write_bytes(_json(report))
    return load_invariant_loop(out)


def load_invariant_loop(directory):
    """Replay the LICM cut and load existing math/host artifacts; never compile."""
    directory=Path(directory).resolve()
    plan=json.loads((directory/'plan.json').read_bytes())
    report=json.loads((directory/'artifact.json').read_bytes())
    if (report.get('schema'),report.get('schema_version'))!=('scar.native-invariant-loop',1):
        raise ValueError('versioned invariant loop artifact required')
    body=region_from_dict(plan['body'])
    expected=plan_invariant_loop(body,carry_parameter=plan['carry_parameter'],
        loop_interface=plan['loop_interface'],memory_budget_bytes=plan['memory_budget_bytes'],
        hoist_invariants=plan['hoist_invariants'])
    if semantic_sha256(expected)!=semantic_sha256(plan) or report['plan_id']!=plan['id']:
        raise ValueError('invariant loop plan differs from original dependency replay')
    if _receipt(body.source.path)['sha256']!=body.source.sha256:
        raise ValueError('invariant loop original body source changed')
    for key,path in (('host',directory/'host.cpp'),('library',directory/'host.so'),('generator',Path(__file__))):
        if _receipt(path)!=report[key]:raise ValueError('invariant loop receipt differs: '+key)
    regions=[*plan['hoisted_regions'],plan['residual_region']]
    if len(report['kernels'])!=len(regions):raise ValueError('invariant loop kernel count differs')
    kernels=[]
    for i,(region,receipt) in enumerate(zip(regions,report['kernels'])):
        folder=directory/('kernel-'+str(i))
        if _receipt(folder/'artifact.json')!=receipt:raise ValueError('invariant loop kernel receipt differs')
        kernel=cpp.load(folder)
        expected=prepare_pipeline_leaf(optimize_region(region_from_dict(region),
            memory_budget_bytes=plan['memory_budget_bytes']))
        if kernel.plan.region!=expected.region or kernel.plan.cost['memory_budget_bytes']!=plan['memory_budget_bytes']:
            raise ValueError('invariant loop kernel differs from original hoist/residual subgraph')
        kernels.append(kernel)
    if _source(plan,kernels)!=(directory/'host.cpp').read_bytes():
        raise ValueError('invariant loop host differs from graph routing replay')
    library=ctypes.CDLL(str(directory/'host.so'));run=library.scar_invariant_loop_v1
    run.argtypes=[ctypes.POINTER(ctypes.POINTER(cpp._Table)),ctypes.POINTER(cpp._Array),
        ctypes.POINTER(cpp._Array),ctypes.c_uint64,ctypes.POINTER(ctypes.c_uint64),ctypes.POINTER(ctypes.c_uint64)]
    run.restype=ctypes.c_int32
    return InvariantLoopArtifact(plan,directory,tuple(kernels),library,run,report)


__all__=['plan_invariant_loop','build_invariant_loop','load_invariant_loop','InvariantLoopArtifact']
