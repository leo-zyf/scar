"""Export an entire typed recursive region as a detached C++20 experiment."""

import argparse
import json
import math
import os
from pathlib import Path
import shlex
from shutil import copy2
import struct
import subprocess
from tempfile import TemporaryDirectory

from scar.analysis.scalar_recursive_region import SCHEMA as REGION
from scar.codegen import scalar_recursive_leaves, scalar_recursive_lowering, scalar_sorted_transition_library
from scar.codegen.native_project import build_native_project, write_native_project
from scar.codegen.scalar_standalone import _FLAGS, _LINK_FLAGS, _dependencies_for, _local, _require
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization import scalar_transition_ir as transition


SCHEMA = 'scar.scalar-recursive-standalone'
CASES = 'scar.scalar-recursive-scope-cases'
_EXE = 'scar_scalar_recursive'
_SOURCE = 'src/main.cpp'
_BINARY = 'bin/' + _EXE
_TEMPLATE = Path(__file__).with_name('scalar_recursive_standalone_v1.cpp.in')
_NATIVE = _TEMPLATE.parents[1] / 'native'
_HEADERS = ('scalar_recursive_runtime.hpp', 'scalar_ordered_state.hpp')


def _state(value, sorted_plan):
    fields = {name:sorted_plan['state_fields'][name] for name in sorted_plan['state_field_order']}
    if 'fields' in value:
        value = dict(zip(fields, value['fields'], strict=True))
    flat = transition._flatten(fields, value)
    return tuple(flat.values())


def _world(data, layout, sorted_plan):
    for field, specification in layout.items():
        value = data[field]
        kind = specification['type']
        if kind == 'state':
            value = _state(value, sorted_plan)
        elif kind == 'set':
            value = sorted(_state(row, sorted_plan) for row in value)
        elif kind == 'float':
            value = float.fromhex(value['float_hex'])
            _require(math.isfinite(value), 'deadline input must be an exact finite value')
        elif kind == 'move_map':
            value = [(_state(key, sorted_plan), tuple(record['fields'])) for key,record in value['items']]
        elif kind == 'entry_map':
            value = [((_state(key[0], sorted_plan),key[1]),tuple(record['fields'])) for key,record in value['items']]
        else:
            _require(kind == 'int' and transition.scalar._int64(value), 'receiver integer port differs from plan')
        yield specification['field'], kind, value


def _encoded(value):
    if type(value) is str:
        return value.encode('ascii').hex() + '\n'
    if type(value) is bool:
        return str(int(value)) + '\n'
    if type(value) is int:
        return str(value) + '\n'
    if type(value) is float:
        return str(struct.unpack('=Q', struct.pack('=d',value))[0]) + '\n'
    return ''.join(map(_encoded,value))


def _case(case, lowering, sorted_plan, table_versions):
    contract = lowering['world']
    version = case['table_version']
    _require(version in table_versions,'typed recursion input has an unrecognized physical scoring table')
    expected = 'SCAR_RECURSION_V1\n'
    result = case['result']
    _require(result is None or transition.scalar._int64(result), 'source recursive result type changed')
    expected += 'stop\n' if result is None else 'return\n' + _encoded(result)
    def state(part):
        return ''.join((_encoded(len(value)) if kind in ('entry_map','move_map','set') else '') + _encoded(value)
                       for _,kind,value in _world(case[part],contract,sorted_plan))

    expected += state('after')
    _, position, threshold, depth, root = lowering['formals']
    actual = case['actuals']
    _require(transition.scalar._int64(actual[threshold]) and transition.scalar._int64(actual[depth])
             and type(actual[root]) is bool, 'source recursive actual ports changed')
    operands = (_state(actual[position],sorted_plan),actual[threshold],actual[depth],actual[root])
    globals_ = tuple(case['module_i64'][name] for name in lowering['constant_ports'])
    _require(all(transition.scalar._int64(value) for value in globals_),'global candidate ports must be exact i64')
    raw = 'SCAR_RECURSIVE_INPUT_V1\n' + version + '\n' + _encoded(operands) + state('before') + _encoded(globals_)
    return raw.encode('ascii'), expected.encode('ascii')


def _resolve(ref):
    return read_artifact(ref)


def _observed_roots(capture, region):
    observation=read_artifact(capture['observation'])
    _require(observation['recursive_region']==region and observation['source']==capture['source'],
             'source probes belong to a different observed recursive region')
    by_activation={row['activation']:row for row in observation['root_probes']}
    actual=[row for row in capture['cases'] if row['oracle_kind']=='CAPTURED_DRIVER_ACTIVATION']
    _require(actual and len({row['activation'] for row in actual})==len(actual),
             'native validation must include distinct original driver activations')
    for case in actual:
        root=by_activation[case['activation']]
        comparable={name:{'items':value['items']} if type(value) is dict and 'ordered_sha256' in value else value
                    for name,value in root['after'].items()}
        _require(case['actuals']==root['actuals'] and case['before']==root['before']
                 and case['after']==comparable and case['result']==root['result']
                 and root['module_i64'].items() <= case['module_i64'].items(),
                 'native validation changed an observed driver activation')


def prepare(region_path, fusion_path, cases_path):
    region_ref = artifact_reference(Path(region_path).resolve(strict=True), REGION)
    fusion_ref = artifact_reference(Path(fusion_path).resolve(strict=True), scalar_sorted_transition_library.SCHEMA)
    cases_ref = artifact_reference(Path(cases_path).resolve(strict=True), CASES)
    region, fusion, capture = map(_resolve,(region_ref,fusion_ref,cases_ref))
    parent = _resolve(fusion['sorted'])
    sorted_plan = _resolve(parent['region'])
    scorer = _resolve(sorted_plan['scorer'])
    trans = _resolve(sorted_plan['transition'])
    _require(capture['source']==region['source']==sorted_plan['source'] and capture['region']==region_ref
             and capture['schema_version']==1 and len(capture['cases'])>=2, 'input cases belong to a different original region')
    _observed_roots(capture,region_ref)
    original_path = Path(fusion_ref['uri']).parent
    members = {key:read_artifact({**ref,'uri':str(original_path/key)}, json_document=False)
               for key,ref in fusion['members'].items() if key.startswith('include/')}
    source = scalar_sorted_transition_library._SOURCE
    original = read_artifact({**fusion['members'][source],'uri':str(original_path/source)},json_document=False).decode()
    leaves = scalar_recursive_leaves.render(sorted_plan,trans)
    lowering = scalar_recursive_lowering.lower(region, sorted_plan, trans, scorer, leaves)
    samples = [_case(item,lowering,sorted_plan,parent['abi']['table_slots']) for item in capture['cases']]
    receive = [*(port['field'] for port in lowering['world'].values()),*lowering['constant_ports'].values()]
    fields = {'COMBINED_SOURCE':original, 'LEAVES':leaves['source'],'BODY':lowering['body'],
              'PUBLISH_FIELDS':'\n'.join(f'    publish(world.{port["field"]});' for port in lowering['world'].values()),
              'RECEIVE_FIELDS':'\n'.join(f'    receive_field(input, world.{field});' for field in receive)}
    text = _TEMPLATE.read_text()
    for name,fragment in fields.items():
        text = text.replace('@SCAR_'+name+'@',fragment)
    _require('@SCAR_' not in text, 'recursive C++ material is incomplete')
    members.update({f'include/{name}':(_NATIVE/name).read_bytes() for name in _HEADERS})
    members[_SOURCE]=text.encode()
    return (region_ref,fusion_ref,cases_ref), members, samples, lowering


def _verify(binary, samples):
    for i, (raw, output) in enumerate(samples):
        run=subprocess.run([str(binary)],input=raw,cwd='/',env={},capture_output=True,timeout=80)
        if run.returncode or run.stderr or run.stdout!=output:
            from difflib import unified_diff

            delta='\n'.join(unified_diff(output.decode().splitlines(),run.stdout.decode().splitlines(),n=3))
            raise ValueError(f'case {i} differs from original state or outcome\n'+run.stderr.decode()+'\n'+delta[:6000])
    return b''.join(output for _, output in samples)


def export(region, fusion, cases, out):
    references, members, oracle, lowering = prepare(region,fusion,cases)
    out=Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    tool=shlex.split(os.environ.get('CXX','c++'))
    with TemporaryDirectory(prefix='.scar-recursive-',dir=out.parent) as temporary:
        root=Path(temporary)/'package'
        for name,data in members.items():
            target=root/name
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(data)
        write_native_project(root,_EXE,include_dirs=('include',),compile_options=_FLAGS,link_options=_LINK_FLAGS)
        binary,_=build_native_project(root,_EXE,tool)
        relocated=Path(temporary)/'relocated recursive probe'
        copy2(binary,relocated)
        output=_verify(relocated,oracle)
        (root/'samples.stdout').write_bytes(output)
        (root/'samples').mkdir()
        for i,(sample,_) in enumerate(oracle):
            (root/'samples'/f'{i}.stdin').write_bytes(sample)
        result=dict(schema=SCHEMA,schema_version=1,cxx_standard=20,region=references[0],fusion=references[1],
            cases=references[2],source=_local(root,_SOURCE,'scar.cpp-source'),binary=_local(root,_BINARY,'scar.elf'),
            output=_local(root,'samples.stdout','application/octet-stream'),dependencies=_dependencies_for(relocated),
            world=lowering['world'],configuration=lowering['constants'],
            configuration_ports=list(lowering['constant_ports']),
            input_protocol='SCAR_RECURSIVE_INPUT_V1', validated_cases=len(oracle),
            native_recursion=True,whole_driver_cpp=False,deployment_authorized=False)
        (root/'artifact.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
        root.rename(out)
    return result


def check(path):
    filename=Path(path).resolve(strict=True)
    result=read_artifact(artifact_reference(filename,SCHEMA))
    root=filename.parent
    refs,members,oracle,lowering=prepare(*(result[key]['uri'] for key in ('region','fusion','cases')))
    _require(tuple(result[key] for key in ('region','fusion','cases'))==refs
             and result['world']==lowering['world'] and result['configuration']==lowering['constants']
             and result['configuration_ports']==list(lowering['constant_ports'])
             and result['input_protocol']=='SCAR_RECURSIVE_INPUT_V1'
             and result['source']==_local(root,_SOURCE,'scar.cpp-source')
             and result['binary']==_local(root,_BINARY,'scar.elf')
             and result['output']==_local(root,'samples.stdout','application/octet-stream')
             and all((root/name).read_bytes()==data for name,data in members.items())
             and result['dependencies']==_dependencies_for(root/_BINARY)
             and result['validated_cases']==len(oracle) and result['native_recursion'] is True
             and result['whole_driver_cpp'] is False and result['deployment_authorized'] is False,
             'recursive artifact differs from exact source, ABI or members')
    _require(all((root/'samples'/f'{i}.stdin').read_bytes()==raw for i,(raw,_) in enumerate(oracle))
             and _verify(root/_BINARY,oracle)==(root/'samples.stdout').read_bytes(),
             'native recursive input or output differs from replay')
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('region','fusion','cases','out','check'):
        p.add_argument('--'+flag,type=Path)
    args=p.parse_args(argv)
    inputs=(args.region,args.fusion,args.cases,args.out)
    if args.check and not any(inputs):
        result=check(args.check)
    elif not args.check and all(inputs):
        result=export(*inputs)
    else:
        p.error('provide --region, --fusion, --cases and --out, or --check')
    print(json.dumps({key:result[key] for key in ('schema','native_recursion','validated_cases','deployment_authorized')}))


if __name__=='__main__':
    main()
