"""Cold-replay a compiled workspace session without compilation or CDLL.

This validates retained source/geometry/protocol and native bytes under the
original producer receipts. It does not execute a getter, prove machine-code
semantics or authorize replacement of an original Python caller.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import struct
from types import SimpleNamespace

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.resource_solver import semantic_sha256


SCHEMA = 'scar.workspace-session-cold-read'
COMPILED = 'scar.compiled-workspace-session'


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _same(left, right):
    return semantic_sha256(left) == semantic_sha256(right)


def _library(path):
    path = Path(path)
    raw = path.read_bytes()
    return dict(library=path.name, sha256=sha256(raw).hexdigest(), bytes=len(raw))


def _elf(reference, symbols):
    """Check ELF64 receipts and defined exported functions, without loading."""
    raw = read_artifact(reference, json_document=False)
    _need(len(raw) >= 64 and raw[:7] == b'\x7fELF\x02\x01\x01'
          and struct.unpack_from('<HH', raw, 16) == (3, 62),
          'original x86-64 little-endian shared ELF required')
    offset = struct.unpack_from('<Q', raw, 40)[0]
    entry_size, count = struct.unpack_from('<HH', raw, 58)
    _need(entry_size == 64 and count > 0 and offset+count*entry_size <= len(raw),
          'bounded ELF64 section table required')
    sections = [struct.unpack_from('<IIQQQQIIQQ', raw, offset+i*entry_size) for i in range(count)]
    exported = set()
    for row in sections:
        if row[1] != 11:  # SHT_DYNSYM
            continue
        _need(row[9] == 24 and row[5] % 24 == 0 and row[6] < count
              and row[4]+row[5] <= len(raw), 'invalid ELF dynamic symbol table')
        strings = sections[row[6]]
        _need(strings[1] == 3 and strings[4]+strings[5] <= len(raw), 'invalid ELF dynamic string table')
        names = raw[strings[4]:strings[4]+strings[5]]
        for at in range(row[4], row[4]+row[5], 24):
            name, info, visibility, section = struct.unpack_from('<IBBH', raw, at)
            if info >> 4 in (1, 2) and info & 15 == 2 and visibility & 3 == 0 and section:
                _need(name < len(names) and names.find(b'\0', name) >= 0, 'invalid exported ELF symbol name')
                exported.add(names[name:names.index(b'\0', name)])
    _need(all(symbol.encode() in exported for symbol in symbols), 'session/base exported ABI function is missing')
    return raw


def inspect_workspace_session(compiled_report):
    from scar.compiler.workspace_session import replay_session, NATIVE
    from scar.codegen.workspace_session import render_workspace_session
    from scar.compiler.export_stateful import project_stateful_contract
    from scar.compiler.stateful_model import stateful_from_dict
    from scar.runtime.native_workspace_session import SCHEMA as RUNTIME, ADMISSION, inspect_bundle, implementation_sha256
    path = Path(compiled_report).resolve()
    if path.is_dir():
        path /= 'result.json'
    reference = artifact_reference(path, COMPILED)
    report = read_artifact(reference)
    _need(type(report.get('schema_version')) is int and report['schema_version'] == 1,
          'versioned compiled session publication required')
    directory = path.parent
    prep_ref, prep, plan, effects, context, residency = replay_session(report['preparation']['uri'])
    _need(prep_ref == report['preparation'], 'compiled session preparation differs from full replay')
    original_cpp = read_artifact(plan['native_blueprint']['cpp_source'], json_document=False)
    source, mapping = render_workspace_session(original_cpp, context['native_metadata'], residency, plan)
    native = directory/'native'
    native_ref = artifact_reference(native/'artifact.json', NATIVE)
    _need(native_ref == report['native'], 'compiled native manifest path or bytes differ')
    metadata = read_artifact(native_ref)
    source_ref = artifact_reference(native/'session.cpp', 'scar.cpp-source')
    library_ref = artifact_reference(native/('session_'+mapping['execution_plan_sha256'][:20]+'.so'), 'scar.native-library')
    expected_metadata = dict(schema=NATIVE, schema_version=1, preparation=prep_ref, session_plan=prep['session_plan'],
        source_mapping=mapping, execution_plan_sha256=mapping['execution_plan_sha256'],
        compiler=context['native_metadata']['compiler'], source=source_ref, library=library_ref)
    _need(_same(metadata, expected_metadata) and read_artifact(source_ref, json_document=False) == source,
          'compiled session source, mapping, geometry, compiler or native receipt differs from renderer replay')
    _need(metadata['execution_plan_sha256'] == semantic_sha256(mapping['execution_plan'])
          and mapping['execution_plan']['parent_execution_plan_sha256'] == context['native_metadata']['execution_plan_sha256']
          and mapping['execution_plan']['session_plan_sha256'] == semantic_sha256(plan)
          and mapping['execution_plan']['residency_plan_sha256'] == semantic_sha256(residency),
          'session execution identity is not independently bound to its source and dirty protocol')
    parent = Path(context['native_manifest']['uri']).parent
    header_refs = {}
    for name in ('stateful_abi_v1.h', 'scar_abi_v1.h'):
        expected = (parent/context['native_metadata']['files'][name]).read_bytes()
        actual = (native/name).read_bytes()
        _need(actual == expected and sha256(actual).hexdigest() == context['native_metadata']['sha256'][name],
              'compiled ABI header differs from original admitted header: '+name)
        header_refs[name] = artifact_reference(native/name, 'scar.cpp-header')
    base_ref = context['native_metadata']['base_native_reference']
    base = read_artifact(base_ref); base_dir = Path(base_ref['uri']).parent
    base_library = base_dir/base['files']['library']
    _need(sha256(base_library.read_bytes()).hexdigest() == base['sha256']['library'], 'original base ELF changed')
    base_library_ref = artifact_reference(base_library, 'scar.native-library')
    base_region = stateful_from_dict(read_artifact(artifact_reference(
        base_dir/base['files']['region'], 'scar.stateful-array-region')))
    step = project_stateful_contract(SimpleNamespace(directory=base_dir, region=base_region,
        plan=context['step_plan'], report=base), selection=None, require_selection=False)
    bundle_path = directory/'bundle'
    _need(report['bundle'] == str(bundle_path), 'compiled bundle path differs')
    bundle = inspect_bundle(bundle_path, expected_admission_sha256=report['expected_admission_sha256'])
    contract, admission = bundle['contract'], bundle['admission']
    compact_effects = [dict(source=row['summary']['source'], shape=row['summary']['roots'][0]['shape'],
        summary_id=row['summary']['id'], writes=[a['rectangle'] for a in row['summary']['accesses']
            if a['kind'] == 'WRITE_COPY_DESTINATION']) for row in effects]
    expected_contract = dict(schema=RUNTIME, schema_version=1, step_contract=step, protocol=plan['protocol'],
        geometry=plan['geometry'], copy_effects=compact_effects, caller_replacement_authorized=False,
        native=dict(base=_library(base_library), session=_library(library_ref['uri']), symbols=plan['symbols'],
            execution_plan_sha256=mapping['execution_plan_sha256']),
        provider_boundary_Q=plan['provider_boundary_Q'], default_backend='NO_OP', trial_only=True)
    _need(_same(contract, expected_contract), 'compact contract differs from source/base/effect/session projection')
    runtime_ref = artifact_reference(bundle_path/'runtime.json', RUNTIME)
    admission_ref = artifact_reference(bundle_path/'admission.json', ADMISSION)
    expected_admission = dict(schema=ADMISSION, schema_version=1, runtime_sha256=runtime_ref['sha256'],
        runtime_implementation_sha256=implementation_sha256(), native=expected_contract['native'], preparation=prep_ref,
        native_manifest=native_ref, session_plan=prep['session_plan'], caller_replacement_authorized=False)
    _need(_same(admission, expected_admission) and admission_ref['sha256'] == report['expected_admission_sha256'],
          'compact admission does not bind the exact rebuilt native/source/runtime contract')
    for role, original in (('base', base_library), ('session', Path(library_ref['uri']))):
        _need((bundle_path/contract['native'][role]['library']).read_bytes() == original.read_bytes(),
              'compact ELF is not byte-identical to its source-admitted original: '+role)
    expected_report = dict(schema=COMPILED, schema_version=1, preparation=prep_ref, native=native_ref,
        bundle=str(bundle_path), expected_admission_sha256=admission_ref['sha256'],
        source_caller_effect_coverage_proven=False, caller_replacement_authorized=False,
        default_backend='NO_OP', deployment_authorized=False)
    _need(_same(report, expected_report), 'compiled publication changed its scope or retained artifact references')
    raw = _elf(library_ref, tuple(plan['symbols'].values()))
    _elf(base_library_ref, ('scar_get_stateful_table_v1',))
    _need(mapping['execution_plan_sha256'].encode() in raw, 'session ELF lacks the independently derived plan constant')
    read_artifact(reference); read_artifact(prep_ref); read_artifact(native_ref)
    return dict(schema=SCHEMA, schema_version=1, status='SOURCE_AND_ARTIFACT_COLD_REPLAY_PASSED',
        compiled=reference, preparation=prep_ref, session_plan=prep['session_plan'], native=native_ref,
        source=source_ref, library=library_ref, headers=header_refs, original_base_manifest=base_ref,
        original_base_library=base_library_ref, runtime=runtime_ref, admission=admission_ref,
        execution_plan_sha256=mapping['execution_plan_sha256'], exported_session_symbols=plan['symbols'],
        exact_source_mapping_replayed=True, independent_session_plan_sha256_checked=True,
        compact_projection_replayed=True, compact_elf_bytes_identical=True, header_bytes_checked=True,
        plan_constant_present_in_elf=True, native_getters_executed=False,
        source_caller_effect_coverage_proven=False, caller_replacement_authorized=False,
        default_backend='NO_OP', deployment_authorized=False, physical_optimality_proven=False,
        machine_code_semantics_proven=False, compiler_invoked=False, native_libraries_loaded=False,
        target_program_executed=False, performance_measured=False,
        checker=artifact_reference(Path(__file__).resolve(), 'scar.python-source'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('compiled_report')
    args = parser.parse_args(argv)
    print(json.dumps(inspect_workspace_session(args.compiled_report), sort_keys=True, indent=2, allow_nan=False))
    return 0


__all__ = ['inspect_workspace_session', 'main']

if __name__ == '__main__':
    raise SystemExit(main())
