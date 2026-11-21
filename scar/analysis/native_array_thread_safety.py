"""Admit a private array task body and the exact bound BLAS dot paths for threads."""

import argparse
from hashlib import sha256
from inspect import getsource
import json
from pathlib import Path

from scar.analysis import seeded_array_task_family
from scar.codegen import native_array_entry, native_array_tape, native_seeded_array_entry as seeded
from scar.codegen import native_numpy_complex, native_numpy_observables, native_numpy_random
from scar.codegen import native_numpy_random_coefficients, native_numpy_vector_normalize
from scar.codegen.native_numpy_complex import _displacement, _instructions, _relocations, _symbols, _unrelocated
from scar.codegen.stateful_numpy_provider import _Image, _tool
from scar.ir.execution_recipe import artifact_reference, read_artifact


SCHEMA = 'scar.native-array-thread-safety'
_BLAS_HASH = '6cad8d2ad994ddc43d2ccdb0fb5d9458373ff1b87ef7ff420f2f94406eb8f082'
_MAX_DOT = 10_000
_DTYPE = '<c16'
_CODEGEN = Path(native_array_entry.__file__).parent

# Each body regeneration must use the reviewed effect boundary. Template hashes
# cover the control between private operations and the diagnostic-only writes.
_TEMPLATES = {
    'native_array_tape.cpp.in': '546bd03f00df4c05143e856409bb1e4227db1940222378eb164fd9a38422a146',
    'native_seeded_array_loop_profile.cpp.in': 'dde16e13ebcb7a8f94e13f1ad64d3d33acaa9c5c997a7c27bed768a596bc726d',
    'native_numpy_vector_normalize.cpp.in': 'd0a70fa21369b9dcf03254efa7b3c2c20c39fe84e4afac2acd6e66f8afb7600a',
    'native_numpy_observables.cpp.in': '600800c4246a6c28fef46050dfd07a03eb2c2151e29216e7de9eaeb26e38664c',
    'native_numpy_random.cpp.in': '83246a44a638adbc4cd8147d1e300df76e83b50a9594262aa04ecb30638a6283',
    'native_numpy_random_coefficients.cpp.in': '281bbf312d1d8c62f2ebd90c1411457cd4f155bd4785739f421d3d95c4fab535',
}
_RENDERERS = {
    native_array_tape._emit: 'b60c818145a5cae589b1179ce5cab041e9e2d2f85b7745207709c1aeea100135',
    seeded._program: 'ac8c432144d3d5ce2cfb47987d11ed7a44bf3be7f1b5a472019540245fe163eb',
}
_RUNTIME = dict(
    parallel_symbol='scipy_openblas_get_parallel64_', parallel_value=1,
    threads_symbol='scipy_openblas_get_num_threads64_', threads_value=1,
    core_symbol='scipy_openblas_get_corename64_', core_value='Haswell',
    initialize_and_check_once_before_team=True, no_provider_mutation=True,
    fenv=dict(rounding='FE_TONEAREST', mxcsr_clear_mask=0xe040, masked_fp_traps=True,
              save_and_restore_each_worker=True),
    exact_linked_provider=True, diagnostics='DISABLED_OR_EXCLUSIVE_PATH_PER_ORDINAL')


def _need(condition, reason):
    if not condition:
        raise ValueError('native array thread safety: ' + reason)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _templates():
    sources = {name: _ref(_CODEGEN / name, 'scar.cpp-audited-template') for name in _TEMPLATES}
    _need(all(sources[name]['sha256'] == expected for name, expected in _TEMPLATES.items())
          and all(sha256(getsource(fn).encode()).hexdigest() == expected for fn, expected in _RENDERERS.items()),
          'task-body lowering has changed and requires a new review')
    return sources


def _body(native, family):
    recipe = read_artifact(native['recipe'])
    binding = read_artifact(native['binding'])
    memory = read_artifact(native['memory'])
    _need(family['loop'] == native['loop'] and family['source_contract'] == recipe['source_contract']
          == binding['source_contract'] and family['state']['kind'] == 'PER_TASK_PRIVATE'
          and family['publication']['after_all_workers'] is True,
          'source does not prove an ordered independent task family')
    route, = (row for row in binding['tapes'] if row['tape'] == memory['tape'])
    tape, schedule = read_artifact(memory['tape']), read_artifact(memory['schedule'])
    _, _, _, periodic, tape, schedule, route, memory = seeded._loop_plan(
        native['loop']['uri'], native['periodic_memory']['uri'], native['binding'], native['memory'],
        tape, schedule, route, memory)
    rng = read_artifact(binding['random'])
    coefficient = read_artifact(binding['coefficient_provider'])
    program, private_tape, count = seeded._program(tape, schedule, route, binding, memory, rng)
    carry = periodic['carry_into_steady']
    cpp = native_array_entry._substitute(seeded._LOOP_PROFILE.read_text(), dict(
        SEEDED_ROUTES=program, STATE_BYTES=carry['bytes'], STEADY_DESTINATION=carry['destination_offset']))
    extents = [row for row in schedule['operations'] if row['kind'] == 'NORMALIZE']
    _need(carry['bytes'] > 0 and carry['bytes'] % 16 == 0 and extents
          and all(row['elements'] == carry['bytes'] // 16 and row['source']['dtype'] == row['target']['dtype'] == _DTYPE
                  for row in extents), 'state-vector extent differs across phases or numerical ABI')
    _need(carry['bytes'] // 16 <= _MAX_DOT and count == native['prepared_runtime_coefficients']
          and native['arena_bytes'] == periodic['arena_bytes'], 'task exceeds audited dot branch or working set')
    norm, _, library, provider = native_array_tape._normalize(recipe['normalize'])
    _, coefficient_code = native_numpy_random_coefficients.render(coefficient['cpython'], coefficient['complex_provider'])
    code = {'src/profile_0.cpp': cpp.encode(), 'src/tape_0.inc': private_tape,
            **norm, **native_numpy_observables.render(provider), **coefficient_code,
            **native_numpy_random.render(rng['provider'])}
    paths = {name: native['files'][name] for name in code if not name.startswith('licenses/')}
    _need(all(read_artifact(ref, json_document=False) == code[name] for name, ref in paths.items()),
          'released body bytes differ from regeneration')
    binary = native['files'][native['binary']]['uri']
    read_artifact(native['files'][native['binary']], json_document=False)
    profile, = (row for row in recipe['profiles'] if row['tape_sha256'] == route['tape']['sha256'])
    native_numpy_complex.audit(binary, read_artifact(profile['package'])['stock_provider'])
    native_numpy_random._audit(binary, rng['provider'])
    native_numpy_vector_normalize._audit_divide(binary, provider)
    return carry['bytes'] // 16, native['files']['provider/objects/' + library], provider, paths


def _function(image, symbols, relocations, name, address=None):
    choices = [row for row in symbols[name] if address is None or row[0] == address]
    _need(len(choices) == 1, 'missing reviewed ELF symbol extent: ' + name)
    (base, size, _), = choices
    _unrelocated(relocations, base, size)
    rows = _instructions(image, base, size)
    return rows, dict(symbol=name, address=base, extent=size, sha256=sha256(image.read(base, size, 5)).hexdigest())


def _dot_closure(image, symbols, relocations, wrapper, target, slot, impl, micro):
    function = lambda name, address=None: _function(image, symbols, relocations, name, address)
    wrapper_rows, entry = function(wrapper)
    rows, kernel = function(target)
    _need(any(f'QWORD PTR [rax+{slot:#x}]' in operand and op in ('call', 'jmp')
              for _, _, op, operand in wrapper_rows)
          and any('<gotoblas>' in operand for _, _, _, operand in wrapper_rows)
          and any(op == 'cmp' and operand == f'rdi,{_MAX_DOT:#x}' for _, _, op, operand in rows),
          'exact public dot route or private threshold changed')
    branches = {_displacement(pos, raw, operands, call=op == 'call') for pos, raw, op, operands in rows
                if op in ('call', 'jmp') and operands.endswith(f'<{impl}>')}
    _need(len(branches) == 1, 'private dot computation target changed')
    address, = branches
    child, private = function(impl, address)
    micros = [(pos, raw, operands) for pos, raw, op, operands in child if op == 'call']
    _need(len(micros) == 1 and micros[0][2].endswith(f'<{micro}>'), 'private dot call closure changed')
    pos, raw, operands = micros[0]
    leaf, microkernel = function(micro, _displacement(pos, raw, operands, call=True))
    _need(all('[rip' not in operand and 'fs:' not in operand and 'gs:' not in operand
              for _, _, op, operand in (*child, *leaf) if op not in ('nop', 'cs', 'data16'))
          and not any(op == 'call' for _, _, op, _ in leaf), 'private dot accesses an unqualified external scope')
    return dict(entry=entry, kernel=kernel, compute=private, microkernel=microkernel,
                dispatch_table_byte_offset=slot, maximum_private_elements=_MAX_DOT)


def _blas(reference, provider):
    _need(reference['sha256'] == provider['blas']['sha256'] == _BLAS_HASH
          and provider['blas_core'] == _RUNTIME['core_value'] and provider['blas_threads'] == 1,
          'unqualified linked OpenBLAS build or core')
    image = _Image(reference)
    symbols, relocations = _symbols(image), _relocations(image)
    table = _tool('readelf', '-Ws', image.path)
    bases = {int(rows[1], 16) for line in table.splitlines() if len(rows := line.split()) == 8
             and rows[3:6] == ['OBJECT', 'GLOBAL', 'PROTECTED'] and rows[-1] == 'gotoblas_HASWELL'}
    _need(len(bases) == 1, 'unqualified Haswell dispatch table')
    base, = bases
    entries = {int(row[0], 16): row[-3] for line in _tool('readelf', '-Wr', image.path).splitlines()
               if len(row := line.split()) == 7 and row[2] == 'R_X86_64_64' and row[-2:] == ['+', '0']}
    _need(entries.get(base + 0x368) == 'ddot_k_HASWELL' and entries.get(base + 0xe00) == 'zdotc_k_HASWELL'
          and 'gotoblas_init' in entries.values(), 'reviewed dispatch and loader-init relocations changed')
    functions = {}
    for key, wrapper, target, slot, compute, micro, stride in (
        ('ddot', 'scipy_cblas_ddot64_', 'ddot_k_HASWELL', 0x368, 'dot_compute', 'ddot_kernel_8.constprop.0', 2),
        ('zdotc', 'scipy_cblas_zdotc_sub64_', 'zdotc_k_HASWELL', 0xe00, 'zdot_compute', 'zdot_kernel_8.constprop.0', 1),
    ):
        functions[key] = dict(_dot_closure(image, symbols, relocations, wrapper, target, slot, compute, micro),
                              x_stride=stride, y_stride=stride)
    init = {name: _function(image, symbols, relocations, name)[1] for name in (
        'gotoblas_init', 'gotoblas_corename', _RUNTIME['parallel_symbol'], _RUNTIME['threads_symbol'],
        _RUNTIME['core_symbol'])}
    return dict(binary=reference, target_table=base, calls=functions, initialization_and_getters=init,
                shared_state='READ_ONLY_BOUND_CORE_TABLE_AND_THREAD_COUNT_AFTER_SERIAL_INIT',
                upstream={
                    'ddot': 'https://github.com/OpenMathLib/OpenBLAS/blob/v0.3.34/kernel/x86_64/ddot.c',
                    'zdot': 'https://github.com/OpenMathLib/OpenBLAS/blob/v0.3.34/kernel/x86_64/zdot.c',
                    'openmp_scope': 'https://www.openmathlib.org/OpenBLAS/docs/faq/',
                })


def project(family_path, native_path):
    family_ref = _ref(family_path, seeded_array_task_family.SCHEMA)
    family = seeded_array_task_family.check(family_ref['uri'])
    native_ref = _ref(native_path, seeded.LOOP_SCHEMA)
    native = read_artifact(native_ref)
    templates = _templates()
    elements, reference, provider, body = _body(native, family)
    return dict(schema=SCHEMA, schema_version=1, family=family_ref, native=native_ref,
                templates=templates, body=body, provider=_blas(reference, provider), runtime_guards=_RUNTIME,
                workload_bound=dict(dtype=_DTYPE, dot_elements=elements, maximum_private_dot_elements=_MAX_DOT),
                effects=dict(rng='PER_TASK_PCG64_STATE', allocation='PER_TASK_PRIVATE_ALIGNED_ARENA',
                             temporary_writes='CALLER_OWNED_DISTINCT_RECORD_AND_TASK_LOCAL_SPANS',
                             publication='OWNER_ONLY_IN_ORDINAL_ORDER_AFTER_JOIN', floating_state='THREAD_LOCAL'),
                qualification='PINNED_PRIVATE_OPENBLAS_DOT_BRANCH', deployment_authorized=False)


def check(path):
    document = read_artifact(_ref(path, SCHEMA))
    _need(document == project(document['family']['uri'], document['native']['uri']),
          'thread safety receipt differs on replay')
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family')
    parser.add_argument('--native')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check')
    args = parser.parse_args(argv)
    if args.check and not any((args.family, args.native, args.out)):
        report = check(args.check)
    elif args.family and args.native and args.out and args.check is None:
        report = project(args.family, args.native)
        with args.out.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
    else:
        parser.error('provide --check RECEIPT or --family FAMILY --native NATIVE --out RECEIPT')
    print(json.dumps(dict(qualification=report['qualification'], workload_bound=report['workload_bound'])))


if __name__ == '__main__':
    main()
