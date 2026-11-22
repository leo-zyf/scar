"""Prove aggregation of sticky floating-point guards over private stock steps."""

from collections import defaultdict
from itertools import groupby
from pathlib import Path
import re

from scar.analysis.cpu_task_arena_reuse import verify_cpu_task_arena_reuse
from scar.codegen import native_numpy_complex as stock, native_seeded_array_task_entry as generator
from scar.codegen.native_provider_bundle import _dependencies
from scar.ir.execution_recipe import artifact_reference, read_artifact


SCHEMA = 'scar.cpu-fp-guard-groups'
_STATUS = 'PROVEN_SCOPED_MASKED_STICKY_FLAGS'
_NAMES = ('FE_DIVBYZERO', 'FE_INVALID', 'FE_OVERFLOW', 'FE_UNDERFLOW')
_DIAGNOSTICS = ('SCAR_ARRAY_ENTRY_STATES', 'SCAR_ARRAY_ENTRY_LANE_REPORT')
_ARITHMETIC = {'add', 'multiply'}
_KINDS = _ARITHMETIC | {'basis', 'literal', 'copy', 'normalize', 'checkpoint'}
_LIBM = '1b87a1a50b496cfead2b0ad134c2ff536705c82608db240c7e8aa48d6c0e4217'
_STOCK_TEXT = {
    'add': '0657cee0872399048f4e89de35362a1223e65ed2873d68ec200e4b026061da07',
    'multiply': 'fabaecfe360d157644fde49dc5b170af2f93c37069c25f076ed96043587220cb',
    'pairwise_sum': '6138a7d72ce5f2ada7d2994d2c0235c692e4a28eea1dfa633010550295f1d190',
}
_GUARDS = ('feclearexcept', 'fetestexcept')
_STEP = re.compile(r'    \{Op::([a-z]+), ([0-9]+), ([0-9]+),(?: \{[-0-9, ]+\},){3} -?[0-9]+\},')
_ROUND = re.compile(r'    \{([0-9]+), ([0-9]+)\},')
_HOOK = re.compile(r'    \{([0-9]+), HookKind::(coefficient|normalization), ([0-9]+)\},')
_SYMBOL = '@GLIBC_2.2.5'


def _need(ok, detail):
    if not ok:
        raise ValueError('CPU FP guard groups: ' + detail)


def _table(source, signature, grammar):
    """Decode a specific emitted constant table from its pinned member."""
    _need(source.count(signature) == 1, 'original generated table is absent or ambiguous')
    body = source.split(signature, 1)[1].split('\n};', 1)[0]
    rows = [grammar.fullmatch(row) for row in body.splitlines()]
    _need(rows and all(row is not None for row in rows), 'unexpected original generated row')
    return [row.groups() for row in rows]


def _phase(kind, begin, end, normalize, checkpoint, steps, hooks):
    applicable = lambda index: steps[index][0] in _ARITHMETIC and index not in hooks
    groups = []
    arithmetic = 0
    calls = 0
    count = 0
    for arithmetic_run, group in groupby(range(begin, normalize), key=applicable):
        members = list(group)
        if not arithmetic_run:
            continue
        count += 1
        arithmetic += len(members)
        calls += sum(steps[index][1] for index in members)
        if len(members) > 1:
            groups.append(dict(begin=members[0], end=members[-1] + 1))
    return dict(phase=kind, begin_compiled_step=begin, end_compiled_step=end,
                normalization_compiled_step=normalize, checkpoint_compiled_step=checkpoint,
                arithmetic_steps=arithmetic, stock_provider_calls=calls, all_groups=count,
                groups=groups, baseline_guard_pairs=arithmetic, candidate_guard_pairs=count)


def _phases(arena):
    members = arena['evidence']['runtime_members']
    raw_tape = read_artifact(members['src/tape_0.inc'], json_document=False).decode('ascii')
    profile = read_artifact(members['src/profile_0.cpp'], json_document=False).decode('ascii')
    periodic = read_artifact(arena['evidence']['storage']['periodic_memory'])
    schedule = read_artifact(arena['evidence']['storage']['schedule'])
    mask = 'constexpr auto fp_errors = ' + ' | '.join(_NAMES) + ';'
    _need(raw_tape.count(mask) == 1, 'original arithmetic guard mask differs')
    raw_steps = _table(raw_tape, 'constexpr Step steps[]{\n', _STEP)
    rounds = [tuple(map(int, row)) for row in _table(raw_tape, 'constexpr std::size_t rounds[][2]{\n', _ROUND)]
    _need(all(kind in _KINDS and int(count) > 0 and int(repeat) > 0 for kind, count, repeat in raw_steps),
          'unknown operation or repeated-step shape')
    steps = [(kind, int(repeat)) for kind, _, repeat in raw_steps]
    hooks = {}
    for coordinate, kind, _ in _table(profile, 'constexpr Hook hooks[]{\n', _HOOK):
        index = int(coordinate)
        _need(0 <= index < len(steps) and index not in hooks
              and steps[index][0] == {'coefficient': 'literal', 'normalization': 'normalize'}[kind],
              'runtime hook overlaps arithmetic or crosses its original coordinates')
        hooks[index] = kind
    compiled = [0]
    for row in schedule['operations'][:periodic['steady']['operations'][-1]]:
        compiled.append(compiled[-1] + (len(row['groups']) if row['kind'] == 'COPY' else 1))
    expected = [tuple(compiled[index] for index in periodic[phase]['operations']) for phase in ('first', 'steady')]
    _need(rounds == expected and rounds[0][0] == 0 and rounds[0][1] == rounds[1][0]
          and rounds[1][1] == len(steps), 'first and steady selectors differ from carried source phases')
    phases = []
    for phase, (begin, end) in zip(('first', 'steady'), rounds, strict=True):
        row = periodic[phase]
        normalize, checkpoint = (compiled[row[name]] for name in ('normalize_operation', 'checkpoint_operation'))
        if phase == 'steady':
            _need(compiled[periodic['carry_into_steady']['operation']] == begin and steps[begin][0] == 'copy',
                  'original steady handoff is not its own barrier')
            begin = compiled[periodic['carry_into_steady']['operation'] + 1]
        _need(begin < normalize < checkpoint < end and checkpoint + 1 == end
              and hooks.get(normalize) == 'normalization' and steps[checkpoint][0] == 'checkpoint'
              and all(steps[index][0] not in ('normalize', 'checkpoint') for index in range(begin, normalize)),
              'arithmetic could cross an original result selection or checkpoint')
        phases.append(_phase(phase.upper(), begin, end, normalize, checkpoint, steps, hooks))
    return phases, dict(runtime_hook_count=len(hooks), compiled_tape_steps=len(steps),
                        phase_barriers=['COPY', 'BASIS', 'LITERAL', 'COEFFICIENT_HOOK',
                                        'NORMALIZATION_HOOK', 'CHECKPOINT', 'END_OF_PHASE'])


def _fenv(binary):
    """Bind the reviewed x86 libm clear/test ABI under the clean ELF loader."""
    dependencies = _dependencies(stock._tool('ldd', binary['uri']))
    provider = artifact_reference(dependencies['libm.so.6'], 'scar.native-provider-library')
    _need(provider['sha256'] == _LIBM, 'unreviewed x86 fenv provider')
    symbols = defaultdict(set)
    imports = set()
    for module in (binary['uri'], *(path for path in dependencies.values() if path is not None)):
        for line in stock._tool('readelf', '-W', '--dyn-syms', module).splitlines():
            row = line.split()
            if len(row) < 8 or row[0][-1:] != ':' or row[4] not in ('GLOBAL', 'WEAK', 'UNIQUE'):
                continue
            name, at, version = row[7].partition('@')
            if name not in _GUARDS:
                continue
            if row[6] != 'UND':
                symbols[name].add((str(Path(module).resolve()), row[7], row[3]))
            elif module == binary['uri'] and row[3] == 'FUNC' and at + version == _SYMBOL:
                imports.add(name)
    _need(imports == set(_GUARDS) and all(symbols[name] == {(provider['uri'], name + '@' + _SYMBOL, 'FUNC')}
                                        for name in _GUARDS),
          'guard call target or version follows another loader route')
    return dict(provider=provider, symbol_version=_SYMBOL[1:], functions=list(_GUARDS), monitored_mask=0x1d,
                clear_normal_return=0, test_nonmutating=True,
                state='X87_AND_SSE_THREAD_FLAGS_ONLY', loader='CLEAN_NON_INTERPOSED_RUNTIME')


def _providers(arena, runtime):
    tape = arena['evidence']['storage']['original_tape']
    recipe = read_artifact(runtime['recipe'])
    profile, = (row for row in recipe['profiles'] if row['tape_sha256'] == tape['sha256'])
    binary = runtime['files'][runtime['binary']]
    checked = stock.audit(binary['uri'], read_artifact(profile['package'])['stock_provider'])
    rows = checked['functions']
    _need(checked == runtime['linkage']['stock']
          and {role: row['normalized_sha256'] for role, row in rows.items()} == _STOCK_TEXT
          and all(transfer['kind'] == 'LITERAL'
                  or transfer['kind'] == 'CALL' and transfer['target'] == 'pairwise_sum'
                  for row in rows.values() for transfer in row['transfers']),
          'stock opcode or closed private helper evidence differs')
    # stock.audit uses the closed opcode registry, including all recursive
    # pairwise calls. No x87/MXCSR status access or external call is admitted.
    return dict(binary=binary, stock=checked, fenv=_fenv(binary))


def derive_cpu_fp_guard_groups(arena_proof_ref):
    """Prove commuting the guard calls under the qualified native contract."""
    arena = verify_cpu_task_arena_reuse(arena_proof_ref)
    runtime = read_artifact(arena['runtime'])
    thread = read_artifact(arena['evidence']['thread_contract'])
    fenv = thread['runtime_guards']['fenv']
    _need(arena['status'] == 'PROVEN_SCOPED_NORMAL_SUCCESS'
          and arena['proof']['arena_pointer_does_not_escape_task'] is True
          and arena['proof']['lane_serial_and_disjoint'] is True
          and fenv == dict(rounding='FE_TONEAREST', mxcsr_clear_mask=0xe040, masked_fp_traps=True,
                           save_and_restore_each_worker=True)
          and runtime['decision']['mode'] == generator._MODE_OPENMP
          and runtime['no_worker_visible_output'] is True,
          'qualified private fixed OpenMP arithmetic with masked traps required')
    phases, barriers = _phases(arena)
    provider = _providers(arena, runtime)
    return dict(schema=SCHEMA, schema_version=1, arena_proof=arena_proof_ref, runtime=arena['runtime'],
                scope=arena['scope'], worker_cpu_ids=arena['worker_cpu_ids'], status=_STATUS, phases=phases,
                evidence=dict(provider=provider, runtime_members=arena['evidence']['runtime_members'],
                    thread_contract=arena['evidence']['thread_contract'], schedule=arena['evidence']['storage']['schedule'],
                    periodic_memory=arena['evidence']['storage']['periodic_memory'], **barriers),
                proof=dict(fp_operation_order_unchanged=True, normal_publication_unchanged=True,
                    flags_are_sticky_within_group=True, traps_masked=True, rounding_mode='FE_TONEAREST',
                    monitored_exceptions=list(_NAMES), guard_provider_exact=True,
                    same_failure_code=3, no_owner_publication_on_failure=True,
                    diagnostic_environment_variables=list(_DIAGNOSTICS), diagnostics_preserved_on_normal_success=True,
                    failure_qualification='CLEAN_NO_DIAGNOSTICS_FIXED_OPENMP_LAUNCH',
                    failure_order='EARLIEST_ORDINAL_FAILURE_REMAINS_AFTER_LANE_JOIN',
                    failed_internal_prefix_preserved=False, failed_sticky_mask_exact=False,
                    requires_no_fenv_observer_or_interposition=True),
                open_effects=['FAILED_PRIVATE_OPERATION_PREFIX_AND_INTERMEDIATE_FLAGS',
                              'FAILURE_WITH_CONCURRENT_DIAGNOSTIC_PUBLICATIONS',
                              'TIMING_AND_ALLOCATOR_OR_EXTERNAL_OBSERVERS'],
                target_executed=False, transformation_authorized=False, deployment_authorized=False)


def verify_cpu_fp_guard_groups(reference):
    saved = read_artifact(reference)
    _need(reference['schema'] == SCHEMA and saved == derive_cpu_fp_guard_groups(saved['arena_proof']),
          'stored FP guard proof differs from cold replay')
    return saved
