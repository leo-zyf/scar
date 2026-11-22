"""Validate a source-compatible C++20 CLI against one exact CPU catalog scope."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
from time import perf_counter_ns

from scar.codegen.native_provider_bundle import _content
from scar.compiler import aten_source_entry, cpu_entry_replay, fixed_grid_source_provider
from scar.compiler.calibrate_array_entry import _write
from scar.compiler.entry_delivery import _aten, _aten_validation, _environment_digest
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.paired_entry_selection import compare_entry_pairs


SCHEMA = 'scar.cpu-fixed-grid-cli-delivery'
_CHAIN = 'scar.whole-entry-main-chain'
_OBSERVATION = 'scar.aten-source-entry-validation'
_PROJECT = fixed_grid_source_provider.PROJECT_SCHEMA
_OUTPUT = fixed_grid_source_provider.OUTPUT_SCHEMA
_ORIGINAL, _NATIVE = 'ORIGINAL_PYTHON', 'NATIVE_CPP'
_TIMING_SCOPE = 'SOURCE_CLI_PROCESS_LAUNCH_TO_EXIT_INCLUDING_FILE_IO'
_PAIRS = 3
_WARMUPS = 1
_OUTPUT_TOKEN = '{output_dir}/final.f32'


def _need(ok, detail):
    if not ok:
        raise ValueError('CPU fixed-grid CLI delivery: ' + detail)


def _raw(reference):
    return read_artifact(reference, json_document=False)


def _regular(path):
    return stat.S_ISREG(path.lstat().st_mode)


def _inputs(main_chain_ref, project_ref):
    main = read_artifact(main_chain_ref)
    _need(main_chain_ref['schema'] == _CHAIN and main['schema_version'] == 1
          and main['plan_only'] is False and main['trial']['deployment_authorized'] is False,
          'retained CPU catalog trial required')
    domain = read_artifact(main['domain'])
    scope = domain['scope']
    trial = main['trial']
    reviewed = trial['provider_replay']
    paired_ref = scope['cpu_paired_observation']
    _need(trial['scope'] == scope and trial['final_action'] == 'KEEP_ORIGINAL'
          and reviewed['reference'] == paired_ref and paired_ref['schema'] == cpu_entry_replay.SCHEMA,
          'retained catalog original replay must bind this exact argument scope')
    pair = cpu_entry_replay.check(paired_ref['uri'])
    _need(pair == read_artifact(paired_ref) and pair['capture'] is None and pair['cpu_override'] is None
          and pair['deployment_authorized'] is False, 'fixed-grid catalog pair does not replay exactly')
    compilation = read_artifact(pair['compilation'])
    compiled = read_artifact(compilation['compilation'])
    authority_ref = compiled['provider']
    authority = read_artifact(authority_ref)
    _need(authority_ref['schema'] == fixed_grid_source_provider.SCHEMA
          and compiled == read_artifact(reviewed['provider_compilation'])
          and compiled['entry'] == compilation['entry'] == authority['entry']
          and compiled['argv'] == compilation['argv'] == scope['argv'],
          'CPU catalog provider does not bind the declared fixed-grid source')
    invocation = pair['invocation']
    original = invocation['arms']['original']
    cpus = invocation['cpu_ids']
    cwd = invocation['cwd']
    environment = original['environment']
    original_executable = original['executable']
    _raw(original_executable)
    argv = scope['argv']
    entry = authority['entry']
    _need(invocation['entry'] == entry and entry['uri'] == scope['entry'] and entry['sha256'] == scope['entry_sha256']
          and cpus == scope['process_affinity_cpu_ids'] and cpus == sorted(set(cpus)) and cpus
          and cwd == scope['working_directory'] and cwd == authority['context']['cwd']
          and Path(cwd).resolve(strict=True) == Path(cwd)
          and _environment_digest(environment) == scope['environment_sha256']
          and original_executable == authority['context']['interpreter']
          and Path(original_executable['uri']).resolve(strict=True) == Path(scope['python_executable'])
          and original['argv'] == [original_executable['uri'], '-B', entry['uri'], *argv[:-1], _OUTPUT_TOKEN],
          'source, launch ABI, environment or CPU catalog original differs')
    parent_ref = authority['source_project']
    parent, project = read_artifact(parent_ref), read_artifact(project_ref)
    rewritten = {'src/main.cpp', 'src/runtime/file_io.hpp', parent['executable'],
                 'validation/build.log', 'validation/ldd.txt'}
    shell_members = {aten_source_entry.ABI_FILE, aten_source_entry.BINDING_FILE, aten_source_entry.LAUNCHER}
    _need(parent_ref['schema'] == project_ref['schema'] == _PROJECT and 'source_entry_cli' not in parent
          and project['source_entry_cli']['parent_manifest_sha256'] == parent_ref['sha256']
          and {key: value for key, value in project.items() if key not in {'files', 'source_entry_cli'}}
              == {key: value for key, value in parent.items() if key != 'files'}
          and set(project['files']) == parent['files'].keys() | shell_members
          and all(project['files'][name] == identity for name, identity in parent['files'].items()
                  if name not in rewritten),
          'source CLI must preserve the exact catalog parent plan and generated body')
    evidence = dict(artifact=project_ref, **_aten(scope, project_ref, project))
    _need(evidence['source'] == entry and evidence['cpu_ids'] == cpus
          and evidence['steps'] == int(argv[1])
          and all(pin['uri'] == row['path'] and all(pin[key] == row[key] for key in ('sha256', 'size_bytes'))
                  for pin, row in zip(authority['inputs'], evidence['file_inputs'], strict=True)),
          'native source CLI CPU cohort, step count or input file identity differs')
    outputs = invocation['files']
    _need(set(outputs) == {'final.f32'}, 'one exact catalog publication required')
    final = outputs['final.f32']
    _raw(final)
    _raw(original['stdout'])
    tools = ('/usr/bin/taskset', '/bin/sh', '/usr/bin/env', '/usr/bin/dirname')
    return dict(main_chain=main_chain_ref, domain=main['domain'], retained_catalog_pair=paired_ref,
                provider=authority_ref, parent_project=parent_ref, project=project_ref, scope=scope, evidence=evidence,
                native_executable=artifact_reference(Path(project_ref['uri']).parent/project['executable'],
                                                      'application/x-executable'),
                working_directory=cwd, environment=environment, original_stdout=original['stdout'],
                final=final, commands={
                    _ORIGINAL: ['/usr/bin/taskset', '-c', ','.join(map(str, cpus)), *original['argv'][:-1], argv[-1]],
                    _NATIVE: [evidence['launcher']['uri'], *argv],
                }, native_placement=read_artifact(evidence['plan'])['placement'],
                launch_tools={tool: artifact_reference(tool, 'application/x-executable') for tool in tools})


def _record(inputs, validation_ref):
    scope, evidence = inputs['scope'], inputs['evidence']
    observed = read_artifact(validation_ref)
    _need(observed['commands'] == inputs['commands'] and observed['working_directory'] == inputs['working_directory']
          and observed['python_provider_environment'] == inputs['environment']
          and observed['full_process_measured'] is True, 'cold observation must match exact bounded native launch')
    decision, _, launch = _aten_validation(scope, evidence, validation_ref, .05)
    _need(all(row['stdout'].encode() == _raw(inputs['original_stdout'])
              and row['output']['sha256'] == inputs['final']['sha256']
              and row['output']['size_bytes'] == inputs['final']['size_bytes']
              for row in observed['observations']), 'standalone observations differ from prior original output bytes')
    modes = observed['publication_modes']
    _need(modes == [dict(pair=row['pair'], mode=row['mode'],
                        permission_bits=stat.S_IMODE(Path(row['output']['path']).lstat().st_mode))
                    for row in observed['observations']], 'permission mode differs from retained output')
    exact_mode = all(modes[index]['permission_bits'] == modes[index+1]['permission_bits']
                     for index in range(0, len(modes), 2))
    return dict(schema=SCHEMA, schema_version=1, status='SCOPED_SOURCE_CLI_VALIDATED',
        main_chain=inputs['main_chain'], domain=inputs['domain'], project=inputs['project'],
        parent_project=inputs['parent_project'], retained_catalog_pair=inputs['retained_catalog_pair'],
        provider=inputs['provider'], observation=validation_ref, source=evidence['source'], target=evidence['target'],
        argv=scope['argv'], cpu_ids=evidence['cpu_ids'], working_directory=inputs['working_directory'],
        environment_sha256=scope['environment_sha256'], file_inputs=evidence['file_inputs'], steps=evidence['steps'],
        storage=evidence['memory'], native_executable=inputs['native_executable'], native_launcher=evidence['launcher'],
        native_placement=inputs['native_placement'], launch_tools=inputs['launch_tools'], launch=launch,
        validated_source_stdout=inputs['original_stdout'], validated_final_tensor=inputs['final'],
        publication_modes=modes,
        validation=dict(exact_stdout=True, exact_stderr=True, exact_exit_status=True,
                        bitwise_final_file=True, matched_retained_original=True, exact_publication_mode=exact_mode),
        selection=observed['decision'], delivery_gate=decision, cxx_standard=20, python_runtime_required=False,
        source_cli_performance_measured=True, automatic_deployment=False, deployment_authorized=False,
        execution_default='KEEP_ORIGINAL', whole_program_equivalence_proven=False)


def _capture(inputs, directory):
    scope, evidence = inputs['scope'], inputs['evidence']
    cpus, argv = scope['process_affinity_cpu_ids'], scope['argv']
    publication = Path(argv[-1])
    _need(set(cpus) <= os.sched_getaffinity(0) and publication.resolve() == publication
          and not publication.exists() and not publication.is_symlink(), 'unavailable CPU cohort or output is not fresh')
    directory.mkdir(parents=True, exist_ok=False)
    expected_stdout, expected_output = _raw(inputs['original_stdout']), inputs['final']
    rows, observations, modes = [], [], []
    for pair in range(_WARMUPS + _PAIRS):
        for mode in ((_ORIGINAL, _NATIVE) if pair % 2 == 0 else (_NATIVE, _ORIGINAL)):
            _need(not publication.exists() and not publication.is_symlink(), 'requested output was occupied before launch')
            start = perf_counter_ns()
            observed = subprocess.run(inputs['commands'][mode], cwd=inputs['working_directory'],
                env=inputs['environment'], stdin=subprocess.DEVNULL, capture_output=True, timeout=120)
            elapsed = perf_counter_ns() - start
            _need(observed.returncode == 0 and observed.stderr == b'' and observed.stdout == expected_stdout,
                  'unchanged source CLI exit, stdout or stderr expectation was not met')
            _need(_regular(publication) and _content(publication) ==
                  {key: expected_output[key] for key in ('sha256', 'size_bytes')}, 'final Tensor publication differs')
            modes.append(dict(pair=pair, mode=mode, permission_bits=stat.S_IMODE(publication.lstat().st_mode)))
            saved = directory/f'pair-{pair}-{mode}.f32'
            shutil.move(publication, saved)
            rows.append(dict(pair=pair, mode=mode, wall_ns=elapsed, returncode=observed.returncode,
                             stdout_sha256=sha256(observed.stdout).hexdigest(), stderr_sha256=sha256(observed.stderr).hexdigest()))
            observations.append(dict(pair=pair, mode=mode, stdout=observed.stdout.decode(), stderr='',
                                     output=dict(path=str(saved), **_content(saved))))
    decision = compare_entry_pairs(rows, incumbent=_ORIGINAL, challenger=_NATIVE,
                                   warmup_pairs=_WARMUPS, measurement_pairs=_PAIRS)
    project = inputs['project']
    witness = dict(schema=_OBSERVATION, schema_version=1, project=str(Path(project['uri']).parent),
        manifest_sha256=project['sha256'], steps=evidence['steps'], inputs=evidence['file_inputs'], argv=argv,
        commands=inputs['commands'], cpu_ids=evidence['cpu_ids'], original_cpu_ids=cpus,
        working_directory=inputs['working_directory'], python_provider_environment=inputs['environment'],
        observations=observations, runs=rows, publication_modes=modes, exact_publication_stdout_stderr_exit_verified=True,
        full_process_measured=True, timing_scope=_TIMING_SCOPE, decision=decision, deployment_authorized=False)
    path = directory/'result.json'
    _write(path, witness)
    return artifact_reference(path, _OBSERVATION)


def measure(main_chain_ref, project_ref, out):
    """Measure three exact source-CLI pairs without rebuilding either arm."""
    inputs = _inputs(main_chain_ref, project_ref)
    out = Path(out).absolute()
    _need(out.resolve() == out and not out.exists() and not out.is_symlink(), 'fresh delivery review root required')
    observation = _capture(inputs, out/'observations')
    _need(aten_source_entry._inputs(read_artifact(
              artifact_reference(Path(project_ref['uri']).parent/aten_source_entry.ABI_FILE, 'scar.fixed-grid-file-entry')),
              Path(inputs['scope']['argv'][0])) == inputs['evidence']['file_inputs'],
          'catalog input files changed during comparison')
    report = _record(inputs, observation)
    _write(out/'result.json', report)
    return report


def verify_cpu_fixed_grid_cli_delivery(reference):
    """Cold-replay the candidate, paired timings and exact output receipts."""
    saved = read_artifact(reference)
    path = Path(reference['uri'])
    _need(reference['schema'] == SCHEMA and path.name == 'result.json' and path.resolve(strict=True) == path,
          'exact native source CLI delivery receipt required')
    _need(saved == _record(_inputs(saved['main_chain'], saved['project']), saved['observation']),
          'saved fixed-grid CLI delivery differs from cold replay')
    return saved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--main-chain', type=Path)
    mode.add_argument('--check', type=Path)
    parser.add_argument('--project', type=Path, help='source-compatible C++20 artifact.json')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if args.check is not None:
        if args.project is not None or args.out is not None:
            parser.error('--check reads an existing delivery review')
        reference = artifact_reference(args.check.resolve(strict=True), SCHEMA)
        report = verify_cpu_fixed_grid_cli_delivery(reference)
    else:
        if args.project is None or args.out is None:
            parser.error('--main-chain requires --project and --out')
        report = measure(artifact_reference(args.main_chain.resolve(strict=True), _CHAIN),
                         artifact_reference(args.project.resolve(strict=True), _PROJECT), args.out)
    print(json.dumps(dict(status=report['status'], selection=report['selection']['verdict'],
                          exact_publication_mode=report['validation']['exact_publication_mode'])))


if __name__ == '__main__':
    raise SystemExit(main())
