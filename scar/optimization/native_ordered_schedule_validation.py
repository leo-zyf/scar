"""Pair clean solver launches on the exact configurations admitted by a credit plan."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median, stdev

from scar.codegen.native_ordered_timing import _run
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization import native_ordered_resource_schedule as schedule


SCHEMA = 'scar.native-ordered-clean-schedule-validation'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _arms(path):
    proposal_ref = artifact_reference(Path(path).resolve(strict=True), schedule.SCHEMA)
    proposal = schedule.check(proposal_ref['uri'])
    _require(proposal['schema_version'] == 2 and proposal == read_artifact(proposal_ref),
             'clean validation requires a checked exact-configuration credit plan')
    selected = proposal['selected_configuration']
    opponents = sorted(key for key, option in proposal['options'].items()
                       if option.get('dominance', {}).get('by') == selected)
    _require(selected and len(opponents) == 1, 'paired validation requires one selected and one dominated configuration')
    arms = {}
    for key in (selected, opponents[0]):
        option = proposal['options'][key]
        quote = option['cost_quote']
        original = read_artifact(quote['native_observation'])
        production = original['production']
        executable = production['executable']
        oracle = production['stdout']
        _require(oracle['sha256'] == quote['exact_stdout_sha256']
                 and original['input'] == quote['exact_input'], 'production solver deviated from its qualified input')
        read_artifact(executable, json_document=False)
        arms[key] = dict(configuration=option['configuration'], executable=executable,
                         input=original['input'], output_sha256=oracle['sha256'])
    left, right = arms.values()
    _require(left['input'] == right['input'] and left['output_sha256'] == right['output_sha256'],
             'paired worker populations must solve the same inputs')
    read_artifact(left['input'], json_document=False)
    return proposal_ref, selected, arms


def _summarize(arms, selected, matches):
    keys = list(arms)
    pairs = len(matches) // 2 - 1
    _require(2 <= pairs <= 16 and len(matches) == (pairs + 1) * 2, 'invalid number of clean pairs')
    measures = {key:[] for key in keys}
    for index, row in enumerate(matches):
        round_, offset = divmod(index, 2)
        key = keys[offset if round_ % 2 == 0 else 1 - offset]
        _require(row['round'] == round_ and row['arm'] == key and row['stdout_sha256'] == arms[key]['output_sha256']
                 and type(row['wall_ns']) is int and row['wall_ns'] > 0, 'clean pair changed arm, input result or order')
        if round_:
            measures[key].append(row['wall_ns'])
    summary = {key:dict(samples_ns=values, median_ns=median(values), min_ns=min(values),
                       max_ns=max(values), sample_stdev_ns=stdev(values)) for key,values in measures.items()}
    alternate = next(key for key in keys if key != selected)
    left, right = summary[selected], summary[alternate]
    supported = left['max_ns'] < right['min_ns']
    return dict(warmup_pairs=1, measured_pairs=pairs, measurements=summary,
                median_speedup=right['median_ns']/left['median_ns'],
                decision='SUPPORT_SELECTED_SOLVER_CONFIGURATION' if supported else 'NO_CONCLUSIVE_SELECTED_ADVANTAGE',
                scope='NATIVE_SOLVER_WITH_MATCHED_PREBUILT_CSR_INPUT',
                original_python_entry_timed=False, deployment_authorized=False)


def collect(path, pairs=3):
    if type(pairs) is not int or not 2 <= pairs <= 16:
        raise ValueError('pairs must be an integer from 2 to 16')
    proposal, selected, arms = _arms(path)
    matches = []
    for round_ in range(pairs + 1):
        for key in (tuple(arms) if round_ % 2 == 0 else tuple(reversed(arms))):
            arm = arms[key]
            stdout, elapsed = _run(Path(arm['executable']['uri']), Path(arm['input']['uri']),
                                   arm['configuration']['launch_cpu_ids'])
            output_sha = hashlib.sha256(stdout).hexdigest()
            _require(output_sha == arm['output_sha256'], 'clean solver output does not match its source oracle')
            matches.append(dict(round=round_, arm=key, wall_ns=elapsed, stdout_sha256=output_sha))
    return dict(schema=SCHEMA, schema_version=1, proposal=proposal, selected=selected, arms=arms, raw=matches,
                **_summarize(arms, selected, matches))


def check(path):
    stored = read_artifact(artifact_reference(Path(path).resolve(strict=True), SCHEMA))
    proposal, selected, arms = _arms(stored['proposal']['uri'])
    expected = dict(schema=SCHEMA, schema_version=1, proposal=proposal, selected=selected, arms=arms, raw=stored['raw'],
                    **_summarize(arms, selected, stored['raw']))
    _require(expected == stored, 'clean solver comparison differs from replay')
    return stored


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--proposal', type=Path)
    inputs.add_argument('--check', type=Path)
    parser.add_argument('--pairs', type=int, default=3)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if args.check and args.out is None:
        result = check(args.check)
    elif args.proposal and args.out:
        result = collect(args.proposal, args.pairs)
        with args.out.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
    else:
        parser.error('provide --proposal and --out, or --check')
    print(json.dumps({key:result[key] for key in ('selected','measured_pairs','median_speedup','decision')}))


if __name__ == '__main__':
    main()
