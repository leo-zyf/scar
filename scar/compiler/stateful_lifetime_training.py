"""Train a complete method cost on one declared real source-entry trajectory."""

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
from hashlib import sha256
import importlib
import inspect
import json
from pathlib import Path
from statistics import median
import sys
from types import FunctionType
import uuid

from scar.compiler import stateful_holdout as sampler
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.runtime.compiled_deployment import thread_environment

DESIGN = 'scar.stateful-lifetime-cost-design'
TRAIN = 'scar.stateful-lifetime-training'
MODEL = 'scar.stateful-lifetime-cost-model'
HOLDOUT = 'scar.stateful-lifetime-holdout'
TICKET = 'scar.stateful-lifetime-holdout-ticket'
RESULT = 'scar.stateful-lifetime-qualification'
FIRST = 'FIRST_CALL_AFTER_ORIGINAL_ENTRY_START'
LATER = 'CONTINUATION_OF_SAME_ORIGINAL_ENTRY'


def _need(value, message):
    if not value:
        raise ValueError(message)


def _stamp():
    return datetime.now(timezone.utc).isoformat()


def _checked(value, schema, digest=None):
    ref = value if type(value) is dict else artifact_reference(Path(value).resolve(), schema)
    doc = read_artifact(ref)
    _need(ref['schema'] == schema and doc['schema_version'] == 1
        and (digest is None or ref['sha256'] == digest), 'lifetime receipt version or SHA differs')
    return ref, doc


def _input_context(state):
    return {key: ({field: row[field] for field in ('kind', 'shape', 'dtype', 'strides')}
        if row['kind'] == 'array' else dict(kind='scalar', representation=row['representation'],
            python_type=row.get('python_type', row.get('type')))) for key, row in state.items()}


def _derive(template_ref, *, directory, repeats, epochs):
    _need(type(repeats) is int and 1 <= repeats <= 7, 'one to seven bounded measurement episodes required')
    _need(set(epochs) == {'training', 'holdout'} and len(set(epochs.values())) == 2
        and all(type(epoch) is str and len(epoch) == 32 and all(c in '0123456789abcdef' for c in epoch)
            for epoch in epochs.values()), 'independent original capture epochs required')
    template = sampler.verify_stateful_holdout_design(template_ref['uri'], expected_sha256=template_ref['sha256'])
    request = template['request']
    prior = read_artifact(request['prior_correctness'])
    original, native = prior['original'], prior['native_trial']
    count = original['boundary_calls']
    _need(prior['argv_template'] == request['argv'] and type(count) is int and 2 <= count <= 64
        and native['boundary_calls'] == count and len(original['trajectory']) == count,
        'correctness admission must establish the same bounded original entry trajectory')
    _need(template['canonical_model_binding']['entry_protocol']['configuration']
        == dict(trial=True, exclusive_inputs=True, selection=None, selection_semantic_sha256=None),
        'one existing explicit native trial protocol required')
    sampler._environment(template)
    policy = {key:template['policy_contract'][key]
        for key in ('max_relative_prediction_error', 'minimum_observed_gain')}
    return dict(schema=DESIGN, schema_version=1, template=template_ref,
        directory=str(directory), repeats=repeats, warmups=2, epochs=epochs, policy=policy,
        method_calls_per_episode=count, phases=[FIRST] + [LATER] * (count - 1),
        input_contexts=[_input_context(step['before']) for step in original['trajectory']],
        context=dict(entry=request['entry'], argv=request['argv'], output_token=request['output_token'],
            target_cost_context=template['context'],
            native_artifact=template['native_artifact'], timer=template['timer'],
            protocol=template['canonical_model_binding']['entry_protocol']['id'],
            caller_cpu_ids=request['caller_cpu_ids'], native_thread_environment=thread_environment()),
        trainer=artifact_reference(Path(__file__), 'scar.implementation-source'),
        time_prediction='PHASE_MEDIANS_OF_EPISODE_MEANS_WEIGHTED_BY_FROZEN_CALL_COUNTS',
        scope='ONE_DECLARED_REAL_ENTRY_ARGUMENT_AND_METHOD_TRAJECTORY',
        training_reads_prior_holdout_timings=False, default_backend='NO_OP'), template


def prepare(template, out, *, repeats=7):
    template_ref = sampler._ref(template, sampler.DESIGN)
    directory = Path(out).resolve()
    design, _ = _derive(template_ref, directory=directory, repeats=repeats,
        epochs={phase:uuid.uuid4().hex for phase in ('training', 'holdout')})
    directory.mkdir(parents=True)
    return sampler._write(directory/'design.json', design)


def _design(value, digest=None):
    ref, saved = _checked(value, DESIGN, digest)
    expected, template = _derive(saved['template'], directory=saved['directory'],
        repeats=saved['repeats'], epochs=saved['epochs'])
    _need(saved == expected and Path(ref['uri']) == Path(saved['directory'])/'design.json',
        'frozen lifetime scenario differs from source, placement, policy or producer')
    return ref, saved, template


@contextmanager
def _runtime(campaign, template):
    from scar.codegen.stateful import load
    from scar.compiler.stateful_runtime import install_stateful_regions
    from scar.runtime.stateful_guard import _function

    candidate = read_artifact(template['candidate'])
    artifact = load(candidate['native_directory'])
    region = artifact.region
    previous_path, previous_bytecode = list(sys.path), sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(region.source.path).parent))
    try:
        module = importlib.import_module(Path(region.source.path).stem)
        _need(Path(vars(module)['__file__']).resolve() == Path(region.source.path).resolve(),
            'ordinary import selected another source')
        owner, name = sampler._owner(module, region)
        saved = vars(owner)[name]
        original = saved.__func__ if type(saved) is staticmethod else saved
        _function(region.source, original)
        signature = inspect.signature(original)
        _need(type(original) is FunctionType and tuple(signature.parameters) == region.parameter_names,
            'original entry callable signature differs')
        _need(thread_environment() == campaign['context']['native_thread_environment'],
            'native thread environment differs')
        with install_stateful_regions(module, [artifact], trial=True, exclusive_inputs=True) as guards:
            yield module, region, signature, original, guards[0]
        _need(vars(owner)[name] is saved, 'original descriptor was not restored')
    finally:
        sys.path[:], sys.dont_write_bytecode = previous_path, previous_bytecode


def _order(campaign, *, paired):
    for phase, count in (('warmup', campaign['warmups']), ('measurement', campaign['repeats'])):
        for index in range(count):
            yield phase, index, (['A', 'B'] if index % 2 == 0 else ['B', 'A']) if paired else ['B']


def _compact_episode(episode):
    return dict(steps=episode['steps'], final=sampler._summary(episode['final']),
        stdout=episode['stdout'], stderr=episode['stderr'],
        outputs={name: sha256(data).hexdigest() for name, data in episode['files'].items()},
        execution=episode['execution'])


def _planned(campaign, name, phase, index, label):
    directory = Path(campaign['directory'])/name/f'{phase}-{index}-{label}'
    token = campaign['context']['output_token']
    output = str(directory/'program-output') if token is not None else None
    return dict(directory=str(directory), program_output_directory=output,
        argv=[output if token is not None and value == token else value for value in campaign['context']['argv']])


def _validate_steps(campaign, label, epoch, row):
    steps = row['steps']
    _need(len(steps) == campaign['method_calls_per_episode'], 'declared method trajectory length differs')
    for position, step in enumerate(steps):
        _need(step['capture_epoch'] == epoch and step['ordinal'] == position
            and type(step['elapsed_ns']) is int and step['elapsed_ns'] > 0
            and step['native_calls'] == int(label == 'B') and step['fallback_calls'] == 0
            and _input_context(step['inputs']) == campaign['input_contexts'][position],
            'method position, admitted input trajectory or successful path differs')
    _need(all(value['tensor_contents_read'] is False for step in steps
        for value in step['inputs'].values() if value['kind'] == 'array'),
        'method timing captures must not read per-call array contents')


def _validate_pair(pair, campaign, phase, index, order, name):
    _need((pair['phase'], pair['index'], pair['order']) == (phase, index, order)
        and set(pair['runs']) == set(order), 'frozen episode order differs')
    for label in order:
        row = pair['runs'][label]
        _validate_steps(campaign, label, campaign['epochs'][name], row)
        execution = row['execution']
        _need(execution['exit_code'] == 0 and execution['argv'] == _planned(campaign, name, phase, index, label)['argv']
            and execution['entry_path'] == campaign['context']['entry']['uri'], 'planned original entry did not complete')
    if len(order) == 2:
        a, b = pair['runs']['A'], pair['runs']['B']
        _need(all(a[key] == b[key] for key in ('final', 'outputs', 'stdout', 'stderr'))
            and [step['inputs'] for step in a['steps']] == [step['inputs'] for step in b['steps']],
            'paired source output or input context differs')


def _collect(design_ref, campaign, template, name):
    paired = name == 'holdout'
    epoch = campaign['epochs'][name]
    capture_dir = Path(campaign['directory'])/name
    capture_dir.mkdir()
    context = deepcopy(template)
    context['request']['capture_epoch'] = epoch
    request = context['request']
    episodes = []
    try:
        with _runtime(campaign, template) as (module, region, signature, original, native):
            for phase, index, order in _order(campaign, paired=paired):
                pair = dict(phase=phase, index=index, order=order, runs={})
                episodes.append(pair)
                raw = {}
                for label in order:
                    raw[label] = sampler._episode(request['entry']['uri'], module, region, signature,
                        original if label == 'A' else native, native, context,
                        _planned(campaign, name, phase, index, label))
                    pair['runs'][label] = _compact_episode(raw[label])
                if paired:
                    _need(raw['A']['final'] == raw['B']['final'] and raw['A']['files'] == raw['B']['files'],
                        'paired complete final state or configured file bytes differ')
                _validate_pair(pair, campaign, phase, index, order, name)
        _need(thread_environment() == campaign['context']['native_thread_environment'],
            'native thread environment changed during capture')
        read_artifact(design_ref)
        for reference in (request['entry'], *template['source_refs'].values(), *template['native_members'].values()):
            read_artifact(reference, json_document=False)
    except Exception as error:
        sampler._write(capture_dir/'failed.json', dict(schema='scar.stateful-lifetime-failed-capture',
            schema_version=1, design=design_ref, epoch=epoch, error_type=type(error).__name__,
            message=str(error), episodes=episodes))
        raise
    return episodes


def _capture(design_ref, campaign, template, name, *, ticket=None):
    started_at = _stamp()
    episodes = _collect(design_ref, campaign, template, name)
    return sampler._write(Path(campaign['directory'])/name/'observations.json',
        dict(schema=TRAIN if name == 'training' else HOLDOUT, schema_version=1, design=design_ref,
            purpose='TRAIN_ONLY' if name == 'training' else 'VALIDATE_ONLY',
            capture_epoch=campaign['epochs'][name], started_at=started_at, finished_at=_stamp(),
            ticket=ticket, episodes=episodes, timer=campaign['context']['timer'],
            per_step_array_payloads_recorded=False))


def _observations(reference, schema, design_ref, campaign, name, *, ticket=None):
    _, capture = _checked(reference, schema)
    epoch = campaign['epochs'][name]
    paired = name == 'holdout'
    _need(capture['design'] == design_ref and capture['capture_epoch'] == epoch
        and capture['purpose'] == ('VALIDATE_ONLY' if paired else 'TRAIN_ONLY')
        and capture['ticket'] == ticket and capture['timer'] == campaign['context']['timer']
        and capture['per_step_array_payloads_recorded'] is False
        and capture['started_at'] < capture['finished_at'], 'capture provenance differs')
    for pair, (phase, index, order) in zip(capture['episodes'], _order(campaign, paired=paired), strict=True):
        _validate_pair(pair, campaign, phase, index, order, name)
    return capture


def _phase_medians(episodes, label):
    first, subsequent = [], []
    for pair in episodes:
        if pair['phase'] == 'measurement':
            first_ns, *later_ns = [step['elapsed_ns'] for step in pair['runs'][label]['steps']]
            first.append(Fraction(first_ns))
            subsequent.append(Fraction(sum(later_ns), len(later_ns)))
    return {FIRST: median(first), LATER: median(subsequent)}


def _fit(training_ref, design_ref, campaign, *, frozen_at):
    training = _observations(training_ref, TRAIN, design_ref, campaign, 'training')
    _need(training['finished_at'] < frozen_at, 'model freeze must follow training')
    phase_costs = _phase_medians(training['episodes'], 'B')
    count = campaign['method_calls_per_episode']
    duration_ns = round((phase_costs[FIRST] + (count - 1) * phase_costs[LATER]) / count)
    return dict(schema=MODEL, schema_version=1, design=design_ref, training=training_ref,
        duration_ns=duration_ns, phase_costs_ns={key:dict(numerator=value.numerator,
            denominator=value.denominator) for key, value in phase_costs.items()},
        frozen_at=frozen_at, context=campaign['context'], scope=campaign['scope'],
        fit_reads_holdout=False, deployment_authorized=False)


def train(design, *, digest):
    design_ref, campaign, template = _design(design, digest)
    observation = _capture(design_ref, campaign, template, 'training')
    model = _fit(observation, design_ref, campaign, frozen_at=_stamp())
    return sampler._write(Path(campaign['directory'])/'model.json', model)


def _ticket(model_ref, model, campaign, *, frozen_at):
    _need(model['frozen_at'] < frozen_at, 'holdout precommit must follow model freeze')
    return dict(schema=TICKET, schema_version=1, model=model_ref, design=model['design'],
        holdout_epoch=campaign['epochs']['holdout'], frozen_at=frozen_at, policy=campaign['policy'])


def _evaluate(model_ref, model, campaign, ticket_ref, holdout_ref):
    _, ticket = _checked(ticket_ref, TICKET)
    observed = _observations(holdout_ref, HOLDOUT, model['design'], campaign, 'holdout', ticket=ticket_ref)
    _need(ticket == _ticket(model_ref, model, campaign, frozen_at=ticket['frozen_at'])
        and ticket['frozen_at'] < observed['started_at'], 'holdout ran before frozen model and policy')
    pairs = [pair for pair in observed['episodes'] if pair['phase'] == 'measurement']
    episode_means = {label:sampler._distribution([
        sum(step['elapsed_ns'] for step in pair['runs'][label]['steps']) /
        campaign['method_calls_per_episode'] for pair in pairs]) for label in ('A', 'B')}
    original, native = (episode_means[label]['median_ns'] for label in ('A', 'B'))
    verdict = sampler._prediction_evaluation(model['duration_ns'], original, native, campaign['policy'])
    native_phases = _phase_medians(pairs, 'B')
    phase_errors = {key:float(abs(Fraction(**model['phase_costs_ns'][key]) - value) / value)
        for key, value in native_phases.items()}
    qualified = (campaign['repeats'] == 7 and verdict['prediction_error_threshold_passed']
        and verdict['observed_gain_threshold_passed'])
    return dict(schema=RESULT, schema_version=1, model=model_ref, holdout_ticket=ticket_ref,
        holdout=holdout_ref, status=('SCENARIO_POINT_PREDICTION_QUALIFIED' if qualified
            else 'SMOKE_ONLY' if campaign['repeats'] < 7 else 'SCENARIO_PREDICTION_REJECTED'),
        evaluation=verdict, observed_episode_mean_distributions=episode_means,
        observed_method_original_over_native_ratio=original/native,
        phase_costs_ns=model['phase_costs_ns'], exact_scope=campaign['scope'],
        observed_phase_medians_ns={key:float(value) for key, value in native_phases.items()},
        descriptive_phase_relative_errors=phase_errors,
        timing_qualification_scope='EPISODE_MEAN_POINT_PREDICTION_ONLY',
        physical_timing_or_resource_guarantee=False, qualified_for_automatic_selection=False,
        deployment_authorized=False, whole_program_speedup_measured=False, default_backend='NO_OP')


def _checked_model(reference, digest=None):
    model_ref, model = _checked(reference, MODEL, digest)
    design_ref, campaign, template = _design(model['design'])
    _need(model == _fit(model['training'], design_ref, campaign, frozen_at=model['frozen_at']),
        'frozen phase model differs from the training-only clocks')
    return model_ref, model, campaign, template


def holdout(model, *, digest):
    model_ref, saved, campaign, template = _checked_model(model, digest)
    out = Path(campaign['directory'])
    ticket = sampler._write(out/'holdout-ticket.json',
        _ticket(model_ref, saved, campaign, frozen_at=_stamp()))
    measured = _capture(saved['design'], campaign, template, 'holdout', ticket=ticket)
    return sampler._write(out/'result.json', _evaluate(model_ref, saved, campaign, ticket, measured))


def check(result, *, digest=None):
    _, result = _checked(result, RESULT, digest)
    model_ref, model, campaign, _ = _checked_model(result['model'])
    _need(result == _evaluate(model_ref, model, campaign, result['holdout_ticket'], result['holdout']),
        'lifetime qualification differs from independent clocks and frozen policy')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    design = commands.add_parser('prepare')
    design.add_argument('--template', required=True)
    design.add_argument('--out', required=True)
    design.add_argument('--repeats', type=int, default=7)
    for command, path in (('train', 'design'), ('holdout', 'model'), ('check', 'result')):
        child = commands.add_parser(command)
        child.add_argument('--'+path, required=True)
        child.add_argument('--sha256', required=True)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        result = prepare(args.template, args.out, repeats=args.repeats)
    elif args.command == 'train':
        result = train(args.design, digest=args.sha256)
    elif args.command == 'holdout':
        result = holdout(args.model, digest=args.sha256)
    else:
        result = check(args.result, digest=args.sha256)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
