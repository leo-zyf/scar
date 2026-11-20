"""Choose an ordered sparse width only at freshly measured runtime counts."""

import argparse
import json
from pathlib import Path

from scar.analysis.replay_transaction import replay_transaction
from scar.codegen.native_parameterized_sparse_entry import _verify_file
from scar.ir.execution_recipe import read_artifact
from scar.optimization import native_parameterized_sparse_plan as sparse
from scar.optimization import scalar_sorted_transition_cost as cost


BENCHMARK = 'scar.native-parameterized-sparse-width-benchmark'
SCHEMA = 'scar.native-parameterized-sparse-width-selection'
_WIDTHS = (2, 6)
_DEFAULT_WIDTH = 6


def _require(condition, detail):
    if not condition:
        raise ValueError('runtime sparse width: ' + detail)


def _without(row, *keys):
    return {key: value for key, value in row.items() if key not in keys}


def _verify_bundle(root, reference):
    bundle_ref = sparse._member(root,reference)
    bundle = read_artifact(bundle_ref)
    directory = Path(bundle_ref['uri']).parent
    for member in bundle['members']:
        linked = sparse._member(directory,dict(member,uri=member['path']))
        _verify_file(Path(linked['uri']),member)


def _arm(arm, width):
    artifact = read_artifact(arm['artifact'])
    reference = artifact['parameterized_plan']
    plan = sparse.check(reference['uri'])
    _require(read_artifact(reference) == plan and plan['decision']['native_experiment_authorized'] is True,
        'native arm no longer binds an admitted source and physical plan')
    sparse._native(arm, plan, reference)
    resources = plan['resources']
    _require(resources['width'] == resources['dispatch']['window'] == width,
        'arm must use its declared fixed-width and fixed-window resources')

    root = Path(arm['artifact']['uri']).parent / 'solver'
    solver = read_artifact(dict(artifact['solver'], uri=str(root / 'artifact.json')))
    executable = sparse._member(root,solver['stream']['executable'])
    _verify_file(Path(executable['uri']),executable)
    providers = {}
    for role in ('numeric','sha256_provider'):
        member = sparse._member(root, solver[role])
        read_artifact(member)
        providers[role] = _without(member,'uri')
    numeric_ref = sparse._member(root,solver['numeric'])
    _verify_bundle(Path(numeric_ref['uri']).parent,read_artifact(numeric_ref)['provider'])
    _verify_bundle(root,solver['sha256_provider'])
    prep = {}
    root = root.parent
    for name, member in artifact['reused_upstream'].items():
        _require(name == member['uri'], 'preparation member path differs from the published package')
        prep[name] = _without(member,'uri')
        linked = sparse._member(root,member)
        _verify_file(Path(linked['uri']),linked)
    return plan, artifact['protocol'], providers, prep


def _context(plan):
    from scar.compiler.compile_triangle_profile import _launch_environment

    profile = read_artifact(plan['artifacts']['fixed_profile_plan'])
    context = read_artifact(profile['anchor']['release'])['context']
    python = str(Path(context['interpreter']['executable']).resolve(strict=True))
    return python,context['cwd'],_launch_environment(context)


def _same_workload(w2, w6, receipt):
    narrow, wide = (arm[0] for arm in (w2,w6))
    scope, other = narrow['scope'], wide['scope']
    _require(narrow['artifacts']['source_count'] == wide['artifacts']['source_count']
        and scope == other and scope['source'] == receipt['source']
        and narrow['resources']['case'] == wide['resources']['case']
        and narrow['source_loop'] == wide['source_loop'] and narrow['invariant_geometry'] == wide['invariant_geometry']
        and narrow['numeric_contract'] == wide['numeric_contract'] and w2[2:] == w6[2:],
        'width arms differ in source count, geometry, or numerical and preparation providers')

    left,right = (arm[1] for arm in (w2,w6))
    _require(all(left[key] == right[key] for key in ('input','source','output','scale','allocation'))
        and _without(left['header'],'recipe_digest_hex') == _without(right['header'],'recipe_digest_hex'),
        'width arms changed runtime count, scale or ordered digest semantics')
    a,b = (row['resources']['placement'] for row in (narrow,wide))
    _require(a['owner'] == b['owner'] and a['owner_cpu_id'] == b['owner_cpu_id']
        and a['cpu_numa_node'] == b['cpu_numa_node'] and a['workers'] == b['workers'][:2]
        and a['worker_cpu_ids'] == b['worker_cpu_ids'][:2]
        and a['launch_cpu_ids'] == sorted([a['owner_cpu_id'],*a['worker_cpu_ids']])
        and b['launch_cpu_ids'] == receipt['cpu_ids'],
        'width arms must use a common launch cohort and the same owner and first physical worker lanes')
    launch = _context(wide)
    _require(_context(narrow) == launch
        and launch == (receipt['interpreter']['python_executable'],receipt['cwd'],receipt['environment']),
        'width arms or the original digest used a different interpreter or provider environment')
    return narrow,wide


def _command(command, receipt, argv):
    _require(type(command) is list and len(command) >= 5 and all(type(arg) is str for arg in command)
        and Path(command[0]).name == 'taskset' and command[1] == '-c'
        and sparse._mask(command[2]) == receipt['cpu_ids'] and command[-1:] == argv,
        'each invocation must use the same CPU cohort and canonical runtime argument')
    return tuple(command[:3]),command[3:]


def _digest(row):
    _require(type(row['returncode']) is int and row['returncode'] == 0,
        'width or original process did not complete')
    output, errors = (read_artifact(row[key],json_document=False) for key in ('stdout','stderr'))
    _require(sparse._DIGEST_LINE.fullmatch(output) and not errors,
        'every observed process must emit only a complete digest')
    return output


def _case(case, receipt, plans):
    count,argv = case['count'],case['argv']
    work = {str(width):sparse.work_for_count(plan,count)
            for width,plan in zip(_WIDTHS,plans,strict=True)}
    _require(argv == [str(count)], 'benchmark argv does not bind the exact runtime count')
    original = case['original']
    launch, command = _command(original['command'],receipt,argv)
    _require(len(command) >= 3 and Path(command[0]).is_absolute()
        and str(Path(command[0]).resolve(strict=True)) == receipt['interpreter']['python_executable']
        and command[-2:] == [receipt['source']['uri'],*argv]
        and all(flag in ('-B','-u') for flag in command[1:-2]),
        'digest oracle must directly execute the selected original source')
    digest = _digest(original)
    rows = case['rows']
    _require(type(rows) is list and len(rows) >= 8 and len(rows) % 2 == 0,
        'expected one warmup pair and at least three clean paired samples per count')

    order = tuple(row['width'] for row in rows[:2])
    _require(set(order) == set(_WIDTHS), 'each pair must include both tested widths')
    samples,orders = {width:[] for width in _WIDTHS},[]
    for index,row in enumerate(rows):
        round_, slot = divmod(index,2)
        current = order if round_%2 == 0 else order[::-1]
        if slot == 0:
            orders.append(list(current))
        width = row['width']
        permitted = ('OBSERVED_WARMUP','WARMUP') if round_ == 0 else ('CLEAN_MEASURED',)
        _require(type(width) is int and width == current[slot] and type(row['round']) is int and row['round'] == round_
            and row['kind'] in permitted and type(row['wall_ns']) is int and row['wall_ns'] > 0,
            'native measurements must preserve paired alternating launch order')
        prefix, command = _command(row['command'],receipt,argv)
        arm = receipt['arms'][str(width)]
        _require(launch == prefix and command == [arm['executable']['uri'],*argv]
            and _digest(row) == digest, 'launch or exact original digest differs across widths')
        if round_:
            samples[width].append(row['wall_ns'])

    measurements, qualification = cost._envelopes({'original':samples[6],'w2':samples[2]}, minimum_gain_bps=0)
    narrow,wide = (measurements[key] for key in ('w2','original'))
    proof = qualification['w2']
    separation = wide['min_ns'] - narrow['max_ns']
    wider_wins = wide['max_ns'] < narrow['min_ns']
    width = 2 if proof['verified_observed_gain'] else _DEFAULT_WIDTH
    return dict(count=count,argv=argv,original=dict(stdout=original['stdout'],stderr=original['stderr']),
        original_digest=digest.decode('ascii').rstrip('\n'), work_by_width=work,orders=orders,warmup_pairs=1,
        clean_pairs=len(samples[6]), measurements={'2':narrow,'6':wide}, selected_width=width,
        separation_w2_gain_ns=separation,
        paired_w6_minus_w2_ns=proof['paired_original_minus_candidate_ns'],
        decision='W2_FASTER_IN_OBSERVED_PAIRS' if width == 2 else (
            'W6_FASTER_IN_OBSERVED_PAIRS' if wider_wins else 'W6_DEFAULT_WITH_OVERLAPPING_ENVELOPES'))


@replay_transaction()
def select(benchmark_path):
    reference = sparse._ref(benchmark_path,BENCHMARK)
    receipt = read_artifact(reference)
    _require(type(receipt['schema_version']) is int and receipt['schema_version'] == 1
        and set(receipt['arms']) == {'2','6'}, 'expected one fixed W2 and one fixed W6 native arm')
    read_artifact(receipt['launcher'],json_document=False)
    interpreter = receipt['interpreter']
    _require(type(interpreter['python_version']) is str and bool(interpreter['python_version'].strip()),
        'original digest interpreter must identify its measured version')
    checked = [_arm(receipt['arms'][str(width)],width) for width in _WIDTHS]
    plans = _same_workload(*checked,receipt)
    cases = [_case(row,receipt,plans) for row in receipt['cases']]
    widths = {str(row['count']):row['selected_width'] for row in cases}
    _require(cases and len(widths) == len(cases), 'expected at least one distinct measured runtime count')
    scope = plans[1]['scope']
    return dict(schema=SCHEMA,schema_version=1,benchmark=reference,source=receipt['source'],
        arms=receipt['arms'], cpu_ids=receipt['cpu_ids'], minimum_count=scope['minimum_count'],
        maximum_count=scope['maximum_count'], exact_count_to_width=widths, default_width=_DEFAULT_WIDTH,
        cases=cases,selection_scope='EXACT_MEASURED_RUNTIME_COUNTS_ONLY',
        default_scope='QUALIFIED_W6_WITH_LIVE_RESOURCE_GUARDS',deployment_authorized=False)


def choose_width(selection, count):
    _require(type(count) is int and selection['minimum_count'] <= count <= selection['maximum_count'],
        'runtime count outside admitted selector range')
    return selection['exact_count_to_width'].get(str(count),selection['default_width'])


def check(path):
    stored = read_artifact(sparse._ref(path,SCHEMA))
    replay = select(stored['benchmark']['uri'])
    _require(sparse._canonical(stored) == sparse._canonical(replay), 'width selection differs from cold replay')
    return replay


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('benchmark','check','out'):
        parser.add_argument('--' + name,type=Path)
    args = parser.parse_args(argv)
    if args.check and not any((args.benchmark,args.out)):
        result = check(args.check)
    elif args.benchmark and args.out and args.check is None:
        result = select(args.benchmark)
        with args.out.open('x') as stream:
            stream.write(json.dumps(result,indent=2,sort_keys=True,allow_nan=False) + '\n')
    else:
        parser.error('provide --check, or --benchmark and --out')
    print(sparse._canonical({key:result[key] for key in ('exact_count_to_width','default_width')}))


if __name__ == '__main__':
    main()
