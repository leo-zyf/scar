"""Join scope-bound import observations to original F4 import requests."""

from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

from scar.analysis.entry_obligations import SCHEMA as ENTRY
from scar.analysis.source_decoded_bodies import reuse_decoded_source_bodies
from scar.analysis.source_declared_catalog import read_source_catalog
from scar.analysis.target_graph import TargetHardwareGraph
from scar.compiler.cpu_entry_replay import SCHEMA as PAIRED
from scar.ir.execution_recipe import read_artifact
from scar.ir.record_codec import encode
from scar.optimization.cpu_entry_decision_ledger import verify_cpu_entry_decision_ledger

SCHEMA = 'scar.f4-cpu-python-import-graph'
_OBSERVATION = 'scar.cpu-original-import-observation'


def _need(ok, detail):
    if not ok:
        raise ValueError('CPU Python import graph: ' + detail)


def _reference(row, schema):
    return dict(uri=row['path'], sha256=row['sha256'], size_bytes=row['bytes'], schema=schema)


def _environment(values):
    return sha256(json.dumps(sorted(values.items()), separators=(',', ':')).encode()).hexdigest()


def _name(value):
    return type(value) is str and bool(value) and all(part.isidentifier() for part in value.split('.'))


def _lookup(rows, index, context):
    _need(type(index) is int and 0 <= index < len(rows), 'invalid original ' + context + ' index')
    return rows[index]


def _verify_scope(original, entry_obligations, observation, decision_ledger):
    program_ref = original['program']
    program = read_artifact(program_ref)
    paired = read_artifact(observation['paired'])
    _need(observation['paired']['schema'] == PAIRED and paired['schema_version'] == 1,
          'observation does not bind an original qualified CPU invocation')
    spec = paired['invocation']
    arm = spec['arms']['original']
    composition = read_artifact(paired['compilation'])
    source = observation['entry']
    _need(source == spec['entry'] == composition['entry'] and program['entry'] == source['uri']
          and observation['interpreter'] == arm['executable'] and observation['cwd'] == spec['cwd']
          and observation['cpu_ids'] == spec['cpu_ids'] and observation['original_argv'] == arm['argv']
          and observation['environment_sha256'] == _environment(arm['environment']),
          'diagnostic differs from original source or paired launch')
    read_artifact(source, json_document=False)

    catalog_receipt = program['f4']['catalog']
    path = Path(catalog_receipt['path'])
    raw = path.read_bytes()
    _need(len(raw) == catalog_receipt['bytes'] and sha256(raw).hexdigest() == catalog_receipt['sha256'],
          'original F4 source catalog bytes differ')
    catalog = read_source_catalog(path.parent)
    _need(json.loads(raw) == catalog.report, 'original F4 source catalog changed')
    unit = program['f4']['entry_unit']
    model = catalog.fact_store.model_units[unit]
    _need((model['path'], model['raw_sha256'], model['source_bytes'])
          == (source['uri'], source['sha256'], source['size_bytes']),
          'diagnostic source differs from the admitted F4 entry bytes')

    target = _reference(program['target'], 'scar.target-hardware-graph')
    graph = TargetHardwareGraph(Path(target['uri']).parent)
    _need(encode(graph._snapshot) == target, 'target differs from the original F4 build')
    constraints = graph.constraints()
    allowed = constraints[0].affinity_cpu_ids if len(constraints) == 1 else None
    cpus = observation['cpu_ids']
    _need(bool(cpus) and (allowed is None or set(cpus) <= set(allowed)),
          'diagnostic CPU cohort differs from original target constraints')

    scope_id = None
    if decision_ledger is not None:
        from scar.compiler.cpu_python_import_diagnostic import check_original_import_scope

        ledger = verify_cpu_entry_decision_ledger(decision_ledger)
        scope = read_artifact(ledger['domain'])['scope']
        _need(scope['program_sha256'] == program_ref['sha256']
              and scope['entry_obligations_sha256'] == entry_obligations['sha256']
              and scope['target'] == target and ledger['target']['graph'] == target,
              'decision ledger belongs to another source, invocation or target')
        check_original_import_scope(scope, observation['paired'])
        scope_id = ledger['scope_sha256']
    return source, target, scope_id


class _ImportGraph:
    def __init__(self):
        self.ids = {}
        self.modules = []
        self.static = []

    def module(self, name):
        if name not in self.ids:
            self.ids[name] = len(self.modules)
            self.modules.append(dict(name=name, identity='MODULE_SPELLING_ONLY', static_sites=[], observed_records=0))
        return self.ids[name]

    def sites(self, original):
        for index, row in enumerate(original['imports']):
            operation = row['operation']
            op = _lookup(original['operations'], operation, 'import operation')
            _need(op['effect_rule']['instruction'] == 'ImportRequest', 'source effect is not an ImportRequest')
            name, level = row['requested_name'], row['relative_level']
            from_names = row['from_names']
            _need(type(name) is str and type(level) is int and level >= 0 and
                  (from_names is None or type(from_names) is list and all(type(item) is str for item in from_names)),
                  'original import request selector differs from the typed grammar')
            choices = defaultdict(set)
            if level == 0 and _name(name):
                choices[name].add('REQUESTED_MODULE')
                pieces = name.split('.')
                for end in range(1, len(pieces)):
                    choices['.'.join(pieces[:end])].add('POSSIBLE_PACKAGE_INITIALIZER')
                for member in from_names or ():
                    if _name(member):
                        choices[name+'.'+member].add('POSSIBLE_FROM_MEMBER_MODULE')
            for alternative in row['alternatives']:
                candidate = alternative['module']
                _need(_name(candidate), 'original module alternative has invalid spelling')
                choices[candidate].add('ORIGINAL_LISTED_MODULE_ALTERNATIVE')
            selected = []
            for spelling, roles in sorted(choices.items()):
                module = self.module(spelling)
                selected.append(module)
                self.modules[module]['static_sites'].append(dict(site=index, roles=sorted(roles)))
            self.static.append(dict(original=f'#/imports/{index}', operation=operation, requested_name=name,
                relative_level=level, from_names=from_names, candidate_modules=selected,
                source_alternative_count=len(row['alternatives']), obligation=row['obligation'],
                target_status='RUNTIME_TARGET_AND_INITIALIZATION_EFFECTS_OPEN'))

    def clock_events(self, records):
        events = []
        for index, row in enumerate(records):
            _need(row['index'] == index and type(row['module']) is str and bool(row['module'])
                  and type(row['cached']) is bool and type(row['indent_spaces']) is int and row['indent_spaces'] >= 0
                  and (row['depth'] is None or type(row['depth']) is int and row['depth'] >= 0),
                  'invalid retained import diagnostic row')
            own, inclusive = row['self_us'], row['cumulative_us']
            _need((own is None and inclusive is None) if row['cached'] else
                  type(own) is int and type(inclusive) is int,
                  'invalid retained import clock fields')
            module = self.module(row['module'])
            self.modules[module]['observed_records'] += 1
            events.append(dict(module=module, record_index=index, self_us=own, cumulative_us=inclusive,
                depth=row['depth'], indent_spaces=row['indent_spaces'],
                exec_index=row['exec_index'], cached=row['cached'],
                clock_status='CACHED_NO_TIMING' if row['cached'] else
                'REPORTED_SELF_AND_CUMULATIVE_US' if 0 <= own <= inclusive else 'CLOCK_ACCOUNTING_OPEN'))
        return events

    def close(self):
        for site in self.static:
            ids = site['candidate_modules']
            if site['relative_level']:
                status = 'RELATIVE_PACKAGE_CONTEXT_OPEN'
            elif any(self.modules[index]['observed_records'] for index in ids):
                status = 'COMPATIBLE_MODULE_SPELLING_OBSERVED'
            else:
                status = 'NO_COMPATIBLE_MODULE_SPELLING_OBSERVED'
            site['status'] = status


def _runtime(graph, observation):
    runs = []
    assigned = True
    for run_index, run in enumerate(observation['runs']):
        tasks, processes = [], defaultdict(list)
        runtime_tids = set()
        for task_index, row in enumerate(run['tasks']):
            tid, tgid = row['tid'], row['tgid']
            _need(type(tid) is int and tid > 0 and tid not in runtime_tids
                  and (tgid is None or type(tgid) is int and tgid > 0),
                  'invalid or duplicate observed task identity')
            runtime_tids.add(tid)
            if tgid is not None:
                processes[tgid].append(tid)
            else:
                assigned = False
            original = f'#/runs/{run_index}/tasks/{task_index}'
            tasks.append(dict(task_id=tid, process_id=tgid, trace=row['trace'], original=original,
                parent_task=None if row['birth'] is None else row['birth']['parent_tid'],
                birth_evidence=None if row['birth'] is None else original+'/birth',
                exec_evidence=None if not row['execs'] else original+'/execs',
                events=graph.clock_events(row['records'])))
        _need(run['ordinal'] == run_index and run['launcher_tid'] in runtime_tids
              and run['interpreter_tid'] in runtime_tids, 'exact original traced launcher/interpreter task missing')
        runs.append(dict(ordinal=run['ordinal'], original=f'#/runs/{run_index}', stderr=run['stderr'], tasks=tasks,
                         launcher_task=run['launcher_tid'], original_interpreter_task=run['interpreter_tid'],
                         processes=[dict(process_id=tgid, task_ids=sorted(tids))
                                    for tgid, tids in sorted(processes.items())]))
    return runs, assigned


@reuse_decoded_source_bodies()
def derive_cpu_python_import_graph(entry_obligations, diagnostic_ref, *, decision_ledger=None):
    """Pin and index original F4 sites and distinct retained runtime records."""
    from scar.compiler import cpu_original_importtime

    original = read_artifact(entry_obligations)
    _need(entry_obligations['schema'] == ENTRY and original['schema_version'] == 1
          and original['scope'] == 'NORMAL_SUCCESS_SOURCE_MAY_GRAPH' and original['effects_closed'] is False,
          'expected published original entry obligations')
    observation = read_artifact(diagnostic_ref)
    _need(diagnostic_ref['schema'] == _OBSERVATION and observation['schema_version'] == 1
          and observation['instrumented'] is True and observation['native_target_executed'] is False
          and cpu_original_importtime.check(diagnostic_ref['uri']) == observation,
          'diagnostic differs from its original captured evidence')
    source, target, scope_id = _verify_scope(original, entry_obligations, observation, decision_ledger)
    graph = _ImportGraph()
    graph.sites(original)
    runs, assigned = _runtime(graph, observation)
    graph.close()
    return dict(schema=SCHEMA, schema_version=1, entry_obligations=entry_obligations, observation=diagnostic_ref,
        program=original['program'], original_source=source, target=target, decision_ledger=decision_ledger,
        scope_sha256=scope_id, static_sites=graph.static, modules=graph.modules, runs=runs,
        spelling_relation='COMPATIBLE_MODULE_SPELLING_ONLY', process_attribution=dict(
            scope='TASK_GROUP_IDENTITY_RETAINED_BY_ORIGINAL_DIAGNOSTIC', reported_tasks_assigned=assigned,
            processes_grouped_only_within_each_run=True, topology_complete=False),
        coverage_frontier=['DYNAMIC_IMPORT_CALLER_SOURCE_SITE_OPEN', 'IMPORT_TARGET_MODULE_OBJECT_AND_ORIGIN_OPEN',
            'IMPORT_HOOK_REGISTRY_AND_INITIALIZATION_EFFECTS_OPEN', 'UNOBSERVED_STATIC_IMPORT_REQUESTS_OPEN',
            'PROCESS_TIMING_OVERLAP_AND_UNOBSERVED_IMPORTS_OPEN'],
        self_times_are_wall_bounds=False, target_executed_by_projection=False,
        call_attribution_proven=False, transformation_authorized=False)


def verify_cpu_python_import_graph(reference):
    document = read_artifact(reference)
    _need(reference['schema'] == SCHEMA and document == derive_cpu_python_import_graph(
        document['entry_obligations'], document['observation'], decision_ledger=document['decision_ledger']),
        'retained import graph differs at cold replay')
    return document
