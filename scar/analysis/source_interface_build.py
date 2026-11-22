"""Source interface build."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import argparse
import gzip
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path
from scar.analysis.normal_facts import SourceFactStore
from scar.analysis.normal_graph import _Emitter
from scar.analysis.source_suspension_interfaces import SuspensionInterfaceIndex, _state_indices
from scar.analysis.source_templates import (
    _json,
    _worker_stores,
    read_template_graph,
    _check_catalog,
)
from scar.analysis.source_productions import _digest
from scar.analysis.source_stage import _durable_bytes, _durable_mkdir
from scar.ir.program import (
    normal_graph as F,
    source_productions as P,
    source_protocols as L,
    suspension_interfaces as I,
)
from scar.ir.program.normal_graph import GlobalPort
from scar.ir.program.structured import PythonProtocolKind as K
from scar.ir.record_codec import decode, encode
from functools import lru_cache


# Read compact local interfaces without recovering complete F4 plans.
@lru_cache(maxsize=1)
def _declaration_files():
    from scar.analysis.source_productions import production_type_catalog
    return tuple(Path(sys.modules[name].__file__).resolve() for name in production_type_catalog()['modules'])


def projection_pins():
    from scar.analysis.source_expression_materials import _implementation
    from scar.analysis.source_templates import _compiler_profile
    package = Path(__file__).resolve().parents[1]
    names = ('analysis/source_interface_build.py', 'analysis/source_interface_build.py',
             'analysis/source_suspension_interfaces.py', 'ir/program/source_protocols.py',
             'ir/program/suspension_interfaces.py', 'ir/program/normal_graph.py')
    paths = {str(package/name) for name in (*names,*_implementation())}
    paths.update(str(path) for path in _declaration_files())
    paths.add(_compiler_profile()['header'])
    return {path:sha256(Path(path).read_bytes()).hexdigest() for path in sorted(paths)}


def file_receipt(path):
    path = Path(path).absolute()
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=sha256(raw).hexdigest())


def _stamp(path):
    value = Path(path).stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


class SourceInterfaceStore:
    def __init__(self, path, facts):
        self.path = Path(path).absolute()
        self.raw = (self.path/'result.json').read_bytes()
        self.report = json.loads(self.raw)
        r = self.report
        if (r['schema'],r['schema_version'],r['status']) != ('scar.local-source-interfaces',1,'LOCAL_INTERFACES_PROJECTED'):
            raise ValueError('expected completed local source interfaces')
        if r['projection_pins'] != projection_pins():
            raise ValueError('local interface projection implementation differs')
        if r['source_facts_sha256'] != sha256(facts.raw).hexdigest() or Path(r['source_facts']) != facts.path:
            raise ValueError('local interfaces belong to another source fact catalog')
        self.facts = facts
        self.rows = {row['unit']:row for row in r['units']}
        if len(self.rows)!=r['selected_sources'] or not self.rows.keys()<=facts.rows.keys():
            raise ValueError('local interface unit directory differs')
        self.loaded = {};self.iterations = {};self.bodies = {};self.mappings = {};self.pins = {}
        self.stats = dict(unit_reads=0,iteration_reads=0,body_reads=0,mapping_reads=0)
        self._admit(dict(path=str(self.path/'result.json'),bytes=len(self.raw),sha256=sha256(self.raw).hexdigest()))
        for receipt in r['parents']:self._admit(receipt)

    def _admit(self, receipt):
        path = Path(receipt['path']);before = _stamp(path)
        if path in self.pins:
            if self.pins[path]!=before:raise ValueError('local interface input changed')
            return
        raw = path.read_bytes()
        if len(raw)!=receipt['bytes'] or sha256(raw).hexdigest()!=receipt['sha256'] or _stamp(path)!=before:
            raise ValueError('local interface byte receipt differs')
        self.pins[path] = before

    def validate_parents(self):
        for path,stamp in self.pins.items():
            if _stamp(path)!=stamp:raise ValueError('local interface snapshot changed during query')

    def covers(self, unit):
        return unit.unit in self.rows

    @property
    def receipt(self):
        return dict(path=str(self.path),sha256=sha256(self.raw).hexdigest(),
                    selected_sources=len(self.rows),catalog_complete=len(self.rows)==len(self.facts.rows))

    def load(self, unit):
        if unit.unit in self.loaded:return self.loaded[unit.unit]
        row = self.rows[unit.unit]
        if row['facts_sha256']!=self.facts.rows[unit.unit]['sha256']:
            raise ValueError('local interface source unit differs')
        for receipt in row['inputs']:self._admit(receipt)
        path = self.path/row['interfaces']
        self._admit(dict(path=str(path),bytes=row['bytes'],sha256=row['sha256']))
        raw = gzip.decompress(path.read_bytes())
        if len(raw)!=row['decoded_bytes'] or sha256(raw).hexdigest()!=row['decoded_sha256']:
            raise ValueError('decoded local interface receipt differs')
        value = decode(L.UnitSourceInterfaces,json.loads(raw),record_tags=P.PRODUCTION_RECORD_TAGS)
        if value.unit!=unit.unit:raise ValueError('local interface unit identity differs')
        expected = {i for i,f in enumerate(unit.functions) if f.environment.kind is not P.DeferredBodyKind.FUNCTION}
        if ({x.fact for x in value.iterations}!=set(range(len(unit.iterations)))
                or len(value.iterations)!=len(unit.iterations)
                or {x.fact for x in value.bodies}!=expected or len(value.bodies)!=len(expected)):
            raise ValueError('local interface family coverage differs')
        for entry in value.iterations:
            self.iterations[unit.unit,unit.iterations[entry.fact]] = entry.site
        for entry in value.bodies:
            self.bodies[unit.unit,unit.functions[entry.fact].value] = entry.site
        for entry in value.mappings:self.mappings[unit.unit,entry.template] = entry
        if len({x.template for x in value.mappings})!=len(value.mappings):
            raise ValueError('duplicate mapping interface template')
        expected_mapping = set()
        for item in unit.containers:
            if item.kind is not F.ContainerKind.MAPPING:continue
            raw = unit.ports[item.value]
            expected_mapping.add((raw['template']['occurrence'],raw['template']['purpose']) if type(raw) is dict
                                 else (raw.template.occurrence,raw.template.purpose.value))
        if {(x.template.occurrence,x.template.purpose.value) for x in value.mappings}!=expected_mapping:
            raise ValueError('mapping interface coverage differs')
        self.loaded[unit.unit]=value;self.stats['unit_reads']+=1
        return value

    def iteration(self, unit, item):
        self.load(unit);self.stats['iteration_reads']+=1
        return self.iterations[unit.unit,item]

    def body(self, unit, function):
        self.load(unit);self.stats['body_reads']+=1
        site = self.bodies[unit.unit,function.value]
        if site.body_scope!=function.body_scope or site.kind is not function.environment.kind:
            raise ValueError('local body interface selects another original function')
        return site

    def mapping(self, unit, template):
        self.load(unit);self.stats['mapping_reads']+=1
        # Missing is a projection/coverage error, never a proven empty effect.
        return self.mappings[unit.unit,template]


# Parallel local interface projection over existing, immutable F4/F5 shards.
_FACTS = None
_PINS = None
_CONTEXTS = {}


def _initialize(facts, pins):
    from scar.analysis.source_templates import _initialize_worker
    _initialize_worker()
    global _FACTS, _PINS
    _FACTS, _PINS = SourceFactStore(facts), pins
    if projection_pins()!=pins:raise ValueError('interface projector changed before worker initialization')


def _verify(receipt):
    if file_receipt(receipt['path'])!=receipt:raise ValueError('interface projection input changed')


def _context(parent):
    key = str(parent.path)
    if key not in _CONTEXTS:
        if (parent.path/'result.json').read_bytes()!=parent.raw:
            raise ValueError('primitive source fact receipt changed')
        root = parent.templates
        raw = (root/'result.json').read_bytes()
        if sha256(raw).hexdigest()!=parent.report['source_result_sha256']:
            raise ValueError('F4 interface root differs from facts6')
        if sha256((parent.model/'manifest.json').read_bytes()).hexdigest()!=parent.report['model_manifest_sha256']:
            raise ValueError('source model manifest differs from facts6')
        report = json.loads(raw)
        specs = tuple((p['domain'],p['materials'],p['verification'],p['collection_sha256'],p['verification_sha256'])
                      for p in report['parents'])
        materials = _worker_stores(specs,report['model_manifest_sha256'])
        refs = [file_receipt(parent.path/'result.json'),file_receipt(parent.model/'manifest.json'),file_receipt(root/'result.json')]
        for material in materials:
            refs.extend(file_receipt(path) for path in (material.root/'result.json',material.verification/'result.json',
                        material.root/'catalog.json',material.verification/'catalog.json'))
        _CONTEXTS[key] = root,report,materials,refs
    return _CONTEXTS[key]


def _project(unit, graph, plans, root_sha, model_sha, *, fact_store=None,
             witness_source=I.WitnessSource.SEALED_TEMPLATE_STORE):
    """Reuse the original port projector, seeded only with admitted sealed plans."""
    index = SuspensionInterfaceIndex(_FACTS if fact_store is None else fact_store)
    nodes = {node.key:node for node in graph.templates}
    for key,plan in plans.items():
        index._accept_plan(unit,key,plan,nodes[key].origin.plan_sha256,
                           witness_source,root_sha,model_sha)
    index._suspending[unit.unit] = {key for key,plan in plans.items()
        if any(type(step.instruction) is P.SuspendExecution for step in plan.steps)}
    iterations = tuple(L.IterationInterface(i,index.iteration(unit,item)) for i,item in enumerate(unit.iterations))
    bodies = tuple(L.DeferredInterface(i,index.body(unit,f)) for i,f in enumerate(unit.functions)
                   if f.environment.kind is not P.DeferredBodyKind.FUNCTION)
    mapping_keys = {index._port(unit,item.value).port.template for item in unit.containers if item.kind is F.ContainerKind.MAPPING}
    mapping_wire = {(key.occurrence,key.purpose.value) for key in mapping_keys}
    points = {}
    for i,raw in enumerate(unit.ports):
        template = raw['template'] if type(raw) is dict else None
        if template is not None:
            if (template['occurrence'],template['purpose']) not in mapping_wire:continue
        elif raw.template not in mapping_keys:continue
        points[index._port(unit,i)] = i
    mappings = []
    for key in sorted(mapping_keys,key=lambda k:(k.occurrence,k.purpose.value)):
        plan = plans[key];emitter = _Emitter(plan,key,{})
        merges = []
        for step in plan.steps:
            if type(step.instruction) is not P.ApplyPythonProtocol or step.instruction.protocol is not K.DICT_MERGE:continue
            arguments = tuple(GlobalPort(unit.unit,emitter.value(v)[0]) for v in step.arguments)
            states = _state_indices(tuple(emitter.value(v)[1] for v in step.arguments))
            if len(states)!=1:raise ValueError('DICT_MERGE requires its original current-world input')
            target,source = arguments[1:3]
            merges.append(L.MappingMergeInterface(index._witness(unit,key,step),arguments[0],target,source,
                arguments[states[0]],tuple(index._completion(unit,emitter,step.id,c) for c in step.completions),
                points.get(target),points.get(source)))
        mappings.append(L.MappingTemplateInterface(key,nodes[key].origin.plan_sha256,
            all(step.region==plan.root for step in plan.steps),tuple(merges)))
    if index.stats['plan_reads'] or index.stats['prefetch_batches']:
        raise ValueError('local projection unexpectedly recovered an unselected plan')
    return L.UnitSourceInterfaces(unit.unit,iterations,bodies,tuple(mappings))


def _project_unit(task):
    identity,out = task
    started,cpu = time.monotonic(),time.process_time()
    if projection_pins()!=_PINS:raise ValueError('interface projector changed during build')
    parent = _FACTS.parents[_FACTS.unit_parent[identity]] if _FACTS.parents else _FACTS
    unit = _FACTS.load(identity)
    root,report,materials,common = _context(parent)
    fact_row = parent.rows[identity]
    inputs = [dict(path=str(parent.path/fact_row['facts']),bytes=fact_row['bytes'],sha256=fact_row['sha256'])]
    index = SuspensionInterfaceIndex(_FACTS)
    required = {index._port(unit,item.value).port.template for item in unit.iterations}
    required.update(f.template for f in unit.functions if f.environment.kind is not P.DeferredBodyKind.FUNCTION)
    required.update(index._port(unit,item.value).port.template for item in unit.containers if item.kind is F.ContainerKind.MAPPING)
    scanned = 0;plans = {}
    if required:
        graph = read_template_graph(root,identity)
        origins = {(n.origin.domain,n.origin.line):n for n in graph.templates}
        seen = set()
        discriminator,tag = P.PRODUCTION_RECORD_TAGS[P.SuspendExecution]
        for material in materials:
            _check_catalog(material,identity)
            basis = material.basis[identity]
            for domain,line,document in material.records(dict(key=identity,**{k:v for k,v in basis.items() if k!='unit'})):
                node = origins.get((domain,line))
                if node is None:continue
                scanned+=1;seen.add((domain,line))
                if node.key not in required and not any(s['instruction'].get(discriminator)==tag for s in document['steps']):continue
                if _digest(document)!=node.origin.plan_sha256:raise ValueError('local interface selected another original plan')
                # Same admission as the immutable facts6 producer: the F4
                # proof/digest is reused; restore exact typed records without
                # replaying source rules or rebuilding their type catalog.
                plan = decode(P.SourceProductionPlan,document,record_tags=P.PRODUCTION_RECORD_TAGS)
                if (plan.source!=P.ProductionOccurrence(node.key.occurrence) or plan.purpose is not node.key.purpose
                        or plan.root!=node.root or plan.input!=graph.input or plan.profile!=graph.profile):
                    raise ValueError('projected plan identity/input/profile differs')
                plans[node.key] = plan
            descriptor = material.rows[identity]['artifact']
            inputs.append(dict(path=str(material.root/descriptor['path']),bytes=descriptor['bytes'],sha256=descriptor['sha256']))
        if set(origins)!=seen or not required<=plans.keys():raise ValueError('local interface template coverage differs')
        value = _project(unit,graph,plans,parent.report['source_result_sha256'],report['model_manifest_sha256'])
        entry = next(row for row in report['units'] if row['unit']==identity)
        header_path = root/entry['header'];header = json.loads(header_path.read_bytes())
        inputs.append(dict(path=str(header_path),bytes=entry['header_bytes'],sha256=entry['header_sha256']))
        for name in ('templates','links'):
            d = header[name];inputs.append(dict(path=str(header_path.parent/d['path']),bytes=d['bytes'],sha256=d['sha256']))
    else:value = L.UnitSourceInterfaces(identity,(),(),())
    for receipt in inputs:_verify(receipt)
    for material in materials:
        _check_catalog(material,identity)
        if not material.unchanged():raise ValueError('interface material proof changed')
    if projection_pins()!=_PINS:raise ValueError('interface projector changed during projection')
    raw = _json(encode(value,record_tags=P.PRODUCTION_RECORD_TAGS));payload = gzip.compress(raw,compresslevel=3,mtime=0)
    path = Path(out)/'units'/identity/'interfaces.json.gz';_durable_mkdir(path.parent);_durable_bytes(path,payload,replace=False)
    row = dict(unit=identity,path=fact_row['path'],interfaces=str(path.relative_to(out)),
        bytes=len(payload),sha256=sha256(payload).hexdigest(),decoded_bytes=len(raw),decoded_sha256=sha256(raw).hexdigest(),
        facts_sha256=fact_row['sha256'],inputs=inputs,parents=common,
        counts=dict(iterations=len(value.iterations),bodies=len(value.bodies),mappings=len(value.mappings),
                    merges=sum(len(m.merges) for m in value.mappings)),
        scanned_templates=scanned,selected_plans=len(plans),worker_pid=os.getpid(),
        seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu)
    _FACTS.loaded.clear()
    for item in _FACTS.parents:item.loaded.clear()
    return row


def build_interfaces(facts,out,*,sources=(),jobs=None):
    from scar.analysis.source_templates import _default_jobs
    started = time.monotonic();facts = SourceFactStore(facts);out = Path(out).absolute()
    jobs = _default_jobs() if jobs is None else jobs
    if type(jobs) is not int or jobs<1 or out.exists():raise ValueError('fresh interface output and positive worker count required')
    wanted = {str(Path(p).absolute()) for p in sources}
    identities = [key for key,row in facts.rows.items() if not wanted or row['path'] in wanted]
    if not identities or wanted and {facts.rows[k]['path'] for k in identities}!=wanted:raise ValueError('interface source selection differs')
    pins = projection_pins();_durable_mkdir(out)
    tasks = [(key,str(out)) for key in identities]
    if jobs==1:
        _initialize(str(facts.path),pins);rows = [_project_unit(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=jobs,mp_context=multiprocessing.get_context('spawn'),
                initializer=_initialize,initargs=(str(facts.path),pins)) as pool:
            rows = list(pool.map(_project_unit,tasks,chunksize=1))
    parents = {str(facts.path/'result.json'):file_receipt(facts.path/'result.json')}
    counts = Counter()
    for row in rows:
        for ref in row.pop('parents'):
            if ref['path'] in parents and parents[ref['path']]!=ref:raise ValueError('interface workers used different snapshots')
            parents[ref['path']] = ref
        counts.update(row['counts'])
    for ref in parents.values():_verify(ref)
    if projection_pins()!=pins or (facts.path/'result.json').read_bytes()!=facts.raw:raise ValueError('interface projection input/implementation changed')
    result = dict(schema='scar.local-source-interfaces',schema_version=1,status='LOCAL_INTERFACES_PROJECTED',
        source_facts=str(facts.path),source_facts_sha256=sha256(facts.raw).hexdigest(),projection_pins=pins,
        selected_sources=len(rows),catalog_sources=len(facts.rows),catalog_complete=len(rows)==len(facts.rows),
        parents=list(parents.values()),units=rows,counts=dict(counts),jobs=jobs,
        worker_pids=sorted({row['worker_pid'] for row in rows}),worker_cpu_seconds=sum(row['cpu_seconds'] for row in rows),
        stored_bytes=sum(row['bytes'] for row in rows),selected_plans=sum(row['selected_plans'] for row in rows),
        source_rules_replayed=False,profile='NORMAL_SUCCESS',first_stage_complete=False,target_executed=False,
        seconds=time.monotonic()-started)
    _durable_bytes(out/'result.json',_json(result),replace=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--facts',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--source',action='append',default=[]);parser.add_argument('--jobs',type=int,default=None)
    args = parser.parse_args();r = build_interfaces(args.facts,args.out,sources=args.source,jobs=args.jobs)
    print(json.dumps({k:r[k] for k in ('status','selected_sources','counts','jobs','worker_pids','stored_bytes','seconds')},sort_keys=True))


if __name__=='__main__':main()
