"""Source templates."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from hashlib import sha256
import json
import argparse
import gzip
import io
import multiprocessing
import os
import re
import sysconfig
import time
import importlib.abc
import sys
from pathlib import Path
from scar.analysis.source_productions import (
    _digest,
    read_production_plan,
    production_type_catalog,
)
from scar.ir.program import source_productions as P, source_declarations as T
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
from dataclasses import fields, is_dataclass
from functools import lru_cache
from scar.analysis.source_expression_materials import _implementation, _basis
from scar.artifacts import _durable_bytes, _durable_mkdir, _sync_directory
from scar.analysis.static.query import _Fragments
from scar.ir.program.control import LanguageCompletionKind as C
from scar.ir.program.model import ValueKind, ValueType
from scar.ir.record_codec import encode, decode
from scar.artifacts import _json, write_table
from contextlib import contextmanager
from threading import RLock
from scar.ir.program.source_declarations import SourceTemplateKey


# Link sealed source templates by original slots, without lowering or running.
#
# Each source unit is independent. Stored nodes reference the original recipe
# bodies; edges bind actual values to ordered parameters and completion ports.
class TemplateLinkError(ValueError):
    pass


def _require(condition, path, detail):
    if not condition:
        raise TemplateLinkError(str(path) + ': ' + detail)


def validate_state_sequence(plan):
    """Check current-world origins in an already shape-checked local recipe.

    Types alone permit using a previous world after an effect. Read-only
    operations may inspect the current world without replacing it; a normal
    state result becomes the following steps' current world. Nested call
    operands are included, without entering their shared/deferred bodies.

    This proves local sequencing, not the physical effects of native calls,
    the value of a mutable referent, or Python reference cleanup semantics.
    """
    regions = {r.id: r for r in plan.regions}
    normal = {s.id: next((c.types for c in s.completions if c.kind is C.FALLTHROUGH), ())
        for s in plan.steps}
    steps = {s.id: s for s in plan.steps}

    def state(kind):
        return type(kind) is ValueType and kind.kind is ValueKind.STATE

    def value_type(value):
        if type(value) is P.ProductionParameter:
            return regions[value.region].parameters[value.index]
        return normal[value.step][value.index]

    def operands(step):
        # The finite instruction schema also contains nested call arguments,
        # captures and saved operands. Do not infer them from display labels.
        pending = [step.instruction, step.arguments]
        while pending:
            item = pending.pop()
            if type(item) in (P.ProductionParameter, P.ProductionResult):
                yield item
            elif type(item) is tuple:
                pending.extend(reversed(item))
            elif is_dataclass(item):
                pending.extend(getattr(item, field.name) for field in reversed(fields(item)))

    uses = results = loops = 0
    for region in plan.regions:
        incoming = [P.ProductionParameter(region.id, i) for i, kind in enumerate(region.parameters) if state(kind)]
        _require(len(incoming) <= 1, region.id, 'one current-world input is permitted per local region')
        current = incoming[0] if incoming else None
        for identity in region.steps:
            step = steps[identity]
            used = [v for v in operands(step) if state(value_type(v))]
            _require(all(v == current for v in used), identity, 'operand uses a stale or foreign world instead of the current world')
            uses += len(used)
            if type(step.instruction) is P.LoopControl:
                loops += 1
                _require(not any(state(value_type(v)) for v in step.arguments[:step.instruction.invariant_count]),
                    identity, 'current world must be loop-carried, never a loop invariant')
            produced = [i for i, kind in enumerate(normal[identity]) if state(kind)]
            _require(len(produced) <= 1, identity, 'one normal current-world result is permitted per step')
            if produced:
                _require(bool(used), identity, 'a normal world result requires a current-world operand')
                current = P.ProductionResult(identity, produced[0])
                results += 1
    return dict(regions=len(regions), steps=len(steps), state_uses=uses, state_results=results, loop_templates=loops)


def build_template_graph(unit, syntax, records, *, max_visits=1_000_000):
    """Check external plan documents before linking the original slots."""
    def admit(_domain, _line, document):
        plan = read_production_plan(document, max_visits=max_visits)
        return plan, _digest(document)

    return _link_template_graph(unit, syntax, records, admit)


def _build_template_graph_from_admitted_plans(unit, syntax, records):
    """Link the current producer's source-matched plans and return their wire rows."""
    documents = {domain: [] for domain in T.SourceMaterialDomain}

    def admit(domain, line, plan):
        _require(type(plan) is P.SourceProductionPlan and line == len(documents[domain]) + 1,
            unit, 'invalid admitted plan or original material line')
        document = encode(plan, record_tags=P.PRODUCTION_RECORD_TAGS)
        canonical = _json(document)[:-1]
        # Each wire value occupies bytes in canonical JSON. For an unusually
        # large record, retain the reader's exact wire visit budget.
        if len(canonical) > 1_000_000:
            read_production_plan(document)
        documents[domain].append(document)
        return plan, sha256(canonical).hexdigest()

    graph = _link_template_graph(unit, syntax, records, admit)
    return graph, documents


def _link_template_graph(unit, syntax, records, admit):
    nodes = {}; contexts = {}; sites = []; common_input = common_profile = None
    syntax_digest = _digest(syntax)
    for domain, line, record in records:
        _require(type(domain) is T.SourceMaterialDomain and type(line) is int and line > 0,
            unit, 'invalid original material coordinates')
        plan, digest = admit(domain, line, record)
        validate_state_sequence(plan)
        _require(type(plan.source) is P.ProductionOccurrence and plan.profile is not None
            and plan.compilation_flow is not None, unit, 'expected a source-bound profiled occurrence template')
        if common_input is None:
            common_input, common_profile = plan.input, plan.profile
        _require(plan.input == common_input and plan.profile == common_profile,
            unit, 'templates mix source inputs or state/exception profiles')
        _require(plan.input.f2_sha256 == syntax_digest and plan.input.path == syntax['input']['path'], unit, 'original F2 input differs')
        key = T.SourceTemplateKey(plan.source.occurrence, plan.purpose)
        if key in nodes:
            _require(nodes[key].origin.plan_sha256 == digest, key, 'conflicting templates for the same source/purpose')
            continue
        regions = {r.id: r for r in plan.regions}; steps = {s.id: s for s in plan.steps}
        root = regions[plan.root]
        nodes[key] = T.SourceTemplateNode(key, root.id, root.parameters, root.completions,
            T.SourceTemplateOrigin(domain, line, digest))
        contexts[key] = next(p for p in plan.compilation_flow.ports if p.region == plan.root)
        child_contexts = {e.step:e.target for e in plan.compilation_flow.edges if e.kind is P.CompilationEdgeKind.CHILD}
        for step in plan.steps:
            instruction = step.instruction
            if type(instruction) is not P.EvaluateChild:
                continue
            field = instruction.field
            _require(field.group == 'FIELD' and 0 <= field.occurrence < len(syntax['nodes']), step.id, 'child has no original field')
            fields = syntax['nodes'][field.occurrence]['fields']
            _require(0 <= field.ordinal < len(fields), step.id, 'child field ordinal outside original F2')
            value = fields[field.ordinal]
            for index in field.item_path:
                _require(value['kind'] == 'LIST' and 0 <= index < len(value['items']), step.id, 'child list path outside original F2')
                value = value['items'][index]
            _require(value['kind'] == 'NODE', step.id, 'child entry is not one original occurrence')
            target = T.SourceTemplateKey(value['reference'], instruction.purpose)
            arguments = []
            for argument in step.arguments:
                if type(argument) is P.ProductionParameter:
                    arguments.append(regions[argument.region].parameters[argument.index])
                else:
                    normal = next(c.types for c in steps[argument.step].completions if c.kind is C.FALLTHROUGH)
                    arguments.append(normal[argument.index])
            sites.append((key, step, target, tuple(arguments), child_contexts[step.id]))
    _require(common_input is not None, unit, 'no executable source templates in the selected material domain')
    links = []
    for caller, step, target, arguments, context in sites:
        _require(target in nodes, step.id, 'missing target template: ' + str(target))
        callee = nodes[target]
        _require(arguments == callee.parameters, step.id, 'actual arguments differ from callee parameters')
        advertised = {(c.kind, c.target): c.types for c in step.completions}
        _require(all(advertised.get((c.kind, c.target)) == c.types for c in callee.completions),
            step.id, 'callee completion is not represented by the caller interface')
        port = contexts[target]
        _require((context.source_scope, context.code_scope) == (port.source_scope, port.code_scope),
            step.id, 'child compilation environment differs from the target entry')
        links.append(T.SourceTemplateLink(caller, step.id, target, step.arguments))
    return T.SourceTemplateGraph(unit, common_input, common_profile, tuple(nodes.values()), tuple(links))


def _compiler_profile():
    header = Path(sysconfig.get_path('include')) / 'internal' / 'pycore_compile.h'
    raw = header.read_bytes()
    limits = re.findall(rb'^#define\s+_PY_STACK_USE_GUIDELINE\s+([0-9]+)\s*$', raw, re.MULTILINE)
    _require(len(limits) == 1 and int(limits[0]) > 0, header, 'current compiler profile lacks a unique positive stack guideline')
    return dict(header=str(header), header_sha256=sha256(raw).hexdigest(), stack_guideline=int(limits[0]))


class _Store:
    def __init__(self, domain, material, verification, manifest_sha256):
        self.domain = T.SourceMaterialDomain(domain)
        self.root = Path(material).absolute(); self.verification = Path(verification).absolute()
        _require(self.root.resolve() == self.root and self.verification.resolve() == self.verification,
            domain, 'parent paths must be canonical')
        raw = (self.root / 'result.json').read_bytes(); checked = (self.verification / 'result.json').read_bytes()
        a, b = json.loads(raw), json.loads(checked)
        _require(a['mode'] == 'COLLECTION' and b['mode'] == 'FRESH_VERIFICATION'
            and a['status'] == b['status'] == 'LOCAL_RULES_VERIFIED', domain, 'parent materials lack completed independent verification')
        for field in ('schema', 'schema_version', 'domain', 'input_basis', 'counts', 'limits', 'catalog_sha256', 'implementation_sha256', 'model_manifest_sha256'):
            _require(a[field] == b[field], domain, 'collection/verification differ: ' + field)
        _require(a['domain'] == domain and a['schema_version'] == 4
            and a['implementation_sha256'] == _implementation() and a['model_manifest_sha256'] == manifest_sha256,
            domain, 'parent domain, model or generator version differs')
        catalog = (self.root / 'catalog.json').read_bytes()
        _require(sha256(catalog).hexdigest() == a['catalog_sha256']
            and (self.verification / 'catalog.json').read_bytes() == catalog, domain, 'parent catalog differs')
        declaration = json.loads(catalog)
        _require(type(declaration) is dict, domain, 'parent catalog must be a record')
        self.catalog_type_sha256 = declaration.get('recipe_type_catalog_sha256')
        self.compiler_profile = declaration.get('call_compiler_profile')
        _require(self.catalog_type_sha256 == production_type_catalog()['content_sha256'],
            domain, 'parent recipe differs from the current type/support catalog')
        _require(self.compiler_profile == _compiler_profile(),
            domain, 'parent recipe differs from the current compiler profile')
        self.profile = decode(P.SourceProductionProfile,
            dict(world=declaration['state_parameter'], exception_abi=declaration['exception_abi']))
        self.rows = {u['unit']: u for u in a['units']}; self.basis = {u['unit']: u for u in a['input_basis']}
        fresh_rows = {u['unit']: u for u in b['units']}
        _require(len(self.rows) == len(a['units']) and len(self.basis) == len(a['input_basis'])
            and len(fresh_rows) == len(b['units']) and set(self.rows) == set(self.basis) == set(fresh_rows),
            domain, 'duplicate or missing parent source units')
        _require(all(self.rows[key][field] == fresh_rows[key][field] for key in self.rows
            for field in ('artifact', 'input', 'catalog_sha256')), domain, 'fresh proof refers to another unit fragment/input')
        self.generator_pins = a['implementation_sha256']
        self.receipt = dict(domain=domain, materials=str(self.root), verification=str(self.verification),
            collection_sha256=sha256(raw).hexdigest(), verification_sha256=sha256(checked).hexdigest())

    def unchanged(self):
        return (sha256((self.root / 'result.json').read_bytes()).hexdigest() == self.receipt['collection_sha256']
            and sha256((self.verification / 'result.json').read_bytes()).hexdigest() == self.receipt['verification_sha256']
            and self.catalog_type_sha256 == production_type_catalog()['content_sha256']
            and self.compiler_profile == _compiler_profile())

    def records(self, unit):
        _require(unit['key'] in self.rows and self.basis[unit['key']] == _basis(unit), unit['path'], 'parent material/input is unavailable or differs')
        row = self.rows[unit['key']]; descriptor = row['artifact']; path = self.root / descriptor['path']
        _require(path.resolve() == path and path.is_relative_to(self.root) and not path.is_symlink(), unit['path'], 'unsafe parent fragment path')
        payload = path.read_bytes()
        _require(len(payload) == descriptor['bytes'] and sha256(payload).hexdigest() == descriptor['sha256'], unit['path'], 'parent compressed receipt differs')
        digest = sha256(); size = 0
        with gzip.GzipFile(fileobj=io.BytesIO(payload)) as stream:
            for line, raw in enumerate(stream, 1):
                digest.update(raw); size += len(raw)
                _require(size <= descriptor['decoded_bytes'], path, 'parent decoded byte budget exceeded')
                record = json.loads(raw)
                if 'plan' in record:
                    yield self.domain, line, record['plan']
        _require(size == descriptor['decoded_bytes'] and digest.hexdigest() == descriptor['decoded_sha256'], path, 'parent decoded receipt differs')


@lru_cache(maxsize=1)
def _worker_stores(specifications, manifest_sha):
    """Process-local immutable parent indexes; no shared graph or descriptor table."""
    stores = []
    for domain, material, verification, collected_sha, verified_sha in specifications:
        store = _Store(domain, material, verification, manifest_sha)
        _require((store.receipt['collection_sha256'], store.receipt['verification_sha256'])
            == (collected_sha, verified_sha), domain, 'worker parent proof differs from its submitted snapshot')
        stores.append(store)
    return tuple(stores)


def _link_unit(task):
    unit, model, out, specifications, manifest_sha, max_visits, linker_pin, types_pin = task
    _require(linker_pin == sha256(Path(__file__).read_bytes()).hexdigest()
        and types_pin == sha256(Path(T.__file__).read_bytes()).hexdigest(), unit['path'], 'worker linker version differs')
    stores = _worker_stores(specifications, manifest_sha)
    try:
        syntax = _Fragments(Path(model)).read(unit['artifacts']['source_syntax'])
        graph = build_template_graph(unit['key'], syntax,
            (record for store in stores for record in store.records(unit)), max_visits=max_visits)
        path = Path(out) / 'units' / unit['key']; _durable_mkdir(path)
        nodes = write_table(path / 'templates.jsonl.gz', graph.templates)
        links = write_table(path / 'links.jsonl.gz', graph.links)
        header = dict(schema=graph.schema, schema_version=graph.schema_version, authority=graph.authority,
            unit=unit['key'], input=encode(graph.input), profile=encode(graph.profile), templates=nodes, links=links)
        header_bytes = _json(header)
        _durable_bytes(path / 'header.json', header_bytes, replace=False); _sync_directory(path)
        return dict(unit=unit['key'], path=unit['path'], header='units/'+unit['key']+'/header.json',
            header_bytes=len(header_bytes), header_sha256=sha256(header_bytes).hexdigest(),
            templates=nodes['records'], links=links['records'], compressed_bytes=nodes['bytes']+links['bytes'])
    except Exception as error:
        error.add_note('SCAR template interfaces source: ' + unit['path'])
        raise


def link_source_templates(model, out, parents, sources=(), *, max_visits=1_000_000, jobs=1):
    started = time.monotonic()
    _require(type(max_visits) is int and max_visits >= 0, 'max_visits', 'expected a nonnegative integer')
    _require(type(jobs) is int and 1 <= jobs <= 32, 'jobs', 'expected 1 through 32 independent source workers')
    model = Path(model).absolute(); out = Path(out).absolute()
    _require(model.resolve() == model and out.resolve() == out and not out.exists(), out, 'output must be a fresh canonical path')
    raw = (model / 'manifest.json').read_bytes(); manifest_sha = sha256(raw).hexdigest(); manifest = json.loads(raw)
    linker_pin = sha256(Path(__file__).read_bytes()).hexdigest(); types_pin = sha256(Path(T.__file__).read_bytes()).hexdigest()
    stores = sorted((_Store(*parent, manifest_sha) for parent in parents), key=lambda s:s.domain.value)
    _require(len(stores) == 3 and {s.domain for s in stores} == set(T.SourceMaterialDomain), 'parents', 'one independently verified store per domain is required')
    _require(all(not out.is_relative_to(p) and not p.is_relative_to(out) for p in
        (model, *(s.root for s in stores), *(s.verification for s in stores))), out, 'output overlaps a parent')
    wanted = {str(Path(p).absolute()) for p in sources}
    units = sorted((u for u in manifest['units'] if not wanted or u['path'] in wanted), key=lambda u:u['key'])
    _require(bool(units) and (not wanted or wanted == {u['path'] for u in units}), 'sources', 'source selection is not exact')
    _require(len({u['key'] for u in units}) == len(units), 'sources', 'duplicate source unit identity')
    _require(all(u['key'] in s.rows for u in units for s in stores), 'parents', 'selected sources lack a complete material domain')
    specifications = tuple((s.domain.value, str(s.root), str(s.verification), s.receipt['collection_sha256'],
        s.receipt['verification_sha256']) for s in stores)
    tasks = [(u, str(model), str(out), specifications, manifest_sha, max_visits, linker_pin, types_pin) for u in units]
    _durable_mkdir(out)
    if jobs == 1:
        rows = [_link_unit(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context('spawn')) as pool:
            rows = list(pool.map(_link_unit, tasks))
    _require((model / 'manifest.json').read_bytes() == raw, model, 'model changed during interface linking')
    _require(all(s.unchanged() and s.generator_pins == _implementation() for s in stores)
        and linker_pin == sha256(Path(__file__).read_bytes()).hexdigest()
        and types_pin == sha256(Path(T.__file__).read_bytes()).hexdigest(), out, 'parent proofs or implementation changed during linking')
    report = dict(schema='scar.source-template-link-store', schema_version=1, status='SOURCE_INTERFACES_LINKED',
        generator_sha256=stores[0].generator_pins, linker_sha256=linker_pin,
        link_types_sha256=types_pin, model_manifest_sha256=manifest_sha,
        parents=[s.receipt for s in stores], selected_sources=len(rows), units=rows, max_visits=max_visits, jobs=jobs,
        templates=sum(r['templates'] for r in rows), links=sum(r['links'] for r in rows),
        compressed_bytes=sum(r['compressed_bytes'] for r in rows),
        local_state_sequence_checked=True,
        parent_catalogs_current=True,
        canonical_graph_generated=False, child_semantics_proven=False, effects_proven=False,
        first_stage_complete=False, target_executed=False, transformation_authorized=False, seconds=time.monotonic()-started)
    _durable_bytes(out / 'result.json', _json(report), replace=False)
    return report


def _read_graph_table(path, descriptor, expected):
    fragment = path.parent / descriptor['path']
    _require(fragment.resolve() == fragment and fragment.parent == path.parent, path, 'unsafe graph table')
    payload = fragment.read_bytes()
    _require(len(payload) == descriptor['bytes'] and sha256(payload).hexdigest() == descriptor['sha256'], path, 'graph table compressed receipt differs')
    digest = sha256(); size = 0; rows = []
    with gzip.GzipFile(fileobj=io.BytesIO(payload), mode='rb') as stream:
        for raw in stream:
            digest.update(raw); size += len(raw)
            _require(size <= descriptor['decoded_bytes'], path, 'graph table decoded budget exceeded')
            rows.append(decode(expected, json.loads(raw), record_tags=P.PRODUCTION_RECORD_TAGS))
    _require(size == descriptor['decoded_bytes'] and digest.hexdigest() == descriptor['decoded_sha256']
        and len(rows) == descriptor['records'], path, 'graph table decoded receipt or record count differs')
    return tuple(rows)


def read_template_graph(store, unit):
    """Read one typed interface graph and verify its sealed parent snapshot.

    Checks stored tables, identities, interface-reference closure and current
    declarations. Source-rule semantics were checked by the producer; original
    bodies are read on expansion. No native-effect or canonical-IR proof is
    granted here. Parent indexes are process-local immutable cache entries.
    """
    store = Path(store).absolute()
    _require(store.resolve() == store and type(unit) is str and Path(unit).name == unit and unit not in ('', '.', '..'), store, 'invalid store or source unit')
    report_bytes = (store / 'result.json').read_bytes(); report = json.loads(report_bytes)
    _require(report['schema'] == 'scar.source-template-link-store' and report['schema_version'] == 1
        and report['status'] == 'SOURCE_INTERFACES_LINKED', store, 'no completed interface link publication')
    _require(report['link_types_sha256'] == sha256(Path(T.__file__).read_bytes()).hexdigest()
        and report['generator_sha256'] == _implementation(), store, 'stored interface or generator declarations differ')
    _require(len(report['units']) == report['selected_sources']
        and len({r['unit'] for r in report['units']}) == len(report['units']), store, 'source index count or identities differ')
    specifications = tuple((p['domain'], p['materials'], p['verification'], p['collection_sha256'], p['verification_sha256']) for p in report['parents'])
    _require(len(specifications) == 3 and {p[0] for p in specifications} == {d.value for d in T.SourceMaterialDomain}, store, 'one parent snapshot per domain is required')
    parents = _worker_stores(specifications, report['model_manifest_sha256'])
    entries = [r for r in report['units'] if r['unit'] == unit]
    _require(len(entries) == 1, unit, 'unit is absent or ambiguous')
    _require(entries[0]['header'] == 'units/' + unit + '/header.json', unit, 'source header identity differs')
    path = store / entries[0]['header']
    _require(path.resolve() == path and path.is_relative_to(store), unit, 'unsafe unit header')
    raw_header = path.read_bytes()
    _require(len(raw_header) == entries[0]['header_bytes'] and sha256(raw_header).hexdigest() == entries[0]['header_sha256'], unit, 'unit header receipt differs')
    header = json.loads(raw_header)
    _require((header['schema'], header['schema_version'], header['authority'], header['unit']) ==
        ('scar.source-template-links', 1, 'SOURCE_TEMPLATE_INTERFACES_ONLY', unit), unit, 'unsupported source graph header')
    _require(all(unit in p.rows and header['input'] == p.rows[unit]['input'] for p in parents), unit, 'graph input differs from its source material')
    profile = decode(P.SourceProductionProfile, header['profile'])
    _require(all(profile == p.profile for p in parents), unit, 'graph state/exception profile differs from its source material')
    templates = _read_graph_table(path, header['templates'], T.SourceTemplateNode)
    links = _read_graph_table(path, header['links'], T.SourceTemplateLink)
    _require(len(templates) == entries[0]['templates'] and len(links) == entries[0]['links']
        and entries[0]['compressed_bytes'] == header['templates']['bytes'] + header['links']['bytes'], unit, 'unit index counts differ from tables')
    keys = {n.key for n in templates}; sites = {(e.caller, e.step) for e in links}
    _require(len(keys) == len(templates) and len(sites) == len(links), unit, 'duplicate template or child-entry site')
    _require(all(e.caller in keys and e.callee in keys and all(v.index >= 0 for v in e.arguments) for e in links), unit, 'dangling template edge or negative argument ordinal')
    _require(all(n.origin.line > 0 and len(n.origin.plan_sha256) == 64
        and all(c in '0123456789abcdef' for c in n.origin.plan_sha256) for n in templates), unit, 'invalid original body coordinates')
    _require((store / 'result.json').read_bytes() == report_bytes
        and all(p.unchanged() and p.generator_pins == _implementation() for p in parents), store, 'graph or parent snapshots changed during reading')
    return T.SourceTemplateGraph(unit, decode(P.ProductionInput, header['input']),
        profile, templates, links)


def _verify_template_unit(task):
    store, unit, reader_pin = task
    _require(sha256(Path(__file__).read_bytes()).hexdigest() == reader_pin, unit, 'worker reader version differs')
    graph = read_template_graph(store, unit)
    return dict(unit=unit, templates=len(graph.templates), links=len(graph.links))


def verify_template_store(store, *, jobs=1):
    """Read every stored graph; return compact receipts, never centralized graphs.

    This checks wire storage and template-reference closure. It does not
    regenerate source rules or prove native effects or cleanup semantics.
    """
    started = time.monotonic(); store = Path(store).absolute()
    _require(type(jobs) is int and 1 <= jobs <= 32, 'jobs', 'expected 1 through 32 source workers')
    _require(store.resolve() == store, store, 'store path must be canonical')
    raw = (store / 'result.json').read_bytes(); report = json.loads(raw)
    reader_pin = sha256(Path(__file__).read_bytes()).hexdigest()
    tasks = [(str(store), row['unit'], reader_pin) for row in report['units']]
    _require(bool(tasks), store, 'no stored source graphs')
    if jobs == 1:
        rows = [_verify_template_unit(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context('spawn')) as pool:
            rows = list(pool.map(_verify_template_unit, tasks))
    templates = sum(r['templates'] for r in rows); links = sum(r['links'] for r in rows)
    _require(len(rows) == report['selected_sources'] and templates == report['templates']
        and links == report['links'] and sum(r['compressed_bytes'] for r in report['units']) == report['compressed_bytes'],
        store, 'global graph counts differ from stored source units')
    _require((store / 'result.json').read_bytes() == raw
        and sha256(Path(__file__).read_bytes()).hexdigest() == reader_pin, store, 'store or reader changed during verification')
    return dict(schema='scar.source-template-store-verification', schema_version=1, status='STORED_INTERFACES_VERIFIED',
        graph_result_sha256=sha256(raw).hexdigest(), reader_sha256=reader_pin,
        link_types_sha256=report['link_types_sha256'], model_manifest_sha256=report['model_manifest_sha256'],
        selected_sources=len(rows), templates=templates, links=links, compressed_bytes=report['compressed_bytes'],
        units=rows, jobs=jobs, seconds=time.monotonic()-started, parent_catalogs_current=True,
        table_receipts_checked=True, stored_template_references_closed=True,
        source_rules_replayed=False, canonical_graph_generated=False, effects_proven=False,
        first_stage_complete=False, target_executed=False, transformation_authorized=False)


def read_template_body(store, unit, key, *, max_visits=1_000_000):
    """Expand one graph node to its original typed source recipe."""
    _require(type(key) is T.SourceTemplateKey, unit, 'invalid template key')
    store = Path(store).absolute()
    graph = read_template_graph(store, unit)
    found = [n for n in graph.templates if n.key == key]
    _require(len(found) == 1, key, 'template is absent or ambiguous')
    node = found[0]
    report = json.loads((store / 'result.json').read_bytes())
    parents = [p for p in report['parents'] if p['domain'] == node.origin.domain.value]
    _require(len(parents) == 1, unit, 'missing original material domain')
    parent = parents[0]; material = _Store(parent['domain'], parent['materials'], parent['verification'], report['model_manifest_sha256'])
    _require(material.receipt == parent, unit, 'parent proof changed since interface linking')
    # _Store.records exhausts and hashes the whole original fragment before
    # the result is returned, including rows after the selected one.
    selected = None
    basis = material.basis[unit]
    for domain, line, document in material.records(dict(key=unit, **{k:v for k,v in basis.items() if k!='unit'})):
        if line == node.origin.line:
            _require(_digest(document) == node.origin.plan_sha256, key, 'original recipe body differs')
            selected = read_production_plan(document, max_visits=max_visits)
    _require(selected is not None and selected.source == P.ProductionOccurrence(key.occurrence)
        and selected.purpose is key.purpose and selected.root == node.root
        and selected.input == graph.input and selected.profile == graph.profile,
        key, 'original recipe identity/input/profile differs')
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True); parser.add_argument('--out', required=True)
    parser.add_argument('--materials', nargs=3, action='append', required=True, metavar=('DOMAIN', 'COLLECTION', 'VERIFICATION'))
    parser.add_argument('--source', action='append', default=[])
    parser.add_argument('--max-visits', type=int, default=1_000_000)
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()
    result = link_source_templates(args.model, args.out, args.materials, args.source, max_visits=args.max_visits, jobs=args.jobs)
    print(json.dumps({k:result[k] for k in ('status','selected_sources','templates','links','compressed_bytes')}, indent=2))


# Read selected original plans with one sealed graph/domain pass per batch.
#
# This is an adapter over existing F4 stores, not a new source frontend. It
# retains full fragment/plan admission and only returns after EOF validation.
_GRAPHS = OrderedDict()


def _graph(store, unit, raw, report):
    """Reuse admitted typed graphs only while their original bytes still match."""
    _require(store.resolve() == store and type(unit) is str
        and Path(unit).name == unit and unit not in ('', '.', '..'), store, 'invalid store or source unit')
    key = str(store), unit, sha256(raw).hexdigest()
    graph = _GRAPHS.get(key)
    if graph is None:
        graph = read_template_graph(store, unit)
    else:
        _require(report['link_types_sha256'] == sha256(Path(T.__file__).read_bytes()).hexdigest()
            and report['generator_sha256'] == _implementation(), store, 'stored interface or generator declarations differ')
        entries = [row for row in report['units'] if row['unit'] == unit]
        _require(len(entries) == 1 and entries[0]['header'] == 'units/' + unit + '/header.json',
            unit, 'source header identity differs')
        row = entries[0]; path = store / row['header']
        _require(path.resolve() == path and path.is_relative_to(store), unit, 'unsafe unit header')
        header = path.read_bytes()
        _require(len(header) == row['header_bytes'] and sha256(header).hexdigest() == row['header_sha256'],
            unit, 'unit header receipt differs')
        header = json.loads(header)
        for name in ('templates', 'links'):
            descriptor = header[name]; fragment = path.parent / descriptor['path']
            _require(fragment.resolve() == fragment and fragment.parent == path.parent, path, 'unsafe graph table')
            payload = fragment.read_bytes()
            _require(len(payload) == descriptor['bytes'] and sha256(payload).hexdigest() == descriptor['sha256'],
                path, 'graph table compressed receipt differs')
    _require((store / 'result.json').read_bytes() == raw, unit, 'template graph root changed during admission')
    _GRAPHS[key] = graph
    _GRAPHS.move_to_end(key)
    while len(_GRAPHS) > 2:
        _GRAPHS.popitem(last=False)
    return graph


def read_template_bodies(store, unit, keys, *, max_visits=1_000_000):
    """Expand multiple original nodes without repeatedly scanning their unit.

    Typed graphs are reused only after checking their original byte receipts
    and current declarations. Parent proofs are checked before returning,
    every selected domain is exhausted once, and every selected plan receives
    its original structural/control-flow admission. No bodies are cached and
    no target Python is imported/executed.
    """
    keys = tuple(dict.fromkeys(keys))
    _require(all(type(key) is T.SourceTemplateKey for key in keys), unit, 'invalid template keys')
    if not keys:
        return {}
    store = Path(store).absolute()
    raw = (store / 'result.json').read_bytes()
    report = json.loads(raw)
    graph = _graph(store, unit, raw, report)
    nodes = {node.key: node for node in graph.templates}
    _require(all(key in nodes for key in keys), unit, 'selected template is absent')
    specifications = tuple((p['domain'], p['materials'], p['verification'],
        p['collection_sha256'], p['verification_sha256']) for p in report['parents'])
    parents = {parent.domain: parent for parent in _worker_stores(specifications, report['model_manifest_sha256'])}
    selected = defaultdict(lambda: defaultdict(list))
    for key in keys:
        node = nodes[key]
        selected[node.origin.domain][node.origin.line].append((key, node))
    result = {}
    for domain, lines in selected.items():
        material = parents[domain]
        parent = next(p for p in report['parents'] if p['domain'] == domain.value)
        _require(material.receipt == parent, unit, 'parent proof changed since interface linking')
        _check_catalog(material, unit)
        basis = material.basis[unit]
        # records() hashes the complete original fragment, including records
        # after the final selected line. Nothing escapes before that check.
        for _, line, document in material.records(dict(key=unit, **{k:v for k,v in basis.items() if k!='unit'})):
            if line not in lines:
                continue
            digest = _digest(document)
            _require(all(digest == node.origin.plan_sha256 for _, node in lines[line]),
                unit, 'original recipe body differs')
            plan = read_production_plan(document, max_visits=max_visits)
            for key, node in lines[line]:
                _require(plan.source == P.ProductionOccurrence(key.occurrence)
                    and plan.purpose is key.purpose and plan.root == node.root
                    and plan.input == graph.input and plan.profile == graph.profile,
                    key, 'original recipe identity/input/profile differs')
                result[key] = plan
    _require(set(result) == set(keys), unit, 'selected recipe is absent from its original domain')
    for parent in report['parents']:
        material = parents[T.SourceMaterialDomain(parent['domain'])]
        _require(material.receipt == parent, unit, 'parent proof changed since interface linking')
        _check_catalog(material, unit)
        _require(material.unchanged() and material.generator_pins == _implementation(),
            unit, 'original material changed during batch reading')
    _require((store / 'result.json').read_bytes() == raw, unit, 'template graph root changed during batch reading')
    return {key: result[key] for key in keys}


def _check_catalog(material, unit):
    # _worker_stores caches admitted catalogs, whereas unchanged() checks
    # reports and the current schema/compiler. Keep fresh on-disk catalog
    # admission as well; a cache hit must not conceal a changed catalog.
    raw = (material.root / 'catalog.json').read_bytes()
    _require(sha256(raw).hexdigest() == material.rows[unit]['catalog_sha256']
        and (material.verification / 'catalog.json').read_bytes() == raw,
        unit, 'parent catalog differs')


# Bounded parallel reads of existing, sealed source production plans.
#
# Tasks carry only store coordinates and the expected root receipt. Every task
# uses the original strict reader; no target module is imported or executed and
# no source operation, graph node, or port is created by this adapter.
class _NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ImportError('torch import is forbidden in static template workers')
        return None


def _check_no_torch():
    if any(name == 'torch' or name.startswith('torch.') for name in sys.modules):
        raise RuntimeError('static template reader process has imported torch')


@contextmanager
def _torch_guard():
    _check_no_torch()
    finder = _NoTorch()
    sys.meta_path.insert(0, finder)
    try:
        yield
        _check_no_torch()
    finally:
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)


def _initialize_worker():
    _check_no_torch()
    sys.meta_path.insert(0, _NoTorch())


def _check_root(root, expected):
    if sha256((Path(root) / 'result.json').read_bytes()).hexdigest() != expected:
        raise ValueError('sealed template root differs from the expected receipt')


def _read_batch(root, unit_id, keys, expected_root_sha256):
    """Spawn-safe task over trusted tool code and original source coordinates."""
    started, cpu = time.monotonic(), time.process_time()
    with _torch_guard():
        _check_root(root, expected_root_sha256)
        from scar.analysis.source_templates import read_template_bodies
        plans = read_template_bodies(root, unit_id, keys)
        _check_root(root, expected_root_sha256)
        if set(plans) != set(keys) or any(type(plan) is not P.SourceProductionPlan
                or plan.source != P.ProductionOccurrence(key.occurrence)
                or plan.purpose is not key.purpose for key,plan in plans.items()):
            raise ValueError('strict template reader returned another source plan')
    return plans, os.getpid(), time.monotonic() - started, time.process_time() - cpu


def _default_jobs():
    try:
        count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        count = getattr(os, 'process_cpu_count', os.cpu_count)() or 1
    return min(48, max(1, count))


class TemplatePlanPool:
    """Reusable spawn pool with an explicit, waiting close operation.

    One-key batches run in the caller. A multi-key batch has at most
    ``min(jobs, key_count)`` submitted tasks; spawn starts workers lazily, so
    small batches do not eagerly create the configured maximum process count.
    ``stats`` reports completed-task worker time, including strict validation.
    """
    def __init__(self, jobs=None):
        jobs = _default_jobs() if jobs is None else jobs
        if type(jobs) is not int or jobs < 1:
            raise ValueError('template jobs must be a positive exact integer')
        self.jobs = jobs
        self._executor = None
        self._closed = False
        self._lock = RLock()
        self._worker_pids = set()
        self._counts = dict(read_many_calls=0, tasks=0, completed_tasks=0, failed_tasks=0,
                            cancelled_tasks=0, serial_tasks=0, parallel_tasks=0,
                            failed_batches=0, peak_inflight=0,worker_batches=0,completed_worker_batches=0)
        self._wall_seconds = self._worker_seconds = self._worker_cpu_seconds = self._close_seconds = 0.0

    @property
    def stats(self):
        with self._lock:
            return dict(self._counts, jobs=self.jobs, plans_per_batch=8,parent_pid=os.getpid(),
                        worker_pids=sorted(self._worker_pids), start_method='spawn',
                        wall_seconds=self._wall_seconds, worker_seconds=self._worker_seconds,
                        worker_cpu_seconds=self._worker_cpu_seconds, close_seconds=self._close_seconds,
                        pool_created=self._executor is not None, closed=self._closed)

    def _accept(self, result, *, parallel):
        plans, pid, seconds, cpu_seconds = result
        if type(plans) is not dict or any(type(plan) is not P.SourceProductionPlan for plan in plans.values()):
            raise ValueError('template worker did not return a typed source plan')
        self._counts['completed_tasks'] += len(plans)
        self._counts['completed_worker_batches'] += 1
        self._worker_seconds += seconds
        self._worker_cpu_seconds += cpu_seconds
        if parallel:
            if pid == os.getpid():
                raise RuntimeError('parallel template task ran in the caller process')
            self._worker_pids.add(pid)
        return plans

    def read_many(self, root, unit_id, keys, expected_root_sha256):
        with self._lock:
            if self._closed:
                raise RuntimeError('template plan pool is closed')
            if type(unit_id) is not str or not unit_id:
                raise ValueError('template reader requires an original unit ID')
            if (type(expected_root_sha256) is not str
                    or re.fullmatch(r'[0-9a-f]{64}', expected_root_sha256) is None):
                raise ValueError('template root SHA must be lowercase SHA256')
            keys = tuple(keys)
            if any(type(key) is not SourceTemplateKey for key in keys):
                raise TypeError('template reader requires original SourceTemplateKey records')
            keys = tuple(dict.fromkeys(keys))
            root = str(Path(root).absolute())
            started = time.monotonic()
            self._counts['read_many_calls'] += 1
            try:
                _check_no_torch()
                _check_root(root, expected_root_sha256)
                if not keys:
                    result = {}
                elif len(keys) == 1 or self.jobs == 1:
                    result = {}
                    self._counts['tasks'] += len(keys)
                    self._counts['serial_tasks'] += len(keys)
                    self._counts['worker_batches'] += 1
                    try:
                        result = self._accept(_read_batch(root, unit_id, keys, expected_root_sha256), parallel=False)
                    except BaseException:
                        self._counts['failed_tasks'] += len(keys)
                        raise
                else:
                    result = self._read_parallel(root, unit_id, keys, expected_root_sha256)
                _check_root(root, expected_root_sha256)
                _check_no_torch()
                return {key: result[key] for key in keys}
            except BaseException:
                self._counts['failed_batches'] += 1
                raise
            finally:
                self._wall_seconds += time.monotonic() - started

    def _read_parallel(self, root, unit_id, keys, expected):
        if self._executor is None:
            self._executor = ProcessPoolExecutor(max_workers=self.jobs,
                mp_context=multiprocessing.get_context('spawn'), initializer=_initialize_worker)
        remaining = iter(tuple(keys[i:i+8]) for i in range(0,len(keys),8))
        pending, results = {}, {}
        window = min(self.jobs, (len(keys)+7)//8)

        def submit():
            batch = next(remaining, None)
            if batch is None:
                return
            future = self._executor.submit(_read_batch, root, unit_id, batch, expected)
            pending[future] = batch
            self._counts['tasks'] += len(batch)
            self._counts['parallel_tasks'] += len(batch)
            self._counts['worker_batches'] += 1
            self._counts['peak_inflight'] = max(self._counts['peak_inflight'], len(pending))

        try:
            for _ in range(window):
                submit()
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    batch = pending.pop(future)
                    try:
                        received = self._accept(future.result(), parallel=True)
                        if set(received)!=set(batch):raise ValueError('template batch selection differs')
                        results.update(received)
                    except BaseException:
                        self._counts['failed_tasks'] += len(batch)
                        raise
                    submit()
        except BaseException:
            # Drain running tasks before propagating the first failure. No
            # partial dictionary escapes and no old batch remains in flight.
            for future in pending:
                future.cancel()
            for future in pending:
                if future.cancelled():
                    self._counts['cancelled_tasks'] += len(pending[future])
                    continue
                try:
                    self._accept(future.result(), parallel=True)
                except BaseException:
                    self._counts['failed_tasks'] += len(pending[future])
            raise
        return results

    def close(self):
        with self._lock:
            if self._closed:
                return
            started = time.monotonic()
            self._closed = True
            try:
                if self._executor is not None:
                    self._executor.shutdown(wait=True, cancel_futures=True)
            finally:
                self._close_seconds += time.monotonic() - started

    def __enter__(self):
        if self._closed:
            raise RuntimeError('template plan pool is closed')
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


__all__ = [
    'TemplatePlanPool',
    'read_template_bodies',
    'TemplateLinkError',
    'validate_state_sequence',
    'build_template_graph',
    'link_source_templates',
    'read_template_graph',
    'verify_template_store',
    'read_template_body',
    'main',
]


if __name__ == '__main__':
    main()
