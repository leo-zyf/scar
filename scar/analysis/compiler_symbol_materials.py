"""Compiler symbol materials."""

from __future__ import annotations

import argparse
import math
import multiprocessing
import re
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from scar.analysis import compiler_topology_materials as _t, source_binding_materials as _bd
from scar.analysis.source_materials import _stamps
from scar.analysis.source_stage import SourceStage
from scar.analysis.source_store import (
    REQUIRED_SOURCE_ARTIFACTS,
    source_input_basis,
    read_source_prerequisites,
    check_replay_limit,
    read_material_manifest,
    pin_material_store,
    check_material_store,
    material_store_members,
)
from scar.analysis.static_program import _json, _write
from scar.analysis.static.query import _Fragments
from dataclasses import replace
from scar.analysis.compiler_topology import _Builder, _FUNCTIONLIKE, TopologyStop
from scar.analysis.source_declaration_view import SourceDeclarationView
from scar.ir.program.source_roles import SlotRef
from scar.ir.program.compiler_symbols import (
    INTRINSIC_NAMES,
    CompilerName,
    SymbolLimits,
    SymbolStop,
    RawCollection,
    RawContribution,
    RawSlot,
    RawParameters,
    SourceDisposition,
)


# Owned C3-S material transactions over the original eight prerequisite views.
_d, _a = _t._d, _t._a
_ARTIFACT = 'compiler_symbols'
_CATALOG = _ARTIFACT + '/catalog.json.gz'
_OWN_DEFAULTS = dict(max_symbol_slots=2_000_000, max_compiler_contributions=4_000_000,
    max_symbol_facts=4_000_000, max_symbol_transforms=2_000_000, max_symbol_steps=20_000_000)
_LIMIT_KEYS = dict(max_symbol_slots='max_symbol_slots', max_compiler_contributions='max_contributions',
    max_symbol_facts='max_facts', max_symbol_transforms='max_transforms', max_symbol_steps='max_steps')
_DEFAULTS = {**_t._DEFAULTS, **{key: _d._DEFAULTS[key] for key in _d._OWN_LIMITS}, **_OWN_DEFAULTS}
_TABLES = ('slots', 'contributions', 'sources', 'facts', 'transforms', 'protocols', 'blocks', 'symbols', 'parameters')
_TOTALS = (*_TABLES, 'steps')
_FLAGS = dict(authority='SOURCE_COMPILER_SYMBOL_PROVENANCE', runtime_binding='NOT_OBSERVED',
    runtime_local_layout='NOT_CLAIMED')


class _TopologyPolicy:
    """Adapt T's scalar totals to the existing prerequisite receipt checker."""
    _catalog_at = staticmethod(_t._catalog_at)
    _inventory = staticmethod(_t._inventory)

    @staticmethod
    def _check_summary(document, rows, counts, digest):
        _t._check_summary(document, rows, digest)

    @staticmethod
    def _report(rows, counts, *, schema, limits, jobs, catalog_sha):
        return _t._report(rows, schema=schema, limits=limits, jobs=jobs, digest=catalog_sha)


def _topology_input(stage, directory, basis, catalog):
    if directory is None:
        return _d._input(stage, 'compiler_topology', None, basis, catalog, _TopologyPolicy)
    root, stored, raw = read_material_manifest(directory, stage.out)
    _t._check_summary(stored, stored['results'], catalog['content_sha256'])
    descriptors = [row['artifact'] for row in stored['results']]
    if (_json(stored.get('basis')) != _json(basis) or len(descriptors) != len(basis)
            or any(row['unit'] != unit['key'] or row['artifact']['path'] != 'units/' + unit['key'] + '/compiler_topology.json.gz'
                for row, unit in zip(stored['results'], basis))):
        raise ValueError('C3-T original prerequisite basis/unit order differs from selected inputs')
    descriptor = stored['catalog']
    pins = pin_material_store(root, [descriptor, *descriptors], raw)
    _t._catalog_at(root, descriptor, catalog)
    return dict(root=root, stored=stored, raw=raw, pins=pins, catalog=descriptor,
        catalog_pins=_stamps(root, (descriptor,)), descriptors=descriptors,
        policy=_TopologyPolicy, catalog_sha=catalog['content_sha256'], artifact='compiler_topology')


def _upstream_rows(unit, f1, f2, roles, contexts, declarations, symbols, topology, descriptors):
    b, c, d = _bd._upstream_rows(unit, f1, f2, roles, contexts, declarations, descriptors[:3])
    a = _a._row(unit['key'], descriptors[3], symbols, f1, f2)
    t = dict(unit=unit['key'], artifact=descriptors[4], status=topology['status'],
        **{key: len(topology[key]) for key in ('blocks', 'scopes', 'rewrites')}, visit_steps=topology['visit_steps'],
        **{key: sum(block['projection'] == value for block in topology['blocks']) for key, value in _t._PROJECTIONS.items()},
        **{key: topology[key] for key in _t._OWN_DEFAULTS}, frontiers=topology['frontiers'],
        upstream_f1_status=f1['status'], upstream_f2_status=f2['status'], upstream_f3b_status=roles['status'],
        upstream_c1_status=contexts['status'], upstream_f3a_status=symbols['status'])
    return b, roles['counts'], c, contexts['counts'], d, a, t


def _symbol_unit(task):
    from scar.analysis.compiler_symbols import compiler_symbols_document, read_compiler_symbols
    from scar.analysis.source_symbols import compiler_hash_context
    model, destination, unit, inputs, prior_root, prior, limits, digest, independent, expected_context = task
    if compiler_hash_context() != expected_context:
        raise ValueError('compiler-symbols worker hash context differs; start analysis with fixed PYTHONHASHSEED')
    f0, f1, f2 = read_source_prerequisites(_Fragments(model), unit, limits['max_nodes'], limits['max_tokens'])
    roles, contexts, declarations, symbols, topology = [_Fragments(root).read(descriptor) for root, descriptor in inputs]
    for doc, keys, label in ((roles, ('max_ports', 'max_slots'), 'F3-B'),
            (contexts, _d._c1._OWN_LIMITS, 'F3-C1'), (declarations, _d._OWN_LIMITS, 'F3-C2a'),
            (symbols, ('max_scopes', 'max_symbols'), 'F3-A'), (topology, _t._OWN_DEFAULTS, 'C3-T')):
        for key in keys:check_replay_limit(doc, key, limits[key], label)
    documents = (f0, f1, f2, roles, contexts, declarations, symbols, topology)
    if prior is not None:
        document = _Fragments(prior_root).read(prior)
        for key, name in _LIMIT_KEYS.items():check_replay_limit(document['limits'], name, limits[key], 'C3-S')
        read_compiler_symbols(document, *documents)
        descriptor = prior
    elif destination is None:
        raise ValueError('compiler symbol artifact is absent')
    else:
        document = compiler_symbols_document(*documents, **{key: limits[key] for key in _OWN_DEFAULTS})
        descriptor = None
    if document['profile']['symbol_catalog_sha256'] != digest:
        raise ValueError('compiler symbol worker catalog differs from its parent')
    if destination is not None and (independent or descriptor is None):
        relative = 'units/' + unit['key'] + '/compiler_symbols.json.gz'
        if (destination / relative).exists():raise FileExistsError('new compiler symbol artifact collides with an existing file')
        descriptor = _write(destination, relative, document)
    row = dict(unit=unit['key'], artifact=descriptor, status=document['status'],
        **{key: len(document[key]) for key in _TABLES}, steps=document['steps'],
        **{key: document['limits'][name] for key, name in _LIMIT_KEYS.items()}, frontiers=document['frontiers'],
        upstream_f1_status=f1['status'], upstream_f2_status=f2['status'], upstream_f3b_status=roles['status'],
        upstream_c1_status=contexts['status'], upstream_c2a_status=declarations['status'],
        upstream_f3a_status=symbols['status'], upstream_topology_status=topology['status'])
    return (row, *_upstream_rows(unit, f1, f2, roles, contexts, declarations, symbols, topology, [item[1] for item in inputs]))


def _report(rows, *, schema, limits, jobs, digest):
    covered = [row['status'] not in ('BUDGET_STOPPED', 'PROJECTION_CONTRADICTION')
        and row['upstream_f1_status'] != 'BUDGET_STOPPED'
        and row['upstream_f2_status'] not in ('BUDGET_STOPPED', 'PARSER_RESOURCE_ERROR')
        and all(row[key] != 'BUDGET_STOPPED' for key in ('upstream_f3b_status', 'upstream_c1_status', 'upstream_c2a_status'))
        and row['upstream_f3a_status'] not in ('BUDGET_STOPPED', 'RESOURCE_ERROR')
        and row['upstream_topology_status'] not in ('BUDGET_STOPPED', 'PROJECTION_CONTRADICTION') for row in rows]
    return dict(schema=schema, schema_version=1, stage='F3-C3S', input_scope='DECLARED_MODEL_SOURCE_UNITS',
        units=len(rows), covered_units=sum(covered), complete_for_declared_units=all(covered),
        all_declared_units_projected=all(row['status'] == 'PROJECTED' for row in rows),
        status_counts=dict(Counter(row['status'] for row in rows)), results=rows,
        **{key: sum(row[key] for row in rows) for key in _TOTALS}, limits=limits, jobs=jobs, catalog_sha256=digest,
        symtable_executed=any(row['upstream_f3a_status'] not in ('UPSTREAM_NOT_PARSED', 'UNSUPPORTED_SYMTABLE_MODE') for row in rows),
        target_executed=False, later_stages_verified=False, whole_program_complete=False, **_FLAGS)


def _check_summary(document, rows, digest):
    from scar.analysis.source_symbols import compiler_hash_context
    limits, jobs = document.get('limits'), document.get('jobs')
    if (type(limits) is not dict or set(limits) != set(_DEFAULTS)
            or any(type(value) is not int or value < 0 for value in limits.values())
            or type(jobs) is not int or not 1 <= jobs <= 32):
        raise ValueError('invalid compiler symbol material budgets/jobs')
    for row in rows:
        for key in _OWN_DEFAULTS:
            if type(row.get(key)) is not int or not 0 <= row[key] <= limits[key]:
                raise ValueError('compiler symbol row budget exceeds stored limits')
    expected = _report(rows, schema='scar.compiler-symbol-materials', limits=limits, jobs=jobs, digest=digest)
    if (_json({key: document.get(key) for key in expected}) != _json(expected)
            or document.get('compiler_hash_context') != compiler_hash_context()
            or document.get('collect_only') is not True or document.get('canonical_model_published') is not False
            or document.get('manifest_unchanged') is not True):
        raise ValueError('compiler symbol material summary differs from freshly verified rows/hash context')
    for key in ('observed_manifest_sha256', 'prior_manifest_sha256', 'published_manifest_sha256'):
        if type(document.get(key)) is not str or re.fullmatch(r'[0-9a-f]{64}', document[key]) is None:
            raise ValueError('invalid observed compiler symbol manifest receipt')
    if 'seconds' in document and (type(document['seconds']) not in (int, float)
            or not math.isfinite(document['seconds']) or document['seconds'] < 0):
        raise ValueError('invalid compiler symbol observed duration')


def _catalog_at(root, descriptor, expected):
    if descriptor.get('path') != _CATALOG or _json(_Fragments(root).read(descriptor)) != _json(expected):
        raise ValueError('compiler symbol catalog receipt/content differs from the current profile')


def _inventory(report, catalog):
    keys = ('stage', 'units', *_TOTALS, 'status_counts', 'catalog_sha256', *_FLAGS,
        'later_stages_verified', 'whole_program_complete')
    return dict({key: report[key] for key in keys}, schema='scar.compiler-symbol-inventory', schema_version=1,
        scope='DECLARED_MODEL_SOURCE_UNITS', artifacts=[catalog])


def model_compiler_symbols(model, out, *, roles_materials=None, contexts_materials=None, declarations_materials=None,
        symbols_materials=None, topology_materials=None, jobs=4, collect_only=False, verify=False, materials=None,
        max_nodes=1_000_000, max_tokens=2_000_000, max_ports=1_000_000, max_slots=10_000_000,
        max_scopes=100_000, max_regions=200_000, max_boundaries=1_000_000, max_symbols=1_000_000,
        max_names=1_000_000, max_contributions=1_000_000, max_conflicts=100_000,
        max_blocks=100_000, max_rewrites=200_000, max_visit_steps=4_000_000,
        max_symbol_slots=2_000_000, max_compiler_contributions=4_000_000, max_symbol_facts=4_000_000,
        max_symbol_transforms=2_000_000, max_symbol_steps=20_000_000):
    from scar.analysis.compiler_symbols import compiler_symbol_catalog
    from scar.analysis.compiler_topology import compiler_topology_catalog
    from scar.analysis.source_declarations import source_declaration_catalog
    from scar.analysis.source_contexts import source_context_catalog
    from scar.analysis.source_role_materials import source_role_catalog
    from scar.analysis.source_symbols import compiler_hash_context
    limits = {key: value for key, value in locals().items() if key in _DEFAULTS}
    external = (roles_materials, contexts_materials, declarations_materials, symbols_materials, topology_materials)
    if (type(jobs) is not int or not 1 <= jobs <= 32 or type(collect_only) is not bool or type(verify) is not bool
            or collect_only and verify or materials is not None and not verify
            or any(value is not None for value in external) and not (collect_only or verify)
            or any(type(value) is not int or value < 0 for value in limits.values())):
        raise ValueError('compiler symbols require bounded distinct modes; external prerequisite capture is collect-only')
    started = time.monotonic();expected_context = compiler_hash_context()
    catalog = compiler_symbol_catalog();digest = catalog['content_sha256']
    with SourceStage(model, out, artifact=_ARTIFACT, verify=verify or collect_only,
            consumes=(*REQUIRED_SOURCE_ARTIFACTS, 'source_roles', 'source_contexts', 'source_declarations',
                'source_symbols', 'compiler_topology', _ARTIFACT)) as stage:
        base = source_input_basis(stage.manifest['units'])
        b_catalog, c_catalog = source_role_catalog(), source_context_catalog()
        b = _d._input(stage, 'source_roles', roles_materials, base, b_catalog, _d._b)
        basis = _d._with_artifacts(base, 'source_roles', b['descriptors'])
        c = _d._input(stage, 'source_contexts', contexts_materials, basis, c_catalog, _d._c1)
        contexts_basis = _d._with_artifacts(basis, 'source_contexts', c['descriptors'])
        d = _bd._declaration_input(stage, declarations_materials, contexts_basis, source_declaration_catalog())
        a = _t._symbols_input(stage, symbols_materials, base)
        topology_basis = _d._with_artifacts(contexts_basis, 'source_symbols', a['descriptors'])
        t = _topology_input(stage, topology_materials, topology_basis, compiler_topology_catalog())
        basis = _d._with_artifacts(_d._with_artifacts(topology_basis, 'source_declarations', d['descriptors']), 'compiler_topology', t['descriptors'])
        existing, canonical_catalog = stage.manifest.get(_ARTIFACT), None
        if existing is not None:
            if type(existing.get('artifacts')) is not list or len(existing['artifacts']) != 1:
                raise ValueError('canonical compiler symbols inventory must own one shared catalog')
            canonical_catalog = existing['artifacts'][0];_catalog_at(stage.root, canonical_catalog, catalog)
        canonical_pins = _stamps(stage.root, (canonical_catalog,)) if canonical_catalog else None
        stored = store_root = store_raw = store_pins = None
        if materials is not None:
            store_root, stored, store_raw = read_material_manifest(materials, stage.out)
            _check_summary(stored, stored['results'], digest)
            if _json(stored.get('basis')) != _json(basis) or len(stored['results']) != len(basis):
                raise ValueError('compiler symbol material basis differs from selected original sources')
            if any(row['unit'] != unit['key'] or row['artifact']['path'] != stage.relative(unit)
                    for row, unit in zip(stored['results'], basis)):
                raise ValueError('compiler symbol original unit/path order differs from the model')
            _catalog_at(store_root, stored['catalog'], catalog)
            store_pins = pin_material_store(store_root, [stored['catalog'], *(row['artifact'] for row in stored['results'])], store_raw)
        destination = stage.out if collect_only else stage.staging
        if destination is not None and (collect_only or canonical_catalog is None):
            if (destination / _CATALOG).exists():raise FileExistsError('new compiler symbol catalog collides with an existing file')
            descriptor = _write(destination, _CATALOG, catalog)
        else:
            descriptor = stored['catalog'] if stored else canonical_catalog
            if descriptor is None:raise ValueError('canonical compiler symbol catalog is absent')
        replay_limits = {key: min(value, stored['limits'][key]) for key, value in limits.items()} if stored else limits
        tasks = [(stage.root, destination, unit, [(item['root'], item['descriptors'][i]) for item in (b,c,d,a,t)],
            store_root or stage.root, stored['results'][i]['artifact'] if stored else stage.manifest['units'][i]['artifacts'].get(_ARTIFACT),
            replay_limits, digest, collect_only, expected_context) for i, unit in enumerate(basis)]
        rows,b_rows,c_rows,d_rows,a_rows,t_rows=[],[],[],[],[],[]
        b_counts=[dict(rule=rule['rule'], **{key:0 for key in _d._b._COUNT_FIELDS}) for rule in b_catalog['rules']]
        c_counts=[dict(rule=rule['rule'], **{key:0 for key in _d._c1._COUNT_FIELDS}) for rule in c_catalog['rules']]
        def accept(packet):
            row,b_row,b_current,c_row,c_current,d_row,a_row,t_row=packet
            rows.append(row);b_rows.append(b_row);c_rows.append(c_row);d_rows.append(d_row);a_rows.append(a_row);t_rows.append(t_row)
            for total,current,fields in ((b_counts,b_current,_d._b._COUNT_FIELDS),(c_counts,c_current,_d._c1._COUNT_FIELDS)):
                if len(total)!=len(current):raise ValueError('compiler symbol prerequisite counter denominator is incomplete')
                for left,right in zip(total,current):
                    if left['rule']!=right['rule']:raise ValueError('compiler symbol prerequisite rule order changed')
                    for key in fields:left[key]+=right[key]
        if jobs==1:
            for packet in map(_symbol_unit,tasks):accept(packet)
        else:
            with ProcessPoolExecutor(max_workers=jobs,mp_context=multiprocessing.get_context('spawn')) as pool:
                for packet in pool.map(_symbol_unit,tasks,chunksize=1):accept(packet)
        _d._check_input(stage,b,b_rows,b_counts,jobs);_d._check_input(stage,c,c_rows,c_counts,jobs)
        _d._check_input(stage,d,d_rows,None,jobs);_t._check_symbols(stage,a,a_rows,jobs)
        _d._check_input(stage,t,t_rows,None,jobs)
        if stored is not None:
            _check_summary(stored,rows,digest)
            check_material_store(store_root,[stored['catalog'],*(row['artifact'] for row in rows)],store_raw,store_pins)
        report=_report(rows,schema=('scar.compiler-symbol-material-verification' if stored else
            'scar.compiler-symbol-verification' if verify else 'scar.compiler-symbol-materials' if collect_only
            else 'scar.compiler-symbol-gate'),limits=limits,jobs=jobs,digest=digest)
        report.update(catalog=descriptor,collect_only=collect_only,canonical_model_published=False,
            observed_manifest_sha256=sha256(stage.original).hexdigest(),compiler_hash_context=expected_context)
        if collect_only:
            report['basis']=basis;material_store_members(stage.out,[descriptor,*(row['artifact'] for row in rows)],result=False)
        elif not verify and report['complete_for_declared_units']:
            stage.publish(rows,_inventory(report,descriptor));report['canonical_model_published']=True
        elif verify and stored is None and _json(existing)!=_json(_inventory(report,descriptor)):
            raise ValueError('canonical compiler symbol inventory differs from freshly verified provenance')
        if canonical_catalog and _stamps(stage.root,(canonical_catalog,))!=canonical_pins:
            raise ValueError('canonical compiler symbol catalog changed before publication')
        if stored is not None:
            report.update(materials=str(store_root),materials_result_sha256=sha256(store_raw).hexdigest(),material_source_basis_unchanged=True)
        report['seconds']=time.monotonic()-started
        return stage.finish(report)


def main(argv=None):
    parser=argparse.ArgumentParser(prog='scar compiler-symbols')
    for key in ('model','out'):parser.add_argument('--'+key,required=True)
    for key in ('roles-materials','contexts-materials','declarations-materials','symbols-materials','topology-materials','materials'):
        parser.add_argument('--'+key)
    parser.add_argument('--jobs',type=int,default=4)
    parser.add_argument('--collect-only',action='store_true');parser.add_argument('--verify',action='store_true')
    for key,value in _DEFAULTS.items():parser.add_argument('--'+key.replace('_','-'),type=int,default=value)
    report=model_compiler_symbols(**vars(parser.parse_args(argv)))
    print(_json({key:report[key] for key in ('stage','units','symbols','parameters','status_counts',
        'complete_for_declared_units','canonical_model_published')}).decode().strip())
    return 0 if report['complete_for_declared_units'] else 1


# C3-S first-pass scalar symbol events over the accepted C3-T visitor.
#
# Native symbols/varnames are never read here. Native metadata is used only by
# C3-T's existing topology replay. Source declarations supply canonical keys and
# provenance, not the flags emitted by this visitor.
RAW_RULES = (
    'SOURCE_BINDER', 'SOURCE_PARAMETER', 'TYPE_PARAMETER', 'NAME_LOAD', 'NAME_STORE',
    'NAME_DELETE', 'ANNOTATED_TARGET', 'PATTERN_BINDER', 'EXCEPT_BINDER', 'IMPORT_BINDER',
    'GLOBAL_DIRECTIVE', 'NONLOCAL_DIRECTIVE', 'WALRUS_ORIGIN_DIRECTIVE', 'WALRUS_HOME_BINDING',
    'GLOBAL_ECHO', 'FORMAT_PARAMETER', 'FORMAT_USE', 'COMPREHENSION_PARAMETER',
    'GENERIC_DEFAULTS_PARAMETER', 'GENERIC_KWDEFAULTS_PARAMETER', 'TYPE_PARAMS_LOCAL',
    'TYPE_PARAMS_USE', 'GENERIC_BASE_LOCAL', 'GENERIC_BASE_USE', 'CLASS_TYPE_PARAMS_BINDING',
    'CLASS_TYPE_PARAMS_USE', 'SUPER_CLASS_USE', 'CLASSDICT_USE', 'CONDITIONAL_ANNOTATION_USE',
)
GLOBAL, LOCAL, PARAM, NONLOCAL, USE = 1, 2, 4, 8, 16
IMPORT, ANNOT, COMP_ITER, TYPE_PARAM = 128, 256, 512, 1024
_CONDITIONAL_FIELDS = {
    'For': frozenset(('body', 'orelse')), 'AsyncFor': frozenset(('body', 'orelse')),
    'While': frozenset(('body', 'orelse')), 'If': frozenset(('body', 'orelse')),
    'Match': frozenset(('cases',)), 'Try': frozenset(('body', 'handlers', 'orelse', 'finalbody')),
    'TryStar': frozenset(('body', 'handlers', 'orelse', 'finalbody')),
    'With': frozenset(('items', 'body')), 'AsyncWith': frozenset(('items', 'body')),
}


class _Collector(_Builder):
    def __init__(self, syntax, roles, contexts, declarations, native, topology, limits):
        self.limits_symbols, self.accepted = limits, topology
        self.symbol_steps = 0
        super().__init__(syntax, roles, contexts, native,
            (max(1, len(topology.blocks)), max(1, len(topology.rewrites)), limits.max_steps))
        SourceDeclarationView(roles, contexts, declarations, contexts=self.view).verify()
        self.declarations = declarations
        self.source_by_site = {}
        for index, row in enumerate(declarations.contributions):
            if row.site in self.source_by_site:
                raise SymbolStop('DUPLICATE_SOURCE_CONTRIBUTION_SITE', 'C2a contributions repeat an original source site', source=index)
            self.source_by_site[row.site] = index
        self.pool = {row.value: index for index, row in enumerate(declarations.names)}
        self.names = {}
        self.raw_slots, self.slot_index, self.events = [], {}, []
        self.source_events = [[] for _ in declarations.contributions]
        self.ignored = set()
        self.param_slots, self.formals_emitted = {}, set()
        self.target_mode, self.iter_expression, self.unevaluated, self.conditional = {}, {}, {}, {}
        self.has_conditional = set()

    def step(self, *, block=None, source=None):
        if self.symbol_steps >= self.limits_symbols.max_steps:
            raise SymbolStop('SYMBOL_STEP_BUDGET', 'symbol visitor budget reached', block=block, source=source, budget=True)
        self.symbol_steps += 1

    def tick(self, occurrence, role=None):
        self.step(block=self.current)
        super().tick(occurrence, role)

    def site(self, occurrence, name, item_path=()):
        return SlotRef(occurrence, 'FIELD', self.fields.fields[self.syntax.nodes[occurrence].constructor][name], item_path)

    def source(self, occurrence, name, item_path=(), *, required=True):
        site = self.site(occurrence, name, item_path)
        index = self.source_by_site.get(site)
        if index is None and required:
            raise SymbolStop('SOURCE_CONTRIBUTION_MISSING', 'visitor name lacks its original C2a contribution', block=self.current)
        return index, site

    def name(self, text):
        if text not in self.names:
            if text in self.pool:
                self.names[text] = CompilerName('SOURCE', source=self.pool[text])
            elif text in INTRINSIC_NAMES:
                self.names[text] = CompilerName('INTRINSIC', intrinsic=text)
            else:
                raise SymbolStop('COMPILER_NAME_SOURCE_MISSING', 'non-intrinsic compiler name is absent from the source pool', block=self.current)
        return self.names[text]

    def source_name(self, source):
        return self.declarations.names[self.declarations.contributions[source].name].value

    def slot(self, block, text):
        name = self.name(text)
        key = block, name
        if key not in self.slot_index:
            if len(self.raw_slots) >= self.limits_symbols.max_symbol_slots:
                raise SymbolStop('SYMBOL_SLOT_BUDGET', 'raw compiler slot budget reached', block=block, budget=True)
            self.slot_index[key] = len(self.raw_slots)
            self.raw_slots.append([block, name, 0, []])
        return self.slot_index[key]

    def emit(self, text, flags, rule, *, source=None, anchor=None, target=None, context=None,
             derived_from=None, echo=True):
        context = self.current if context is None else context
        target = context if target is None else target
        self.step(block=context, source=source)
        if rule not in RAW_RULES:
            raise ValueError('compiler collector used an undeclared raw rule')
        if len(self.events) >= self.limits_symbols.max_contributions:
            raise SymbolStop('COMPILER_CONTRIBUTION_BUDGET', 'raw compiler contribution budget reached',
                block=context, source=source, budget=True)
        slot = self.slot(target, text)
        old = self.raw_slots[slot][2]
        if flags & PARAM and old & PARAM or flags & TYPE_PARAM and old & TYPE_PARAM:
            raise SymbolStop('DUPLICATE_COMPILER_PARAMETER', 'first-pass parameter declaration is repeated', block=target, source=source)
        effective = flags
        if self.target_mode[context] and rule != 'GLOBAL_ECHO':
            if (old | flags) & (GLOBAL | NONLOCAL):
                raise SymbolStop('COMPREHENSION_TARGET_DIRECTIVE_CONFLICT', 'comprehension target conflicts with a directive', block=target, source=source)
            effective |= COMP_ITER
        event = len(self.events)
        self.events.append(RawContribution(context, slot, effective, rule, source, anchor, derived_from))
        self.raw_slots[slot][2] |= effective
        self.raw_slots[slot][3].append(event)
        if source is not None:
            self.source_events[source].append(event)
        if flags & PARAM:
            self.param_slots[target].append(slot)
        elif flags & GLOBAL and echo:
            # The helper mirrors the original flag, not target-mode decoration,
            # into st_global immediately. The root self-echo is idempotent but
            # retained as explicit causal provenance.
            self.emit(text, flags, 'GLOBAL_ECHO', source=source, anchor=anchor, target=0,
                context=context, derived_from=event, echo=False)
        return event

    def emit_source(self, occurrence, name, flags, rule, item_path=(), *, target=None):
        source, anchor = self.source(occurrence, name, item_path)
        return self.emit(self.source_name(source), flags, rule, source=source, anchor=anchor, target=target)

    def block(self, occurrence, role, *, can_see=False, attach=True):
        index = super().block(occurrence, role, can_see=can_see, attach=attach)
        parent = self.blocks[index].parent
        self.target_mode[index] = False
        self.iter_expression[index] = 0 if parent is None else self.iter_expression[parent]
        self.unevaluated[index] = False
        self.conditional[index] = False
        self.param_slots[index] = []
        origin = self.blocks[index].origin
        if role in ('HEADER_ANNOTATIONS', 'VARIABLE_ANNOTATIONS', 'TYPE_PARAMETER_BOUND',
                    'TYPE_PARAMETER_DEFAULT', 'TYPE_ALIAS'):
            field = ('args' if role == 'HEADER_ANNOTATIONS' else 'bound' if role == 'TYPE_PARAMETER_BOUND'
                else 'default_value' if role == 'TYPE_PARAMETER_DEFAULT' else 'value' if role == 'TYPE_ALIAS' else None)
            anchor = (self.site(occurrence, field) if field is not None else
                self.contexts.scopes[self.blocks[index].source_scope].anchor)
            self.emit('.format', PARAM, 'FORMAT_PARAMETER', context=index, anchor=anchor)
            self.emit('.format', USE, 'FORMAT_USE', context=index, anchor=anchor)
            if can_see:
                self.emit('__classdict__', USE, 'CLASSDICT_USE', context=index, anchor=anchor)
        elif role == 'TYPE_PARAMETERS':
            anchor = self.site(occurrence, 'type_params')
            if can_see:
                self.emit('__classdict__', USE, 'CLASSDICT_USE', context=index, anchor=anchor)
            kind = self.syntax.nodes[occurrence].constructor
            if kind == 'ClassDef':
                for text, local_rule, use_rule in (('.type_params', 'TYPE_PARAMS_LOCAL', 'TYPE_PARAMS_USE'),
                        ('.generic_base', 'GENERIC_BASE_LOCAL', 'GENERIC_BASE_USE')):
                    self.emit(text, LOCAL, local_rule, context=index, anchor=anchor)
                    self.emit(text, USE, use_rule, context=index, anchor=anchor)
            elif kind in ('FunctionDef', 'AsyncFunctionDef'):
                arguments = self.field(occurrence, 'args').reference
                # The pinned source parser supplies a non-NULL defaults ASDL
                # sequence even for []; official code tests NULL, not length.
                self.emit('.defaults', PARAM, 'GENERIC_DEFAULTS_PARAMETER', context=index,
                    anchor=self.site(arguments, 'defaults'))
                if any(value.kind == 'NODE' for value in self.field(arguments, 'kw_defaults').items):
                    self.emit('.kwdefaults', PARAM, 'GENERIC_KWDEFAULTS_PARAMETER', context=index,
                        anchor=self.site(arguments, 'kw_defaults'))
        elif role == 'CLASS_BODY':
            if self.field(occurrence, 'type_params').items:
                anchor = self.site(occurrence, 'type_params')
                self.emit('__type_params__', LOCAL, 'CLASS_TYPE_PARAMS_BINDING', context=index, anchor=anchor)
                self.emit('.type_params', USE, 'CLASS_TYPE_PARAMS_USE', context=index, anchor=anchor)
        elif role in ('COMPREHENSION', 'GENERATOR_EXPRESSION'):
            self.emit('.0', PARAM, 'COMPREHENSION_PARAMETER', context=index, anchor=self.site(occurrence, 'generators', (0,)))
        return index

    def enter(self, block, fields):
        role = self.blocks[block].origin.role
        if role in ('FUNCTION_BODY', 'LAMBDA_BODY') and block not in self.formals_emitted:
            self.formals_emitted.add(block)
            occurrence = self.blocks[block].origin.occurrence
            arguments = self.field(occurrence, 'args').reference
            previous, self.current = self.current, block
            try:
                for name in ('posonlyargs', 'args', 'kwonlyargs', 'vararg', 'kwarg'):
                    value = self.field(arguments, name)
                    members = value.items if value.kind == 'LIST' else (value,)
                    for item in members:
                        if item.kind == 'NODE':
                            self.emit_source(item.reference, 'arg', PARAM, 'SOURCE_PARAMETER')
            finally:
                self.current = previous
        return super().enter(block, fields)

    def visit_field(self, occurrence, name):
        if self.syntax.nodes[occurrence].constructor == 'comprehension' and name in ('target', 'iter'):
            block = self.current
            if name == 'target':
                old, self.target_mode[block] = self.target_mode[block], True
                try:
                    return super().visit_field(occurrence, name)
                finally:
                    self.target_mode[block] = old
            self.iter_expression[block] += 1
            try:
                return super().visit_field(occurrence, name)
            finally:
                self.iter_expression[block] -= 1
        return super().visit_field(occurrence, name)

    def variable_annotation(self, occurrence):
        target = self.field(occurrence, 'target').reference
        value = self.field(occurrence, 'value')
        if self.syntax.nodes[target].constructor == 'Name':
            simple = self.field(occurrence, 'simple').scalar.to_python()
            if simple:
                self.emit_source(target, 'id', LOCAL | ANNOT, 'ANNOTATED_TARGET')
            elif value.kind == 'NODE':
                self.emit_source(target, 'id', LOCAL, 'ANNOTATED_TARGET')
        else:
            self.visit_field(occurrence, 'target')
        creator = self.current
        if (self.blocks[creator].origin.role == 'ROOT'
                or self.blocks[creator].origin.role == 'CLASS_BODY' and self.conditional[creator]):
            if creator not in self.has_conditional:
                self.has_conditional.add(creator)
                self.emit('__conditional_annotations__', USE, 'CONDITIONAL_ANNOTATION_USE',
                    anchor=self.site(occurrence, 'annotation'))
        if creator not in self.variable_annotations:
            block = self.block(self.blocks[creator].origin.occurrence, 'VARIABLE_ANNOTATIONS',
                can_see=self.blocks[creator].origin.role == 'CLASS_BODY' and not self.future)
            self.variable_annotations[creator] = block
        annotation = self.variable_annotations[creator]
        # All FunctionBlock kinds share this rule; nested blocks reset the bit.
        old = self.unevaluated[annotation]
        self.unevaluated[annotation] = self.blocks[creator].origin.role in (
            'FUNCTION_BODY', 'LAMBDA_BODY', 'COMPREHENSION', 'GENERATOR_EXPRESSION')
        try:
            self.enter(annotation, ((occurrence, 'annotation'),))
        finally:
            self.unevaluated[annotation] = old
        self.visit_field(occurrence, 'value')

    def named_expression(self, occurrence):
        if self.iter_expression[self.current]:
            raise SymbolStop('WALRUS_IN_COMPREHENSION_ITERABLE', 'named expression in an inherited comprehension iterable context', block=self.current)
        if self.blocks[self.current].origin.role not in ('COMPREHENSION', 'GENERATOR_EXPRESSION'):
            return
        target = self.field(occurrence, 'target').reference
        source, anchor = self.source(target, 'id')
        text = self.source_name(source)
        block = self.current
        while block is not None:
            self.step(block=block, source=source)
            role = self.blocks[block].origin.role
            if role in ('COMPREHENSION', 'GENERATOR_EXPRESSION'):
                slot = self.slot_index.get((block, self.name(text)))
                flags = 0 if slot is None else self.raw_slots[slot][2]
                if flags & COMP_ITER and flags & LOCAL:
                    raise SymbolStop('WALRUS_REBINDS_COMP_TARGET', 'named expression conflicts with a comprehension target', block=block, source=source)
            elif role in ('FUNCTION_BODY', 'LAMBDA_BODY'):
                slot = self.slot_index.get((block, self.name(text)))
                flags = 0 if slot is None else self.raw_slots[slot][2]
                self.emit(text, GLOBAL if flags & GLOBAL else NONLOCAL, 'WALRUS_ORIGIN_DIRECTIVE', source=source, anchor=anchor)
                self.emit(text, LOCAL, 'WALRUS_HOME_BINDING', source=source, anchor=anchor, target=block)
                return
            elif role == 'ROOT':
                self.emit(text, GLOBAL, 'WALRUS_ORIGIN_DIRECTIVE', source=source, anchor=anchor)
                self.emit(text, GLOBAL, 'WALRUS_HOME_BINDING', source=source, anchor=anchor, target=block)
                return
            else:
                raise SymbolStop('WALRUS_FORBIDDEN_COMPILER_HOME', 'named expression escapes into a class/type compiler scope', block=block, source=source)
            block = self.blocks[block].parent
        raise SymbolStop('WALRUS_COMPILER_HOME_MISSING', 'named expression has no enclosing compiler home', source=source)

    def generic(self, occurrence):
        node, block = self.syntax.nodes[occurrence], self.current
        constructor = self.constructors[node.constructor]
        for rule_id in constructor.fields:
            rule = self.catalog.fields[rule_id]
            if rule.dispatch not in ('ORDERED_VISIT', 'HANDLER_FIELD'):
                continue
            old = self.conditional[block]
            if rule.name in _CONDITIONAL_FIELDS.get(node.constructor, ()):
                self.conditional[block] = True
            try:
                self.value(node.fields[rule.ordinal])
            finally:
                self.conditional[block] = old

    def visit(self, occurrence):
        node = self.syntax.nodes[occurrence]
        kind = node.constructor
        if kind == 'Name':
            self.tick(occurrence)
            source, anchor = self.source(occurrence, 'id', required=False)
            if source is None:
                raise SymbolStop('NAME_SOURCE_CONTRIBUTION_MISSING', 'visited Name has no original source contribution', block=self.current)
            if self.unevaluated[self.current]:
                self.ignored.add(source)
                return
            ctx = self.syntax.nodes[self.field(occurrence, 'ctx').reference].constructor
            flags, rule = (USE, 'NAME_LOAD') if ctx == 'Load' else (LOCAL, 'NAME_DELETE' if ctx == 'Del' else 'NAME_STORE')
            self.emit(self.source_name(source), flags, rule, source=source, anchor=anchor)
            if ctx == 'Load' and self.source_name(source) == 'super' and self.blocks[self.current].origin.role in _FUNCTIONLIKE:
                self.emit('__class__', USE, 'SUPER_CLASS_USE', anchor=anchor)
            return
        if kind in ('FunctionDef', 'AsyncFunctionDef', 'ClassDef'):
            self.emit_source(occurrence, 'name', LOCAL, 'SOURCE_BINDER')
        elif kind in ('TypeVar', 'TypeVarTuple', 'ParamSpec'):
            self.emit_source(occurrence, 'name', LOCAL | TYPE_PARAM, 'TYPE_PARAMETER')
        elif kind in ('Global', 'Nonlocal'):
            for index, _ in enumerate(self.field(occurrence, 'names').items):
                self.emit_source(occurrence, 'names', GLOBAL if kind == 'Global' else NONLOCAL,
                    'GLOBAL_DIRECTIVE' if kind == 'Global' else 'NONLOCAL_DIRECTIVE', (index,))
        elif kind == 'alias':
            # Only the binder contribution, not IMPORT_*_SPEC or a star marker,
            # is in the C2a contribution table.
            for field in ('asname', 'name'):
                source, anchor = self.source(occurrence, field, required=False)
                if source is not None:
                    self.emit(self.source_name(source), IMPORT, 'IMPORT_BINDER', source=source, anchor=anchor)
                    break
        elif kind == 'MatchStar':
            source, anchor = self.source(occurrence, 'name', required=False)
            if source is not None:
                self.emit(self.source_name(source), LOCAL, 'PATTERN_BINDER', source=source, anchor=anchor)
        elif kind == 'NamedExpr':
            self.named_expression(occurrence)
        elif kind == 'ExceptHandler':
            self.tick(occurrence)
            self.visit_field(occurrence, 'type')
            source, anchor = self.source(occurrence, 'name', required=False)
            if source is not None:
                self.emit(self.source_name(source), LOCAL, 'EXCEPT_BINDER', source=source, anchor=anchor)
            self.visit_field(occurrence, 'body')
            return
        constructor = self.constructors[kind]
        if constructor.dispatch in ('ROOT', 'ORDERED_CHILDREN'):
            self.tick(occurrence)
            self.generic(occurrence)
        else:
            super().visit(occurrence)
        if kind in ('MatchMapping', 'MatchAs'):
            source, anchor = self.source(occurrence, 'rest' if kind == 'MatchMapping' else 'name', required=False)
            if source is not None:
                self.emit(self.source_name(source), LOCAL, 'PATTERN_BINDER', source=source, anchor=anchor)

    def collect(self):
        scopes = super().run()
        if (tuple(self.blocks) != self.accepted.blocks or scopes != self.accepted.scopes
                or tuple(self.rewrites) != self.accepted.rewrites):
            raise SymbolStop('COMPILER_TOPOLOGY_REPLAY_CHANGED', 'scalar visitor changed the accepted C3-T graph')
        sources = []
        for source, row in enumerate(self.declarations.contributions):
            events = tuple(self.source_events[source])
            if events:
                disposition = 'DETACHED_RAW' if row.analysis_visibility == 'DETACHED_FUTURE_ANNOTATION' else 'EMITTED'
            elif source in self.ignored:
                disposition = 'IGNORED_LOCAL_ANNOTATION'
            else:
                raise SymbolStop('SOURCE_CONTRIBUTION_UNDISPOSED', 'source contribution lacks a first-pass event or explicit suppression', source=source)
            sources.append(SourceDisposition(source, disposition, events))
        return RawCollection(tuple(RawSlot(block, name, flags, tuple(events)) for block, name, flags, events in self.raw_slots),
            tuple(self.events), tuple(sources), tuple(RawParameters(i, tuple(self.param_slots[i])) for i in range(len(self.blocks))),
            tuple(sorted(self.has_conditional)), self.symbol_steps)


def collect_symbols(syntax, roles, contexts, declarations, native, topology, *, limits: SymbolLimits) -> RawCollection:
    if type(limits) is not SymbolLimits:
        raise ValueError('raw compiler collection requires typed SymbolLimits')
    if topology.status == 'SOURCE_ONLY_MODE':
        return RawCollection((), (), tuple(SourceDisposition(i, 'GRAMMAR_ONLY', ())
            for i in range(len(declarations.contributions))), (), (), 0)
    if topology.status != 'PROJECTED' or declarations.status != 'DECLARED':
        raise SymbolStop('UPSTREAM_NOT_READY', 'raw symbol collection requires accepted source declarations and compiler topology')
    try:
        return _Collector(syntax, roles, contexts, declarations, native, topology, limits).collect()
    except TopologyStop as error:
        raise SymbolStop(error.frontier.code, error.frontier.detail, budget=error.budget) from error


__all__ = [
    'model_compiler_symbols',
    'main',
    'RAW_RULES',
    'collect_symbols',
    'GLOBAL',
    'LOCAL',
    'PARAM',
    'NONLOCAL',
    'USE',
    'IMPORT',
    'ANNOT',
    'COMP_ITER',
    'TYPE_PARAM',
]
