"""Compiler topology materials."""

from __future__ import annotations

import argparse
import math
import multiprocessing
import re
import time
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from scar.analysis import source_declaration_materials as _d, source_symbol_materials as _a
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
from scar.ir.program.compiler_symbols import CompilerTopologyMaterial
from scar.ir.program.source_contexts import SourceContextsMaterial
from scar.ir.program.source_roles import SourceRolesMaterial
from scar.ir.program.source_paths import SourceSymbolsMaterial
from scar.ir.record_codec import encode


# C3-T material transactions: topology only, with the native fact reader intact.
_ARTIFACT = 'compiler_topology'
_CATALOG = _ARTIFACT + '/catalog.json.gz'
_OWN_DEFAULTS = dict(max_blocks=100_000, max_rewrites=200_000, max_visit_steps=4_000_000)
_DEFAULTS = dict(max_nodes=1_000_000, max_tokens=2_000_000, max_ports=1_000_000, max_slots=10_000_000,
    max_scopes=100_000, max_regions=200_000, max_boundaries=1_000_000, max_symbols=1_000_000, **_OWN_DEFAULTS)
_PROJECTIONS = dict(public_blocks='PUBLIC', inlined_blocks='INLINED',
    future_detached_blocks='FUTURE_DETACHED', detached_descendant_blocks='DETACHED_DESCENDANT')
_TOTALS = ('blocks', 'scopes', 'rewrites', 'visit_steps', *_PROJECTIONS)
_FLAGS = dict(authority='SOURCE_COMPILER_TOPOLOGY', symbol_origin_projection='NOT_COMPUTED_IN_C3T',
    parameter_projection='NOT_COMPUTED_IN_C3T', runtime_execution='NOT_OBSERVED')
_A_INVENTORY = ('stage', 'units', 'scopes', 'symbols', 'status_counts', 'authority', 'ast_join',
    'complete_lexical_scopes', 'later_stages_verified', 'whole_program_complete')


def _symbols_input(stage, directory, basis):
    if directory is not None:
        root, stored, raw, pins = _a._read_store(directory, stage.out)
        descriptors = [row['artifact'] for row in stored['results']]
        if _json(stored['basis']) != _json(basis) or len(descriptors) != len(basis):
            raise ValueError('F3-A source basis differs from selected original sources')
        if any(row['unit'] != unit['key'] or row['artifact']['path'] != 'units/' + unit['key'] + '/source_symbols.json.gz'
                for row, unit in zip(stored['results'], basis)):
            raise ValueError('F3-A original unit/path order differs from the model')
    else:
        root, stored, raw, pins = stage.root, None, None, None
        inventory = stage.manifest.get('source_symbols')
        if type(inventory) is not dict or inventory.get('schema') != 'scar.source-symbol-inventory':
            raise ValueError('compiler-topology requires canonical F3-A or explicit --symbols-materials')
        try:descriptors = [unit['artifacts']['source_symbols'] for unit in stage.manifest['units']]
        except KeyError as exc:raise ValueError('canonical F3-A unit artifact is absent') from exc
    return dict(root=root, stored=stored, raw=raw, pins=pins, descriptors=descriptors)


def _check_symbols(stage, handle, rows, jobs):
    if handle['stored'] is not None:
        _a._check_summary(handle['stored'], rows)
        check_material_store(handle['root'], handle['descriptors'], handle['raw'], handle['pins'])
    else:
        report = _a._report(rows, schema='', limits={}, jobs=jobs)
        expected = {key: report[key] for key in _A_INVENTORY}
        expected.update(schema='scar.source-symbol-inventory', schema_version=1, scope='DECLARED_MODEL_SOURCE_UNITS')
        if _json(stage.manifest['source_symbols']) != _json(expected):
            raise ValueError('canonical F3-A inventory differs from freshly verified native facts')


def _upstream_rows(unit, f1, f2, roles, contexts, symbols, descriptors):
    common = dict(unit=unit['key'], upstream_f1_status=f1['status'], upstream_f2_status=f2['status'])
    b = dict(common, artifact=descriptors[0], status=roles['status'], slots=roles['slots'],
        ports=len(roles['ports']), list_items=roles['list_items'], max_ports=roles['max_ports'],
        max_slots=roles['max_slots'], frontiers=roles['frontiers'])
    c = dict(common, artifact=descriptors[1], status=contexts['status'],
        **{key: len(contexts[key]) for key in ('scopes', 'regions', 'private_contexts', 'boundaries')},
        ports=len(roles['ports']) if contexts['status'] == 'CONTEXTUALIZED' else 0, slots=contexts['slots'],
        **{key: contexts[key] for key in _d._c1._OWN_LIMITS}, frontiers=contexts['frontiers'], upstream_f3b_status=roles['status'])
    return b, c, _a._row(unit['key'], descriptors[2], symbols, f1, f2)


def _topology_unit(task):
    from scar.analysis.compiler_topology import compiler_topology_document, read_compiler_topology
    from scar.analysis.source_symbols import compiler_hash_context
    model, destination, unit, inputs, prior_root, prior, limits, digest, independent, expected_context = task
    if compiler_hash_context() != expected_context:
        raise ValueError('compiler-topology worker hash context differs; start analysis with fixed PYTHONHASHSEED')
    f0, f1, f2 = read_source_prerequisites(_Fragments(model), unit, limits['max_nodes'], limits['max_tokens'])
    roles, contexts, symbols = [_Fragments(root).read(descriptor) for root, descriptor in inputs]
    for doc, keys, label in ((roles, ('max_ports', 'max_slots'), 'F3-B'),
            (contexts, _d._c1._OWN_LIMITS, 'F3-C1'), (symbols, ('max_scopes', 'max_symbols'), 'F3-A')):
        for key in keys:check_replay_limit(doc, key, limits[key], label)
    if prior is not None:
        document = _Fragments(prior_root).read(prior)
        for key in _OWN_DEFAULTS:check_replay_limit(document, key, limits[key], 'C3-T')
        read_compiler_topology(document, f0, f1, f2, roles, contexts, symbols)
        descriptor = prior
    elif destination is None:
        raise ValueError('compiler topology artifact is absent')
    else:
        document = compiler_topology_document(f0, f1, f2, roles, contexts, symbols,
            **{key: limits[key] for key in _OWN_DEFAULTS})
        descriptor = None
    if document['profile']['topology_catalog_sha256'] != digest:
        raise ValueError('compiler topology worker catalog differs from its parent')
    if destination is not None and (independent or descriptor is None):
        relative = 'units/' + unit['key'] + '/compiler_topology.json.gz'
        if (destination / relative).exists():raise FileExistsError('new topology artifact collides with an existing file')
        descriptor = _write(destination, relative, document)
    row = dict(unit=unit['key'], artifact=descriptor, status=document['status'],
        **{key: len(document[key]) for key in ('blocks', 'scopes', 'rewrites')}, visit_steps=document['visit_steps'],
        **{key: sum(block['projection'] == value for block in document['blocks']) for key, value in _PROJECTIONS.items()},
        **{key: document[key] for key in _OWN_DEFAULTS}, frontiers=document['frontiers'],
        upstream_f1_status=f1['status'], upstream_f2_status=f2['status'], upstream_f3b_status=roles['status'],
        upstream_c1_status=contexts['status'], upstream_f3a_status=symbols['status'])
    b, c, a = _upstream_rows(unit, f1, f2, roles, contexts, symbols, [item[1] for item in inputs])
    return row, b, roles['counts'], c, contexts['counts'], a


def _report(rows, *, schema, limits, jobs, digest):
    covered = [row['status'] not in ('BUDGET_STOPPED', 'PROJECTION_CONTRADICTION')
        and row['upstream_f1_status'] != 'BUDGET_STOPPED'
        and row['upstream_f2_status'] not in ('BUDGET_STOPPED', 'PARSER_RESOURCE_ERROR')
        and all(row[key] != 'BUDGET_STOPPED' for key in ('upstream_f3b_status', 'upstream_c1_status'))
        and row['upstream_f3a_status'] not in ('BUDGET_STOPPED', 'RESOURCE_ERROR') for row in rows]
    return dict(schema=schema, schema_version=1, stage='F3-C3T', input_scope='DECLARED_MODEL_SOURCE_UNITS',
        units=len(rows), covered_units=sum(covered), complete_for_declared_units=all(covered),
        all_declared_units_projected=all(row['status'] == 'PROJECTED' for row in rows),
        status_counts=dict(Counter(row['status'] for row in rows)), results=rows,
        **{key: sum(row[key] for row in rows) for key in _TOTALS}, limits=limits, jobs=jobs,
        catalog_sha256=digest, symtable_executed=any(row['upstream_f3a_status'] not in
            ('UPSTREAM_NOT_PARSED', 'UNSUPPORTED_SYMTABLE_MODE') for row in rows), target_executed=False,
        later_stages_verified=False, whole_program_complete=False, **_FLAGS)


def _check_summary(document, rows, digest):
    from scar.analysis.source_symbols import compiler_hash_context
    limits, jobs = document.get('limits'), document.get('jobs')
    if (type(limits) is not dict or set(limits) != set(_DEFAULTS)
            or any(type(value) is not int or value < 0 for value in limits.values())
            or type(jobs) is not int or not 1 <= jobs <= 32):
        raise ValueError('invalid topology material budgets/jobs')
    for row in rows:
        for key in _OWN_DEFAULTS:
            if type(row.get(key)) is not int or not 0 <= row[key] <= limits[key]:
                raise ValueError('topology row budget exceeds stored limits')
    expected = _report(rows, schema='scar.compiler-topology-materials', limits=limits, jobs=jobs, digest=digest)
    if (_json({key: document.get(key) for key in expected}) != _json(expected)
            or document.get('compiler_hash_context') != compiler_hash_context()
            or document.get('collect_only') is not True or document.get('canonical_model_published') is not False
            or document.get('manifest_unchanged') is not True):
        raise ValueError('topology material summary differs from freshly verified rows/hash context')
    for key in ('observed_manifest_sha256', 'prior_manifest_sha256', 'published_manifest_sha256'):
        if type(document.get(key)) is not str or re.fullmatch(r'[0-9a-f]{64}', document[key]) is None:
            raise ValueError('invalid observed topology manifest receipt')
    if 'seconds' in document and (type(document['seconds']) not in (int, float)
            or not math.isfinite(document['seconds']) or document['seconds'] < 0):
        raise ValueError('invalid topology observed duration')


def _catalog_at(root, descriptor, expected):
    if descriptor.get('path') != _CATALOG or _json(_Fragments(root).read(descriptor)) != _json(expected):
        raise ValueError('topology catalog receipt/content differs from the current profile')


def _inventory(report, catalog):
    keys = ('stage', 'units', *_TOTALS, 'status_counts', 'catalog_sha256', *_FLAGS,
        'later_stages_verified', 'whole_program_complete')
    return dict({key: report[key] for key in keys}, schema='scar.compiler-topology-inventory', schema_version=1,
        scope='DECLARED_MODEL_SOURCE_UNITS', artifacts=[catalog])


def model_compiler_topology(model, out, *, roles_materials=None, contexts_materials=None, symbols_materials=None,
        jobs=4, collect_only=False, verify=False, materials=None, max_nodes=1_000_000, max_tokens=2_000_000,
        max_ports=1_000_000, max_slots=10_000_000, max_scopes=100_000, max_regions=200_000,
        max_boundaries=1_000_000, max_symbols=1_000_000, max_blocks=100_000, max_rewrites=200_000,
        max_visit_steps=4_000_000):
    from scar.analysis.compiler_topology import compiler_topology_catalog
    from scar.analysis.source_contexts import source_context_catalog
    from scar.analysis.source_role_materials import source_role_catalog
    from scar.analysis.source_symbols import compiler_hash_context
    limits = {key: value for key, value in locals().items() if key in _DEFAULTS}
    external = (roles_materials, contexts_materials, symbols_materials)
    if (type(jobs) is not int or not 1 <= jobs <= 32 or type(collect_only) is not bool or type(verify) is not bool
            or collect_only and verify or materials is not None and not verify
            or any(value is not None for value in external) and not (collect_only or verify)
            or any(type(value) is not int or value < 0 for value in limits.values())):
        raise ValueError('topology requires bounded distinct modes; external prerequisite capture is collect-only')
    started = time.monotonic();expected_context = compiler_hash_context()
    catalog = compiler_topology_catalog();digest = catalog['content_sha256']
    with SourceStage(model, out, artifact=_ARTIFACT, verify=verify or collect_only,
            consumes=(*REQUIRED_SOURCE_ARTIFACTS, 'source_roles', 'source_contexts', 'source_symbols', _ARTIFACT)) as stage:
        base = source_input_basis(stage.manifest['units'])
        b_catalog, c_catalog = source_role_catalog(), source_context_catalog()
        b = _d._input(stage, 'source_roles', roles_materials, base, b_catalog, _d._b)
        basis = _d._with_artifacts(base, 'source_roles', b['descriptors'])
        c = _d._input(stage, 'source_contexts', contexts_materials, basis, c_catalog, _d._c1)
        a = _symbols_input(stage, symbols_materials, base)
        basis = _d._with_artifacts(_d._with_artifacts(basis, 'source_contexts', c['descriptors']), 'source_symbols', a['descriptors'])
        existing, canonical_catalog = stage.manifest.get(_ARTIFACT), None
        if existing is not None:
            if type(existing.get('artifacts')) is not list or len(existing['artifacts']) != 1:
                raise ValueError('canonical topology inventory must own one shared catalog')
            canonical_catalog = existing['artifacts'][0];_catalog_at(stage.root, canonical_catalog, catalog)
        canonical_pins = _stamps(stage.root, (canonical_catalog,)) if canonical_catalog else None
        stored = store_root = store_raw = store_pins = None
        if materials is not None:
            store_root, stored, store_raw = read_material_manifest(materials, stage.out)
            _check_summary(stored, stored['results'], digest)
            if _json(stored.get('basis')) != _json(basis) or len(stored['results']) != len(basis):
                raise ValueError('topology material basis differs from selected original sources')
            if any(row['unit'] != unit['key'] or row['artifact']['path'] != stage.relative(unit)
                    for row, unit in zip(stored['results'], basis)):
                raise ValueError('topology original unit/path order differs from the model')
            _catalog_at(store_root, stored['catalog'], catalog)
            store_pins = pin_material_store(store_root, [stored['catalog'], *(row['artifact'] for row in stored['results'])], store_raw)
        destination = stage.out if collect_only else stage.staging
        if destination is not None and (collect_only or canonical_catalog is None):
            if (destination / _CATALOG).exists():raise FileExistsError('new topology catalog collides with an existing file')
            descriptor = _write(destination, _CATALOG, catalog)
        else:
            descriptor = stored['catalog'] if stored else canonical_catalog
            if descriptor is None:raise ValueError('canonical topology catalog is absent')
        replay_limits = {key: min(value, stored['limits'][key]) for key, value in limits.items()} if stored else limits
        tasks = [(stage.root, destination, unit, [(item['root'], item['descriptors'][i]) for item in (b,c,a)],
            store_root or stage.root, stored['results'][i]['artifact'] if stored else stage.manifest['units'][i]['artifacts'].get(_ARTIFACT),
            replay_limits, digest, collect_only, expected_context) for i, unit in enumerate(basis)]
        rows,b_rows,c_rows,a_rows=[],[],[],[]
        b_counts=[dict(rule=rule['rule'], **{key:0 for key in _d._b._COUNT_FIELDS}) for rule in b_catalog['rules']]
        c_counts=[dict(rule=rule['rule'], **{key:0 for key in _d._c1._COUNT_FIELDS}) for rule in c_catalog['rules']]
        def accept(packet):
            row,b_row,b_current,c_row,c_current,a_row=packet
            rows.append(row);b_rows.append(b_row);c_rows.append(c_row);a_rows.append(a_row)
            for total,current,fields in ((b_counts,b_current,_d._b._COUNT_FIELDS),(c_counts,c_current,_d._c1._COUNT_FIELDS)):
                if len(total)!=len(current):raise ValueError('topology prerequisite counter denominator is incomplete')
                for left,right in zip(total,current):
                    if left['rule']!=right['rule']:raise ValueError('topology prerequisite rule order changed')
                    for key in fields:left[key]+=right[key]
        if jobs==1:
            for packet in map(_topology_unit,tasks):accept(packet)
        else:
            with ProcessPoolExecutor(max_workers=jobs,mp_context=multiprocessing.get_context('spawn')) as pool:
                for packet in pool.map(_topology_unit,tasks,chunksize=1):accept(packet)
        _d._check_input(stage,b,b_rows,b_counts,jobs);_d._check_input(stage,c,c_rows,c_counts,jobs)
        _check_symbols(stage,a,a_rows,jobs)
        if stored is not None:
            _check_summary(stored,rows,digest)
            check_material_store(store_root,[stored['catalog'],*(row['artifact'] for row in rows)],store_raw,store_pins)
        report=_report(rows,schema=('scar.compiler-topology-material-verification' if stored else
            'scar.compiler-topology-verification' if verify else 'scar.compiler-topology-materials' if collect_only
            else 'scar.compiler-topology-gate'),limits=limits,jobs=jobs,digest=digest)
        report.update(catalog=descriptor,collect_only=collect_only,canonical_model_published=False,
            observed_manifest_sha256=sha256(stage.original).hexdigest(),compiler_hash_context=expected_context)
        if collect_only:
            report['basis']=basis;material_store_members(stage.out,[descriptor,*(row['artifact'] for row in rows)],result=False)
        elif not verify and report['complete_for_declared_units']:
            stage.publish(rows,_inventory(report,descriptor));report['canonical_model_published']=True
        elif verify and stored is None and _json(existing)!=_json(_inventory(report,descriptor)):
            raise ValueError('canonical topology inventory differs from freshly verified topology')
        if canonical_catalog and _stamps(stage.root,(canonical_catalog,))!=canonical_pins:
            raise ValueError('canonical topology catalog changed before publication')
        if stored is not None:
            report.update(materials=str(store_root),materials_result_sha256=sha256(store_raw).hexdigest(),material_source_basis_unchanged=True)
        report['seconds']=time.monotonic()-started
        return stage.finish(report)


def main(argv=None):
    parser=argparse.ArgumentParser(prog='scar compiler-topology')
    for key in ('model','out'):parser.add_argument('--'+key,required=True)
    for key in ('roles-materials','contexts-materials','symbols-materials','materials'):parser.add_argument('--'+key)
    parser.add_argument('--jobs',type=int,default=4)
    parser.add_argument('--collect-only',action='store_true');parser.add_argument('--verify',action='store_true')
    for key,value in _DEFAULTS.items():parser.add_argument('--'+key.replace('_','-'),type=int,default=value)
    report=model_compiler_topology(**vars(parser.parse_args(argv)))
    print(_json({key:report[key] for key in ('stage','units','blocks','scopes','rewrites','status_counts',
        'complete_for_declared_units','canonical_model_published')}).decode().strip())
    return 0 if report['complete_for_declared_units'] else 1


# C3-T source/compiler topology queries; symbol origins remain unprojected.
def _digest(document):
    return sha256(json.dumps(document, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('ascii')).hexdigest()


def _index(index, size, label):
    if type(index) is not int or not 0 <= index < size:
        raise ValueError('C3-T ' + label + ' is outside its original table')
    return index


_SOURCE_ROLES = {
    'ROOT': 'ROOT', 'FUNCTION_BODY': 'FUNCTION_BODY', 'CLASS_BODY': 'CLASS_BODY',
    'LAMBDA_BODY': 'LAMBDA_BODY', 'TYPE_PARAMETERS': 'TYPE_PARAMETERS',
    'HEADER_ANNOTATIONS': 'HEADER_ANNOTATIONS', 'VARIABLE_ANNOTATIONS': 'VARIABLE_ANNOTATIONS',
    'TYPE_ALIAS_VALUE': 'TYPE_ALIAS', 'TYPE_PARAMETER_BOUND': 'TYPE_PARAMETER_BOUND',
    'TYPE_PARAMETER_DEFAULT': 'TYPE_PARAMETER_DEFAULT', 'COMPREHENSION': 'COMPREHENSION',
    'GENERATOR_EXPRESSION': 'GENERATOR_EXPRESSION',
}
_ORIGIN_CONSTRUCTORS = {
    'ROOT': ('Module', 'Interactive', 'Expression'),
    'FUNCTION_BODY': ('FunctionDef', 'AsyncFunctionDef'), 'CLASS_BODY': ('ClassDef',),
    'LAMBDA_BODY': ('Lambda',), 'TYPE_PARAMETERS': ('FunctionDef', 'AsyncFunctionDef', 'ClassDef', 'TypeAlias'),
    'HEADER_ANNOTATIONS': ('FunctionDef', 'AsyncFunctionDef'),
    'VARIABLE_ANNOTATIONS': ('Module', 'Interactive', 'FunctionDef', 'AsyncFunctionDef', 'ClassDef'),
    'TYPE_ALIAS': ('TypeAlias',), 'TYPE_PARAMETER_BOUND': ('TypeVar',),
    'TYPE_PARAMETER_DEFAULT': ('TypeVar', 'ParamSpec', 'TypeVarTuple'),
    'COMPREHENSION': ('ListComp', 'SetComp', 'DictComp'),
    'GENERATOR_EXPRESSION': ('GeneratorExp',),
}


class CompilerTopologyView:
    """Queries retain C1 scope/region IDs and original F3-A public row IDs."""

    def __init__(self, roles, c1, native, material, *, contexts=None):
        from scar.analysis.compiler_topology import _catalog
        if (type(roles) is not SourceRolesMaterial or roles.status != 'CLASSIFIED'
                or type(c1) is not SourceContextsMaterial or c1.status != 'CONTEXTUALIZED'
                or type(native) is not SourceSymbolsMaterial
                or type(material) is not CompilerTopologyMaterial
                or material.status not in ('PROJECTED', 'SOURCE_ONLY_MODE')):
            raise ValueError('C3-T queries require typed complete source/topology inputs')
        self.roles, self.contexts, self.native, self.material = roles, c1, native, material
        self.context_view, self.catalog = contexts, _catalog()
        basis = material.input
        if (basis.source_scope_count != len(c1.scopes) or basis.native_scope_count != len(native.scopes)
                or basis.c1_status != c1.status or basis.f3a_status != native.status
                or basis.f3b_document_sha256 != _digest(encode(roles))
                or basis.c1_document_sha256 != _digest(encode(c1))
                or basis.f3a_document_sha256 != _digest(encode(native))
                or any(getattr(basis, field) != getattr(native.input, field)
                    or getattr(basis, field) != getattr(c1.input, field)
                    for field in ('f0_document_sha256', 'f1_document_sha256', 'f2_document_sha256', 'path'))
                or basis.mode != native.mode or basis.mode != c1.input.mode
                or material.profile.native_profile_sha256 != _digest(encode(native.profile))
                or material.profile.topology_catalog_sha256 != self.catalog.content_sha256
                or material.profile.context_catalog_sha256 != c1.profile.context_catalog_sha256
                or material.profile.f2_schema_sha256 != c1.profile.f2_schema_sha256):
            raise ValueError('C3-T source/compiler receipts or versioned catalogs differ')
        if contexts is not None and (contexts.roles != roles or contexts.material != c1):
            raise ValueError('C3-T source context view belongs to another source unit')
        self._owners = {}
        for index, scope in enumerate(c1.scopes):
            if scope.owner in self._owners:
                raise ValueError('C3-T requires a unique source environment per typed C1 owner')
            self._owners[scope.owner] = index

    def block(self, index):
        return self.material.blocks[_index(index, len(self.material.blocks), 'compiler block reference')]

    def scope(self, index):
        return self.material.scopes[_index(index, len(self.material.scopes), 'source scope reference')]

    def region(self, index):
        from scar.ir.program.source_contexts import OwnerRef
        region = self.contexts.regions[_index(index, len(self.contexts.regions), 'source region reference')]
        owner = region.owner
        if owner.kind == 'FORMAL_INTERFACE':
            # A function's parameter interface belongs to its body environment,
            # although the containing arguments slot is evaluated by its creator.
            candidates = [self._owners[key] for kind in ('FUNCTION_BODY', 'LAMBDA_BODY')
                if (key := OwnerRef(owner.occurrence, kind)) in self._owners]
            if len(candidates) != 1:
                raise ValueError('C3-T formal region lacks its unique source function/lambda body')
            scope = candidates[0]
        else:
            if owner not in self._owners:
                raise ValueError('C3-T region has no original source owner environment')
            scope = self._owners[owner]
        return self.scope(scope)

    def port(self, index):
        _index(index, len(self.roles.ports), 'source port reference')
        if self.context_view is None:
            raise ValueError('C3-T port queries require the original SourceContextView')
        return self.scope(self.context_view.context_for_port(index).state.environment)

    def _empty_header(self, occurrence):
        fields = self.context_view.role_view
        if fields.field(occurrence, 'returns').kind == 'NODE':
            return False
        arguments = fields.field(occurrence, 'args').reference
        args = []
        for name in ('posonlyargs', 'args', 'kwonlyargs'):
            args.extend(item.reference for item in fields.field(arguments, name).items)
        for name in ('vararg', 'kwarg'):
            value = fields.field(arguments, name)
            if value.kind == 'NODE':
                args.append(value.reference)
        return all(fields.field(arg, 'annotation').kind != 'NODE' for arg in args)

    def verify(self):
        from scar.analysis.compiler_topology import _projection_children
        data = self.material
        if tuple(row.scope for row in data.scopes) != tuple(range(len(self.contexts.scopes))):
            raise ValueError('C3-T source scope projection does not preserve every original ordinal')
        if data.status == 'SOURCE_ONLY_MODE':
            if data.input.mode != 'func_type' or self.native.status != 'UNSUPPORTED_SYMTABLE_MODE':
                raise ValueError('C3-T grammar-only projection has a different official compiler boundary')
            for index in range(len(self.contexts.regions)):
                self.region(index)
            return dict(blocks=0, scopes=len(data.scopes), regions=len(self.contexts.regions),
                public_blocks=0, public_edges=0, rewrites=0)
        if self.native.status != 'CAPTURED' or not data.blocks:
            raise ValueError('C3-T public projection requires a captured official scope graph')
        if data.blocks[0].parent is not None or data.blocks[0].origin.role != 'ROOT':
            raise ValueError('C3-T original compiler construction graph has a different root')
        origins, child_refs, public_refs, expected_rewrites = set(), [[] for _ in data.blocks], {}, set()
        detached = [False] * len(data.blocks)
        for index, block in enumerate(data.blocks):
            origin = block.origin
            key = origin.occurrence, origin.role
            if key in origins or origin.role not in self.catalog.origin_roles:
                raise ValueError('C3-T compiler origin is duplicated or has an undeclared role')
            origins.add(key)
            if self.context_view is not None:
                syntax = self.context_view.syntax
                _index(origin.occurrence, len(syntax.nodes), 'source occurrence')
                if syntax.nodes[origin.occurrence].constructor not in _ORIGIN_CONSTRUCTORS[origin.role]:
                    raise ValueError('C3-T compiler origin role does not match its original F2 constructor')
            if (block.attachment not in self.catalog.attachments or block.projection not in self.catalog.projections
                    or type(block.can_see_class_scope) is not bool):
                raise ValueError('C3-T compiler block has an undeclared attachment/projection policy')
            if index:
                _index(block.parent, index, 'previously created compiler parent')
            elif block.attachment != 'DIRECT' or block.projection != 'PUBLIC':
                raise ValueError('C3-T public root cannot be detached or inlined')
            if block.parent is not None:
                detached[index] = detached[block.parent]
            if block.attachment == 'FUTURE_DETACHED':
                if origin.role not in ('HEADER_ANNOTATIONS', 'VARIABLE_ANNOTATIONS'):
                    raise ValueError('C3-T future detachment requires an annotation origin')
                detached[index] = True
                expected = 'FUTURE_DETACHED'
                expected_rewrites.add(('FUTURE_DETACH', index, block.parent))
            elif detached[index]:
                expected = 'DETACHED_DESCENDANT'
            elif origin.role == 'COMPREHENSION' and not self.block(block.parent).can_see_class_scope:
                expected = 'INLINED'
                expected_rewrites.add(('INLINE_COMPREHENSION', index, block.parent))
            else:
                expected = 'PUBLIC'
            if block.projection != expected:
                raise ValueError('C3-T projection ignores the original reachable-parent/annotation rule')
            if block.projection == 'PUBLIC':
                native = _index(block.public_scope, len(self.native.scopes), 'original public compiler row')
                if native in public_refs:
                    raise ValueError('C3-T public row has multiple compiler-template identities')
                public_refs[native] = index
            elif block.public_scope is not None:
                raise ValueError('C3-T elided compiler template cannot claim an observed public row')
            if len(set(block.children)) != len(block.children):
                raise ValueError('C3-T original child attachment is duplicated')
            for child in block.children:
                _index(child, len(data.blocks), 'compiler child')
                if child <= index or self.block(child).parent != index or self.block(child).attachment != 'DIRECT':
                    raise ValueError('C3-T child was not attached to its original construction parent')
                child_refs[child].append(index)
            if block.source_scope is None:
                if origin.role != 'HEADER_ANNOTATIONS' or (self.context_view is not None and not self._empty_header(origin.occurrence)):
                    raise ValueError('C3-T source-less compiler block lacks an empty header origin')
            else:
                source = self.contexts.scopes[_index(block.source_scope, len(self.contexts.scopes), 'original source scope')]
                if (source.owner.occurrence, _SOURCE_ROLES.get(source.owner.kind)) != key:
                    raise ValueError('C3-T block is not the role of its original typed source owner')
        for index, block in enumerate(data.blocks):
            expected = [] if index == 0 or block.attachment == 'FUTURE_DETACHED' else [block.parent]
            if child_refs[index] != expected:
                raise ValueError('C3-T original parent/child attachment inverse is incomplete')
        for row in data.scopes:
            block = self.block(row.block)
            if block.source_scope != row.scope or row.disposition != block.projection:
                raise ValueError('C3-T source projection does not round-trip to its compiler template')
            source = self.contexts.scopes[row.scope]
            parent = None if source.parent is None else self.scope(source.parent).block
            if block.parent != parent:
                raise ValueError('C3-T source parent does not match the original compiler environment')
        actual_rewrites = {(row.rule, row.block, row.parent) for row in data.rewrites}
        if len(actual_rewrites) != len(data.rewrites) or actual_rewrites != expected_rewrites:
            raise ValueError('C3-T rewrite witnesses are missing, duplicated or unsupported')
        if set(public_refs) != set(range(len(self.native.scopes))) or data.blocks[0].public_scope != self.native.root:
            raise ValueError('C3-T public correspondence does not cover the original compiler rows exactly once')
        children, edges = _projection_children(data.blocks), 0
        for native, index in public_refs.items():
            projected = tuple(self.block(child).public_scope for child in children[index])
            if projected != self.native.scopes[native].children:
                raise ValueError('C3-T original public child edges differ from the witnessed splice result')
            edges += len(projected)
        for index in range(len(self.contexts.regions)):
            self.region(index)
        return dict(blocks=len(data.blocks), scopes=len(data.scopes), regions=len(self.contexts.regions),
            public_blocks=len(public_refs), public_edges=edges, rewrites=len(data.rewrites))


__all__ = [
    'model_compiler_topology',
    'main',
    'CompilerTopologyView',
]
