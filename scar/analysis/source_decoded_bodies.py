"""Bounded per-replay reuse of decoded original F4 source bodies."""

from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import fields
from enum import Enum
from hashlib import sha256
import json
from threading import get_ident

from scar.analysis.source_productions import read_production_plan
from scar.analysis.source_template_batch import _check_catalog, _graph
from scar.analysis.source_template_links import _implementation, _require, _worker_stores
from scar.analysis.source_world_material import WorldMaterial
from scar.ir.program import source_productions as P, source_declarations as T

_ACTIVE = ContextVar('scar_decoded_source_bodies', default=None)
_MAX_ENTRIES = 1536
_MAX_WIRE_BYTES = 16 * 1024 * 1024
_VISITS = 1_000_000


def _immutable(plan):
    pending = [plan]
    while pending:
        value = pending.pop()
        kind = type(value)
        if kind in (str, int, bool, type(None)) or isinstance(value, Enum):
            continue
        if kind in (tuple, frozenset):
            pending.extend(value)
            continue
        parameters = getattr(kind, '__dataclass_params__', None)
        if parameters is None or not parameters.frozen:
            return False
        pending.extend(getattr(value, field.name) for field in fields(kind))
    return True


class _DecodedPlans:
    def __init__(self):
        self.owner = get_ident()
        self.entries = OrderedDict()
        self.wire_bytes = 0
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if key in self.entries:
            self.entries.move_to_end(key)
            self.hits += 1
            return self.entries[key][1]
        self.misses += 1
        return None

    def admit(self, key, size, plan):
        if key in self.entries or size > _MAX_WIRE_BYTES or not _immutable(plan):
            return
        while len(self.entries) >= _MAX_ENTRIES or self.wire_bytes + size > _MAX_WIRE_BYTES:
            _, (evicted, _) = self.entries.popitem(last=False)
            self.wire_bytes -= evicted
        self.entries[key] = size, plan
        self.wire_bytes += size


def _current():
    plans = _ACTIVE.get()
    return plans if plans is not None and plans.owner == get_ident() else None


@contextmanager
def reuse_decoded_source_bodies():
    """Keep only immutable body material within one caller-owned replay."""
    active = _current()
    if active is not None:
        yield active
        return
    plans = _DecodedPlans()
    token = _ACTIVE.set(plans)
    try:
        yield plans
    finally:
        plans.entries.clear()
        plans.wire_bytes = 0
        _ACTIVE.reset(token)


def _read(material, keys, cache):
    """Keep the original template and worker checks around a single decode."""
    root, unit = material.root, material.unit
    raw = (root/'result.json').read_bytes()
    _require(raw == material._template_raw, unit, 'template graph root changed since original admission')
    report = json.loads(raw)
    graph = _graph(root, unit, raw, report)
    decoder = tuple(report['generator_sha256'][name] for name in
                    ('analysis/source_productions.py', 'ir/program/source_productions.py', 'ir/record_codec.py'))
    nodes = {node.key: node for node in graph.templates}
    _require(all(key in nodes for key in keys), unit, 'selected template is absent')
    declarations = {T.SourceMaterialDomain(row['domain']): row for row in report['parents']}
    specs = tuple((row['domain'], row['materials'], row['verification'],
                   row['collection_sha256'], row['verification_sha256']) for row in report['parents'])
    parents = {parent.domain: parent for parent in _worker_stores(specs, report['model_manifest_sha256'])}
    selected = defaultdict(lambda: defaultdict(list))
    for key in keys:
        node = nodes[key]
        selected[node.origin.domain][node.origin.line].append((key, node))
    result, pending = {}, []
    for domain, lines in selected.items():
        parent = parents[domain]
        _require(parent.receipt == declarations[domain], unit, 'parent proof changed since interface linking')
        _check_catalog(parent, unit)
        basis = {name: value for name, value in parent.basis[unit].items() if name != 'unit'}
        for _, line, document in parent.records(dict(basis, key=unit)):
            if line not in lines:
                continue
            wire = json.dumps(document, sort_keys=True, separators=(',', ':'),
                              ensure_ascii=True, allow_nan=False).encode('ascii')
            digest = sha256(wire).hexdigest()
            _require(all(digest == node.origin.plan_sha256 for _, node in lines[line]),
                     unit, 'original recipe body differs')
            cache_key = decoder, digest, _VISITS
            plan = cache.get(cache_key)
            if plan is None:
                plan = read_production_plan(document, max_visits=_VISITS)
                pending.append((cache_key, len(wire), plan))
            for key, node in lines[line]:
                _require(plan.source == P.ProductionOccurrence(key.occurrence)
                         and plan.purpose is key.purpose and plan.root == node.root
                         and plan.input == graph.input and plan.profile == graph.profile,
                         key, 'original recipe identity/input/profile differs')
                result[key] = plan
    _require(set(result) == set(keys), unit, 'selected recipe is absent from its original domain')
    for domain, parent in parents.items():
        _require(parent.receipt == declarations[domain], unit, 'parent proof changed since interface linking')
        _check_catalog(parent, unit)
        _require(parent.unchanged() and parent.generator_pins == _implementation(),
                 unit, 'original material changed during batch reading')
    _require((root/'result.json').read_bytes() == raw, unit, 'template graph root changed during batch reading')
    material.validate_parents()
    for key, size, plan in pending:
        cache.admit(key, size, plan)
    return result


class _WorldBodyAccess:
    def __init__(self, material, cache):
        self.original = material
        self.cache = cache

    def __getattr__(self, name):
        return getattr(self.original, name)

    closure = WorldMaterial.closure

    def plans(self, keys):
        if _current() is not self.cache:
            return self.original.plans(keys)
        keys = tuple(dict.fromkeys(keys))
        if any(type(key) is not T.SourceTemplateKey or key not in self.interfaces['nodes'] for key in keys):
            raise ValueError('world material template key is absent or foreign')
        self.validate_parents()
        stored = self.original._plans
        wanted = tuple(key for key in keys if key not in stored)
        if wanted:
            stored.update(_read(self.original, wanted, self.cache))
        return {key: stored[key] for key in keys}


def source_body_access(material):
    """Adapt an original ordinary-source reader inside an explicit reuse scope."""
    active = _current()
    if active is None or type(material) is not WorldMaterial:
        return material
    return _WorldBodyAccess(material, active)
