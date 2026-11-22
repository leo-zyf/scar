"""Observed code correspondence."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass, fields
from hashlib import sha256
from pathlib import Path
import re
import sys
import gzip
import json
import time
from types import CodeType
from scar.analysis.call_trace_reader import CallOperandTrace
from scar.analysis.static.compilation import StaticCodeEvidence
from scar.ir.program.call_edges import CodeCapture
from scar.ir.program.providers import ProviderArtifact
from scar.ir.program.source_call_bindings import SourceBodyNamespace
from scar.ir.program.source_subjects import ModuleUnitSource, SourceCompilationCodeRef
from scar.ir.record_codec import decode, encode, loads
from scar.ir.semantics_v2 import SourceSnapshot
from scar.trace.code_snapshot import code_key, describe_code
from scar.trace.lean_capture import LeanCapturePlan
from contextlib import nullcontext
from copy import deepcopy
from scar.artifacts import _json
from scar.analysis import compiler_stage_cache
from scar.ir.program import source_productions as P, normal_graph as F, compiler_symbols as CV, compiler_symbols as BV, compiler_symbols as NV, compiler_symbols as TV, compiler_symbols as SV
from scar.ir.program.source_declarations import SourceTemplateGraph, SourceTemplateKey
from scar.ir.program.source_protocols import UnitSourceInterfaces


# Fresh, code-scope-only correspondence to retained static source.
#
# A current file is material, not a capture-time module version. Matching a
# compiled child never identifies its creator, globals, defaults or class object.
# Saved correspondence records are audit DTOs, not reusable admission tokens.
_ORIGINS = ('HISTORICAL_SOURCE_SNAPSHOT', 'CURRENT_STATIC_SOURCE_CANDIDATE')
_SCOPE = 'MATCHED_CODE_SCOPE_ONLY'


@dataclass(frozen=True, slots=True)
class SourceScopeAuthority:
    run_scope: str
    capture: CodeCapture
    source: SourceSnapshot
    historical_snapshot: SourceSnapshot | None
    plan: ProviderArtifact | None
    source_origin: str
    scope: str = _SCOPE

    def __post_init__(self):
        if (type(self.run_scope) is not str or re.fullmatch(r'[0-9a-f]{64}', self.run_scope) is None
                or type(self.capture) is not CodeCapture or type(self.source) is not SourceSnapshot
                or type(self.source_origin) is not str or self.source_origin not in _ORIGINS
                or self.scope != _SCOPE or type(self.scope) is not str):
            raise ValueError('code correspondence needs original capture/current source and fixed scope')
        self.capture.code.__post_init__()
        self.capture.anchor.__post_init__()
        if self.capture.code.filename != self.source.path:
            raise ValueError('code correspondence changed its original source locator')
        if self.source_origin == 'HISTORICAL_SOURCE_SNAPSHOT':
            if (type(self.historical_snapshot) is not SourceSnapshot or type(self.plan) is not ProviderArtifact
                    or self.historical_snapshot.path != self.source.path
                    or self.historical_snapshot.fingerprint != self.source.fingerprint):
                raise ValueError('historical source scope needs its matching original snapshot/plan')
            self.plan.__post_init__()
        elif self.historical_snapshot is not None or self.plan is not None:
            raise ValueError('current source candidate cannot invent a historical whole-file snapshot')


@dataclass(frozen=True, slots=True)
class CodeScopeCorrespondence(SourceScopeAuthority):
    body: SourceBodyNamespace | None = None
    compilation: SourceCompilationCodeRef | None = None

    def __post_init__(self):
        SourceScopeAuthority.__post_init__(self)
        if type(self.body) is not SourceBodyNamespace or type(self.compilation) is not SourceCompilationCodeRef:
            raise TypeError('code/body correspondence needs original body and static compilation reference')
        self.body.__post_init__()
        self.compilation.__post_init__()
        if (self.body.definition != self.compilation.definition
                or self.compilation.code_layout_key != self.capture.code.key):
            raise ValueError('correspondence changed its original definition or matched Code scope')
        subject = self.compilation.source
        if type(subject) is ModuleUnitSource:
            if self.body.source is not None or subject.snapshot != self.source:
                raise ValueError('module Code correspondence changed its static source subject')
        elif (subject != self.body.source or subject.path != self.source.path
                or subject.fingerprint != self.source.fingerprint):
            raise ValueError('child Code correspondence changed its original static source occurrence')


class ObservedCodeSources:
    """One fresh trace/profile owner; code matches grant no environment facts.

    The supplied trace/run are the existing normal reader results. This owner
    checks membership, compiler profiles and original lean-plan receipts without
    repeating the frame/operand/target scan or accepting validation booleans.
    """
    def __init__(self, trace, run_scope):
        if (type(trace) is not CallOperandTrace or type(run_scope) is not str
                or re.fullmatch(r'[0-9a-f]{64}', run_scope) is None):
            raise TypeError('observed source scopes require a normal typed trace and original run scope')
        self.trace, self.run_scope = trace, run_scope
        self._captures = set(trace.frame_scan.captures)
        self._metadata = {row.process_id: row for row in trace.metadata}
        if len(self._metadata) != len(trace.metadata):
            raise ValueError('source-scope compiler metadata has duplicate processes')
        self._snapshots, self._plans, self._roots = {}, {}, OrderedDict()
        directory = Path(trace.directory)
        if not directory.is_absolute() or directory.resolve() != directory or directory.is_symlink():
            raise ValueError('observed source scopes require the original canonical trace directory')
        total = 0
        for metadata in trace.frame_scan.metadata:
            wire = metadata.get('capture_plan')
            if wire is None:
                continue
            process = metadata['process_id']
            receipt = decode(ProviderArtifact, wire)
            path = Path(receipt.uri)
            if (path != directory / f'lean-plan.{process}.json' or path.resolve() != path
                    or path.is_symlink() or not path.is_file() or receipt.size_bytes > 16_000_000
                    or total + receipt.size_bytes > trace.frame_budget.max_raw_bytes):
                raise ValueError('source-scope plan differs from its original namespace/byte bound')
            with path.open('rb') as stream:
                raw = stream.read(receipt.size_bytes + 1)
            if len(raw) != receipt.size_bytes or sha256(raw).hexdigest() != receipt.sha256:
                raise ValueError('source-scope original plan bytes/SHA differ')
            total += len(raw)
            plan = LeanCapturePlan.from_dict(loads(raw), max_codes=trace.frame_budget.max_codes)
            if plan.namespace_root != trace.project_root:
                raise ValueError('source-scope original plan namespace policy differs')
            if process in self._plans:
                raise ValueError('source-scope process has duplicate original plans')
            self._plans[process] = receipt
            for body in plan.bodies:
                key = process, body.source.path
                previous = self._snapshots.setdefault(key, body.source)
                if previous != body.source:
                    raise ValueError('one original plan path has conflicting whole-file snapshots')

    def require_source(self, captured, current_snapshot):
        """Check material/profile scope; not itself a compiled-code match."""
        if type(captured) is not CodeCapture or captured not in self._captures:
            raise ValueError('source correspondence requires the retained original CodeCapture/anchor')
        if type(current_snapshot) is not SourceSnapshot or current_snapshot.path != captured.code.filename:
            raise ValueError('source correspondence changed the captured filename/current snapshot')
        metadata = self._metadata.get(captured.process_id)
        if (metadata is None or metadata.implementation != sys.implementation.name
                or metadata.python != sys.version):
            raise ValueError('CAPTURE_COMPILER_PROFILE_UNSUPPORTED')
        historical = self._snapshots.get((captured.process_id, current_snapshot.path))
        if historical is not None and historical.fingerprint != current_snapshot.fingerprint:
            raise ValueError('CAPTURE_SOURCE_VERSION_CHANGED: original whole-file snapshot differs')
        return SourceScopeAuthority(self.run_scope, captured, current_snapshot, historical,
            None if historical is None else self._plans[captured.process_id],
            _ORIGINS[0] if historical is not None else _ORIGINS[1])

    def require(self, captured, body, code, compilation):
        """Verify this retained body/compiler path, never a parent environment."""
        if type(code) is not CodeType or type(compilation) is not SourceCompilationCodeRef:
            raise TypeError('code correspondence requires the actual static Code and typed compilation')
        namespace, snapshot = body.namespace, body.metadata.source
        if type(namespace) is not SourceBodyNamespace or type(snapshot) is not SourceSnapshot:
            raise TypeError('code correspondence requires original body namespace/source records')
        authority = self.require_source(captured, snapshot)
        compilation.__post_init__()
        if (namespace.region != body.graph.root or namespace.entry != body.metadata.entry
                or compilation.definition != namespace.definition
                or compilation.compiler_implementation != sys.implementation.name
                or compilation.compiler_version != tuple(sys.version_info[:3])
                or compilation.compiler_python != sys.version
                or compilation.compiler_cache_tag != sys.implementation.cache_tag):
            raise ValueError('source-scope body/definition/compiler profile differs')
        text = body.context.docs['source']['text']
        if type(text) is not str or 'sha256:' + sha256(text.encode()).hexdigest() != snapshot.fingerprint:
            raise ValueError('source-scope current retained text differs from its exact fingerprint')
        root = self._roots.get(snapshot)
        if root is None:
            root = compile(text, snapshot.path, 'exec', flags=0, dont_inherit=True, optimize=0)
            self._roots[snapshot] = root
            while len(self._roots) > 16:
                self._roots.popitem(last=False)
        else:
            self._roots.move_to_end(snapshot)
        selected = root
        for position in compilation.code_path:
            if position >= len(selected.co_consts) or type(selected.co_consts[position]) is not CodeType:
                raise ValueError('source-scope compiler path does not select an original child Code')
            selected = selected.co_consts[position]
        if (describe_code(selected) != encode(captured.code)
                or describe_code(code) != encode(captured.code)):
            raise ValueError('CAPTURE_CODE_DESCRIPTOR_MISMATCH')
        evidence = StaticCodeEvidence(snapshot, body.context.unit['namespace_root'], namespace.definition,
            compilation.source, compilation.code_path, selected, code_key(selected),
            sys.implementation.name, tuple(sys.version_info[:3]), sys.version,
            sys.implementation.cache_tag, root_code_layout_key=code_key(root))
        if evidence.to_reference() != compilation:
            raise ValueError('source-scope original compilation differs from retained-source replay')
        if body.document.get('module') is not None:
            original = body.module.declaration.compilation
            if original != compilation:
                raise ValueError('source-scope original module CodeRef differs')
        elif body.document.get('class') is not None:
            if body.class_entry.compilation != compilation:
                raise ValueError('source-scope original class CodeRef differs')
        else:
            wire = body.document.get('static_compilation')
            expected = evidence.to_dict()
            if type(wire) is dict and 'layout' not in wire:
                expected.pop('layout')
            if encode(expected) != wire:
                raise ValueError('source-scope original function compilation envelope differs')
        return CodeScopeCorrespondence(authority.run_scope, authority.capture, authority.source,
            authority.historical_snapshot, authority.plan, authority.source_origin,
            authority.scope, namespace, compilation)


# Local F0--F4 material for retained compilation text and a replay recipe.
#
# Replay compiles but never executes the payload. Its parameters are a compatible
# analysis choice, not a reconstruction of the caller's original compile request.
# This format has no exec/read events, namespace solver or importable module grant.
SCHEMA = 'scar.observed-compiled-local-material'
FACTS_FORMAT = 'OBSERVED_COMPILED_LOCAL_JSON'
LOWERING = 'CODE_EQUIVALENT_REPLAY_NOT_ORIGINAL_REQUEST'
_CLAIMS = ('target_executed', 'original_request_proven', 'runtime_namespace_proven',
    'source_body_activated', 'global_namespace_resolved', 'first_stage_complete')
# These v1 producers wrote the same canonical plans and receipts.
_COMPATIBLE_V1_PRODUCERS = (
    '6c56d389738e99cb01ae4d05e32b58b3d36b0b7b9105b2d8c38f65a7bf779050',
    'b64e9ba9651f689a0f87c4f4a04a0a8069aae4d5842b23b141847357bf81bf18',
)


def _pins():
    from scar.analysis.source_f4_variants import pins
    return {**pins(), 'compiler_stage_cache':
        sha256(Path(compiler_stage_cache.__file__).read_bytes()).hexdigest()}


def _stamp(path):
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _recipe(value):
    names = {'filename', 'mode', 'flags', 'dont_inherit', 'optimize'}
    if (type(value) is not dict or set(value) != names
            or type(value['filename']) is not str or value['mode'] not in ('exec', 'eval', 'single')
            or type(value['flags']) is not int or value['flags'] < 0
            or value['dont_inherit'] is not True or type(value['optimize']) is not int
            or value['optimize'] not in (0, 1, 2)):
        raise ValueError('explicit compatible filename/mode/flags/dont_inherit=True/optimize recipe required')
    return dict(value)


def _write_document(root, name, document):
    from scar.artifacts import _durable_bytes
    raw = _json(document); payload = gzip.compress(raw, compresslevel=3, mtime=0)
    path = name + '.json.gz'; _durable_bytes(root/path, payload, replace=False)
    return dict(path=path, bytes=len(payload), sha256=sha256(payload).hexdigest(),
        decoded_bytes=len(raw), decoded_sha256=sha256(raw).hexdigest())


def _locator(root, receipt, schema):
    from scar.ir.program.source_references import SourceArtifactLocator
    artifact = ProviderArtifact(str(root/receipt['path']), receipt['sha256'], receipt['bytes'], schema)
    return SourceArtifactLocator(artifact, receipt['decoded_sha256'], receipt['decoded_bytes'], 'gzip-canonical-json')


def build_local_material(payload, recipe, out, *, raw=None, worker=None):
    """Seal one pure local graph; an optional native worker may serve a batch.

    ``payload`` is retained Unicode compilation text or bytes. ``raw`` can retain
    its known original bytes; absent those, UTF-8 is an analysis representation.
    Paths under ``out`` label these retained snapshots, not original file reads.
    """
    from scar.analysis.source_bytes import source_bytes_document, decoded_source_bytes
    from scar.analysis.source_tokens import source_tokens_document
    from scar.analysis.source_syntax import source_syntax_document
    from scar.analysis.source_role_materials import source_roles_document
    from scar.analysis.source_context_variants import context_variant_document
    from scar.analysis.source_binding_variants import binding_variant_document
    from scar.analysis.source_symbol_variants import symbol_variant_document, SymbolVariantWorker
    from scar.analysis.source_topology_variants import topology_variant_document
    from scar.analysis.source_compiler_symbol_variants import compiler_symbol_variant_document
    from scar.analysis.source_f4_variants import construct_graph
    from scar.analysis.source_symbols import _digest
    from scar.analysis.call_compilation import supported_future_flag_mask
    from scar.artifacts import _durable_bytes, _durable_mkdir
    from scar.ir.program.source_bytes import SourceBytesProfile
    from scar.ir.program.source_roles import SourceRolesMaterial
    from scar.ir.program.compiler_symbols import SymbolSourceParent, SymbolTextInput
    from scar.ir.program.source_bytes import TextOpenInterpretation
    from scar.ir.program.compiler_symbols import EffectiveCompileContext
    from scar.trace.code_snapshot import describe_code

    started = time.monotonic(); recipe = _recipe(recipe); out = Path(out).absolute()
    if out.resolve() != out or out.exists():
        raise ValueError('fresh canonical local compiled material output required')
    if type(payload) is bytes:
        if raw is not None and raw != payload:
            raise ValueError('retained raw payloads differ')
        raw = payload; payload = None
    elif type(payload) is not str:
        raise TypeError('retained compilation payload must be str or bytes')
    original_raw_known = raw is not None
    if raw is None: raw = payload.encode('utf-8')
    if type(raw) is not bytes: raise TypeError('raw source snapshot must be immutable bytes')
    _durable_mkdir(out)
    snapshot = out/'source.py'; _durable_bytes(snapshot, raw, replace=False)
    f0 = source_bytes_document(raw, path=str(snapshot), namespace_root=str(out))
    interpreted = decoded_source_bytes(f0)
    if payload is None: payload = interpreted
    if type(payload) is not str or interpreted != payload:
        raise ValueError('F0 source decoding must equal the retained compilation text')
    f1 = source_tokens_document(f0)
    f2 = source_syntax_document(f0, f1, mode=recipe['mode'])
    roles = source_roles_document(f0, f1, f2)
    documents = dict(source_bytes=f0, source_tokens=f1, source_syntax=f2, source_roles=roles)
    materials = {name:_write_document(out, name, doc) for name,doc in documents.items()}
    parents = tuple(SymbolSourceParent(name, _locator(out, materials[name], documents[name]['schema']),
        _digest(documents[name])) for name in documents)
    code = compile(payload, recipe['filename'], recipe['mode'], flags=recipe['flags'],
        dont_inherit=True, optimize=recipe['optimize'])
    context = EffectiveCompileContext(recipe['filename'], recipe['mode'], recipe['flags'], True,
        0, recipe['flags'], code.co_flags & supported_future_flag_mask(), code.co_flags,
        recipe['optimize'], decode(SourceBytesProfile, f0['profile']), lowering=LOWERING)
    source_unit = sha256(_json(dict(raw_sha256=sha256(raw).hexdigest(),
        text_utf8_sha256=sha256(payload.encode('utf-8')).hexdigest()))).hexdigest()
    # Reuse immutable analyzer inputs only within this owned compilation.
    # Native worker observations and reader validation results stay uncached.
    with compiler_stage_cache.compiler_stage_session():
        cdoc = context_variant_document(source_unit, f0, f1, f2, roles, context, payload)
        bdoc = binding_variant_document(f0, f1, f2, roles, cdoc, payload)
        text = SymbolTextInput(parents[0].source, parents[0].document_sha256,
            sha256(raw).hexdigest(), len(raw),
            TextOpenInterpretation(f0['decoding']['encoding'], 'strict', '', True, False,
                rule='RETAINED_COMPILATION_PAYLOAD_NO_RUNTIME_READ'),
            sha256(payload.encode('utf-8')).hexdigest(), len(payload), (LOWERING,), ())
        with (SymbolVariantWorker() if worker is None else nullcontext(worker)) as active:
            ndoc = symbol_variant_document(cdoc, f0, f1, f2, roles, payload, (text,), active, parents=parents)
            tdoc = topology_variant_document(cdoc, ndoc, f0, f1, f2, roles, payload, active, parents=parents)
            sdoc = compiler_symbol_variant_document(cdoc, bdoc, ndoc, tdoc,
                f0, f1, f2, roles, payload, active, parents=parents)
            adapter = dict(active.parent)
    context_variant = decode(CV.SourceContextVariant, cdoc)
    binding = decode(BV.SourceBindingVariant, bdoc)
    native = decode(NV.SourceSymbolVariant, ndoc)
    topology = decode(TV.SourceTopologyVariant, tdoc)
    symbols = decode(SV.SourceCompilerSymbolVariant, sdoc)
    graph, records, dispositions, edges, facts = construct_graph(context_variant, binding,
        native, topology, symbols, f0, f2, decode(SourceRolesMaterial, roles))
    documents.update(context_variant=cdoc, binding_variant=bdoc, native_variant=ndoc,
        topology_variant=tdoc, compiler_symbol_variant=sdoc,
        source_contexts=encode(context_variant.material), source_bindings=encode(binding.bindings),
        source_declarations=encode(binding.declarations), source_symbols=encode(native.material),
        compiler_topology=encode(topology.material), compiler_symbols=encode(symbols.material))
    for name,document in documents.items():
        if name not in materials: materials[name] = _write_document(out, name, document)
    tables = dict(graph=_write_document(out, 'graph', encode(graph, record_tags=P.PRODUCTION_RECORD_TAGS)),
        facts=_write_document(out, 'facts', encode(facts, record_tags=P.PRODUCTION_RECORD_TAGS)),
        dispositions=_write_document(out, 'dispositions', dict(records=dispositions)),
        normal_edges=_write_document(out, 'normal_edges', dict(records=encode(tuple(edges), record_tags=P.PRODUCTION_RECORD_TAGS))))
    for domain, rows in records.items():
        tables[domain.value] = _write_document(out, domain.value, dict(records=rows))
    # These three auxiliary interface tables really are empty for plain lambda
    # functions. Richer regions need a separate honest witness-aware projector.
    empty_interfaces = (not facts.iterations
        and all(c.kind is not F.ContainerKind.MAPPING for c in facts.containers)
        and all(f.environment.kind is P.DeferredBodyKind.FUNCTION for f in facts.functions))
    if empty_interfaces:
        tables['interfaces'] = _write_document(out, 'interfaces', encode(UnitSourceInterfaces(facts.unit, (), (), ())))
    counts = {field.name:len(getattr(facts, field.name)) for field in fields(F.UnitFacts)
        if type(getattr(facts, field.name)) is tuple}
    model = dict(key=facts.unit, source_unit=source_unit, path=str(snapshot), namespace_root=str(out),
        raw_sha256=sha256(raw).hexdigest(), decoded_sha256=sha256(payload.encode('utf-8')).hexdigest(),
        artifacts=materials, module_names=[], module_mappings=[],
        source_origin='RETAINED_COMPILATION_PAYLOAD', raw_input_original=original_raw_known)
    report = dict(schema=SCHEMA, schema_version=1, status='LOCAL_COMPILED_F4_MATERIAL_BUILT',
        unit=facts.unit, source_unit=source_unit, model_unit=model, recipe=recipe, lowering=LOWERING,
        replayed_root_code=describe_code(code), context=encode(context),
        materials=materials, tables=tables, counts=counts, native_adapter=adapter,
        snapshot=dict(path='source.py',bytes=len(raw),sha256=sha256(raw).hexdigest()),
        local_interfaces='EMPTY_BY_ACTUAL_FACTS' if empty_interfaces else 'NOT_PROJECTED',
        pins=_pins(), implementation_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        claims={name:False for name in _CLAIMS}, seconds=time.monotonic()-started)
    _durable_bytes(out/'result.json', _json(report), replace=False)
    ObservedCompiledLocalMaterial(out)
    return report


class ObservedCompiledLocalMaterial:
    """Receipt-bound shared F4 graph, independent of any source catalog."""
    def __init__(self, path):
        self.path = self.root = Path(path).absolute()
        if self.path.resolve() != self.path: raise ValueError('canonical local material required')
        self._report_path = self.path/'result.json'
        self._report_raw = self._report_path.read_bytes()
        self.report = report = json.loads(self._report_raw)
        if ((report.get('schema'), report.get('schema_version')) != (SCHEMA, 1)
                or report['implementation_sha256'] not in (*_COMPATIBLE_V1_PRODUCERS,
                    sha256(Path(__file__).read_bytes()).hexdigest())
                or report['pins'] != _pins() or report['lowering'] != LOWERING
                or set(report['claims']) != set(_CLAIMS) or any(report['claims'].values())):
            raise ValueError('local compiled material producer or contract differs')
        _recipe(report['recipe'])
        self.unit, self.source_unit = report['unit'], report['source_unit']
        self._report_snapshot = deepcopy(report)
        self._parents = {self._report_path:self._report_raw}
        self._stamps = {}; self._plans = {}
        for descriptor in (*report['materials'].values(), *report['tables'].values(), report['snapshot']):
            self._read_raw(descriptor)
        adapter = Path(report['native_adapter']['path'])
        raw = adapter.read_bytes()
        if sha256(raw).hexdigest() != report['native_adapter']['sha256']:
            raise ValueError('local compiled native adapter receipt differs')
        self._parents[adapter] = raw; self._stamps[adapter] = _stamp(adapter)
        self.graph = decode(SourceTemplateGraph, self._document(report['tables']['graph']), record_tags=P.PRODUCTION_RECORD_TAGS)
        if self.graph.unit != self.unit: raise ValueError('local graph unit differs')
        self.interfaces = dict(nodes={node.key:node for node in self.graph.templates},
            links={(link.caller,link.step):link for link in self.graph.links})
        self.result_artifact = ProviderArtifact(str(self.path/'result.json'), sha256(self._report_raw).hexdigest(), len(self._report_raw), SCHEMA)
        self.fact_store = LocalCompiledFacts(self)
        self.validate_parents()

    def _read_raw(self, descriptor):
        relative = Path(descriptor['path']); path = self.path/relative
        if relative.is_absolute() or '..' in relative.parts or path.resolve() != path or path.is_symlink():
            raise ValueError('local compiled artifact escapes its root')
        before = _stamp(path); raw = path.read_bytes()
        if (before != _stamp(path) or len(raw) != descriptor['bytes']
                or sha256(raw).hexdigest() != descriptor['sha256']):
            raise ValueError('local compiled artifact receipt differs')
        self._stamps[path] = before
        return raw

    def _document(self, descriptor):
        raw = gzip.decompress(self._read_raw(descriptor))
        if len(raw) != descriptor['decoded_bytes'] or sha256(raw).hexdigest() != descriptor['decoded_sha256']:
            raise ValueError('local compiled decoded artifact receipt differs')
        return json.loads(raw)

    def material_document(self, name):
        self.validate_parents()
        return self._document(self.report['materials'][name])

    def validate_parents(self):
        if (self.path/'result.json' != self._report_path or self.report != self._report_snapshot
                or self._report_path.read_bytes() != self._report_raw):
            raise ValueError('local compiled report changed')
        if any(path != self._report_path and path.read_bytes() != raw for path,raw in self._parents.items()):
            raise ValueError('local compiled parent changed')
        if any(_stamp(path) != stamp for path,stamp in self._stamps.items()):
            raise ValueError('local compiled material table changed')
        return len(self._stamps)+len(self._parents)

    def guard_stamps(self):
        self.validate_parents()
        return {path:_stamp(path) for path in self.guard_paths()}

    def guard_paths(self):
        return tuple(dict.fromkeys((*self._parents, *self._stamps)))

    def plans(self, keys):
        from scar.analysis.observed.dependency_export import PlanRowCache
        from scar.analysis.source_productions import _digest, read_production_plan
        keys = tuple(dict.fromkeys(keys)); self.validate_parents()
        if any(type(key) is not SourceTemplateKey or key not in self.interfaces['nodes'] for key in keys):
            raise ValueError('foreign local compiled template key')
        wanted = defaultdict(dict)
        for key in keys:
            if key not in self._plans:
                node = self.interfaces['nodes'][key]; wanted[node.origin.domain.value][node.origin.line] = key
        for domain, lines in wanted.items():
            owner = getattr(self, 'catalog', self)
            cache = getattr(owner, '_compiled_plan_rows', None)
            if cache is None:
                cache = owner._compiled_plan_rows = PlanRowCache()
            descriptor = self.report['tables'][domain]
            table_path = self.path / descriptor['path']
            cache_key = table_path, descriptor['sha256'], descriptor['decoded_sha256']
            records = cache.load(cache_key, descriptor, lambda: gzip.decompress(self._read_raw(descriptor)))
            for line,key in lines.items():
                node = self.interfaces['nodes'][key]; document = records.row(line)
                if _digest(document) != node.origin.plan_sha256:
                    raise ValueError('local compiled plan digest differs')
                plan = read_production_plan(document)
                if (plan.source.occurrence != key.occurrence or plan.purpose is not key.purpose
                        or plan.root != node.root or plan.input != self.graph.input or plan.profile != self.graph.profile):
                    raise ValueError('local compiled plan identity/input/profile differs')
                self._plans[key] = plan
        self.validate_parents()
        return {key:self._plans[key] for key in keys}


class LocalCompiledFacts:
    """The actual local UnitFacts JSON and its metadata, never a facts6 store."""
    def __init__(self, material):
        self.material = material; self.path = material.path; self.report = material.report
        self.raw = material._report_raw; self.unit = material.unit; self.source_unit = material.source_unit
        descriptor = self.report['tables']['facts']
        self.row = dict(unit=self.unit, path=self.report['model_unit']['path'],
            facts=descriptor['path'], facts_format=FACTS_FORMAT, counts=self.report['counts'],
            **{name:value for name,value in descriptor.items() if name != 'path'})
        self.rows = {self.unit:self.row}; self.model_units = {self.unit:self.report['model_unit']}
        self.sources = {self.row['path']:self.unit}; self.source_units = {self.row['path']:(self.unit,)}
        self.offsets = {self.unit:0}; self.size = self.report['counts']['ports']; self.loaded = {}
        self.original_source_units = {}; self.importable_units = frozenset()
        self._source_parents = {self.unit:self}; self.parents = (); self.unit_parent = {}
        self.supplemental_sources = ()

    def validate_parents(self): return self.material.validate_parents()
    def source_parent_for(self, identity): self._identity(identity); return self
    def model_for(self, identity): self._identity(identity); return self.path
    def units_for_source(self, source): return tuple(self.source_units.get(str(Path(source).absolute()), ()))

    def _identity(self, identity):
        if identity != self.unit: raise KeyError(identity)

    def document(self, identity=None):
        self._identity(self.unit if identity is None else identity); self.validate_parents()
        document = self.material._document(self.report['tables']['facts'])
        if document['unit'] != self.unit: raise ValueError('local compiled fact unit differs')
        return document

    def load(self, identity=None):
        identity = self.unit if identity is None else identity; self._identity(identity); self.validate_parents()
        if identity not in self.loaded:
            unit = decode(F.UnitFacts, self.document(identity), record_tags=P.PRODUCTION_RECORD_TAGS)
            if unit.root_scope is None or unit.root_scope.owner.unit != self.unit:
                raise ValueError('local compiled facts have no original root')
            if len(unit.ports) != self.size: raise ValueError('local compiled port denominator differs')
            self.loaded[identity] = unit
        return self.loaded[identity]

    def locate(self, point):
        if type(point) is not int or not 0 <= point < self.size: raise ValueError('local fact point outside catalog')
        return self.unit, point


__all__ = [
    'ObservedCodeSources',
    'SourceScopeAuthority',
    'CodeScopeCorrespondence',
    'SCHEMA',
    'FACTS_FORMAT',
    'LOWERING',
    'build_local_material',
    'ObservedCompiledLocalMaterial',
    'LocalCompiledFacts',
]
