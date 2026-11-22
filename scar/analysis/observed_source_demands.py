"""Observed source demands."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import sys
import time
import ast
import hashlib
from types import CodeType
from scar.analysis.observed_call_bindings import _json, _read_trace
from scar.analysis.program_dependency_session import _stamp
from scar.analysis.program_observed_dependencies import _inputs
from scar.analysis.python_call_targets import read_python_call_targets
from scar.analysis.static_calls import StaticCallBudget, StaticCallSession
from scar.analysis.static_sources import (
    _ByteBudget,
    _Changed,
    _has_symlink_component,
    _read_source,
)
from scar.ir.program.call_operands import LEAN_PORTS_RULE
from scar.ir.program.providers import ProviderArtifact
from scar.ir.program.static_sources import StaticSourceFile
from scar.ir.record_codec import decode, encode, loads
from scar.trace.code_snapshot import code_key, describe_code
from scar.trace.lean_capture import LeanCapturePlan
from dataclasses import dataclass
from scar.analysis.observed_code_correspondence import ObservedCodeSources, SourceScopeAuthority
from scar.analysis.static_compilation import StaticCodeEvidence, _scope_nodes
from scar.analysis.static_rules import _at_path, plan_static_entry
from scar.analysis.static_body import null_body_diagnostic
from scar.ir.program.call_edges import CodeCapture
from scar.ir.program.source_references import SourceArtifactLocator
from scar.ir.program.source_subjects import ModuleUnitSource, SourceCompilationCodeRef
from scar.ir.semantics_v2 import SourceSnapshot
from scar.ir.v2.common import SourceReference
from scar.ir.v2.ids import OperationDefinitionID


# Admit missing source materials selected by original captured Code records.
#
# A filename locates material only. A historical snapshot must agree when one
# exists. Otherwise a current static source candidate needs an actual selected
# code and its unique complete compiler correspondence. Neither path establishes
# file execution, import/class identity, runtime globals or optimization legality.
def expand_observed_sources(model, trace, out, budget=None, *, max_source_files=8,
                            max_source_bytes=8 * 1024 * 1024):
    """Append only missing files with a receipt-bound source/compiler match.

    File/byte ceilings bound attempted source reads, including reservations
    for failed reads. Unsupported source-version profiles remain explicit gaps.
    Every input stream is closed before the existing static publisher runs.
    """
    started = time.monotonic()
    limits = StaticCallBudget() if budget is None else budget
    if type(limits) is not StaticCallBudget:
        raise TypeError('observed source demands require StaticCallBudget')
    if any(type(value) is not int or value < 1 for value in (max_source_files, max_source_bytes)):
        raise ValueError('observed source file/byte budgets must be positive exact integers')
    model, out = Path(model).absolute(), Path(out).absolute()
    if (model.resolve() != model or model.is_symlink() or out.resolve() != out or out.is_symlink()
            or out.is_relative_to(model)
            or out.exists() and (not out.is_dir() or any(out.iterdir()))):
        raise FileExistsError('source-demand reports require a fresh canonical directory outside the model')
    manifest_path = model / 'manifest.json'
    before_model = _stamp(manifest_path)
    if (manifest_path.is_symlink() or manifest_path.resolve() != manifest_path
            or before_model[2] > limits.max_manifest_bytes):
        raise ValueError('source demands require the original bounded model manifest')
    with manifest_path.open('rb') as stream:
        original = stream.read(limits.max_manifest_bytes + 1)
    if len(original) != before_model[2] or _stamp(manifest_path) != before_model:
        raise ValueError('source model changed during demand discovery')
    manifest = loads(original)
    if (type(manifest) is not dict or manifest.get('schema') != 'scar.static-program'
            or type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 1
            or type(manifest.get('units')) is not list):
        raise ValueError('source demands require the original static program model')
    prior = sha256(original).hexdigest()
    trace_path = Path(getattr(trace, 'directory', trace)).absolute()
    if trace_path.resolve() != trace_path or trace_path.is_symlink():
        raise ValueError('source demands require the original canonical trace directory')
    trace_before = _inputs(trace_path)
    trace = _read_trace(trace)
    observed = read_python_call_targets(trace)
    actual_codes = {(target.caller.process_id, target.code) for target in observed.targets if target.code is not None}
    actual_codes.update((entry.callee.process_id, entry.code) for entry in observed.entries)
    present = {unit['path'] for unit in manifest['units']}
    groups = defaultdict(list)
    for capture in trace.frame_scan.captures:
        if capture.code.filename not in present:
            groups[capture.code.filename].append(capture)
    metadata = {item.process_id: item for item in trace.metadata}
    frame_metadata = {item['process_id']: item for item in trace.frame_scan.metadata}
    snapshots, plan_receipts, issues = {}, {}, []
    input_bytes = len(original) + sum(item.raw_bytes for item in (*trace.frame_scan.files, *trace.files))

    def check():
        if time.monotonic() - started > limits.max_seconds:
            raise TimeoutError('observed source demand wall budget exhausted')

    def issue(kind, filename, detail, **fields):
        issues.append({'kind': kind, 'file': filename, 'detail': detail, **fields})

    # _read_trace has independently checked these plan declarations against
    # the original frame/operand receipts. Recheck bytes before using content.
    for process in sorted({capture.process_id for values in groups.values() for capture in values}):
        check()
        if process not in metadata or metadata[process].rule != LEAN_PORTS_RULE:
            continue
        receipt = decode(ProviderArtifact, frame_metadata[process]['capture_plan'])
        path = Path(receipt.uri)
        if (path != trace_path / f'lean-plan.{process}.json' or path.is_symlink()
                or path.resolve() != path or receipt.size_bytes > 16_000_000
                or input_bytes + receipt.size_bytes > limits.max_input_bytes):
            raise ValueError('observed source plan changed its original artifact namespace')
        with path.open('rb') as stream:
            raw = stream.read(receipt.size_bytes + 1)
        if len(raw) != receipt.size_bytes or sha256(raw).hexdigest() != receipt.sha256:
            raise ValueError('observed source plan changed its original bytes')
        input_bytes += len(raw)
        plan = LeanCapturePlan.from_dict(loads(raw), max_codes=trace.frame_budget.max_codes)
        if plan.namespace_root != trace.project_root:
            raise ValueError('source demand plan changed its original namespace policy')
        plan_receipts[process] = receipt
        for body in plan.bodies:
            snapshots[process, body.source.path] = body.source

    sources, demands = [], []
    attempted = charged = retained = 0
    for filename, captures in sorted(groups.items()):
        check()
        path = Path(filename)
        candidates = [capture for capture in captures
            if capture.process_id in metadata
            and metadata[capture.process_id].implementation == sys.implementation.name
            and metadata[capture.process_id].python == sys.version
            and ((capture.process_id, filename) in snapshots
                 or (capture.process_id, capture.code.key) in actual_codes)]
        if not candidates:
            issue('CAPTURE_SOURCE_VERSION_UNRECORDED', filename,
                'No matching compiler profile with an original snapshot or actual selected code; incidental captures do not demand current source.')
            continue
        try:
            available = (path.is_absolute() and path.resolve() == path
                and _has_symlink_component(path) is None and path.suffix == '.py' and path.is_file())
            required = path.stat().st_size if available else None
        except (OSError, ValueError, RuntimeError):
            available, required = False, None
        if not available:
            issue('CAPTURE_SOURCE_FILE_UNAVAILABLE', filename,
                'The captured filename does not locate a canonical regular Python source file.')
            continue
        if attempted >= max_source_files:
            issue('OBSERVED_SOURCE_FILE_BUDGET', filename, 'The source-read file budget was reached.')
            continue
        if required > max_source_bytes - charged or input_bytes + required > limits.max_input_bytes:
            issue('OBSERVED_SOURCE_BYTE_BUDGET', filename, 'The file exceeds the remaining source-read byte budget.')
            continue
        attempted += 1
        charged += required
        input_bytes += required
        try:
            raw, text, encoding = _read_source(path, required)
            raw_sha, decoded_sha = sha256(raw).hexdigest(), sha256(text.encode()).hexdigest()
            matched_versions = []
            for capture in candidates:
                snapshot = snapshots.get((capture.process_id, filename))
                if snapshot is not None and snapshot.fingerprint != 'sha256:' + decoded_sha:
                    issue('CAPTURE_SOURCE_VERSION_CHANGED', filename,
                        'The current file conflicts with this capture scope\'s original whole-file snapshot.',
                        process_id=capture.process_id, code=capture.code.key)
                    continue
                matched_versions.append(capture)
            if not matched_versions:
                continue
            compiled = compile(text, filename, 'exec', flags=0, dont_inherit=True, optimize=0)
            pending, codes = [((), compiled)], []
            while pending:
                compiler_path, code = pending.pop()
                codes.append((compiler_path, code))
                pending.extend(((*compiler_path, index), value) for index, value in enumerate(code.co_consts)
                    if type(value) is CodeType)
            proofs = []
            for capture in matched_versions:
                check()
                matches = [(compiler_path, code) for compiler_path, code in codes
                    if describe_code(code) == encode(capture.code)]
                if len(matches) != 1:
                    issue('CAPTURE_SOURCE_COMPILER_CODE_NOT_UNIQUE', filename,
                        'Complete captured/compiler descriptors have no unique original co_consts path.',
                        code=capture.code.key, candidate_count=len(matches))
                    continue
                compiler_path, _ = matches[0]
                snapshot = snapshots.get((capture.process_id, filename))
                historical = None if snapshot is None else encode(snapshot)
                plan = plan_receipts.get(capture.process_id)
                proofs.append({'capture': encode(capture), 'source_snapshot': historical,
                    'historical_snapshot': historical,
                    'source_origin': 'HISTORICAL_SOURCE_SNAPSHOT' if snapshot is not None else 'CURRENT_STATIC_SOURCE_CANDIDATE',
                    'plan': None if plan is None else encode(plan), 'compiler_path': list(compiler_path),
                    'root_code_layout_key': code_key(compiled), 'implementation': sys.implementation.name,
                    'python': sys.version, 'cache_tag': sys.implementation.cache_tag})
            if not proofs:
                continue
            source = StaticSourceFile(filename, str(path.parent), (), text, raw_sha,
                decoded_sha, encoding, len(raw), ())
            sources.append(source)
            retained += len(raw)
            demands.append({'source': source.to_manifest_dict(),
                'source_bytes': encode(ProviderArtifact(filename, raw_sha, len(raw), 'scar.observed-source-bytes')),
                'proofs': proofs, 'import_or_class_origin_established': False,
                'file_execution_established': False, 'runtime_globals_established': False})
        except (OSError, UnicodeError, SyntaxError, ValueError, _ByteBudget, _Changed) as error:
            issue('CAPTURE_SOURCE_MATERIAL_OPEN', filename, type(error).__name__ + ': ' + str(error))
    if _inputs(trace_path) != trace_before:
        raise ValueError('original trace/plan membership or bytes changed during source demands')
    if _stamp(manifest_path) != before_model:
        raise ValueError('original source model changed before material publication')
    provenance = {'schema': 'scar.observed-source-demands', 'schema_version': 1,
        'prior_model_manifest': encode(ProviderArtifact(str(manifest_path), prior, len(original), 'scar.static-program')),
        'trace': trace.directory, 'run_scope': observed.source_sha256,
        'capture_files': encode((*trace.frame_scan.files, *trace.files)),
        'plans': encode(tuple(plan_receipts.values())), 'demands': demands,
        'namespace_policy': 'INDEPENDENT_FILE_PARENT_WITHOUT_IMPORT_MAPPING',
        'target_executed': False, 'import_or_class_origin_established': False}
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'plan.json').open('xb') as stream:
        stream.write(_json({**provenance, 'issues': issues, 'readers_closed_before_publication': True}))
    # Readers above are eager and all files were opened through context
    # managers. Only this existing owner may now mutate the static model.
    old_keys = {unit['key'] for unit in manifest['units']}
    if sources:
        from scar.analysis.static_dependencies import append_static_source_units
        updated = append_static_source_units(model, tuple(sources), provenance=provenance,
            jobs=1, max_bytes=max_source_bytes, expected_manifest_sha256=prior)
    else:
        updated = manifest
    if _inputs(trace_path) != trace_before:
        raise ValueError('original trace changed while the closed-reader source publication was in progress')
    added = [unit for unit in updated['units'] if unit['key'] not in old_keys]
    published = sha256(manifest_path.read_bytes()).hexdigest()
    report = {'schema': 'scar.observed-source-expansion', 'schema_version': 1,
        'model': str(model), 'trace': trace.directory, 'run_scope': observed.source_sha256,
        'prior_manifest_sha256': prior, 'published_manifest_sha256': published,
        'added_files': [unit['path'] for unit in added], 'added_unit_keys': [unit['key'] for unit in added],
        'added_source_count': len(added), 'source_file_read_attempts': attempted,
        'charged_source_bytes': charged, 'retained_source_bytes': retained,
        'max_source_files': max_source_files, 'max_source_bytes': max_source_bytes,
        'issues': issues, 'plan': str(out / 'plan.json'), 'readers_closed_before_publication': True,
        'fresh_bindings_and_protocols_required': bool(added), 'target_executed': False,
        'whole_program_complete': False, 'analyzer_seconds': time.monotonic() - started}
    with (out / 'result.json').open('xb') as stream:
        stream.write(_json(report))
    return report


# Captured code selects missing static bodies through original compiler paths.
#
# Observation supplies a demand, not a replacement graph. Retained source is
# compiled without execution, lexical occurrences and complete code descriptors
# are checked, all readers close, and the ordinary static lowerer builds only
# previously unmaterialized entries. Existing canonical bodies are not replaced.
@dataclass(frozen=True, slots=True)
class ObservedBodyDemand:
    run_scope: str
    capture: CodeCapture
    unit_number: int
    entry_key: str
    lexical_entry: SourceArtifactLocator
    source_artifact: SourceArtifactLocator
    compiler: SourceCompilationCodeRef
    source_authority: SourceScopeAuthority

    def __post_init__(self):
        for value, kind in ((self.capture, CodeCapture), (self.lexical_entry, SourceArtifactLocator),
                (self.source_artifact, SourceArtifactLocator), (self.compiler, SourceCompilationCodeRef),
                (self.source_authority, SourceScopeAuthority)):
            if type(value) is not kind:
                raise TypeError('observed body demand needs exact ' + kind.__name__)
            post = getattr(value, '__post_init__', None)
            if post is not None:
                post()
        compiler_source = (self.compiler.source.snapshot if type(self.compiler.source) is ModuleUnitSource
                           else self.compiler.source)
        if (type(self.run_scope) is not str or len(self.run_scope) != 64
                or type(self.unit_number) is not int or self.unit_number < 0
                or type(self.entry_key) is not str or not self.entry_key.startswith('lexical:')
                or self.compiler.code_layout_key != self.capture.code.key
                or self.source_authority.run_scope != self.run_scope
                or self.source_authority.capture != self.capture
                or self.source_authority.source.path != compiler_source.path
                or self.source_authority.source.fingerprint != compiler_source.fingerprint):
            raise ValueError('observed body demand changed original captured/compiler identity')


def _span(value):
    return value.start_line, value.end_line, value.start_column, value.end_column


def _source(owner, number):
    unit = owner.units[number]
    catalog = owner._catalog(number)
    source = owner._read(unit['artifacts']['source'])
    if (source.get('schema') != 'scar.static-source-text' or source.get('schema_version') != 1
            or source.get('path') != unit['path'] or source.get('namespace_root') != unit['namespace_root']
            or source.get('decoded_sha256') != unit['decoded_sha256']
            or hashlib.sha256(source['text'].encode()).hexdigest() != unit['decoded_sha256']):
        raise ValueError('retained source text/namespace differs from its original unit receipt')
    text = source['text']
    tree = ast.parse(text, filename=unit['path'])
    root = compile(text, unit['path'], 'exec', flags=0, dont_inherit=True, optimize=0)
    scopes, issues = {}, []
    _scope_nodes(tree, root, (), ast.Module, scopes, issues)
    return catalog, tree, root, scopes, issues


def _entry_code(entry, tree, root, scopes, snapshot):
    node = _at_path(tree, entry['ast_path'])
    ids = entry.get('semantic_definition_ids') or ()
    if len(ids) != 1:
        raise LookupError('LEXICAL_ENTRY_DEFINITION_NOT_UNIQUE')
    definition = decode(OperationDefinitionID, ids[0])
    if entry['kind'] == 'MODULE' and type(node) is ast.Module:
        return definition, ModuleUnitSource(snapshot, entry['namespace_root'], definition), (), root
    expected = {'FUNCTION': ast.FunctionDef, 'ASYNC_FUNCTION': ast.AsyncFunctionDef, 'CLASS': ast.ClassDef}
    if expected.get(entry['kind']) is not type(node):
        raise LookupError('LEXICAL_COMPILER_SCOPE_KIND_UNSUPPORTED')
    source = decode(SourceReference, entry['source_ref'])
    if (source.path != snapshot.path or source.fingerprint != snapshot.fingerprint
            or (node.lineno, node.end_lineno, node.col_offset, node.end_col_offset) != _span(source)):
        raise ValueError('lexical entry occurrence differs from its retained source reference')
    rows = [(path, code) for path, code, scope in scopes.get(_span(source), ()) if scope is node]
    if len(rows) != 1:
        raise LookupError('LEXICAL_COMPILER_PATH_NOT_UNIQUE')
    path, code = rows[0]
    return definition, source, path, code


def expand_observed_bodies(model, trace, out, budget=None, max_bodies=32, *,
                           max_source_files=8, max_source_bytes=8 * 1024 * 1024):
    """Plan from original capture, close readers, then add missing static bodies.

    ``out`` is a separate expansion report directory. The observed-calls CLI
    subsequently reconnects the same retained trace using a fresh reader.
    """
    started = time.monotonic()
    out, model = Path(out).absolute(), Path(model).absolute()
    if (out.resolve() != out or out.is_symlink() or out.is_relative_to(model)
            or out.exists() and (not out.is_dir() or any(out.iterdir()))):
        raise FileExistsError('body-demand report needs a fresh canonical directory outside the model')
    limits = StaticCallBudget() if budget is None else budget
    if type(limits) is not StaticCallBudget:
        raise TypeError('observed body demands need StaticCallBudget')
    if type(max_bodies) is not int or max_bodies < 1 or max_bodies > limits.max_bodies:
        raise ValueError('max_bodies must be positive and within the static body budget')
    from scar.analysis.observed_source_demands import expand_observed_sources
    source_expansion = expand_observed_sources(model, trace, out / 'sources', limits,
        max_source_files=max_source_files, max_source_bytes=max_source_bytes)
    # Source publication closes its readers before mutation. Reopen original
    # capture and the updated canonical source owner for the separate body phase.
    trace = _read_trace(trace)
    observed = read_python_call_targets(trace)
    code_sources = ObservedCodeSources(trace, observed.source_sha256)
    metadata = {meta.process_id: meta for meta in trace.metadata}
    groups = defaultdict(list)
    issues = []
    for capture in trace.frame_scan.captures:
        meta = metadata.get(capture.process_id)
        if meta is None or meta.implementation != sys.implementation.name or meta.python != sys.version:
            issues.append({'kind': 'CAPTURE_COMPILER_PROFILE_UNSUPPORTED', 'code': capture.code.key,
                           'process_id': capture.process_id})
            continue
        groups[capture.code.filename].append(capture)
    requests, already, matches, unmatched_reasons = {}, {}, {}, {}
    owner = StaticCallSession(model, out=out / 'source-basis', budget=limits)
    prior = owner.manifest_sha256
    try:
        by_path = defaultdict(list)
        for number, unit in enumerate(owner.units):
            by_path[unit['path']].append(number)
        for filename, captures in groups.items():
            try:
                owner._check()
                numbers = by_path.get(filename, ())
                if not numbers:
                    kind = ('CAPTURE_SOURCE_UNIT_UNAVAILABLE' if Path(filename).is_absolute()
                            else 'CAPTURE_NONFILESYSTEM_SOURCE_UNAVAILABLE')
                    issues.append({'kind': kind, 'file': filename,
                                   'codes': sorted({item.code.key for item in captures})})
                    continue
                for number in numbers:
                    owner._check()
                    unit = owner.units[number]
                    if any(kind not in unit['artifacts'] for kind in ('source', 'lexical')):
                        raise LookupError('RETAINED_SOURCE_OR_LEXICAL_INPUT_UNAVAILABLE')
                    catalog, tree, root, scopes, scope_issues = _source(owner, number)
                    snapshot = SourceSnapshot(filename, 'sha256:' + unit['decoded_sha256'],
                                              'utf-8', f'{sys.version_info.major}.{sys.version_info.minor}')
                    rows = {row['entry_key']: row for row in unit['bodies']}
                    root_key = code_key(root)
                    wanted = defaultdict(list)
                    for capture in captures:
                        wanted[capture.code.qualname, capture.code.first_line].append(capture)
                    for ordinal, entry in enumerate(catalog['entries']):
                        owner._check()
                        if entry['kind'] not in ('MODULE', 'FUNCTION', 'ASYNC_FUNCTION', 'CLASS'):
                            continue
                        try:
                            definition, source, path, code = _entry_code(entry, tree, root, scopes, snapshot)
                        except LookupError:
                            continue  # Unmatched captures receive an explicit issue below.
                        selected = wanted.get((code.co_qualname, code.co_firstlineno), ())
                        if not selected:
                            continue
                        description = describe_code(code)
                        for capture in selected:
                            if description != encode(capture.code):
                                continue
                            try:
                                source_authority = code_sources.require_source(capture, snapshot)
                            except (LookupError, ValueError, TypeError) as error:
                                issues.append({'kind': 'CAPTURE_CODE_SOURCE_SCOPE_OPEN',
                                    'code': capture.code.key, 'process_id': capture.process_id,
                                    'detail': type(error).__name__ + ': ' + str(error)})
                                continue
                            record_key = capture.process_id, capture.code.key
                            matches.setdefault(record_key, []).append(entry['entry_key'])
                            row = rows.get(entry['entry_key'])
                            if row is None or row.get('definitions') != entry.get('semantic_definition_ids'):
                                raise ValueError('original body ledger changed the selected lexical definition')
                            if 'artifact' in row:
                                if row['status'] == 'LOWERED_WITH_GAPS':
                                    already[entry['entry_key']] = None
                                    continue
                                if not null_body_diagnostic(unit, row, owner._read(row['artifact'])):
                                    issues.append({'kind': 'CAPTURE_EXISTING_BODY_REFINEMENT_REQUIRED',
                                        'code': capture.code.key, 'entry': entry['entry_key'],
                                        'detail': 'Existing core ports cannot be replaced by diagnostic materialization.'})
                                    continue
                            current_plan = plan_static_entry(tree, entry, path=unit['path'],
                                fingerprint='sha256:' + unit['decoded_sha256'])
                            if (current_plan['status'] not in ('READY', 'NEEDS_STATIC_COMPILER')
                                    or current_plan.get('selected_profile') is None):
                                issues.append({'kind': 'CAPTURE_BODY_STATIC_RULE_OPEN',
                                    'code': capture.code.key, 'entry': entry['entry_key'],
                                    'status': current_plan['status'],
                                    'unsupported_shapes': current_plan['unsupported_shapes'],
                                    'entry_requirements': current_plan['rules'][0]['prerequisites'],
                                    'detail': 'Material and code match; the current static adapter still cannot build this body.'})
                                continue
                            evidence = StaticCodeEvidence(snapshot, unit['namespace_root'], definition, source,
                                path, code, code_key(code), sys.implementation.name, tuple(sys.version_info[:3]),
                                sys.version, sys.implementation.cache_tag, root_code_layout_key=root_key)
                            demand = ObservedBodyDemand(observed.source_sha256, capture, number, entry['entry_key'],
                                owner._locator(unit['artifacts']['lexical'], 'scar.static-lexical-catalog/1',
                                    '#/entries/' + str(ordinal), f'#/units/{number}/artifacts/lexical'),
                                owner._locator(unit['artifacts']['source'], 'scar.static-source-text/1', '#',
                                    f'#/units/{number}/artifacts/source'), evidence.to_reference(),
                                source_authority)
                            requests.setdefault(entry['entry_key'], demand)
                    # Distinguish missing scope adapters from actual code
                    # mismatches without guessing from a display name.
                    unresolved = [item for item in captures
                        if not matches.get((item.process_id, item.code.key))]
                    for capture in unresolved:
                        unmatched_reasons.setdefault((capture.process_id, capture.code.key),
                                                     'CAPTURE_CODE_DESCRIPTOR_MISMATCH')
                    pending = [root]
                    while pending and unresolved:
                        owner._check()
                        compiled_code = pending.pop()
                        pending.extend(item for item in compiled_code.co_consts
                                       if type(item) is type(root))
                        for capture in unresolved:
                            if ((compiled_code.co_qualname, compiled_code.co_firstlineno)
                                    != (capture.code.qualname, capture.code.first_line)):
                                continue
                            same = describe_code(compiled_code) == encode(capture.code)
                            if same:
                                unmatched_reasons[capture.process_id, capture.code.key] = (
                                    'CAPTURE_COMPILER_SCOPE_ADAPTER_UNAVAILABLE')
            except (LookupError, ValueError, SyntaxError, TypeError, OverflowError, TimeoutError) as error:
                issues.append({'kind': 'CAPTURE_SOURCE_COMPILER_MAPPING_OPEN', 'file': filename,
                               'detail': type(error).__name__ + ': ' + str(error)})
                if isinstance(error, (OverflowError, TimeoutError)):
                    break
        ambiguous = set()
        for capture in trace.frame_scan.captures:
            candidates = matches.get((capture.process_id, capture.code.key), ())
            if len(candidates) != 1:
                ambiguous.update(candidates)
                reason = ('CAPTURE_LEXICAL_ENTRY_AMBIGUOUS' if candidates else
                    unmatched_reasons.get((capture.process_id, capture.code.key),
                                          'CAPTURE_LEXICAL_CODE_MAPPING_UNAVAILABLE'))
                issues.append({'kind': reason, 'code': capture.code.key,
                    'file': capture.code.filename, 'candidates': candidates,
                    'detail': 'Generated/unsupported/mismatched code is not selected by spelling or line number.'})
        for entry in ambiguous:
            requests.pop(entry, None)
        selected = list(requests)[:max_bodies]
        if len(requests) > max_bodies:
            issues.append({'kind': 'OBSERVED_BODY_DEMAND_BUDGET', 'remaining': len(requests) - max_bodies})
        plan = {'schema': 'scar.observed-body-demand-plan', 'schema_version': 1,
            'model': str(model), 'prior_manifest_sha256': prior, 'run_scope': observed.source_sha256,
            'source_expansion_result': str(out / 'sources' / 'result.json'),
            'requests': encode(tuple(requests[key] for key in selected)), 'requested_entries': selected,
            'already_materialized_entries': list(already), 'issues': issues,
            'target_executed': False, 'runtime_class_identity_proved': False,
            'qualification': 'Complete code/source correspondence selects static lowering, not transformation.'}
    finally:
        owner.close()  # All artifact/source owners close before model mutation.
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'plan.json').open('xb') as stream:
        stream.write(_json(plan))
    results = []
    if selected:
        from scar.analysis.static_body import expand_static_bodies
        updated = expand_static_bodies(model, jobs=1, entries=tuple(selected),
                                       max_bodies=len(selected), additive_only=True, retry_diagnostics=True)
        wanted = set(selected)
        results = [dict(row) for unit in updated['units'] for row in unit['bodies'] if row['entry_key'] in wanted]
    published = hashlib.sha256((model / 'manifest.json').read_bytes()).hexdigest()
    expanded = [row['entry_key'] for row in results
                if row.get('status') == 'LOWERED_WITH_GAPS' and 'artifact' in row]
    report = {'schema': 'scar.observed-body-expansion', 'schema_version': 1,
        'model': str(model), 'trace': trace.directory, 'plan': str(out / 'plan.json'),
        'prior_manifest_sha256': prior, 'published_manifest_sha256': published,
        'source_expansion': source_expansion,
        'requested_entries': selected, 'request_count': len(selected),
        'expanded_entries': expanded, 'expanded_body_count': len(expanded),
        'already_materialized_entries': list(already), 'results': results, 'issues': issues,
        'readers_closed_before_expansion': True, 'additive_only': True,
        'target_executed': False, 'whole_program_complete': False,
        'analyzer_seconds': time.monotonic() - started}
    with (out / 'result.json').open('xb') as stream:
        stream.write(_json(report))
    return report


__all__ = [
    'expand_observed_sources',
    'ObservedBodyDemand',
    'expand_observed_bodies',
]
