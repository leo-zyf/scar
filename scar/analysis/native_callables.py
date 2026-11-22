"""Native callables."""

from __future__ import annotations

import hashlib
import json
import re
import gzip
from pathlib import Path
from scar.analysis.call_trace_reader import CallOperandTrace
from scar.analysis.python_call_targets import read_python_call_targets
from scar.ir.program.call_edges import CallRecordAnchor, CallTraceFile
from scar.ir.program.native_callables import (
    ADDRESS_FIELDS,
    NativeCallableCapture,
    NativeCallableFact,
    NativeCallableMetadata,
    build_fields,
)
from scar.ir.record_codec import decode, loads
from scar.trace.reader import open_binary, resolve
from dataclasses import dataclass
from hashlib import sha256
from scar.ir.program.codec import dumps as dump_graph
from scar.ir.program.functions import ArgumentKind
from scar.ir.program.ids import (
    BlockID,
    LocationID,
    NativeBoundaryID,
    OperationID,
    ProviderID,
    RegionID,
    UseID,
    ValueID,
)
from scar.ir.program.model import (
    Behavior,
    Block,
    BlockArgument,
    CompletionResult,
    CompletionSignature,
    CompletionSuccessor,
    ExitKind,
    ExitPort,
    Operation,
    ProgramGraph,
    ReferenceValue,
    Region,
    StateLocation,
    StateVersionValue,
    Terminator,
    TerminatorKind,
    Use,
    UseRole,
    value_type,
)
from scar.ir.program.native import (
    CoreInputPort,
    CoreOutputPort,
    EntryPortRef,
    ExitPortRef,
    NativeBoundary,
    NativeInputSlot,
    NativeOutputSlot,
)
from scar.ir.program.providers import (
    ProviderArtifact,
    ProviderArtifactKind,
    ProviderBinding,
    ProviderCaptureMode,
    ProviderCollectionDomain,
    ProviderCollectionObservation,
    ProviderCoverage,
    ProviderCoverageDomain,
    ProviderElement,
    ProviderElementKind,
    ProviderEnvelope,
    ProviderFactReference,
    ProviderRegion,
    ProviderRegionRelation,
    ProviderStage,
)
from scar.ir.program.source_references import SourceArtifactLocator
from scar.ir.program.validation import require_valid
from scar.ir.v2.common import Completeness


# Strict optional native carrier companion over the original CALL streams.
#
# Reading this evidence neither reclassifies Python target v2 nor proves native
# effects, dispatch completion, source provenance or cross-event pointer identity.
_META = re.compile(r'native-callable-meta\.([1-9][0-9]*)\.json$')
_EVENT = re.compile(r'native-callables\.([1-9][0-9]*)\.jsonl(?:\.gz)?$')
_ROW_KEYS = {'schema', 'schema_version', 'event_index', 'caller', 'attempt_index',
             'callable_reference', 'carrier', 'fields'}
_LEGACY_FAMILIES = {'codes', 'events', 'meta', 'operands', 'operand-meta'}


def _canonical_file(logical, *, archive=False):
    path = Path(logical)
    if not path.is_absolute() or path.resolve() != path or path.is_symlink():
        raise ValueError('native callable inputs must be original canonical paths')
    if archive:
        compressed = Path(str(path) + '.gz')
        if path.exists() and compressed.exists():
            raise ValueError('ambiguous plain/gzip native callable artifact')
        physical = resolve(path)
    else:
        physical = path
    if (not physical.is_file() or physical.is_symlink()
            or physical.resolve() != physical):
        raise ValueError('native callable input is absent, nonregular or symlinked')
    return path, physical


def _verify_receipt(receipt, *, maximum, expected=None):
    if expected is not None and receipt.path != str(expected):
        raise ValueError('native callable receipt changed its original artifact namespace')
    path, _ = _canonical_file(receipt.path)
    if receipt.bytes > maximum or path.stat().st_size != receipt.bytes:
        raise ValueError('native callable receipt exceeds budget or changed size')
    with path.open('rb') as stream:
        payload = stream.read(receipt.bytes + 1)
    if len(payload) != receipt.bytes or hashlib.sha256(payload).hexdigest() != receipt.sha256:
        raise ValueError('native callable input bytes/SHA differ')
    return payload


def _fact(document):
    if (type(document) is not dict or set(document) != _ROW_KEYS
            or document['schema'] != 'scar.native-callables'
            or type(document['schema_version']) is not int or document['schema_version'] != 1):
        raise ValueError('unsupported native callable event schema/fields')
    return decode(NativeCallableFact,
        {key: value for key, value in document.items() if key not in ('schema', 'schema_version')})


def read_native_callables(trace, *, max_bytes=32_000_000, max_rows=100_000):
    """Freshly verify the optional companion; None is legacy NOT_RECORDED.

    A supported complete profile has exactly one fact per original attempt,
    including UNKNOWN carrier facts. Partial/failed companions are rejected,
    never converted to an empty set or a claim that missing calls did not occur.
    """
    if type(trace) is not CallOperandTrace:
        raise TypeError('native callable reader requires the original CallOperandTrace')
    if any(type(value) is not int or value < 1 for value in (max_bytes, max_rows)):
        raise ValueError('native callable read ceilings must be positive exact integers')
    root = Path(trace.directory)
    if not root.is_absolute() or root.resolve() != root or root.is_symlink():
        raise ValueError('native callable directory must be original and canonical')
    metas, event_processes = {}, set()
    for path in root.iterdir():
        match = _META.fullmatch(path.name)
        if match:
            metas[int(match[1])] = path
        elif (match := _EVENT.fullmatch(path.name)) is not None:
            event_processes.add(int(match[1]))
        elif path.name.startswith('native-callable-meta.') and path.name.endswith('.json.gz'):
            raise ValueError('native callable metadata must be retained as original JSON')
    if not metas and not event_processes:
        return None
    if set(metas) != event_processes:
        raise ValueError('native callable events and metadata must be retained together')
    observed = read_python_call_targets(trace)
    attempts = {(row.caller.process_id, row.attempt_index): row for row in observed.targets}
    expected_processes = {row.process_id for row in trace.metadata}
    if set(metas) != expected_processes:
        raise ValueError('native callable capture does not cover every original CALL process')
    facts, anchors, metadata, files, target_basis = [], [], [], [], []
    total = 0
    seen = set()
    originals = (*trace.frame_scan.files, *trace.files)
    for process, meta_path in sorted(metas.items(), key=lambda row: str(row[1])):
        _canonical_file(str(meta_path))
        if meta_path.stat().st_size > min(max_bytes - total, 1_000_000):
            raise ValueError('native callable metadata byte ceiling exceeded')
        with meta_path.open('rb') as stream:
            payload = stream.read(min(max_bytes - total, 1_000_000) + 1)
        if len(payload) > min(max_bytes - total, 1_000_000):
            raise ValueError('native callable metadata grew beyond its byte ceiling')
        total += len(payload)
        meta = decode(NativeCallableMetadata, loads(payload))
        if (meta.process_id != process or meta.project_root != trace.project_root
                or meta.stopped or meta.io_failed or meta.errors):
            raise ValueError('native callable issuer/scope differs or capture is partial/failed')
        expected_inputs = tuple(row for row in originals if row.process_id == process)
        if ({row.family for row in meta.legacy_inputs} != _LEGACY_FAMILIES
                or len(expected_inputs) != 5 or set(meta.legacy_inputs) != set(expected_inputs)):
            raise ValueError('native callable companion changed the five original CALL receipts')
        target_raw = _verify_receipt(meta.target_meta, maximum=min(1_000_000, max_bytes - total),
            expected=root / f'python-call-target-meta.{process}.json')
        total += len(target_raw)
        target_meta = loads(target_raw)
        target_basis.append((str(meta_path.parent / f'python-call-target-meta.{process}.json'),
            meta.target_meta.sha256, target_meta['file']['path'], target_meta['file']['sha256']))
        for receipt in (meta.probe_source, meta.probe_binary):
            _verify_receipt(receipt, maximum=max_bytes - total)
            total += receipt.bytes
        expected_attempts = {key for key in attempts if key[0] == process}
        if (meta.counts.attempts != len(expected_attempts) or meta.counts.rows != meta.file.rows
                or meta.file.rows != len(expected_attempts)
                or meta.file.rows > meta.budget.max_records
                or meta.file.bytes > meta.budget.max_bytes):
            raise ValueError('native callable denominator/output counts or declared budgets differ')
        expected_path = root / f'native-callables.{process}.jsonl'
        if meta.file.path != str(expected_path):
            raise ValueError('native callable output belongs to another artifact namespace')
        logical, physical = _canonical_file(meta.file.path, archive=True)
        if physical.stat().st_size > max_bytes or total + meta.file.bytes > max_bytes:
            raise ValueError('native callable stored/decoded byte ceiling exceeded')
        files.append(CallTraceFile(str(meta_path), 'native-callable-meta', process,
            hashlib.sha256(payload).hexdigest(), len(payload), payload.count(b'\n')))
        digest = hashlib.sha256()
        size = rows = 0
        line_limit = min(meta.budget.max_line_bytes, 2_000_000)
        with open_binary(logical) as stream:
            while raw := stream.readline(line_limit + 1):
                size += len(raw)
                total += len(raw)
                rows += 1
                if (len(raw) > line_limit or not raw.endswith(b'\n') or not raw.strip()
                        or total > max_bytes or len(facts) >= max_rows
                        or rows > meta.budget.max_records or size > meta.budget.max_bytes):
                    raise ValueError('native callable event/line/decoded byte budget or framing differs')
                digest.update(raw)
                fact = _fact(loads(raw))
                key = process, fact.attempt_index
                target = attempts.get(key)
                if (fact.event_index != rows or fact.caller.process_id != process
                        or target is None or key in seen or fact.caller != target.caller
                        or fact.callable_reference != target.callable_reference):
                    raise ValueError('native callable fact changed original CALL/target identity or sequence')
                for name in ADDRESS_FIELDS:
                    field = getattr(fact.fields, name)
                    if field.value is not None and field.value >= 1 << meta.abi.pointer_bits:
                        raise ValueError('native pointer exceeds the original declared ABI width')
                seen.add(key)
                facts.append(fact)
                anchors.append(CallRecordAnchor(str(logical), rows, hashlib.sha256(raw).hexdigest(), rows))
        if (size, rows, digest.hexdigest()) != (meta.file.bytes, meta.file.rows, meta.file.sha256):
            raise ValueError('native callable original EOF/bytes/rows/SHA differ')
        metadata.append(meta)
        files.append(CallTraceFile(str(logical), 'native-callables', process,
            digest.hexdigest(), size, rows))
    if seen != set(attempts):
        raise ValueError('native callable capture left original attempts unrecorded')
    run = hashlib.sha256(json.dumps(target_basis, sort_keys=True).encode()).hexdigest()
    if run != observed.source_sha256:
        raise ValueError('native callable target metadata changed within its original run basis')
    return NativeCallableCapture(str(root), observed.source_sha256, tuple(facts),
        tuple(anchors), tuple(metadata), tuple(files))


# Shared declared CPython CALL interfaces, without invoking a callable.
#
# Templates own Python-side ports and an open dispatch boundary. C layouts are
# conditional contracts, not observed C entry, arguments, allocations or returns.
SCHEMA = 'scar.cpython-native-protocol'
PROTOCOL_VERSION = 1


@dataclass(frozen=True, slots=True)
class NativeProtocolShape:
    carrier: str
    flags: int
    arguments: tuple[tuple[str, str | None], ...]

    def __post_init__(self):
        if (type(self.carrier) is not str or self.carrier not in ('METHOD_DEF', 'WRAPPER_SLOT')
                or type(self.flags) is not int or not 0 <= self.flags <= 0xffffffff
                or type(self.arguments) is not tuple):
            raise ValueError('native protocol shape requires a recognized carrier and exact flags/arguments')
        for row in self.arguments:
            if type(row) is not tuple or len(row) != 2 or type(row[0]) is not str:
                raise TypeError('native source argument shape must be a typed kind/keyword tuple')
            kind, keyword = row
            if kind not in tuple(item.value for item in ArgumentKind):
                raise ValueError('native protocol argument kind is unsupported')
            if (kind == ArgumentKind.KEYWORD.value and
                    (type(keyword) is not str or not keyword or '\0' in keyword)
                    or kind != ArgumentKind.KEYWORD.value and keyword is not None):
                raise ValueError('only original explicit keyword arguments have a keyword selector')


def _json(document):
    return (json.dumps(document, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=True, allow_nan=False) + '\n').encode('ascii')


def _shape(shape):
    if type(shape) is not NativeProtocolShape:
        raise TypeError('expected exact NativeProtocolShape')
    shape.__post_init__()
    return {'carrier': shape.carrier, 'flags': shape.flags,
            'arguments': [[kind, keyword] for kind, keyword in shape.arguments]}


def native_protocol_key(shape):
    """Only the versioned protocol and source-port shape select a template."""
    return sha256(_json({'protocol_version': PROTOCOL_VERSION, 'shape': _shape(shape)})).hexdigest()


def _cabi_contract(shape):
    if shape.carrier == 'WRAPPER_SLOT':
        convention = 'WRAPPER_KEYWORDS' if shape.flags & 1 else 'WRAPPER_POSITIONAL'
        if shape.flags & ~1:
            convention = 'UNSUPPORTED_WRAPPER_FLAGS'
        formals = ['SELF', 'POSITIONAL_TUPLE', 'WRAPPED_SLOT']
        if shape.flags & 1:
            formals.append('KEYWORD_DICT')
        roles = ['EXACT_WRAPPER_DESCRIPTOR_ROLE_REQUIRED']
    else:
        # These are the exact CPython dispatch mask/alternatives; binding and
        # coexistence flags are preserved in the shape, not mistaken for ABI.
        conventions = {
            0x04: ('NOARGS', ['SELF', 'C_NULL']),
            0x08: ('O', ['SELF', 'OBJECT']),
            0x01: ('VARARGS', ['SELF', 'POSITIONAL_TUPLE']),
            0x03: ('VARARGS_KEYWORDS', ['SELF', 'POSITIONAL_TUPLE', 'KEYWORD_DICT']),
            0x80: ('FASTCALL', ['SELF', 'POSITIONAL_ARRAY', 'POSITIONAL_COUNT']),
            0x82: ('FASTCALL_KEYWORDS', ['SELF', 'POSITIONAL_AND_KEYWORD_ARRAY',
                                      'POSITIONAL_COUNT', 'KEYWORD_NAMES']),
            0x282: ('METHOD_FASTCALL_KEYWORDS', ['SELF', 'DEFINING_CLASS',
                'POSITIONAL_AND_KEYWORD_ARRAY', 'POSITIONAL_COUNT', 'KEYWORD_NAMES']),
        }
        convention, formals = conventions.get(shape.flags & 0x28f,
                                               ('UNSUPPORTED_METHOD_FLAGS', []))
        roles = ['EXACT_PYC_FUNCTION_OR_PYC_METHOD_ROLE_REQUIRED',
                 'EXACT_METHOD_DESCRIPTOR_ROLE_REQUIRED',
                 'EXACT_CLASSMETHOD_DESCRIPTOR_ROLE_REQUIRED']
    return {'qualification': 'CONDITIONAL_LAYOUT_NOT_OBSERVED_C_ARGUMENTS',
        'abi_profile': 'CPYTHON_3_14_7_GIL_LINUX_X86_64',
        'flags_domain': 'PyMethodDef.ml_flags' if shape.carrier == 'METHOD_DEF' else 'wrapperbase.flags',
        'calling_convention': convention, 'implementation_formal_roles': formals,
        'alternative_carrier_role_conditions': roles,
        'source_call_prefix': 'UNRESOLVED_UNLESS_SEPARATELY_ESTABLISHED',
        'argument_expansions': 'OPEN' if any(kind in ('STARRED', 'DOUBLE_STARRED')
                                          for kind, _ in shape.arguments) else 'EXPLICIT_SOURCE_PORTS',
        'applicability_conditions': ['ORIGINAL_CALLABLE_CARRIER_AND_FLAGS_STILL_MATCH',
            'SOURCE_TO_DISPATCHER_ARGUMENT_LAYOUT_VALID', 'ADAPTER_REACHES_IMPLEMENTATION'],
        'possible_preimplementation_failures': ['RECEIVER_OR_ARGUMENT_CHECK',
            'KEYWORD_OR_ARITY_CHECK', 'RECURSION_CHECK', 'ARGUMENT_EXPANSION_OR_PACKING'],
        'arity_used_to_prune_paths': False, 'implementation_entry_observed': False,
        'physical_packing_observed': False, 'effects_complete': False,
        'normal_return_observed': False}


def build_native_protocol(shape):
    """Build a validated shared graph from a shape; perform no I/O or target call."""
    shape_document = _shape(shape)
    key = native_protocol_key(shape)
    prefix = 'cpython-native-protocol-v1:' + key
    region, provider = RegionID(prefix), ProviderID(prefix + ':provider')
    entry, normal, failed = (BlockID(prefix + ':' + name) for name in ('entry', 'normal', 'raised'))
    invoke, returned, raised = (OperationID(prefix + ':' + name) for name in ('dispatch', 'return', 'raise'))
    location = LocationID(prefix + ':world')
    labels = ('callable', *(f'explicit{i}' for i in range(len(shape.arguments))))
    before = [ReferenceValue(ValueID(prefix + ':' + label), region, BlockArgument(entry, i))
              for i, label in enumerate(labels)]
    before.append(StateVersionValue(ValueID(prefix + ':world.before'), region,
        BlockArgument(entry, len(before)), location))
    after = (ReferenceValue(ValueID(prefix + ':result'), region, BlockArgument(normal, 0)),
        StateVersionValue(ValueID(prefix + ':world.normal'), region, BlockArgument(normal, 1), location))
    error = (ReferenceValue(ValueID(prefix + ':exception'), region, BlockArgument(failed, 0)),
        StateVersionValue(ValueID(prefix + ':world.raised'), region, BlockArgument(failed, 1), location))

    def uses(operation, values):
        return tuple(Use(UseID(operation.value + ':use:' + str(i)), value.id,
            UseRole.STATE_CONTENT if type(value) is StateVersionValue else UseRole.REFERENCE)
            for i, value in enumerate(values))

    inputs, outputs, errors = uses(invoke, before), uses(returned, after), uses(raised, error)
    normal_types, error_types = tuple(map(value_type, after)), tuple(map(value_type, error))
    operations = (
        Operation(invoke, 'CPython callable adapter and open implementation dispatch',
            (Behavior.CONTROL, Behavior.READ, Behavior.WRITE), inputs, effects_complete=False,
            terminator=Terminator(TerminatorKind.INVOKE, (
                CompletionSuccessor(ExitKind.NORMAL, normal, (CompletionResult(0), CompletionResult(1))),
                CompletionSuccessor(ExitKind.RAISE, failed, (CompletionResult(0), CompletionResult(1))))),
            completions=(CompletionSignature(ExitKind.NORMAL, normal_types),
                         CompletionSignature(ExitKind.RAISE, error_types)),
            invoke_arguments=tuple(use.id for use in inputs)),
        Operation(returned, 'possible Python object normal completion', (Behavior.CONTROL,), outputs,
            terminator=Terminator(TerminatorKind.RETURN, exit='normal', values=tuple(use.id for use in outputs))),
        Operation(raised, 'possible adapter or implementation exception', (Behavior.CONTROL,), errors,
            terminator=Terminator(TerminatorKind.RAISE, exit='raised', values=tuple(use.id for use in errors))))
    declaration = {'schema': SCHEMA + '.definition', 'schema_version': PROTOCOL_VERSION,
                   'shape': shape_document, 'cabi_contract': _cabi_contract(shape)}
    declared = _json(declaration)
    # This is an inline declaration identity, not a fabricated disk/source
    # receipt. The same shape/contract is retained in the stored wrapper.
    artifact = ProviderArtifact('urn:scar:cpython-native-protocol:' + sha256(declared).hexdigest(),
        sha256(declared).hexdigest(), len(declared), declaration['schema'])
    fact = lambda suffix='': ProviderFactReference(artifact, '#' + suffix)
    envelope = ProviderEnvelope(provider, 'SCAR declared CPython callable protocol', '1',
        'CPython-3.14.7-interface-contract', artifact, ProviderArtifactKind.DEFINITION_GRAPH,
        ProviderCaptureMode.STATIC_ANALYSIS, None, (),
        (ProviderStage('declared Python-side interface', fact('/shape')),), (),
        tuple(ProviderCoverage(domain, Completeness.PARTIAL if domain in (
            ProviderCoverageDomain.STRUCTURE, ProviderCoverageDomain.DATAFLOW) else Completeness.UNKNOWN,
            fact() if domain in (ProviderCoverageDomain.STRUCTURE, ProviderCoverageDomain.DATAFLOW) else None,
            'Declared adapter boundary; runtime selection, implementation and effects remain open.')
            for domain in ProviderCoverageDomain),
        tuple(ProviderCollectionObservation(domain, None, 'No target execution or measurement.')
              for domain in ProviderCollectionDomain))
    boundary = NativeBoundary(NativeBoundaryID(prefix + ':boundary'), provider, region,
        fact('/shape'), None, None,
        tuple(NativeInputSlot(i, 'PYTHON_CALLABLE' if i == 0 else f'PYTHON_EXPLICIT_ARGUMENT:{i-1}',
            fact('/shape' if i == 0 else '/shape/arguments/' + str(i - 1)), EntryPortRef(region, value.id))
            for i, value in enumerate(before[:-1])),
        (NativeOutputSlot(0, 'POSSIBLE_PYTHON_OBJECT_RESULT', fact('/cabi_contract'),
                         ExitPortRef(region, 'normal', 0)),),
        (CoreInputPort(EntryPortRef(region, before[-1].id), 'Open world before adapter dispatch.'),),
        (CoreOutputPort(ExitPortRef(region, 'normal', 1), 'Open normal post-world.'),
         CoreOutputPort(ExitPortRef(region, 'raised', 0), 'Possible exception including preimplementation failure.'),
         CoreOutputPort(ExitPortRef(region, 'raised', 1), 'Open exceptional post-world.')))
    graph = ProgramGraph(region, (Region(region, 'shared CPython callable protocol', entry,
        (Block(entry, tuple(value.id for value in before), (invoke,)),
         Block(normal, tuple(value.id for value in after), (returned,)),
         Block(failed, tuple(value.id for value in error), (raised,))),
        (ExitPort('normal', ExitKind.NORMAL, normal_types), ExitPort('raised', ExitKind.RAISE, error_types))),),
        operations, tuple((*before, *after, *error)), (StateLocation(location, 'Open callable adapter state.'),),
        provider_envelopes=(envelope,),
        provider_regions=(ProviderRegion(provider, key, region, ProviderRegionRelation.DEFINES_REPRESENTATION),),
        provider_bindings=(ProviderBinding(provider, region,
            ProviderElement(ProviderElementKind.OPERATION, key), invoke),), native_boundaries=(boundary,))
    require_valid(graph)
    return graph


def _document(shape, graph):
    core = json.loads(dump_graph(graph))
    return {'schema': SCHEMA, 'schema_version': PROTOCOL_VERSION, 'shape': _shape(shape),
        'cabi_contract': _cabi_contract(shape), 'graph': core,
        'target_executed': False, 'transformation_authorized': False}


def native_protocol_document(shape):
    """Freshly derive the whole wrapper, including its unobserved contracts."""
    return _document(shape, build_native_protocol(shape))


def write_native_protocol(shape, fresh_path):
    """Store one deterministic gzip wrapper; never overwrite an existing file."""
    graph = build_native_protocol(shape)
    document = _document(shape, graph)
    path = Path(fresh_path).absolute()
    if path.resolve() != path or path.is_symlink() or path.exists() or path.suffix != '.gz':
        raise FileExistsError('native protocol output must be a fresh canonical gzip file')
    raw = _json(document)
    stored = gzip.compress(raw, compresslevel=3, mtime=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(stored)
    artifact = ProviderArtifact(str(path), sha256(stored).hexdigest(), len(stored), SCHEMA + '/1')
    locator = SourceArtifactLocator(artifact, sha256(raw).hexdigest(), len(raw),
                                    'gzip-canonical-json', '#/graph')
    core = document['graph']
    return locator, core['schema'] + '/' + str(core['schema_version']), graph


__all__ = [
    'build_fields',
    'read_native_callables',
    'NativeProtocolShape',
    'build_native_protocol',
    'write_native_protocol',
    'native_protocol_key',
    'native_protocol_document',
    'SCHEMA',
    'PROTOCOL_VERSION',
]
