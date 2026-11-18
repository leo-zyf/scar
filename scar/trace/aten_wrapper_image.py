"""Resolve observed ATen wrapper entries against the capture's mapped ELF backing."""
from __future__ import annotations

from base64 import b64decode
from collections import defaultdict
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import stat
import struct
import subprocess

from scar.analysis.call_operand_reader import read_call_operands
from scar.analysis.native_callables import read_native_callables
from scar.ir.program.native_callables import NativeCarrier
from scar.ir.record_codec import encode
from scar.trace.collector_execution_witness import PROCESS_PROFILE, bind_collector_execution_witness
from scar.trace.process_image_receipts import read_process_image_receipts


SCHEMA = 'scar.aten-wrapper-images'
_PHASES = ('NATIVE_START', 'NATIVE_CLOSE')
_MAX_ELF_BYTES = 64_000_000


def _receipt_bytes(path, receipt, ceiling):
    if not 0 <= receipt['size_bytes'] <= ceiling or path.is_symlink():
        raise ValueError('unsupported wrapper-image receipt member size or path')
    with path.open('rb') as stream:
        payload = stream.read(receipt['size_bytes'] + 1)
    if len(payload) != receipt['size_bytes'] or sha256(payload).hexdigest() != receipt['sha256']:
        raise ValueError('wrapper-image member changed during replay')
    return payload


def _image_inputs(coordinates):
    member = coordinates.report.get('collector_execution_membership', {})
    if member.get('profile') != PROCESS_PROFILE:
        return None, 'V2_PROCESS_IMAGE_WITNESS_NOT_RETAINED'
    witness_ref = member['manifest']
    witness_path = Path(witness_ref['uri'])
    witness = bind_collector_execution_witness(
        witness_path, expected_sha256=witness_ref['sha256'], coordinates=coordinates)
    if (witness.get('process_birth_join') or {}).get('status') != 'BOUND_VIA_EXACT_OWNED_ARTIFACT_PUBLICATION':
        return None, 'WRAPPER_IMAGE_PROCESS_INCARNATION_UNBOUND'
    manifest = json.loads(_receipt_bytes(witness_path, witness_ref, 1_000_000))
    image_ref = manifest['process_image_receipts']
    image_path = Path(image_ref['uri'])
    image = read_process_image_receipts(image_path, expected_sha256=image_ref['sha256'])
    phases = {row['phase']: row for row in image['snapshots']}
    if any(phase not in phases or phases[phase]['maps_status'] != 'ALL_ORIGINAL_ROWS_RETAINED'
           for phase in _PHASES):
        return None, 'WRAPPER_IMAGE_NATIVE_BOUNDARY_MAPS_UNAVAILABLE'
    rows, refs = {}, {}
    for phase in _PHASES:
        receipt = phases[phase]['rows']
        path = image_path.parent / receipt['path']
        payload = _receipt_bytes(path, receipt, image['budgets']['max_output_bytes'])
        rows[phase] = tuple(item for item in map(json.loads, payload.splitlines())
                            if item['mapping'] is not None and item['mapping']['perms'][2] == 'x')
        refs[phase] = dict(uri=str(path), schema=receipt['schema'],
                           sha256=receipt['sha256'], size_bytes=receipt['size_bytes'])
    return (witness_ref, image_ref, image['backings'], rows, refs), None


def _mapped_entry(rows, backings, address):
    owners = [tuple(row for row in rows[phase]
                    if row['mapping']['start'] <= address < row['mapping']['end']) for phase in _PHASES]
    if any(len(found) != 1 for found in owners):
        return None, 'WRAPPER_ENTRY_MAPPING_UNAVAILABLE_OR_AMBIGUOUS'
    start, close = (found[0] for found in owners)
    mapping = start['mapping']
    if close['mapping'] != mapping or mapping['perms'][1] == 'w':
        return None, 'WRAPPER_ENTRY_MAPPING_UNQUALIFIED_AT_NATIVE_BOUNDARIES'
    index = start['backing'].get('backing_index')
    if index is None or close['backing'].get('backing_index') != index:
        return None, 'WRAPPER_ENTRY_MAPPING_HAS_NO_COMMON_FILE_BACKING'
    name = b64decode(mapping['pathname_base64'], validate=True)
    path = Path(os.fsdecode(name))
    if not path.is_absolute() or path.name != 'libtorch_python.so' or b'\\' in name:
        return None, 'WRAPPER_ENTRY_IS_OUTSIDE_SUPPORTED_TORCH_IMAGE'
    backing = backings[index]
    offset = mapping['offset'] + (address - mapping['start'])
    if not 0 <= offset < backing['size_bytes']:
        return None, 'WRAPPER_ENTRY_OFFSET_IS_OUTSIDE_RETAINED_FILE'
    return (path, backing['sha256'], backing['size_bytes'], offset, index, start, close), None


def _has_exact_bytes(stream, digest, nbytes):
    observed = sha256()
    remaining = nbytes
    while remaining:
        block = stream.read(min(1_048_576, remaining))
        if not block:
            return False
        observed.update(block)
        remaining -= len(block)
    return not stream.read(1) and observed.hexdigest() == digest


def _elf_functions(path, digest, nbytes, offsets):
    """Read one hash-matched regular fd, then inspect that fd with readelf."""
    if not 0 < nbytes <= _MAX_ELF_BYTES:
        return None
    try:
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW), 'rb') as stream:
            fd = stream.fileno()
            before = os.fstat(fd)
            if (not stat.S_ISREG(before.st_mode) or before.st_size != nbytes
                    or not _has_exact_bytes(stream, digest, nbytes)):
                return None
            stream.seek(0)
            header = stream.read(64)
            if (len(header) != 64 or header[:7] != b'\x7fELF\x02\x01\x01'
                    or struct.unpack_from('<HH', header, 16) != (3, 62)):
                return None
            # The inspection follows the previously opened descriptor even if
            # the analysis-machine pathname changes after byte verification.
            listing = subprocess.run(['readelf', '--wide', '--program-headers', '--syms', '--demangle',
                                        f'/proc/self/fd/{fd}'], capture_output=True, check=True,
                                       pass_fds=(fd,), timeout=30, env={**os.environ, 'LC_ALL': 'C'}).stdout
            after = os.fstat(fd)
            if ((before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                    != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
                return None
    except (OSError, subprocess.SubprocessError):
        return None
    virtuals_by_offset, symbols = defaultdict(list), defaultdict(set)
    sought_virtuals = set()
    for line in BytesIO(listing):
        line = line.rstrip(b'\n')
        fields = line.split(maxsplit=7)
        if not fields:
            continue
        if fields[0] == b'LOAD':
            parts = line.split()
            if b'E' in parts[6:-1]:
                start, virtual, size = (int(parts[i], 16) for i in (1, 2, 4))
                if not 0 <= start <= start + size <= nbytes:
                    return None
                for offset in offsets:
                    if start <= offset < start + size:
                        actual = virtual + offset - start
                        virtuals_by_offset[offset].append(actual)
                        sought_virtuals.add(actual)
        elif (len(fields) == 8 and fields[0].endswith(b':') and fields[3] == b'FUNC'
              and fields[6].isdigit()):
            virtual = int(fields[1], 16)
            if virtual in sought_virtuals:
                symbols[virtual].add(fields[7].decode('utf-8', errors='replace'))
    resolved = {}
    for offset, virtuals in virtuals_by_offset.items():
        if len(virtuals) == 1:
            virtual, = virtuals
            names = symbols[virtual]
            if len(names) == 1:
                symbol, = names
                resolved[offset] = (virtual, symbol)
    return resolved


def _source_attempts(capture, trace):
    coordinates = capture.coordinates
    pid = coordinates.target_process.pid
    root = Path(coordinates.report['local_root']) / 'trace'
    if trace is None:
        trace = read_call_operands(root, coordinates.report['call_collector_declaration']['project_root'])
    if Path(trace.directory) != root:
        raise ValueError('wrapper evidence uses another original CALL trace root')
    operand_files = {row.path: row for row in trace.files if row.process_id == pid and row.family == 'operands'}
    attempts = {event.index: event for event in trace.attempts if event.process_id == pid}
    selected = {}
    for call in capture.calls:
        event = attempts.get(call['attempt'])
        origin = call['operand']['artifact']
        member = operand_files.get(origin['uri'])
        if (event is None or member is None or (member.raw_bytes, member.raw_sha256)
                != (origin['size_bytes'], origin['sha256'])
                or event.anchor.path != origin['uri']
                or call['operand']['locator'] != f'#/lines/{event.anchor.line_number}'
                or call['invocation'] != dict(process_id=event.process_id, thread_id=event.thread_id,
                                               invocation_id=event.invocation_id)
                or call['site'] != encode(event.site)):
            raise ValueError('wrapper scope and operand reader address different original CALL bytes')
        selected[call['attempt']] = event
    return trace, selected


def recover_aten_wrapper_images(capture, trace=None):
    """Bind scoped CALL entries to file-backed FUNC symbols at two image boundaries.

    The native callable companion replays the original CALL artifacts. Its
    publication is not covered by the image receipt, and this association
    carries no claim about loaded pages or exact C++ source/effects.
    """
    coordinates = capture.coordinates
    calls = [dict(attempt=row['attempt'], site=row['site'], invocation=row['invocation'],
                  operand=row['operand'], scope=row['declaration'], status='UNRESOLVED', reason=None)
             for row in capture.calls]
    report = dict(schema=SCHEMA, schema_version=1, run=encode(coordinates.run.id),
                  process=encode(coordinates.target_process.id), coordinate_binding=encode(coordinates.artifact),
                  provider_revision=capture.revision, calls=calls, witness=None, image_receipt=None,
                  native_receipts=[],
                  independent_native_sidecar_publication_established=False,
                  continuous_mapping_established=False, executed_pages_verified=False,
                  source_build_association_established=False, effects_established=False,
                  transformation_authorized=False)
    images, reason = _image_inputs(coordinates)
    if images is None:
        for call in calls:
            call['reason'] = reason
        return report
    report['witness'], report['image_receipt'], backings, rows, refs = images
    trace, attempts = _source_attempts(capture, trace)
    native = read_native_callables(trace)
    if native is None:
        for call in calls:
            call['reason'] = 'NATIVE_CALLABLE_COMPANION_NOT_RETAINED'
        return report
    pid = coordinates.target_process.pid
    report['native_receipts'] = [encode(row) for row in native.files if row.process_id == pid]
    facts = {fact.attempt_index: (fact, anchor) for fact, anchor in zip(native.facts, native.anchors, strict=True)
             if fact.caller.process_id == pid}
    groups = defaultdict(list)
    for call in calls:
        event = attempts[call['attempt']]
        fact, anchor = facts[call['attempt']]
        if encode(fact.caller) != call['invocation'] or fact.callable_reference != event.dispatch.reference:
            raise ValueError('native callable and observed scope have different CALL identities')
        fields = fact.fields
        address = fields.entrypoint_address.value
        call.update(native_event=encode(anchor), callable_reference=fact.callable_reference,
                    carrier=fact.carrier.value, callable_type=fields.exact_type_name.value,
                    method_name=fields.method_name.value, method_flags=fields.method_flags.value,
                    observed_entrypoint_address=address)
        if fact.carrier is not NativeCarrier.METHOD_DEF or address is None or fields.method_name.value is None:
            call['reason'] = 'NATIVE_METHOD_ENTRY_UNOBSERVED'
            continue
        mapped, reason = _mapped_entry(rows, backings, address)
        if mapped is None:
            call['reason'] = reason
            continue
        path, digest, nbytes, offset, index, start, close = mapped
        groups[path, digest, nbytes].append((call, offset, index, start, close))
    for (path, digest, nbytes), group in groups.items():
        functions = _elf_functions(path, digest, nbytes, {row[1] for row in group})
        for call, offset, index, start, close in group:
            if functions is None:
                call['reason'] = 'MATCHING_WRAPPER_ELF_BYTES_OR_INSPECTION_UNAVAILABLE'
                continue
            entry = functions.get(offset)
            if entry is None or not entry[1].startswith(f"torch::autograd::THPVariable_{call['method_name']}("):
                call['reason'] = 'EXACT_TORCH_WRAPPER_FUNCTION_SYMBOL_UNAVAILABLE'
                continue
            call.update(status='QUALIFIED_IMAGE_ENTRY', reason=None, entry_vaddr=entry[0], entry_symbol=entry[1],
                        backing=dict(uri=str(path), size_bytes=nbytes, sha256=digest,
                                     index=index, file_offset=offset),
                        mappings={phase: dict(artifact=refs[phase], locator=f"#/lines/{row['ordinal']}")
                                  for phase, row in zip(_PHASES, (start, close), strict=True)})
    return report
