"""Small deployment loader for an admitted, unchanged loop shared library.

Only bundle bytes, a pinned producer-admission hash and the live process/CPU
boundary are checked here. Compiler proofs and provenance archives belong to
export admission. Original Python callable/source guards belong to loop_guard.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import re
import stat
import struct
import sys

from .loop_contract import validate_admission, validate_contract

_JSON_LIMIT = 2 * 1024 * 1024


class GuardFailure(RuntimeError):
    """Native execution declined; the installer may call the original once."""


class _Array(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("rank", ctypes.c_int64),
                ("shape", ctypes.c_int64 * 2), ("strides", ctypes.c_int64 * 2),
                ("lower_bound", ctypes.c_size_t), ("upper_bound", ctypes.c_size_t)]


_Run = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(_Array), ctypes.c_uint64,
                       ctypes.POINTER(_Array))


class _Table(ctypes.Structure):
    _fields_ = [("abi_version", ctypes.c_uint32), ("struct_size", ctypes.c_uint32),
                ("trip_count", ctypes.c_uint64), ("run", _Run),
                ("loop_sha256", ctypes.c_char_p), ("strategy_sha256", ctypes.c_char_p),
                ("execution_plan_sha256", ctypes.c_char_p), ("contract_sha256", ctypes.c_char_p)]


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _regular(path):
    _need(stat.S_ISREG(path.lstat().st_mode), "bundle member must be a regular file: " + path.name)


def _pairs(values):
    result = {}
    for key, value in values:
        _need(key not in result, "duplicate JSON field")
        result[key] = value
    return result


def _invalid_number(value):
    raise ValueError("nonfinite JSON number: " + value)


def _small_raw(path):
    _regular(path)
    with path.open("rb") as stream:
        raw = stream.read(_JSON_LIMIT + 1)
    _need(len(raw) <= _JSON_LIMIT, "bundle JSON exceeds 2 MiB")
    return raw


def _json(raw):
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=_invalid_number)


def _file_hash_and_header(path):
    _regular(path)
    digest = sha256()
    with path.open("rb") as stream:
        header = stream.read(64)
        digest.update(header)
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest(), header


def _cpu_available(cpu_ids):
    try:
        available = os.sched_getaffinity(0)
    except (AttributeError, OSError) as error:
        raise GuardFailure("live CPU allocation cannot be verified") from error
    if not set(cpu_ids) <= available:
        raise GuardFailure("planned CPU allocation unavailable")


def _process(contract):
    cpu = contract["cpu"]
    if (sys.platform != cpu["os"] or platform.machine().lower() not in ("x86_64", "amd64")
            or ctypes.sizeof(ctypes.c_void_p) * 8 != cpu["pointer_bits"]
            or sys.byteorder != cpu["byteorder"] or ctypes.sizeof(ctypes.c_double) != 8
            or ctypes.sizeof(_Array) != 64 or ctypes.sizeof(_Table) != 56):
        raise GuardFailure("runtime process does not satisfy the Linux x86_64 loop ABI")
    # SSE2 is part of the x86_64 ABI. The unchanged SIMD kernel additionally
    # checks CPUID on every executing bound worker; no AVX/OSXSAVE is assumed.
    if cpu["required_isa"] not in ("BASELINE_X86_64", "SSE2_X86_64"):
        raise GuardFailure("unsupported runtime ISA")
    _cpu_available(cpu["ids"])


def _descriptor(array, np):
    if type(array) is not np.ndarray:
        raise GuardFailure("runtime requires an exact numpy.ndarray")
    if array.ndim != 2 or array.dtype != np.dtype("float64") or not array.dtype.isnative:
        raise GuardFailure("runtime requires native float64 rank-two arrays")
    if not array.flags.aligned or any(stride % 8 for stride in array.strides):
        raise GuardFailure("runtime requires float64-aligned data and strides")
    owner, seen = array, set()
    while owner.base is not None:
        if id(owner) in seen or type(owner.base) is not np.ndarray:
            raise GuardFailure("runtime array backing allocation cannot be verified")
        seen.add(id(owner)); owner = owner.base
    if not owner.flags.owndata or not (owner.flags.c_contiguous or owner.flags.f_contiguous):
        raise GuardFailure("runtime array needs a contiguous owning allocation")
    lower = int(owner.ctypes.data); upper = lower + owner.nbytes
    pointer = int(array.ctypes.data)
    limit = (1 << (ctypes.sizeof(ctypes.c_size_t) * 8)) - 1
    if not 0 <= lower <= upper <= limit or not 0 <= pointer <= limit:
        raise GuardFailure("runtime allocation exceeds pointer bounds")
    if array.size:
        offsets = [(extent - 1) * stride for extent, stride in zip(array.shape, array.strides)]
        low = pointer + sum(min(0, offset) for offset in offsets)
        high = pointer + sum(max(0, offset) for offset in offsets) + 8
        if low < lower or high > upper:
            raise GuardFailure("runtime array strides exceed backing allocation")
    return _Array(pointer, 2, (ctypes.c_int64 * 2)(*array.shape),
                  (ctypes.c_int64 * 2)(*array.strides), lower, upper)


@dataclass
class NativeLoop:
    directory: Path
    _contract: dict
    _report: dict
    _library: object
    _table: object
    _run: object

    @property
    def contract(self):
        return json.loads(json.dumps(self._contract))

    @property
    def report(self):
        return json.loads(json.dumps(self._report))

    def run(self, state, steps):
        if type(steps) is not int or steps != self._contract["trip_count"] or steps <= 0:
            raise GuardFailure("runtime trip count differs from exact specialization")
        _cpu_available(self._contract["cpu"]["ids"])
        import numpy as np
        descriptor = _descriptor(state, np)
        shape = tuple(self._contract["array"]["shape"])
        if state.shape != shape or not state.flags.c_contiguous:
            raise GuardFailure("runtime array shape/layout differs from contract")
        size = self._contract["array"]["final_bytes"]
        if state.nbytes != size or size > self._contract["memory_budget_bytes"] or size > sys.maxsize:
            raise GuardFailure("runtime fresh final exceeds payload budget")
        try:
            output = np.empty(shape, dtype=np.float64)
        except (MemoryError, ValueError, OverflowError) as error:
            raise GuardFailure("runtime fresh final cannot be allocated") from error
        if output is state or output.shape != shape or output.nbytes != size or not output.flags.owndata:
            raise GuardFailure("runtime allocation is not the independent fresh final")
        output_descriptor = _descriptor(output, np)
        if (descriptor.lower_bound < output_descriptor.upper_bound and
                output_descriptor.lower_bound < descriptor.upper_bound):
            raise GuardFailure("runtime final allocation aliases the input owner")
        status = self._run(ctypes.byref(descriptor), steps, ctypes.byref(output_descriptor))
        if status:
            reason = {1: "pointer/rank/bounds/alias", 2: "shape", 3: "floating-point or non-finite carry",
                      4: "CPU affinity", 5: "trip count", 6: "payload budget", 7: "SSE2 ISA"}.get(status, str(status))
            raise GuardFailure("native loop " + reason + " guard")
        return output


def load(bundle, *, expected_admission_sha256, allow_unqualified=False):
    """Load exactly the bundle pinned by the build producer's admission hash."""
    _need(type(expected_admission_sha256) is str and
          re.fullmatch("[0-9a-f]{64}", expected_admission_sha256) is not None,
          "explicit expected admission SHA256 must be 64 lowercase hexadecimal characters")
    directory = Path(bundle).resolve()
    _need(directory.is_dir(), "runtime bundle directory required")
    admission_raw = _small_raw(directory / "admission.json")
    _need(sha256(admission_raw).hexdigest() == expected_admission_sha256, "admission SHA256 differs from explicit trust pin")
    admission = _json(admission_raw)
    validate_admission(admission)
    contract_raw = _small_raw(directory / "runtime.json")
    _need(sha256(contract_raw).hexdigest() == admission["runtime_contract_sha256"], "runtime contract byte SHA256 differs")
    contract = _json(contract_raw)
    validate_contract(contract)
    _need(contract['schema_version'] == 1 or allow_unqualified is True,
          'unqualified loop requires explicit pipeline validation or release')
    native = contract["native"]
    _need(native["ids"] == admission["native_ids"] and
          native["library_sha256"] == admission["library_sha256"], "runtime/native admission identities differ")
    core = {"admission.json", "runtime.json", native["library"]}
    members = {p.name for p in directory.iterdir()}
    _need(core <= members <= core | {"export-report.json"},
          "runtime bundle has missing core files or unknown extra members")
    library_path = directory / native["library"]
    library_hash, header = _file_hash_and_header(library_path)
    _need(library_hash == native["library_sha256"], "runtime library byte SHA256 differs")
    _process(contract)
    _need(len(header) >= 64 and header[:6] == b"\x7fELF\x02\x01" and
          struct.unpack_from("<H", header, 16)[0] == 3 and struct.unpack_from("<H", header, 18)[0] == 62,
          "runtime library requires an ELF64 little-endian x86_64 shared object")
    library = ctypes.CDLL(str(library_path))
    getter = library.scar_get_loop_table_v1
    getter.argtypes = []; getter.restype = ctypes.POINTER(_Table)
    table = getter()
    _need(bool(table) and table.contents.abi_version == native["loop_abi_version"] and
          table.contents.struct_size >= ctypes.sizeof(_Table) and bool(table.contents.run) and
          table.contents.trip_count == contract["trip_count"], "native loop table ABI/count differs")
    for key in ("loop_sha256", "strategy_sha256", "execution_plan_sha256", "contract_sha256"):
        _need(getattr(table.contents, key) == native["ids"][key].encode(), "native loop embedded identity differs: " + key)
    report = {**native["ids"], "admission_sha256": expected_admission_sha256,
              "runtime_contract_sha256": admission["runtime_contract_sha256"], "library_sha256": library_hash,
              "loop_abi_version": native["loop_abi_version"], "shape": contract["array"]["shape"],
              "cpu": contract["cpu"], "trust_policy": admission["trust_policy"]}
    return NativeLoop(directory, contract, report, library, table, table.contents.run)


__all__ = ["NativeLoop", "GuardFailure", "load"]
