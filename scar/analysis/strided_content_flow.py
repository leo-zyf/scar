"""Byte-level reaching definitions for an ordered, fixed-geometry loop body.

The caller supplies admitted storage identities and accesses on the normal
backedge. The analysis neither selects those accesses from source nor proves
their provider effects. An unresolved write remains a clobber even if its
destination is unknown, and the same body is used for every iteration.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from scar.ir.provenance import Overlap, _byte_bounds, _width
from scar.ir.v2 import StorageAllocationID, StorageRegion


ByteIntervals = tuple[tuple[int, int], ...]


class AccessKind(str, Enum):
    READ = 'READ'
    WRITE = 'WRITE'
    READ_WRITE = 'READ_WRITE'
    MAY_WRITE = 'MAY_WRITE'


@dataclass(frozen=True, slots=True)
class ContentAccess:
    access_id: str
    kind: AccessKind
    region: StorageRegion | None = None
    allocation: StorageAllocationID | None = None

    def __post_init__(self):
        if type(self.access_id) is not str or not self.access_id:
            raise ValueError('expected an original access identity')
        if type(self.kind) is not AccessKind:
            raise TypeError('expected an exact access kind')
        if self.region is not None and type(self.region) is not StorageRegion:
            raise TypeError('expected an original storage region')
        if self.allocation is not None and type(self.allocation) is not StorageAllocationID:
            raise TypeError('expected an original storage allocation')
        if self.region is not None and self.allocation not in (None, self.region.allocation):
            raise ValueError('access region and allocation identities differ')
        if self.kind in (AccessKind.WRITE, AccessKind.READ_WRITE) and self.region is None:
            raise ValueError('definite writes need a declared region; use MAY_WRITE for an open destination')


class ContentOrigin(str, Enum):
    ENTRY = 'ENTRY'
    WRITE = 'WRITE'
    UNKNOWN = 'UNKNOWN'


@dataclass(frozen=True, slots=True)
class ReachingContent:
    kind: ContentOrigin
    intervals: ByteIntervals | None
    writer: str | None = None
    iteration_distance: int | None = None


@dataclass(frozen=True, slots=True)
class ReadFlow:
    read: str
    allocation: StorageAllocationID | None
    first_iteration: tuple[ReachingContent, ...]
    later_iterations: tuple[ReachingContent, ...]


def _geometry_inputs(region):
    if type(region) is not StorageRegion or type(region.allocation) is not StorageAllocationID:
        raise TypeError('expected a typed storage region and allocation')
    if (type(region.offset) is not int or region.offset < 0
            or type(region.shape) is not tuple or type(region.strides) is not tuple
            or len(region.shape) != len(region.strides)
            or any(type(size) is not int or size < 0 for size in region.shape)
            or any(type(stride) is not int for stride in region.strides)
            or any(type(value) is not str or not value for value in (region.dtype, region.layout, region.device))):
        raise ValueError('invalid typed storage region geometry')
    if 0 in region.shape:
        return None, ()
    bounds = _byte_bounds(region)
    if bounds is not None and bounds[0] < 0:
        raise ValueError('strided region addresses bytes before its allocation')
    return bounds, None


def _budget(max_intervals):
    if type(max_intervals) is not int or max_intervals <= 0:
        raise ValueError('expected a positive geometry interval budget')


def _merge(intervals):
    merged = []
    for low, high in sorted(intervals):
        if merged and low <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], high))
        else:
            merged.append((low, high))
    return tuple(merged)


def _geometry(region, budget):
    bounds, intervals = _geometry_inputs(region)
    if bounds is None:
        return bounds, intervals
    width = _width(region)
    dimensions = sorted((abs(stride), size) for size, stride in
                        zip(region.shape, region.strides, strict=True) if size > 1 and stride != 0)
    # Sorted axes with stride <= the current span translate a contiguous
    # block onto itself or its neighbor. They add no interval fragments.
    dense = 1
    for index, (stride, size) in enumerate(dimensions):
        if stride > dense:
            break
        dense += (size - 1) * stride
    else:
        index = len(dimensions)
    low = bounds[0]
    intervals = ((low, low + dense * width),)
    for stride, size in dimensions[index:]:
        if size > budget // len(intervals):
            return bounds, None
        step = stride * width
        intervals = _merge((a + n * step, b + n * step)
                           for n in range(size) for a, b in intervals)
    return bounds, intervals


def byte_intervals(region, *, max_intervals=4096):
    """Return exact half-open byte intervals, or None at a geometry frontier."""
    _budget(max_intervals)
    return _geometry(region, max_intervals)[1]


def _split(remaining, coverage):
    """Partition sorted, disjoint read intervals by sorted write intervals."""
    hits, misses = [], []
    j = 0
    for cursor, high in remaining:
        while j < len(coverage) and coverage[j][1] <= cursor:
            j += 1
        while j < len(coverage) and coverage[j][0] < high:
            start, end = coverage[j]
            if cursor < start:
                misses.append((cursor, start))
            hits.append((max(cursor, start), min(high, end)))
            cursor = min(high, end)
            if cursor == high:
                break
            j += 1
        if cursor < high:
            misses.append((cursor, high))
    return tuple(hits), tuple(misses)


def storage_overlap(left, right, *, max_intervals=4096):
    """Compare admitted regions without treating intersecting bounds as proof."""
    _budget(max_intervals)
    bounds_a, a = _geometry(left, max_intervals)
    bounds_b, b = _geometry(right, max_intervals)
    if left.allocation != right.allocation or a == () or b == ():
        return Overlap.DISJOINT
    if left.device != right.device:
        raise ValueError('one storage allocation has conflicting devices')
    if bounds_a is not None and bounds_b is not None and (
            bounds_a[1] <= bounds_b[0] or bounds_b[1] <= bounds_a[0]):
        return Overlap.DISJOINT
    if a is None or b is None:
        return Overlap.POSSIBLE
    return Overlap.OVERLAPS if _split(a, b)[0] else Overlap.DISJOINT


@dataclass(frozen=True, slots=True)
class _Access:
    record: ContentAccess
    allocation: StorageAllocationID | None
    bounds: tuple[int, int] | None
    intervals: ByteIntervals | None


def _resolve(read, remaining, writes, distance):
    parts = []
    for write in reversed(writes):
        if write.allocation not in (None, read.allocation):
            continue
        exact = write.intervals
        coverage = exact if exact is not None else (write.bounds,) if write.bounds is not None else remaining
        covered, remaining = _split(remaining, coverage)
        if covered:
            definite = exact is not None and write.record.kind is not AccessKind.MAY_WRITE
            parts.append(ReachingContent(ContentOrigin.WRITE if definite else ContentOrigin.UNKNOWN,
                covered, write.record.access_id, distance if definite else None))
        if not remaining:
            break
    return tuple(parts), remaining


def _entry(intervals):
    return (ReachingContent(ContentOrigin.ENTRY, intervals),) if intervals else ()


def analyze_loop_content(accesses, *, max_intervals=4096):
    """Return per-read reaching content on the initial and steady iterations.

    The sequence is a single normal ordered body whose accesses have the same
    geometry and storage identities across the backedge. WRITE overwrites
    exactly its region. READ_WRITE reads old bytes before its normal write.
    Conditional or unqualified writes use MAY_WRITE and invalidate certainty
    until a later definite write covers the affected bytes.
    """
    _budget(max_intervals)
    accesses = tuple(accesses)
    if any(type(access) is not ContentAccess for access in accesses):
        raise TypeError('expected original typed content accesses')
    if len({access.access_id for access in accesses}) != len(accesses):
        raise ValueError('access identities must be unique within a loop body')
    prepared, devices = [], {}
    for access in accesses:
        access.__post_init__()
        region = access.region
        allocation, bounds, intervals = access.allocation, None, None
        if region is not None:
            bounds, intervals = _geometry(region, max_intervals)
            allocation = region.allocation
            if devices.setdefault(allocation, region.device) != region.device:
                raise ValueError('one storage allocation has conflicting devices')
        prepared.append(_Access(access, allocation, bounds, intervals))
    writes = [access for access in prepared if access.record.kind is not AccessKind.READ]
    prior_writes, reads = [], []
    for access in prepared:
        if access.record.kind in (AccessKind.READ, AccessKind.READ_WRITE):
            if access.intervals is None:
                first = later = (ReachingContent(ContentOrigin.UNKNOWN, None),)
            elif not access.intervals:
                first = later = ()
            else:
                current, left = _resolve(access, access.intervals, prior_writes, 0)
                first = current + _entry(left)
                carried, remaining = _resolve(access, left, writes, 1) if left else ((), ())
                later = current + carried + _entry(remaining)
            reads.append(ReadFlow(access.record.access_id, access.allocation, first, later))
        if access.record.kind is not AccessKind.READ:
            prior_writes.append(access)
    return tuple(reads)
