from scar.analysis.strided_content_flow import (
    AccessKind, ContentAccess, ContentOrigin, ReachingContent, analyze_loop_content,
    byte_intervals, storage_overlap,
)
from scar.ir.provenance import Overlap
from scar.ir.v2 import StorageAllocationID, StorageRegion, StorageRegionID


def _view(name, offset, shape, strides, *, allocation='a', dtype='uint8'):
    return StorageRegion(StorageRegionID(name), StorageAllocationID(allocation),
                         offset, shape, strides, dtype, 'cpu')


def test_geometry_compresses_dense_axes_and_detects_exact_byte_overlap():
    reverse = _view('dense', 999_999_999, (1_000, 1_000_000), (-1_000_000, -1), dtype='float64')
    assert byte_intervals(reverse, max_intervals=1) == ((0, 8_000_000_000),)
    broadcast = _view('broadcast', 1, (500_000, 7), (0, 0), dtype='float64')
    assert byte_intervals(broadcast, max_intervals=1) == ((8, 16),)
    even, odd = _view('even', 0, (8,), (2,)), _view('odd', 1, (8,), (2,))
    assert storage_overlap(even, odd) is Overlap.DISJOINT
    assert storage_overlap(even, odd, max_intervals=1) is Overlap.POSSIBLE
    assert storage_overlap(even, broadcast) is Overlap.OVERLAPS


def test_exact_reaching_writes_are_partitioned_by_read_and_backedge():
    entire = _view('entire', 0, (12,), (1,))
    first = _view('first', 0, (6,), (1,))
    overlap = _view('overlap', 3, (6,), (1,))
    separate = _view('separate', 10, (2,), (1,), allocation='b')
    pre, update, post = analyze_loop_content((
        ContentAccess('pre', AccessKind.READ, entire),
        ContentAccess('update', AccessKind.READ_WRITE, first),
        ContentAccess('unrelated', AccessKind.WRITE, separate),
        ContentAccess('post', AccessKind.READ, entire),
        ContentAccess('last', AccessKind.WRITE, overlap),
    ))
    entry, write = ContentOrigin.ENTRY, ContentOrigin.WRITE
    assert pre.first_iteration == (ReachingContent(entry, ((0, 12),)),)
    assert pre.later_iterations == (ReachingContent(write, ((3, 9),), 'last', 1),
                                    ReachingContent(write, ((0, 3),), 'update', 1),
                                    ReachingContent(entry, ((9, 12),)))
    assert update.first_iteration == (ReachingContent(entry, ((0, 6),)),)
    assert post.first_iteration == (ReachingContent(write, ((0, 6),), 'update', 0),
                                     ReachingContent(entry, ((6, 12),)))
    assert post.later_iterations[1] == ReachingContent(write, ((6, 9),), 'last', 1)


def test_unresolved_clobber_limits_claims_until_a_definite_write():
    entire = _view('entire', 0, (12,), (1,))
    tail = _view('tail', 4, (8,), (1,))
    stripe = _view('stripe', 0, (3,), (2,))
    pre, post, unknown = analyze_loop_content((
        ContentAccess('pre', AccessKind.READ, entire),
        ContentAccess('possible', AccessKind.MAY_WRITE, stripe),
        ContentAccess('all', AccessKind.MAY_WRITE, allocation=StorageAllocationID('b')),
        ContentAccess('overwrite', AccessKind.WRITE, tail),
        ContentAccess('post', AccessKind.READ, entire),
        ContentAccess('unknown', AccessKind.READ, stripe),
    ), max_intervals=1)
    assert pre.later_iterations == (ReachingContent(ContentOrigin.WRITE, ((4, 12),), 'overwrite', 1),
                                   ReachingContent(ContentOrigin.UNKNOWN, ((0, 4),), 'possible'))
    assert post.first_iteration == (ReachingContent(ContentOrigin.WRITE, ((4, 12),), 'overwrite', 0),
                                     ReachingContent(ContentOrigin.UNKNOWN, ((0, 4),), 'possible'))
    assert unknown.first_iteration == unknown.later_iterations == (
        ReachingContent(ContentOrigin.UNKNOWN, None),)
