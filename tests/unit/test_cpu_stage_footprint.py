from scar.optimization.cpu_stage_footprint import _access_set,_line_bounds,_merge,_partition


def test_static_partition_union_and_unknown_address_phase_bounds():
    partitions=_partition(7,[14,15,42])
    assert [(p.first_row,p.end_row) for p in partitions]==[(0,3),(3,5),(5,7)]
    layouts=[dict(storage_group=0,stride=7,offset=1),dict(storage_group=0,stride=7,offset=8),
        dict(storage_group=1,stride=5,offset=0)]
    union=_access_set(partitions[:2],layouts,4)
    expected={}
    for p in partitions[:2]:
        for row in range(p.first_row,p.end_row):
            for layout in layouts:
                expected.setdefault(layout['storage_group'],set()).update(
                    layout['offset']+row*layout['stride']+col for col in range(4))
    assert {key:{v for a,b in intervals for v in range(a,b)} for key,intervals in union.items()}==expected
    assert _merge([(0,4),(3,6),(6,8),(12,13)])==((0,8),(12,13))
    for group,intervals in union.items():
        bounds=_line_bounds(intervals,8,64,8)
        counts=[len({(element*8+byte+phase)//64 for element in expected[group] for byte in range(8)})
            for phase in range(0,64,8)]
        assert (bounds['lower'],bounds['upper'])==(min(counts),max(counts))
        assert bounds['base_address_known'] is False
