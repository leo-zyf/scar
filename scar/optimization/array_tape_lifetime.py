"""Place private array-tape roots with exact address residues and last uses."""

from collections import defaultdict

from scar.optimization.pipeline_memory import _solve


def pack_roots(roots, accesses, *, search_budget=1024):
    """Reuse slots only across disjoint inclusive instruction lifetimes."""
    lifetimes = {}
    for ordinal, (reads, writes) in enumerate(accesses):
        for key in set(reads) | set(writes):
            if key not in lifetimes:
                if key in reads:
                    raise ValueError('the tape reads a root before its first write')
                lifetimes[key] = [ordinal, ordinal]
            else:
                lifetimes[key][1] = ordinal
    if set(roots) != set(lifetimes):
        raise ValueError('tape roots and accessed roots differ')
    groups = defaultdict(list)
    unique_bytes = 0
    for key, (first, last) in lifetimes.items():
        root = roots[key]
        extent, residue = root['size_bytes'], root['address_mod_64']
        if type(extent) is not int or extent <= 0 or type(residue) is not int or residue not in range(0, 64, 8):
            raise ValueError('each private root needs a bounded extent and exact stock-loop alignment')
        padded = (extent + residue + 63) & ~63
        groups[residue].append(dict(id=key, nbytes=padded, producer_index=first, last_use=last))
        unique_bytes += padded

    offsets = {}
    capacity = lower_bound = states = 0
    complete = True
    for residue, items in sorted(groups.items()):
        items.sort(key=lambda row: (row['producer_index'], row['id']))
        # The solver uses dense event indices only for its overlap lower bound.
        endpoints = {ordinal: i for i, ordinal in enumerate(sorted(
            {row[key] for row in items for key in ('producer_index', 'last_use')}))}
        intervals = [dict(row, producer_index=endpoints[row['producer_index']],
                          last_use=endpoints[row['last_use']]) for row in items]
        slots, assignment, lower, explored, solved = _solve(intervals, len(endpoints), search_budget)
        lower_bound += lower
        states += explored
        complete &= solved
        starts = []
        for extent, _ in slots:
            starts.append(capacity)
            capacity += extent
        for row, index in zip(items, assignment, strict=True):
            offsets[row['id']] = starts[index] + residue
    return dict(offset_bytes=offsets, lifetimes=lifetimes, arena_bytes=capacity,
        independent_root_bytes=unique_bytes, lower_bound_bytes=lower_bound,
        exact_minimum_in_residue_partition=complete, states_explored=states,
        budget_scope='PRIVATE_TAPE_ROOTS_WITH_ALIGNMENT_PADDING',
        excludes=['IMMUTABLE_COEFFICIENTS', 'LOWERING_WORKSPACE', 'PROCESS_RUNTIME'])
