from dataclasses import replace
import json

import pytest

from scar.ir.record_codec import encode
from scar.ir.versioned_schedule import (StorageRegion, StorageSlot, VersionedLogicalValue as V,
    VersionedMaterialization as M, VersionedResourceSchedule as S, VersionedTask as T, VersionedTaskKind as K)
from scar.optimization.schedule import (LogicalValue, Materialization, ResourceSchedule,
    schedule_document, schedule_from_dict, validate_schedule)


def ordered_state_schedule():
    values = (V('initial'), V('rhs', 'compute'), V('state1', 'write'),
              V('later', 'reuse'), V('committed', 'commit'))
    slots = (StorageSlot('external', 'host', 16, 'EXTERNAL'),
             StorageSlot('shadow', 'host', 16, allocated_by='allocate', released_by='release'),
             StorageSlot('temp', 'host', 16, allocated_by='allocate', released_by='release'))
    region = lambda slot: StorageRegion(slot, 0, 16)
    materials = (M('external0', 'initial', region('external'), 0, external_ready=True),
        M('shadow0', 'initial', region('shadow'), 0, 'snapshot'),
        M('rhs0', 'rhs', region('temp'), 0, 'compute'),
        M('shadow1', 'state1', region('shadow'), 1, 'write'),
        M('rhs1', 'later', region('temp'), 1, 'reuse'),
        M('external1', 'committed', region('external'), 1, 'commit'))
    tasks = (T('allocate', K.ALLOCATE, allocates=('shadow', 'temp'), capability='CPU_ALLOCATE'),
        T('snapshot', K.SNAPSHOT, ('external0',), ('shadow0',), ('allocate',), capability='CPU_HOST_SNAPSHOT'),
        T('compute', K.COMPUTE, ('shadow0',), ('rhs0',), ('snapshot',), capability='CPU_SCALAR_DAG'),
        T('write', K.STATE_WRITE, ('shadow0', 'rhs0'), ('shadow1',), ('compute',), capability='CPU_STATE_WRITE'),
        T('reuse', K.COMPUTE, ('shadow1',), ('rhs1',), ('write',), capability='CPU_SCALAR_DAG'),
        T('fence', K.SUCCESS_FENCE, ('shadow1',), depends_on=('reuse',),
            validates=('allocate', 'snapshot', 'compute', 'write', 'reuse'), capability='CPU_SUCCESS_FENCE'),
        T('commit', K.COMMIT, ('external0', 'shadow1'), ('external1',), ('fence',), capability='CPU_COMMIT'),
        T('release', K.RELEASE, depends_on=('commit',), releases=('shadow', 'temp'), capability='CPU_RELEASE'))
    return S('original-state-region', values, materials, slots, tasks,
        budget_bytes=48, budget_scope='EXTERNAL_STATE_PLUS_FIXED_WORKSPACE',
        budget_refs=({'original_budget_pointer': '#/memory_budget_bytes'},), executable=True)


def test_state_versions_slot_reuse_and_transaction_require_real_happens_before():
    schedule = ordered_state_schedule()
    report = validate_schedule(schedule)
    assert report['fixed_capacity_bytes'] == 48
    assert schedule_from_dict(json.loads(json.dumps(schedule_document(schedule)))) == schedule
    # Serialization tuple order is not causality: a reversed tuple with the
    # same edges remains valid, while an apparently ordered tuple can race.
    assert validate_schedule(replace(schedule, tasks=tuple(reversed(schedule.tasks))))['tasks'] == 8
    tasks = tuple(replace(t, reads=('external0',), depends_on=('allocate',)) if t.id == 'reuse' else
                  replace(t, depends_on=('reuse', 'write')) if t.id == 'fence' else t for t in schedule.tasks)
    with pytest.raises(ValueError, match='unordered producers|old-version consumer'):
        validate_schedule(replace(schedule, tasks=tasks))
    tasks = tuple(replace(t, depends_on=('reuse',)) if t.id == 'commit' else t for t in schedule.tasks)
    with pytest.raises(ValueError, match='complete precommit success fence'):
        validate_schedule(replace(schedule, tasks=tasks))
    tasks = tuple(replace(t, validates=('allocate', 'snapshot', 'compute', 'write')) if t.id == 'fence'
                  else t for t in schedule.tasks)
    with pytest.raises(ValueError, match='every precommit'):
        validate_schedule(replace(schedule, tasks=tasks))
    tasks = tuple(replace(t, kind=K.EXPORT, capability='CPU_EXPORT') if t.id == 'commit'
                  else t for t in schedule.tasks)
    with pytest.raises(ValueError, match='fence/export cannot modify'):
        validate_schedule(replace(schedule, tasks=tasks))
    materials = tuple(replace(m, value='initial') if m.id == 'shadow1' else m for m in schedule.materializations)
    values = tuple(v for v in schedule.values if v.id != 'state1')
    with pytest.raises(ValueError, match='distinct new immutable logical value'):
        validate_schedule(replace(schedule, materializations=materials, values=values))


def test_cpu_snapshot_cannot_authorize_gpu_copy_and_v1_wire_is_unchanged():
    schedule = ordered_state_schedule()
    slots = tuple(replace(s, device='cuda') if s.id == 'shadow' else s for s in schedule.storage_slots)
    with pytest.raises(ValueError, match='do not admit device copies'):
        validate_schedule(replace(schedule, storage_slots=slots))
    tasks = tuple(replace(t, kind=K.COPY) if t.id == 'snapshot' else t for t in schedule.tasks)
    with pytest.raises(ValueError, match='generic COPY/device transfer'):
        validate_schedule(replace(schedule, tasks=tasks))
    legacy = ResourceSchedule('old-region', (LogicalValue('arg'),),
        (Materialization('arg-mat', 'arg', 'host', external_ready=True, ownership='EXTERNAL'),), ())
    expected = {'schema': 'scar.resource-schedule', 'schema_version': 1, **encode(legacy)}
    assert json.dumps(schedule_document(legacy), sort_keys=True) == json.dumps(expected, sort_keys=True)
    assert schedule_from_dict(expected) == legacy
    assert validate_schedule(legacy)['tasks'] == 0


def test_final_commit_materializes_one_existing_value_without_inventing_a_producer():
    original = ordered_state_schedule()
    materials = tuple(replace(m, value='state1') if m.id == 'external1' else m for m in original.materializations)
    values = tuple(v for v in original.values if v.id != 'committed')
    identity = dict(logical_identity='VALUE_PRESERVING_FINAL_ROOT_COMMIT',
        identity_source_material='shadow1',
        identity_proof_ref={'schedule_pointer': '#/proof/final_root_commit_identity/root'})
    tasks = tuple(replace(t, attrs=identity) if t.id == 'commit' else t for t in original.tasks)
    schedule = replace(original, values=values, materializations=materials, tasks=tasks,
        proof={'final_root_commit_identity': {'root': {'profile_replay': 'ALL_SOURCE_ROIS_AND_UNCHANGED_HALO'}}})
    validate_schedule(schedule)
    assert schedule_from_dict(schedule_document(schedule)) == schedule
    replicas = [m for m in schedule.materializations if m.value == 'state1']
    assert {m.ready_task for m in replicas} == {'write', 'commit'}
    assert next(v.producer for v in schedule.values if v.id == 'state1') == 'write'
    with pytest.raises(ValueError, match='profile identity proof'):
        validate_schedule(replace(schedule, proof={}))
    partial = tuple(replace(m, region=replace(m.region, nbytes=8)) if m.id == 'external1' else m
                    for m in schedule.materializations)
    with pytest.raises(ValueError, match='full-result materialization'):
        validate_schedule(replace(schedule, materializations=partial))


def test_fused_copy_preserves_logical_values_and_rejects_cross_copy_alias():
    values = (V('left'), V('right'))
    slots = (StorageSlot('external:0', 'host', 16, 'EXTERNAL'),
        StorageSlot('external:1', 'host', 16, 'EXTERNAL'),
        StorageSlot('private', 'host', 32, allocated_by='allocate', released_by='release'))
    materials = (M('in:0', 'left', StorageRegion('external:0', 0, 16), 0, external_ready=True),
        M('in:1', 'right', StorageRegion('external:1', 0, 16), 0, external_ready=True),
        M('out:0', 'left', StorageRegion('private', 0, 16), 0, 'fused'),
        M('out:1', 'right', StorageRegion('private', 16, 16), 1, 'fused'))
    fused = T('fused', K.COPY, ('in:0', 'in:1'), ('out:0', 'out:1'), ('allocate',),
        capability='CPU_HOST_COPY_REGION', source_refs=({'source': 'two-private-raw-copies'},),
        attrs=dict(stage_kind='FUSED_SNAPSHOT_ROOT', dtype='float64', shape=[1, 2],
            child_task_ids=['copy:0', 'copy:1'], copy_pairs=[
                dict(read='in:0', write='out:0', source_task='copy:0'),
                dict(read='in:1', write='out:1', source_task='copy:1')]))
    schedule = S('copy-region', values, materials, slots, (
        T('allocate', K.ALLOCATE, allocates=('private',), capability='CPU_ALLOCATE'), fused,
        T('release', K.RELEASE, depends_on=('fused',), releases=('private',), capability='CPU_RELEASE')))
    assert validate_schedule(schedule)['tasks'] == 3
    assert schedule_from_dict(schedule_document(schedule)) == schedule
    # Two destination versions cannot be concurrently produced in overlapping
    # storage, even if their logical IDs and source inputs differ.
    overlapping = tuple(replace(m, region=StorageRegion('private', 8, 16)) if m.id == 'out:1' else m
        for m in materials)
    with pytest.raises(ValueError, match='unordered producers|disjoint private writes'):
        validate_schedule(replace(schedule, materializations=overlapping))
    corrupted = replace(fused, attrs={**fused.attrs,
        'copy_pairs': list(reversed(fused.attrs['copy_pairs']))})
    with pytest.raises(ValueError, match='exact independent raw-copy ports'):
        validate_schedule(replace(schedule, tasks=(schedule.tasks[0], corrupted, schedule.tasks[-1])))
    with pytest.raises(ValueError, match='generic COPY/device transfer'):
        validate_schedule(replace(schedule, executable=True))
