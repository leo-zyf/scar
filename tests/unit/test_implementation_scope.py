"""Source scope and integer cut algebra only; no native/target execution."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from test_implementation_envelope import original_and_envelope, _ancestors
from scar.analysis.control_milestone import derive_scope_control_milestone
from scar.analysis.implementation_scope import derive_stateful_scope_envelope
from scar.analysis.schedule_time import attach_task_time
from scar.codegen.scope_boundary_probe import render_scope_boundary_probe
from scar.codegen.stateful import implementation_sha256
from scar.compiler.calibrate_scope_boundary import (
    CONDITIONS, CUTS, MISSING, PHASES, SCHEMA, scope_boundary_statistics,
)
from scar.ir.execution_recipe import artifact_reference, make_recipe
from scar.ir.versioned_schedule import (
    TemporalDurationBasis as Basis, TemporalOccupancyKind as OK,
    VersionedTaskKind as K, schedule_document, validate_versioned_schedule,
)


def test_source_cleanup_preserves_original_graph_and_only_certified_fence_has_zero_additional_service(
        original_and_envelope, tmp_path):
    original, _ = original_and_envelope
    path = tmp_path/'scope-component.json'
    path.write_text(json.dumps({'schema': 'scar.test-source-component'}))
    artifact = artifact_reference(path, 'scar.test-source-component')
    recipe = make_recipe('cpu-stateful-array-region-v1',
        semantic_boundary={'id': original.region_id,
            'inputs': [{'id': 'input:0', 'kind': 'array', 'shape': [1, 2]}]},
        control={'fixed_source_order': True, 'spatial_partition': 'SERIAL_ROWS'},
        task_graph=schedule_document(original), resources={'fixture': 'NO_RUNTIME_HARDWARE_CLAIM'},
        implementation={'family': 'CPP_SYNCHRONOUS_CPU_TRANSACTIONAL_STATEFUL_REGION',
            'backend_inputs': {'region': 'region', 'plan': 'stage_plan'},
            'backend_capability_record': {'implementation_sha256': implementation_sha256()}},
        artifacts={'region': artifact, 'stage_plan': artifact})
    scope = derive_stateful_scope_envelope(original, recipe)
    assert scope.values == original.values and scope.materializations == original.materializations
    assert scope.storage_slots == original.storage_slots and scope.order_constraints == original.order_constraints
    assert scope.temporal is None and scope.executable is False
    before, after = _ancestors(original), _ancestors(scope)
    assert all(ancestors <= after[name] for name, ancestors in before.items())
    tasks = {row.id: row for row in scope.tasks}
    for old in original.tasks:
        new = tasks[old.id]
        assert (new.kind, new.reads, new.writes, new.allocates, new.releases, new.attrs,
                new.source_refs) == (old.kind, old.reads, old.writes, old.allocates,
                old.releases, old.attrs, old.source_refs)
    cleanup = next(t for t in scope.tasks if t.attrs.get('phase_kind') == 'NATIVE_HELPER_CLEANUP')
    commits = [t for t in scope.tasks if t.kind is K.COMMIT]
    release = next(t for t in scope.tasks if t.kind is K.RELEASE)
    fence = next(t for t in scope.tasks if t.kind is K.SUCCESS_FENCE)
    assert scope.tasks.index(commits[-1])+1 == scope.tasks.index(cleanup)
    assert scope.tasks.index(cleanup)+1 == scope.tasks.index(release)
    assert {t.id for t in commits} <= after[cleanup.id] and cleanup.id in after[release.id]
    assert cleanup.attrs['failure_domain'] == 'POSTCOMMIT' and cleanup.attrs['may_fail'] is False
    assert cleanup.id not in fence.validates
    proof = scope.proof['implementation_scope_cleanup']
    assert proof['reverse_raii_order'] == ['FloatingScope', 'ThreadAffinity', 'WORKSPACE_POOL_UNIQUE_PTR_ARRAY']
    assert proof['source']['source_pins'] and proof['source']['fragments']
    assert proof['milestone']['measurement_claimed'] is False and proof['costs_inferred'] is False

    # The portable collective RELEASE leaves its internal slot order unexpanded.
    # Once individual frees are represented, C++ std::array destruction must
    # follow the reverse declaration order, rather than retaining a legacy HB.
    physical_order = 'CPP_STD_ARRAY_REVERSE_DESTRUCTION'
    slots = tuple(replace(slot, released_by='release:'+slot.id)
        if slot.ownership == 'SCHEDULE' else slot for slot in original.storage_slots)
    old_release = next(t for t in original.tasks if t.kind is K.RELEASE)
    slot_order = [slot.id for slot in reversed(slots) if slot.ownership == 'SCHEDULE']
    releases = []
    for slot in slot_order:
        releases.append(replace(old_release, id='release:'+slot, releases=(slot,),
            depends_on=(releases[-1].id,) if releases else old_release.depends_on))
    split = replace(original, storage_slots=slots,
        tasks=(*(t for t in original.tasks if t.kind is not K.RELEASE), *releases),
        proof={**original.proof, 'pool_release_order': physical_order,
            'physical_release_slot_order': slot_order})
    def split_recipe(control):
        return make_recipe(recipe['profile'], task_graph=schedule_document(split), control=control,
            **{key: recipe[key] for key in ('semantic_boundary', 'resources', 'implementation',
                                          'artifacts', 'obligations')})
    modern = derive_stateful_scope_envelope(split, split_recipe({**recipe['control'],
        'pool_release_order': physical_order}))
    assert [t.releases[0] for t in modern.tasks if t.kind is K.RELEASE] == ['temp', 'shadow']
    assert modern.proof['implementation_scope_cleanup']['pool_release_order'] == physical_order
    with pytest.raises(ValueError, match='reverse physical pool release order'):
        derive_stateful_scope_envelope(split, split_recipe(recipe['control']))
    with pytest.raises(ValueError, match='reverse physical pool release order'):
        derive_stateful_scope_envelope(split, split_recipe({**recipe['control'],
            'pool_release_order': 'UNKNOWN'}))

    timed = attach_task_time(scope)
    marked = derive_scope_control_milestone(timed, recipe)
    marker = next(a for a in marked.temporal.activities if a.task == fence.id)
    assert marker.duration_basis is Basis.SOURCE_CONTROL_MILESTONE
    assert (marker.duration_lower_ticks, marker.duration_upper_ticks) == (0, 0)
    assert any(ref.get('schema') == 'scar.source-control-milestone' for ref in marker.source_refs)
    assert not any(o.kind is OK.EXECUTION_DEMAND and o.activity == marker.id
                   for o in marked.temporal.occupancies)
    assert validate_versioned_schedule(marked)['temporal']['source_control_milestones'] == 1

    def with_marker(activity, *, proof=marked.proof):
        return replace(marked, proof=proof, temporal=replace(marked.temporal,
            activities=tuple(activity if a.id == marker.id else a for a in marked.temporal.activities)))
    with pytest.raises(ValueError, match='UNKNOWN'):
        validate_versioned_schedule(with_marker(replace(marker, duration_basis=Basis.UNKNOWN)))
    with pytest.raises(ValueError, match='source-proven control milestone'):
        validate_versioned_schedule(with_marker(replace(marker, source_refs=())))
    no_proof = deepcopy(marked.proof); no_proof.pop('implementation_scope_cleanup')
    with pytest.raises(ValueError, match='source-proven control milestone'):
        validate_versioned_schedule(with_marker(marker, proof=no_proof))
    old_demand = next(o for o in timed.temporal.occupancies
                      if o.kind is OK.EXECUTION_DEMAND and o.activity == marker.id)
    with pytest.raises(ValueError, match='execution demand'):
        validate_versioned_schedule(replace(marked, temporal=replace(marked.temporal,
            occupancies=(*marked.temporal.occupancies, old_demand),
            resource_capacities=timed.temporal.resource_capacities)))


def test_source_probe_has_cleanup_cut_and_nine_cut_samples_partition_without_overlap():
    source, provenance = render_scope_boundary_probe(
        {'arrays': 1, 'scalars': 1, 'providers': 0, 'array_shapes': [[2, 3]],
         'workspace_slot_capacities': [48]},
        {'workspace_payload_bytes': 48, 'external_array_bytes': 48,
         'total_region_array_payload_bytes': 96, 'memory_budget_bytes': 96},
        [0], {'region_sha256': '0'*64, 'plan_sha256': '1'*64, 'provider_runtime': {}})
    assert len(provenance['cut_names']) == 7 and provenance['cut_names'][-1] == 'helper_cleanup_ready'
    assert source.index('ThreadAffinity caller_affinity;') < source.index('if (!record_cut(5))')
    assert source.index('if (!record_cut(5))') < source.index('\n    }\n    // Normal-success reverse RAII:')
    assert source.index('\n    }\n    // Normal-success reverse RAII:') < source.index('if (!record_cut(6))')
    assert 'std::array<uint64_t, 7> boundary_cuts' in source
    assert provenance['implementation_scope_source']['helper_cleanup_before_pool_release'] is True
    assert provenance['implementation_scope_source']['pool_release_rule']['required_value'] == (
        'CPP_STD_ARRAY_REVERSE_DESTRUCTION')
    assert provenance['scope_transformation']['pool_release_measured'] is False
    assert provenance['performance_measured'] is False and provenance['native_compiled'] is False
    assert len(CUTS) == 9 and len(PHASES) == 8
    vectors = ([100, 105, 111, 118, 126, 135, 145, 156, 168],
               [200, 206, 213, 221, 230, 240, 251, 263, 276],
               [300, 307, 315, 324, 334, 345, 357, 370, 384])
    document = dict(schema=SCHEMA, schema_version=1, cuts=list(CUTS),
        phases=deepcopy(list(PHASES)), conditions=list(CONDITIONS), missing_terms=list(MISSING),
        repetitions=3, warmup=1,
        clock=dict(api='clock_gettime', clock_id='CLOCK_MONOTONIC', unit='ns'),
        samples=[dict(id='scope:'+str(i), timestamps_ns=list(v)) for i, v in enumerate(vectors)])
    statistics = scope_boundary_statistics(document)
    assert len(statistics) == 8
    for i, vector in enumerate(vectors):
        assert sum(row['samples_ns'][i] for row in statistics) == vector[-1]-vector[0]
    assert statistics[-2]['phase'] == 'helper_cleanup'
    invalid = []
    bad = deepcopy(document); bad['samples'][0]['timestamps_ns'].pop(7); invalid.append(bad)
    bad = deepcopy(document); bad['samples'][0]['timestamps_ns'].insert(7, 151); invalid.append(bad)
    bad = deepcopy(document); bad['samples'][0]['timestamps_ns'][4] = 117; invalid.append(bad)
    bad = deepcopy(document); bad['cuts'].pop(7); invalid.append(bad)
    bad = deepcopy(document); bad['phases'][-1]['start'] = 'restore_ready'; invalid.append(bad)
    for bad in invalid:
        with pytest.raises(ValueError):
            scope_boundary_statistics(bad)
