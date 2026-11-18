"""Adapt admitted whole-stateful-call profiles to the common cost record.

Stage timings and repeated-session totals are not entry costs.  A cost known
under the original profile contract does not establish whole-entry physical
CPU demands, memory residency, source admission or deployment permission.
"""
from copy import deepcopy
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import statistics
import struct
import subprocess
import sys

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.implementation_plan import CostEstimate
from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.record_codec import encode
from scar.optimization.deployment_cost import _digest
from scar.optimization.implementation_costs import CostQuery, query_key
from scar.optimization.stateful_cpu_choices import (
    _cost, _domain, stateful_cpu_cost_context,
)


SCHEMA = 'scar.stateful-entry-cost'
SCOPE = 'ONE_ORIGINAL_METHOD_CALL_SITE'
_DOMAIN_SCHEMA = 'scar.stateful-cpu-domain'
_PATH_TYPES = (Path, type(Path()))


def _refs(rows):
    result, seen = [], set()
    for row in rows:
        key = _digest(row)
        if key not in seen:
            seen.add(key); result.append(deepcopy(row))
    return result


def _domain_material(value):
    reference = None
    if isinstance(value, (str, Path)):
        path = Path(value).resolve()
        if path.is_dir():
            path = path/'domain.json'
        reference = artifact_reference(path, _DOMAIN_SCHEMA)
        value = read_artifact(reference)
    return _domain(value), reference


def _entry_json_tree(value):
    """Inspect exact builtin containers only, without custom object protocols."""
    pending, active = [(value, False)], set()
    while pending:
        current, leaving = pending.pop()
        kind = type(current)
        if leaving:
            active.remove(id(current))
        elif current is None or kind in (bool, int, str):
            continue
        elif kind is float:
            if not math.isfinite(current):
                return False
        elif kind in (dict, list):
            if id(current) in active:
                return False
            if kind is dict and any(type(key) is not str for key in current):
                return False
            active.add(id(current))
            pending.append((current, True))
            pending.extend((item, False) for item in (current.values() if kind is dict else current))
        else:
            return False
    return True


def _entry_cacheable_input(value, *, domain=False):
    if type(value) is str or type(value) in _PATH_TYPES:
        return True
    if not domain and type(value) is ProviderFactReference:
        artifact = value.artifact
        return (type(artifact) is ProviderArtifact and type(value.locator) is str
            and all(type(item) is str for item in (artifact.uri, artifact.sha256, artifact.schema))
            and type(artifact.size_bytes) is int)
    return _entry_json_tree(value)


def _entry_input_key(value, *, domain=False):
    """Bind the original representation without changing admission semantics."""
    if type(value) is str or type(value) in _PATH_TYPES:
        path = Path(value).resolve() if domain else Path(value).absolute()
        if domain and path.is_dir():
            path = path/'domain.json'
        raw = path.read_bytes()
        return dict(kind='DOMAIN_PATH' if domain else 'PROFILE_PATH', path=str(path),
            sha256=sha256(raw).hexdigest(), size_bytes=len(raw))
    # Only the exact SCAR FactReference is materialized for a profile key.
    # Domain ArtifactRefs/FactReferences remain subject to the original reader.
    if not domain and type(value) is ProviderFactReference:
        artifact = value.artifact
        return dict(kind='PROFILE_FACT_REFERENCE', content=dict(locator=value.locator,
            artifact=dict(uri=artifact.uri, sha256=artifact.sha256,
                size_bytes=artifact.size_bytes, schema=artifact.schema)))
    return dict(kind='INLINE', content=value)


def _entry_live_key():
    """Recheck cheap live inputs which a file-read capture cannot observe."""
    import numpy as np
    from scar.codegen import cpp
    from numpy._core import _multiarray_umath
    from scar.runtime.compiled_deployment import thread_environment
    command = shlex.split(os.environ.get('CXX', 'g++'))
    if not command or not shutil.which(command[0]):
        raise ValueError('C++20 compiler unavailable')
    identity = subprocess.run([*command, '--version'], capture_output=True,
        text=True, check=True).stdout.splitlines()[0]
    return dict(affinity=sorted(os.sched_getaffinity(0)), environment=dict(os.environ),
        thread_environment=thread_environment(), cwd=str(Path.cwd().resolve()),
        compiler=dict(command=command, executable=shutil.which(command[0]), identity=identity,
            options=list(cpp.COMPILER_OPTIONS)),
        runtime=dict(platform=sys.platform, byteorder=sys.byteorder, maxsize=sys.maxsize,
            pointer_bits=8*struct.calcsize('P'), python=sys.version,
            numpy_version=np.__version__, ndarray_c_abi=int(_multiarray_umath._get_ndarray_c_version()),
            numpy_power_identity=id(np.power), original_power_identity=id(_multiarray_umath.power),
            ufunc_type_identity=id(np.ufunc)),
        callables=[id(value) for value in (_domain, _cost, stateful_cpu_cost_context,
            _digest, query_key, encode, CostEstimate, _derive_stateful_entry_cost_uncached)])


def _entry_implementation_key(transaction):
    paths = {Path(__file__).resolve(), Path(sys.modules[CostEstimate.__module__].__file__).resolve()}
    for fn in (_domain, _cost, stateful_cpu_cost_context, _digest, query_key, encode):
        paths.add(Path(fn.__code__.co_filename).resolve())
    result = []
    for path in sorted(paths):
        raw = path.read_bytes(); digest = sha256(raw).hexdigest()
        transaction.watch(path, digest, len(raw))
        result.append(dict(path=str(path), sha256=digest, size_bytes=len(raw)))
    return result


def _watch_entry_implementations(transaction):
    """Capture lazy-imported Python implementation dependencies as well as data.

    Already imported code need not open its source while executing.  Watching
    loaded SCAR source modules here keeps such implementation changes from
    being hidden by the result memo.  This runs inside its read capture.
    """
    root = Path(__file__).resolve().parents[1]
    paths = set()
    for name, module in tuple(sys.modules.items()):
        if name != 'scar' and not name.startswith('scar.'):
            continue
        filename = getattr(module, '__file__', None)
        if filename is None:
            continue
        path = Path(filename).resolve()
        if path.suffix == '.py' and path.is_relative_to(root):
            paths.add(path)
    for path in sorted(paths):
        raw = path.read_bytes()
        transaction.watch(path, sha256(raw).hexdigest(), len(raw))


def derive_stateful_entry_cost(domain, candidate_id, profiles=()):
    """Replay once per cold transaction; retain the original uncached API.

    Reuse is confined to a successful Python metadata read capture.  The
    transaction rechecks its complete captured dependencies on hits and exit;
    callers cannot supply a flag asserting prior verification.
    """
    from scar.analysis.replay_transaction import current_transaction
    transaction = current_transaction()
    if (transaction is None or type(profiles) not in (tuple, list) or type(candidate_id) is not str
            or not _entry_cacheable_input(domain, domain=True)
            or not all(_entry_cacheable_input(value) for value in profiles)):
        return _derive_stateful_entry_cost_uncached(domain, candidate_id, profiles)
    try:
        inputs = dict(domain=_entry_input_key(domain, domain=True), candidate_id=candidate_id,
            profiles=[_entry_input_key(value) for value in profiles],
            live=_entry_live_key())
        # Test serializability before enrolling implementation dependencies.
        _digest(inputs)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, ImportError, subprocess.SubprocessError):
        # Unsupported key representations keep their former error/UNKNOWN
        # behavior instead of acquiring a new cache-admission policy.
        return _derive_stateful_entry_cost_uncached(domain, candidate_id, profiles)
    # Transaction integrity errors must propagate, never fall back around a
    # changed source dependency and return an apparently successful report.
    inputs['implementation'] = _entry_implementation_key(transaction)
    key = _digest(inputs)

    def factory():
        _watch_entry_implementations(transaction)
        result = _derive_stateful_entry_cost_uncached(domain, candidate_id, profiles)
        _watch_entry_implementations(transaction)
        return result

    return transaction.memo('stateful-entry-cost/v1', key, factory, capture_reads=True,
        identity_queries=(tuple(inputs['live']['compiler']['command']+['--version']),))


def _verify_entry_profile_model(document):
    """Replay recognized model derivations through their official producers."""
    basis=document.get('derivation',{}).get('basis')
    if basis=='SOURCE_ACCOUNTED_SERIAL_ACTIVITY_POINT_MODEL':
        from scar.optimization.stateful_activity_model import verify_activity_entry_profile
        return verify_activity_entry_profile(document)
    if basis=='MATCHED_CANONICAL_GUARDED_ENTRY_POINT_MODEL':
        from scar.compiler.calibrate_stateful_entry import verify_canonical_entry_profile
        return verify_canonical_entry_profile(document)
    return None


def _derive_stateful_entry_cost_uncached(domain, candidate_id, profiles=()):
    """Replay existing whole-call cost admission without timing or loading code.

    Profiles may contain entries for other candidates; those remain rejected
    offers.  Multiple distinct applicable offers are ambiguous, never searched
    for the fastest estimate.  The shared stateful domain reader may query the
    compiler identity, but this adapter never compiles or invokes target code.
    """
    domain, domain_reference = _domain_material(domain)
    context = stateful_cpu_cost_context(domain, candidate_id)
    scope = context['scope']
    if (scope.get('boundary') != 'WARM_SESSION' or scope.get('call_scope') != SCOPE
            or type(scope.get('reuse_count')) is not int or scope['reuse_count'] != 1
            or scope.get('includes') != 'ALL_IMPLEMENTATION_WORK_FROM_CALL_DISPATCH_THROUGH_RETURN'):
        raise ValueError('exact warm single whole-method entry scope required')
    if type(profiles) not in (tuple, list):
        raise ValueError('explicit list or tuple of whole-stateful-call profiles required')
    base = read_artifact(domain['base_stage_plan'])
    candidate = next((row for row in domain['candidates'] if row['id'] == candidate_id), None)
    sources = [dict(artifact=domain[name], locator='#') for name in
        ('region', 'base_stage_plan', 'target_snapshot', 'inventory')]
    if domain_reference is not None:
        sources.append(dict(artifact=domain_reference, locator='#'))
    sources.append(dict(domain_id=domain['domain_id'], candidate_id=candidate_id,
        pointer='#/original_no_op' if candidate is None else
            '#/candidates/' + str(domain['candidates'].index(candidate))))
    sources = _refs(sources)
    query = CostQuery(SCOPE, context, tuple(sources))
    offers, accepted = [], {}
    for index, profile in enumerate(profiles):
        cost, issues, document = _cost(profile, context)
        issues = list(issues)
        if cost is not None:
            try:
                _verify_entry_profile_model(document)
            except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
                issues.append('whole-entry model derivation replay failed: ' + str(error))
        if cost is not None:
            if cost['tick_seconds'] != 1e-9:
                issues.append('whole-call profile must use the existing integer-nanosecond writer convention')
            if type(cost['duration_ticks']) is not int or not 0 <= cost['duration_ticks'] < 2**60:
                issues.append('whole-call duration must fit the common nonnegative integer-nanosecond contract')
        admitted = cost is not None and not issues
        offers.append(dict(index=index, admitted=admitted,
            profile_sha256=_digest(document) if document is not None else None,
            source=deepcopy(cost['source']) if cost is not None else None,
            frontier=issues))
        if admitted:
            accepted.setdefault(_digest(document), (cost, document))
    frontier = []
    if not accepted:
        frontier.append(dict(field='cost', reason='NO_ADMITTED_EXACT_CONTEXT_WHOLE_METHOD_CALL_PROFILE'))
        for offer in offers:
            frontier.extend(dict(field='profiles.' + str(offer['index']), reason=reason)
                for reason in offer['frontier'])
    elif len(accepted) > 1:
        frontier.append(dict(field='cost', reason='MULTIPLE_DISTINCT_APPLICABLE_WHOLE_METHOD_PROFILES'))
    accepted_cost, accepted_profile = next(iter(accepted.values())) if len(accepted) == 1 else (None, None)
    if accepted_cost is not None:
        for key in ('context_evidence',):
            sources.extend(accepted_profile.get(key, ()))
        sources.extend(accepted_profile['estimate'].get('evidence', ()))
        sources.extend(accepted_profile['derivation'].get('evidence', ()))
        for name in ('source_profile', 'rebase_proof'):
            if name in accepted_profile:
                sources.append(dict(artifact=accepted_profile[name], locator='#'))
        sources = _refs(sources)
        # Canonical composition and observed-original forward expectations
        # remain declared models; neither creates observed target components.
        estimate = CostEstimate(accepted_cost['duration_ticks'], None, None,
            'DECLARED_MODEL', query_key(query), tuple(sources), ())
    else:
        estimate = CostEstimate(None, None, None, 'UNKNOWN', query_key(query), tuple(sources),
            tuple(item['reason'] for item in frontier))
    obligations = [
        dict(field='whole_entry_cpu_execution_domain',
            reason=('Original caller affinity and native-thread environment do not establish a whole-entry CPU demand or reservation.'
                if candidate is None else
                'Candidate binding covers internal native stages; caller guards, packing, allocation, providers and return do not have a proven whole-entry CPU demand.'),
            source_refs=deepcopy(sources)),
        dict(field='whole_entry_opaque_memory_and_residency',
            reason='Original payload budget does not establish opaque runtime memory, original NumPy peak memory, process RSS or physical NUMA residency.',
            source_refs=deepcopy(sources)),
    ]
    return dict(schema=SCHEMA, schema_version=1, candidate_id=candidate_id,
        domain_id=domain['domain_id'], domain_sha256=_digest(domain),
        region=deepcopy(domain['region']), target=deepcopy(domain['target_snapshot']),
        context=deepcopy(context), query=dict(scope=SCOPE, context=deepcopy(context)),
        cost=encode(estimate), source_refs=deepcopy(sources), frontier=frontier,
        profiles=offers, admitted_profile_sha256=_digest(accepted_profile) if accepted_profile is not None else None,
        derivation=deepcopy(accepted_cost['derivation']) if accepted_cost is not None else None,
        candidate_qualification=dict(original_no_op=candidate is None,
            legal_under_guards=True if candidate is None else candidate['legal_under_guards'],
            legality_frontier=[] if candidate is None else deepcopy(candidate['legality_frontier']),
            lowering_admitted=False if candidate is None else candidate.get('lowering_admitted', False)),
        resource_qualification=dict(status='UNRESOLVED',
            candidate_internal_stage_binding=None if candidate is None else deepcopy(candidate['binding']),
            stage_binding_is_whole_entry_binding=False,
            caller_affinity_cpu_ids=deepcopy(domain['live_constraints']['process_affinity_cpu_ids']),
            caller_affinity_is_exclusive_reservation=False,
            known_native_payload=dict(external_array_bytes=base['external_array_bytes'],
                workspace_payload_bytes=base['workspace_payload_bytes'],
                total_region_array_payload_bytes=base['total_region_array_payload_bytes'],
                memory_budget_bytes=base['memory_budget_bytes'], budget_scope=base['budget_scope']),
            original_numpy_peak_memory='UNKNOWN', opaque_runtime_memory='UNKNOWN',
            physical_numa_residency='UNKNOWN', whole_entry_physical_feasibility_proven=False),
        unresolved_resources=obligations,
        scope_interpretation='WHOLE_METHOD_PROFILE_REPLAY_NOT_STAGE_OR_SESSION_REPEAT_COST',
        source_admission='RETAINED_DOMAIN_REPLAY_NOT_NEW_F4_OR_NATIVE_ARTIFACT_ADMISSION',
        cost_interpretation='DECLARED_FORWARD_EXPECTATION_NOT_OBSERVED_COMPONENTS_OR_PHYSICAL_BOUNDS',
        target_program_executed=False, native_loaded=False, compiled=False,
        deployment_authorized=False)


def verify_stateful_entry_cost(report, domain, candidate_id, profiles=()):
    """Recompute domain, profile evidence, query identity and resource qualifiers."""
    expected = derive_stateful_entry_cost(domain, candidate_id, profiles)
    if _digest(report) != _digest(expected):
        raise ValueError('stateful entry cost differs from whole-call profile/domain replay')
    return expected


def _evaluate_legacy_stateful_entry_holdout(model, holdout, policy):
    """Qualify prediction accuracy from retained paired complete-method clocks.

    Rejected predictions stay available as declared analysis models. Their
    raw profiles are not returned as qualified inputs for automatic selection.
    This never changes the frozen prediction, threshold or target samples.
    """
    from scar.optimization.stateful_activity_model import read_activity_entry_model
    def reference(value, schema):
        return value if type(value) is dict else artifact_reference(Path(value).resolve(), schema)
    model_ref=reference(model,'scar.stateful-activity-entry-model')
    holdout_ref=reference(holdout,'scar.real-caller-cost-holdout')
    policy_ref=reference(policy,'scar.conditional-entry-holdout-policy')
    model=read_activity_entry_model(model_ref['uri'])
    if read_artifact(model_ref)!=model:raise ValueError('frozen model differs during source replay')
    observed,rules=read_artifact(holdout_ref),read_artifact(policy_ref)
    if observed.get('schema_version')!=1 or rules.get('schema_version')!=1:
        raise ValueError('supported holdout and frozen policy versions required')
    facts=read_artifact(model['facts']);activity=read_artifact(model['activity_costs'])
    binding=observed['frozen_model_binding'];protocol=observed['protocol']
    if (rules['frozen_model']!=model_ref or observed['frozen_model']!=model_ref
            or rules['predicted_native_method_ns']!=model['duration_ns']
            or rules['policy_frozen_before_holdout'] is not True
            or rules['no_fit_or_threshold_adjustment_from_holdout'] is not True
            or observed['model_fitted_or_updated_from_holdout'] is not False
            or binding['facts']!=model['facts'] or binding['activity_costs']!=model['activity_costs']
            or binding['domain']!=activity['domain'] or binding['candidate_id']!=activity['candidate_id']
            or protocol['method_scope']['call_scope']!=SCOPE
            or protocol['method_scope']['boundary']!=facts['context']['scope']['boundary']
            or protocol['caller_cpu_ids']!=read_artifact(activity['domain'])['live_constraints']['process_affinity_cpu_ids']
            or protocol['per_step_array_content_reads_by_observer']!=0
            or protocol['caller_affinity_changed_by_harness'] is not False):
        raise ValueError('holdout differs from frozen model, policy, source domain or method boundary')
    for name in ('harness','correctness_helpers'):
        read_artifact(observed[name],json_document=False)
    prior=read_artifact(observed['prior_independent_full_trajectory_validation'])
    specialization=read_artifact(observed['specialization'])
    candidate=read_artifact(specialization['candidate'])
    native_metadata=read_artifact(observed['native_artifact'])
    if (prior['status']!='PASSED_EXPLICIT_NATIVE_TRIAL'
            or prior['specialization']!=observed['specialization']
            or prior['native_artifact']!=observed['native_artifact']
            or candidate['native_artifact']!=observed['native_artifact']
            or candidate['candidate_id']!=binding['candidate_id']
            or candidate['program']!=binding['program']
            or candidate['program']!=specialization['program']
            or specialization['domain']!=binding['domain']
            or specialization['input_shapes']!=facts['context']['input_shapes']
            or specialization['region_sha256']!=facts['context']['region_sha256']
            or native_metadata['region_sha256']!=facts['context']['region_sha256']
            or read_artifact(binding['program'])['entry']!=observed['entry']['uri']):
        raise ValueError('holdout candidate, entry and independent correctness evidence differ')
    read_artifact(observed['entry'],json_document=False)
    from scar.compiler.stateful_model import stateful_from_dict
    domain=read_artifact(activity['domain'])
    region=stateful_from_dict(read_artifact(domain['region']))
    before,after=observed['original_source_refs_before'],observed['original_source_refs_after']
    source_hashes={row.path:row.sha256 for row in (region.source,*(callee.source for callee in region.callees))}
    source_hashes[observed['entry']['uri']]=observed['entry']['sha256']
    if before!=after or set(before)!=set(source_hashes):
        raise ValueError('complete original source version set changed during holdout')
    for path,reference in before.items():
        if reference['uri']!=path or reference['sha256']!=source_hashes[path]:
            raise ValueError('holdout source versions differ from original region and entry')
        read_artifact(reference,json_document=False)
    count,warmup=rules['implementation_pairs'],rules['warmup_pairs']
    if (type(count) is not int or count<3 or type(warmup) is not int or warmup<1
            or protocol['measured_pairs']!=count or protocol['warmup_pairs']!=warmup
            or len(observed['measured_pairs'])!=count or len(observed['warmups'])!=warmup):
        raise ValueError('complete frozen paired sample and warmup counts required')
    groups={'original':[],'native_trial':[]};native_calls=fallbacks=0
    required=('final_complete_arrays_halo_and_scalar_types_bytes_equal',
        'step_shape_dtype_and_scalar_context_trajectory_equal','stdout_equal','stderr_equal',
        'output_file_names_and_bytes_equal','every_native_step_executed')
    correct=True
    for phase,rows in (('warmup',observed['warmups']),('measurement',observed['measured_pairs'])):
        if [row['index'] for row in rows]!=list(range(len(rows))):
            raise ValueError('unique ordered pair indices required')
        for pair in rows:
            if pair['order']!=(['A','B'] if pair['index']%2==0 else ['B','A']):
                raise ValueError('frozen alternating pair order differs')
            a,b=pair['original']['steps'],pair['native_trial']['steps']
            if not a or len(a)!=len(b):raise ValueError('matching nonempty complete method trajectories required')
            for steps in (a,b):
                if [row['ordinal'] for row in steps]!=list(range(len(steps))):
                    raise ValueError('complete consecutive method ordinals required')
                for step in steps:
                    inputs=step['inputs']
                    if set(inputs)!={row.id for row in region.inputs}:
                        raise ValueError('holdout declared input ID set differs')
                    for spec in region.inputs:
                        current=inputs[spec.id]
                        if current['path']!=list(spec.path) or current['kind']!=spec.kind:
                            raise ValueError('holdout declared input path or kind differs')
                        if spec.kind=='array' and (current['shape']!=facts['context']['input_shapes'][spec.id]
                                or current['dtype']!='<f8' or current['device']!='cpu'
                                or current['c_contiguous'] is not True or current['tensor_contents_read'] is not False):
                            raise ValueError('holdout array geometry or measurement regime differs')
            if any(row['guard_statistics']['native_calls']!=0 or row['guard_statistics']['fallback_calls']!=0 for row in a):
                raise ValueError('original timing path contains guarded or native calls')
            correct &= all(pair['comparison'].get(key) is True for key in required)
            for key,steps in (('original',a),('native_trial',b)):
                if any(type(row['elapsed_ns']) is not int or row['elapsed_ns']<=0 for row in steps):
                    raise ValueError('positive original integer method clock samples required')
                if phase=='measurement':groups[key].append(statistics.mean(row['elapsed_ns'] for row in steps))
            correct &= [row['inputs'] for row in a]==[row['inputs'] for row in b]
            for step in b:
                stats=step['guard_statistics']
                correct &= stats['native_calls']==1 and stats['fallback_calls']==0
                if phase=='measurement':
                    native_calls+=stats['native_calls'];fallbacks+=stats['fallback_calls']
    original=statistics.median(groups['original']);native=statistics.median(groups['native_trial'])
    error=abs(model['duration_ns']-native)/native;gain=1-native/original
    threshold,minimum=rules['max_relative_prediction_error'],rules['minimum_observed_gain']
    if any(type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<1
           for value in (threshold,minimum)):
        raise ValueError('finite frozen relative-error and gain thresholds in [0,1) required')
    if rules['relative_error_denominator']!='OBSERVED_NATIVE_MEDIAN_OF_SEVEN_EPISODE_MEAN_METHOD_DURATIONS':
        raise ValueError('frozen prediction error metric differs')
    # Legacy samples can retain useful observations without receiving a new
    # policy retroactively. A future capture must bind all three before work.
    policy_bound=(observed.get('frozen_policy')==policy_ref
        and observed.get('policy_frozen_before_target') is True)
    timer_bound=(rules.get('method_scope')==protocol['method_scope']
        and rules.get('holdout_harness')==observed['harness'])
    qualified=correct and error<=threshold and policy_bound and timer_bound
    for ref in (model_ref,holdout_ref,policy_ref):read_artifact(ref)
    return dict(schema='scar.stateful-entry-model-qualification',schema_version=1,
        model=model_ref,holdout=holdout_ref,policy=policy_ref,
        status='HOLDOUT_CONTEXT_QUALIFIED' if qualified else 'HOLDOUT_PREDICTION_REJECTED',
        predicted_native_ns=model['duration_ns'],original_median_ns=original,native_median_ns=native,
        method_original_over_native_ratio=original/native,relative_prediction_error=error,
        prediction_error_threshold=threshold,prediction_error_threshold_passed=error<=threshold,
        policy_precommitted_in_holdout=policy_bound,full_timer_scope_and_harness_bound=timer_bound,
        qualification_frontiers=[reason for passed,reason in (
            (correct,'PAIRED_SOURCE_STATE_OR_NATIVE_PATH_CORRECTNESS_FAILED'),
            (error<=threshold,'FROZEN_PREDICTION_ERROR_THRESHOLD_EXCEEDED'),
            (policy_bound,'POLICY_REFERENCE_NOT_BOUND_BEFORE_TARGET_IN_CAPTURE'),
            (timer_bound,'FULL_TIMER_SCOPE_OR_HARNESS_NOT_BOUND_IN_FROZEN_POLICY')) if not passed],
        observed_gain=gain,observed_gain_threshold=minimum,observed_gain_threshold_passed=gain>=minimum,
        measured_native_calls=native_calls,measured_fallback_calls=fallbacks,correctness_passed=bool(correct),
        qualified_profile=model['profile'] if qualified else None,
        original_declared_profile_retained_for_analysis=model['profile'],
        target_samples_fitted=False,model_or_threshold_changed=False,
        cost_qualification_scope='ONLY_RETAINED_SAME_SOURCE_SHAPE_CALLER_AND_GUARDED_ENTRY_CONTEXT',
        physical_resource_qualification='SEPARATE_UNRESOLVED',whole_program_speedup_claimed=False,
        default_backend='NO_OP',deployment_authorized=False)


def _future_holdout_state(state,region,context,*,final):
    """Validate recorded geometry/scalars without reading target array data."""
    import struct
    if type(state)is not dict or set(state)!={row.id for row in region.inputs}:
        raise ValueError('future holdout declared input ID set differs')
    for spec in region.inputs:
        row=state[spec.id]
        if type(row)is not dict or row.get('path')!=list(spec.path) or row.get('kind')!=spec.kind:
            raise ValueError('future holdout declared input path or kind differs')
        if spec.kind=='array':
            fields={'kind','path','dtype','shape','strides','device','c_contiguous','aligned','tensor_contents_read'}
            if final:fields|={'payload_bytes','payload_sha256'}
            shape=context['input_shapes'][spec.id]
            if (set(row)!=fields or row['shape']!=shape or row['dtype']!='<f8' or row['device']!='cpu'
                    or row['c_contiguous']is not True or row['aligned']is not True
                    or row['tensor_contents_read']is not final
                    or row['strides']!=[shape[1]*8,8]):
                raise ValueError('future holdout array geometry or content-read regime differs')
            if final and (row['payload_bytes']!=8*shape[0]*shape[1]
                    or type(row['payload_sha256'])is not str or len(row['payload_sha256'])!=64
                    or any(c not in '0123456789abcdef'for c in row['payload_sha256'])):
                raise ValueError('complete final array receipt required')
        else:
            scalar_type=row.get('type');fields={'kind','path','type','representation'}
            if scalar_type in ('builtins.float','numpy.float64'):fields.add('bits')
            if set(row)!=fields:raise ValueError('complete known scalar observation fields required')
            try:
                if scalar_type=='builtins.int':
                    value=int(row['representation'])
                    valid=str(value)==row['representation'] and abs(value)<=2**53
                elif scalar_type in ('builtins.float','numpy.float64'):
                    value=float.fromhex(row['representation'])
                    valid=math.isfinite(value) and value.hex()==row['representation'] and struct.pack('=d',value).hex()==row['bits']
                else:valid=False
            except (ValueError,TypeError,OverflowError):valid=False
            if not valid:raise ValueError('finite exact declared scalar observation required')


def _evaluate_future_stateful_entry_holdout(model_ref,holdout_ref,policy_ref):
    from scar.compiler.stateful_holdout import (DESIGN,POLICY,REPORT,TIMER,
        verify_stateful_holdout_design,_distribution,_holdout_model_material,_prediction_evaluation)
    from scar.compiler.stateful_model import stateful_from_dict
    from hashlib import sha256
    observed=read_artifact(holdout_ref);rules=read_artifact(policy_ref)
    fields={'schema','schema_version','status','design','capture_epoch','observation_origin','frozen_policy',
        'frozen_model','timer','harness','implementation_sources','original_source_refs_before',
        'original_source_refs_after','policy_frozen_before_target','design_readback_before_target',
        'design_readback_after_target','caller_affinity_changed_by_harness','original_descriptors_restored',
        'pairs','summaries','guard_statistics_including_warmups','frozen_policy_evaluation',
        'per_step_array_content_reads_by_observer','final_state_reads_per_episode','independent_prior_correctness',
        'performance_measured','model_fitted_from_holdout','model_qualification_promoted',
        'deployment_authorized','whole_program_speedup_measured','output_comparison_scope','noise_limitations'}
    if (set(observed)!=fields or (observed['schema'],observed['schema_version'])!=(REPORT,2)
            or observed['status']!='OBSERVED_FIXED_IMPLEMENTATIONS'
            or (rules.get('schema'),rules.get('schema_version'))!=(POLICY,2)):
        raise ValueError('complete known future holdout v2 and frozen policy v2 required')
    design_ref=observed['design']
    if design_ref.get('schema')!=DESIGN:raise ValueError('future source design reference required')
    design=verify_stateful_holdout_design(design_ref['uri'],expected_sha256=design_ref['sha256'])
    if read_artifact(design_ref)!=design:raise ValueError('future design reference differs from cold replay')
    request=design['request'];epoch=request['capture_epoch']
    expected_sources={**design['source_refs'],request['entry']['uri']:request['entry']}
    if (observed['frozen_model']!=model_ref or request['model']!=model_ref
            or observed['frozen_policy']!=policy_ref or design['frozen_policy']!=policy_ref
            or rules!=design['policy_contract'] or rules.get('purpose')!='FOR_FUTURE_CAPTURE'
            or rules.get('prior_samples_authorized')is not False
            or observed['capture_epoch']!=epoch or rules.get('capture_epoch')!=epoch
            or observed['timer']!=TIMER or rules.get('timer')!=TIMER
            or observed['harness']!=rules.get('harness')
            or observed['harness']!=design['implementation_sources']['compiler/stateful_holdout.py']
            or observed['implementation_sources']!=design['implementation_sources']
            or observed['original_source_refs_before']!=expected_sources
            or observed['original_source_refs_after']!=expected_sources
            or observed['independent_prior_correctness']!=request['prior_correctness']
            or observed['per_step_array_content_reads_by_observer']!=0
            or observed['final_state_reads_per_episode']!=1
            or observed['output_comparison_scope']!='CONFIGURED_OUTPUT_DIRECTORY_ONLY'):
        raise ValueError('future capture reference/timer/harness/source/epoch binding differs')
    for key in ('policy_frozen_before_target','design_readback_before_target','design_readback_after_target','original_descriptors_restored'):
        if observed[key]is not True:raise ValueError('future precommit and restoration evidence required')
    for key in ('caller_affinity_changed_by_harness','model_fitted_from_holdout','model_qualification_promoted',
                'deployment_authorized','whole_program_speedup_measured'):
        if observed[key]is not False:raise ValueError('future capture cannot alter policy, placement or qualification')
    origin=observed['observation_origin']
    if origin not in ('COLLECTOR_TARGET_EXECUTION','SYNTHETIC_METADATA_VALIDATION'):
        raise ValueError('known future capture observation origin required')
    real=origin=='COLLECTOR_TARGET_EXECUTION'
    if observed['performance_measured']is not real:raise ValueError('synthetic metadata cannot claim target measurement')
    for ref in (*design['implementation_sources'].values(),*expected_sources.values()):
        read_artifact(ref,json_document=False)
    model,facts,activity,canonical_binding=_holdout_model_material(model_ref)
    domain=_domain(read_artifact(design['domain']));candidate=read_artifact(design['candidate'])
    context=stateful_cpu_cost_context(domain,candidate['candidate_id'])
    if (facts['context']!=context or design['context']!=context
            or request['caller_cpu_ids']!=domain['live_constraints']['process_affinity_cpu_ids']):
        raise ValueError('future captured caller/model context differs from original domain')
    if canonical_binding is None:
        if activity['domain']!=design['domain'] or activity['candidate_id']!=candidate['candidate_id']:
            raise ValueError('future source-accounted activity model domain differs')
    else:
        from scar.analysis.stateful_entry_protocol import derive_stateful_entry_protocol
        protocol=derive_stateful_entry_protocol(design['native_artifact'],
            trial=True,exclusive_inputs=True,selection=None)
        if (design.get('canonical_model_binding')!=canonical_binding or design['activity_costs'] is not None
                or model['domain']!=design['domain'] or model['candidate_id']!=candidate['candidate_id']
                or model['specialization']!=request['specialization'] or model['entry_protocol']!=protocol):
            raise ValueError('future canonical model or exact guarded entry protocol differs')
    region=stateful_from_dict(read_artifact(domain['region']))
    pairs=observed['pairs'];planned=design['episodes']
    if type(pairs)is not list or len(pairs)!=9 or len(planned)!=18:
        raise ValueError('complete future two warmup and seven measurement pairs required')
    groups={'A':[],'B':[]};pooled={'A':[],'B':[]};native_calls=fallbacks=0;total_native=total_fallback=0
    correct=True
    for ordinal,pair in enumerate(pairs):
        phase='warmup'if ordinal<2 else'measurement';index=ordinal if ordinal<2 else ordinal-2
        order=['A','B']if index%2==0 else['B','A']
        if (set(pair)!={'phase','index','order','checks','implementations'} or pair['phase']!=phase
                or pair['index']!=index or pair['order']!=order or set(pair['implementations'])!={'A','B'}):
            raise ValueError('future frozen pair phase/index/order differs')
        implementations=pair['implementations']
        for label in order:
            row=implementations[label]
            if set(row)!={'steps','final_state','argv','execution','stdout','stderr','files'}:
                raise ValueError('complete known future episode fields required')
            plan=next(p for p in planned if(p['phase'],p['index'],p['label'])==(phase,index,label))
            if (plan['capture_epoch']!=epoch or row['argv']!=plan['argv']
                    or row['execution'].get('entry_path')!=request['entry']['uri']
                    or row['execution'].get('argv')!=plan['argv'] or row['execution'].get('exit_code')!=0):
                raise ValueError('future original entry argv/output token or epoch differs')
            if type(row['stdout'])is not str or type(row['stderr'])is not str:
                raise ValueError('original text streams required')
            output=plan['program_output_directory']
            actual={}
            if output is not None:
                root=Path(output)
                for path in sorted(root.rglob('*')):
                    if path.is_file():
                        raw=path.read_bytes();actual[str(path.relative_to(root))]=dict(size_bytes=len(raw),sha256=sha256(raw).hexdigest())
            if row['files']!=actual:raise ValueError('configured output files differ from captured receipts')
            _future_holdout_state(row['final_state'],region,context,final=True)
            steps=row['steps']
            if type(steps)is not list or not steps:raise ValueError('nonempty future method trajectory required')
            for step_index,step in enumerate(steps):
                if (set(step)!={'ordinal','elapsed_ns','inputs','capture_epoch','native_calls','fallback_calls'}
                        or step['ordinal']!=step_index or step['capture_epoch']!=epoch
                        or type(step['elapsed_ns'])is not int or step['elapsed_ns']<=0
                        or type(step['native_calls'])is not int or type(step['fallback_calls'])is not int):
                    raise ValueError('positive complete future clocks and epoch required')
                _future_holdout_state(step['inputs'],region,context,final=False)
                if label=='A' and(step['native_calls']!=0 or step['fallback_calls']!=0):
                    raise ValueError('original future path contains native or guarded calls')
                if label=='B':
                    correct &= step['native_calls']==1 and step['fallback_calls']==0
                    total_native+=step['native_calls'];total_fallback+=step['fallback_calls']
                    if phase=='measurement':native_calls+=step['native_calls'];fallbacks+=step['fallback_calls']
            if phase=='measurement':
                groups[label].append(statistics.mean(s['elapsed_ns']for s in steps))
                pooled[label].extend(s['elapsed_ns']for s in steps)
        a,b=implementations['A'],implementations['B']
        native=sum(s['native_calls']for s in b['steps']);fallback=sum(s['fallback_calls']for s in b['steps'])
        checks=dict(final_state_equal=a['final_state']==b['final_state'],stdout_equal=a['stdout']==b['stdout'],
            stderr_equal=a['stderr']==b['stderr'],configured_output_files_equal=a['files']==b['files'],
            context_trajectory_equal=[s['inputs']for s in a['steps']]==[s['inputs']for s in b['steps']],
            native_success=native==len(b['steps'])and native>0 and fallback==0)
        if pair['checks']!=checks:raise ValueError('future pair correctness claims differ from recorded evidence')
        correct &= all(checks.values())
    expected_summaries={label:dict(episode_means=_distribution(groups[label]),method_calls=_distribution(pooled[label]))for label in ('A','B')}
    if observed['summaries']!=expected_summaries:raise ValueError('future summary differs from original raw clock samples')
    total=observed['guard_statistics_including_warmups']
    if (set(total)!={'native_calls','fallback_calls','reasons'} or total['native_calls']!=total_native
            or total['fallback_calls']!=total_fallback or type(total['reasons'])is not dict
            or total_fallback==0 and total['reasons']!={}):
        raise ValueError('future total guard statistics differ')
    original=statistics.median(groups['A']);native=statistics.median(groups['B'])
    threshold=rules['max_relative_prediction_error'];minimum=rules['minimum_observed_gain']
    expected_evaluation=_prediction_evaluation(model['duration_ns'],original,native,rules)
    error=expected_evaluation['relative_prediction_error'];gain=expected_evaluation['observed_method_gain']
    if observed['frozen_policy_evaluation']!=expected_evaluation:raise ValueError('future policy evaluation differs from raw samples')
    qualified=real and bool(correct)and error<=threshold and gain>=minimum
    verify_stateful_holdout_design(design_ref['uri'],expected_sha256=design_ref['sha256'])
    for ref in (model_ref,holdout_ref,policy_ref):read_artifact(ref)
    result=dict(schema='scar.stateful-entry-model-qualification',schema_version=1,model=model_ref,
        holdout=holdout_ref,policy=policy_ref,design=design_ref,capture_epoch=epoch,
        status='HOLDOUT_CONTEXT_QUALIFIED'if qualified else'HOLDOUT_PREDICTION_REJECTED',
        future_metadata_consistent=True,target_measurement_observed=real,predicted_native_ns=model['duration_ns'],
        original_median_ns=original,native_median_ns=native,method_original_over_native_ratio=original/native,
        relative_prediction_error=error,prediction_error_threshold=threshold,prediction_error_threshold_passed=error<=threshold,
        observed_gain=gain,observed_gain_threshold=minimum,observed_gain_threshold_passed=gain>=minimum,
        policy_precommitted_in_holdout=True,full_timer_scope_and_harness_bound=True,
        qualification_frontiers=[reason for passed,reason in ((real,'SYNTHETIC_METADATA_IS_NOT_TARGET_OBSERVATION'),
            (correct,'PAIRED_SOURCE_STATE_OR_NATIVE_PATH_CORRECTNESS_FAILED'),
            (error<=threshold,'FROZEN_PREDICTION_ERROR_THRESHOLD_EXCEEDED'),(gain>=minimum,'FROZEN_OBSERVED_GAIN_THRESHOLD_NOT_MET'))if not passed],
        measured_native_calls=native_calls,measured_fallback_calls=fallbacks,correctness_passed=bool(correct),
        qualified_profile=model['profile']if qualified else None,original_declared_profile_retained_for_analysis=model['profile'],
        target_samples_fitted=False,model_or_threshold_changed=False,
        cost_qualification_scope='ONLY_RETAINED_SAME_SOURCE_SHAPE_CALLER_AND_GUARDED_ENTRY_CONTEXT',
        physical_resource_qualification='SEPARATE_UNRESOLVED',whole_program_speedup_claimed=False,
        default_backend='NO_OP',deployment_authorized=False)
    if canonical_binding is not None:
        result.update(entry_protocol=deepcopy(model['entry_protocol']),
            entry_protocol_scope='ONLY_EXACT_COMPILER_ARTIFACT_ADAPTER_GUARDED_ENTRY_PROTOCOL',
            compact_or_other_trial_configuration_qualified=False)
    return result


def evaluate_stateful_entry_holdout(model,holdout,policy):
    """Dispatch by the actual protocol; future policies cannot relabel legacy clocks."""
    def reference(value):
        if type(value)is dict:return value
        path=Path(value).resolve();schema=json.loads(path.read_bytes())['schema']
        return artifact_reference(path,schema)
    model_ref,holdout_ref,policy_ref=map(reference,(model,holdout,policy))
    observed,rules=read_artifact(holdout_ref),read_artifact(policy_ref)
    if (observed.get('schema'),observed.get('schema_version'))==('scar.stateful-method-holdout',2):
        return _evaluate_future_stateful_entry_holdout(model_ref,holdout_ref,policy_ref)
    if (observed.get('schema'),observed.get('schema_version'))==('scar.real-caller-cost-holdout',1):
        if(rules.get('schema'),rules.get('schema_version'))!=('scar.conditional-entry-holdout-policy',1):
            raise ValueError('legacy capture requires its original legacy policy; future policy cannot authorize old samples')
        return _evaluate_legacy_stateful_entry_holdout(model_ref,holdout_ref,policy_ref)
    raise ValueError('unknown holdout capture schema/version')


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description='Evaluate a frozen stateful entry model against an independent holdout')
    for name in ('model','holdout','policy','out'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args(argv)
    report=evaluate_stateful_entry_holdout(args.model,args.holdout,args.policy)
    with Path(args.out).open('x') as stream:
        json.dump(report,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({key:report[key] for key in ('status','relative_prediction_error',
        'method_original_over_native_ratio','qualified_profile','deployment_authorized')},sort_keys=True))
    return 0


__all__ = ['SCHEMA', 'SCOPE', 'derive_stateful_entry_cost', 'verify_stateful_entry_cost',
           'evaluate_stateful_entry_holdout']


if __name__=='__main__':
    raise SystemExit(main())
