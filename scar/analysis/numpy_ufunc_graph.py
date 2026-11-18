"""Replay exact NumPy ufunc evidence onto a bounded F4 CALL graph."""
import argparse
from hashlib import sha256
from importlib.machinery import EXTENSION_SUFFIXES
import json
from pathlib import Path

from scar.analysis.build import BuildSession
from scar.analysis.program_dependencies import DependencyWalk
from scar.analysis.program_dependency_export import export_dependency_walk
from scar.compiler.call_binding import plan_digest
from scar.ir.execution_recipe import artifact_reference
from scar.ir.program.dependencies import (DependencyCondition as Condition, DependencyEdge as Edge,
    DependencyEvidence as Evidence, DependencyKind as K, DependencyMeaning as M,
    DependencyPoint as Point, DependencyStrength as Strength)
from scar.ir.program.ids import ProviderID
from scar.ir.program.providers import ProviderArtifact
from scar.ir.program.source_references import SourceArtifactLocator
from scar.ir.record_codec import encode
from scar.trace.numpy_ufunc_calls import CORE_MODULES, SCHEMA as OBSERVATION_SCHEMA


SCHEMA = 'scar.numpy-ufunc-call-graph'
_TYPE = {'module': 'numpy', 'qualname': 'ufunc'}


def _need(value, detail):
    if not value:
        raise ValueError(detail)


def _locator(artifact, pointer):
    return SourceArtifactLocator(artifact, artifact.sha256, artifact.size_bytes, 'canonical-json', pointer)


def _native_library(export):
    origin = Path(export['extension_origin'])
    _need(origin.is_absolute() and origin.name.endswith(tuple(EXTENSION_SUFFIXES)),
        'recorded NumPy core extension origin must identify a native module file')
    path = origin.resolve(strict=True)
    return artifact_reference(path, 'scar.native-library-file')


def _ufunc(row):
    value = row['ufunc']
    if value is None:
        return None
    _need(type(value) is dict and set(value) == {'name', 'input_count', 'output_count',
            'core_signature', 'numpy_version', 'public_core_export'} and row['target_type'] == _TYPE,
        'recorded ufunc metadata differs from its exact target type')
    _need(type(value['name']) is str and 0 < len(value['name']) <= 256
        and all(type(value[key]) is int and 0 <= value[key] <= 64 for key in ('input_count', 'output_count'))
        and (value['core_signature'] is None or type(value['core_signature']) is str and len(value['core_signature']) <= 512)
        and (value['numpy_version'] is None or type(value['numpy_version']) is str and len(value['numpy_version']) <= 128),
        'invalid bounded ufunc descriptor metadata')
    export = value['public_core_export']
    if export is not None:
        _need(type(export) is dict and set(export) == {'module', 'symbol', 'extension_origin'}
            and export['module'] in CORE_MODULES and export['symbol'] == value['name']
            and type(export['extension_origin']) is str,
            'ufunc library candidate requires its exact retained public/core export')
    return value


def derive_ufunc_call_graph(program, observation):
    path = Path(observation).resolve(strict=True)
    _need(path.stat().st_size <= 100_000, 'bounded ufunc observation required')
    raw = path.read_bytes()
    document = json.loads(raw)
    _need(type(document) is dict and (document.get('schema'), document.get('schema_version')) == (OBSERVATION_SCHEMA, 1)
        and document.get('errors') == [] and document.get('evidence_scope') == 'FIRST_LOCAL_CALL_PER_SELECTED_SITE'
        and all(document.get(key) is False for key in ('global_events_enabled', 'operand_values_collected', 'callee_completed')),
        'original successful sidecar observation required')
    pid, sites, rows = document['process_id'], document['sites'], document['rows']
    _need(type(pid) is int and pid > 0 and type(sites) is list and 1 <= len(sites) <= 32
        and all(type(site) is str for site in sites) and len(set(sites)) == len(sites) and type(rows) is list,
        'invalid bounded sidecar site set')
    sidecar = ProviderArtifact(str(path), sha256(raw).hexdigest(), len(raw), OBSERVATION_SCHEMA)
    roots, points, edges, bindings, candidates, libraries = {}, {}, [], [], {}, {}
    with BuildSession(program) as session:
        build = session._f4
        _need(build is not None, 'admitted F4 ProgramBuild required')
        providers = [provider for name, provider in build.providers.items() if name.startswith('bounded_call_bindings:')
            and document['plan_sha256'] == plan_digest(provider.plan) and provider.plan['max_targets_per_site'] == 1
            and any(row['process_id'] == pid for row in provider.summary['observations'])]
        _need(len(providers) == 1, 'unique original same-process one-attempt CALL provider required')
        owner = providers[0]; owner.check()
        _need(set(sites) <= owner.operations.keys(), 'sidecar selects a foreign F4 CALL site')
        attempts = {}
        for site in sites:
            operation = owner.operations[site]
            for point in owner.by_call[operation.namespace, operation.node]:
                attempt = owner.original(point)
                if attempt.process_id == pid:
                    _need(site not in attempts, 'more than one same-process first attempt')
                    attempts[site] = operation, point, attempt
        _need(all(type(row) is dict and set(row) == {'site_id', 'caller_code_key', 'offset', 'thread_id', 'target_type', 'ufunc'}
                and type(row['site_id']) is str for row in rows)
            and len(rows) == len(attempts) and {row['site_id'] for row in rows} == set(attempts),
            'sidecar rows differ from the original first attempts')
        for index, row in enumerate(rows):
            original, point, attempt = attempts[row['site_id']]
            _need(type(row['offset']) is int and type(row['thread_id']) is int
                and (row['caller_code_key'], row['offset'], row['thread_id'], row['target_type']) ==
                    (attempt.caller_site.code, attempt.caller_site.offset, attempt.thread_id,
                        json.loads(attempt.metadata_json)['type']), 'CALL sidecar and original attempt disagree')
            ufunc = _ufunc(row)
            observed, = (edge for edge in owner.outgoing(original).edges if edge.target == point and edge.kind is K.CALL_TARGET)
            roots[original] = original
            points.update({original:original, point:point})
            edges.append(observed)
            pointer = '#/rows/'+str(index)
            binding = dict(call=encode(original), original_call_port=encode(attempt.source_call),
                original_attempt=encode(attempt.evidence), observed_target=encode(point), sidecar_pointer=pointer,
                ufunc=ufunc, candidate=None)
            bindings.append(binding)
            export = None if ufunc is None else ufunc['public_core_export']
            if export is None:
                continue
            origin = export['extension_origin']
            if origin not in libraries:
                libraries[origin] = _native_library(export)
            library = libraries[origin]
            key = library['sha256'], library['uri'], export['module'], export['symbol']
            candidate_id = ProviderID(sha256(json.dumps(key, separators=(',', ':')).encode()).hexdigest())
            candidate = Point(original.namespace, candidate_id, None)
            points[candidate] = candidate
            library_point = Point(original.namespace, ProviderID(library['sha256']), None)
            points[library_point] = library_point
            proof = (_locator(sidecar, pointer), _locator(attempt.evidence.artifact, attempt.evidence.locator))
            edges.extend((Edge(point, candidate, K.IMPLEMENTATION, M.STRUCTURE,
                (Evidence(SCHEMA+'.public-core-export-identity', proof, (point,candidate)),),
                (Condition(SCHEMA, 'CANDIDATE_LIBRARY_OPERATION_INNER_DISPATCH_UNKNOWN', candidate),), Strength.MAY),
                Edge(candidate, library_point, K.RESOURCE, M.STRUCTURE,
                    (Evidence(SCHEMA+'.extension-file-origin', proof, (candidate,library_point)),),
                    (Condition(SCHEMA, 'REPORTED_EXTENSION_ORIGIN_FILE_BYTES', library_point),), Strength.MAY)))
            binding['candidate'] = encode(candidate)
            candidates[candidate] = dict(point=encode(candidate), library_point=encode(library_point),
                provider='numpy', module=export['module'], symbol=export['symbol'], library=library,
                selected_inner_loop=None, dispatch='UNKNOWN')
        owner.check()
        manifest = artifact_reference(build.path/'manifest.json', build.report['schema'])
    _need(path.read_bytes() == raw, 'ufunc observation changed during replay')
    traversal = DependencyWalk(tuple(roots), tuple(points), tuple(edges), (), 'outgoing', False,
        (K.CALL_TARGET, K.IMPLEMENTATION, K.RESOURCE))
    return dict(schema=SCHEMA, schema_version=1, program=manifest, observation=encode(sidecar),
        graph=export_dependency_walk(traversal), attempts=bindings, candidates=list(candidates.values()),
        qualification=dict(first_attempt_association='SAME_PROCESS_PLAN_SITE_CODE_OFFSET_THREAD_AND_TARGET_TYPE',
            same_caller_activation_proven=False, actual_implementation_selected=False,
            loaded_binary_memory_bytes_proven=False, operand_values_or_types_observed=False,
            dispatch_override_or_dtype_resolution_proven=False, native_effects_closed=False,
            target_executed_by_replay=False, first_stage_complete=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--program', required=True)
    parser.add_argument('--observation', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--out', type=Path)
    mode.add_argument('--check', type=Path)
    args = parser.parse_args()
    report = derive_ufunc_call_graph(args.program, args.observation)
    if args.check:
        _need(json.loads(args.check.read_bytes()) == report, 'stored ufunc graph differs from independent replay')
    else:
        with args.out.open('x') as stream:
            json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps(dict(attempts=len(report['attempts']), candidates=len(report['candidates']),
        edges=report['graph']['summary']['edge_kinds'], replay_verified=args.check is not None)))


if __name__ == '__main__':
    main()
