"""Source-bound GCC vector cost diagnostics, without runtime cost conversion.

Only the observed GCC 13.3.0 format-1 text block is recognized.  All versions
remain separate.  These target-relative compiler scores are inputs to future
models, never nanoseconds, observed execution counts or qualified profiles.
"""
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import io
import json
from pathlib import Path
import re

from scar.analysis.gcc_optimization import _raw
from scar.compiler.collect_native_optimization import (
    SCHEMA as COLLECTION_SCHEMA, check_gcc_optimization, read_gcc_feedback,
)
from scar.ir.execution_recipe import _json, artifact_reference
from scar.optimization.deployment_cost import _pointer


SCHEMA = 'scar.gcc-vector-cost-features'
_MARKER = 'Cost model analysis:'
_HEADER = re.compile(r'(?m)^[ \t]*Cost model analysis:[ \t]*\r?$')
_FIELDS = (
    ('Vector inside of loop cost', 'vector_inside_cost', 'GCC_TARGET_RELATIVE_COST'),
    ('Vector prologue cost', 'vector_prologue_cost', 'GCC_TARGET_RELATIVE_COST'),
    ('Vector epilogue cost', 'vector_epilogue_cost', 'GCC_TARGET_RELATIVE_COST'),
    ('Scalar iteration cost', 'scalar_iteration_cost', 'GCC_TARGET_RELATIVE_COST'),
    ('Scalar outside cost', 'scalar_outside_cost', 'GCC_TARGET_RELATIVE_COST'),
    ('Vector outside cost', 'vector_outside_cost', 'GCC_TARGET_RELATIVE_COST'),
    ('prologue iterations', 'prologue_iterations', 'GCC_MODEL_ITERATIONS'),
    ('epilogue iterations', 'epilogue_iterations', 'GCC_MODEL_ITERATIONS'),
    ('Calculated minimum iters for profitability', 'minimum_profitable_iterations', 'GCC_MODEL_ITERATIONS'),
)
_LABELS = {label:(name, unit) for label, name, unit in _FIELDS}


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _record(reference):
    _, raw = _raw(reference, 16*2**20)
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
        decoded = stream.read(64*2**20+1)
    _need(len(decoded) <= 64*2**20, 'GCC vector feature decoded record budget exceeded')
    return _json(decoded)


def _evidence(collection_ref, collection, diagnostic, index):
    return dict(collection_ref=deepcopy(collection_ref), record_ref=deepcopy(collection['files']['record']),
        diagnostic_pointer=diagnostic['id'], message_pointer=diagnostic['raw_message_pointer'],
        feedback_ref=deepcopy(collection['files']['feedback']), feedback_pointer='#/diagnostics/'+str(index),
        pointer_encoding='RFC6901_ON_DECOMPRESSED_JSON')


def _primary_stages(diagnostic, projection):
    locations = {row['id']:row for row in projection['locations']}
    files = {row['id']:row for row in projection['files']}
    spans = {row['index']:row for row in projection['stage_spans']}
    source = Path(projection['source_ref']['uri']).resolve()
    selected = []
    for pointer in diagnostic['primary_location_ids']:
        row = locations[pointer]
        file = files.get(row['file_id'])
        if (row['role'] != 'PRIMARY' or row['diagnostic'] != diagnostic['id']
                or row['mapping_status'] != 'MAPPED_TO_STAGE_SPANS' or file is None
                or file['source_identity_status'] != 'SOURCE_REF_PATH_MATCH'
                or file['resolved_path'] != str(source)):
            continue
        for index in row['stage_span_indices']:
            span = spans[index]
            if span['start_line'] <= row['line'] <= span['end_line']:
                selected.append(dict(stage_id=span['id'], span_index=index,
                    primary_location_pointer=pointer, line=row['line'], column=row['column']))
    return selected


def _stage_associations(selected, collection, collection_ref):
    subject = collection.get('subject')
    view = subject.get('view') if type(subject) is dict else None
    stages = view.get('stages', []) if type(view) is dict else []
    result = []
    for stage_id in dict.fromkeys(row['stage_id'] for row in selected):
        matches = [(index, row) for index, row in enumerate(stages) if row.get('id') == stage_id]
        actions, refs = [], []
        known = bool(matches)
        for index, stage in matches:
            identifiers = stage.get('source_action_ids')
            if type(identifiers) is not list or any(type(name) is not str for name in identifiers):
                known = False
            else:
                actions.extend(identifiers)
                refs.append(dict(artifact=deepcopy(collection_ref),
                    locator='#/subject/view/stages/'+str(index)+'/source_action_ids'))
        result.append(dict(stage_id=stage_id,
            span_indices=sorted({row['span_index'] for row in selected if row['stage_id'] == stage_id}),
            primary_locations=[deepcopy(row) for row in selected if row['stage_id'] == stage_id],
            source_ref=deepcopy(collection['source']),
            source_action_ids=list(dict.fromkeys(actions)) if known else None,
            source_action_mapping='COLLECTION_VIEW_REPLAYED' if known else 'UNKNOWN_NO_COMPLETE_VIEW_MAPPING',
            source_action_refs=refs if known else []))
    return result


def _text_spans(chunks, pointer, start, end):
    """Offsets count Unicode codepoints within original message strings."""
    result, position = [], 0
    for index, chunk in enumerate(chunks):
        following = position + len(chunk)
        if max(start, position) < min(end, following):
            result.append(dict(pointer=pointer+'/'+str(index),
                start=max(start, position)-position, end_exclusive=min(end, following)-position))
        position = following
        if position >= end:
            break
    return result


def _block_features(text, chunks, pointer, start, end):
    occurrences = {name:[] for _, name, _ in _FIELDS}
    frontier, position = [], start
    for line in text[start:end].splitlines(keepends=True):
        following = position + len(line)
        stripped = line.strip()
        if not stripped or stripped == _MARKER:
            position = following
            continue
        key, separator, value = stripped.partition(':')
        location = _text_spans(chunks, pointer, position, following)
        if separator and key in _LABELS:
            name, _ = _LABELS[key]
            value = value.strip()
            try:
                parsed = int(value) if re.fullmatch(r'[+-]?[0-9]+', value) else None
            except ValueError:
                parsed = None
            occurrences[name].append(dict(value=parsed,
                status='REPORTED_INTEGER' if parsed is not None else 'MALFORMED_INTEGER',
                message_text_spans=location))
        else:
            frontier.append(dict(reason='UNRECOGNIZED_COST_BLOCK_LINE', message_text_spans=location))
        position = following
    features = {}
    for _, name, unit in _FIELDS:
        rows = occurrences[name]
        if not rows:
            reason = 'MISSING_FIELD'
        elif len(rows) != 1:
            reason = 'DUPLICATE_FIELD_NOT_DISAMBIGUATED'
        elif rows[0]['value'] is None:
            reason = 'MALFORMED_INTEGER'
        else:
            reason = None
        features[name] = dict(value=rows[0]['value'] if reason is None else None,
            status='REPORTED_COMPILER_MODEL_INTEGER' if reason is None else 'UNKNOWN',
            unit=unit, reported_occurrences=rows)
        if reason is not None:
            frontier.append(dict(field=name, reason=reason))
    return features, frontier


def project_gcc_vector_costs(collection_ref):
    """Cold-check a collection and preserve every supported mapped cost block."""
    if isinstance(collection_ref, (str, Path)):
        collection_ref = artifact_reference(Path(collection_ref).resolve(), COLLECTION_SCHEMA)
    else:
        collection_ref = deepcopy(collection_ref)
    collection = check_gcc_optimization(collection_ref)
    projection = read_gcc_feedback(collection['files']['feedback'])
    _need(projection['source_ref'] == collection['source']
        and projection['record_ref'] == collection['files']['record'], 'checked feedback/source/record binding differs')
    header = projection['observed_wire_header']
    generator = header.get('generator')
    supported = (header.get('format') == '1' and type(generator) is dict
        and generator.get('version') == '13.3.0' and generator.get('name') in ('GNU C17', 'GNU C++20'))
    blocks, frontier = [], []
    if not supported:
        frontier.append(dict(reason='UNSUPPORTED_GCC_PRODUCER_OR_WIRE_FORMAT',
            record_ref=deepcopy(collection['files']['record']), raw_pointer='#/0',
            observed_wire_header=deepcopy(header)))
    else:
        record = _record(collection['files']['record'])
        passes = {row['id']:row for row in projection['passes']}
        for index, diagnostic in enumerate(projection['diagnostics']):
            pointer = diagnostic.get('raw_message_pointer')
            if pointer is None:
                continue
            try:
                message = _pointer(record, pointer)
            except (KeyError, IndexError, TypeError, ValueError) as error:
                raise ValueError('checked GCC diagnostic message pointer differs') from error
            text_message = type(message) is list and all(type(chunk) is str for chunk in message)
            candidate = (_MARKER in ''.join(message) if text_message else
                type(message) is str and _MARKER in message or type(message) is list
                and any(type(chunk) is str and _MARKER in chunk for chunk in message))
            if not candidate:
                continue
            evidence = _evidence(collection_ref, collection, diagnostic, index)
            names = [passes[pass_id].get('name') for pass_id in diagnostic['pass_ids']]
            selected = _primary_stages(diagnostic, projection)
            if diagnostic.get('kind') != 'note':
                frontier.append(dict(reason='COST_BLOCK_REQUIRES_NOTE_DIAGNOSTIC', evidence=evidence))
                continue
            if not names or any(name != 'vect' for name in names):
                frontier.append(dict(reason='COST_BLOCK_REQUIRES_UNAMBIGUOUS_VECT_PASS', evidence=evidence))
                continue
            if not selected:
                frontier.append(dict(reason='PRIMARY_LOCATION_NOT_MAPPED_TO_VERIFIED_SOURCE_STAGE', evidence=evidence))
                continue
            if type(message) is not list or any(type(chunk) is not str for chunk in message):
                frontier.append(dict(reason='OPAQUE_NON_TEXT_COST_BLOCK_MESSAGE', evidence=evidence))
                continue
            text = ''.join(message)
            markers = list(_HEADER.finditer(text))
            if not markers:
                frontier.append(dict(reason='MALFORMED_COST_BLOCK_HEADER', evidence=evidence))
                continue
            associations = _stage_associations(selected, collection, collection_ref)
            for ordinal, marker in enumerate(markers):
                end = markers[ordinal+1].start() if ordinal+1 < len(markers) else len(text)
                features, unresolved = _block_features(text, message, pointer, marker.start(), end)
                identity = diagnostic['id']+'/cost-block/'+str(ordinal)
                for issue in unresolved:
                    frontier.append(dict(block=identity, evidence=deepcopy(evidence), **issue))
                blocks.append(dict(id=identity, block_ordinal=ordinal, evidence=deepcopy(evidence),
                    diagnostic_kind=diagnostic['kind'], function=diagnostic.get('function'),
                    pass_ids=deepcopy(diagnostic['pass_ids']),
                    pass_record_pointers=deepcopy(diagnostic['pass_ids']),
                    message_text_spans=_text_spans(message, pointer, marker.start(), end),
                    text_offset_unit='UNICODE_CODEPOINTS_IN_ORIGINAL_MESSAGE_STRINGS',
                    stage_ids=[row['stage_id'] for row in associations], stage_associations=deepcopy(associations),
                    features=features, frontier=unresolved,
                    status='ALL_REQUESTED_FIELDS_REPORTED' if not unresolved else 'PARTIAL_OR_MALFORMED',
                    basis='OBSERVED_COMPILER_DIAGNOSTIC_NOT_OBSERVED_RUNTIME_COST',
                    scope='ONE_REPORTED_VECTOR_COST_MODEL_ANALYSIS_BLOCK', qualified_profile=None))
    stage_blocks = {row['id']:[] for row in projection['stage_spans']}
    for block in blocks:
        for stage_id in block['stage_ids']:
            stage_blocks.setdefault(stage_id, []).append(block['id'])
    # Recheck the original collection bytes without rerunning a producer.
    _, raw_collection = _raw(collection_ref, 64*2**20)
    _need(_json(raw_collection) == collection, 'GCC collection changed during feature projection')
    return dict(schema=SCHEMA, schema_version=1, collection_ref=collection_ref,
        feedback_ref=deepcopy(collection['files']['feedback']), record_ref=deepcopy(collection['files']['record']),
        source_ref=deepcopy(collection['source']), observed_wire_header=deepcopy(header),
        implementation_source=artifact_reference(__file__, 'scar.python-source'),
        units=dict(target_cost='GCC_TARGET_RELATIVE_COST', model_iteration='GCC_MODEL_ITERATIONS'),
        blocks=blocks, stage_blocks=stage_blocks,
        stage_coverage={stage_id:('REPORTED_VECTOR_MODEL_FEATURES' if ids else 'UNKNOWN_NO_SUPPORTED_VECTOR_COST_BLOCK')
            for stage_id, ids in stage_blocks.items()}, frontier=frontier,
        summary=dict(blocks=len(blocks), complete_blocks=sum(not row['frontier'] for row in blocks),
            partial_blocks=sum(bool(row['frontier']) for row in blocks), frontier_entries=len(frontier),
            frontier_reasons=dict(sorted(Counter(row['reason'] for row in frontier).items()))),
        qualifications=dict(collection_cold_replayed=True, primary_source_stage_mapping_required=True,
            all_versions_retained=True, best_version_selected=False, stage_costs_summed=False,
            target_relative_scores_are_nanoseconds=False, model_iterations_are_observed_execution_counts=False,
            latency_measured=False, elapsed_time_estimated=False, timing_qualified=False,
            semantic_source_admission_replayed=False, deployment_authorized=False,
            compiler_invoked_now=False, target_program_executed=False, native_loaded=False),
        qualified_profile=None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    document = project_gcc_vector_costs(args.collection)
    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(document, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(result=artifact_reference(output, SCHEMA), blocks=len(document['blocks']),
        frontier=len(document['frontier'])), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


__all__ = ['project_gcc_vector_costs']
