"""Select a bounded next capture from entry producer CodeIDs and original sites."""

import argparse
from collections import defaultdict
from dataclasses import replace
import json
import os
from pathlib import Path

from scar.analysis.entry_call_dependence import SCHEMA as DEPENDENCE
from scar.analysis.program_view import ProgramView
from scar.compiler.call_binding import (
    plan_call_bindings, validate_call_binding_plan, write_call_binding_plan,
)
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.record_codec import decode
from scar.optimization.resource_solver import semantic_sha256
from scar.trace.lean_capture import LeanBodyPlan, LeanCapturePlan


RETURN_DEMANDS = 'scar.stage1-producer-return-demands-observation'


def _observation(report, pointer):
    prefix = '#/observed_targets/'
    if type(pointer) is not str or not pointer.startswith(prefix) or not pointer[len(prefix):].isdecimal():
        raise ValueError('entry producer lacks its original observation reference')
    return report['observed_targets'][int(pointer[len(prefix):])]


def _producer_targets(dependence, report):
    pointers = {pointer for item in dependence['live_ins']
        for call in item['observed_calls'] for pointer in call['observations']}
    targets = {}
    for pointer in sorted(pointers):
        row = _observation(report, pointer)
        code = row['observation']['target_code']
        if row['entry_scope_reachable'] and code is not None:
            targets[code['key']] = code
    return targets


def plan_entry_producer_calls(dependence_path, catalog):
    """Reconstruct selectors in the expanded catalog and retain existing caps."""
    reference = artifact_reference(dependence_path, DEPENDENCE)
    dependence = read_artifact(reference)
    report = read_artifact(dependence['entry_calls'])
    return _plan_calls(catalog, report, _producer_targets(dependence, report),
        'PRIOR_OBSERVED_SITES_AND_IMMEDIATE_PRODUCER_BODIES', {'source_dependence': reference})


def plan_return_producer_calls(demand_path, catalog):
    """Follow selected return dependencies without widening the original sample."""
    reference = artifact_reference(demand_path, RETURN_DEMANDS)
    demand = read_artifact(reference)
    if demand['schema_version'] != 1:
        raise ValueError('unsupported return producer demand version')
    entry = demand['entry_projection']
    if entry['schema'] != 'scar.bounded-entry-calls':
        raise ValueError('return producer demands require the original entry projection')
    report = read_artifact(entry)
    summary = read_artifact(demand['source_summary'])
    header = summary['header'] if summary['schema_version'] == 2 else summary
    if (header['schema'] != 'scar.interprocedural-summary'
            or header['entry_projection'] != dict(sha256=semantic_sha256(report),
                program=report['program'], producer=report['producer'])):
        raise ValueError('return demands and source summary name different entry projections')
    targets = {}
    for path, rows in demand['observed_source_targets'].items():
        for selected in rows:
            row = _observation(report, selected['observation'])
            code = row['observation']['target_code']
            if (not row['entry_scope_reachable'] or code is None
                    or (code['filename'], code['key'], code['qualname'])
                    != (path, selected['code_key'], selected['qualname'])):
                raise ValueError('return producer differs from its observed code identity')
            targets[code['key']] = code
    return _plan_calls(catalog, report, targets, 'PRIOR_OBSERVED_SITES_AND_RETURN_PRODUCER_BODIES',
        {'source_return_demands': reference, 'source_entry_projection': entry})


def _plan_calls(catalog, report, targets, selection, provenance):
    view = ProgramView(catalog)
    store = view.catalog.fact_store
    previous = defaultdict(set)
    prior_keys = set()
    for row in report['observed_targets']:
        unit = row['unit']
        path = store.model_units.get(unit, {}).get('path') or store.rows[unit].get('path')
        if type(path) is not str or not Path(path).is_absolute():
            raise ValueError('prior observed CALL has no original source in the expanded catalog')
        previous[path].add(row['query'])
        site = row['observation']['caller_site']
        prior_keys.add((site['code'], site['offset']))

    sites, bodies, frontiers = {}, {}, []
    details = None

    def admit(plan, wanted):
        if plan['catalog'] != details['catalog']:
            raise ValueError('expanded source catalog changed while selecting calls')
        for row in plan['sites']:
            if not wanted(row):
                continue
            previous = sites.setdefault(row['id'], row)
            if previous != row:
                raise ValueError('selected original CALL has conflicting source identities')
        for record in (plan['selector'] or {}).get('bodies', ()):
            body = decode(LeanBodyPlan, record)
            existing = bodies.setdefault(body.code.key, body)
            if (existing.source, existing.code) != (body.source, body.code):
                raise ValueError('selected call body changed its source or CodeID')

    for path, indices in sorted(previous.items()):
        for entry in (None, '<module>'):
            plan = plan_call_bindings(view, path, entry=entry, call_indices=indices, max_targets_per_site=1)
            if details is None:
                details = plan
            admit(plan, lambda row: (row['code_key'], row['offset']) in prior_keys)
    if details is None or {(r['code_key'], r['offset']) for r in sites.values()} != prior_keys:
        raise ValueError('expanded catalog lost a previously observed exact CALL site')

    for key, code in targets.items():
        path = code['filename']
        if not view.catalog.units_for_source(path):
            frontiers.append(dict(field='producer.body', code_key=key, path=path,
                reason='PRODUCER_SOURCE_OUTSIDE_EXPANDED_CATALOG'))
            continue
        plan = plan_call_bindings(view, path, entry=code['qualname'], max_targets_per_site=1)
        if not any(row['code_key'] == key for row in plan['sites']):
            frontiers.append(dict(field='producer.call_sites', code_key=key, path=path,
                reason='NO_EXACT_ORIGINAL_CALL_SITE_IN_PRODUCER_BODY'))
            continue
        admit(plan, lambda row: row['code_key'] == key)

    offsets = defaultdict(set)
    for row in sites.values():
        offsets[row['code_key']].add(row['offset'])
    selected = tuple(replace(bodies[key], call_offsets=tuple(sorted(values)),
        unsupported_offsets=tuple(offset for offset in bodies[key].unsupported_offsets if offset in values))
        for key, values in offsets.items())
    sources = sorted({body.source.path for body in selected})
    namespace = os.path.commonpath([str(Path(path).parent) for path in sources])
    for reference in provenance.values():
        read_artifact(reference, json_document=False)
    return validate_call_binding_plan({**details,
        'source_selection': selection, **provenance, 'sites': list(sites.values()), 'frontiers': frontiers,
        'sources': sources, 'module_bodies_included': True,
        'selector': LeanCapturePlan(namespace, selected, max_codes=64).to_dict()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--entry-dependence', type=Path)
    source.add_argument('--return-demands', type=Path)
    parser.add_argument('--catalog', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    plan = (plan_entry_producer_calls(args.entry_dependence, args.catalog) if args.entry_dependence
            else plan_return_producer_calls(args.return_demands, args.catalog))
    if args.out.exists():
        raise FileExistsError(args.out)
    write_call_binding_plan(plan, args.out)
    print(json.dumps(dict(selected_codes=len(plan['selector']['bodies']),
        selected_sites=len(plan['sites']), producer_frontiers=len(plan['frontiers']))))


if __name__ == '__main__':
    main()
