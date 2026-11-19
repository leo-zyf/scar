"""Select F4 formal binding instances from already retained frame entries."""

import argparse
from collections import Counter, defaultdict
import json

from scar.analysis.source_reference_slice import source_port
from scar.ir.program.normal_targets import Body
from scar.ir.program.source_invocations import SourceCallScope
from scar.ir.program.source_name_definitions import NameDefinitionKind
from scar.ir.program.source_observed_formals import ObservedFormalFrontier, ObservedFormalInstance
from scar.ir.program.source_productions import DeferredBodyKind
from scar.ir.program.source_state import BindingBoundness
from scar.ir.record_codec import encode


SCHEMA = 'scar.source-observed-formal-instances'


def _select(session, unit):
    f4 = session._f4
    if f4 is None or 'binding_states' not in f4.providers:
        raise ValueError('formal occurrence replay requires an observed F4 binding-state provider')
    unit = f4.report['f4']['entry_unit'] if unit is None else unit
    provider = f4.providers['binding_states']
    if unit not in provider.headers:
        raise ValueError('requested unit has no observed binding-state publication')
    state = provider.ensure(unit)
    codes = provider.index.codes
    codes.check()
    run = codes.targets.source_sha256
    facts = state.source.facts
    functions = {source_port(facts, row.value): row for row in facts.functions}
    versions, origins = defaultdict(list), defaultdict(list)
    for version in state.versions.values():
        location = state.locations[version.location]
        if (version.kind is NameDefinitionKind.FORMAL and version.boundness is BindingBoundness.BOUND
                and location.interface.kind == 'LEXICAL_BINDING' and location.key == version.namespace.name):
            versions[version.source_function, version.payload, version.namespace.name].append(version)
    for link in codes.records.values():
        for capture in link.captures:
            origins[capture.process_id, capture.code.key].append((link, capture.code))

    instances, frontiers = [], []
    entries = 0
    for frame in codes.trace.entries:
        choices = origins.get((frame.invocation.process_id, frame.code), ())
        if not any(link.body.unit == unit for link, _ in choices):
            continue
        entries += 1
        scope = SourceCallScope(run, frame.invocation)

        def frontier(reason, formal=None):
            frontiers.append(ObservedFormalFrontier(scope, frame.anchor, reason, formal))

        if len(choices) != 1:
            frontier('FORMAL_ENTRY_SOURCE_BODY_NOT_UNIQUE')
            continue
        link, code = choices[0]
        function = functions.get(link.function)
        if (function is None or Body(unit, function.template, function.body) != link.body
                or function.body_scope != link.scope or function.environment.kind is not DeferredBodyKind.FUNCTION):
            frontier('FORMAL_ENTRY_FUNCTION_BODY_MISMATCH')
            continue
        if frame.suspended or not frame.parameters_complete:
            frontier('FORMAL_ENTRY_FRAME_PARAMETERS_UNAVAILABLE')
            continue
        names = function.parameter_names
        formals = function.formals[1:-1]
        refs = frame.parameters
        if (len(names) != len(formals) or len(names) != len(set(names)) or set(code.parameters) != set(names)
                or tuple(ref.name for ref in refs) != code.parameters):
            frontier('FORMAL_ENTRY_PARAMETER_LAYOUT_MISMATCH')
            continue
        if names:
            packed = frame.caller is not None and frame.caller.opname == 'CALL_FUNCTION_EX'
            frontier('CALLER_PACK_BINDING_UNRESOLVED' if packed else 'CALLER_ACTUAL_ROUTE_NOT_ESTABLISHED')
        refs_by_name = {ref.name: ref for ref in refs}
        for ordinal, (index, name) in enumerate(zip(formals, names, strict=True)):
            port = source_port(facts, index)
            candidates = versions.get((link.function, port, name), ())
            if not candidates:
                frontier('FORMAL_BINDING_VERSION_TEMPLATE_NOT_TRACKED', port)
                continue
            if len(candidates) != 1:
                frontier('FORMAL_BINDING_VERSION_TEMPLATE_NOT_UNIQUE', port)
                continue
            version, = candidates
            instances.append(ObservedFormalInstance(scope, frame.anchor, link.id, link.function, port, ordinal,
                                                     version.location, version.id, refs_by_name[name]))
    state.check()
    provider.check()
    codes.check()
    return unit, entries, tuple(instances), tuple(frontiers)


def derive_observed_formal_instances(session, *, unit=None) -> tuple[ObservedFormalInstance, ...]:
    """Return typed reference observations for one admitted F4 source unit."""
    return _select(session, unit)[2]


def export_observed_formal_instances(session, *, unit=None) -> dict:
    """Produce JSON-ready occurrences and independent open-scope diagnostics."""
    unit, entries, instances, frontiers = _select(session, unit)
    return dict(schema=SCHEMA, schema_version=1, unit=unit, candidate_entries=entries,
                instances=encode(instances), frontiers=encode(frontiers),
                summary=dict(formal_observations=len(instances),
                             observed_frames=len({row.call_scope for row in instances}),
                             frontier_reasons=dict(Counter(row.reason for row in frontiers))),
                reference_authority='ORIGINAL_FRAME_ENTRY_POINT_ONLY', caller_actual_routes_established=False,
                object_content_versions_established=False, effects_closed=False, transformation_authorized=False)


def main(argv=None) -> int:
    from scar.analysis.build import BuildSession

    parser = argparse.ArgumentParser(prog='scar observed-formals', description=__doc__)
    parser.add_argument('build', help='F4 build with observed binding states')
    parser.add_argument('--unit', help='source unit identity, defaults to the entry unit')
    args = parser.parse_args(argv)
    with BuildSession(args.build) as session:
        result = export_observed_formal_instances(session, unit=args.unit)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
