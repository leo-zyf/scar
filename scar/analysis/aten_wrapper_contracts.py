"""Keep the observed entry, declared argument routing and effects separate."""

from collections import defaultdict


SCHEMA = 'scar.aten-wrapper-contracts'


def _declared_effects(operators):
    effects = []
    for operator in operators:
        declared = operator['declared_schema_effects']
        if declared['may_mutate']:
            effects.append(dict(kind='NATIVE_SCHEMA_MAY_MUTATE', operator=operator['eto'],
                                ports=[p for p in declared['alias_ports']
                                       if p['direction'] == 'input' and p['alias']['is_write']]))
        outputs = [p for p in declared['alias_ports'] if p['direction'] == 'output']
        if outputs:
            inputs = [p for p in declared['alias_ports'] if p['direction'] == 'input']
            observed = []
            for inp in inputs:
                for out in outputs:
                    common = (set(inp['alias']['after']) & set(out['alias']['after'])) - {'*'}
                    left = operator['inputs'][inp['ordinal']].get('descriptor_id')
                    right = operator['outputs'][out['ordinal']].get('descriptor_id')
                    if common and left is not None and left == right:
                        observed.append(dict(input=inp['port'], output=out['port'],
                                             input_ordinal=inp['ordinal'], output_ordinal=out['ordinal']))
            effects.append(dict(kind='NATIVE_SCHEMA_RETURN_ALIAS', operator=operator['eto'], ports=outputs,
                                observed_same_descriptor_aliases=observed))
    return effects


def _by_attempt(rows, attempts, layer):
    indexed = {row['attempt']: row for row in rows}
    if len(indexed) != len(rows) or set(indexed) - attempts:
        raise ValueError(layer + ' refers to different original attempts')
    return indexed


def project_aten_wrapper_contracts(source, native, images, declared):
    """Retain exact observations without implying source-build equivalence."""
    attempts = {row['attempt'] for row in source['calls']}
    operators = defaultdict(list)
    for row in native['operators']:
        operators[row['call_attempt']].append(row)
    if (set(operators) != attempts or native['run'] != images['run']
            or native['process'] != images['process']
            or native['coordinate_binding'] != images['coordinate_binding']
            or declared is not None and declared['coordinate_binding'] != native['coordinate_binding']):
        raise ValueError('wrapper evidence belongs to a different original capture')
    entries = _by_attempt(images['calls'], attempts, 'executable image')
    routes = _by_attempt(declared['site_rows'], attempts, 'generated declaration') if declared else {}

    records = []
    for row in source['calls']:
        attempt, site = row['attempt'], row['site']
        entry = entries.get(attempt)
        route = routes.get(attempt)
        selected = route.get('selection') if route else None
        if entry is not None and (entry['site'] != site or entry['invocation'] != source['invocation']):
            raise ValueError('wrapper entry changed its original CALL site or invocation')
        native_refs = [dict(schema=op['schema'], eto=op['eto'], kineto=op['kineto']) for op in operators[attempt]]
        operand, scope = (operators[attempt][0][key] for key in ('operand', 'scope'))
        if entry and (entry['operand'], entry['scope']) != (operand, scope):
            raise ValueError('wrapper entry changed its original CALL or scope bytes')
        if route and (route['source_call'], route['operators'], route['operand'], route['scope']) != (
                row['source_call'], native_refs, operand, scope):
            raise ValueError('declared wrapper route changed its original CALL or native event')
        if selected and (len(operators[attempt]) != 1 or
                [r['observed_port'] for r in selected['ordered_routes']]
                != [port['source'] for port in operators[attempt][0]['inputs']]):
            raise ValueError('declared input routes changed the original observed port order')

        agreement = None
        if (entry and entry['status'] == 'QUALIFIED_IMAGE_ENTRY'
                and selected and selected.get('cpp_symbol')):
            stem = entry['entry_symbol'].split('(', 1)[0].rsplit('::', 1)[-1]
            agreement = stem == selected['cpp_symbol']

        effects = _declared_effects(operators[attempt])
        if selected and selected['deprecated']:
            effects.append(dict(kind='PYTHON_DEPRECATED_SIGNATURE_MAY_WARN', declaration=selected['declaration']))
        if selected and selected.get('supplied_unmapped_actuals'):
            effects.append(dict(kind='PYTHON_ACTUALS_REQUIRE_WRAPPER_HANDLING',
                                declaration=selected['declaration'], actuals=selected['supplied_unmapped_actuals']))

        records.append(dict(attempt=attempt, site=site,
            image_status=entry['status'] if entry else 'UNRESOLVED',
            argument_status=route['status'] if route else 'NOT_REQUESTED',
            entry_name_agreement=agreement, parser_variant=None if not selected else selected['parser_variant'],
            effects=effects, source_call=row['source_call'],
            wrapper_build_correspondence_proven=False, effects_complete=False, replacement_authorized=False))

    return dict(schema=SCHEMA, schema_version=1, run=native['run'], process=native['process'],
                invocation=source['invocation'], calls=records, deployment_authorized=False)
