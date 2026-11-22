"""Project original F4 source initializer candidates and their local effects."""

from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

from scar.analysis.build import BuildSession
from scar.analysis.entry_obligations import SCHEMA as OBLIGATIONS
from scar.analysis.source_decoded_bodies import reuse_decoded_source_bodies, source_body_access
from scar.analysis.source_effects import classify_instruction
from scar.analysis.source_reference_slice import ReferenceSlicer, source_port
from scar.ir.execution_recipe import read_artifact
from scar.ir.program import normal_graph as G, source_productions as P
from scar.ir.program.source_declarations import SourceTemplateKey
from scar.ir.record_codec import decode, encode

SCHEMA = 'scar.f4-cpu-entry-import-initializers'
_DETAILS = {'DefineLiteral', 'ConstructValue', 'DeferRegion', 'BuildFunction', 'ImportRequest'}
_OPEN = ('RUNTIME_MODULE_SELECTION_AND_IMPORTER_EFFECTS_OPEN',
         'IMPORT_CACHE_OR_RELOAD_AND_INITIALIZER_ACTIVATION_OPEN',
         'DYNAMIC_PROVIDER_EFFECTS_AND_EXCEPTION_ROUTES_OPEN')


def _need(ok, detail):
    if not ok:
        raise ValueError('CPU entry import initializers: ' + detail)


def _row(rows, index):
    _need(type(index) is int and 0 <= index < len(rows), 'invalid original table ordinal')
    return rows[index]


def _edge(kind, source, target, evidence):
    return dict(kind=kind, source=source, target=target, original=evidence)


def _requests(original):
    requests, edges, candidates = [], [], set()
    for index, row in enumerate(original['imports']):
        pointer = f'#/imports/{index}'
        operation = _row(original['operations'], row['operation'])
        _need(operation['effect_rule']['instruction'] == 'ImportRequest',
              'original request is outside its typed import operation')
        alternatives = []
        for ordinal, option in enumerate(row['alternatives']):
            body_index = option['body']
            evidence = f'{pointer}/alternatives/{ordinal}'
            status = option['status']
            if body_index is not None:
                body = _row(original['bodies'], body_index)
                _need(status == 'POSSIBLE_MODULE_BODY' and option['point'] == body['point'],
                      'module candidate changed its original source body')
                candidates.add(body_index)
                edges.append(_edge('MAY_SELECT_SOURCE_INITIALIZER', pointer, f'#/bodies/{body_index}', evidence))
            alternatives.append(dict(original=evidence, body=body_index, status=status))
        requests.append(dict(original=pointer, operation=row['operation'], alternatives=alternatives,
            status='SOURCE_INITIALIZER_CANDIDATES' if any(row['body'] is not None for row in alternatives)
            else 'SOURCE_INITIALIZER_OPEN', loader_selection_proven=False))
    return requests, edges, candidates


def _coordinates(original, body):
    coordinates, inventory = {}, defaultdict(list)
    for index in body['operations']:
        row = _row(original['operations'], index)
        port = _row(original['ports'], row['reach_port'])
        owner = _row(original['owners'], port['owner'])
        template = _row(original['templates'], owner['template'])
        _need(row['unit'] == template['unit'] == body['unit']
              and port['kind'] == G.PortKind.ENTER.value and row['template'] == template['template']
              and owner['original'].partition(':')[0] == P.ProductionStepID.prefix,
              'initializer operation differs from its original typed entry port')
        key = decode(SourceTemplateKey, template['template'])
        coordinate = key, owner['original']
        _need(coordinate not in coordinates, 'duplicate operation in original initializer body')
        coordinates[coordinate] = index
        inventory[row['effect_rule']['instruction']].append(index)
    return coordinates, dict(sorted(inventory.items()))


def _detail(original, source, coordinates, inventory, slicer):
    wanted = {index for kind in _DETAILS for index in inventory.get(kind, ())}
    selected = {key: index for key, index in coordinates.items() if index in wanted}
    plans = source_body_access(source.material).plans(key for key, _ in selected)
    literals, declarations, constructions, imports = [], [], [], []
    selected_literals = set()
    for key, plan in plans.items():
        for step in plan.steps:
            index = selected.get((key, step.id.wire))
            if index is None:
                continue
            wanted.remove(index)
            instruction = step.instruction
            original_rule = original['operations'][index]['effect_rule']
            _need(encode(classify_instruction(instruction)) == original_rule,
                  'initializer instruction differs from its retained typed rule')
            if type(instruction) is P.DefineLiteral:
                literals.append(dict(operation=index, literal=encode(instruction.value), origin='ORIGINAL_IMPLICIT_LITERAL'))
            elif type(instruction) is P.ConstructValue and instruction.construction is P.ConstructionKind.LITERAL:
                selected_literals.add(index)
            elif type(instruction) is P.DeferRegion:
                _need(any(region.id == instruction.body for region in plan.regions),
                      'deferred declaration has no original sealed region')
                declarations.append(dict(operation=index, body=dict(unit=source.unit, template=encode(key),
                    region=instruction.body.wire), kind=instruction.environment.kind.value,
                    binding_scope=instruction.environment.scope, executes_body_at_declaration=False))
            elif type(instruction) is P.BuildFunction:
                constructions.append(dict(operation=index, kind=instruction.environment.kind.value,
                    binding_scope=instruction.environment.scope,
                    default_source_count=sum(param.default_source is not None
                                             for param in instruction.environment.parameters),
                    captures=len(instruction.environment.captures), annotations=instruction.annotations is not None))
            elif type(instruction) is P.ImportRequest:
                imports.append(dict(operation=index, requested_name=instruction.requested_name,
                    relative_level=instruction.relative_level, from_names=encode(instruction.from_names)))
    _need(not wanted, 'initializer original operations are missing from template recipes')
    for index, fact in enumerate(source.facts.constants):
        port = source_port(source.facts, fact.value).port
        operation = coordinates.get((port.template, port.owner.wire))
        if operation in selected_literals:
            literals.append(dict(operation=operation, literal=encode(fact.literal),
                origin='ORIGINAL_LITERAL_FACT', fact=encode(slicer.fact(source.facts, 'constants', index))))
    return dict(literals=sorted(literals, key=lambda row: row['operation']),
        deferred_bodies=sorted(declarations, key=lambda row: row['operation']),
        callable_constructions=sorted(constructions, key=lambda row: row['operation']),
        import_operations=sorted(imports, key=lambda row: row['operation']))


def _initializers(session, original, candidates, edges):
    output = []
    requests = {row['operation']: index for index, row in enumerate(original['imports'])}
    calls = {row['operation']: index for index, row in enumerate(original['calls'])}
    slicer = ReferenceSlicer(session.view.catalog)
    for index in sorted(candidates):
        body = original['bodies'][index]
        pointer = f'#/bodies/{index}'
        result = dict(original=pointer, source_region=body['point'], status=body['status'],
                      operation_inventory=None, effects=None, call_sites=[], nested_import_sites=[],
                      dynamic_initialization_proven=False)
        if body['status'] != 'PROJECTED':
            output.append(result)
            continue
        unit = _row(original['units'], body['unit'])
        view = session.view
        source = view.unit(unit['unit'])
        root = source.facts.root_scope
        point = view.source_region_ref(source.unit, root.template, root.region)
        _need(encode(point) == body['point'] and encode(source.namespace) == unit['namespace'],
              'candidate body is not the original declared module initializer')
        coordinates, inventory = _coordinates(original, body)
        history = body['operations']
        details = _detail(original, source, coordinates, inventory, slicer)
        for import_op in details['import_operations']:
            original_row = original['imports'][requests[import_op['operation']]]
            _need(all(import_op[key] == original_row[key] for key in
                  ('requested_name', 'relative_level', 'from_names')), 'typed import request differs from F4 row')
        for operation in history:
            if operation in calls:
                slot = f'#/calls/{calls[operation]}'
                result['call_sites'].append(slot)
                edges.append(_edge('CONTAINS_ORIGINAL_CALL_SITE', pointer, slot, slot))
            if operation in requests:
                slot = f'#/imports/{requests[operation]}'
                result['nested_import_sites'].append(slot)
                edges.append(_edge('CONTAINS_ORIGINAL_IMPORT_REQUEST', pointer, slot, slot))
        source.material.validate_parents()
        result.update(operation_inventory=dict(original=pointer+'/operations', by_instruction=inventory,
            scope='PUBLISHED_NORMAL_SUCCESS_INITIALIZER_BODY'), effects=details)
        output.append(result)
    return output


@reuse_decoded_source_bodies()
def derive_cpu_entry_import_initializers(entry_obligations_ref, *, session=None):
    """Read sealed source candidates and executable initializer operations mechanically."""
    original = read_artifact(entry_obligations_ref)
    _need(entry_obligations_ref['schema'] == OBLIGATIONS and original['schema_version'] == 1
          and original['scope'] == 'NORMAL_SUCCESS_SOURCE_MAY_GRAPH' and original['effects_closed'] is False,
          'original published F4 may-graph obligations required')
    program = original['program']
    manifest = read_artifact(program)
    _need(manifest['backend'] == 'F4' and manifest['workflow_complete'] is True,
          'expected an original F4 program')
    requests, edges, candidates = _requests(original)
    initializers = []
    if candidates:
        build_path = Path(program['uri']).parent
        with (BuildSession(build_path) if session is None else nullcontext(session)) as active:
            _need(active.path == build_path.resolve() and active.report == manifest, 'original F4 build differs')
            initializers = _initializers(active, original, candidates, edges)
    return dict(schema=SCHEMA, schema_version=1, entry_obligations=entry_obligations_ref, program=program,
        status='SOURCE_INITIALIZER_INVENTORIES_PROJECTED' if any(row['operation_inventory'] is not None
            for row in initializers) else 'SOURCE_INITIALIZER_BODIES_OPEN',
        requests=requests, initializers=initializers, edges=edges, coverage_frontier=list(_OPEN),
        runtime_import_effects_closed=False, whole_entry_effect_equivalence_proven=False,
        transformation_authorized=False, target_executed=False)


def verify_cpu_entry_import_initializers(reference, *, session=None):
    """Cold-replay the manifest against the exact original F4 material."""
    saved = read_artifact(reference)
    _need(reference['schema'] == SCHEMA
          and saved == derive_cpu_entry_import_initializers(saved['entry_obligations'], session=session),
          'saved initializer projection differs from original F4 replay')
    return saved
