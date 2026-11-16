"""Source-bound stage structure, separate from source identity and observations.

Entry residency and scalar readiness come from the original renderer capture;
this projection never reconstructs a second cross-stage execution history.
"""
from copy import deepcopy
from hashlib import sha256
import math

from scar.compiler.stateful_model import validate_stateful_region, stateful_digest
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.stateful-stage-signature'
_COMPONENTS = dict(COMPUTE='FUSED_SCALAR_DAG_ROW_STAGE_V2', SCALAR='TAGGED_SCALAR_DAG_SETUP_STAGE_V2')
_ARITHMETIC = frozenset(('add', 'sub', 'mul', 'div', 'neg'))


def project_stage_signature(region, plan, capture, stage_index):
    """Project one closed serial stage from a fresh original-source capture.

    The caller/reader retains and independently reproduces the capture. A
    signature is not a runtime logical version, a measurement, or authority to
    execute a replacement. Original IDs occur only in mapping/entry evidence.
    """
    region = validate_stateful_region(region)
    if (type(plan.get('cpu_threads')) is not int or plan['cpu_threads'] != 1
            or plan.get('region_id') != region.id or plan.get('region_sha256') != stateful_digest(region)):
        raise ValueError('original serial stage plan and source-bound region required')
    if type(stage_index) is not int or stage_index < 0:
        raise ValueError('original nonnegative stage index required')
    stages = [row for row in plan['stages'] if row['index'] == stage_index]
    captures = [row for row in capture['stages'] if row['index'] == stage_index]
    if len(stages) != 1 or len(captures) != 1:
        raise ValueError('one original stage and renderer entry capture required')
    stage, captured = stages[0], captures[0]
    if stage['kind'] not in _COMPONENTS:
        raise ValueError('only closed scalar/array compute stages have structural DAG signatures')
    body = captured['body']
    if type(body) is not str or sha256(body.encode()).hexdigest() != captured['body_sha256']:
        raise ValueError('original captured stage body bytes differ')
    original = {node.id: node for node in region.nodes}; positions = {key: i for i, key in enumerate(original)}
    inputs = {row.id: row for row in region.inputs}; input_positions = {key: i for i, key in enumerate(inputs)}
    shapes, views = captured['shapes'], captured['views']
    available = captured['entry_available']; ready = set(captured['entry_scalar_ready'])
    if (set(shapes) != set(original) or not set(views) <= set(original)
            or not set(available) <= set(original) or not ready <= set(original)
            or any(shapes[key] is None for key in available) or any(shapes[key] is not None for key in ready)):
        raise ValueError('renderer entry shape/residency/scalar-readiness domain differs')
    root = stage['root_expr']; closed = set(stage['expr_ids'])
    if (root not in original or not closed <= set(original)
            or (shapes[root] is None) != (stage['kind'] == 'SCALAR')):
        raise ValueError('stage root/closure/value kind differs from the original capture')
    slots = {row['id']: row for row in plan['workspace_slots']}
    groups = {}; group_rows = []; group_sources = []; nodes = []; projected = {}; mapping = []
    uniform_groups = {}; uniforms = []

    def group(kind, identity):
        key = (kind, identity)
        if key not in groups:
            if kind == 'WORKSPACE_SLOT':
                capacity = slots[identity]['capacity_bytes']
            else:
                shape = plan['input_shapes'][identity]; capacity = 8 * math.prod(shape)
            if type(capacity) is not int or capacity < 0:
                raise ValueError('original backing capacity required')
            index = len(groups); groups[key] = index
            group_rows.append(dict(id=index, kind=kind, capacity_bytes=capacity))
            group_sources.append(dict(group=index, kind=kind, original=identity))
        return groups[key]

    def payload_group(payload):
        return group('WORKSPACE_SLOT', plan['assignment'][payload]['slot_id'])

    def emit(key, record):
        record = dict(id=len(nodes), **record); nodes.append(record); projected[key] = record['id']
        return record['id']

    def uniform(key, kind, original_identity):
        identity = (kind, original_identity)
        if identity not in uniform_groups:
            index = len(uniforms); uniform_groups[identity] = index
            uniforms.append(dict(id=index, kind=kind, representation='FLOAT64_WITH_INTEGER_TAG'))
            return emit(key, dict(kind=kind, uniform=index, representation='TAGGED_SCALAR'))
        index = uniform_groups[identity]
        existing = next(row['id'] for row in nodes if row.get('uniform') == index)
        projected[key] = existing
        return existing

    def visit(key):
        if key in projected:
            return projected[key]
        node = original[key]; shape = shapes[key]
        if shape is not None and (key in available or key in views):
            if key in available:
                backing = payload_group(available[key]); offset = 0; stride = shape[1]
            else:
                name, row, col, rows, cols = views[key]
                if name not in inputs or inputs[name].kind != 'array' or list(shape) != [rows, cols]:
                    raise ValueError('original view root/extent differs')
                stride = plan['input_shapes'][name][1]; offset = row * stride + col
                backing = payload_group(plan['shadows'][name]) if name in plan['shadows'] else group('EXTERNAL_ARRAY', name)
            if (any(type(v) is not int or v < 0 for v in (*shape, offset, stride))
                    or len(shape) != 2 or stride < shape[1]):
                raise ValueError('bounded original unit-step row layout required')
            end = offset if not all(shape) else offset + (shape[0] - 1) * stride + shape[1]
            if end * 8 > group_rows[backing]['capacity_bytes']:
                raise ValueError('original array load exceeds its backing group')
            value = emit(key, dict(kind='ARRAY_LOAD', group=backing, offset=offset, stride=stride,
                shape=list(shape), representation='FLOAT64'))
        elif shape is None and key in ready:
            value = uniform(key, 'READY_SCALAR', key)
        elif node.op == 'const':
            constant = node.attrs['value']
            value = emit(key, dict(kind='CONSTANT', float64_hex=float(constant).hex(),
                integer_tag=type(constant) is int, representation='TAGGED_SCALAR'))
        elif node.op == 'input' and shape is None:
            name = node.attrs['name']
            if name not in inputs or inputs[name].kind != 'scalar':
                raise ValueError('original runtime scalar input required')
            value = uniform(key, 'RUNTIME_SCALAR', name)
        else:
            if node.op not in _ARITHMETIC or key not in closed:
                raise ValueError('nonresident provider or expression outside original closed stage')
            operands = [visit(operand) for operand in node.inputs]
            value = emit(key, dict(kind='SCALAR_OP' if shape is None else 'ARRAY_OP', op=node.op,
                inputs=operands, representation='TAGGED_SCALAR' if shape is None else 'FLOAT64'))
        mapping.append(dict(node=value, expression=key, region_node_ref=dict(artifact_role='region', pointer='#/nodes/'+str(positions[key])),
            original_op=node.op, original_inputs=list(node.inputs), original_attrs=deepcopy(node.attrs),
            input_ref=(dict(artifact_role='region', pointer='#/inputs/'+str(input_positions[node.attrs['name']]))
                       if node.op == 'input' else None)))
        return value

    result = visit(root)
    if stage['kind'] == 'COMPUTE':
        shape = stage['shape']
        if list(shapes[root]) != list(shape) or stage['output_payload'] is None:
            raise ValueError('original array output shape/payload differs')
        backing = payload_group(stage['output_payload'])
        if 8 * math.prod(shape) > group_rows[backing]['capacity_bytes']:
            raise ValueError('original output exceeds its physical backing capacity')
        output = dict(kind='ARRAY_STORE', group=backing, offset=0, stride=shape[1], shape=list(shape))
    else:
        shape = None
        if stage['output_payload'] is not None or root not in captured['exit_scalar_ready']:
            raise ValueError('original scalar stage result must become a ready scalar port')
        output = dict(kind='READY_SCALAR_RESULT', representation='FLOAT64_WITH_INTEGER_TAG')
    parameters = dict(stage_kind=stage['kind'], shape=None if shape is None else list(shape), workers=1,
        dag=dict(nodes=nodes, root=result, uniform_inputs=uniforms, backing_groups=group_rows, output=output),
        numeric_contract=dict(dtype='float64', arithmetic=plan['numerical_contract']['arithmetic'],
            scalar_representation='FLOAT64_WITH_INTEGER_TAG', integer_range='EXACT_PLUS_MINUS_2_POW_53_OR_PRECOMMIT_GUARD',
            ordered_operands=True, reassociation=False, contraction=False))
    component = _COMPONENTS[stage['kind']]
    return dict(schema=SCHEMA, schema_version=1, component=component, parameters=parameters,
        key=semantic_sha256(dict(component=component, parameters=parameters)),
        source_mapping=dict(region_id=region.id, region_sha256=stateful_digest(region),
            source_ref=dict(artifact_role='region', pointer='#/source'),
            stage_id=stage['id'], stage_index=stage_index, root_expression=root,
            stage_ref=dict(artifact_role='base_stage_plan', pointer='#/stages/'+str(plan['stages'].index(stage))),
            source_action_ids=list(stage['source_action_ids']), nodes=mapping, backing_groups=group_sources,
            body_sha256=captured['body_sha256'], full_source_sha256=capture['full_source_sha256'],
            renderer_provenance=deepcopy(capture['provenance'])),
        entry=dict(available=deepcopy(available), scalar_ready=list(captured['entry_scalar_ready']),
            views=deepcopy(views), shapes=deepcopy(shapes)),
        runtime_logical_identity_claimed=False, measurement_claimed=False, transformation_authorized=False)


__all__ = ['SCHEMA', 'project_stage_signature']
