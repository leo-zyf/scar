"""Small presentation adapters; recorded relations remain the authority."""
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

from scar.ir.record_codec import encode
from scar.ir.versioned_schedule import TemporalScopeKind, VersionedTaskKind
from scar.optimization.schedule import schedule_document, schedule_from_dict, validate_schedule


def _id(kind, value):
    from scar.optimization.resource_solver import semantic_sha256
    return kind+'-'+semantic_sha256(value)[:32]


def project_schedule(schedule_or_document):
    """Show task/value edges and shared loop bodies of a versioned graph."""
    schedule = (schedule_from_dict(schedule_or_document) if type(schedule_or_document) is dict
                else schedule_or_document)
    validate_schedule(schedule)
    document = schedule_document(schedule)
    if document['schema_version'] not in (2, 3, 4):
        raise ValueError('viewer task adapter requires the versioned schedule')
    if document['schema_version'] == 4:
        return _project_schedule_v4(schedule, document)
    from scar.optimization.resource_solver import semantic_sha256
    namespace = semantic_sha256(document)
    identifier = lambda kind, name: _id(kind, (namespace, name))
    root = identifier('scope', schedule.call_scope)
    loop = None
    if schedule.temporal is not None:
        loops = [row for row in schedule.temporal.scopes if row.kind is TemporalScopeKind.LOOP]
        if len(loops) > 1:
            raise ValueError('task presentation currently supports one shared loop scope')
        loop = loops[0] if loops else None
    body = identifier('scope', loop.id) if loop else root
    root_part = identifier('part', 'root')
    body_part = identifier('part', 'body') if loop else root_part
    root_node = dict(id=root, label=schedule.region_id, kind='REGION',
        details=dict(original_call_scope=schedule.call_scope, original_region_id=schedule.region_id,
            source_budget_scope=schedule.budget_scope, source_budget_bytes=schedule.budget_bytes,
            representation='VERSIONED_TASK_GRAPH_NOT_FULL_PROGRAM_SOURCE', executable=schedule.executable))
    body_node = (dict(id=body, label='循环 '+(loop.trip_count_symbol or str(loop.trip_count)),
        kind='REGION', role='LOOP', details=dict(original=encode(loop), iterations_unrolled=0),
        child_parts=[body_part]) if loop else root_node)
    nodes = [body_node]; edges = []; parts = []
    materials = {row.id:row for row in schedule.materializations}
    values = {row.id:row for row in schedule.values}
    slots = {row.id:row for row in schedule.storage_slots}
    writers = {name:task.id for task in schedule.tasks for name in task.writes}
    def edge(source, target, kind, layer, details, label=''):
        return dict(id=identifier('edge', (source, target, kind, details)), source=source, target=target,
            kind=kind, layer=layer, strength='EXACT', label=label or kind, details=details)
    for task in schedule.tasks:
        task_id = identifier('task', task.id); port_part = identifier('ports', task.id)
        nodes.append(dict(id=task_id, label=task.kind.value+' · '+task.id, kind='OPERATION',
            child_parts=[port_part], details=dict(original_task_id=task.id, record=encode(task))))
        for before in task.depends_on:
            edges.append(edge(identifier('task', before), task_id, 'TASK_DEPENDENCY', 'control',
                dict(original_before=before, original_after=task.id, original_task=task.id)))
        port_nodes = [dict(id=task_id, label=task.kind.value+' · '+task.id, kind='OPERATION',
            details=dict(original_task_id=task.id, record=encode(task)))]; port_edges=[]
        for name in dict.fromkeys((*task.reads, *task.writes)):
            material=materials[name]; slot=slots[material.region.slot]; value=values[material.value]
            material_id=identifier('material', name)
            port_nodes.append(dict(id=material_id, label=name, kind='VALUE',
                details=dict(materialization=encode(material), logical_value=encode(value),
                    storage=encode(slot), version_scope='CALL_LOCAL_STORAGE_EPOCH_NOT_CONTENT_EQUALITY')))
            if name in task.reads:
                port_edges.append(edge(material_id, task_id, 'READ', 'data',dict(materialization=name)))
                if name in writers:
                    edges.append(edge(identifier('task',writers[name]),task_id,'MATERIALIZATION_FLOW','data',
                        dict(materialization=name,producer=writers[name],consumer=task.id),label=name))
            if name in task.writes:
                port_edges.append(edge(task_id,material_id,'WRITE','state' if task.kind in (
                    VersionedTaskKind.STATE_WRITE,VersionedTaskKind.COMMIT) else 'data',dict(materialization=name)))
        for name in (*task.allocates,*task.releases):
            slot=slots[name];slot_id=identifier('slot',name)
            port_nodes.append(dict(id=slot_id,label=name+' · '+str(slot.capacity_bytes)+' B',kind='RESOURCE',
                details=dict(original_storage_slot=encode(slot),allocation_is_not_value_readiness=True)))
            source,target=(task_id,slot_id) if name in task.allocates else (slot_id,task_id)
            port_edges.append(edge(source,target,'ALLOCATES' if name in task.allocates else 'RELEASES',
                'resource',dict(original_storage_slot=name)))
        if len(port_nodes)>1:
            parts.append(dict(id=port_part, root=task_id, label='读写端口与存储', nodes=port_nodes,
                edges=port_edges, frontiers=[],boundaries=[],complete=True))
        else:
            nodes[-1]['child_parts']=[]
    parts.insert(0,dict(id=body_part,root=body,label=schedule.region_id,nodes=nodes,edges=edges,
        frontiers=list(schedule.unresolved),boundaries=[],loops_unrolled=False,complete=True))
    if loop:
        parts.insert(0,dict(id=root_part,root=root,label=schedule.region_id,nodes=[root_node,body_node],
            edges=[edge(root,body,'SCOPE_CONTAINS','structure',dict(original_scope=loop.id))],
            frontiers=[],boundaries=[],loops_unrolled=False,complete=True))
    return dict(schema='scar.graph-viewer-projection',schema_version=1,title='SCAR · '+schedule.region_id,
        root=root_part,parts=parts,source=dict(kind='VERSIONED_TASK_GRAPH',sha256=namespace),
        coverage=dict(complete=True,scope='RETAINED_TASK_RECORDS_ONLY_NOT_FULL_PROGRAM',
            loops_unrolled=False,target_program_executed=False),
        stats=dict(parts=len(parts),tasks=len(schedule.tasks),materials=len(materials)))


def _project_schedule_v4(schedule, document):
    """Keep root workspace and one iteration template in distinct lazy parts."""
    from scar.optimization.resource_solver import semantic_sha256
    temporal = schedule.temporal
    roots = [s for s in temporal.scopes if s.kind is TemporalScopeKind.ROOT]
    loops = [s for s in temporal.scopes if s.kind is TemporalScopeKind.LOOP]
    if len(roots) == len(temporal.scopes) == 1 and not loops and not schedule.tasks:
        if temporal.activities or schedule.values or schedule.materializations or schedule.storage_slots:
            raise ValueError('empty v4 root must not hide active data or storage records')
        namespace = semantic_sha256(document)
        root = _id('scope', (namespace, roots[0].id)); root_part = _id('part', (namespace, 'root'))
        condition = encode(temporal.path_condition)
        return dict(schema='scar.graph-viewer-projection', schema_version=1,
            title='SCAR · '+schedule.region_id, root=root_part,
            parts=[dict(id=root_part, root=root, label=schedule.region_id,
                nodes=[dict(id=root, label=schedule.region_id+' · 空执行路径', kind='REGION', parent=None,
                    details=dict(original_scope=encode(roots[0]), source_proof=encode(schedule.proof),
                        path_condition=condition, active_tasks=0, storage_allocations=0))],
                edges=[], frontiers=list(schedule.unresolved), boundaries=[], complete=True, loops_unrolled=False)],
            source=dict(kind='VERSIONED_TASK_GRAPH', schema_version=4, sha256=namespace),
            coverage=dict(complete=True, scope='RETAINED_TASK_RECORDS_ONLY_NOT_FULL_PROGRAM',
                path_condition=condition, loops_unrolled=False, target_program_executed=False),
            stats=dict(parts=1, tasks=0, materials=0, root_tasks=0, shared_body_tasks=0, iterations_unrolled=0))
    if (len(roots) != 1 or len(loops) != 1 or len(temporal.scopes) != 2
            or loops[0].parent != roots[0].id):
        raise ValueError('v4 task presentation supports one root and one direct shared loop only')
    root_scope, loop = roots[0], loops[0]
    namespace = semantic_sha256(document)
    identifier = lambda kind, name: _id(kind, (namespace, name))
    scope_nodes = {root_scope.id: identifier('scope', root_scope.id), loop.id: identifier('scope', loop.id)}
    part_ids = {root_scope.id: identifier('part', 'root'), loop.id: identifier('part', 'body')}
    activities = defaultdict(list)
    for activity in temporal.activities:
        activities[activity.task].append(activity)
    task_scopes = {}
    for task in schedule.tasks:
        scopes = {a.scope for a in activities[task.id]}
        if len(scopes) != 1 or next(iter(scopes)) not in scope_nodes:
            raise ValueError('each v4 task needs activities in one explicit supported scope')
        task_scopes[task.id] = next(iter(scopes))
    materials = {m.id: m for m in schedule.materializations}
    values = {v.id: v for v in schedule.values}
    slots = {s.id: s for s in schedule.storage_slots}
    tasks = {t.id: t for t in schedule.tasks}
    events = {event.id: event for event in temporal.events}
    local_materials = set(loop.iteration_local_materializations)
    local_values = set(loop.iteration_local_values)
    iteration = loop.iteration_symbol or 'i'
    condition = encode(temporal.path_condition)
    conditions = [] if condition is None else [condition]
    root = scope_nodes[root_scope.id]; body = scope_nodes[loop.id]
    root_node = dict(id=root, label=schedule.region_id, kind='REGION', parent=None,
        details=dict(original_call_scope=schedule.call_scope, original_region_id=schedule.region_id,
            original_scope=encode(root_scope), source_budget_scope=schedule.budget_scope,
            source_budget_bytes=schedule.budget_bytes, executable=schedule.executable,
            representation='VERSIONED_SHARED_LOOP_TASK_GRAPH', path_condition=condition))
    body_node = dict(id=body, label='循环 '+(loop.trip_count_symbol or str(loop.trip_count))+' · 共享正文',
        kind='REGION', role='LOOP', parent=root, child_parts=[part_ids[loop.id]],
        details=dict(original_scope=encode(loop), iterations_unrolled=0,
            iteration_coordinate='('+iteration+', epoch)', path_condition=condition))
    containers = {
        root_scope.id: dict(id=part_ids[root_scope.id], root=root, label=schedule.region_id,
            nodes=[root_node, body_node], edges=[], frontiers=list(schedule.unresolved), boundaries=[],
            loops_unrolled=False, complete=True),
        loop.id: dict(id=part_ids[loop.id], root=body, label='共享循环正文',
            nodes=[{**body_node, 'child_parts': []}], edges=[], frontiers=[], boundaries=[],
            loops_unrolled=False, complete=True),
    }
    parts = []; node_ids = {key: {n['id'] for n in part['nodes']} for key, part in containers.items()}

    def append_node(scope, node):
        if node['id'] not in node_ids[scope]:
            containers[scope]['nodes'].append(node); node_ids[scope].add(node['id'])

    def edge(source, target, kind, layer, details, label=None):
        return dict(id=identifier('edge', (source, target, kind, details)), source=source, target=target,
            kind=kind, layer=layer, strength='EXACT', label=label or kind,
            conditions=conditions, details={**details, 'path_condition': condition})

    def task_node(task, *, expandable=True):
        return dict(id=identifier('task', task.id), label=task.kind.value+' · '+task.id, kind='OPERATION',
            parent=scope_nodes[task_scopes[task.id]],
            child_parts=[identifier('ports', task.id)] if expandable else [],
            details=dict(original_task_id=task.id, record=encode(task),
                original_scope=task_scopes[task.id], iteration_template=task_scopes[task.id] == loop.id,
                original_activities=[encode(a) for a in activities[task.id]]))

    def material_node(name):
        material = materials[name]; local = name in local_materials
        coordinate = dict(iteration=iteration if local else None, epoch=material.version,
            logical_value_iteration_local=material.value in local_values)
        return dict(id=identifier('material', name), label=name+(
            ' · ('+iteration+', epoch '+str(material.version)+')' if local else ' · epoch '+str(material.version)),
            kind='VALUE', parent=body if local else root,
            details=dict(materialization=encode(material), logical_value=encode(values[material.value]),
                storage=encode(slots[material.region.slot]), iteration_coordinate=coordinate,
                version_scope='ITERATION_LOCAL_TEMPLATE_EPOCH' if local else 'ROOT_CALL_LOCAL_STORAGE_EPOCH',
                storage_allocation_scope=root_scope.id,
                physical_slot_is_not_a_new_allocation_per_iteration=True))

    slot_parts = {name: identifier('storage', name) for name in slots}
    def slot_node(name, *, expandable=True):
        return dict(id=identifier('slot', name), label=name+' · '+str(slots[name].capacity_bytes)+' B',
            kind='RESOURCE', parent=root, child_parts=[slot_parts[name]] if expandable else [],
            details=dict(original_storage_slot=encode(slots[name]), allocation_scope=root_scope.id,
                allocation_is_not_value_readiness=True, iteration_instances_allocated=False))

    for task in schedule.tasks:
        scope = task_scopes[task.id]; node = task_node(task); append_node(scope, node)
        port_nodes = {node['id']: task_node(task, expandable=False)}; port_edges = []
        for name in dict.fromkeys((*task.reads, *task.writes)):
            material = material_node(name); port_nodes[material['id']] = material
            if name in task.reads:
                port_edges.append(edge(material['id'], node['id'], 'READ', 'data', dict(materialization=name)))
            if name in task.writes:
                layer = 'state' if task.kind in (VersionedTaskKind.STATE_WRITE, VersionedTaskKind.COMMIT) else 'data'
                port_edges.append(edge(node['id'], material['id'], 'WRITE', layer, dict(materialization=name)))
        for name in dict.fromkeys((*task.allocates, *task.releases)):
            if scope != root_scope.id:
                raise ValueError('v4 shared workspace allocation/release must stay in root scope')
            storage = slot_node(name); port_nodes[storage['id']] = storage
            source, target = (node['id'], storage['id']) if name in task.allocates else (storage['id'], node['id'])
            port_edges.append(edge(source, target, 'ALLOCATES' if name in task.allocates else 'RELEASES',
                'resource', dict(original_storage_slot=name)))
        if len(port_nodes) > 1:
            parts.append(dict(id=identifier('ports', task.id), root=node['id'], label='读写端口与存储',
                nodes=list(port_nodes.values()), edges=port_edges, frontiers=[], boundaries=[], complete=True))
        else:
            node['child_parts'] = []

    def task_relation(before, after, kind, layer, details, label=None):
        left, right = task_scopes[before], task_scopes[after]
        scope = left if left == right else root_scope.id
        source = identifier('task', before) if left == scope else body
        target = identifier('task', after) if right == scope else body
        details = {**details, 'original_before': before, 'original_after': after,
            'original_before_scope': left, 'original_after_scope': right,
            'presentation_scope_summary': left != right, 'original_endpoints_replaced_in_IR': False}
        containers[scope]['edges'].append(edge(source, target, kind, layer, details, label))
        if left != right:
            containers[scope]['boundaries'].append(dict(kind=kind, **details))

    for task in schedule.tasks:
        for before in task.depends_on:
            task_relation(before, task.id, 'TASK_DEPENDENCY', 'control', dict(original_task=task.id))
        for name in task.reads:
            producer = materials[name].ready_task
            if producer is not None:
                task_relation(producer, task.id, 'MATERIALIZATION_FLOW', 'data',
                    dict(materialization=name, original_region=encode(materials[name].region)), name)
    containers[root_scope.id]['edges'].append(edge(root, body, 'SCOPE_CONTAINS', 'structure',
        dict(original_scope=loop.id, presentation_only=True)))
    for carry in getattr(loop, 'content_carries', ()):
        before, after = materials[carry.before], materials[carry.after]
        containers[loop.id]['edges'].append(edge(body, body, 'LOOP_CONTENT_CARRY', 'data',
            dict(original_scope=loop.id, record=encode(carry), before_materialization=carry.before,
                after_materialization=carry.after, iteration_distance=carry.iteration_distance,
                source_refs=encode(carry.source_refs),
                before_coordinate=dict(iteration=iteration, epoch=before.version),
                after_coordinate=dict(iteration=iteration+' + '+str(carry.iteration_distance), epoch=after.version),
                presentation_scope_summary=True, iteration_instances_created=False,
                time_constraint_replaced=False),
            '内容递推 ('+iteration+', epoch '+str(before.version)+') → ('+iteration+' + '
                +str(carry.iteration_distance)+', epoch '+str(after.version)+')'))

    # Workspace is shown once at root; expansion exposes its original lifetime
    # and versioned body materials, never another allocation inside the loop.
    for name, slot in slots.items():
        node = slot_node(name); append_node(root_scope.id, node)
        storage_nodes = {node['id']: slot_node(name, expandable=False)}; storage_edges = []
        for task_id, kind in ((slot.allocated_by, 'ALLOCATES'), (slot.released_by, 'RELEASES')):
            if task_id is None:
                continue
            if task_scopes[task_id] != root_scope.id:
                raise ValueError('v4 physical workspace lifetime cannot be owned by a loop-body task')
            task = task_node(tasks[task_id]); storage_nodes[task['id']] = task
            source, target = (task['id'], node['id']) if kind == 'ALLOCATES' else (node['id'], task['id'])
            storage_edges.append(edge(source, target, kind, 'resource', dict(original_storage_slot=name)))
        for material in schedule.materializations:
            if material.region.slot == name:
                item = material_node(material.id); storage_nodes[item['id']] = item
                storage_edges.append(edge(node['id'], item['id'], 'MATERIALIZED_IN_SLOT', 'resource',
                    dict(original_materialization=material.id, presentation_only=True,
                        physical_slot_reallocated=False)))
        occupancies = [encode(o) for o in temporal.occupancies if o.storage_slot == name]
        parts.append(dict(id=slot_parts[name], root=node['id'], label='外层存储生命周期 · '+name,
            nodes=list(storage_nodes.values()), edges=storage_edges, frontiers=[], boundaries=[],
            original_physical_occupancies=occupancies, complete=True))
        node['details']['physical_occupancies'] = occupancies

    def selection_label(selection):
        return None if selection is None else getattr(selection.selector, 'value', selection.selector)

    def event_endpoint(event, scope, selection):
        if event.scope == scope and event.task is not None:
            return identifier('task', event.task)
        # Ports are finite symbolic FIRST/LAST boundary declarations, not
        # expanded iterations or dynamic invocations of the shared template.
        port = identifier('event-port', (event.id, encode(selection), scope))
        owner = scope_nodes[event.scope]
        append_node(scope, dict(id=port, label=event.kind.value+(
            ' · '+selection_label(selection) if selection is not None else ''),
            kind='BOUNDARY_PORT', parent=owner,
            details=dict(original_event_id=event.id, record=encode(event),
                iteration_selection=encode(selection), presentation_only=True, runtime_invocation=False)))
        return port

    for constraint in temporal.constraints:
        before, after = events[constraint.before], events[constraint.after]
        scope = before.scope if before.scope == after.scope else root_scope.id
        source = event_endpoint(before, scope, constraint.before_iteration)
        target = event_endpoint(after, scope, constraint.after_iteration)
        # Distinct same-task event endpoints must not look like a task cycle.
        if source == target:
            for event, selection in ((before, constraint.before_iteration), (after, constraint.after_iteration)):
                port = identifier('event-port', (event.id, encode(selection), scope))
                append_node(scope, dict(id=port, label=event.kind.value, kind='BOUNDARY_PORT',
                    parent=scope_nodes[event.scope], details=dict(original_event_id=event.id,
                        record=encode(event), iteration_selection=encode(selection), presentation_only=True)))
            source = identifier('event-port', (before.id, encode(constraint.before_iteration), scope))
            target = identifier('event-port', (after.id, encode(constraint.after_iteration), scope))
        labels = [selection_label(s) for s in (constraint.before_iteration, constraint.after_iteration) if s is not None]
        details = dict(original_constraint_id=constraint.id, record=encode(constraint),
            original_before_event=encode(before), original_after_event=encode(after),
            source_refs=encode(constraint.source_refs), presentation_only_endpoints=True)
        label = constraint.kind.value+(' · '+' → '.join(labels) if labels else '')
        containers[scope]['edges'].append(edge(source, target, 'TEMPORAL_CONSTRAINT', 'control', details, label))
        if labels:
            containers[scope]['boundaries'].append(details)
    ordered_parts = [containers[root_scope.id], containers[loop.id], *parts]
    for part in ordered_parts:
        ids = {node['id'] for node in part['nodes']}
        if any(e['source'] not in ids or e['target'] not in ids for e in part['edges']):
            raise ValueError('v4 presentation edge must retain both displayed endpoints in its own part')
    return dict(schema='scar.graph-viewer-projection', schema_version=1, title='SCAR · '+schedule.region_id,
        root=part_ids[root_scope.id], parts=ordered_parts,
        source=dict(kind='VERSIONED_TASK_GRAPH', schema_version=4, sha256=namespace),
        coverage=dict(complete=True, scope='RETAINED_TASK_RECORDS_ONLY_NOT_FULL_PROGRAM',
            path_condition=condition, loops_unrolled=False, target_program_executed=False),
        stats=dict(parts=len(ordered_parts), tasks=len(tasks), materials=len(materials),
            root_tasks=sum(scope == root_scope.id for scope in task_scopes.values()),
            shared_body_tasks=sum(scope == loop.id for scope in task_scopes.values()), iterations_unrolled=0))


def project_hardware(target):
    """Keep every resource and connection in a kind-sharded inventory view."""
    from scar.analysis.target_graph import TargetHardwareGraph
    from scar.ir.execution_recipe import artifact_reference
    from scar.ir.program import target_hardware as H
    graph = target if isinstance(target, TargetHardwareGraph) else TargetHardwareGraph(Path(target))
    snapshot = artifact_reference(graph.path/'result.json', 'scar.target-hardware-graph')
    records=graph.resources(); links=graph.connections(); groups=defaultdict(list)
    for row in records:groups[type(row).__name__].append(row)
    identifier=lambda kind, name:_id(kind,(snapshot,name))
    node_id=lambda row:identifier('hardware',str(row.id))
    part_id=lambda name:identifier('hardware-part',name)
    root=identifier('hardware-group','root')
    root_nodes=[dict(id=root,label='硬件资源库存',kind='REGION',layer='resource',
        details=dict(snapshot=snapshot,physical_use_established=False,presentation_only=True))]
    parts=[]; by_id={row.id:row for row in records}
    def resource_node(row):
        name=type(row).__name__
        label=(name+' CPU '+str(row.cpu_id) if isinstance(row,H.HardwareThread) else
               name+' '+str(getattr(row,'name',None) or getattr(row,'numa_node',None) or row.id))
        return dict(id=node_id(row),label=label,kind='RESOURCE',role=name,layer='resource',
            details=dict(record=encode(row),snapshot=snapshot,physical_use_established=False,
                reported_capacity_is_available_budget=False), child_parts=[])
    for name, rows in sorted(groups.items()):
        group=identifier('hardware-group',name)
        root_nodes.append(dict(id=group,label=f'{name} · {len(rows)}',kind='REGION',layer='resource',
            child_parts=[part_id(name)],details=dict(presentation_only=True,original_resource_type=name)))
        nodes={group:dict(id=group,label=name,kind='REGION',layer='resource',details=dict(presentation_only=True))}
        own={row.id for row in rows}; edges=[]
        for row in rows:nodes[node_id(row)]=resource_node(row)
        for index,link in enumerate(links):
            if link.source not in own and link.target not in own:continue
            for endpoint in (link.source,link.target):
                row=by_id[endpoint]
                if node_id(row) not in nodes:
                    n=resource_node(row);n['child_parts']=[part_id(type(row).__name__)]
                    n['details']['reference_to_other_resource_group']=True;nodes[n['id']]=n
            edges.append(dict(id=identifier('hardware-edge',index),source=node_id(by_id[link.source]),
                target=node_id(by_id[link.target]),kind=link.kind.value,layer='resource',
                strength='EXACT' if link.kind is H.ConnectionKind.CONTAINS else 'MAY',
                label=link.kind.value,details=dict(record=encode(link),original_connection_index=index,
                    inventory_relation_not_actual_workload_use=True)))
        parts.append(dict(id=part_id(name),root=group,label=name,nodes=list(nodes.values()),edges=edges,
            frontiers=[],boundaries=[],complete=True))
    parts.insert(0,dict(id=part_id('root'),root=root,label='硬件拓扑',nodes=root_nodes,edges=[],
        frontiers=[],boundaries=[],complete=True))
    return dict(root=part_id('root'),parts=parts,source=snapshot,
        stats=dict(resources=len(records),connections=len(links)),inventory_only=True)
