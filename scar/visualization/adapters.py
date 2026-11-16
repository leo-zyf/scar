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
    """Show task/value edges and shared loop bodies of an existing v2/v3 graph."""
    schedule = (schedule_from_dict(schedule_or_document) if type(schedule_or_document) is dict
                else schedule_or_document)
    validate_schedule(schedule)
    document = schedule_document(schedule)
    if document['schema_version'] not in (2, 3):
        raise ValueError('viewer task adapter requires the versioned schedule')
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
