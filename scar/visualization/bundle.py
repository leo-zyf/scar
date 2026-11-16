"""Package bounded graph projections as an offline, classic-script viewer."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil

ASSETS=Path(__file__).parent/'assets'


def _safe_json(value):
    return json.dumps(value,ensure_ascii=True,separators=(',',':'),allow_nan=False).replace(
        '<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')


def write_view_bundle(projection,out,*,title=None,hardware=None):
    if projection.get('schema')!='scar.graph-viewer-projection' or projection.get('schema_version')!=1:
        raise ValueError('original graph viewer projection required')
    out=Path(out).resolve()
    if out.exists():raise ValueError('fresh viewer output required; existing files are not overwritten')
    parts=deepcopy(projection['parts']);resource_root=None;resource_stats=None
    if hardware is not None:
        from .adapters import project_hardware
        resources=project_hardware(hardware);parts.extend(resources['parts'])
        resource_root=resources['root'];resource_stats=resources['stats']
    ids={part['id'] for part in parts}
    if len(ids)!=len(parts) or projection['root'] not in ids:raise ValueError('unique graph parts and original root required')
    manifest=dict(schema='scar.offline-graph-view',schema_version=1,
        title=title or projection.get('title','SCAR 图查看器'),root=projection['root'],resource_root=resource_root,
        source=projection.get('source'),coverage=projection.get('coverage',{}),limits=projection.get('limits',{}),
        stats=projection.get('stats',{}),resource_stats=resource_stats,parts={})
    prepared=[];total=0
    for index,part in enumerate(parts):
        node_ids={node['id'] for node in part['nodes']}
        if len(node_ids)!=len(part['nodes']):raise ValueError('duplicate node within a presentation part')
        for node in part['nodes']:
            children=node.get('children_parts',node.get('child_parts',[]))
            if any(child not in ids for child in children):raise ValueError('child part is not exported')
            node['children_parts']=list(dict.fromkeys(children))
            node.setdefault('details',{})
            for field in ('point','source','owner','role','neighborhood_queried','original_kind'):
                if field in node:node['details'].setdefault(field,node[field])
            node['kind']=node.get('kind') or node.get('role') or 'PORT'
        for edge in part['edges']:
            if edge['source'] not in node_ids or edge['target'] not in node_ids:
                raise ValueError('every displayed edge must retain both original endpoints')
            edge.setdefault('label',edge['kind']);edge.setdefault('layer','other');edge.setdefault('strength','UNKNOWN')
            edge.setdefault('details',{})
            for field in ('meaning','conditions','evidence','runtime_occurrence'):
                if field in edge:edge['details'].setdefault(field,edge[field])
        part.setdefault('label',part.get('title',part['id']))
        file=f'parts/part-{index:04d}.js'
        raw=('window.SCAR_VIEW.register(JSON.parse('+_safe_json(_safe_json(part))+'));\n').encode()
        total+=len(raw)
        if total>64*1024**2:raise ValueError('viewer projection exceeds 64 MiB; select smaller scope or limits')
        prepared.append((file,raw))
        manifest['parts'][part['id']]=dict(file=file,label=part['label'],node_count=len(part['nodes']),
            edge_count=len(part['edges']),boundary_count=len(part.get('boundaries',[])),
            frontier_count=len(part.get('frontiers',[])),root=part.get('root'))
    files={'index.html':ASSETS/'viewer.html','viewer.js':ASSETS/'viewer.js','viewer.css':ASSETS/'viewer.css',
        'vendor/cytoscape.min.js':ASSETS/'vendor/cytoscape.min.js',
        'vendor/cytoscape.LICENSE':ASSETS/'vendor/cytoscape.LICENSE',
        'vendor/cytoscape.json':ASSETS/'vendor/cytoscape.json'}
    if any(not path.is_file() for path in files.values()):raise ValueError('offline viewer assets are incomplete')
    out.mkdir(parents=True)
    for name,path in files.items():
        dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    for file,raw in prepared:
        path=out/file;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    (out/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,ensure_ascii=False,indent=2)+'\n')
    (out/'manifest.js').write_text('window.SCAR_GRAPH_MANIFEST=JSON.parse('+_safe_json(_safe_json(manifest))+');\n')
    report=dict(schema='scar.graph-view-export',schema_version=1,entry=str(out/'index.html'),
        manifest=str(out/'manifest.json'),parts=len(parts),graph_data_bytes=total,
        source=manifest['source'],coverage=manifest['coverage'],resource_stats=resource_stats,
        target_program_executed=False,ir_modified=False,network_required_at_view_time=False,
        files={str(path.relative_to(out)):dict(bytes=path.stat().st_size,sha256=sha256(path.read_bytes()).hexdigest())
            for path in out.rglob('*') if path.is_file()})
    (out/'result.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    return report
