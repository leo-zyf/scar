"""Publish and independently replay typed F4 binding state tables."""

import argparse
from collections import Counter
from hashlib import sha256
from itertools import chain, zip_longest
import json
import os
from pathlib import Path
import time

from scar.analysis.build_reader_versions import STATE_OWNER_PREDECESSOR, STATE_OWNER_READER
from scar.analysis.program_view import ProgramView
from scar.analysis.source_effects import catalog
from scar.analysis.source_value_flow import _artifact
from scar.artifacts import _durable_bytes, _durable_mkdir, _json, write_table
from scar.artifacts import read_record_table as _table
from scar.ir.program import binding_ports as B, state_ports as S, source_productions as P
from scar.ir.program import data_materials as C, data_materials as L, normal_graph as F, view as V
from scar.ir.program.ids import OperationID
from scar.ir.record_codec import _decode, _record_tags, decode, encode, loads


SCHEMA = 'scar.source-state-view'
ENCODING = 'SHARED_CONTROL_DEFINITION_SETS_V1'
_FIELDS = frozenset(S.StoredBindingTrack.__dataclass_fields__)
_STATIC_COUNTS = ('operations', 'binding_accesses', 'content_accesses', 'content_addresses',
    'content_states', 'content_relations', 'content_lineage_records', 'content_lineage_relations',
    'missing_address_rules')
_STATIC_TABLES = ('rules', 'requests', 'accesses', 'content_addresses', 'content_states',
    'content_accesses', 'content_edges', 'content_lineage', 'content_lineage_edges')
_NO_AUTHORITY = ('loops_unrolled', 'target_executed', 'tensor_content_versions_proven',
                 'effect_membership_closed', 'first_stage_complete')


def _pins():
    from scar.analysis import source_state as source_state, source_effects as source_effects, source_name_definitions
    from scar.analysis import source_stack_routes as source_name_access_anchors, source_value_flow, program_view
    from scar.analysis import source_namespace_contents as source_implicit_namespaces
    from scar.analysis import annotation_state as source_annotation_namespaces
    from scar.analysis import source_content, source_content_interfaces
    from scar.analysis import class_bindings as source_environment_values
    from scar.analysis import source_content_lineage
    from scar.ir.program import class_ports as environment_types
    from scar.ir.program import source_effects as effects_types
    return {name:sha256(Path(module.__file__).read_bytes()).hexdigest() for name,module in (
        ('producer', __import__(__name__, fromlist=['__file__'])), ('state', source_state),
        ('state_types', S), ('effect_catalog', source_effects), ('effect_types', effects_types),
        ('definitions', source_name_definitions), ('anchors', source_name_access_anchors),
        ('implicit_addresses', source_implicit_namespaces),
        ('annotation_addresses', source_annotation_namespaces),
        ('content', source_content), ('content_types', C), ('content_interfaces', source_content_interfaces),
        ('content_lineage', source_content_lineage), ('content_lineage_types', L),
        ('environment_values', source_environment_values), ('environment_value_types', environment_types),
        ('body_projection', source_value_flow), ('unified_adapter', program_view))}


def storage_pins():
    return dict(codec=sha256(Path(__file__).read_bytes()).hexdigest(),
                types=sha256(Path(B.__file__).read_bytes()).hexdigest())


def compatible_storage_pins(recorded, current=None):
    active = storage_pins() if current is None else current
    return recorded == active or (active['codec'] == STATE_OWNER_READER['program_state_view.py']
        and recorded == dict(active, codec=STATE_OWNER_PREDECESSOR['binding_track_storage.py']))


def compatible_state_pins(recorded, current=None):
    active = _pins() if current is None else current
    return recorded == active or (active.get('producer') == STATE_OWNER_READER['program_state_view.py']
        and active.get('state') == STATE_OWNER_READER['source_state.py']
        and recorded == _previous_state_pins(active))


def _previous_state_pins(active):
    return dict(active, producer=STATE_OWNER_PREDECESSOR['program_state_view.py'],
                state=STATE_OWNER_PREDECESSOR['source_state.py'])


def _mask(values,count):
    bits=0;previous=-1
    for value in values:
        if type(value)is not int or not previous<value<count:
            raise ValueError('definition set must contain ordered unique track-local ordinals')
        bits|=1<<value;previous=value
    return format(bits,'x')


def _values(mask,count):
    if (type(mask)is not str or not mask or any(c not in '0123456789abcdef' for c in mask)
            or len(mask)>1 and mask[0]=='0'):
        raise ValueError('noncanonical definition bitset')
    bits=int(mask,16)
    if bits.bit_length()>count:raise ValueError('definition bitset exceeds its track domain')
    result=[]
    while bits:
        bit=bits&-bits;result.append(bit.bit_length()-1);bits^=bit
    return result


class TrackPacker:
    def __init__(self):self.controls=[];self._controls={}

    def _control(self,control):
        # A wire key is stable for both typed publication and legacy repacking.
        typed=type(control)is B.SharedBindingControl
        key=_json((encode(control.source_function),control.max_states,control.max_stack)) if typed else _json(
            (control['source_function'],control['max_states'],control['max_stack']))
        if key in self._controls:
            ordinal=self._controls[key]
            prior=self.controls[ordinal]
            if type(prior)is not type(control) or control!=prior:
                raise ValueError('one function/control budget has different matched topology')
            return ordinal
        ordinal=len(self.controls);self._controls[key]=ordinal;self.controls.append(control);return ordinal

    def pack(self,track):
        if type(track)is not S.StoredBindingTrack:raise TypeError('expected a stored binding track')
        control=B.SharedBindingControl(track.source_function,track.max_states,track.max_stack,
            tuple((s.port,s.stack) for s in track.states),track.transitions,track.traversal_complete)
        ordinal=self._control(control);sets=[];by_mask={};by_identity={}
        def intern(values):
            # The input track owns all these immutable tuples for this call.
            # Identity avoids repeatedly hashing long identical IR tuples.
            identity=id(values)
            if identity in by_identity:return by_identity[identity]
            mask=_mask(values,len(track.definitions));i=by_mask.get(mask)
            if i is None:i=len(sets);by_mask[mask]=i;sets.append(mask)
            by_identity[identity]=i;return i
        states=tuple((intern(s.incoming),intern(s.outgoing)) for s in track.states)
        reads=tuple((r.state,r.access,r.anchors,intern(r.definitions)) for r in track.reads)
        return B.PackedBindingTrack(ordinal,track.namespace,track.max_updates,track.definitions,
            tuple(sets),states,reads,track.frontiers,track.dataflow_complete)

    def pack_wire(self,track):
        if set(track)!=_FIELDS:raise ValueError('legacy binding-track field domain differs')
        control={k:track[k] for k in ('source_function','max_states','max_stack','transitions','traversal_complete')}
        control['states']=[[s['port'],s['stack']] for s in track['states']]
        ordinal=self._control(control);sets=[];by_values={}
        def intern(values):
            key=tuple(values);i=by_values.get(key)
            if i is None:i=len(sets);by_values[key]=i;sets.append(_mask(key,len(track['definitions'])))
            return i
        return dict(control=ordinal,namespace=track['namespace'],max_updates=track['max_updates'],
            definitions=track['definitions'],definition_sets=sets,
            states=[[intern(s['incoming']),intern(s['outgoing'])] for s in track['states']],
            reads=[[r['state'],r['access'],r['anchors'],intern(r['definitions'])] for r in track['reads']],
            frontiers=track['frontiers'],dataflow_complete=track['dataflow_complete'])


def unpack_wire(track,controls):
    if set(track)!=set(B.PackedBindingTrack.__dataclass_fields__):raise ValueError('packed track field domain differs')
    i=track['control']
    if type(i)is not int or not 0<=i<len(controls):raise ValueError('track control reference is outside its unit')
    control=controls[i];n=len(control['states']);definitions=track['definitions']
    sets=[_values(mask,len(definitions)) for mask in track['definition_sets']]
    if len(set(track['definition_sets']))!=len(sets) or len(track['states'])!=n:
        raise ValueError('track set pool or control-state denominator differs')
    used=set()
    def get(i):
        if type(i)is not int or not 0<=i<len(sets):raise ValueError('definition-set reference is outside its track')
        used.add(i);return sets[i]
    states=[]
    for coordinate,indices in zip(control['states'],track['states'],strict=True):
        if len(coordinate)!=2 or len(indices)!=2:raise ValueError('control/set coordinate arity differs')
        port,stack=coordinate
        states.append(dict(port=port,stack=stack,incoming=get(indices[0]),outgoing=get(indices[1])))
    reads=[]
    for state,access,anchors,definition_set in track['reads']:
        if type(state)is not int or not 0<=state<n or type(access)is not int or access<0:
            raise ValueError('read state/access reference is invalid')
        reads.append(dict(state=state,access=access,anchors=anchors,definitions=get(definition_set)))
    if used!=set(range(len(sets))):raise ValueError('packed track has unused definition sets')
    return dict(source_function=control['source_function'],namespace=track['namespace'],
        max_states=control['max_states'],max_stack=control['max_stack'],max_updates=track['max_updates'],
        definitions=definitions,states=states,transitions=control['transitions'],reads=reads,
        frontiers=track['frontiers'],traversal_complete=control['traversal_complete'],
        dataflow_complete=track['dataflow_complete'])


def read_tracks(folder,header):
    folder=Path(folder)
    if header['schema_version']==1:
        yield from _table(folder,header['tables']['tracks']);return
    if ((header['schema_version'], header.get('track_encoding')) != (2, ENCODING)
            or not compatible_storage_pins(header.get('storage_pins'))):
        raise ValueError('unsupported binding-track storage contract')
    controls=list(_table(folder,header['tables']['controls']))
    for wire in _table(folder,header['tables']['tracks']):yield unpack_wire(wire,controls)


def repack(source,out):
    """Reencode a completed unit, checking every legacy record roundtrip."""
    started=time.monotonic();source=Path(source).resolve();out=Path(out).absolute()
    if out.exists() or out.resolve()!=out:raise ValueError('fresh canonical repack output required')
    raw=(source/'result.json').read_bytes();header=json.loads(raw)
    if (header['schema'],header['schema_version'])!=('scar.source-state-view',1):
        raise ValueError('repacking requires an original schema-1 state unit')
    original_pins=storage_pins();_durable_mkdir(out);packer=TrackPacker();count=0
    def rows():
        nonlocal count
        for wire in _table(source,header['tables']['tracks']):
            packed=packer.pack_wire(wire)
            if unpack_wire(packed,packer.controls)!=wire:raise ValueError('binding-track codec changed a legacy field')
            count+=1;yield packed
    tables=dict(tracks=write_table(out/'tracks.jsonl.gz',rows()),
        controls=write_table(out/'controls.jsonl.gz',packer.controls))
    for name,receipt in header['tables'].items():
        if name=='tracks':continue
        if receipt['path']!=name+'.jsonl.gz':raise ValueError('legacy state table path differs')
        p=source/receipt['path'];data=p.read_bytes()
        if len(data)!=receipt['bytes'] or sha256(data).hexdigest()!=receipt['sha256']:
            raise ValueError('legacy state table receipt changed')
        os.link(p,out/receipt['path']);tables[name]=receipt
    if count!=header['counts']['tracks'] or raw!=(source/'result.json').read_bytes() or storage_pins()!=original_pins:
        raise ValueError('state input, codec or track denominator changed while repacking')
    report=dict(header,schema_version=2,track_encoding=ENCODING,storage_pins=original_pins,tables=tables,
        repacked_from=dict(path=str(source/'result.json'),bytes=len(raw),sha256=sha256(raw).hexdigest()),
        all_legacy_track_fields_roundtripped=True,repacking_seconds=time.monotonic()-started)
    _durable_bytes(out/'result.json',_json(report),replace=False)
    return report


def _edges(state):
    for rows in state.out_edges.values():
        for edge in rows:
            if edge.source.namespace != state.namespace or edge.target.namespace != state.namespace:
                raise ValueError('binding edge left its owning source unit')
            conditions = tuple(V.StoredDependencyCondition(c.domain,c.name,
                None if c.subject is None else c.subject.node,c.index,tuple(p.node for p in c.operands)) for c in edge.conditions)
            yield S.StoredBindingEdge(edge.source.node,edge.target.node,edge.kind,edge.meaning,
                edge.strength,edge.evidence[0].rule,conditions)


def _rows(state):
    content=state.content()
    lineage=state.content_lineage.ensure_all()
    return dict(rules=(type(catalog()[0]),catalog()),
        requests=(S.SourceEffectRequest,state.requests.values()),
        locations=(S.BindingLocationTemplate,state.locations.values()),
        accesses=(S.BindingAccessTemplate,state.accesses),
        versions=(S.BindingVersionTemplate,state.versions.values()),
        selections=(S.BindingReadSelection,state.selections),
        tracks=(S.StoredBindingTrack,_tracks(state)),edges=(S.StoredBindingEdge,_edges(state)),
        content_addresses=(C.ContentAddressTemplate,content.addresses.values()),
        content_states=(C.ContentStateProjection,content.projections.values()),
        content_accesses=(C.ContentAccessTemplate,content.accesses),
        content_edges=(S.StoredBindingEdge,_edges(content)),
        content_lineage=(L.ContentValueLineage,lineage.records),
        content_lineage_edges=(S.StoredBindingEdge,_edges(lineage)))


def _counts(state):
    content=state.content()
    lineage=state.content_lineage.ensure_all()
    return dict(operations=len(state.requests),binding_accesses=len(state.accesses),locations=len(state.locations),
        versions=len(state.versions),read_selections=len(state.selections),tracks=len(state.tracks),
        relations=sum(len(rows) for rows in state.out_edges.values()),
        content_accesses=len(content.accesses),content_addresses=len(content.addresses),
        content_states=len(content.projections),content_relations=sum(len(rows) for rows in content.out_edges.values()),
        content_lineage_records=len(lineage.records),
        content_lineage_relations=sum(len(rows) for rows in lineage.out_edges.values()),
        missing_address_rules=dict(Counter(r.rule.instruction for r in state.requests.values() if r.missing_rules)))


def _stored_track(graph, limits):
    return S.StoredBindingTrack(graph.source_function, graph.namespace, **limits,
        definitions=graph.definitions, states=graph.states, transitions=graph.transitions,
        reads=graph.reads, frontiers=graph.frontiers, traversal_complete=graph.traversal_complete,
        dataflow_complete=graph.dataflow_complete)


def _tracks(state):
    for graph, limits in zip(state.tracks, state.track_limits, strict=True):
        yield _stored_track(graph, limits)


def _write_state_tables(out, records, *, packed=False):
    timings = {}
    if packed:
        packer = TrackPacker()
        tracks = records['tracks']
        records = {**records, 'tracks': (packer.pack(track) for track in tracks), 'controls': packer.controls}
    tables = {name: write_table(out / (name + '.jsonl.gz'), rows, timings=timings)
              for name, rows in records.items()}
    return tables, timings


def _write_state_view(state, out, tables, counts, *, started, pins, storage, timings=None,
                      parent=None, static_inventory=None):
    catalog = state.view.catalog
    if parent is None:
        parent = _artifact(catalog.path / 'result.json', catalog.report['schema'])
    report = dict(schema=SCHEMA, schema_version=1 if storage is None else 2,
        catalog=str(catalog.path), parent=encode(parent), unit=state.material.unit,
        namespace=encode(state.namespace), source_material=encode(state.material.result_artifact),
        binding_materials=encode(state.materials), pins=pins, tables=tables, counts=counts,
        original_body_bytes_copied=0, seconds=time.monotonic()-started,
        **dict.fromkeys(_NO_AUTHORITY, False))
    if storage is not None:
        report.update(storage_pins=storage, track_encoding=ENCODING, table_seconds=timings)
    if static_inventory is not None:
        report['static_inventory'] = static_inventory
    state.check()
    if pins != _pins() or storage is not None and storage != storage_pins():
        raise ValueError('state view rules or storage codec changed during publication')
    _durable_bytes(out / 'result.json', _json(report), replace=False)
    return report


def publish_state_view(state, out, *, packed=False):
    """Publish the typed rows as a plain or shared-control state view."""
    started = time.monotonic()
    pins, storage = _pins(), storage_pins() if packed else None
    out = Path(out).absolute()
    if out.exists() or packed and out.resolve() != out:
        raise ValueError('fresh state-view output required')
    state.check()
    _durable_mkdir(out)
    tables, timings = _write_state_tables(out, {name: rows for name, (_, rows) in _rows(state).items()}, packed=packed)
    return _write_state_view(state, out, tables, _counts(state),
                             started=started, pins=pins, storage=storage, timings=timings)


def _restore_request_order(state, path, tables):
    """Replay saved demand order; row references retain their original meaning.

    Operation IDs resolve only through the original sealed plans. This pass
    does not trust saved effects: all rows are decoded and compared with the
    independently rederived records below.
    """
    source=state.source
    source._admit(tuple(source.material.interfaces['nodes']))
    owners={source._identity(OperationID,key,step.id):(key,step.id)
        for key,plan in source.plans.items()for step in plan.steps}
    for wire in _table(path,tables['requests']):
        operation=decode(OperationID,wire['operation'])
        if operation not in owners:raise ValueError('saved effect operation is absent from original source plans')
        key,step=owners[operation]
        state.ensure_operation(source._node(key,step),include_content=False)
    content=state.content(complete=False)
    for wire in _table(path,tables['content_accesses']):
        operation=decode(OperationID,wire['operation'])
        if operation not in state.requests:raise ValueError('saved content access lacks its original effect request')
        content.ensure_operation(source.nodes[operation].point)


def _legacy_environment(report, pins):
    admitted=report['pins']
    facades={'producer','unified_adapter','state','implicit_addresses',
        'annotation_addresses','content_interfaces'}
    introduced={'environment_values','environment_value_types'}
    legacy_environment=introduced.isdisjoint(admitted)
    omitted=introduced if legacy_environment else set()
    if ({k:v for k,v in admitted.items() if k not in facades}
            != {k:v for k,v in pins.items() if k not in facades|omitted}):
        raise ValueError('state view schema or producer contract differs')
    for name in facades:
        if name not in admitted and name not in {'producer','unified_adapter','state'}:
            continue
        if (type(admitted.get(name)) is not str or len(admitted[name])!=64
                or any(c not in '0123456789abcdef' for c in admitted[name])):
            raise ValueError('state view facade provenance is not a SHA256')
    return legacy_environment


def _replay_tracks(state, path, tables, *, packed):
    controls = list(_table(path, tables['controls'])) if packed else ()
    packer = TrackPacker() if packed else None
    count = 0
    for wire in _table(path, tables['tracks']):
        if packed:
            index = wire['control']
            if type(index) is not int or not 0 <= index < len(controls):
                raise ValueError('packed binding control reference is outside its unit')
            control = controls[index]
            function = decode(F.GlobalPort, control['source_function'])
            namespace = decode(F.Namespace, wire['namespace'])
            limits = dict(max_states=control['max_states'], max_stack=control['max_stack'], max_updates=wire['max_updates'])
        else:
            track = decode(S.StoredBindingTrack, wire, record_tags=P.PRODUCTION_RECORD_TAGS)
            function, namespace = track.source_function, track.namespace
            limits = dict(max_states=track.max_states, max_stack=track.max_stack, max_updates=track.max_updates)
        replayed = _stored_track(state.track(function, namespace, **limits), limits)
        if (encode(packer.pack(replayed)) != wire if packed else replayed != track):
            raise ValueError('state view track differs from its source fixed point')
        count += 1
    if count != len(state.tracks) or (packed and controls != [encode(row) for row in packer.controls]):
        raise ValueError('state view track order, multiplicity or shared control differs')


def read_state_view(path):
    """Rederive saved state from original F4 rules in either wire version."""
    path = Path(path).resolve()
    raw = (path / 'result.json').read_bytes()
    report = json.loads(raw)
    pins = _pins()
    version = report['schema_version']
    if report['schema'] != SCHEMA or version not in (1, 2):
        raise ValueError('unsupported state view wire')
    packed = version == 2
    storage = storage_pins() if packed else None
    if packed and (report['track_encoding'] != ENCODING
            or not compatible_storage_pins(report['storage_pins'], storage)
            or not compatible_state_pins(report['pins'], pins)):
        raise ValueError('state/storage replay contract differs')
    legacy_environment = not packed and _legacy_environment(report, pins)
    if report['original_body_bytes_copied'] != 0 or any(report[key] is not False for key in _NO_AUTHORITY):
        raise ValueError('state view claims results outside its static binding scope')
    parent = _artifact(Path(report['catalog']) / 'result.json', report['parent']['schema'])
    if encode(parent) != report['parent']:
        raise ValueError('state view catalog receipt differs')
    view = ProgramView(report['catalog'])
    state = view.state(report['unit'])
    if legacy_environment and state.environment_values:
        raise ValueError('legacy state view needs republication with retained environment projections')
    if (encode(state.namespace) != report['namespace']
            or encode(state.material.result_artifact) != report['source_material']
            or encode(state.materials) != report['binding_materials']):
        raise ValueError('state view source address material differs')
    tables = report['tables']
    _restore_request_order(state, path, tables)
    _replay_tracks(state, path, tables, packed=packed)
    rows = _rows(state)
    if set(tables) != (set(rows) | {'controls'} if packed else set(rows)):
        raise ValueError('state view table domain differs')
    sentinel = object()
    for name, (cls, expected) in rows.items():
        if name == 'tracks':
            continue
        actual = (decode(cls, row, record_tags=P.PRODUCTION_RECORD_TAGS) for row in _table(path, tables[name]))
        if name in ('locations', 'versions', 'edges'):
            remaining = set(expected)
            for record in actual:
                if record not in remaining:
                    raise ValueError('state view source declaration or relation differs: ' + name)
                remaining.remove(record)
            if remaining:
                raise ValueError('state view source declaration or relation is missing: ' + name)
        elif any(a != b for a, b in zip_longest(actual, expected, fillvalue=sentinel)):
            raise ValueError('state view source rederivation differs: ' + name)
    if _counts(state) != report['counts']:
        raise ValueError('state view coverage counts differ')
    state.check()
    if (pins != _pins() or storage is not None and storage != storage_pins()
            or (path / 'result.json').read_bytes() != raw):
        raise ValueError('state view contract changed during read')
    return report, view, state


class BindingInventory:
    """Consume one admitted static shard while leaving the source recipes authoritative."""

    def __init__(self, state, inventory, index_sha256):
        from scar.analysis.program_view_shard_reader import open_source_graph_shard

        self.state = state
        self.shard = shard = open_source_graph_shard(inventory, state.material.unit, index_sha256=index_sha256)
        header_path = Path(shard.unit_header.uri)
        raw = header_path.read_bytes()
        if sha256(raw).hexdigest() != shard.unit_header.sha256:
            raise ValueError('static binding shard changed after header admission')
        self.folder = header_path.parent / 'state'
        header = loads(raw)
        self.tables = tables = header['state_tables']
        self.counts = counts = header['counts']
        catalog = state.view.catalog
        if (shard.catalog != _artifact(catalog.path / 'result.json', catalog.report['schema'])
                or shard.source_material != state.material.result_artifact or shard.namespace != state.namespace
                or shard.namespace.graph != state.namespace.graph
                or set(tables) != {*_STATIC_TABLES, 'locations', 'edges', 'versions', 'selections', 'tracks'}
                or any(tables[n]['records'] != 0 for n in ('versions', 'selections', 'tracks'))
                or any(counts[n] != 0 for n in ('versions', 'read_selections', 'tracks'))
                or any(row['path'] != name + '.jsonl.gz' for name, row in tables.items())):
            raise ValueError('static binding shard uses a different graph or analysis contract')
        self.tags = _record_tags(P.PRODUCTION_RECORD_TAGS)

    def _records(self, name, cls):
        self._path(name)
        return (_decode(cls, row, self.tags) for row in _table(self.folder, self.tables[name]))

    def _path(self, name):
        path = self.folder / self.tables[name]['path']
        if path.resolve(strict=True) != path:
            raise ValueError('stored binding table must retain its original path: ' + name)
        return path

    def tracks(self, functions):
        """Load the static binding index and select the existing solver's inputs."""
        from scar.analysis.source_state import local_read_tracks

        state = self.state
        if state.requests or state.accesses or state.locations or state.tracks or state.content_provider is not None:
            raise ValueError('static binding input requires a fresh source state provider')
        accesses = list(self._records('accesses', S.BindingAccessTemplate))
        locations = {row.id: row for row in self._records('locations', S.BindingLocationTemplate)}
        requests = {row.operation: row for row in self._records('requests', S.SourceEffectRequest)}
        counts = self.counts
        missing = Counter(row.rule.instruction for row in requests.values() if row.missing_rules)
        if ((len(accesses), len(locations), len(requests)) !=
                (counts['binding_accesses'], counts['locations'], counts['operations'])
                or any(len(rows) != self.tables[name]['records'] for name, rows in
                       (('accesses', accesses), ('locations', locations), ('requests', requests)))
                or dict(missing) != counts['missing_address_rules']
                or any(row.unit != state.material.unit for row in locations.values())
                or sorted(i for row in requests.values() for i in row.accesses) != list(range(len(accesses)))
                or any(access.operation not in requests or i not in requests[access.operation].accesses
                    or any(alt.location not in locations for alt in access.alternatives)
                    for i, access in enumerate(accesses))):
            raise ValueError('stored binding accesses, requests and locations do not close')
        state.locations.update(locations)
        state.accesses.extend(accesses)
        state.requests.update(requests)
        state.analyzed_operations.update(requests)
        state.bindings_complete = True
        state.check()
        selected = frozenset(functions)
        return tuple(pair for pair in local_read_tracks(state) if pair[0] in selected)

    def publish(self, out):
        """Publish the existing state-view wire with unchanged static tables."""
        started = time.monotonic()
        pins, storage = _pins(), storage_pins()
        state = self.state
        if not state.bindings_complete or state.content_provider is not None:
            raise ValueError('static binding output requires the completed binding-only overlay')
        _durable_mkdir(out)
        tables = {}
        for name in _STATIC_TABLES:
            receipt = self.tables[name]
            data = self._path(name).read_bytes()
            if (len(data), sha256(data).hexdigest()) != (receipt['bytes'], receipt['sha256']):
                raise ValueError('static source state table changed: ' + name)
            _durable_bytes(out / receipt['path'], data, replace=False)
            tables[name] = receipt
        self._path('edges')
        updates = dict(locations=state.locations.values(), versions=state.versions.values(),
            selections=state.selections, tracks=_tracks(state),
            edges=chain(_table(self.folder, self.tables['edges']), _edges(state)))
        published, timings = _write_state_tables(out, updates, packed=True)
        tables.update(published)
        counts = {name: self.counts[name] for name in _STATIC_COUNTS}
        counts.update(locations=len(state.locations), versions=len(state.versions),
            tracks=len(state.tracks), read_selections=len(state.selections), relations=tables['edges']['records'])
        if (len(state.requests), len(state.accesses), tables['edges']['records']) != (
                counts['operations'], counts['binding_accesses'],
                self.counts['relations'] + sum(len(rows) for rows in state.out_edges.values())):
            raise ValueError('binding solve changed its static input domain')
        return _write_state_view(state, out, tables, counts, started=started, pins=pins, storage=storage,
            parent=self.shard.catalog, timings=timings,
            static_inventory=dict(index=encode(self.shard.index), unit=encode(self.shard.unit_header),
                reused_tables=list(_STATIC_TABLES), binding_solver_replayed=True))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog');parser.add_argument('--unit');parser.add_argument('--out')
    parser.add_argument('--read')
    parser.add_argument('--packed', action='store_true', help='publish shared-control state tables')
    parser.add_argument('--repack', help='losslessly transform an existing v1 view to the packed format')
    parser.add_argument('--slot',action='append',type=int,default=[],help='original C2 local-binding slot; owner function is resolved mechanically')
    parser.add_argument('--interfaces-only',action='store_true',help='publish addresses without solving local binding tracks')
    args=parser.parse_args(argv)
    if args.repack:
        if not args.out or any((args.read, args.catalog, args.unit, args.slot, args.interfaces_only, args.packed)):
            parser.error('--repack requires only --out')
        report = repack(args.repack, args.out)
    elif args.read:
        if any((args.catalog,args.unit,args.out,args.slot,args.interfaces_only,args.packed)):parser.error('--read stands alone')
        report,_,_=read_state_view(args.read)
    else:
        if not all((args.catalog,args.unit,args.out)):parser.error('--catalog --unit --out are required')
        from scar.analysis.source_references import source_port
        from scar.ir.program import normal_graph as F
        state=ProgramView(args.catalog).state(args.unit)
        if args.interfaces_only and args.slot:parser.error('--interfaces-only cannot select slots')
        if not args.slot and not args.interfaces_only:
            from scar.analysis.source_state import track_local_reads
            track_local_reads(state)
        for index in dict.fromkeys(args.slot):
            if not 0<=index<len(state.bindings.bindings):parser.error('slot is outside original C2 binding domain')
            slot=state.bindings.bindings[index]
            functions=[f for f in state.source.facts.functions if f.environment.scope==slot.owner]
            if slot.domain!='LOCAL_BINDING' or len(functions)!=1:parser.error('slot needs one owning source function')
            state.track(source_port(state.source.facts,functions[0].value),F.Namespace(F.NamespaceKind.LEXICAL,index,state._name(slot.name)))
        report=publish_state_view(state,args.out,packed=args.packed)
    print(json.dumps(dict(unit=report['unit'],counts=report['counts'],first_stage_complete=False)))


if __name__ == '__main__':
    main()
