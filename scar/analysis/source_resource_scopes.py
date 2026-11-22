"""Connect owned boundary markers, original CALLs, OS work and source ports.

Only per-thread syscall order and explicit marker offsets are used. Markers
are collector work. Enclosure requires a balanced observed native call or
ordinary Python activation. Enclosure is not exclusive program origin.
"""
from collections import Counter,defaultdict
from bisect import bisect_left
from hashlib import sha256
import argparse
import inspect
import json
from pathlib import Path
import time

from scar.trace.control_transfers import bind_capture_coordinates
from scar.trace.resource_scopes import read_resource_scopes,RECORD
from scar.trace.os_resources import _scan,_number,_interface_family,_WRITE
from scar.analysis.call_operand_reader import read_call_operands
from scar.analysis.python_call_targets import read_python_call_targets
from scar.analysis.source_body_activations import _SourceAnchors
from scar.analysis.source_body_interfaces import BodyInterfaceResolver
from scar.analysis.source_observed_codes import ObservedSourceCodeProvider
from scar.analysis.source_observed_calls import ObservedSourceCallsiteIndex
from scar.analysis.program_view import ProgramView,_digest
from scar.artifacts import _json
from scar.artifacts import _durable_bytes, _durable_mkdir
from scar.analysis.source_consumer_parents import _stamp
from scar.artifacts import read_record_table as _table
from scar.artifacts import write_table
from scar.ir.program import resource_observations as R
from scar.ir.program import source_productions as P
from scar.ir.program.call_edges import FrameCompletion
from scar.ir.program.ids import ProviderID
from scar.ir.program.providers import ProviderArtifact,ProviderFactReference
from scar.ir.record_codec import encode,decode


SCHEMA='scar.source-resource-scopes'
_RESUMABLE=inspect.CO_GENERATOR|inspect.CO_COROUTINE|inspect.CO_ASYNC_GENERATOR|inspect.CO_ITERABLE_COROUTINE


def _receipt(path):
    path=Path(path);raw=path.read_bytes()
    return dict(path=str(path),sha256=sha256(raw).hexdigest(),bytes=len(raw))


def _pins():
    from scar.trace import resource_scopes, os_resources, control_transfers as capture_coordinates
    from scar.analysis import python_call_targets, source_body_activations as source_body_activations, source_observed_codes, source_observed_calls as source_observed_calls
    return {**source_observed_codes.request_code_pins(),**{name:sha256(Path(module.__file__).read_bytes()).hexdigest() for name,module in (
        ('producer',__import__(__name__,fromlist=['__file__'])),('types',R),
        ('markers',resource_scopes),('os_reader',os_resources),('coordinates',capture_coordinates),
        ('source_anchors',source_body_activations),('call_index',source_observed_calls),
        ('python_targets',python_call_targets))}}


def _physical_members(coordinates,artifacts):
    """Use exact retained logical/physical aliases, never hash-only PID guesses."""
    result={}
    for index,artifact in enumerate(artifacts):
        rows=[row for row in coordinates.report['membership']
            if row['local_path']==artifact.uri and
            (row['artifact']['sha256'],row['artifact']['size_bytes'],row['artifact']['schema'])==
            (artifact.sha256,artifact.size_bytes,artifact.schema)]
        if len(rows)==1:result[index]=decode(ProviderArtifact,rows[0]['artifact'])
    return result


def _python_activations(trace,coordinates,membership):
    targets=coordinates.report.get('python_call_target_membership')
    if (targets is None or targets['manifest']!=membership.get('python_target_manifest')
            or targets['events']!=membership.get('python_target_events')):
        raise ValueError('Python resource markers have no exact bound callee handoff')
    declaration=targets['manifest']
    expected=dict(path=declaration['uri'],sha256=declaration['sha256'],bytes=declaration['size_bytes'])
    if _receipt(declaration['uri'])!=expected:
        raise ValueError('Python target handoff differs from its coordinate membership')
    observed=read_python_call_targets(trace)
    if _receipt(declaration['uri'])!=expected:
        raise ValueError('Python target handoff changed while reading its entries')
    bindings={(entry.caller,entry.attempt_index):(entry,anchor)
        for entry,anchor in zip(observed.entries,observed.entry_anchors,strict=True)}
    entries={entry.invocation:entry for entry in trace.entries}
    exits=defaultdict(list)
    for exit_ in trace.exits:exits[exit_.invocation].append(exit_)
    codes={(capture.process_id,capture.code.key):capture for capture in trace.frame_scan.captures}
    return bindings,entries,exits,codes


def _python_scope(fields,attempt,activation,claimed):
    bindings,entries,exits,codes=activation
    begin=fields[1];binding=bindings.get((begin.caller,begin.attempt_index))
    if binding is None:raise ValueError('Python resource scope has no original actual callee entry')
    entry,anchor=binding
    frame=entries[entry.callee];ends=exits[entry.callee]
    code=codes.get((entry.callee.process_id,entry.code))
    if (entry.callee in claimed or code is None or code.code.flags&_RESUMABLE or frame.suspended
            or frame.anchor.index!=entry.entry_index or attempt.frame_index>=entry.entry_index
            or len(ends)!=1 or ends[0].code!=entry.code or ends[0].return_site.code!=entry.code
            or ends[0].completion is FrameCompletion.SUSPENDED
            or ends[0].anchor.index is None or ends[0].anchor.index<=entry.entry_index):
        raise ValueError('Python resource scope has no unique ordinary frame lifetime')
    claimed.add(entry.callee)
    return R.ObservedPythonResourceScope(*fields,entry.callee,code.anchor,anchor,frame.anchor,ends[0].anchor)


def _enclosing_scopes(calls,boundaries):
    """Sweep exact per-TID completion lines and select the deepest full enclosure."""
    states={}
    for tid,rows in boundaries.items():
        states[tid]=(iter(sorted(rows)),[],[])
    next_boundary={tid:next(state[0]) for tid,state in states.items()}
    for call in calls:
        tid=call['tid'];state=states.get(tid)
        if state is None:yield None;continue
        following,openings,identities=state
        boundary=next_boundary[tid];last=call['line'];first=call['first_line']
        while boundary is not None and boundary[0]<=last:
            line,opened,identity=boundary
            if opened:
                openings.append(line);identities.append(identity)
            else:
                if identities.pop()!=identity:raise ValueError('balanced scope ranges are not nested')
                openings.pop()
            boundary=next(following,None)
        next_boundary[tid]=boundary
        # A resumed call may have started before a currently enclosing inner
        # frame. Select only an interval that brackets both original lines.
        depth=len(openings)-1 if openings and openings[-1]<first else bisect_left(openings,first)-1
        yield identities[depth] if depth>=0 else None


def derive_resource_scopes(coordinates,capture,trace,*,include_positive_writes=False):
    membership=coordinates.report.get('resource_scope_membership')
    if membership is None:raise ValueError('capture has no bound resource scope marker channel')
    declaration=membership['manifest'];metadata=Path(declaration['uri'])
    markers=read_resource_scopes(metadata,expected_sha256=declaration['sha256'])
    if markers.metadata['process_id']!=coordinates.target_process.pid:
        raise ValueError('resource marker process differs from its bound capture')
    if encode(markers.artifact)!=membership['events']:
        raise ValueError('resource markers differ from the coordinate membership')
    python=(_python_activations(trace,coordinates,membership) if markers.metadata['schema_version']==2 else None)
    attempts={(row.invocation,row.index):row for row in trace.attempts}
    files,artifacts,lines,calls=_scan(Path(capture)/'os',128_000_000,2_000_000,interface_work=True)
    members=_physical_members(coordinates,artifacts)
    events={event.sequence:event for event in markers.events};os_markers=defaultdict(list)
    candidates=[];unassigned=[];frontiers=[]
    def fact(call,entry=False):
        artifact=members.get(call['file'],artifacts[call['file']])
        return ProviderFactReference(artifact,'#/lines/'+str(call['first_line'] if entry else call['line']))
    for call in calls:
        if not call.get('interface_record'):continue
        artifact=members.get(call['file']);process=(coordinates.process_for(artifact,call['tid']) if artifact else None)
        if process is None or process.id!=coordinates.target_process.id:
            unassigned.append((fact(call),'OUTSIDE_VERIFIED_TARGET_PROCESS'));continue
        a=call['args'];is_marker=False
        if call['name']=='pwrite64' and len(a)==4 and _number(a[0])==markers.metadata['marker_fd']:
            offset=_number(a[3]);sequence=(offset//RECORD.size+1 if offset is not None and offset>=0 and offset%RECORD.size==0 else None)
            event=events.get(sequence)
            if (event is not None and event.native_tid==call['tid']
                    and _number(a[2])==RECORD.size and call['result']==RECORD.size):
                os_markers[sequence].append(call);is_marker=True
        if not is_marker:candidates.append(call)
    if set(os_markers)!=set(events) or any(len(v)!=1 for v in os_markers.values()):
        raise ValueError('marker offsets have no unique exact native-TID pwrite receipts')
    by_tid=defaultdict(list)
    for event in markers.events:by_tid[event.native_tid].append(event)
    scopes=[];boundaries=defaultdict(list);scope_calls={};relations={};claimed=set();claimed_attempts=set()
    for tid,rows in by_tid.items():
        stack=[];previous=0
        for event in rows:
            call=os_markers[event.sequence][0]
            if call['first_line']<=previous:raise ValueError('marker file and per-thread syscall orders disagree')
            previous=call['line']
            if event.kind is R.ScopeEventKind.BEGIN:
                attempt_key=(event.caller,event.attempt_index)
                if attempt_key not in attempts or attempt_key in claimed_attempts:
                    raise ValueError('resource scope has no distinct original CALL attempt')
                claimed_attempts.add(attempt_key);stack.append(event);continue
            if not stack or stack[-1].scope!=event.scope:
                raise ValueError('resource marker completions are not properly nested')
            begin=stack.pop();attempt=attempts[begin.caller,begin.attempt_index]
            opening=os_markers[begin.sequence][0]
            identity=ProviderID(_digest((SCHEMA,coordinates.run.id,begin,event)))
            fields=(identity,begin,event,attempt.anchor,fact(opening),fact(call,True))
            if event.kind is R.ScopeEventKind.PY_EXIT:
                if python is None:raise ValueError('Python resource scope requires V2 marker membership')
                row=_python_scope(fields,attempt,python,claimed)
                relations[identity]='ENCLOSED_BY_OBSERVED_PYTHON_ACTIVATION'
            else:
                if python is not None and (begin.caller,begin.attempt_index) in python[0]:
                    raise ValueError('native resource completion contradicts an observed Python activation')
                row=R.ObservedResourceScope(*fields)
                relations[identity]='ENCLOSED_BY_OBSERVED_NATIVE_CALL'
            scopes.append(row);scope_calls[identity]=attempt
            boundaries[tid].extend(((opening['line'],True,identity),(call['first_line'],False,identity)))
        frontiers.extend(dict(reason=('UNCLOSED_RESOURCE_SCOPE' if python is not None else 'UNCLOSED_NATIVE_SCOPE'),
            begin=encode(event.source)) for event in stack)
    if (len(scopes),len(frontiers))!=(markers.metadata['completed'],markers.metadata['unclosed']):
        raise ValueError('resource scope boundary totals differ from marker handoff')
    work=[];positive_write_fds={}
    for call,selected in zip(candidates,_enclosing_scopes(candidates,boundaries),strict=True):
        source=fact(call)
        if include_positive_writes and call['name'] in _WRITE and (call['result'] or 0)>0:
            positive_write_fds[source]=_number(call['args'][0]) if call['args'] else None
        if selected is None:
            unassigned.append((source,'NO_BALANCED_ENCLOSING_SCOPE'));continue
        work.append(R.ScopedResourceWork(selected,source,call['name'],
            _interface_family(call['name']),call['result'],relations[selected]))
    derived=dict(scopes=tuple(sorted(scopes,key=lambda row:row.id.value)),work=tuple(work),
        markers=tuple((events[i].source,fact(os_markers[i][0])) for i in sorted(events)),
        unassigned=tuple(unassigned),frontiers=tuple(frontiers))
    if include_positive_writes:derived['positive_write_fds']=positive_write_fds
    return derived,scope_calls,markers


def _source_bindings(index,rows,attempts):
    resolver=BodyInterfaceResolver(index.view)
    for unit,contexts in index._contexts.items():
        resolver.plans.update(((unit,key),context) for key,context in contexts.items())
    anchors=_SourceAnchors(index,resolver);bindings=[]
    for scope in rows['scopes']:
        for anchor in anchors.select(scope.begin.caller.process_id,attempts[scope.id].site):
            bindings.append(R.SourceResourceScopeBinding(scope.id,anchor))
    return tuple(bindings)


def build_resource_scopes(capture,out,*,catalog=None,project_root=None,graph_check_out=None,index=None):
    started=time.monotonic();capture=Path(capture).resolve();out=Path(out).absolute()
    if out.exists() or out.resolve()!=out:raise ValueError('fresh canonical resource scope output required')
    pins=_pins();coordinate_path=capture/'capture-coordinates.json'
    coordinates=bind_capture_coordinates(capture,max_bytes=512_000_000,max_lines=2_000_000,profile_uri=str(coordinate_path))
    if coordinate_path.read_bytes()!=coordinates.payload:raise ValueError('stored capture coordinates differ')
    trace_path=capture/'trace'
    if index is not None:
        if (Path(index.codes.trace.directory).resolve()!=trace_path or
                catalog is not None and Path(catalog).resolve()!=index.view.catalog.path):
            raise ValueError('resource builder index belongs to another original source/capture')
        view=index.view;catalog=view.catalog.path;trace=index.codes.trace
    elif catalog is not None:
        view=ProgramView(catalog);codes=ObservedSourceCodeProvider(view,trace_path,project_root)
        index=ObservedSourceCallsiteIndex(codes);trace=codes.trace
    else:
        index=None;trace=read_call_operands(trace_path,project_root)
    rows,attempts,markers=derive_resource_scopes(coordinates,capture,trace)
    bindings=();source_frontiers=[]
    if index is not None:
        bindings=_source_bindings(index,rows,attempts)
        bound={row.scope for row in bindings}
        source_frontiers=[dict(scope=encode(scope.id),reason='SOURCE_OPERATION_NOT_CORRESPONDED')
            for scope in rows['scopes'] if scope.id not in bound]
        index.check()
    _durable_mkdir(out);tables={}
    tables['scopes']=write_table(out/'scopes.jsonl.gz',rows['scopes'])
    tables['work']=write_table(out/'work.jsonl.gz',rows['work'])
    tables['markers']=write_table(out/'markers.jsonl.gz',rows['markers'])
    tables['unassigned']=write_table(out/'unassigned.jsonl.gz',rows['unassigned'])
    tables['frontiers']=write_table(out/'frontiers.jsonl.gz',(*rows['frontiers'],*source_frontiers))
    shards={}
    for unit in sorted({row.anchor.operation.unit for row in bindings}):
        folder=out/'units'/unit;_durable_mkdir(folder)
        table=write_table(folder/'bindings.jsonl.gz',(row for row in bindings if row.anchor.operation.unit==unit))
        shards[unit]=dict(path='units/'+unit,bindings=table)
    summary=dict(scopes=len(rows['scopes']),enclosed_work=len(rows['work']),
        excluded_marker_writes=len(rows['markers']),unassigned_work=len(rows['unassigned']),
        source_bindings=len(bindings),source_bound_scopes=len({r.scope for r in bindings}),
        source_frontiers=len(source_frontiers),unclosed_scopes=len(rows['frontiers']),
        syscalls=dict(Counter(row.syscall for row in rows['work'])))
    version=markers.metadata['schema_version']
    if version==2:
        summary['python_activation_scopes']=sum(isinstance(row,R.ObservedPythonResourceScope) for row in rows['scopes'])
    report=dict(schema=SCHEMA,schema_version=version,capture=str(capture),catalog=None if catalog is None else str(Path(catalog).resolve()),
        project_root=project_root,coordinates=encode(coordinates.artifact),pins=pins,tables=tables,shards=shards,summary=summary,
        scope_ownership_condition=markers.metadata['ownership_condition'],
        non_marker_work_may_include_nested_callbacks_and_collector_overhead=True,
        direct_program_origin_proven=False,effects_closed=False,readiness_established=False,
        clock_matching_used=False,source_bodies_copied=False,first_stage_complete=False,seconds=time.monotonic()-started)
    if pins!=_pins():raise ValueError('resource scope implementation changed during build')
    _durable_bytes(out/'result.json',_json(report),replace=False)
    if graph_check_out is not None:
        from scar.analysis.source_resource_scope_view import verify_resource_scope_graph
        verify_resource_scope_graph(out,graph_check_out,index=index)
    return report


class ResourceScopeStore:
    """Read stored typed scopes against original OS, CALL and source inputs."""
    def __init__(self,path,*,index=None):
        self.path=Path(path).resolve();self.raw=(self.path/'result.json').read_bytes();r=json.loads(self.raw);self.report=r
        if (r['schema']!=SCHEMA or type(r['schema_version']) is not int or r['schema_version'] not in (1,2)
                or r['pins']!=_pins()):
            raise ValueError('resource scope schema or implementation differs')
        if (any(r[name]is not False for name in ('direct_program_origin_proven','effects_closed',
                'readiness_established','clock_matching_used','source_bodies_copied','first_stage_complete'))
                or r['non_marker_work_may_include_nested_callbacks_and_collector_overhead']is not True):
            raise ValueError('resource scope publication claims unproved effects or ownership')
        self.coordinates=bind_capture_coordinates(r['capture'],max_bytes=512_000_000,max_lines=2_000_000,
            profile_uri=r['coordinates']['uri'])
        if encode(self.coordinates.artifact)!=r['coordinates']:raise ValueError('resource scope capture changed')
        if index is None and r['catalog'] is not None:
            view=ProgramView(r['catalog']);codes=ObservedSourceCodeProvider(view,Path(r['capture'])/'trace',r['project_root'])
            index=ObservedSourceCallsiteIndex(codes)
        self.index=index
        trace=(index.codes.trace if index is not None else read_call_operands(Path(r['capture'])/'trace',r['project_root']))
        if (Path(trace.directory).resolve()!=Path(r['capture'])/'trace'
                or index is not None and str(index.view.catalog.path)!=r['catalog']):
            raise ValueError('resource scope index uses another source catalog or capture')
        expected,attempts,markers=derive_resource_scopes(self.coordinates,r['capture'],trace)
        if r['schema_version']!=markers.metadata['schema_version']:
            raise ValueError('stored resource scope output changes the original marker schema')
        expected_bindings=() if index is None else _source_bindings(index,expected,attempts)
        scope_kind=R.ObservedResourceScope if r['schema_version']==1 else (
            R.ObservedPythonResourceScope|R.ObservedResourceScope)
        kinds=dict(scopes=scope_kind,work=R.ScopedResourceWork,
            markers=tuple[ProviderFactReference,ProviderFactReference],
            unassigned=tuple[ProviderFactReference,str],frontiers=None)
        self.rows={};self.guards={}
        for name,kind in kinds.items():
            receipt=r['tables'][name];p=self.path/receipt['path'];before=_stamp(p)
            values=tuple(decode(kind,row,record_tags=P.PRODUCTION_RECORD_TAGS) if kind else row for row in _table(self.path,receipt))
            if _stamp(p)!=before:raise ValueError('resource scope table changed while reading')
            self.guards[p]=before;self.rows[name]=values
            if name!='frontiers' and values!=expected[name]:raise ValueError('stored resource scopes differ from original marker/CALL/OS evidence')
        bound={row.scope for row in expected_bindings}
        expected_frontiers=(*expected['frontiers'],*(dict(scope=encode(scope.id),reason='SOURCE_OPERATION_NOT_CORRESPONDED')
            for scope in expected['scopes'] if index is not None and scope.id not in bound))
        if self.rows['frontiers']!=tuple(expected_frontiers):raise ValueError('resource scope frontier denominator differs')
        loaded=[]
        for unit,s in r['shards'].items():
            if s['path']!='units/'+unit:raise ValueError('resource scope unit shard differs')
            folder=self.path/s['path'];p=folder/s['bindings']['path'];before=_stamp(p)
            rows=tuple(decode(R.SourceResourceScopeBinding,row,record_tags=P.PRODUCTION_RECORD_TAGS) for row in _table(folder,s['bindings']))
            if any(row.anchor.operation.unit!=unit for row in rows) or _stamp(p)!=before:
                raise ValueError('resource scope binding unit changed')
            loaded.extend(rows);self.guards[p]=before
        order=lambda row:_digest(row)
        if tuple(sorted(loaded,key=order))!=tuple(sorted(expected_bindings,key=order)):
            raise ValueError('stored resource source ports differ from original source/CALL correspondence')
        self.bindings=tuple(loaded);self.guards[self.path/'result.json']=_stamp(self.path/'result.json')
        self.check()

    def check(self):
        if any(_stamp(path)!=stamp for path,stamp in self.guards.items()) or self.raw!=(self.path/'result.json').read_bytes():
            raise ValueError('admitted resource scope publication changed')
        if self.index is not None:self.index.codes._check_trace()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capture',required=True)
    p.add_argument('--catalog');p.add_argument('--project-root',required=True);p.add_argument('--out',required=True)
    p.add_argument('--graph-check-out')
    a=p.parse_args();r=build_resource_scopes(a.capture,a.out,catalog=a.catalog,project_root=a.project_root,
        graph_check_out=a.graph_check_out)
    print(json.dumps(r['summary'],indent=2))


if __name__=='__main__':main()
