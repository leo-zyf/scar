"""Derive guarded owner-scoped residency for a loop's native plot inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import re

from scar.analysis.source_controller_plot import SCHEMA as PLOT, check as check_plot
from scar.codegen.native_controller_plot import ASSETS, HEADER
from scar.codegen.native_live_plot import SCHEMA as PROVIDER
from scar.codegen.native_stateful_entry import SCHEMA as ENTRY
from scar.ir.execution_recipe import artifact_reference, read_artifact


SCHEMA = 'scar.source-controller-plot-residency'
OBSERVATION = 'scar.native-plot-asset-loop-observation'
_OPEN = re.compile(r'openat\(AT_FDCWD, ("(?:[^"\\]|\\.)*"), O_RDONLY\) = (\d+)')


def _require(ok, detail):
    if not ok:
        raise ValueError('plot residency: ' + detail)


def _one(rows, detail):
    rows = tuple(rows)
    _require(len(rows)==1, detail)
    return rows[0]


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True),schema)


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _member(reference, declaration, name):
    path = Path(name)
    _require(bool(name) and not path.is_absolute() and '..' not in path.parts and declaration['uri']==name,
             'expected a package-relative member')
    target = Path(reference['uri']).parent / path
    _require(target.resolve(strict=True)==target and target.is_file(),
             'package member is not a canonical local file')
    material = dict(**declaration)
    material['uri'] = str(target)
    return material,read_artifact(material,json_document=False)


def _original_loop(source, startup, baseline):
    controller_ref = startup['controller_source']
    controller = read_artifact(controller_ref)
    site = source['effect']['site']
    selected = source['effect']['controller_call']
    _require(selected['site']==site and selected['role']=='ORIGINAL_EFFECT'
             and source['effect']['ordered_subcall_attempts'],
             'needs one original ordered plot effect')
    hits = []
    def visit(rows, loops=()):
        for current in rows:
            if current['kind']=='CALL' and current.get('site')==site:
                hits.append((current,loops))
            for field in ('condition','body','then','otherwise'):
                enclosing = (*loops,current['header']) if current['kind']=='WHILE' and field=='body' else loops
                visit(current.get(field,()),enclosing)
    visit(controller['program'])
    matched,loops = _one(hits,'plot CALL needs one exact original controller occurrence')
    _require(matched==selected and len(loops)==1 and loops[0]==controller['original_loop_header'],
             'original plot effect is outside the admitted original loop')
    native = baseline['controller']['plot']['binding']
    effect = _one((r for r in baseline['controller']['effects'] if r['site']==site),
                  'generated plot effect lacks an exact original controller route')
    _require(native['plot_ordinal']==effect['ordinal'] and effect['source_fact']==selected['source_fact']
             and native['time_argument']==source['title']['controller_scalar']['argument_ordinal'],
             'generated native plot order or current time binding differs')
    return dict(controller=controller_ref,loop_header=loops[0],effect_site=site,
                original_source_fact=effect['source_fact'],callback_ordinal=effect['ordinal'])


def _scoped_axes(source, startup, baseline, baseline_ref, provider):
    native = baseline['controller']
    image_name = native['startup']['image']
    image_ref,image = _member(baseline_ref,baseline['files'][image_name],image_name)
    _require(image_ref['sha256']==native['startup']['image_sha256']
             and len(image)==native['startup']['image_bytes'],
             'startup image binding differs from its native bytes')
    coordinates = native['plot']['binding']['read_only_roots']
    axes = source['arrays']
    _require(set(coordinates)=={'x','y'} and set(axes)=={'x','y','z'}
             and provider['profile']['roots'].keys()==axes.keys(),
             'original plot ABI needs exactly two immutable coordinates and one live field')
    view_keys = ('byte_strides','root_shape','shape','start')
    for role,route in axes.items():
        actual = provider['profile']['roots'][role]
        _require(all(actual[key]==route[key] for key in view_keys)
                 and actual['field_path']==route['controller_path'],
                 'original plot and provider geometry differ at '+role)
    numeric_inputs = {tuple(row['source_path']) for row in baseline['ports'] if row['kind']=='array'}
    outputs = startup['materialization']['outputs']
    slices = read_artifact(startup['source_slice'])['ports']
    placements, cursor = {},0
    for item in outputs:
        if item['kind']=='ARRAY':
            length = item['expected']['size_bytes']
            placements[item['id']] = (cursor,length,item)
            cursor += length
    _require(cursor==len(image),'startup layout does not account for the image extent')
    roots, packed = {},{}
    for role,coordinate in coordinates.items():
        route = axes[role]
        identity = route['id']
        offset,length,output = placements[identity]
        geometry = output['geometry']
        producer = _one((r for r in slices if r['id']==identity),'one original coordinate startup slice required')
        proof = _one((r for r in startup['decisions'] if r['port']==identity),'one coordinate startup decision required')
        _require(route['kind']=='STARTUP_ASSET' and coordinate in native['startup']['auxiliary_arrays']
                 and coordinate['id']==identity and coordinate['bound_to_controller'] is False
                 and coordinate['path']==route['controller_path']==output['path']==producer['controller_path']
                 and tuple(coordinate['path']) not in numeric_inputs
                 and coordinate['offset']==offset and coordinate['size_bytes']==length
                 and coordinate['shape']==route['root_shape']==geometry['shape']
                 and proof['action']=='BUILD_EXECUTE_SOURCE_AND_VERIFY_EXACT_OUTPUT'
                 and proof['source']==route['startup_material']==output['source']
                 and not any(producer[key] for key in ('supplied_inputs','open_value_sources','control_guards')),
                 'coordinates must come from a private qualified nonnumeric startup asset')
        full = memoryview(image)[offset:offset+length]
        height,width = coordinate['shape']
        rows,cols = route['shape']
        start_row,start_col = route['start']
        _require(geometry['dtype']=='<f8' and geometry['itemsize']==8 and geometry['byte_offset']==0
                 and length==height*width*8 and route['byte_strides']==geometry['byte_strides']==[width*8,8]
                 and 0 <= start_row < start_row+rows <= height and 0 <= start_col < start_col+cols <= width
                 and _sha(full)==output['expected']['sha256'],
                 'startup asset geometry or original coordinate bytes differ')
        intervals = [(8*(row*width+start_col), 8*(row*width+start_col+cols))
                     for row in range(start_row,start_row+rows)]
        payload = b''.join(full[a:b] for a,b in intervals)
        packed[role] = payload
        roots[role] = dict(id=identity,offset=offset,size_bytes=length,
            source=route['startup_material'],original_slice=route['source_slice'],
            root_sha256=_sha(full),packed_sha256=_sha(payload),packed_bytes=len(payload),
            **{key:route[key] for key in view_keys})
    x,y = (roots[role] for role in ('x','y'))
    _require(x['offset']+x['size_bytes']<=y['offset'] or y['offset']+y['size_bytes']<=x['offset'],
             'immutable coordinate byte ranges overlap')
    z = axes['z']
    ports = [row for row in baseline['ports'] if row['kind']=='array']
    zidx = native['plot']['binding']['current_root_ordinal']
    _require(type(zidx) is int and 0<=zidx<len(ports) and ports[zidx]['id']==z['id']
             and ports[zidx]['output'] is True and ports[zidx]['shape']==z['root_shape']
             and z['kind']=='CONTROLLER_ARRAY' and z['current_array']==z['id'],
             'controller mutable field may not move into the prepared context')
    _require(native['startup']['mode']=='ORIGINAL_CONSTRUCTOR_BUILDTIME_MATERIALIZATION',
             'unknown startup storage owner')
    return roots,packed,image_ref,dict(id=z['id'],root_ordinal=zidx,controller_path=z['controller_path'],
                                     original_slice=z['source_slice'],
                                     **{key:z[key] for key in view_keys})


def _check_coordinate_evidence(source, payloads):
    plan = read_artifact(source['contour'])
    result = []
    for sample in plan['samples']:
        raw = read_artifact(sample['input'],json_document=False)
        offsets = sample['grid']['offsets']
        for role,expected in payloads.items():
            begin = offsets[role]
            _require(begin>=0 and raw[begin:begin+len(expected)]==expected,
                     'startup coordinate packing differs at original observed contour port')
        result.append(dict(frame_index=sample['frame_index'],input=sample['input']))
    _require(len(result)>=2,'need multiple original contour observations')
    return result


def _assets(source,baseline,baseline_ref,provider_ref,provider):
    native = baseline['controller']['plot']
    _require(provider['symbol']=='scar_live_plot_v1' and provider['exact_original_png_bytes'] is True
             and provider['schema_version']==1 and provider['in_process'] is True
             and native['synchronous'] is True,
             'prior native provider must be the pinned synchronous in-process renderer')
    layout = read_artifact(provider['sources']['layout'])
    _require(layout['contour_plan']==source['contour'] and layout['title_plan']==source['text'],
             'provider lowering references different original fit inputs')
    assets = {}
    layouts = {row['template'] for row in layout['scenes']}
    template = _one(layouts,'original scenes do not share one qualified read-only layout')
    layout_ref = provider['sources']['layout']
    _,expected_layout = _member(layout_ref,layout['files'][template],template)
    expected_font = read_artifact(provider['sources']['font'],json_document=False)
    for role in ('layout','font','library'):
        key = provider[role]
        old_ref,old = _member(provider_ref,provider['files'][key],key)
        installed = str(ASSETS / key)
        installed_ref,payload = _member(baseline_ref,baseline['files'][installed],installed)
        _require(installed in native['binding']['provider_members'] and payload==old,
                 'installed native plot member differs from its prior provider')
        if role in ('font','layout'):
            _require(payload==(expected_font if role=='font' else expected_layout),
                     'installed asset differs from the qualified original plot source at '+role)
        assets[role] = dict(member=installed,size_bytes=len(payload),sha256=installed_ref['sha256'],
                            installed_material=installed_ref,provider_material=old_ref)
    _require(len({Path(assets[role]['member']).name for role in ('font','layout')})==2,
             'plot asset names cannot collide in the observation')
    return assets


def _producers(baseline,baseline_ref,provider,provider_ref):
    declarations = {
        'owner': (baseline_ref,baseline,HEADER,dict(
            admission='for (const auto& asset : assets) verify_asset(package, asset)',
            dispatch='const auto result = render_(&x_, &y_',
            startup_x='x_(startup.borrow_array(',
            startup_y='y_(startup.borrow_array(',
        )),
        'render': (provider_ref,provider,'src/live.cpp',dict(
            load_layout='const auto source = profile_bytes(layout,',
            load_font='const auto face = profile_bytes(font,',
            pack_x='selected.x = borrow_lattice(tape, selected, x)',
            pack_y='selected.y = borrow_lattice(tape, selected, y)',
            pack_z='selected.z = borrow_lattice(tape, selected, z)',
            validate_x='guard_plot_coordinates(tape, grid.x)',
            validate_y='guard_plot_coordinates(tape, grid.y)',
            live_time='generate_glyphs(face, time)',
        )),
        'glyph': (provider_ref,provider,'src/glyph.cpp',dict(
            open_memory_font='FT_New_Memory_Face(library,',
        )),
    }
    result = {}
    for phase,(receipt,document,name,expected) in declarations.items():
        reference,payload = _member(receipt,document['files'][name],name)
        lines = payload.splitlines(keepends=True)
        actions = {}
        for role,needle in expected.items():
            index,line = _one(((i,line) for i,line in enumerate(lines,1) if needle.encode() in line),
                              'missing exact pinned v1 producer anchor: '+role)
            actions[role] = dict(line=index,sha256=_sha(line))
        result[phase] = dict(material=reference,actions=actions)
    return result


def _observe(reference,assets,source):
    summary = read_artifact(reference)
    path = Path(reference['uri']).parent
    trace = _ref(path/'openat.log','scar.filtered-strace-openat')
    raw = read_artifact(trace,json_document=False)
    count = summary['frames']
    _require(summary['schema_version']==1 and summary['instrumented'] is True and summary['cost_ns'] is None
             and type(count) is int and 2<=count<=4096,
             'bounded original multi-frame open observation required')
    lookup = {assets[key]['installed_material']['uri']:key for key in ('layout','font')}
    ordered, events = [],{}
    for index,line in enumerate(raw.splitlines(keepends=True),1):
        match = _OPEN.fullmatch(line.decode().rstrip('\n'))
        _require(match is not None,'filtered trace must contain successful read-only asset opens')
        name = json.loads(match.group(1))
        _require(name in lookup,'asset open path does not name this baseline package')
        role = lookup[name]
        ordered.append(role)
        events.setdefault(role,[]).append(dict(line=index,sha256=_sha(line),fd=int(match.group(2))))
    preflight = sorted(('layout','font'),key=lambda role:assets[role]['member'])
    _require(ordered==preflight+['layout','font']*count,
             'filtered evidence does not match old controller preflight and per-frame provider opens')
    _require(set(summary['load_counts'])=={Path(key).name for key in lookup},
             'asset manifest and captured open paths differ')
    counts = {}
    for role in ('layout','font'):
        asset = assets[role]
        measured = summary['load_counts'][Path(asset['member']).name]
        _require(measured['total_successful_opens']==len(events[role])==count+1
                 and measured['provider_asset_sha256']==asset['sha256']
                 and measured['file_bytes']==asset['size_bytes']
                 and measured['logical_bytes_when_fully_read']==asset['size_bytes']*(count+1),
                 'filtered open count or known resource identity differs')
        counts[role] = dict(observed_successful_opens=count+1, proposed_successful_opens=1,
                            constructor_open=events[role][0], per_frame_opens=events[role][1:],
                            estimated_duplicate_full_reads_bytes=count*asset['size_bytes'])
    counter = source['counter']
    filename = source['output']['filename']
    paths = [path/'frames'/(filename['prefix'] + format(counter['initial']+i*counter['increment'],
                                               filename['format'])+filename['suffix']) for i in range(count)]
    frames = [_ref(member,'image/png') for member in paths]
    _require(all(read_artifact(frame,json_document=False).startswith(b'\x89PNG\r\n\x1a\n') for frame in frames),
             'paired original output does not cover all retained snapshot calls')
    return dict(trace=trace,trace_is_path_filtered=True,frame_count=count,frames=frames,
                invocation=summary['invocation'],asset_opens=counts,timing_ns=None)


def _derive(source_ref,baseline_ref,observation):
    source = check_plot(source_ref['uri'])
    baseline = read_artifact(baseline_ref)
    _require(source==read_artifact(source_ref) and baseline['schema_version']==1
             and baseline['controller']['plot']['planning']==source_ref,
             'baseline executable must use the exact original plot binding')
    native = baseline['controller']
    startup = read_artifact(source['startup'])
    _require(native['startup']['planning']==source['startup'] and native['planning']==startup['controller_plan'],
             'baseline must use the exact original startup and controller plans')
    provider_ref = native['plot']['provider']
    provider = read_artifact(provider_ref)
    loop = _original_loop(source,startup,baseline)
    roots,packed,image,z = _scoped_axes(source,startup,baseline,baseline_ref,provider)
    evidence = _check_coordinate_evidence(source,packed)
    assets = _assets(source,baseline,baseline_ref,provider_ref,provider)
    observed = _observe(observation,assets,source)
    producers = _producers(baseline,baseline_ref,provider,provider_ref)
    coord_bytes = sum(row['packed_bytes'] for row in roots.values())
    asset_bytes = sum(assets[role]['size_bytes'] for role in ('layout','font'))
    return dict(schema=SCHEMA,schema_version=1,source_plot=source_ref,source_provider=provider_ref,
                baseline_entry=baseline_ref,observation=observation,original_loop=loop,assets=assets,
                original_native_producers=producers,
                startup_image=image,roots=roots,original_coordinate_observations=evidence,
                dynamic=dict(z=z,time=source['title']['controller_scalar']),
                owner=dict(kind='PRIVATE_SYNCHRONOUS_CONTROLLER_PLOT_CONTEXT',
                    create_after=['STARTUP_IMAGE_ADMISSION','REPLACEMENT_PROVIDER_PREFLIGHT'],
                    create_before='ORIGINAL_CONTROLLER_LOOP',
                    destroy_after='LAST_USE_WITHIN_SAME_CONTROLLER_ACTIVATION',
                    package_bytes_must_be_immutable_during_activation=True,
                    snapshot_regular_assets_exact_size_and_sha256=True,
                    roots_have_no_numeric_or_provider_write_alias=True,
                    may_share_across_activations=False,
                    guarded_fallback='BASELINE_V1_AVAILABLE_AT_PLAN_SELECTION',
                    guard_failure_before_controller_effects=True,
                    automatic_mixed_version_fallback=False),
                evidence=observed, resource_model=dict(
                    resident_payload_bytes=asset_bytes+coord_bytes,
                    retained_coordinate_bytes=coord_bytes,
                    removed_coordinate_copy_payload_bytes=(observed['frame_count']-1)*coord_bytes,
                    duplicate_asset_full_read_bytes=observed['frame_count']*asset_bytes,
                    resident_payload_excludes_library_and_decode_allocations=True,
                    io_bytes_measured=False, performance_measured=False),
                preserved_dependencies=[
                    dict(before='CURRENT_Z_READ',after='GENERATE_LIVE_GEOMETRY_AND_COLORS'),
                    dict(before='CURRENT_TIME_READ',after='GENERATE_LIVE_TITLE'),
                    dict(before='GENERATE_LIVE_GEOMETRY_AND_COLORS',after='ORIGINAL_SNAPSHOT_PUBLICATION'),
                    dict(before='GENERATE_LIVE_TITLE',after='ORIGINAL_SNAPSHOT_PUBLICATION'),
                    dict(before='ORIGINAL_SNAPSHOT_PUBLICATION',after='SUCCESS_ONLY_FILENAME_INCREMENT')],
                legality='CONDITIONAL_EXPLICIT_TRIAL_WITH_IMMUTABLE_COMPILER_OWNED_PACKAGE',
                default_backend='NO_OP',whole_entry_authorized=False,deployment_authorized=False)


def project(source_plot,baseline_entry,observation):
    return _derive(_ref(source_plot,PLOT),_ref(baseline_entry,ENTRY),_ref(observation,OBSERVATION))


def check(path):
    result = read_artifact(_ref(path,SCHEMA))
    _require(result==_derive(result['source_plot'],result['baseline_entry'],result['observation']),
             'residency receipt differs from original source, image, assets and observed opens')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--source-plot')
    mode.add_argument('--check')
    parser.add_argument('--baseline-entry')
    parser.add_argument('--observation')
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    if args.check and not (args.baseline_entry or args.observation or args.out):
        result = check(args.check)
    elif all((args.source_plot,args.baseline_entry,args.observation,args.out)):
        result = project(args.source_plot,args.baseline_entry,args.observation)
        with Path(args.out).open('x',encoding='utf-8') as stream:
            json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False)
            stream.write('\n')
    else:
        parser.error('provide --check or --source-plot with --baseline-entry, --observation and --out')
    print(json.dumps(dict(schema=SCHEMA,frames=result['evidence']['frame_count'],
                          payload_bytes=result['resource_model']['resident_payload_bytes'])))


if __name__=='__main__':
    main()
