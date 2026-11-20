"""Bind a measured native service to one visible mount and logical CPU."""

import os
from pathlib import Path, PurePosixPath
import re
import stat

from scar.analysis.target_inventory import _cpu_list


SCHEMA = 'scar.native-service-environment'
_PROC = Path('/proc/self')
_CPU = Path('/sys/devices/system/cpu')
_ESCAPE = re.compile(r'\\(040|011|012|134)')
_LIMITS = dict(filesystem_scope='VISIBLE_MOUNT_METADATA_AT_OUTPUT_DIRECTORY',
               cpu_scope='SELECTED_LINUX_LOGICAL_CPU_TOPOLOGY',
               physical_storage_io_observed=False, publication_durability_proven=False,
               cpu_capacity_or_exclusivity_proven=False)


def _require(ok, detail):
    if not ok:
        raise ValueError('native service environment: ' + detail)


def _keys(value, expected, detail):
    _require(type(value) is dict and value.keys()==expected, detail)


def _int(value, minimum=0):
    return type(value) is int and value>=minimum


def _absolute(value):
    if type(value) is not str or '\0' in value or not value.startswith('/') or value.startswith('//'):
        return False
    path = PurePosixPath(value)
    return str(path)==value and '..' not in path.parts


def _covers(mount, directory):
    return mount=='/' or directory==mount or directory.startswith(mount+'/')


def _unescape(value):
    return _ESCAPE.sub(lambda match: chr(int(match[1],8)),value)


def _mount_record(line):
    _require(type(line) is str and '\n' not in line and '\0' not in line, 'invalid mountinfo evidence line')
    prefix,separator,suffix = line.partition(' - ')
    rows = prefix.split(' ')
    provider = suffix.split(' ',2)
    _require(bool(separator) and len(rows)>=6 and len(provider)==3 and all(rows) and all(provider)
             and re.fullmatch(r'[0-9]+',rows[0]) and re.fullmatch(r'[0-9]+',rows[1])
             and re.fullmatch(r'[0-9]+:[0-9]+',rows[2]), 'malformed mountinfo record')
    major,minor = map(int,rows[2].split(':'))
    root,mount_point = map(_unescape,rows[3:5])
    _require(_absolute(root) and _absolute(mount_point), 'mountinfo paths need canonical Linux roots')
    return dict(id=int(rows[0]),parent_id=int(rows[1]),device_major=major,device_minor=minor,
                root=root,path=mount_point,fstype=provider[0],source=_unescape(provider[1]),
                options=rows[5],optional_fields=rows[6:],super_options=provider[2])


def _select_mount(directory, device, fd_mount):
    candidate, depth = None, -1
    tied = False
    with (_PROC/'mountinfo').open('rb') as stream:
        for raw in stream:
            line = os.fsdecode(raw.removesuffix(b'\n'))
            row = _mount_record(line)
            if ((row['device_major'],row['device_minor'])!=(os.major(device),os.minor(device))
                or not _covers(row['path'],directory)
                or (fd_mount is not None and row['id']!=fd_mount)):
                continue
            level = len(PurePosixPath(row['path']).parts)
            if level>depth:
                candidate,depth,tied = (row,line),level,False
            elif level==depth:
                tied = True
    _require(candidate is not None and not tied,
             'output directory needs one device-matched visible mount')
    return candidate


def _descriptor_mount(fd, inode):
    fields = {}
    try:
        with (_PROC/'fdinfo'/str(fd)).open('rb') as stream:
            for line in stream:
                key,_,value = line.partition(b':')
                if key in (b'mnt_id',b'ino'):
                    _require(key not in fields and value.strip().isdigit(), 'malformed fdinfo identity')
                    fields[key] = int(value.strip())
    except (FileNotFoundError,PermissionError):
        return None
    _require(b'ino' not in fields or fields[b'ino']==inode, 'directory descriptor inode differs from fdinfo')
    return fields.get(b'mnt_id')


def _identity(status):
    return dict(device=status.st_dev,inode=status.st_ino)


def _filesystem(output_directory):
    path = Path(output_directory).resolve(strict=True)
    namespace = _PROC/'ns/mnt'
    before = _identity(namespace.stat())
    descriptor = os.open(path,os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        selected = os.fstat(descriptor)
        mount_id = _descriptor_mount(descriptor,selected.st_ino)
        row,line = _select_mount(str(path),selected.st_dev,mount_id)
        current = path.stat(follow_symlinks=False)
        _require(stat.S_ISDIR(current.st_mode) and _identity(selected)==_identity(current)
                 and before==_identity(namespace.stat()), 'output directory or mount namespace changed during capture')
    finally:
        os.close(descriptor)
    return dict(output_directory=str(path),**_identity(selected),mount=row,mountinfo_line=line,
                descriptor_mount_id=mount_id,mount_namespace=before,
                selection='DESCRIPTOR_MOUNT_ID' if mount_id is not None else 'LONGEST_DEVICE_MATCH')


def _cpu(cpu_id):
    _require(_int(cpu_id) and cpu_id in os.sched_getaffinity(0),
             'requested logical CPU must be in the current allowed affinity')
    path = _CPU/f'cpu{cpu_id}'
    topology = path/'topology'
    package = int((topology/'physical_package_id').read_text().strip())
    core = int((topology/'core_id').read_text().strip())
    siblings = _cpu_list((topology/'thread_siblings_list').read_text())
    try:
        die = int((topology/'die_id').read_text().strip())
    except FileNotFoundError:
        die = -1
    nodes = [int(member.name[4:]) for member in path.iterdir()
             if re.fullmatch(r'node[0-9]+',member.name)]
    _require(len(nodes)<=1, 'logical CPU belongs to multiple reported NUMA nodes')
    return dict(id=cpu_id,package_id=package,core_id=core,die_id=None if die==-1 else die,
                thread_siblings=siblings,numa_node=nodes[0] if nodes else None,required_affinity=[cpu_id])


def capture(output_directory, cpu_id):
    """Capture only the requested Linux directory and CPU identity."""
    return validate(dict(schema=SCHEMA,schema_version=1,filesystem=_filesystem(output_directory),
                         cpu=_cpu(cpu_id),limits=dict(_LIMITS)))


def validate(record):
    """Validate a stored observation without opening its paths or reading topology."""
    _keys(record,{'schema','schema_version','filesystem','cpu','limits'},'wrong environment receipt fields')
    _require(record['schema']==SCHEMA and type(record['schema_version']) is int
             and record['schema_version']==1, 'unsupported environment receipt version')
    limits = record['limits']
    _keys(limits,_LIMITS.keys(),'wrong environment boundaries')
    _require(all(type(limits[key]) is type(value) and limits[key]==value for key,value in _LIMITS.items()),
             'receipt exceeds mount or CPU topology scope')

    filesystem = record['filesystem']
    _keys(filesystem,{'output_directory','device','inode','mount','mountinfo_line',
                      'descriptor_mount_id','mount_namespace','selection'},'wrong filesystem receipt fields')
    directory,device,inode = (filesystem[key] for key in ('output_directory','device','inode'))
    _require(_absolute(directory) and _int(device) and _int(inode,1),'invalid output directory identity')
    mount = _mount_record(filesystem['mountinfo_line'])
    _keys(filesystem['mount'],mount.keys(),'wrong mount receipt fields')
    _require(all(type(filesystem['mount'][key]) is type(value) and filesystem['mount'][key]==value
                 for key,value in mount.items())
             and _int(mount['id'],1) and _int(mount['parent_id'],1)
             and _covers(mount['path'],directory),'mount receipt differs from its observed path')
    try:
        device_number = os.makedev(mount['device_major'],mount['device_minor'])
    except OverflowError as error:
        raise ValueError('native service environment: invalid Linux mount device') from error
    _require(device==device_number,'directory st_dev differs from the visible mount device')
    namespace = filesystem['mount_namespace']
    _keys(namespace,{'device','inode'},'wrong mount namespace identity fields')
    _require(_int(namespace['device']) and _int(namespace['inode'],1),'invalid mount namespace identity')
    mount_id = filesystem['descriptor_mount_id']
    _require((mount_id is None and filesystem['selection']=='LONGEST_DEVICE_MATCH')
             or (_int(mount_id,1) and mount_id==mount['id'] and filesystem['selection']=='DESCRIPTOR_MOUNT_ID'),
             'mount selection differs from the recorded descriptor')

    cpu = record['cpu']
    _keys(cpu,{'id','package_id','core_id','die_id','thread_siblings','numa_node','required_affinity'},
          'wrong CPU topology receipt fields')
    _require(all(_int(cpu[key]) for key in ('id','package_id','core_id'))
             and all(cpu[key] is None or _int(cpu[key]) for key in ('die_id','numa_node')),
             'invalid CPU, socket, core, die or NUMA identifier')
    siblings, affinity = cpu['thread_siblings'],cpu['required_affinity']
    _require(type(siblings) is list and all(_int(item) for item in siblings)
             and siblings==sorted(set(siblings)) and cpu['id'] in siblings
             and type(affinity) is list and len(affinity)==1 and _int(affinity[0]) and affinity[0]==cpu['id'],
             'SMT siblings or required single CPU affinity differ')
    return record
