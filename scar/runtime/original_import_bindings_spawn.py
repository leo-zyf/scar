"""Observe lexical imports in an original CPython process and optional direct spawn workers."""

import builtins
import sys


_POSIX = sys.modules['posix']
_IMPORT = builtins.__import__
_MODULE = type(sys)
_SPEC = sys.modules['_frozen_importlib'].ModuleSpec
_MISSING = object()
_FUNCTION = type(lambda: None)
_SPAWN_PREFIX = 'from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd='
_SCOPE = 'DIRECT_STANDARD_CPYTHON_SPAWN'


def _text(value):
    if type(value) is bytes:
        return value.decode(sys.getfilesystemencoding(), sys.getfilesystemencodeerrors())
    return value


def spawn_payload_fds(original_argv):
    """Recognize only the POSIX stdlib command to bootstrap a worker pipe."""
    if len(original_argv) < 4 or original_argv[-3] != '-c' or original_argv[-1] != '--multiprocessing-fork':
        return None
    payload = original_argv[-2]
    if not payload.startswith(_SPAWN_PREFIX) or not payload.endswith(')'):
        return None
    tracker, delimiter, pipe = payload[len(_SPAWN_PREFIX):-1].partition(', pipe_handle=')
    if not delimiter or not all(fd.isascii() and fd.isdecimal() and len(fd) <= 10 for fd in (tracker, pipe)):
        return None
    return int(tracker), int(pipe)


def diagnostic_spawn_argv(original_argv, bootstrap, request, directory, slot):
    """Keep the executable and flags, changing only the worker's -c payload."""
    argv = list(original_argv)
    namespace = dict(__name__='_scar_spawn_observer', __file__=bootstrap,
                     _SCAR_CHILD=(request, directory, slot, argv.copy()))
    argv[-2] = f"exec(compile(open({bootstrap!r}, 'rb').read(), {bootstrap!r}, 'exec'), {namespace!r})"
    return argv


def _read_config(request):
    with open(request, 'rb') as stream:
        literal = compile(stream.read(), request, 'eval', dont_inherit=True)
    return eval(literal, {'__builtins__': {}})


def _names(name, namespace, fromlist, level):
    if level:
        package = namespace.get('__package__')
        if package is None:
            spec = namespace.get('__spec__')
            if type(spec) is _SPEC:
                package = spec.parent
            else:
                package = namespace.get('__name__', '')
                if type(package) is str and '__path__' not in namespace:
                    package = package.rpartition('.')[0]
        if type(package) is not str or not package or level > package.count('.') + 1:
            return []
        base = package.rsplit('.', level - 1)[0] if level > 1 else package
        name = base + '.' + name if name else base
    if not name:
        return []
    parts = name.split('.')
    names = ['.'.join(parts[:i]) for i in range(1, len(parts) + 1)]
    names.extend(name + '.' + item for item in fromlist if item != '*')
    return list(dict.fromkeys(names))


def _origin(module):
    if type(module) is not _MODULE:
        return None, 'UNKNOWN'
    spec = module.__dict__.get('__spec__')
    if type(spec) is not _SPEC:
        return None, 'UNKNOWN'
    origin = spec.origin
    if type(origin) is str:
        if origin.startswith('/'):
            return origin, 'ABSOLUTE_FILE'
        if origin in ('built-in', 'frozen'):
            return None, 'BUILT_IN' if origin == 'built-in' else 'FROZEN'
    if origin == 'namespace' or origin is None and spec.submodule_search_locations is not None:
        return None, 'NAMESPACE'
    return None, 'UNKNOWN'


class _SpawnObserver:
    def __init__(self, config, request, directory, record):
        self.config = config
        self.request = request
        self.directory = directory
        self.record = record
        self.module = None
        self.original = None
        self.guard = None
        self.next_slot = 1

    def install_if_ready(self):
        if self.module is not None:
            return
        module = sys.modules.get('multiprocessing.util')
        if type(module) is not _MODULE or module.__dict__.get('__file__') != self.config['spawn_util_file']:
            return
        spec = module.__dict__.get('__spec__')
        original = module.__dict__.get('spawnv_passfds')
        threading = sys.modules.get('threading')
        if spec is None or getattr(spec, '_initializing', False) or type(original) is not _FUNCTION \
                or original.__code__.co_filename != self.config['spawn_util_file'] or type(threading) is not _MODULE:
            return
        self.guard = threading.Lock()
        self.original = original
        self.wrapper = self.spawn
        self.module = module
        module.spawnv_passfds = self.wrapper

    def restore(self):
        if self.module is not None and self.module.__dict__.get('spawnv_passfds') is self.wrapper:
            self.module.spawnv_passfds = self.original

    def spawn(self, path, args, passfds):
        caller = sys._getframe(1).f_code
        popen = sys.modules.get('multiprocessing.popen_spawn_posix')
        if type(popen) is not _MODULE or caller.co_filename != self.config['spawn_launch_file'] \
                or popen.__dict__.get('__file__') != self.config['spawn_launch_file'] \
                or caller is not popen.Popen._launch.__code__:
            return self.original(path, args, passfds)

        original_argv = [_text(arg) for arg in args]
        executable = _text(path)
        fds = spawn_payload_fds(original_argv)
        with self.guard:
            eligible = (original_argv[0] == executable == self.config['original_argv'][0]
                        and fds is not None and all(fd in passfds for fd in fds)
                        and self.next_slot <= self.config['max_spawn_children'])
            if eligible:
                slot = self.next_slot
                self.next_slot += 1
            else:
                self.record['direct_spawns_omitted'] += 1
        if not eligible:
            return self.original(path, args, passfds)

        diagnostic = diagnostic_spawn_argv(original_argv, __file__, self.request, self.directory, slot)
        argv = list(args)
        argv[-2] = diagnostic[-2]
        pid = self.original(path, argv, passfds)
        with self.guard:
            self.record['children'].append(dict(slot=slot, pid=pid, executable=executable,
                original_argv=original_argv, diagnostic_argv=diagnostic))
        return pid


def _run(config, request, directory, slot, execute):
    follow_spawn = 'observer_core' in config
    record = dict(entries=[])
    spawn = budget = None
    destination = directory
    if follow_spawn:
        record.update(schema='scar.cpu-original-import-process', schema_version=1, slot=slot,
                      pid=_POSIX.getpid(), ppid=_POSIX.getppid(), cpu_ids=sorted(_POSIX.sched_getaffinity(0)),
                      entries_omitted=0, children=[], direct_spawns_omitted=0,
                      children_scope=_SCOPE if slot == 0 else 'NOT_INSTRUMENTED')
        spawn = _SpawnObserver(config, request, directory, record) if slot == 0 else None
        budget = iter(range(config['max_entries_per_process']))
        destination = f'{directory}/process-{slot:04d}.txt'
    sites = config['sites']
    opcode = config['import_opcode']

    def observe(name, globals=None, locals=None, fromlist=(), level=0):
        caller = sys._getframe(1)
        code = caller.f_code
        file_sites = sites.get(code.co_filename)
        offset = caller.f_lasti
        position = None if file_sites is None else file_sites.get((code.co_firstlineno, code.co_qualname, offset, name))
        watched = position is not None and offset < len(code.co_code) and code.co_code[offset] == opcode
        if watched:
            requested = list(fromlist or ())
            names = _names(name, globals or {}, requested, level)
            before = [sys.modules.get(module, _MISSING) for module in names]
        result = _IMPORT(name, globals, locals, fromlist, level)
        if spawn is not None and spawn.module is None:
            spawn.install_if_ready()
        if watched:
            if budget is not None and next(budget, None) is None:
                record['entries_omitted'] += 1
            else:
                modules = []
                for name_, old in zip(names, before):
                    present = sys.modules.get(name_, _MISSING)
                    cached = old is not _MISSING and old is not None
                    available = present is not _MISSING and present is not None
                    origin, kind = _origin(present)
                    modules.append(dict(name=name_, origin=origin, origin_kind=kind, cached_before=cached,
                                        present_after=available, same_cached_object=cached and present is old))
                record['entries'].append(dict(caller=code.co_filename, position=position, offset=offset,
                    code_first_line=code.co_firstlineno, code_name=code.co_qualname,
                    requested_name=name, level=level, fromlist=requested, modules=modules))
        return result

    builtins.__import__ = observe
    try:
        if spawn is not None:
            spawn.install_if_ready()
        execute()
    finally:
        if builtins.__import__ is observe:
            builtins.__import__ = _IMPORT
        if spawn is not None:
            spawn.restore()
        if follow_spawn:
            record['children'].sort(key=lambda child: child['slot'])
        original_error = sys.exception()
        try:
            with open(destination, 'x', encoding='utf-8') as output:
                output.write(repr(record if follow_spawn else record['entries']) + '\n')
        except OSError:
            if original_error is None:
                raise


def _primary(config, request, directory):
    original_argv = config['original_argv']
    source_index = config['source_index']
    script = original_argv[source_index]
    sys.argv[:] = original_argv[source_index:]
    sys.orig_argv[:] = original_argv
    if not sys.flags.safe_path:
        sys.path[0] = config['script_directory']
    main = _MODULE('__main__')
    main.__dict__.update(__file__=script, __cached__=None, __builtins__=builtins,
                         __loader__=sys.modules['_frozen_importlib_external'].SourceFileLoader('__main__', script))
    sys.modules['__main__'] = main
    with open(script, 'rb') as stream:
        code = compile(stream.read(), script, 'exec', dont_inherit=True)
    _run(config, request, directory, 0, lambda: exec(code, main.__dict__))


def _child(request, directory, slot, original_argv):
    sys.orig_argv[:] = original_argv
    config = _read_config(request)
    code = compile(original_argv[-2], '<string>', 'exec', dont_inherit=True)
    _run(config, request, directory, slot, lambda: exec(code, sys.modules['__main__'].__dict__))


if __name__ == '__main__':
    request, directory = sys.argv[1:]
    _primary(_read_config(request), request, directory)
elif '_SCAR_CHILD' in globals():
    _child(*_SCAR_CHILD)
