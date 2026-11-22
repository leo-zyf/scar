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


def _capture_imports(config, request, directory, slot, execute):
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


_CODE = type((lambda: None).__code__)
_CODE_SCALARS = ('co_argcount', 'co_posonlyargcount', 'co_kwonlyargcount', 'co_nlocals', 'co_stacksize',
    'co_flags', 'co_code', 'co_names', 'co_varnames', 'co_freevars', 'co_cellvars', 'co_filename', 'co_name',
    'co_qualname', 'co_firstlineno', 'co_linetable', 'co_exceptiontable')


def _same(actual, expected, depth=0):
    """Compare only the exact Python builtin shapes produced by the compiler."""
    kind = type(expected)
    if type(actual) is not kind or depth >= 128:
        return False
    if kind is _CODE:
        return (all(getattr(actual, key) == getattr(expected, key) for key in _CODE_SCALARS)
                and _same(actual.co_consts, expected.co_consts, depth + 1))
    if kind is tuple:
        return len(actual) == len(expected) and all(_same(a, b, depth + 1) for a, b in zip(actual, expected))
    if kind is slice:
        return all(_same(a, b, depth + 1) for a, b in zip(
            (actual.start, actual.stop, actual.step), (expected.start, expected.stop, expected.step)))
    if kind is frozenset:
        return len(actual) == len(expected) and all(any(_same(a, b, depth + 1) for a in actual) for b in expected)
    if kind is float:
        return expected == expected and actual.hex() == expected.hex()
    if kind is complex:
        return _same(actual.real, expected.real, depth + 1) and _same(actual.imag, expected.imag, depth + 1)
    if kind in (str, bytes, int, bool) or expected is None or expected is Ellipsis:
        return actual == expected
    return False


class _Initializers:
    def __init__(self, config, slot):
        self.config = config
        external = sys.modules['_frozen_importlib_external']
        self.loader_type = external.SourceFileLoader
        self.exec_module_code = external._LoaderBasics.exec_module.__code__
        self.module_ref = sys.modules['_weakref'].ref
        self.expected = {(row['source_path'], row['module_name']):
            compile(row['source_bytes'], row['source_path'], 'exec', dont_inherit=True)
            for row in config['initializer_sources']}
        self.source_paths = {path for path, _ in self.expected}
        self.target_names = {name for _, name in self.expected}
        self.record = dict(schema='scar.cpu-import-initializer-process', schema_version=1, slot=slot,
            pid=_POSIX.getpid(), ppid=_POSIX.getppid(), cpu_ids=sorted(_POSIX.sched_getaffinity(0)),
            activations=[], import_completions=[], activations_omitted=0, rejections={})
        self.monitor = sys.monitoring
        self.events = self.monitor.events
        self.tool = None
        self.observer_code = None
        self.active = {}
        self.watches = {}
        self.identities = {}
        self.serial = 0
        self.identity_serial = 0
        self.unwind_enabled = False

    def tick(self):
        self.serial += 1
        return self.serial

    def reject(self, reason):
        output = self.record['rejections']
        output[reason] = output.get(reason, 0) + 1

    def update_events(self):
        interested = bool(self.active or self.watches)
        if interested != self.unwind_enabled:
            self.unwind_enabled = interested
            self.monitor.set_events(self.tool, self.events.PY_START | (self.events.PY_UNWIND if interested else 0))

    def identity(self, name, module, *, introduce=False):
        for token, reference in self.identities.get(name, ()):
            if reference() is module:
                return token
        if not introduce:
            return None
        self.identity_serial += 1
        self.identities.setdefault(name, []).append((self.identity_serial, self.module_ref(module)))
        return self.identity_serial

    def request(self, observer, *, watched_only=False):
        source = observer.f_back
        if source is None:
            return None
        code = source.f_code
        path = code.co_filename
        if watched_only and path not in self.config['sites']:
            return None
        offset = source.f_lasti
        raw = code.co_code
        if not path.startswith('/') or not 0 <= offset < len(raw) or raw[offset] != self.config['import_opcode']:
            return None
        arguments = observer.f_locals
        name, level = arguments.get('name'), arguments.get('level')
        if type(name) is not str or type(level) is not int:
            return None
        key = code.co_firstlineno, code.co_qualname, offset, name
        position = self.config['sites'].get(path, {}).get(key)
        if position is None and not watched_only:
            positions = code.co_positions()
            for _ in range(offset // 2):
                next(positions, None)
            observed = next(positions, None)
            if observed is not None and all(type(item) is int for item in observed):
                position = list(observed)
        if position is None:
            return None
        return dict(caller=path, position=position, offset=offset, code_first_line=key[0], code_name=key[1],
                    requested_name=name, level=level, fromlist=list(arguments.get('fromlist') or ()))

    def loader(self, frame, name, path):
        namespace = frame.f_globals
        spec = namespace.get('__spec__')
        if type(spec) is not _SPEC or spec.name != name or spec.origin != path:
            return None
        loader = spec.loader
        module = sys.modules.get(name)
        if (type(loader) is not self.loader_type or type(module) is not _MODULE or module.__dict__ is not namespace
                or loader.name != name or loader.path != path or namespace.get('__loader__') is not loader):
            return None
        frame = frame.f_back
        while frame is not None:
            if frame.f_code is self.exec_module_code:
                owner = frame.f_locals
                if owner.get('self') is loader and owner.get('module') is module:
                    return frame, module
                return None
            frame = frame.f_back
        return None

    def ancestry(self, frame):
        requester, enclosing = None, None
        frame = frame.f_back
        while frame is not None:
            if frame.f_code is self.observer_code:
                if requester is None:
                    requester = self.request(frame)
                enclosing = self.watches.get(id(frame))
                if enclosing is not None and enclosing['frame'] is frame:
                    break
                enclosing = None
            frame = frame.f_back
        return requester, enclosing

    def start(self, code, offset):
        if code is self.observer_code:
            frame = sys._getframe(1)
            request = self.request(frame, watched_only=True)
            if request is not None:
                self.watches[id(frame)] = dict(frame=frame, request=request, start_sequence=self.tick(), rows=[])
                self.update_events()
            return None
        if code.co_name != '<module>' or code.co_filename not in self.source_paths:
            return self.monitor.DISABLE
        frame = sys._getframe(1)
        path, name = code.co_filename, frame.f_globals.get('__name__')
        expected = self.expected.get((path, name)) if type(name) is str else None
        if expected is None:
            self.reject('MODULE_NAMESPACE_DIFFERS')
            return None
        owner = self.loader(frame, name, path)
        if owner is None:
            self.reject('STANDARD_SOURCE_LOADER_NOT_BOUND')
            return None
        if not _same(code, expected):
            self.reject('STRUCTURAL_CODE_DIFFERS')
            return None
        rows = self.record['activations']
        if len(rows) >= self.config['max_initializer_activations']:
            self.record['activations_omitted'] += 1
            return None
        loader_frame, module = owner
        requester, enclosing = self.ancestry(loader_frame)
        row = dict(module_name=name, source_path=path, start_sequence=self.tick(), terminal_sequence=None,
                   terminal='OPEN', code_match=True, loader='STANDARD_SOURCE_FILE_LOADER_EXEC_MODULE',
                   module_identity=self.identity(name, module, introduce=True), requester=requester,
                   watched_request=None if enclosing is None else enclosing['request'], watched_entry_index=None)
        rows.append(row)
        if enclosing is not None:
            enclosing['rows'].append(row)
        self.active[id(frame)] = frame, row
        self.monitor.set_local_events(self.tool, code, self.events.PY_RETURN)
        self.update_events()
        return None

    def import_return(self, frame):
        watch = self.watches.pop(id(frame), None)
        if watch is None or watch['frame'] is not frame:
            return
        values = frame.f_locals
        modules = values.get('modules')
        if modules is None:
            return
        entries = values['record']['entries']
        entry_index = next((index for index in range(len(entries) - 1, -1, -1)
                            if entries[index]['modules'] is modules), None)
        if entry_index is None:
            return
        identities = []
        for index, row in enumerate(modules):
            name = row['name']
            if name in self.target_names:
                # The retained v2 variables identify the object of its cache
                # snapshot even if another thread replaces the entry later.
                module = (values['before'][index] if row['same_cached_object']
                          else values.get('present') if index == len(modules) - 1 else None)
                if module is not None:
                    token = self.identity(name, module)
                    if token is not None:
                        identities.append(dict(name=name, identity=token))
        self.record['import_completions'].append(dict(entry_index=entry_index, start_sequence=watch['start_sequence'],
            return_sequence=self.tick(), module_identities=identities))
        for row in watch['rows']:
            row['watched_entry_index'] = entry_index

    def finish(self, frame, code, terminal):
        current = self.active.pop(id(frame), None)
        if current is None or current[0] is not frame:
            return
        current[1].update(terminal=terminal, terminal_sequence=self.tick())
        if not any(active.f_code is code for active, _ in self.active.values()):
            self.monitor.set_local_events(self.tool, code, 0)
        self.update_events()

    def returned(self, code, offset, value):
        frame = sys._getframe(1)
        if code is self.observer_code:
            self.import_return(frame)
            self.update_events()
        else:
            self.finish(frame, code, 'PY_RETURN')

    def unwound(self, code, offset, exception):
        if code is self.observer_code:
            self.watches.pop(id(sys._getframe(1)), None)
            self.update_events()
        elif code.co_name == '<module>' and code.co_filename in self.source_paths:
            self.finish(sys._getframe(1), code, 'PY_UNWIND')

    def run(self, execute):
        self.observer_code = builtins.__import__.__code__
        self.tool = next(slot for slot in range(5, -1, -1) if self.monitor.get_tool(slot) is None)
        callbacks = ((self.events.PY_START, self.start), (self.events.PY_RETURN, self.returned),
                     (self.events.PY_UNWIND, self.unwound))
        self.monitor.use_tool_id(self.tool, 'SCAR original source initializer')
        try:
            for event, callback in callbacks:
                self.monitor.register_callback(self.tool, event, callback)
            self.monitor.set_local_events(self.tool, self.observer_code, self.events.PY_RETURN)
            self.monitor.set_events(self.tool, self.events.PY_START)
            execute()
        finally:
            self.monitor.set_events(self.tool, 0)
            self.monitor.set_local_events(self.tool, self.observer_code, 0)
            for frame, _ in self.active.values():
                self.monitor.set_local_events(self.tool, frame.f_code, 0)
            for event, _ in callbacks:
                self.monitor.register_callback(self.tool, event, None)
            self.monitor.free_tool_id(self.tool)
            self.active.clear()
            self.watches.clear()

    def write(self):
        directory = self.config['initializer_directory']
        slot = self.record['slot']
        original_error = sys.exception()
        try:
            with open(f'{directory}/initializer-process-{slot:04d}.txt', 'x', encoding='utf-8') as output:
                output.write(repr(self.record) + '\n')
        except OSError:
            if original_error is None:
                raise


def _run(config, request, directory, slot, execute):
    if 'initializer_sources' not in config:
        return _capture_imports(config, request, directory, slot, execute)
    monitor = _Initializers(config, slot)
    try:
        return _capture_imports(config, request, directory, slot, lambda: monitor.run(execute))
    finally:
        monitor.write()


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
