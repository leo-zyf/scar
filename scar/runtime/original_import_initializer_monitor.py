"""Record exact source initializer activations beside existing lexical import captures."""

import builtins
import sys


_POSIX = sys.modules['posix']
_EXTERNAL = sys.modules['_frozen_importlib_external']
_SPEC = sys.modules['_frozen_importlib'].ModuleSpec
_LOADER = _EXTERNAL.SourceFileLoader
_EXEC_MODULE = _EXTERNAL._LoaderBasics.exec_module.__code__
_CODE = type((lambda: None).__code__)
_MODULE = type(sys)
_WEAKREF = sys.modules['_weakref'].ref
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


def _read_config(request):
    with open(request, 'rb') as stream:
        code = compile(stream.read(), request, 'eval', dont_inherit=True)
    return eval(code, {'__builtins__': {}})


class _Initializers:
    def __init__(self, config, slot):
        self.config = config
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
        self.identities.setdefault(name, []).append((self.identity_serial, _WEAKREF(module)))
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
        if (type(loader) is not _LOADER or type(module) is not _MODULE or module.__dict__ is not namespace
                or loader.name != name or loader.path != path or namespace.get('__loader__') is not loader):
            return None
        frame = frame.f_back
        while frame is not None:
            if frame.f_code is _EXEC_MODULE:
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


def _spawn_runtime(request):
    config = _read_config(request)
    source = config['spawn_observer']
    namespace = dict(__name__='_scar_spawn_observer_helper', __file__=__file__)
    with open(source, 'rb') as stream:
        code = compile(stream.read(), source, 'exec', dont_inherit=True)
    exec(code, namespace)
    run = namespace['_run']

    def observed(config, request, directory, slot, execute):
        monitor = _Initializers(config, slot)
        try:
            return run(config, request, directory, slot, lambda: monitor.run(execute))
        finally:
            monitor.write()

    namespace['_run'] = observed
    return config, namespace


if __name__ == '__main__':
    request, directory = sys.argv[1:]
    config, spawn_runtime = _spawn_runtime(request)
    spawn_runtime['_primary'](config, request, directory)
elif '_SCAR_CHILD' in globals():
    _, spawn_runtime = _spawn_runtime(_SCAR_CHILD[0])
    spawn_runtime['_child'](*_SCAR_CHILD)
