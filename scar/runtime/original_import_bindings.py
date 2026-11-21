"""Run a file-style CPython entry with a narrow lexical import observer."""

import builtins
import sys


_IMPORT = builtins.__import__
_MODULE = type(sys)
_SPEC = sys.modules['_frozen_importlib'].ModuleSpec
_MISSING = object()


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


def _run(config, destination):
    entries = []
    sites = config['sites']
    import_opcode = config['import_opcode']

    def observe(name, globals=None, locals=None, fromlist=(), level=0):
        caller = sys._getframe(1)
        code = caller.f_code
        file_sites = sites.get(code.co_filename)
        if file_sites is None:
            return _IMPORT(name, globals, locals, fromlist, level)
        offset = caller.f_lasti
        coordinate = code.co_firstlineno, code.co_qualname, offset, name
        position = file_sites.get(coordinate)
        if position is None or offset >= len(code.co_code) or code.co_code[offset] != import_opcode:
            return _IMPORT(name, globals, locals, fromlist, level)

        requested = list(fromlist or ())
        names = _names(name, globals or {}, requested, level)
        before = [sys.modules.get(module, _MISSING) for module in names]
        # The original importer and exceptions propagate directly to the file.
        result = _IMPORT(name, globals, locals, fromlist, level)
        modules = []
        for module, old in zip(names, before):
            present = sys.modules.get(module, _MISSING)
            cached = old is not _MISSING and old is not None
            available = present is not _MISSING and present is not None
            origin, kind = _origin(present)
            modules.append(dict(name=module, origin=origin, origin_kind=kind, cached_before=cached,
                                present_after=available, same_cached_object=cached and present is old))
        entries.append(dict(caller=code.co_filename, position=position, offset=offset,
                            code_first_line=code.co_firstlineno, code_name=code.co_qualname,
                            requested_name=name, level=level, fromlist=requested, modules=modules))
        return result

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
    builtins.__import__ = observe
    try:
        exec(code, main.__dict__)
    finally:
        if builtins.__import__ is observe:
            builtins.__import__ = _IMPORT
        original_error = sys.exception()
        try:
            with open(destination, 'x', encoding='utf-8') as output:
                output.write(repr(entries) + '\n')
        except OSError:
            if original_error is None:
                raise


if __name__ == '__main__':
    request, destination = sys.argv[1:]
    with open(request, 'rb') as stream:
        literal = compile(stream.read(), request, 'eval', dont_inherit=True)
    _run(eval(literal, {'__builtins__': {}}), destination)
