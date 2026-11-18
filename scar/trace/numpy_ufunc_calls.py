"""Optional exact-ufunc metadata beside an original bounded CALL capture."""
import argparse
from importlib.machinery import ExtensionFileLoader, ModuleSpec
import json
import os
from pathlib import Path
import runpy
import sys
import threading
import types

from scar.compiler.call_binding import plan_digest
from scar.trace.call_binding import _type_metadata, install
from scar.trace.code_snapshot import code_key


SCHEMA = 'scar.numpy-ufunc-call-observations'
CORE_MODULES = ('numpy._core._multiarray_umath', 'numpy.core._multiarray_umath')


def ufunc_metadata(value):
    """Read only exact C descriptors on the already loaded NumPy ufunc type."""
    numpy = sys.modules.get('numpy')
    kind = type(value)
    if (type(numpy) is not types.ModuleType or vars(numpy).get('ufunc') is not kind
            or _type_metadata(kind) != {'module': 'numpy', 'qualname': 'ufunc'}):
        return None
    fields = {}
    for field in ('__name__', 'nin', 'nout', 'signature'):
        descriptor = vars(kind).get(field)
        if type(descriptor) is not types.GetSetDescriptorType or descriptor.__objclass__ is not kind:
            return None
        fields[field] = descriptor.__get__(value, kind)
    name, inputs, outputs, signature = (fields[key] for key in ('__name__', 'nin', 'nout', 'signature'))
    if (type(name) is not str or not 0 < len(name) <= 256 or type(inputs) is not int
            or type(outputs) is not int or not 0 <= inputs <= 64 or not 0 <= outputs <= 64
            or signature is not None and (type(signature) is not str or len(signature) > 512)):
        return None
    version = vars(numpy).get('__version__')
    result = dict(name=name, input_count=inputs, output_count=outputs, core_signature=signature,
        numpy_version=version if type(version) is str and len(version) <= 128 else None,
        public_core_export=None)
    if vars(numpy).get(name) is not value:
        return result
    for module_name in CORE_MODULES:
        module = sys.modules.get(module_name)
        if type(module) is not types.ModuleType or vars(module).get(name) is not value:
            continue
        members = vars(module)
        spec, origin = members.get('__spec__'), members.get('__file__')
        if (type(spec) is ModuleSpec and type(spec.loader) is ExtensionFileLoader
                and spec.name == module_name and type(origin) is str and spec.origin == origin):
            result['public_core_export'] = dict(module=module_name, symbol=name, extension_origin=origin)
            break
    return result


class _Sidecar:
    def __init__(self, base, site_ids):
        if (base is None or base.plan is None or base.closed or base.stopped
                or not base.hook_installed or not base._owns() or base.roots
                or base.plan['max_targets_per_site'] != 1):
            raise ValueError('sidecar requires an unstarted target and original single-attempt CALL capture')
        self.base, self.monitor = base, base.monitor
        self.pid = base.pid
        self.sites = {key: row['id'] for key, row in base.sites.items() if row['id'] in site_ids}
        if (not self.sites or set(self.sites.values()) != set(site_ids) or len(self.sites) != len(site_ids)
                or len(self.sites) > 32):
            raise ValueError('select one to thirty-two unique original CALL site IDs')
        self.keys = {key for key, _ in self.sites}; self.codes = {}; self.rows = {}; self.closed = False
        self.errors = []
        self.name = 'scar.numpy-ufunc-call.'+str(self.pid)
        self.tool = next((tool for tool in range(6) if self.monitor.get_tool(tool) is None), None)
        if self.tool is None:
            raise ValueError('no independent monitoring slot for the ufunc CALL sidecar')
        self.monitor.use_tool_id(self.tool, self.name)
        self.monitor.register_callback(self.tool, self.monitor.events.CALL, self.call)
        sys.addaudithook(self.audit)

    def owns_slot(self):
        return (os.getpid() == self.pid and self.monitor.get_tool(self.tool) == self.name
            and self.monitor.get_events(self.tool) == self.monitor.get_events(self.base.tool_id) == 0)

    def audit(self, event, args):
        if self.closed or event != 'exec' or not args or self.base.roots.get(id(args[0])) is not args[0]:
            return
        try:
            if not self.owns_slot() or not self.base._owns():
                raise ValueError('monitoring tool changed')
            for identity, code in self.base.codes.items():
                if identity not in self.codes and code_key(code) in self.keys:
                    self.codes[identity] = code
                    self.monitor.set_local_events(self.tool, code, self.monitor.events.CALL)
        except Exception as error:
            self.errors.append(type(error).__name__)
            self.close()

    def call(self, code, offset, value, _arg0):
        if self.closed or not self.owns_slot() or self.codes.get(id(code)) is not code:
            return self.monitor.DISABLE
        key = code_key(code), offset
        site = self.sites.get(key)
        if site is not None and site not in self.rows:
            try:
                if sys._getframe(1).f_code is not code:
                    raise ValueError('CALL frame changed')
                self.rows[site] = dict(site_id=site, caller_code_key=key[0], offset=offset,
                    thread_id=threading.get_ident(), target_type=_type_metadata(type(value)),
                    ufunc=ufunc_metadata(value))
            except Exception as error:
                self.errors.append(type(error).__name__)
        return self.monitor.DISABLE

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.monitor.get_tool(self.tool) != self.name:
            self.errors.append('MONITORING_SLOT_OWNERSHIP_LOST')
            return
        for code in self.codes.values():
            self.monitor.set_local_events(self.tool, code, 0)
        self.monitor.register_callback(self.tool, self.monitor.events.CALL, None)
        self.monitor.free_tool_id(self.tool)

    def document(self):
        return dict(schema=SCHEMA, schema_version=1, plan_sha256=plan_digest(self.base.plan),
            process_id=self.pid, sites=sorted(self.sites.values()), rows=list(self.rows.values()),
            errors=self.errors, evidence_scope='FIRST_LOCAL_CALL_PER_SELECTED_SITE',
            global_events_enabled=False, operand_values_collected=False, callee_completed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('entry', type=Path)
    parser.add_argument('args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('fresh ufunc sidecar publication required')
    entry = args.entry.resolve(strict=True)
    addon = _Sidecar(install(), args.site)
    old_argv, old_path = sys.argv, sys.path[:]
    try:
        sys.argv = [str(entry), *args.args]
        sys.path.insert(0, str(entry.parent))
        runpy.run_path(str(entry), run_name='__main__')
    finally:
        sys.argv, sys.path[:] = old_argv, old_path
        addon.close()
        if os.getpid() == addon.pid:
            with args.out.open('x') as stream:
                json.dump(addon.document(), stream, sort_keys=True, indent=2, allow_nan=False)
                stream.write('\n')


if __name__ == '__main__':
    main()
