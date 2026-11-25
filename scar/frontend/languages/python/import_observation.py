"""Record the actual module-level member bindings of one from-import in CPython."""

import argparse
import dis
from hashlib import sha256
import os
from pathlib import Path
import sys
from types import CodeType, FunctionType, ModuleType
import warnings

from scar.contract import write_json
from .environment import code_digest, run_script


def _binding(namespace, name):
    if name not in namespace:
        return {"kind": "absent"}
    value = namespace[name]
    if type(value) is FunctionType or type(value) is type:
        if type(value) is FunctionType:
            kind, module, name = "function", value.__module__, value.__qualname__
        else:
            kind = "type"
            module = type.__dict__["__module__"].__get__(value)
            name = type.__dict__["__qualname__"].__get__(value)
        if all(
            type(label) is str and 0 < len(label) <= 4096 for label in (module, name)
        ):
            # These mutable labels describe a sample, not a callable identity.
            return dict(kind=kind, module=module, qualname=name)
    if type(value) is ModuleType:
        module = vars(value).get("__name__")
        if type(module) is str:
            return {
                "kind": "module",
                "name": module,
                "same_registry_instance": (
                    sys.modules.get(module) is value
                    if type(sys.modules) is dict
                    else None
                ),
            }
    if (type(value) is int or type(value) is bool) and value.bit_length() <= 64:
        return {"kind": type(value).__name__, "value": value}
    return {"kind": "none" if value is None else "other"}


class ImportBindingObserver:
    """Monitor only a matching code object's instructions. No import is replayed."""

    def __init__(self, source, line, limit):
        self.source = source.resolve(strict=True)
        self.data = self.source.read_bytes()
        self.line, self.limit = line, limit
        self.pid = os.getpid()
        self.tool = None
        self.codes = {}
        self.events = []
        self.dropped = 0

    def audit(self, event, args):
        if event != "exec" or self.tool is None or self.pid != os.getpid():
            return
        code = args[0]
        if type(code) is not CodeType or code.co_name != "<module>":
            return
        if os.path.basename(code.co_filename) != self.source.name:
            return
        if code.co_filename != str(self.source) and (
            code.co_filename.startswith("<")
            or Path(code.co_filename).resolve() != self.source
        ):
            return
        # Recompilation identifies a loaded code object. Diagnostics from this
        # shadow compilation must not be emitted a second time by the observer.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                expected = compile(
                    self.data,
                    code.co_filename,
                    "exec",
                    dont_inherit=True,
                    optimize=sys.flags.optimize,
                )
        except (SyntaxError, ValueError):
            return
        if code_digest(code) != code_digest(expected):
            return
        instructions = list(dis.get_instructions(code))
        points = {}
        for i, ins in enumerate(instructions):
            if (
                ins.opname != "IMPORT_NAME"
                or ins.positions.lineno != self.line
                or ins.positions.col_offset is None
            ):
                continue
            pos = ins.positions
            site = dict(
                line=pos.lineno,
                end_line=pos.end_lineno,
                column=pos.col_offset + 1,
                end_column=pos.end_col_offset + 1,
            )
            j = i + 1
            index = 0
            while j + 2 < len(instructions):
                member, store, after = instructions[j : j + 3]
                if (
                    member.opname != "IMPORT_FROM"
                    or store.opname not in ("STORE_NAME", "STORE_GLOBAL")
                    or after.is_jump_target
                    or after.opname not in {"IMPORT_FROM", "POP_TOP"}
                ):
                    break
                point = dict(
                    location=site,
                    binding_index=index,
                    member=member.argval,
                    binding=store.argval,
                )
                points.setdefault(after.offset, []).append(point)
                j += 2
                index += 1
        if points:
            self.codes[code] = points
            sys.monitoring.set_local_events(
                self.tool, code, sys.monitoring.events.INSTRUCTION
            )

    def instruction(self, code, offset):
        points = self.codes.get(code, {}).get(offset)
        if not points:
            return sys.monitoring.DISABLE
        if self.pid != os.getpid():
            return
        frame = sys._getframe(1)
        if (
            frame.f_code is not code
            or frame.f_locals is not frame.f_globals
            or type(frame.f_globals) is not dict
        ):
            return
        owner = frame.f_globals.get("__name__")
        if type(owner) is not str:
            return
        for point in points:
            if len(self.events) >= self.limit:
                self.dropped += 1
            else:
                self.events.append(
                    dict(
                        **point,
                        owner=owner,
                        result=_binding(frame.f_globals, point["binding"]),
                    )
                )

    def start(self):
        monitor = getattr(sys, "monitoring", None)
        if monitor is None or sys.implementation.name != "cpython":
            raise ValueError("import binding capture requires CPython 3.12+")
        for tool in (3, 4, 2, 1, 0, 5):
            if monitor.get_tool(tool) is None:
                monitor.use_tool_id(tool, "scar.import-observation")
                self.tool = tool
                break
        if self.tool is None:
            raise ValueError("no free CPython monitoring slot")
        monitor.register_callback(
            self.tool, monitor.events.INSTRUCTION, self.instruction
        )
        sys.addaudithook(self.audit)

    def stop(self):
        if self.pid != os.getpid() or self.tool is None:
            return
        tool, self.tool = self.tool, None
        for code in self.codes:
            sys.monitoring.set_local_events(tool, code, 0)
        sys.monitoring.register_callback(tool, sys.monitoring.events.INSTRUCTION, None)
        sys.monitoring.free_tool_id(tool)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--line", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("script")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.line < 1 or not 1 <= args.limit <= 4096:
        parser.error("expected a positive line and an event limit in [1,4096]")
    out = args.out.resolve()
    entry = Path(args.script).resolve(strict=True)
    data = entry.read_bytes()
    observer = ImportBindingObserver(args.source, args.line, args.limit)
    observer.start()
    try:
        run_script(args.script, args.arguments, os.getcwd(), data=data)
    finally:
        observer.stop()
        if observer.pid == os.getpid():
            write_json(
                out,
                dict(
                    schema="scar.python-import-observation",
                    schema_version=1,
                    scope="instrumented_script_successful_module_bindings",
                    interpreter=f"{sys.implementation.name} {sys.version.split()[0]}",
                    entry_sha256=sha256(data).hexdigest(),
                    source_sha256=sha256(observer.data).hexdigest(),
                    source=str(observer.source),
                    line=observer.line,
                    events=observer.events,
                    dropped=observer.dropped,
                ),
            )


if __name__ == "__main__":
    main()
