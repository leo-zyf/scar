"""Attach an admitted loop while executing one complete original entry code.

Only this module code receives local instruction events. Static STORE offsets
remain candidates: an observed normal edge and the live hierarchical function
guards must both succeed before a binding changes. This is not a call trace or
a claim of complete early-call/alias coverage.
"""
from contextlib import contextmanager
import dis
from functools import wraps
from hashlib import sha256
from pathlib import Path
import sys
from types import ModuleType

from scar.trace.code_snapshot import code_key
from .entry_runner import _Scope, _arguments, _script_state, run_original_script
from .loop_contract import canonical_sha256


_TOOL_NAME = "scar-main-attachment-v1"
_MAX_INSTRUCTIONS = 100_000


def _prepare(entry, artifact, plan):
    contract = artifact.contract
    if (type(plan) is not dict or plan.get("schema") != "scar.entry-attachment-plan"
            or type(plan.get("schema_version")) is not int or plan["schema_version"] != 1
            or plan.get("entry") != str(entry)
            or plan.get("runtime_contract_sha256") != canonical_sha256(contract)):
        raise ValueError("entry_attachment_contract_mismatch")
    if Path(contract["source"]["path"]).resolve() != entry:
        raise ValueError("main_attachment_requires_entry_owned_loop")
    expected_interpreter = dict(implementation=sys.implementation.name,
        version=list(sys.version_info[:3]), cache_tag=sys.implementation.cache_tag)
    if plan.get("interpreter") != expected_interpreter:
        raise ValueError("module_compile_interpreter_mismatch")
    rows = [row for row in plan.get("modules", ()) if row.get("path") == str(entry)]
    if len(rows) != 1:
        raise ValueError("one_exact_entry_module_required")
    row = rows[0]
    if (row.get("mode") != "ENTRY_MAIN_LOCAL_POST_STORE_CANDIDATES"
            or row.get("attachment_target") is not True or row.get("frontiers")):
        raise ValueError("main_module_identity_unresolved")
    recipe = dict(filename=str(entry), mode="exec", flags=0, dont_inherit=True,
                  optimize=sys.flags.optimize)
    if row.get("compile_recipe") != recipe:
        raise ValueError("whole_module_compile_recipe_mismatch")
    limit = plan.get("limits", {}).get("max_bytes")
    if type(limit) is not int or not 0 < limit <= 8 * 1024 * 1024:
        raise ValueError("bounded_module_source_required")
    with entry.open("rb") as source:
        raw = source.read(limit + 1)
    if (len(raw) > limit or sha256(raw).hexdigest() != row.get("source_sha256")
            or row["source_sha256"] != contract["source"]["sha256"]):
        raise ValueError("whole_module_source_identity_mismatch")
    code = compile(raw, **recipe)
    if code_key(code) != row.get("module_code_key"):
        raise ValueError("whole_module_code_identity_mismatch")
    names = {contract["source"]["qualname"]}
    names.update(guard["environment_path"][0] for guard in contract["callee_guards"])
    body_source = next(guard["source"] for guard in contract["callee_guards"]
                       if len(guard["environment_path"]) == 1)
    if Path(body_source["path"]).resolve() == entry:
        names.update(guard["environment_path"][-1] for guard in contract["callee_guards"]
                     if len(guard["environment_path"]) == 2)
    if not all(type(name) is str and name.isidentifier() for name in names):
        raise ValueError("direct_global_function_bindings_required")
    if row.get("required_global_names") != sorted(names):
        raise ValueError("required_module_binding_set_mismatch")
    instructions = tuple(dis.get_instructions(code, adaptive=False, show_caches=False))
    normal = {before.offset: after for before, after in zip(instructions, instructions[1:])}
    by_offset = {instruction.offset: instruction for instruction in instructions}
    exceptions = tuple(dis.Bytecode(code).exception_entries)
    anchors = {}
    for anchor in row.get("anchors", ()):
        store, after = anchor.get("store_offset"), anchor.get("after_offset")
        if type(store) is not int or type(after) is not int:
            raise ValueError("exact_module_instruction_offsets_required")
        instruction, following = by_offset.get(store), normal.get(store)
        if (instruction is None or following is None or following.offset != after
                or instruction.opname not in ("STORE_NAME", "STORE_GLOBAL")
                or anchor.get("store_opname") != instruction.opname
                or anchor.get("binding") != instruction.argval or instruction.argval not in names
                or anchor.get("predecessor_execution_proven") is not False
                or anchor.get("guard_readiness_proven") is not False):
            raise ValueError("anchor_differs_from_original_module_instructions")
        # RAISE is not a local monitoring event. Exclude an exceptional edge
        # that could have the same observed offsets as normal STORE fallthrough.
        if any(item.start <= store < item.end and item.target == after for item in exceptions):
            continue
        anchors[(store, after)] = instruction.argval
    if not anchors:
        raise ValueError("no_verified_normal_store_successor_candidate")
    return code, anchors


@contextmanager
def _main_module():
    missing = object()
    previous = sys.modules.get("__main__", missing)
    module = ModuleType("__main__")
    sys.modules["__main__"] = module
    try:
        yield module
    finally:
        if previous is missing:
            sys.modules.pop("__main__", None)
        else:
            sys.modules["__main__"] = previous


def _reserve(monitoring):
    for tool in (3, 4, 0, 1, 2, 5):
        if monitoring.get_tool(tool) is not None:
            continue
        try:
            monitoring.use_tool_id(tool, _TOOL_NAME)
        except ValueError:
            continue
        if monitoring.get_events(tool) == 0:
            return tool
        monitoring.free_tool_id(tool)
    return None


def _original(entry, argv, reason):
    report = run_original_script(entry, argv)
    report["reason"] = reason
    report["frontier"] = [reason]
    report["attachment_mode"] = "MAIN_LOCAL_INSTRUCTION"
    return report


def execute_main(entry, argv, *, artifact, attachment_plan, exclusive_inputs=False):
    """Execute original bytes once; install only after live module readiness.

    The caller admits/loads the bundle separately. Unsupported monitoring or
    unresolved attachment metadata runs the original script with a frontier.
    Target exceptions and nonzero SystemExit keep their normal propagation.
    """
    entry, argv = _arguments(entry, argv)
    if exclusive_inputs is not True:
        return _original(entry, argv, "exclusive_inputs_not_explicitly_true")
    monitoring = getattr(sys, "monitoring", None)
    if sys.implementation.name != "cpython" or sys.version_info < (3, 12) or monitoring is None:
        return _original(entry, argv, "module_local_monitoring_unavailable")
    try:
        code, anchors = _prepare(entry, artifact, attachment_plan)
    except (ValueError, KeyError, TypeError, OSError, SyntaxError) as error:
        return _original(entry, argv, "main_attachment_admission_rejected:" + str(error))
    scope = _Scope(artifact)
    tool = _reserve(monitoring)
    if tool is None:
        return _original(entry, argv, "module_local_monitoring_tool_unavailable")
    state = dict(previous=None, instructions=0, observed_store_edges=0, installed=False)
    exited = False
    previous_callback = None
    try:
        with _main_module() as module:
            namespace = vars(module)

            def instruction(current_code, offset):
                if current_code is not code:
                    return
                frame = sys._getframe(1)
                if frame.f_code is not code or frame.f_globals is not namespace or frame.f_locals is not namespace:
                    state["previous"] = None
                    return
                state["instructions"] += 1
                if state["instructions"] > _MAX_INSTRUCTIONS:
                    scope.frontier.append("module_instruction_attachment_budget_exhausted")
                    monitoring.set_local_events(tool, code, 0)
                    return monitoring.DISABLE
                previous, state["previous"] = state["previous"], offset
                if (previous, offset) not in anchors:
                    return
                state["observed_store_edges"] += 1
                original = namespace.get(scope.name)
                try:
                    guarded = scope.make_guard(original, artifact, exclusive_inputs=True)
                except (scope.mismatch, ValueError, KeyError, OSError):
                    return  # Later original STOREs may complete the environment.
                @wraps(original)
                def binding(*args, **kwargs):
                    return guarded(*args, **kwargs) if scope.active else original(*args, **kwargs)
                binding.scar_stats = guarded.scar_stats
                binding.scar_artifact = artifact
                namespace[scope.name] = binding
                scope.bindings.append(dict(module=module, module_name="__main__", original=original,
                    binding=binding, stats=guarded.scar_stats, natural_import=False))
                state["installed"] = True
                monitoring.set_local_events(tool, code, 0)
                return monitoring.DISABLE

            previous_callback = monitoring.register_callback(tool, monitoring.events.INSTRUCTION, instruction)
            # The temporary main has no __file__ during preloaded-module scan.
            # The original full module is then executed with normal script globals.
            with _script_state(entry, argv, scope):
                namespace.update(__file__=str(entry), __cached__=None, __loader__=None,
                                 __package__="", __spec__=None)
                monitoring.set_local_events(tool, code, monitoring.events.INSTRUCTION)
                try:
                    exec(code, namespace, namespace)
                except SystemExit as error:
                    status = error.code
                    if status is not None and not (isinstance(status, int) and int.__int__(status) == 0):
                        raise
                    exited = True
                finally:
                    monitoring.set_local_events(tool, code, 0)
    finally:
        if monitoring.get_tool(tool) == _TOOL_NAME:
            monitoring.set_local_events(tool, code, 0)
            monitoring.register_callback(tool, monitoring.events.INSTRUCTION, previous_callback)
            monitoring.free_tool_id(tool)
    if not state["installed"]:
        scope.frontier.append("no_observed_store_edge_with_complete_live_guard_readiness")
    statistics = dict(native_calls=0, fallback_calls=0, reasons={})
    for row in scope.bindings:
        for name in ("native_calls", "fallback_calls"):
            statistics[name] += row["stats"][name]
        for reason, count in row["stats"]["reasons"].items():
            statistics["reasons"][reason] = statistics["reasons"].get(reason, 0) + count
    return dict(schema="scar.entry-runner-execution", schema_version=1,
        entry_path=str(entry), argv=argv, decision="GUARDED_ATTACHMENT" if state["installed"] else "NO_OP",
        reason="original_module_local_store_boundary", attachment_mode="MAIN_LOCAL_INSTRUCTION",
        statistics=statistics, installed_bindings=len(scope.bindings),
        observed_boundary_calls=statistics["native_calls"] + statistics["fallback_calls"],
        boundary_calls_observation_complete=False,
        boundary_observation_scope="INSTALLED_BINDING_ONLY_NO_COMPLETE_EARLY_CALL_OR_ALIAS_TRACE",
        attachments=[dict(source=str(entry), module=row["module_name"], natural_import=row["natural_import"])
                     for row in scope.bindings],
        frontier=scope.frontier, system_exit_zero=exited, exit_code=0, full_program_cpp=False,
        monitoring=dict(scope="EXACT_ORIGINAL_MODULE_CODE_ONLY", instruction_events=state["instructions"],
            observed_normal_store_edges=state["observed_store_edges"], tools_released=True,
            function_events_enabled=False, source_rewritten=False, module_code_key=code_key(code)))


__all__ = ["execute_main"]
