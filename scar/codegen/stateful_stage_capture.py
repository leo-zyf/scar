"""Capture existing stage source and entry state without changing the backend.

Only the trusted host-side renderer receives two read-only hooks. Generated
C++ must match an independent call to the untouched renderer byte for byte.
No target entry, compiler, native stage or sampler is run by this module.
"""
import ast
from copy import deepcopy
from hashlib import sha256
from pathlib import Path


SCHEMA = 'scar.stateful-stage-source-capture'
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
_ENTRY = '_scar_stage_capture_entry'
_EXIT = '_scar_stage_capture_exit'


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _pin(path):
    raw = path.read_bytes()
    return dict(path=str(path.relative_to(_ROOT)), sha256=sha256(raw).hexdigest(), bytes=len(raw))


def _same_ast(left, right):
    return ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False)


def _hook(name, arguments, location):
    return ast.copy_location(ast.Expr(value=ast.Call(func=ast.Name(id=name, ctx=ast.Load()),
        args=[ast.Name(id=argument, ctx=ast.Load()) for argument in arguments], keywords=[])), location)


def _instrument(source, path):
    parsed = ast.parse(source, filename=str(path))
    functions = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == '_source']
    _need(len(functions) == 1 and not functions[0].decorator_list, 'one undecorated original _source renderer required')
    original = functions[0]
    _need(not any(isinstance(node, ast.Name) and node.id in (_ENTRY, _EXIT) for node in ast.walk(original)),
          'capture hook names collide with the original renderer')
    function = deepcopy(original)
    expected_loop = ast.parse("for stage in plan['stages']:\n    pass").body[0]
    loops = [node for node in function.body if isinstance(node, ast.For)
             and _same_ast(node.target, expected_loop.target) and _same_ast(node.iter, expected_loop.iter)]
    _need(len(loops) == 1, 'one original stage-generation loop required')
    loop = loops[0]
    skip = ast.parse("if kind == 'COMMIT':\n    continue").body[0]
    skips = [index for index, node in enumerate(loop.body) if _same_ast(node, skip)]
    assignment = ast.parse("current_exprs = set(stage['expr_ids'])").body[0]
    entries = [index for index, node in enumerate(loop.body) if _same_ast(node, assignment)]
    extend = ast.parse('stage_text.extend(body)').body[0]
    exits = [index for index, node in enumerate(loop.body) if _same_ast(node, extend)]
    _need(len(skips) == len(entries) == len(exits) == 1 and skips[0] < entries[0] < exits[0],
          'unique non-COMMIT entry and stage-body append anchors required')
    entry_anchor, exit_anchor = loop.body[entries[0]], loop.body[exits[0]]
    entry = _hook(_ENTRY, ('stage', 'available', 'scalar_ready', 'views', 'shapes'), entry_anchor)
    exit = _hook(_EXIT, ('stage', 'body', 'scalar_ready'), exit_anchor)
    amended = []
    for index, node in enumerate(loop.body):
        if index == exits[0]:
            amended.append(exit)
        amended.append(node)
        if index == entries[0]:
            amended.append(entry)
    loop.body = amended
    # Remove precisely our two inserts and compare every original mathematical
    # and control AST node, including closure evaluation and release handling.
    check = deepcopy(function)
    check_loop = next(node for node in check.body if isinstance(node, ast.For)
                      and _same_ast(node.target, expected_loop.target) and _same_ast(node.iter, expected_loop.iter))
    check_loop.body = [node for node in check_loop.body if not (
        isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name) and node.value.func.id in (_ENTRY, _EXIT))]
    _need(_same_ast(original, check), 'capture transformation changed original renderer AST')
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    receipt = dict(original_function_lines=[original.lineno, original.end_lineno],
        entry_anchor_line=entry_anchor.lineno, exit_anchor_line=exit_anchor.lineno,
        original_ast_sha256=sha256(ast.dump(original, include_attributes=False).encode()).hexdigest(),
        transformed_ast_sha256=sha256(ast.dump(function, include_attributes=False).encode()).hexdigest(),
        insertions=2, unique_stage_loop=True, non_commit_only=True,
        entry_after='current_exprs = set(stage[expr_ids])', exit_before='stage_text.extend(body)',
        removing_capture_hooks_restores_original_ast=True, mathematical_ast_preserved=True)
    return module, receipt


def capture_stateful_stage_sources(region, plan, binding, contract):
    """Capture serial stage fragments from caller-admitted backend inputs.

    Admission of source/plan/binding belongs to the caller. This is not a new
    semantic proof or a numerical/performance observation of the source stage.
    """
    from scar.codegen import stateful
    _need(type(plan) is dict and type(plan.get('cpu_threads')) is int and plan['cpu_threads'] == 1
          and len(binding.cpu_ids) == 1, 'capture currently supports the original serial stateful renderer only')
    backend = Path(stateful.__file__).resolve()
    _need(backend.parent == _HERE and Path(stateful._HERE).resolve() == _HERE,
          'the original local stateful backend and templates are required')
    paths = (Path(__file__).resolve(), backend, _HERE/'stateful_v1.cpp.in', _HERE/'loop_v1.cpp.in',
             _HERE/'stateful_abi_v1.h', _HERE/'scar_abi_v1.h')
    pins = [_pin(path) for path in paths]
    module, ast_receipt = _instrument(backend.read_text(), backend)
    stages = []
    active = None

    def enter(stage, available, scalar_ready, views, shapes):
        nonlocal active
        _need(active is None and stage.get('kind') != 'COMMIT'
              and type(stage.get('index')) is int, 'stage capture entry order differs from the original plan')
        active = dict(index=stage['index'], entry_available=deepcopy(available),
            entry_scalar_ready=sorted(scalar_ready), views=deepcopy(views), shapes=deepcopy(shapes))

    def leave(stage, body, scalar_ready):
        nonlocal active
        _need(active is not None and active['index'] == stage['index']
              and type(body) is list and all(type(line) is str for line in body),
              'stage capture exit differs from the original entry/body')
        text = '\n'.join(body)
        stages.append(dict(**active, body=text, body_sha256=sha256(text.encode()).hexdigest(),
                           exit_scalar_ready=sorted(scalar_ready)))
        active = None

    namespace = dict(stateful._source.__globals__)
    namespace.update({_ENTRY: enter, _EXIT: leave})
    # Compile only a trusted Python renderer copy, not the target program or
    # generated C++. The original module globals and function are untouched.
    exec(compile(module, str(backend), 'exec', dont_inherit=True), namespace)
    captured = namespace['_source'](region, plan, binding, contract)
    original = stateful._source(region, plan, binding, contract)
    _need(active is None and captured == original, 'captured C++ differs from the unmodified backend renderer')
    expected = [stage['index'] for stage in plan['stages'] if stage['kind'] != 'COMMIT']
    _need([stage['index'] for stage in stages] == expected and len(set(expected)) == len(expected),
          'capture must cover every original non-COMMIT stage exactly once')
    _need(pins == [_pin(path) for path in paths], 'backend/capture source changed during rendering')
    digest = sha256(original.encode()).hexdigest()
    return dict(schema=SCHEMA, schema_version=1, full_source_sha256=digest, stages=stages,
        provenance=dict(source_pins=pins, ast_transform=ast_receipt,
            body_encoding='ORIGINAL_BODY_LINES_JOINED_WITH_NEWLINE_WITHOUT_OUTER_12_SPACE_INDENT',
            snapshots='DEEPCOPIED_HOST_CODEGEN_METADATA_NOT_RUNTIME_TENSOR_CONTENTS',
            full_source_equal_to_original_renderer=True, unmodified_full_source_sha256=digest,
            stage_order=expected, caller_supplied_source_and_plan_admission_required=True,
            host_renderer_ast_instrumented=True, target_program_executed=False,
            native_compiled=False, performance_measured=False, optimization_authorized=False))


__all__ = ['SCHEMA', 'capture_stateful_stage_sources']
