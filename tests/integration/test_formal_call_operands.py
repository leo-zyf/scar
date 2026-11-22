"""Replay original F4 operands without requiring a workload adapter."""

from scar.analysis.program_view import ProgramView
from scar.analysis.call_inputs import call_operands
from scar.analysis.source_frontend import build_frontend
from scar.analysis.source_reference_slice import source_port
from scar.compiler.call_binding import plan_call_bindings
from scar.ir.effects import Knowledge
from scar.ir.program.dependencies import DependencyKind as K
from scar.ir.program.binding_ports import ArgumentRole
from scar.ir.program.content_routes import ContentRequestKind, WholeOwner
from scar.ir.program.source_productions import ApplyInplaceOperator, ApplyPythonProtocol, ProductionStep
from scar.ir.program.structured import PythonProtocolKind


def test_call_operands_preserve_compile_time_strings_and_live_formatting(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    source = root / 'entry.py'
    source.write_text('''
def consume(text):
    return text

def render(name):
    consume("caption")
    consume(f"caption")
    consume(text=f"caption {name}")

def accumulate(view, delta):
    view += delta
    return view
''', encoding='utf-8')
    frontend = tmp_path / 'frontend'
    report = build_frontend((root,), frontend, sources=(source,))
    assert report['workflow_complete'] and not report['target_executed']

    view = ProgramView(frontend / 'catalog')
    binding = plan_call_bindings(view, source, entry='render')
    sites = sorted(binding['sites'], key=lambda row: row['site']['position'])
    assert len(sites) == 3 and not binding['frontiers']
    for line, site in enumerate(sites, start=6):
        assert site['site']['position'][0] == line
        facts = view.unit(site['unit']).facts
        _, operand = call_operands(view, site['unit'], site['query'])
        assert operand.ordinal == 0
        assert (operand.role, operand.keyword) == (
            (ArgumentRole.KEYWORD, 'text') if line == 8 else (ArgumentRole.POSITIONAL, None))
        assert view.original(operand.edge.target).owner.wire == site['original_step']
        assert operand.edge in view.consumers(operand.edge.source).edges
        reach = view.dependencies.walk((operand.edge.source,), direction='incoming',
            kinds=(K.VALUE_USE, K.RESULT, K.OPERAND), max_points=100)
        records = [view.original(point) for point in reach.points]
        protocols = {step.instruction.protocol for step in records if type(step) is ProductionStep
                     and type(step.instruction) is ApplyPythonProtocol}
        constants = {row.literal.to_python() for row in facts.constants if
                     view.source_port(source_port(facts, row.value)) in reach.points}
        if line < 8:
            assert constants == {'caption'} and not protocols
        else:
            assert 'caption' not in constants and 'caption ' in constants
            assert {PythonProtocolKind.FSTRING_FORMAT, PythonProtocolKind.BUILD_STRING} <= protocols
            assert any(not row.write and row.namespace.name == 'name' and
                       view.source_port(source_port(facts, row.value)) in reach.points for row in facts.names)

    unit = sites[0]['unit']
    provider = view.unit(unit)
    function, = [fn for fn in provider.facts.functions if fn.parameter_names == ('view', 'delta')]
    key, step = next((key, step) for key, plan in provider.material.closure(function.template).items()
                     for step in plan.steps if type(step.instruction) is ApplyInplaceOperator)
    operation = provider._node(key, step.id)
    content = view.state(unit).content(complete=False)
    index, = content.ensure_operation(operation)
    request = content.accesses[index]
    assert request.kind is ContentRequestKind.UPDATE and request.result is not None
    assert isinstance(content.addresses[request.address].selector, WholeOwner)
    assert request.payload != request.result
    assert view.effects(operation).writes.knowledge is Knowledge.UNKNOWN
