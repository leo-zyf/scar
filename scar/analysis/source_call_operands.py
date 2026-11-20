"""Query evaluated call operands and their original F4 input edges."""

from dataclasses import dataclass

from scar.analysis.source_reference_consumers import argument_points
from scar.analysis.source_reference_slice import source_port
from scar.ir.program.dependencies import DependencyEdge, DependencyKind
from scar.ir.program.normal_graph import PortKind
from scar.ir.program.reference_consumers import ArgumentRole


@dataclass(frozen=True, slots=True)
class SourceCallOperand:
    role: ArgumentRole
    ordinal: int
    keyword: str | None
    edge: DependencyEdge


def call_operands(view, unit, call_index, *, context=None):
    """Return reference operand routes in the original call ABI order."""
    provider = view.unit(unit)
    facts = provider.facts
    if type(call_index) is not int or not 0 <= call_index < len(facts.calls):
        raise ValueError('expected an original call fact ordinal')
    call = facts.calls[call_index]
    result = view.source_port(source_port(facts, call.result), context=context)
    operation = view.get(result).owner
    inputs = {}
    for point in provider.expand(operation):
        port = view.original(point)
        if getattr(port, 'kind', None) is PortKind.ARGUMENT:
            inputs[port.ordinal] = point
    operands = []
    for index, (role, ordinal, keyword, value) in enumerate(argument_points(call)):
        actual = view.source_port(source_port(facts, value), context=context)
        use = inputs.get(index)
        if use is None:
            raise ValueError('original call operand lacks its declared input port')
        edges = view.dependencies.incoming(use, kinds=(DependencyKind.VALUE_USE,)).edges
        if len(edges) != 1 or edges[0].source != actual:
            raise ValueError('original call fact disagrees with its value-to-operand edge')
        operands.append(SourceCallOperand(role, ordinal, keyword, edges[0]))
    return tuple(operands)
