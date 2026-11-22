"""Common input builders for retained compiler tests."""

from scar.analysis.source_expression_rules import SourceExpressionRules
from scar.analysis.source_productions import ProductionSourceIndex
from scar.ir.program import source_productions as P
from scar.ir.program.ids import LocationID
from scar.ir.program.model import ValueKind, ValueType
from scar.ir.program.protocol_signatures import ExceptionABI
WORLD = ValueType(ValueKind.STATE, location=LocationID('test-source-world'))

REF = ValueType(ValueKind.REFERENCE)


def _documents(text,mode='exec'):
    from scar.analysis.source_bytes import source_bytes_document
    from scar.analysis.source_tokens import source_tokens_document
    from scar.analysis.source_syntax import source_syntax_document
    from scar.analysis.source_role_materials import source_roles_document
    from scar.analysis.source_contexts import source_contexts_document
    from scar.analysis.source_declarations import source_declarations_document
    from scar.analysis.source_bindings import source_bindings_document
    from scar.analysis.source_symbols import source_symbols_document
    from scar.analysis.compiler_topology import compiler_topology_document
    from scar.analysis.compiler_symbols import compiler_symbols_document
    f0 = source_bytes_document(text.encode(), path='/source/schedules.py', namespace_root='/source')
    f1 = source_tokens_document(f0)
    f2 = source_syntax_document(f0, f1,mode=mode)
    b = source_roles_document(f0, f1, f2)
    c = source_contexts_document(f0, f1, f2, b)
    d = source_declarations_document(f0, f1, f2, b, c)
    q = source_bindings_document(f0, f1, f2, b, c, d)
    a = source_symbols_document(f0, f1, f2)
    t = compiler_topology_document(f0, f1, f2, b, c, a)
    s = compiler_symbols_document(f0, f1, f2, b, c, d, a, t)
    return f0, f1, f2, b, c, d, q, a, t, s


def _rules(documents):
    from scar.analysis.frontend_bindings import read_frontend_bindings
    from scar.analysis.source_contexts import SourceContextView
    from scar.ir.program.source_syntax import SourceSyntaxMaterial
    from scar.ir.record_codec import decode
    f0, f1, f2, b, c, d, q, a, t, s = documents
    frontend = read_frontend_bindings(f0, f1, f2, b, c, d, q, a, t, s)
    context = SourceContextView(decode(SourceSyntaxMaterial, f2), frontend.roles, frontend.contexts)
    return SourceExpressionRules(ProductionSourceIndex(f0, context, frontend),
                                 world=WORLD, exception_abi=ExceptionABI.EXCEPTION_WORLD)

def purpose(rules, occurrence):
    kind = rules.source.context.syntax.nodes[occurrence].constructor
    if kind not in ('Name', 'Attribute', 'Subscript', 'List', 'Tuple'):
        return P.ProductionPurpose.VALUE
    context = rules.source.roles.context(occurrence)
    return {'Load': P.ProductionPurpose.VALUE, 'Store': P.ProductionPurpose.STORE,
            'Del': P.ProductionPurpose.DELETE}[context]
