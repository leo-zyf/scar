"""Lower a checked free-form Fortran subset to the fixed-form transport dialect.

The compiler establishes kinds and constant values. Fparser supplies the scope,
declaration, and statement structure. The transport source is built anew from
these inputs and never substitutes routines from an earlier upstream release.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys

from fparser.common.readfortran import FortranFileReader
from fparser.two import Fortran2003 as F
from fparser.two.parser import ParserFactory
from fparser.two.utils import walk


_SCOPES = re.compile(r'^\s*procedure name = (\w+)\s*$', re.MULTILINE)
_SYMBOLS = re.compile(r"^\s*symtree: '([^']+)'\s+\|\| symbol: '[^']+'[^\n]*\n", re.MULTILINE)
_KINDS = {('REAL', 4): 'REAL', ('REAL', 8): 'DOUBLE PRECISION',
          ('INTEGER', 4): 'INTEGER', ('LOGICAL', 4): 'LOGICAL'}
_FORTRAN_STRINGS = re.compile(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"")
_KINDED_NUMBER = re.compile(r'(?<![\w.])((?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eEdD][+-]?\d+)?)_(\w+)\b')
_SOURCE_RELATIONS = re.compile(r'<=|>=|==|/=|<|>')
_RELATIONS = {'<=': '.LE.', '>=': '.GE.', '==': '.EQ.', '/=': '.NE.', '<': '.LT.', '>': '.GT.'}
_PROGRAM_UNITS = (F.Main_Program, F.Subroutine_Subprogram, F.Function_Subprogram)
_CONTAINERS = (F.Module_Subprogram_Part, F.Internal_Subprogram_Part)
_STATEMENTS = {
    'Assignment_Stmt', 'Block_Label_Do_Construct', 'Block_Nonlabel_Do_Construct', 'Call_Stmt',
    'Continue_Stmt', 'Else_If_Stmt', 'Else_Stmt', 'End_Do_Stmt', 'End_If_Stmt', 'Format_Stmt',
    'Goto_Stmt', 'If_Construct', 'If_Stmt', 'If_Then_Stmt', 'Label_Do_Stmt', 'Nonlabel_Do_Stmt',
    'Read_Stmt', 'Return_Stmt', 'Stop_Stmt', 'Write_Stmt',
}


@dataclass(frozen=True)
class Symbol:
    type_and_kind: tuple[str, int]
    attributes: str
    value: str | None

    def type_name(self):
        if self.type_and_kind not in _KINDS:
            raise ValueError(f'unsupported Fortran type and kind {self.type_and_kind}')
        return _KINDS[self.type_and_kind]


@dataclass(frozen=True)
class Procedure:
    name: str
    node: object
    specification: tuple
    execution: tuple
    symbols: dict


def _symbol_tables(dump):
    scopes = list(_SCOPES.finditer(dump))
    tables = {}
    imports = []
    for index, scope in enumerate(scopes):
        end = scopes[index + 1].start() if index + 1 < len(scopes) else len(dump)
        text = dump[scope.end():end].split('\n  code:', 1)[0]
        entries = list(_SYMBOLS.finditer(text))
        symbols = {}
        for item_index, item in enumerate(entries):
            tail = entries[item_index + 1].start() if item_index + 1 < len(entries) else len(text)
            fields = text[item.end():tail]
            type_spec = re.search(r'^\s*type spec\s*:\s*\((\w+)\s+(\d+)\)', fields, re.MULTILINE)
            if not type_spec:
                origin = re.search(r"from namespace '(\w+)'", item[0])
                if origin:
                    imports.append((scope[1].lower(), item[1].lower(), origin[1].lower()))
                continue
            attributes = re.search(r'^\s*attributes:\s*(.*)$', fields, re.MULTILINE)
            value = re.search(r'^\s*value:\s*(.*)$', fields, re.MULTILINE)
            symbols[item[1].lower()] = Symbol((type_spec[1], int(type_spec[2])),
                                             attributes[1] if attributes else '', value[1] if value else None)
        tables[scope[1].lower()] = symbols
    for scope, name, origin in imports:
        inherited = tables.get(origin, {}).get(name)
        if inherited is not None:
            tables[scope][name] = inherited
    return tables


def _source_text(node, symbols):
    def convert(fragment):
        def number(match):
            value, kind = match.groups()
            if not kind.isdecimal():
                fact = symbols.get(kind.lower())
                if not fact or not fact.value or not fact.value.isdecimal():
                    raise ValueError(f'missing compiler-resolved Fortran kind {kind}')
                kind = fact.value
            if kind == '4':
                return value
            if kind == '8' and re.search(r'[.eEdD]', value):
                return re.sub(r'[eEdD]', 'D', value) if re.search(r'[eEdD]', value) else value + 'D0'
            raise ValueError(f'unsupported Fortran literal kind in {match[0]}')

        fragment = _KINDED_NUMBER.sub(number, fragment)
        fragment = _SOURCE_RELATIONS.sub(lambda match: _RELATIONS[match[0]], fragment)
        # Preserve conversions to compiler-resolved real kinds in the fixed dialect.
        def real_conversion(match):
            value, kind = match.groups()
            fact = symbols.get(kind.lower())
            kind = fact.value if fact else kind
            if kind not in {'4', '8'}:
                raise ValueError(f'unsupported real conversion kind {kind}')
            return ('DBLE' if kind == '8' else 'REAL') + '(' + value + ')'

        return re.sub(r'\bREAL\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)', real_conversion,
                      fragment, flags=re.IGNORECASE)

    source = str(node)
    chunks = []
    cursor = 0
    for literal in _FORTRAN_STRINGS.finditer(source):
        chunks.extend((convert(source[cursor:literal.start()]), literal[0]))
        cursor = literal.end()
    chunks.append(convert(source[cursor:]))
    return ''.join(chunks)


def _literal(value, symbol):
    if symbol.type_and_kind == ('REAL', 8):
        value = re.sub(r'_8\b', '', value)
        if re.search(r'[eEdD]', value):
            return re.sub(r'[eEdD]', 'D', value)
        return value + ('D0' if '.' in value else '.0D0')
    return re.sub(r'_4\b', '', value)


def _parameter_lines(symbols, used):
    for name in sorted(used & symbols.keys()):
        symbol = symbols[name]
        if 'PARAMETER' not in symbol.attributes:
            continue
        if not symbol.value:
            raise ValueError(f'missing compiler-resolved Fortran parameter {name}')
        if symbol.value.startswith('(/') and symbol.value.endswith('/)'):
            values = [_literal(value.strip(), symbol) for value in symbol.value[2:-2].split(',')]
            yield f'{symbol.type_name()} {name}({len(values)})'
            yield f'DATA {name} /{", ".join(values)}/'
        else:
            yield f'{symbol.type_name()} {name}'
            yield f'PARAMETER ({name} = {_literal(symbol.value, symbol)})'


def _collect_procedures(node, tables, inherited, procedures):
    if isinstance(node, F.Module):
        module = str(node.content[0].items[1]).lower()
        if module not in tables:
            raise ValueError(f'no compiler symbol table for module {module}')
        inherited = {name: symbol for name, symbol in tables[module].items()
                     if 'PARAMETER' in symbol.attributes}
    if isinstance(node, _PROGRAM_UNITS):
        name = str(node.content[0].items[1]).lower()
        if name in procedures or name not in tables:
            raise ValueError(f'open or ambiguous Fortran procedure scope {name}')
        specification = tuple(
            statement for child in node.content if isinstance(child, F.Specification_Part)
            for item in child.content
            for statement in (item.content if isinstance(item, F.Implicit_Part) else (item,)))
        execution = next((tuple(child.content) for child in node.content
                          if isinstance(child, F.Execution_Part)), ())
        procedures[name] = Procedure(name, node, specification, execution, {**inherited, **tables[name]})
    for child in getattr(node, 'content', ()):
        if isinstance(child, (*_PROGRAM_UNITS, *_CONTAINERS, F.Module)):
            _collect_procedures(child, tables, inherited, procedures)


def _declarations(procedure, known):
    symbols = procedure.symbols
    dimensions = {}
    defined = set()
    declarations = []
    for statement in procedure.specification:
        if isinstance(statement, (F.Use_Stmt, F.Implicit_Stmt)):
            continue
        if isinstance(statement, F.Procedure_Declaration_Stmt):
            declarations.append('EXTERNAL ' + str(statement.items[-1]))
            continue
        if isinstance(statement, (F.External_Stmt, F.Intrinsic_Stmt)):
            declarations.append(str(statement))
            continue
        if not isinstance(statement, F.Type_Declaration_Stmt):
            raise ValueError(f'unsupported Fortran declaration {statement}')
        _, attr_list, entities = statement.items
        attrs = attr_list.items if attr_list else ()
        shape = next((item.items[1] for item in attrs if isinstance(item, F.Dimension_Attr_Spec)), None)
        for attr in attrs:
            if not isinstance(attr, (F.Dimension_Attr_Spec, F.Intent_Attr_Spec)) and str(attr).upper() != 'PARAMETER':
                raise ValueError(f'unsupported Fortran declaration attribute {attr}')
        for entity in entities.items:
            entity_name, array, character, initializer = entity.items
            name = str(entity_name).lower()
            symbol = symbols.get(name)
            if not symbol:
                raise ValueError(f'missing compiler information for Fortran variable {name}')
            if 'PARAMETER' in symbol.attributes:
                continue
            if character or initializer:
                raise ValueError(f'unsupported Fortran declaration initializer {entity}')
            extent = array or shape
            declaration = f'{symbol.type_name()} {name}'
            if extent:
                declaration += f'({_source_text(extent, symbols)})'
                dimensions[name] = extent
            declarations.append(declaration)
            defined.add(name)
    for name in sorted(known):
        candidate = known[name]
        if name not in defined and isinstance(candidate.node, F.Function_Subprogram):
            declarations.append(f'{candidate.symbols[name].type_name()} {name}')
        if name not in defined and name not in symbols:
            declarations.append(f'EXTERNAL {name}')
        elif name not in defined and 'DUMMY' not in symbols[name].attributes:
            declarations.append(f'EXTERNAL {name}')
    return declarations, dimensions


def _all_names(parts):
    return {str(name).lower() for part in parts for name in walk(part, F.Name)}


class FixedSource:
    def __init__(self, procedure, columns):
        self.procedure = procedure
        self.columns = columns
        self.lines = []
        self.indices = []
        self.used = _all_names((*procedure.specification, *procedure.execution))

    def line(self, contents, label=None):
        self.lines.append(f'{str(label or ""):>5} {contents}')

    def emit(self, node):
        kind = type(node).__name__
        if kind not in _STATEMENTS:
            raise ValueError(f'unsupported Fortran statement {kind}. {node}')
        if getattr(node, 'content', None) is not None:
            for child in node.content:
                self.emit(child)
            return
        symbols = self.procedure.symbols
        label = getattr(getattr(node, 'item', None), 'label', None)
        if isinstance(node, F.Assignment_Stmt) and isinstance(node.items[0], F.Name):
            name = str(node.items[0]).lower()
            bounds = self.columns.get(name)
            if bounds:
                self.broadcast(node, name, bounds, label)
                return
        self.line(_source_text(node, symbols), label)

    def broadcast(self, assignment, array, bounds, label):
        rhs = assignment.items[2]
        if not isinstance(rhs, (F.Int_Literal_Constant, F.Real_Literal_Constant,
                                F.Signed_Int_Literal_Constant, F.Signed_Real_Literal_Constant,
                                F.Logical_Literal_Constant, F.Level_2_Unary_Expr)):
            raise ValueError(f'unsupported whole array assignment {assignment}')
        if walk(rhs, F.Name):
            raise ValueError(f'whole array assignment needs a scalar literal {assignment}')
        if not isinstance(bounds, F.Explicit_Shape_Spec_List):
            raise ValueError(f'whole array assignment needs explicit bounds {assignment}')
        indices = []
        for dimension in range(len(bounds.items)):
            serial = len(self.indices) + 1
            index = f'scar_i{serial}'
            while index in self.used:
                serial += 1
                index = f'scar_i{serial}'
            self.used.add(index)
            self.indices.append(index)
            indices.append(index)
        for index, extent in reversed(list(zip(indices, bounds.items))):
            first, last = extent.items
            self.line(f'DO {index} = {first or 1}, {last}', label)
            label = None
        self.line(f'{array}({", ".join(indices)}) = {_source_text(rhs, self.procedure.symbols)}')
        for _ in indices:
            self.line('END DO')


def _lower_procedure(procedure, known):
    declarations, dimensions = _declarations(procedure, known)
    source = FixedSource(procedure, dimensions)
    for node in procedure.execution:
        source.emit(node)
    start = procedure.node.content[0]
    if isinstance(start, F.Function_Stmt):
        if start.items[-1] is not None:
            raise ValueError(f'unsupported Fortran function suffix {start.items[-1]}')
        dtype = procedure.symbols[procedure.name].type_name()
        declaration = f'{dtype} FUNCTION {procedure.name}({start.items[2] or ""})'
    elif isinstance(start, F.Subroutine_Stmt):
        if start.items[0] is not None or start.items[-1] is not None:
            raise ValueError(f'unsupported Fortran subroutine modifiers {start}')
        declaration = f'SUBROUTINE {procedure.name}({start.items[2] or ""})'
    else:
        declaration = str(start)
    parameters = tuple(_parameter_lines(procedure.symbols, source.used))
    fixed = [declaration, 'IMPLICIT NONE',
             *(line for line in parameters if not line.startswith('DATA ')), *declarations]
    fixed.extend(f'INTEGER {name}' for name in source.indices)
    fixed.extend(line for line in parameters if line.startswith('DATA '))
    return '\n'.join([*('      ' + line for line in fixed), *source.lines, '      END', ''])


def main(request):
    reader = ParserFactory().create(std='f2008')
    procedures = {}
    for row in request['units']:
        tree = reader(FortranFileReader(row['source'], ignore_comments=True))
        semantics = Path(row['semantics']).read_text(encoding='utf-8')
        _collect_procedures(tree, _symbol_tables(semantics), {}, procedures)
    entries = [node.name for node in procedures.values() if isinstance(node.node, F.Main_Program)]
    if len(entries) != 1 or request['entry'].lower() not in ('main', entries[0]):
        raise ValueError('expected one supported free-form program entry')
    functions = {name for name, procedure in procedures.items() if not isinstance(procedure.node, F.Main_Program)}
    pending = entries.copy()
    output = {}
    while pending:
        name = pending.pop()
        if name in output:
            continue
        procedure = procedures[name]
        calls = _all_names(procedure.execution) & functions - {name}
        output[name] = _lower_procedure(procedure, {target: procedures[target] for target in calls})
        pending.extend(sorted(calls, reverse=True))
    return output


if __name__ == '__main__':
    request = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    print(json.dumps(main(request)))
