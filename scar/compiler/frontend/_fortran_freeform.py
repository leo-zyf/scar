"""Lower a checked free-form Fortran subset to the fixed-form transport dialect.

The compiler establishes kinds and constant values. Fparser supplies the scope,
declaration, and statement structure. The transport source is built anew from
these inputs and never substitutes routines from an earlier upstream release.
"""

from copy import copy
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
_SYMBOLS = re.compile(r"^[ \t]*symtree: '([^']+)'[ \t]*\|\| symbol: '[^']+'[^\n]*\n", re.MULTILINE)
_KINDS = {('REAL', 4): 'REAL', ('REAL', 8): 'DOUBLE PRECISION',
          ('INTEGER', 4): 'INTEGER', ('LOGICAL', 4): 'LOGICAL'}
# Netlib f2c's default REAL function convention widens the return to double.
# REAL arguments still refer to float storage.
_C_TYPES = {('INTEGER', 4): ('int', 'integer', 'integer'),
            ('REAL', 4): ('float', 'real', 'doublereal'),
            ('REAL', 8): ('double', 'doublereal', 'doublereal')}
_IDENTIFIER = re.compile(r'(?<![\w.])[A-Za-z_]\w*', re.ASCII)
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
    c_interop: bool
    binding: str | None

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
            type_spec = re.search(r'^\s*type spec\s*:\s*\((\w+)\s+(\d+)[^\n]*\)', fields, re.MULTILINE)
            if not type_spec:
                origin = re.search(r"from namespace '(\w+)'", item[0])
                if origin:
                    imports.append((scope[1].lower(), item[1].lower(), origin[1].lower()))
                continue
            attributes = re.search(r'^\s*attributes:\s*(.*)$', fields, re.MULTILINE)
            value = re.search(r'^\s*value:\s*(.*)$', fields, re.MULTILINE)
            binding = re.search(r"binding_label: '([^']+)'", item[0])
            symbols[item[1].lower()] = Symbol((type_spec[1], int(type_spec[2])),
                                             attributes[1] if attributes else '', value[1] if value else None,
                                             'C_INTEROP' in type_spec[0], binding[1] if binding else None)
        tables[scope[1].lower()] = symbols
    for scope, name, origin in imports:
        inherited = tables.get(origin, {}).get(name)
        if inherited is not None:
            tables[scope][name] = inherited
    return tables


def _source_text(node, symbols, aliases=None):
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

        fragment = re.sub(r'\bREAL\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)', real_conversion,
                          fragment, flags=re.IGNORECASE)
        if aliases:
            fragment = _IDENTIFIER.sub(lambda match: aliases.get(match[0].lower(), match[0]), fragment)
        return fragment

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


def _declarations(procedure, known, *, result=None, bridge=False, descriptors=None):
    symbols = procedure.symbols
    descriptors = descriptors or {}
    dimensions = {}
    defined = set()
    declarations = []
    allowed = {'PARAMETER', 'VALUE'} if bridge else {'PARAMETER'}
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
            if not isinstance(attr, (F.Dimension_Attr_Spec, F.Intent_Attr_Spec)) and str(attr).upper() not in allowed:
                raise ValueError(f'unsupported Fortran declaration attribute {attr}')
        for entity in entities.items:
            entity_name, array, character, initializer = entity.items
            name = str(entity_name).lower()
            symbol = symbols.get(name)
            if not symbol:
                raise ValueError(f'missing compiler information for Fortran variable {name}')
            if 'PARAMETER' in symbol.attributes or name == result:
                continue
            if character or initializer:
                raise ValueError(f'unsupported Fortran declaration initializer {entity}')
            extent = array or shape
            if name in descriptors:
                descriptor = descriptors[name]
                declarations.append(f'{symbol.type_name()} {descriptor["array"]}(*)')
                for axis in descriptor['axes']:
                    declarations.append('INTEGER*8 ' + ', '.join(axis[key] for key in _CFI_AXIS_FIELDS))
                defined.add(name)
                continue
            if extent and _all_names((extent,)) & descriptors.keys():
                raise ValueError(f'CFI array used in a declaration of {name}')
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


_CFI_AXIS_FIELDS = ('extent', 'stride', 'origin', 'lower', 'upper')
_CFI_INQUIRIES = {'SIZE': 'extent', 'LBOUND': 'lower', 'UBOUND': 'upper'}


def _integer_constant(node, symbols):
    if isinstance(node, F.Name):
        fact = symbols.get(str(node).lower())
        if not fact or fact.type_and_kind != ('INTEGER', 4) or 'PARAMETER' not in fact.attributes:
            raise ValueError(f'CFI shape needs a default-integer constant. {node}')
        value = _literal(fact.value or '', fact)
    else:
        value = _source_text(node, symbols)
    if not re.fullmatch(r'[+-]?\s*\d+', value):
        raise ValueError(f'CFI shape needs a default-integer constant. {node}')
    number = int(value.replace(' ', ''))
    if not -(2**31) <= number < 2**31:
        raise ValueError(f'CFI shape constant outside the GNU default integer range. {node}')
    return number


def _cfi_inquiry(node, descriptors, symbols):
    intrinsic, parameters = node.items
    arguments = parameters.items if parameters else ()
    if not arguments or not isinstance(arguments[0], F.Name):
        return None
    descriptor = descriptors.get(str(arguments[0]).lower())
    if descriptor is None:
        return None
    intrinsic = str(intrinsic).upper()
    if intrinsic not in _CFI_INQUIRIES or len(arguments) != 2:
        raise ValueError(f'unsupported CFI array inquiry. {node}')
    dim = arguments[1]
    if isinstance(dim, F.Actual_Arg_Spec):
        keyword, dim = dim.items
        if str(keyword).upper() != 'DIM':
            raise ValueError(f'CFI array inquiry needs an explicit dimension. {node}')
    number = _integer_constant(dim, symbols)
    if not 1 <= number <= len(descriptor['axes']):
        raise ValueError(f'CFI inquiry dimension outside the declared rank. {node}')
    return f'INT({descriptor["axes"][number - 1][_CFI_INQUIRIES[intrinsic]]})'


def _cfi_index(node, descriptors, symbols):
    if isinstance(node, F.Name):
        fact = symbols.get(str(node).lower())
        return (fact is not None and fact.type_and_kind == ('INTEGER', 4)
                and 'DIMENSION' not in fact.attributes and 'PROCEDURE' not in fact.attributes)
    if isinstance(node, F.Int_Literal_Constant):
        _integer_constant(node, symbols)
        return True
    if isinstance(node, F.Intrinsic_Function_Reference):
        return _cfi_inquiry(node, descriptors, symbols) is not None
    if isinstance(node, (F.Level_2_Unary_Expr, F.Level_2_Expr, F.Add_Operand, F.Parenthesis)):
        return all(_cfi_index(item, descriptors, symbols) for item in node.items
                   if not isinstance(item, str))
    return False


def _cfi_expression(node, descriptors, symbols):
    if isinstance(node, F.Intrinsic_Function_Reference):
        inquiry = _cfi_inquiry(node, descriptors, symbols)
        if inquiry is not None:
            return inquiry
    if isinstance(node, F.Part_Ref):
        name, subscripts = node.items
        descriptor = descriptors.get(str(name).lower())
        if descriptor is not None:
            axes = descriptor['axes']
            if len(subscripts.items) != len(axes) or not all(
                    _cfi_index(index, descriptors, symbols) for index in subscripts.items):
                raise ValueError(f'CFI element access needs one scalar default-integer subscript per axis. {node}')
            terms = []
            for index, axis in zip(subscripts.items, axes):
                index = _cfi_expression(index, descriptors, symbols)
                terms.append(f'(({index}) - {axis["origin"]}) * {axis["stride"]}')
            return f'{descriptor["array"]}(1 + {" + ".join(terms)})'
        if _all_names((subscripts,)) & descriptors.keys():
            fact = symbols.get(str(name).lower())
            if not fact or 'DIMENSION' not in fact.attributes:
                raise ValueError(f'CFI values cannot be passed to an unlowered function. {node}')
    if isinstance(node, F.Call_Stmt) and _all_names((node,)) & descriptors.keys():
        raise ValueError(f'CFI values cannot be passed by reference to a procedure. {node}')
    if isinstance(node, F.Name) and str(node).lower() in descriptors:
        raise ValueError(f'CFI array requires an element access or scalar shape inquiry. {node}')
    if isinstance(node, tuple):
        return tuple(_cfi_expression(item, descriptors, symbols) for item in node)
    if hasattr(node, 'items'):
        replacement = copy(node)
        replacement.items = _cfi_expression(node.items, descriptors, symbols)
        return replacement
    return node


class FixedSource:
    def __init__(self, procedure, columns, aliases, descriptors):
        self.procedure = procedure
        self.columns = columns
        self.aliases = aliases
        self.descriptors = descriptors
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
        if self.descriptors:
            node = _cfi_expression(node, self.descriptors, symbols)
        self.line(_source_text(node, symbols, self.aliases), label)

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


def _function_result(procedure):
    start = procedure.node.content[0]
    if isinstance(start, F.Function_Stmt):
        suffix = start.items[-1]
        return str(suffix.items[0]).lower() if suffix and suffix.items[0] else procedure.name
    return None


def _binding_spec(procedure):
    start = procedure.node.content[0]
    if isinstance(start, F.Function_Stmt):
        return start.items[-1].items[1] if start.items[-1] else None
    if isinstance(start, F.Subroutine_Stmt):
        return start.items[-1]
    return None


def _c_export(procedure, transport):
    start = procedure.node.content[0]
    symbol = procedure.symbols.get(procedure.name)
    if not symbol or not symbol.binding or not _IDENTIFIER.fullmatch(symbol.binding):
        raise ValueError(f'BIND(C) requires a compiler-resolved C identifier. {procedure.name}')
    spec = _binding_spec(procedure)
    spelling = str(spec.items[0])[1:-1] if spec.items[0] else procedure.name
    if symbol.binding != spelling or 'BIND(C)' not in symbol.attributes:
        raise ValueError(f'BIND(C) name differs from compiler evidence. {procedure.name}')

    dimensions = {}
    for statement in procedure.specification:
        if not isinstance(statement, F.Type_Declaration_Stmt):
            continue
        _, attributes, entities = statement.items
        shared = next((attr.items[1] for attr in attributes.items
                       if isinstance(attr, F.Dimension_Attr_Spec)), None) if attributes else None
        for entity in entities.items:
            name, shape, _, _ = entity.items
            dimensions[str(name).lower()] = shape or shared

    used = set(procedure.symbols)
    descriptor_serial = 0

    def abi(name, *, result=False):
        nonlocal descriptor_serial
        fact = procedure.symbols[name]
        attributes = set(re.findall(r'[A-Z]+', fact.attributes))
        unsupported = {'POINTER', 'ALLOCATABLE', 'OPTIONAL', 'VOLATILE', 'ASYNCHRONOUS'}
        if not fact.c_interop or fact.type_and_kind not in _C_TYPES or attributes & unsupported:
            raise ValueError(f'unsupported interoperable type {name} in {procedure.name}')
        shape = dimensions.get(name)
        by_value = 'VALUE' in attributes
        c_type, argument_type, return_type = _C_TYPES[fact.type_and_kind]
        transport_type = return_type if result else argument_type
        row = dict(c_type=c_type, transport_type=transport_type, by_value=by_value,
                   read_only='DUMMY(IN)' in fact.attributes, shape=str(shape) if shape is not None else None)
        if shape is not None or 'DIMENSION' in attributes:
            if result or by_value:
                raise ValueError(f'unsupported C array ABI for {name} in {procedure.name}')
            if isinstance(shape, F.Assumed_Shape_Spec_List):
                if attributes & {'CONTIGUOUS', 'TARGET'}:
                    raise ValueError(f'CFI CONTIGUOUS/TARGET dummy is outside the borrowed element ABI. {name}')
                bounds = [_integer_constant(axis.items[0], procedure.symbols) if axis.items[0] is not None else 1
                          for axis in shape.items]
                while True:
                    descriptor_serial += 1
                    array = f'scarcfia{descriptor_serial}'
                    axes = [{field: f'{array}{axis}{field[0]}' for field in _CFI_AXIS_FIELDS}
                            for axis in range(len(bounds))]
                    identifiers = {array, *(value for axis in axes for value in axis.values())}
                    if not identifiers & used:
                        used.update(identifiers)
                        break
                row['descriptor'] = {'schema': 'scar.gnu-cfi-assumed-shape', 'schema_version': 1,
                                     'rank': len(bounds), 'cfi_type': f'CFI_type_{c_type}',
                                     'lower_bounds': bounds, 'array': array, 'axes': axes}
            elif not isinstance(shape, (F.Explicit_Shape_Spec_List, F.Assumed_Size_Spec)):
                raise ValueError(f'unsupported C array ABI for {name} in {procedure.name}')
        return row

    arguments = start.items[2]
    inputs = [abi(str(item).lower()) for item in arguments.items] if arguments else []
    result = _function_result(procedure)
    return dict(symbol=symbol.binding, transport=transport + '_', arguments=inputs,
                result=abi(result, result=True) if result else None)


def _lower_procedure(procedure, known, export=None):
    result = _function_result(procedure)
    transport = export['transport'][:-1] if export else None
    name = transport or procedure.name
    if _binding_spec(procedure) and transport is None:
        raise ValueError(f'BIND(C) requires an admitted transport. {procedure.name}')
    symbol = procedure.symbols.get(procedure.name)
    if symbol and 'RECURSIVE' in symbol.attributes:
        raise ValueError(f'recursive Fortran needs reentrant transport. {procedure.name}')
    start = procedure.node.content[0]
    arguments = start.items[2] if isinstance(start, (F.Function_Stmt, F.Subroutine_Stmt)) else None
    descriptors = {}
    transport_arguments = []
    if export and arguments is not None:
        for argument, abi in zip(arguments.items, export['arguments'], strict=True):
            original = str(argument).lower()
            descriptor = abi.get('descriptor')
            if descriptor is None:
                transport_arguments.append(original)
                continue
            descriptors[original] = descriptor
            transport_arguments.append(descriptor['array'])
            for axis in descriptor['axes']:
                transport_arguments.extend(axis[field] for field in _CFI_AXIS_FIELDS)
    signature = ', '.join(transport_arguments) if descriptors else str(arguments or '')
    declarations, dimensions = _declarations(procedure, known, result=result, bridge=transport is not None,
                                             descriptors=descriptors)
    source = FixedSource(procedure, dimensions, {result: name} if result and name != result else {}, descriptors)
    for node in procedure.execution:
        source.emit(node)
    if isinstance(start, F.Function_Stmt):
        dtype = procedure.symbols[result].type_name()
        declaration = f'{dtype} FUNCTION {name}({signature})'
    elif isinstance(start, F.Subroutine_Stmt):
        if start.items[0] is not None:
            raise ValueError(f'unsupported Fortran subroutine modifiers {start}')
        declaration = f'SUBROUTINE {name}({signature})'
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
    transports, exports = {}, {}
    if request.get('mixed'):
        if entries:
            raise ValueError('a mixed free-form provider must export procedures without a main program')
        for name, procedure in procedures.items():
            if _binding_spec(procedure) is not None:
                index = len(transports) + 1
                while (transport := f'scarbind{index}') in procedures or transport in transports.values():
                    index += 1
                exports[name] = _c_export(procedure, transport)
                transports[name] = transport
        if not exports or len({row['symbol'] for row in exports.values()}) != len(exports):
            raise ValueError('mixed free-form units require distinct compiler-resolved BIND(C) entries')
        entries = list(transports)
    elif len(entries) != 1 or request['entry'].lower() not in ('main', entries[0]):
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
        if calls & transports.keys():
            raise ValueError('Fortran calls into a C ABI export require explicit value transport')
        output[name] = _lower_procedure(procedure, {target: procedures[target] for target in calls},
                                        exports.get(name))
        pending.extend(sorted(calls, reverse=True))
    descriptor_units = {}
    for name, row in exports.items():
        arrays = [arg['descriptor']['array'] for arg in row['arguments'] if 'descriptor' in arg]
        if arrays:
            descriptor_units[name + '.f'] = arrays
    return dict(sources=output, exports=list(exports.values()), descriptor_units=descriptor_units)


if __name__ == '__main__':
    request = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    print(json.dumps(main(request)))
