"""Normalize verified Fortran procedures to the fixed-form transport dialect.

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
from fparser.two.utils import FortranSyntaxError, walk

_SCOPES = re.compile(r"^\s*procedure name = (\w+)\s*$", re.MULTILINE)
_SYMBOLS = re.compile(
    r"^[ \t]*symtree: '([^']+)'[ \t]*\|\| symbol: '[^']+'[^\n]*\n", re.MULTILINE
)
_KINDS = {
    ("REAL", 4): "REAL",
    ("REAL", 8): "DOUBLE PRECISION",
    ("INTEGER", 4): "INTEGER",
    ("LOGICAL", 4): "LOGICAL",
    ("COMPLEX", 4): "COMPLEX",
    ("COMPLEX", 8): "DOUBLE COMPLEX",
    ("CHARACTER", 1): "CHARACTER",
}
# Netlib f2c's default REAL function convention widens the return to double.
# REAL arguments still refer to float storage.
_C_TYPES = {
    ("INTEGER", 4): ("int", "integer", "integer"),
    ("REAL", 4): ("float", "real", "doublereal"),
    ("REAL", 8): ("double", "doublereal", "doublereal"),
}
_GNU_C_TYPES = {**_C_TYPES, ("LOGICAL", 4): ("int", "integer", "integer")}
_GNU_C_TYPES[("CHARACTER", 1)] = ("char", "char", "char")
_IDENTIFIER = re.compile(r"(?<![\w.])[A-Za-z_]\w*", re.ASCII)
_FORTRAN_STRINGS = re.compile(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"")
_KINDED_NUMBER = re.compile(
    r"(?<![\w.])((?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eEdD][+-]?\d+)?)_(\w+)\b"
)
_SOURCE_RELATIONS = re.compile(r"<=|>=|==|/=|<|>")
_RELATIONS = {
    "<=": ".LE.",
    ">=": ".GE.",
    "==": ".EQ.",
    "/=": ".NE.",
    "<": ".LT.",
    ">": ".GT.",
}
_PROGRAM_UNITS = (F.Main_Program, F.Subroutine_Subprogram, F.Function_Subprogram)
_CONTAINERS = (F.Module_Subprogram_Part, F.Internal_Subprogram_Part)
_STATEMENTS = {
    "Assignment_Stmt",
    "Block_Label_Do_Construct",
    "Block_Nonlabel_Do_Construct",
    "Call_Stmt",
    "Continue_Stmt",
    "Close_Stmt",
    "Else_If_Stmt",
    "Else_Stmt",
    "End_Do_Stmt",
    "End_If_Stmt",
    "Format_Stmt",
    "Goto_Stmt",
    "If_Construct",
    "If_Stmt",
    "If_Then_Stmt",
    "Label_Do_Stmt",
    "Nonlabel_Do_Stmt",
    "Open_Stmt",
    "Read_Stmt",
    "Return_Stmt",
    "Stop_Stmt",
    "Write_Stmt",
}
_ELEMENTWISE_EXPRESSIONS = (
    F.Add_Operand,
    F.Int_Literal_Constant,
    F.Level_2_Expr,
    F.Level_2_Unary_Expr,
    F.Logical_Literal_Constant,
    F.Mult_Operand,
    F.Name,
    F.Parenthesis,
    F.Real_Literal_Constant,
    F.Signed_Int_Literal_Constant,
    F.Signed_Real_Literal_Constant,
    str,
    type(None),
)


@dataclass(frozen=True)
class Symbol:
    type_and_kind: tuple[str, int]
    attributes: str
    value: str | None
    c_interop: bool
    binding: str | None

    def type_name(self):
        if self.type_and_kind not in _KINDS:
            raise ValueError(f"unsupported Fortran type and kind {self.type_and_kind}")
        return _KINDS[self.type_and_kind]


@dataclass(frozen=True)
class Procedure:
    name: str
    node: object
    specification: tuple
    execution: tuple
    symbols: dict
    unit: str
    external: bool


def _symbol_tables(dump):
    scopes = list(_SCOPES.finditer(dump))
    tables = {}
    imports = []
    for index, scope in enumerate(scopes):
        end = scopes[index + 1].start() if index + 1 < len(scopes) else len(dump)
        text = dump[scope.end() : end].split("\n  code:", 1)[0]
        entries = list(_SYMBOLS.finditer(text))
        symbols = {}
        for item_index, item in enumerate(entries):
            tail = (
                entries[item_index + 1].start()
                if item_index + 1 < len(entries)
                else len(text)
            )
            fields = text[item.end() : tail]
            type_spec = re.search(
                r"^\s*type spec\s*:\s*\((\w+)\s+(\d+)[^\n]*\)", fields, re.MULTILINE
            )
            character = re.search(
                r"^\s*type spec\s*:\s*\(CHARACTER .* (\d+)\)", fields, re.MULTILINE
            )
            if character:
                type_and_kind = ("CHARACTER", int(character[1]))
            elif type_spec:
                type_and_kind = (type_spec[1], int(type_spec[2]))
            else:
                type_and_kind = None
            if type_and_kind is None:
                origin = re.search(r"from namespace '(\w+)'", item[0])
                if origin:
                    imports.append(
                        (scope[1].lower(), item[1].lower(), origin[1].lower())
                    )
                continue
            attributes = re.search(r"^\s*attributes:\s*(.*)$", fields, re.MULTILINE)
            value = re.search(r"^\s*value:\s*(.*)$", fields, re.MULTILINE)
            binding = re.search(r"binding_label: '([^']+)'", item[0])
            symbols[item[1].lower()] = Symbol(
                type_and_kind,
                attributes[1] if attributes else "",
                value[1] if value else None,
                bool(type_spec and "C_INTEROP" in type_spec[0]),
                binding[1] if binding else None,
            )
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
                    raise ValueError(f"missing compiler-resolved Fortran kind {kind}")
                kind = fact.value
            if kind == "4":
                return value
            if kind == "8" and re.search(r"[.eEdD]", value):
                return (
                    re.sub(r"[eEdD]", "D", value)
                    if re.search(r"[eEdD]", value)
                    else value + "D0"
                )
            raise ValueError(f"unsupported Fortran literal kind in {match[0]}")

        fragment = _KINDED_NUMBER.sub(number, fragment)
        fragment = _SOURCE_RELATIONS.sub(lambda match: _RELATIONS[match[0]], fragment)
        if aliases:
            fragment = _IDENTIFIER.sub(
                lambda match: aliases.get(match[0].lower(), match[0]), fragment
            )
        return fragment

    source = str(node)
    chunks = []
    cursor = 0
    for literal in _FORTRAN_STRINGS.finditer(source):
        chunks.extend((convert(source[cursor : literal.start()]), literal[0]))
        cursor = literal.end()
    chunks.append(convert(source[cursor:]))
    return "".join(chunks)


def _literal(value, symbol):
    if symbol.type_and_kind == ("REAL", 8):
        value = re.sub(r"_8\b", "", value)
        if re.search(r"[eEdD]", value):
            return re.sub(r"[eEdD]", "D", value)
        return value + ("D0" if "." in value else ".0D0")
    return re.sub(r"_4\b", "", value)


def _parameter_lines(symbols, used):
    for name in sorted(used & symbols.keys()):
        symbol = symbols[name]
        if "PARAMETER" not in symbol.attributes:
            continue
        if not symbol.value:
            raise ValueError(f"missing compiler-resolved Fortran parameter {name}")
        if symbol.value.startswith("(/") and symbol.value.endswith("/)"):
            values = [
                _literal(value.strip(), symbol)
                for value in symbol.value[2:-2].split(",")
            ]
            yield f"{symbol.type_name()} {name}({len(values)})"
            yield f'DATA {name} /{", ".join(values)}/'
        else:
            yield f"{symbol.type_name()} {name}"
            yield f"PARAMETER ({name} = {_literal(symbol.value, symbol)})"


def _collect_procedures(node, tables, inherited, procedures, unit, *, external=False):
    if isinstance(node, F.Module):
        module = str(node.content[0].items[1]).lower()
        if module not in tables:
            raise ValueError(f"no compiler symbol table for module {module}")
        inherited = {
            name: symbol
            for name, symbol in tables[module].items()
            if "PARAMETER" in symbol.attributes
        }
    if isinstance(node, _PROGRAM_UNITS):
        name = str(node.content[0].items[1]).lower()
        if name in procedures or name not in tables:
            raise ValueError(f"open or ambiguous Fortran procedure scope {name}")
        specification = tuple(
            statement
            for child in node.content
            if isinstance(child, F.Specification_Part)
            for item in child.content
            for statement in (
                item.content if isinstance(item, F.Implicit_Part) else (item,)
            )
        )
        execution = next(
            (
                tuple(child.content)
                for child in node.content
                if isinstance(child, F.Execution_Part)
            ),
            (),
        )
        procedures[name] = Procedure(
            name,
            node,
            specification,
            execution,
            {**inherited, **tables[name]},
            unit,
            external,
        )
    for child in getattr(node, "content", ()):
        if isinstance(child, (*_PROGRAM_UNITS, *_CONTAINERS, F.Module)):
            _collect_procedures(
                child,
                tables,
                inherited,
                procedures,
                unit,
                external=isinstance(node, F.Program),
            )


def _declarations(
    procedure, known, *, result=None, bridge=False, descriptors=None, aliases=None
):
    symbols = procedure.symbols
    descriptors = descriptors or {}
    aliases = aliases or {}
    dimensions = {}
    defined = set()
    declarations = []
    allowed = {"PARAMETER", "VALUE"} if bridge else {"PARAMETER"}
    for statement in procedure.specification:
        if isinstance(statement, (F.Use_Stmt, F.Implicit_Stmt)):
            continue
        if isinstance(statement, F.Procedure_Declaration_Stmt):
            names = _source_text(statement.items[-1], symbols, aliases)
            declarations.append(f"EXTERNAL {names}")
            continue
        if isinstance(statement, (F.External_Stmt, F.Intrinsic_Stmt)):
            keyword = (
                "EXTERNAL" if isinstance(statement, F.External_Stmt) else "INTRINSIC"
            )
            names = _source_text(statement.items[-1], symbols, aliases)
            declarations.append(f"{keyword} {names}")
            continue
        if not isinstance(statement, F.Type_Declaration_Stmt):
            raise ValueError(f"unsupported Fortran declaration {statement}")
        _, attr_list, entities = statement.items
        attrs = attr_list.items if attr_list else ()
        shape = next(
            (
                item.items[1]
                for item in attrs
                if isinstance(item, F.Dimension_Attr_Spec)
            ),
            None,
        )
        for attr in attrs:
            if str(attr).upper() == "CONTIGUOUS" and all(
                descriptors.get(str(entity.items[0]).lower(), {}).get("contiguous")
                for entity in entities.items
            ):
                continue
            if (
                not isinstance(attr, (F.Dimension_Attr_Spec, F.Intent_Attr_Spec))
                and str(attr).upper() not in allowed
            ):
                raise ValueError(f"unsupported Fortran declaration attribute {attr}")
        for entity in entities.items:
            entity_name, array, character, initializer = entity.items
            name = str(entity_name).lower()
            symbol = symbols.get(name)
            if not symbol:
                raise ValueError(
                    f"missing compiler information for Fortran variable {name}"
                )
            if "PARAMETER" in symbol.attributes or name == result:
                continue
            if character or initializer:
                raise ValueError(
                    f"unsupported Fortran declaration initializer {entity}"
                )
            extent = array or shape
            if name in descriptors:
                descriptor = descriptors[name]
                declarations.append(f'{symbol.type_name()} {descriptor["array"]}(*)')
                for axis in descriptor["axes"]:
                    declarations.append(
                        "INTEGER*8 " + ", ".join(axis[key] for key in _CFI_AXIS_FIELDS)
                    )
                defined.add(name)
                continue
            if extent and _all_names((extent,)) & descriptors.keys():
                raise ValueError(f"CFI array used in a declaration of {name}")
            dtype = symbol.type_name()
            if symbol.type_and_kind == ("CHARACTER", 1):
                selector = statement.items[0].items[1]
                if selector is None:
                    length = "1"
                elif isinstance(selector, F.Char_Selector):
                    length = _source_text(selector.items[0], symbols)
                else:
                    length = _source_text(selector.items[1], symbols)
                dtype = f"CHARACTER*({length})"
            declaration = f"{dtype} {aliases.get(name, name)}"
            if extent:
                declaration += f"({_source_text(extent, symbols)})"
                dimensions[name] = extent
            declarations.append(declaration)
            defined.add(name)
    for name in sorted(known):
        candidate = known[name]
        transport_name = aliases.get(name, name)
        if name not in defined and isinstance(candidate.node, F.Function_Subprogram):
            result_type = candidate.symbols[_function_result(candidate)].type_name()
            declarations.append(f"{result_type} {transport_name}")
        if name not in defined and name not in symbols:
            declarations.append(f"EXTERNAL {transport_name}")
        elif name not in defined and "DUMMY" not in symbols[name].attributes:
            declarations.append(f"EXTERNAL {transport_name}")
    return declarations, dimensions


def _all_names(parts):
    return {str(name).lower() for part in parts for name in walk(part, F.Name)}


_CFI_AXIS_FIELDS = ("extent", "stride", "origin", "lower", "upper")
_CFI_INQUIRIES = {"SIZE": "extent", "LBOUND": "lower", "UBOUND": "upper"}


def _integer_constant(node, symbols):
    if isinstance(node, F.Name):
        fact = symbols.get(str(node).lower())
        if (
            not fact
            or fact.type_and_kind != ("INTEGER", 4)
            or "PARAMETER" not in fact.attributes
        ):
            raise ValueError(f"CFI shape needs a default-integer constant. {node}")
        value = _literal(fact.value or "", fact)
    else:
        value = _source_text(node, symbols)
    if not re.fullmatch(r"[+-]?\s*\d+", value):
        raise ValueError(f"CFI shape needs a default-integer constant. {node}")
    number = int(value.replace(" ", ""))
    if not -(2**31) <= number < 2**31:
        raise ValueError(
            f"CFI shape constant outside the GNU default integer range. {node}"
        )
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
        raise ValueError(f"unsupported CFI array inquiry. {node}")
    dim = arguments[1]
    if isinstance(dim, F.Actual_Arg_Spec):
        keyword, dim = dim.items
        if str(keyword).upper() != "DIM":
            raise ValueError(f"CFI array inquiry needs an explicit dimension. {node}")
    number = _integer_constant(dim, symbols)
    if not 1 <= number <= len(descriptor["axes"]):
        raise ValueError(f"CFI inquiry dimension outside the declared rank. {node}")
    return f'INT({descriptor["axes"][number - 1][_CFI_INQUIRIES[intrinsic]]})'


def _cfi_index(node, descriptors, symbols):
    if isinstance(node, F.Name):
        fact = symbols.get(str(node).lower())
        return (
            fact is not None
            and fact.type_and_kind == ("INTEGER", 4)
            and "DIMENSION" not in fact.attributes
            and "PROCEDURE" not in fact.attributes
        )
    if isinstance(node, F.Int_Literal_Constant):
        _integer_constant(node, symbols)
        return True
    if isinstance(node, F.Intrinsic_Function_Reference):
        return _cfi_inquiry(node, descriptors, symbols) is not None
    if isinstance(
        node, (F.Level_2_Unary_Expr, F.Level_2_Expr, F.Add_Operand, F.Parenthesis)
    ):
        return all(
            _cfi_index(item, descriptors, symbols)
            for item in node.items
            if not isinstance(item, str)
        )
    return False


def _lower_expression(node, descriptors, symbols):
    if isinstance(node, F.Intrinsic_Function_Reference):
        inquiry = _cfi_inquiry(node, descriptors, symbols)
        if inquiry is not None:
            return inquiry
        intrinsic, arguments = node.items
        fact = symbols.get("real")
        if (
            str(intrinsic).upper() == "REAL"
            and fact
            and "INTRINSIC" in fact.attributes
            and arguments
            and len(arguments.items) == 2
        ):
            actual = {}
            for name, value in zip(("a", "kind"), arguments.items):
                if isinstance(value, F.Actual_Arg_Spec):
                    label, value = value.items
                    name = str(label).lower()
                actual[name] = value
            if actual.keys() != {"a", "kind"}:
                raise ValueError(f"unsupported REAL intrinsic arguments. {node}")
            kind = _integer_constant(actual["kind"], symbols)
            if kind not in {4, 8}:
                raise ValueError(f"unsupported real conversion kind {kind}")
            name = "DBLE" if kind == 8 else "REAL"
            value = _lower_expression(actual["a"], descriptors, symbols)
            return f"{name}({value})"
    if isinstance(node, F.Part_Ref):
        name, subscripts = node.items
        descriptor = descriptors.get(str(name).lower())
        if descriptor is not None:
            axes = descriptor["axes"]
            if len(subscripts.items) != len(axes) or not all(
                _cfi_index(index, descriptors, symbols) for index in subscripts.items
            ):
                raise ValueError(
                    f"CFI element access needs one scalar default-integer subscript per axis. {node}"
                )
            terms = []
            for index, axis in zip(subscripts.items, axes):
                index = _lower_expression(index, descriptors, symbols)
                terms.append(f'(({index}) - {axis["origin"]}) * {axis["stride"]}')
            return f'{descriptor["array"]}(1 + {" + ".join(terms)})'
        if _all_names((subscripts,)) & descriptors.keys():
            fact = symbols.get(str(name).lower())
            if not fact or "DIMENSION" not in fact.attributes:
                raise ValueError(
                    f"CFI values cannot be passed to an unlowered function. {node}"
                )
    if isinstance(node, F.Call_Stmt) and _all_names((node,)) & descriptors.keys():
        raise ValueError(
            f"CFI values cannot be passed by reference to a procedure. {node}"
        )
    if isinstance(node, F.Name) and str(node).lower() in descriptors:
        raise ValueError(
            f"CFI array requires an element access or scalar shape inquiry. {node}"
        )
    if isinstance(node, tuple):
        return tuple(_lower_expression(item, descriptors, symbols) for item in node)
    if isinstance(node, list):
        return [_lower_expression(item, descriptors, symbols) for item in node]
    if hasattr(node, "items"):
        replacement = copy(node)
        replacement.items = _lower_expression(node.items, descriptors, symbols)
        return replacement
    return node


class FixedSource:
    def __init__(self, procedure, columns, aliases, descriptors):
        self.procedure = procedure
        self.columns = columns
        self.aliases = aliases
        self.descriptors = descriptors
        real = procedure.symbols.get("real")
        self.normalize_expressions = descriptors or (
            real and "INTRINSIC" in real.attributes
        )
        self.lines = []
        self.indices = []
        self.used = _all_names((*procedure.specification, *procedure.execution))

    def line(self, contents, label=None):
        self.lines.append(f'{str(label or ""):>5} {contents}')

    def emit(self, node):
        kind = type(node).__name__
        if kind not in _STATEMENTS:
            raise ValueError(f"unsupported Fortran statement {kind}. {node}")
        if getattr(node, "content", None) is not None:
            for child in node.content:
                self.emit(child)
            return
        symbols = self.procedure.symbols
        label = getattr(getattr(node, "item", None), "label", None)
        if isinstance(node, F.Assignment_Stmt) and isinstance(node.items[0], F.Name):
            name = str(node.items[0]).lower()
            bounds = self.columns.get(name)
            if bounds:
                self.assign_array(node, name, bounds, label)
                return
        if self.normalize_expressions:
            node = _lower_expression(node, self.descriptors, symbols)
        self.line(_source_text(node, symbols, self.aliases), label)

    def assign_array(self, assignment, array, bounds, label):
        rhs = assignment.items[2]
        if any(not isinstance(node, _ELEMENTWISE_EXPRESSIONS) for node in walk(rhs)):
            raise ValueError(f"unsupported whole array assignment {assignment}")
        if not isinstance(bounds, F.Explicit_Shape_Spec_List):
            raise ValueError(
                f"whole array assignment needs explicit bounds {assignment}"
            )
        symbols = self.procedure.symbols
        for name in _all_names((bounds,)):
            fact = symbols.get(name)
            if fact is None or not (
                "PARAMETER" in fact.attributes or "DUMMY(IN)" in fact.attributes
            ):
                raise ValueError(
                    f"mutable bound in whole array assignment {assignment}"
                )
        inputs = _all_names((rhs,))
        for name in inputs:
            fact = symbols.get(name)
            shape = self.columns.get(name)
            if fact is None or (
                "DIMENSION" in fact.attributes
                and (shape is None or str(shape).lower() != str(bounds).lower())
            ):
                raise ValueError(f"array shape is unresolved in {assignment}")
        rank = len(bounds.items)
        while len(self.indices) < rank:
            serial = len(self.indices) + 1
            index = f"scar_i{serial}"
            while index in self.used:
                serial += 1
                index = f"scar_i{serial}"
            self.used.add(index)
            self.indices.append(index)
        indices = self.indices[:rank]
        for index, extent in reversed(list(zip(indices, bounds.items))):
            first, last = extent.items
            self.line(f"DO {index} = {first or 1}, {last}", label)
            label = None
        suffix = f'({", ".join(indices)})'
        aliases = {
            **self.aliases,
            **{name: name + suffix for name in inputs if name in self.columns},
        }
        self.line(f"{array}{suffix} = {_source_text(rhs, symbols, aliases)}")
        for _ in indices:
            self.line("END DO")


def _function_result(procedure):
    start = procedure.node.content[0]
    if isinstance(start, F.Function_Stmt):
        suffix = start.items[-1]
        return (
            str(suffix.items[0]).lower()
            if suffix and suffix.items[0]
            else procedure.name
        )
    return None


def _binding_spec(procedure):
    start = procedure.node.content[0]
    if isinstance(start, F.Function_Stmt):
        return start.items[-1].items[1] if start.items[-1] else None
    if isinstance(start, F.Subroutine_Stmt):
        return start.items[-1]
    return None


def _c_export(procedure, transport, signature=None):
    start = procedure.node.content[0]
    symbol = procedure.symbols.get(procedure.name)
    spec = _binding_spec(procedure)
    legacy = spec is None
    if legacy:
        if signature is None or not _IDENTIFIER.fullmatch(signature["symbol"]):
            raise ValueError(f"unresolved GNU external ABI in {procedure.name}")
        spelling = signature["symbol"]
    else:
        spelling = str(spec.items[0])[1:-1] if spec.items[0] else procedure.name
        if (
            not symbol
            or symbol.binding != spelling
            or not _IDENTIFIER.fullmatch(spelling)
            or "BIND(C)" not in symbol.attributes
        ):
            raise ValueError(
                f"BIND(C) name differs from compiler evidence. {procedure.name}"
            )

    dimensions = {}
    for statement in procedure.specification:
        if not isinstance(statement, F.Type_Declaration_Stmt):
            continue
        _, attributes, entities = statement.items
        shared = (
            next(
                (
                    attr.items[1]
                    for attr in attributes.items
                    if isinstance(attr, F.Dimension_Attr_Spec)
                ),
                None,
            )
            if attributes
            else None
        )
        for entity in entities.items:
            name, shape, _, _ = entity.items
            dimensions[str(name).lower()] = shape or shared

    used = set(procedure.symbols)
    descriptor_serial = 0

    def abi(name, *, result=False):
        nonlocal descriptor_serial
        fact = procedure.symbols.get(name)
        if fact is None:
            raise ValueError(
                f"{procedure.unit}: unsupported GNU ABI parameter "
                f"{procedure.name}.{name}"
            )
        attributes = set(re.findall(r"[A-Z]+", fact.attributes))
        unsupported = {"POINTER", "ALLOCATABLE", "OPTIONAL", "VOLATILE", "ASYNCHRONOUS"}
        types = _GNU_C_TYPES if legacy else _C_TYPES
        if (
            (not legacy and not fact.c_interop)
            or fact.type_and_kind not in types
            or attributes & unsupported
        ):
            raise ValueError(
                f"unsupported external ABI type {name} in {procedure.name}"
            )
        shape = dimensions.get(name)
        by_value = "VALUE" in attributes
        c_type, argument_type, return_type = types[fact.type_and_kind]
        transport_type = return_type if result else argument_type
        row = dict(
            name=name,
            c_type=c_type,
            transport_type=transport_type,
            by_value=by_value,
            read_only="DUMMY(IN)" in fact.attributes,
            shape=str(shape) if shape is not None else None,
        )
        if fact.type_and_kind == ("CHARACTER", 1):
            if result or by_value or shape is not None:
                raise ValueError(f"unsupported character result or array in {procedure.name}")
            hidden = next(
                (item for item in signature["arguments"] if item["name"] == "_" + name),
                None,
            )
            if hidden is None or hidden["dtype"] != "integer(kind=8)" or not hidden["by_value"]:
                raise ValueError(f"unresolved GNU character length ABI in {procedure.name}")
            row["character_length"] = "size_t"
        if shape is not None or "DIMENSION" in attributes:
            if result or by_value:
                raise ValueError(
                    f"unsupported C array ABI for {name} in {procedure.name}"
                )
            if isinstance(shape, F.Assumed_Shape_Spec_List):
                if legacy:
                    raise ValueError(
                        f"GNU assumed-shape descriptor is unsupported. {name}"
                    )
                if "TARGET" in attributes:
                    raise ValueError(
                        f"CFI TARGET dummy is outside the nonescaping element ABI. {name}"
                    )
                bounds = [
                    (
                        _integer_constant(axis.items[0], procedure.symbols)
                        if axis.items[0] is not None
                        else 1
                    )
                    for axis in shape.items
                ]
                while True:
                    descriptor_serial += 1
                    array = f"scarcfia{descriptor_serial}"
                    axes = [
                        {
                            field: f"{array}{axis}{field[0]}"
                            for field in _CFI_AXIS_FIELDS
                        }
                        for axis in range(len(bounds))
                    ]
                    identifiers = {
                        array,
                        *(value for axis in axes for value in axis.values()),
                    }
                    if not identifiers & used:
                        used.update(identifiers)
                        break
                row["descriptor"] = {
                    "schema": "scar.gnu-cfi-assumed-shape",
                    "schema_version": 1,
                    "rank": len(bounds),
                    "cfi_type": f"CFI_type_{c_type}",
                    "lower_bounds": bounds,
                    "array": array,
                    "axes": axes,
                }
                if "CONTIGUOUS" in attributes:
                    intent = re.search(r"\bDUMMY\((IN|OUT|INOUT)\)", fact.attributes)
                    row["descriptor"].update(
                        contiguous=True, intent=intent[1].lower() if intent else "inout"
                    )
            elif not isinstance(
                shape, (F.Explicit_Shape_Spec_List, F.Assumed_Size_Spec)
            ):
                raise ValueError(
                    f"unsupported C array ABI for {name} in {procedure.name}"
                )
            elif isinstance(shape, F.Assumed_Size_Spec) and shape.items[0] is None:
                lower = shape.items[1]
                try:
                    row["lower"] = (
                        _integer_constant(lower, procedure.symbols)
                        if lower is not None
                        else 1
                    )
                except ValueError:
                    # Dynamic lower bounds still use the original transport.
                    pass
        return row

    arguments = start.items[2]
    inputs = [abi(str(item).lower()) for item in arguments.items] if arguments else []
    result_name = _function_result(procedure)
    result = abi(result_name, result=True) if result_name else None
    if legacy:
        _check_gnu_signature(procedure, signature, inputs, result_name, result)
    return dict(
        unit=procedure.unit,
        procedure=procedure.name,
        symbol=spelling,
        transport=transport + "_",
        arguments=inputs,
        result=result,
    )


def _check_gnu_signature(procedure, signature, arguments, result_name, result):
    def compiler_type(name):
        dtype, kind = procedure.symbols[name].type_and_kind
        return f"{dtype.lower()}(kind={kind})"

    expected_arguments = [
        dict(
            name=argument["name"],
            dtype=compiler_type(argument["name"]),
            by_value=argument["by_value"],
            array=argument["shape"] is not None or "character_length" in argument,
        )
        for argument in arguments
    ]
    expected_arguments.extend(
        dict(name="_" + argument["name"], dtype="integer(kind=8)", by_value=True, array=False)
        for argument in arguments
        if "character_length" in argument
    )
    expected_result = compiler_type(result_name) if result_name else "void"
    actual_result = signature["result"]
    if expected_result == "real(kind=4)" and actual_result == "real(kind=8)":
        result["c_type"] = "double"
        expected_result = actual_result
    if signature["arguments"] != expected_arguments or actual_result != expected_result:
        raise ValueError(
            f"GNU ABI differs from Fortran declarations in {procedure.name}"
        )


def _lower_procedure(procedure, known, export=None, call_aliases=None):
    result = _function_result(procedure)
    transport = export["transport"][:-1] if export else None
    name = transport or procedure.name
    if _binding_spec(procedure) and transport is None:
        raise ValueError(f"BIND(C) requires an admitted transport. {procedure.name}")
    symbol = procedure.symbols.get(procedure.name)
    if symbol and "RECURSIVE" in symbol.attributes:
        raise ValueError(
            f"recursive Fortran needs reentrant transport. {procedure.name}"
        )
    start = procedure.node.content[0]
    arguments = (
        start.items[2]
        if isinstance(start, (F.Function_Stmt, F.Subroutine_Stmt))
        else None
    )
    descriptors = {}
    transport_arguments = []
    if export and arguments is not None:
        for argument, abi in zip(arguments.items, export["arguments"], strict=True):
            original = str(argument).lower()
            descriptor = abi.get("descriptor")
            if descriptor is None:
                transport_arguments.append(original)
                continue
            descriptors[original] = descriptor
            transport_arguments.append(descriptor["array"])
            for axis in descriptor["axes"]:
                transport_arguments.extend(axis[field] for field in _CFI_AXIS_FIELDS)
    signature = ", ".join(transport_arguments) if descriptors else str(arguments or "")
    declarations, dimensions = _declarations(
        procedure,
        known,
        result=result,
        bridge=transport is not None,
        descriptors=descriptors,
        aliases=call_aliases,
    )
    aliases = dict(call_aliases or {})
    if result and name != result:
        aliases[result] = name
    source = FixedSource(
        procedure,
        dimensions,
        aliases,
        descriptors,
    )
    for node in procedure.execution:
        source.emit(node)
    if isinstance(start, F.Function_Stmt):
        dtype = procedure.symbols[result].type_name()
        declaration = f"{dtype} FUNCTION {name}({signature})"
    elif isinstance(start, F.Subroutine_Stmt):
        if start.items[0] is not None:
            raise ValueError(f"unsupported Fortran subroutine modifiers {start}")
        declaration = f"SUBROUTINE {name}({signature})"
    else:
        declaration = str(start)
    parameters = tuple(_parameter_lines(procedure.symbols, source.used))
    fixed = [
        declaration,
        "IMPLICIT NONE",
        *(line for line in parameters if not line.startswith("DATA ")),
        *declarations,
    ]
    fixed.extend(f"INTEGER {name}" for name in source.indices)
    fixed.extend(line for line in parameters if line.startswith("DATA "))
    return "\n".join(
        [*("      " + line for line in fixed), *source.lines, "      END", ""]
    )


def main(request):
    reader = ParserFactory().create(std="f2008")
    procedures = {}
    passthrough = set()
    signatures = {}
    for row in request["units"]:
        try:
            tree = reader(FortranFileReader(row["source"], ignore_comments=True))
        except FortranSyntaxError as error:
            raise ValueError(f"{row['unit']}: {error}") from None
        semantics = Path(row["semantics"]).read_text(encoding="utf-8")
        before = set(procedures) if row.get("passthrough") else None
        _collect_procedures(
            tree, _symbol_tables(semantics), {}, procedures, row["unit"]
        )
        signatures.update(
            ((row["unit"], record["name"]), record) for record in row["signatures"]
        )
        if before is not None:
            passthrough.update(procedures.keys() - before)
    entries = [
        node.name
        for node in procedures.values()
        if isinstance(node.node, F.Main_Program)
    ]
    transports, exports = {}, {}
    bound = set()
    if request.get("mixed"):
        if entries:
            raise ValueError(
                "a mixed free-form provider must export procedures without a main program"
            )
        bound = {name for name, proc in procedures.items() if _binding_spec(proc)}
        if bound & passthrough:
            raise ValueError("fixed-form BIND(C) requires a separate ABI transport")
        reserved = {row["symbol"] for row in signatures.values()}
        for name, procedure in procedures.items():
            if name in passthrough or (name not in bound and not procedure.external):
                continue
            index = len(transports) + 1
            while (
                transport := f"scarbind{index}"
            ) in procedures or transport + "_" in reserved:
                index += 1
            signature = signatures.get((procedure.unit, name))
            exports[name] = _c_export(procedure, transport, signature)
            transports[name] = transport
            reserved.add(transport + "_")
        if not exports or len({row["symbol"] for row in exports.values()}) != len(
            exports
        ):
            raise ValueError(
                "mixed free-form units require distinct compiler-resolved external entries"
            )
        entries = list(transports)
    elif len(entries) != 1 or request["entry"].lower() not in ("main", entries[0]):
        raise ValueError("expected one supported free-form program entry")
    functions = {
        name
        for name, procedure in procedures.items()
        if not isinstance(procedure.node, F.Main_Program)
    }
    pending = [*entries, *sorted(passthrough)]
    output, seen = {}, set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        procedure = procedures[name]
        calls = {
            target
            for target in _all_names(procedure.execution) & functions - {name}
            if target in procedure.symbols
            and "PROCEDURE" in procedure.symbols[target].attributes
            and "INTRINSIC" not in procedure.symbols[target].attributes
        }
        bridged_calls = calls & transports.keys()
        if bridged_calls & bound or any(
            argument["by_value"]
            for target in bridged_calls
            for argument in exports[target]["arguments"]
        ):
            raise ValueError(
                "Fortran calls into a value or BIND(C) export require an explicit bridge"
            )
        if name in passthrough and bridged_calls:
            raise ValueError(f"fixed-form procedure calls a renamed transport. {name}")
        if name not in passthrough:
            output[name] = _lower_procedure(
                procedure,
                {target: procedures[target] for target in calls},
                exports.get(name),
                {target: transports[target] for target in bridged_calls},
            )
        pending.extend(sorted(calls, reverse=True))
    descriptor_units = {}
    for name, row in exports.items():
        arrays = [
            arg["descriptor"]["array"]
            for arg in row["arguments"]
            if "descriptor" in arg
        ]
        if arrays:
            descriptor_units[name + ".f"] = arrays
    return dict(
        sources=output,
        exports=list(exports.values()),
        descriptor_units=descriptor_units,
    )


if __name__ == "__main__":
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    try:
        normalized = main(request)
    except ValueError as error:
        sys.exit(str(error))
    print(json.dumps(normalized))
