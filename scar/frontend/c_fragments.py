"""Adapt compiler-typed C source to C++ fragments and relocate clean body spans."""

from bisect import bisect_left, bisect_right
from collections import defaultdict
from hashlib import sha256
import json
import re

from ..contract import read_artifact


def read_c_family_types(unit, captures, source):
    name = unit["path"]
    if name not in captures:
        raise ValueError(f"no sealed C-family type evidence for {name}")
    typed = json.loads(read_artifact(captures[name]))
    fingerprint = typed.get("source_sha256")
    if (
        typed["path"] != name
        or typed["flags"] != unit.get("flags", [])
        or fingerprint is not None
        and fingerprint != sha256(source).hexdigest()
    ):
        raise ValueError(f"C type evidence has a different source or profile: {name}")
    return typed


def collect_c_adaptations(units, source_bytes, preserve_abi, captures):
    declarations = defaultdict(list)
    conversions = defaultdict(dict)
    definitions = defaultdict(dict)
    cfi_headers = []
    for unit in units:
        name = unit["path"]
        typed = read_c_family_types(unit, captures, source_bytes[name])
        cfi_headers.extend(typed.get("gnu_cfi", ()))
        records = typed["records"]
        functions, parameters, bodies = {}, defaultdict(list), {}
        for index, record in enumerate(records):
            path = record["path"]
            if path != name and not preserve_abi:
                continue
            if (
                preserve_abi
                and record["external_linkage"]
                and record["kind"] in ("FunctionDecl", "VarDecl")
            ):
                declarations[path].append(record)
            parent = record["parent"]
            if record["kind"] == "FunctionDecl" and record["definition"]:
                functions[index] = record
            elif parent in functions:
                if record["kind"] == "ParmDecl":
                    parameters[parent].append(record)
                elif record["kind"] == "CompoundStmt":
                    bodies[parent] = record
            if (
                parent is None
                or record["canonical_type"] != "void *"
                or not record["span"]
            ):
                continue
            conversion = records[parent]
            if (
                conversion["kind"] != "UnexposedExpr"
                or conversion["path"] != path
                or not conversion["canonical_type"].endswith(" *")
                or conversion["canonical_type"] == "void *"
            ):
                continue
            span = tuple(record["span"])
            target = conversion["type"]
            if conversions[path].setdefault(span, target) != target:
                raise ValueError(
                    f"ambiguous C pointer conversion in {path} at byte {span[0]}"
                )
        for index, arguments in parameters.items():
            if index not in bodies:
                continue
            function = functions[index]
            path = function["path"]
            if path not in source_bytes:
                raise ValueError(f"C definition has no locked source bytes: {path}")
            header = _c_identifier_definition(
                source_bytes[path], function, arguments, bodies[index]
            )
            if header is not None:
                span, declarators = header
                if definitions[path].setdefault(span, declarators) != declarators:
                    raise ValueError(
                        f'ambiguous old-style C function definition. {path}:{function["line"]}'
                    )
    return declarations, conversions, definitions, cfi_headers


_TRIVIA = re.compile(rb"(?:\s+|/\*[\s\S]*?\*/|//[^\n]*\n)*")
_C_IDENTIFIER = re.compile(rb"[A-Za-z_]\w*")


def _c_identifier_definition(data, function, parameters, body):
    """Record declarators for an identifier-list definition with unchanged call ABI."""
    token = _C_IDENTIFIER.match(data, function["offset"])
    if token is None:
        return None
    start = _TRIVIA.match(data, token.end()).end()
    if data[start : start + 1] != b"(":
        return None
    cursor = start + 1
    for index, parameter in enumerate(parameters):
        name = _C_IDENTIFIER.match(data, _TRIVIA.match(data, cursor).end())
        if name is None or name[0] != parameter["name"].encode():
            return None
        delimiter = _TRIVIA.match(data, name.end()).end()
        expected = b")" if index + 1 == len(parameters) else b","
        if data[delimiter : delimiter + 1] != expected:
            return None
        cursor = delimiter + 1

    location = f'{function["path"]}:{function["line"]}'
    if (
        token[0] != function["name"].encode()
        or not function["span"]
        or not body["span"]
        or body["path"] != function["path"]
    ):
        raise ValueError(
            f"old-style C definition has no original function/body range. {location}"
        )
    body_start = body["span"][0]
    if not function["span"][0] <= start < cursor <= body_start < function["span"][1]:
        raise ValueError(
            f"old-style C definition has inconsistent source ranges. {location}"
        )

    # Clang's function type already records the default parameter promotions
    # for a definition written with an identifier list. ParmDecl records the
    # local type before those promotions. Only use an ANSI declarator if each
    # has the same type at both sides of the call boundary.
    types = [parameter["canonical_type"] for parameter in parameters]
    _, _, signature = function["canonical_type"].partition("(")
    if not signature.endswith(")") or signature[:-1].split(", ") != types:
        raise ValueError(
            f"old-style C parameter promotion or adjustment requires an ABI bridge. {location}"
        )

    declared = []
    names_end = cursor
    for parameter in parameters:
        span = parameter["span"]
        if not span or parameter["path"] != function["path"]:
            raise ValueError(
                f"old-style C parameter has no original declaration range. {location}"
            )
        begin, stop = span
        spelling = parameter["name"].encode()
        if (
            start < begin < stop < names_end
            and data[begin:stop] == spelling
            and parameter["type"] == "int"
        ):
            declared.append(dict(implicit_int=parameter["name"]))
            continue
        name = _C_IDENTIFIER.match(data, parameter["offset"])
        end = _TRIVIA.match(data, stop).end()
        if (
            not name
            or name[0] != spelling
            or not cursor
            <= begin
            <= name.start()
            < name.end()
            <= stop
            <= end
            < body_start
            or _TRIVIA.match(data, cursor).end() != begin
            or data[end : end + 1] != b";"
        ):
            raise ValueError(
                f"old-style C parameters need independent ordered declaration ranges. {location}"
            )
        declared.append(dict(span=[begin, stop]))
        cursor = end + 1
    if (
        _TRIVIA.match(data, cursor).end() != body_start
        or data[body_start : body_start + 1] != b"{"
    ):
        raise ValueError(
            f"old-style C definition has an unsupported parameter preamble. {location}"
        )
    return (start, body_start), declared


def _c_abi_spans(path, data, declarations):
    # Clang can emit one VarDecl per declarator in a comma-separated declaration.
    groups = {}
    for record in declarations:
        span = record["span"]
        offset = record["offset"]
        spelling = _C_IDENTIFIER.match(data, offset)
        # The token can be a macro alias. Use the cursor's full declaration
        # extent, then qualify the resulting external symbol at link time.
        if (
            not span
            or not spelling
            or not span[0] <= offset < spelling.end() <= span[1] <= len(data)
        ):
            raise ValueError(
                f'C linkage declaration lacks a source range. {path}:{record["line"]}'
            )
        start, stop = span
        body = record["kind"] == "FunctionDecl" and record["definition"]
        if start not in groups or stop > groups[start][0]:
            groups[start] = stop, body
    for start, (stop, body) in sorted(groups.items()):
        if body:
            if data[stop - 1 : stop] != b"}":
                raise ValueError(f"unanchored C function body. {path}:{start}")
        elif data[stop - 1 : stop] != b";":
            end = _TRIVIA.match(data, stop).end()
            if data[end : end + 1] != b";":
                raise ValueError(f"unanchored C declaration end. {path}:{start}")
            stop = end + 1
        yield start, stop


def _c_adaptation(path, data, declarations, conversions, definitions):
    linkage = list(_c_abi_spans(path, data, declarations))
    prior = 0
    for start, stop in linkage:
        if start < prior:
            raise ValueError(f"overlapping C linkage declarations in {path}")
        prior = stop
    prior = 0
    casts = []
    for (start, stop), target in sorted(conversions.items()):
        if not 0 <= start < stop <= len(data):
            continue
        if start < prior:
            raise ValueError(
                f"overlapping C pointer conversions in {path} at byte {start}"
            )
        prior = stop
        casts.append(dict(span=[start, stop], type=target))
    headers = [
        dict(span=span, parameters=parameters)
        for span, parameters in sorted(definitions.items())
    ]
    return dict(
        linkage=linkage, pointer_conversions=casts, identifier_definitions=headers
    )


def _c_region_anchors(adaptation, regions):
    """Retain closed bodies whose interiors have no declaration or type edits."""
    insertions = sorted(
        {offset for span in adaptation["linkage"] for offset in span}
        | {
            offset
            for row in adaptation["pointer_conversions"]
            for offset in row["span"]
        }
    )
    replacements = [tuple(row["span"]) for row in adaptation["identifier_definitions"]]
    starts = [start for start, _ in replacements]
    for region in regions:
        start, stop = region["source"]["span"]
        cut = bisect_right(insertions, start)
        header = bisect_left(starts, stop)
        if cut < len(insertions) and insertions[cut] < stop:
            continue
        if header and replacements[header - 1][1] > start:
            continue
        yield region["region"], region["source"]["span"]


_C_ADAPTATIONS = {"linkage", "pointer_conversions", "identifier_definitions"}
_C_PARAMETER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _source_span(span, size, name):
    if (
        type(span) not in (tuple, list)
        or len(span) != 2
        or any(type(offset) is not int for offset in span)
        or not 0 <= span[0] < span[1] <= size
    ):
        raise ValueError(f"invalid C adaptation span in {name}")
    return tuple(span)


def _ansi_parameters(data, begin, end, parameters, name):
    if (
        type(parameters) not in (tuple, list)
        or not parameters
        or data[begin : begin + 1] != b"("
        or data[end : end + 1] != b"{"
    ):
        raise ValueError(
            f"invalid C identifier-list definition in {name} at byte {begin}"
        )
    declared, prior = [], begin + 1
    for parameter in parameters:
        if type(parameter) is not dict:
            raise ValueError(f"invalid C parameter record in {name}")
        if set(parameter) == {"span"}:
            start, stop = _source_span(parameter["span"], end, name)
            if start < prior:
                raise ValueError(
                    f"unordered C parameter spans in {name} at byte {start}"
                )
            declared.append(data[start:stop])
            prior = stop
        elif set(parameter) == {"implicit_int"}:
            spelling = parameter["implicit_int"]
            if type(spelling) is not str or not _C_PARAMETER.fullmatch(spelling):
                raise ValueError(f"invalid implicit int parameter in {name}")
            declared.append(b"int " + spelling.encode())
        else:
            raise ValueError(f"invalid C parameter record in {name}")
    replacement = b"(" + b", ".join(declared) + b")"
    return replacement + b"\n" * (
        data[begin:end].count(b"\n") - replacement.count(b"\n")
    )


def _c_source_edits(name, data, adaptation):
    """Render sealed C declaration, conversion and identifier-list records."""
    if (
        type(data) is not bytes
        or type(adaptation) is not dict
        or set(adaptation) != _C_ADAPTATIONS
        or any(type(rows) not in (tuple, list) for rows in adaptation.values())
    ):
        raise ValueError(f"invalid C ABI adaptation in {name}")
    size = len(data)
    insertions, replacements = {}, []
    prior = 0
    for start, stop in sorted(
        _source_span(span, size, name) for span in adaptation["linkage"]
    ):
        if start < prior:
            raise ValueError(f"overlapping C linkage declarations in {name}")
        insertions.setdefault(start, []).append(b'extern "C" { ')
        insertions.setdefault(stop, []).append(b" }")
        prior = stop

    prior = 0
    for conversion in adaptation["pointer_conversions"]:
        if type(conversion) is not dict or set(conversion) != {"span", "type"}:
            raise ValueError(f"invalid C pointer conversion in {name}")
        start, stop = _source_span(conversion["span"], size, name)
        target = conversion["type"]
        if type(target) is not str or not target.strip():
            raise ValueError(f"invalid C pointer target type in {name}")
        if start < prior:
            raise ValueError(
                f"overlapping or unordered C pointer conversions in {name} at byte {start}"
            )
        insertions.setdefault(start, []).append(f"static_cast<{target}>(".encode())
        insertions.setdefault(stop, []).insert(0, b")")
        prior = stop

    prior = 0
    for definition in adaptation["identifier_definitions"]:
        if type(definition) is not dict or set(definition) != {"span", "parameters"}:
            raise ValueError(f"invalid C identifier-list record in {name}")
        start, stop = _source_span(definition["span"], size, name)
        if start < prior:
            raise ValueError(
                f"overlapping or unordered C definitions in {name} at byte {start}"
            )
        replacements.append(
            (
                start,
                stop,
                _ansi_parameters(data, start, stop, definition["parameters"], name),
            )
        )
        prior = stop
    return insertions, replacements


def adapt_c_source(name, data, declarations, conversions, definitions, regions):
    """Return C++-compatible bytes and relocated spans unchanged by the edits."""
    adaptation = _c_adaptation(name, data, declarations, conversions, definitions)
    insertions, replacements = _c_source_edits(name, data, adaptation)
    edits = {start: (stop, replacement) for start, stop, replacement in replacements}
    chunks, position = [], 0
    for offset in sorted(insertions.keys() | edits.keys()):
        if offset < position:
            raise ValueError(f"overlapping C source edits in {name}")
        chunks.extend((data[position:offset], *insertions.get(offset, ())))
        position = offset
        if offset in edits:
            position, replacement = edits[offset]
            chunks.append(replacement)
    converted = b"".join((*chunks, data[position:]))
    anchors = []
    for region, (start, stop) in _c_region_anchors(adaptation, regions):
        shift = sum(
            len(edit)
            for offset, edits_at_offset in insertions.items()
            if offset <= start
            for edit in edits_at_offset
        ) + sum(
            len(replacement) - (end - begin)
            for begin, end, replacement in replacements
            if end <= start
        )
        new_span = start + shift, stop + shift
        if converted[new_span[0] : new_span[1]] != data[start:stop]:
            raise ValueError(f"unbound C region source span in {name}")
        anchors.append((region, new_span))
    return converted, anchors
