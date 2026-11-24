"""M2 scalar presentation, container display and ordered formatted fields."""

import re
import unicodedata

from .values import (
    SCALARS,
    Value,
    literal,
    rank,
    runtime,
    protocols,
    wire,
    scalar_spelling,
    _UNKNOWN,
)

SCALAR_FORMAT = dict(
    boolean_true="True",
    boolean_false="False",
    unit="None",
    integer_base=10,
    float="shortest_binary64",
)
_CONVERSION = {-1: "none", ord("s"): "str", ord("r"): "repr", ord("a"): "ascii"}
_GRAMMAR = re.compile(
    r"(?:(?P<fill>.)?(?P<align>[<>=^]))?(?P<sign>[+\- ])?(?P<negative_zero>z)?"
    r"(?P<alternate>#)?(?P<zero>0)?(?P<width>\d*)(?P<grouping>[,_])?"
    r"(?:\.(?P<precision>\d+))?(?P<presentation>[bcdeEfFgGnosxX%])?",
    re.DOTALL,
)


def _profile(frame):
    return frame.project.text_profiles.get(frame.scope.get("text_profile"), {})


def _interpreter(frame):
    match = re.match(r"cpython (\d+)\.(\d+)", _profile(frame).get("interpreter", ""))
    return tuple(map(int, match.groups())) if match else ()


def _spec(frame, node, source):
    if source.kind != "text" or type(source.exact) is not str:
        frame.fail(
            node, "format_spec", "the format specification requires a proven text value"
        )
    raw = source.exact
    fields = _GRAMMAR.fullmatch(raw)
    if not fields:
        if re.search(r"\.\d*[,_]", raw):
            frame.fail(
                node,
                "format_fraction_grouping",
                "fractional digit grouping needs a separate presentation rule",
            )
        frame.fail(node, "format_spec", f"unsupported format specification {raw!r}")
    if fields["negative_zero"] and not (3, 11) <= _interpreter(frame) < (3, 15):
        frame.fail(
            node,
            "format_grammar_version",
            "negative-zero normalization requires a verified source format grammar",
        )

    def quantity(key, default, maximum):
        digits = fields[key]
        if not digits:
            return default
        if (
            not digits.isascii()
            and _profile(frame).get("unicode_version") != unicodedata.unidata_version
        ):
            frame.fail(
                node,
                "format_digits",
                "non-ASCII width or precision requires matching source Unicode decimal classifications",
            )
        value = 0
        for digit in digits:
            value = 10 * value + int(digit)
            if value > maximum:
                frame.fail(
                    node,
                    "format_size",
                    f"{key} exceeds the supported size of {maximum}",
                )
        return value

    return dict(
        fill=fields["fill"] or " ",
        align=fields["align"] or "auto",
        sign=fields["sign"] or "-",
        negative_zero=bool(fields["negative_zero"]),
        alternate=bool(fields["alternate"]),
        zero=bool(fields["zero"]),
        width=quantity("width", 0, (1 << 63) - 1),
        grouping=fields["grouping"] or "",
        precision=quantity("precision", -1, (1 << 31) - 1),
        presentation=fields["presentation"] or "",
        sign_explicit=fields["sign"] is not None,
        fill_explicit=fields["fill"] is not None,
        empty=not raw,
    )


def _concat(frame, parts, steps, discard=False):
    if not parts:
        return literal("")
    if len(parts) == 1:
        return parts[0]
    facts = {}
    if all(type(v.exact) is str for v in parts):
        facts["exact"] = "".join(v.exact for v in parts)
    return frame.effect(
        dict(
            op="text_concat",
            parts=[v.value for v in parts],
            type="text",
            effects=["VAL", "MEM"],
        ),
        "text",
        steps,
        discard,
        facts=facts,
    )


def array_text(frame, node, value, steps):
    frame.project.array_ops.add("array_format")
    return frame.effect(
        dict(
            op="array_format",
            value=frame.scalar(value, node),
            type="text",
            options=dict(
                precision=8,
                threshold=1000,
                edgeitems=3,
                linewidth=75,
                suppress=False,
                sign="-",
                floatmode="maxprec",
                legacy=False,
            ),
            effects=["VAL", "MEM", "STATE"],
        ),
        "text",
        steps,
    )


def _numpy_major(frame, node):
    provider = frame.project.providers.get("numpy", {})
    major = provider.get("version", "").partition(".")[0]
    if (
        major not in {"1", "2"}
        or provider.get("distribution", "").lower() != "numpy"
        or not provider.get("sha256")
    ):
        frame.fail(
            node,
            "scalar_repr",
            "NumPy scalar display needs the verified source distribution",
        )
    return major


def _scalar_field(frame, node, value, conversion, spec, steps, discard=False):
    profile = frame.scope.get("text_profile")
    if (
        conversion == "repr"
        and value.kind == "text"
        and profile not in frame.project.text_profiles
    ):
        frame.fail(
            node,
            "format_unicode",
            "text representation requires the source Unicode printability profile",
        )
    if conversion == "none" and value.kind in {
        "bool",
        "numpy_bool",
        "i64",
        "f64",
        "numpy_f64",
    }:
        if spec["presentation"] == "n":
            frame.fail(
                node,
                "format_locale",
                "locale-aware numeric presentation requires the source numeric locale",
            )
        if spec["presentation"] == "c" and value.kind == "i64":
            lo, hi = value.bounds
            if lo <= 0xDFFF and hi >= 0xD800:
                frame.fail(
                    node,
                    "format_character",
                    "the character range can produce a surrogate outside the text encoding contract",
                )
    facts = {}
    spelling = SCALAR_FORMAT
    if value.kind == "numpy_bool":
        major = _numpy_major(frame, node)
        if major == "2" and conversion in {"repr", "ascii"}:
            spelling = dict(
                SCALAR_FORMAT, boolean_true="np.True_", boolean_false="np.False_"
            )
    if spec["empty"] and conversion in {"none", "str"}:
        # Preserve small exact specifications assembled from literal text and integer
        # fields. Runtime rendering still occurs at the field's source evaluation point.
        if (rendered := scalar_spelling(value.exact, spelling)) is not _UNKNOWN:
            facts["exact"] = rendered
    return frame.effect(
        dict(
            op="format_value",
            value=frame.scalar(value, node),
            conversion=conversion,
            spec=spec,
            type="text",
            unicode_profile=profile,
            scalar_format=spelling,
            effects=["VAL", "MEM"],
            may_raise=True,
        ),
        "text",
        steps,
        discard,
        facts=facts,
    )


def _container(frame, node, value, steps, *, ascii_only=False):
    opening, closing = ("[", "]") if value.kind == "list" else ("(", ")")
    if value.kind == "tuple" and len(value.value) == 1:
        closing = ",)"
    parts = [literal(opening)]
    for i, item in enumerate(value.value):
        if i:
            parts.append(literal(", "))
        parts.append(
            render(
                frame, node, item, "ascii" if ascii_only else "repr", literal(""), steps
            )
        )
    return _concat(frame, [*parts, literal(closing)], steps)


def display(frame, node, value, steps):
    """Prepare aggregates for the existing stream-print scalar contract."""
    if value.kind in {"list", "tuple"}:
        return _container(frame, node, value, steps)
    if rank(value.kind) is not None:
        return array_text(frame, node, value, steps)
    if value.kind == "numpy_bool":
        _numpy_major(frame, node)
        # The stream receives str(value); the original program binding keeps its kind.
        return Value("bool", value.value, value.facts)
    if len(protocols(value.kind)) > 1:
        # str() shares the same contract on these primitive protocol domains.
        # The source binding is unchanged, and no cast or runtime tag is emitted.
        return Value(wire(value.kind), value.value, value.facts)
    return value


def render(frame, node, value, conversion, source_spec, steps, discard=False):
    spec = _spec(frame, node, source_spec)
    if len(protocols(value.kind)) > 1:
        if conversion in {"repr", "ascii"}:
            frame.fail(
                node,
                "scalar_observation",
                "representation observes a selected scalar's source protocol; no common representation contract is proven",
            )
        value = Value(wire(value.kind), value.value, value.facts)
    source = value.kind
    if rank(source) is not None:
        if conversion in {"repr", "ascii"}:
            frame.fail(
                node,
                "array_repr",
                "array constructor-style representation needs a dtype and shape display contract",
            )
        if conversion == "none" and not spec["empty"]:
            frame.fail(
                node,
                "array_format_spec",
                "nonempty array presentation requires a rank-specific format contract",
            )
        value = array_text(frame, node, value, steps)
    elif source in {"list", "tuple"}:
        if conversion == "none" and not spec["empty"]:
            frame.fail(
                node,
                "sequence_format_spec",
                "container formatting requires an empty specification or explicit text conversion",
            )
        value = _container(frame, node, value, steps, ascii_only=conversion == "ascii")
    elif source == "handle:error" and (
        conversion == "str" or conversion == "none" and spec["empty"]
    ):
        value = frame.effect(
            dict(
                op="host_call",
                interface="error.message",
                semantics_version=1,
                args=[frame.scalar(value, node)],
                type="text",
                effects=["VAL"],
                reads=["error.value"],
                writes=[],
                may_raise=True,
            ),
            "text",
            steps,
        )
    elif source == "numpy_f64" and conversion in {"repr", "ascii"}:
        if _numpy_major(frame, node) == "2":
            inner = _scalar_field(
                frame, node, value, "repr", _spec(frame, node, literal("")), steps
            )
            value = _concat(frame, [literal("np.float64("), inner, literal(")")], steps)
        else:
            return _scalar_field(frame, node, value, conversion, spec, steps, discard)
    elif source in SCALARS:
        return _scalar_field(frame, node, value, conversion, spec, steps, discard)
    else:
        frame.fail(
            node,
            "format_dispatch",
            f"{source} requires a closed conversion and format method",
        )
    if spec["empty"]:
        return value
    return _scalar_field(frame, node, value, "none", spec, steps, discard)


def evaluate(frame, node, steps, discard=False):
    if node["op"] == "format_string":
        return _concat(
            frame,
            [frame.evaluate(item, steps) for item in node["items"]],
            steps,
            discard,
        )
    value = frame.evaluate(node["value"], steps)
    source_spec = literal("")
    conversion = _CONVERSION.get(node["conversion"])
    if conversion is None:
        frame.fail(node, "format_conversion", "unrecognized formatted-field conversion")
    if node.get("format") is not None:
        if conversion != "none":
            version = _interpreter(frame)
            if (3, 13) <= version < (3, 15):
                # CONVERT_VALUE precedes evaluation of the nested spec on these versions.
                value = render(frame, node, value, conversion, source_spec, steps)
                conversion = "none"
            elif not (3, 10) <= version < (3, 13):
                frame.fail(
                    node,
                    "format_evaluation_order",
                    "conversion and nested specification require a verified interpreter evaluation order",
                )
        # The width/precision expressions may rebind the variable holding the value.
        if runtime(value.kind) and value.value["op"] != "literal":
            value = frame.effect(value.value, value.kind, steps, facts=value.facts)
        source_spec = frame.evaluate(node["format"], steps)
    return render(frame, node, value, conversion, source_spec, steps, discard)
