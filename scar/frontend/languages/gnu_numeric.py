"""Common F1 interpretation of GNU compiler -Q option rows."""

import re

_NUMERIC_OPTIONS = {
    "math_errno": "-fmath-errno",
    "rounding_math": "-frounding-math",
    "trapping_math": "-ftrapping-math",
    "signed_zeros": "-fsigned-zeros",
    "finite_only": "-ffinite-math-only",
    "fp_contract": "-ffp-contract",
    "unsafe_math": "-funsafe-math-optimizations",
    "associative_math": "-fassociative-math",
    "reciprocal_math": "-freciprocal-math",
    "signaling_nans": "-fsignaling-nans",
}


def numeric_option_rows(result, *, fortran=False, extra_rows=()):
    options = dict(_NUMERIC_OPTIONS)
    if fortran:
        options["protect_parens"] = "-fprotect-parens"
    wanted = {*options.values(), *extra_rows}
    rows = {}
    for line in (result.stdout + result.stderr).decode(errors="replace").splitlines():
        match = re.match(r"^\s*(-[fm][\w-]+)(?:=\[[^]]+\])?\s+(.*?)\s*$", line)
        if match and match[1] in wanted:
            rows[match[1]] = match[2]
    policy, unresolved = {}, []
    for name, option in options.items():
        chosen = rows.get(option)
        if chosen in {"[enabled]", "[disabled]"}:
            policy[name] = chosen == "[enabled]"
        elif name == "fp_contract" and chosen in {"off", "on", "fast"}:
            policy[name] = chosen
        else:
            policy[name] = None
            unresolved.append(
                {"field": name, "reason": "option_query_has_no_explicit_value"}
            )
    return rows, policy, unresolved
