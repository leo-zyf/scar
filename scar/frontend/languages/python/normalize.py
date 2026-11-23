"""F2: normalize captured Python facts into structured operation bodies."""

import ast
from .module import _ModuleGraph
from .numeric import Unsupported, _array_function, _names, _scalar


def normalize_unit(parsed):
    """F2: turn captured syntax and lexical facts into source-independent operations."""
    path, root, abi, tree = parsed.path, parsed.root, parsed.abi, parsed.tree
    if tree is None:
        return dict(functions=[], externs=[], diagnostics=list(parsed.diagnostics))
    if abi.get("kind") == "python_module":
        return _ModuleGraph(parsed, abi).translate()
    definitions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    exports, imports = abi.get("exports", {}), abi.get("imports", {})
    symbols = {**{name: exports.get(name, name) for name in definitions}, **imports}
    functions, externs, diagnostics = [], [], []
    partial = set()
    initialization = False

    def problem(node, error, symbol=None):
        diagnostics.append(
            dict(
                code="python." + getattr(error, "kind", type(node).__name__),
                detail=str(error),
                symbol=symbol,
                file=str(path.relative_to(root)),
                line=getattr(error, "line", node.lineno),
            )
        )

    for fn in definitions.values():
        symbol = symbols[fn.name]
        try:
            names = _names(fn)
            if abi.get("kind") == "binary_f64_arrays":
                f = _array_function(fn, names, symbols, fn.name == abi["entry"])
            else:
                explicit = fn.name in exports
                if not explicit and any(
                    not isinstance(a.annotation, ast.Name) or a.annotation.id != "float"
                    for a in fn.args.args
                ):
                    raise Unsupported(
                        fn,
                        "function parameters need a typed binding or float annotation",
                    )
                if len(fn.body) != 1 or not isinstance(fn.body[0], ast.Return):
                    raise Unsupported(fn, "scalar function needs a single value return")
                f = dict(
                    name=symbol,
                    result="f64",
                    params=[dict(name=n, type="f64") for n in names],
                    body=[
                        dict(
                            op="return", value=_scalar(fn.body[0].value, names, symbols)
                        )
                    ],
                )
            f["location"] = dict(file=str(path.relative_to(root)), line=fn.lineno)
            functions.append(f)
        except Unsupported as exc:
            problem(fn, exc, symbol)
            partial.add(symbol)
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) and not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            problem(
                node,
                Unsupported(
                    node, "module initialization requires executable semantics"
                ),
            )
            initialization = True
    for fn in definitions.values():
        for call in ast.walk(fn):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id in imports
            ):
                externs.append(
                    dict(
                        name=imports[call.func.id],
                        params=[
                            dict(name=f"p{i}", type="f64")
                            for i in range(len(call.args))
                        ],
                        result="f64",
                    )
                )
    if partial or initialization:
        module = ".".join(path.relative_to(root).with_suffix("").parts)
        profile = dict(
            module=module,
            main=False,
            project={
                module: {
                    "path": path.relative_to(root).as_posix(),
                    "package": path.name == "__init__.py",
                    "exports": parsed.exports,
                }
            },
        )
        evidence = _ModuleGraph(parsed, profile, aliases=symbols)
        unit = evidence.translate()
        if initialization:
            partial.add(evidence.root_symbol)
        functions.extend(
            f
            for f in unit["functions"]
            if f["name"] in partial or ".<locals>." in f["name"]
        )
        diagnostics.extend(
            d
            for d in unit["diagnostics"]
            if d["symbol"] in partial or ".<locals>." in d["symbol"]
        )
        externs.extend(unit["externs"])
    return dict(functions=functions, externs=externs, diagnostics=diagnostics)
