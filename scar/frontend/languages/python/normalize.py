"""F2: normalize captured Python facts into structured operation bodies."""

import ast
from collections import Counter

from scar.contract import walk_operations

from .module import _ModuleGraph
from .numeric import Unsupported, _array_function, _names, _scalar_function


def normalize_unit(parsed):
    """F2: turn captured syntax and lexical facts into source-independent operations."""
    path, root, abi, tree = parsed.path, parsed.root, parsed.abi, parsed.tree
    if tree is None:
        return dict(functions=[], externs=[], diagnostics=list(parsed.diagnostics))
    if abi.get("kind") == "python_module":
        return _ModuleGraph(parsed, abi).translate()
    definitions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    declaration_counts = Counter(
        n.name for n in tree.body if isinstance(n, ast.FunctionDef)
    )
    exports, imports = abi.get("exports", {}), abi.get("imports", {})
    symbols = {**{name: exports.get(name, name) for name in definitions}, **imports}
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
    protocols = {f["name"]: f for f in unit["functions"]}
    # The scalar ABI recognizes the standard builtin float annotation. Other
    # annotation protocols and type parameters require executable module setup.
    pinned_float = "float" not in parsed.exports and "float" not in imports
    annotation_setup = {
        creation["location"]["line"]
        for creation in protocols[evidence.root_symbol]["partial_body"]
        if creation["op"] == "python_define"
        and (
            creation.get("type_params")
            or any(
                not (
                    pinned_float
                    and (
                        a.get("value", {}).get("op") == "python_builtin"
                        and a["value"].get("name") == "float"
                        or a.get("syntax")
                        == {"syntax": "Name", "fields": [["id", "float"]]}
                    )
                )
                for a in creation.get("annotations", ())
            )
        )
    }
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
            if fn.lineno in annotation_setup:
                raise Unsupported(
                    fn,
                    "function needs its source annotation or type-parameter protocol",
                )
            if declaration_counts[fn.name] != 1:
                raise Unsupported(
                    fn, "repeated definitions require sequential module binding"
                )
            _names(fn)
            if symbol not in protocols:
                raise Unsupported(
                    fn, "function definition does not have one closed binding"
                )
            if abi.get("kind") == "binary_f64_arrays":
                f = _array_function(protocols[symbol], fn.name == abi["entry"])
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
                f = _scalar_function(protocols[symbol])
            f["location"] = dict(file=str(path.relative_to(root)), line=fn.lineno)
            functions.append(f)
        except Unsupported as exc:
            problem(fn, exc, symbol)
            partial.add(symbol)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and (
            node.decorator_list
            or node.args.defaults
            or any(d is not None for d in node.args.kw_defaults)
            or node.lineno in annotation_setup
        ):
            problem(
                node,
                Unsupported(
                    node, "function creation requires module initialization effects"
                ),
            )
            initialization = True
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
    imported = set(imports.values())
    for call in walk_operations(functions):
        if call["op"] == "call" and call["callee"] in imported:
            externs.append(
                dict(
                    name=call["callee"],
                    params=[
                        dict(name=f"p{i}", type="f64") for i in range(len(call["args"]))
                    ],
                    result="f64",
                )
            )
    if partial or initialization:
        if initialization:
            partial.add(evidence.root_symbol)
        completed = {f["name"] for f in functions}
        functions.extend(
            f
            for f in unit["functions"]
            if f["name"] not in completed
            and (initialization or f["name"] != evidence.root_symbol)
        )
        diagnostics.extend(
            d
            for d in unit["diagnostics"]
            if d["symbol"] not in completed
            and (initialization or d["symbol"] != evidence.root_symbol)
        )
        externs.extend(unit["externs"])
    return dict(functions=functions, externs=externs, diagnostics=diagnostics)
