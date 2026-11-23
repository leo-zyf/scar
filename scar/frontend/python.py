"""Build Python module/control/value graphs and lower sealed numeric entry ABIs."""

import ast
import builtins
from collections import deque
from importlib.util import resolve_name
import math
from pathlib import Path
import shlex
import sys
import symtable
import tokenize

_OPS = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div"}


class Unsupported(ValueError):
    def __init__(self, node, detail):
        self.line = getattr(node, "lineno", 1)
        self.kind = type(node).__name__
        super().__init__(detail)


def var(name):
    return dict(op="var", name=name)


def literal(value):
    if isinstance(value, str):
        return dict(op="string", value=value)
    return dict(op="literal", type="f64", value=float(value))


def _coefficient(node):
    if (
        type(node.value) not in (int, float)
        or not math.isfinite(node.value)
        or abs(node.value) > 2**53
    ):
        raise Unsupported(
            node, "constant exceeds the declared float64 arithmetic domain"
        )
    return literal(node.value)


def _names(fn):
    args = fn.args
    if (
        fn.decorator_list
        or args.defaults
        or args.posonlyargs
        or args.kwonlyargs
        or args.vararg
        or args.kwarg
    ):
        raise Unsupported(
            fn, "Python callable requires runtime argument binding or decorators"
        )
    return [a.arg for a in args.args]


def _scalar(node, bound, symbols):
    if isinstance(node, ast.Name) and node.id in bound:
        return var(node.id)
    if isinstance(node, ast.Constant):
        return _coefficient(node)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return dict(
            op="unary",
            operator="neg" if isinstance(node.op, ast.USub) else "pos",
            value=_scalar(node.operand, bound, symbols),
        )
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        if isinstance(node.op, ast.Div) and not (
            isinstance(node.right, ast.Constant) and node.right.value != 0
        ):
            raise Unsupported(
                node, "division needs the Python zero-divisor exception contract"
            )
        return dict(
            op="binary",
            operator=_OPS[type(node.op)],
            left=_scalar(node.left, bound, symbols),
            right=_scalar(node.right, bound, symbols),
        )
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in symbols
        and not node.keywords
    ):
        return dict(
            op="call",
            callee=symbols[node.func.id],
            args=[_scalar(x, bound, symbols) for x in node.args],
        )
    raise Unsupported(node, "unresolved Python operation or dynamic dispatch")


def _array_function(fn, names, symbols, entry):
    array_names = names[:-1] if entry else names
    if not array_names:
        raise Unsupported(fn, "array entry needs at least one numerical input")
    element, count, output = "$index", "$count", "$output"
    n = var("$elements")
    ptr = {"kind": "ptr", "to": {"kind": "const", "of": "f64"}}
    internal = {}
    prefix = []
    serial = 0

    def at(base):
        return dict(op="index", base=base, indices=[var(element)])

    def copy(target, expr):
        return dict(
            op="map",
            index=element,
            extent=n,
            independent=True,
            proof="fresh private result or owned destination; same-index accesses",
            body=[dict(op="assign", target=at(var(target)), value=expr)],
        )

    def scratch(statements):
        nonlocal serial
        serial += 1
        name = f"$temporary{serial}"
        statements.append(dict(op="buffer", name=name, type="f64", extent=n))
        return name

    def evaluate(node, statements, bound):
        if isinstance(node, ast.Name) and node.id in bound:
            return var(internal.get(node.id, node.id)), True
        if isinstance(node, ast.Constant):
            return _coefficient(node), False
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value, array = evaluate(node.operand, statements, bound)
            if not array:
                return dict(op="unary", operator="neg", value=value), False
            dest = scratch(statements)
            statements.append(
                copy(dest, dict(op="unary", operator="neg", value=at(value)))
            )
            return var(dest), True
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            a, aa = evaluate(node.left, statements, bound)
            b, ba = evaluate(node.right, statements, bound)
            if not (aa or ba):
                raise Unsupported(
                    node, "scalar-only subtree needs Python scalar promotion"
                )
            if isinstance(node.op, ast.Div):
                raise Unsupported(
                    node, "array divide requires NumPy error-mode effects"
                )
            dest = scratch(statements)
            expr = dict(
                op="binary",
                operator=_OPS[type(node.op)],
                left=at(a) if aa else a,
                right=at(b) if ba else b,
            )
            statements.append(copy(dest, expr))
            return var(dest), True
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in symbols
            and not node.keywords
        ):
            args = [evaluate(a, statements, bound) for a in node.args]
            if not all(is_array for _, is_array in args):
                raise Unsupported(node, "array call has a nonarray argument")
            dest = scratch(statements)
            statements.append(
                dict(
                    op="expr",
                    value=dict(
                        op="call",
                        callee=symbols[node.func.id],
                        args=[n, *[v for v, _ in args], var(dest)],
                    ),
                )
            )
            return var(dest), True
        raise Unsupported(
            node, "dynamic Python expression outside the declared array ABI"
        )

    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name = node.targets[0].id
            if name not in internal:
                internal[name] = "$local_" + name
                prefix.append(
                    dict(op="buffer", name=internal[name], type="f64", extent=n)
                )
                if name in array_names:
                    prefix.append(copy(internal[name], at(var(name))))

    def statements(nodes, bound, inner=False):
        result = []
        for node in nodes:
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                continue
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                name = node.targets[0].id
                v, is_array = evaluate(node.value, result, bound)
                if not is_array:
                    raise Unsupported(node, "array variable assigned a scalar")
                result.append(copy(internal[name], at(v)))
                bound.add(name)
            elif isinstance(node, ast.Return) and not inner:
                v, is_array = evaluate(node.value, result, bound)
                if not is_array:
                    raise Unsupported(node, "array entry returned a scalar")
                result.extend([copy(output, at(v)), dict(op="return")])
            elif (
                entry
                and isinstance(node, ast.For)
                and isinstance(node.target, ast.Name)
                and not node.orelse
                and isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "range"
                and not node.iter.keywords
                and len(node.iter.args) == 1
                and isinstance(node.iter.args[0], ast.Name)
                and node.iter.args[0].id == names[-1]
                and node.target.id not in bound
                and node.target.id != names[-1]
            ):
                i = "$iteration"
                result.append(
                    dict(
                        op="for",
                        init=[
                            dict(
                                op="decl",
                                name=i,
                                type="i64",
                                init=dict(op="literal", type="i64", value=0),
                            )
                        ],
                        condition=dict(
                            op="binary", operator="lt", left=var(i), right=var(count)
                        ),
                        step=dict(op="unary", operator="preinc", value=var(i)),
                        body=statements(node.body, bound.copy(), True),
                    )
                )
            else:
                raise Unsupported(node, "statement outside the closed array profile")
        return result

    if not isinstance(fn.body[-1], ast.Return):
        raise Unsupported(fn, "array function needs a total return")
    params = [dict(name="$elements", type="i64")]
    if entry:
        params.append(dict(name=count, type="i64"))
    params.extend(dict(name=name, type=ptr) for name in array_names)
    params.append(dict(name=output, type={"kind": "ptr", "to": "f64"}))
    return dict(
        name=symbols[fn.name],
        result="void",
        params=params,
        body=prefix + statements(fn.body, set(array_names)),
    )


def translate(path, flags, *, root, abi):
    try:
        tree = _parse(path)
    except SyntaxError as error:
        location = dict(
            file=path.relative_to(root).as_posix(),
            line=error.lineno,
            column=error.offset,
        )
        issue = dict(code="python.syntax", detail=error.msg, field="syntax", **location)
        return dict(functions=[], externs=[], diagnostics=[issue])
    if abi.get("kind") == "python_module":
        return _ModuleGraph(path, root, abi, tree).translate()
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
                    "exports": _catalog(tree, module),
                }
            },
        )
        evidence = _ModuleGraph(path, root, profile, tree, aliases=symbols)
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


def _parse(path):
    # Respect the encoding declaration without importing or executing user code.
    with tokenize.open(path) as source:
        return ast.parse(source.read(), filename=str(path))


def _relative(node, module, package):
    if not node.level:
        return node.module or ""
    parent = module if package else module.rpartition(".")[0]
    try:
        return resolve_name("." * node.level + (node.module or ""), parent)
    except (ValueError, ImportError):
        return ""


def _provider(name):
    base = name.split(".", 1)[0]
    return (
        "python_stdlib"
        if base in sys.stdlib_module_names or base == "builtins"
        else "python_package"
    )


def _py_type(name=None):
    return {
        "kind": "python_object",
        "class": name or {"status": "unresolved", "field": "runtime_class"},
    }


def _function(module, qualified):
    return f"py:{module}:{qualified}"


def _literal_fact(value):
    return {"class": type(value).__name__, "values": [value]}


def _catalog(tree, module):
    """Export lexical names and initial values. Mutable globals remain state edges."""
    exports = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            exports[node.name] = {
                "class": "type" if isinstance(node, ast.ClassDef) else "function",
                "symbol": _function(module, node.name),
            }
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            for target in targets:
                if isinstance(target, ast.Name):
                    fact = {}
                    if isinstance(value, ast.Constant) and type(value.value) in (
                        bool,
                        str,
                        int,
                        float,
                        type(None),
                    ):
                        fact = _literal_fact(value.value)
                    exports[target.id] = {**fact, "state": _function(module, target.id)}
    return exports


def project_units(entry: Path, root: Path) -> list[dict]:
    """Discover source-root modules and package initializers for syntactic imports."""
    root = Path(root).resolve()
    entry = Path(entry)
    entry = (root / entry).resolve() if not entry.is_absolute() else entry.resolve()

    def module_path(path):
        relative = path.relative_to(root).with_suffix("")
        pieces = relative.parts[:-1] if relative.name == "__init__" else relative.parts
        return ".".join(pieces) or "__main__"

    def resolve(module):
        if not module or not all(part.isidentifier() for part in module.split(".")):
            return None
        stem = root.joinpath(*module.split("."))
        # CPython's FileFinder tries the package before its sibling module.
        for path in (stem / "__init__.py", stem.with_suffix(".py")):
            if path.is_file() and path.resolve().is_relative_to(root):
                return path.resolve()
        return None

    entry_module = module_path(entry)
    catalog = {}
    pending = deque([(entry_module, entry)])
    scheduled = {entry_module}

    def enqueue(module):
        parts = module.split(".")
        for i in range(1, len(parts) + 1):
            candidate = ".".join(parts[:i])
            path = resolve(candidate)
            if path and candidate not in scheduled:
                scheduled.add(candidate)
                pending.append((candidate, path))

    while pending:
        module, path = pending.popleft()
        package = path.name == "__init__.py"
        try:
            tree = _parse(path)
        except SyntaxError:
            catalog[module] = {
                "path": path.relative_to(root).as_posix(),
                "package": package,
                "exports": {},
            }
            continue
        catalog[module] = {
            "path": path.relative_to(root).as_posix(),
            "package": package,
            "exports": _catalog(tree, module),
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    enqueue(name.name)
            elif isinstance(node, ast.ImportFrom):
                source = _relative(node, module, package)
                if source:
                    enqueue(source)
                    for name in node.names:
                        enqueue(source + "." + name.name)
    return [
        dict(
            path=spec["path"],
            language="python",
            abi=dict(
                kind="python_module",
                module=name,
                main=name == entry_module,
                project=catalog,
            ),
        )
        for name, spec in catalog.items()
    ]


def _join_facts(*facts):
    """Only retain facts shared by every incoming control path."""
    if not facts:
        return {}
    unbound = any(f.get("maybe_unbound") for f in facts)
    facts = [f for f in facts if f != {"maybe_unbound": True}]
    if not facts:
        return {"maybe_unbound": True}
    common = {
        k: v
        for k, v in facts[0].items()
        if k not in {"values", "maybe_unbound"}
        and all(f.get(k) == v for f in facts[1:])
    }
    if all("values" in f for f in facts):
        values = []
        for f in facts:
            for v in f["values"]:
                if not any(type(v) == type(w) and v == w for w in values):
                    values.append(v)
        if len(values) <= 16:
            common["values"] = values
    if unbound:
        common["maybe_unbound"] = True
    return common


def _command_shapes(values, shell):
    """Expose literal argument vectors. Shell expansion needs runtime state."""
    shapes = []
    for value in values:
        if isinstance(value, str) and shell is False:
            words = [value]
        elif (
            isinstance(value, list)
            and all(isinstance(word, str) for word in value)
            and shell is False
        ):
            words = value
        elif (
            isinstance(value, str)
            and shell is True
            and not any(c in value for c in "|&;<>$`*?[]{}~#\n")
        ):
            try:
                words = shlex.split(value)
            except ValueError:
                continue
        else:
            continue
        if words:
            shape = {"argv": words, "executable": words[0]}
            if Path(words[0]).name in {"mpirun", "mpiexec", "orterun"}:
                start = 1
                ranks = None
                if (
                    len(words) > 3
                    and words[1] in {"-n", "-np", "--np"}
                    and words[2].isdecimal()
                ):
                    start = 3
                    ranks = int(words[2])
                if len(words) > start and not words[start].startswith("-"):
                    shape["mpi_program"] = {
                        "argv": words[start:],
                        "executable": words[start],
                        "ranks": ranks,
                    }
            shapes.append(shape)
    return shapes


def _static_values(operator, left, right):
    if operator != "add" or "values" not in left or "values" not in right:
        return {}
    result = []
    for a in left["values"]:
        for b in right["values"]:
            if type(a) is not str or type(b) is not str or len(a) + len(b) > 4096:
                return {}
            if a + b not in result:
                result.append(a + b)
            if len(result) > 16:
                return {}
    return {"class": "str", "values": result}


class _ModuleGraph:
    """Own one file and retain structured evidence separately from executable ABI bodies."""

    def __init__(self, path, root, abi, tree, aliases=None):
        self.path = path.relative_to(root).as_posix()
        self.module = abi["module"]
        self.main = abi["main"]
        self.tree = tree
        self.project = abi["project"]
        self.functions = []
        self.externs = {}
        self.diagnostics = []
        self.aliases = aliases or {}
        self._seen = set()
        self.initial = {}
        self.package = self.project[self.module]["package"]
        self.root_symbol = "__main__" if self.main else _function(self.module, "<init>")
        with tokenize.open(path) as source:
            self.table = symtable.symtable(source.read(), str(path), "exec")

    def location(self, node):
        return dict(
            file=self.path,
            line=getattr(node, "lineno", 1),
            column=getattr(node, "col_offset", 0) + 1,
            end_line=getattr(node, "end_lineno", 1),
            end_column=getattr(node, "end_col_offset", 0) + 1,
        )

    def issue(self, node, symbol, field, detail):
        location = self.location(node)
        key = (symbol, location["line"], location["column"], field)
        if key not in self._seen:
            self._seen.add(key)
            self.diagnostics.append(
                dict(
                    code="python." + field,
                    symbol=symbol,
                    field=field,
                    detail=detail,
                    **location,
                )
            )
        return {"status": "unresolved", "field": field, "reason": detail}

    def dependency(self, target):
        self.externs.setdefault(
            "py:external:" + target,
            dict(
                name="py:external:" + target,
                provider=_provider(target),
                symbol=target,
                calling_convention="python",
                params=[],
                variadic=True,
                result=_py_type(),
            ),
        )
        return "py:external:" + target

    def module_bindings(self, node):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                module = alias.name if alias.asname else bound
                yield alias.name, bound, {"class": "module", "module": module}
            return
        base = _relative(node, self.module, self.package)
        for alias in node.names:
            name = alias.name
            sub = ".".join(x for x in (base, name) if x)
            if sub in self.project:
                yield sub, alias.asname or name, {"class": "module", "module": sub}
            elif base in self.project:
                fact = self.project[base]["exports"].get(
                    name, {"state": _function(base, name)}
                )
                yield base, alias.asname or name, fact
            else:
                yield base, alias.asname or name, {"external": sub}

    def translate(self):
        # Function bodies may resolve imports from module state. Execution order
        # of a module initializer is maintained separately by its own graph.
        self.initial = {**self.project[self.module]["exports"]}
        self.initial.update(
            {n: {"class": "function", "symbol": s} for n, s in self.aliases.items()}
        )
        for node in self.tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for _, name, fact in self.module_bindings(node):
                    self.initial[name] = fact
        root = _Scope(self, self.root_symbol, "", self.table, globals_only=True)
        root.env["__name__"] = _literal_fact("__main__" if self.main else self.module)
        self.functions.append(root.definition(self.tree, self.tree.body, [], {}))
        return dict(
            functions=self.functions,
            externs=list(self.externs.values()),
            diagnostics=self.diagnostics,
        )


class _Scope:
    def __init__(self, unit, symbol, qualified, table, parent=None, globals_only=False):
        self.unit = unit
        self.symbol = symbol
        self.qualified = qualified
        self.module_scope = globals_only
        self.table = table
        self.parent = parent
        self.env = {}
        self.calls = set()
        symbols = table.get_symbols() if table else []
        self.locals = {s.get_name() for s in symbols if s.is_local()}
        self.global_names = {s.get_name() for s in symbols if s.is_declared_global()}
        self.nonlocals = {
            s.get_name() for s in symbols if s.is_free() or s.is_nonlocal()
        }
        self.cells = {
            s.get_name()
            for child in (table.get_children() if table else [])
            for s in child.get_symbols()
            if s.is_free()
        } & self.locals

    def issue(self, node, field, detail):
        return self.unit.issue(node, self.symbol, field, detail)

    def record(self, node, op, **fields):
        return {"op": op, "location": self.unit.location(node), **fields}

    def variable(self, name):
        if not self.module_scope and (name in self.cells or name in self.nonlocals):
            owner = self
            while owner and name not in owner.locals:
                owner = owner.parent
            if owner and not owner.module_scope:
                return {
                    **var(name),
                    "cell": {
                        "owner": owner.symbol,
                        "name": name,
                        "identity": "lexical_activation",
                    },
                }
        if (
            self.module_scope
            or name in self.global_names
            or (name not in self.locals and name not in self.nonlocals)
        ):
            return {**var(_function(self.unit.module, name)), "scope": "global"}
        return var(name)

    def fact(self, name):
        if name in self.env:
            return self.env[name]
        if name in self.nonlocals and self.parent:
            return self.parent.fact(name)
        if name in self.locals:
            return {}
        if not self.module_scope and name in self.unit.initial:
            return self.unit.initial[name]
        if hasattr(builtins, name):
            return {"external": "builtins." + name}
        return {}

    def definition(self, node, body, params, initial):
        # CPython's symbol table resolves lexical bindings across all branches.
        self.env.update(initial)
        for p in params:
            if p["name"] in self.cells:
                p["cell"] = self.variable(p["name"])["cell"]
        prelude = (
            []
            if self.module_scope
            else [
                dict(
                    op="decl",
                    name=n,
                    type=_py_type(),
                    initial_binding="unbound",
                    **({"cell": self.variable(n)["cell"]} if n in self.cells else {}),
                )
                for n in sorted(self.locals - {p["name"] for p in params})
            ]
        )
        steps = self.statements(body)
        self.issue(
            body[0] if body else node,
            "native_binding",
            "Python object and effect contracts require target lowering before execution",
        )
        return dict(
            name=self.symbol,
            params=params,
            result="void" if self.module_scope else _py_type(),
            location=self.unit.location(node),
            calling_convention="python",
            calls=sorted(self.calls),
            partial_body=[*prelude, *steps],
        )

    def _external_value(self, target, args, keywords):
        if target == "builtins.str" and len(args) == 1 and "values" in args[0]:
            return {"class": "str", "values": [str(v) for v in args[0]["values"]]}
        if target in {
            "builtins.print",
            "os.chdir",
            "os.mkdir",
            "os.makedirs",
            "shutil.rmtree",
            "time.sleep",
        }:
            return _literal_fact(None)
        if target in {
            "os.path.exists",
            "os.path.isfile",
            "os.path.isdir",
            "builtins.isinstance",
        }:
            return {"class": "bool"}
        if target in {"builtins.len", "builtins.int", "subprocess.Popen.wait"}:
            return {"class": "int"}
        if target == "builtins.range":
            return {"class": "range"}
        if target == "builtins.open":
            return {"class": "io.IOBase"}
        if target == "subprocess.Popen":
            return {"class": "subprocess.Popen"}
        if target in {"builtins.str", "builtins.input", "builtins.format"}:
            return {"class": "str"}
        if target in {
            "numpy.array",
            "numpy.zeros",
            "numpy.empty",
            "numpy.arange",
            "numpy.linspace",
            "numpy.fromfile",
        }:
            result = {"class": "numpy.ndarray"}
            if target in {"numpy.zeros", "numpy.empty"}:
                shape = args[0] if args else keywords.get("shape", {})
                if len(shape.get("values", [])) == 1:
                    dims = shape["values"][0]
                    dims = [dims] if isinstance(dims, int) else dims
                    if isinstance(dims, list) and all(
                        type(d) == int and d >= 0 for d in dims
                    ):
                        result["shape"] = dims
                token = args[1] if len(args) > 1 else keywords.get("dtype")
                if token is None:
                    result["dtype"] = "float64"
                elif len(token.get("values", [])) == 1 and isinstance(
                    token["values"][0], str
                ):
                    result["declared_dtype"] = token["values"][0]
                elif token.get("external"):
                    result["declared_dtype"] = token["external"]
            return result
        return {}

    def expr(self, node):
        R = self.record
        kind = type(node).__name__
        if node is None:
            return {
                "op": "literal",
                "type": _py_type("NoneType"),
                "value": None,
            }, _literal_fact(None)
        if isinstance(node, ast.Constant):
            v = node.value
            if type(v) in (str, int, float, bool, type(None)):
                f = _literal_fact(v)
                value = v
            elif isinstance(v, complex):
                f = {"class": "complex"}
                value = [v.real, v.imag]
            elif isinstance(v, bytes):
                f = {"class": "bytes"}
                value = list(v)
            else:
                f = {"class": type(v).__name__}
                value = {"singleton": type(v).__name__}
            return R(node, "literal", type=_py_type(f["class"]), value=value), f
        if isinstance(node, ast.Name):
            f = self.fact(node.id)
            record = self.variable(node.id)
            if f.get("external", "").startswith("builtins."):
                record = {"op": "python_builtin", "name": node.id}
            evidence = {
                k: f[k]
                for k in (
                    "symbol",
                    "module",
                    "external",
                    "shape",
                    "dtype",
                    "declared_dtype",
                    "maybe_unbound",
                )
                if k in f
            }
            return {
                **record,
                "location": self.unit.location(node),
                "type": _py_type(f.get("class")),
                **({"binding_facts": evidence} if evidence else {}),
            }, f
        if isinstance(node, ast.Attribute):
            base, b = self.expr(node.value)
            name = node.attr
            f = {}
            member = None
            if "module" in b:
                module = b["module"]
                sub = module + "." + name
                if sub in self.unit.project:
                    f = {"class": "module", "module": sub}
                elif module in self.unit.project:
                    f = self.unit.project[module]["exports"].get(name, {})
                    member = _function(module, name)
                else:
                    f = {"external": sub}
            elif "external" in b:
                f = {"external": b["external"] + "." + name}
            elif b.get("class") in {"io.IOBase", "subprocess.Popen", "numpy.ndarray"}:
                f = {"external": b["class"] + "." + name}
            fields = {"base": base, "field": name, "type": _py_type(f.get("class"))}
            if member:
                fields["binding"] = self.variable_reference(member)
            return R(node, "member", **fields), f
        if isinstance(node, ast.Call):
            target, t = self.expr(node.func)
            arguments = [self.expr(n) for n in node.args]
            args = [op for op, _ in arguments]
            facts = [f for _, f in arguments]
            key_items = [(k.arg, self.expr(k.value)) for k in node.keywords]
            keys = [{"name": key, "value": v} for key, (v, _) in key_items]
            key_facts = {key: f for key, (_, f) in key_items}
            fields = dict(target=target, args=args, keywords=keys)
            if "symbol" in t and t.get("class") == "function":
                self.calls.add(t["symbol"])
                return (
                    R(
                        node,
                        "call",
                        callee=t["symbol"],
                        binding="python_lexical",
                        **fields,
                    ),
                    {},
                )
            if "external" in t:
                name = t["external"]
                self.calls.add(self.unit.dependency(name))
                f = self._external_value(name, facts, key_facts)
                op = "library_call"
                effect = ["CTRL", "STATE"]
                if name in {
                    "subprocess.Popen",
                    "subprocess.run",
                    "subprocess.call",
                    "subprocess.check_call",
                    "subprocess.check_output",
                    "os.system",
                }:
                    op = "process_launch"
                    effect = ["CTRL", "STATE", "IO", "ORDER"]
                    command = facts[0] if facts else key_facts.get("args", {})
                    fields["command_argument"] = 0 if facts else "args"
                    shell = (
                        True
                        if name == "os.system"
                        else key_facts.get("shell", {"values": [False]}).get(
                            "values", [None]
                        )[0]
                    )
                    if "values" in command:
                        fields["command_values"] = {
                            "status": "inferred",
                            "values": command["values"],
                            "basis": "literal assignments and module initial values",
                            "maybe_unbound": command.get("maybe_unbound", False),
                        }
                        fields["launch_forms"] = _command_shapes(
                            command["values"], shell
                        )
                    else:
                        fields["command_values"] = self.issue(
                            node,
                            "process_command",
                            "executable command depends on runtime values",
                        )
                    fields["cwd"] = next(
                        (k["value"] for k in keys if k["name"] == "cwd"),
                        self.variable_reference("process.cwd"),
                    )
                elif name.startswith(("os.", "shutil.", "glob.", "io.")) or name in {
                    "builtins.open",
                    "builtins.print",
                    "builtins.input",
                }:
                    effect.append("IO")
                if name == "os.chdir":
                    fields["state_writes"] = ["process.cwd"]
                if name in {
                    "subprocess.Popen.wait",
                    "subprocess.Popen.communicate",
                    "time.sleep",
                }:
                    effect.append("ORDER")
                if "shape" in f or "dtype" in f or "declared_dtype" in f:
                    fields["result_facts"] = {
                        k: v for k, v in f.items() if k != "class"
                    }
                self.issue(
                    node, "provider_binding", "native target binding for " + name
                )
                return (
                    R(
                        node,
                        op,
                        provider=_provider(name),
                        symbol=name,
                        type=_py_type(f.get("class")),
                        effects=effect,
                        **fields,
                    ),
                    f,
                )
            return (
                R(
                    node,
                    "python_invoke",
                    dispatch=self.issue(
                        node,
                        "dispatch",
                        "runtime callable or receiver class is not fixed",
                    ),
                    effects=["CTRL", "STATE"],
                    **fields,
                ),
                {},
            )
        if isinstance(node, ast.BinOp):
            l, a = self.expr(node.left)
            r, b = self.expr(node.right)
            operator = type(node.op).__name__.lower()
            f = _static_values(operator, a, b)
            return (
                R(
                    node,
                    "binary",
                    operator=operator,
                    left=l,
                    right=r,
                    protocol="python_numeric",
                    type=_py_type(f.get("class")),
                ),
                f,
            )
        if isinstance(node, ast.UnaryOp):
            value, f = self.expr(node.operand)
            op = type(node.op).__name__.lower()
            return R(node, "unary", operator=op, value=value, protocol="python"), (
                {"class": "bool"} if op == "not" else {}
            )
        if isinstance(node, ast.Subscript):
            return (
                R(
                    node,
                    "index",
                    base=self.expr(node.value)[0],
                    indices=[self.expr(node.slice)[0]],
                    protocol="python_subscription",
                ),
                {},
            )
        if isinstance(node, ast.Slice):
            return R(
                node,
                "slice",
                **{
                    key: self.expr(getattr(node, key))[0]
                    for key in ("lower", "upper", "step")
                },
            ), {"class": "slice"}
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            items = [self.expr(v) for v in node.elts]
            f = {"class": kind.lower()}
            if len(items) <= 64 and all(
                len(x.get("values", [])) == 1 for _, x in items
            ):
                f["values"] = [[x["values"][0] for _, x in items]]
            return (
                R(node, "aggregate", kind=kind.lower(), items=[op for op, _ in items]),
                f,
            )
        if isinstance(node, ast.Compare):
            return (
                R(
                    node,
                    "compare_chain",
                    left=self.expr(node.left)[0],
                    operators=[type(o).__name__.lower() for o in node.ops],
                    operands=[self.expr(n)[0] for n in node.comparators],
                    short_circuit=True,
                ),
                {},
            )
        if isinstance(node, ast.BoolOp):
            return (
                R(
                    node,
                    "short_circuit",
                    operator=type(node.op).__name__.lower(),
                    operands=[self.expr(n)[0] for n in node.values],
                ),
                {},
            )
        if isinstance(node, ast.JoinedStr):
            return R(
                node, "format_string", items=[self.expr(n)[0] for n in node.values]
            ), {"class": "str"}
        if isinstance(node, ast.FormattedValue):
            return R(
                node,
                "format_field",
                value=self.expr(node.value)[0],
                conversion=node.conversion,
                format=self.expr(node.format_spec)[0] if node.format_spec else None,
            ), {"class": "str"}
        return self.unhandled(node), {}

    @staticmethod
    def variable_reference(symbol):
        return {"op": "var", "scope": "global", "name": symbol}

    def unhandled(self, node):
        children = []
        for field, v in ast.iter_fields(node):
            for child in (v if isinstance(v, list) else [v]):
                if isinstance(child, ast.stmt):
                    children.extend(self.statements([child]))
                elif isinstance(child, ast.expr):
                    children.append(self.expr(child)[0])
                elif isinstance(child, ast.AST) and not isinstance(
                    child,
                    (
                        ast.expr_context,
                        ast.operator,
                        ast.unaryop,
                        ast.cmpop,
                        ast.boolop,
                    ),
                ):
                    children.append(self.unhandled(child))
        return self.record(
            node,
            "unsupported",
            construct=type(node).__name__,
            children=children,
            missing=self.issue(
                node, "protocol", "unbound " + type(node).__name__ + " protocol"
            ),
        )

    def destination(self, node, fact):
        if isinstance(node, ast.Name):
            self.env[node.id] = fact
            return {**self.variable(node.id), "location": self.unit.location(node)}
        if isinstance(node, (ast.List, ast.Tuple)):
            return self.record(
                node, "unpack", items=[self.destination(n, {}) for n in node.elts]
            )
        return self.expr(node)[0]

    def branch(self, body, env):
        previous = self.env
        self.env = dict(env)
        result = self.statements(body)
        outgoing = self.env
        self.env = previous
        return result, outgoing

    def merge(self, *branches):
        self.env = {
            n: _join_facts(*(br.get(n, {"maybe_unbound": True}) for br in branches))
            for n in set().union(*branches)
        }

    def statements(self, body):
        result = []
        for n in body:
            result.extend(self.statement(n))
        return result

    def statement(self, node):
        R = self.record
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            steps = []
            for module, bound, fact in self.unit.module_bindings(node):
                deps = []
                for i in range(1, len(module.split(".")) + 1):
                    path = ".".join(module.split(".")[:i])
                    if path in self.unit.project:
                        deps.append(_function(path, "<init>"))
                self.calls.update(deps)
                if bound == "*" or not module:
                    self.issue(
                        node, "import_binding", "dynamic or relative wildcard binding"
                    )
                binding = self.destination(
                    ast.copy_location(ast.Name(id=bound, ctx=ast.Store()), node), fact
                )
                steps.append(
                    R(
                        node,
                        "python_import",
                        module=module,
                        binding=binding,
                        member=bound,
                        provider=(
                            "project"
                            if module in self.unit.project
                            else _provider(module)
                        ),
                        initializers=deps,
                        cache="sys.modules",
                        effects=["CTRL", "STATE", "IO"],
                    )
                )
            return steps
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            qual = (self.qualified + ".<locals>." if self.qualified else "") + node.name
            symbol = self.unit.aliases.get(qual, _function(self.unit.module, qual))
            if isinstance(node, ast.ClassDef):
                return [self.unhandled(node)]
            params = []
            facts = {}
            a = node.args
            for category, group in (
                ("posonly", a.posonlyargs),
                ("positional", a.args),
                ("varargs", [a.vararg] if a.vararg else []),
                ("keywordonly", a.kwonlyargs),
                ("kwargs", [a.kwarg] if a.kwarg else []),
            ):
                for arg in group:
                    params.append(
                        dict(
                            name=arg.arg,
                            type=_py_type(),
                            binding=category,
                            location=self.unit.location(arg),
                        )
                    )
                    facts[arg.arg] = {}
            table = (
                next(
                    (
                        c
                        for c in self.table.get_children()
                        if c.get_name() == node.name and c.get_lineno() == node.lineno
                    ),
                    None,
                )
                if self.table
                else None
            )
            if table is None:
                self.issue(
                    node, "scope_binding", "compiler scope unavailable for " + node.name
                )
            nested = _Scope(self.unit, symbol, qual, table, parent=self)
            self.unit.functions.append(
                nested.definition(node, node.body, params, facts)
            )
            fact = (
                {"class": "function", "symbol": symbol}
                if not node.decorator_list
                else {}
            )
            dest = self.destination(
                ast.copy_location(ast.Name(id=node.name, ctx=ast.Store()), node), fact
            )
            return [
                R(
                    node,
                    "python_define",
                    symbol=symbol,
                    target=dest,
                    decorators=[self.expr(d)[0] for d in node.decorator_list],
                    defaults=[self.expr(d)[0] for d in a.defaults],
                    keyword_defaults=[
                        self.expr(d)[0] if d is not None else None
                        for d in a.kw_defaults
                    ],
                )
            ]
        if isinstance(node, ast.Expr):
            return [R(node, "expr", value=self.expr(node.value)[0])]
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            if node.value is None:
                return [
                    R(
                        node,
                        "annotation",
                        target=self.destination(node.target, {}),
                        type_expression=self.expr(node.annotation)[0],
                    )
                ]
            value, f = self.expr(node.value)
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if len(targets) == 1:
                return [
                    R(
                        node,
                        "assign",
                        target=self.destination(targets[0], f),
                        value=value,
                    )
                ]
            return [
                R(
                    node,
                    "assign_many",
                    targets=[self.destination(t, f) for t in targets],
                    value=value,
                )
            ]
        if isinstance(node, ast.AugAssign):
            value = self.expr(node.value)[0]
            return [
                R(
                    node,
                    "update",
                    operator=type(node.op).__name__.lower(),
                    target=self.destination(node.target, {}),
                    value=value,
                )
            ]
        if isinstance(node, ast.If):
            condition = self.expr(node.test)[0]
            before = self.env.copy()
            yes, left = self.branch(node.body, before)
            no, right = self.branch(node.orelse, before)
            self.merge(left, right)
            return [R(node, "if", condition=condition, then=yes, **{"else": no})]
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            before = self.env.copy()
            if isinstance(node, ast.While):
                op = "while"
                fields = {"condition": self.expr(node.test)[0]}
            else:
                iterator = self.expr(node.iter)[0]
                target = self.destination(node.target, {})
                op = "for"
                fields = {
                    "init": [],
                    "range": iterator,
                    "target": target,
                    "protocol": (
                        "async_iterator"
                        if isinstance(node, ast.AsyncFor)
                        else "iterator"
                    ),
                }
            steps, after = self.branch(node.body, self.env)
            self.merge(before, after)
            otherwise, after_else = self.branch(node.orelse, self.env)
            self.merge(self.env, after_else)
            return [R(node, op, body=steps, **fields, **{"else": otherwise})]
        if isinstance(node, (ast.With, ast.AsyncWith)):
            resources = []
            for item in node.items:
                value, f = self.expr(item.context_expr)
                resources.append(
                    dict(
                        manager=value,
                        binding=(
                            self.destination(item.optional_vars, f)
                            if item.optional_vars
                            else None
                        ),
                    )
                )
            return [
                R(
                    node,
                    "with_context",
                    resources=resources,
                    body=self.statements(node.body),
                    effects=["CTRL", "STATE", "ORDER"],
                )
            ]
        if isinstance(node, ast.Try):
            before = self.env.copy()
            body, normal = self.branch(node.body, before)
            otherwise, normal = self.branch(node.orelse, normal)
            arms = []
            states = [normal]
            for h in node.handlers:
                scope = before.copy()
                if h.name:
                    scope[h.name] = {"class": "BaseException"}
                handler, outgoing = self.branch(h.body, scope)
                states.append(outgoing)
                arms.append(
                    R(
                        h,
                        "exception_handler",
                        test=self.expr(h.type)[0] if h.type else None,
                        name=h.name,
                        body=handler,
                    )
                )
            self.merge(*states)
            final = self.statements(node.finalbody)
            return [
                R(
                    node,
                    "try",
                    body=body,
                    handlers=arms,
                    **{"else": otherwise, "finally": final},
                )
            ]
        if isinstance(node, ast.Return):
            return [R(node, "return", value=self.expr(node.value)[0])]
        if isinstance(node, ast.Raise):
            return [
                R(
                    node,
                    "raise",
                    value=self.expr(node.exc)[0],
                    cause=self.expr(node.cause)[0],
                )
            ]
        if isinstance(node, ast.Delete):
            return [
                R(
                    node,
                    "unbind",
                    targets=[self.destination(t, {}) for t in node.targets],
                )
            ]
        if isinstance(node, (ast.Break, ast.Continue, ast.Pass)):
            return [R(node, type(node).__name__.lower())]
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            return [
                R(
                    node,
                    "binding_scope",
                    kind=type(node).__name__.lower(),
                    names=node.names,
                )
            ]
        return [self.unhandled(node)]
