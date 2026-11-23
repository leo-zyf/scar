"""F2 module initialization, lexical state, dispatch and external process operations."""

import ast
from pathlib import Path
import shlex

from scar.contract import digest, json_bytes

from .facts import (
    _function,
    _literal_fact,
    _relative,
    assigned_names,
    provider,
    scope_nodes,
)

_PROCESS_METHODS = {"wait", "communicate", "poll"}
_STREAM_DIRECTIONS = {
    "stdin": "parent_to_child",
    "stdout": "child_to_parent",
    "stderr": "child_to_parent",
}


def _annotation_items(node):
    args = node.args
    # CPython visits ordinary positional annotations before positional-only ones.
    parameters = [
        *args.args,
        *args.posonlyargs,
        args.vararg,
        *args.kwonlyargs,
        args.kwarg,
    ]
    for arg in parameters:
        if arg is not None and arg.annotation is not None:
            yield arg.arg, arg.annotation
    if node.returns is not None:
        yield "return", node.returns


def _deferred_syntax(value):
    """Preserve unevaluated syntax without introducing execution ports or effects."""
    if isinstance(value, ast.AST):
        return {
            "syntax": type(value).__name__,
            "fields": [
                [k, _deferred_syntax(v)]
                for k, v in ast.iter_fields(value)
                if k != "ctx"
            ],
        }
    if isinstance(value, list):
        return [_deferred_syntax(v) for v in value]
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, float):
        return {"float": value.hex()}
    if isinstance(value, complex):
        return {"complex": [value.real.hex(), value.imag.hex()]}
    if value is Ellipsis:
        return {"singleton": "Ellipsis"}
    return value


def _type_parameters(node):
    return [
        {
            "name": param.name,
            "kind": type(param).__name__,
            **{
                field: {
                    "evaluation": "deferred",
                    "syntax": _deferred_syntax(getattr(param, field)),
                }
                for field in ("bound", "default_value")
                if getattr(param, field, None) is not None
            },
        }
        for param in getattr(node, "type_params", ())
    ]


def _declared(slot):
    definitions = slot.get("definitions", ())
    result = {"state": slot["state"]}
    if len(definitions) == 1 and not definitions[0]["decorated"]:
        result.update({k: definitions[0][k] for k in ("class", "symbol")})
    elif definitions:
        result["possible_symbols"] = [d["symbol"] for d in definitions]
    if slot.get("literal_sites"):
        result["examples"] = [s["value"] for s in slot["literal_sites"]]
    return result


def _examples(*facts):
    """Bounded, nonexhaustive scalar examples, always subject to a value guard."""
    examples = {}
    for fact in facts:
        for v in fact.get("values", fact.get("examples", ())):
            if type(v) in {str, int, float, bool, type(None)}:
                examples[type(v), v] = v
                if len(examples) == 16:
                    return list(examples.values())
    return list(examples.values())


def _guarded(fact, state):
    """A late lookup may retain an identity candidate, never a captured literal."""
    result = {"state": state}
    expected = {k: fact[k] for k in ("module", "symbol", "external") if k in fact}
    for key in ("process_handles", "bound_process_handles"):
        if key in fact:
            expected[key] = fact[key]
    if expected:
        guard = {"state": state, "expected": expected}
        # The captured object's identity is checked at this lookup. An alias
        # remains valid even if the slot it was copied from has been rebound.
        prior = (
            []
            if ("process_handles" in expected or "bound_process_handles" in expected)
            else fact.get("guards", [])
        )
        result.update(expected, guards=prior if guard in prior else [*prior, guard])
        if "class" in fact:
            result["class"] = fact["class"]
    for key in (
        "possible_symbols",
        "maybe_unbound",
        "process_channels",
        "import_sources",
    ):
        if key in fact:
            result[key] = fact[key]
    if examples := _examples(fact):
        result["examples"] = examples
    return result


def _py_type(name=None):
    return {
        "kind": "python_object",
        "class": name or {"status": "unresolved", "field": "runtime_class"},
    }


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
        if k not in {"values", "examples", "maybe_unbound"}
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
    if any(f.get("guards") for f in facts) and "guards" not in common:
        # An identity proved under different guards is not unconditional at the join.
        for key in (
            "class",
            "module",
            "symbol",
            "external",
            "values",
            "shape",
            "dtype",
        ):
            common.pop(key, None)
    if "values" not in common and (examples := _examples(*facts)):
        common["examples"] = examples
    if all(f.get("process_handles") for f in facts):
        # A reaching object is one instance from one of these sites.
        common["process_handles"] = sorted(
            {s for f in facts for s in f["process_handles"]}
        )
    if "symbol" not in common:
        candidates = set()
        for fact in facts:
            candidates.update(fact.get("possible_symbols", ()))
            if fact.get("class") == "function" and fact.get("symbol"):
                candidates.add(fact["symbol"])
        if candidates:
            common["possible_symbols"] = sorted(candidates)
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
    if operator != "add":
        return {}
    known = "values" in left and "values" in right
    result = []
    for a in left.get("values", left.get("examples", ())):
        for b in right.get("values", right.get("examples", ())):
            if type(a) is not str or type(b) is not str or len(a) + len(b) > 4096:
                return {}
            if a + b not in result:
                result.append(a + b)
            if len(result) > 16:
                return {}
    return (
        ({"class": "str", "values": result} if known else {"examples": result})
        if result
        else {}
    )


def _popen_argument(node, args, keywords, name, index, default):
    if name in keywords:
        number = next(i for i, k in enumerate(node.keywords) if k.arg == name)
        return keywords[name], {"operand": ["keywords", number, "value"]}
    if len(args) > index and not any(
        isinstance(n, ast.Starred) for n in node.args[: index + 1]
    ):
        return args[index], {"operand": ["args", index]}
    if any(isinstance(n, ast.Starred) for n in node.args) or None in keywords:
        return {}, {"status": "unresolved", "field": name}
    return _literal_fact(default), None


def _popen_channels(node, args, keywords):
    """Describe the parent-side endpoints chosen by Popen's arguments."""
    channels = {}
    for index, channel in enumerate(_STREAM_DIRECTIONS, start=3):
        fact, _ = _popen_argument(node, args, keywords, channel, index, None)
        if fact.get("external") == "subprocess.PIPE":
            route = "pipe"
        elif channel == "stderr" and fact.get("external") == "subprocess.STDOUT":
            route = "redirected_to_stdout"
        elif fact.get("values") == [None]:
            route = "inherited"
        elif fact.get("external") == "subprocess.DEVNULL":
            route = "devnull"
        else:
            route = "runtime"
        channels[channel] = {"route": route}
        if fact.get("guards"):
            channels[channel]["conditions"] = fact["guards"]
    return channels


class _ModuleGraph:
    """Own one file and retain structured evidence separately from executable ABI bodies."""

    def __init__(self, parsed, abi, aliases=None):
        self.path = parsed.path.relative_to(parsed.root).as_posix()
        self.module = abi["module"]
        self.main = abi["main"]
        self.tree = parsed.tree
        self.source_hash = parsed.source_hash
        version = tuple(int(v) for v in parsed.compiler_id.split()[1].split(".")[:2])
        future = any(
            isinstance(n, ast.ImportFrom)
            and n.level == 0
            and n.module == "__future__"
            and any(a.name == "annotations" for a in n.names)
            for n in self.tree.body
        )
        self.annotation_protocol = {
            "evaluation": (
                "stringized"
                if future
                else "deferred" if version >= (3, 14) else "eager"
            ),
            "compiler": parsed.compiler_id,
        }
        self.project = abi["project"]
        self.functions = []
        self.externs = {}
        self.diagnostics = []
        self.aliases = aliases or {}
        self._seen = set()
        self.initial = {}
        self.package = self.project[self.module]["package"]
        self.root_symbol = "__main__" if self.main else _function(self.module, "<init>")
        self.table = parsed.scopes
        self.builtins = parsed.builtins
        self.libraries = parsed.libraries
        self.providers = parsed.providers

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
                provider=provider(target, self.libraries),
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
                yield alias.name, bound, _guarded(
                    {"class": "module", "module": module}, "sys.modules." + module
                )
            return
        base = _relative(node, self.module, self.package)
        for alias in node.names:
            name = alias.name
            sub = ".".join(x for x in (base, name) if x)
            state = _function(base, name)
            if base in self.project:
                slot = self.project[base]["exports"].get(name)
                fact = _guarded(_declared(slot or {"state": state}), state)
            else:
                slot = None
                fact = _guarded({"external": sub}, sub)
            if sub in self.project and name != "*":
                if not slot:
                    fact = _guarded({"class": "module", "module": sub}, state)
                fact["import_sources"] = [
                    {"kind": "attribute", "module": base, "name": name, "state": state},
                    {
                        "kind": "submodule",
                        "module": sub,
                        "when": "package_attribute_missing",
                    },
                ]
            yield base, alias.asname or name, fact

    def translate(self):
        # Function bodies may resolve imports from module state. Execution order
        # of a module initializer is maintained separately by its own graph.
        self.initial = {
            name: _declared(slot)
            for name, slot in self.project[self.module]["exports"].items()
        }
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
    def __init__(
        self,
        unit,
        symbol,
        qualified,
        table,
        parent=None,
        globals_only=False,
        class_scope=False,
    ):
        self.unit = unit
        self.symbol = symbol
        self.qualified = qualified
        self.module_scope = globals_only
        self.class_scope = class_scope
        self.table = table
        self.parent = parent
        self.env = {}
        self.calls = set()
        self.locals = table.local if table else frozenset()
        self.global_names = table.declared_global if table else frozenset()
        self.nonlocals = table.free if table else frozenset()
        self.cells = table.cell if table else frozenset()

    def issue(self, node, field, detail):
        return self.unit.issue(node, self.symbol, field, detail)

    def record(self, node, op, **fields):
        return {"op": op, "location": self.unit.location(node), **fields}

    def cell_owner(self, name):
        if name in self.cells and not self.class_scope:
            return self
        if name in self.nonlocals:
            owner = self.parent
            while owner and not owner.module_scope:
                if owner.class_scope:
                    if name == "__class__":
                        return owner
                elif name in owner.locals:
                    return owner
                owner = owner.parent
        return None

    def variable(self, name):
        owner = self.cell_owner(name)
        if owner:
            return {
                "op": "var",
                "name": name,
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
            return {
                "op": "var",
                "name": _function(self.unit.module, name),
                "scope": "global",
            }
        return {"op": "var", "name": name}

    def fact(self, name):
        if name in self.env:
            return self.env[name]
        if name in self.nonlocals:
            binding = self.variable(name)
            owner = self.cell_owner(name)
            if owner and not (owner.class_scope and name == "__class__"):
                return _guarded(owner.fact(name), binding["cell"])
            return {"state": binding.get("cell", binding["name"])}
        if name in self.locals and not (self.module_scope or self.class_scope):
            return {"maybe_unbound": True}
        if not self.module_scope and name in self.unit.aliases:
            return {
                "class": "function",
                "symbol": self.unit.aliases[name],
                "abi_binding": True,
            }
        if not self.module_scope and name in self.unit.initial:
            return _guarded(self.unit.initial[name], _function(self.unit.module, name))
        if name in self.unit.builtins:
            return _guarded(
                {"external": "builtins." + name}, _function(self.unit.module, name)
            )
        return {"maybe_unbound": True}

    def invalidate_shared(self):
        for name, fact in self.env.items():
            binding = self.variable(name)
            if (
                binding.get("scope") == "global"
                or "cell" in binding
                or self.class_scope
            ):
                self.env[name] = _guarded(fact, binding.get("cell") or binding["name"])
            elif fact.get("class") in {"dict", "set", "list", "numpy.ndarray"}:
                self.env[name] = {
                    k: v for k, v in fact.items() if k in {"class", "guards"}
                }

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
                    **(
                        {"cell": self.variable(n)["cell"]}
                        if n in self.cells and not self.class_scope
                        else {}
                    ),
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
        kind = (
            "module"
            if self.module_scope
            else "class" if self.class_scope else "function"
        )
        signature = dict(
            name=self.symbol,
            params=params,
            result="void" if self.module_scope else _py_type(),
            location=self.unit.location(node),
            calling_convention="python",
            python_scope=dict(
                kind=kind,
                module=self.unit.module,
                initializer=self.unit.root_symbol,
                main=self.unit.main,
            ),
            calls=sorted(self.calls),
            partial_body=[*prelude, *steps],
        )
        if kind == "function":
            yields = any(
                isinstance(n, (ast.Yield, ast.YieldFrom)) for n in scope_nodes(body)
            )
            if isinstance(node, ast.AsyncFunctionDef):
                protocol = "async_generator" if yields else "coroutine"
            else:
                protocol = "generator" if yields else "regular"
            signature["call_protocol"] = {"kind": protocol}
        return signature

    def _external_value(self, target, args, keywords):
        if target == "builtins.str" and len(args) == 1:
            if "values" in args[0]:
                return {"class": "str", "values": [str(v) for v in args[0]["values"]]}
            return {"examples": [str(v) for v in _examples(args[0])]}
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
        if target == "subprocess.Popen.communicate":
            return {"class": "tuple"}
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
                    "possible_symbols",
                    "process_handles",
                    "bound_process_handles",
                    "state",
                    "guards",
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
            handles = b.get("process_handles")
            if handles and name in _STREAM_DIRECTIONS:
                guards = [
                    {
                        "operand": "base",
                        "expected": {
                            "process_handles": handles,
                            "original_stream": name,
                        },
                    }
                ]
                return (
                    R(
                        node,
                        "process_stream",
                        base=base,
                        field=name,
                        channel=name,
                        direction=_STREAM_DIRECTIONS[name],
                        handle={"operand": ["base"], "sites": handles},
                        channel_binding=b.get("process_channels", {}).get(
                            name, {"route": "runtime"}
                        ),
                        binding_guard={
                            "status": "conditional",
                            "conditions": guards,
                            "require_bound": bool(b.get("maybe_unbound")),
                            "on_failure": "runtime_lookup",
                        },
                        type=_py_type(),
                        effects=["STATE"],
                    ),
                    {},
                )
            if handles and name in _PROCESS_METHODS:
                f = {
                    "external": "subprocess.Popen." + name,
                    "bound_process_handles": handles,
                }
            elif "module" in b:
                module = b["module"]
                sub = module + "." + name
                if sub in self.unit.project:
                    f = {"class": "module", "module": sub}
                elif module in self.unit.project:
                    member = _function(module, name)
                    slot = self.unit.project[module]["exports"].get(
                        name, {"state": member}
                    )
                    f = _guarded(_declared(slot), member)
                else:
                    f = {"external": sub}
            elif "external" in b:
                f = {"external": b["external"] + "." + name}
            elif b.get("class") in {"io.IOBase", "numpy.ndarray"}:
                f = {"external": b["class"] + "." + name}
            if b.get("guards") and not (handles and name in _PROCESS_METHODS):
                f["guards"] = [*b["guards"], *f.get("guards", [])]
            if handles and name in _PROCESS_METHODS:
                f.setdefault("guards", []).append(
                    {
                        "operand": "target",
                        "expected": {
                            "bound_receiver_from": handles,
                            "method": "subprocess.Popen." + name,
                        },
                    }
                )
            if b.get("maybe_unbound"):
                f["maybe_unbound"] = True
            if "external" in f and not (handles and name in _PROCESS_METHODS):
                f = _guarded(f, f["external"])
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
            if t.get("guards") or t.get("maybe_unbound"):
                fields["binding_guard"] = dict(
                    status="conditional",
                    conditions=t.get("guards", []),
                    require_bound=bool(t.get("maybe_unbound")),
                    on_failure="runtime_lookup",
                )
            self.invalidate_shared()
            if "symbol" in t and t.get("class") == "function":
                self.calls.add(t["symbol"])
                return (
                    R(
                        node,
                        "call",
                        callee=t["symbol"],
                        binding=(
                            "python_guarded"
                            if "binding_guard" in fields
                            else (
                                "python_abi"
                                if t.get("abi_binding")
                                else "python_lexical"
                            )
                        ),
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
                    if name == "subprocess.Popen":
                        site = digest(
                            json_bytes(
                                [
                                    self.unit.source_hash,
                                    self.symbol,
                                    self.unit.location(node),
                                ]
                            )
                        )[:32]
                        channels = _popen_channels(node, facts, key_facts)
                        fields.update(
                            result_handle={
                                "site": site,
                                "instance": "per_successful_invocation",
                            },
                            channels=channels,
                            waits_for_exit=False,
                        )
                        f.update(process_handles=[site], process_channels=channels)
                    command = facts[0] if facts else key_facts.get("args", {})
                    fields["command_argument"] = 0 if facts else "args"
                    shell_fact, shell_arg = _popen_argument(
                        node, facts, key_facts, "shell", 8, False
                    )
                    options = shell_fact.get("values", [])
                    shell = (
                        True
                        if name == "os.system"
                        else options[0] if len(options) == 1 else None
                    )
                    fields["shell"] = (
                        {"value": True, "source": "provider"}
                        if name == "os.system"
                        else shell_arg or {"value": False, "source": "default"}
                    )
                    if "values" in command:
                        fields["command_values"] = {
                            "status": "inferred",
                            "values": command["values"],
                            "basis": "reaching assignments on captured paths",
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
                        if examples := _examples(command):
                            fields["command_values"]["guarded_examples"] = [
                                {
                                    "guard": {
                                        "argument": fields["command_argument"],
                                        "equals": v,
                                        "type": type(v).__name__,
                                    },
                                    "launch_forms": _command_shapes([v], shell),
                                }
                                for v in examples
                            ]
                    _, cwd_arg = _popen_argument(node, facts, key_facts, "cwd", 9, None)
                    fields["cwd"] = cwd_arg or self.variable_reference("process.cwd")
                elif t.get("bound_process_handles"):
                    method = name.rsplit(".", 1)[-1]
                    op = "process_" + method
                    # Capture the evaluated bound method; an aliased method retains
                    # its receiver even after the original variable is rebound.
                    handle = {
                        "operand": ["target"],
                        "projection": "bound_receiver",
                        "sites": t["bound_process_handles"],
                    }
                    if target.get("op") == "member":
                        handle.update(operand=["target", "base"], projection="identity")
                    fields.update(handle=handle, waits_for_exit=method != "poll")
                    if method == "communicate":
                        effect += ["IO", "ORDER"]
                    elif method == "wait":
                        effect.append("ORDER")
                elif name.startswith(("os.", "shutil.", "glob.", "io.")) or name in {
                    "builtins.open",
                    "builtins.print",
                    "builtins.input",
                }:
                    effect.append("IO")
                if name == "os.chdir":
                    fields["state_writes"] = ["process.cwd"]
                if (
                    name
                    in {
                        "subprocess.Popen.wait",
                        "subprocess.Popen.communicate",
                        "time.sleep",
                    }
                    and "ORDER" not in effect
                ):
                    effect.append("ORDER")
                if "shape" in f or "dtype" in f or "declared_dtype" in f:
                    fields["result_facts"] = {
                        k: v for k, v in f.items() if k != "class"
                    }
                self.issue(
                    node, "provider_binding", "native target binding for " + name
                )
                if "binding_guard" in fields:
                    if "values" in f:
                        f["examples"] = _examples(f)
                        del f["values"]
                    f["guards"] = t.get("guards", [])
                return (
                    R(
                        node,
                        op,
                        provider=provider(name, self.unit.libraries),
                        symbol=name,
                        type=_py_type(f.get("class")),
                        effects=effect,
                        **fields,
                    ),
                    f,
                )
            if t.get("possible_symbols"):
                fields["candidate_callees"] = [
                    {
                        "symbol": symbol,
                        "guard": {"operand": "target", "expected": {"symbol": symbol}},
                    }
                    for symbol in t["possible_symbols"]
                ]
                self.calls.update(t["possible_symbols"])
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
        target = self.expr(node)[0]
        if isinstance(node, (ast.Subscript, ast.Attribute)):
            # Attribute/subscript writes may modify any alias of the receiver.
            self.invalidate_shared()
        return target

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

    def widen(self, statements):
        """Invalidate bindings possibly written before this control-flow point."""
        self.invalidate_shared()
        bound = {
            name for name, fact in self.env.items() if not fact.get("maybe_unbound")
        }
        examples = {name: _examples(fact) for name, fact in self.env.items()}
        for part in scope_nodes(statements):
            for name in assigned_names(part):
                if isinstance(part, ast.Name) and isinstance(part.ctx, ast.Del):
                    bound.discard(name)
                self.env[name] = {} if name in bound else {"maybe_unbound": True}
                if examples.get(name):
                    self.env[name]["examples"] = examples[name]

    def statements(self, body):
        result = []
        for n in body:
            result.extend(self.statement(n))
        return result

    def annotated_assignment(self, node):
        target = node.target
        steps = []
        if node.value is not None:
            value, fact = self.expr(node.value)
            steps.append(
                self.record(
                    node, "assign", target=self.destination(target, fact), value=value
                )
            )
        elif isinstance(target, (ast.Attribute, ast.Subscript)):
            # Without an RHS the base and index expressions still run, with
            # no final attribute or item lookup/store.
            operands = [target.value] + (
                [target.slice] if isinstance(target, ast.Subscript) else []
            )
            steps.extend(
                self.record(t, "expr", value=self.expr(t)[0]) for t in operands
            )
        namespace = self.module_scope or self.class_scope
        stored = bool(namespace and node.simple)
        mode = self.unit.annotation_protocol["evaluation"]
        if not namespace or not stored and mode != "eager":
            mode = "not_evaluated"
        fields = {
            "target_syntax": _deferred_syntax(target),
            "stored": stored,
            "annotation_protocol": {
                **self.unit.annotation_protocol,
                "evaluation": mode,
            },
        }
        if mode == "eager":
            fields["value"] = self.expr(node.annotation)[0]
        else:
            fields["syntax"] = _deferred_syntax(node.annotation)
        if stored:
            fields["effects"] = ["STATE"]
        steps.append(self.record(node, "annotation", **fields))
        return steps

    def statement(self, node):
        R = self.record
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            steps = []
            from_import = isinstance(node, ast.ImportFrom)
            fromlist = [a.name for a in node.names] if from_import else []
            for i, (module, bound, fact) in enumerate(self.unit.module_bindings(node)):
                alias = node.names[i]
                self.invalidate_shared()
                deps = []
                conditional = []
                if not from_import or i == 0:
                    parts = module.split(".")
                    for depth in range(1, len(parts) + 1):
                        path = ".".join(parts[:depth])
                        if path in self.unit.project:
                            deps.append(_function(path, "<init>"))
                    for member in fromlist:
                        child = module + "." + member
                        if child in self.unit.project:
                            conditional.append(
                                {
                                    "module": child,
                                    "initializer": _function(child, "<init>"),
                                    "when": {
                                        "kind": "package_attribute_missing",
                                        "module": module,
                                        "name": member,
                                    },
                                }
                            )
                self.calls.update(deps)
                self.calls.update(row["initializer"] for row in conditional)
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
                        member=alias.name,
                        alias=alias.asname,
                        bound_value=fact,
                        request={
                            "form": "from" if from_import else "import",
                            "module": (
                                (node.module or "") if from_import else alias.name
                            ),
                            "level": node.level if from_import else 0,
                            "fromlist": fromlist,
                            "binding_index": i if from_import else 0,
                        },
                        provider=(
                            "project"
                            if module in self.unit.project
                            else provider(module, self.unit.libraries)
                        ),
                        **(
                            {
                                "source_provider": self.unit.providers[
                                    module.split(".")[0]
                                ]
                            }
                            if module.split(".")[0] in self.unit.providers
                            else {}
                        ),
                        initializers=deps,
                        conditional_initializers=conditional,
                        cache="sys.modules",
                        effects=["CTRL", "STATE", "IO"],
                    )
                )
            return steps
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            separator = "." if self.class_scope else ".<locals>."
            qual = (self.qualified + separator if self.qualified else "") + node.name
            symbol = self.unit.aliases.get(qual, _function(self.unit.module, qual))
            if self.table and sum(n == node.name for n, _ in self.table.children) > 1:
                symbol += f"@{node.lineno}"
            table = (
                self.table.children.get((node.name, node.lineno))
                if self.table
                else None
            )
            if table is None:
                self.issue(
                    node, "scope_binding", "compiler scope unavailable for " + node.name
                )
            decorators = [self.expr(d)[0] for d in node.decorator_list]
            type_params = _type_parameters(node)
            creation = {"type_params": type_params} if type_params else {}
            if isinstance(node, ast.ClassDef):
                bases = [self.expr(b)[0] for b in node.bases]
                keywords = [
                    {"name": k.arg, "value": self.expr(k.value)[0]}
                    for k in node.keywords
                ]
                body_symbol = symbol + ".<class_body>"
                nested = _Scope(
                    self.unit, body_symbol, qual, table, parent=self, class_scope=True
                )
                body = nested.definition(node, node.body, [], {})
                body.update(scope_kind="class", owner_class=symbol)
                self.unit.functions.append(body)
                self.calls.add(body_symbol)
                self.invalidate_shared()
                name = ast.copy_location(ast.Name(id=node.name, ctx=ast.Store()), node)
                dest = self.destination(name, {})
                return [
                    R(
                        node,
                        "python_define_class",
                        symbol=symbol,
                        target=dest,
                        body_symbol=body_symbol,
                        bases=bases,
                        keywords=keywords,
                        decorators=decorators,
                        effects=["CTRL", "STATE"],
                        class_protocol=self.issue(
                            node,
                            "class_protocol",
                            "runtime class namespace, metaclass and MRO",
                        ),
                        **creation,
                    )
                ]
            params = []
            facts = {}
            a = node.args
            defaults = [self.expr(d)[0] for d in a.defaults]
            keyword_defaults = [
                self.expr(d)[0] if d is not None else None for d in a.kw_defaults
            ]
            annotations = []
            for name, expression in _annotation_items(node):
                if self.unit.annotation_protocol["evaluation"] != "eager":
                    field = {"syntax": _deferred_syntax(expression)}
                elif type_params:
                    # Generic function annotations see the compiler-created
                    # type-parameter scope. Outer names alone cannot resolve them.
                    field = {
                        "value": R(
                            expression,
                            "annotation_evaluation",
                            scope="type_parameters",
                            expression=_deferred_syntax(expression),
                            effects=["CTRL", "STATE", "OPAQUE"],
                        )
                    }
                    self.invalidate_shared()
                else:
                    field = {"value": self.expr(expression)[0]}
                annotations.append({"name": name, **field})
            if annotations:
                creation.update(
                    annotations=annotations,
                    annotation_protocol=self.unit.annotation_protocol,
                )
            positional = [*a.posonlyargs, *a.args]
            default_from = {
                arg.arg: {"field": "defaults", "index": i}
                for i, arg in enumerate(positional[len(positional) - len(a.defaults) :])
            }
            default_from.update(
                {
                    arg.arg: {"field": "keyword_defaults", "index": i}
                    for i, (arg, value) in enumerate(zip(a.kwonlyargs, a.kw_defaults))
                    if value is not None
                }
            )
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
                            required=category not in {"varargs", "kwargs"}
                            and arg.arg not in default_from,
                            **(
                                {"default_from": default_from[arg.arg]}
                                if arg.arg in default_from
                                else {}
                            ),
                            location=self.unit.location(arg),
                        )
                    )
                    facts[arg.arg] = {}
            nested = _Scope(self.unit, symbol, qual, table, parent=self)
            definition = nested.definition(node, node.body, params, facts)
            if self.class_scope:
                definition["owner_class"] = self.symbol.removesuffix(".<class_body>")
            self.unit.functions.append(definition)
            fact = (
                {"class": "function", "symbol": symbol}
                if not node.decorator_list
                else {}
            )
            if decorators:
                self.invalidate_shared()
            dest = self.destination(
                ast.copy_location(ast.Name(id=node.name, ctx=ast.Store()), node), fact
            )
            return [
                R(
                    node,
                    "python_define",
                    symbol=symbol,
                    target=dest,
                    decorators=decorators,
                    defaults=defaults,
                    keyword_defaults=keyword_defaults,
                    **creation,
                )
            ]
        if isinstance(node, ast.Expr):
            return [R(node, "expr", value=self.expr(node.value)[0])]
        if isinstance(node, ast.AnnAssign):
            return self.annotated_assignment(node)
        if isinstance(node, ast.Assign):
            value, f = self.expr(node.value)
            targets = node.targets
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
            iterator = None if isinstance(node, ast.While) else self.expr(node.iter)[0]
            before = self.env.copy()
            self.widen(
                [node.test if isinstance(node, ast.While) else node.target, *node.body]
            )
            header = self.env.copy()
            if isinstance(node, ast.While):
                op = "while"
                fields = {"condition": self.expr(node.test)[0]}
            else:
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
            self.merge(before, header, after)
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
            # Any prefix of the protected body may precede a handler.
            self.widen(node.body)
            exceptional = self.env.copy()
            for h in node.handlers:
                self.env = exceptional.copy()
                test = self.expr(h.type)[0] if h.type else None
                if h.name:
                    self.env[h.name] = {}
                handler, outgoing = self.branch(h.body, self.env)
                if h.name:
                    outgoing[h.name] = {"maybe_unbound": True}
                states.append(outgoing)
                arms.append(
                    R(
                        h,
                        "exception_handler",
                        test=test,
                        name=h.name,
                        body=handler,
                    )
                )
            self.merge(*states, *([exceptional] if node.finalbody else []))
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
                    targets=[
                        self.destination(t, {"maybe_unbound": True})
                        for t in node.targets
                    ],
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
