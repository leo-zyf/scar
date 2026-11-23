"""B2a control lowering: protected regions, scoped exits, and process completion."""

from dataclasses import dataclass, field

from .model import (
    Decl,
    Expr,
    Stmt,
    Type,
    binary,
    call,
    cast,
    declaration,
    expression_stmt,
    integer,
    member,
    name,
    named,
    symbol_name,
    unary,
    utf8_literal,
    variable,
)

TEXT_STREAM_TYPE = {"kind": "handle", "resource": "text_stream"}
ERROR_TYPE = {"kind": "handle", "resource": "error"}
_COMPLETIONS = {"return": 1, "break": 2, "continue": 3}


@dataclass
class _Cleanup:
    number: int
    break_depth: int
    continue_depth: int
    causes: set[str] = field(default_factory=set)
    finalizing: bool = False

    @property
    def saved(self):
        return f"scar_return_{self.number}"


class ControlFlow:
    """Own the control state of lowering; operand and statement lowering are callbacks."""

    statement_kinds = frozenset(
        {
            "try",
            "resource_scope",
            "throw",
            "rethrow",
            "label",
            "goto",
            "return",
            "break",
            "continue",
        }
    )

    def __init__(self, *, expression, statements, type_name, headers, use_sdk):
        self._expression = expression
        self._statements = statements
        self._type_name = type_name
        self._headers = headers
        self._use_sdk = use_sdk
        self._cleanups = []
        self._jump_domains = {}
        self._break_depth = self._continue_depth = self._control_id = 0
        self._function_result = None
        self._deferred_return = False
        self._handles_exit = False

    def begin_function(self, result):
        self._function_result = result
        self._deferred_return = False
        self._jump_domains.clear()

    def finish_function(self, symbol, body):
        if (
            self._deferred_return
            and symbol != "main"
            and self._function_result != "void"
            and (not body or body[-1].kind not in ("throw", "return"))
        ):
            # A sealed value-returning function must yield a value on every
            # successful exit, even when its return crossed a cleanup callback.
            self._headers.add("stdexcept")
            body += (
                Stmt(
                    "throw",
                    value=Expr(
                        "construct",
                        type=named("std::logic_error"),
                        args=(
                            Expr(
                                "string",
                                value="function completed without its declared result",
                            ),
                        ),
                    ),
                ),
            )
        return body

    def require_exit_handler(self):
        self._handles_exit = True

    def entry_body(self, body):
        if not self._handles_exit:
            return body
        self._use_sdk("exceptions.hpp")
        parameter = Decl(
            "param",
            name="scar_signal",
            type=Type("ref", element=Type("const", element=named("scar::exit_signal"))),
        )
        handler = (
            parameter,
            (Stmt("return", value=call("scar::report_exit", name("scar_signal"))),),
        )
        return (Stmt("try", body=body, handlers=(handler,)),)

    def error_make(self, node):
        category, message = node.get("category"), node["message"]
        if (
            not isinstance(category, str)
            or not category
            or node.get("type") != ERROR_TYPE
            or message.get("type") != "text"
        ):
            raise ValueError("invalid error construction")
        self._use_sdk("exceptions.hpp")
        msg = cast(named("std::string"), self._expression(message))
        if category != "system_exit":
            if "exit_code" in node or "exit_message" in node:
                raise ValueError("unexpected exit payload on error")
            return call("scar::make_error", utf8_literal(category, "std::string"), msg)
        self.require_exit_handler()
        code, display = node.get("exit_code", {}), node.get("exit_message", {})
        if code.get("type") != "i64" or display.get("type") not in ("unit", "text"):
            raise ValueError("system_exit requires status and optional text")
        self._headers.update(("tuple", "optional"))
        string = named("std::string")
        optional = Type("template", name="std::optional", args=(string,))
        displayed = self._expression(display)
        if display["type"] == "unit":
            displayed = binary(
                ",", cast(named("void"), displayed), name("std::nullopt")
            )
        types = (string, named("std::int64_t"), optional)
        payload = Expr(
            "construct",
            type=Type("template", name="std::tuple", args=types),
            qualifier="brace",
            args=(
                msg,
                cast(types[1], self._expression(code)),
                Expr("construct", type=optional, args=(displayed,), qualifier="brace"),
            ),
        )
        names = ("scar_message", "scar_code", "scar_display")
        make = Expr(
            "lambda",
            params=tuple(Decl("param", type=t, name=n) for t, n in zip(types, names)),
            body=(
                Stmt(
                    "return",
                    value=call(
                        "scar::make_exit_error",
                        name(names[1]),
                        call("std::move", name(names[0])),
                        call("std::move", name(names[2])),
                    ),
                ),
            ),
        )
        return call("std::apply", make, payload)

    def loop_body(self, items, bindings=(), switch=False):
        self._break_depth += 1
        self._continue_depth += not switch
        try:
            return self._statements(items, bindings=bindings)
        finally:
            self._continue_depth -= not switch
            self._break_depth -= 1

    def _fresh_id(self):
        self._control_id += 1
        return self._control_id

    def _completion(self, kind, value=None):
        region = self._cleanups[-1] if self._cleanups else None
        if (
            region is None
            or kind == "break"
            and self._break_depth > region.break_depth
            or kind == "continue"
            and self._continue_depth > region.continue_depth
        ):
            return [Stmt(kind, value=value)]
        if kind in ("break", "continue") and not (
            self._break_depth if kind == "break" else self._continue_depth
        ):
            raise ValueError(f"{kind} has no enclosing control target")
        region.causes.add(kind)
        code = []
        if kind == "return":
            if self._function_result == "void":
                if value is not None:
                    code.append(expression_stmt(value))
            elif value is None:
                raise ValueError("non-void function return lacks a value")
            else:
                code.append(
                    expression_stmt(call(member(name(region.saved), "emplace"), value))
                )
        return [*code, Stmt("return", value=integer(_COMPLETIONS[kind]))]

    def _protected(self, body, cleanup):
        """Execute cleanup once, keeping the original exception active in a catch."""
        number = self._fresh_id()
        region = _Cleanup(number, self._break_depth, self._continue_depth)
        self._cleanups.append(region)
        try:
            protected = tuple(body())
            region.finalizing = True
            final = tuple(cleanup())
        finally:
            self._cleanups.pop()
        final_name = f"scar_final_{number}"
        finish = call(final_name)
        if not region.causes:
            fn = Expr("lambda", captures=("&",), body=final)
            handler = (expression_stmt(finish), Stmt("throw"))
            return (
                Stmt(
                    "block",
                    body=(
                        declaration(variable(final_name, named("auto"), fn)),
                        Stmt("try", body=protected, handlers=((None, handler),)),
                        expression_stmt(finish),
                    ),
                ),
            )

        normal = Stmt("return", value=integer(0))

        def action(stmts):
            return Expr(
                "lambda", captures=("&",), type=named("int"), body=(*stmts, normal)
            )

        before = []
        if "return" in region.causes and self._function_result != "void":
            self._deferred_return = True
            self._headers.add("optional")
            ty = self._function_result
            ref = isinstance(ty, dict) and ty.get("kind") == "ref"
            if ref:
                self._headers.add("functional")
                stored = Type(
                    "template",
                    name="std::reference_wrapper",
                    args=(self._type_name(ty["to"]),),
                )
            else:
                stored = self._type_name(ty)
            before.append(
                declaration(
                    variable(
                        region.saved,
                        Type("template", name="std::optional", args=(stored,)),
                    )
                )
            )
            slot = unary("*", name(region.saved))
            returned = call(member(slot, "get")) if ref else call("std::move", slot)
        else:
            returned = None
        before.append(declaration(variable(final_name, named("auto"), action(final))))
        reason = f"scar_completion_{number}"
        override = f"scar_override_{number}"
        after_final = declaration(variable(override, named("int"), finish))
        guard = Stmt(
            "try",
            body=(expression_stmt(binary("=", name(reason), call(action(protected)))),),
            handlers=(
                (
                    None,
                    (
                        after_final,
                        Stmt(
                            "if",
                            condition=name(override),
                            body=(Stmt("return", value=name(override)),),
                        ),
                        Stmt("throw"),
                    ),
                ),
            ),
        )
        resolve = Expr(
            "lambda",
            captures=("&",),
            type=named("int"),
            body=(
                declaration(variable(reason, named("int"))),
                guard,
                after_final,
                Stmt(
                    "return",
                    value=Expr(
                        "conditional",
                        args=(name(override), name(override), name(reason)),
                    ),
                ),
            ),
        )
        before.append(declaration(variable(reason, named("int"), call(resolve))))
        for kind, code in _COMPLETIONS.items():
            if kind in region.causes:
                before.append(
                    Stmt(
                        "if",
                        condition=binary("==", name(reason), integer(code)),
                        body=tuple(
                            self._completion(
                                kind, returned if kind == "return" else None
                            )
                        ),
                    )
                )
        return (Stmt("block", body=tuple(before)),)

    def _try_region(self, stmt):
        self._use_sdk("exceptions.hpp")
        handlers = stmt.get("handlers", ())

        def body():
            work = self._statements(stmt["body"])
            alternate = self._statements(stmt.get("else", ()))
            if not handlers:
                return (*work, *alternate)
            n = self._fresh_id()
            caught = f"scar_caught_{n}"
            following = (Stmt("throw"),)
            for handler in reversed(handlers):
                categories = handler["categories"]
                if not isinstance(categories, list) or not all(
                    isinstance(c, str) and c for c in categories
                ):
                    raise ValueError("invalid error category list")
                binding = handler.get("binding")
                if binding is None:
                    code = self._statements(handler["body"])
                elif isinstance(binding, str):
                    code = (
                        declaration(
                            variable(
                                symbol_name(binding, "v"),
                                self._type_name(ERROR_TYPE),
                                name(caught),
                            )
                        ),
                        *self._statements(handler["body"], bindings=(binding,)),
                    )
                elif (
                    isinstance(binding, dict)
                    and binding.get("op") == "var"
                    and binding.get("type") == ERROR_TYPE
                ):
                    dest = self._expression(binding)
                    code = (
                        expression_stmt(binary("=", dest, name(caught))),
                        *self._protected(
                            lambda: self._statements(handler["body"]),
                            lambda: (
                                expression_stmt(binary("=", dest, Expr("nullptr"))),
                            ),
                        ),
                    )
                else:
                    raise ValueError("handler binding requires a typed error variable")
                checks = [
                    call(
                        "scar::matches",
                        name(caught),
                        utf8_literal(c, "std::string_view"),
                    )
                    for c in categories
                ]
                condition = checks[0] if checks else Expr("literal", value=False)
                for value in checks[1:]:
                    condition = binary("||", condition, value)
                following = (
                    Stmt("if", condition=condition, body=code, else_body=following),
                )
            catches = (
                (
                    None,
                    (
                        declaration(
                            variable(caught, named("auto"), call("scar::capture_error"))
                        ),
                        *following,
                    ),
                ),
            )
            if not alternate:
                return (Stmt("try", body=work, handlers=catches),)
            success = f"scar_success_{n}"
            return (
                declaration(
                    variable(success, named("bool"), Expr("literal", value=False))
                ),
                Stmt(
                    "try",
                    body=(
                        *work,
                        expression_stmt(
                            binary("=", name(success), Expr("literal", value=True))
                        ),
                    ),
                    handlers=catches,
                ),
                Stmt("if", condition=name(success), body=alternate),
            )

        if stmt.get("finally"):
            return self._protected(body, lambda: self._statements(stmt["finally"]))
        return (Stmt("block", body=tuple(body())),)

    def _resource_scope(self, stmt):
        resource = stmt["resource"]
        if resource.get("type") != TEXT_STREAM_TYPE:
            raise ValueError("unsupported resource scope")
        self._use_sdk("host.hpp")
        symbol = f"scar_resource_{self._fresh_id()}"
        enter = declaration(
            variable(
                symbol,
                self._type_name(TEXT_STREAM_TYPE),
                call("scar::host::stream_enter", self._expression(resource)),
            )
        )
        cleanup = self._protected(
            lambda: self._statements(stmt["body"]),
            lambda: (expression_stmt(call("scar::host::stream_close", name(symbol))),),
        )
        return [Stmt("block", body=(enter, *cleanup))]

    def _check_jump_domain(self, symbol):
        current = tuple((region.number, region.finalizing) for region in self._cleanups)
        if self._jump_domains.setdefault(symbol, current) != current:
            raise ValueError("an unstructured jump crosses a cleanup boundary")

    def statement(self, stmt):
        op = stmt["op"]
        if op == "try":
            return self._try_region(stmt)
        if op == "resource_scope":
            return self._resource_scope(stmt)
        if op == "throw":
            if stmt["value"].get("type") != ERROR_TYPE:
                raise ValueError("throw requires an error handle")
            self._use_sdk("exceptions.hpp")
            return [
                expression_stmt(
                    call("scar::raise_error", self._expression(stmt["value"]))
                )
            ]
        if op == "rethrow":
            self._use_sdk("exceptions.hpp")
            return [expression_stmt(call("scar::safe_rethrow"))]
        if op in ("label", "goto"):
            self._check_jump_domain(stmt["name"])
            return [Stmt(op, name=symbol_name(stmt["name"], "label"))]
        if op in _COMPLETIONS:
            value = (
                self._expression(stmt["value"])
                if op == "return" and stmt.get("value") is not None
                else None
            )
            return self._completion(op, value)
        raise ValueError(f"unsupported control statement {op!r}")
