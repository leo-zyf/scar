"""Close lexical Python protocols from the bound project operation graph."""

import math

from scar.contract import json_bytes, python_provider_id, walk_operations
from . import (
    arrays,
    effects,
    expressions,
    exceptions,
    graphics,
    integers,
    host_protocols,
    mappings,
    namespace_operations,
    objects,
    obligations,
    processes,
    products,
    scalars,
    sequences,
    text,
    values,
)
from .values import Value as _Value, literal as _literal
from .import_state import ImportState, namespace_slot


def _reachable(roots, edges):
    successors = {}
    for edge in edges:
        successors.setdefault(edge["owner"], set()).add(edge["target"])
    result, pending = set(), list(roots)
    while pending:
        current = pending.pop()
        if current not in result:
            result.add(current)
            pending.extend(successors.get(current, ()))
    return result


def _join(paths, sequence_domain=None):
    """Only bindings present on every control predecessor are definitely bound."""
    result = {}
    for name, first in paths[0].items():
        incoming = [path.get(name) for path in paths[1:]]
        if all(values.identical(v, first) for v in incoming):
            result[name] = first
        elif values.is_sequence(first.kind) and all(
            v is not None
            and values.is_sequence(v.kind)
            and isinstance(v.value, dict)
            and v.value.get("op") == first.value.get("op") == "var"
            and (v.value.get("name"), v.value.get("scope"))
            == (first.value.get("name"), first.value.get("scope"))
            for v in incoming
        ):
            if sequence_domain is not None and all(
                sequence_domain.compatible(first.kind, v.kind) for v in incoming
            ):
                kind = sequence_domain.resolve_kind(first.kind)
                facts = (
                    first.facts
                    if all(values.identical(v.facts, first.facts) for v in incoming)
                    else {}
                )
                result[name] = _Value(
                    kind, {**first.value, "type": values.wire(kind)}, facts
                )
        elif values.is_array(first.kind) and all(
            v is not None
            and values.is_array(v.kind)
            and values.array_element(v.kind) == values.array_element(first.kind)
            and isinstance(v.value, dict)
            and isinstance(first.value, dict)
            and v.value.get("op") == first.value.get("op") == "var"
            and (v.value.get("name"), v.value.get("scope"))
            == (first.value.get("name"), first.value.get("scope"))
            for v in incoming
        ):
            kind = values.join_arrays([first.kind, *(v.kind for v in incoming)])
            facts = {}
            if "shape" in first.facts and all(
                values.identical(v.facts.get("shape"), first.facts["shape"])
                for v in incoming
            ):
                facts["shape"] = first.facts["shape"]
            result[name] = _Value(
                kind, {**first.value, "type": values.wire(kind)}, facts
            )
        elif all(
            v is not None
            and v.kind == first.kind
            and values.identical(v.value, first.value)
            for v in incoming
        ):
            facts = {}
            if first.kind in {"i64", "integer"}:
                facts.update(values.join_integer_ranges([first, *incoming]))
            if "stream_origin" in first.facts and all(
                values.identical(
                    v.facts.get("stream_origin"), first.facts["stream_origin"]
                )
                for v in incoming
            ):
                facts["stream_origin"] = first.facts["stream_origin"]
            result[name] = _Value(first.kind, first.value, facts)
    return result


def _global_bindings(functions):
    """Identify single bindings and the complete writer set for module slots."""
    writes, repeated = {}, set()
    for function in functions.values():
        pending = [(function["body"], function["python_scope"]["kind"] != "module")]
        while pending:
            node, multiple = pending.pop()
            if isinstance(node, list):
                pending.extend((item, multiple) for item in node)
                continue
            if not isinstance(node, dict):
                continue
            op = node.get("op")
            multiple = multiple or op in {
                "for",
                "while",
                "exception_handler",
                "mapping_comprehension",
                "sequence_comprehension",
            }
            for target in values.written_targets(node):
                if target.get("op") == "var" and target.get("scope") == "global":
                    name = target["name"]
                    writes.setdefault(name, []).append((function["name"], id(target)))
                    if multiple:
                        repeated.add(name)
            pending.extend((child, multiple) for child in node.values())
    single = {name for name, sites in writes.items() if len(sites) == 1} - repeated
    multiple = {
        name: (sites[0][0], dict.fromkeys(target for _, target in sites))
        for name, sites in writes.items()
        if len(sites) > 1
        and name not in repeated
        and all(owner == sites[0][0] for owner, _ in sites)
    }
    return set(writes), single, multiple


class _Unclosed(Exception):
    def __init__(self, name, node, field, detail):
        self.issue = dict(
            code="python." + field,
            symbol=name,
            location=node.get("location"),
            **({"site": node["site"]} if node.get("site") is not None else {}),
            detail=detail,
        )
        super().__init__(detail)


class _Region:
    """A statement boundary and its in-progress child regions in the target buffer."""

    def __init__(self, steps, parent, source):
        self.steps = steps
        self.start = len(steps) if parent is None or steps is parent.steps else 0
        self.end = self.start
        self.source = source
        self.current = None
        self.children = []
        self.handler_entries = {}

    def boundary(self, node):
        self.current, self.end = node, len(self.steps)
        self.children.clear()
        self.handler_entries.clear()

    def fragment(self):
        record = dict(
            completion="complete" if self.current is None else "prefix",
            body=self.steps[self.start : self.end],
        )
        if self.current is None:
            return record
        node = self.current
        frontier = dict(operator=node["op"], **values.source_site(node), regions=[])
        # Constant-folded branches may share the caller's destination. Their
        # operations remain inside the corresponding source branch in this record.
        before_child = min(
            (child.start for child in self.children if child.steps is self.steps),
            default=len(self.steps),
        )
        setup = []
        for statement in self.steps[self.end : before_child]:
            if statement["op"] not in {"decl", "assign", "expr"} and not statement.get(
                "pending_import_effect"
            ):
                break
            setup.append(statement)
        if setup:
            frontier["evaluation"] = setup
        for child in self.children:
            path = next(
                (
                    [key]
                    for key in ("body", "then", "else", "finally")
                    if node.get(key) is child.source
                ),
                None,
            )
            if path is None:
                path = next(
                    (
                        ["handlers", i, "body"]
                        for i, handler in enumerate(node.get("handlers", ()))
                        if handler.get("body") is child.source
                    ),
                    None,
                )
            if path is not None:
                region = dict(path=path, **child.fragment())
                if path[0] == "handlers" and path[1] in self.handler_entries:
                    region["handler"] = self.handler_entries[path[1]]
                frontier["regions"].append(region)
        record["frontier"] = frontier
        return record


def _binding_use(node, evaluating):
    """Follow only connected operand edges of the current normalization visit."""
    if not evaluating or evaluating[-1] is not node:
        return None
    attributes = []
    child = node
    for index in range(len(evaluating) - 2, -1, -1):
        parent = evaluating[index]
        site = parent.get("site")
        if not site:
            break
        if (
            parent.get("op") in {"member", "process_stream"}
            and parent.get("base") is child
        ):
            attributes.append(dict(name=parent["field"], site=site))
            child = parent
            continue
        if parent.get("target") is child and parent["op"] in {
            "call",
            "library_call",
            "python_invoke",
            "process_launch",
            "process_wait",
            "process_poll",
            "process_communicate",
        }:
            return dict(
                attributes=attributes, consumer=dict(role="call_target", site=site)
            )
        for arg, item in enumerate(parent.get("args", ())):
            if item is child:
                return dict(
                    attributes=attributes,
                    consumer=dict(role="argument", index=arg, site=site),
                )
        break
    if attributes:
        return dict(attributes=attributes)
    return None


class _Project:
    def __init__(self, functions, *, entry_environment=None, function_definitions=None):
        self.functions = functions
        self.function_definitions = function_definitions or {}
        self.initializers = {
            scope["module"]: name
            for name, f in functions.items()
            if (scope := f.get("python_scope", {})).get("kind") == "module"
        }
        self.imports = ImportState(
            self.initializers,
            main_namespace=next(
                (
                    owner
                    for owner, symbol in self.initializers.items()
                    if functions[symbol]["python_scope"].get("main")
                ),
                None,
            ),
        )
        self.names = {}
        self.globals = {}
        self.types = {}
        self.closed = {}
        self.activations = {}
        self.prefixes = []
        self.result_kinds = {}
        self.result_facts = {}
        self.mapping_parameters = {}
        self.pending_signatures = {}
        self.recursive_uses = set()
        self.function_defaults = {}
        self.normalized_sites = set()
        self.providers = {}
        self.array_ops = set()
        self.process_ops = set()
        self.process_requests = {}
        self.required_operators = {}
        self.execution_images = []
        self.array_state = arrays.LiteralState()
        self.objects = objects.ObjectModel()
        self.graphics = graphics.Graphics()
        self.decisions = []
        self.skipped_sites = {}
        self.skipped_edges = {}
        self.live_sites = set()
        self.actual_calls = set()
        self.active = []
        self.global_reads = {}
        self.registry_reads = {}
        self.registry_writes = {}
        self.import_markers = []
        self.source_imports = host_protocols.ImportOccurrences(
            functions, entry_environment=entry_environment
        )
        self.shadow, self.single_bind_globals, self.global_writers = _global_bindings(
            functions
        )
        self.sequences = sequences.Domains(functions, self.single_bind_globals)

    def require_operator(self, family, operation, **contract):
        requirement = self.required_operators.setdefault(
            family, dict(operations=set(), **contract)
        )
        if any(requirement.get(key) != value for key, value in contract.items()):
            raise ValueError("conflicting logical operator requirements")
        requirement["operations"].add(operation)

    def operator_requirements(self):
        return {
            family: {**requirement, "operations": sorted(requirement["operations"])}
            for family, requirement in self.required_operators.items()
        }

    def structured_parameters(self, name):
        """Infer a closed value-family constraint from a real builtin type test."""
        if name not in self.mapping_parameters:
            fn = self.functions[name]
            formals = {p["name"]: i for i, p in enumerate(fn["params"])}
            selected = set()
            for operation in walk_operations(fn["body"]):
                target = operation.get("target", {})
                args = operation.get("args", ())
                if (
                    operation["op"] in {"call", "library_call"}
                    and target.get("op") == "python_builtin"
                    and target.get("name") == "isinstance"
                    and len(args) == 2
                    and args[1].get("op") == "python_builtin"
                    and args[1].get("name") == "dict"
                    and args[0].get("op") == "var"
                    and args[0].get("scope") != "global"
                    and args[0].get("name") in formals
                ):
                    selected.add(formals[args[0]["name"]])
            self.mapping_parameters[name] = frozenset(selected)
        return self.mapping_parameters[name]

    def _check_recursive_bindings(self, function):
        """A provisional signature cannot license folding changing global bindings."""
        pending, seen = [function], set()
        while pending:
            current = pending.pop()
            if current["name"] in seen:
                continue
            seen.add(current["name"])
            for operation in walk_operations(current["body"]):
                if any(
                    target.get("scope") == "global" or target.get("cell")
                    for target in values.written_targets(operation)
                ):
                    raise _Unclosed(
                        function["name"],
                        operation,
                        "recursive_state",
                        "recursive normalization requires closed local bindings",
                    )
                if operation["op"] == "call":
                    callee = operation.get("callee")
                    if callee == function["name"]:
                        continue
                    if callee not in self.closed:
                        raise _Unclosed(
                            function["name"],
                            operation,
                            "recursive_effects",
                            "recursive callee effects require a closed body",
                        )
                    pending.append(self.closed[callee])

    def import_may_write(self, body):
        """Summarize closed canonical callees without trusting unknown effects.

        Do not cache a negative result: a deferred import marker may acquire its
        actual producer after a later namespace consumer is analyzed.
        """
        pending, seen = [body], set()
        while pending:
            for operation in walk_operations(pending.pop()):
                writes = operation.get("writes", ())
                if effects.has_open_effect(operation) or {
                    "module.registry",
                    "module.namespace",
                }.intersection(writes):
                    return True
                opcode = operation["op"]
                if opcode == "call":
                    name = operation.get("callee")
                    function = self.closed.get(name)
                    if (
                        function is None
                        or name in self.registry_writes
                        or self.functions.get(name, {})
                        .get("python_scope", {})
                        .get("kind")
                        == "module"
                    ):
                        return True
                    if name not in seen:
                        seen.add(name)
                        pending.append(function["body"])
        return False

    def global_read(self, name, value, node=None, evaluating=()):
        parts = name.split(":", 2)
        if len(parts) == 3 and parts[0] == "py":
            self.observe_import(parts[1], "namespace", node, parts[2], evaluating)
        for active in self.active:
            if self.functions[active]["python_scope"]["kind"] == "function":
                self.global_reads[active][name] = (value.kind, value.value)

    def read_registry(self, name, proof, node=None):
        self.observe_import(name, "registry", node)
        for active in self.active:
            if self.functions[active]["python_scope"]["kind"] == "function":
                self.registry_reads[active][name] = proof

    def scoped_registry_effect(
        self, owner, node, operation, steps, *, before=None, completed=None
    ):
        # Compare around the original action without executing it again. A later
        # consumer adds only the state ports it actually requires.
        marker = dict(op="block", body=[], pending_import_effect=True)
        pending = False
        baseline = self.imports.snapshot() if before is None else before
        epoch = baseline["binding_epoch"]
        completed = completed or {}
        for name, entry in baseline["registry"].items():
            prior = baseline["import_windows"].get(name, ())
            proof = self.imports.normal_identity(entry, prior)
            if proof is None:
                continue
            if (
                name in operation["direct_registry_keys"]
                or name in operation["conditional_registry_keys"]
            ):
                if name not in completed:
                    self.imports.bind_registry(
                        name,
                        dict(
                            state="unresolved", owner=owner, **values.source_site(node)
                        ),
                    )
                self.imports.drop_windows(name)
                continue
            if not prior and name in baseline["namespace_writes"]:
                # No comparison covers the earlier opaque write. Starting a
                # new comparison here would conceal the missing prefix.
                self.imports.invalidate_entry(
                    name,
                    dict(
                        state="unresolved",
                        epoch=self.imports.next_epoch(),
                        owner=owner,
                        **values.source_site(node),
                    ),
                )
                continue
            self.imports.bind_windows(
                name, [*prior, (marker, operation, proof, epoch)], proof
            )
            pending = True
        if pending:
            self.import_markers.append(marker)
            steps.append(marker)

    def captured_registry_effect(self, owner, node, steps, before):
        capture = node["capture"]
        node["registry_state_capture"] = dict(
            identity=capture,
            owner=owner,
            scope="per_dynamic_execution_of_original_action",
            before="immediately_before_original_action",
            after="normal_return_before_any_later_action",
            result="retain_original_captured_result_without_reexecution",
        )
        operation = dict(
            op="module_state_preserves",
            semantics_version=1,
            type="bool",
            operator_family="module_registry",
            source_protocol="captured_effect_state_comparison",
            occurrence="action/" + capture,
            producer_capture=capture,
            producer_owner=owner,
            direct_registry_keys=[],
            conditional_registry_keys=[],
            effects=["CTRL", "STATE"],
            reads=["captured_action.before", "captured_action.after"],
            writes=[],
            may_raise=False,
            checks={},
            **values.source_site(node),
        )
        self.scoped_registry_effect(owner, node, operation, steps, before=before)

    def registry_entry(self, name, node):
        """Use an earlier identity only on explicit successful comparison edges."""
        current = self.imports.registry_entry(name)
        windows = self.imports.windows(name)
        if current.get("state") != "unresolved" or not windows:
            return current
        expected = self.imports.normal_identity(current, windows)
        if expected is None or not self.imports.covers_registry_lookup(
            name, expected, windows
        ):
            return current
        self.observe_import(name, "registry", node)
        self.imports.bind_registry(name, expected)
        self.decisions.append(
            dict(
                code="guarded_module_registry_continuation",
                module=name,
                state=expected["state"],
                guards=[window[1]["occurrence"] for window in windows],
                proof="normal edges of explicit state comparisons; failed edges require source-equivalent continuation",
                **values.source_site(node),
            )
        )
        return expected

    def observe_import(self, name, domain, node, member=None, evaluating=()):
        key = self.imports.registry_key(name) if domain == "namespace" else name
        version = (
            self.imports.binding_version(namespace_slot(name, member))
            if member is not None
            else 0
        )
        if domain == "namespace" and (mutation := self.imports.namespace_change(key)):
            if type(mutation.get("epoch")) is int and version > mutation["epoch"]:
                mutation = None
        else:
            mutation = None
        if mutation is not None and not self.imports.windows(key):
            owner = self.active[-1]
            error = _Unclosed(
                owner,
                node or self.functions[owner],
                "module_namespace_after_effect",
                f"an opaque effect may have changed {name}.{member} through a retained module alias",
            )
            error.issue["source_evidence"] = dict(
                module=name, registry_key=key, member=member, last_change=mutation
            )
            raise error
        for marker, operation, proof, epoch in self.imports.windows(key):
            if domain == "namespace" and member is not None and version > epoch:
                continue
            checks = operation["checks"].setdefault(key, {"before": proof})
            checks.setdefault(
                domain,
                dict(
                    requires=(
                        (
                            "same_module_instance_and_complete"
                            if proof["state"] == "ready"
                            else "same_module_instance_and_initializing"
                        )
                        if domain == "registry"
                        else "same_namespace_bindings_and_reachable_value_state"
                    ),
                    first_consumer=self.active[-1],
                    **values.source_site(node or {}),
                    **({"first_member": member} if member is not None else {}),
                    **({"lexical_namespace": name} if domain == "namespace" else {}),
                ),
            )
            if domain == "namespace" and member is not None:
                requirement = checks[domain]
                reads = requirement.setdefault("binding_reads", [])
                read = dict(
                    member=member,
                    analysis_binding_epoch=version,
                    consumer=self.active[-1],
                    **values.source_site(node or {}),
                )
                use = _binding_use(node, evaluating)
                if use:
                    read["operand_use"] = use
                if read not in reads:
                    reads.append(read)
                # A field or call edge identifies a consumer. Its implementation
                # may still observe additional state or escape the object.
                requirement["value_state_projection"] = "unproven"
            if marker["op"] == "block":
                captured_effect = (
                    operation.get("source_protocol")
                    == "captured_effect_state_comparison"
                )
                continuation = (
                    "state_effect_continuation_required"
                    if captured_effect
                    else "module_import_continuation_required"
                )
                resume = {}
                if captured_effect:
                    resume = dict(
                        producer_capture=operation["producer_capture"],
                        resume="continue_after_original_action_with_captured_result_and_actual_post_state",
                    )
                elif operation.get("source_protocol") == "module_state_comparison":
                    ledger = self.source_imports
                    row = ledger.rows.get(operation["occurrence"])
                    if row is not None:
                        if any(
                            p["opcode"] == "module_namespace_initialize"
                            and p["role"] == "source_import"
                            and p.get("captures_import_result_and_state")
                            for p in row["producer_sites"]
                        ):
                            resume = dict(
                                occurrence=row["id"],
                                resume="continue_from_original_import_result_and_post_state_without_reexecuting_loader",
                            )
                for op in (operation["op"], continuation):
                    namespace_operations.require_registry_operator(self, op)
                if captured_effect:
                    namespace_operations.require_registry_operator(
                        self, "capture_effect_state"
                    )
                marker.clear()
                marker.update(
                    op="if",
                    condition=operation,
                    then=[],
                    **{
                        "else": [
                            dict(
                                op="expr",
                                value=dict(
                                    op=continuation,
                                    type="void",
                                    operator_family="module_registry",
                                    noreturn=True,
                                    effects=["CTRL", "OPAQUE"],
                                    semantics="transfer_to_source_equivalent_continuation_or_prove_edge_unreachable_before_delivery",
                                    **resume,
                                    **values.source_site(operation),
                                ),
                            )
                        ]
                    },
                )

    def finish_imports(self):
        unused = {
            id(marker) for marker in self.import_markers if marker["op"] == "block"
        }
        unused.update(self.source_imports.empty_markers())
        for marker, _ in self.source_imports.initialization_points.values():
            marker.pop("pending_import_effect", None)
        if unused:
            # Prefix reports and completed bodies may have copied a statement
            # list. They share the original marker objects.
            pending = [self.prefixes, *self.closed.values()]
            while pending:
                item = pending.pop()
                if isinstance(item, list):
                    if any(id(value) in unused for value in item):
                        item[:] = [value for value in item if id(value) not in unused]
                    pending.extend(item)
                elif isinstance(item, dict):
                    pending.extend(item.values())

    def unknown_registry(self, owner, node):
        """Record the call effect separately from the owned import-state change."""
        change = self.imports.invalidate(owner, values.source_site(node))
        for active in self.active:
            if self.functions[active]["python_scope"]["kind"] == "function":
                self.registry_writes[active] = change

    def inactive_bodies(self, activation):
        """Only proven edge removal can turn a required body into an inactive body."""
        if not activation:
            return set()
        roots, edges = activation["roots"], activation["edges"]
        removed, kept = [], []
        for edge in edges:
            key = edge["owner"], edge.get("site")
            edge_key = (
                *key,
                edge["target"],
                edge.get("role"),
                json_bytes(edge.get("condition")),
            )
            if edge_key in self.skipped_edges:
                removed.append(dict(edge, proof=self.skipped_edges[edge_key]))
            elif (
                edge.get("site") is not None
                and key in self.skipped_sites
                and key not in self.live_sites
            ):
                removed.append(dict(edge, proof=self.skipped_sites[key]))
            else:
                kept.append(edge)
        kept.extend(
            dict(owner=owner, target=target) for owner, target in self.actual_calls
        )
        inactive = (
            (_reachable(roots, edges) - _reachable(roots, kept))
            .intersection(self.functions)
            .difference(self.closed)
        )
        if removed:
            self.decisions.append(
                dict(
                    code="inactive_activation_edges",
                    roots=roots,
                    removed_edges=removed,
                    inactive_bodies=sorted(inactive),
                    proof="root reachability changes only after excluding proven untaken source sites; hard edges and actual calls remain",
                )
            )
        return inactive

    def compile(self, name, types, *, source=None):
        fn = self.functions[name]
        if name in self.active:
            prototype = self.pending_signatures.get(name)
            if prototype is not None and prototype["argument_kinds"] == types:
                self.recursive_uses.add(name)
                return prototype
            raise _Unclosed(
                name,
                fn,
                "cycle",
                "recursive calls or partially initialized modules need a recursive contract",
            )
        if name in self.closed:
            if len(self.types[name]) != len(types) or not all(
                self.sequences.compatible(left, right)
                for left, right in zip(self.types[name], types)
            ):
                raise _Unclosed(
                    name,
                    fn,
                    "argument_types",
                    f"call types {types} conflict with earlier {self.types[name]}",
                )
            for module, proof in self.registry_reads.get(name, {}).items():
                if self.imports.registry_entry(module) != proof:
                    error = _Unclosed(
                        name,
                        source or fn,
                        "import_module_cache",
                        f"the preceding initialization proof for {module} no longer holds at this call",
                    )
                    current = self.imports.registry_entry(module)
                    error.issue["source_evidence"] = dict(
                        module=module,
                        required=proof,
                        registry_state=current.get(
                            "state", "no_scheduled_initialization"
                        ),
                        last_change=current,
                    )
                    raise error
                self.read_registry(module, proof, source or fn)
            for slot, required in self.global_reads.get(name, {}).items():
                current = self.names.get(slot)
                if current is None or (current.kind, current.value) != required:
                    raise _Unclosed(
                        name,
                        fn,
                        "name_binding",
                        f"cached function requires initialized binding {slot} at this call",
                    )
                self.global_read(slot, current, source or fn)
            if write := self.registry_writes.get(name):
                self.unknown_registry(write["owner"], write)
            return self.closed[name]
        scope = fn.get("python_scope", {})
        kind = scope.get("kind")
        if (
            kind not in {"module", "function"}
            or kind == "function"
            and fn.get("call_protocol", {}).get("kind") != "regular"
        ):
            raise _Unclosed(
                name,
                fn,
                "activation",
                "a regular function or module initialization contract is required",
            )
        if len(types) != len(fn["params"]):
            raise _Unclosed(
                name,
                fn,
                "arity",
                "the invocation must bind every formal parameter once",
            )
        if source is not None:
            self.activations[name] = dict(
                owner=self.active[-1], **values.source_site(source)
            )
        self.active.append(name)
        self.global_reads[name] = {}
        self.registry_reads[name] = {}
        registry_prefix = self.imports.snapshot() if kind == "module" else None
        registry_key = self.imports.registry_key(scope["module"])
        if kind == "module" and not self.imports.registry_entry(registry_key):
            self.imports.bind_registry(
                registry_key,
                dict(
                    state="initializing",
                    initializer=name,
                    owner=name,
                    **values.source_site(source or fn),
                ),
            )
        saved_contents = None
        if kind == "function":
            saved_contents = (
                self.array_state.contents,
                self.sequences.contents,
                self.sequences.states,
            )
            (
                self.array_state.contents,
                self.sequences.contents,
                self.sequences.states,
            ) = ({}, {}, {})
        try:
            frame = _Frame(self, fn, types)
            if kind == "function" and self.structured_parameters(name):
                self.pending_signatures[name] = dict(
                    name=name,
                    params=frame.params,
                    result=values.wire("structured_value"),
                    argument_kinds=types,
                    recursive_prototype=True,
                )
                self.result_kinds[name] = "structured_value"
                self.result_facts[name] = {}
            try:
                result = frame.lower()
            except _Unclosed:
                if registry_prefix is not None:
                    # No caller may use a proof that required the failed
                    # initializer to return. Its executed prefix is reported.
                    self.imports.restore(registry_prefix)
                self.prefixes.append(frame.prefix_fragment())
                raise
            if name in self.recursive_uses:
                self._check_recursive_bindings(result)
                result["recursive_contract"] = dict(
                    semantics_version=1,
                    result="closed_builtin_structured_value",
                    bindings="no_recursive_global_or_cell_writes",
                )
            self.types[name] = types
            self.closed[name] = result
            return result
        finally:
            self.pending_signatures.pop(name, None)
            if saved_contents is not None:
                (
                    self.array_state.contents,
                    self.sequences.contents,
                    self.sequences.states,
                ) = saved_contents
            self.active.pop()


class _Frame:
    def __init__(self, project, fn, types):
        self.project, self.fn = project, fn
        self.name, self.scope = fn["name"], fn["python_scope"]
        self.module = self.scope["module"]
        self.initializing = self.scope["kind"] == "module"
        self.env = {}
        self.locals = {}
        self.returns = set()
        self.return_facts = None
        self.next_temp = 0
        self.loops = []
        self.handler_depth = 0
        self.region = None
        self.root_region = None
        self.evaluating = []
        self.dynamic = 0
        self.control_regions = []
        self.params = []
        self.project.sequences.bind_frame(fn, types)
        for formal, kind in zip(fn["params"], types):
            if formal.get("cell") or formal.get("binding") not in {
                "positional",
                "posonly",
                "keywordonly",
            }:
                self.fail(
                    formal,
                    "formal",
                    "captured or variadic parameter needs a binding contract",
                )
            name = formal["name"]
            self.params.append(dict(name=name, type=values.wire(kind)))
            facts = (
                {"mapping_origin": {"owner": self.name, "parameter": name}}
                if kind == "structured_value"
                else {}
            )
            self.env[name] = _Value(kind, None, facts).at(name)
            self.locals[name] = kind

    def fail(self, node, field, message, *, source_evidence=None):
        error = _Unclosed(self.name, node, field, message)
        if source_evidence:
            error.issue["source_evidence"] = source_evidence
        raise error

    def scalar(self, value, node):
        if not values.runtime(value.kind):
            self.fail(
                node, "value_contract", "symbolic binding escapes its closed namespace"
            )
        return value.value

    def skip_expression(self, node, reason):
        for part in walk_operations(node):
            if part.get("site") is not None:
                self.project.skipped_sites[self.name, part["site"]] = reason

    def temporary(self):
        self.next_temp += 1
        return f"@evaluation/{self.next_temp}"

    def observe_effect(self, node):
        writes = node.get("writes", ())
        before_effect = None
        if (
            effects.has_open_effect(node)
            or "module.registry" in writes
            and node.get("source_protocol") != "python.import"
        ):
            self.project.array_state.invalidate()
            self.project.sequences.invalidate()
            if isinstance(node.get("capture"), str) and node["capture"]:
                before_effect = self.project.imports.snapshot()
            self.project.unknown_registry(self.name, node)
        return before_effect

    def effect(self, node, kind, steps, discard=False, *, facts=None):
        source = (self.evaluating[-1] if self.evaluating else None) or (
            self.region.current if self.region is not None else None
        )
        node = {**values.source_site(source or {}), **node}
        before_effect = self.observe_effect(node)
        if discard:
            steps.append(dict(op="expr", value=node))
            result = _literal(None)
        else:
            name = self.temporary()
            steps.append(dict(op="decl", name=name, type=values.wire(kind), init=node))
            result = _Value(kind, None, facts or {}).at(name)
        if before_effect is not None:
            self.project.captured_registry_effect(self.name, node, steps, before_effect)
        return result

    def global_value(self, name, value):
        self.project.global_read(
            name,
            value,
            (self.evaluating[-1] if self.evaluating else None)
            or (self.region.current if self.region else None),
            self.evaluating,
        )
        if (
            self.initializing
            or not values.runtime(value.kind)
            or value.value["op"] != "var"
        ):
            return value
        facts = {}
        if name in self.project.single_bind_globals:
            if value.kind in values.SCALARS:
                facts = value.facts
            elif values.is_array(value.kind) and "shape" in value.facts:
                # A fixed handle and immutable region layout do not prove content.
                facts = {"shape": value.facts["shape"]}
            elif values.is_product(value.kind):
                facts = products.returned_facts(value)
            elif "stream_origin" in value.facts:
                facts = {"stream_origin": value.facts["stream_origin"]}
        elif value.kind == "integer" and name in self.project.global_writers:
            owner, writers = self.project.global_writers[name]
            # A completed compilation supplies all writes in this initializer's
            # code. Retain only an interval true of every possible assignment.
            if owner in self.project.closed and all(writers.values()):
                facts = values.join_integer_ranges(writers.values())
        return _Value(value.kind, value.value, facts)

    def ref(self, node):
        name = node["name"]
        if node["op"] == "python_builtin" and self.project.imports.open_namespace:
            self.fail(
                node,
                "builtin_after_callback",
                f"lookup of {name!r} needs the current globals and builtins after a Python callback",
                source_evidence=self.project.imports.open_namespace,
            )
        if node.get("cell"):
            self.fail(node, "cell", "a lexical cell needs a closed activation lifetime")
        if node.get("scope") == "global":
            result = (
                None
                if self.project.imports.open_namespace
                else self.project.names.get(name)
            )
            if result is not None:
                result = self.global_value(name, result)
            elif publication := self.project.imports.publications.get(name):
                self.fail(
                    node,
                    "global_binding_after_import",
                    f"using {name} needs the actual value in the current frame globals after import callbacks",
                    source_evidence=dict(
                        slot=name,
                        prior_publication=publication,
                        last_change=self.project.imports.open_namespace,
                        required="lookup_actual_current_binding_and_qualify_its_representation_or_propagate_failure",
                    ),
                )
        elif node["op"] == "python_builtin":
            slot = namespace_slot(self.module, name)
            result = self.env.get(name, self.project.names.get(slot))
            if result is None and slot not in self.project.shadow:
                if name in {
                    "print",
                    "range",
                    "int",
                    "float",
                    "abs",
                    "min",
                    "max",
                    "len",
                    "tuple",
                    "sorted",
                    "dict",
                    "isinstance",
                    "bool",
                }:
                    result = _Value("builtin", name)
                else:
                    result = exceptions.builtin(name) or host_protocols.builtin(
                        self, node, name
                    )
                    if result is None:
                        self.fail(
                            node,
                            "provider_contract",
                            f"builtins.{name} needs a typed provider contract",
                        )
        else:
            result = self.env.get(name)
        if result is None:
            self.fail(
                node,
                "name_binding",
                f"{name} has no definitely initialized binding at this point",
            )
        return result

    def array_readonly_call(self, node, rebound):
        """A nonactivating lookup for loop effects; unresolved dispatch stays open."""

        for part in walk_operations(node.get("target", {})):
            if part["op"] in {"var", "python_builtin"} and (
                part["name"] in rebound
                or namespace_slot(self.module, part["name"]) in rebound
            ):
                return False

        def lookup(expr):
            op = expr.get("op")
            if op in {"var", "python_builtin"}:
                name = expr["name"]
                if op == "python_builtin":
                    slot = namespace_slot(self.module, name)
                    found = self.env.get(name, self.project.names.get(slot))
                    if found is None and slot not in self.project.shadow:
                        if name in {
                            "print",
                            "range",
                            "int",
                            "float",
                            "abs",
                            "min",
                            "max",
                            "len",
                            "tuple",
                            "sorted",
                        }:
                            return _Value("builtin", name)
                        return exceptions.builtin(name) or host_protocols.builtin(
                            self, expr, name
                        )
                    return found
                scope = (
                    self.project.names if expr.get("scope") == "global" else self.env
                )
                return scope.get(name)
            if op == "member":
                base = lookup(expr["base"])
                if base and base.kind == "module":
                    return self.project.names.get(
                        namespace_slot(base.value, expr["field"])
                    )
                if base and base.kind == "host_namespace":
                    return host_protocols.readonly_callable(base, expr["field"])
            return None

        target = lookup(node.get("target", {}))
        return target is not None and (
            target.kind in {"builtin", "exception_class"}
            or host_protocols.array_effect_free(target)
        )

    def store(self, target, value, steps):
        if target.get("op") == "unpack":
            items = target["items"]
            for item in items:
                if item.get("op") == "starred" or item.get("construct") == "Starred":
                    self.fail(
                        item,
                        "unpack_starred",
                        "a starred target requires a list of the remaining values",
                    )
            if values.is_product(value.kind):
                value = self.effect(value.value, value.kind, steps, facts=value.facts)
            elements = products.elements(self, target, value, steps)
            if elements is None:
                self.fail(
                    target,
                    "unpack_iterable",
                    f"unpacking {value.kind} requires a closed iterator contract",
                )
            if len(items) != len(elements):
                self.fail(
                    target,
                    "unpack_arity",
                    f"assignment expects {len(items)} items, received {len(elements)}",
                )
            # Check each nested sequence upon reaching its target. Earlier targets
            # can rebind names used to evaluate a later receiver or index.
            for item, element in zip(items, elements):
                self.store(item, element, steps)
            return
        if target.get("op") == "index":
            base = self.evaluate(target["base"], steps)
            if values.runtime(base.kind):
                base = self.effect(base.value, base.kind, steps, facts=base.facts)
            if base.kind in {"mapping_ref", "structured_value"}:
                indices = target.get("indices", [])
                if len(indices) != 1:
                    self.fail(
                        target, "mapping_key", "mapping access needs one text key"
                    )
                mappings.put(
                    self, target, base, self.evaluate(indices[0], steps), value, steps
                )
            else:
                arrays.index(self, target, steps, rhs=value, evaluated_base=base)
            return
        if target.get("op") == "member":
            if values.runtime(value.kind) and value.value["op"] != "literal":
                value = self.effect(value.value, value.kind, steps)
            base = self.evaluate(target["base"], steps)
            self.project.objects.write(
                self, target, base, target["field"], value, steps
            )
            return
        if target.get("op") != "var" or target.get("cell"):
            self.fail(
                target,
                "store",
                "attribute, subscript or closure assignment needs a state contract",
            )
        name = target["name"]
        global_ = target.get("scope") == "global"
        if global_ and self.project.imports.open_namespace:
            self.fail(
                target,
                "global_store_after_callback",
                f"replacing {name} must preserve the current binding and its release effects",
                source_evidence=self.project.imports.open_namespace,
            )
        if global_ and not self.initializing:
            self.fail(target, "global_write", "a function writes a module namespace")
        if global_ and name in {
            namespace_slot(self.module, field)
            for field in (
                "__builtins__",
                "__name__",
                "__package__",
                "__path__",
                "__spec__",
                "__loader__",
            )
        }:
            self.fail(
                target, "rebind", "assignment changes the module or import environment"
            )
        scope = self.project.names if global_ else self.env
        previous = scope.get(name)
        if value.kind == "sequence_method":
            value = sequences.persist_method(self, target, value, steps)
        if global_ and value.kind == "project_method" and not self.dynamic:
            value = objects.persist_method(self, target, value, steps)
        if (
            global_
            and value.kind in {"array_copy_method", "array_reshape_method"}
            and not self.dynamic
        ):
            value = arrays.persist_method(self, target, value, steps)
        if global_ and value.kind == "process_method" and not self.dynamic:
            value = processes.persist_method(self, target, value, steps)
        if not values.runtime(value.kind):
            if (
                self.dynamic
                and not self.protected_normal_context()
                or value.kind == "range"
                and global_
            ):
                self.fail(
                    target,
                    "conditional_binding",
                    f"{value.kind} needs a persistent runtime binding here",
                )
            if (
                previous is not None
                and previous != value
                and not (
                    self.initializing
                    and global_
                    and previous.kind == value.kind == "project_class"
                )
            ):
                self.fail(target, "rebind", f"symbolic identity of {name} changes")
            scope[name] = value
            if global_:
                self.project.imports.binding_written(name)
            return
        if previous is not None and previous.kind == "integer" and value.kind == "i64":
            value = integers.promote(self, target, value, steps=steps)
        if previous is not None and not self.project.sequences.compatible(
            previous.kind, value.kind
        ):
            self.fail(
                target,
                "rebind",
                f"{name} changes type from {previous.kind} to {value.kind}",
            )
        types = self.project.globals if global_ else self.locals
        storage = (
            previous.value["name"]
            if previous is not None
            and isinstance(previous.value, dict)
            and previous.value.get("binding_slot") == name
            else name
        )
        if storage in types and not self.project.sequences.compatible(
            types[storage], value.kind
        ):
            self.fail(
                target, "local_type", f"{name} has incompatible reaching value types"
            )
        types[storage] = (
            values.join_arrays([types[storage], value.kind])
            if storage in types and values.is_array(value.kind)
            else value.kind
        )
        variable = value.at(
            storage,
            **({"scope": "global"} if global_ else {}),
            **({"binding_slot": name} if storage != name else {}),
        )
        scope[name] = variable
        if global_:
            self.project.imports.binding_written(name)
            if value.kind == "integer" and name in self.project.global_writers:
                _, writers = self.project.global_writers[name]
                site = id(target)
                if site in writers:
                    previous_write = writers[site]
                    incoming = (
                        (value,) if previous_write is None else (previous_write, value)
                    )
                    writers[site] = _Value(
                        "integer", None, values.join_integer_ranges(incoming)
                    )
        steps.append(dict(op="assign", target=variable.value, value=value.value))

    @property
    def provider_sources(self):
        initializer = self.project.functions.get(self.scope.get("initializer"), {})
        scope = initializer.get("python_scope", {})
        if scope.get("kind") == "module" and scope.get("module") == self.module:
            return scope.get("provider_sources", {})
        return {}

    def _import_evidence(self, module):
        """Summarize where a name could come from when initialization is unclosed."""
        sources = {}
        for name, record in self.provider_sources.items():
            if name != module and not name.startswith(module + "."):
                continue
            row = {
                k: record[k]
                for k in (
                    "status",
                    "provider_id",
                    "member",
                    "sha256",
                    "resolution",
                    "reason",
                )
                if k in record
            }
            exports = {}
            for member, evidence in record.get("exports", {}).items():
                exports[member] = dict(
                    candidates=[
                        {
                            **{
                                k: candidate[k]
                                for k in ("module", "member", "name", "line", "sha256")
                                if k in candidate
                            },
                            **(
                                {"conditional_paths": True}
                                if candidate.get("conditions")
                                or any(
                                    route.get("conditions")
                                    for route in candidate.get("alternate_paths", ())
                                )
                                else {}
                            ),
                        }
                        for candidate in evidence.get("candidates", ())
                    ],
                    unresolved=sorted(
                        {u["reason"] for u in evidence.get("unresolved", ())}
                    ),
                )
            if exports:
                row["exports"] = exports
            sources[name] = row
        return (
            dict(scope="source_candidates_only", modules=sources) if sources else None
        )

    def source_export(self, node, module, member, steps, namespace=None):
        if namespace is None and node.get("op") == "python_import":
            namespace = host_protocols.source_namespace(self, node, module)
        return host_protocols.source_export(self, node, namespace, member, steps)

    def source_namespace(self, node, module):
        return host_protocols.source_namespace(self, node, module)

    def import_origin(self, node):
        """Resolve the requested module independently of its execution contract."""
        module = node["module"]
        root = module.partition(".")[0]
        local = node.get("provider") == "project"
        source = node.get("source_provider", {})
        exact = "import_resolution" in node
        if not exact and local:
            self.require_import_instance(node, module)
            return None
        resolved = node.get("import_resolution") if exact else source
        if not isinstance(resolved, dict) or not isinstance(source, dict):
            self.fail(
                node, "import_identity", f"invalid resolution record for {module}"
            )
        # A per-site identity and its sealed unit must agree independently of
        # which subset of exports either record was asked to describe.
        identities = [(root, source)]
        if exact:
            identities.append((module, resolved))
        for requested, evidence in identities:
            stored = self.provider_sources.get(requested)
            if (
                stored
                and evidence
                and any(
                    stored.get(key) != evidence.get(key)
                    for key in (
                        "module",
                        "status",
                        "origin",
                        "sha256",
                        "distribution",
                        "version",
                        "member",
                    )
                )
            ):
                self.fail(
                    node,
                    "import_identity",
                    f"site and unit disagree about the source of {requested}",
                )
        if exact and resolved.get("module") != module:
            self.fail(
                node,
                "import_identity",
                f"source evidence for {resolved.get('module')!r} cannot identify {module}",
            )
        status = resolved.get("status")
        if status == "project":
            self.require_import_instance(node, module)
            if not local or module not in self.project.initializers:
                self.fail(
                    node,
                    "import_initializer",
                    f"the resolved project module {module} has no matching graph initializer",
                )
            return None
        if status == "namespace":
            self.fail(
                node,
                "import_namespace",
                f"{module} is a namespace package whose search locations and bindings need an initialization contract",
            )
        if status not in {"verified_stdlib", "verified_distribution"}:
            kind = resolved.get("resolution")
            detail = (
                resolved.get("reason") or "no verified origin for this import request"
            )
            code = "import_missing" if kind == "missing" else "import_resolution"
            origin = f" at {resolved['origin']}" if resolved.get("origin") else ""
            self.fail(
                node, code, f"resolution of {module}{origin} is incomplete ({detail})"
            )
        if local:
            self.fail(
                node,
                "import_identity",
                f"project binding for {module} conflicts with the resolved external source",
            )
        if exact:
            digest = resolved.get("sha256")
            if (
                not resolved.get("origin")
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                self.fail(
                    node,
                    "import_identity",
                    f"{module} lacks an origin and matching file or interpreter digest",
                )
            if status == "verified_distribution" and not (
                resolved.get("distribution") and resolved.get("version")
            ):
                self.fail(
                    node,
                    "import_identity",
                    f"{module} lacks the identity and version of its owning distribution",
                )
        if exact and module == root:
            keys = ["module", "status", "origin", "sha256"]
            if status == "verified_distribution":
                keys += ["distribution", "version"]
            if any(source.get(key) != resolved.get(key) for key in keys):
                self.fail(
                    node,
                    "import_identity",
                    f"root provider and exact resolution disagree for {module}",
                )
        if module != root:
            alias = source.get("aliases", {}).get(module)
            if exact and alias is not None:
                if resolved.get("canonical_module") != alias.get("module") or any(
                    resolved.get(k) != alias.get(k)
                    for k in ("status", "origin", "sha256")
                ):
                    self.fail(
                        node,
                        "import_identity",
                        f"the selected source and provider alias disagree for {module}",
                    )
            elif not exact:
                if (
                    not alias
                    or alias.get("status") != status
                    or not alias.get("sha256")
                ):
                    self.fail(
                        node,
                        "import_resolution",
                        f"the graph identifies {root} without a source identity for {module}",
                    )
        if node.get("graph_backed"):
            initializer = self.project.initializers.get(module)
            scope = self.project.functions.get(initializer, {}).get("python_scope", {})
            captured = scope.get("source_provider") or {}
            if (
                not initializer
                or scope.get("graph_backed") is not True
                or python_provider_id(captured) != python_provider_id(resolved)
            ):
                self.fail(
                    node,
                    "import_initializer",
                    f"the source graph initializer does not identify {module}",
                )
            return None
        return resolved

    def skip_activation(self, node, target, role, condition, proof):
        if node.get("site") is not None:
            key = self.name, node["site"], target, role, json_bytes(condition)
            self.project.skipped_edges[key] = proof

    def module_member(self, node, module, member, steps, *, optional=False):
        """Read current bindings before considering a missing-attribute hook."""
        module = self.project.imports.namespace_owner(module)
        self.project.observe_import(module, "namespace", node, member, self.evaluating)
        slot = namespace_slot(module, member)
        candidates = list(node.get("missing_attribute_hooks", ()))
        if candidate := node.get("missing_attribute_hook"):
            candidates.append(candidate)
        candidates = [
            c
            for c in candidates
            if c.get("when")
            == dict(kind="package_attribute_missing", module=module, name=member)
        ]
        value = self.project.names.get(slot)
        if value is not None:
            for candidate in candidates:
                self.skip_activation(
                    node,
                    candidate["symbol"],
                    "conditional_attribute",
                    candidate["when"],
                    dict(rule="module_member_bound", slot=slot),
                )
            return self.global_value(slot, value)
        if slot in self.project.shadow:
            self.fail(
                node,
                "module_attribute_state",
                f"the current binding state of {module}.{member} is unresolved",
            )
        hook = self.project.names.get(namespace_slot(module, "__getattr__"))
        if hook is not None:
            permitted = {c["symbol"] for c in candidates}
            if hook.kind != "function" or hook.value not in permitted:
                self.fail(
                    node,
                    "module_attribute_hook",
                    f"the current lookup hook of {module} has no matching source edge",
                )
            for candidate in candidates:
                if candidate["symbol"] != hook.value:
                    self.skip_activation(
                        node,
                        candidate["symbol"],
                        "conditional_attribute",
                        candidate["when"],
                        dict(rule="module_hook_binding", selected=hook.value),
                    )
            return self.invoke_project(
                node, hook.value, [_literal(member)], {}, steps, False
            )
        if namespace_slot(module, "__getattr__") in self.project.shadow:
            self.fail(
                node,
                "module_attribute_state",
                f"the current lookup hook binding of {module} is unresolved",
            )
        if optional:
            return None
        self.fail(node, "attribute", f"unbound module attribute {module}.{member}")

    def require_import_instance(self, node, module):
        if self.project.imports.registry_key(module) != module:
            self.fail(
                node,
                "import_entry_instance",
                f"importing {module!r} needs a second module namespace and initialization for the running script",
                source_evidence=dict(
                    lexical_namespace=module,
                    running_registry_key="__main__",
                    requested_registry_key=module,
                    graph_initializer=self.project.initializers[module],
                ),
            )

    def cached_module(self, node, module):
        """Prove the graph-owned module instance from its preceding activation."""
        self.require_import_instance(node, module)
        namespace = self.project.imports.namespace_owner(module)
        symbol = self.project.initializers.get(namespace)
        row = self.project.registry_entry(module, node)
        if (
            row.get("state") != "ready"
            and not (
                row.get("state") == "initializing" and symbol in self.project.active
            )
            or not symbol
            or row.get("initializer") != symbol
        ):
            state = row.get("state", "no_scheduled_initialization")
            self.fail(
                node,
                "import_module_cache",
                f"{module!r} has no proven completed module-registry entry ({state}); cold loading, import hooks and loader failures require a runtime contract",
                source_evidence={
                    "module": module,
                    "registry_state": state,
                    "initializer": symbol,
                    "scope": "process_module_registry",
                    **({"last_change": row} if row else {}),
                },
            )
        self.project.read_registry(module, row, node)
        self.project.decisions.append(
            dict(
                code="module_registry_hit",
                scope="process_module_registry",
                module=module,
                owner=self.name,
                initialized_by=row,
                entry_phase=row["state"],
                **values.source_site(node),
            )
        )
        return _Value("module", namespace)

    def package_fromlist(self, node, occurrence, *, dynamic=False):
        """Keep the package and member candidates at this IMPORT_NAME occurrence."""
        module = node["module"]
        members = node["request"].get("fromlist", ())
        phase = "during_IMPORT_NAME_before_IMPORT_FROM"
        checks = node.get("fromlist_checks")
        children = {}
        for child in node.get("conditional_initializers", ()):
            name = child["when"].get("name")
            if (
                name not in members
                or child["when"]
                != dict(kind="package_attribute_missing", module=module, name=name)
                or child["module"] != module + "." + name
                or child["initializer"]
                != self.project.initializers.get(child["module"])
            ):
                self.fail(
                    node,
                    "import_guard",
                    "conditional child does not match the package request",
                )
            children[name] = child
        if checks is not None and (
            not isinstance(checks, dict)
            or checks.get("phase") != phase
            or not isinstance(checks.get("items"), list)
            or len(checks["items"]) != len(members)
            or any(not isinstance(item, dict) for item in checks["items"])
        ):
            self.fail(
                node,
                "import_guard",
                "fromlist checks do not cover the source request in order",
            )

        if not members:
            self.fail(node, "import_guard", "package from-import has no fromlist")
        ordered_checks = []
        hook_slot = namespace_slot(module, "__getattr__")
        hook = self.project.names.get(hook_slot)
        for index, member in enumerate(members):
            when = dict(kind="package_attribute_missing", module=module, name=member)
            child = children.get(member)
            item = checks["items"][index] if checks is not None else {}
            supplied_child = item.get("child", {})
            if checks is not None and (
                item.get("binding_index") != index
                or item.get("member") != member
                or item.get("when") != when
                or not isinstance(supplied_child, dict)
                or (
                    supplied_child
                    and supplied_child.get("module") != module + "." + member
                )
                or supplied_child.get("initializer") != (child or {}).get("initializer")
            ):
                self.fail(
                    node,
                    "import_guard",
                    "fromlist member or conditional initializer differs from its source edge",
                )
            slot = namespace_slot(module, member)
            candidate = item.get("child") or (
                {"module": child["module"], "initializer": child["initializer"]}
                if child
                else None
            )
            check = dict(
                binding_index=index,
                member=member,
                when=when,
                child_candidate=candidate,
                graph_binding=(
                    "bound" if slot in self.project.names else "no_graph_binding"
                ),
                binding_epoch=self.project.imports.binding_version(slot),
                declared_binding=slot in self.project.shadow,
            )
            hooks = item.get("missing_attribute_hooks") or [
                h
                for h in [
                    *node.get("missing_attribute_hooks", ()),
                    node.get("missing_attribute_hook"),
                ]
                if h and h.get("when") == when
            ]
            if hooks:
                check["hook_candidates"] = [h["symbol"] for h in hooks]
            if candidate:
                check["child_registry_state"] = self.project.imports.registry_entry(
                    candidate["module"]
                ).get("state", "no_graph_initialization")
            ordered_checks.append(check)

        # A graph initializer models its body, not the calling __import__.
        # It may change even an already bound member, raise, or return another
        # object whose successive IMPORT_FROM lookups produce distinct values.
        first = ordered_checks[0]
        evidence = dict(
            **first,
            occurrence=occurrence["id"],
            phase=phase,
            package=module,
            ordered_checks=ordered_checks,
            registry_state=self.project.imports.registry_entry(module).get("state"),
            hook_binding=(
                "bound"
                if hook is not None
                else (
                    "unresolved"
                    if hook_slot in self.project.shadow
                    else "no_graph_binding"
                )
            ),
            actual_lookup="unproven_after_current_import_callbacks",
            import_effects=occurrence["import_effects"],
            outcomes=dict(
                present="skip_child_initialization_then_continue_fromlist",
                attribute_error="attempt_child_import_with_current_registry_loader_and_missing_submodule_handling",
                other_exception="propagate_at_this_source_check",
            ),
            import_result="retain_actual_IMPORT_NAME_result_or_propagate_exception_before_binding",
            subsequent_bindings="separate_IMPORT_FROM_lookups_on_returned_object_in_source_order",
            bindings=occurrence["bindings"],
        )
        if change := self.project.imports.namespace_change(
            self.project.imports.registry_key(module)
        ):
            evidence["last_namespace_change"] = change
        if dynamic:
            return evidence
        self.fail(
            node,
            (
                "import_from_hook"
                if hook is not None and first["graph_binding"] != "bound"
                else "module_attribute_state"
            ),
            f"the package fromlist lookup of {module}.{first['member']} has no proven "
            "outcome after the original import callbacks",
            source_evidence=evidence,
        )

    def imported(self, node, steps):
        req = node.get("request", {})
        occurrence = self.project.source_imports.get(self.name, node)
        primary = req.get("form") != "from" or req.get("binding_index", 0) == 0
        if not primary and occurrence["activation"] != "visited":
            self.fail(
                node,
                "import_binding_sequence",
                "IMPORT_FROM requires its preceding source module request",
            )
        occurrence["activation"] = "visited"
        mod = node["module"]
        if (
            self.initializing
            and self.project.imports.open_namespace
            and req.get("form") == "import"
        ):
            if namespace_operations.publish_plain_import(self, node, occurrence, steps):
                return
            self.fail(
                node,
                "import_publication_evidence",
                f"import of {mod} needs an identified source IMPORT_NAME result and matching module store",
                source_evidence=dict(
                    occurrence=occurrence["id"],
                    request=occurrence["request"],
                    bindings=occurrence["bindings"],
                    last_change=self.project.imports.open_namespace,
                ),
            )
        if self.dynamic and not self.protected_normal_context():
            self.fail(
                node,
                "conditional_import",
                f"runtime-dependent initialization of {mod} requires an import-state contract",
            )
        if not self.initializing:
            self.fail(
                node,
                "import_scope",
                f"function-local import of {mod} needs an activation-time initialization and binding contract",
            )
        if req.get("form") not in {"from", "import"} or node.get("member") == "*":
            self.fail(
                node,
                "import_form",
                f"import of {mod} needs an explicit request form and exported binding",
            )
        if not req.get("level") and "module" in req and req["module"] != mod:
            self.fail(
                node,
                "import_identity",
                f"absolute request for {req['module']} points to a different module {mod}",
            )
        origin = self.import_origin(node)
        if self.project.imports.open_namespace:
            self.fail(
                node,
                "import_callable_after_callback",
                f"import of {mod} needs the current __import__ callable, registry and returned bindings",
                source_evidence=dict(
                    request=req, last_change=self.project.imports.open_namespace
                ),
            )
        if origin is not None:
            before = self.project.imports.snapshot()
            self.project.array_state.invalidate()
            self.project.sequences.invalidate()
            if req.get("level"):
                self.fail(
                    node,
                    "import_parent",
                    f"relative external import of {mod} needs a closed parent package",
                )
            with self.project.source_imports.external_request(
                self, occurrence, node, origin, steps, primary=primary
            ):
                status = origin["status"]
                if status == "verified_stdlib" and host_protocols.supports_import(mod):
                    value = host_protocols.import_provider(self, node, steps)
                    receipt = self.project.providers["stdlib:" + mod.partition(".")[0]]
                    receipt["import_effects"] = node.get("effects", ["UNKNOWN"])
                elif status == "verified_distribution" and mod == "numpy":
                    value = arrays.import_provider(self, node, steps)
                elif status == "verified_distribution" and graphics.handles_import(
                    node
                ):
                    value = self.project.graphics.import_provider(self, node, steps)
                elif host_protocols.literal_exports(self, mod):
                    value = host_protocols.source_provider(self, node, origin, steps)
                else:
                    identity = (
                        f"{origin['distribution']} {origin['version']}"
                        if origin.get("distribution") and origin.get("version")
                        else "the standard library"
                    )
                    self.fail(
                        node,
                        "import_semantics",
                        f"{mod} resolves to {identity} at {origin.get('origin')}. Its initialization, effects and exported bindings have no execution contract.",
                        source_evidence=self._import_evidence(mod),
                    )
            if primary:
                scoped = host_protocols.registry_import(self, node, origin, occurrence)
                if scoped:
                    completed = (
                        {mod: value.facts["namespace_instance"]}
                        if value.facts.get("namespace_instance")
                        else {}
                    )
                    self.project.scoped_registry_effect(
                        self.name,
                        node,
                        scoped,
                        steps,
                        before=before,
                        completed=completed,
                    )
                else:
                    self.project.unknown_registry(self.name, node)
            self.store(node["binding"], value, steps)
            return

        def initialize(symbol):
            self.project.actual_calls.add((self.name, symbol))
            target = self.project.functions.get(symbol, {})
            scope = target.get("python_scope", {})
            if scope.get("kind") != "module":
                self.fail(
                    node, "import_initializer", f"missing module initializer {symbol}"
                )
            if mod != "__main__":
                self.require_import_instance(node, scope["module"])
            name = self.project.imports.registry_key(scope["module"])
            previous = self.project.registry_entry(name, node) or None
            if previous is not None and previous.get("state") == "absent":
                previous = None
            if (
                previous is not None
                and previous.get("state") == "initializing"
                and previous.get("initializer") == symbol
                and symbol in self.project.active
            ):
                # The loader already published this exact parent instance before
                # entering its body. Reusing it does not prove completion.
                self.project.read_registry(name, previous, node)
                return
            if previous is not None and previous.get("state") != "ready":
                self.fail(
                    node,
                    "import_initializer_state",
                    f"an earlier import effect left the registry binding of {name} unresolved",
                    source_evidence=dict(
                        module=name,
                        initializer=symbol,
                        registry_state=previous["state"],
                        last_change=previous,
                    ),
                )
            if previous is not None and previous.get("initializer") != symbol:
                self.fail(
                    node,
                    "import_initializer_identity",
                    f"the current entry for {name} does not match its graph initializer",
                )
            if previous is not None:
                self.project.observe_import(name, "registry", node)
            if previous is None:
                publication = self.project.imports.begin_initializer(
                    name, symbol, self.name, occurrence["id"], values.source_site(node)
                )
                analyzed_body = symbol not in self.project.closed
                fn = self.project.compile(symbol, (), source=node)
                if fn.get("noreturn"):
                    self.fail(
                        node,
                        "import_initializer_completion",
                        f"the initializer for {name} has no proven normal continuation",
                    )
                call = dict(
                    op="call", callee=symbol, args=[], **values.source_site(node)
                )
                self.project.source_imports.producer(occurrence, call, "module_body")
                steps.append(dict(op="expr", value=call))
                # Only the continuation of this emitted call has an initialized
                # module. Merely compiling or finding a cached body proves less.
                still_published = (
                    self.project.imports.registry_entry(name) is publication
                )
                if not still_published:
                    self.project.registry_entry(name, node)
                if not self.project.imports.complete_initializer(
                    name,
                    publication,
                    call,
                    analyzed_body=analyzed_body and still_published,
                ):
                    self.fail(
                        node,
                        "import_initializer_identity",
                        f"{name} did not retain the published instance on initializer return",
                    )
            parent, _, leaf = scope["module"].rpartition(".")
            if parent:
                slot = namespace_slot(parent, leaf)
                value = _Value("module", scope["module"])
                if slot in self.project.names and self.project.names[slot] != value:
                    self.fail(
                        node,
                        "rebind",
                        f"import replaces existing package member {slot}",
                    )
                self.project.names[slot] = value
                self.project.imports.binding_written(slot)

        for symbol in node["initializers"]:
            initialize(symbol)
        target_state = self.project.imports.registry_entry(mod)
        if target_state.get("state") != "ready" and not (
            target_state.get("state") == "initializing"
            and target_state.get("initializer") in self.project.active
        ):
            self.fail(
                node, "import_initializer", f"incomplete or circular import of {mod}"
            )
        if req["form"] == "from" and primary:
            scope = self.project.functions[self.project.initializers[mod]][
                "python_scope"
            ]
            if scope.get("package"):
                self.package_fromlist(node, occurrence)
        elif node.get("fromlist_checks") or node.get("conditional_initializers"):
            self.fail(
                node, "import_guard", "only IMPORT_NAME may carry fromlist prechecks"
            )
        if req["form"] == "import":
            value = _Value("module", mod if node.get("alias") else mod.split(".")[0])
        else:
            if (
                namespace_slot(mod, node["member"]) not in self.project.names
                and namespace_slot(mod, "__getattr__") in self.project.names
            ):
                self.fail(
                    node,
                    "import_from_hook",
                    "IMPORT_FROM needs its lookup failure and ImportError conversion contract",
                )
            value = self.module_member(node, mod, node["member"], steps)
        self.store(node["binding"], value, steps)

    def evaluate(self, node, steps, discard=False, *, truth_context=False):
        self.evaluating.append(node)
        try:
            result = self._evaluate(node, steps, discard, truth_context=truth_context)
            if node.get("site") is not None:
                self.project.normalized_sites.add((self.name, node["site"]))
            return result
        finally:
            self.evaluating.pop()

    def _evaluate(self, node, steps, discard=False, *, truth_context=False):
        op = node["op"]
        if self.project.imports.open_namespace:
            result = namespace_operations.open_object_expression(
                self, node, steps, truth_context=truth_context, discard=discard
            )
            if result is not None:
                return result
        if op == "python_select":
            if node.get("site") is not None:
                self.project.live_sites.add((self.name, node["site"]))
            return expressions.select(
                self, node, steps, discard=discard, truth_context=truth_context
            )
        if op == "assignment_expression":
            value = self.evaluate(node["value"], steps)
            if values.runtime(value.kind) and value.value.get("op") != "literal":
                value = self.effect(value.value, value.kind, steps, facts=value.facts)
            self.store(node["target"], value, steps)
            return self.truth(value, node) if truth_context else value
        if op == "mapping":
            result = mappings.literal(self, node, steps)
            return self.truth(result, node) if truth_context else result
        if op in {"mapping_comprehension", "sequence_comprehension"}:
            result = self.comprehension(node, steps)
            return self.truth(result, node) if truth_context else result
        if (
            op == "compare_chain"
            and len(node.get("operators", ())) == 1
            and node["operators"][0] in {"is", "isnot"}
        ):
            return self._evaluate(
                {
                    **node,
                    "op": "binary",
                    "operator": node["operators"][0],
                    "right": node["operands"][0],
                },
                steps,
                discard,
                truth_context=truth_context,
            )
        if node.get("site") is not None:
            self.project.live_sites.add((self.name, node["site"]))
        if truth_context:
            if op in {"binary", "compare_chain", "short_circuit"} and not (
                op == "binary" and node.get("operator") in {"is", "isnot"}
            ):
                return expressions.evaluate(self, node, steps, truth_context=True)
            return self.truth(self.evaluate(node, steps), node)
        if op == "literal":
            value = node["value"]
            if (
                isinstance(node.get("type"), dict)
                and node["type"].get("class") == "complex"
            ):
                if (
                    not isinstance(value, list)
                    or len(value) != 2
                    or any(
                        type(part) is not float or not math.isfinite(part)
                        for part in value
                    )
                ):
                    self.fail(
                        node,
                        "literal_contract",
                        "complex literal requires two finite binary64 components",
                    )
                return _Value(
                    "python_complex", dict(op="literal", type="c128", value=value)
                )
            if type(value) is int and not -(1 << 63) <= value < 1 << 63:
                return integers.literal(self, node, value, steps=steps)
            if type(value) not in (bool, int, float, str, type(None)):
                self.fail(
                    node,
                    "literal_contract",
                    "literal requires another numeric or object representation",
                )
            if type(value) is str:
                try:
                    value.encode("utf-8")
                except UnicodeEncodeError:
                    self.fail(node, "encoding", "text contains an unpaired surrogate")
            return _literal(value)
        if op in {"var", "python_builtin"}:
            return self.ref(node)
        if op == "binary" and node.get("operator") in {"is", "isnot"}:
            left = self.evaluate(node["left"], steps)
            if values.runtime(left.kind) and left.value.get("op") != "literal":
                left = self.effect(left.value, left.kind, steps, facts=left.facts)
            right = self.evaluate(node["right"], steps)
            result = sequences.identity(node, left, right)
            return (
                result
                if result is not None
                else expressions.apply_binary(
                    self, node, node["operator"], left, right, steps, discard
                )
            )
        if op in {"format_string", "format_field"}:
            return text.evaluate(self, node, steps, discard)
        if op == "aggregate":
            return arrays.capture_sequence(self, node, steps)
        if op == "index":
            base = self.evaluate(node["base"], steps)
            if values.runtime(base.kind):
                base = self.effect(base.value, base.kind, steps, facts=base.facts)
            if base.kind in {"mapping_ref", "structured_value"}:
                if len(node.get("indices", [])) != 1:
                    self.fail(node, "mapping_key", "mapping access needs one text key")
                return mappings.get(
                    self, node, base, self.evaluate(node["indices"][0], steps), steps
                )
            return arrays.index(self, node, steps, discard=discard, evaluated_base=base)
        if op == "process_stream":
            base = self.evaluate(node["base"], steps)
            result = processes.member(self, node, base, node["field"], steps)
            if result is None:
                self.fail(
                    node,
                    "process_receiver",
                    "the process channel has no proven receiver",
                )
            return result
        if op == "member":
            base = self.evaluate(node["base"], steps)
            text_member = text.member(self, node, base, node["field"])
            if text_member is not None:
                return text_member
            mapping_member = mappings.member(self, node, base, node["field"], steps)
            if mapping_member is not None:
                return mapping_member
            if base.kind in {"python_complex", "numpy_c128"} and node["field"] in {
                "real",
                "imag",
            }:
                return self.effect(
                    dict(
                        op="intrinsic",
                        name=node["field"],
                        args=[base.value],
                        type="f64",
                    ),
                    "numpy_f64" if base.kind == "numpy_c128" else "f64",
                    steps,
                )
            if values.object_name(base.kind) is not None:
                return self.project.objects.member(
                    self, node, base, node["field"], steps
                )
            if base.kind == "project_class":
                method = self.project.objects.layouts[base.value].methods.get(
                    node["field"]
                )
                if method:
                    return _Value("function", method)
                self.fail(
                    node,
                    "class_attribute",
                    f"no ordinary method binding for {base.value}.{node['field']}",
                )
            if base.kind == "handle:text_stream" or processes.is_process(base.kind):
                base = self.effect(base.value, base.kind, steps, facts=base.facts)
            process_member = processes.member(self, node, base, node["field"], steps)
            if process_member is not None:
                return process_member
            host_member = host_protocols.member(self, node, base, node["field"], steps)
            if host_member is not None:
                return host_member
            if base.kind == "provider_namespace":
                return arrays.provider_member(self, node, node["field"], steps, base)
            if base.kind == "graphics_namespace":
                return self.project.graphics.member(self, node, base, node["field"])
            if values.is_array(base.kind):
                return arrays.member(self, node, base, node["field"], steps)
            sequence_member = sequences.member(self, node, base, node["field"])
            if sequence_member is not None:
                return sequence_member
            if base.kind == "module":
                return self.module_member(node, base.value, node["field"], steps)
            self.fail(
                node,
                "attribute",
                f"unbound module or dynamic attribute {node['field']}",
            )
        if op in {
            "call",
            "library_call",
            "python_invoke",
            "process_launch",
            "process_wait",
            "process_poll",
            "process_communicate",
        }:
            return self.call(node, steps, discard)
        if op == "unary" and node["operator"] != "not":
            return arrays.scalar_unary(
                self, node, self.evaluate(node["value"], steps), steps, discard
            )
        if op == "unary" and node["operator"] == "not":
            value = self.evaluate(node["value"], steps, truth_context=True)
            if value.value["op"] == "literal":
                return _literal(not value.value["value"])
            return _Value(
                "bool", dict(op="unary", operator="not", type="bool", value=value.value)
            )
        if op in {"binary", "compare_chain", "short_circuit"}:
            return expressions.evaluate(self, node, steps, discard)

        self.fail(node, "operation_contract", f"unclosed expression {op}")

    def call(self, node, steps, discard):
        if not isinstance(node.get("target"), dict):
            self.fail(
                node,
                "call_target",
                "the source call has no captured callable expression",
            )
        target = self.evaluate(node["target"], steps)
        if target.kind == "sequence_method":
            receiver, method = target.value
            target = _Value(
                target.kind,
                (
                    self.effect(
                        receiver.value, receiver.kind, steps, facts=receiver.facts
                    ),
                    method,
                ),
            )
        if target.kind == "project_method":
            method, receiver = target.value
            if receiver.value.get("scope") == "global":
                receiver = self.effect(receiver.value, receiver.kind, steps)
            target = _Value("project_method", (method, receiver))

        def argument(expr):
            value = self.evaluate(expr, steps)
            if values.runtime(value.kind) and value.value["op"] != "literal":
                return self.effect(value.value, value.kind, steps, facts=value.facts)
            return value

        args = [argument(v) for v in node["args"]]
        kwargs = {}
        for item in node.get("keywords", ()):
            key = item["name"]
            if key is None or key in kwargs:
                self.fail(node, "keyword_binding", "dynamic or repeated keyword")
            kwargs[key] = argument(item["value"])
        if target.kind == "builtin" and target.value == "isinstance":
            if (
                kwargs
                or len(args) != 2
                or args[1].kind != "builtin"
                or args[1].value != "dict"
            ):
                self.fail(
                    node,
                    "mapping_isinstance",
                    "isinstance requires the closed builtin dict type",
                )
            return mappings.is_mapping(self, node, args[0])
        mapping_call = mappings.invoke(self, node, target, args, kwargs, steps, discard)
        if mapping_call is not None:
            return mapping_call
        text_call = text.invoke(self, node, target, args, kwargs, steps)
        if text_call is not None:
            return text_call
        if target.kind == "builtin" and target.value == "bool":
            if kwargs or len(args) > 1:
                self.fail(
                    node,
                    "boolean_conversion",
                    "bool accepts zero or one positional value",
                )
            return self.truth(args[0], node) if args else _literal(False)
        sequence_call = sequences.invoke(
            self, node, target, args, kwargs, steps, discard
        )
        if sequence_call is not None:
            return sequence_call
        if target.kind == "project_class":
            return self.project.objects.construct(
                self, node, target, args, kwargs, steps
            )
        if target.kind == "project_method":
            method, receiver = target.value
            return self.invoke_project(
                node, method, [receiver, *args], kwargs, steps, discard
            )
        if target.kind in {
            "array_constructor",
            "array_copy_method",
            "array_reshape_method",
        }:
            return arrays.construct(self, node, target, args, kwargs, steps, discard)
        process_call = processes.invoke(
            self, node, target, args, kwargs, steps, discard
        )
        if process_call is not None:
            return process_call
        if target.kind == "builtin" and target.value in {"min", "max"}:
            return scalars.extreme(self, node, target.value, args, kwargs, steps)
        if target.kind == "builtin" and target.value == "len":
            if not kwargs and len(args) == 1 and args[0].kind == "mapping_ref":
                return mappings.size(self, node, args[0])
            if not kwargs and len(args) == 1 and values.is_sequence(args[0].kind):
                return self.project.sequences.size(args[0])
            if kwargs or len(args) != 1 or not products.is_sequence(args[0]):
                self.fail(node, "length_contract", "len requires one closed sequence")
            return _literal(
                len(values.product_members(args[0].kind))
                if values.is_product(args[0].kind)
                else len(args[0].value)
            )
        if target.kind == "builtin" and target.value == "tuple":
            if (
                kwargs
                or len(args) != 1
                or args[0].kind != "tuple"
                and not values.is_product(args[0].kind)
            ):
                self.fail(
                    node,
                    "product_constructor",
                    "tuple conversion requires an existing immutable tuple",
                )
            return args[0]
        if target.kind == "builtin" and target.value in {"int", "float", "abs"}:
            result = scalars.construct(
                self, node, target.value, args, kwargs, steps=steps
            )
            self.project.decisions.append(
                dict(
                    code=(
                        "numeric_intrinsic"
                        if target.value == "abs"
                        else "numeric_conversion"
                    ),
                    symbol=self.name,
                    location=node.get("location"),
                    constructor=target.value,
                    source_types=[v.kind for v in args],
                    source_facts=[
                        {k: v for k, v in item.facts.items() if k in {"exact", "range"}}
                        for item in args
                    ],
                    result=result.kind,
                    result_facts=result.facts,
                    proof="source numeric protocol and proven representation domain; arguments retain evaluation order",
                )
            )
            return result
        if target.kind == "graphics_callable":
            return self.project.graphics.invoke(
                self, node, target.value, args, kwargs, steps, discard
            )
        if target.kind == "exception_class":
            return exceptions.construct(
                self, node, target, args, kwargs, steps, discard
            )
        if target.kind == "builtin" and target.value == "range":
            if (
                kwargs
                or not 1 <= len(args) <= 3
                or any(v.kind not in {"i64", "integer"} for v in args)
            ):
                self.fail(
                    node,
                    "range_contract",
                    "range requires one to three mathematical integers",
                )
            if len(args) == 3 and (
                type(args[2].exact) is not int or args[2].exact == 0
            ):
                self.fail(node, "range_step", "a nonzero constant step is required")
            return _Value("range", tuple(args))
        if target.kind == "builtin" and target.value == "print":
            args = [text.display(self, node, value, steps) for value in args]
            if any(
                a.kind not in {"bool", "i64", "f64", "numpy_f64", "text", "unit"}
                for a in args
            ):
                self.fail(
                    node,
                    "print_contract",
                    "no text formatting contract for this scalar or aggregate",
                )
            if set(kwargs) - {"sep", "end", "flush", "file"}:
                self.fail(node, "print_contract", "unrecognized print keyword")
            sep, end = (kwargs.get(k, _literal(None)) for k in ("sep", "end"))
            sep = _literal(" ") if sep.kind == "unit" else sep
            end = _literal("\n") if end.kind == "unit" else end
            flush = kwargs.get("flush", _literal(False))
            stream = kwargs.get("file", _literal(None))
            if (
                sep.kind != "text"
                or end.kind != "text"
                or flush.kind != "bool"
                or stream.kind not in {"unit", "handle:text_stream"}
            ):
                self.fail(
                    node,
                    "print_contract",
                    "print requires a text stream, text separators and a Boolean flush flag",
                )
            call = dict(
                op="text_print",
                args=[self.scalar(a, node) for a in args],
                sep=sep.value,
                end=end.value,
                flush=flush.value,
                type="unit",
                effects=["IO", "STATE"],
                encoding="utf-8",
                **values.source_site(node),
                **values.argument_evidence(args, sep=sep, end=end, flush=flush),
                scalar_format=dict(
                    boolean_true="True",
                    boolean_false="False",
                    unit="None",
                    integer_base=10,
                ),
                keyword_order=[
                    "stream" if k == "file" else k
                    for k in kwargs
                    if k != "file" or stream.kind != "unit"
                ],
            )
            if stream.kind != "unit":
                call["stream"] = stream.value
            if any(a.kind in {"f64", "numpy_f64"} for a in args):
                call["scalar_format"]["float"] = "shortest_binary64"
            return self.effect(call, "unit", steps, discard)
        host_call = host_protocols.invoke(
            self, node, target, args, kwargs, steps, discard
        )
        if host_call is not None:
            return host_call
        if target.kind != "function" or target.value not in self.project.functions:
            self.fail(
                node,
                "dispatch_contract",
                f"a callable has no reachable project body ({target.kind}: {target.value})",
            )
        return self.invoke_project(node, target.value, args, kwargs, steps, discard)

    def invoke_project(self, node, name, args, kwargs, steps, discard):
        formals = self.project.functions[name]["params"]
        if len(args) > len(formals):
            self.fail(
                node, "argument_binding", f"too many positional arguments for {name}"
            )
        bound = []
        for i, formal in enumerate(formals):
            param, mode = formal["name"], formal.get("binding")
            if (
                i < len(args)
                and mode in {"posonly", "positional"}
                and param not in kwargs
            ):
                bound.append(args[i])
            elif (
                i >= len(args)
                and mode in {"positional", "keywordonly"}
                and param in kwargs
            ):
                bound.append(kwargs.pop(param))
            elif (
                i >= len(args)
                and param not in kwargs
                and param in self.project.function_defaults.get(name, {})
            ):
                bound.append(self.project.function_defaults[name][param])
            else:
                self.fail(
                    node,
                    "argument_binding",
                    f"missing, repeated or unsupported binding of {name}.{param}",
                )
        if kwargs:
            self.fail(node, "argument_binding", f"unconsumed keywords for {name}")
        for index in self.project.structured_parameters(name):
            if name in self.project.active:
                required = {"owner": name, "parameter": formals[index]["name"]}
                if bound[index].facts.get("mapping_descendant_of") != required:
                    self.fail(
                        node,
                        "recursive_progress",
                        "recursive structured arguments must descend through the same input mapping",
                    )
            bound[index] = mappings.pack(self, node, bound[index])
        operands = [self.scalar(v, node) for v in bound]
        function = self.project.compile(name, tuple(v.kind for v in bound), source=node)
        parameter_types = [param["type"] for param in function["params"]]
        if self.project.sequences.resolve(
            parameter_types, strict=False
        ) != self.project.sequences.resolve(
            [values.wire(v.kind) for v in bound], strict=False
        ):
            self.fail(
                node,
                "argument_types",
                "bound arguments disagree with the closed parameter types",
            )
        result = _Value(
            self.project.result_kinds[name], None, self.project.result_facts[name]
        )
        result_fact = (
            None if function.get("noreturn") else values.scalar_evidence(result)
        )
        self.project.actual_calls.add((self.name, name))
        for domain, ports in effects.call_writes(function, self.project.closed).items():
            state = (
                self.project.array_state
                if domain == "arrays"
                else self.project.sequences
            )
            if ports is None:
                state.invalidate()
            else:
                for i in ports:
                    state.invalidate(bound[i])
        call = dict(
            op="call",
            callee=name,
            args=operands,
            **values.source_site(node),
            **values.argument_evidence(bound),
            **({"result_facts": result_fact} if result_fact is not None else {}),
            type=function["result"],
            parameter_types=parameter_types,
            **({"noreturn": True} if function.get("noreturn") else {}),
        )
        return self.effect(
            call,
            result.kind,
            steps,
            discard,
            facts=result.facts,
        )

    def record_return(self, value):
        self.returns.add(value.kind)
        facts = products.returned_facts(value)
        if self.return_facts is None:
            self.return_facts = facts
        elif not values.identical(self.return_facts, facts):
            self.return_facts = (
                values.join_integer_ranges(
                    [
                        _Value(value.kind, None, self.return_facts),
                        _Value(value.kind, None, facts),
                    ]
                )
                if self.returns == {"integer"}
                else {}
            )

    def truth(self, value, node):
        if value.kind == "integer":
            zero = integers.promote(self, node, _literal(0))
            return integers.apply(
                self, node, "ne", [value, zero], "bool", may_raise=False
            )
        if value.kind in {"mapping_ref", "structured_value"}:
            return mappings.truth(self, node, value)
        if values.is_sequence(value.kind):
            return _Value(
                "bool",
                dict(
                    op="binary",
                    operator="ne",
                    type="bool",
                    left=self.project.sequences.size(value).value,
                    right=_literal(0).value,
                ),
            )
        if products.is_sequence(value):
            return _literal(
                bool(values.product_members(value.kind))
                if values.is_product(value.kind)
                else bool(value.value)
            )
        if values.object_name(value.kind) is not None:
            return _literal(True)
        if values.is_array(value.kind):
            self.fail(
                node,
                "array_truth",
                "an array requires an explicit scalar truth or reduction contract",
            )
        expr = self.scalar(value, node)
        if "exact" in value.facts:
            return _literal(bool(value.exact))
        if value.kind in {"bool", "numpy_bool", "boolean_choice"}:
            return _Value("bool", value.value, value.facts)
        if value.kind == "unit":
            return _literal(False)
        if value.kind not in {"i64", "f64", "numpy_f64", "binary64_choice", "text"}:
            self.fail(
                node, "truth_contract", f"{value.kind} has no closed truth conversion"
            )
        return _Value(
            "bool",
            dict(
                op="binary",
                operator="ne",
                left=expr,
                right=_literal(
                    {
                        "i64": 0,
                        "f64": 0.0,
                        "numpy_f64": 0.0,
                        "binary64_choice": 0.0,
                        "text": "",
                    }[value.kind]
                ).value,
                type="bool",
            ),
        )

    def enter_control(self, kind, node):
        self.control_regions.append(dict(kind=kind, **values.source_site(node)))

    def leave_control(self):
        self.control_regions.pop()

    def protected_normal_context(self):
        return (
            self.dynamic == len(self.control_regions)
            and bool(self.control_regions)
            and all(
                region["kind"] in {"try_body", "try_else"}
                for region in self.control_regions
            )
        )

    def snapshot(self):
        return (
            self.env.copy(),
            self.project.names.copy(),
            self.project.imports.snapshot(),
        )

    def copy_state(self, state):
        return state[0].copy(), state[1].copy(), ImportState.copy_snapshot(state[2])

    def restore(self, state):
        self.env, self.project.names, imports = self.copy_state(state)
        self.project.imports.restore(imports)

    def merge(self, states):
        self.env = _join([state[0] for state in states], self.project.sequences)
        self.project.names = _join(
            [state[1] for state in states], self.project.sequences
        )
        self.project.imports.restore(
            ImportState.join_snapshots([state[2] for state in states])
        )

    def exceptional_snapshot(self, start, after, body):
        result = self.copy_state(start)
        after = self.snapshot() if after is None else after
        may_write = self.project.import_may_write(body)
        imports = self.project.imports.exceptional_prefix(
            start[2], after[2], may_write=may_write
        )
        if (
            may_write
            and not imports["open_namespace"]
            and any(
                operation.get("frame_globals", {}).get("identity")
                == "current_activation_globals"
                for operation in walk_operations(body)
            )
        ):
            # Both endpoints may exclude a throwing arm after an open effect.
            imports["open_namespace"] = {
                "reason": "open_namespace_on_exceptional_prefix"
            }
        return result[0], {} if imports["open_namespace"] else result[1], imports

    def project_final_state(self, path, before, after):
        result = self.copy_state(path)
        imports = self.project.imports.project_final(path[2], before[2], after[2])
        return result[0], {} if imports["open_namespace"] else result[1], imports

    def conditional(self, node, steps):
        check = self.evaluate(node["condition"], steps, truth_context=True)
        if check.value["op"] == "literal":
            self.skip_expression(
                node["else"] if check.value["value"] else node["then"],
                dict(
                    rule="constant_condition",
                    guard_site=node.get("site"),
                    truth=check.value["value"],
                ),
            )
            return self.sequence(
                node["then"] if check.value["value"] else node["else"], steps
            )
        start = self.snapshot()
        arms, exits = [], []
        self.dynamic += 1
        self.enter_control("branch", node)
        try:
            for side in ("then", "else"):
                self.restore(start)
                out = []
                if self.sequence(node[side], out):
                    exits.append(self.snapshot())
                arms.append(out)
        finally:
            self.leave_control()
            self.dynamic -= 1
        steps.append(
            dict(op="if", condition=check.value, then=arms[0], **{"else": arms[1]})
        )
        if exits:
            self.merge(exits)
        return bool(exits)

    def comprehension(self, node, steps):
        """Lower one lexical comprehension region, retaining structured loops."""
        identity = node.get("scope", {}).get("id")
        if not isinstance(identity, str):
            self.fail(
                node,
                "comprehension_scope",
                "comprehension requires a captured lexical scope",
            )
        prefix = "@comprehension/" + identity + "/"

        def rename(value):
            if isinstance(value, list):
                return [rename(item) for item in value]
            if not isinstance(value, dict):
                return value
            result = {key: rename(item) for key, item in value.items()}
            if (
                result.get("op") == "var"
                and result.get("cell", {}).get("owner") == identity
            ):
                result["name"] = prefix + result["name"]
                result.pop("cell", None)
                result.pop("scope", None)
            return result

        try:
            node = rename(node)
            if node["op"] == "mapping_comprehension":
                return mappings.comprehension(
                    self, node, steps, self._comprehension_generators
                )
            return sequences.comprehension(
                self, node, steps, self._comprehension_generators
            )
        finally:
            for name in tuple(self.env):
                if name.startswith(prefix):
                    self.env.pop(name)

    def _comprehension_generators(self, node, steps, make_result, put_entry):
        generators = node.get("generators", [])
        if not generators or any(g.get("asynchronous") for g in generators):
            self.fail(
                node,
                "comprehension_iteration",
                "comprehension requires synchronous generators",
            )
        outer = self.evaluate(generators[0]["iterable"], steps)
        outer_iteration = self.range_loop(
            dict(
                op="for",
                protocol="iterator",
                target=generators[0]["target"],
                body=[],
                **values.source_site(node),
            ),
            steps,
            domain=outer,
        )
        if outer_iteration[-1]:
            self.fail(
                node,
                "comprehension_iteration",
                "comprehension iterator requires an explicit next-value contract",
            )
        result = make_result(steps)

        def generate(level, out):
            generator = generators[level]
            domain = outer if level == 0 else self.evaluate(generator["iterable"], out)
            source_loop = dict(
                op="for",
                protocol="iterator",
                target=generator["target"],
                body=[],
                **values.source_site(node),
            )
            # This is a may-write summary, never an execution schedule. It lets
            # the ordinary loop analysis invalidate facts carried by callbacks,
            # walrus assignments and nested generator clauses before lowering.
            repeated = [
                expression
                for index, clause in enumerate(generators[level:], start=level)
                for expression in (
                    ([clause["iterable"]] if index != level else [])
                    + list(clause.get("filters", ()))
                )
            ]
            repeated.extend(
                node[field] for field in ("key", "value", "element") if field in node
            )
            source_loop["body"] = [dict(op="expr", value=term) for term in repeated]
            start = self.snapshot()
            loop, item, count, empty, _, setup = (
                outer_iteration
                if level == 0
                else self.range_loop(source_loop, out, domain=domain)
            )
            if setup:
                self.fail(
                    node,
                    "comprehension_iteration",
                    "comprehension iterator requires an explicit next-value contract",
                )
            if empty:
                self.skip_expression(
                    source_loop["body"],
                    dict(rule="empty_loop", guard_site=node.get("site")),
                )
                out.append(loop)
                return
            allocation_mark = self.project.array_state.enter_loop(self, source_loop)
            self.widen_loop(source_loop, item, count)
            self.dynamic += 1
            self.enter_control("loop", node)
            try:
                self.store(generator["target"], item, loop["body"])

                def filtered(position, body):
                    filters = generator.get("filters", [])
                    if position == len(filters):
                        if level + 1 < len(generators):
                            generate(level + 1, body)
                        else:
                            put_entry(result, self, body)
                        return
                    predicate = self.evaluate(
                        filters[position], body, truth_context=True
                    )
                    if predicate.exact is False:
                        return
                    if predicate.exact is True:
                        filtered(position + 1, body)
                        return
                    before = self.snapshot()
                    accepted = []
                    filtered(position + 1, accepted)
                    self.merge([before, self.snapshot()])
                    body.append(
                        dict(
                            op="if",
                            condition=predicate.value,
                            then=accepted,
                            **{"else": []},
                        )
                    )

                filtered(0, loop["body"])
                after = self.snapshot()
            finally:
                self.leave_control()
                self.dynamic -= 1
                self.project.array_state.leave_loop(allocation_mark)
            out.append(loop)
            self.merge([start, after])

        generate(0, steps)
        return result

    def range_loop(self, node, steps, *, domain=None):
        domain = self.evaluate(node["range"], steps) if domain is None else domain
        if domain.kind == "mapping_items_ref" and node.get("protocol") == "iterator":
            return mappings.iteration(self, node, domain, steps)
        if (
            values.is_sequence(domain.kind)
            and node.get("protocol") == "iterator"
            and node["target"].get("op") == "var"
        ):
            return sequences.iteration(self, node, domain, steps)
        if (
            domain.kind == "handle:text_stream"
            and node.get("protocol") == "iterator"
            and node["target"].get("op") == "var"
        ):
            stream = self.effect(domain.value, domain.kind, steps, facts=domain.facts)
            setup = []
            line = processes.next_line(self, stream, setup)
            condition = dict(
                op="binary",
                operator="ne",
                type="bool",
                left=line.value,
                right=_literal("").value,
            )
            return (
                dict(op="while", body=[], condition=condition),
                line,
                None,
                False,
                False,
                setup,
            )
        if (
            node.get("protocol") != "iterator"
            or domain.kind != "range"
            or node["target"].get("op") != "var"
        ):
            self.fail(
                node,
                "iterator_contract",
                "counted iteration requires a closed range and one lexical target",
            )
        args = domain.value
        start, stop = (_literal(0), args[0]) if len(args) == 1 else args[:2]
        step = args[2] if len(args) == 3 else _literal(1)
        stride = step.exact
        if type(stride) is not int or not stride:
            self.fail(node, "range_step", "a nonzero constant step is required")
        if any(value.kind == "integer" for value in (start, stop, step)):
            return self.wide_range_loop(node, start, stop, step, steps)
        # Range operands are already captured in source order by call().
        positive = stride > 0
        al, ah = start.bounds
        bl, bh = stop.bounds
        empty = al >= bh if positive else ah <= bl
        nonempty = ah < bl if positive else al > bh
        distance = bh - al if positive else ah - bl
        count = max(0, (distance + abs(stride) - 1) // abs(stride))
        low, high = (al, bh - 1) if positive else (bl + 1, ah)
        cursor = self.temporary()
        iv = _Value("i64", None, {"range": (low, high)}).at(cursor)
        condition = dict(
            op="binary",
            operator="lt" if positive else "gt",
            type="bool",
            left=iv.value,
            right=stop.value,
        )
        advance = dict(
            op="binary", operator="add", type="i64", left=iv.value, right=step.value
        )
        if not empty and (high + stride > (1 << 63) - 1 or low + stride < -(1 << 63)):
            # The terminal mathematical index may exceed int64. Substituting
            # the exclusive end when that happens leaves precisely the same visits.
            limit = (1 << 63) - 1 - stride if positive else -(1 << 63) - stride
            guard = dict(
                op="binary",
                operator="gt" if positive else "lt",
                left=iv.value,
                right=_literal(limit).value,
                type="bool",
            )
            advance = dict(
                op="select",
                condition=guard,
                then=stop.value,
                **{"else": advance},
                type="i64",
            )
        loop = dict(
            op="for",
            init=[dict(op="decl", name=cursor, type="i64", init=start.value)],
            condition=condition,
            step=dict(
                op="update", operator="set", target=iv.value, value=advance, type="i64"
            ),
            body=[],
        )
        return loop, iv, count, empty, nonempty, []

    def widen_integer_binding(self, target, steps):
        """Give a widened source binding a new slot; earlier ports stay typed."""
        name = target["name"]
        global_ = target.get("scope") == "global"
        scope = self.project.names if global_ else self.env
        previous = scope.get(name)
        if previous is None or previous.kind != "i64":
            return
        converted = integers.promote(self, target, previous)
        storage = self.temporary()
        types = self.project.globals if global_ else self.locals
        types[storage] = "integer"
        current = converted.at(
            storage, binding_slot=name, **({"scope": "global"} if global_ else {})
        )
        steps.append(dict(op="assign", target=current.value, value=converted.value))
        scope[name] = current

    def wide_range_loop(self, node, start, stop, step, steps):
        """A lazy range needs a mathematical cursor, never a materialized length."""
        stride = step.exact
        positive = stride > 0
        a, b = start.bounds, stop.bounds
        empty = nonempty = False
        count, facts = None, {}
        if a is not None and b is not None:
            al, ah = a
            bl, bh = b
            empty = al >= bh if positive else ah <= bl
            nonempty = ah < bl if positive else al > bh
            distance = bh - al if positive else ah - bl
            count = max(0, (distance + abs(stride) - 1) // abs(stride))
            if not empty:
                facts["range"] = (al, bh - 1) if positive else (bl + 1, ah)
        first, last, stride_value = [
            integers.promote(self, node, value, steps=steps)
            for value in (start, stop, step)
        ]
        if not empty:
            self.widen_integer_binding(node["target"], steps)
        cursor = self.temporary()
        iv = _Value("integer", None, facts).at(cursor)
        condition = integers.apply(
            self, node, "lt" if positive else "gt", [iv, last], "bool", may_raise=False
        )
        advance = integers.apply(self, node, "add", [iv, stride_value])
        loop = dict(
            op="for",
            init=[dict(op="decl", name=cursor, type="integer", init=first.value)],
            condition=condition.value,
            step=dict(
                op="update",
                operator="set",
                target=iv.value,
                value=advance.value,
                type="integer",
            ),
            body=[],
        )
        self.project.decisions.append(
            dict(
                code="mathematical_range_iteration",
                symbol=self.name,
                cursor="integer",
                constant_step=stride,
                length_materialized=False,
                bound_status="proven" if a is not None and b is not None else "unknown",
                proof="captured immutable arguments; sign-directed exclusive bound; private mathematical cursor; source target is assigned only on visits",
                **values.source_site(node),
            )
        )
        return loop, iv, count, empty, nonempty, []

    def widen_loop(self, node, iv, count):
        self.project.sequences.invalidate_contents()
        body = node["body"]
        writes = {}
        nested = False
        for op in walk_operations(body):
            if op["op"] == "exception_handler" and (count is None or count > 1):
                alias = op.get("binding")
                if alias and alias.get("op") == "var":
                    env = (
                        self.project.names
                        if alias.get("scope") == "global"
                        else self.env
                    )
                    env.pop(alias["name"], None)
            if op["op"] in {"for", "while"}:
                nested = True
            for dest in values.written_targets(op):
                if dest.get("op") == "var":
                    writes.setdefault(
                        (dest.get("scope") == "global", dest["name"]), []
                    ).append(op)
        if iv is not None and node["target"].get("op") == "var":
            target = node["target"]
            index = (target.get("scope") == "global", target["name"])
            writes.setdefault(index, [])
        else:
            index = None

        def delta(port):
            if port.get("op") == "literal" and type(port.get("value")) is int:
                return (port["value"],) * 2
            if port.get("op") == "var":
                key = port.get("scope") == "global", port["name"]
                if key == index and not writes[key]:
                    return iv.bounds
                binding = (self.project.names if key[0] else self.env).get(key[1])
                if key not in writes and binding and binding.kind == "i64":
                    return binding.bounds
            return None

        for (global_, name), sites in writes.items():
            env = self.project.names if global_ else self.env
            value = env.get(name)
            if value is None:
                continue
            if not values.runtime(value.kind):
                self.fail(
                    node,
                    "loop_binding",
                    f"loop-carried symbolic binding {name} needs a runtime representation",
                )
            facts = {}
            if value.kind == "i64" and count and not nested and sites:
                down = up = 0
                for site in sites:
                    rhs = site.get("value", {})
                    op = site.get("operator")
                    if (
                        site["op"] == "assign"
                        and site["target"].get("op") == "var"
                        and rhs.get("op") == "binary"
                        and rhs["left"].get("op") == "var"
                        and rhs["left"]["name"] == name
                        and (rhs["left"].get("scope") == "global") == global_
                    ):
                        op, rhs = rhs["operator"], rhs["right"]
                    elif site["op"] != "update":
                        break
                    interval = delta(rhs)
                    if op not in {"add", "sub"} or interval is None:
                        break
                    a, b = interval
                    if op == "sub":
                        a, b = -b, -a
                    down += min(a, 0)
                    up += max(b, 0)
                else:
                    lower = value.bounds[0] + (count - 1) * down
                    upper = value.bounds[1] + (count - 1) * up
                    if -(1 << 63) <= lower <= upper < 1 << 63:
                        facts["range"] = (lower, upper)
            env[name] = _Value(value.kind, value.value, facts)

    def refine_loop_test(self, test):
        if test.get("op") != "binary":
            return
        op = test["operator"]
        left, right = test["left"], test["right"]
        if right.get("op") == "var" and left.get("op") == "literal":
            left, right = right, left
            op = {"gt": "lt", "ge": "le", "lt": "gt", "le": "ge", "eq": "eq"}.get(op)
        if (
            left.get("op") != "var"
            or right.get("op") != "literal"
            or type(right["value"]) is not int
        ):
            return
        env = self.project.names if left.get("scope") == "global" else self.env
        v = env.get(left["name"])
        if v is None or v.kind != "i64":
            return
        low, high = v.bounds
        b = right["value"]
        if op in {"le", "lt", "eq"}:
            high = min(high, b - (op == "lt"))
        if op in {"ge", "gt", "eq"}:
            low = max(low, b + (op == "gt"))
        if low <= high:
            env[left["name"]] = _Value("i64", v.value, {"range": (low, high)})

    def loop(self, node, steps):
        counted = node["op"] == "for"
        if counted:
            loop, iv, count, empty, nonempty, test_setup = self.range_loop(node, steps)
            if empty and not test_setup:
                # Iterator evaluation is complete, but no target binding or
                # body/back-edge effect occurs. The else suite still executes.
                self.skip_expression(
                    node["body"], dict(rule="empty_loop", guard_site=node.get("site"))
                )
                steps.append(loop)
                return self.sequence(node.get("else", []), steps)
        else:
            iv, count = None, None
            loop = dict(op="while", body=[])
            test_setup = []
        allocation_mark = self.project.array_state.enter_loop(self, node)
        start = self.snapshot()
        self.widen_loop(node, iv, count)
        header = self.snapshot()
        flag = self.temporary() if node.get("else") else None
        if counted:
            condition = loop["condition"]
        else:
            test = self.evaluate(node["condition"], test_setup, truth_context=True)
            empty, nonempty = test.exact is False, test.exact is True
            condition = test.value
            self.refine_loop_test(condition)
        loop["condition"] = condition
        if test_setup or flag:
            # Mark exhaustion at the condition edge. A pending break can be
            # replaced by a continue or a return inside source-level finally.
            finish = []
            if flag:
                steps.append(
                    dict(op="decl", name=flag, type="bool", init=_literal(False).value)
                )
                finish.append(
                    dict(
                        op="assign",
                        target=_Value("bool", None).at(flag).value,
                        value=_literal(True).value,
                    )
                )
            loop["condition"] = _literal(True).value
            loop["body"].extend(test_setup)
            loop["body"].append(
                dict(
                    op="if",
                    condition=dict(
                        op="unary", operator="not", value=condition, type="bool"
                    ),
                    then=[*finish, dict(op="break")],
                    **{"else": []},
                )
            )
        flow = {"break": [], "continue": []}
        self.loops.append(flow)
        self.dynamic += 1
        self.enter_control("loop", node)
        try:
            if not empty:
                if counted:
                    self.store(node["target"], iv, loop["body"])
                if self.sequence(node["body"], loop["body"]):
                    flow["continue"].append(self.snapshot())
            else:
                self.skip_expression(
                    node["body"], dict(rule="empty_loop", guard_site=node.get("site"))
                )
        finally:
            self.leave_control()
            self.dynamic -= 1
            self.loops.pop()
        broke = flow["break"]
        steps.append(loop)
        if empty:
            normal = [start]
        elif counted:
            normal = ([start] if not nonempty else []) + flow["continue"]
        else:
            normal = [] if nonempty else [header, *flow["continue"]]
        exits = list(broke)
        if normal:
            self.merge(normal)
            else_body = []
            self.dynamic += bool(broke)
            self.enter_control("loop_else", node)
            try:
                if self.sequence(node.get("else", []), else_body):
                    exits.append(self.snapshot())
            finally:
                self.leave_control()
                self.dynamic -= bool(broke)
            if else_body:
                if broke:
                    steps.append(
                        dict(
                            op="if",
                            condition=dict(op="var", type="bool", name=flag),
                            then=else_body,
                            **{"else": []},
                        )
                    )
                else:
                    steps.extend(else_body)
        if exits:
            self.merge(exits)
        self.project.array_state.leave_loop(allocation_mark)
        return bool(exits)

    def sequence(self, body, steps):
        parent = self.region
        self.region = region = _Region(steps, parent, body)
        if parent is None:
            self.root_region = region
        else:
            parent.children.append(region)
        try:
            falls = self._sequence(body, steps, region)
            region.boundary(None)
            return falls
        finally:
            self.region = parent

    def _sequence(self, body, steps, region):
        def stop():
            self.skip_expression(
                body[index + 1 :],
                dict(
                    rule="no_normal_successor",
                    guard_site=node.get("site"),
                    operator=node["op"],
                ),
            )
            return False

        next_statement = 0
        for index, node in enumerate(body):
            region.boundary(node)
            if node.get("site") is not None:
                self.project.live_sites.add((self.name, node["site"]))
            if index < next_statement:
                # This binding was emitted inside its one source IMPORT_NAME.
                continue
            op = node["op"]
            first = len(steps)
            if (
                op == "decl"
                and node.get("initial_binding") == "unbound"
                or op == "pass"
            ):
                continue
            if op == "expr":
                self.evaluate(node["value"], steps, discard=True)
            elif op == "assign":
                if not (
                    namespace_operations.store_open_module_constant(self, node, steps)
                    or namespace_operations.open_object_statement(self, node, steps)
                ):
                    self.store(
                        node["target"], self.evaluate(node["value"], steps), steps
                    )
            elif op == "python_import":
                group = namespace_operations.global_fromlist(self, body, index)
                if group is None:
                    self.imported(node, steps)
                else:
                    occurrence, bindings = group
                    self.import_origin(node)
                    occurrence["activation"] = "visited"
                    evidence = self.package_fromlist(node, occurrence, dynamic=True)
                    namespace_operations.namespace_effect(
                        self,
                        node,
                        steps,
                        "module_fromlist_publish",
                        source_protocol="python.import",
                        occurrence=occurrence["id"],
                        request=occurrence["request"],
                        resolved_module=node["module"],
                        bindings=bindings,
                        fromlist_checks=evidence["ordered_checks"],
                        initializers=node["initializers"],
                        source_effects=occurrence["source_effects"],
                        dispatch="call_current_frame_importer_once_with_original_arguments",
                        initialization="only_as_invoked_by_actual_importer_including_ordered_fromlist_prechecks",
                        lookup="successive_IMPORT_FROM_on_retained_actual_return_including_circular_registry_fallback",
                        publication="STORE_NAME_after_each_successful_lookup_in_current_frame_globals",
                        release="preserve_replaced_values_operand_stack_unwind_and_final_import_result_release",
                        callbacks="preserve_source_opcode_tracing_reentry_and_unraisable_finalizer_errors",
                    )
                    for binding in bindings:
                        self.project.imports.publications[binding["slot"]] = dict(
                            occurrence=occurrence["id"], **binding
                        )
                    next_statement = index + len(bindings)
            elif op == "unbind" and self.project.imports.open_namespace:
                if not namespace_operations.open_object_statement(self, node, steps):
                    self.fail(
                        node,
                        "source_delete_evidence",
                        "deletion in the open namespace requires each source mapping, name and ordered DELETE instruction",
                    )
            elif op == "python_define_class":
                self.store(
                    node["target"],
                    self.project.objects.define(self, node, steps),
                    steps,
                )
            elif op == "python_define":
                if not objects.plain_definition(node):
                    defined = objects.bind_function_definition(self, node, steps)
                else:
                    defined = _Value("function", node["symbol"])
                self.store(node["target"], defined, steps)
            elif op == "raise":
                exceptions.throw(self, node, steps)
                return stop()
            elif op == "try":
                if not exceptions.protected(self, node, steps):
                    return stop()
            elif op == "with_context":
                if not exceptions.resources(self, node, steps):
                    return stop()
            elif op == "return":
                value = (
                    self.evaluate(node["value"], steps)
                    if "value" in node
                    else _literal(None)
                )
                if self.project.structured_parameters(self.name):
                    value = mappings.pack(self, node, value)
                steps.append(dict(op="return", value=self.scalar(value, node)))
                self.record_return(value)
                return stop()
            elif op == "update" and node["target"].get("op") in {"var", "member"}:
                target = node["target"]
                receiver = None
                if target["op"] == "member":
                    receiver = self.evaluate(target["base"], steps)
                    receiver = self.effect(
                        self.scalar(receiver, target), receiver.kind, steps
                    )
                    left = self.project.objects.member(
                        self, target, receiver, target["field"], steps
                    )
                else:
                    left = self.ref(target)
                left = self.effect(left.value, left.kind, steps, facts=left.facts)
                right = self.evaluate(node["value"], steps)
                if arrays.update(self, node, left, right, steps):
                    result = left
                else:
                    result = expressions.apply_binary(
                        self, node, node["operator"], left, right, steps
                    )
                if receiver is not None:
                    self.project.objects.write(
                        self, target, receiver, target["field"], result, steps
                    )
                else:
                    self.store(target, result, steps)
            elif op in {"break", "continue"} and self.loops:
                flow = self.loops[-1]
                flow[op].append(self.snapshot())
                steps.append(dict(op=op))
                return stop()
            elif op == "if":
                if not self.conditional(node, steps):
                    return stop()
            elif op in {"for", "while"}:
                if not self.loop(node, steps):
                    return stop()
            else:
                self.fail(node, "statement_contract", f"unclosed statement {op}")
            for statement in steps[first:]:
                if statement["op"] in {"decl", "expr", "assign"}:
                    value = (
                        statement.get("init")
                        if statement["op"] == "decl"
                        else statement.get("value")
                    )
                    if value and value.get("noreturn"):
                        return stop()
        return True

    def prefix_fragment(self):
        region = self.root_region
        record = region.fragment() if region else dict(body=[])
        record.update(
            completion="prefix",
            symbol=self.name,
            params=self.params,
            **values.source_site((region.current if region else None) or self.fn),
        )
        if self.name in self.project.activations:
            record["activation"] = self.project.activations[self.name]
        return record

    def lower(self):
        if self.initializing:
            self.project.names[namespace_slot(self.module, "__name__")] = _literal(
                "__main__" if self.scope["main"] else self.module
            )
            self.project.imports.binding_written(
                namespace_slot(self.module, "__name__")
            )
        body = []
        falls = self.sequence(self.fn["body"], body)
        if falls and not self.initializing:
            value = _literal(None)
            if self.project.structured_parameters(self.name):
                value = mappings.pack(self, self.fn, value)
            body.append(dict(op="return", value=value.value))
            self.record_return(value)
        if self.returns and all(values.is_sequence(kind) for kind in self.returns):
            first = next(iter(self.returns))
            for kind in self.returns:
                if not self.project.sequences.compatible(first, kind):
                    self.fail(
                        self.fn,
                        "result_contract",
                        "return paths have heterogeneous sequence element protocols",
                    )
            self.returns = {self.project.sequences.resolve_kind(first)}
        if len(self.returns) > 1 or self.initializing and self.returns:
            self.fail(
                self.fn,
                "result_contract",
                f"incompatible return paths {sorted(self.returns)}",
            )
        result = "void" if self.initializing else next(iter(self.returns), "unit")
        self.project.result_kinds[self.name] = result
        self.project.result_facts[self.name] = self.return_facts or {}
        parameters = {p["name"] for p in self.params}
        declarations = [
            dict(op="decl", name=n, type=values.wire(t))
            for n, t in self.locals.items()
            if n not in parameters
        ]
        return dict(
            name=self.name,
            params=self.params,
            result=values.wire(result),
            location=self.fn.get("location"),
            body=[*declarations, *body],
            **({"noreturn": True} if not falls and not self.returns else {}),
        )


def normalize_protocols(module):
    """Certify each reachable protocol, retaining closed signatures on a partial result."""
    functions = {
        f["name"]: f
        for f in module["functions"]
        if f.get("execution_kind") == "python_protocol"
    }
    if not functions:
        return module, [], []
    project = _Project(
        functions,
        entry_environment=module.get("python_entry_environment"),
        function_definitions=module.get("python_definitions"),
    )
    project.execution_images = module.get("execution_images", [])
    project.text_profiles = module.get("text_profiles", {})
    issue = None
    omitted = set()
    try:
        entry = module["entry"]
        if (
            entry not in functions
            or functions[entry].get("python_scope", {}).get("kind") != "module"
        ):
            raise _Unclosed(
                entry,
                functions.get(entry, {}),
                "project_entry",
                "module entry required for closed Python name binding",
            )
        project.compile(entry, ())
        omitted = set(module.get("python_candidate_only", ())).difference(
            project.closed
        )
        omitted.update(project.objects.class_bodies)
        omitted.update(project.inactive_bodies(module.get("activation_graph")))
        for name, fn in functions.items():
            if name not in project.closed and name not in omitted:
                raise _Unclosed(
                    name,
                    fn,
                    "reachable_protocol",
                    "no proven activation or call signature for this reachable body",
                )
        _, open_obligations = obligations.resolve(
            module.get("protocol_obligations", ()),
            functions=functions,
            closed=project.closed,
            omitted=omitted,
            class_proofs=project.objects.proofs,
            normalized_sites=project.normalized_sites,
            skipped_sites=project.skipped_sites,
            live_sites=project.live_sites,
        )
        if open_obligations:
            obligation = open_obligations[0]
            raise _Unclosed(
                obligation["symbol"],
                {
                    "location": {
                        k: obligation[k]
                        for k in ("file", "line", "column")
                        if k in obligation
                    }
                },
                "frontend_obligation",
                obligation["detail"],
            )
    except _Unclosed as error:
        issue = error.issue
    # Deferred region ownership uses object identity. Resolve those regions
    # before freezing types, which copies the canonical operations.
    project.finish_imports()
    if issue is None:
        try:
            project.sequences.freeze(project)
        except ValueError as error:
            issue = _Unclosed(
                entry, functions[entry], "sequence_element_type", str(error)
            ).issue
    if issue is not None:
        project.sequences.freeze(project, partial=True)
    obligation_analysis, _ = obligations.resolve(
        module.get("protocol_obligations", ()),
        functions=functions,
        closed=project.closed,
        omitted=omitted,
        class_proofs=project.objects.proofs,
        normalized_sites=project.normalized_sites,
        skipped_sites=project.skipped_sites,
        live_sites=project.live_sites,
    )
    module["source_obligations"] = obligation_analysis
    module["source_imports"] = project.source_imports.product(
        project.skipped_sites, omitted
    )
    contracts = [
        dict(
            symbol=name,
            params=[p["type"] for p in fn["params"]],
            result=fn["result"],
        )
        for name, fn in project.closed.items()
    ]
    module["protocol_analysis"] = dict(
        entry=module["entry"],
        closed=contracts,
        single_bind_globals=sorted(project.single_bind_globals),
        status="INCOMPLETE" if issue else "READY",
        source_obligation_groups=obligation_analysis["groups"],
        **(
            {"operator_requirements": project.operator_requirements()}
            if project.required_operators
            else {}
        ),
        **({"first_gap": issue} if issue else {}),
        **(
            {"process_requirements": list(project.process_requests.values())}
            if project.process_requests
            else {}
        ),
        **(
            {"graphics_operations": sorted(project.graphics.operations)}
            if project.graphics.operations
            else {}
        ),
        **(
            {
                "providers": list(project.providers.values()),
                "array_operations": sorted(project.array_ops),
            }
            if project.providers
            else {}
        ),
    )
    if issue:
        module["protocol_analysis"].update(
            closed_fragments=[
                dict(
                    symbol=name,
                    completion="complete",
                    function=fn,
                    **values.source_site(functions[name]),
                    **(
                        {"activation": project.activations[name]}
                        if name in project.activations
                        else {}
                    ),
                )
                for name, fn in project.closed.items()
            ],
            prefix_fragments=list(reversed(project.prefixes)),
        )
        return module, project.decisions, [issue]
    changes = project.decisions + [
        dict(
            **contract,
            code="closed_python_protocol",
            proof="ordered module initialization, reaching lexical bindings and consistent call types",
        )
        for contract in contracts
    ]
    for layout in project.objects.layouts.values():
        methods = set(layout.methods.values())
        changes.append(
            dict(
                code="closed_instance_layout",
                record=layout.name,
                namespace=layout.body_symbol,
                fields=sorted(layout.fields),
                activated_methods=sorted(methods & project.closed.keys()),
                inactive_methods=sorted(methods - project.closed.keys()),
                proof="ordinary class and method bindings close throughout the entry call graph with shared identity, checked field presence and observable writes",
            )
        )
    module["functions"] = [
        project.closed.get(fn["name"], fn)
        for fn in module["functions"]
        if fn["name"] not in omitted
    ]
    module["records"].extend(
        layout.record() for layout in project.objects.layouts.values()
    )
    module["globals"].extend(
        dict(op="decl", name=n, type=values.wire(t)) for n, t in project.globals.items()
    )
    if project.array_ops:
        elements = {
            spec["element"]
            for node in walk_operations(list(project.closed.values()))
            if isinstance(spec := node.get("type"), dict)
            and spec.get("kind") == "ndarray"
        }
        requirement = dict(device="cpu", operations=sorted(project.array_ops))
        if len(elements) == 1:
            requirement["element"] = next(iter(elements))
        elif elements:
            requirement["elements"] = sorted(elements)
        module.setdefault("operator_requirements", {})["ndarray"] = requirement
    if project.process_ops:
        module.setdefault("operator_requirements", {})["process"] = dict(
            operations=sorted(project.process_ops),
            binding=(
                "explicit_targets" if project.process_requests else "declared_images"
            ),
        )
    if project.required_operators:
        module.setdefault("operator_requirements", {}).update(
            project.operator_requirements()
        )
    if project.graphics.operations:
        module.setdefault("operator_requirements", {})[
            "graphics"
        ] = project.graphics.requirement()
    module["externs"] = [
        e for e in module["externs"] if e.get("calling_convention") != "python"
    ]
    declarations = module["declarations"]
    for symbol in omitted:
        declarations.pop(symbol, None)
    module.pop("python_class_candidates", None)
    module.pop("python_candidate_only", None)
    module.pop("python_definitions", None)
    for name, fn in project.closed.items():
        # Keep independent native declarations for the downstream ABI comparison.
        # Only graph-level Python object placeholders acquire the inferred contract.
        variants = [
            d
            for d in declarations.get(name, ())
            if all(
                not (isinstance(t, dict) and t.get("kind") == "python_object")
                for t in (d["result"], *(p["type"] for p in d["params"]))
            )
        ]
        declarations[name] = [
            *variants,
            {k: fn[k] for k in ("name", "params", "result")},
        ]
    module.pop("protocol_obligations", None)
    return project.sequences.resolve(module), changes, []
