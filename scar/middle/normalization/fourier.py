"""One-dimensional complex transforms and their source binding obligations."""

from scar.contract import digest, json_bytes, python_provider_id

from . import host_protocols, namespace_operations
from .values import Value, array_kind, literal, source_site, wire

_ROOT = "numpy"
_NAMESPACE = "numpy.fft"
_IMPLEMENTATION = "numpy.fft._pocketfft"
_SOURCES = {
    _ROOT: (
        "numpy/__init__.py",
        "09295a80660f17925ae23765ce8cbd7ff7ceae968d5f2f89349f1cb74c0b9e11",
    ),
    _NAMESPACE: (
        "numpy/fft/__init__.py",
        "251bd35d9a814b98e076ab2e0da2a5a7eabbfb11427a7e873ae127d5a26137c0",
    ),
    _IMPLEMENTATION: (
        "numpy/fft/_pocketfft.py",
        "c809e89c05cf7b97d75abd73fdda8454403fe67772d4e435aa35bc4ce723eeb6",
    ),
}
_DIRECTIONS = {"fft": "forward", "ifft": "inverse"}
_PROVIDER = {
    "name": "pocketfft",
    "revision": "33ae5dc94c9cdc7f1c78346504a85de87cadaa12",
}
_PROFILE = "complex128-rank1-ortho-v1"


def _source(frame, node, module):
    source = frame.provider_sources.get(module, {})
    if not (
        source.get("status") == "verified_distribution"
        and source.get("distribution", "").lower() == _ROOT
        and source.get("version") == "2.5.5"
        and source.get("module") == module
        and (source.get("member"), source.get("sha256")) == _SOURCES[module]
        and source.get("proof_scope") == "installed_file_with_digest"
        and source.get("origin")
        and source.get("provider_id") == python_provider_id(source)
    ):
        frame.fail(
            node,
            "fourier_source",
            f"{module} needs its verified numerical source profile",
        )
    return source


def _identity(source):
    return {
        key: source.get(key) for key in ("module", "member", "provider_id", "sha256")
    }


def _definition(frame, node, child, name):
    export = child.get("exports", {}).get(name, {})
    candidates = export.get("candidates", ())
    if export.get("unresolved") or len(candidates) != 1:
        frame.fail(node, "fourier_export", f"no unique sealed definition for {name}")
    candidate = candidates[0]
    via = candidate.get("via", ())
    if not (
        export.get("status") == "source_candidates"
        and candidate.get("kind", "definition") == "definition"
        and candidate.get("module") == _IMPLEMENTATION
        and candidate.get("distribution", "").lower() == _ROOT
        and candidate.get("version") == "2.5.5"
        and candidate.get("name") == name
        and (candidate.get("member"), candidate.get("sha256"))
        == _SOURCES[_IMPLEMENTATION]
        and candidate.get("origin")
        and type(candidate.get("line")) is int
        and candidate["line"] > 0
        and isinstance(candidate.get("provider_id"), str)
        and len(candidate["provider_id"]) == 64
        and not candidate.get("conditions")
        and not candidate.get("alternate_paths")
        and len(via) == 2
        and all(step.get("name") == name for step in via)
        and via[0].get("kind") == "from"
        and _identity(child) == _identity(via[0])
        and via[1].get("kind") == "definition"
        and _identity(candidate) == _identity(via[1])
    ):
        frame.fail(node, "fourier_export", f"{name} has a different export provenance")
    return candidate


def _lookup(frame, node, namespace, member, candidate, steps):
    """Preserve the single actual lookup and both continuations at its source site."""
    capture = digest(
        json_bytes([frame.name, source_site(node), frame.temporary(), member])
    )
    common = dict(
        operator_family="module_registry",
        semantics_version=1,
        capture=capture,
        owner=frame.name,
        instance_scope="per_dynamic_lookup_in_current_activation",
        namespace_binding=namespace,
        member=member,
        **source_site(node),
    )
    resolve = dict(
        op="module_attribute_resolve",
        type="bool",
        **common,
        candidate=candidate,
        lookup="IMPORT_FROM" if node.get("op") == "python_import" else "attribute",
        checks=[
            "capture_actual_lookup_once_including_attribute_hooks",
            "qualify_original_namespace_or_callable_and_its_active_implementation",
        ],
        reads=[
            "source.import_binding",
            "lookup.capture",
            "module.namespace",
            "module.registry",
        ],
        writes=["lookup.capture", "import.external_effects"],
        effects=["CTRL", "STATE", "IO", "MEM", "OPAQUE"],
        may_raise=True,
        external_effect="UNKNOWN",
    )
    _guard(
        frame,
        steps,
        resolve,
        common,
        "resume_source_after_lookup_with_actual_object_preserving_remaining_argument_evaluation",
    )
    return dict(
        capture=capture,
        owner=frame.name,
        source_site=node.get("site"),
        normal_edge=(
            "matched_callable"
            if candidate["kind"] == "exported_callable"
            else "matched_namespace"
        ),
        required_family="module_registry",
    )


def _guard(frame, steps, operation, common, continuation):
    fallback = dict(
        op="module_export_continuation_required",
        type="void",
        **common,
        noreturn=True,
        reads=["lookup.capture", "source.continuation"],
        writes=["source.continuation"],
        effects=["CTRL", "OPAQUE"],
        external_effect="UNKNOWN",
        semantics=continuation,
    )
    for item in (operation, fallback):
        namespace_operations.require_registry_operator(frame.project, item["op"])
    matched = frame.effect(operation, "bool", steps)
    steps.append(
        {
            "op": "if",
            "condition": matched.value,
            "then": [],
            "else": [dict(op="expr", value=fallback)],
        }
    )


def member(frame, node, namespace, name, steps):
    if namespace.value == _ROOT and name == "fft":
        root = _source(frame, node, _ROOT)
        child = _source(frame, node, _NAMESPACE)
        instance = namespace.facts.get("source_namespace")
        if not instance or instance.get("source_provider_id") != root["provider_id"]:
            frame.fail(
                node, "fourier_namespace", "the lookup has no captured parent namespace"
            )
        row = host_protocols._initialize_source_namespace(frame, node, instance)
        binding = _lookup(
            frame,
            node,
            dict(occurrence=row["id"], **_identity(root)),
            name,
            dict(
                kind="source_namespace",
                **_identity(child),
                on_missing="execute_source___getattr___and_preserve_cold_or_cached_import_hooks_and_failures",
            ),
            steps,
        )
        return Value("provider_namespace", _NAMESPACE, dict(binding=binding))
    if namespace.value != _NAMESPACE:
        return None
    if name not in _DIRECTIONS:
        frame.fail(
            node, "fourier_export", f"no registered transform for {_NAMESPACE}.{name}"
        )
    root = _source(frame, node, _ROOT)
    child = _source(frame, node, _NAMESPACE)
    definition = _definition(frame, node, child, name)
    profile = dict(
        id="numpy.fft-2.5.5-c2c-v1",
        distribution="numpy",
        version="2.5.5",
        function=f"{_NAMESPACE}.{name}",
        root=_identity(root),
        namespace=_identity(child),
        definition=dict(**_identity(definition), name=name, line=definition["line"]),
    )
    binding = _lookup(
        frame,
        node,
        namespace.facts["binding"],
        name,
        dict(
            kind="exported_callable",
            **profile["definition"],
            dispatch="original_decorated_array_function_for_exact_builtin_ndarray",
        ),
        steps,
    )
    return Value(
        "array_constructor",
        profile["function"],
        dict(
            source_profile=profile,
            binding=binding,
            diagnostic_file=definition["origin"],
        ),
    )


def invoke(frame, node, target, args, keywords, steps, discard):
    profile = target.facts["source_profile"]
    name = profile["definition"]["name"]
    parameters = ("a", "n", "axis", "norm", "out")
    if (
        len(args) > len(parameters)
        or set(keywords) - set(parameters)
        or set(parameters[: len(args)]) & keywords.keys()
    ):
        frame.fail(
            node, "fourier_arguments", f"{name} has invalid or duplicate arguments"
        )
    bound = {**dict(zip(parameters, args)), **keywords}
    array = bound.get("a")
    kind = array_kind(1, "c128")
    if array is None or array.kind != kind:
        frame.fail(
            node,
            "fourier_input",
            "the one-dimensional transform requires an explicit complex128 array",
        )
    if any(bound[key].kind != "unit" for key in ("n", "out") if key in bound):
        frame.fail(
            node,
            "fourier_arguments",
            "transform length and output storage must use their source defaults",
        )
    axis = bound.get("axis", literal(-1))
    if axis.kind != "i64" or axis.exact not in (-1, 0):
        frame.fail(
            node,
            "fourier_axis",
            "the transform axis must resolve to the input's only dimension",
        )
    norm = bound.get("norm")
    if norm is None or norm.kind != "text" or norm.exact != "ortho":
        frame.fail(
            node,
            "fourier_normalization",
            "the numerical profile requires orthonormal scaling",
        )
    common = dict(
        operator_family="module_registry",
        semantics_version=1,
        capture=target.facts["binding"]["capture"],
        owner=frame.name,
        source_binding=target.facts["binding"],
        arguments=[v.value for v in args],
        keywords={k: v.value for k, v in keywords.items()},
        **source_site(node),
    )
    dispatch = dict(
        op="module_callable_matches",
        type="bool",
        **common,
        source_profile=profile["id"],
        reads=[
            "lookup.capture",
            "callable.implementation",
            "module.namespace",
            "input.type",
            "floating_point.error_state",
        ],
        writes=[],
        effects=["CTRL", "STATE"],
        may_raise=False,
        checks=[
            "captured_callable_current_implementation_and_transitive_bindings",
            "exact_builtin_ndarray_no_override_dispatch",
            "registered_numpy_error_and_warning_policy",
        ],
    )
    _guard(
        frame,
        steps,
        dispatch,
        common,
        "call_captured_actual_callable_with_already_evaluated_arguments_and_resume_source",
    )
    frame.project.require_operator(
        "fourier", "array_fft", profile=_PROFILE, required_provider=_PROVIDER
    )
    warning = dict(
        operation=name,
        module=_IMPLEMENTATION,
        location=dict(diagnostic_file=target.facts["diagnostic_file"], line=101),
        source_line="return ufunc(a, fct, axes=[(axis,), (), (axis,)], out=out)",
    )
    operation = dict(
        op="array_fft",
        input=array.value,
        type=wire(kind),
        axis=0,
        direction=_DIRECTIONS[name],
        normalization="ortho",
        operator_family="fourier",
        source_profile=profile,
        source_binding=target.facts["binding"],
        required_provider=_PROVIDER,
        numeric_policy="numpy_default",
        warning=warning,
        effects=["MEM", "VAL", "STATE", "IO"],
        may_raise=True,
        reads=["lookup.capture", "input.elements", "floating_point.error_state"],
        writes=["new.array_storage", "warnings"],
        storage="fresh",
        empty_axis=dict(
            exception="ValueError",
            message="Invalid number of FFT data points (0) specified.",
        ),
        **source_site(node),
    )
    return frame.effect(
        operation,
        kind,
        steps,
        discard,
        facts=frame.project.array_state.create(array.facts.get("shape", (None,))),
    )
