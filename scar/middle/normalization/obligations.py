"""M2 source obligations, resolved by typed operations and control evidence."""

from collections import Counter

from scar.contract import walk_operations

RULES = {
    "scope_binding": ("F1_CAPTURE", "lexical_owner"),
    "protocol": ("F2_NORMALIZE", "source_construct"),
    "import_binding": ("M1_BIND", "import_target"),
    "dispatch": ("M2_NORMALIZE", "call_target"),
    "native_binding": ("M2_NORMALIZE", "canonical_body"),
    "provider_binding": ("M2_NORMALIZE", "logical_library_interface"),
    "class_protocol": ("M2_NORMALIZE", "class_creation"),
    "process_command": ("M2_NORMALIZE", "process_arguments"),
}


def _position(symbol, location):
    return symbol, *(location.get(k) for k in ("file", "line", "column"))


def resolve(
    obligations,
    *,
    functions,
    closed,
    omitted,
    class_proofs,
    normalized_sites,
    skipped_sites,
    live_sites,
):
    """Return an auditable partition; a prefix is never a completed function."""
    positions = {}
    for symbol, function in functions.items():
        for node in walk_operations(function["body"]):
            if node.get("site") is not None:
                positions.setdefault(
                    _position(symbol, node.get("location", {})), []
                ).append((symbol, node["site"]))
    records, unresolved = [], []
    for index, obligation in enumerate(obligations):
        symbol, field = obligation.get("symbol"), obligation.get("field")
        layer, rule = RULES.get(
            field, ("M2_NORMALIZE", "unregistered_source_obligation")
        )
        sites = positions.get(_position(symbol, obligation), [])
        proof = None
        status = "OPEN"
        if symbol in omitted:
            status, proof = "PROVEN_INACTIVE", {"rule": "closed_activation_graph"}
        elif (
            field == "class_protocol" and _position(symbol, obligation) in class_proofs
        ):
            status, proof = "M2_RESOLVED", {"rule": "qualified_class_creation"}
        elif (
            field != "native_binding"
            and sites
            and all(site in skipped_sites and site not in live_sites for site in sites)
        ):
            status, proof = "PROVEN_INACTIVE", {
                "rule": "unselected_control_region",
                "guards": [skipped_sites[site] for site in sites],
            }
        elif field == "native_binding" and symbol in closed:
            status, proof = "DEFERRED_TO_B1", {
                "rule": "closed_canonical_body",
                "symbol": symbol,
            }
        elif (
            field in {"dispatch", "provider_binding"}
            and symbol in closed
            and any(site in normalized_sites for site in sites)
        ):
            status = "M2_RESOLVED" if field == "dispatch" else "DEFERRED_TO_B1"
            proof = {
                "rule": "normalized_source_operation",
                "sites": [site[1] for site in sites if site in normalized_sites],
            }
        if status == "OPEN":
            unresolved.append(obligation)
        records.append(
            {
                "obligation": index,
                "layer": layer,
                "rule": rule,
                "status": status,
                **({"proof": proof} if proof is not None else {}),
            }
        )
    counts = Counter((r["layer"], r["rule"], r["status"]) for r in records)
    return {
        "semantics_version": 1,
        "scope": "source_normalization_only",
        "native_delivery_proven": False,
        "source_obligations": [
            {
                key: value
                for key, value in obligation.items()
                if key not in {"source_line", "diagnostic_file"}
            }
            for obligation in obligations
        ],
        "records": records,
        "groups": [
            dict(layer=layer, rule=rule, status=status, sites=count)
            for (layer, rule, status), count in sorted(counts.items())
        ],
    }, unresolved
