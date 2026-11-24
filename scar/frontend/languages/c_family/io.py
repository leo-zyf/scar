"""F2 byte-buffer and process projections of captured public libc C bindings."""

_HEADERS = {
    "sprintf": "stdio.h",
    "snprintf": "stdio.h",
    "strcpy": "string.h",
    "strncpy": "string.h",
    "strcat": "string.h",
    "memcpy": "string.h",
    "memmove": "string.h",
    "system": "stdlib.h",
}
_SIGNED = {"i8", "i16", "i32", "i64"}
_UNSIGNED = {"u8", "u16", "u32", "u64"}


def _unqualified(spec):
    while isinstance(spec, dict) and spec.get("kind") == "const":
        spec = spec["of"]
    return spec


def _pointer(spec, element, readonly=False):
    spec = _unqualified(spec)
    if not isinstance(spec, dict) or spec.get("kind") != "ptr":
        return False
    target = spec.get("to")
    const = isinstance(target, dict) and target.get("kind") == "const"
    return const == readonly and _unqualified(target) == element


def _signature(declaration):
    name = declaration["name"]
    params = [_unqualified(p.get("type")) for p in declaration.get("params", ())]
    result = _unqualified(declaration.get("result"))
    if name in {"sprintf", "snprintf"}:
        count = 3 if name == "snprintf" else 2
        return (
            len(params) == count
            and declaration.get("variadic") is True
            and isinstance(result, str)
            and result in _SIGNED
            and _pointer(params[0], "char")
            and _pointer(params[-1], "char", True)
            and (
                name == "sprintf"
                or isinstance(params[1], str)
                and params[1] in _UNSIGNED
            )
        )
    if declaration.get("variadic"):
        return False
    if name == "system":
        return (
            len(params) == 1
            and isinstance(result, str)
            and result in _SIGNED
            and _pointer(params[0], "char", True)
        )
    element = "void" if name in {"memcpy", "memmove"} else "char"
    count = 3 if name in {"strncpy", "memcpy", "memmove"} else 2
    return (
        len(params) == count
        and _pointer(result, element)
        and _pointer(params[0], element)
        and _pointer(params[1], element, True)
        and (count == 2 or isinstance(params[2], str) and params[2] in _UNSIGNED)
    )


def project(declaration, args, *, c_usr, application_definition=False):
    """Project a conditional library effect without replacing the actual call."""
    name = declaration.get("name")
    if (
        application_definition
        or declaration.get("provider") != "libc"
        or name not in _HEADERS
        or declaration.get("header") != _HEADERS[name]
        or c_usr != "c:@F@" + name
        or not _signature(declaration)
        or len(args) < len(declaration["params"])
        or not declaration.get("variadic")
        and len(args) != len(declaration["params"])
    ):
        return None
    effect = dict(
        semantics_version=1,
        interface="process.system" if name == "system" else "bytes.copy",
        operands={},
        defaults={},
        obligations=[
            "linked_external_binding",
            "call_argument_binding",
            "defined_execution",
        ],
        contract=dict(semantics_version=1, source_interface=name, count_unit="bytes"),
    )
    operands, contract = effect["operands"], effect["contract"]
    if name == "system":
        operands["command"] = ["args", 0]
        contract.update(
            command_read="through_first_nul_when_nonnull",
            null_command="query_command_processor_availability",
            process_effect="external_command_processor_when_nonnull",
            return_value="implementation_status_not_direct_exit_code",
        )
        effect["obligations"].extend(
            [
                "command_nullness_and_terminated_bytes",
                "command_processor_identity",
                "source_environment_and_cwd",
                "external_process_effects_unmodeled",
            ]
        )
        return effect
    operands["dst"] = ["args", 0]
    contract["destination_access"] = "write"
    effect["obligations"].extend(
        ["destination_storage_and_capacity", "source_alias_and_extent"]
    )
    if name in {"sprintf", "snprintf"}:
        effect["interface"] = "bytes.format_into"
        start = 3 if name == "snprintf" else 2
        operands["format"] = ["args", start - 1]
        operands.update(
            {
                f"value_{index - start}": ["args", index]
                for index in range(start, len(args))
            }
        )
        contract.update(
            format_family="c_printf",
            format_read="through_first_nul",
            overlap="forbidden_for_formatted_copy_sources",
            additional_writes="format_percent_n_targets_may_be_written",
        )
        effect["obligations"].extend(
            [
                "format_string_and_variadic_types",
                "format_percent_n_targets",
                "formatted_source_destination_nonoverlap",
                "locale_and_encoding",
            ]
        )
        if name == "snprintf":
            operands["capacity"] = ["args", 1]
            contract.update(
                capacity="explicit_api_byte_limit",
                termination="nul_if_capacity_positive_and_result_nonnegative",
                return_count="required_characters_excluding_nul_on_success",
                write_extent="min_result_capacity_minus_one_plus_nul_if_capacity_positive_on_success",
                zero_capacity="no_destination_write_and_null_destination_allowed",
            )
        else:
            contract.update(
                capacity="not_bounded_by_interface",
                termination="nul_after_formatted_characters_on_success",
                return_count="written_characters_excluding_nul_on_success",
                write_extent="return_count_plus_nul_on_success",
            )
        return effect
    operands["src"] = ["args", 1]
    contract.update(source_access="read", return_value="destination_pointer")
    if name in {"strncpy", "memcpy", "memmove"}:
        operands["count"] = ["args", 2]
    if name == "strcat":
        effect["interface"] = "bytes.concat"
        contract.update(
            destination_access="read_write",
            destination_scan="through_first_nul",
            write_offset="destination_string_length",
            source_scan="through_first_nul",
            write_extent="source_length_plus_nul",
            termination="copies_source_nul_at_append_position",
        )
        effect["obligations"].append("destination_initial_termination")
    elif name == "strcpy":
        contract.update(
            source_scan="through_first_nul",
            write_extent="source_length_plus_nul",
            termination="copies_source_nul",
        )
    elif name == "strncpy":
        contract.update(
            source_scan="up_to_count_or_first_nul",
            write_extent="exact_count",
            termination="not_guaranteed_when_source_length_at_least_count",
            padding="nul_to_count_when_source_shorter",
        )
    else:
        contract.update(
            source_scan="exact_count",
            write_extent="exact_count",
            termination="not_added",
            padding="none",
        )
    contract["overlap"] = (
        "allowed_as_if_temporary_copy" if name == "memmove" else "forbidden"
    )
    if name != "memmove":
        effect["obligations"].append("source_destination_nonoverlap")
    return effect
