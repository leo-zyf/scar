"""Independent C-family capture and normalization entry points."""


def parse_unit(
    path,
    flags=(),
    *,
    root,
    language=None,
    source_hashes=None,
    working_directory=None,
    source_operand=None,
    driver_options=None,
):
    from .facts import parse_unit as capture

    return capture(
        path,
        flags,
        root=root,
        language=language,
        source_hashes=source_hashes,
        working_directory=working_directory,
        source_operand=source_operand,
        driver_options=driver_options,
    )


def normalize_unit(parsed):
    from .normalize import normalize_unit as normalize

    return normalize(parsed)
