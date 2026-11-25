"""Independent Fortran capture and normalization entry points."""


def parse_unit(
    path, flags=(), *, root, abi=None, working_directory=None, source_operand=None
):
    from .facts import parse_unit as capture

    return capture(
        path,
        flags,
        root=root,
        abi=abi,
        working_directory=working_directory,
        source_operand=source_operand,
    )


def normalize_unit(parsed):
    from .normalize import normalize_unit as normalize

    return normalize(parsed)
