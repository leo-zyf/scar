"""Independent Python capture and normalization entry points."""


def project_units(entry, root):
    from .facts import project_units as discover

    return discover(entry, root)


def capture_project(entry, root, *, buffers=None, source_hashes=None):
    from .facts import capture_project as capture

    return capture(entry, root, buffers=buffers, source_hashes=source_hashes)


def parse_unit(
    path, flags=(), *, root, abi=None, source_bytes=None, source_hashes=None
):
    from .facts import parse_unit as capture

    return capture(
        path,
        flags,
        root=root,
        abi=abi,
        source_bytes=source_bytes,
        source_hashes=source_hashes,
    )


def normalize_unit(parsed):
    from .normalize import normalize_unit as normalize

    return normalize(parsed)
