"""Versioned source interfaces shared by capture and normalization."""

import re

# A capture boundary names a public interface candidate, never a purity proof.
PYTHON_HOST_MODULES = frozenset(
    {
        "enum",
        "glob",
        "importlib",
        "os",
        "os.path",
        "shutil",
        "time",
        "sys",
        "math",
        "subprocess",
    }
)

_PYTHON_SOURCE_INTERFACES = {
    "numpy": ("numpy", "numpy-public-v1", (1, 24), (3, 0)),
    "matplotlib": ("matplotlib", "pyplot-3.11.2", (3, 11, 2), (3, 11, 3)),
    "matplotlib.pyplot": (
        "matplotlib",
        "pyplot-3.11.2",
        (3, 11, 2),
        (3, 11, 3),
    ),
    "mpl_toolkits.mplot3d": (
        "matplotlib",
        "pyplot-3.11.2",
        (3, 11, 2),
        (3, 11, 3),
    ),
}


def python_source_boundary(record):
    """Return an interface candidate, never permission to erase initialization.

    Frontends may stop source expansion at registered source interfaces. Their
    execution contracts, effects and individual operations still require the
    consuming layer's independent qualification.
    """
    if record.get("status") == "verified_stdlib":
        if (
            record.get("module") in PYTHON_HOST_MODULES
            and re.fullmatch(r"cpython 3\.\d+\.\d+", str(record.get("interpreter", "")))
            and re.fullmatch(r"[a-f0-9]{64}", str(record.get("sha256", "")))
        ):
            return "python-host-public-v1"
        return None
    if record.get("status") != "verified_distribution":
        return None
    interface = _PYTHON_SOURCE_INTERFACES.get(record.get("module"))
    if interface is None:
        return None
    distribution, profile, lower, upper = interface
    release = re.fullmatch(
        r"(\d+)\.(\d+)\.(\d+)(?:\.post\d+)?(?:\+[A-Za-z0-9.]+)?",
        str(record.get("version", "")),
    )
    if (
        record.get("distribution", "").lower() == distribution
        and release
        and lower <= tuple(map(int, release.group(1, 2, 3))) < upper
        and re.fullmatch(r"[a-f0-9]{64}", str(record.get("sha256", "")))
    ):
        return profile
    return None
