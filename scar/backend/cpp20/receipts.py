"""Pure SDK and fingerprint validation shared by the B2 artifact consumers."""

import re

from scar.contract import digest, json_bytes

# Both sides of the target package validate the same finite SDK output paths.
SDK_HEADERS = {
    name: f"include/scar/{name}"
    for name in (
        "array.hpp",
        "array_io.hpp",
        "exceptions.hpp",
        "format.hpp",
        "fourier.hpp",
        "host.hpp",
        "integer.hpp",
        "mapping.hpp",
        "sequence.hpp",
        "text.hpp",
    )
}


_GMP_MEMBERS = {
    "include/gmp.h",
    "include/gmpxx.h",
    "lib/libgmp.a",
    "COPYRIGHT",
    "COPYING.LGPLv3",
    "COPYING.GPLv2",
    "COPYING.GPLv3",
}


def gmp_files(binding):
    """Validate a scheduled or sealed GMP SDK without consulting an installation."""
    if (
        not isinstance(binding, dict)
        or binding.get("provider") != "gmp-cpu"
        or type(binding.get("semantics_version")) is not int
        or binding["semantics_version"] != 1
        or binding.get("device") != "cpu"
        or binding.get("profile") != "signed-arbitrary-precision-v1"
        or not isinstance(binding.get("version"), str)
        or re.fullmatch(r"6\.[0-9]+\.[0-9]+", binding["version"]) is None
        or not isinstance(binding.get("files"), dict)
        or binding["files"].keys() != _GMP_MEMBERS
    ):
        raise ValueError("target requires a sealed qualified GMP CPU SDK")
    files = binding["files"]
    for row in files.values():
        if (
            not isinstance(row, dict)
            or not is_digest(row.get("sha256"))
            or type(row.get("size")) is not int
            or row["size"] <= 0
        ):
            raise ValueError("invalid GMP SDK member identity")
    identity = {
        key: binding[key]
        for key in ("provider", "semantics_version", "device", "profile", "version")
    }
    identities = {
        name: {key: row[key] for key in ("sha256", "size")}
        for name, row in files.items()
    }
    identity["files"] = identities
    if binding.get("sdk_sha256") != digest(json_bytes(identity)):
        raise ValueError("GMP SDK member set differs from its qualified identity")
    abi = binding.get("abi", {})
    if (
        not isinstance(abi, dict)
        or not isinstance(abi.get("architecture"), str)
        or not abi["architecture"]
        or type(abi.get("limb_bits")) is not int
        or abi.get("limb_bits") not in {32, 64}
        or type(abi.get("pointer_bits")) is not int
        or abi.get("pointer_bits") not in {32, 64}
        or abi.get("link") != "static"
        or abi.get("api") != "gmp-mpz-c"
    ):
        raise ValueError("GMP SDK lacks its qualified static ABI")
    return identities


_POCKETFFT_REVISION = "33ae5dc94c9cdc7f1c78346504a85de87cadaa12"
_POCKETFFT_FILES = {
    "include/pocketfft_hdronly.h": dict(
        size=111974,
        sha256="50f4ddbe60432325f560c706b6f1c49f9e0949fc7f20ae6499f2299e37d9790c",
    ),
    "LICENSE.md": dict(
        size=1498,
        sha256="a85ca13fdf90160b64a0698215868c13b74d835ad0a4e2ba44713b8c5058a056",
    ),
}


def pocketfft_files(binding):
    """Validate the fixed native Fourier SDK independently of its installation path."""
    if (
        not isinstance(binding, dict)
        or binding.get("provider") != "pocketfft-cpu"
        or type(binding.get("semantics_version")) is not int
        or binding["semantics_version"] != 1
        or binding.get("device") != "cpu"
        or binding.get("revision") != _POCKETFFT_REVISION
        or binding.get("profile") != "c2c-c128-rank1-ortho"
        or not isinstance(binding.get("files"), dict)
        or binding["files"].keys() != _POCKETFFT_FILES.keys()
        or any(
            not isinstance(binding["files"][name], dict)
            or any(
                binding["files"][name].get(key) != value
                for key, value in expected.items()
            )
            for name, expected in _POCKETFFT_FILES.items()
        )
    ):
        raise ValueError("target requires the fixed qualified PocketFFT SDK")
    return _POCKETFFT_FILES


def xtensor_headers(receipt):
    if (
        receipt.get("schema") != "scar.cpu-sdk"
        or receipt.get("schema_version") != 1
        or {k: v["version"] for k, v in receipt.get("packages", {}).items()}
        != {"xtensor": "0.27.1", "xtl": "0.8.0"}
    ):
        raise ValueError("incompatible installed operator SDK receipt")
    headers = {}
    for name, sha in receipt["files"].items():
        if not name.startswith("include/"):
            continue
        if not re.fullmatch(
            r"include/(?:xtensor\.hpp|(?:xtensor|xtl)/(?:[\w-]+/)*[\w-]+\.hpp)", name
        ) or not re.fullmatch("[a-f0-9]{64}", sha):
            raise ValueError(f"invalid xtensor CPU SDK header {name!r}")
        headers[name] = sha
    if (
        not {
            "include/xtensor/containers/xarray.hpp",
            "include/xtensor/containers/xadapt.hpp",
            "include/xtl/xtl_config.hpp",
        }
        <= headers.keys()
    ):
        raise ValueError("incomplete xtensor CPU SDK header closure")
    return headers


def validate_mpi_sdk(receipt):
    if (
        not isinstance(receipt, dict)
        or receipt.get("provider") != "mpi-c"
        or receipt.get("api") != "c"
        or receipt.get("semantics_version") != 1
        or not re.fullmatch(r"[0-9]+\.[0-9]+", str(receipt.get("standard_version")))
        or not all(
            isinstance(receipt.get(k), list) and receipt[k]
            for k in ("libraries", "headers")
        )
    ):
        raise ValueError("target requires a qualified MPI C SDK receipt")
    for item in (*receipt["headers"], *receipt["libraries"]):
        if not isinstance(item, dict) or not re.fullmatch(
            r"[a-f0-9]{64}", str(item.get("sha256"))
        ):
            raise ValueError("invalid scheduled MPI SDK fingerprint")
    if not any(h.get("relative") == "mpi.h" for h in receipt["headers"]):
        raise ValueError("MPI SDK receipt omits the public header")
    for header in receipt["headers"]:
        index, relative = header.get("root_index"), header.get("relative")
        if (
            type(index) is not int
            or index < 0
            or not isinstance(relative, str)
            or not re.fullmatch(r"[A-Za-z0-9_+.-]+(?:/[A-Za-z0-9_+.-]+)*", relative)
            or any(part in {".", ".."} for part in relative.split("/"))
        ):
            raise ValueError("invalid scheduled MPI SDK header location")


def is_digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None
