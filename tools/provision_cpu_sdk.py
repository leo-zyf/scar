"""Install SCAR's pinned CPU array headers. CMAKE_COMMAND may select CMake >= 3.29."""

import argparse
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory
from urllib.request import urlopen

SDK = Path.home() / ".local/share/scar/sdk"
PACKAGES = (
    (
        "xtl",
        "0.8.0",
        "ee38153b7dd0ec84cee3361f5488a4e7e6ddd26392612ac8821cbc76e740273a",
    ),
    (
        "xtensor",
        "0.27.1",
        "117c192ae3b7c37c0156dedaa88038e0599a6b264666c3c6c2553154b500fe23",
    ),
)
RECEIPT = "scar-sdk.json"
REQUIRED = {
    "share/cmake/xtl/xtlConfig.cmake",
    "share/cmake/xtensor/xtensorConfig.cmake",
    "include/xtl/xtl_config.hpp",
    "include/xtensor/containers/xarray.hpp",
    "include/xtensor/containers/xadapt.hpp",
    "include/xtensor/views/xstrided_view.hpp",
}


def checksum(path):
    return sha256(path.read_bytes()).hexdigest()


def archive(cache, name, package):
    path = cache / f"{name}-{package['version']}.tar.gz"
    if path.is_file() and checksum(path) == package["sha256"]:
        return path
    with NamedTemporaryFile(dir=cache, delete=False) as download:
        temporary = Path(download.name)
        try:
            with urlopen(package["url"], timeout=60) as source:
                shutil.copyfileobj(source, download)
            download.flush()
            if checksum(temporary) != package["sha256"]:
                raise ValueError(f"upstream archive checksum mismatch: {name}")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return path


def current(prefix, packages):
    try:
        receipt = json.loads((prefix / RECEIPT).read_bytes())
        files = receipt["files"]
        return (
            receipt["schema"] == "scar.cpu-sdk"
            and receipt["schema_version"] == 1
            and receipt["packages"] == packages
            and receipt["prefix"] == str(prefix)
            and REQUIRED.issubset(files)
            and all(
                not Path(name).is_absolute()
                and ".." not in Path(name).parts
                and checksum(prefix / name) == expected
                for name, expected in files.items()
            )
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def cmake(*args, cwd=None):
    # Keep stdout available for the machine-readable installation receipt.
    local = SDK.parent / "toolchain/cmake-3.31.6/cmake/data/bin/cmake"
    command = os.environ.get("CMAKE_COMMAND") or (local if local.is_file() else "cmake")
    subprocess.run([command, *args], cwd=cwd, check=True, stdout=sys.stderr)


def provision(prefix: Path, cache: Path) -> Path:
    prefix, cache = (
        Path(prefix).expanduser().resolve(),
        Path(cache).expanduser().resolve(),
    )
    packages = {
        name: {
            "version": version,
            "sha256": sha,
            "url": f"https://codeload.github.com/xtensor-stack/{name}/tar.gz/refs/tags/{version}",
        }
        for name, version, sha in PACKAGES
    }
    cache.mkdir(parents=True, exist_ok=True)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with (
        (prefix.parent / f".{prefix.name}.scar-sdk.lock").open("a") as prefix_lock,
        (cache / ".provision.lock").open("a") as cache_lock,
    ):
        fcntl.flock(prefix_lock, fcntl.LOCK_EX)
        fcntl.flock(cache_lock, fcntl.LOCK_EX)
        if not current(prefix, packages):
            if prefix.exists() and any(prefix.iterdir()):
                try:
                    previous = json.loads((prefix / RECEIPT).read_bytes())
                    owned = (
                        previous.get("schema") == "scar.cpu-sdk"
                        and previous.get("packages") == packages
                        and previous.get("prefix") == str(prefix)
                    )
                except (OSError, ValueError):
                    owned = False
                if not owned:
                    raise ValueError(
                        f"refusing to overlay an unowned SDK prefix: {prefix}"
                    )
            with TemporaryDirectory(dir=prefix.parent, prefix=".scar-sdk-") as temp:
                root = Path(temp)
                stage = root / "install"
                for name, p in packages.items():
                    source = root / f"{name}-{p['version']}"
                    cmake("-E", "tar", "xzf", archive(cache, name, p), cwd=root)
                    build = root / f"build-{name}"
                    options = (
                        [f"-Dxtl_DIR={stage / 'share/cmake/xtl'}"]
                        if name == "xtensor"
                        else []
                    )
                    cmake(
                        "-S",
                        source,
                        "-B",
                        build,
                        f"-DCMAKE_INSTALL_PREFIX={prefix}",
                        "-DBUILD_TESTS=OFF",
                        "-DCMAKE_INSTALL_MESSAGE=NEVER",
                        *options,
                    )
                    cmake("--install", build, "--prefix", stage)
                files = {
                    p.relative_to(stage).as_posix(): checksum(p)
                    for p in sorted(stage.rglob("*"))
                    if p.is_file()
                }
                if missing := REQUIRED.difference(files):
                    raise ValueError(
                        f"SDK installation missing files: {sorted(missing)}"
                    )
                (stage / RECEIPT).write_text(
                    json.dumps(
                        {
                            "schema": "scar.cpu-sdk",
                            "schema_version": 1,
                            "prefix": str(prefix),
                            "packages": packages,
                            "files": files,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                if not prefix.exists():
                    stage.rename(prefix)
                else:
                    for relative in (*files, RECEIPT):
                        destination = prefix / relative
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        (stage / relative).replace(destination)
    return prefix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix",
        type=Path,
        default=os.environ.get("SCAR_CPU_SDK") or SDK / "xtensor-0.27.1",
    )
    parser.add_argument("--cache", type=Path, default=SDK / "cache")
    args = parser.parse_args()
    prefix = provision(args.prefix, args.cache)
    installed = json.loads((prefix / RECEIPT).read_bytes())
    print(
        json.dumps(
            {
                "prefix": str(prefix),
                "receipt": str(prefix / RECEIPT),
                "packages": installed["packages"],
            },
            indent=2,
        )
    )
