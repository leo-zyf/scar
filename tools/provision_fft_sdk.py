"""Explicitly install the pinned standalone PocketFFT C++ header and its license."""

import argparse
import fcntl
from hashlib import sha256
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

REVISION = "33ae5dc94c9cdc7f1c78346504a85de87cadaa12"
FILES = {
    "include/pocketfft_hdronly.h": "50f4ddbe60432325f560c706b6f1c49f9e0949fc7f20ae6499f2299e37d9790c",
    "LICENSE.md": "a85ca13fdf90160b64a0698215868c13b74d835ad0a4e2ba44713b8c5058a056",
}


def provision(prefix):
    prefix = prefix.expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with (prefix.parent / f".{prefix.name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if prefix.exists():
            if all(
                (prefix / n).is_file()
                and (prefix / n).resolve().is_relative_to(prefix)
                and sha256((prefix / n).read_bytes()).hexdigest() == sha
                for n, sha in FILES.items()
            ):
                return prefix
            raise ValueError(f"existing prefix differs from this SDK: {prefix}")
        with TemporaryDirectory(dir=prefix.parent, prefix=".scar-pocketfft-") as work:
            stage = Path(work) / "install"
            for name, expected in FILES.items():
                output = stage / name
                upstream = f"https://raw.githubusercontent.com/mreineck/pocketfft/{REVISION}/{output.name}"
                with urlopen(upstream, timeout=30) as source:
                    data = source.read()
                if sha256(data).hexdigest() != expected:
                    raise ValueError(f"upstream digest differs for {name}")
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(data)
            stage.rename(prefix)
    return prefix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix",
        type=Path,
        default=os.environ.get("SCAR_FOURIER_SDK")
        or Path.home() / ".local/share/scar/sdk" / ("pocketfft-" + REVISION),
    )
    print(provision(parser.parse_args().prefix))
