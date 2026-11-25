"""Qualify the installed standalone CPU complex DFT provider for sealed operations."""

import os
from pathlib import Path

from scar.contract import array_rank_domain, digest, json_bytes
from .runtime import QualificationFailure, qualification_check

REVISION = "33ae5dc94c9cdc7f1c78346504a85de87cadaa12"
REQUIRED_PROVIDER = {"name": "pocketfft", "revision": REVISION}
PROFILE = "complex128-rank1-ortho-v1"
FILES = {
    "include/pocketfft_hdronly.h": {
        "size": 111974,
        "sha256": "50f4ddbe60432325f560c706b6f1c49f9e0949fc7f20ae6499f2299e37d9790c",
    },
    "LICENSE.md": {
        "size": 1498,
        "sha256": "a85ca13fdf90160b64a0698215868c13b74d835ad0a4e2ba44713b8c5058a056",
    },
}


def _use(node, families):
    vector = dict(kind="ndarray", element="c128", rank=1)
    direction = node.get("direction")
    source = node.get("input")
    if (
        node.get("op") != "array_fft"
        or node.get("operator_family") != "fourier"
        or node.get("type") != vector
        or type(node["type"]["rank"]) is not int
        or not isinstance(source, dict)
        or source.get("type") != vector
        or type(source["type"]["rank"]) is not int
        or type(node.get("axis")) is not int
        or node["axis"] != 0
        or direction not in ("forward", "inverse")
        or node.get("normalization") != "ortho"
        or node.get("may_raise") is not True
        or node.get("required_provider") != REQUIRED_PROVIDER
        or {"n", "out", "axes", "workers", "in_place"}.intersection(node)
    ):
        for key, ty in (
            ("input.type", source.get("type") if isinstance(source, dict) else None),
            ("type", node.get("type")),
        ):
            with qualification_check(
                "semantic_contract",
                "fourier.array_type",
                site=node.get("site"),
                field=key,
                observed=ty,
            ):
                domain = array_rank_domain(ty)
                if domain is None or len(domain) != 1:
                    raise ValueError("the Fourier provider requires a proven rank")
        missing = [
            key
            for key in (
                "axis",
                "direction",
                "normalization",
                "required_provider",
                "operator_family",
                "may_raise",
            )
            if node.get(key) is None
        ]
        stage = (
            "semantic_contract"
            if missing or node.get("may_raise") is not True
            else "implementation_profile"
        )
        raise QualificationFailure(
            stage,
            "fourier.c2c_signature",
            "requires a rank-one complex128 orthonormal DFT of the complete input",
            site=node.get("site"),
            missing_fields=missing,
            result_type=node.get("type"),
            input_type=source.get("type"),
            axis=node.get("axis"),
            direction=direction,
            normalization=node.get("normalization"),
            provider=node.get("required_provider"),
        )
    policy = node.get("numeric_policy")
    effects = set(node.get("effects", ()))
    if policy == "numpy_default":
        warning = node.get("warning", {})
        location = warning.get("location", {}) if isinstance(warning, dict) else None
        if (
            effects != {"MEM", "VAL", "STATE", "IO"}
            or not isinstance(location, dict)
            or warning.get("operation")
            != {"forward": "fft", "inverse": "ifft"}[direction]
            or not isinstance(warning.get("module"), str)
            or not warning["module"]
            or not isinstance(location.get("diagnostic_file"), str)
            or not location["diagnostic_file"]
            or type(location.get("line")) is not int
            or location["line"] <= 0
            or not isinstance(warning.get("source_line"), str)
        ):
            raise ValueError(
                "Fourier warning policy requires the qualified diagnostic location and observable effects"
            )
    elif policy is not None or effects != {"MEM", "VAL"} or "warning" in node:
        raise QualificationFailure(
            "implementation_profile" if policy is not None else "semantic_contract",
            "fourier.numeric_effect_policy",
            "unregistered Fourier numeric or diagnostic policy",
            numeric_policy=policy,
            effects=sorted(effects),
            site=node.get("site"),
        )
    source_binding = node.get("source_binding")
    if source_binding is not None and (
        not isinstance(source_binding, dict)
        or not isinstance(source_binding.get("required_family"), str)
        or source_binding["required_family"] not in families
    ):
        raise QualificationFailure(
            "external_effects",
            "fourier.source_dispatch",
            "Fourier source dispatch requires its separate operator binding",
            required_family=(
                source_binding.get("required_family")
                if isinstance(source_binding, dict)
                else None
            ),
            site=node.get("site"),
        )
    source = node.get("source_profile")
    if source is not None and not isinstance(source, dict):
        raise ValueError("invalid Fourier source profile")
    # B2a checks this occurrence after scheduling. The input expression can be
    # rewritten independently; its rank/dtype are rechecked at consumption.
    fingerprint = digest(json_bytes({k: v for k, v in node.items() if k != "input"}))
    return fingerprint, dict(
        direction=direction,
        site=node.get("site"),
        numeric_policy=policy,
        source_profile_sha256=(
            digest(json_bytes(source)) if source is not None else None
        ),
        source_binding_sha256=(
            digest(json_bytes(source_binding)) if source_binding else None
        ),
    )


def qualify(requirement, nodes, families):
    if (
        not isinstance(requirement, dict)
        or requirement.keys() - {"profile", "required_provider", "operations"}
        or requirement.get("profile", PROFILE) != PROFILE
        or requirement.get("required_provider", REQUIRED_PROVIDER) != REQUIRED_PROVIDER
        or requirement.get("operations") != ["array_fft"]
        or not nodes
    ):
        if (
            not isinstance(requirement, dict)
            or not nodes
            or not isinstance(requirement.get("operations"), list)
        ):
            raise ValueError(
                "Fourier requires a declared operation and typed execution occurrence"
            )
        raise QualificationFailure(
            "implementation_profile",
            "fourier.registered_profile",
            "unregistered Fourier operation, profile or provider revision",
            operations=requirement.get("operations"),
            profile=requirement.get("profile", PROFILE),
            required_provider=requirement.get("required_provider", REQUIRED_PROVIDER),
        )
    uses = dict(_use(node, families) for node in nodes)
    default = Path.home() / ".local/share/scar/sdk" / ("pocketfft-" + REVISION)
    try:
        prefix = (
            Path(os.environ.get("SCAR_FOURIER_SDK") or default)
            .expanduser()
            .resolve(strict=True)
        )
        for name, identity in FILES.items():
            path = (prefix / name).resolve(strict=True)
            if (
                not path.is_relative_to(prefix)
                or not path.is_file()
                or path.stat().st_size != identity["size"]
                or digest(path.read_bytes()) != identity["sha256"]
            ):
                raise ValueError(f"pinned PocketFFT SDK member differs: {name}")
    except (ValueError, OSError) as error:
        detail = (
            str(error)
            if isinstance(error, ValueError)
            else f"missing pinned native Fourier SDK; run tools/provision_fft_sdk.py: {error}"
        )
        raise QualificationFailure(
            "target_platform",
            "fourier.sdk_identity",
            detail,
            provider=REQUIRED_PROVIDER,
            required_files=FILES,
        ) from error
    return dict(
        provider="pocketfft-cpu",
        semantics_version=1,
        revision=REVISION,
        profile="c2c-c128-rank1-ortho",
        device="cpu",
        prefix=str(prefix),
        files=FILES,
        calls=uses,
        directions=sorted({use["direction"] for use in uses.values()}),
        abi=dict(
            interface="pocketfft::c2c<double>",
            rank=1,
            axes=[0],
            extent="input.shape[0]",
            input_stride="region element stride in bytes",
            output_stride_bytes=16,
            normalization="1/sqrt(binary64(input.shape[0]))",
        ),
        resources=dict(
            workers=1,
            thread_policy="single transform, one provider worker",
            output="fresh row-major array; 16 bytes per element",
            temporary="provider-managed transform plan and workspace",
            temporary_bytes=None,
            predicted_duration_ns=None,
        ),
    )
