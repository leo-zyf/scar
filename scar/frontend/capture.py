"""Discover F1 tasks and join captured project catalogs before unit processing."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from scar.contract import json_bytes, validate_images

if TYPE_CHECKING:
    from .languages.python.environment import ImportEnvironment


@dataclass(frozen=True)
class CapturedProject:
    profiles: list
    parsed_units: list
    images: list | None
    root_image: str | None
    source_hashes: dict
    build: dict | None
    build_bindings: list
    python_environment: "ImportEnvironment | None" = None


def capture_project(request):
    """Intern full compilation profiles across script and build discoveries.

    An observed link binds an artifact to an image. It does not prove that a
    runtime build operation, its diagnostics or its filesystem effects can go.
    """
    python_environment = None
    if request.python_entries:
        from .languages.python.environment import capture_import_environment

        python_environment = capture_import_environment(request.python_entries)
    sources = dict(request.source_hashes)
    profiles, parsed_units, indices = [], [], {}

    def add_sources(receipts):
        for name, revision in receipts.items():
            if sources.setdefault(name, revision) != revision:
                raise ValueError(
                    f"project captures observed different source revisions: {name}"
                )

    def intern(profile, parsed=None):
        key = json_bytes(profile)
        if key not in indices:
            indices[key] = len(profiles)
            profiles.append(profile)
            parsed_units.append(parsed)
        elif parsed is not None:
            parsed_units[indices[key]] = parsed
        return indices[key]

    for profile in request.profiles:
        intern(profile)
    images, root_image = request.images, request.root_image
    if request.project_entry or any("source" in image for image in images or ()):
        from .languages.python.facts import ProjectCapture

        scripts = ProjectCapture(
            request.root,
            buffers=request.python_buffers,
            source_hashes=request.source_hashes,
        )

        def script(path):
            project = scripts.capture(path)
            add_sources(project.source_digests)
            return [
                intern(profile, parsed)
                for profile, parsed in zip(project.profiles, project.units)
            ]

        if request.project_entry:
            # The initial profile only locks this script. Discovery supplies its
            # real module ABI and all project import units instead.
            profiles.clear()
            parsed_units.clear()
            indices.clear()
            script(request.project_entry)
        else:
            images = [
                (
                    {key: value for key, value in image.items() if key != "source"}
                    | {"units": script(image["source"])}
                    if "source" in image
                    else image
                )
                for image in images
            ]
    build, bindings = None, []
    if request.build_trace:
        from .builds.facts import capture_build
        from .builds.normalize import normalize_build

        trace = request.build_trace
        facts = capture_build(
            trace["raw"],
            root=request.root,
            provider=trace["provider"],
            process_observation=trace.get("process_observation"),
            process_tree=trace.get("process_tree"),
        )
        build = normalize_build(
            facts,
            targets=trace["targets"],
            entry="main" if images is not None else request.entry,
            require_root=images is None,
        )
        add_sources(build["source_hashes"])
        remap = [intern(profile) for profile in build["profiles"]]
        nodes = {node["id"]: node for node in build["build_graph"]["nodes"]}
        built_images = []
        for image in build["images"]:
            image = {**image, "units": [remap[index] for index in image["units"]]}
            built_images.append(image)
            artifact = nodes[image["build_artifact"]]
            bindings.append(
                dict(
                    image_id=image["id"],
                    target_artifact=artifact["id"],
                    source_path=artifact["path"],
                    unit_indices=image["units"],
                    trace_sha256=build["build_graph"]["trace_sha256"],
                    evidence="captured_compile_and_link",
                    obligations=dict(
                        historical_input_versions="unproven",
                        recipe_effects="unproven",
                        diagnostic_stream="unobserved",
                    ),
                )
            )
        if images is None:
            images, root_image = built_images or None, build["root_image"]
        else:
            images = [*images, *built_images]
    if images is not None:
        validate_images(images, root_image, unit_count=len(profiles))
    return CapturedProject(
        profiles,
        parsed_units,
        images,
        root_image,
        sources,
        build,
        bindings,
        python_environment,
    )
