"""Static dependencies."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace, dataclass
import gzip
import hashlib
import json
import os
import stat
import multiprocessing
import shutil
import tempfile
import time
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
from scar.ir.program.discovery import ModuleMapping
from scar.ir.program.providers import ProviderArtifact
from scar.ir.program.source_references import SourceArtifactLocator
from scar.ir.program.static_dependencies import (
    DependencyFrontier,
    DependencyImportRelation,
    DependencyLimits,
    DependencyMaterial,
    DependencyMaterialKind as Kind,
    DependencyModuleQuery,
    DependencyRelationCount,
    DependencySearchStatus as Search,
    DependencyVersionStatus as Version,
    StaticDependencyInventory,
)
from scar.ir.v2.common import SourceVersion
from concurrent.futures import ProcessPoolExecutor, as_completed
from scar.analysis.static.sources import _ByteBudget, _Changed, _has_symlink_component, _read_source
from scar.ir.program.static_sources import StaticSourceFile


# Locate materials named by an existing import index, without importing targets.
#
# The only searches are fixed module-derived paths under caller-supplied roots.
# There is no environment walk, importlib resolution, import hook, package/plugin
# initialization, setup.py execution, or second definition/value graph.
def _stamp(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _symlink(path):
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            return str(current)
    return None


def _descriptor_locator(root, descriptor, schema):
    relative = Path(descriptor["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("dependency source artifact escapes its model")
    path = root / relative
    return SourceArtifactLocator(
        ProviderArtifact(str(path), descriptor["sha256"], descriptor["bytes"], schema),
        descriptor["decoded_sha256"], descriptor["decoded_bytes"], "gzip-canonical-json")


def _load_inputs(directory):
    root = Path(os.path.abspath(directory))
    if _symlink(root) is not None:
        raise ValueError("static dependency model must not traverse symlinks")
    path = root / "manifest.json"
    if _symlink(path) is not None:
        raise ValueError("static dependency manifest must not traverse symlinks")
    payload = path.read_bytes()
    manifest = json.loads(payload)
    if manifest.get("schema") != "scar.static-program" or manifest.get("schema_version") != 1:
        raise ValueError("unsupported static program manifest")
    descriptor = manifest.get("artifacts", {}).get("imports")
    if descriptor is None:
        raise ValueError("static program has no original import index; refine its lexical/import index first")
    # These immutable fragment receipts survive adding this inventory to the
    # same manifest. An optional declaration receipt to the mutable manifest
    # would become stale (or require a circular self-hash) on publication.
    locator = _descriptor_locator(root, descriptor, "scar.static-import-index")
    import_path = Path(locator.artifact.uri)
    if _symlink(import_path) is not None:
        raise ValueError("import index must not traverse symlinks")
    stored = import_path.read_bytes()
    if len(stored) != locator.artifact.size_bytes or hashlib.sha256(stored).hexdigest() != locator.artifact.sha256:
        raise ValueError("original import index stored receipt differs")
    decoded = gzip.decompress(stored)
    if len(decoded) != locator.decoded_bytes or hashlib.sha256(decoded).hexdigest() != locator.decoded_sha256:
        raise ValueError("original import index decoded receipt differs")
    imports = json.loads(decoded)
    if imports.get("schema") != "scar.static-import-index" or imports.get("schema_version") != 1:
        raise ValueError("unsupported original import index")
    return root, manifest, imports, locator


def _read_receipt(path, before, limit, spent):
    """Hash complete stable bytes; stat alone is never a content identity."""
    if before.st_size > limit:
        raise ValueError("file exceeds available content-read budget")
    spent[0] += 1
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        if _stamp(os.fstat(descriptor)) != _stamp(before):
            raise OSError("material changed between path check and open")
        digest, read = hashlib.sha256(), 0
        while read < before.st_size:
            chunk = os.read(descriptor, min(1024 * 1024, before.st_size - read))
            if not chunk:
                break
            read += len(chunk)
            spent[1] += len(chunk)
            digest.update(chunk)
        if (read != before.st_size or _stamp(os.fstat(descriptor)) != _stamp(before)
                or _stamp(path.lstat()) != _stamp(before) or _symlink(path) is not None):
            raise OSError("material changed while its version was read")
        return digest.hexdigest(), read
    finally:
        os.close(descriptor)


def read_static_dependencies(directory):
    """Read the published view against its original current import endpoints.

    External receipts describe bytes inventoried earlier; this reader does not
    import those files or assert that they are still their recorded version.
    Source ingestion must check that version again before using new bytes.
    """
    from dataclasses import fields
    from scar.analysis.static_program import _read_fragment
    from scar.ir.record_codec import decode

    root, manifest, imports, locator = _load_inputs(directory)
    descriptor = manifest.get("artifacts", {}).get("dependencies")
    if descriptor is None:
        raise ValueError("static program has no published dependency inventory")
    relative = Path(descriptor["path"])
    if relative.is_absolute() or ".." in relative.parts or _symlink(root / relative) is not None:
        raise ValueError("dependency inventory escapes its model or traverses symlinks")
    document = _read_fragment(root, descriptor)
    if (document.get("schema") != StaticDependencyInventory.SCHEMA
            or document.get("schema_version") != StaticDependencyInventory.SCHEMA_VERSION):
        raise ValueError("unsupported dependency inventory")
    names = {field.name for field in fields(StaticDependencyInventory)}
    if set(document) != names | {"schema", "schema_version", "summary", "qualification"}:
        raise ValueError("dependency inventory has unexpected or missing fields")
    result = decode(StaticDependencyInventory, {key: document[key] for key in names})
    if result.import_index != locator:
        raise ValueError("dependency inventory refers to a different original import index")
    if document["summary"] != result.summary() or document["qualification"] != result.to_dict()["qualification"]:
        raise ValueError("dependency inventory summary or qualification differs")
    if len(result.relations) != len(imports["relations"]):
        raise ValueError("dependency inventory does not cover the original import relations")
    for origin, relation in zip(imports["relations"], result.relations):
        module = origin.get("candidate_module")
        if relation.module_query is None:
            if module:
                raise ValueError("dependency inventory omitted an original module query")
        elif result.module_queries[relation.module_query].module != module:
            raise ValueError("dependency module query differs from the original relation")
    units = {unit["key"]: unit for unit in manifest["units"]}
    for material in result.materials:
        if material.supplied_unit is None:
            continue
        unit = units.get(material.supplied_unit)
        if (unit is None or unit["path"] != material.path
                or material.supplied_source != _descriptor_locator(root, unit["artifacts"]["source"], "scar.static-source-text")
                or material.supplied_version != SourceVersion(unit["path"], "sha256:" + unit["decoded_sha256"])):
            raise ValueError("dependency supplied snapshot differs from its current original unit")
    return result


def inventory_static_dependencies(directory, *, search_roots, max_modules=4096, max_files=4096,
                                  max_bytes=64 * 1024 * 1024, max_file_bytes=8 * 1024 * 1024,
                                  native_suffixes=None):
    """Return a typed, read-only material view of the original import relations.

    ``search_roots`` are explicit import-root *candidates*, not a claim about a
    target process's sys.path. ``native_suffixes`` defaults to this analyzer's
    public interpreter suffix metadata; cross-environment clients must supply
    their target profile. Dotted names probe only their component-derived file
    names; ``pkg-stubs`` is a declaration candidate, never a runtime package.

    The module budget bounds filesystem queries. File/byte budgets bound only
    attempted external file reads (including failed reads), with source/stub
    files prioritized over native binaries. Supplied snapshots retain original receipts and are not
    reread from the application's source tree. Missing paths are scoped to
    these roots and patterns; from-export/submodule ambiguity remains in the
    original relation records and is never resolved by file presence alone.
    """
    limits = DependencyLimits(max_modules, max_files, max_bytes, max_file_bytes)
    if isinstance(search_roots, (str, bytes, Path)):
        raise TypeError("search_roots must be an iterable of explicit root paths")
    roots = tuple(dict.fromkeys(str(Path(os.path.abspath(x))) for x in search_roots))
    suffixes = tuple(dict.fromkeys(EXTENSION_SUFFIXES if native_suffixes is None else native_suffixes))
    if not suffixes or any(type(x) is not str or not x.startswith(".") or "/" in x for x in suffixes):
        raise ValueError("native suffixes must be nonempty explicit filename suffixes")
    root, manifest, imports, locator = _load_inputs(directory)
    roots_frontiers, valid_roots = [], []
    for supplied_root in roots:
        path = Path(supplied_root)
        try:
            link = _symlink(path)
            if link is not None:
                roots_frontiers.append(DependencyFrontier("SEARCH_ROOT_SYMLINK", supplied_root, "Not followed: " + link))
            elif not stat.S_ISDIR(path.lstat().st_mode):
                roots_frontiers.append(DependencyFrontier("SEARCH_ROOT_NOT_DIRECTORY", supplied_root, "No directory available."))
            else:
                valid_roots.append(path)
        except OSError as error:
            roots_frontiers.append(DependencyFrontier("SEARCH_ROOT_UNAVAILABLE", supplied_root, str(error)))

    material_rows, material_keys, mappings, supplied = [], {}, defaultdict(set), defaultdict(set)

    def add_material(material):
        key = (material.path, material.kind, material.supplied_unit)
        if key not in material_keys:
            material_keys[key] = len(material_rows)
            material_rows.append(material)
        number = material_keys[key]
        mappings[number].update(material.mappings)
        return number

    for unit in manifest["units"]:
        pairs = tuple(ModuleMapping(x["name"], x["source_root"]) for x in unit.get("module_mappings", ()))
        if not pairs:
            pairs = tuple(ModuleMapping(x, unit["namespace_root"]) for x in unit.get("module_names", ()))
        descriptor = unit.get("artifacts", {}).get("source")
        if not pairs or descriptor is None:
            continue
        material = DependencyMaterial(unit["path"], Kind.STUB if unit["path"].endswith(".pyi") else Kind.SOURCE,
            tuple(sorted(set(pairs), key=lambda x: (x.source_root, x.name))), Version.SUPPLIED_SNAPSHOT,
            supplied_unit=unit["key"],
            supplied_source=_descriptor_locator(root, descriptor, "scar.static-source-text"),
            supplied_version=SourceVersion(unit["path"], "sha256:" + unit["decoded_sha256"]),
            version_detail="Original supplied snapshot; current application source file was not reread or imported.")
        index = add_material(material)
        for mapping in pairs:
            supplied[mapping.name].add(index)

    # Retain only referenced supplied materials in the published view; do not
    # confuse every file in an analysis scope with an actually used import.
    names = sorted({x["candidate_module"] for x in imports["relations"] if type(x.get("candidate_module")) is str
                    and x["candidate_module"]})
    queries, query_by_name, searched = [], {}, 0
    for name in names:
        indices, frontiers = set(supplied.get(name, ())), []
        if not all(x.isidentifier() for x in name.split(".")):
            status = Search.INVALID_MODULE_NAME
        elif searched >= limits.max_modules:
            status = Search.MODULE_BUDGET
            frontiers.append(DependencyFrontier("MODULE_SEARCH_BUDGET", None, "No filesystem query for this module was performed."))
        else:
            status = Search.SEARCHED_EXPLICIT_ROOTS
            searched += 1
            parts = name.split(".")
            for search_root in valid_roots:
                stem = search_root.joinpath(*parts)
                candidates = [(Path(str(stem) + ".py"), Kind.SOURCE),
                              (stem / "__init__.py", Kind.SOURCE),
                              (Path(str(stem) + ".pyi"), Kind.STUB), (stem / "__init__.pyi", Kind.STUB)]
                for suffix in suffixes:
                    candidates.extend(((Path(str(stem) + suffix), Kind.NATIVE), (stem / ("__init__" + suffix), Kind.NATIVE)))
                stub_stem = search_root.joinpath(parts[0] + "-stubs", *parts[1:])
                if len(parts) == 1:
                    candidates.append((stub_stem / "__init__.pyi", Kind.STUB))
                else:
                    candidates.extend(((Path(str(stub_stem) + ".pyi"), Kind.STUB), (stub_stem / "__init__.pyi", Kind.STUB)))
                # A .pyi initializer is a declaration and does not turn a
                # directory into an executable regular Python package.
                runtime_initializer_found = False
                for path, kind in candidates:
                    try:
                        link = _symlink(path)
                        if link is not None:
                            frontiers.append(DependencyFrontier("MATERIAL_SYMLINK_NOT_FOLLOWED", str(path), "Not followed: " + link))
                            continue
                        info = path.lstat()
                    except FileNotFoundError:
                        continue
                    except OSError as error:
                        frontiers.append(DependencyFrontier("MATERIAL_PATH_UNAVAILABLE", str(path), str(error)))
                        continue
                    if not stat.S_ISREG(info.st_mode):
                        frontiers.append(DependencyFrontier("MATERIAL_NOT_REGULAR_FILE", str(path), "Candidate is not a regular file."))
                        continue
                    if path.parent == stem and kind in (Kind.SOURCE, Kind.NATIVE):
                        runtime_initializer_found = True
                    # A supplied snapshot already covers this exact mapping;
                    # avoid inventing a second file identity for its disk path.
                    matched = [i for i in indices if material_rows[i].path == str(path)
                               and material_rows[i].supplied_unit is not None
                               and ModuleMapping(name, str(search_root)) in material_rows[i].mappings]
                    if matched:
                        continue
                    index = add_material(DependencyMaterial(str(path), kind, (ModuleMapping(name, str(search_root)),),
                        Version.CONTENT_NOT_READ, stat_stamp=_stamp(info), version_detail="Content version not read yet."))
                    indices.add(index)
                try:
                    if _symlink(stem) is None and stat.S_ISDIR(stem.lstat().st_mode) and not runtime_initializer_found:
                        indices.add(add_material(DependencyMaterial(str(stem), Kind.NAMESPACE_DIRECTORY,
                            (ModuleMapping(name, str(search_root)),), Version.DIRECTORY_ONLY,
                            stat_stamp=_stamp(stem.lstat()), version_detail="Namespace directory candidate, not a file/content identity.")))
                except FileNotFoundError:
                    pass
                except OSError as error:
                    frontiers.append(DependencyFrontier("NAMESPACE_PATH_UNAVAILABLE", str(stem), str(error)))
        query_by_name[name] = len(queries)
        queries.append(DependencyModuleQuery(name, status, tuple(sorted(indices)), tuple(frontiers)))

    used = sorted({i for query in queries for i in query.materials})
    old_to_new = {old: new for new, old in enumerate(used)}
    material_rows = [replace(material_rows[i], mappings=tuple(sorted(mappings[i], key=lambda x: (x.source_root, x.name)))) for i in used]
    queries = [replace(x, materials=tuple(sorted(old_to_new[i] for i in x.materials))) for x in queries]

    read_files, spent = 0, [0, 0]
    order = sorted(range(len(material_rows)), key=lambda i: (material_rows[i].kind is Kind.NATIVE, material_rows[i].path))
    for index in order:
        material = material_rows[index]
        if material.version_status is not Version.CONTENT_NOT_READ:
            continue
        required = material.stat_stamp[3]
        available = min(limits.max_file_bytes, limits.max_bytes - spent[1])
        if spent[0] >= limits.max_files or required > available:
            material_rows[index] = replace(material, version_detail="Content version not read: file/count/byte budget. Stat is not a content identity.")
            continue
        try:
            path = Path(material.path)
            if _symlink(path) is not None:
                raise OSError("material path now traverses a symbolic link")
            before = path.lstat()
            if not stat.S_ISREG(before.st_mode) or _stamp(before) != material.stat_stamp:
                raise OSError("material changed since candidate discovery")
            digest, count = _read_receipt(path, before, available, spent)
            material_rows[index] = replace(material, version_status=Version.CONTENT_HASHED,
                artifact=ProviderArtifact(material.path, digest, count, "scar.dependency-material-bytes"),
                version_detail="Complete stable file bytes hashed; file was not imported, parsed or executed.")
            read_files += 1
        except (OSError, ValueError) as error:
            material_rows[index] = replace(material, version_status=Version.READ_FAILED,
                                           version_detail=type(error).__name__ + ": " + str(error))

    relation_rows, counts = [], Counter()
    for number, relation in enumerate(imports["relations"]):
        query_index = query_by_name.get(relation.get("candidate_module"))
        relation_rows.append(DependencyImportRelation("#/relations/" + str(number), query_index,
            relation.get("status", "ORIGINAL_MODULE_UNRESOLVED") if query_index is None else None))
        if query_index is None:
            availability = "ORIGINAL_MODULE_UNRESOLVED"
        else:
            query = queries[query_index]
            if query.materials:
                availability = "MATERIAL_CANDIDATES"
            elif query.status is not Search.SEARCHED_EXPLICIT_ROOTS:
                availability = query.status.value
            elif query.frontiers or roots_frontiers:
                availability = "SEARCH_FRONTIER"
            else:
                availability = "NOT_LOCATED_IN_EXPLICIT_ROOTS"
        counts[(relation["relation_kind"], availability)] += 1
    return StaticDependencyInventory(locator, roots, suffixes, limits, tuple(material_rows), tuple(queries),
        tuple(relation_rows), tuple(DependencyRelationCount(kind, availability, count)
            for (kind, availability), count in sorted(counts.items())), tuple(roots_frontiers), read_files, spent[1], spent[0])


# Feed selected dependency materials to the existing, source-only frontend.
#
# This helper owns newly written unit fragments in a caller-owned staging
# directory. It does not publish a manifest, scan package directories, select a
# runtime import implementation, execute target callbacks, or claim a closure.
# Normal units are produced by static_program._run_file, not another graph IR.
@dataclass(frozen=True, slots=True)
class DependencyExpansionFrontier:
    material: int
    namespace_root: str | None
    kind: str
    detail: str


@dataclass(frozen=True, slots=True)
class DependencyUnitOrigin:
    material: int
    unit_key: str
    namespace_root: str
    mappings: tuple[ModuleMapping, ...]
    analyzed_file: ProviderArtifact
    module_queries: tuple[int, ...]
    import_relations: tuple[str, ...]
    reused: bool = False


@dataclass(frozen=True, slots=True)
class DependencyExpansion:
    """Normal frontend units and references to their conditional input basis."""

    import_index: SourceArtifactLocator
    output: str
    selected_materials: tuple[int, ...]
    unselected_materials: tuple[int, ...]
    units: tuple[dict, ...]
    origins: tuple[DependencyUnitOrigin, ...]
    frontiers: tuple[DependencyExpansionFrontier, ...]
    read_files: int
    source_bytes: int
    charged_bytes: int
    max_files: int
    max_bytes: int
    jobs: int

    def summary(self):
        from collections import Counter
        return {"selected_materials": len(self.selected_materials), "unselected_materials": len(self.unselected_materials),
                "source_files_read": self.read_files, "source_bytes": self.source_bytes,
                "source_read_budget_charged_bytes": self.charged_bytes,
                "units": len(self.units), "reused_units": sum(x.reused for x in self.origins),
                "statuses": dict(Counter(x["status"] for x in self.units)),
                "frontiers": dict(Counter(x.kind for x in self.frontiers)),
                "target_executed": False, "phase": "graph", "runtime_import_result_established": False,
                "dependency_closure_complete": False, "canonical_bodies_expanded": False}

    def to_dict(self):
        from scar.ir.record_codec import encode
        return {"schema": "scar.static-dependency-expansion", "schema_version": 1,
                **encode(self), "summary": self.summary(),
                "qualification": "Units are alternative source interpretations grouped by namespace root. "
                    "They do not select a runtime import implementation. Stub catalogs supply declarations, not bodies/effects. "
                    "Original import relation pointers apply to the exact input fragment receipt; a publisher rebuilding "
                    "that index must retain or mechanically relocate this proof basis. Native and unresolved selection "
                    "boundaries stay open; unselected material indices are continuation inputs, not dead work."}


def _existing_receipts_match(output, unit, source):
    """An existing unit may be reused, never overwritten or silently rebound."""
    from scar.analysis.static_program import _read_fragment
    if any(unit.get(name) != expected for name, expected in (
            ("path", source.path), ("namespace_root", source.namespace_root),
            ("raw_sha256", source.raw_sha256), ("decoded_sha256", source.decoded_sha256),
            ("encoding", source.encoding), ("source_bytes", source.size_bytes),
            ("module_names", list(source.module_names)),
            ("module_mappings", [{"name": x.name, "source_root": x.source_root} for x in source.module_mappings]))):
        return False
    descriptors = [*unit["artifacts"].values(), *(x["artifact"] for x in unit.get("bodies", ()) if "artifact" in x)]
    for descriptor in descriptors:
        relative = Path(descriptor["path"])
        if relative.is_absolute() or ".." in relative.parts:
            return False
        path = output / relative
        if _has_symlink_component(path) is not None:
            return False
        stored = path.read_bytes()
        if len(stored) != descriptor["bytes"] or hashlib.sha256(stored).hexdigest() != descriptor["sha256"]:
            return False
        raw = gzip.decompress(stored)
        if len(raw) != descriptor["decoded_bytes"] or hashlib.sha256(raw).hexdigest() != descriptor["decoded_sha256"]:
            return False
    snapshot = _read_fragment(output, unit["artifacts"]["source"])
    if (snapshot.get("text") != source.text or snapshot.get("path") != source.path
            or snapshot.get("namespace_root") != source.namespace_root
            or snapshot.get("raw_sha256") != source.raw_sha256
            or snapshot.get("decoded_sha256") != source.decoded_sha256):
        return False
    return True


def expand_static_dependency_materials(inventory, *, out, material_indices=None, max_files=16,
                                      max_bytes=8 * 1024 * 1024, jobs=1, existing_units=(), progress=None):
    """Build bounded external .py/.pyi units in ``out`` without publication.

    ``material_indices`` selects candidates for static analysis, not runtime
    implementations. Omitted indices select external source/stub candidates.
    Each namespace-root interpretation is a distinct normal frontend unit;
    mappings from different roots are never merged into one interpretation.
    Existing units are reusable only when explicitly supplied and every input
    identity and stored fragment receipt matches. Other existing paths remain
    untouched and generate a frontier.

    ``max_files`` bounds attempted physical file reads; ``max_bytes`` bounds
    reserved raw-source reads, including rejected versions. A failed read
    conservatively consumes its reserved allowance, not a claimed measurement
    of bytes read. Remaining indices/frontiers support later rounds.
    The caller publishes units and refreshes global indices atomically; this
    helper never modifies a current manifest. ``progress`` is an optional
    caller-owned SCAR observer, never a callback discovered in target source.
    """
    from scar.analysis.static_program import _json, _run_file
    if type(inventory) is not StaticDependencyInventory:
        raise TypeError("dependency expansion requires the typed original inventory")
    inventory.__post_init__()
    for name, value in (("max_files", max_files), ("max_bytes", max_bytes)):
        if type(value) is not int or value < 1:
            raise ValueError(name + " must be a positive exact integer")
    if type(jobs) is not int or not 1 <= jobs <= 32:
        raise ValueError("jobs must be in [1, 32]")
    output = Path(os.path.abspath(out))
    if _has_symlink_component(output) is not None:
        raise ValueError("dependency staging output must not traverse symlinks")
    if material_indices is None:
        selected = tuple(i for i, material in enumerate(inventory.materials)
                         if material.supplied_unit is None and material.kind in (Kind.SOURCE, Kind.STUB))
    else:
        selected = tuple(material_indices)
        if any(type(x) is not int or not 0 <= x < len(inventory.materials) for x in selected):
            raise ValueError("selected material index is outside the original inventory")
        if len(set(selected)) != len(selected):
            raise ValueError("selected material indices must be unique")
    unselected = tuple(i for i in range(len(inventory.materials)) if i not in set(selected))
    old = {}
    for unit in existing_units:
        if unit["key"] in old:
            raise ValueError("existing units contain a duplicate key")
        old[unit["key"]] = unit
    output.mkdir(parents=True, exist_ok=True)
    if _has_symlink_component(output / "units") is not None:
        raise ValueError("dependency unit directory must not traverse symlinks")
    query_indices = defaultdict(set)
    query_relations = defaultdict(set)
    for number, query in enumerate(inventory.module_queries):
        for index in query.materials:
            query_indices[index].add(number)
    for relation in inventory.relations:
        if relation.module_query is not None:
            query_relations[relation.module_query].add(relation.origin)

    frontiers, origins, units, prepared = [], [], {}, []
    read_files, source_bytes, charged_bytes = 0, 0, 0

    def frontier(index, root, kind, detail):
        frontiers.append(DependencyExpansionFrontier(index, root, kind, detail))

    for index in selected:
        material = inventory.materials[index]
        if material.supplied_unit is not None:
            frontier(index, None, "ALREADY_SUPPLIED_SNAPSHOT", "Material already names an original supplied unit; no disk reread was attempted.")
            continue
        if material.kind not in (Kind.SOURCE, Kind.STUB):
            frontier(index, None, "NON_SOURCE_MATERIAL", "Native implementation contracts and namespace selection are not source files.")
            continue
        if read_files >= max_files:
            frontier(index, None, "FILE_BUDGET", "Physical source read budget reached; this candidate was not read.")
            continue
        path = Path(material.path)
        if _has_symlink_component(path) is not None:
            frontier(index, None, "SOURCE_SYMLINK", "Candidate source traverses a symbolic link; no source was read.")
            continue
        expected_bytes = material.artifact.size_bytes if material.artifact is not None else (
            material.stat_stamp[3] if material.stat_stamp is not None else 0)
        if expected_bytes > max_bytes - charged_bytes:
            frontier(index, None, "BYTE_BUDGET", "Candidate exceeds remaining source-byte budget; it remains available for another round.")
            continue
        read_files += 1
        read_limit = min(expected_bytes or max_bytes - charged_bytes, max_bytes - charged_bytes)
        try:
            raw, text, encoding = _read_source(path, read_limit)
            charged_bytes += len(raw)
            raw_sha = hashlib.sha256(raw).hexdigest()
            if material.artifact is not None and (raw_sha != material.artifact.sha256 or len(raw) != material.artifact.size_bytes):
                frontier(index, None, "MATERIAL_VERSION_CHANGED", "Current stable source bytes differ from the material's original receipt.")
                continue
        except _ByteBudget as error:
            charged_bytes += read_limit
            frontier(index, None, "MATERIAL_VERSION_CHANGED" if material.artifact is not None else "SOURCE_SIZE_CHANGED",
                     "Current source size exceeds the inventoried/reserved read size; refresh this material before expansion.")
            continue
        except _Changed as error:
            charged_bytes += read_limit
            frontier(index, None, "SOURCE_CHANGED_DURING_READ", str(error))
            continue
        except (OSError, UnicodeError, LookupError, SyntaxError) as error:
            charged_bytes += read_limit
            frontier(index, None, "SOURCE_READ_FAILED", type(error).__name__ + ": " + str(error))
            continue
        source_bytes += len(raw)
        decoded_sha = hashlib.sha256(text.encode()).hexdigest()
        receipt = ProviderArtifact(material.path, raw_sha, len(raw), "scar.dependency-material-bytes")
        by_root = defaultdict(set)
        for mapping in material.mappings:
            by_root[mapping.source_root].add(mapping)
        for namespace, mapping_set in sorted(by_root.items()):
            mappings = tuple(sorted(mapping_set, key=lambda x: (x.source_root, x.name)))
            if not path.is_relative_to(namespace):
                frontier(index, namespace, "MATERIAL_MAPPING_OUTSIDE_ROOT", "Original mapping root does not contain this file.")
                continue
            source = StaticSourceFile(material.path, namespace, tuple(sorted(x.name for x in mappings)), text,
                                      raw_sha, decoded_sha, encoding, len(raw), mappings)
            key = hashlib.sha256(_json([namespace, source.path, decoded_sha])).hexdigest()[:24]
            if key in units:
                frontier(index, namespace, "DUPLICATE_UNIT_INTERPRETATION", "This identical source/root unit is already selected in this batch.")
                continue
            queries = tuple(sorted(query_indices[index]))
            pointers = tuple(sorted({p for q in queries for p in query_relations[q]}))
            origin = DependencyUnitOrigin(index, key, namespace, mappings, receipt, queries, pointers)
            destination = output / "units" / key
            if destination.exists() or key in old:
                try:
                    matched = key in old and _existing_receipts_match(output, old[key], source)
                except (OSError, ValueError, KeyError, TypeError):
                    matched = False
                if matched:
                    from dataclasses import replace
                    units[key] = old[key]
                    origins.append(replace(origin, reused=True))
                else:
                    frontier(index, namespace, "EXISTING_UNIT_NOT_EXACT", "Existing unit/path was not overwritten: its declaration or receipts do not match.")
                continue
            if destination.is_symlink():
                frontier(index, namespace, "EXISTING_UNIT_NOT_EXACT", "Existing symlink unit path was not overwritten.")
                continue
            # Mark scheduled keys, so parallel workers can never write the
            # same unit namespace even if a malformed inventory repeats it.
            units[key] = None
            prepared.append((source, origin))

    def completed(unit, origin):
        units[origin.unit_key] = unit
        origins.append(origin)
        if unit["status"] in ("ERROR", "SYNTAX_ERROR"):
            frontier(origin.material, origin.namespace_root, "FRONTEND_FAILED", "Normal source frontend reported " + unit["status"])
        elif unit["path"].endswith(".py") and not {"semantic", "semantics", "control"} <= unit["artifacts"].keys():
            frontier(origin.material, origin.namespace_root, "SOURCE_LAYERS_MISSING", "Normal source frontend retained explicit layer-construction issues.")
        if progress is not None:
            progress(len(origins), total_units, unit)

    total_units = len(prepared) + len(origins)
    if jobs == 1:
        for source, origin in prepared:
            completed(_run_file(source, str(output), "graph", 10000), origin)
    elif prepared:
        with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(_run_file, source, str(output), "graph", 10000): (source, origin)
                       for source, origin in prepared}
            for future in as_completed(futures):
                source, origin = futures[future]
                try:
                    unit = future.result()
                except Exception as error:
                    from scar.analysis.static_program import _failed_file
                    unit = _failed_file(source, str(output), error)
                completed(unit, origin)
    return DependencyExpansion(inventory.import_index, str(output), selected, unselected,
        tuple(units[key] for key in sorted(units) if units[key] is not None),
        tuple(sorted(origins, key=lambda x: (x.unit_key, x.material))), tuple(frontiers), read_files, source_bytes, charged_bytes,
        max_files, max_bytes, jobs)


# Publish import-driven source expansion into the one canonical static model.
def append_static_source_units(directory, sources, *, provenance, jobs=4,
                               max_bytes=64 * 1024 * 1024, progress=None,
                               expected_manifest_sha256=None):
    """Append explicit source interpretations without replacing old inputs.

    Retained relationship/import/material publications describe their original
    source basis only. No loader, source import or target execution occurs.
    """
    from concurrent.futures import ProcessPoolExecutor
    import json
    import multiprocessing
    import re
    from scar.analysis.static_program import (_artifacts, _json, _lowering_route,
        _read_fragment, _run_file, _write, read_static_program)
    from scar.analysis.static.sources import _read_source
    from scar.analysis.static_body import _stamp
    from scar.analysis.static.calls import input_basis_unit
    from scar.analysis.static.basis_compatibility import (
        _same,
        _source_expansion_valid,
        source_units_identity,
    )
    from scar.ir.program.static_sources import StaticSourceFile

    if type(sources) is not tuple or any(type(source) is not StaticSourceFile for source in sources):
        raise TypeError('source append requires an immutable tuple of exact StaticSourceFile records')
    if type(jobs) is not int or not 1 <= jobs <= 32:
        raise ValueError('source append jobs must be in [1,32]')
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError('source append byte budget must be positive')
    if type(provenance) is not dict or not provenance or not _same(provenance, provenance):
        raise ValueError('source append requires explicit JSON provenance')
    if expected_manifest_sha256 is not None and (type(expected_manifest_sha256) is not str
            or re.fullmatch(r'[0-9a-f]{64}', expected_manifest_sha256) is None):
        raise ValueError('expected manifest receipt must be a lowercase SHA256')
    if progress is not None and not callable(progress):
        raise TypeError('source append progress must be callable')
    output = Path(directory).absolute()
    manifest = read_static_program(output, verify=False)
    original = (output / 'manifest.json').read_bytes()
    if json.loads(original) != manifest or (expected_manifest_sha256 is not None
            and hashlib.sha256(original).hexdigest() != expected_manifest_sha256):
        raise ValueError('current static manifest changed before source append')
    if not sources:
        return manifest
    old_basis = [input_basis_unit(unit) for unit in manifest['units']]
    ledger = manifest.get('source_expansion')
    if ledger is not None and not _source_expansion_valid(ledger, old_basis):
        raise ValueError('current source expansion ledger is not an exact append-only chain')
    old_keys = {unit['key'] for unit in manifest['units']}
    old_coordinates = {(unit['namespace_root'], unit['path']) for unit in manifest['units']}
    if len(old_keys) != len(manifest['units']) or len(old_coordinates) != len(manifest['units']):
        raise ValueError('source append cannot inherit duplicate original unit identities')
    coordinates = [(source.namespace_root, source.path) for source in sources]
    if len(set(coordinates)) != len(sources) or any(key in old_coordinates for key in coordinates):
        raise ValueError('source append cannot replace or duplicate an original source interpretation')
    if sum(source.size_bytes for source in sources) > max_bytes:
        raise OverflowError('source append byte budget exhausted')
    requested_keys = [hashlib.sha256(_json([source.namespace_root, source.path,
        source.decoded_sha256])).hexdigest()[:24] for source in sources]
    if len(set(requested_keys)) != len(sources) or any(key in old_keys for key in requested_keys):
        raise ValueError('source append would overwrite an original canonical unit key')

    def verify_sources():
        for source in sources:
            source.__post_init__()
            if source.module_names or source.module_mappings:
                raise ValueError('observed source append cannot assert candidate import identities')
            path = Path(source.path)
            if path.resolve() != path or path.is_symlink():
                raise ValueError('source append requires original canonical file paths')
            raw, text, encoding = _read_source(path, min(max_bytes, source.size_bytes))
            if (len(raw) != source.size_bytes or hashlib.sha256(raw).hexdigest() != source.raw_sha256
                    or text != source.text or encoding != source.encoding
                    or hashlib.sha256(text.encode('utf8')).hexdigest() != source.decoded_sha256):
                raise ValueError('source append differs from its exact retained bytes/text/encoding')

    verify_sources()
    temporary = Path(tempfile.mkdtemp(prefix='.static-source-append-', dir=output.parent))
    try:
        shutil.copytree(output, temporary, dirs_exist_ok=True, copy_function=os.link)
        shared = {descriptor['path']: _stamp(temporary / descriptor['path'])
                  for descriptor in _artifacts(manifest)}
        if jobs == 1:
            new_units = []
            for source in sources:
                new_units.append(_run_file(source, str(temporary), 'graph', 10000))
                if progress is not None:
                    progress(len(new_units), len(sources), new_units[-1])
        else:
            with ProcessPoolExecutor(max_workers=jobs,
                    mp_context=multiprocessing.get_context('spawn')) as pool:
                # Preserve the requested unit order regardless of worker completion.
                futures = [pool.submit(_run_file, source, str(temporary), 'graph', 10000)
                           for source in sources]
                new_units = []
                for future in futures:
                    new_units.append(future.result())
                    if progress is not None:
                        progress(len(new_units), len(sources), new_units[-1])
        if len({unit['key'] for unit in new_units}) != len(new_units) or any(
                unit['key'] in old_keys for unit in new_units):
            raise ValueError('source append would overwrite an existing canonical unit key')
        for unit in new_units:
            if 'lexical' not in unit['artifacts']:
                continue
            catalog = _read_fragment(temporary, unit['artifacts']['lexical'])
            for entry in catalog['entries']:
                entry['lowering_route'] = _lowering_route(entry)
            unit['module_initializer_ids'] = [identity for entry in catalog['entries']
                if entry['kind'] == 'MODULE' for identity in entry['semantic_definition_ids'] or ()]
            unit['lexical_completion'] = catalog.get('completion', {})
            unit['lexical_summary'] = catalog['summary']
            unit['artifacts']['lexical'] = _write(temporary, unit['artifacts']['lexical']['path'], catalog)
        step = {'before_count': len(old_basis), 'after_count': len(old_basis) + len(new_units),
            'prior_manifest_sha256': hashlib.sha256(original).hexdigest(),
            'before_identity_sha256': source_units_identity(old_basis),
            'appended_identity_sha256': source_units_identity([input_basis_unit(unit) for unit in new_units]),
            'unit_keys': [unit['key'] for unit in new_units], 'provenance': json.loads(_json(provenance))}
        manifest['units'].extend(new_units)
        manifest['source_expansion'] = {'schema': 'scar.static-source-expansion-ledger',
            'schema_version': 1, 'coverage': 'PRIOR_SOURCE_BASIS_ONLY',
            'steps': [*(ledger['steps'] if ledger is not None else ()), step]}
        manifest['dependency_source_roots'] = list(dict.fromkeys((
            *manifest.get('dependency_source_roots', ()), *(source.namespace_root for source in sources))))
        summary = manifest['summary']
        for unit in new_units:
            for name, number in unit['counts'].items():
                summary[name] = summary.get(name, 0) + number
        summary.update(files=len(manifest['units']), processed_files=len(manifest['units']),
            source_bytes=sum(unit['source_bytes'] for unit in manifest['units']),
            physical_source_files=len({(unit['path'], unit['raw_sha256']) for unit in manifest['units']}),
            statuses=dict(Counter(unit['status'] for unit in manifest['units'])),
            body_statuses=dict(Counter(body['status'] for unit in manifest['units'] for body in unit['bodies'])),
            issues_by_stage=dict(Counter(issue['stage'] for unit in manifest['units'] for issue in unit['issues'])))
        manifest['completion'].update(body_expansion_complete=False, dependency_closure_complete=False,
            whole_program_model_complete=False, language_ledger_complete=False,
            static_rule_ledgers_present=all('rules' in unit['artifacts'] for unit in manifest['units']))
        for name in ('explicit_ast_catalog_complete', 'compiler_symbol_tables_cataloged'):
            if name in manifest['completion']:
                manifest['completion'][name] = all(unit.get('lexical_completion', {}).get(name, False)
                                                   for unit in manifest['units'])
        manifest['storage'] = {'files': len(list(temporary.rglob('*.gz'))),
            'compressed_bytes': sum(path.stat().st_size for path in temporary.rglob('*.gz'))}
        if not _source_expansion_valid(manifest['source_expansion'],
                [input_basis_unit(unit) for unit in manifest['units']]):
            raise ValueError('source append failed its strict prefix/ledger identity check')
        if any(_stamp(temporary / path) != stamp for path, stamp in shared.items()):
            raise ValueError('an original source/body/publication artifact changed during source append')
        (temporary / 'manifest.json').unlink()
        (temporary / 'manifest.json').write_bytes(_json(manifest))
        read_static_program(temporary, verify=False)
        verify_sources()
        read_static_program(output, verify=False)
        if (output / 'manifest.json').read_bytes() != original:
            raise ValueError('current model changed before source append publication')
        previous = Path(tempfile.mkdtemp(prefix='.static-replacing-', dir=output.parent))
        previous.rmdir()
        output.rename(previous)
        try:
            temporary.rename(output)
        except BaseException:
            previous.rename(output)
            raise
        shutil.rmtree(previous)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def expand_static_dependency_program(directory, *, material_indices=None, max_files=4096,
                                     max_bytes=64 * 1024 * 1024, jobs=4, progress=None):
    """Analyze available source candidates, retaining their conditional origins.

    A round consumes the current material inventory and refreshes import and
    material indices. Newly discovered materials remain explicit next-round
    inputs. No runtime import choice or complete dependency closure is claimed.
    """
    from scar.analysis.static_program import (_discard_static_calls,
        _discard_static_dependencies, _json, _lowering_route, _read_fragment, _write,
        read_static_program)
    from scar.analysis.static.dependencies import inventory_static_dependencies, read_static_dependencies
    from scar.analysis.static.dependencies import expand_static_dependency_materials
    from scar.analysis.static.imports import build_import_index
    from scar.analysis.static_body import _stamp
    from scar.ir.record_codec import encode
    from scar.ir.program.providers import ProviderArtifact
    from scar.ir.program.source_references import SourceArtifactLocator

    output = Path(directory).absolute()
    manifest = read_static_program(output, verify=False)
    original = (output / "manifest.json").read_bytes()
    inventory = read_static_dependencies(output)
    if (output / "manifest.json").read_bytes() != original:
        raise ValueError("current model changed while reading dependency inputs")
    started = time.monotonic()
    temporary = Path(tempfile.mkdtemp(prefix=".static-dependency-expansion-", dir=output.parent))
    try:
        shutil.copytree(output, temporary, dirs_exist_ok=True, copy_function=os.link)
        _discard_static_calls(temporary, manifest)
        # Preserve only the actual proof inputs. They are content addressed,
        # owned current-model fragments, not old checkouts or old manifests.
        prior_import = manifest["artifacts"]["imports"]
        import_basis_path = "indexes/dependency-inputs/imports-" + prior_import["sha256"] + ".json.gz"
        import_basis = {**prior_import, "path": import_basis_path}
        destination = temporary / import_basis_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            os.link(temporary / prior_import["path"], destination)
        _read_fragment(temporary, import_basis)
        manifest["artifacts"]["dependency-input-imports-" + prior_import["sha256"]] = import_basis
        input_inventory = replace(inventory, import_index=replace(inventory.import_index,
            artifact=replace(inventory.import_index.artifact, uri=str(output / import_basis_path))))
        basis_document = input_inventory.to_dict()
        basis_key = hashlib.sha256(_json(basis_document)).hexdigest()
        inventory_basis_path = "indexes/dependency-inputs/materials-" + basis_key + ".json.gz"
        inventory_basis = _write(temporary, inventory_basis_path, basis_document)
        manifest["artifacts"]["dependency-input-materials-" + basis_key] = inventory_basis
        shared = {descriptor["path"]: _stamp(temporary / descriptor["path"])
            for unit in manifest["units"]
            for descriptor in (*unit["artifacts"].values(),
                *(body["artifact"] for body in unit["bodies"] if "artifact" in body))}
        expansion = expand_static_dependency_materials(input_inventory, out=temporary,
            material_indices=material_indices, max_files=max_files, max_bytes=max_bytes,
            jobs=jobs, existing_units=manifest["units"], progress=progress)
        old_keys = {unit["key"] for unit in manifest["units"]}
        new_units = [unit for unit in expansion.units if unit["key"] not in old_keys]
        # Existing units and their canonical bodies retain all original IDs.
        # A new namespace interpretation enters the same normal unit schema.
        for unit in new_units:
            if "lexical" not in unit["artifacts"]:
                continue
            catalog = _read_fragment(temporary, unit["artifacts"]["lexical"])
            for entry in catalog["entries"]:
                entry["lowering_route"] = _lowering_route(entry)
            unit["module_initializer_ids"] = [identity for entry in catalog["entries"]
                if entry["kind"] == "MODULE" for identity in entry["semantic_definition_ids"] or ()]
            unit["lexical_completion"] = catalog.get("completion", {})
            unit["lexical_summary"] = catalog["summary"]
            unit["artifacts"]["lexical"] = _write(temporary, unit["artifacts"]["lexical"]["path"], catalog)
        manifest["units"].extend(new_units)
        manifest["dependency_source_roots"] = list(dict.fromkeys((
            *manifest.get("dependency_source_roots", ()), *(unit["namespace_root"] for unit in new_units))))
        summary = manifest["summary"]
        for unit in new_units:
            for key, number in unit["counts"].items():
                summary[key] = summary.get(key, 0) + number
        summary.update(files=len(manifest["units"]), processed_files=len(manifest["units"]),
            source_bytes=sum(unit["source_bytes"] for unit in manifest["units"]),
            physical_source_files=len({(unit["path"], unit["raw_sha256"]) for unit in manifest["units"]}),
            statuses=dict(Counter(unit["status"] for unit in manifest["units"])),
            body_statuses=dict(Counter(body["status"] for unit in manifest["units"] for body in unit["bodies"])),
            issues_by_stage=dict(Counter(issue["stage"] for unit in manifest["units"] for issue in unit["issues"])))
        syntax, kinds, scopes, routes = Counter(), Counter(), Counter(), Counter()
        def lexical(unit):
            if "lexical" not in unit["artifacts"]:
                return {"imports": []}
            catalog = _read_fragment(temporary, unit["artifacts"]["lexical"])
            syntax.update(catalog["syntax"])
            kinds.update(entry["kind"] for entry in catalog["entries"])
            scopes.update(table["kind"] for table in catalog["symbol_tables"])
            routes.update(entry.get("lowering_route", _lowering_route(entry))["status"] for entry in catalog["entries"])
            return catalog
        imports = build_import_index(manifest["units"], lexical)
        _discard_static_dependencies(temporary, manifest)
        manifest["artifacts"]["imports"] = _write(temporary, "indexes/imports.json.gz", imports)
        ledger = {"schema": "scar.static-language-inventory", "schema_version": 1,
            "syntax_occurrences": dict(sorted(syntax.items())), "entry_kinds": dict(kinds),
            "compiler_scope_kinds": dict(scopes), "entry_routes": dict(routes), "language_rules_complete": False,
            "qualification": "Source inventory and entry routes, including conditional dependency candidates; canonical rules remain open."}
        manifest["artifacts"]["language"] = _write(temporary, "indexes/language.json.gz", ledger)
        input_locator = SourceArtifactLocator(ProviderArtifact(str(output / inventory_basis_path),
            inventory_basis["sha256"], inventory_basis["bytes"], input_inventory.SCHEMA),
            inventory_basis["decoded_sha256"], inventory_basis["decoded_bytes"], "gzip-canonical-json")
        report = {"schema": "scar.static-dependency-expansion-index", "schema_version": 1,
            "input_inventory": encode(input_locator), "import_index": encode(input_inventory.import_index),
            "selected_materials": list(expansion.selected_materials), "unit_keys": [unit["key"] for unit in expansion.units],
            "origins": encode(expansion.origins), "frontiers": encode(expansion.frontiers),
            "summary": {**expansion.summary(), "new_units": len(new_units)},
            "qualification": "Original material/import pointers refer to the immutable input basis; units are conditional source interpretations, not selected runtime implementations."}
        report_key = hashlib.sha256(_json(report)).hexdigest()
        manifest["artifacts"]["dependency-expansion-" + report_key] = _write(temporary,
            "indexes/dependency-expansions/" + report_key + ".json.gz", report)
        manifest["dependency_expansion"] = {"analyzer_seconds": time.monotonic() - started,
            "summary": report["summary"], "report": "dependency-expansion-" + report_key}
        for property in ("explicit_ast_catalog_complete", "compiler_symbol_tables_cataloged"):
            manifest["completion"][property] = all(unit.get("lexical_completion", {}).get(property, False)
                                                   for unit in manifest["units"])
        manifest["completion"].update(body_expansion_complete=False, dependency_closure_complete=False,
            whole_program_model_complete=False, language_ledger_complete=False, implicit_scope_semantics_complete=False)
        manifest["catalog_refinement"] = {"updated_stage": "dependency-source-expansion+imports",
            "source_graphs_rebuilt": 0, "new_source_units": len(new_units), "entry_routes": dict(routes),
            "compiler_scope_kinds": dict(scopes), "import_summary": imports["summary"]}
        # Refresh candidate materials against the enlarged source scope. Only
        # URI roots change on publication; exact stored/decoded receipts do not.
        (temporary / "manifest.json").unlink()
        (temporary / "manifest.json").write_bytes(_json(manifest))
        next_inventory = inventory_static_dependencies(temporary, search_roots=inventory.search_roots,
            max_modules=inventory.limits.max_modules, max_files=inventory.limits.max_files,
            max_bytes=inventory.limits.max_bytes, max_file_bytes=inventory.limits.max_file_bytes,
            native_suffixes=inventory.native_suffixes)
        def relocate(locator):
            return replace(locator, artifact=replace(locator.artifact,
                uri=str(output / Path(locator.artifact.uri).relative_to(temporary))))
        next_inventory = replace(next_inventory, import_index=relocate(next_inventory.import_index),
            materials=tuple(replace(material, supplied_source=relocate(material.supplied_source))
                if material.supplied_source is not None else material for material in next_inventory.materials))
        manifest["artifacts"]["dependencies"] = _write(temporary, "indexes/dependencies.json.gz", next_inventory.to_dict())
        manifest["dependency_inventory"] = {"summary": next_inventory.summary(),
            "search_roots": list(next_inventory.search_roots)}
        manifest["completion"]["dependency_inventory_present"] = True
        manifest["storage"] = {"files": len(list(temporary.rglob("*.gz"))),
            "compressed_bytes": sum(path.stat().st_size for path in temporary.rglob("*.gz"))}
        if any(_stamp(temporary / path) != stamp for path, stamp in shared.items()):
            raise ValueError("an original source/body fragment changed during dependency expansion")
        (temporary / "manifest.json").write_bytes(_json(manifest))
        read_static_program(temporary, verify=False)
        read_static_program(output, verify=False)
        if (output / "manifest.json").read_bytes() != original:
            raise ValueError("current model changed before dependency source publication")
        previous = Path(tempfile.mkdtemp(prefix=".static-replacing-", dir=output.parent))
        previous.rmdir()
        output.rename(previous)
        try:
            temporary.rename(output)
        except BaseException:
            previous.rename(output)
            raise
        shutil.rmtree(previous)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
