"""Pure directory-name, identity and parent facts for conditional normal paths.

Names do not identify inodes. Existing components may be symlinks, so only
fresh directory creation establishes a lexical parent. Graph construction and
operation interpretation remain in the caller of this state domain.
"""

import posixpath


def calls_libc_exit(node, externs):
    """A direct call to the selected C exit(int) has no ordinary return edge."""
    if node.get("op") != "call" or node.get("callee") != "exit":
        return False
    decl = externs.get("exit", {})
    return (
        (decl.get("provider"), decl.get("header"), decl.get("result"))
        == ("libc", "stdlib.h", "void")
        and not decl.get("variadic")
        and [p["type"] for p in decl.get("params", ())] == ["i32"]
        and node.get("parameter_types") == ["i32"]
        and len(node.get("args", ())) == 1
    )


class Directories:
    def __init__(self, create, parent_edge, scope):
        self.create, self.parent_edge, self.scope = create, parent_edge, scope
        self.entries, self.parents, self.labels = {}, {}, {}
        self.generation = 0
        self.root = self.create(
            "filesystem_root", [scope, "root"], None, None, None, (), (None, "/", True)
        )
        self.cwd = self.create(
            "entry",
            [scope, "cwd"],
            None,
            None,
            None,
            (),
            ("invocation.cwd", ".", False),
        )
        self.parents[self.root] = self.root
        self.labels[self.root], self.labels[self.cwd] = (None, "/", True), (
            "invocation.cwd",
            ".",
            False,
        )

    def clone(self):
        other = object.__new__(type(self))
        other.__dict__ = dict(self.__dict__)
        other.entries, other.parents, other.labels = (
            dict(self.entries),
            dict(self.parents),
            dict(self.labels),
        )
        return other

    def _label(self, parent, name, physical=False):
        anchor, path, known = self.labels.get(
            parent, ("directory:" + parent, ".", False)
        )
        return anchor, posixpath.join(path, name), physical and known

    def _parent(self, directory, site, known_only):
        if directory not in self.parents and not known_only:
            identity = self.create(
                "symbolic_parent",
                [self.scope, directory, self.generation],
                None,
                None,
                site,
                (),
                None,
            )
            self.parents[directory] = identity
            self.labels[identity] = ("directory:" + identity, ".", False)
            self.parent_edge(directory, identity, site)
        return self.parents.get(directory)

    def directory(self, path, site=None, *, known_only=False):
        if type(path) is not str or not path or "\0" in path:
            return None
        current = self.root if posixpath.isabs(path) else self.cwd
        for name in path.split("/"):
            if name in {"", "."}:
                continue
            if name == "..":
                current = self._parent(current, site, known_only)
            else:
                key = current, name
                target = self.entries.get(key)
                if target is None and not known_only:
                    label = self._label(current, name)
                    target = self.create(
                        "existing",
                        [self.scope, key, self.generation],
                        None,
                        key,
                        site,
                        (),
                        label,
                    )
                    self.entries[key] = target
                    self.labels[target] = label
                current = target
            if current is None:
                return None
        return current

    def file(self, path, site=None):
        if type(path) is not str or not path or "\0" in path:
            return None
        parent, name = posixpath.split(path)
        if name in {"", ".", ".."}:
            return None
        directory = self.directory(parent or ".", site)
        return (directory, name) if directory is not None else None

    def label(self, key):
        return self._label(*key)[:2]

    def mkdir(self, path, site, *, existing_allowed=False):
        key = self.file(path, site)
        if key is None:
            return None
        if existing_allowed:
            return self.directory(path, site)
        self.generation += 1
        label = self._label(*key, physical=True)
        target = self.create(
            "created",
            [self.scope, key, site, self.generation],
            key[0],
            key,
            site,
            (),
            label,
        )
        self.entries[key] = target
        self.parents[target], self.labels[target] = key[0], label
        return target

    def chdir(self, path, site):
        directory = self.directory(path, site)
        if directory is None:
            self.forget(site, changes_cwd=True)
        else:
            self.cwd = directory
        return self.cwd

    def getcwd(self):
        anchor, path, physical = self.labels[self.cwd]
        return path if physical and anchor is None else None

    def forget(self, site, *, changes_cwd=False):
        self.generation += 1
        self.entries.clear()
        self.parents.clear()
        self.labels = {key: ("directory:" + key, ".", False) for key in self.labels}
        if changes_cwd:
            self.root = self.create(
                "filesystem_root",
                [self.scope, "root-after", site, self.generation],
                None,
                None,
                site,
                (),
                (None, "/", True),
            )
            self.cwd = self.create(
                "entry",
                [self.scope, "cwd-after", site, self.generation],
                None,
                None,
                site,
                (),
                None,
            )
        self.parents[self.root] = self.root
        self.labels[self.root] = (None, "/", True)
        self.labels.setdefault(self.cwd, ("directory:" + self.cwd, ".", False))

    @classmethod
    def join(cls, left, right, site):
        joined = left.clone()
        joined.entries, joined.parents, joined.labels = (
            {},
            {},
            {**left.labels, **right.labels},
        )
        joined.generation = max(left.generation, right.generation) + 1
        remaps, ambiguous = [{}, {}], [set(), set()]

        def remap(index, before, after):
            old = remaps[index].get(before, after)
            if old != after:
                ambiguous[index].add(before)
            remaps[index][before] = after

        def merge(a, b, key=None):
            if a == b:
                result = a
            else:
                parents = [left.parents.get(a), right.parents.get(b)]
                parents = [remaps[i].get(p, p) for i, p in enumerate(parents)]
                parent = parents[0] if parents[0] == parents[1] else None
                label = (
                    left.labels.get(a)
                    if left.labels.get(a) == right.labels.get(b)
                    else None
                )
                result = joined.create(
                    "phi",
                    [joined.scope, site, a, b, key],
                    parent,
                    key,
                    site,
                    (a, b),
                    label,
                )
                joined.labels[result] = label or ("directory:" + result, ".", False)
                if parent is not None:
                    joined.parents[result] = parent
            remap(0, a, result)
            remap(1, b, result)
            return result

        joined.root = merge(left.root, right.root)
        joined.parents[joined.root] = joined.root
        joined.cwd = merge(left.cwd, right.cwd)
        joined.entries.update(
            {
                key: left.entries[key]
                for key in left.entries.keys() & right.entries.keys()
                if left.entries[key] == right.entries[key]
            }
        )

        def names(domain, index):
            result = {}
            for (parent, name), target in domain.entries.items():
                if parent in ambiguous[index]:
                    continue
                key = remaps[index].get(parent, parent), name
                if key in result and result[key] != target:
                    result[key] = None
                else:
                    result[key] = target
            return {key: value for key, value in result.items() if value is not None}

        changed = True
        while changed:
            changed = False
            normalized = [names(left, 0), names(right, 1)]
            for key in normalized[0].keys() & normalized[1].keys():
                if key not in joined.entries:
                    joined.entries[key] = merge(
                        normalized[0][key], normalized[1][key], key
                    )
                    changed = True
        for directory in left.parents.keys() & right.parents.keys():
            a, b = left.parents[directory], right.parents[directory]
            if a == b:
                joined.parents[directory] = a
        for index in range(2):
            for directory in ambiguous[index]:
                remaps[index][directory] = None
        return joined, remaps[0], remaps[1]


class FileCandidates:
    """Guarded provenance, kept separate from definitely current file versions.

    Keys are directory-name identities, never inode identities. A barrier only
    adds a proof obligation; it never promotes one of these records to current.
    """

    def __init__(self):
        self.records = {}

    def clone(self):
        other = type(self)()
        other.records = {version: dict(row) for version, row in self.records.items()}
        return other

    def add(self, version, key, locator, *, invalidated_by=(), requires=()):
        previous = self.records.get(version, {})
        keys = set(previous.get("keys", ()))
        if key is not None:
            keys.add(key)
        self.records[version] = dict(
            version=version,
            locator=locator,
            keys=tuple(sorted(keys)),
            invalidated_by=tuple(
                sorted(set(previous.get("invalidated_by", ())) | set(invalidated_by))
            ),
            requires=tuple(sorted(set(previous.get("requires", ())) | set(requires))),
        )

    def overwrite(self, keys):
        keys = set(keys)
        removed = []
        for version, row in list(self.records.items()):
            if keys.intersection(row["keys"]):
                removed.append(version)
                del self.records[version]
        return removed

    def havoc(self, versions, locators, barrier, *, retain=(), overwritten=()):
        retain, overwritten = set(retain), set(overwritten)
        removed = self.overwrite(overwritten)
        for key, version in versions.items():
            if key not in retain and key not in overwritten:
                self.add(version, key, locators[version])
        for row in self.records.values():
            if not retain.intersection(row["keys"]):
                row["invalidated_by"] = tuple(
                    sorted(set(row["invalidated_by"]) | {barrier})
                )
                row["requires"] = tuple(
                    sorted(
                        set(row["requires"])
                        | {
                            "referenced_barriers_preserve_file_content_and_locator",
                            "candidate_is_current_at_consumer",
                        }
                    )
                )
        return removed

    def rows(self, key=None):
        return [
            {
                "version": row["version"],
                "locator": row["locator"],
                "invalidated_by": list(row["invalidated_by"]),
                "requires": list(row["requires"]),
            }
            for _, row in sorted(self.records.items())
            if key is None or key in row["keys"]
        ]

    @classmethod
    def join(cls, left, right):
        result = cls()
        for source in (left, right):
            for row in source.records.values():
                for key in row["keys"] or (None,):
                    result.add(
                        row["version"],
                        key,
                        row["locator"],
                        invalidated_by=row["invalidated_by"],
                        requires=(
                            *row["requires"],
                            "candidate_available_on_selected_normal_predecessor",
                        ),
                    )
        return result
