"""Pure directory-name, identity and parent facts for conditional normal paths.

Names do not identify inodes. Existing components may be symlinks, so only
fresh directory creation establishes a lexical parent. Graph construction and
operation interpretation remain in the caller of this state domain.
"""

import posixpath


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
