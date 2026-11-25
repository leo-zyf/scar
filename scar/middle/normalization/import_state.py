"""Owned import-state dataflow, independent of protocol compilation and guards.

Snapshots copy their containers. Borrowed deferred markers and operation records
retain identity so a guard later emitted at its original source site stays shared.
No compiler, frame, evaluation or error-reporting service enters this component.
"""

from scar.contract import json_bytes


def namespace_slot(module, name):
    return f"py:{module}:{name}"


class ImportState:
    def __init__(self, namespaces=(), *, main_namespace=None):
        self._namespaces = frozenset(namespaces)
        self._main_namespace = main_namespace
        self._registry = {}
        self._namespace_writes = {}
        self._binding_versions = {}
        self._windows = {}
        self._epoch = 0

    def registry_key(self, namespace):
        return "__main__" if namespace == self._main_namespace else namespace

    def namespace_owner(self, registry_key):
        if registry_key == "__main__" and self._main_namespace is not None:
            return self._main_namespace
        return registry_key

    def registry_entry(self, key):
        """Borrow a proof record; updates replace it through bind_registry."""
        return self._registry.get(key, {})

    def bind_registry(self, key, proof):
        self._registry[key] = proof

    def windows(self, key):
        return tuple(self._windows.get(key, ()))

    @staticmethod
    def normal_identity(entry, windows):
        """Identity now, conditional on success at each covered effect site."""
        if entry.get("state") in {"initializing", "ready"}:
            return entry
        if entry.get("state") == "unresolved" and windows:
            return entry.get("on_preservation")

    def bind_windows(self, key, windows, expected):
        self._windows[key] = list(windows)
        entry = self._registry[key]
        if entry["state"] == "unresolved":
            self._registry[key] = dict(entry, on_preservation=expected)

    def drop_windows(self, key):
        self._windows.pop(key, None)

    def invalidate_entry(self, key, change):
        previous = self._registry.get(key, {})
        self._registry[key] = change
        self.drop_windows(key)
        if (
            previous.get("state") != "absent"
            and self.namespace_owner(key) in self._namespaces
        ):
            self._namespace_writes[key] = change

    def namespace_change(self, key):
        return self._namespace_writes.get(key)

    def binding_version(self, slot):
        return self._binding_versions.get(slot, 0)

    def next_epoch(self):
        self._epoch += 1
        return self._epoch

    def binding_written(self, slot):
        self._binding_versions[slot] = self.next_epoch()

    def snapshot(self):
        return self.copy_snapshot(
            dict(
                binding_epoch=self._epoch,
                registry=self._registry,
                namespace_writes=self._namespace_writes,
                binding_versions=self._binding_versions,
                import_windows=self._windows,
            )
        )

    def restore(self, snapshot):
        state = self.copy_snapshot(snapshot)
        self._registry = state["registry"]
        self._namespace_writes = state["namespace_writes"]
        self._binding_versions = state["binding_versions"]
        self._windows = state["import_windows"]
        # Restoring an older branch must never reuse an epoch already issued
        # on another analyzed path.
        self._epoch = max(self._epoch, state["binding_epoch"])

    def invalidate(self, owner, source):
        change = dict(
            state="unresolved", epoch=self.next_epoch(), owner=owner, **source
        )
        for key in self._registry:
            # Unknown contents still have a version. Another possible write
            # advances it, including a previously absent registry entry.
            self.invalidate_entry(key, change)
        return change

    @staticmethod
    def copy_snapshot(state):
        return dict(
            binding_epoch=state.get("binding_epoch", 0),
            registry=state["registry"].copy(),
            namespace_writes=state["namespace_writes"].copy(),
            binding_versions=state.get("binding_versions", {}).copy(),
            import_windows={
                name: list(windows) for name, windows in state["import_windows"].items()
            },
        )

    @staticmethod
    def _unresolved(records, reason):
        states, initializers = set(), set()
        for record in records:
            states.update(record.get("possible_states", [record["state"]]))
            initializers.update(record.get("initializers", ()))
            if record.get("initializer"):
                initializers.add(record["initializer"])
        return dict(
            state="unresolved",
            reason=reason,
            possible_states=sorted(states),
            initializers=sorted(initializers),
        )

    @staticmethod
    def join_snapshots(states):
        result = dict(
            binding_epoch=max(state.get("binding_epoch", 0) for state in states),
            registry={},
            namespace_writes={},
            binding_versions={},
            import_windows={},
        )
        for name in set().union(*(state["registry"] for state in states)):
            incoming = [
                state["registry"].get(name, {"state": "absent"}) for state in states
            ]
            if all(record["state"] == "absent" for record in incoming):
                continue
            result["registry"][name] = (
                incoming[0]
                if all(record == incoming[0] for record in incoming[1:])
                else ImportState._unresolved(incoming, "normal_control_join")
            )
        for name in set().union(*(state["namespace_writes"] for state in states)):
            incoming = [state["namespace_writes"].get(name) for state in states]
            result["namespace_writes"][name] = (
                incoming[0]
                if incoming[0] is not None
                and all(record == incoming[0] for record in incoming[1:])
                else dict(
                    state="unresolved",
                    reason="namespace_may_change_on_normal_path",
                    epoch=max((record or {}).get("epoch", 0) for record in incoming),
                )
            )
        common = set.intersection(
            *(set(state.get("binding_versions", {})) for state in states)
        )
        for name in common:
            result["binding_versions"][name] = min(
                state["binding_versions"][name] for state in states
            )
        for name, proof in result["registry"].items():
            histories = [state["import_windows"].get(name, ()) for state in states]
            if not any(histories):
                continue
            identities = [
                ImportState.normal_identity(state["registry"].get(name, {}), history)
                for state, history in zip(states, histories)
            ]
            expected = identities[0]
            if (
                expected is None
                or any(value != expected for value in identities)
                or any(
                    state["namespace_writes"].get(name) and not history
                    for state, history in zip(states, histories)
                )
            ):
                # A guard reached through another arm cannot account for an
                # uncovered mutation or a different module lifetime here.
                result["namespace_writes"][name] = dict(
                    state="unresolved",
                    reason="uncovered_effect_on_normal_path",
                    epoch=result["binding_epoch"],
                )
                continue
            windows, seen = [], set()
            for history in histories:
                for marker, operation, before, epoch in history:
                    key = (id(marker), id(operation), json_bytes(before), epoch)
                    if key not in seen:
                        seen.add(key)
                        windows.append((marker, operation, before, epoch))
            if windows:
                result["import_windows"][name] = windows
                if proof["state"] == "unresolved":
                    result["registry"][name] = dict(proof, on_preservation=expected)
        return result

    def exceptional_prefix(self, start, after, *, may_write):
        imports = self.join_snapshots([start, after])
        if may_write:
            for module in set(start["registry"]) | set(after["registry"]):
                incoming = [
                    state["registry"].get(module, {"state": "absent"})
                    for state in (start, after)
                ]
                imports["registry"][module] = self._unresolved(
                    incoming, "exceptional_import_prefix"
                )
                imports["registry"][module]["possible_states"] = [
                    "absent",
                    "initializing",
                    "ready",
                    "unresolved",
                ]
                imports["import_windows"].pop(module, None)
                if self.namespace_owner(module) in self._namespaces:
                    imports["namespace_writes"][module] = dict(
                        state="unresolved",
                        reason="exceptional_import_prefix",
                        epoch=self.next_epoch(),
                    )
        return imports

    @staticmethod
    def project_final(path, before, after):
        result = ImportState.copy_snapshot(path)
        for domain in (
            "registry",
            "namespace_writes",
            "binding_versions",
            "import_windows",
        ):
            prior, current = before[domain], after[domain]
            for key in prior.keys() | current.keys():
                if key in prior and key in current and prior[key] == current[key]:
                    continue
                if key in current:
                    result[domain][key] = (
                        list(current[key])
                        if domain == "import_windows"
                        else current[key]
                    )
                else:
                    result[domain].pop(key, None)
        return result
