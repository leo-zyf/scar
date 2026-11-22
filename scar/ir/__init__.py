"""Public execution and source-model records, resolved from their owning module."""

from importlib import import_module


_PUBLIC_MODULES = {
    'ids': ('CodeID', 'InvocationID', 'LogicalVersion', 'ObjectID', 'StorageID'),
    'effects': ('ActionLabel', 'Effect', 'Knowledge'),
    'entities': ('Materialization', 'ObjectEntity', 'Region', 'classify_region_overlap', 'tensor_entity'),
    'events': ('Event', 'Opportunity'),
    'proofs': ('ProofObligation', 'ProofStatus', 'proof', 'proof_summary'),
    'graph': ('ExecutionGraph', 'GraphEdge', 'GraphNode', 'NodeKind', 'ProgramGraph'),
    'correspondence': ('SourceCorrespondence', 'add_correspondences', 'link_events', 'summarize_correspondence'),
    'static': ('from_project', 'from_source'),
    'versions': ('StorageVersionRegistry', 'default_registry', 'mark_external_write', 'mark_storage_write',
                 'storage_version'),
}
_EXPORTS = {name: module for module, names in _PUBLIC_MODULES.items() for name in names}
__all__ = list(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f'{__name__!r} has no attribute {name!r}')
    value = getattr(import_module(f'.{module}', __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(globals().keys() | _EXPORTS.keys())
