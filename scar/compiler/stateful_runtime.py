"""Compatibility facade for the shared lightweight stateful boundary guards."""
from contextlib import contextmanager

from scar.runtime import stateful_guard as _guard
from scar.runtime.native_stateful import GuardFailure as _RuntimeGuardFailure
from scar.runtime.stateful_guard import (_ordinary, _field,
    _input, _lookup, _function_state, _same_function_state)
from scar.compiler.runtime import SourceMismatch


def _function(source, value):
    try:
        return _guard._function(source, value)
    except _guard.SourceMismatch as error:
        raise SourceMismatch(str(error)) from error


class _ArtifactAdapter:
    def __init__(self, artifact, selection):
        from scar.compiler.stateful_model import validate_stateful_region
        from scar.compiler.export_stateful import project_stateful_contract
        validate_stateful_region(artifact.region)
        self.original = artifact
        self.contract = project_stateful_contract(artifact, selection, require_selection=False)
        self.report = artifact.report

    def run(self, arrays, scalars):
        from scar.codegen.stateful import GuardFailure
        try:
            return self.original.run(arrays, scalars)
        except GuardFailure as error:
            raise _RuntimeGuardFailure(str(error)) from error


def guarded_stateful_function(original, artifact, *, owner=None,
                              exclusive_inputs=False, selection=None, trial=False):
    adapter = artifact if hasattr(artifact, 'contract') else _ArtifactAdapter(artifact, selection)
    try:
        wrapper = _guard.guarded_stateful_function(original, adapter, owner=owner,
            exclusive_inputs=exclusive_inputs, selection=selection, trial=trial)
    except _guard.SourceMismatch as error:
        raise SourceMismatch(str(error)) from error
    wrapper.scar_artifact = artifact
    return wrapper


@contextmanager
def install_stateful_regions(module, artifacts, *, exclusive_inputs=False,
                             selection=None, trial=False):
    artifacts = list(artifacts)
    adapters = [artifact if hasattr(artifact, 'contract') else _ArtifactAdapter(artifact, selection)
                for artifact in artifacts]
    try:
        with _guard.install_stateful_regions(module, adapters, exclusive_inputs=exclusive_inputs,
                                            selection=selection, trial=trial) as wrappers:
            for wrapper, artifact in zip(wrappers, artifacts):
                wrapper.scar_artifact = artifact
            yield wrappers
    except _guard.SourceMismatch as error:
        raise SourceMismatch(str(error)) from error


__all__ = ['SourceMismatch', 'guarded_stateful_function', 'install_stateful_regions']
