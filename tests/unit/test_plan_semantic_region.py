"""Cost KEEP never installs the native artifact built for isolated validation."""

from types import ModuleType, SimpleNamespace

from scar.compiler import plan_semantic_region


def test_keep_with_compiled_validation_keeps_the_live_entry(monkeypatch):
    module = ModuleType('trial_source')

    def original(value):
        return value * 2

    module.entry = original
    pipeline = SimpleNamespace(source=SimpleNamespace(qualname='entry'))
    build = dict(decision='KEEP', native_artifact={'uri': '/unused/artifact.json'})
    monkeypatch.setattr(plan_semantic_region, 'check',
        lambda _: (build, pipeline, None, None, None))

    def unexpected_trial(*args, **kwargs):
        raise AssertionError('installation entered native admission for a KEEP decision')

    monkeypatch.setattr(plan_semantic_region, '_trial', unexpected_trial)
    with plan_semantic_region.install(module, 'result.json', trial=True, exclusive_inputs=True) as entry:
        assert entry is module.entry is original
        assert entry(2) == 4
    assert module.entry is original
