"""Exact controlled-planner transitions for the source summary readers."""

from scar.analysis.build_reader_versions import CONTROLLED_TRANSITION, compatible_pins


_BINDING = 'compiler/source_summary_binding.py'
_BINDING_PREDECESSOR = ('1f1c1ef4a50c21f2cdbebcb435b2f7d48ea319e1b008e49baebb73ec5c9cba7d', 9943)
_BINDING_CURRENT = ('910a8126ee3f5b2f08c0722982fe1a39369db14f0b028c114bebe34fdc7ff1be', 10917)
_TRANSITION = {
    'compiler/plan_controlled_region.py': (
        '41bec4712b4d5e23fc2d633d0a4de99b5f432b2e1602ad6490d27d4fb4ad4686',
        '69a8e4b024f514baa25f16145fded08ff6c11507e5c48fb55acaa9ae57752a71',
    ),
    _BINDING: (_BINDING_PREDECESSOR[0], _BINDING_CURRENT[0]),
}
_STORAGE = 'analysis/source_summary_storage.py'
_STORAGE_CURRENT = '7ab3f63b08a29c471a431e4e3e823f901753f0d62ed21b4340893e903d0a493f'
_ADDED = (_STORAGE, 'compiler/source_summary_reader_versions.py')


def compatible_plan_pins(recorded, active):
    if recorded == active:
        return True
    if (active.get(_STORAGE) != _STORAGE_CURRENT
            or any(active.get(name) != new for name, (_, new) in _TRANSITION.items())):
        return False
    previous = {name: digest for name, digest in active.items() if name not in _ADDED}
    previous.update({name: old for name, (old, _) in _TRANSITION.items()})
    return compatible_pins(recorded, previous, CONTROLLED_TRANSITION)


def compatible_binding_producer(recorded, active):
    if recorded == active:
        return True
    if (active['sha256'], active['size_bytes']) != _BINDING_CURRENT:
        return False
    old_sha, old_size = _BINDING_PREDECESSOR
    return recorded == {**active, 'sha256': old_sha, 'size_bytes': old_size}
