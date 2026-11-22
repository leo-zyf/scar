"""Exact predecessor pins for independently replayed ordered publications."""
from scar.analysis.build_reader_versions import compatible_pins


# The immediately preceding owners are the only changed producer files.
# The complete remaining rule map and every original typed table are checked.
_LIFETIME_TRANSITION = {
    'producer': ('b1e0bf529585c730e3fe24cf29daeadb0d7d03247ac3cdc17442da26d5f6e455',
                 '58460a40967c25abd1ff2ba6fec562465def57f4c0b5416168db448bd11f152a'),
}
_TRANSFER_TRANSITION = {
    'storage': ('5ac193f0b47dc71288ecc4e91b0aaa3cf2915da02770001da57e523d9cb50977',
                '3f77dc95bcb4609584213055ec1275e5251055d9f7cbc027a56025362e8fdce8'),
    'lifetime_replayer': ('e5e8bc0eb11b384745c01de882baa33c496f6486dd30914e436a5755d48975d1',
                          '3f77dc95bcb4609584213055ec1275e5251055d9f7cbc027a56025362e8fdce8'),
}


def compatible_lifetime_pins(recorded, active):
    return compatible_pins(recorded, active, _LIFETIME_TRANSITION)


def compatible_transfer_pins(recorded, active):
    if recorded == active:
        return True
    if type(recorded) is not dict or set(recorded) != set(active):
        return False
    return (recorded['rules'] != active['rules'] and recorded['parent_lifetime'] != active['parent_lifetime']
            and compatible_pins(recorded['rules'], active['rules'], _TRANSFER_TRANSITION)
            and compatible_lifetime_pins(recorded['parent_lifetime'], active['parent_lifetime']))
