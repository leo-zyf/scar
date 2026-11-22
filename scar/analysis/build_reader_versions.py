"""Exact reader-only transitions for unchanged published graph schemas."""

# Change these entries only after replaying the old and current readers.
BUILD_TRANSITION = {
    'build.py': (
        'ffe32e2f9f15a2a7ba652309dbda5f71b7c5f35f8b9cc2bc5bef1925958b3425',
        '7365d3d10936b00fb6970277e566d9ea384195e5eb5106849ca8c9e500b8b4ef',
    ),
    'build_f4.py': (
        '31196fab20f5f75f335e7f23935df33e495aecf2b8da7ac6063faa1cedb0b525',
        '1370a359755aba7655a986ff0a33d14dda481be5beebb8b0a3760874353850f5',
    ),
    'source_call_binding_view.py': (
        '00bcffb9b9ab02e03275c6221b938d12028865036db2870f26d63b84b68ece44',
        '1ac363cc7db0c4bb04a8fc17923947dd1bc61136f5c582c055141c60633babcd',
    ),
}

_BUILD_PREDECESSOR = {
    'build.py': '14ebe380a60123229f410a817b650412b1eb2d51331665abf867adc1da444042',
    'build_f4.py': 'a77718605d3c7f399b4f85dfdb047c0d07cc69f62e88bdfff20cb3b544251c24',
    'build_reader_versions.py': 'a76a12c4ca908ce080bfaa9ff76684ce379e2f07dac008468ed57ff7f26cdb61',
}

# Observation state publication and lazy replay share one owner.
# The pre-merge producer and reader remain reproducible at fc594dc8.
_OBSERVED_STATE_PREDECESSOR = {
    'build.py': 'a846d76fb8b5cd284657d2c7623172cd4fb5037fd71e246261eea7aef98de8e2',
    'build_f4.py': 'e1c7aa6f147cb725d9ede6aa100869e8e717f4f7d569efa56d642d9d53ed7608',
    'build_reader_versions.py': '4c887af0a0d14929e470033d8eb559654fc91d2fa281a66849302c36114e12e6',
    'source_observed_state.py': '3fa70ecbf068bb13da1c022bbc28b704532ac2d38b4f228f6b8caf9f7afb83a5',
    'source_observed_state_view.py': 'e9a68375d863f5630b148a7c70d0ae0c6e636abe487a9e758d2180c575d2dea0',
}
_OBSERVED_STATE_READER = {
    'build.py': 'ba6bd5372e4387b5d3a6a3920304ad2041e0df4c6e9f9f54035694cb2ef860de',
    'build_f4.py': 'bf3851276afecb329271f28f9cceae4a5b5b4e1ef4938ecedda16c9472e97ca7',
    'source_observed_inventory_state.py': 'b669edfe93a007af019fb6b33ae4781f946d64f8108f42d2b4d83799326dc110',
}

# F4 state inventory and fixed-point storage ownership was consolidated.
STATE_OWNER_READER = {
    'build.py': 'b6b875e2246f7713109f47dffa65153a7be838ee6138800265afdfa33a48d78c',
    'build_f4.py': '1a6146a97b844f339367f1dabb5b0bd3f633ed47201cea67b8b66873df40d32e',
    'program_state_view.py': 'db1afacc5ca01a0b4d16c460998961f224ffe41af45e258f306cef8d5837500a',
    'program_state_coverage.py': 'fa8049c68ab8954fefa372cf6ba8b4ecf9546068c14b3ba9826b3c04ce2c9d82',
    'program_inventory.py': '9a3d0ef3b490dd0a4badae31f4054a84c521650fb10372db68a07b1937fdd0f8',
    'source_state.py': 'f30566d4767255568191d24a49cc78eb4027d9ae84bf58e372906c9f060adf66',
}
STATE_OWNER_PREDECESSOR = {
    'build.py': 'ba6bd5372e4387b5d3a6a3920304ad2041e0df4c6e9f9f54035694cb2ef860de',
    'build_f4.py': 'bf3851276afecb329271f28f9cceae4a5b5b4e1ef4938ecedda16c9472e97ca7',
    'program_state_view.py': '77b487f00caf55bea21705e089b706726f6df527059a309bf6e20d2f86867f7c',
    'program_state_coverage.py': '67162cb4cffe1fffd5f677a18e20177c510c114b6a63fc7f12852f4dfd633a04',
    'program_inventory.py': '915fccd69f2b25b44a6ec415c9d6d448127c567bdcabbbda84cb42ad37a90de3',
    'source_state.py': 'd564e50e9abe1edd4cfed445f18c3fe98d433199f01565bbac6323f8a7259344',
    'binding_track_storage.py': 'c4e0f9b0464dc6988f0122a8b087f646e0223a735a464e87700d4278ddd1adff',
    'source_observed_inventory_state.py': 'b669edfe93a007af019fb6b33ae4781f946d64f8108f42d2b4d83799326dc110',
    'build_reader_versions.py': 'c1b9e453e68a89ffc5af8888c732de8944454ba5a8aba25bf75afadaf6a3c96a',
}

LOOP_TRANSFER_TRANSITION = {
    'storage': (
        'bbe9c5ba441b19ff17b05aac9d7f01ea3fd19e9c3dacc9ca5b9ea73ccedaa4aa',
        'f944847204af869de1cc6407718c20d6c003704e86cecb7014d4c3f7b7a4fea1',
    ),
}

CONTROLLED_TRANSITION = {
    'compiler/plan_controlled_region.py': (
        '2d10825e12e2bd6aa099b7b6d3d9bde356ae0ce34119e3336eec43b5bcbe0ec3',
        '41bec4712b4d5e23fc2d633d0a4de99b5f432b2e1602ad6490d27d4fb4ad4686',
    ),
    'analysis/bounded_entry_calls.py': (
        'b7e19d362cf63f765b7c7c9357ec294a0ed699994481b6f51306f30005973523',
        '7d9aa45e1aecb5405b437e617ecede2724cef9b37d8175cef4b7a344424362e3',
    ),
    'compiler/admission_context.py': (
        'f8b57fec159b78a18ce29d3b8b57698a6c549de11a05c12bf45ebf420b70d60a',
        '40b4ee568bc30e1633cd38bf82ff4040fd7fd3d1db6b025654153b45c8242fde',
    ),
}

_ENTRY_PRODUCERS = (('b7e19d362cf63f765b7c7c9357ec294a0ed699994481b6f51306f30005973523', 34950),
                    ('7d9aa45e1aecb5405b437e617ecede2724cef9b37d8175cef4b7a344424362e3', 35189))

def compatible_pins(recorded, active, transition, *, added=()):
    """Accept an exact predecessor alongside current immutable reader pins.

    Only listed files can differ. Earlier data receipts are still verified
    and their semantics are rederived by the active reader.
    """
    if recorded == active:
        return True
    if type(recorded) is not dict:
        return False
    if any(active.get(name) != new for name, (_, new) in transition.items()):
        return False
    previous = {name: digest for name, digest in active.items() if name not in added}
    previous.update({name: old for name, (old, _) in transition.items()})
    return recorded == previous


def _previous_build_reader_matches(recorded, active):
    if compatible_pins(recorded, active, BUILD_TRANSITION, added=('build_reader_versions.py',)):
        return True
    return (all(active.get(name) == new for name, (_, new) in BUILD_TRANSITION.items())
            and recorded == dict(active, **_BUILD_PREDECESSOR))


def _compatible_observed_build_pins(recorded, active):
    if _previous_build_reader_matches(recorded, active):
        return True
    if any(active.get(name) != digest for name, digest in _OBSERVED_STATE_READER.items()):
        return False
    previous = {name: digest for name, digest in active.items() if name != 'source_observed_inventory_state.py'}
    previous.update(_OBSERVED_STATE_PREDECESSOR)
    return _previous_build_reader_matches(recorded, previous)


def compatible_build_pins(recorded, active):
    if _compatible_observed_build_pins(recorded, active):
        return True
    if any(active.get(name) != digest for name, digest in STATE_OWNER_READER.items()):
        return False
    predecessor = dict(active, **STATE_OWNER_PREDECESSOR)
    return _compatible_observed_build_pins(recorded, predecessor)


def compatible_entry_producer(recorded, active):
    if recorded == active:
        return True
    old, current = _ENTRY_PRODUCERS
    if (active['sha256'], active['size_bytes']) != current:
        return False
    return recorded == {**active, 'sha256': old[0], 'size_bytes': old[1]}
