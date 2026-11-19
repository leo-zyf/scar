"""Exact reader-only transitions for unchanged published graph schemas."""

# Change these entries only after replaying the old and current readers.
BUILD_TRANSITION = {
    'build.py': (
        'ffe32e2f9f15a2a7ba652309dbda5f71b7c5f35f8b9cc2bc5bef1925958b3425',
        '14ebe380a60123229f410a817b650412b1eb2d51331665abf867adc1da444042',
    ),
    'build_f4.py': (
        '31196fab20f5f75f335e7f23935df33e495aecf2b8da7ac6063faa1cedb0b525',
        'a77718605d3c7f399b4f85dfdb047c0d07cc69f62e88bdfff20cb3b544251c24',
    ),
    'source_call_binding_view.py': (
        '00bcffb9b9ab02e03275c6221b938d12028865036db2870f26d63b84b68ece44',
        '1ac363cc7db0c4bb04a8fc17923947dd1bc61136f5c582c055141c60633babcd',
    ),
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


def compatible_entry_producer(recorded, active):
    if recorded == active:
        return True
    old, current = _ENTRY_PRODUCERS
    if (active['sha256'], active['size_bytes']) != current:
        return False
    return recorded == {**active, 'sha256': old[0], 'size_bytes': old[1]}
