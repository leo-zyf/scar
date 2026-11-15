"""Small array recurrence used to exercise source-derived loop placement."""


def advance(payload, fixed):
    prepared = fixed * fixed
    return payload + prepared


def simulate(initial, fixed, count):
    for unused in range(count):
        initial = advance(initial, fixed)
    return initial


def changing_input(initial, fixed, count):
    for unused in range(count):
        fixed[...] += 1
        initial = advance(initial, fixed)
    return initial


def index_dependent(initial, fixed, count):
    for index in range(count):
        initial = advance(initial, fixed) + index
    return initial
