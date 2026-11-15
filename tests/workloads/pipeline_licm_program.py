"""A pure composition exposes fixed preprocessing below two call boundaries."""


def prepare(fixed):
    return fixed * fixed


def combine(payload, prepared):
    return payload + prepared


def wrapper(payload, fixed):
    prepared = prepare(fixed)
    return combine(payload, prepared)


def relay(payload, fixed):
    return wrapper(payload, fixed)


def simulate(initial, fixed, count):
    for unused in range(count):
        initial = relay(initial, fixed)
    return initial


def changing_input(initial, fixed, count):
    for unused in range(count):
        fixed[...] += 1
        initial = relay(initial, fixed)
    return initial


def cyclic(payload, fixed):
    return cyclic(payload, fixed)


def recursive_loop(initial, fixed, count):
    for unused in range(count):
        initial = cyclic(initial, fixed)
    return initial
