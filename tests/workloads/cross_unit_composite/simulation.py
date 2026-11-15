"""One array carry with read-only preprocessing across module globals."""
from pipeline import relay as advance, ambiguous_wrapper, cyclic


def simulate(initial, fixed, count):
    for unused in range(count):
        initial = advance(initial, fixed)
    return initial


def ambiguous_loop(initial, fixed, count):
    for unused in range(count):
        initial = ambiguous_wrapper(initial, fixed)
    return initial


def recursive_loop(initial, fixed, count):
    for unused in range(count):
        initial = cyclic(initial, fixed)
    return initial


def changing_input(initial, fixed, count):
    for unused in range(count):
        fixed[...] += 1
        initial = advance(initial, fixed)
    return initial
