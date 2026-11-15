"""Ordinary from-import aliases transport function references across files."""
from scalar_ops import prepare as fixed_part, combine as join, chosen_prepare as uncertain


def wrapper(payload, fixed):
    prepared = fixed_part(fixed)
    return join(payload, prepared)


def relay(payload, fixed):
    return wrapper(payload, fixed)


def ambiguous_wrapper(payload, fixed):
    prepared = uncertain(fixed)
    return join(payload, prepared)


def cyclic(payload, fixed):
    return cyclic(payload, fixed)
