"""Small pure array operations; the environment-controlled export is ambiguous."""
import os


def prepare(fixed):
    return fixed * fixed


def alternate(fixed):
    return fixed + fixed


def combine(payload, prepared):
    return payload + prepared


if os.environ.get('SCAR_CROSS_UNIT_CHOICE'):
    chosen_prepare = prepare
else:
    chosen_prepare = alternate
