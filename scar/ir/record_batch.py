"""Batch adapters for the unchanged strict record codec.

A batch admits one local tag-policy snapshot. Every wire value still passes
through the original decoder, including Union trials, nested record validation
and live post-init invariants. Neither user records nor validation results are
cached. A later batch admits the supplied mapping again.
"""
from .record_codec import _decode, _record_tags


def decode_records(expected, iterable_of_wire_values, *, record_tags=None):
    """Eagerly return a tuple of fully decoded records under one tag snapshot.

    Tags are validated before consuming the iterable, including an empty one.
    Changes to the original mapping while its values are consumed do not alter
    this batch's policy. Any row failure propagates the original exception;
    no partially decoded tuple is returned.
    """
    tags = _record_tags(record_tags)
    return tuple(_decode(expected, value, tags) for value in iterable_of_wire_values)


__all__ = ["decode_records"]
