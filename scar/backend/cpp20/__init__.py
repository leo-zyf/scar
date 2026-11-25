"""Public, independently loadable B2a and B2b artifact consumers."""

__all__ = ("lower_scheduled", "print_target")


def __getattr__(name):
    if name == "lower_scheduled":
        from .target import lower_scheduled

        return lower_scheduled
    if name == "print_target":
        from .project import print_target

        return print_target
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
