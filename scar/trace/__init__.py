__all__ = ["load", "TraceSession"]


def __getattr__(name):
    # Code identity and selective capture are also used by deployment guards.
    # Importing a trace submodule must not initialize the recorder/whole IR.
    if name == 'load':
        from scar.trace.reader import load
        globals()[name] = load
        return load
    if name == 'TraceSession':
        from .session import TraceSession
        globals()[name] = TraceSession
        return TraceSession
    raise AttributeError(name)
