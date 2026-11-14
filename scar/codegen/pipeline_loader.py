"""Load only admitted pipeline artifact families through their own readers."""
import json
from pathlib import Path


def load(path):
    path = Path(path).absolute()
    metadata = json.loads((path / 'artifact.json').read_bytes())
    schema = metadata.get('schema')
    if schema == 'scar.native-array-pipeline':
        from .pipeline import load as read
    elif schema == 'scar.native-fused-pipeline-artifact':
        from .fused_pipeline import load as read
    else:
        raise ValueError('unsupported native pipeline artifact schema')
    return read(path)
