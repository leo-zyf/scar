from hashlib import sha256
from types import CodeType

import pytest

from scar.analysis.fixed_grid_entry_abi import derive_fixed_grid_file_entry
from scar.trace.code_snapshot import code_key


_SOURCE = '''import json as js
from pathlib import Path as File
import sys
import torch as tensor
from torchdiffeq import odeint as integrate

def rate(previous, parameter):
    return tensor.mm(previous, parameter)

def cli():
    tensor.set_num_threads(1)
    tensor.set_num_interop_threads(1)
    count = int(sys.argv[1])
    destination = File(sys.argv[2])
    root = File(sys.argv[3])
    width = 3
    names = ('initial', 'param')
    state, *constants = [tensor.from_file(str(root / (name + '.raw')), False,
                        width * width, dtype=tensor.float32).view(width, width) for name in names]
    time = tensor.tensor([0.0, count / 16], dtype=tensor.float64)
    solution = integrate(lambda _t, value: rate(value, *constants), state, time,
                         method='euler', options={'step_size': 1.0 / 16.0})
    final = solution[-1].contiguous()
    destination.write_bytes(final.numpy().tobytes())
    print(js.dumps({'z': 'euler', 'size': list(final.shape), 'iterations': count}, sort_keys=True))

if __name__ == '__main__':
    cli()
'''


def _source(text, path):
    raw = text.encode()
    code = next(item for item in compile(raw, str(path), 'exec', dont_inherit=True).co_consts
                if type(item) is CodeType and item.co_qualname == 'rate')
    declaration = dict(source=dict(path=str(path), size_bytes=len(raw), sha256=sha256(raw).hexdigest()),
        function='rate', torch_global='tensor', parameters=['previous', 'parameter'], input_order=[1, 0],
        code_key=code_key(code))
    return declaration, raw


def test_fixed_grid_file_entry_replays_original_ports_and_publishes_in_source_order(tmp_path):
    source, raw = _source(_SOURCE, tmp_path / 'subject.py')
    abi = derive_fixed_grid_file_entry(source, 'cli', source_bytes=raw)
    assert abi['argv'] == {'input_root': 3, 'steps': 1, 'output': 2, 'count': 3}
    assert abi['solver']['state_input_index'] == 1
    assert [row['path'] for row in abi['inputs']] == ['param.raw', 'initial.raw']
    assert [row['load_order'] for row in abi['inputs']] == [1, 0]
    assert abi['publication']['stdout'] == [dict(key='iterations', kind='STEP_COUNT'),
        dict(key='size', kind='FINAL_SHAPE'), dict(key='z', kind='LITERAL', value='euler')]
    assert dict(source='file', target='stdout', kind='ORDER') in abi['graph']['edges']
    with pytest.raises(ValueError, match='differs from the admitted RHS source'):
        derive_fixed_grid_file_entry(source, 'cli', source_bytes=raw + b'\n')


@pytest.mark.parametrize(('rewrite', 'detail'), [
    (lambda text: text.replace('    time =', '    print("in between")\n    time ='), 'unqualified setup'),
    (lambda text: text.replace('    time =', '    unused = int(sys.argv[4])\n    time ='), 'distinct argument parses'),
    (lambda text: text.replace('width * width,', '9.0,'), 'file read extent'),
])
def test_fixed_grid_file_entry_refuses_unknown_normal_completion_effects(tmp_path, rewrite, detail):
    source, raw = _source(rewrite(_SOURCE), tmp_path / 'subject.py')
    with pytest.raises(ValueError, match=detail):
        derive_fixed_grid_file_entry(source, 'cli', source_bytes=raw)
