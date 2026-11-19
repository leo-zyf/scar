"""Index the existing canonical JSON plan arrays for repeated source lookups."""

from array import array
from collections import OrderedDict
from hashlib import sha256
import json


_PREFIX = '{"records":['
_SUFFIX = ']}\n'


class PlanRows:
    """A single wire document and its row offsets, with no retained JSON tree."""

    def __init__(self, decoded):
        self.text = text = decoded.decode('ascii')
        self.decoder = json.JSONDecoder()
        self.offsets = offsets = array('Q')
        if not text.startswith(_PREFIX):
            raise ValueError('expected an original canonical plan table')
        position = len(_PREFIX)
        if not text.startswith(_SUFFIX, position):
            while position < len(text):
                start = position
                row, position = self.decoder.raw_decode(text, start)
                if type(row) is not dict:
                    raise ValueError('compiled plan table must contain object records')
                offsets.append(start)
                if position == len(text) or text[position] != ',':
                    break
                position += 1
        if text[position:] != _SUFFIX:
            raise ValueError('compiled plan table has an invalid record boundary')

    @property
    def byte_cost(self):
        return len(self.text) + self.offsets.itemsize * len(self.offsets)

    def row(self, line):
        if type(line) is not int or not 1 <= line <= len(self.offsets):
            raise ValueError('compiled plan ordinal is outside the original table')
        return self.decoder.raw_decode(self.text, self.offsets[line-1])[0]


class PlanRowCache:
    """Bound retained wire bytes to one admitted catalog's working set."""

    def __init__(self, limit=64 * 1024**2):
        self.limit = limit
        self.bytes = 0
        self.tables = OrderedDict()

    def load(self, key, descriptor, read):
        table = self.tables.get(key)
        if table is not None:
            self.tables.move_to_end(key)
            return table
        decoded = read()
        if (len(decoded) != descriptor['decoded_bytes']
                or sha256(decoded).hexdigest() != descriptor['decoded_sha256']):
            raise ValueError('indexed compiled plan table differs from the original receipt')
        table = PlanRows(decoded)
        cost = table.byte_cost
        if cost <= self.limit:
            while self.bytes + cost > self.limit:
                _, removed = self.tables.popitem(last=False)
                self.bytes -= removed.byte_cost
            self.tables[key] = table
            self.bytes += cost
        return table
