"""Observe the original filled-contour boundary and its exact drawing uses."""

from collections import defaultdict
from contextlib import contextmanager
import json
from pathlib import Path
import struct
import sys

from scar.ir.execution_recipe import artifact_reference
from scar.trace.code_snapshot import code_key

SCHEMA = 'scar.contour-scene-observation'
INPUT_MAGIC = b'SCARCTI1'
OUTPUT_MAGIC = b'SCARCTO1'
_CAP = 64_000_000


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _enum(value):
    return str(value).split('.')[-1]


class ContourProbe:
    """Capture dimensions, operands and normal results at the existing C++ provider."""

    def __init__(self, root, numpy, project):
        self.root = root
        self.np = numpy
        self.project = project
        self.sources = {}
        self.paths = []
        self.events = []
        self.active = None
        self.normalized = None

    @contextmanager
    def instrument(self):
        import contourpy
        from contourpy import _contourpy
        from matplotlib.contour import ContourSet, QuadContourSet

        original = contourpy._class_lookup['mpl2014']
        fill = original.create_filled_contour
        paths = ContourSet._make_paths_from_contour_generator
        normalize = QuadContourSet._contour_args
        self.source = dict(code_key=code_key(paths.__code__), qualname=paths.__qualname__,
                           material=_ref(paths.__code__.co_filename, 'scar.python-source'))
        self.binary = _ref(_contourpy.__file__, 'scar.native-provider-library')
        owner = self

        class Constructor:
            def __getattr__(self, name):
                return getattr(original, name)

            def __call__(self, x, y, z, mask, **kwargs):
                actual = original(x, y, z, mask, **kwargs)
                lineage = owner._lineage(x, y, z)
                if lineage is None and owner.normalized is not None:
                    vertices, binding = owner.normalized
                    if all(owner._same_values(a, b)
                           for a, b in zip(vertices, (x, y, z), strict=True)):
                        lineage = binding
                owner.normalized = None
                owner.sources[id(actual)] = (actual, x, y, z, mask, kwargs, lineage)
                return actual

        def observe_normalize(chart, args, kwargs):
            result = normalize(chart, args, kwargs)
            owner.normalized = None
            if len(args) >= 3:
                binding = owner._lineage(*args[:3])
                if binding is not None and all(owner._same_values(a, b)
                                                for a, b in zip(args[:3], result, strict=True)):
                    owner.normalized = result, dict(binding, normalization_same_observed_values=True)
            return result

        def observe_fill(actual, lower, upper):
            result = fill(actual, lower, upper)
            if owner.active is not None and owner.active[0] is actual:
                owner.active[1].append((lower, upper, result))
            return result

        def observe_paths(chart):
            actual = chart._contour_generator
            if not chart.filled or chart._paths is not None or id(actual) not in owner.sources:
                return paths(chart)
            if owner.active is not None:
                raise ValueError('nested original contour construction remains open')
            calls = []
            owner.active = (actual, calls)
            try:
                result = paths(chart)
            finally:
                owner.active = None
            if not calls or len(calls) != len(result):
                raise ValueError('original filled calls and drawing paths differ in count')
            owner.paths.append(dict(original=tuple(result), constructor=owner.sources[id(actual)],
                                    calls=calls, version=contourpy.__version__, collections=[]))
            return result

        contourpy._class_lookup['mpl2014'] = Constructor()
        original.create_filled_contour = observe_fill
        ContourSet._make_paths_from_contour_generator = observe_paths
        QuadContourSet._contour_args = observe_normalize
        try:
            yield
        finally:
            contourpy._class_lookup['mpl2014'] = original
            original.create_filled_contour = fill
            ContourSet._make_paths_from_contour_generator = paths
            QuadContourSet._contour_args = normalize

    def collection(self, paths, draw, start, end):
        if not paths:
            return
        matches = []
        for row in self.paths:
            original = row['original']
            if len(original) == len(paths) and all(a is b for a, b in zip(original, paths, strict=True)):
                matches.append((row, list(range(len(original)))))
                continue
            indices = defaultdict(list)
            for i, path in enumerate(original):
                indices[id(path)].append(i)
            if all(len(indices[id(path)]) == 1 for path in paths):
                matches.append((row, [indices[id(path)][0] for path in paths]))
        if len(matches) != 1:
            return
        row, bands = matches[0]
        row['collections'].append(dict(draw_ordinal=draw, path_span=dict(start=start, end=end), bands=bands,
            simplify_threshold=[float(path.simplify_threshold) for path in paths],
            path_simplifies=[bool(path.should_simplify) for path in paths]))

    def _write(self, path, data, schema):
        if len(data) > _CAP:
            raise ValueError('contour witness exceeds capture budget')
        path.write_bytes(data)
        return _ref(path, schema)

    def _lineage(self, x, y, z):
        frame = sys._getframe(1)
        while frame is not None:
            code = frame.f_code
            if code.co_argcount and Path(code.co_filename).is_relative_to(self.project):
                parameter = code.co_varnames[0]
                receiver = frame.f_locals.get(parameter)
                try:
                    fields = object.__getattribute__(receiver, '__dict__')
                except AttributeError:
                    fields = None
                if type(fields) is dict:
                    bindings = {axis: self._view(values, parameter, fields)
                                for axis, values in zip(('x', 'y', 'z'), (x, y, z), strict=True)}
                    if all(bindings.values()):
                        return dict(code_key=code_key(code), source=_ref(code.co_filename, 'scar.python-source'),
                                    bindings=bindings, sampled_alias_relation=True)
            frame = frame.f_back
        return None

    def _same_values(self, a, b):
        a, b = self.np.asarray(a), self.np.asarray(b)
        return a.shape == b.shape and a.dtype.str == b.dtype.str == '<f8' and a.tobytes() == b.tobytes()

    def _view(self, values, parameter, fields):
        if type(values) is not self.np.ndarray or values.ndim != 2 or values.dtype.str != '<f8':
            return None
        rows, columns = values.shape
        found = []
        for name, root in fields.items():
            if (type(root) is not self.np.ndarray or root.ndim != 2 or root.dtype != values.dtype
                    or not root.flags.c_contiguous or values.strides != root.strides
                    or not self.np.shares_memory(values, root) or root.shape[1] == 0):
                continue
            distance = values.ctypes.data - root.ctypes.data
            if distance < 0 or distance % 8:
                continue
            r, c = divmod(distance // 8, root.shape[1])
            if r + rows <= root.shape[0] and c + columns <= root.shape[1]:
                found.append(dict(field_path=[parameter, name], root_shape=list(root.shape),
                                  start=[r, c], shape=[rows, columns], byte_strides=list(values.strides)))
        return found[0] if len(found) == 1 else None

    def finish_frame(self, index):
        for row in self.paths:
            if not row['collections']:
                continue
            self._retain(index, row)
        self.paths.clear()
        self.sources.clear()

    def _retain(self, frame, row):
        actual, x0, y0, z0, mask0, kwargs, lineage = row['constructor']
        x, y, z = (self.np.asarray(a, dtype='<f8', order='C') for a in (x0, y0, z0))
        if x.shape != y.shape or y.shape != z.shape or z.ndim != 2:
            raise ValueError('original contour provider requires three equal-shaped lattices')
        mask = self.np.asarray(mask0, dtype='u1', order='C') if mask0 is not None else None
        if mask is not None and mask.shape != z.shape:
            raise ValueError('original contour provider mask has different geometry')
        calls = row['calls']
        levels = [float(calls[0][0])]
        for lower, upper, _ in calls:
            if struct.pack('<d', lower) != struct.pack('<d', levels[-1]) or not lower < upper:
                raise ValueError('original contours require an exact contiguous sequence of intervals')
            levels.append(float(upper))
        y_chunk, x_chunk = int(kwargs['y_chunk_size']), int(kwargs['x_chunk_size'])
        header = INPUT_MAGIC + struct.pack('<7Q', *z.shape, len(levels), int(actual.corner_mask),
                                           x_chunk, y_chunk, int(mask is not None))
        input_bytes = header + x.tobytes() + y.tobytes() + z.tobytes() + struct.pack(f'<{len(levels)}d', *levels)
        if mask is not None:
            input_bytes += mask.tobytes()
        polygons = bytearray(OUTPUT_MAGIC + struct.pack('<Q', len(calls)))
        for _, _, (vertices, codes) in calls:
            if len(vertices) != len(codes):
                raise ValueError('original filled-contour polygon rows differ')
            polygons.extend(struct.pack('<Q', len(vertices)))
            for points, path_codes in zip(vertices, codes, strict=True):
                points = self.np.asarray(points, dtype='<f8', order='C')
                path_codes = self.np.asarray(path_codes, dtype='u1', order='C')
                if points.shape != (len(path_codes), 2):
                    raise ValueError('original filled-contour vertex codes differ')
                polygons.extend(struct.pack('<Q', len(path_codes)) + points.tobytes() + path_codes.tobytes())
        prefix = self.root / f'contour-{len(self.events)}'
        offsets = {name: len(header) + i * z.nbytes for i, name in enumerate(('x', 'y', 'z', 'levels'))}
        if mask is not None:
            offsets['mask'] = offsets['levels'] + len(levels) * 8
        provider = dict(name='contourpy', version=row['version'], algorithm='mpl2014', binary=self.binary,
                        fill_type=_enum(actual.fill_type), line_type=_enum(actual.line_type),
                        corner_mask=bool(actual.corner_mask), chunk_size=[y_chunk, x_chunk],
                        thread_count=int(actual.thread_count), masked=mask is not None)
        self.events.append(dict(frame_index=frame, source=self.source, provider=provider,
            input=self._write(prefix.with_suffix('.cti'), input_bytes, 'scar.contour-scene-input'),
            expected_paths=self._write(prefix.with_suffix('.cto'), polygons, 'scar.contour-scene-paths'),
            grid=dict(shape=list(z.shape), dtype='<f8', offsets=offsets, lineage=lineage),
            intervals=[dict(lower=a, upper=b) for a, b in zip(levels[:-1], levels[1:], strict=True)],
            collections=row['collections']))

    def report(self, original_agg):
        if not self.events:
            raise ValueError('no exact filled-contour to AGG path binding observed')
        report = dict(schema=SCHEMA, schema_version=1, agg_scene=_ref(original_agg, 'scar.agg-scene-observation'),
                      samples=self.events, scene_control_resolved=False, capture_is_diagnostic=True)
        (self.root / 'contour-result.json').write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
