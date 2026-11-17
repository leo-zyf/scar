"""Interpret original F4 ndarray indexing copies as ordered root-relative ROIs.

This is an effect profile, not an ArrayRegion/codegen profile. Integer axes
squeeze logical dimensions while retaining physical row/column coordinates.
Target code is never imported or executed; retained F0 is compiled for CodeID.
"""
from dataclasses import dataclass

from scar.analysis.source_body_interfaces import BodyInterfaceResolver
from scar.analysis.source_reference_slice import source_port
from scar.compiler.graph_frontend import (_GraphLowerer, _Scalar, _Slice, _WORLD, _ENV,
    _origin, _reject, _source_identity, _metadata, GraphRegionRejected)
from scar.compiler.stateful_frontend import _StateLowerer
from scar.compiler.model import SourceRef
from scar.ir.program import source_productions as P, normal_links as F
from scar.ir.program.control import LanguageCompletionKind as C
from scar.ir.program.normal_targets import Body
from scar.ir.record_codec import encode
from scar.trace.code_snapshot import code_key
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.array-copy-effects'


@dataclass(frozen=True)
class _Root:
    formal: int
    shape: tuple[int, int] | None
    origin: dict


@dataclass(frozen=True)
class _Selection:
    root: _Root
    rectangle: tuple[tuple[int, int], tuple[int, int]]
    squeezed: tuple[int, ...]
    origin: dict
    get_access: str | None = None
    captured_epoch: int | None = None

    @property
    def shape(self):
        return tuple(b-a for i,(a,b) in enumerate(self.rectangle) if i not in self.squeezed)


def _intersects(left, right):
    return all(max(a[0],b[0]) < min(a[1],b[1]) for a,b in zip(left,right))


class _CopyLowerer(_StateLowerer):
    def __init__(self, original, function, resolver, syntax, shapes):
        super().__init__(original, function, resolver, syntax, {}, {}, {})
        self.shapes = shapes; self.accesses = []; self.copies = []; self.root_epochs = {}

    def construct(self, context, step, args, evaluated, origin):
        ins = step.instruction
        if ins.construction is P.ConstructionKind.TUPLE and ins.tuple_layout is P.TupleConstructionLayout.ITEMS:
            if not all(type(v) is _Slice or type(v) is _Scalar and type(v.value) is int for v in args[:-1]):
                _reject('COPY_SELECTOR_TUPLE', 'only original integer or unit-slice axes are supported', origin)
            return (tuple(args[:-1]), _WORLD)
        return super().construct(context, step, args, evaluated, origin)

    def select(self, receiver, axes, origin):
        if type(axes) is not tuple or len(axes) != 2:
            _reject('COPY_RANK2_SELECTOR', 'two original constant axes required', origin)
        if type(receiver) is _Root:
            if receiver.shape is None:
                _reject('COPY_SHAPE_FACT_MISSING', 'array formal has no original declared rank-two shape', origin)
            root = receiver; base = tuple((0,n) for n in root.shape)
        elif type(receiver) is _Selection and not receiver.squeezed:
            root, base = receiver.root, receiver.rectangle
        else:
            _reject('COPY_RECEIVER_PROFILE', 'indexing requires a formal rank-two root or unsqueezed fixed view', origin)
        rectangle, squeezed = [], []
        for i,(axis,(start,stop)) in enumerate(zip(axes,base)):
            extent = stop-start
            if type(axis) is _Scalar and type(axis.value) is int:
                index = axis.value+extent if axis.value < 0 else axis.value
                if not 0 <= index < extent:
                    _reject('COPY_INDEX_OUTSIDE_SHAPE', 'constant integer axis is outside the declared shape', origin)
                rectangle.append((start+index,start+index+1)); squeezed.append(i)
            elif type(axis) is _Slice:
                a,b,step = slice(*axis.bounds).indices(extent)
                if step != 1:
                    _reject('COPY_SLICE_STEP', 'only original unit-step slices are supported', origin)
                rectangle.append((start+a,start+max(a,b)))
            else:
                _reject('COPY_DYNAMIC_SELECTOR', 'each selector must be an original integer or unit slice', origin)
        return _Selection(root, tuple(rectangle), tuple(squeezed), origin)

    def access(self, kind, selection, origin, **extra):
        identity = 'access:'+str(len(self.accesses))
        self.accesses.append(dict(id=identity, kind=kind, formal_index=selection.root.formal,
            rectangle=encode(selection.rectangle), visible_shape=list(selection.shape),
            squeezed_axes=list(selection.squeezed), source=origin, **extra))
        return identity

    def action(self, context, step, args, origin):
        kind = step.instruction.action
        if kind in (P.LanguageActionKind.ITEM_GET, P.LanguageActionKind.ITEM_SET):
            expected = 3 if kind is P.LanguageActionKind.ITEM_GET else 4
            if len(args) != expected or args[-1] is not _WORLD:
                _reject('COPY_ITEM_ABI', 'original receiver/selectors/payload/world arity differs', origin)
            selected = self.select(args[0], args[1], origin)
            if kind is P.LanguageActionKind.ITEM_GET:
                scalar = len(selected.squeezed) == 2
                identity = self.access('READ_SCALAR_VALUE' if scalar else 'CREATE_ALIAS_VIEW', selected, origin,
                    reads_content_now=scalar, local_epoch=self.root_epochs[selected.root.formal] if scalar else None)
                return (_Selection(selected.root, selected.rectangle, selected.squeezed, origin, identity,
                    self.root_epochs[selected.root.formal] if scalar else None), _WORLD)
            value = args[2]
            if type(value) is _Root:
                if value.shape is None:
                    _reject('COPY_RHS_SHAPE', 'copy source formal lacks its declared shape', origin)
                value = _Selection(value, tuple((0,n) for n in value.shape), (), value.origin)
            if type(value) is not _Selection or value.shape != selected.shape:
                _reject('COPY_RHS_PROFILE', 'only original root/view copies with identical squeezed shapes are supported', origin)
            read_epoch = value.captured_epoch if value.captured_epoch is not None else self.root_epochs[value.root.formal]
            read = value.get_access if value.captured_epoch is not None else self.access('READ_COPY_SOURCE', value, origin,
                local_epoch=read_epoch, alias_view_creation=value.get_access, reads_content_now=True)
            before = self.root_epochs[selected.root.formal]
            write = self.access('WRITE_COPY_DESTINATION', selected, origin, local_epoch_before=before, local_epoch_after=before+1)
            same = value.root.formal == selected.root.formal
            self.copies.append(dict(id='copy:'+str(len(self.copies)), source_access=read, destination_access=write,
                source_formal=value.root.formal, destination_formal=selected.root.formal,
                source_rectangle=encode(value.rectangle), destination_rectangle=encode(selected.rectangle),
                source_epoch=read_epoch, destination_epoch_before=before, destination_epoch_after=before+1,
                alias_relation='SAME_FORMAL_ROOT' if same else 'DIFFERENT_FORMALS_MAY_ALIAS_AT_RUNTIME',
                source_destination_overlap=_intersects(value.rectangle,selected.rectangle) if same else None,
                overlap_handling='ORIGINAL_NUMPY_SETITEM_ASSIGNMENT_NOT_A_MEMCPY_REWRITE', source=origin))
            self.root_epochs[selected.root.formal] = before+1
            return (_WORLD,)
        if kind is P.LanguageActionKind.UNARY:
            operation = self.operations.get(context.result(step).port)
            if (operation is not None and operation.selector.value in ('USub','UAdd') and len(args) == 2
                    and type(args[0]) is _Scalar and type(args[0].value) is int
                    and tuple(source_port(self.facts,i).port for i in operation.operands)
                        == tuple(context.value(v).port for v in step.arguments[:-1])):
                return (_Scalar(-args[0].value if operation.selector.value == 'USub' else args[0].value, origin), _WORLD)
        _reject('COPY_UNSUPPORTED_ACTION', kind.value+' has no closed copy-only effect contract', origin)

    def inplace(self, context, step, args, origin):
        _reject('COPY_INPLACE_ARITHMETIC', 'in-place arithmetic is outside the copy-only effect profile', origin)

    def lower(self):
        fn = self.function
        if (fn.body_scope is None or fn.environment.kind is not P.DeferredBodyKind.FUNCTION
                or fn.environment.captures or fn.environment.owned_cells
                or any(p.kind.value in ('VAR_POSITIONAL','VAR_KEYWORD') for p in fn.environment.parameters)):
            _reject('COPY_FUNCTION_INTERFACE', 'ordinary fixed-signature function without captures required')
        context = self.resolver.plan(self.source.unit, fn.template)
        _GraphLowerer.check_decorators(self, context)
        arguments = [_ENV]
        for i,_ in enumerate(fn.parameter_names):
            formal = source_port(self.facts, fn.formals[i+1])
            origin = _origin(context,fn.body,value=encode(formal))
            root = _Root(i, self.shapes.get(i), origin); arguments.append(root); self.root_epochs[i] = 0
            namespaces = self.names.get((formal.port,True),())
            if len(namespaces) != 1 or namespaces[0].kind is not F.NamespaceKind.LEXICAL:
                _reject('COPY_FORMAL_BINDING', 'formal lacks its original unique lexical binding', origin)
            self.locals[namespaces[0]] = root
        kind,result = self.run(context,fn.body,(*arguments,_WORLD))
        if kind is not C.RETURN or len(result) != 2 or type(result[0]) is not _Scalar or result[0].value is not None:
            _reject('COPY_RETURN_PROFILE', 'copy-only effect summary requires original None return without escaping arrays')
        if not self.copies:
            _reject('COPY_NO_ASSIGNMENTS', 'no supported original array copy assignment found')


def derive_array_copy_effects(view, *, unit, function_ordinal, input_shapes, resolver=None, source_refs=()):
    """Summarize one original function; shape keys are formal parameter ordinals."""
    original = view.unit(unit); functions = original.facts.functions
    if type(function_ordinal) is not int or not 0 <= function_ordinal < len(functions):
        raise ValueError('an original function ordinal is required')
    fn = functions[function_ordinal]
    if type(input_shapes) is not dict or any(type(i) is not int or not 0 <= i < len(fn.parameter_names)
        or type(shape) not in (tuple,list) or len(shape) != 2
        or any(type(n) is not int or not 0 <= n < 2**63 for n in shape) for i,shape in input_shapes.items()):
        raise ValueError('declared rank-two shapes indexed by exact formal ordinals required')
    resolver = resolver or BodyInterfaceResolver(view)
    resolver.prepare(Body(unit,fn.template,fn.body))
    model,syntax,codes,recipe,digest = _source_identity(view.catalog,unit)
    name,span,lines = _metadata(syntax,fn.template.occurrence)
    candidates = [c for c in codes if c.co_name == name and c.co_firstlineno in lines
        and c.co_varnames[:c.co_argcount+c.co_kwonlyargcount] == fn.parameter_names]
    if len(candidates) != 1:
        raise ValueError('original function has no unique compile-only complete CodeID')
    code = candidates[0]
    source = SourceRef(model['path'],digest,code.co_qualname,span['line'],span['column'],code_key(code),
        (dict(kind='source_identity',method='COMPILE_ONLY_RETAINED_F0',unit=unit,recipe=recipe,
              occurrence=fn.template.occurrence,semantic_authority='ORIGINAL_F4_NOT_COMPILED_AST'),))
    lowerer = _CopyLowerer(original,fn,resolver,syntax,{i:tuple(s) for i,s in input_shapes.items()})
    frontier = []
    try:
        lowerer.lower()
    except GraphRegionRejected as error:
        frontier.append(dict(code=error.code,detail=error.detail,source=error.origin))
    roots = [dict(formal_index=i, parameter=fn.parameter_names[i], shape=list(s),
        formal_port=encode(source_port(original.facts,fn.formals[i+1])),
        reads=[a['id'] for a in lowerer.accesses if a['formal_index'] == i and a['kind'] in ('READ_COPY_SOURCE','READ_SCALAR_VALUE')],
        writes=[a['id'] for a in lowerer.accesses if a['formal_index'] == i and a['kind'] == 'WRITE_COPY_DESTINATION'])
        for i,s in sorted(input_shapes.items())]
    result = dict(schema=SCHEMA,schema_version=1,status='FRONTIER' if frontier else 'CLOSED_COPY_EFFECTS_CONDITIONAL',
        source=encode(source),unit=unit,function_ordinal=function_ordinal,body=encode(fn.body_scope),
        parameters=list(fn.parameter_names),roots=roots,accesses=lowerer.accesses,copies=lowerer.copies,frontiers=frontier,
        original_f4_artifact=encode(original.material.result_artifact),source_refs=encode(tuple(source_refs)),
        coordinate_scope='ONE_ACTIVATION_FORMAL_ARRAY_ROOT_NOT_RUNTIME_STORAGE_ID',
        local_epochs_are_ordered_writes_not_global_logical_versions=True,
        guards=['EXACT_NUMPY_NDARRAY_FLOAT64_RANK2','ORIGINAL_FUNCTION_CODE_ID_AND_BINDING',
            'ARRAY_SHAPE_EQUALS_DECLARED_SOURCE_FACTS','C_CONTIGUOUS_NONOVERLAPPING_ROOT_STORAGE',
            'CALLER_HOLDS_ROOT_ARRAY_REFERENCES_THROUGH_ACTIVATION',
            'STANDARD_NUMPY_INDEXING_AND_ASSIGNMENT_PROTOCOL_NO_SUBCLASS_CALLBACKS'],
        effects=dict(array_reads=True,array_writes=True,aliases='EXPLICIT_VIEWS',
            cross_formal_aliases='UNKNOWN_UNLESS_CALLER_GUARDS_ORIGINAL_ACTUAL_BINDING',
            may_raise='ORIGINAL_INDEXING_AND_ASSIGNMENT_EXCEPTIONS_AND_VIEW_OR_OVERLAP_ALLOCATION',
            allocation='VIEW_METADATA_AND_POSSIBLE_NUMPY_OVERLAP_TEMPORARY',
            external_effect='FORMAL_ARRAY_CONTENT_MUTATIONS',rng_effect='NONE_UNDER_EXACT_NDARRAY_PROTOCOL'),
        source_execution_observed=False,array_region_compilable=False,pure=False,
        whole_program_effects_closed=False,optimization_authorized=False)
    original.check()
    result['id'] = 'array-copy-effects:'+semantic_sha256(result)
    return result


def prove_copy_writes_disjoint(summary, *, formal_index, rectangles):
    """Check geometry only; caller must establish the exact actual/formal root."""
    document = dict(summary); identity = document.pop('id', None)
    if (summary.get('schema'),summary.get('schema_version')) != (SCHEMA,1) or identity != 'array-copy-effects:'+semantic_sha256(document):
        raise ValueError('exact original copy-effect summary identity required')
    if summary.get('status') != 'CLOSED_COPY_EFFECTS_CONDITIONAL':
        raise ValueError('closed original copy-effect summary required')
    roots = [r for r in summary['roots'] if r['formal_index'] == formal_index]
    if len(roots) != 1:
        raise ValueError('one original formal-array root required')
    shape = roots[0]['shape']
    if any(type(r) not in (tuple,list) or len(r) != 2 or any(type(axis) not in (tuple,list)
        or len(axis) != 2 or any(type(v) is not int for v in axis)
        or not 0 <= axis[0] <= axis[1] <= shape[i] for i,axis in enumerate(r)) for r in rectangles):
        raise ValueError('protected rectangles must be bounded by the same formal root')
    writes = [a for a in summary['accesses'] if a['formal_index'] == formal_index and a['kind'] == 'WRITE_COPY_DESTINATION']
    checks = [dict(write=a['id'],protected_rectangle=i,overlap=_intersects(a['rectangle'],r),source=a['source'])
              for a in writes for i,r in enumerate(rectangles)]
    return dict(schema='scar.array-copy-write-disjointness',schema_version=1,summary_id=summary['id'],
        formal_index=formal_index,status='OVERLAP' if any(c['overlap'] for c in checks) else 'DISJOINT_UNDER_ROOT_GUARDS',
        checks=checks,original_actual_formal_mapping_required=True,guards=summary['guards'],optimization_authorized=False)


__all__ = ['SCHEMA','derive_array_copy_effects','prove_copy_writes_disjoint']
