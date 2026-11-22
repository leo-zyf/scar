"""Source value interfaces."""

import argparse
import json
import time
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from scar.analysis import source_value_flow as values
from scar.analysis.source_consumer_parents import _stamp
from scar.analysis.source_relation_snapshot import RelationSnapshot, _local_path
from scar.artifacts import (
    write_table,
    read_record_table as _table,
    _durable_bytes,
    _durable_mkdir,
    _json,
)
from scar.ir.program import (
    binding_ports as R,
    source_productions as P,
    normal_graph as T,
)
from scar.ir.record_codec import encode, decode
from scar.analysis.source_reference_consumers import argument_points, bind_consumer_interface
from scar.analysis.source_references import ReferenceSlicer, source_port


# Connect derived call inputs to original receipt-bound source interfaces.
#
# The parent supplies a syntactic may-dependency from its root formal. That
# dependency never becomes an alias between the root and a derived actual.
# Function bodies remain shared and are not entered by this adapter.
_CONDITIONS = (
    'PARENT_SOURCE_VALUE_REACH_MAY_DEPENDENCY_SELECTED',
    'ROOT_TO_DERIVED_ACTUAL_REFERENCE_IDENTITY_NOT_ESTABLISHED',
    'HISTORICAL_TARGET_IS_MAY_ALTERNATIVE_NOT_MATCHED_CALLER_CONTEXT',
    'ARGUMENT_VALUE_IS_NOT_EFFECT_OR_ESCAPE_SUMMARY',
    'NO_CONTENT_PRESERVATION_OR_RUNTIME_DISPATCH_PROOF',
)


def _index(rows, ordinal):
    if type(ordinal) is not int or not 0 <= ordinal < len(rows):
        raise ValueError('value interface index is outside its original parent table')
    return rows[ordinal]


def _original_call(slicer, unit, record):
    prefix = slicer.fact(unit, 'calls', 0).locator.rsplit('/', 1)[0] + '/'
    locator = record.fact.locator
    if not locator.startswith(prefix):
        raise ValueError('value interface call reference differs from its original source table')
    suffix = locator[len(prefix):]
    if not suffix.isascii() or not suffix.isdecimal() or str(int(suffix)) != suffix:
        raise ValueError('value interface call needs its canonical original row ordinal')
    ordinal = int(suffix)
    call = _index(unit.calls, ordinal)
    if record.call != call or record.fact != slicer.fact(unit, 'calls', ordinal):
        raise ValueError('value interface body call changed its original fact or artifact receipt')
    return call


def connect_value_interfaces(snapshot, bodies, reaches):
    """Join every parent call input using one complete historical calls scan.

    ``SourceConsumerBinding.use`` and frontier.use index the returned uses.
    ``ReferenceArgumentUse.node`` and frontier.node are parent reach ordinals,
    not world-return node indices. No body entry, caller-context admission or
    root-to-actual reference equality is inferred. The caller retains snapshot
    parent guards and is responsible for its final validate_parents() check.
    """
    slicer = ReferenceSlicer(snapshot)
    uses = []; sites = []; wanted = set(); wanted_units = set(); body_calls = {}
    for reach_ordinal, reach in enumerate(reaches):
        body = _index(bodies, reach.body)
        if reach.formal.unit != body.body.unit or body.source_function.unit != body.body.unit:
            raise ValueError('value interface reach formal belongs to a different source body')
        slicer.local(reach.formal)
        provider, function_port = slicer.local(body.source_function)
        function = next((f for f in provider.functions if f.value == function_port), None)
        if (function is None or T.Body(provider.unit, function.template, function.body) != body.body
                or reach.formal not in tuple(source_port(provider, p) for p in function.formals[1:-1])):
            raise ValueError('value interface root is not an original parameter of its source function')
        if reach.body not in body_calls:
            unit, *_ = slicer.unit(body.body.unit)
            records = {}
            for record in body.calls:
                call = _original_call(slicer, unit, record)
                if record.fact in records and records[record.fact] != call:
                    raise ValueError('value interface body has conflicting original call records')
                records[record.fact] = call
            body_calls[reach.body] = unit, records
        unit, records = body_calls[reach.body]
        for input_ordinal, value in enumerate(reach.call_inputs):
            if value.call not in records:
                raise ValueError('dependent call input has no exact original SourceBodyCall reference')
            call = records[value.call]
            try:
                role = R.ArgumentRole(value.role)
            except (TypeError, ValueError) as error:
                raise ValueError('dependent call input has an unknown original argument role') from error
            if type(value.ordinal) is not int or not any(
                    role is expected_role and value.ordinal == ordinal and value.keyword == keyword
                    and value.actual == source_port(unit, local)
                    for expected_role, ordinal, keyword, local in argument_points(call)):
                raise ValueError('dependent call input differs from its original actual, role, position or keyword')
            before, after = source_port(unit, call.world), source_port(unit, call.result_world)
            conditions = tuple(dict.fromkeys((*reach.conditions, *_CONDITIONS)))
            use = R.ReferenceArgumentUse(reach_ordinal, value.call, role, value.ordinal, value.keyword,
                value.actual, (), False, before, after, conditions)
            uses.append(R.SourceValueInterfaceUse(reach_ordinal, input_ordinal, use))
            result = source_port(unit, call.result)
            sites.append((call, result, source_port(unit, call.callee)))
            wanted.add(_json(encode(result))); wanted_units.add(unit.unit)

    targets = defaultdict(list)
    # Decode only wanted records while RelationSnapshot checks both complete
    # receipts. All reaches and call inputs share this single original scan.
    for _, link, evidence in snapshot.rows('calls', lambda wire:
            wire['result']['unit'] in wanted_units and _json(wire['result']) in wanted, T.CallLink):
        targets[link.result].append((link, evidence))
    bindings = []; frontiers = []
    for use_ordinal, item in enumerate(uses):
        use = item.use
        call, result, callee = sites[use_ordinal]
        matched = False
        for link, evidence in targets.get(result, ()):
            if link.result != result or link.callee != callee:
                continue
            matched = True
            records, reason = bind_consumer_interface(slicer, use_ordinal, use, call, link, evidence)
            bindings.extend(records)
            if reason is not None:
                frontiers.append(R.ConsumerFrontier(item.reach, use_ordinal, use.actual, evidence, reason))
        if not matched:
            frontiers.append(R.ConsumerFrontier(item.reach, use_ordinal, use.actual, None,
                'HISTORICAL_CALL_TARGET_INTERFACE_UNRESOLVED'))
    return R.SourceValueInterfaceGraph(tuple(uses), tuple(bindings), tuple(frontiers))


# Receipt-bound argument interfaces downstream of source value operations.
#
# One parent certification is shared within a build/verification session. This
# does not select dispatch, erase parent dependency conditions or enter bodies.
class ValueInterfaceParents:
    """Retain exactly one fully certified source value-flow parent in memory."""

    def __init__(self, path):
        started = time.monotonic(); self.root = Path(path).absolute()
        self.report, (self.bodies, self.reaches) = values.read_value_flow(self.root)
        self.raw = (self.root/'result.json').read_bytes()
        if json.loads(self.raw) != self.report:
            raise ValueError('value interface parent changed after its strict read')
        self.artifact = values._artifact(self.root/'result.json', self.report['schema'])
        self.tables = {}
        for receipt in self.report['tables'].values():
            table = _local_path(self.root, receipt['path']); before = _stamp(table); raw = table.read_bytes()
            if (sha256(raw).hexdigest() != receipt['sha256'] or len(raw) != receipt['bytes']
                    or before != _stamp(table)):
                raise ValueError('value interface parent table changed after certification')
            self.tables[table] = before
        consumer_root = Path(self.report['consumers'])
        consumer = json.loads((consumer_root/'result.json').read_bytes())
        if encode(values._artifact(consumer_root/'result.json', 'scar.reference-consumer-graph')) != self.report['parent']:
            raise ValueError('certified value-flow consumer report changed')
        self.catalog, self._validate_sources = values._source_catalog(consumer_root, consumer)
        # These exact report links were certified above. They locate the inert
        # historical relation snapshot, whose own receipts are checked below.
        returns = json.loads((Path(consumer['returns'])/'result.json').read_bytes())
        world = json.loads((Path(returns['world'])/'result.json').read_bytes())
        join = json.loads((Path(world['join'])/'result.json').read_bytes())
        declared = json.loads((Path(join['contexts'])/'result.json').read_bytes())
        fields = json.loads((Path(declared['parent'])/'result.json').read_bytes())
        self.snapshot = RelationSnapshot(fields['relations'], fields['catalog'])
        if (encode(self.snapshot.result_artifact) != fields['relation_parent']
                or encode(self.snapshot.catalog_artifact) != fields['catalog_parent']):
            raise ValueError('value interface relations differ from the certified field parent')
        self.validate()
        self.certification_seconds = time.monotonic()-started

    def validate(self):
        if self.raw != (self.root/'result.json').read_bytes() or self.report['pins'] != values._pins():
            raise ValueError('certified value-flow report or rules changed')
        if any(_stamp(path) != stamp for path, stamp in self.tables.items()):
            raise ValueError('certified value-flow tables changed')
        self._validate_sources(); self.snapshot.validate_parents()


def _pins():
    from scar.analysis import source_reference_consumers

    pins = {name: sha256(Path(path).read_bytes()).hexdigest() for name, path in (
        ('producer', __file__), ('types', R.__file__),
        ('binder', source_reference_consumers.__file__), ('values', values.__file__))}
    pins['interfaces'] = pins['producer']
    return pins


def _counts(graph):
    return dict(uses=len(graph.uses), bindings=len(graph.bindings),
        bound_uses=len({b.use for b in graph.bindings}),
        source_functions=len({b.source_callable for b in graph.bindings}),
        activations=dict(Counter(b.activation.value for b in graph.bindings)),
        frontiers=len(graph.frontiers), frontier_reasons=dict(Counter(f.reason for f in graph.frontiers)))


def _parents(path, parents):
    parent = parents if parents is not None else ValueInterfaceParents(path)
    if parent.root != Path(path).absolute(): raise ValueError('certified value parent path differs')
    parent.validate()
    return parent


def build_value_interface_graph(path, out, *, parents=None):
    started = time.monotonic(); pins = _pins(); out = Path(out).absolute()
    if out.exists(): raise ValueError('fresh value interfaces output required')
    parents = _parents(path, parents)
    graph = connect_value_interfaces(parents.snapshot, parents.bodies, parents.reaches)
    parents.validate()
    if _pins() != pins: raise ValueError('value interface producer changed')
    _durable_mkdir(out)
    tables = {name: write_table(out/(name+'.jsonl.gz'), getattr(graph, name))
        for name in ('uses', 'bindings', 'frontiers')}
    report = dict(schema='scar.source-value-interfaces', schema_version=1,
        status='CONDITIONAL_SOURCE_VALUE_CALL_INTERFACES', values=str(parents.root),
        parent=encode(parents.artifact), pins=pins, tables=tables, counts=_counts(graph),
        runtime_dispatch_proven=False, root_to_argument_identity_proven=False,
        target_executed=False, first_stage_complete=False, seconds=time.monotonic()-started)
    _durable_bytes(out/'result.json', _json(report), replace=False)
    return report


def read_value_interface_graph(path, *, parents=None):
    path = Path(path).absolute(); raw = (path/'result.json').read_bytes(); report = json.loads(raw)
    if (report['schema'], report['schema_version']) != ('scar.source-value-interfaces', 1) or report['pins'] != _pins():
        raise ValueError('value interface producer differs')
    parents = _parents(report['values'], parents)
    if report['parent'] != encode(parents.artifact): raise ValueError('value interface parent differs')
    classes = dict(uses=R.SourceValueInterfaceUse, bindings=R.SourceConsumerBinding, frontiers=R.ConsumerFrontier)
    graph = R.SourceValueInterfaceGraph(**{name: tuple(decode(cls, row, record_tags=P.PRODUCTION_RECORD_TAGS)
        for row in _table(path, report['tables'][name])) for name, cls in classes.items()})
    if (graph != connect_value_interfaces(parents.snapshot, parents.bodies, parents.reaches)
            or _counts(graph) != report['counts']):
        raise ValueError('value interface bindings differ from original call inputs and source interfaces')
    if any(report[k] is not False for k in ('runtime_dispatch_proven', 'root_to_argument_identity_proven',
            'target_executed', 'first_stage_complete')):
        raise ValueError('value interface graph cannot prove runtime dispatch or root identity')
    parents.validate()
    if raw != (path/'result.json').read_bytes(): raise ValueError('value interface graph changed during verification')
    return report, graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--values'); parser.add_argument('--out'); parser.add_argument('--read')
    args = parser.parse_args()
    if args.read: report, _ = read_value_interface_graph(args.read)
    else:
        if not args.values or not args.out: parser.error('--values and --out required')
        report = build_value_interface_graph(args.values, args.out)
    print(json.dumps(dict(status=report['status'], counts=report['counts'], seconds=report['seconds'])), flush=True)


if __name__ == '__main__': main()
