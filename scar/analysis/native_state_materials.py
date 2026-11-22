"""Native state materials."""

from dataclasses import dataclass
from hashlib import sha256
import gzip
import json
import argparse
import time
from pathlib import Path
from scar.analysis import (
    source_binding_native_graph as native_graph,
    source_native_binding_materials as generated,
)
from scar.analysis.static import native_interfaces as static_native_interfaces
from scar.analysis.native_call_materials import StaticNativeDeclaration, StaticNativeMaterials
from scar.artifacts import (
    read_record_table as _table,
    _durable_bytes,
    _durable_mkdir,
    _json,
    write_table as _stream,
)
from scar.ir.program import codec, native_contracts as N, native_contracts as R
from scar.ir.program.dependencies import GraphNamespace
from scar.ir.program.native import NativeBoundary
from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.program.queries import ProgramIndex
from scar.ir.record_codec import decode
from scar.analysis.python_binding_query import PythonBindingGraph


# Read existing independent native schema units without source-call replay.
#
# Only the native batches of a binding/native publication are certified here.
# Python normalization, caller relations and runtime dispatch are not rederived.
# The retained parser output is re-lowered and byte-compared by the existing
# native reader; neither parser workers nor target programs are started.
@dataclass(frozen=True, slots=True)
class NativeStateUnit:
    namespace: GraphNamespace
    index: ProgramIndex
    boundary: NativeBoundary
    declaration: StaticNativeDeclaration
    declaration_fact: ProviderFactReference


def _stamp(path):
    row = path.stat()
    return row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_ctime_ns


class NativeStateMaterials:
    """Lazily authenticate native batches; retain original graph locators.

    ``path`` is a publication containing ``batches`` and the original links
    table (currently program-binding-native-current). The generated Python
    declaration catalog alone has no canonical native units. ``check`` guards
    all materials actually loaded in this session; unread batches stay unread.
    """

    def __init__(self, path):
        self.path = Path(path).absolute()
        self._stamps = {}; self._bytes = {}; self._receipts = {}; self._batches = {}
        with generated._target_guard():
            self.raw = self._read(self.path/'result.json')
            self.report = r = json.loads(self.raw)
            if ((r.get('schema'), r.get('schema_version'), r.get('status')) !=
                    (native_graph.SCHEMA, 1, 'SOURCE_NORMALIZATION_TO_CANONICAL_NATIVE_SCHEMAS')):
                raise ValueError('native state materials require an existing canonical native-batch publication')
            self._pins = native_graph._pins()
            if (r['pins'] != self._pins or r['native_schemas_connected'] is not True
                    or any(r[name] is not False for name in native_graph.NO_PROOF)):
                raise ValueError('native batch producer or declaration-only authority differs')

            # This original report supplies package/revision metadata only. Its
            # receipt is anchored in the native report; no source ancestry or
            # generated Python definitions table is loaded or certified here.
            binding_path = Path(r['bindings']).absolute()/'result.json'
            parents = [decode(ProviderArtifact, row) for row in r['parents']
                if row['uri'] == str(binding_path)]
            if len(parents) != 1 or parents[0].schema != generated._SCHEMA:
                raise ValueError('native batches lack their original declaration-version report receipt')
            self._binding_raw = self._artifact(parents[0], binding_path)
            self.binding_report = json.loads(self._binding_raw)
            b = self.binding_report
            if ((b['schema'], b['schema_version'], b['status']) !=
                    (generated._SCHEMA, 1, generated._STATUS) or b['pins'] != generated._pins()
                    or any(b[name] is not False for name in generated._FALSE_FLAGS)):
                raise ValueError('native declaration-version report or producer differs')

            receipt = r['tables']['links']
            if receipt['path'] != 'links.jsonl.gz':
                raise ValueError('native schema-link table path differs')
            self._read(self.path/receipt['path'], receipt['sha256'], receipt['bytes'])
            links = tuple(decode(N.BindingSchemaLink, row) for row in _table(self.path, receipt))
            if len(links) != r['counts']['binding_links'] or len({l.binding for l in links}) != len(links):
                raise ValueError('native schema-link coverage or binding identities differ')
            self.references = {}
            for link in links:
                previous = self.references.setdefault(link.native.selector, link.native)
                if previous != link.native:
                    raise ValueError('native schema selector has conflicting original graph references')
            selectors = []
            for ordinal, batch in enumerate(r['batches']):
                if (batch['path'] != 'native/'+f'{ordinal:04d}'
                        or not 1 <= len(batch['selectors']) <= 64):
                    raise ValueError('native batch path or selector budget differs')
                selectors.extend(batch['selectors'])
            if (tuple(selectors) != tuple(sorted(self.references))
                    or len(selectors) != len(set(selectors)) or len(selectors) != r['counts']['selectors']):
                raise ValueError('native batch/reference selector coverage differs')
            self.check()

    def _read(self, path, digest=None, size=None):
        path = Path(path).absolute()
        if path.resolve() != path or not path.is_file():
            raise ValueError('native state material is not a canonical physical file: '+str(path))
        stamp = _stamp(path)
        if path in self._stamps and self._stamps[path] != stamp:
            raise ValueError('native state material changed: '+str(path))
        raw = self._bytes.get(path)
        if raw is None:
            raw = path.read_bytes()
            if _stamp(path) != stamp:
                raise ValueError('native state material changed while reading: '+str(path))
            self._bytes[path] = raw
            self._receipts[path] = sha256(raw).hexdigest(), len(raw)
            self._stamps[path] = stamp
        observed_digest, observed_size = self._receipts[path]
        if ((digest is not None and digest != observed_digest)
                or (size is not None and size != observed_size)):
            raise ValueError('native state material receipt differs: '+str(path))
        return raw

    def _artifact(self, artifact, expected_path):
        if artifact.uri != str(Path(expected_path).absolute()):
            raise ValueError('native material artifact points outside its original batch')
        return self._read(expected_path, artifact.sha256, artifact.size_bytes)

    def _batch(self, ordinal, batch):
        folder = self.path/batch['path']
        artifact = decode(ProviderArtifact, batch['material'])
        if artifact.schema != static_native_interfaces.SCHEMA:
            raise ValueError('native parser material schema differs')
        wire = json.loads(self._artifact(artifact, folder/'materials.json'))
        if (wire.get('schema'), wire.get('schema_version')) != (static_native_interfaces.SCHEMA, 1):
            raise ValueError('native parser material document schema differs')
        materials = decode(StaticNativeMaterials,
            {k:v for k,v in wire.items() if k not in ('schema', 'schema_version')})
        # This existing reader checks package/revision, all source/tool spans,
        # isolation, exact lowerer output bytes, and original schema references.
        natives = native_graph._read_native(self.path, (batch,), self.references, self.binding_report)
        spans = [*materials.package_sources, materials.parser.worker, *materials.parser.tools]
        for declaration in materials.declarations:
            spans.extend((declaration.header_source, declaration.yaml_source, declaration.schema_source))
            spans.extend(row.source for row in (*declaration.dispatch, *declaration.fields))
        for span in spans:
            self._read(span.path, span.sha256, span.size_bytes)
        result = []
        for position, declaration in enumerate(materials.declarations):
            reference, _, original_boundary = natives[declaration.selector]
            expected = folder/(sha256(declaration.selector.encode()).hexdigest()[:24]+'.json.gz')
            locator = reference.graph
            if locator.pointer != '#' or locator.representation != 'gzip-canonical-json':
                raise ValueError('native graph locator changed its original codec representation')
            raw = gzip.decompress(self._artifact(locator.artifact, expected))
            if len(raw) != locator.decoded_bytes or sha256(raw).hexdigest() != locator.decoded_sha256:
                raise ValueError('native decoded graph receipt differs')
            document = json.loads(raw)
            if (document.get('schema') != codec.SCHEMA or type(document.get('schema_version')) is not int
                    or not 1 <= document['schema_version'] <= codec.SCHEMA_VERSION):
                raise ValueError('native graph codec metadata differs')
            graph = codec.loads(raw.decode('utf-8')); index = ProgramIndex(graph)
            boundary = index.native_boundary(reference.boundary)
            fact = ProviderFactReference(artifact, '#/declarations/'+str(position))
            if (boundary != original_boundary or boundary.region != graph.root
                    or boundary.signature != ProviderFactReference(artifact, fact.locator+'/schema')
                    or len(declaration.inputs) != len(boundary.inputs)
                    or len(declaration.outputs) != len(boundary.outputs)):
                raise ValueError('native boundary differs from its original declaration coordinates')
            for ports, slots in ((declaration.inputs, boundary.inputs), (declaration.outputs, boundary.outputs)):
                for i, (port, slot) in enumerate(zip(ports, slots, strict=True)):
                    if port.position != i or slot.position != i or port.kind != slot.role:
                        raise ValueError('native declaration port position/role differs')
            namespace = GraphNamespace(locator, document['schema']+'/'+str(document['schema_version']), graph.root)
            result.append(NativeStateUnit(namespace, index, boundary, declaration, fact))
        self.check()
        self._batches[ordinal] = tuple(result)
        return self._batches[ordinal]

    def units(self):
        self.check()
        for ordinal, batch in enumerate(self.report['batches']):
            with generated._target_guard():
                rows = self._batches.get(ordinal)
                if rows is None:
                    rows = self._batch(ordinal, batch)
            yield from rows
        self.check()

    def check(self):
        with generated._target_guard():
            if (json.loads(self.raw) != self.report or json.loads(self._binding_raw) != self.binding_report
                    or native_graph._pins() != self._pins
                    or any(_stamp(path) != stamp for path, stamp in self._stamps.items())):
                raise ValueError('native state original materials or producer changed')


# Project exact resource-related input types and dispatch declarations.
def _projections(type_,path=(),pointer=''):
    if type_['kind'] in ('OptionalType','ListType'):
        projection=R.TypeProjection.OPTIONAL_ELEMENT if type_['kind']=='OptionalType' else R.TypeProjection.LIST_ELEMENT
        yield from _projections(type_['element'],(*path,projection),pointer+'/element')
    elif type_['kind']=='BaseType':
        kind={'Tensor':R.ResourceOperandKind.TENSOR_ARGUMENT,'Device':R.ResourceOperandKind.DEVICE_ARGUMENT,
            'Generator':R.ResourceOperandKind.GENERATOR_ARGUMENT}.get(type_.get('name'))
        if kind is not None:yield kind,type_['text'],path,pointer


def build_resource_declarations(bindings,out):
    started=time.monotonic();view=PythonBindingGraph(bindings);out=Path(out).absolute()
    if out.exists():raise ValueError('fresh resource declaration output required')
    operands=[];dispatch=[];seen=set();materials={}
    for definition in view.definitions():
        if definition.native is None or definition.native.boundary in seen:continue
        graph=view.expand_native(definition);boundary=next(b for b in graph.native_boundaries if b.id==definition.native.boundary)
        seen.add(boundary.id);artifact=boundary.signature.artifact
        if artifact.sha256 not in materials:
            raw=Path(artifact.uri).read_bytes()
            if len(raw)!=artifact.size_bytes or sha256(raw).hexdigest()!=artifact.sha256:raise ValueError('native declaration materials differ')
            materials[artifact.sha256]=json.loads(raw)
        pointer=boundary.signature.locator.rsplit('/',1)[0];row=materials[artifact.sha256]
        for key in pointer.removeprefix('#/').split('/'):row=row[int(key)] if type(row) is list else row[key]
        for index,(parameter,slot) in enumerate(zip(row['inputs'],boundary.inputs,strict=True)):
            for category,type_,path,tail in _projections(parameter['type']):
                operands.append(R.ResourceOperand(boundary.id,slot.port,category,type_,path,
                    ProviderFactReference(artifact,pointer+'/inputs/'+str(index)+'/type'+tail)))
        for index,candidate in enumerate(row['dispatch']):
            dispatch.append(R.DeclaredDispatchCandidate(boundary.id,tuple(candidate['keys']),candidate['symbol'],
                ProviderFactReference(artifact,pointer+'/dispatch/'+str(index))))
    _durable_mkdir(out);ot=_stream(out/'operands.jsonl.gz',operands);dt=_stream(out/'dispatch.jsonl.gz',dispatch)
    result=dict(schema='scar.native-resource-declarations',schema_version=1,status='DECLARED_OPERANDS_AND_DISPATCH',
        bindings=str(view.path),bindings_result_sha256=sha256((view.path/'result.json').read_bytes()).hexdigest(),
        native_boundaries=len(seen),resource_operands=ot,dispatch_candidates=dt,
        physical_resources_bound=False,resource_uses_created=0,kernels_observed=0,copies_observed=0,
        type_is_not_consumption_proof=True,target_executed=False,first_stage_complete=False,
        implementation_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),types_sha256=sha256(Path(R.__file__).read_bytes()).hexdigest(),
        seconds=time.monotonic()-started)
    _durable_bytes(out/'result.json',_json(result),replace=False);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--bindings',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();r=build_resource_declarations(args.bindings,args.out)
    print(json.dumps({k:r[k] for k in ('status','native_boundaries','seconds')},indent=2))


if __name__=='__main__':main()
