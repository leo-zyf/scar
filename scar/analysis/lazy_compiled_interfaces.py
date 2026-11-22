"""Receipt admission now; original compiled-interface type checks on demand.

This directory adapter keeps published leaf producers unchanged. It admits
all original header/input bytes without decoding UnitFacts or another graph.
The existing reader remains the sole authority for typed interfaces, family
coverage, source ports and suspension/mapping witnesses when queried.
"""
from hashlib import sha256
import json
from pathlib import Path
from threading import RLock
from types import MappingProxyType

from scar.analysis.compiled_local_interfaces import (CompiledLocalInterfaces,SCHEMA,
    _FALSE,_FAMILIES,_producer,projection_pins)
from scar.analysis.observed_code_correspondence import ObservedCompiledLocalMaterial
from scar.ir.program.suspension_interfaces import WitnessSource


_COUNTS=('iterations','bodies','mappings','merges','suspensions')
_READ_COUNTS=('unit_reads','iteration_reads','body_reads','mapping_reads')
_EMPTY=MappingProxyType({})


def _stamp(path):
    value=path.stat()
    return value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns


class LazyCompiledInterfaces:
    """One original leaf's guarded header plus an optional original reader.

    ``material`` is an already admitted ObservedCompiledLocalMaterial. Its
    existing graph admission is unchanged; this adapter performs no graph or
    UnitFacts decoding. A cold directory's declared counts are retained
    separately from the actual read counters, which start at zero.
    """
    def __init__(self,path,material):
        if not isinstance(material,ObservedCompiledLocalMaterial):
            raise TypeError('lazy interfaces require an already admitted compiled material')
        self.material=material;self.path=Path(path).absolute()
        if self.path.resolve()!=self.path or self.path.is_symlink():
            raise ValueError('canonical lazy compiled interface directory required')
        self._reader=None;self._lock=RLock();self._pins={};self._receipts={}
        self.pins=MappingProxyType(self._pins)
        material.validate_parents()
        root=self.path/'result.json';before=_stamp(root);self.raw=root.read_bytes()
        if _stamp(root)!=before:raise ValueError('compiled interface report changed during admission')
        self.report=report=json.loads(self.raw)
        if ((report.get('schema'),report.get('schema_version'),report.get('status'))!=
                (SCHEMA,1,'COMPILED_LOCAL_INTERFACES_PROJECTED')
                or type(report.get('schema_version'))is not int
                or report.get('implementation_sha256')!=_producer()
                or report.get('projection_pins')!=projection_pins()
                or report.get('unit')!=material.unit
                or report.get('witness_source')!=WitnessSource.SEALED_COMPILED_LOCAL_MATERIAL.value
                or report.get('families_complete')!={name:True for name in _FAMILIES}
                or any(report.get(name)is not False for name in _FALSE)):
            raise ValueError('compiled interface producer/material/coverage contract differs')
        counts=report.get('counts');row=report.get('row')
        if (type(counts)is not dict or set(counts)!=set(_COUNTS)
                or any(type(value)is not int or value<0 for value in counts.values())
                or type(row)is not dict or row.get('unit')!=material.unit or row.get('counts')!=counts):
            raise ValueError('compiled interface declared unit/count denominator differs')
        material_path=material.path/'result.json';facts=material.report['tables']['facts']
        expected_facts=dict(path=str(material.path/facts['path']),bytes=facts['bytes'],sha256=facts['sha256'])
        if (report.get('material',{}).get('path')!=str(material_path)
                or report.get('facts')!=expected_facts or row.get('facts_sha256')!=facts['sha256']):
            raise ValueError('compiled interface original material/fact parent differs')
        self.facts=material.fact_store
        self.rows=MappingProxyType({material.unit:row})
        self.declared_counts=MappingProxyType(dict(counts))
        self._admit(dict(path=str(root),bytes=len(self.raw),sha256=sha256(self.raw).hexdigest()))
        self._admit(report['material']);self._admit(report['facts'])
        for receipt in report['parents']:self._admit(receipt)
        for receipt in row['inputs']:self._admit(receipt)
        relative=Path(row['interfaces'])
        if relative.is_absolute()or'..'in relative.parts:
            raise ValueError('compiled interface table escapes its original directory')
        self._admit(dict(path=str(self.path/relative),bytes=row['bytes'],sha256=row['sha256']))
        self.validate_parents()

    def _admit(self,receipt):
        if (type(receipt)is not dict or set(receipt)!={'path','bytes','sha256'}
                or type(receipt['path'])is not str or type(receipt['bytes'])is not int or receipt['bytes']<0
                or type(receipt['sha256'])is not str or len(receipt['sha256'])!=64):
            raise ValueError('exact original interface byte receipt required')
        path=Path(receipt['path']);before=_stamp(path)
        if not path.is_absolute()or path.resolve()!=path or path.is_symlink()or not path.is_file():
            raise ValueError('canonical original compiled interface input required')
        if path in self._receipts:
            if self._receipts[path]!=receipt or self._pins[path]!=before:
                raise ValueError('compiled interface repeated input receipt differs')
            return
        raw=path.read_bytes()
        if len(raw)!=receipt['bytes']or sha256(raw).hexdigest()!=receipt['sha256']or _stamp(path)!=before:
            raise ValueError('compiled interface byte receipt differs')
        self._receipts[path]=dict(receipt);self._pins[path]=before

    @property
    def is_loaded(self):return self._reader is not None

    @property
    def stats(self):return dict.fromkeys(_READ_COUNTS,0)if self._reader is None else self._reader.stats

    @property
    def loaded(self):return _EMPTY if self._reader is None else self._reader.loaded

    @property
    def iterations(self):return _EMPTY if self._reader is None else self._reader.iterations

    @property
    def bodies(self):return _EMPTY if self._reader is None else self._reader.bodies

    @property
    def mappings(self):return _EMPTY if self._reader is None else self._reader.mappings

    @property
    def receipt(self):
        return dict(path=str(self.path),sha256=sha256(self.raw).hexdigest(),selected_sources=len(self.rows),
            catalog_complete=len(self.rows)==len(self.facts.rows))

    def admission_summary(self):
        return dict(header_admitted=True,typed_interfaces_loaded=self.is_loaded,
            status='ORIGINAL_READER_TYPES_AND_WITNESSES_VALIDATED'if self.is_loaded else
                'HEADER_RECEIPTS_ADMITTED_TYPES_DEFERRED',
            declared_counts=dict(self.declared_counts),read_counts=dict(self.stats))

    def validate_parents(self):
        self.material.validate_parents()
        if json.loads(self.raw)!=self.report or self.declared_counts!=self.report['counts']:
            raise ValueError('compiled interface in-memory report/counts changed')
        if any(_stamp(path)!=stamp for path,stamp in self._pins.items()):
            raise ValueError('compiled interface original input changed')
        if self._reader is not None:self._reader.validate_parents()
        return len(self._pins)

    def covers(self,unit):
        self.validate_parents()
        return unit.unit in self.rows

    def _demand(self,unit):
        if unit.unit not in self.rows:raise KeyError(unit.unit)
        with self._lock:
            self.validate_parents()
            if self._reader is None:
                reader=CompiledLocalInterfaces(self.path,self.material)
                reader.validate_parents();self.validate_parents()
                if reader.report!=self.report or reader.rows!=self.rows:
                    raise ValueError('original interface reader differs from admitted header')
                # Do not publish an instance whose constructor/type/family/
                # witness verification or final cold guards raised.
                self._reader=reader
            return self._reader

    def load(self,unit):return self._demand(unit).load(unit)
    def iteration(self,unit,item):return self._demand(unit).iteration(unit,item)
    def body(self,unit,function):return self._demand(unit).body(unit,function)
    def mapping(self,unit,template):return self._demand(unit).mapping(unit,template)


__all__=['LazyCompiledInterfaces']
