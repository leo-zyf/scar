"""C++20 repetition of one already-admitted transactional stateful ABI v1 call.

This is an explicit fixed-input composition, not a proof that an original
Python caller loop is replaceable. Arrays carry state between calls; the
array identities/descriptors, scalar arguments and providers are invariant.
Each inner call keeps its allocations, team, guards and commit boundary.
No Python callback executes between successful native iterations.
"""
from dataclasses import dataclass, replace
from hashlib import sha256
import ctypes
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from scar.codegen import cpp
from scar.codegen import stateful as backend

SCHEMA = 'scar.native-stateful-repeat-adapter'
_TEMPLATE = Path(__file__).resolve().parents[1] / 'codegen/stateful_repeat_v1.cpp.in'
CONDITIONS = (
    'FIXED_EXACT_NONNEGATIVE_INTEGER_REPETITION_COUNT',
    'IDENTICAL_ARRAY_IDENTITIES_DESCRIPTORS_SCALARS_AND_PROVIDERS_EACH_STEP',
    'ONLY_ARRAY_CONTENTS_ARE_LOOP_CARRIED_STATE',
    'NO_INTERSTEP_PYTHON_CODE_HOOKS_OBSERVERS_OR_CONCURRENT_STATE_ACCESS',
    'NO_INTERSTEP_PYTHON_SIGNAL_OR_CANCELLATION_RESPONSIVENESS_CONTRACT',
    'ALL_ORIGINAL_SINGLE_CALL_GUARDS_AND_EXTERNAL_ASSUMPTIONS',
)


def _hash(raw): return sha256(raw).hexdigest()
def _json(value): return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
def _receipt(path):
    path=Path(path).resolve();raw=path.read_bytes()
    return dict(path=str(path),sha256=_hash(raw),bytes=len(raw))
def _source(base):
    text=_TEMPLATE.read_text()
    for name in ('region_sha256','plan_sha256','contract_sha256'):
        text=text.replace('@'+name.upper()+'@',base.report[name])
    if '@' in text: raise ValueError('unresolved repetition adapter source')
    return text.encode()


class RepeatGuardFailure(backend.GuardFailure):
    """Progress is committed; resume at completed, never replay earlier steps."""
    def __init__(self, requested, completed, status, detail):
        super().__init__(detail)
        self.requested=requested;self.completed=completed;self.status=status


@dataclass
class StatefulRepeatArtifact:
    base: object
    directory: Path
    metadata: dict
    library: object
    native_repeat: object

    @property
    def report(self):
        return json.loads(json.dumps(self.metadata))

    def run(self, arrays, scalars, repetitions):
        """Execute a declared homogeneous call sequence; report exact progress.

        The caller must first establish CONDITIONS for the intended sequence.
        This API performs no automatic replacement of an application loop.
        A partial failure has committed precisely completed successful calls;
        the failing call is still untouched under the original ABI contract.
        """
        if type(repetitions) is not int or not 0 <= repetitions < 2**64:
            raise ValueError('repetitions requires an exact nonnegative uint64 integer')
        if repetitions==0:
            return dict(requested=0,completed=0,native_calls=0)
        progress=ctypes.c_uint64();status=None
        def invoke(*args):
            nonlocal status
            status=self.native_repeat(self.base._table,*args,repetitions,ctypes.byref(progress))
            return status
        # Reuse the exact existing Python descriptor/scalar/provider checks,
        # packing and provider lock. Only the terminal native dispatch changes.
        # replace() does not modify the base table, runtime, backend or ABI.
        facade=replace(self.base,_table=SimpleNamespace(contents=SimpleNamespace(run=invoke)))
        try:facade.run(arrays,scalars)
        except backend.GuardFailure as error:
            raise RepeatGuardFailure(repetitions,progress.value,status,str(error)) from error
        return dict(requested=repetitions,completed=progress.value,native_calls=progress.value)

    def run_with_fallback(self, arrays, scalars, repetitions, original_step):
        """On a precommit guard, execute only the uncommitted suffix in Python.

        original_step is the caller's original single step for the same fixed
        inputs. Its exception propagates normally; prior commits stay visible.
        """
        try:
            result=self.run(arrays,scalars,repetitions)
            return {**result,'fallback_calls':0,'native_status':0}
        except RepeatGuardFailure as error:
            for _ in range(error.completed,repetitions):original_step()
            return dict(requested=repetitions,completed=repetitions,native_calls=error.completed,
                fallback_calls=repetitions-error.completed,native_status=error.status)


def build_repeat(native_directory, out):
    """Compile one small host-control adapter, reusing original native math."""
    base=backend.load(native_directory);out=Path(out).resolve()
    if out.exists():raise ValueError('fresh stateful repetition output required')
    metadata=base.report
    if metadata['failure_policy']!='GUARD_FAILURE_ONLY_BEFORE_EXTERNAL_COMMIT_ORIGINAL_ONCE':
        raise ValueError('per-call precommit-only fallback ABI required')
    compiler=metadata['compiler'];source=_source(base)
    identity_line=subprocess.run([*compiler['argv'],'--version'],capture_output=True,text=True,check=True).stdout.splitlines()[0]
    if identity_line!=compiler['identity']:raise ValueError('stateful compiler identity changed before adapter build')
    headers={name:(base.directory/name).read_bytes() for name in ('stateful_abi_v1.h','scar_abi_v1.h')}
    identity=dict(source_sha256=_hash(source),headers={k:_hash(v) for k,v in headers.items()},
        compiler=compiler,base_artifact=_receipt(base.directory/'artifact.json'),
        generator=_receipt(__file__),template=_receipt(_TEMPLATE))
    build_id=_hash(_json(identity));out.mkdir(parents=True)
    files=dict(source='repeat.cpp',library='repeat_'+build_id[:20]+'.so')
    (out/files['source']).write_bytes(source)
    for name,raw in headers.items():(out/name).write_bytes(raw)
    command=[*compiler['argv'],*compiler['options'],str(out/files['source']),'-o',str(out/files['library'])]
    result=subprocess.run(command,capture_output=True,text=True)
    if result.returncode:raise RuntimeError('stateful repetition adapter compilation failed:\n'+result.stderr)
    report=dict(schema=SCHEMA,schema_version=1,build_id=build_id,identity=identity,
        files=files,sha256={k:_hash((out/v).read_bytes()) for k,v in files.items()},
        stateful_abi_version=1,cpp_standard=20,conditions=list(CONDITIONS),
        sequence='FIXED_INPUTS_CALL_COMPOSITION_NOT_ORIGINAL_CALLER_LOOP_PROOF',
        commit_policy='EACH_SUCCESSFUL_ORIGINAL_ABI_CALL_COMMITS_BEFORE_NEXT_STEP',
        failure_policy='STOP_BEFORE_FAILED_STEP_COMMIT_RESUME_ONLY_UNCOMMITTED_SUFFIX',
        native_math_changed=False,per_iteration_python_callbacks=False,
        allocations_or_team_setup_hoisted=False,target_loop_replacement_authorized=False,
        selected_for_use=False,cost_status='UNCALIBRATED',physical_optimality_proven=False)
    (out/'artifact.json').write_bytes(_json(report))
    return load_repeat(out)


def load_repeat(directory):
    directory=Path(directory).resolve();report=json.loads((directory/'artifact.json').read_bytes())
    if (report.get('schema'),report.get('schema_version'),report.get('stateful_abi_version'))!=(SCHEMA,1,1):
        raise ValueError('supported stateful repetition adapter required')
    identity=report['identity']
    if _hash(_json(identity))!=report['build_id']:raise ValueError('repetition build identity differs')
    for key in ('generator','template','base_artifact'):
        if _receipt(identity[key]['path'])!=identity[key]:raise ValueError('repetition source or base receipt differs')
    base=backend.load(Path(identity['base_artifact']['path']).parent)
    for key,name in report['files'].items():
        if Path(name).name!=name or _hash((directory/name).read_bytes())!=report['sha256'][key]:
            raise ValueError('repetition artifact checksum differs: '+key)
    if _source(base)!=(directory/report['files']['source']).read_bytes():
        raise ValueError('repetition source differs from original ABI replay')
    for name,digest in identity['headers'].items():
        if Path(name).name!=name or _hash((directory/name).read_bytes())!=digest:
            raise ValueError('repetition header receipt differs')
    library=ctypes.CDLL(str(directory/report['files']['library']))
    run=library.scar_repeat_stateful_v1
    run.argtypes=[ctypes.POINTER(backend._Table),ctypes.POINTER(cpp._Array),ctypes.c_uint64,
        ctypes.POINTER(backend._Scalar),ctypes.c_uint64,ctypes.POINTER(backend._Provider),ctypes.c_uint64,
        ctypes.c_uint64,ctypes.POINTER(ctypes.c_uint64)]
    run.restype=ctypes.c_int32
    return StatefulRepeatArtifact(base,directory,report,library,run)


__all__=['CONDITIONS','RepeatGuardFailure','StatefulRepeatArtifact','build_repeat','load_repeat']
