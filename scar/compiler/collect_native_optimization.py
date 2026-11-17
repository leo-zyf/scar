"""Collect source-bound GCC feedback without loading or running generated code.

This recompiles one retained translation unit with its declared options plus
GCC's diagnostic flag. The result is compiler evidence, never a latency model,
semantic admission, or replacement for the originally admitted native binary.
"""
import argparse
from copy import deepcopy
from hashlib import sha256
import gzip
import io
import json
from pathlib import Path
import shutil
import subprocess

from scar.ir.execution_recipe import artifact_reference, read_artifact

SCHEMA = 'scar.gcc-native-optimization-collection'
_ROOT = Path(__file__).resolve().parents[1]
_SOURCES = ('compiler/collect_native_optimization.py', 'analysis/gcc_optimization.py',
    'compiler/native_lowering_view.py')


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _reference(value, schema):
    reference = artifact_reference(Path(value).absolute(), schema) if isinstance(value, (str, Path)) else deepcopy(value)
    _bytes(reference)
    return reference


def _bytes(reference):
    path = Path(reference['uri'])
    _need(path.is_absolute(), 'absolute retained file reference required')
    raw = path.read_bytes()
    _need(len(raw) == reference['size_bytes'] and sha256(raw).hexdigest() == reference['sha256'],
        'compiler evidence file identity differs: '+str(path))
    return raw


def _write(path, document):
    path.write_text(json.dumps(document, sort_keys=True, indent=2, allow_nan=False)+'\n')
    return artifact_reference(path, document['schema'])


def _write_feedback(path, document):
    raw = json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    path.write_bytes(gzip.compress(raw, mtime=0))
    return artifact_reference(path, document['schema']+'+gzip')


def read_gcc_feedback(reference):
    """Read the bounded gzip JSON projection; raw GCC messages remain separate."""
    raw = _bytes(reference)
    _need(len(raw) <= 16*2**20, 'compressed GCC projection exceeds budget')
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
        decoded = stream.read(64*2**20+1)
    _need(len(decoded) <= 64*2**20, 'decoded GCC projection exceeds budget')
    document = json.loads(decoded)
    _need(reference['schema'] == document['schema']+'+gzip'
        and (document['schema'], document['schema_version']) == ('scar.gcc-optimization-projection', 1),
        'versioned compressed GCC projection required')
    return document


def _pins():
    return {name: sha256((_ROOT/name).read_bytes()).hexdigest() for name in _SOURCES}


def _compiler(value):
    _need(type(value) is dict and set(value) == {'argv', 'identity', 'options'},
        'complete retained compiler argv, identity and options required')
    for name in ('argv', 'options'):
        _need(type(value[name]) is list and value[name] and all(type(s) is str and s for s in value[name]),
            'nonempty explicit compiler '+name+' required')
    _need(type(value['identity']) is str and value['identity'], 'retained compiler identity required')
    # This collector has one explicit shared-library profile. It does not
    # translate another build system or silently change optimization options.
    _need('-shared' in value['options'] and '-fPIC' in value['options'],
        'current collector requires the retained shared-library profile')
    _need(not any(s in ('-o', '-c', '-S', '-E') or s.startswith(('-fopt-info', '-dump', '-save-temps'))
        or s == '-fsave-optimization-record' for s in value['options']),
        'retained options must not override collector outputs or diagnostics')
    return deepcopy(value)


def _command(compiler, source, directory, link_options):
    return [*compiler['argv'], *compiler['options'], '-fsave-optimization-record',
        source['uri'], '-o', str(directory/'analysis.so'), *link_options]


def _projection(record, source, spans, directory):
    from scar.analysis.gcc_optimization import parse_gcc_optimization_record
    return parse_gcc_optimization_record(record, source_ref=source,
        stage_spans=spans, compilation_directory=directory)


def _view_spans(view):
    # Ambiguous textual candidates are retained by the view for inspection,
    # but cannot establish ownership of a compiler diagnostic.
    return [dict(id='stage:'+str(stage['index']), **span)
        for stage in view['stages'] if not stage.get('spans_are_candidates', False)
        for span in stage['source_line_spans']]


def collect_gcc_optimization(source, compiler, out, *, headers=(), stage_spans=(),
                             link_options=('-lm',), subject=None):
    """Recompile one known source; preserve all options and original source path."""
    source = _reference(source, 'scar.cpp-source')
    headers = [_reference(row, 'scar.cpp-header') for row in headers]
    compiler = _compiler(compiler)
    spans = json.loads(json.dumps(list(stage_spans), allow_nan=False))
    links = list(link_options)
    _need(all(type(s) is str and s.startswith('-l') and len(s) > 2 for s in links),
        'explicit library link options required')
    directory = Path(out).absolute()
    _need(not directory.exists(), 'fresh compiler evidence output required')
    pins = _pins()
    executable = shutil.which(compiler['argv'][0])
    _need(executable is not None, 'retained compiler executable is unavailable')
    observed = subprocess.run([*compiler['argv'], '--version'], text=True,
        capture_output=True, check=True).stdout.splitlines()[0]
    _need(observed == compiler['identity'], 'live compiler differs from retained compiler')
    working_directory = str(Path.cwd().absolute())
    directory.mkdir(parents=True)
    command = _command(compiler, source, directory, links)
    process = subprocess.run(command, text=True, capture_output=True, cwd=working_directory)
    log = directory/'build.log'
    log.write_text(process.stdout+process.stderr)
    _need(process.returncode == 0, 'GCC feedback compilation failed: '+process.stderr[-2000:])
    records = list(directory.glob('*.opt-record.json.gz'))
    _need(len(records) == 1, 'one GCC optimization record required for this translation unit')
    for reference in (source, *headers):
        _bytes(reference)
    _need(pins == _pins(), 'feedback collector implementation changed during compilation')
    record = artifact_reference(records[0], 'gcc.optimization-record-json-gzip')
    projection = _projection(record, source, spans, working_directory)
    projected = _write_feedback(directory/'feedback.json.gz', projection)
    document = dict(schema=SCHEMA, schema_version=1, source=source, headers=headers,
        compiler=compiler, observed_compiler_identity=observed,
        observed_compiler_executable=artifact_reference(Path(executable).resolve(), 'scar.compiler-executable'),
        command=command, working_directory=working_directory, link_options=links,
        diagnostic_options=['-fsave-optimization-record'], stage_spans=spans,
        feedback_encoding='GZIP_CANONICAL_JSON',
        subject=deepcopy(subject), implementation_pins=pins,
        files=dict(record=record, feedback=projected,
            library=artifact_reference(directory/'analysis.so', 'scar.analysis-only-native-library'),
            build_log=artifact_reference(log, 'scar.compiler-build-log')),
        summary=projection['summary'], compiler_invoked=True, source_fragment_compiled=True,
        target_program_executed=False, native_loaded=False, latency_measured=False,
        timing_qualified=False, deployment_authorized=False,
        original_admitted_binary_replaced=False, semantic_source_admission_replayed=False,
        limits=['GCC records only transformations that emit optimization diagnostics.',
            'Estimated execution counts and diagnostic counts are not measured execution counts.',
            'Matching source and compiler options do not prove component costs compose.',
            'System headers and full compiler toolchain closure are not pinned by this collector.'])
    _write(directory/'result.json', document)
    return document


def check_gcc_optimization(result):
    """Cold re-read of receipts and projection; no compiler or native loading."""
    reference = _reference(result, SCHEMA)
    document = read_artifact(reference)
    _need((document.get('schema'), document.get('schema_version')) == (SCHEMA, 1),
        'versioned GCC collection required')
    _need(document['implementation_pins'] == _pins(), 'GCC evidence reader implementation differs')
    compiler = _compiler(document['compiler'])
    directory = Path(reference['uri']).parent
    _need(document['command'] == _command(compiler, document['source'], directory, document['link_options'])
        and document['diagnostic_options'] == ['-fsave-optimization-record']
        and document['observed_compiler_identity'] == compiler['identity'],
        'GCC recorded compiler command differs')
    for row in (document['source'], *document['headers'], *document['files'].values(),
                document['observed_compiler_executable']):
        _bytes(row)
    projection = _projection(document['files']['record'], document['source'],
        document['stage_spans'], document['working_directory'])
    _need(document['feedback_encoding'] == 'GZIP_CANONICAL_JSON'
        and read_gcc_feedback(document['files']['feedback']) == projection
        and document['summary'] == projection['summary'], 'GCC feedback differs from raw record replay')
    subject = document['subject']
    if type(subject) is dict and subject.get('kind') == 'STATEFUL_NATIVE_LOWERING':
        from scar.compiler.native_lowering_view import derive_stateful_lowering_view
        retained = subject['view']
        view = derive_stateful_lowering_view(retained['native_manifest'],
            stage_capture=retained['capture']['artifact'])
        _need(view == retained and view['source'] == document['source']
            and view['compiler'] == compiler and list(view['headers'].values()) == document['headers']
            and _view_spans(view) == document['stage_spans'],
            'native compiler stage mapping differs from original lowering replay')
    _need(document['compiler_invoked'] is True and document['source_fragment_compiled'] is True,
        'compiler feedback must disclose its compilation')
    for field in ('target_program_executed', 'native_loaded', 'latency_measured', 'timing_qualified',
                  'deployment_authorized', 'original_admitted_binary_replaced', 'semantic_source_admission_replayed'):
        _need(document[field] is False, 'compiler feedback cannot claim '+field)
    _bytes(reference)
    return document


def collect_native_lowering(native, out, *, stage_capture=None):
    from scar.compiler.native_lowering_view import derive_stateful_lowering_view
    view = derive_stateful_lowering_view(native, stage_capture=stage_capture)
    result = collect_gcc_optimization(view['source'], view['compiler'], out,
        headers=list(view['headers'].values()), stage_spans=_view_spans(view),
        subject=dict(kind='STATEFUL_NATIVE_LOWERING', view=view))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', help='retained stateful artifact.json; no native loading')
    parser.add_argument('--stage-capture', help='optional retained source-capture reference')
    parser.add_argument('--out')
    parser.add_argument('--check')
    args = parser.parse_args(argv)
    if args.check:
        document = check_gcc_optimization(args.check)
    else:
        if not args.native or not args.out:
            parser.error('--native and --out are required')
        document = collect_native_lowering(args.native, args.out, stage_capture=args.stage_capture)
    print(json.dumps(dict(schema=document['schema'], summary=document['summary'],
        files=document['files'], latency_measured=document['latency_measured']), sort_keys=True))


if __name__ == '__main__':
    main()
