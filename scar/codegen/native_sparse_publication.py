"""Qualify an independent native SHA-256 publication from exact source ports."""

import argparse
import ast
import ctypes
from dis import get_instructions
import json
from pathlib import Path
from types import CodeType

from scar.analysis.replay_transaction import replay_transaction
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.runtime.csr_region_entry import interpreter_context
from scar.trace.code_snapshot import code_key


SCHEMA = 'scar.native-sparse-publication'
STREAM = 'scar.native-ordered-source-digest-stream'
ENTRY = 'scar.entry-call-dependence'
RELEASE = 'scar.sparse-parallel-entry-selection'
PROFILE = dict(algorithm='SHA256', input='ORDERED_CONTIGUOUS_LITTLE_ENDIAN_BINARY64_SOLUTION_BYTES',
    stdout='LOWERCASE_ASCII_HEX_PLUS_LF', emission='ONLY_AFTER_EXACT_COUNT_AND_NORMAL_COMPLETION',
    mode='EXPLICIT_INDEPENDENT_NATIVE_STDOUT', original_python_effects_discharged=False)
SYSTEM_LIBRARIES = ('libc.so.6', 'libdl.so.2', 'libpthread.so.0')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _record(document, locator, table):
    prefix = f'#/{table}/'
    _require(type(locator) is str and locator.startswith(prefix) and locator[len(prefix):].isdecimal(),
        'publication locator does not address its admitted table')
    return document[table][int(locator[len(prefix):])]


def _native_call(report, locator, carrier, caller, module=None):
    record = _record(report, locator, 'observed_targets')
    observation = record['observation']
    identity = json.loads(observation['metadata_json'])['native_boundary']
    _require(record['entry_scope_reachable'] and observation['caller_site']['code'] == caller
        and identity['qualname'] == carrier and (module is None or identity.get('module') == module),
        'publication observed carrier or source CodeID differs')
    return observation


def _verify_arguments(calls, code, observation, count):
    site = observation['caller_site']
    opcode, = (op for op in get_instructions(code) if op.offset == site['offset'])
    _require(opcode.opname == site['opname'] and list(opcode.positions) == site['position'],
        'publication source call offset differs from the sealed source')
    lineno, end_lineno, col, end_col = site['position']
    call, = (node for node in calls if (node.lineno, node.end_lineno, node.col_offset, node.end_col_offset)
        == (lineno, end_lineno, col, end_col))
    _require(len(call.args) == count and all(type(arg) is not ast.Starred for arg in call.args) and not call.keywords,
        'publication call differs from the fixed positional ABI')


class _DlInfo(ctypes.Structure):
    _fields_ = [(name, ctypes.c_void_p) for name in ('path', 'base', 'symbol', 'address')]


def _crypto_provider():
    import _hashlib

    root = Path(_hashlib.__file__).resolve(strict=True)
    handle = ctypes.CDLL(str(root))
    address_info = ctypes.CDLL(None).dladdr
    address_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DlInfo)]
    address_info.restype = ctypes.c_int
    identities = set()
    for name in ('OPENSSL_init_crypto', 'EVP_sha256', 'EVP_MD_CTX_new', 'EVP_MD_CTX_free', 'EVP_DigestInit_ex',
            'EVP_DigestUpdate', 'EVP_DigestFinal_ex'):
        row = _DlInfo()
        _require(address_info(getattr(handle, name), ctypes.byref(row)) == 1 and row.path,
            'cannot qualify the original hashlib linked SHA-256 library')
        identities.add(Path(ctypes.string_at(row.path).decode()).resolve(strict=True))
    provider, = identities
    return dict(original_hashlib=artifact_reference(root, 'scar.elf-library'),
        libcrypto=artifact_reference(provider, 'scar.elf-library'),
        abi='OPENSSL_EVP_SHA256', external_config_loaded=False)


@replay_transaction()
def project(release_ref, entry_ref, *, replay_entry=False):
    selected, entry = read_artifact(release_ref), read_artifact(entry_ref)
    _require(release_ref['schema'] == RELEASE and entry_ref['schema'] == ENTRY and entry['schema_version'] == 1
        and selected['decision'] == 'PARALLEL' and selected['study_parallel'] is False
        and selected['context']['interpreter'] == interpreter_context(),
        'publication requires an exact qualified release and its source interpreter')
    if replay_entry:
        from scar.analysis.entry_call_dependence import check

        check(entry_ref['uri'])
        read_artifact(entry_ref, json_document=False)
    source_ref = selected['source']
    scope = read_artifact(source_ref)
    publication = scope['dependences']['ordered_publication']
    projection = read_artifact(scope['pipeline_scope'])['source_graphs']['sparse_solve']
    _require(entry['entry_calls'] == projection and publication['kind'] == 'ORDERED_HASH_ACCUMULATION'
        and publication['requires_original_provider_and_normal_completion'] is True
        and all(publication[key] for key in ('constructor_to_binding', 'original_return_transport', 'ordered_world'))
        and publication['solve_result_to_publication']['path'],
        'source publication F4 routes are incomplete or not from the same entry graph')
    consumer, = entry['return_consumers']
    printer, = consumer['observations']
    _require((consumer['role'], consumer['slot'], consumer['effect_discharge_proven']) == ('POSITIONAL', 0, False)
        and consumer['data_route'] and consumer['world_route'] and entry['qualification']['content_effects_discharged'] is False,
        'original print consumer must retain its single ordered data and world paths')
    report = read_artifact(projection)
    caller = scope['source_body']['code_key']
    entry_source = selected['entry']
    raw = read_artifact(entry_source, json_document=False)
    _require(entry_source == scope['entry']['source'] and scope['source_body']['path'] == entry_source['uri'],
        'publication body source pin differs')
    module = compile(raw, entry_source['uri'], 'exec', dont_inherit=True, optimize=selected['context']['interpreter']['optimize'])
    body, = (code for code in module.co_consts if type(code) is CodeType and code_key(code) == caller)
    syntax = [node for node in ast.walk(ast.parse(raw, filename=entry_source['uri'])) if type(node) is ast.Call]
    calls = dict(constructor=publication['original_constructor'], update=publication['original_update'],
        final=publication['original_final'], stdout=printer)
    for key, carrier, module_name, code, arity in (
            ('constructor', 'openssl_sha256', '_hashlib', body, 0), ('update', 'HASH.update', None, body, 1),
            ('final', 'HASH.hexdigest', None, body, 0), ('stdout', 'print', 'builtins', module, 1)):
        observation = _native_call(report, calls[key], carrier, code_key(code), module_name)
        _verify_arguments(syntax, code, observation, arity)

    return dict(schema=SCHEMA, schema_version=1, release=release_ref, execution_recipe=selected['execution_recipe'],
        entry_source=entry_source, parallel_scope=source_ref, entry_dependence=entry_ref, source_calls=projection,
        body_code_key=caller, original_carrier_calls=calls, provider=_crypto_provider(), profile=PROFILE)


def project_stream(source, contract):
    """Project success-only output. The final stdout write itself is external I/O."""
    _require(contract['schema'] == SCHEMA and contract['schema_version'] == 1 and contract['profile'] == PROFILE
        and contract['body_code_key'] == source['source']['body_code_key'],
        'native digest stream requires the exact publication contract and source body')
    old_output = source['output']
    sizes = source['bytes']
    return dict(source, schema=STREAM,
        output=dict(profile=PROFILE, success_bytes=65, failure_stdout_bytes=0,
            provider_sha256=contract['provider']['libcrypto']['sha256']),
        binary_consumer=dict(argument='--consume-records', input=old_output, completion='EXACT_TERMINAL_AND_EOF'),
        bytes={**sizes, 'binary_consumer_input_complete': sizes['output_complete'], 'output_complete': 65})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--entry-dependence', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    release = artifact_reference(args.release.resolve(strict=True), RELEASE)
    entry = artifact_reference(args.entry_dependence.resolve(strict=True), ENTRY)
    document = project(release, entry)
    with args.out.open('x') as output:
        json.dump(document, output, sort_keys=True, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps(artifact_reference(args.out, SCHEMA)))


if __name__ == '__main__':
    main()
