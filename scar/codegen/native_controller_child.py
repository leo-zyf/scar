"""Lower a source-qualified child effect using the controller ABI."""

from dataclasses import dataclass
from pathlib import Path
import re

from scar.ir.execution_recipe import read_artifact


HEADER = 'include/scar_generated/native_controller_child.hpp'
_TEMPLATE = Path(__file__).with_name('native_controller_child_v1.hpp.in')


def _need(condition, detail):
    if not condition:
        raise ValueError('native controller child: ' + detail)


def _literal(value):
    _need(type(value) is str and '\0' not in value, 'malformed original argument string')
    return '"' + ''.join(f'\\x{byte:02x}' for byte in value.encode('utf-8')) + '"'


@dataclass(frozen=True, slots=True)
class NativeControllerChild:
    reference: dict
    site: dict
    contract: dict
    source: bytes

    def members(self):
        return {HEADER: ('scar.cpp-header', self.source)}

    def artifacts(self):
        return dict(planning=self.reference, effect_site=self.site, contract=self.contract,
                    mode='NATIVE_SYNCHRONOUS_CHILD_EXPLICIT_VALIDATION',
                    executable_guard='RUNTIME_SHA256_AND_FILE_STAMP_BEFORE_ABSOLUTE_SPAWN',
                    remaining_effects_preserved=True, whole_program_translated=False,
                    deployment_authorized=False)


def _source(ordinal, executable, argv):
    arguments = []
    for item in argv:
        if item['kind'] == 'LITERAL':
            arguments.append(_literal(item['value']))
        elif item['kind'] == 'DIRECTORY_JOIN':
            arguments.append('joined(' + _literal(item['component']) + ')')
        else:
            raise ValueError('native controller child: unsupported original argv operand')
    template = _TEMPLATE.read_text()
    replacements = dict(ORDINAL=ordinal, EXECUTABLE=_literal(executable['uri']),
                        EXECUTABLE_SHA256=executable['sha256'], EXECUTABLE_SIZE=executable['size_bytes'],
                        ARGUMENT_COUNT=len(argv), ARGUMENTS=',\n            '.join(arguments))
    for key, value in replacements.items():
        template = template.replace(f'@{key}@', str(value))
    return template.encode()


def prepare(reference, *, controller_reference, controller_plan):
    """Bind one checked process effect to the replayed controller route."""
    from scar.compiler.plan_controller_child import read_controller_child_plan

    report = read_controller_child_plan(reference['uri'])
    _need(report == read_artifact(reference) and report['controller_plan'] == controller_reference,
          'process plan differs from the current controller planning receipt')
    _need(report['status'] == 'CONDITIONAL_EXPLICIT_TRIAL_ONLY'
          and report['whole_entry_authorized'] is False and report['deployment_authorized'] is False,
          'qualified explicit child trial scope required')
    site = report['effect_site']
    ordinal = site['ordinal']
    effects = controller_plan['effect_sites']
    _need(type(ordinal) is int and 0 <= ordinal < len(effects)
          and effects[ordinal]['ordinal'] == ordinal
          and effects[ordinal]['site'] == site['site']
          and effects[ordinal]['source_fact'] == site['source_fact']
          and effects[ordinal]['positional_count'] == 0 and not effects[ordinal]['keywords'],
          'selected child differs from the original zero-argument effect')
    contract = report['child']
    _need(contract['runtime_directory'] == 'EXPLICIT_ABSOLUTE_ARGUMENT'
          and all(contract[key] is True for key in ('inherit_cwd', 'inherit_environment', 'inherit_stdio',
                                                      'close_other_fds', 'restore_default_signals'))
          and contract['shell'] is contract['check_exit_status'] is False
          and contract['completion'] == 'SYNCHRONOUS_WAITPID'
          and contract['nonzero_and_signal'] == 'RETURN_NORMALLY_AND_DISCARD_ORIGINAL_COMPLETED_PROCESS'
          and contract['launch_or_wait_failure'] == 'PROPAGATE_BACKEND_FAILURE',
          'unsupported original subprocess contract')
    path = contract['directory_field_path']
    _need(type(path) is list and len(path) >= 2 and all(type(part) is str and part.isidentifier() for part in path),
          'original directory field route missing')
    executable = contract['executable']
    digest = executable['sha256']
    _need(type(executable['uri']) is str and Path(executable['uri']).is_absolute()
          and type(executable['size_bytes']) is int and 0 < executable['size_bytes'] < (1 << 63)
          and re.fullmatch(r'[0-9a-f]{64}', digest), 'absolute pinned original program image required')
    read_artifact(executable, json_document=False)

    argv = contract['argv']
    _need(type(argv) is list and 0 < len(argv) <= 256 and argv[0]['kind'] == 'LITERAL' and argv[0]['value'],
          'original ordered argv needs its literal argv[0]')
    return NativeControllerChild(reference, site, contract, _source(ordinal, executable, argv))
