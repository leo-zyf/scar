"""Lower a qualified source console effect to a bounded native overlay."""

from dataclasses import dataclass
from pathlib import Path

from scar.ir.execution_recipe import read_artifact


HEADER = 'include/scar_generated/native_controller_console.hpp'
_TEMPLATE = Path(__file__).with_name('native_controller_console_v1.hpp.in')


def _need(condition, detail):
    if not condition:
        raise ValueError('native controller console: ' + detail)


@dataclass(frozen=True, slots=True)
class NativeControllerConsole:
    reference: dict
    site: dict
    contract: dict
    source: bytes

    def members(self):
        return {HEADER: ('scar.cpp-header', self.source)}

    def artifacts(self):
        return dict(planning=self.reference, effect_site=self.site, contract=self.contract,
                    mode='NATIVE_BUFFERED_CONSOLE_EXPLICIT_VALIDATION',
                    runtime_requires_exclusive_stdout=True, remaining_effects_preserved=True,
                    whole_program_translated=False, deployment_authorized=False)


def prepare(reference, *, controller_reference, controller_plan):
    """Replay the selected proof and bind its exact effect to this controller."""
    from scar.compiler.plan_controller_console import read_controller_console_plan

    report = read_controller_console_plan(reference['uri'])
    _need(report == read_artifact(reference) and report['controller_plan'] == controller_reference,
          'console plan must replay against the current controller planning receipt')
    _need(report['status'] == 'CONDITIONAL_EXPLICIT_TRIAL_ONLY'
          and report['whole_entry_authorized'] is False and report['deployment_authorized'] is False,
          'console needs the qualified explicit trial scope')
    site = report['effect_site']
    ordinal = site['ordinal']
    effects = controller_plan['effect_sites']
    _need(type(ordinal) is int and 0 <= ordinal < len(effects)
          and effects[ordinal]['ordinal'] == ordinal
          and effects[ordinal]['site'] == site['site']
          and effects[ordinal]['source_fact'] == site['source_fact']
          and effects[ordinal]['positional_count'] == 1 and not effects[ordinal]['keywords'],
          'selected console call differs from the replayed unary effect')
    contract = report['console']
    bound = contract['max_buffer_bytes']
    _need(contract['encoding'] == 'UTF-8' and contract['ascii_only'] is True
          and contract['append_lf'] is True and contract['destination_fd'] == 1
          and contract['destination_kind'] == 'REGULAR_FILE'
          and contract['publication_horizon'] == 'NORMAL_PROCESS_EXIT'
          and contract['exclusive_destination_contract_required'] is True
          and type(bound) is int and 0 < bound < 8192,
          'unsupported stdout stream, text domain or publication contract')
    source = _TEMPLATE.read_text().replace('@ORDINAL@', str(ordinal)).replace('@CAPACITY@', str(bound))
    return NativeControllerConsole(reference, site, contract, source.encode())
