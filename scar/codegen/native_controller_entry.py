"""Bind a source controller to the qualified native update and external effects."""
from dataclasses import dataclass
import json
from pathlib import Path
import re

from scar.codegen.native_controller_child import NativeControllerChild
from scar.codegen.native_controller_console import NativeControllerConsole
from scar.codegen.native_update_entry import NativeUpdate, prepare as prepare_update
from scar.compiler.plan_native_controller import read_native_controller_plan
from scar.ir.execution_recipe import read_artifact


SOURCE = 'src/controller.cpp'
ABI = 'include/scar_generated/controller_effect_abi_v1.h'
LOADER = 'include/scar_generated/native_controller_effects.hpp'


def _render(name, values):
    source = Path(__file__).with_name(name).read_text()
    return re.sub(r'@([A-Z_]+)@', lambda match: str(values[match[1]]), source)


def _unique(items, reason):
    matches = list(items)
    if len(matches) != 1:
        raise ValueError(reason)
    return matches[0]


@dataclass(frozen=True, slots=True)
class NativeController:
    reference: dict
    plan: dict
    update: NativeUpdate
    source: bytes
    numeric_slots: tuple[tuple[int, int], ...]
    extra_inputs: tuple[tuple[int, str], ...]
    guards: tuple[tuple[str, bool], ...]
    outputs: tuple[int, ...]
    call: dict
    carried_index: int
    console: NativeControllerConsole | None = None
    child: NativeControllerChild | None = None

    def members(self):
        members = {
            SOURCE: ('scar.cpp-source', self.source),
            ABI: ('scar.cpp-header', Path(__file__).with_name('controller_effect_abi_v1.h').read_bytes()),
            LOADER: ('scar.cpp-header', Path(__file__).with_name('native_controller_effects.hpp.in').read_bytes()),
        }
        if self.console is not None:
            members.update(self.console.members())
        if self.child is not None:
            members.update(self.child.members())
        return members

    def artifacts(self):
        summary = self.plan
        report = dict(planning=self.reference, controller_sha256=summary['plan_sha256'],
                    effects=summary['effect_sites'], update=self.call,
                    guard_inputs=[dict(argument=path, expected=value) for path, value in self.guards],
                    scalar_inputs=[dict(slot=i, argument=path) for i, path in self.extra_inputs],
                    numeric_bindings=[dict(slot=i, numeric_ordinal=index) for i, index in self.numeric_slots],
                    state_outputs=[summary['slots'][index] for index in self.outputs],
                    state_encoding='LE_RECORDS_U32_KIND_U32_RESERVED_I64_INTEGER_F64_FLOATING',
                    opaque_effect_provider='REQUIRED_EXPLICIT_NATIVE_ABI_V1',
                    whole_program_translated=False, deployment_authorized=False)
        if self.console is not None:
            report['console'] = self.console.artifacts()
        if self.child is not None:
            report['child'] = self.child.artifacts()
            report['child']['directory_argument'] = '.'.join(self.child.contract['directory_field_path'])
        return report


def prepare(reference, metadata, ports, *, console_reference=None, child_reference=None):
    from scar.codegen.native_controller import prepare as generate

    report, _ = read_native_controller_plan(reference['uri'])
    if report != read_artifact(reference):
        raise ValueError('native controller plan changed during replay')
    update = prepare_update(report['update_plan'], metadata, ports)
    original = read_artifact(report['source'])
    plan, source = generate(original, array_ids=[row['id'] for row in ports if row['kind'] == 'array'])
    site = _unique(plan['update_sites'], 'exactly one original update call site required')
    receiver = _unique((row for row in site['formal_routes'] if row['binding']['role'] == 'ATTRIBUTE_RECEIVER'),
                       'original update requires exactly one bound receiver')
    carried = _unique((row['binding'] for row in site['formal_routes']
                       if row['parameter'] == update.scalar_argument
                       and row['binding']['role'] in ('POSITIONAL', 'KEYWORD')),
                      'original update scalar needs one explicit caller actual')
    carried_index = carried['ordinal'] + (site['positional_count'] if carried['role'] == 'KEYWORD' else 0)
    update_proof = read_artifact(read_artifact(report['update_plan'])['source'])

    def map_path(path):
        if path[0] != receiver['parameter']:
            raise ValueError('original update input is independent of its selected receiver')
        return (*receiver['binding']['source_path'], *path[1:])

    by_id = {row['id']: map_path(row['parent_path']) for row in update_proof['scalar_inputs']}
    numeric_indices = {by_id[row['id']]: index for index, row in enumerate(
        port for port in ports if port['kind'] == 'scalar')}
    guard_paths = {map_path(path.split('.')): expected for path, expected in update.entry_guards}
    numeric_slots, extra_inputs, outputs = [], [], []
    for slot in plan['slots']:
        index, path = slot['ordinal'], tuple(slot['path'])
        if slot['input']:
            if slot['kind'] == 'FIELD' and path in numeric_indices:
                numeric_slots.append((index, numeric_indices[path]))
            else:
                extra_inputs.append((index, '.'.join(path)))
        if slot['output']:
            outputs.append(index)
    guards = tuple(('.'.join(path), expected) for path, expected in guard_paths.items())
    if any(slot['kind'] == 'FIELD' and tuple(slot['path']) in guard_paths for slot in plan['slots']):
        raise ValueError('controller and update dispatch guard require one common state slot')
    if console_reference is not None and child_reference is not None:
        raise ValueError('child and buffered console need a shared inherited stdout contract')
    console = child = None
    if console_reference is not None:
        from scar.codegen.native_controller_console import prepare as prepare_console
        console = prepare_console(console_reference, controller_reference=reference, controller_plan=plan)
    if child_reference is not None:
        from scar.codegen.native_controller_child import prepare as prepare_child
        child = prepare_child(child_reference, controller_reference=reference, controller_plan=plan)
    return NativeController(reference, plan, update, source, tuple(numeric_slots), tuple(extra_inputs),
                            guards, tuple(outputs), site, carried_index, console, child)


def render(controller, setup, selected):
    def quote(value):
        return json.dumps(value, ensure_ascii=False)

    initial_numbers = []
    updated_numbers = []
    for slot, index in controller.numeric_slots:
        value = f'arguments.values[{index}]'
        initial_numbers.append(f'''    controller_slots[{slot}] = {{
        {value}.integer ? SCAR_CONTROLLER_INT64_V1 : SCAR_CONTROLLER_FLOAT64_V1,
        0, {value}.integer ? static_cast<std::int64_t>({value}.value) : 0, {value}.value, nullptr, 0}};''')
        updated_numbers.append(f'''        {{
            const auto& value = current[{slot}];
            if (value.kind == SCAR_CONTROLLER_FLOAT64_V1) {{
                if (!std::isfinite(value.floating)) return -2;
                {value} = {{value.floating, 0}};
            }} else if (value.kind == SCAR_CONTROLLER_INT64_V1 && value.integer >= -limit && value.integer <= limit) {{
                {value} = {{static_cast<double>(value.integer), 1}};
            }} else {{
                return -2;
            }}
        }}''')
    updated_numbers.extend(
        f'        if (arguments.values[{index}].integer) return -2;'
        for index in controller.update.scalar_reads)
    plan = controller.plan
    with_console = controller.console is not None
    with_child = controller.child is not None
    effect_arguments = ''
    if with_console:
        effect_arguments = '''    require(next < argc && std::string_view(argv[next]) == "--exclusive-stdout",
            "native console requires an explicit --exclusive-stdout effect contract");
    ++next;
'''
    if with_child:
        path = quote('.'.join(controller.child.contract['directory_field_path']))
        effect_arguments = f'''    require(next + 2 < argc && std::string_view(argv[next]) == "--effect-directory" &&
            std::string_view(argv[next+1]) == {path},
            "native child requires its explicit original directory field binding");
    const auto* child_directory = argv[next+2];
    next += 3;
'''
    entry = _render('native_controller_arguments_v1.cpp.in', {
        'SLOT_COUNT': len(plan['slots']),
        'EFFECT_ARGUMENTS': effect_arguments,
        'CONTROLLER_INPUTS': ''.join(f'    controller_slots[{index}] = read({quote(path)});\n'
                                     for index, path in controller.extra_inputs),
        'GUARDS': ''.join(f'''    require(next + 2 < argc && std::string_view(argv[next]) == "--guard" &&
            std::string_view(argv[next+1]) == {quote(path)} &&
            std::string_view(argv[next+2]) == "{str(value).lower()}", "original update branch guard differs");
    next += 3;
''' for path, value in controller.guards),
        'NUMBER_INPUTS': '\n'.join(initial_numbers),
    })
    declarations = '''#include "controller_effect_abi_v1.h"
#include "native_controller_effects.hpp"

extern "C" int scar_array_copy_preflight_v1(const scar_array_v1*, std::uint64_t) noexcept;
extern "C" int scar_array_copy_run_v1(const scar_array_v1*, std::uint64_t) noexcept;
extern "C" const char* scar_array_copy_plan_sha256_v1() noexcept;

'''
    selected_overlay = 'console' if with_console else 'child' if with_child else None
    dispatch = '''        return context.effects->call(call, values, count, roots, root_count);'''
    overlay_setup = overlay_member = overlay_pointer = ''
    if selected_overlay is not None:
        declarations = f'#include "native_controller_{selected_overlay}.hpp"\n' + declarations
        overlay_type = f'Controller{selected_overlay.title()}Overlay'
        overlay_argument = 'stdout' if with_console else 'child_directory'
        overlay_setup = f'    {overlay_type} {selected_overlay}({overlay_argument});\n'
        overlay_member = f'        {overlay_type}* {selected_overlay};\n'
        overlay_pointer = f'&{selected_overlay}, '
        dispatch = f'        return context.{selected_overlay}->dispatch(call, values, count, roots, root_count,\n' + '''
            [&](auto ordinal, auto* args, auto nargs, auto* arrays, auto narrays) {
                return context.effects->call(ordinal, args, nargs, arrays, narrays);
            });'''.lstrip('\n')
    body = _render('native_controller_entry_v1.cpp.in', {
        'IDENTITY': plan['plan_sha256'],
        'COPY_IDENTITY': controller.update.copies['plan_sha256'],
        'EFFECT_COUNT': len(plan['effect_sites']),
        'PROVIDER_SETUP': setup,
        'PROVIDER_POINTER': '&provider' if selected else 'nullptr',
        'PROVIDER_COUNT': int(selected),
        'OVERLAY_SETUP': overlay_setup,
        'OVERLAY_CONTEXT_MEMBER': overlay_member,
        'OVERLAY_POINTER': overlay_pointer,
        'EFFECT_DISPATCH': dispatch,
        'UPDATE_ORDINAL': controller.call['ordinal'],
        'UPDATE_ARITY': controller.call['positional_count'] + len(controller.call['keywords']),
        'SLOT_COUNT': len(plan['slots']),
        'CARRIED_INDEX': controller.carried_index,
        'UPDATE_NUMBER_BINDINGS': '\n'.join(updated_numbers) + '\n',
        'SCALAR_RETURN': controller.update.scalar_code,
    })
    slots = ', '.join(str(index) for index in controller.outputs)
    output = f'''    std::vector<double> state_words;
    for (const auto index : std::array<std::size_t, {len(controller.outputs)}>{{{slots}}}) {{
        const auto& state = controller_slots[index];
        require(state.kind != SCAR_CONTROLLER_UTF8_V1, "unsupported text controller state output");
        state_words.push_back(std::bit_cast<double>(static_cast<std::uint64_t>(state.kind)));
        state_words.push_back(std::bit_cast<double>(state.integer));
        state_words.push_back(state.floating);
    }}
    auto controller_output = std::make_unique<PendingFile>(controller_out);
    controller_output->write(state_words);
    pending.push_back(std::move(controller_output));
'''
    finalize = '''    require(console.finish() == 0, "native console declined final stdout publication");
''' if with_console else ''
    return dict(UPDATE_DECLARATIONS=declarations, ENTRY_ARGUMENTS=entry, RUN_BODY=body,
                UPDATE_OUTPUT_COUNT=' + 1', UPDATE_OUTPUT=output, FINALIZE_EFFECTS=finalize)
