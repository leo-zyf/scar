"""Bind a source controller to the qualified native update and external effects."""
from dataclasses import dataclass
import json
from pathlib import Path
import re

from scar.codegen.native_controller_child import NativeControllerChild
from scar.codegen.native_controller_console import NativeControllerConsole
from scar.codegen.native_controller_plot import NativeControllerPlot
from scar.codegen.native_controller_startup import NativeControllerStartup
from scar.codegen.native_update_entry import NativeUpdate, prepare as prepare_update
from scar.compiler.plan_native_controller import read_native_controller_plan
from scar.ir.execution_recipe import read_artifact


SOURCE = 'src/controller.cpp'
ABI = 'include/scar_generated/controller_effect_abi_v1.h'
LOADER = 'include/scar_generated/native_controller_effects.hpp'
_EXTERNAL_ARGS = '''    require(argc >= 7 && std::string_view(argv[1]) == "--effect-library" &&
            std::string_view(argv[3]) == "--iteration-limit" && std::string_view(argv[5]) == "--state-out",
            "expected --effect-library FILE --iteration-limit N --state-out FILE");
    const auto* effect_path = argv[2];
    const auto iteration_limit = number<std::uint64_t>(argv[4]);
    require(iteration_limit > 0, "positive iteration limit required");
    const auto requested = fs::absolute(fs::path(argv[6]));
    require(!requested.filename().empty() && requested.filename() != "." && requested.filename() != "..",
            "controller state output must be a file");
    const auto controller_out = fs::canonical(requested.parent_path()) / requested.filename();
'''
_BUILTIN_ARGS = '''    require(argc >= 6 && std::string_view(argv[1]) == "--iteration-limit",
            "expected --iteration-limit N [--state-out FILE] --effect-directory PORT DIRECTORY");
    const auto iteration_limit = number<std::uint64_t>(argv[2]);
    require(iteration_limit > 0, "positive iteration limit required");
    fs::path controller_out;
    int next = 3;
    if (std::string_view(argv[next]) == "--state-out") {
        const auto requested = fs::absolute(fs::path(argv[next + 1]));
        require(!requested.filename().empty() && requested.filename() != "." && requested.filename() != "..",
                "controller state output must be a file");
        controller_out = fs::canonical(requested.parent_path()) / requested.filename();
        next += 2;
    }
'''


def _render(name, values):
    source = Path(__file__).with_name(name).read_text()
    return re.sub(r'@([A-Z_]+)@', lambda match: str(values[match[1]]), source)


def _unique(items, reason):
    matches = list(items)
    if len(matches) != 1:
        raise ValueError(reason)
    return matches[0]


@dataclass(frozen=True, slots=True)
class PublicControllerArgv:
    reference: dict
    header: str
    source: bytes

    def members(self):
        return {self.header: ('scar.cpp-header', self.source)}

    def artifacts(self):
        return dict(planning=self.reference, header=self.header, driver='NATIVE_ORIGINAL_OPTION_SPELLINGS',
                    same_process=True, whole_program_translated=False, deployment_authorized=False)


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
    startup: NativeControllerStartup | None = None
    plot: NativeControllerPlot | None = None
    public_argv: PublicControllerArgv | None = None

    def members(self):
        members = {
            SOURCE: ('scar.cpp-source', self.source),
            ABI: ('scar.cpp-header', Path(__file__).with_name('controller_effect_abi_v1.h').read_bytes()),
            LOADER: ('scar.cpp-header', Path(__file__).with_name('native_controller_effects.hpp.in').read_bytes()),
        }
        if self.console is not None and self.plot is None:
            members.update(self.console.members())
        if self.child is not None:
            members.update(self.child.members())
        if self.startup is not None:
            members.update(self.startup.members())
        if self.plot is not None:
            members.update(self.plot.members())
        if self.public_argv is not None:
            members.update(self.public_argv.members())
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
        if self.startup is not None:
            report['startup'] = self.startup.artifacts()
        if self.plot is not None:
            report['plot'] = self.plot.artifacts()
            report['opaque_effect_provider'] = 'NONE_IN_ADMITTED_NATIVE_CONTROLLER_SCOPE'
            report['console']['mode'] = 'NATIVE_ASCII_STDOUT_WITH_INHERITED_CHILD'
            report['console']['runtime_requires_exclusive_stdout'] = False
            report['final_state_files_optional'] = True
        if self.public_argv is not None:
            report['public_argv'] = self.public_argv.artifacts()
        return report


def _prepare_public_argv(reference, controller_source, startup, child, plot, extra_inputs, guards, ports):
    from scar.analysis.source_controller_argv import check
    from scar.codegen.native_controller_argv import HEADER, render

    if startup is None or child is None or plot is None:
        raise ValueError('public original arguments require the qualified startup and native effects')
    recipe = check(reference['uri'])
    if (recipe != read_artifact(reference) or recipe['startup'] != startup.reference
            or recipe['controller'] != controller_source):
        raise ValueError('public original arguments must bind the current startup/controller route')
    arrays = [row for row in ports if row['kind'] == 'array']
    if set(startup.array_ordinals) != set(range(len(arrays))) or any(
            path not in startup.guard_values for path, _ in guards):
        raise ValueError('public arguments leave a numerical array or branch ingress unbound')
    slots = [path for index, path in extra_inputs if index not in startup.slot_ordinals]
    numerics = [row['id'] for index, row in enumerate(item for item in ports if item['kind'] == 'scalar')
                if index not in startup.numeric_ordinals]
    directory = '.'.join(child.contract['directory_field_path'])
    return PublicControllerArgv(reference, HEADER, render(recipe, slot_arguments=slots,
                                                          numeric_ports=numerics, directory_argument=directory))


def prepare(reference, metadata, ports, *, console_reference=None, child_reference=None,
            startup_reference=None, plot_reference=None, plot_provider=None, plot_residency=None,
            argv_reference=None, package=None):
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
    if console_reference is not None and child_reference is not None and plot_reference is None:
        raise ValueError('child and buffered console need a shared inherited stdout contract')
    if any((plot_reference, plot_provider, plot_residency)) and not all(
            (plot_reference, plot_provider, startup_reference, console_reference, child_reference)):
        raise ValueError('native plot entry needs source, provider, startup, console and child proofs')
    console = child = startup = None
    if console_reference is not None:
        from scar.codegen.native_controller_console import prepare as prepare_console
        console = prepare_console(console_reference, controller_reference=reference, controller_plan=plan)
    if child_reference is not None:
        from scar.codegen.native_controller_child import prepare as prepare_child
        child = prepare_child(child_reference, controller_reference=reference, controller_plan=plan)
    if startup_reference is not None:
        from scar.codegen.native_controller_startup import prepare as prepare_startup
        startup = prepare_startup(startup_reference, controller_reference=reference, controller_plan=plan,
                                  ports=ports, package=package, borrow_arrays=plot_reference is not None)
        for path, expected in guards:
            if path in startup.guard_values and startup.guard_values[path] != expected:
                raise ValueError('initial constructor guard differs from the original numerical branch')
    plot = None
    if plot_reference is not None:
        from scar.codegen.native_controller_plot import prepare as prepare_plot

        plot = prepare_plot(plot_reference, plot_provider, controller_plan=plan,
                            startup=startup, console=console, child=child, ports=ports, residency=plot_residency)
    public_argv = None
    if argv_reference is not None:
        public_argv = _prepare_public_argv(argv_reference, report['source'],
                                          startup, child, plot, extra_inputs, guards, ports)
    return NativeController(reference, plan, update, source, tuple(numeric_slots), tuple(extra_inputs),
                            guards, tuple(outputs), site, carried_index, console, child, startup, plot, public_argv)


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
    with_plot = controller.plot is not None
    startup = controller.startup
    effect_arguments = ''
    if with_console and not with_plot:
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
    read_inputs = controller.extra_inputs
    read_guards = controller.guards
    initialize = ''
    if startup is not None:
        read_inputs = tuple(row for row in read_inputs if row[0] not in startup.slot_ordinals)
        read_guards = tuple(row for row in read_guards if row[0] not in startup.guard_values)
        initialize = '    ControllerStartup startup;\n'
    entry = _render('native_controller_arguments_v1.cpp.in', {
        'PREFIX': _BUILTIN_ARGS if with_plot else _EXTERNAL_ARGS,
        'LEGACY_NEXT': '' if with_plot else '    int next = 7;\n',
        'SLOT_COUNT': len(plan['slots']),
        'EFFECT_ARGUMENTS': effect_arguments,
        'OPTIONAL_STATE_BEGIN': 'controller_out.empty() || (' if with_plot else '',
        'OPTIONAL_STATE_END': ')' if with_plot else '',
        'CONTROLLER_INPUTS': ''.join(f'    controller_slots[{index}] = read({quote(path)});\n'
                                     for index, path in read_inputs),
        'GUARDS': ''.join(f'''    require(next + 2 < argc && std::string_view(argv[next]) == "--guard" &&
            std::string_view(argv[next+1]) == {quote(path)} &&
            std::string_view(argv[next+2]) == "{str(value).lower()}", "original update branch guard differs");
    next += 3;
''' for path, value in read_guards),
        'STARTUP': initialize,
        'STARTUP_SCALARS': ''.join(
            f'    require(startup.initialize_scalar({index}, arguments.values[{index}]), "missing startup scalar");\n'
            for index in startup.numeric_ordinals) if startup is not None else '',
        'NUMBER_INPUTS': '\n'.join(initial_numbers),
        'STARTUP_SLOTS': ''.join(
            f'\n    require(startup.initialize_slot({index}, controller_slots[{index}]), "missing startup field");'
            for index in startup.slot_ordinals) if startup is not None else '',
    })
    declarations = '''#include "controller_effect_abi_v1.h"
#include "native_controller_effects.hpp"

extern "C" int scar_array_copy_preflight_v1(const scar_array_v1*, std::uint64_t) noexcept;
extern "C" int scar_array_copy_run_v1(const scar_array_v1*, std::uint64_t) noexcept;
extern "C" const char* scar_array_copy_plan_sha256_v1() noexcept;

'''
    selected_overlay = 'console' if with_console and not with_plot else 'child' if with_child else None
    dispatch = '''        return context.effects->call(call, values, count, roots, root_count);'''
    overlay_setup = overlay_member = overlay_pointer = ''
    if selected_overlay is not None:
        declarations = f'#include "native_controller_{selected_overlay}.hpp"\n' + declarations
        overlay_type = f'Controller{selected_overlay.title()}Overlay'
        overlay_argument = 'stdout' if selected_overlay == 'console' else 'child_directory'
        overlay_setup = f'    {overlay_type} {selected_overlay}({overlay_argument});\n'
        overlay_member = f'        {overlay_type}* {selected_overlay};\n'
        overlay_pointer = f'&{selected_overlay}, '
        dispatch = f'        return context.{selected_overlay}->dispatch(call, values, count, roots, root_count,\n' + '''
            [&](auto ordinal, auto* args, auto nargs, auto* arrays, auto narrays) {
                return context.effects->call(ordinal, args, nargs, arrays, narrays);
            });'''.lstrip('\n')
    loader = f'    ControllerEffects effects(effect_path, identity, {len(plan["effect_sites"])});\n'
    effect_member = '        const ControllerEffects* effects;\n'
    effect_pointer = '&effects, '
    if with_plot:
        declarations = '#include "native_controller_plot.hpp"\n' + declarations
        overlay_setup += '    ControllerPlotOverlay plot(startup, child_directory);\n'
        overlay_member += '        ControllerPlotOverlay* plot;\n'
        overlay_pointer += '&plot, '
        loader = effect_member = effect_pointer = ''
        dispatch = '''        return context.plot->dispatch(call, values, count, roots, root_count,
            [&](auto ordinal, auto* args, auto nargs, auto* arrays, auto narrays) {
                return context.child->dispatch(ordinal, args, nargs, arrays, narrays,
                    [](auto, auto*, auto, auto*, auto) { return -1; });
            });'''
    body = _render('native_controller_entry_v1.cpp.in', {
        'IDENTITY': plan['plan_sha256'],
        'COPY_IDENTITY': controller.update.copies['plan_sha256'],
        'EFFECT_LOADER': loader,
        'EFFECT_MEMBER': effect_member,
        'EFFECT_POINTER': effect_pointer,
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
''' if with_console and not with_plot else ''
    rendered = dict(UPDATE_DECLARATIONS=declarations, ENTRY_ARGUMENTS=entry, RUN_BODY=body,
                    UPDATE_OUTPUT_COUNT=' + 1', UPDATE_OUTPUT=output, FINALIZE_EFFECTS=finalize)
    if startup is not None:
        rendered['UPDATE_DECLARATIONS'] = '#include "native_controller_startup.hpp"\n' + declarations
        rendered['BIND_COUNT'] = (f'1 + 3 * (arrays.size() + scalars.size() + output_count'
                                  f' - {len(startup.array_ordinals) + len(startup.numeric_ordinals)})')
        if startup.numeric_ordinals:
            reject = ' && '.join(f'slot != {index}' for index in startup.numeric_ordinals)
            rendered['CHECK_SCALAR_INGRESS'] = f'            require({reject}, "startup owns this scalar");\n'
        if startup.array_ordinals:
            reject = ' && '.join(f'slot != {index}' for index in startup.array_ordinals)
            rendered['CHECK_ARRAY_INGRESS'] = f'            require({reject}, "startup owns this array");\n'
        rendered['MAY_INITIALIZE_ARRAY'] = (
            '        if (startup.initialize_array(i, payloads[i], descriptors[i])) continue;\n')
        if not with_plot:
            rendered['AFTER_ARRAYS_READY'] = '    startup.release_image();\n'
    if with_plot:
        rendered['ALTERNATE_BIND_COUNT'] = f' || static_cast<std::size_t>(argc) == {rendered["BIND_COUNT"]} - 3 * output_count'
        rendered['CHECK_BINDINGS'] = f'''    require(std::count_if(result.inputs.begin(), result.inputs.end(),
        [](const auto& path) {{ return !path.empty(); }}) == arrays.size() - {len(startup.array_ordinals)},
        "original array input is missing");
    require(std::count(seen.begin(), seen.end(), true) == scalars.size() - {len(startup.numeric_ordinals)},
        "original numerical input is missing");
'''
        rendered['OPTIONAL_ARRAY_OUTPUT'] = ' || arguments.outputs[i].empty()'
        rendered['UPDATE_OUTPUT'] = '    if (!controller_out.empty()) {\n' + ''.join(
            '    ' + line for line in output.splitlines(keepends=True)) + '    }\n'
        rendered['FINALIZE_EFFECTS'] = '    require(std::fflush(stdout) == 0, "cannot publish original stdout");\n'
    if controller.public_argv is not None:
        header = Path(controller.public_argv.header).name
        rendered['UPDATE_DECLARATIONS'] = f'#include "{header}"\n' + rendered['UPDATE_DECLARATIONS']
        rendered['MAIN_DISPATCH'] = '''        auto lowered = lower_controller_argv(argc, argv);
        if (lowered.empty()) return 0;
        std::vector<char*> internal;
        internal.reserve(lowered.size() + 1);
        for (auto& argument : lowered) internal.push_back(argument.data());
        internal.push_back(nullptr);
        run(static_cast<int>(lowered.size()), internal.data());'''
        rendered['MAIN_ARGV_ERROR'] = ''' catch (const ScarPublicArgvError& error) {
        std::cerr << error.what() << '\\n';
        return 2;
    }'''
    return rendered
