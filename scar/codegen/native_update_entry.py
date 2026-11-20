"""Connect an original ordered update to the standalone array entry."""
from dataclasses import dataclass
import json

from scar.codegen.native_array_copy import prepare as prepare_copies
from scar.compiler.plan_native_update import read_native_update_plan
from scar.ir.execution_recipe import read_artifact
from scar.optimization.resource_solver import semantic_sha256


COPY_SOURCE = 'src/array_copy.cpp'
_OPERATORS = {'ADD': '+', 'SUBTRACT': '-', 'MULTIPLY': '*', 'TRUE_DIVIDE': '/'}


@dataclass(frozen=True, slots=True)
class NativeUpdate:
    planning: dict
    copies: dict
    copy_source: bytes
    scalar_argument: str
    scalar_expression: dict
    scalar_reads: tuple[int, ...]
    scalar_code: str
    entry_guards: tuple[tuple[str, bool], ...]

    def artifacts(self):
        return dict(planning=self.planning, copies=self.copies, scalar_argument=self.scalar_argument,
                    scalar_expression=self.scalar_expression,
                    entry_guards=[dict(path=path, expected=value) for path, value in self.entry_guards],
                    supported_execution='REPEATED_ORIGINAL_UPDATE',
                    trip_count='EXPLICIT_RUNTIME_COUNT', outer_original_driver_lowered=False)


def prepare(reference, metadata, ports):
    report, _ = read_native_update_plan(reference['uri'])
    if report != read_artifact(reference):
        raise ValueError('native update planning receipt changed during replay')
    source = read_artifact(report['source'])
    recipe = read_artifact(report['numeric_recipe'])
    from scar.compiler.stateful_model import stateful_digest, stateful_from_dict

    region = stateful_from_dict(read_artifact(recipe['artifacts']['region']))
    stage_plan = read_artifact(recipe['artifacts']['stage_plan'])
    if (metadata['region_sha256'] != stateful_digest(region)
            or metadata['plan_sha256'] != semantic_sha256(stage_plan)
            or metadata['implementation_sha256'] != recipe['implementation']['backend_capability_record']['implementation_sha256']):
        raise ValueError('native update must keep the original numerical region, plan, and backend')
    arrays = [row for row in ports if row['kind'] == 'array']
    scalars = {row['id']: i for i, row in enumerate(row for row in ports if row['kind'] == 'scalar')}
    copies, cpp = prepare_copies(source, array_ids=[row['id'] for row in arrays])
    if any(copy['shape'] != array['shape'] for copy, array in zip(copies['arrays'], arrays, strict=True)):
        raise ValueError('native update copy and numerical ports differ in shape')
    written = {row['destination']['array_index'] for row in copies['ordered_moves']}
    for index in written:
        arrays[index]['output'] = True
    tail = source['scalar_tail']
    expression = tail['return_expression']
    if expression is None or tail['frontiers']:
        raise ValueError('native update needs an original closed scalar return expression')
    arguments, reads = set(), set()

    def lower(node):
        match node:
            case {'operator': operator, 'operands': [left, right]} if operator in _OPERATORS:
                return f'({lower(left)} {_OPERATORS[operator]} {lower(right)})'
            case {'kind': 'ARGUMENT', 'parameter': parameter}:
                arguments.add(parameter)
                return 'carried_scalar'
            case {'kind': 'REGION_INPUT', 'id': identity} if identity in scalars:
                slot = scalars[identity]
                reads.add(slot)
                return f'arguments.values[{slot}].value'
            case _:
                raise ValueError('unsupported original scalar return operand or operation')

    scalar_code = lower(expression)
    if len(arguments) != 1:
        raise ValueError('native update needs one original carried scalar argument')
    prefix = source['entry_guards']
    if prefix['frontiers']:
        raise ValueError('native update original dispatch prefix is open')
    guards = []
    for row in prefix['guards']:
        if (row['kind'] != 'EXACT_BUILTIN_BOOL' or type(row['expected']) is not bool
                or not row['path'] or any(not part.isidentifier() for part in row['path'])):
            raise ValueError('native update entry needs known boolean path guards')
        guards.append(('.'.join(row['path']), row['expected']))
    return NativeUpdate(reference, copies, cpp, next(iter(arguments)), expression,
                        tuple(sorted(reads)), scalar_code, tuple(guards))


def render(update, provider_setup, provider_args):
    flags = '''    require(argc >= 7 && std::string_view(argv[1]) == "--steps" &&
            std::string_view(argv[3]) == "--initial-scalar" && std::string_view(argv[5]) == "--scalar-out",
            "expected --steps N --initial-scalar VALUE --scalar-out FILE before array ports");
    const auto steps = number<std::uint64_t>(argv[2]);
    double carried_scalar = number<double>(argv[4]);
    require(std::isfinite(carried_scalar), "nonfinite original carried scalar argument");
    const auto requested = fs::absolute(fs::path(argv[6]));
    require(!requested.filename().empty() && requested.filename() != "." && requested.filename() != "..",
            "scalar output must be a file");
    const auto scalar_out = fs::canonical(requested.parent_path()) / requested.filename();'''
    next_argument = 7
    for path, value in update.entry_guards:
        name, literal = json.dumps(path, ensure_ascii=False), 'true' if value else 'false'
        flags += f'''
    require(argc > {next_argument+2} && std::string_view(argv[{next_argument}]) == "--guard" &&
            std::string_view(argv[{next_argument+1}]) == {name} &&
            std::string_view(argv[{next_argument+2}]) == "{literal}", "original update branch guard differs");'''
        next_argument += 3
    shift = next_argument - 1
    flags += f'''
    const auto arguments = bind(argc - {shift}, argv + {shift});
    require(!fs::exists(fs::symlink_status(scalar_out)) &&
            std::find(arguments.outputs.begin(), arguments.outputs.end(), scalar_out) == arguments.outputs.end(),
            "scalar output must have a fresh distinct path");'''
    copy_identity = update.copies['plan_sha256']
    require_float = ''.join(f'    require(arguments.values[{index}].integer == 0, "expected binary64 scalar field");\n'
                            for index in update.scalar_reads)
    body = f'''    const auto* copy_identity = scar_array_copy_plan_sha256_v1();
    require(copy_identity && copy_identity == std::string_view("{copy_identity}"), "copy plan identity differs");
    require(scar_array_copy_preflight_v1(descriptors.data(), descriptors.size()) == 0, "array copy preflight failed");
{require_float}{provider_setup}    for (std::uint64_t step = 0; step < steps; ++step) {{
        const auto status = table->run(descriptors.data(), descriptors.size(),
                                       arguments.values.data(), arguments.values.size(), {provider_args});
        if (status != 0) throw std::runtime_error("native stateful region declined: " + std::to_string(status));
        require(scar_array_copy_run_v1(descriptors.data(), descriptors.size()) == 0, "array copy declined");
        carried_scalar = {update.scalar_code};
        require(std::isfinite(carried_scalar), "original scalar result outside finite binary64 profile");
    }}'''
    declarations = '''extern "C" int scar_array_copy_preflight_v1(const scar_array_v1*, std::uint64_t) noexcept;
extern "C" int scar_array_copy_run_v1(const scar_array_v1*, std::uint64_t) noexcept;
extern "C" const char* scar_array_copy_plan_sha256_v1() noexcept;

'''
    output = '''    auto scalar_output = std::make_unique<PendingFile>(scalar_out);
    scalar_output->write(std::vector<double>{carried_scalar});
    pending.push_back(std::move(scalar_output));
'''
    return dict(UPDATE_DECLARATIONS=declarations, ENTRY_ARGUMENTS=flags, RUN_BODY=body,
                UPDATE_OUTPUT_COUNT=' + 1', UPDATE_OUTPUT=output)
