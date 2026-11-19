"""Replay the two-stream CUDA copy lowering against an installed toolkit."""

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess

from scar.codegen.cuda_copy_schedule import render_cuda_copy_schedule
from scar.codegen.native_project import build_native_project, write_native_project
from scar.ir.implementation_plan import (
    CostEstimate, ImplementationOption, PlanningDataEdge, PlanningOperation,
    PlanningProblem, PlanningResource, TransferOption, problem_document,
)
from scar.ir.program.providers import ProviderArtifact
from scar.optimization.constructive_planner import construct_implementations
from scar.optimization.material_routes import expand_material_route


def fixture():
    refs = ({'kind': 'DECLARED_OFFLINE_CUDA_COPY_FIXTURE'},)

    def cost(value):
        return CostEstimate(value, value, value, 'DECLARED_MODEL', 'OFFLINE_COPY_SHAPE', refs)

    target = ProviderArtifact('declared:cuda-copy-fixture', '0' * 64, 0, 'scar.target-hardware-graph')
    resources = tuple(PlanningResource(name, kind, capacity, refs) for name, kind, capacity in (
        ('host', 'MEMORY', 256), ('device', 'MEMORY', 256), ('copy-in', 'CHANNEL', 1),
        ('copy-out', 'CHANNEL', 1), ('bus', 'BANDWIDTH_CLASS', 1)))
    operations, edges = [], []
    for index in range(2):
        name, value, output = f'write-{index}', f'input-{index}', f'device-{index}'
        copy = ImplementationOption('h2d', 'CUDA_MEMCPY_ASYNC', refs, {'copy-in': 1, 'bus': 1}, {},
            {value: 'host'}, {output: 'device'}, cost(3), dict(logical_value=value,
                source_memory='host', destination_memory='device', completion='DESTINATION_READY'))
        operations.append(PlanningOperation(name, refs, (), (copy,), copy.id))
        edges.extend((PlanningDataEdge(name + ':input', value, None, name, 64, 'host', source_refs=refs),
                      PlanningDataEdge(f'publish-{index}', output, name, 'return', 64, None, source_refs=refs)))
    host_return = ImplementationOption('return', 'CUDA_COPY_RETURN', refs, {}, {},
        {'device-0': 'host', 'device-1': 'host'}, {}, cost(0), {})
    operations.append(PlanningOperation('return', refs, (), (host_return,), host_return.id))
    result = PlanningProblem(target, refs, resources, tuple(operations), tuple(edges))
    download = TransferOption('d2h', 'device', 'host', {'copy-out': 1, 'bus': 1}, cost(5), 'CUDA_MEMCPY_ASYNC', refs)
    for index in range(2):
        result = expand_material_route(result, f'publish-{index}', (download,))
    return result


def main_source(inputs, outputs):
    indices = ', '.join(map(str, (*inputs, *outputs)))
    return '#define FIXTURE_SLOT_INDICES ' + indices + '\n' + r'''
#include "generated.hpp"

#include <algorithm>
#include <iostream>
#include <memory>
#include <string_view>

int main(int argc, char** argv) {
    using namespace scar_cuda_generated;
    if (argc == 2 && std::string_view(argv[1]) == "--describe") {
        std::size_t waits = 0;
        for (const auto& step : steps) waits += step.wait_for.size();
        std::cout << buffers.size() << ' ' << steps.size() << ' ' << waits << '\n';
        return 0;
    }
    try {
        int count = 0;
        const auto status = cudaGetDeviceCount(&count);
        if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
            (status == cudaSuccess && count <= device)) {
            std::cerr << "CUDA device unavailable: " << cudaGetErrorString(status) << '\n';
            return 77;
        }
        detail::check(status, "cudaGetDeviceCount");
        std::array<std::unique_ptr<Buffer>, buffers.size()> storage;
        std::array<Buffer*, buffers.size()> bindings;
        for (std::size_t index = 0; index < buffers.size(); ++index) {
            storage[index] = std::make_unique<Buffer>(device, buffers[index].memory, buffers[index].size);
            bindings[index] = storage[index].get();
        }
        constexpr std::array<std::size_t, 4> slots{FIXTURE_SLOT_INDICES};
        const auto bytes = [&bindings](std::size_t index) {
            return static_cast<unsigned char*>(bindings[index]->data());
        };
        std::fill_n(bytes(slots[0]), 64, 19);
        std::fill_n(bytes(slots[1]), 64, 73);
        run(bindings);
        if (!std::equal(bytes(slots[0]), bytes(slots[0]) + 64, bytes(slots[2])) ||
            !std::equal(bytes(slots[1]), bytes(slots[1]) + 64, bytes(slots[3]))) {
            throw std::runtime_error("CUDA copy altered the output bytes");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cuda-root', type=Path, default=Path('/usr/local/cuda'))
    args = parser.parse_args()
    toolkit, root = args.cuda_root.resolve(), args.out.absolute()
    library = toolkit / 'lib64/libcudart.so.13'
    if not library.is_file():
        parser.error('a CUDA 13 toolkit with libcudart.so.13 under lib64 is required')
    (root / 'src').mkdir(parents=True)

    problem = fixture()
    schedule = construct_implementations(problem)
    binding = dict(device=0, memory_kinds={'host': 'PinnedHost', 'device': 'Device'},
                   stream_channels=('copy-in', 'copy-out'))
    source = render_cuda_copy_schedule(problem, schedule, **binding)
    tight = replace(problem, resources=tuple(replace(row, capacity=192) if row.id == 'host' else row
                                            for row in problem.resources))
    try:
        render_cuda_copy_schedule(tight, construct_implementations(tight), **binding)
    except ValueError as error:
        if 'remain allocated' not in str(error):
            raise
    else:
        raise ValueError('CUDA lowering exceeded the long-lived pinned backing capacity')
    slots = {name: index for index, name in enumerate(source.buffer_values)}
    (root / 'src/main.cpp').write_text(main_source([slots[f'input-{i}'] for i in range(2)],
                                                  [slots[f'$copy:publish-{i}:0:material'] for i in range(2)]))
    header = root / 'src/generated.hpp'
    header.write_text(source.header)
    shutil.copyfile(Path(__file__).resolve().parents[1] / 'scar/codegen/cuda_copy_schedule_v1.h',
                    root / 'src/cuda_copy_schedule_v1.h')
    write_native_project(root, 'cuda-copy', include_dirs=(str(toolkit / 'include'),),
                         compile_options=('-Wall', '-Wextra', '-Werror'),
                         libraries=(str(library),), runtime_dirs=(str(library.parent),))
    executable, dependencies = build_native_project(root, 'cuda-copy', ['c++'], expected_libraries=(library,))
    description = subprocess.check_output([str(executable), '--describe'], text=True, env={}).strip()
    if description != '6 4 1':
        raise ValueError('compiled CUDA program has incorrect buffer, copy or event counts')
    run = subprocess.run([str(executable)], text=True, env={}, capture_output=True)
    if run.returncode not in (0, 77):
        raise ValueError('CUDA native roundtrip failed: ' + run.stderr.strip())

    def write(name, value):
        path = root / name
        path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
        return sha256(path.read_bytes()).hexdigest()

    receipt = dict(declared_fixture_only=True, cpp20_compile_link_passed=True, streams=2, copies=4,
        cross_stream_events=1, insufficient_pinned_backing_rejected=True,
        device_bytes_verified=run.returncode == 0, device_probe=run.stderr.strip(),
        problem_sha256=write('problem.json', problem_document(problem)), schedule_sha256=write('schedule.json', schedule),
        bindings_sha256=write('bindings.json', dict(binding, buffer_values=source.buffer_values)),
        generated_header_sha256=sha256(header.read_bytes()).hexdigest(), native_dependencies=dependencies)
    write('validation/offline.json', receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
