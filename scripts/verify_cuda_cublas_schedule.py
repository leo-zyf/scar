"""Build a shared host-copy/GEMM schedule with the installed CUDA and cuBLAS toolkit."""

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess

from scar.codegen.cuda_copy_schedule import render_cuda_cublas_schedule
from scar.codegen.native_project import build_native_project, write_native_project
from scar.ir.implementation_plan import (
    CostEstimate, ImplementationOption, PlanningDataEdge, PlanningOperation,
    PlanningProblem, PlanningResource, TransferOption, problem_document,
)
from scar.ir.program.providers import ProviderArtifact
from scar.optimization.constructive_planner import construct_implementations
from scar.optimization.material_routes import expand_material_route


def fixture():
    refs = ({'kind': 'DECLARED_OFFLINE_CUBLAS_SGEMM_FIXTURE'},)

    def cost(value):
        return CostEstimate(value, value, value, 'DECLARED_MODEL', 'OFFLINE_CUBLAS_SGEMM_FIXTURE', refs)

    target = ProviderArtifact('declared:cuda-cublas-fixture', '0' * 64, 0, 'scar.target-hardware-graph')
    resources = tuple(PlanningResource(name, kind, capacity, refs) for name, kind, capacity in (
        ('host', 'MEMORY', 128), ('device', 'MEMORY', 32768), ('copy-in', 'CHANNEL', 1),
        ('compute-stream', 'CHANNEL', 1), ('copy-out', 'CHANNEL', 1), ('gpu', 'EXECUTION', 1)))
    gemm = ImplementationOption('sgemm', 'CUBLAS_SGEMM_F32_COLUMN_MAJOR_V1', refs,
        {'compute-stream': 1, 'gpu': 1}, {'device': 16384}, {'a': 'device', 'b': 'device'},
        {'c': 'device'}, cost(7), dict(left_edge='gemm-a', right_edge='gemm-b',
            rows=2, columns=2, reduction=3, completion='SAME_STREAM'))
    output = ImplementationOption('return', 'CUDA_COPY_RETURN', refs, {}, {},
        {'c': 'host'}, {}, cost(0), {})
    operations = (PlanningOperation('gemm', refs, (), (gemm,), gemm.id),
                  PlanningOperation('return', refs, (), (output,), output.id))
    data = (PlanningDataEdge('gemm-a', 'a', None, 'gemm', 24, 'host', source_refs=refs),
            PlanningDataEdge('gemm-b', 'b', None, 'gemm', 24, 'host', source_refs=refs),
            PlanningDataEdge('publish', 'c', 'gemm', 'return', 16, None, source_refs=refs))
    problem = PlanningProblem(target, refs, resources, operations, data)
    upload = TransferOption('h2d', 'host', 'device', {'copy-in': 1}, cost(3), 'CUDA_MEMCPY_ASYNC', refs)
    download = TransferOption('d2h', 'device', 'host', {'copy-out': 1}, cost(5), 'CUDA_MEMCPY_ASYNC', refs)
    for edge_id, option in (('gemm-a', upload), ('gemm-b', upload), ('publish', download)):
        problem = expand_material_route(problem, edge_id, (option,))
    return problem


def main_source(slot_indices):
    slots = ', '.join(map(str, slot_indices))
    return '#define FIXTURE_SLOT_INDICES ' + slots + '\n' + r'''
#include "generated.hpp"

#include <cstring>
#include <iostream>
#include <memory>
#include <string_view>

int main(int argc, char** argv) {
    using namespace scar_cuda_generated;
    if (argc == 2 && std::string_view(argv[1]) == "--describe") {
        std::size_t copies = 0, provider_calls = 0, waits = 0;
        for (const auto& step : steps) {
            copies += std::holds_alternative<Copy>(step.operation);
            provider_calls += std::holds_alternative<MatrixMultiply>(step.operation);
            waits += step.wait_for.size();
        }
        std::cout << buffers.size() << ' ' << copies << ' ' << provider_calls << ' ' << waits << '\n';
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
        scar::cuda_copy_v1::detail::check(status, "cudaGetDeviceCount");
        std::array<std::unique_ptr<Buffer>, buffers.size()> storage;
        std::array<Buffer*, buffers.size()> bindings;
        for (std::size_t index = 0; index < buffers.size(); ++index) {
            storage[index] = std::make_unique<Buffer>(device, buffers[index].memory, buffers[index].size);
            bindings[index] = storage[index].get();
        }
        constexpr std::array<std::size_t, 3> slots{FIXTURE_SLOT_INDICES};
        constexpr std::array<float, 6> a{1, 4, 2, 5, 3, 6};
        constexpr std::array<float, 6> b{7, 9, 11, 8, 10, 12};
        constexpr std::array<float, 4> c{58, 139, 64, 154};
        std::memcpy(bindings[slots[0]]->data(), a.data(), sizeof(a));
        std::memcpy(bindings[slots[1]]->data(), b.data(), sizeof(b));
        run(bindings);
        if (std::memcmp(bindings[slots[2]]->data(), c.data(), sizeof(c))) {
            throw std::runtime_error("cuBLAS SGEMM output differs from the exact fixture");
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
    libraries = tuple(toolkit / 'lib64' / name for name in ('libcudart.so.13', 'libcublas.so.13'))
    if not all(path.is_file() for path in libraries):
        parser.error('a CUDA 13 toolkit with libcudart and libcublas is required')
    (root / 'src').mkdir(parents=True)

    problem = fixture()
    plan = construct_implementations(problem)
    binding = dict(device=0, memory_kinds={'host': 'PinnedHost', 'device': 'Device'},
                   stream_channels=('copy-in', 'compute-stream', 'copy-out'), compute_resource='gpu')
    source = render_cuda_cublas_schedule(problem, plan, **binding)
    gemm, *others = problem.operations
    mode, = gemm.options
    wrong = replace(mode, parameters=dict(mode.parameters, reduction=4))
    wrong_problem = replace(problem, operations=(replace(gemm, options=(wrong,)), *others))
    try:
        render_cuda_cublas_schedule(wrong_problem, construct_implementations(wrong_problem), **binding)
    except ValueError as error:
        if 'matrix byte extents' not in str(error):
            raise
    else:
        raise ValueError('CUDA lowering accepted an inconsistent typed GEMM operand')

    slots = {name: index for index, name in enumerate(source.buffer_values)}
    (root / 'src/main.cpp').write_text(main_source([slots[name] for name in ('a', 'b', '$copy:publish:0:material')]))
    header = root / 'src/generated.hpp'
    header.write_text(source.header)
    codegen = Path(__file__).resolve().parents[1] / 'scar/codegen'
    for name in ('cuda_copy_schedule_v1.h', 'cuda_cublas_schedule_v1.h'):
        shutil.copyfile(codegen / name, root / 'src' / name)
    write_native_project(root, 'cuda-cublas', include_dirs=(str(toolkit / 'include'),),
                         compile_options=('-Wall', '-Wextra', '-Werror'),
                         libraries=tuple(map(str, libraries)), runtime_dirs=(str(toolkit / 'lib64'),))
    executable, dependencies = build_native_project(root, 'cuda-cublas', ['c++'], expected_libraries=libraries)
    description = subprocess.check_output([str(executable), '--describe'], text=True, env={}).strip()
    if description != '6 3 1 2':
        raise ValueError('compiled CUDA/cuBLAS program has incorrect buffers, copies, provider calls or waits')
    run = subprocess.run([str(executable)], text=True, env={}, capture_output=True)
    if run.returncode not in (0, 77):
        raise ValueError('native cuBLAS result check failed: ' + run.stderr.strip())

    def archive(name, record):
        path = root / name
        path.write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
        return sha256(path.read_bytes()).hexdigest()

    receipt = dict(declared_fixture_only=True, cpp20_compile_link_passed=True,
        streams=3, copies=3, cublas_sgemm_calls=1, cross_stream_waits=2, device_workspace_bytes=16384,
        mismatched_gemm_dimensions_rejected=True, device_result_verified=run.returncode == 0,
        device_probe=run.stderr.strip(), problem_sha256=archive('problem.json', problem_document(problem)),
        schedule_sha256=archive('schedule.json', plan), native_dependencies=dependencies,
        bindings_sha256=archive('bindings.json', dict(binding, buffer_values=source.buffer_values)),
        generated_header_sha256=sha256(header.read_bytes()).hexdigest())
    archive('validation/offline.json', receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
