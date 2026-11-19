#pragma once

#include "cuda_cublas_schedule_v1.h"/* SCAR_ACTIVITY_INCLUDE */

#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <ATen/ops/add_ops.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/relu_ops.h>

#include <cstdint>
#include <string_view>

namespace scar::cuda_hybrid_v1 {

using cuda_cublas_v1::Buffer;
using cuda_cublas_v1::BufferSpec;
using cuda_cublas_v1::Copy;
using cuda_cublas_v1::MatrixMultiply;
using cuda_cublas_v1::Memory;

struct HostTensor {
    std::size_t slot;
    std::int64_t rows;
    std::int64_t columns;
};

struct HostRelu {
    HostTensor input;
    HostTensor output;
};

struct HostAdd {
    HostTensor left;
    HostTensor right;
    HostTensor output;
    std::int64_t alpha;
};

struct Step {
    std::size_t stream;
    std::variant<Copy, MatrixMultiply, HostRelu, HostAdd> operation;
    std::span<const std::size_t> wait_for;
    bool record_completion;

    bool host() const noexcept {
        return std::holds_alternative<HostRelu>(operation) || std::holds_alternative<HostAdd>(operation);
    }
};

namespace detail {

static_assert(std::string_view(at::_ops::relu_out::schema_str) ==
              "relu.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)");
static_assert(std::string_view(at::_ops::add_out::schema_str) ==
              "add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)");

inline at::Tensor host_view(HostTensor port, std::span<Buffer* const> buffers) {
    return at::from_blob(buffers[port.slot]->data(), {port.rows, port.columns},
                         at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
}

inline void host_call(const Step& step, std::span<Buffer* const> buffers) {
    HostTensor output_slot;
    at::Tensor output;
    if (const auto* relu = std::get_if<HostRelu>(&step.operation)) {
        output_slot = relu->output;
        output = host_view(output_slot, buffers);
        at::_ops::relu_out::call(host_view(relu->input, buffers), output);
    } else {
        const auto& add = std::get<HostAdd>(step.operation);
        output_slot = add.output;
        output = host_view(output_slot, buffers);
        at::_ops::add_out::call(host_view(add.left, buffers), host_view(add.right, buffers), at::Scalar(add.alpha), output);
    }
    if (output.data_ptr() != buffers[output_slot.slot]->data() || output.dim() != 2
            || output.size(0) != output_slot.rows || output.size(1) != output_slot.columns || !output.is_contiguous()) {
        throw std::runtime_error("ATen output differs from its pinned material contract");
    }
}

/* SCAR_ACTIVITY_TEMPLATE */inline void execute(int device, std::span<const std::size_t> workspace_bytes, std::span<const BufferSpec> specs,
                    std::span<const Step> steps, std::span<Buffer* const> buffers/* SCAR_ACTIVITY_ARGUMENT */) {/* SCAR_ACTIVITY_SETUP_START */
    using cuda_copy_v1::detail::check;
    if (specs.size() != buffers.size()) throw std::invalid_argument("hybrid buffer count differs");
    std::unordered_set<Buffer*> identities;
    identities.reserve(buffers.size());
    for (std::size_t index = 0; index < buffers.size(); ++index) {
        const auto* buffer = buffers[index];
        if (!buffer || buffer->device() != device || buffer->memory() != specs[index].memory
                || buffer->size() != specs[index].size || !identities.insert(buffers[index]).second) {
            throw std::invalid_argument("hybrid buffer identity, device, memory kind or byte extent differs");
        }
    }
    at::set_num_threads(1);
    const at::NoGradGuard no_grad;
    const cuda_copy_v1::detail::DeviceScope scope(device);/* SCAR_ACTIVITY_INIT */
    cuda_copy_v1::detail::Queues queues(workspace_bytes.size(), steps.size());
    cuda_cublas_v1::detail::Dispatcher gpu(device, workspace_bytes);
    const cuda_copy_v1::detail::DrainQueuedWork drain{queues};
    for (auto& stream : queues.streams) {
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    }
    gpu.prepare(queues.streams);
    for (std::size_t index = 0; index < steps.size(); ++index) {
        if (steps[index].record_completion) {
            check(cudaEventCreateWithFlags(&queues.events[index], cudaEventDisableTiming), "cudaEventCreateWithFlags");
        }
    }/* SCAR_ACTIVITY_SETUP_END */
    for (std::size_t index = 0; index < steps.size(); ++index) {
        const auto& step = steps[index];
        if (step.host()) {/* SCAR_ACTIVITY_HOST_WAIT_START */
            for (const auto predecessor : step.wait_for) {
                check(cudaEventSynchronize(queues.events[predecessor]), "cudaEventSynchronize");
            }/* SCAR_ACTIVITY_HOST_START */
            host_call(step, buffers);/* SCAR_ACTIVITY_HOST_END */
            continue;
        }
        const auto stream = queues.streams[step.stream];
        for (const auto predecessor : step.wait_for) {
            check(cudaStreamWaitEvent(stream, queues.events[predecessor], 0), "cudaStreamWaitEvent");
        }/* SCAR_ACTIVITY_DEVICE_START */
        if (const auto* copy = std::get_if<Copy>(&step.operation)) {
            cuda_copy_v1::detail::enqueue_copy(stream, copy->source, copy->destination, copy->direction, specs, buffers);
        } else {
            gpu.enqueue(cuda_cublas_v1::Step{step.stream, std::get<MatrixMultiply>(step.operation),
                                             step.wait_for, step.record_completion},
                        stream, specs, buffers);
        }/* SCAR_ACTIVITY_DEVICE_END */
        if (step.record_completion) {
            check(cudaEventRecord(queues.events[index], stream), "cudaEventRecord");
        }
    }/* SCAR_ACTIVITY_JOIN_START */
    queues.join();/* SCAR_ACTIVITY_JOIN_END */
}

}  // namespace detail

}  // namespace scar::cuda_hybrid_v1
