#pragma once

#include "cuda_copy_schedule_v1.h"

#include <cublas_v2.h>

#include <limits>
#include <memory>
#include <variant>

namespace scar::cuda_cublas_v1 {

using cuda_copy_v1::Buffer;
using cuda_copy_v1::BufferSpec;
using cuda_copy_v1::Memory;

struct Copy {
    std::size_t source;
    std::size_t destination;
    cudaMemcpyKind direction;
};

struct MatrixMultiply {
    std::size_t left;
    std::size_t right;
    std::size_t output;
    int rows;
    int columns;
    int reduction;
};

struct Step {
    std::size_t stream;
    std::variant<Copy, MatrixMultiply> operation;
    std::span<const std::size_t> wait_for;
    bool record_completion;
};

namespace detail {

inline void check(cublasStatus_t status, const char* api) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(api) + ": " + cublasGetStatusString(status));
    }
}

class Context {
public:
    Context() = default;
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    ~Context() {
        if (handle_) cublasDestroy(handle_);
    }

    void bind(int device, cudaStream_t stream, std::size_t workspace_bytes) {
        workspace_ = std::make_unique<Buffer>(device, Memory::Device, workspace_bytes);
        check(cublasCreate(&handle_), "cublasCreate");
        check(cublasSetStream(handle_, stream), "cublasSetStream");
        check(cublasSetWorkspace(handle_, workspace_->data(), workspace_->size()), "cublasSetWorkspace");
        check(cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode");
        check(cublasSetMathMode(handle_, CUBLAS_PEDANTIC_MATH), "cublasSetMathMode");
    }

    void multiply(const MatrixMultiply& operation, std::span<Buffer* const> buffers) const {
        static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
        const float alpha = 1.0F;
        const float beta = 0.0F;
        const auto* left = static_cast<const float*>(buffers[operation.left]->data());
        const auto* right = static_cast<const float*>(buffers[operation.right]->data());
        auto* output = static_cast<float*>(buffers[operation.output]->data());
        check(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, operation.rows, operation.columns,
                          operation.reduction, &alpha, left, operation.rows, right, operation.reduction,
                          &beta, output, operation.rows), "cublasSgemm");
    }

private:
    cublasHandle_t handle_ = nullptr;
    std::unique_ptr<Buffer> workspace_;
};

class Dispatcher {
public:
    Dispatcher(int device, std::span<const std::size_t> workspace_bytes)
        : device_(device), workspace_bytes_(workspace_bytes), contexts_(workspace_bytes.size()) {}

    void prepare(std::span<const cudaStream_t> streams) {
        for (std::size_t i = 0; i < streams.size(); ++i) {
            if (workspace_bytes_[i]) contexts_[i].bind(device_, streams[i], workspace_bytes_[i]);
        }
    }

    void enqueue(const Step& step, cudaStream_t stream, std::span<const BufferSpec> specs,
                 std::span<Buffer* const> buffers) const {
        if (const auto* copy = std::get_if<Copy>(&step.operation)) {
            cuda_copy_v1::detail::enqueue_copy(stream, copy->source, copy->destination,
                                               copy->direction, specs, buffers);
        } else {
            contexts_[step.stream].multiply(std::get<MatrixMultiply>(step.operation), buffers);
        }
    }

private:
    int device_;
    std::span<const std::size_t> workspace_bytes_;
    std::vector<Context> contexts_;
};

inline void execute(int device, std::span<const std::size_t> workspace_bytes,
                    std::span<const BufferSpec> specs, std::span<const Step> steps,
                    std::span<Buffer* const> buffers) {
    cuda_copy_v1::detail::execute_queued(device, workspace_bytes.size(), specs, steps, buffers,
        [device, workspace_bytes] { return Dispatcher{device, workspace_bytes}; });
}

}  // namespace detail

}  // namespace scar::cuda_cublas_v1
