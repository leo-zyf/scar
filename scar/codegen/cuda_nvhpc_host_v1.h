#pragma once

#include "cuda_copy_schedule_v1.h"

#include <cblas.h>

#include <cfenv>
#include <limits>

namespace scar::cuda_nvhpc_host_v1 {

using cuda_copy_v1::Buffer;
using cuda_copy_v1::Memory;

static_assert(sizeof(blasint) == 4 && sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);

struct HostTensor {
    std::size_t slot;
    blasint rows;
    blasint columns;
};

struct HostMm {
    HostTensor left;
    HostTensor right;
    HostTensor output;
};

inline void configure(int threads) {
    if (std::fegetround() != FE_TONEAREST) {
        throw std::runtime_error("NVHPC host GEMM requires normal FP32 rounding");
    }
    openblas_set_num_threads(threads);
    if (openblas_get_num_threads() != threads) {
        throw std::runtime_error("NVHPC host GEMM could not honor the planned thread count");
    }
}

inline float* host_port(HostTensor port, std::span<Buffer* const> buffers) {
    if (port.slot >= buffers.size() || port.rows <= 0 || port.columns <= 0) {
        throw std::invalid_argument("NVHPC host GEMM matrix port is outside the planned ABI");
    }
    const auto* buffer = buffers[port.slot];
    if (buffer->memory() != Memory::PinnedHost || buffer->size() != sizeof(float) * std::size_t(port.rows) * port.columns) {
        throw std::invalid_argument("NVHPC host GEMM requires the exact pinned-host material extent");
    }
    return static_cast<float*>(buffer->data());
}

inline void multiply(const HostMm& mm, std::span<Buffer* const> buffers) {
    if (mm.left.columns != mm.right.rows || mm.output.rows != mm.left.rows || mm.output.columns != mm.right.columns
            || mm.left.slot == mm.output.slot || mm.right.slot == mm.output.slot) {
        throw std::invalid_argument("NVHPC host GEMM matrix geometry or output ownership differs");
    }
    const auto* left = host_port(mm.left, buffers);
    const auto* right = host_port(mm.right, buffers);
    auto* output = host_port(mm.output, buffers);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, mm.output.rows, mm.output.columns, mm.left.columns,
                1.0F, left, mm.left.columns, right, mm.right.columns, 0.0F, output, mm.output.columns);
}

}  // namespace scar::cuda_nvhpc_host_v1
