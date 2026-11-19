#pragma once

#include <ATen/ops/from_blob.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <system_error>
#include <sys/mman.h>

namespace scar_aten_graph {

struct ReadyReplica {
    std::size_t input_index;
    std::size_t lane;
};

inline at::Tensor materialize_ready_replica(const at::Tensor& input) {
    if (!input.defined() || !input.device().is_cpu() || input.layout() != c10::kStrided
            || input.scalar_type() != at::kFloat || !input.is_contiguous()
            || input.requires_grad() || input.is_conj() || input.is_neg()) {
        throw std::invalid_argument("ready replica requires a contiguous CPU float32 tensor");
    }
    const auto elements = input.numel();
    if (elements <= 0 || static_cast<std::uint64_t>(elements) > std::numeric_limits<std::size_t>::max() / 4) {
        throw std::invalid_argument("invalid ready replica input size");
    }
    const auto bytes = 4 * static_cast<std::size_t>(elements);
    void* address = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (address == MAP_FAILED) {
        throw std::system_error(errno, std::generic_category(), "mmap ready replica");
    }
    const std::shared_ptr<void> backing(address, [bytes](void* data) noexcept { ::munmap(data, bytes); });
    auto result = at::from_blob(backing.get(), input.sizes(), input.strides(), [backing](void*) {}, input.options());
    // The synchronous provider copy first-touches the new pages on the calling lane.
    result.copy_(input);
    return result;
}

}  // namespace scar_aten_graph
