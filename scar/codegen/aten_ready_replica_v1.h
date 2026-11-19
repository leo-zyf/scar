#pragma once

#include <ATen/ops/from_blob.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <sched.h>
#include <stdexcept>
#include <system_error>
#include <sys/mman.h>

namespace scar_aten_graph {

struct ReadyReplica {
    std::size_t input_index;
    std::size_t lane;
};

namespace detail {

inline std::size_t ready_replica_bytes(const at::Tensor& input) {
    static_assert(sizeof(float) == 4);
    if (!input.defined() || !input.device().is_cpu() || input.layout() != c10::kStrided
            || input.scalar_type() != at::kFloat || !input.is_contiguous()
            || input.requires_grad() || input.is_conj() || input.is_neg()) {
        throw std::invalid_argument("ready replica requires a contiguous CPU float32 tensor");
    }
    const auto elements = input.numel();
    if (elements <= 0 || static_cast<std::uint64_t>(elements) > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::invalid_argument("invalid ready replica input size");
    }
    return sizeof(float) * static_cast<std::size_t>(elements);
}

inline at::Tensor copy_ready_replica(const at::Tensor& input, std::size_t bytes) {
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

}  // namespace detail

inline at::Tensor materialize_ready_replica(const at::Tensor& input) {
    return detail::copy_ready_replica(input, detail::ready_replica_bytes(input));
}

// The caller serializes submissions and guarantees exclusive source access
// for each submission. Only the nominated lane may acquire the replica, and
// operators and diagnostic observers must keep it read-only and unescaped.
class ReadyRetainedReplica {
public:
    ReadyRetainedReplica() = default;
    ReadyRetainedReplica(const ReadyRetainedReplica&) = delete;
    ReadyRetainedReplica& operator=(const ReadyRetainedReplica&) = delete;

    at::Tensor acquire(const at::Tensor& input) {
        const auto bytes = detail::ready_replica_bytes(input);
        const int cpu = ::sched_getcpu();
        if (cpu < 0) throw std::system_error(errno, std::generic_category(), "get ready replica CPU");
        if (slot_.defined() && cpu_ == cpu && slot_.sizes() == input.sizes() && slot_.strides() == input.strides()
                && std::memcmp(slot_.const_data_ptr<float>(), input.const_data_ptr<float>(), bytes) == 0) {
            ++hits_;
            return slot_;
        }

        // Drop the old backing before creating another one. A failed copy
        // leaves the slot empty and a later submission can retry.
        slot_.reset();
        slot_ = detail::copy_ready_replica(input, bytes);
        cpu_ = cpu;
        ++misses_;
        return slot_;
    }

    // Successful acquisitions, readable after a submission has joined.
    std::size_t hits() const noexcept { return hits_; }
    std::size_t misses() const noexcept { return misses_; }

private:
    at::Tensor slot_;
    int cpu_ = -1;
    std::size_t hits_ = 0;
    std::size_t misses_ = 0;
};

}  // namespace scar_aten_graph
