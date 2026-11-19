#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace scar::cuda_copy_v1 {

enum class Memory { PinnedHost, Device };

namespace detail {

inline void check(cudaError_t status, const char* api) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(api) + ": " + cudaGetErrorString(status));
    }
}

class DeviceScope {
public:
    explicit DeviceScope(int device) : requested_(device) {
        check(cudaGetDevice(&previous_), "cudaGetDevice");
        if (requested_ != previous_) check(cudaSetDevice(requested_), "cudaSetDevice");
    }
    DeviceScope(const DeviceScope&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;
    ~DeviceScope() {
        if (requested_ != previous_) cudaSetDevice(previous_);
    }

private:
    int requested_;
    int previous_;
};

}  // namespace detail

class Buffer {
public:
    Buffer(int device, Memory memory, std::size_t size) : device_(device), memory_(memory), size_(size) {
        if (size == 0) throw std::invalid_argument("CUDA copy buffer requires a positive byte extent");
        detail::DeviceScope scope(device);
        if (memory == Memory::Device) {
            detail::check(cudaMalloc(&data_, size), "cudaMalloc");
        } else {
            detail::check(cudaHostAlloc(&data_, size, cudaHostAllocDefault), "cudaHostAlloc");
        }
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() {
        if (memory_ == Memory::PinnedHost) {
            cudaFreeHost(data_);
        } else {
            int previous;
            if (cudaGetDevice(&previous) != cudaSuccess) return;
            if (previous != device_ && cudaSetDevice(device_) != cudaSuccess) return;
            cudaFree(data_);
            if (previous != device_) cudaSetDevice(previous);
        }
    }

    void* data() const noexcept { return data_; }
    int device() const noexcept { return device_; }
    Memory memory() const noexcept { return memory_; }
    std::size_t size() const noexcept { return size_; }

private:
    int device_;
    Memory memory_;
    std::size_t size_;
    void* data_ = nullptr;
};

struct BufferSpec {
    Memory memory;
    std::size_t size;
};

struct Step {
    std::size_t stream;
    std::size_t source;
    std::size_t destination;
    cudaMemcpyKind direction;
    std::span<const std::size_t> wait_for;
    bool record_completion;
};

namespace detail {

class Queues {
public:
    Queues(std::size_t stream_count, std::size_t step_count)
        : streams(stream_count), events(step_count) {}
    Queues(const Queues&) = delete;
    Queues& operator=(const Queues&) = delete;
    ~Queues() {
        // Keep caller storage in use until queued work has been drained, including when submission fails.
        if (!joined_) for (auto stream : streams) if (stream) cudaStreamSynchronize(stream);
        for (auto event : events) if (event) cudaEventDestroy(event);
        for (auto stream : streams) if (stream) cudaStreamDestroy(stream);
    }

    void join() {
        for (auto stream : streams) check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
        joined_ = true;
    }

    std::vector<cudaStream_t> streams;
    std::vector<cudaEvent_t> events;

private:
    bool joined_ = false;
};

// The generated plan owns the ordering. The caller owns non-overlapping buffers
// and supplies their initial values before entering this synchronous boundary.
inline void execute(int device, std::size_t stream_count, std::span<const BufferSpec> specs,
                    std::span<const Step> steps, std::span<Buffer* const> buffers) {
    if (specs.size() != buffers.size()) throw std::invalid_argument("CUDA copy buffer count differs");
    std::unordered_set<Buffer*> identities;
    identities.reserve(buffers.size());
    for (std::size_t index = 0; index < buffers.size(); ++index) {
        const auto* buffer = buffers[index];
        if (!buffer || buffer->device() != device || buffer->memory() != specs[index].memory
            || buffer->size() != specs[index].size || !identities.insert(buffers[index]).second) {
            throw std::invalid_argument("CUDA copy buffer identity, device, memory kind or byte extent differs");
        }
    }
    DeviceScope scope(device);
    Queues queues(stream_count, steps.size());
    for (auto& stream : queues.streams) {
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    }
    for (std::size_t index = 0; index < steps.size(); ++index) {
        if (steps[index].record_completion) {
            check(cudaEventCreateWithFlags(&queues.events[index], cudaEventDisableTiming), "cudaEventCreateWithFlags");
        }
    }
    for (std::size_t index = 0; index < steps.size(); ++index) {
        const auto& step = steps[index];
        const auto stream = queues.streams[step.stream];
        for (const auto predecessor : step.wait_for) {
            check(cudaStreamWaitEvent(stream, queues.events[predecessor], 0), "cudaStreamWaitEvent");
        }
        check(cudaMemcpyAsync(buffers[step.destination]->data(), buffers[step.source]->data(),
                              specs[step.source].size, step.direction, stream), "cudaMemcpyAsync");
        if (step.record_completion) {
            check(cudaEventRecord(queues.events[index], stream), "cudaEventRecord");
        }
    }
    queues.join();
}

}  // namespace detail

}  // namespace scar::cuda_copy_v1
