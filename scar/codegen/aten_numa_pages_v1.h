#pragma once

#include <ATen/core/Tensor.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <ostream>
#include <stdexcept>
#include <sys/syscall.h>
#include <unistd.h>

namespace scar_aten_graph {

struct NumaPageSample {
    std::size_t storage_bytes = 0;
    std::size_t page_size_bytes = 0;
    std::size_t storage_pages = 0;
    std::size_t sampled_pages = 0;
    std::map<int, std::size_t> node_pages;
    std::size_t unknown_pages = 0;
    int query_errno = 0;
};

inline NumaPageSample observe_numa_pages(const void* data, std::size_t storage_bytes) {
    NumaPageSample result;
    result.storage_bytes = storage_bytes;
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        result.query_errno = EINVAL;
        return result;
    }
    result.page_size_bytes = static_cast<std::size_t>(page_size);
    if (storage_bytes == 0) return result;

    const auto begin = reinterpret_cast<std::uintptr_t>(data);
    if (data == nullptr || storage_bytes - 1 > std::numeric_limits<std::uintptr_t>::max() - begin) {
        result.query_errno = data == nullptr ? EFAULT : EOVERFLOW;
        return result;
    }
    const auto first_page = begin / result.page_size_bytes;
    const auto last_page = (begin + (storage_bytes - 1)) / result.page_size_bytes;
    result.storage_pages = last_page - first_page + 1;

    constexpr std::size_t max_samples = 32;
    result.sampled_pages = std::min(max_samples, result.storage_pages);
#if defined(SYS_move_pages)
    std::array<void*, max_samples> pages{};
    const auto span = result.storage_pages - 1;
    const auto steps = std::max<std::size_t>(1, result.sampled_pages - 1);
    for (std::size_t i = 0; i < result.sampled_pages; ++i) {
        // Evenly cover the storage range, without multiplying its unbounded extent.
        const auto offset = (span / steps) * i + ((span % steps) * i) / steps;
        pages[i] = reinterpret_cast<void*>((first_page + offset) * result.page_size_bytes);
    }
    std::array<int, max_samples> nodes;
    nodes.fill(-ENODATA);
    // A null target node array requests residency only. Zero flags prevent page migration.
    const auto queried = syscall(SYS_move_pages, 0, static_cast<unsigned long>(result.sampled_pages),
                                 pages.data(), static_cast<const int*>(nullptr), nodes.data(), 0);
    if (queried == -1) {
        result.query_errno = errno;
        result.unknown_pages = result.sampled_pages;
        return result;
    }
    for (std::size_t i = 0; i < result.sampled_pages; ++i) {
        if (nodes[i] >= 0) {
            ++result.node_pages[nodes[i]];
        } else {
            ++result.unknown_pages;
        }
    }
#else
    result.query_errno = ENOSYS;
    result.unknown_pages = result.sampled_pages;
#endif
    return result;
}

inline NumaPageSample observe_numa_pages(const at::Tensor& tensor) {
    if (!tensor.defined() || !tensor.device().is_cpu() || tensor.layout() != at::kStrided
            || !tensor.has_storage() || !tensor.storage().device().is_cpu()) {
        throw std::invalid_argument("NUMA observation requires a strided CPU Tensor storage");
    }
    const auto storage = tensor.storage();
    return observe_numa_pages(storage.data(), storage.nbytes());
}

inline void write_numa_pages(std::ostream& stream, const NumaPageSample& sample) {
    const char* status = "OBSERVED";
    if (sample.query_errno != 0) {
        status = "UNAVAILABLE";
    } else if (sample.storage_bytes == 0) {
        status = "EMPTY";
    } else if (sample.unknown_pages != 0) {
        status = "PARTIAL";
    }
    stream << "{\"status\":\"" << status << "\",\"storage_bytes\":" << sample.storage_bytes
           << ",\"page_size_bytes\":" << sample.page_size_bytes << ",\"storage_pages\":" << sample.storage_pages
           << ",\"sampled_pages\":" << sample.sampled_pages << ",\"unknown_pages\":" << sample.unknown_pages
           << ",\"query_errno\":";
    if (sample.query_errno == 0) {
        stream << "null";
    } else {
        stream << sample.query_errno;
    }
    stream << ",\"node_pages\":{";
    for (auto it = sample.node_pages.begin(); it != sample.node_pages.end(); ++it) {
        if (it != sample.node_pages.begin()) stream << ',';
        stream << '\"' << it->first << "\":" << it->second;
    }
    stream << "}}";
}

}  // namespace scar_aten_graph
