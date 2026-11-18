#pragma once

#include <cstdint>
#include <span>
#include <string_view>

namespace scar {

inline constexpr int aten_boundary_abi_version = 1;

enum class AtenMode { Serial, TwoCpu };

struct AtenBoundaryPlan {
    int abi_version;
    AtenMode mode;
    int caller_cpu;
    int worker_cpu;
    int intraop;
    int interop;
    std::span<const std::int64_t> shape;
    std::span<const std::int64_t> strides;
    std::string_view dtype;
    std::string_view boundary;
    std::string_view measured_source_sha256;
};

}  // namespace scar

int scar_aten_boundary_main(const scar::AtenBoundaryPlan&);
