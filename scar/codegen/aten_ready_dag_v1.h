#pragma once

#include "aten_native_team_v1.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace scar_aten_graph {

template <std::size_t Count>
inline constexpr auto source_ready_order = [] {
    std::array<std::size_t, Count> order{};
    for (std::size_t index = 0; index < Count; ++index) order[index] = index;
    return order;
}();

template <std::size_t Count, std::size_t Edges>
void validate_ready_order(const std::array<std::size_t, Count>& order,
                          const std::array<std::size_t, Count + 1>& offsets,
                          const std::array<std::size_t, Edges>& predecessors) {
    std::array<std::size_t, Count> rank;
    rank.fill(Count);
    for (std::size_t position = 0; position < Count; ++position) {
        const auto index = order[position];
        if (index >= Count || rank[index] != Count) {
            throw std::invalid_argument("ready DAG order must name each operation exactly once");
        }
        rank[index] = position;
    }
    for (std::size_t index = 0; index < Count; ++index) {
        for (auto edge = offsets[index]; edge < offsets[index + 1]; ++edge) {
            if (rank[predecessors[edge]] >= rank[index]) {
                throw std::invalid_argument("ready DAG order violates a Tensor dependency");
            }
        }
    }
}

// Cross-lane Tensor edges publish and wait on completion. Independent use
// counters release backing handles without introducing scheduling constraints.
template <std::size_t Count, std::size_t Edges, class Invoke, class Release, class Prepare>
void execute_ready_dag(CpuTeam& team, const std::array<std::uint8_t, Count>& lanes,
                       const std::array<std::size_t, Count + 1>& offsets,
                       const std::array<std::size_t, Edges>& predecessors,
                       const std::array<std::size_t, Count>& consumer_counts,
                       const std::array<bool, Count>& exported, Invoke&& invoke, Release&& release,
                       Prepare&& prepare,
                       const std::array<std::size_t, Count>& order = source_ready_order<Count>) {
    enum Completion : std::uint8_t { pending, complete, cancelled };

    // The generated source order has already been checked against its edges.
    if (&order != &source_ready_order<Count>) validate_ready_order(order, offsets, predecessors);

    std::array<bool, 3> used{};
    for (const auto lane : lanes) {
        if (lane >= used.size()) throw std::invalid_argument("invalid native DAG lane");
        used[lane] = true;
    }

    std::array<bool, Count> has_remote_reader{};
    for (std::size_t index = 0; index < Count; ++index) {
        for (auto edge = offsets[index]; edge < offsets[index + 1]; ++edge) {
            const auto parent = predecessors[edge];
            if (lanes[parent] != lanes[index]) has_remote_reader[parent] = true;
        }
    }

    std::array<std::atomic<std::size_t>, Count> remaining{};
    for (std::size_t i = 0; i < Count; ++i) {
        if (!exported[i] && consumer_counts[i] > 1) {
            remaining[i].store(consumer_counts[i], std::memory_order_relaxed);
        }
    }
    std::array<std::atomic<Completion>, Count> state{};
    std::atomic<bool> stopping = false;
    const auto publish = [&](std::size_t index, Completion outcome) noexcept {
        Completion expected = pending;
        if (state[index].compare_exchange_strong(expected, outcome, std::memory_order_release,
                                                std::memory_order_relaxed)) {
            state[index].notify_all();
        }
    };
    const auto abort = [&]() noexcept {
        if (stopping.exchange(true, std::memory_order_acq_rel)) return;
        for (std::size_t index = 0; index < Count; ++index) {
            if (has_remote_reader[index]) publish(index, cancelled);
        }
    };
    const auto run_lane = [&](std::size_t lane) {
        if (stopping.load(std::memory_order_acquire)) return;
        prepare(lane);
        for (const auto index : order) {
            if (lanes[index] != lane) continue;
            for (auto edge = offsets[index]; edge < offsets[index + 1]; ++edge) {
                const auto parent = predecessors[edge];
                if (lanes[parent] == lane) continue;
                state[parent].wait(pending, std::memory_order_acquire);
                if (state[parent].load(std::memory_order_acquire) == cancelled) return;
            }
            if (stopping.load(std::memory_order_acquire)) return;
            invoke(index);
            for (auto edge = offsets[index]; edge < offsets[index + 1]; ++edge) {
                const auto parent = predecessors[edge];
                if (exported[parent]) continue;
                // The last read-modify-write acquires the earlier readers'
                // releases before resetting their shared Tensor handle.
                if (consumer_counts[parent] == 1 ||
                        remaining[parent].fetch_sub(1, std::memory_order_acq_rel) == 1) {
                    release(parent);
                }
            }
            if (!exported[index] && consumer_counts[index] == 0) release(index);
            if (has_remote_reader[index]) publish(index, complete);
        }
    };

    team.run_lanes_with_abort(used, run_lane, abort);
}

}  // namespace scar_aten_graph
