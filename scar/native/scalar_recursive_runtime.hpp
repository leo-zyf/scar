// Included in the generated TU after the original scalar leaf types and helpers.
#include "scalar_ordered_state.hpp"

#include <chrono>
#include <coroutine>
#include <exception>
#include <unordered_set>

namespace {
using Pair = std::tuple<Integer, Integer>;
using Key = std::tuple<State, Integer>;
using Ranked = std::tuple<std::optional<Integer>, std::optional<Move>>;

struct TupleHash {
    template<class Value> std::size_t operator()(const Value& value) const {
        if constexpr (requires { std::tuple_size<Value>::value; }) {
            return std::apply([&](const auto&... item) {
                std::size_t hash = 0;
                ((hash ^= (*this)(item) + 0x9e3779b9U + (hash << 6) + (hash >> 2)), ...);
                return hash;
            }, value);
        } else {
            return std::hash<Value>{}(value);
        }
    }
};

using EntryMap = scar::native::OrderedMap<Key, Pair, TupleHash>;
using MoveMap = scar::native::OrderedMap<State, Move, TupleHash>;
using History = std::unordered_set<State, TupleHash>;

std::optional<Move> find_move(const MoveMap& map, const State& key) {
    const auto* found = map.find(key);
    return found ? std::optional<Move>(*found) : std::nullopt;
}

template<class Value>
const Value& read(const std::optional<Value>& local) {
    if (!local) throw Decline{};
    return *local;
}

Integer checked_size(std::size_t count) {
    if (count > static_cast<std::size_t>(high)) throw Decline{};
    return static_cast<Integer>(count);
}

Integer remainder(Integer a, Integer b) {
    if (b == 0) throw Decline{};
    if (b == -1) return 0;
    const Integer r = a % b;
    return r && ((r < 0) != (b < 0)) ? r + b : r;
}

struct SourceStop {};
double exact_clock() {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

struct RecursionLimit {
    inline static thread_local unsigned active = 0;
    RecursionLimit() {
        if (active >= 256) throw Decline{};
        ++active;
    }
    ~RecursionLimit() { --active; }
    RecursionLimit(const RecursionLimit&) = delete;
    RecursionLimit& operator=(const RecursionLimit&) = delete;
};

template<class Value>
class Stream {
public:
    struct promise_type {
        std::optional<Value> value;
        std::exception_ptr pending;
        Stream get_return_object() {
            return Stream(std::coroutine_handle<promise_type>::from_promise(*this));
        }
        std::suspend_always initial_suspend() const noexcept { return {}; }
        std::suspend_always final_suspend() const noexcept { return {}; }
        std::suspend_always yield_value(Value next) {
            value.emplace(std::move(next));
            return {};
        }
        void return_void() const noexcept {}
        void unhandled_exception() noexcept { pending = std::current_exception(); }
    };
    explicit Stream(std::coroutine_handle<promise_type> step) noexcept : step_(step) {}
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    Stream(Stream&& other) noexcept : step_(std::exchange(other.step_, {})) {}
    ~Stream() { if (step_) step_.destroy(); }
    bool advance() {
        if (!step_ || step_.done()) return false;
        step_.promise().value.reset();
        step_.resume();
        if (step_.promise().pending) std::rethrow_exception(step_.promise().pending);
        return step_.promise().value.has_value();
    }
    const Value& current() const { return read(step_.promise().value); }
private:
    std::coroutine_handle<promise_type> step_;
};
}  // namespace
