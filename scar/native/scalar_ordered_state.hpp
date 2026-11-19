#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <unordered_map>
#include <utility>

namespace scar::native {

// A single owner holds the recursive state. Rehashing preserves element
// addresses, so the order links can share the hash nodes' key and value.
template<class Key, class Value, class Hash = std::hash<Key>, class Equal = std::equal_to<Key>>
class OrderedMap {
public:
    explicit OrderedMap(const Hash& hash = Hash{}, const Equal& equal = Equal{})
        : entries_(0, hash, equal) {}

    OrderedMap(const OrderedMap&) = delete;
    OrderedMap& operator=(const OrderedMap&) = delete;
    OrderedMap(OrderedMap&&) = delete;
    OrderedMap& operator=(OrderedMap&&) = delete;

    [[nodiscard]] bool empty() const noexcept { return entries_.empty(); }
    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }
    void reserve(std::size_t count) { entries_.reserve(count); }

    void clear() noexcept {
        entries_.clear();
        first_ = nullptr;
        last_ = nullptr;
    }

    [[nodiscard]] Value* find(const Key& key) {
        const auto found = entries_.find(key);
        return found == entries_.end() ? nullptr : std::addressof(found->second.value);
    }

    [[nodiscard]] const Value* find(const Key& key) const {
        const auto found = entries_.find(key);
        return found == entries_.end() ? nullptr : std::addressof(found->second.value);
    }

    [[nodiscard]] Value get_or(const Key& key, Value fallback) const {
        if (const auto* value = find(key)) {
            return *value;
        }
        return fallback;
    }

    // Return whether a new key was inserted. Updating an existing value
    // preserves both its key and its place in the insertion order.
    template<class K, class V>
    bool insert_or_assign(K&& key, V&& value) {
        auto [position, inserted] = entries_.try_emplace(std::forward<K>(key), std::forward<V>(value));
        auto& entry = position->second;
        if (!inserted) {
            entry.value = std::forward<V>(value);
            return false;
        }
        entry.key = std::addressof(position->first);
        entry.previous = last_;
        if (last_) {
            last_->next = std::addressof(entry);
        } else {
            first_ = std::addressof(entry);
        }
        last_ = std::addressof(entry);
        return true;
    }

    // Borrowed key/value addresses survive rehash and are valid until that
    // entry is erased or the owner is cleared or destroyed.
    [[nodiscard]] const Key* first_key() const noexcept {
        return first_ ? first_->key : nullptr;
    }

    [[nodiscard]] const Key* first_key_except(const Key& keep) const {
        if (!first_) {
            return nullptr;
        }
        if (!entries_.key_eq()(*first_->key, keep)) {
            return first_->key;
        }
        return first_->next ? first_->next->key : nullptr;
    }

    bool erase(const Key& key) {
        const auto position = entries_.find(key);
        if (position == entries_.end()) {
            return false;
        }
        const auto& entry = position->second;
        if (entry.previous) {
            entry.previous->next = entry.next;
        } else {
            first_ = entry.next;
        }
        if (entry.next) {
            entry.next->previous = entry.previous;
        } else {
            last_ = entry.previous;
        }
        entries_.erase(position);
        return true;
    }

    bool erase_first() {
        const auto* key = first_key();
        return key && erase(*key);
    }

    bool erase_first_except(const Key& keep) {
        const auto* key = first_key_except(keep);
        return key && erase(*key);
    }

    template<class Visitor>
    void for_each(Visitor&& visitor) const {
        for (const auto* entry = first_; entry; entry = entry->next) {
            visitor(*entry->key, entry->value);
        }
    }

private:
    struct Entry {
        template<class V>
        explicit Entry(V&& initial) : value(std::forward<V>(initial)) {}

        Value value;
        const Key* key = nullptr;
        Entry* previous = nullptr;
        Entry* next = nullptr;
    };

    std::unordered_map<Key, Entry, Hash, Equal> entries_;
    Entry* first_ = nullptr;
    Entry* last_ = nullptr;
};

// Borrow the owner's set without copying it or exposing mutations to a callee.
template<class Set>
class ReadOnlySetView {
public:
    explicit ReadOnlySetView(const Set& owner) noexcept : owner_(owner) {}
    ReadOnlySetView(const Set&&) = delete;

    template<class K>
    [[nodiscard]] bool contains(const K& key) const { return owner_.contains(key); }

    [[nodiscard]] std::size_t size() const { return owner_.size(); }

private:
    const Set& owner_;
};

}  // namespace scar::native
