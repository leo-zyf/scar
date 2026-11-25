#pragma once

#include <algorithm>
#include <cstddef>
#include <exception>
#include <list>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace scar::imports {

// Storage for a native provider that owns its registry. Lookup distinguishes
// an absent key from a present None value. Erase unlinks before destruction
// so a finalizer can reenter and observe the updated registry.
template <class Object> class ordered_registry {
  std::list<std::pair<std::string, Object>> entries_;
  auto find(std::string_view name) {
    return std::find_if(entries_.begin(), entries_.end(),
                        [&](const auto &item) { return item.first == name; });
  }

public:
  using object = Object;
  std::optional<Object> lookup_module(std::string_view name) {
    const auto it = find(name);
    if (it == entries_.end())
      return std::nullopt;
    return it->second;
  }
  void publish_module(std::string_view name, Object value) {
    const auto it = find(name);
    if (it == entries_.end())
      entries_.emplace_back(name, std::move(value));
    else
      std::swap(it->second, value);
  }
  Object pop_module(std::string_view name) {
    const auto it = find(name);
    if (it == entries_.end())
      throw std::out_of_range(std::string(name));
    auto value = std::move(it->second);
    entries_.erase(it);
    return value;
  }
  void erase_module_if_present(std::string_view name) {
    auto it = find(name);
    if (it != entries_.end()) {
      auto retained = std::move(it->second);
      entries_.erase(it);
    }
  }
};

// Adapter to a live import domain. All cache, spec and object operations use
// that domain's actual mappings, locking and reference semantics. The default
// path is only for top-level absolute names and a source-capable modern loader.
// find_spec/create_module/exec_module include their observable callbacks.
template <class Host, class Spec>
auto load_selected(Host &host, Spec &spec) -> typename Host::object {
  if (!host.supports_modern_loader(spec))
    return host.continue_other_loader(spec);
  auto module = host.create_module(spec);
  host.set_initializing(spec, true);
  try {
    host.publish_module(host.spec_name(spec), module);
    try {
      host.exec_module(spec, module);
    } catch (...) {
      host.erase_module_if_present(host.spec_name(spec));
      throw;
    }
    // Replacing the local result releases a displaced module before
    // republishing the actual returned object at the end of the registry.
    module = host.pop_module(host.spec_name(spec));
    host.publish_module(host.spec_name(spec), module);
    host.verbose_import(spec);
  } catch (...) {
    host.set_initializing(spec, false);
    throw;
  }
  host.set_initializing(spec, false);
  return module;
}

template <class Host>
auto default_absolute(Host &host, std::string_view name, unsigned minor) ->
    typename Host::object {
  {
    auto cached = host.lookup_module(name);
    if (cached && !host.is_none(*cached)) {
      host.ensure_initialized(name, *cached);
      return std::move(*cached);
    }
  }
  host.audit_import(name);
  for (;;) {
    // Both the audit hook and the initialization check can mutate the cache.
    auto cached = host.lookup_module(name);
    if (!cached || host.is_initializing(*cached)) {
      bool loaded = false;
      host.with_module_lock(name, [&] {
        cached = host.lookup_module(name);
        if (!cached) {
          auto spec = host.find_spec(name);
          cached = load_selected(host, spec);
          loaded = true;
        }
      });
      if (loaded)
        return std::move(*cached);
      host.wait_for_module(name);
    } else if (minor == 14 && !host.same_cached_module(name, *cached)) {
      continue;
    }
    if (host.is_none(*cached))
      return host.raise_halted_import(name);
    return std::move(*cached);
  }
}

enum class opcode {
  constant,
  small_int,
  import_name,
  store_name,
  store_global,
  push_null,
  load_name,
  attribute,
  call,
  discard
};
struct instruction {
  opcode operation;
  unsigned offset;
  unsigned operand;
  std::string_view name;
};

struct region {
  std::string_view code_sha256;
  unsigned minor;
  std::span<const instruction> instructions;
};

enum class outcome { unmatched, continued_in_host, completed };

// The host supplies the actual active frame and live owned operand slots, one
// optional<object> per slot (disengaged is the interpreter's call NULL).
// replace applies the source opcode's stack cleanup and release semantics.
// A callback or exception can transfer control with all prior effects intact.
template <class Frame> outcome run(Frame &frame, const region &program) {
  if (program.instructions.empty() ||
      (program.minor != 12 && program.minor != 14) || !frame.enter(program))
    return outcome::unmatched;
  auto &stack = frame.stack();
  const auto base = stack.size();
  for (auto ins : program.instructions) {
    try {
      if (!frame.before(ins.offset))
        return outcome::continued_in_host;
      if (stack.size() < base)
        return outcome::continued_in_host;
      auto slots = std::span(stack).subspan(base);
      switch (ins.operation) {
      case opcode::constant:
        stack.emplace_back(frame.constant(ins.operand));
        break;
      case opcode::small_int:
        stack.emplace_back(frame.small_int(ins.operand));
        break;
      case opcode::import_name: {
        // The active frame resolves __import__ now. Only a dispatch of this
        // original absolute request to the default importer uses the kernel.
        if (slots.size() < 2 || !slots[slots.size() - 2] || !slots.back())
          return outcome::continued_in_host;
        auto value = frame.import_name(ins.operand, *slots[slots.size() - 2],
                                       *slots.back(), [&](auto &host) {
                                         return default_absolute(host, ins.name,
                                                                 program.minor);
                                       });
        frame.replace(2, std::move(value));
        break;
      }
      case opcode::store_name:
      case opcode::store_global:
        if (slots.empty() || !slots.back())
          return outcome::continued_in_host;
        frame.publish(ins.operand, ins.operation == opcode::store_global,
                      *slots.back());
        frame.replace(1, std::nullopt);
        break;
      case opcode::push_null:
        stack.emplace_back(std::nullopt);
        break;
      case opcode::load_name:
        stack.emplace_back(frame.load_name(ins.operand));
        break;
      case opcode::attribute: {
        if (slots.empty() || !slots.back())
          return outcome::continued_in_host;
        auto member = frame.attribute(*slots.back(), ins.operand);
        frame.replace(1, std::move(member));
        break;
      }
      case opcode::call: {
        if (slots.size() < 2 || ins.operand > slots.size() - 2)
          return outcome::continued_in_host;
        auto args = slots.last(ins.operand);
        auto prefix = slots.subspan(slots.size() - ins.operand - 2, 2);
        auto callable = program.minor == 12 ? 1 : 0;
        if (!prefix[callable] || prefix[1 - callable] ||
            !std::all_of(args.begin(), args.end(),
                         [](const auto &s) { return s.has_value(); }))
          return outcome::continued_in_host;
        auto result = frame.invoke(*prefix[callable], args);
        frame.replace(ins.operand + 2, std::move(result));
        break;
      }
      case opcode::discard:
        if (slots.empty() || !slots.back())
          return outcome::continued_in_host;
        frame.replace(1, std::nullopt);
        break;
      }
    } catch (...) {
      frame.unwind(ins.offset, std::current_exception());
      return outcome::continued_in_host;
    }
  }
  return outcome::completed;
}
} // namespace scar::imports
