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

enum class body_opcode {
  push_null,
  attribute,
  unpack_sequence,
  load_global,
  load_global_call,
  call,
  discard,
  load_name,
  build_list,
  build_defaults_tuple,
  list_extend,
  store_subscript,
  default_tuple,
  child_code,
  set_function_defaults,
  make_function,
  store_name,
  store_global,
  constant,
  small_int,
  return_const,
  return_value
};
struct body_instruction {
  body_opcode operation;
  unsigned offset;
  unsigned operand;
  std::string_view name;
  unsigned tuple_size = 0;
};
struct constant_slot {
  unsigned index;
  std::string_view kind;
  std::string_view value;
};
struct body_program {
  std::string_view symbol;
  std::string_view code_sha256;
  std::span<const body_instruction> instructions;
  std::span<const constant_slot> constants;
  bool returns;
  unsigned resume_offset;
  unsigned minor = 0;
};

enum class outcome { unmatched, continued_in_host, completed };
struct body_provider;
template <class Frame>
outcome run_body(Frame &, const body_program &, body_provider);

struct body_provider {
  std::span<const body_program> programs{};
  template <class Frame>
  outcome enter(Frame &frame, std::string_view code_sha256,
                std::string_view symbol) const {
    return enter_at(frame, code_sha256, symbol, 2);
  }
  template <class Frame>
  outcome enter_at(Frame &frame, std::string_view code_sha256,
                   std::string_view symbol, unsigned pc) const {
    for (const auto &p : programs) {
      if (p.code_sha256 == code_sha256 && p.symbol == symbol &&
          !p.instructions.empty() && p.instructions.front().offset == pc)
        return run_body(frame, p, *this);
    }
    return outcome::unmatched;
  }
};

// Adapter to a live import domain. All cache, spec and object operations use
// that domain's actual mappings, locking and reference semantics. The default
// path is only for top-level absolute names and a source-capable modern loader.
// find_spec/create_module/exec_module include their observable callbacks.
template <class Host, class Spec>
auto load_selected(Host &host, Spec &spec, body_provider bodies = {}) ->
    typename Host::object {
  if (!host.supports_modern_loader(spec))
    return host.continue_other_loader(spec);
  auto module = host.create_module(spec);
  host.set_initializing(spec, true);
  try {
    host.publish_module(host.spec_name(spec), module);
    try {
      if constexpr (requires { host.exec_module(spec, module, bodies); })
        host.exec_module(spec, module, bodies);
      else
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
auto default_absolute(Host &host, std::string_view name, unsigned minor,
                      body_provider bodies = {}) -> typename Host::object {
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
          cached = load_selected(host, spec, bodies);
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
  import_request,
  import_from,
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
  body_provider bodies{};
  bool named_request{};
};

template <class Frame, class Slots>
bool invoke_slots(Frame &frame, Slots slots, unsigned count, unsigned minor,
                  body_provider bodies) {
  if (slots.size() < 2 || count > slots.size() - 2)
    return false;
  auto args = slots.last(count);
  auto prefix = slots.subspan(slots.size() - count - 2, 2);
  auto callable = minor == 12 ? 1 : 0;
  if (!prefix[callable] || prefix[1 - callable] ||
      !std::all_of(args.begin(), args.end(),
                   [](const auto &s) { return s.has_value(); }))
    return false;
  auto result = [&] {
    if constexpr (requires { frame.invoke(*prefix[callable], args, bodies); })
      return frame.invoke(*prefix[callable], args, bodies);
    else
      return frame.invoke(*prefix[callable], args);
  }();
  frame.replace(count + 2, std::move(result));
  return true;
}

// The host supplies the actual active frame and live owned operand slots, one
// optional<object> per slot (disengaged is the interpreter's call NULL).
// replace applies the source opcode's stack cleanup and release semantics.
// A callback or exception can transfer control with all prior effects intact.
template <class Frame> outcome run(Frame &frame, const region &program) {
  if (program.instructions.empty() ||
      (program.minor != 12 && program.minor != 14) || !frame.enter(program))
    return outcome::unmatched;
  auto &stack = frame.stack();
  // A legacy frame can decline the entire named-import region before taking
  // ownership of operands. The two services use the actual source importer
  // and IMPORT_FROM lookup (including its circular-import fallback).
  if constexpr (!requires {
                  frame.import_request(0u, *stack.back(), *stack.back());
                  frame.import_from(*stack.back(), 0u);
                })
    if (program.named_request)
      return outcome::unmatched;
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
                                         return default_absolute(
                                             host, ins.name, program.minor,
                                             program.bodies);
                                       });
        frame.replace(2, std::move(value));
        break;
      }
      case opcode::import_request:
        if constexpr (requires {
                        frame.import_request(ins.operand, *stack.back(),
                                             *stack.back());
                      }) {
          if (!program.named_request || slots.size() < 2 ||
              !slots[slots.size() - 2] || !slots.back())
            return outcome::continued_in_host;
          auto returned = frame.import_request(
              ins.operand, *slots[slots.size() - 2], *slots.back());
          frame.replace(2, std::move(returned));
          break;
        } else {
          return outcome::unmatched;
        }
      case opcode::import_from:
        if constexpr (requires {
                        frame.import_from(*stack.back(), ins.operand);
                      }) {
          if (!program.named_request || slots.empty() || !slots.back())
            return outcome::continued_in_host;
          auto value = frame.import_from(*slots.back(), ins.operand);
          stack.emplace_back(std::move(value));
          break;
        } else {
          return outcome::unmatched;
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
        if (!invoke_slots(frame, slots, ins.operand, program.minor,
                          program.bodies))
          return outcome::continued_in_host;
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
// This entry follows actual loader or callable selection. The source frame
// owns locals, globals, builtins and code objects. A mismatch after an opcode
// leaves its original stack intact for the host continuation.
template <class Frame>
outcome run_body(Frame &frame, const body_program &program,
                 body_provider bodies) {
  if (program.instructions.empty() || !frame.enter_body(program))
    return outcome::unmatched;
  auto &stack = frame.stack();
  // Admit a defaults-bearing body only when this frame supplies every
  // corresponding operation. Earlier module publications cannot be repeated.
  constexpr bool checks_tuple =
      requires { frame.matches_defaults_tuple(*stack.back(), 0u); };
  constexpr bool creates_with_defaults =
      requires { frame.make_function(*stack.back(), *stack.back()); };
  constexpr bool sets_defaults =
      requires { frame.set_function_defaults(*stack.back(), *stack.back()); };
  constexpr bool loads_globals = requires { frame.load_global(0u); };
  constexpr bool invokes = requires {
    frame.invoke(*stack.back(), std::span(stack), bodies);
  } || requires { frame.invoke(*stack.back(), std::span(stack)); };
  constexpr bool loads_names = requires { frame.load_name(0u); };
  constexpr bool builds_tuples =
      requires { frame.build_tuple(std::span(stack)); };
  constexpr bool builds_lists =
      requires { frame.build_list(std::span(stack)); };
  constexpr bool extends_lists =
      requires { frame.extend_list(*stack.back(), *stack.back()); };
  constexpr bool reads_attributes =
      requires { frame.attribute(*stack.back(), 0u); };
  constexpr bool unpacks = requires {
    frame.admit_unpack_store(program);
    frame.unpack_sequence(*stack.back(), 0u);
  };
  constexpr bool stores_items = requires {
    frame.store_subscript(*stack.back(), *stack.back(), *stack.back());
  };
  bool uses_unpack = false;
  for (const auto &ins : program.instructions) {
    uses_unpack |= ins.operation == body_opcode::unpack_sequence;
    if ((!reads_attributes && ins.operation == body_opcode::attribute) ||
        (!unpacks && ins.operation == body_opcode::unpack_sequence) ||
        (!stores_items && ins.operation == body_opcode::store_subscript) ||
        (!loads_globals && (ins.operation == body_opcode::load_global ||
                            ins.operation == body_opcode::load_global_call)) ||
        (!invokes && ins.operation == body_opcode::call) ||
        (!checks_tuple && ins.operation == body_opcode::default_tuple) ||
        (!creates_with_defaults &&
         ins.operation == body_opcode::make_function && ins.operand == 1) ||
        (!sets_defaults &&
         ins.operation == body_opcode::set_function_defaults) ||
        (!loads_names && ins.operation == body_opcode::load_name) ||
        (!builds_tuples &&
         ins.operation == body_opcode::build_defaults_tuple) ||
        (!builds_lists && ins.operation == body_opcode::build_list) ||
        (!extends_lists && ins.operation == body_opcode::list_extend))
      return outcome::unmatched;
  }
  if (uses_unpack) {
    if constexpr (unpacks) {
      // The active frame certifies source-compatible iteration, ownership,
      // exception dispatch, and operand capacity before the first operation.
      if (!frame.admit_unpack_store(program))
        return outcome::unmatched;
    }
  } else if (program.instructions.back().operation ==
             body_opcode::store_subscript) {
    if constexpr (stores_items &&
                  requires { frame.admit_subscript_region(program); }) {
      // Admission guarantees source-equivalent dispatch for this live frame.
      // It must precede allocation, mapping lookups and all source callbacks.
      if (!frame.admit_subscript_region(program))
        return outcome::unmatched;
    } else {
      return outcome::unmatched;
    }
  }
  const auto base = stack.size();
  bool pending_defaults = false;
  for (const auto &ins : program.instructions) {
    try {
      if (!frame.before(ins.offset))
        return outcome::continued_in_host;
      switch (ins.operation) {
      case body_opcode::push_null:
        stack.emplace_back(std::nullopt);
        break;
      case body_opcode::attribute:
        if constexpr (reads_attributes) {
          if (stack.size() <= base || !stack.back())
            return outcome::continued_in_host;
          auto value = frame.attribute(*stack.back(), ins.operand);
          frame.replace(1, std::move(value));
        }
        break;
      case body_opcode::unpack_sequence:
        if constexpr (unpacks) {
          if (stack.size() != base + 1 || !stack.back())
            return outcome::continued_in_host;
          // A successful provider returns exactly the source arity, left to
          // right. A short/long iterable or a failing iterator raises before
          // the first assignment target is evaluated.
          auto iterable = std::move(*stack.back());
          stack.pop_back();
          auto items = frame.unpack_sequence(std::move(iterable), ins.operand);
          if (items.size() != ins.operand)
            throw std::logic_error(
                "invalid result from admitted unpack provider");
          for (auto i = items.rbegin(); i != items.rend(); ++i)
            stack.emplace_back(std::move(*i));
        }
        break;
      case body_opcode::load_global:
      case body_opcode::load_global_call:
        if constexpr (loads_globals) {
          auto value = frame.load_global(ins.operand);
          bool pair = ins.operation == body_opcode::load_global_call;
          if (pair && program.minor == 12)
            stack.emplace_back(std::nullopt);
          stack.emplace_back(std::move(value));
          if (pair && program.minor == 14)
            stack.emplace_back(std::nullopt);
        }
        break;
      case body_opcode::call:
        if constexpr (invokes) {
          if (stack.size() < base ||
              !invoke_slots(frame, std::span(stack).subspan(base), ins.operand,
                            program.minor, bodies))
            return outcome::continued_in_host;
        }
        break;
      case body_opcode::discard:
        if (stack.size() <= base || !stack.back())
          return outcome::continued_in_host;
        frame.replace(1, std::nullopt);
        break;
      case body_opcode::load_name:
        if constexpr (loads_names)
          stack.emplace_back(frame.load_name(ins.operand));
        break;
      case body_opcode::build_list:
        if constexpr (builds_lists) {
          if (stack.size() < base || ins.operand > stack.size() - base)
            return outcome::continued_in_host;
          auto items = std::span(stack).last(ins.operand);
          auto list = frame.build_list(items);
          frame.replace(ins.operand, std::move(list));
        }
        break;
      case body_opcode::build_defaults_tuple:
        if constexpr (builds_tuples) {
          if (pending_defaults || stack.size() - base != ins.operand)
            return outcome::continued_in_host;
          auto tuple = frame.build_tuple(std::span(stack).last(ins.operand));
          frame.replace(ins.operand, std::move(tuple));
          pending_defaults = true;
        }
        break;
      case body_opcode::list_extend:
        if constexpr (extends_lists) {
          if (stack.size() < base + 2 || !stack.back() ||
              !stack[stack.size() - 2])
            return outcome::continued_in_host;
          frame.extend_list(*stack[stack.size() - 2], *stack.back());
          frame.replace(1, std::nullopt);
        }
        break;
      case body_opcode::store_subscript:
        if constexpr (stores_items) {
          if (stack.size() < base + 3)
            return outcome::continued_in_host;
          auto slots = std::span(stack).last(3);
          if (!slots[0] || !slots[1] || !slots[2])
            return outcome::continued_in_host;
          frame.store_subscript(*slots[1], *slots[2], *slots[0]);
          frame.replace(3, std::nullopt);
        }
        break;
      case body_opcode::default_tuple:
        if constexpr (checks_tuple) {
          if (stack.size() != base)
            return outcome::continued_in_host;
          auto tuple = frame.constant(ins.operand);
          if (!frame.matches_defaults_tuple(tuple, ins.tuple_size))
            return outcome::continued_in_host;
          stack.emplace_back(std::move(tuple));
          pending_defaults = true;
        }
        break;
      case body_opcode::child_code:
      case body_opcode::constant:
        stack.emplace_back(frame.constant(ins.operand));
        break;
      case body_opcode::small_int:
        stack.emplace_back(frame.small_int(ins.operand));
        break;
      case body_opcode::make_function: {
        if (stack.size() != base + 1 + pending_defaults || !stack.back() ||
            !frame.matches_code(*stack.back(), ins.name))
          return outcome::continued_in_host;
        // MAKE_FUNCTION takes its globals and effective builtins from this
        // activation, including an updated __builtins__ entry in its globals.
        if (ins.operand == 1) {
          if constexpr (creates_with_defaults) {
            if (!pending_defaults || !stack[base])
              return outcome::continued_in_host;
            auto function = frame.make_function(*stack.back(), *stack[base]);
            frame.replace(2, std::move(function));
            pending_defaults = false;
          }
        } else {
          auto function = frame.make_function(*stack.back());
          frame.replace(1, std::move(function));
        }
        break;
      }
      case body_opcode::set_function_defaults: {
        if constexpr (sets_defaults) {
          if (!pending_defaults || stack.size() != base + 2 || !stack[base] ||
              !stack.back())
            return outcome::continued_in_host;
          frame.set_function_defaults(*stack.back(), *stack[base]);
          auto function = std::move(stack.back());
          frame.replace(2, std::move(function));
          pending_defaults = false;
        }
        break;
      }
      case body_opcode::store_name:
      case body_opcode::store_global:
        if (stack.size() <= base || !stack.back())
          return outcome::continued_in_host;
        frame.publish(ins.operand, ins.operation == body_opcode::store_global,
                      *stack.back());
        frame.replace(1, std::nullopt);
        break;
      case body_opcode::return_const:
        if (stack.size() != base)
          return outcome::continued_in_host;
        frame.return_constant(ins.operand);
        return outcome::completed;
      case body_opcode::return_value:
        if (stack.size() != base + 1 || !stack.back())
          return outcome::continued_in_host;
        frame.return_top();
        return outcome::completed;
      }
    } catch (...) {
      frame.unwind(ins.offset, std::current_exception());
      return outcome::continued_in_host;
    }
  }
  frame.resume_at(program.resume_offset);
  return outcome::continued_in_host;
}
template <class Frame>
outcome run_body(Frame &frame, const body_program &program) {
  return run_body(frame, program, {});
}
} // namespace scar::imports
