#pragma once

#include "import.hpp"
#include <iostream>
#include <map>
#include <memory>
#include <vector>

namespace scar::imports {
struct project_module {
  std::string_view name, symbol, code;
  void (*initialize)() = nullptr;
};

namespace isolated {
struct value;
using object = std::shared_ptr<value>;
using slot = std::optional<object>;
using scope = std::map<std::string, object, std::less<>>;
struct value {
  enum class type { scalar, module, code, function, print };
  type kind;
  std::string display;
  const body_program *body{};
  scope members{};
  bool initializing{};
  explicit value(type k, std::string_view text = {}) : kind(k), display(text) {}
};
inline object make(value::type kind, std::string_view text = {}) {
  return std::make_shared<value>(kind, text);
}
struct operands {
  std::vector<slot> values;
  auto &stack() { return values; }
  bool before(unsigned) const { return true; }
  void replace(std::size_t n, slot next) {
    values.resize(values.size() - n);
    if (next)
      values.emplace_back(std::move(next));
  }
  [[noreturn]] void unwind(unsigned, std::exception_ptr error) {
    std::rethrow_exception(error);
  }
};
struct activation : operands {
  const body_program &body;
  body_provider provider;
  object owner, returned;
  bool enter_body(const body_program &program) const {
    return &program == &body;
  }
  object small_int(unsigned n) {
    return make(value::type::scalar, std::to_string(n));
  }
  object constant(unsigned index) {
    for (const auto &c : body.constants)
      if (c.index == index)
        return make(value::type::scalar,
                    c.kind == "NoneType" ? "None" : c.value);
    for (const auto &i : body.instructions)
      if (i.operation == body_opcode::child_code && i.operand == index)
        for (const auto &candidate : provider.programs)
          if (candidate.code_sha256 == i.name) {
            object code = make(value::type::code);
            code->body = &candidate;
            return code;
          }
    throw std::logic_error("unbound local constant");
  }
  bool matches_code(const object &code, std::string_view sha) const {
    return code->kind == value::type::code && code->body->code_sha256 == sha;
  }
  object make_function(const object &code) {
    object result = make(value::type::function);
    result->body = code->body;
    return result;
  }
  void publish(unsigned index, bool, const object &item) {
    for (const auto &i : body.instructions)
      if ((i.operation == body_opcode::store_name ||
           i.operation == body_opcode::store_global) &&
          i.operand == index) {
        owner->members[std::string(i.name)] = item;
        return;
      }
    throw std::logic_error("unbound module publication");
  }
  void return_constant(unsigned index) { returned = constant(index); }
  void return_top() {
    returned = *values.back();
    values.pop_back();
  }
  void resume_at(unsigned) {
    throw std::logic_error("partial local activation");
  }
  object execute() {
    if (run_body(*this, body) != outcome::completed || !returned)
      throw std::logic_error("incomplete local activation");
    return returned;
  }
};

struct process : operands, ordered_registry<object> {
  std::span<const project_module> catalog;
  const region *current{};
  scope globals;
  object none = make(value::type::scalar, "None");
  object zero = make(value::type::scalar, "0");
  object printer = make(value::type::print);
  explicit process(std::span<const project_module> m) : catalog(m) {}
  bool enter(const region &r) const { return &r == current && values.empty(); }
  std::string_view name_at(unsigned i) const {
    for (const auto &ins : current->instructions)
      if (ins.operand == i && !ins.name.empty())
        return ins.name;
    throw std::logic_error("unbound local name");
  }
  object small_int(unsigned n) {
    if (n != 0)
      throw std::logic_error("unsupported import level");
    return zero;
  }
  object constant(unsigned i) {
    const auto first = current->instructions[0],
               second = current->instructions[1];
    if (first.operation == opcode::constant && i == first.operand)
      return zero;
    if (second.operation == opcode::constant && i == second.operand)
      return none;
    throw std::logic_error("unexpected import constant");
  }
  object load_name(unsigned i) {
    auto name = name_at(i);
    if (auto it = globals.find(name); it != globals.end())
      return it->second;
    if (name == "print")
      return printer;
    throw std::logic_error("unbound local entry name");
  }
  void publish(unsigned i, bool, const object &v) {
    globals[std::string(name_at(i))] = v;
  }
  object attribute(const object &v, unsigned i) {
    return v->members.at(std::string(name_at(i)));
  }
  object invoke(const object &callable, std::span<slot> args,
                body_provider provider) {
    if (callable->kind == value::type::function && args.empty()) {
      activation f{{}, *callable->body, provider, nullptr, nullptr};
      return f.execute();
    }
    if (callable != printer)
      throw std::logic_error("unbound callable");
    for (std::size_t i = 0; i < args.size(); ++i) {
      if (i)
        std::cout.put(' ');
      const auto &s = (*args[i])->display;
      std::cout.write(s.data(), static_cast<std::streamsize>(s.size()));
    }
    std::cout.put('\n');
    return none;
  }
  template <class F>
  object import_name(unsigned, const object &level, const object &from,
                     F fetch) {
    if (level != zero || from != none)
      throw std::logic_error("unsupported local import arguments");
    return fetch(*this);
  }
  struct spec {
    const project_module *definition;
    object module;
  };
  spec find_spec(std::string_view name) {
    for (auto &m : catalog)
      if (m.name == name)
        return {&m, {}};
    throw std::logic_error("missing sealed local module");
  }
  bool supports_modern_loader(spec) const { return true; }
  object continue_other_loader(spec) {
    throw std::logic_error("unsupported loader");
  }
  object create_module(spec &s) {
    s.module = make(value::type::module);
    return s.module;
  }
  std::string_view spec_name(spec s) { return s.definition->name; }
  void exec_module(spec s, const object &m, body_provider provider) {
    if (s.definition->initialize) {
      s.definition->initialize();
      return;
    }
    for (auto &b : provider.programs)
      if (b.symbol == s.definition->symbol &&
          b.code_sha256 == s.definition->code) {
        activation f{{}, b, provider, m, nullptr};
        f.execute();
        return;
      }
    throw std::logic_error("unbound local module code");
  }
  void set_initializing(spec s, bool flag) { s.module->initializing = flag; }
  void ensure_initialized(std::string_view, const object &) {}
  bool is_initializing(const object &m) { return m->initializing; }
  bool is_none(const object &v) { return v == none; }
  bool same_cached_module(std::string_view name, const object &m) {
    return lookup_module(name) == m;
  }
  template <class F> void with_module_lock(std::string_view, F f) { f(); }
  void wait_for_module(std::string_view) {}
  object raise_halted_import(std::string_view) {
    throw std::logic_error("halted local import");
  }
  void verbose_import(spec) {}
  void audit_import(std::string_view) {}
};
} // namespace isolated
// A standalone compiled entry owns one registry across all of its imports.
// An exception leaves the caller's binding untouched; load_selected removes
// the newly inserted module and rethrows the same exception object.
inline void import_compiled(std::string_view name, std::string_view binding,
                            unsigned minor, void (*initialize)()) {
  static isolated::process domain({});
  const project_module request{name, {}, {}, initialize};
  struct restore_catalog {
    isolated::process &domain;
    std::span<const project_module> previous;
    ~restore_catalog() { domain.catalog = previous; }
  } restore{domain, domain.catalog};
  domain.catalog = std::span(&request, 1);
  auto result = default_absolute(domain, name, minor);
  domain.globals[std::string(binding)] = std::move(result);
}
inline int run_project(std::span<const region> regions,
                       std::span<const project_module> modules) {
  isolated::process entry(modules);
  for (const auto &part : regions) {
    entry.current = &part;
    if (run(entry, part) != outcome::completed || !entry.values.empty())
      throw std::logic_error("incomplete sealed local entry");
  }
  return 0;
}
} // namespace scar::imports
