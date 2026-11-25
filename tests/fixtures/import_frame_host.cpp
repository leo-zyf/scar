#include "regions.hpp"
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>
using namespace scar::imports;
struct value {
  std::string text;
  std::shared_ptr<value> member;
};
using handle = std::shared_ptr<value>;
using slot = std::optional<handle>;
using text = std::string;
handle object_value(text label) {
  return std::make_shared<value>(value{std::move(label), {}});
}

struct frame : ordered_registry<handle> {
  text mode, code, source_value, module_name;
  int audits = 0, executions = 0;
  bool initialized = false, stored = false;
  std::vector<slot> values;
  std::map<unsigned, handle> globals;
  std::map<unsigned, text> names;
  handle none = object_value("None"),
         replacement = object_value("replacement module");
  frame(char **args) : mode(args[1]), code(args[2]), source_value(args[3]) {
    replacement->member = object_value(args[4]);
    for (auto ins : generated::regions[0].instructions) {
      if (!ins.name.empty())
        names[ins.operand] = ins.name;
      if (ins.operation == opcode::import_name)
        module_name = ins.name;
    }
    if (mode == "hit")
      publish_module(module_name, replacement);
  }
  bool is_none(const handle &v) { return v == none; }
  handle raise_halted_import(std::string_view) {
    throw std::runtime_error("blocked");
  }
  void ensure_initialized(std::string_view, const handle &) {}
  bool is_initializing(const handle &) { return initialized; }
  void wait_for_module(std::string_view) {}
  bool same_cached_module(std::string_view name, const handle &module) {
    return lookup_module(name) == module;
  }
  bool supports_modern_loader(const text &) { return true; }
  handle continue_other_loader(const text &) {
    throw std::logic_error("unbound loader");
  }
  const text &spec_name(const text &spec) { return spec; }
  void verbose_import(const text &) {}
  void audit_import(std::string_view name) {
    ++audits;
    if (mode == "deny")
      throw std::runtime_error("denied");
    if (mode == "audit_insert")
      publish_module(name, replacement);
  }
  template <class F> void with_module_lock(std::string_view, F run) { run(); }
  text find_spec(std::string_view name) { return text(name); }
  handle create_module(const text &) { return object_value("original module"); }
  void set_initializing(const text &, bool state) { initialized = state; }
  void exec_module(const text &name, const handle &module) {
    ++executions;
    if (!initialized || lookup_module(name) != module)
      std::abort();
    module->member = object_value(source_value);
    if (mode == "recursive" &&
        default_absolute(*this, name, generated::regions[0].minor) != module)
      std::abort();
    if (mode == "replace" || mode == "failure")
      publish_module(name, replacement);
    if (mode == "failure")
      throw std::runtime_error("exec failure");
  }
  auto &stack() { return values; }
  bool enter(const region &r) {
    return mode != "mismatch" && code == r.code_sha256 && values.empty();
  }
  bool before(unsigned) { return true; }
  handle constant(unsigned i) {
    return i == generated::regions[0].instructions[1].operand
               ? none
               : object_value("0");
  }
  handle small_int(unsigned i) { return object_value(std::to_string(i)); }
  template <class F>
  handle import_name(unsigned i, const handle &level, const handle &from,
                     F original) {
    if (names.at(i) != module_name || level->text != "0" || from != none)
      std::abort();
    return mode == "hook" ? replacement : original(*this);
  }
  void replace(size_t n, slot next) {
    while (n--)
      values.pop_back();
    if (next)
      values.push_back(std::move(next));
  }
  void publish(unsigned i, bool, const handle &v) {
    globals[i] = v;
    stored = true;
  }
  handle load_name(unsigned i) {
    return names.at(i) == "print" ? object_value("print") : globals.at(i);
  }
  handle attribute(const handle &base, unsigned) { return base->member; }
  handle invoke(const handle &callable, std::span<slot> args) {
    if (callable->text == "print") {
      std::cout << (*args.front())->text << '\n';
      return none;
    }
    return object_value(callable->text);
  }
  void unwind(unsigned, std::exception_ptr error) {
    std::rethrow_exception(error);
  }
};
int main(int argc, char **argv) {
  if (argc != 5)
    return 3;
  frame f(argv);
  int status = 0;
  try {
    if (run(f, generated::regions[0]) != outcome::completed)
      status = 2;
  } catch (const std::exception &) {
    status = 1;
  }
  if (f.initialized)
    return 3;
  std::cerr << f.audits << " " << f.executions << " " << f.stored << " "
            << f.lookup_module(f.module_name).has_value() << '\n';
  return status;
}
