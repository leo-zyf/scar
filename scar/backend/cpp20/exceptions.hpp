#pragma once

#include <cstdint>
#include <cstdio>
#include <exception>
#include <ios>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace scar {

struct error_record;
using error_value = std::shared_ptr<const error_record>;

struct exit_payload {
  int status;
  std::optional<std::string> display;
};

struct error_record {
  std::string category, message;
  std::optional<exit_payload> exit;
  // Only originally foreign exceptions need an exception_ptr. Managed signals
  // already own their record; storing that signal back here would form a cycle.
  std::exception_ptr native;
  void (*raise_with_identity)(const error_value &) = nullptr;
  mutable std::mutex cause_mutex;
  mutable error_value cause;
  mutable bool self_cause = false, suppress_context = false;

  error_record(std::string kind, std::string text)
      : category(std::move(kind)), message(std::move(text)) {}
};

inline const error_record &error(const error_value &value) {
  if (!value)
    throw std::logic_error("unbound exception value");
  return *value;
}

inline error_value make_error(std::string category, std::string message) {
  return std::make_shared<error_record>(std::move(category),
                                        std::move(message));
}

inline error_value make_exit_error(std::int64_t status, std::string message,
                                   std::optional<std::string> exit_message) {
  auto value =
      std::make_shared<error_record>("system_exit", std::move(message));
  value->exit.emplace(static_cast<unsigned char>(status),
                      std::move(exit_message));
  return value;
}

// A native throw owns the same handle that exception bindings and causes see.
struct error_carrier {
  error_value value;
  explicit error_carrier(error_value handle) : value(std::move(handle)) {}
};

class program_error : public std::exception, public error_carrier {
public:
  explicit program_error(error_value value) : error_carrier(std::move(value)) {}
  program_error(std::string category, std::string message)
      : program_error(make_error(std::move(category), std::move(message))) {}
  const char *what() const noexcept override { return value->message.c_str(); }
  const std::string &category() const noexcept { return value->category; }
  const std::string &message() const noexcept { return value->message; }
};

class exit_signal : public std::exception, public error_carrier {
public:
  const int status;
  explicit exit_signal(error_value handle)
      : error_carrier(std::move(handle)), status(value->exit->status) {}
  explicit exit_signal(std::int64_t code, std::string message = {},
                       std::optional<std::string> exit_message = std::nullopt)
      : exit_signal(make_exit_error(code, std::move(message),
                                    std::move(exit_message))) {}
  const char *what() const noexcept override { return value->message.c_str(); }
  const std::string &message() const noexcept { return value->message; }
  const std::optional<std::string> &exit_message() const noexcept {
    return value->exit->display;
  }
};

namespace exception_detail {
template <class Native> struct attributed_native : Native, error_carrier {
  attributed_native(const Native &original, error_value handle)
      : Native(original), error_carrier(std::move(handle)) {}
  const char *what() const noexcept override { return value->message.c_str(); }
};

template <class Native>
error_value capture_native(std::exception_ptr ptr, const Native &original,
                           std::string category) {
  auto value =
      std::make_shared<error_record>(std::move(category), original.what());
  value->native = std::move(ptr);
  value->raise_with_identity = [](const error_value &handle) {
    try {
      std::rethrow_exception(handle->native);
    } catch (const Native &caught) {
      throw attributed_native<Native>(caught, handle);
    }
  };
  return value;
}
} // namespace exception_detail

inline error_value
capture_error(std::exception_ptr ptr = std::current_exception()) {
  if (!ptr)
    throw std::logic_error("no exception is active");
  using exception_detail::capture_native;
  try {
    std::rethrow_exception(ptr);
  } catch (const error_carrier &caught) {
    return caught.value;
  } catch (const std::out_of_range &caught) {
    return capture_native(ptr, caught, "index_error");
  } catch (const std::invalid_argument &caught) {
    return capture_native(ptr, caught, "value_error");
  } catch (const std::domain_error &caught) {
    return capture_native(ptr, caught, "value_error");
  } catch (const std::overflow_error &caught) {
    return capture_native(ptr, caught, "overflow_error");
  } catch (const std::ios_base::failure &caught) {
    return capture_native(ptr, caught, "os_error");
  } catch (const std::system_error &caught) {
    return capture_native(ptr, caught, "os_error");
  } catch (const std::bad_alloc &caught) {
    return capture_native(ptr, caught, "memory_error");
  } catch (const std::exception &caught) {
    return capture_native(ptr, caught, "runtime_error");
  } catch (...) {
    auto value =
        std::make_shared<error_record>("foreign_exception", "native exception");
    value->native = std::move(ptr);
    return value;
  }
}

inline bool matches(const error_value &value, std::string_view category) {
  const auto &actual = error(value).category;
  if (category == actual || category == "base_exception")
    return true;
  if (category == "exception")
    return actual != "system_exit" && actual != "foreign_exception";
  if (category == "import_error")
    return actual == "module_not_found_error";
  if (category == "lookup_error")
    return actual == "index_error" || actual == "key_error";
  if (category == "arithmetic_error")
    return actual == "overflow_error" || actual == "zero_division_error";
  if (actual == "unicode_decode_error")
    return category == "unicode_error" || category == "value_error";
  if (actual == "unsupported_operation")
    return category == "os_error" || category == "value_error";
  if (actual == "unicode_error")
    return category == "value_error";
  return false;
}

inline std::string error_message(const error_value &value) {
  return error(value).message;
}

inline error_value error_cause(const error_value &value) {
  const auto &record = error(value);
  const std::lock_guard lock(record.cause_mutex);
  return record.self_cause ? value : record.cause;
}

inline bool error_suppresses_context(const error_value &value) {
  const auto &record = error(value);
  const std::lock_guard lock(record.cause_mutex);
  return record.suppress_context;
}

[[noreturn]] inline void raise_error(const error_value &value) {
  const auto &record = error(value);
  if (record.exit)
    throw exit_signal(value);
  if (record.native) {
    if (!error_suppresses_context(value))
      std::rethrow_exception(record.native);
    if (record.raise_with_identity)
      record.raise_with_identity(value);
  }
  throw program_error(value);
}

struct explicit_raise {
  error_value value, cause;
  explicit_raise(error_value handle, error_value reason)
      : value(std::move(handle)), cause(std::move(reason)) {
    error(cause);
  }
  explicit_raise(error_value handle, std::nullptr_t)
      : value(std::move(handle)) {}
};

[[noreturn]] inline void raise_error_from(explicit_raise request) {
  const auto &record = error(request.value);
  {
    const std::lock_guard lock(record.cause_mutex);
    record.self_cause = request.value == request.cause;
    record.cause = record.self_cause ? nullptr : std::move(request.cause);
    record.suppress_context = true;
  }
  raise_error(request.value);
}

[[noreturn]] inline void safe_rethrow() {
  if (auto current = std::current_exception())
    std::rethrow_exception(current);
  throw program_error("runtime_error", "No active exception to reraise");
}

inline int report_exit(const exit_signal &signal) noexcept {
  bool failed = false;
  if (const auto &message = signal.exit_message()) {
    failed = std::fwrite(message->data(), 1, message->size(), stderr) !=
             message->size();
    failed = std::fputc('\n', stderr) == EOF || failed;
  }
  failed = std::fflush(stdout) != 0 || failed;
  failed = std::fflush(stderr) != 0 || failed;
  return failed ? 120 : signal.status;
}

} // namespace scar
