#pragma once

#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace scar {

class program_error : public std::exception {
  std::string category_, message_;

public:
  program_error(std::string category, std::string message)
      : category_(std::move(category)), message_(std::move(message)) {}
  const char *what() const noexcept override { return message_.c_str(); }
  const std::string &category() const noexcept { return category_; }
  const std::string &message() const noexcept { return message_; }
};

class exit_signal : public std::exception {
  std::string message_;
  std::optional<std::string> exit_message_;

public:
  const int status;
  explicit exit_signal(std::int64_t code, std::string message = {},
                       std::optional<std::string> exit_message = std::nullopt)
      : message_(std::move(message)), exit_message_(std::move(exit_message)),
        status(static_cast<unsigned char>(code)) {}
  const char *what() const noexcept override { return message_.c_str(); }
  const std::string &message() const noexcept { return message_; }
  const std::optional<std::string> &exit_message() const noexcept {
    return exit_message_;
  }
};

struct error_record {
  std::exception_ptr exception;
  std::string category, message;
};
using error_value = std::shared_ptr<const error_record>;

inline error_value
capture_error(std::exception_ptr ptr = std::current_exception()) {
  if (!ptr)
    throw std::logic_error("no exception is active");
  try {
    std::rethrow_exception(ptr);
  } catch (const program_error &error) {
    return std::make_shared<error_record>(ptr, error.category(),
                                          error.message());
  } catch (const exit_signal &error) {
    return std::make_shared<error_record>(ptr, "system_exit", error.message());
  } catch (const std::out_of_range &error) {
    return std::make_shared<error_record>(ptr, "index_error", error.what());
  } catch (const std::invalid_argument &error) {
    return std::make_shared<error_record>(ptr, "value_error", error.what());
  } catch (const std::domain_error &error) {
    return std::make_shared<error_record>(ptr, "value_error", error.what());
  } catch (const std::overflow_error &error) {
    return std::make_shared<error_record>(ptr, "overflow_error", error.what());
  } catch (const std::system_error &error) {
    return std::make_shared<error_record>(ptr, "os_error", error.what());
  } catch (const std::bad_alloc &error) {
    return std::make_shared<error_record>(ptr, "memory_error", error.what());
  } catch (const std::exception &error) {
    return std::make_shared<error_record>(ptr, "runtime_error", error.what());
  } catch (...) {
    // An unclassified native exception may propagate, but is not silently
    // qualified as a source-language exception or caught by a narrower handler.
    return std::make_shared<error_record>(ptr, "foreign_exception",
                                          "native exception");
  }
}

inline const error_record &error(const error_value &value) {
  if (!value)
    throw std::logic_error("unbound exception value");
  return *value;
}

inline bool matches(const error_value &value, std::string_view category) {
  const auto &actual = error(value).category;
  if (category == actual || category == "base_exception")
    return true;
  if (category == "exception")
    return actual != "system_exit" && actual != "foreign_exception";
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

inline error_value make_error(std::string category, std::string message) {
  auto ptr = std::make_exception_ptr(
      program_error(std::move(category), std::move(message)));
  return capture_error(ptr);
}

inline error_value make_exit_error(std::int64_t status, std::string message,
                                   std::optional<std::string> exit_message) {
  return capture_error(std::make_exception_ptr(
      exit_signal(status, std::move(message), std::move(exit_message))));
}

[[noreturn]] inline void raise_error(const error_value &value) {
  std::rethrow_exception(error(value).exception);
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
