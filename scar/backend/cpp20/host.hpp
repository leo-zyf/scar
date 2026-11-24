#pragma once

#include "exceptions.hpp"
#include "runtime.hpp"
#include <cerrno>
#include <dirent.h>
#include <fcntl.h>
#include <filesystem>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <signal.h>
#include <spawn.h>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/wait.h>
#include <sys/xattr.h>
#include <time.h>
#include <unistd.h>
#include <vector>

namespace scar::host {
namespace fs = std::filesystem;

[[noreturn]] inline void os_error(const char *operation) {
  throw std::system_error(errno, std::generic_category(), operation);
}

inline const std::string &file_path(const std::string &path) {
  if (path.find('\0') != std::string::npos)
    throw std::invalid_argument("embedded NUL in filesystem path");
  return path;
}

inline std::string path_join(std::initializer_list<std::string> parts) {
  std::string result;
  for (const auto &part : parts) {
    if (!part.empty() && part.front() == '/')
      result = part;
    else {
      if (!result.empty() && result.back() != '/')
        result += '/';
      result += part;
    }
  }
  return result;
}

inline bool fs_exists(const std::string &path) {
  struct stat st;
  return path.find('\0') == std::string::npos && ::stat(path.c_str(), &st) == 0;
}
inline bool fs_is_file(const std::string &path) {
  struct stat st;
  return path.find('\0') == std::string::npos &&
         ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}
inline bool fs_is_dir(const std::string &path) {
  struct stat st;
  return path.find('\0') == std::string::npos &&
         ::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}
inline std::nullptr_t fs_mkdir(const std::string &path) {
  if (::mkdir(file_path(path).c_str(), 0777) != 0)
    os_error("mkdir");
  return nullptr;
}
struct path_option {
  std::string path;
  bool flag;
};
inline std::nullptr_t fs_makedirs(const path_option &request) {
  if (!fs::create_directories(file_path(request.path)) && !request.flag)
    throw std::system_error(EEXIST, std::generic_category(), "mkdir");
  return nullptr;
}
// An ignored failure in one child must not prevent removal of its siblings.
// Descend through opened directory descriptors and never follow a child link.
inline void remove_children_best_effort(int fd) noexcept {
  DIR *directory = ::fdopendir(fd);
  if (!directory) {
    ::close(fd);
    return;
  }
  while (auto *entry = ::readdir(directory)) {
    const char *name = entry->d_name;
    if (std::strcmp(name, ".") == 0 || std::strcmp(name, "..") == 0)
      continue;
    const int child =
        ::openat(fd, name, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (child >= 0) {
      remove_children_best_effort(child);
      ::unlinkat(fd, name, AT_REMOVEDIR);
    } else if (::unlinkat(fd, name, 0) != 0) {
      ::unlinkat(fd, name, AT_REMOVEDIR);
    }
  }
  ::closedir(directory);
}
inline std::nullptr_t fs_remove_tree(const path_option &request) {
  const auto &path = file_path(request.path);
  if (request.flag) {
    const int fd =
        ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (fd >= 0) {
      remove_children_best_effort(fd);
      ::rmdir(path.c_str());
    }
    return nullptr;
  }
  std::error_code error;
  auto status = fs::symlink_status(path, error);
  if (!error && !fs::is_directory(status))
    error = std::make_error_code(fs::exists(status)
                                     ? std::errc::not_a_directory
                                     : std::errc::no_such_file_or_directory);
  if (!error)
    fs::remove_all(path, error);
  if (error)
    throw fs::filesystem_error("remove tree", path, error);
  return nullptr;
}

class descriptor {
  int fd_;

public:
  explicit descriptor(int fd) : fd_(fd) {
    if (fd < 0)
      os_error("open");
  }
  descriptor(const descriptor &) = delete;
  descriptor &operator=(const descriptor &) = delete;
  ~descriptor() {
    if (fd_ >= 0)
      ::close(fd_);
  }
  int get() const { return fd_; }
  int release() { return std::exchange(fd_, -1); }
  void move_above_stdio() {
    if (fd_ <= STDERR_FILENO) {
      const int high = ::fcntl(fd_, F_DUPFD_CLOEXEC, STDERR_FILENO + 1);
      if (high < 0)
        os_error("move pipe descriptor");
      ::close(std::exchange(fd_, high));
    }
  }
  void close() {
    if (::close(release()) != 0)
      os_error("close");
  }
};

inline struct stat status_of(int fd) {
  struct stat st;
  if (::fstat(fd, &st) < 0)
    os_error("fstat");
  return st;
}
inline bool unsupported_xattr(int e) {
  return e == ENOTSUP || e == ENODATA || e == EINVAL;
}
inline void copy_xattrs(int source, int destination) {
  std::vector<char> names;
  for (;;) {
    auto size = ::flistxattr(source, nullptr, 0);
    if (size < 0) {
      if (unsupported_xattr(errno))
        return;
      os_error("list extended attributes");
    }
    if (!size)
      return;
    names.resize(size);
    size = ::flistxattr(source, names.data(), names.size());
    if (size >= 0) {
      names.resize(size);
      break;
    }
    if (errno != ERANGE)
      os_error("list extended attributes");
  }
  for (std::size_t i = 0; i < names.size();) {
    const char *key = names.data() + i;
    i += std::strlen(key) + 1;
    for (;;) {
      auto size = ::fgetxattr(source, key, nullptr, 0);
      if (size >= 0) {
        std::vector<char> value(static_cast<std::size_t>(size) + 1);
        size = ::fgetxattr(source, key, value.data(), value.size());
        if (size < 0 && errno == ERANGE)
          continue;
        if (size >= 0 &&
            ::fsetxattr(destination, key, value.data(), size, 0) == 0)
          break;
      }
      if (unsupported_xattr(errno) || errno == EPERM || errno == EACCES)
        break;
      os_error("copy extended attribute");
    }
  }
}
enum class copy_attributes { content, mode, stat };
inline void copystat(int source, int destination, copy_attributes policy) {
  if (policy == copy_attributes::content)
    return;
  const auto st = status_of(source);
  if (policy == copy_attributes::stat) {
    timespec times[2]{st.st_atim, st.st_mtim};
    if (::futimens(destination, times) < 0)
      os_error("copy timestamps");
    copy_xattrs(source, destination);
  }
  if (::fchmod(destination, st.st_mode & 07777) < 0)
    os_error("copy permissions");
}

struct path_pair {
  std::string source, destination;
};
inline void copy_regular(const fs::path &source, const fs::path &destination,
                         copy_attributes policy) {
  descriptor src(::open(source.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK));
  const auto source_info = status_of(src.get());
  if (!S_ISREG(source_info.st_mode))
    throw std::invalid_argument("copy requires a regular source file");
  descriptor dst(::open(destination.c_str(),
                        O_WRONLY | O_CREAT | O_CLOEXEC | O_NONBLOCK, 0666));
  const auto dest_info = status_of(dst.get());
  if (!S_ISREG(dest_info.st_mode) || (dest_info.st_dev == source_info.st_dev &&
                                      dest_info.st_ino == source_info.st_ino))
    throw std::invalid_argument("copy requires a distinct regular destination");
  if (::ftruncate(dst.get(), 0) < 0)
    os_error("truncate destination");
  std::array<char, 65536> buffer;
  for (;;) {
    auto n = ::read(src.get(), buffer.data(), buffer.size());
    if (n < 0 && errno == EINTR)
      continue;
    if (n < 0)
      os_error("copy read");
    if (!n)
      break;
    for (ssize_t i = 0; i < n;) {
      auto written = ::write(dst.get(), buffer.data() + i, n - i);
      if (written < 0 && errno == EINTR)
        continue;
      if (written < 0)
        os_error("copy write");
      if (!written)
        throw std::ios_base::failure("short file copy");
      i += written;
    }
  }
  copystat(src.get(), dst.get(), policy);
  dst.close();
}
inline std::string copy_to(path_pair paths, copy_attributes policy) {
  file_path(paths.source);
  file_path(paths.destination);
  if (policy != copy_attributes::content && fs_is_dir(paths.destination))
    paths.destination = path_join(
        {paths.destination, fs::path(paths.source).filename().native()});
  copy_regular(paths.source, paths.destination, policy);
  return paths.destination;
}
inline std::string fs_copy_file(path_pair paths) {
  return copy_to(std::move(paths), copy_attributes::content);
}
inline std::string fs_copy(path_pair paths) {
  return copy_to(std::move(paths), copy_attributes::mode);
}
inline std::string fs_copy2(path_pair paths) {
  return copy_to(std::move(paths), copy_attributes::stat);
}

using directory_id = std::pair<dev_t, ino_t>;
inline void copy_directory(const fs::path &source, const fs::path &destination,
                           std::vector<directory_id> &ancestors,
                           directory_id &output) {
  descriptor src(::open(source.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
  auto st = status_of(src.get());
  directory_id id{st.st_dev, st.st_ino};
  if ((!ancestors.empty() && id == output) ||
      std::find(ancestors.begin(), ancestors.end(), id) != ancestors.end())
    throw std::invalid_argument("directory copy traverses a cycle");
  std::vector<fs::path> names;
  for (const auto &entry : fs::directory_iterator(source))
    names.push_back(entry.path().filename());
  if (!fs::create_directories(destination))
    throw std::system_error(EEXIST, std::generic_category(),
                            "copy tree destination");
  descriptor dst(
      ::open(destination.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
  if (ancestors.empty()) {
    auto root = status_of(dst.get());
    output = {root.st_dev, root.st_ino};
  }
  ancestors.push_back(id);
  for (const auto &n : names) {
    if (fs::is_directory(source / n))
      copy_directory(source / n, destination / n, ancestors, output);
    else
      copy_regular(source / n, destination / n, copy_attributes::stat);
  }
  ancestors.pop_back();
  copystat(src.get(), dst.get(), copy_attributes::stat);
}
inline std::string fs_copy_tree(const path_pair &paths) {
  std::vector<directory_id> ancestors;
  directory_id output{};
  copy_directory(file_path(paths.source), file_path(paths.destination),
                 ancestors, output);
  return paths.destination;
}
inline std::nullptr_t process_chdir(const std::string &path) {
  fs::current_path(file_path(path));
  return nullptr;
}

inline unsigned utf8_width(unsigned b) {
  if (b < 0x80)
    return 1;
  if (b >= 0xc2 && b <= 0xdf)
    return 2;
  if (b >= 0xe0 && b <= 0xef)
    return 3;
  if (b >= 0xf0 && b <= 0xf4)
    return 4;
  throw std::invalid_argument("invalid UTF-8 leading byte");
}
inline void utf8_scalar(std::string_view bytes) {
  const auto b = static_cast<unsigned char>(bytes.front());
  const auto width = utf8_width(b);
  if (bytes.size() < width)
    throw std::invalid_argument("truncated UTF-8 scalar");
  for (unsigned i = 1; i < width; ++i) {
    const auto next = static_cast<unsigned char>(bytes[i]);
    if ((next & 0xc0) != 0x80 ||
        (i == 1 && ((b == 0xe0 && next < 0xa0) || (b == 0xed && next >= 0xa0) ||
                    (b == 0xf0 && next < 0x90) || (b == 0xf4 && next >= 0x90))))
      throw std::invalid_argument("invalid UTF-8 continuation");
  }
}
inline std::int64_t text_length(std::string_view bytes) {
  std::int64_t count = 0;
  while (!bytes.empty()) {
    utf8_scalar(bytes);
    bytes.remove_prefix(utf8_width(static_cast<unsigned char>(bytes.front())));
    ++count;
  }
  return count;
}

struct console_request {
  std::string prompt, errors;
};
inline std::string console_input(const console_request &request) {
  if (request.errors != "strict" && request.errors != "surrogateescape")
    throw std::invalid_argument("unsupported console text error policy");
  if (std::fflush(stderr) != 0)
    os_error("flush standard error");
  write_stdout(request.prompt, false);
  flush_stdout();
  auto byte = [] {
    const int c = std::fgetc(stdin);
    if (c == EOF && std::ferror(stdin))
      throw std::ios_base::failure("standard input read failed");
    return c;
  };
  std::string result;
  for (;;) {
    const int c = byte();
    if (c == EOF) {
      if (result.empty())
        throw program_error("eof", "EOF when reading a line");
      break;
    }
    if (c == '\n')
      break;
    if (c == '\r') {
      const int next = byte();
      if (next != EOF && next != '\n' && std::ungetc(next, stdin) == EOF)
        throw std::ios_base::failure("standard input pushback failed");
      break;
    }
    result.push_back(static_cast<char>(c));
  }
  if (request.errors == "strict")
    text_length(result);
  // With surrogateescape, non-UTF-8 bytes retain their original encoding.
  // The binding contract requires matching UTF-8/surrogateescape output.
  return result;
}

[[noreturn]] inline std::nullptr_t process_exit(const error_value &value) {
  raise_error(value);
}

inline std::string process_getcwd() {
  auto cwd = fs::current_path().native();
  text_length(cwd);
  return cwd;
}

class stream_state {
  std::FILE *file_;
  bool reading_, line_buffered_;
  std::FILE *file() const {
    if (!file_)
      throw std::invalid_argument("I/O operation on closed file.");
    return file_;
  }
  int byte() {
    int c = std::fgetc(file_);
    if (c == EOF && std::ferror(file_))
      throw std::ios_base::failure("text stream read failed");
    return c;
  }

public:
  stream_state(std::FILE *f, bool reading)
      : file_(f), reading_(reading),
        line_buffered_(!reading && ::isatty(::fileno(f))) {}
  stream_state(const stream_state &) = delete;
  void require_open() const { static_cast<void>(file()); }
  ~stream_state() {
    if (file_)
      std::fclose(file_);
  }
  void close() {
    if (file_ && std::fclose(std::exchange(file_, nullptr)) != 0)
      os_error("close text stream");
  }
  void flush() {
    auto *f = file();
    if (!reading_ && std::fflush(f) != 0)
      os_error("flush text stream");
  }
  std::int64_t write(std::string_view text) {
    auto *f = file();
    if (reading_)
      throw program_error("unsupported_operation", "not writable");
    auto count = text_length(text);
    if ((!text.empty() &&
         std::fwrite(text.data(), 1, text.size(), f) != text.size()) ||
        std::ferror(f))
      throw std::ios_base::failure("text stream write failed");
    if (line_buffered_ && text.find_first_of("\r\n") != std::string_view::npos)
      flush();
    return count;
  }
  std::string read(std::int64_t count, bool line) {
    file();
    if (!reading_)
      throw program_error("unsupported_operation", "not readable");
    std::string result;
    while (count != 0) {
      int c = byte();
      if (c == EOF)
        break;
      if (c == '\r') {
        const int next = byte();
        if (next != EOF && next != '\n' && std::ungetc(next, file_) == EOF)
          throw std::ios_base::failure("text stream pushback failed");
        c = '\n';
      }
      char scalar[4]{static_cast<char>(c)};
      const auto width = utf8_width(c);
      for (unsigned k = 1; k < width; ++k) {
        int next = byte();
        if (next == EOF)
          throw std::invalid_argument("truncated UTF-8 scalar");
        scalar[k] = static_cast<char>(next);
      }
      utf8_scalar({scalar, width});
      result.append(scalar, width);
      if (line && c == '\n')
        break;
      if (count > 0)
        --count;
    }
    return result;
  }
};
using text_stream = std::shared_ptr<stream_state>;
inline text_stream adopt_text_stream(descriptor &fd, const char *mode) {
  auto *f = ::fdopen(fd.get(), mode);
  if (!f)
    os_error("fdopen");
  fd.release();
  struct close_file {
    void operator()(std::FILE *file) const { std::fclose(file); }
  };
  std::unique_ptr<std::FILE, close_file> owner(f);
  auto s = std::make_shared<stream_state>(f, mode[0] == 'r');
  owner.release();
  return s;
}

inline stream_state &stream(const text_stream &s) {
  if (!s)
    throw std::ios_base::failure("unbound text stream");
  return *s;
}
inline text_stream stream_enter(const text_stream &value) {
  stream(value).require_open();
  return value;
}
struct stream_options {
  std::string path, mode, encoding, newline;
};
inline text_stream stream_open(const stream_options &o) {
  if (o.encoding != "utf-8" || o.newline != "universal")
    throw std::invalid_argument(
        "unsupported text stream encoding or newline policy");
  const auto &m = o.mode;
  if (m.empty() || (m.size() != 1 && !(m.size() == 2 && m[1] == 't')))
    throw std::invalid_argument("unsupported text stream mode");
  int flags = O_CLOEXEC;
  const char *mode;
  switch (m[0]) {
  case 'r':
    flags |= O_RDONLY;
    mode = "r";
    break;
  case 'a':
    flags |= O_CREAT | O_WRONLY | O_APPEND;
    mode = "a";
    break;
  case 'w':
    flags |= O_CREAT | O_WRONLY | O_TRUNC;
    mode = "w";
    break;
  case 'x':
    flags |= O_CREAT | O_WRONLY | O_EXCL;
    mode = "w";
    break;
  default:
    throw std::invalid_argument("unsupported text stream mode");
  }
  descriptor fd(::open(file_path(o.path).c_str(), flags, 0666));
  if (S_ISDIR(status_of(fd.get()).st_mode))
    throw std::system_error(EISDIR, std::generic_category(),
                            "open text stream");
  return adopt_text_stream(fd, mode);
}
struct stream_data {
  text_stream handle;
  std::string text;
};
struct stream_count {
  text_stream handle;
  std::int64_t count;
};
inline std::int64_t stream_write(const stream_data &r) {
  return stream(r.handle).write(r.text);
}
inline std::string stream_read(const stream_count &r) {
  return stream(r.handle).read(r.count, false);
}
inline std::string stream_readline(const stream_count &r) {
  return stream(r.handle).read(r.count, true);
}
inline std::nullptr_t stream_flush(const text_stream &s) {
  stream(s).flush();
  return nullptr;
}
inline std::nullptr_t stream_close(const text_stream &s) {
  stream(s).close();
  return nullptr;
}

// An explicit wait owns completion. Released live children are checked again
// on a later launch without changing unrelated children's wait status.
class unclaimed_children {
  std::list<pid_t> children_;
  std::mutex lock_;

  static bool finished(pid_t child) noexcept {
    int status;
    const pid_t result = ::waitpid(child, &status, WNOHANG);
    return (result == child && (WIFEXITED(status) || WIFSIGNALED(status))) ||
           (result < 0 && errno == ECHILD);
  }

public:
  void collect() {
    std::lock_guard guard(lock_);
    children_.remove_if(finished);
  }
  void adopt(std::list<pid_t> &ticket) {
    if (!finished(ticket.front())) {
      std::lock_guard guard(lock_);
      children_.splice(children_.end(), ticket);
    }
  }
};

class process_state {
  std::shared_ptr<unclaimed_children> registry_;
  std::list<pid_t> child_{-1};
  std::optional<std::int64_t> status_;
  std::mutex wait_lock_;

public:
  text_stream output;
  explicit process_state(std::shared_ptr<unclaimed_children> registry)
      : registry_(std::move(registry)) {}
  void started(pid_t pid) noexcept { child_.front() = pid; }
  ~process_state() {
    if (child_.front() > 0)
      registry_->adopt(child_);
  }
  std::int64_t wait() {
    std::lock_guard guard(wait_lock_);
    while (!status_) {
      if (child_.front() <= 0)
        throw std::invalid_argument("process has no waitable child");
      int status;
      if (::waitpid(child_.front(), &status, 0) < 0) {
        if (errno == EINTR)
          continue;
        if (errno == ECHILD)
          child_.front() = -1;
        os_error("waitpid");
      }
      if (WIFEXITED(status))
        status_ = WEXITSTATUS(status);
      else if (WIFSIGNALED(status))
        status_ = -WTERMSIG(status);
    }
    child_.front() = -1;
    return *status_;
  }
};
using process_handle = std::shared_ptr<process_state>;

inline void spawn_check(int error, const char *operation) {
  if (error != 0)
    throw std::system_error(error, std::generic_category(), operation);
}
struct spawn_actions {
  posix_spawn_file_actions_t value;
  spawn_actions() {
    spawn_check(::posix_spawn_file_actions_init(&value),
                "initialize child file actions");
  }
  spawn_actions(const spawn_actions &) = delete;
  ~spawn_actions() { ::posix_spawn_file_actions_destroy(&value); }
};
struct spawn_attributes {
  posix_spawnattr_t value;
  spawn_attributes() {
    spawn_check(::posix_spawnattr_init(&value),
                "initialize process attributes");
  }
  spawn_attributes(const spawn_attributes &) = delete;
  ~spawn_attributes() { ::posix_spawnattr_destroy(&value); }
};
struct spawn_request {
  std::string output_image;
  std::vector<std::string> argv;
  bool pipe_stdout, stderr_to_stdout;
};
inline process_handle process_spawn(spawn_request request) {
  if (request.argv.empty())
    throw std::invalid_argument("empty child argument vector");
  const fs::path output(file_path(request.output_image));
  if (output.empty() || output.has_parent_path() || output == "." ||
      output == "..")
    throw std::invalid_argument("invalid executable image name");
  const auto image = fs::read_symlink("/proc/self/exe").parent_path() / output;
  if (fs::is_symlink(fs::symlink_status(image)))
    throw std::invalid_argument("executable image must be a published file");
  std::vector<char *> argv;
  argv.reserve(request.argv.size() + 1);
  for (auto &arg : request.argv) {
    if (arg.find('\0') != std::string::npos)
      throw std::invalid_argument("embedded NUL in process argument");
    argv.push_back(arg.data());
  }
  argv.push_back(nullptr);
  static const auto registry = std::make_shared<unclaimed_children>();
  registry->collect();
  auto child = std::make_shared<process_state>(registry);
  spawn_actions actions;
  spawn_attributes attrs;
  sigset_t defaults;
  ::sigemptyset(&defaults);
  ::sigaddset(&defaults, SIGPIPE);
#ifdef SIGXFSZ
  ::sigaddset(&defaults, SIGXFSZ);
#endif
#ifdef SIGXFZ
  ::sigaddset(&defaults, SIGXFZ);
#endif
  spawn_check(::posix_spawnattr_setsigdefault(&attrs.value, &defaults),
              "set child default signals");
  spawn_check(::posix_spawnattr_setflags(&attrs.value, POSIX_SPAWN_SETSIGDEF),
              "set child signal policy");
  std::optional<descriptor> writer;
  if (request.pipe_stdout) {
    int pipe[2];
    if (::pipe2(pipe, O_CLOEXEC) != 0)
      os_error("create child output pipe");
    descriptor reader(pipe[0]);
    writer.emplace(pipe[1]);
    reader.move_above_stdio();
    writer->move_above_stdio();
    child->output = adopt_text_stream(reader, "r");
    spawn_check(::posix_spawn_file_actions_adddup2(
                    &actions.value, writer->get(), STDOUT_FILENO),
                "route child stdout");
  }
  if (request.stderr_to_stdout)
    spawn_check(::posix_spawn_file_actions_adddup2(
                    &actions.value, STDOUT_FILENO, STDERR_FILENO),
                "route child stderr");
#if defined(__GLIBC__) &&                                                      \
    (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 34))
  spawn_check(::posix_spawn_file_actions_addclosefrom_np(&actions.value,
                                                         STDERR_FILENO + 1),
              "close child auxiliary descriptors");
#else
  throw std::system_error(ENOTSUP, std::generic_category(),
                          "close child auxiliary descriptors");
#endif
  pid_t pid;
  spawn_check(::posix_spawn(&pid, image.c_str(), &actions.value, &attrs.value,
                            argv.data(), ::environ),
              "launch executable image");
  child->started(pid);
  return child;
}

inline text_stream process_stdout(const process_handle &child) {
  if (!child || !child->output)
    throw std::invalid_argument("process has no stdout pipe");
  return child->output;
}
inline std::int64_t process_wait(const process_handle &child) {
  if (!child)
    throw std::invalid_argument("unbound process handle");
  return child->wait();
}

inline double time_wall() {
  timespec ts;
  if (::clock_gettime(CLOCK_REALTIME, &ts) < 0)
    os_error("clock_gettime");
  return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) / 1e9;
}
inline std::nullptr_t time_sleep(double seconds) {
  const auto ns = std::ceil(static_cast<long double>(seconds) * 1000000000.L);
  if (!std::isfinite(ns) || seconds < 0 ||
      ns > std::numeric_limits<std::int64_t>::max())
    throw std::invalid_argument("invalid sleep duration");
  const auto n = static_cast<std::int64_t>(ns);
  timespec ts{static_cast<time_t>(n / 1000000000),
              static_cast<long>(n % 1000000000)};
  while (::nanosleep(&ts, &ts) < 0)
    if (errno != EINTR)
      os_error("nanosleep");
  return nullptr;
}

using scar::text_role;
using printable = std::variant<std::string, bool, std::nullptr_t, std::int64_t,
                               double, text_stream>;
struct text_operand {
  text_role role;
  printable value;
};
struct text_output {
  std::initializer_list<text_operand> operands;
  scar::text_format format;
};
inline std::nullptr_t text_print(const text_output &out) {
  for (const auto &operand : out.operands)
    if (operand.role == text_role::stream) {
      auto &sink = stream(std::get<text_stream>(operand.value));
      return scar::print_sequence(
          out, [&](std::string_view s) { sink.write(s); },
          [&] { sink.flush(); });
    }
  throw std::invalid_argument("missing print stream operand");
}
} // namespace scar::host
