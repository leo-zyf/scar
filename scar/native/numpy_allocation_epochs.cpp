#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_22_API_VERSION
#define NPY_TARGET_VERSION NPY_1_22_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <unistd.h>

namespace {
constexpr const char* capsule_name = "mem_handler";
constexpr const char* handler_name = "scar_numpy_allocation_epochs";

struct State {
    PyDataMem_Handler handler{};
    PyDataMemAllocator original{};
    PyObject* original_capsule;
    const pid_t process = getpid();
    const size_t max_live;
    std::mutex mutex;
    std::unordered_map<uintptr_t, uint64_t> active;
    uint64_t cursor = 0;
    bool complete = true;
    bool collecting = true;

    State(PyObject* capsule, PyDataMemAllocator allocator, size_t limit)
        : original(allocator), original_capsule(capsule), max_live(limit) {}

    void observe(uintptr_t drop, uintptr_t add) noexcept {
        if (getpid() != process) return;
        std::lock_guard guard(mutex);
        observe_locked(drop, add);
    }

    void observe_locked(uintptr_t drop, uintptr_t add) noexcept {
        if (!collecting || !complete) return;
        if (cursor == UINT64_MAX || (drop && active.erase(drop) != 1)) {
            complete = false;
            return;
        }
        ++cursor;
        if (!add) return;
        try {
            if (active.size() == max_live || !active.emplace(add, cursor).second) complete = false;
        } catch (...) {
            complete = false;
        }
    }
};

void* allocate(void* ctx, size_t size) noexcept {
    auto& state = *static_cast<State*>(ctx);
    void* p = state.original.malloc(state.original.ctx, size);
    const int saved_errno = errno;
    if (p) state.observe(0, reinterpret_cast<uintptr_t>(p));
    errno = saved_errno;
    return p;
}

void* allocate_zeroed(void* ctx, size_t count, size_t size) noexcept {
    auto& state = *static_cast<State*>(ctx);
    void* p = state.original.calloc(state.original.ctx, count, size);
    const int saved_errno = errno;
    if (p) state.observe(0, reinterpret_cast<uintptr_t>(p));
    errno = saved_errno;
    return p;
}

void* reallocate(void* ctx, void* p, size_t size) noexcept {
    auto& state = *static_cast<State*>(ctx);
    if (getpid() != state.process) return state.original.realloc(state.original.ctx, p, size);
    const auto old = reinterpret_cast<uintptr_t>(p);
    std::lock_guard guard(state.mutex);
    void* result = state.original.realloc(state.original.ctx, p, size);
    const int saved_errno = errno;
    if (result || size == 0) state.observe_locked(old, reinterpret_cast<uintptr_t>(result));
    errno = saved_errno;
    return result;
}

void release(void* ctx, void* p, size_t size) noexcept {
    auto& state = *static_cast<State*>(ctx);
    const int saved_errno = errno;
    if (p) state.observe(reinterpret_cast<uintptr_t>(p), 0);
    errno = saved_errno;
    state.original.free(state.original.ctx, p, size);
}

State* owned_state(PyObject* capsule) {
    auto* handler = static_cast<PyDataMem_Handler*>(PyCapsule_GetPointer(capsule, capsule_name));
    if (!handler) return nullptr;
    if (handler->allocator.malloc != allocate) {
        PyErr_SetString(PyExc_ValueError, "expected a SCAR allocation tracker");
        return nullptr;
    }
    return static_cast<State*>(handler->allocator.ctx);
}

void destroy(PyObject* capsule) {
    auto* state = owned_state(capsule);
    if (!state) {
        PyErr_Clear();
        return;
    }
    // The child may inherit the table while a different parent thread was
    // changing it. Only the owning process destroys tracker internals.
    if (getpid() != state->process) return;
    Py_DECREF(state->original_capsule);
    delete state;
}

PyObject* start(PyObject*, PyObject* args) {
    Py_ssize_t limit;
    if (!PyArg_ParseTuple(args, "n", &limit)) return nullptr;
    if (limit < 1 || limit > 262144) {
        PyErr_SetString(PyExc_ValueError, "expected an allocation table limit from 1 to 262144");
        return nullptr;
    }
    PyObject* previous = PyDataMem_GetHandler();
    if (!previous) return nullptr;
    const auto* original = static_cast<const PyDataMem_Handler*>(PyCapsule_GetPointer(previous, capsule_name));
    if (!original || original->version != 1 || previous != PyDataMem_DefaultHandler) {
        Py_DECREF(previous);
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "original NumPy default allocation handler required");
        return nullptr;
    }
    State* state;
    try {
        state = new State(previous, original->allocator, static_cast<size_t>(limit));
    } catch (...) {
        Py_DECREF(previous);
        return PyErr_NoMemory();
    }
    std::strcpy(state->handler.name, handler_name);
    state->handler.version = 1;
    state->handler.allocator = {state, allocate, allocate_zeroed, reallocate, release};
    PyObject* capsule = PyCapsule_New(&state->handler, capsule_name, destroy);
    if (!capsule) {
        Py_DECREF(previous);
        delete state;
        return nullptr;
    }
    PyObject* replaced = PyDataMem_SetHandler(capsule);
    if (!replaced) {
        Py_DECREF(capsule);
        return nullptr;
    }
    Py_DECREF(replaced);
    return capsule;
}

PyObject* lookup(PyObject*, PyObject* args) {
    PyObject* capsule;
    PyObject* object;
    if (!PyArg_ParseTuple(args, "OO", &capsule, &object)) return nullptr;
    auto* state = owned_state(capsule);
    if (!state) return nullptr;
    if (getpid() != state->process || !PyArray_CheckExact(object)) Py_RETURN_NONE;
    auto* array = reinterpret_cast<PyArrayObject*>(object);
    if (!PyArray_CHKFLAGS(array, NPY_ARRAY_OWNDATA) || PyArray_BASE(array)
        || PyArray_HANDLER(array) != capsule) Py_RETURN_NONE;
    uint64_t epoch;
    uint64_t cursor;
    {
        std::lock_guard guard(state->mutex);
        auto iterator = state->active.find(reinterpret_cast<uintptr_t>(PyArray_DATA(array)));
        if (!state->collecting || !state->complete || iterator == state->active.end()) Py_RETURN_NONE;
        epoch = iterator->second;
        cursor = state->cursor;
    }
    return Py_BuildValue("KK", static_cast<unsigned long long>(epoch), static_cast<unsigned long long>(cursor));
}

PyObject* stop(PyObject*, PyObject* capsule) {
    auto* state = owned_state(capsule);
    if (!state) return nullptr;
    if (getpid() != state->process) Py_RETURN_NONE;
    {
        std::lock_guard guard(state->mutex);
        state->collecting = false;
        state->active.clear();
    }
    PyObject* current = PyDataMem_GetHandler();
    if (!current) return nullptr;
    if (current == capsule) {
        PyObject* replaced = PyDataMem_SetHandler(state->original_capsule);
        Py_DECREF(current);
        if (!replaced) return nullptr;
        Py_DECREF(replaced);
    } else {
        Py_DECREF(current);
    }
    Py_RETURN_NONE;
}

PyMethodDef methods[] = {
    {"start", start, METH_VARARGS, nullptr},
    {"lookup", lookup, METH_VARARGS, nullptr},
    {"stop", stop, METH_O, nullptr},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {PyModuleDef_HEAD_INIT, "_scar_numpy_allocation_epochs", nullptr, -1, methods};
}

PyMODINIT_FUNC PyInit__scar_numpy_allocation_epochs() {
    import_array();
    return PyModule_Create(&module);
}
