#include <Python.h>

#include <array>
#include <cstdint>
#include <limits>

namespace {
int watcher = -1;
std::uint64_t stamp = 0;
std::array<PyObject*, 64> watched{};
std::size_t active = 0;

int changed(PyDict_WatchEvent, PyObject*, PyObject*, PyObject*) noexcept {
    if (stamp != std::numeric_limits<std::uint64_t>::max()) {
        ++stamp;
    }
    return 0;
}

int watch(PyObject* dictionary) {
    for (std::size_t i = 0; i < active; ++i) {
        if (watched[i] == dictionary) {
            return 0;
        }
    }
    if (watcher < 0 || active == watched.size() || !PyDict_CheckExact(dictionary)) {
        return 1;
    }
    if (PyDict_Watch(watcher, dictionary) < 0) {
        return -1;
    }
    watched[active++] = Py_NewRef(dictionary);
    return 0;
}
}

#define SCAR_PUBLIC extern "C" __attribute__((visibility("default")))

SCAR_PUBLIC int scar_study_epoch_open_v1() {
    if (watcher >= 0) {
        return 1;
    }
    watcher = PyDict_AddWatcher(changed);
    stamp = 1;
    return watcher < 0 ? -1 : 0;
}

SCAR_PUBLIC int scar_study_epoch_watch_dict_v1(PyObject* dictionary) {
    return watch(dictionary);
}

SCAR_PUBLIC int scar_study_epoch_watch_type_v1(PyObject* type) {
    if (!PyType_CheckExact(type)) {
        return 1;
    }
    PyObject* namespace_dict = PyType_GetDict(reinterpret_cast<PyTypeObject*>(type));
    if (!namespace_dict) {
        return -1;
    }
    const int result = watch(namespace_dict);
    Py_DECREF(namespace_dict);
    return result;
}

SCAR_PUBLIC std::uint64_t scar_study_epoch_value_v1() {
    return watcher >= 0 ? stamp : 0;
}

SCAR_PUBLIC int scar_study_epoch_close_v1() {
    if (watcher < 0) {
        return 1;
    }
    for (std::size_t i = 0; i < active; ++i) {
        if (PyDict_Unwatch(watcher, watched[i]) < 0) {
            return -1;
        }
        Py_CLEAR(watched[i]);
    }
    active = 0;
    const int result = PyDict_ClearWatcher(watcher);
    watcher = -1;
    return result;
}
