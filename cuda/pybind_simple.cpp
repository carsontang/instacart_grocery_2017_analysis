#include <pybind11/pybind11.h>

int add(int i, int j) {
    return i + j;
}

namespace py = pybind11;

PYBIND11_PLUGIN(pybind_simple) {
    py::module m("pybind_simple", "pybind11 example plugin");

    m.def("add", &add, "A function which adds two numbers");

    return m.ptr();
}