#include <pybind11/pybind11.h>
#include "pybind_simple.h"

PYBIND11_PLUGIN(pybind_simple) {
    pybind11::module m("pybind_simple", "pybind11 example plugin");

    m.def("add", &add, "A function which adds two numbers");

    return m.ptr();
}