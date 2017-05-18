#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "pybind_simple.h"

pybind11::array_t<int> add_arrays(pybind11::array_t<int> arr1, pybind11::array_t<int> arr2) {
    auto buf1 = arr1.request(),
         buf2 = arr2.request();

    auto result = pybind11::array_t<int>(buf1.size);
    auto buf3 = result.request();

    int *ptr1 = (int *) buf1.ptr,
        *ptr2 = (int *) buf2.ptr,
        *ptr3 = (int *) buf3.ptr;

    parallel_add(ptr1, ptr2, ptr3, buf1.size);

    return result;
}

// TODO (ctang): Add more arguments to CSRMV
pybind11::array_t<int> csrmv(
    pybind11::array_t<int> arr1,
    pybind11::array_t<int> arr2

) {
    auto buf1 = arr1.request(),
         buf2 = arr2.request();

    auto result = pybind11::array_t<int>(buf1.size);
    auto buf3 = result.request();

    int *ptr1 = (int *) buf1.ptr,
        *ptr2 = (int *) buf2.ptr,
        *ptr3 = (int *) buf3.ptr;

    parallel_add(ptr1, ptr2, ptr3, buf1.size);

    return result;
}

PYBIND11_PLUGIN(pybind_simple) {
    pybind11::module m("pybind_simple", "pybind11 example plugin");

    m.def("add", &add, "A function which adds two numbers");
    m.def("add_arrays", &add_arrays, "A function that adds two Numpy arrays");

    return m.ptr();
}