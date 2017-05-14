#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>

#include "pybind_gpu.h"

pybind11::array_t<int> add(pybind11::array_t<int> arr1, pybind11::array_t<int> arr2) {
    auto buf1 = arr1.request(),
         buf2 = arr2.request();

    auto result = py::array_t<int>(buf1.size);
    auto buf3 = result.request();

    int *ptr1 = (int *) buf1.ptr,
        *ptr2 = (int *) buf2.ptr,
        *ptr3 = (int *) buf3.ptr;

    parallel_add<<<32, 32>>>(ptr1, ptr2, ptr3, buf1.size);

    return arr1;
}

PYBIND11_PLUGIN(pybind_gpu) {
  pybind11::module m("pybind_gpu", "pybind11 bindings to CUDA kernels");

  m.def("add", "Add two Numpy arrays");

  return m.ptr();
}
