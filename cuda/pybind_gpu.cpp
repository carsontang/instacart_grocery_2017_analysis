#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "pybind_gpu.h"

// TODO: make sparse_mv_dot take in a sparse matrix and a vector
// The sparse matrix will be 3 Numpy arrays that comprise the CSR format of a
// sparse matrix.
// write a simpler method that simply adds two Numpy arrays.
/*
 * Take the dot product of a sparse matrix and a vector.
 * "sparse_mv_dot" means "sparse matrix-vector dot product"
 */

pybind11::array_t<int> add(pybind11::array_t<int> arr1, pybind11::array_t<int> arr2) {
    return arr1;
}

void csr_mv_dot(
  pybind11::array_t<int> val,
  pybind11::array_t<int> col_ind,
  pybind11::array_t<int> row_ptr,
  pybind11::array_t<int> vector
) {
    return;
}

PYBIND11_PLUGIN(pybind_gpu) {
  pybind11::module m("pybind_gpu", "pybind11 bindings to CUDA kernels");

  m.def("add", "Add two Numpy arrays");
  m.def("csr_mv_dot", "Take the dot product of a sparse matrix and a vector");

  return m.ptr();
}
