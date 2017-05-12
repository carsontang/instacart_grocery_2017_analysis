#ifndef PYBIND_GPU_H
#define PYBIND_GPU_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#define NUM_CUDA_BLOCKS  1
#define NUM_CUDA_THREADS 2

pybind11::array_t<int> add(pybind11::array_t<int> arr1, pybind11::array_t<int> arr2);

pybind11::array_t<int> csr_mv_dot(pybind11::array val,
                                  pybind11::array col_ind,
                                  pybind11::array row_ptr,
                                  pybind11::array vector);

#endif // PYBIND_GPU_H
