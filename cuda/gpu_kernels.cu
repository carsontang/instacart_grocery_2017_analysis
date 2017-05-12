#include "pybind_gpu.h"

pybind11::array_t<int> csr_mv_dot(
        pybind11::array_t<int> val,
        pybind11::array_t<int> col_ind,
        pybind11::array_t<int> row_ptr,
        pybind11::array_t<int> vector
) {
    _csv_mv_dot_kernel<<<NUM_CUDA_BLOCKS,NUM_CUDA_THREADS>>>();
}

__global__ void _csv_mv_dot_kernel() {
    return;
}
