#include "pybind_gpu.h"

void parallel_add(int* arr1, int* arr2, int* result, int size) {
    gpu_add(arr1, arr2, result, size);
}

__global__ void gpu_add(int* arr1, int* arr2, int* result, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = arr1[idx] + arr2[idx];
    }
}
