#include "pybind_simple.h"

__global__ void kernel_add(int *i, int *j, int *result) {
    *result = *i + *j;
}

int add(int i, int j) {
    size_t nBytes = sizeof(i);
    int result = 0;

    int *d_i, *d_j, *d_result;
    cudaMalloc( (int **)&d_i, nBytes);
    cudaMalloc( (int **)&d_j, nBytes);
    cudaMalloc( (int **)&d_result, nBytes);

    cudaMemcpy(d_i, &i, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_j, &j, nBytes, cudaMemcpyHostToDevice);

    kernel_add<<<1, 1>>>(d_i, d_j, d_result);

    cudaMemcpy(&result, d_result, cudaMemcpyDeviceToHost);

    cudaFree(d_i);
    cudaFree(d_j);
    cudaFree(d_result);

    return result;
}