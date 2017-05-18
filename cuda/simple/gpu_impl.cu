#include "pybind_simple.h"

__global__ void kernel_add(int *i, int *j, int *result) {
    *result = *i + *j;
}

__global__ void kernel_add_arrays(int *arr1, int *arr2, int *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = arr1[idx] + arr2[idx];
    }
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

    cudaMemcpy(&result, d_result, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_i);
    cudaFree(d_j);
    cudaFree(d_result);

    cudaDeviceReset();

    return result;
}

void parallel_add(int* arr1, int* arr2, int* result, int n) {
    size_t nBytes = n * sizeof(int);
    int *d_arr1, *d_arr2, *d_result;

    cudaMalloc( (int **)&d_arr1, nBytes);
    cudaMalloc( (int **)&d_arr2, nBytes);
    cudaMalloc( (int **)&d_result, nBytes);

    cudaMemcpy(d_arr1, arr1, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr2, arr2, nBytes, cudaMemcpyHostToDevice);

    kernel_add_arrays<<<32, 32>>>(d_arr1, d_arr2, d_result, n);

    cudaMemcpy(result, d_result, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_arr1);
    cudaFree(d_arr2);
    cudaFree(d_result);

    cudaDeviceReset();
}