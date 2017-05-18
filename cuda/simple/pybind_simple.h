#ifndef PYBIND_SIMPLE_H
#define PYBIND_SIMPLE_H

int add(int i, int j);

void parallel_add(int* arr1, int* arr2, int* result, int size);

/**
    parallel_csrmv - This function multiplies a sparse matrix and a regular vector,
    and returns the resulting vector in result.

    The sparse matrix is stored in the CSR format, which consists of 3 arrays:
    csr_vals, csr_cols, and csr_rows.

    nnz - number of nonzeroes.
*/
void parallel_csrmv(
    float* csr_vals,
    int* csr_cols,
    int* csr_rows,
    float* vector,
    float* result,
    const int nnz,
    const int n_rows);

#endif //PYBIND_SIMPLE_H