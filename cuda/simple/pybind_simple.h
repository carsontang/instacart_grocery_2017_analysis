#ifndef PYBIND_SIMPLE_H
#define PYBIND_SIMPLE_H

int add(int i, int j);

void parallel_add(int* arr1, int* arr2, int* result, int size);

#endif //PYBIND_SIMPLE_H