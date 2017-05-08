import unittest
import numpy as np
from sparse_matrix import SparseMatrix

class TestSparseMatrix(unittest.TestCase):

    def test_build(self):
        matrix = np.matrix('10 0 0 0 -2 0; 3 9 0 0 0 3; 0 7 8 7 0 0; 3 0 8 7 5 0; 0 8 0 9 9 13; 0 4 0 0 2 -1')
        val = [10, -2, 3, 9, 3, 7, 8, 7, 3, 8, 7, 5, 8, 9, 9, 13, 4, 2, -1]
        col_ind = [0, 4, 0, 1, 5, 1, 2, 3, 0, 2, 3, 4, 1, 3, 4, 5, 1, 4, 5]
        row_ptr = [0, 2, 5, 8, 12, 16, 19]
        sparse = SparseMatrix.build(matrix)

        self.assertEqual(len(sparse.row_ptr), len(row_ptr))
        for i in range(len(sparse.row_ptr)):
            self.assertEqual(sparse.row_ptr[i], row_ptr[i])

        self.assertEqual(len(sparse.col_ind), len(col_ind))
        for i in range(len(sparse.col_ind)):
            self.assertEqual(sparse.col_ind[i], col_ind[i], "%d" % i)

        self.assertEqual(len(sparse.val), len(val))
        for i in range(len(sparse.val)):
            self.assertEqual(sparse.val[i], val[i])

    def test_get(self):
        matrix = np.matrix('10 0 0 0 -2 0; 3 9 0 0 0 3; 0 7 8 7 0 0; 3 0 8 7 5 0; 0 8 0 9 9 13; 0 4 0 0 2 -1')
        sparse = SparseMatrix.build(matrix)
        rows, cols = matrix.shape
        for i in range(rows):
            for j in range(cols):
                self.assertEqual(sparse.get(i, j), matrix[i, j])

if __name__ == '__main__':
    unittest.main()
