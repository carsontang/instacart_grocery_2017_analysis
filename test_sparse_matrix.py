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

    def test_eq(self):
        m1 = SparseMatrix.build(np.matrix('10 0 0 0 -2 0; 3 9 0 0 0 3; 0 7 8 7 0 0; 3 0 8 7 5 0; 0 8 0 9 9 13; 0 4 0 0 2 -1'))
        m2 = SparseMatrix.build(np.matrix('10 0 0 0 -2 0; 3 9 0 0 0 3; 0 7 8 7 0 0; 3 0 8 7 5 0; 0 8 0 9 9 13; 0 4 0 0 2 -1'))
        m3 = SparseMatrix.build(np.matrix('3 9 0 0 0 3; 0 7 8 7 0 0; 3 0 8 7 5 0; 0 8 0 9 9 13; 0 4 0 0 2 -1; 10 0 0 0 -2 0'))

        self.assertEqual(m1, m2)
        self.assertNotEqual(m1, m3)

    def test_wrap(self):
        val = [10, -2, 3, 9, 3, 7, 8, 7, 3, 8, 7, 5, 8, 9, 9, 13, 4, 2, -1]
        col_ind = [0, 4, 0, 1, 5, 1, 2, 3, 0, 2, 3, 4, 1, 3, 4, 5, 1, 4, 5]
        row_ptr = [0, 2, 5, 8, 12, 16, 19]
        matrix = np.matrix('10 0 0 0 -2 0; 3 9 0 0 0 3; 0 7 8 7 0 0; 3 0 8 7 5 0; 0 8 0 9 9 13; 0 4 0 0 2 -1')
        sparse = SparseMatrix.build(matrix)
        wrapped = SparseMatrix.wrap(val, col_ind, row_ptr)

        self.assertTrue(np.array_equal(sparse.val, wrapped.val))
        self.assertTrue(np.array_equal(sparse.col_ind, wrapped.col_ind))
        self.assertTrue(np.array_equal(sparse.row_ptr, wrapped.row_ptr))

    def test_append(self):
        upper_half = SparseMatrix.build(np.matrix('10 0 0 0 -2 0; 3 9 0 0 0 3; 0 7 8 7 0 0'))
        lower_half = SparseMatrix.build(np.matrix('3 0 8 7 5 0; 0 8 0 9 9 13; 0 4 0 0 2 -1'))
        full_matrix = SparseMatrix.build(np.matrix('10 0 0 0 -2 0; 3 9 0 0 0 3; 0 7 8 7 0 0; 3 0 8 7 5 0; 0 8 0 9 9 13; 0 4 0 0 2 -1'))

        self.assertEqual(full_matrix, upper_half.append(lower_half))


if __name__ == '__main__':
    unittest.main()
