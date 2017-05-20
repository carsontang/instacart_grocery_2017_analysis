import numpy as np
import sys

# Implements the Compressed Row Storage (CRS) format of a sparse matrix
# See http://netlib.org/linalg/html_templates/node91.html

class SparseMatrix(object):
    def __init__(self, val, col_ind, row_ptr):
        self.val = val
        self.col_ind = col_ind
        self.row_ptr = row_ptr

    def get(self, row, col):
        # Refer to http://netlib.org/linalg/html_templates/node91.html
        cur_row_start_pos = self.row_ptr[row]
        next_row_start_pos = self.row_ptr[row + 1]

        for index in range(cur_row_start_pos, next_row_start_pos):
            if self.col_ind[index] == col:
                return self.val[index]

        return 0

    def approx_size_in_bytes(self):
        print "val (length = %d): %d bytes + 96 bytes for np.array" % (len(self.val), sys.getsizeof(self.val) - 96)
        print "col_ind (length = %d): %d bytes + 96 bytes for np.array" % (len(self.col_ind), sys.getsizeof(self.col_ind) - 96)
        print "row_ptr (length = %d): %d bytes + 96 bytes for np.array" % (len(self.row_ptr), sys.getsizeof(self.row_ptr) - 96)

        return sys.getsizeof(self.val) \
                + sys.getsizeof(self.col_ind) \
                + sys.getsizeof(self.row_ptr)

    def dot(self, vector):
        pass

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return np.array_equal(self.val, other.val) \
                    and np.array_equal(self.row_ptr, other.row_ptr) \
                    and np.array_equal(self.col_ind, other.col_ind)
        return False

    def __ne__(self, other):
        return not self.__eq__(other)


    @staticmethod
    def wrap(val, col_ind, row_ptr):
        return SparseMatrix(val, col_ind, row_ptr)

    @staticmethod
    def build(matrix):
        rows, cols = matrix.shape
        val = np.array([], dtype=np.int)
        col_ind = np.array([], dtype=np.int)
        row_ptr = np.array([], dtype=np.int)

        nnz = 0

        for i in range(rows):
            row_ptr = np.append(row_ptr, nnz)
            for j in range(cols):
                if matrix[i, j] != 0:
                    val = np.append(val, matrix[i, j])
                    col_ind = np.append(col_ind, j)
                    nnz += 1

        row_ptr = np.append(row_ptr, nnz)
        return SparseMatrix(val, col_ind, row_ptr)

    @staticmethod
    def concatenate(first, second):
        col_ind = np.concatenate((first.col_ind, second.col_ind))
        val = np.concatenate((first.val, second.val))

        # Use self.row_ptr's last element as the
        # first element of other.row_ptr. other.row_ptr then
        # requires some modifications
        other_row_ptr = map(lambda x: x-second.row_ptr[0], second.row_ptr)
        other_row_ptr = other_row_ptr[1:]
        other_row_ptr = map(lambda x: x + first.row_ptr[-1], other_row_ptr)

        row_ptr = np.concatenate((first.row_ptr, other_row_ptr))
        return SparseMatrix.wrap(val, col_ind, row_ptr)