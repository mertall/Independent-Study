import numpy as np
from typing import Tuple

def qr(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m = len(A) # Number of rows
    n = len(A[0]) # Number of columns
    Q = np.eye(m)
    R = A.copy()

    for column_number in range(n):
        # Grab column with all values below diagonal
        R_T = R.T
        x = R_T[column_number][column_number:m]

        # Build Householder vector
        e = np.zeros(len(x)) # Set basis vector
        e[0] = 1 * -np.sign(x[0]) # Flip the sign of x so during calculation of v we don't approach zero
        x_norm = np.sqrt(x.T @ x) # norm of x
        v = x - (x_norm * e) 

        # Householder reflection matrix
        v_inner = v.T @ v
        v_outer = np.outer(v,v)
        H = np.eye(m)
        H[column_number:m, column_number:m] -= 2/v_inner * v_outer # Only apply section of H that in R hasn't been eliminated yet

        # Apply H_n to R only on non eliminated vectors
        R = H @ R
        # Accumlate Q as H_1...H_n
        Q = Q @ H
    return Q[:,:n], R[:n,:n]


# First test: 4×4 matrix
A_4x4 = np.array([
    [1,     2,   3,  4],
    [2,     6,   7,  8],
    [3,     7,   9, 10],
    [4,     8,  10, 11]
], dtype=float)

Q, R = qr(A_4x4)
print(np.allclose(Q @ R, A_4x4))

# Overdetermined test 1: 5×3 tall matrix
A_5x3 = np.array([
    [1.,  2.,  3.],
    [4.,  5.,  6.],
    [7.,  8.,  9.],
    [2.,  3.,  4.],
    [5.,  6.,  7.]
], dtype=float)

Q, R = qr(A_5x3)
print(np.allclose(Q @ R, A_5x3))


# Overdetermined test 2: 4×2 tall matrix
A_4x2 = np.array([
    [1., 0.],
    [0., 1.],
    [1., 1.],
    [2., 3.]
], dtype=float)

Q, R = qr(A_4x2)
print(np.allclose(Q @ R, A_4x2))