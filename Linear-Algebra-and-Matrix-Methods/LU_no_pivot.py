import numpy as np
from typing import Tuple

def lu_no_pivot(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    LU Decomposition implementation in numpy, no pivoting
    
    Input: A; numpy.ndarray that is square

    Ouput: (L,U); tuple of numpy.ndarrays
    L (np.ndarray): Unit lower-triangular matrix.
    U (np.ndarray): Upper-triangular matrix.
    '''

    # Index variables
    i = len(A) # number of rows
    j = len(A[0]) # number of columns 
    # we could use just i since matrix is square;
    # but this will make the code clear to read

    assert i == j

    # Matrices to preform Gaussian Eliminiation with
    L = np.eye(i,j) # Identity matrix size of A for L
    U = A.copy()
    assert L.size == A.size
    assert U.size == A.size

    for column_number in range(j):
        row_below_diagonal=column_number+1
        # Loop across row values below diagonal
        for row_number in range(row_below_diagonal,i):

            # Observe value above entry we are trying to eliminate
            # If we are looking at a value in row 2, column 1, 
            # we need divide our current value by the value on the diagonal in same column
            # NOTICE: if our diagonal entry is 0 here, we cannot proceed. We can fix this with partial pivoting.
            factor=U[row_number][column_number]/U[column_number][column_number]

            # Take the row in our U, that corresponds to the column number we are currently observing
            modified_row=factor*U[column_number]

            # To eliminate a value below the diagonal in a given column, 
            # subtract a scaled version of the current row—whose diagonal element is assumed non-zero—from each lower row.
            U_new_row=U[row_number]-modified_row

            # Store new row in U
            U[row_number]=U_new_row

            # Store factor in L
            L[row_number][column_number]=factor

    return (L,U)

A_3x3 = np.array([
    [2, 4, 1],
    [0, 3, 5],
    [1, -2, 1]
], dtype=float)

L,U =lu_no_pivot(A_3x3)

print(np.allclose(L @ U, A_3x3))

A_4x4 = np.array([
    [4, 2, 3, 1],
    [0, 5, 1, 2],
    [1, 1, 6, 3],
    [2, 4, 1, 7]
], dtype=float)

L,U =lu_no_pivot(A_4x4)

print(np.allclose(L @ U, A_4x4))

A_5x5 = np.array([
    [5, 2, 1, 3, 4],
    [0, 6, 2, 1, 3],
    [1, 0, 7, 4, 2],
    [2, 1, 3, 8, 1],
    [3, 4, 2, 0, 9]
], dtype=float)

L,U =lu_no_pivot(A_5x5)

print(np.allclose(L @ U, A_5x5))

