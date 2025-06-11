import numpy as np
from typing import Tuple

def lu_with_pivot(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    LU Decomposition implementation in numpy, no pivoting
    
    Input: A; numpy.ndarray that is square

    Ouput: (L,U); tuple of numpy.ndarrays
    L (np.ndarray): Unit lower-triangular matrix.
    U (np.ndarray): Upper-triangular matrix.
    '''

    # Index variables
    i=len(A) # number of rows
    j=len(A[0]) # number of columns 
    # we could use just i since matrix is square;
    # but this will make the code clear to read

    assert i == j

    # Matrices to preform Gaussian Eliminiation with Pivoting
    L=np.eye(i,j) # Identity matrix size of A for L
    U=A.copy()
    P=np.eye(i,j)
    assert L.size==A.size
    assert U.size==A.size

    for column_number in range(j):
        row_below_diagonal=column_number+1
        # Loop across row values below diagonal
        for row_number in range(row_below_diagonal,i):
            
            # Observe value above entry we are trying to eliminate
            # If we are looking at a value in row 2, column 1, 
            # we need divide our current value by the value on the diagonal in same column
            # NOTICE: if our diagonal entry is 0 here, we cannot proceed. We can fix this with partial pivoting.
            if U[column_number][column_number]==0:
                print("pivoting partially")
                # Find max value in given column
                current_max=-np.inf
                row_swap = column_number  # default: no swap unless better row found
                for row_value in range(column_number,i):
                    potential_value=U[row_value][column_number]
                    if potential_value > current_max:
                        current_max = potential_value
                        row_swap = row_value
                # Swap row with zero with max value row in U
                U[[column_number, row_swap]] = U[[row_swap, column_number]]
                # Swap row with zero with max value row in P
                P[[column_number, row_swap]] = P[[row_swap, column_number]]
                # Swap the portions of rows in L that were already computed (i.e., all columns before the current pivot column).
                # This maintains consistency in L after row swaps in U, since these entries represent previous elimination steps.
                if column_number > 0:
                    L[[column_number, row_swap], :column_number]=L[[row_swap, column_number], :column_number]
                
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

    return (P, L,U)

# This matrix has a zero in the (0,0) position — LU will fail without row swapping
A_3x3=np.array([
    [0, 2, 3],
    [1, 1, 1],
    [4, 5, 6]
], dtype=float)

try:
    P,L,U=lu_with_pivot(A_3x3)
    print(np.allclose(L @ U, P @ A_3x3))
except ZeroDivisionError as e:
    print("LU failed on A_3x3 without pivoting:", e)

# Diagonal contains small values that cause numerical instability
A_4x4 = np.array([
    [0, 2, 3, 4],
    [1, 5, 6, 7],
    [2, 8, 9, 10],
    [3, 11, 12, 13]
], dtype=float)


try:
    P,L,U=lu_with_pivot(A_4x4)
    print(np.allclose(L @ U, P @ A_4x4))
except ZeroDivisionError as e:
    print("LU failed on A_4x4 without pivoting:", e)

# Zero pivot at (1,1) — will trigger division by zero
A_5x5 = np.array([
    [0, 2, 3, 4, 1],
    [3, 0, 2, 5, 2],  # zero pivot here
    [1, 1, 0, 3, 2],
    [4, 2, 1, 0, 3],
    [2, 3, 2, 1, 9]
], dtype=float)

try:
    P,L,U=lu_with_pivot(A_5x5)
    print(np.allclose(L @ U, P @ A_5x5))
except ZeroDivisionError as e:
    print("LU failed on A_5x5 without pivoting:", e)

