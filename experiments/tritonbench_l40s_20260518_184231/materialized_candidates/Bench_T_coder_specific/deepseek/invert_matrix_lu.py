import triton
import numpy as np

# Create a Tensor object representing the input matrix
A = triton.tensor(np.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]], dtype=np.float32))

# Compute the inverse of the matrix using LU decomposition
A_inv = triton.linalg.invert_matrix_lu(A, pivot=True)

# Print the result
print(A_inv.numpy())
