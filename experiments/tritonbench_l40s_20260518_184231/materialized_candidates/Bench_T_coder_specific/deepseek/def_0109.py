import triton
import numpy as np

# Create a tensor from a numpy array
A = triton.tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))

# Compute the pseudoinverse
A_pinv = triton.functional.pseudoinverse_svd(A)

# Convert the result back to a numpy array
A_pinv_np = A_pinv.numpy()
