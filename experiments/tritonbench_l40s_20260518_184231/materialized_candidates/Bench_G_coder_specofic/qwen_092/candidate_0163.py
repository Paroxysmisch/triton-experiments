import triton
import triton.language as tl
import numpy as np

# Define the dimensions and strides
M, K = 128, 128
K, N = 128, 128

# Initialize input matrices A and B
A = np.random.rand(M, K).astype(np.float32)
B = np.random.rand(K, N).astype(np.float32)

# Allocate memory for the output matrix C
C = np.zeros((M, N), dtype=np.float32)

# Convert input matrices to Triton tensors
A_tensor = triton.from_numpy(A)
B_tensor = triton.from_numpy(B)
C_tensor = triton.from_numpy(C)

# Define block sizes
BLOCK_SIZE_M = 32
BLOCK_SIZE_N = 32
BLOCK_SIZE_K = 32

# Define the activation function
activation = "leaky_relu"

# Call the matmul function
matmul(
    A_tensor.data_ptr(), B_tensor.data_ptr(), C_tensor.data_ptr(),
    A_tensor.stride(0), A_tensor.stride(1),
    B_tensor.stride(0), B_tensor.stride(1),
    C_tensor.stride(0), C_tensor.stride(1),
    M, N, K,
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
    activation
)

# Convert the output tensor back to a NumPy array
C_result = C_tensor.numpy()

# Print the result
print(C_result)
