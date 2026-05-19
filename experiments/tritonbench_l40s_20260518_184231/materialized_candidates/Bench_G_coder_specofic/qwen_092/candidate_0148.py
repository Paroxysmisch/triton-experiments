import triton
import triton.language as tl
import numpy as np

# Initialize input tensor
n_elements = 1024
x = np.random.rand(n_elements).astype(np.float32)

# Allocate output tensor for triton_mul2
y = np.empty_like(x)

# Call triton_mul2
triton_mul2[triton.cdiv(n_elements, BLOCK_SIZE), BLOCK_SIZE](x, y, n_elements, BLOCK_SIZE)

# Call triton_mul2_inplace
triton_mul2_inplace[triton.cdiv(n_elements, BLOCK_SIZE), BLOCK_SIZE](x, n_elements, BLOCK_SIZE)

print("Original x:", x)
print("Result of triton_mul2:", y)
print("Result of triton_mul2_inplace:", x)
