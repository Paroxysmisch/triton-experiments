import triton
import triton.language as tl

# Define the block size as a constant
BLOCK_SIZE = 128

# Triton kernel for out-of-place multiplication by 2
@triton.jit
def mul2_kernel(X, Z, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(X + offsets, mask=mask)
    z = x * 2
    tl.store(Z + offsets, z, mask=mask)

# Triton kernel for in-place multiplication by 2
@triton.jit
def mul2_inplace_kernel(X, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(X + offsets, mask=mask)
    z = x * 2
    tl.store(X + offsets, z, mask=mask)

# Wrapper function for out-of-place multiplication by 2
def triton_mul2(X, n_elements):
    Z = tl.zeros_like(X)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    mul2_kernel[grid](X, Z, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return Z

# Wrapper function for in-place multiplication by 2
def triton_mul2_inplace(X, n_elements):
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    mul2_inplace_kernel[grid](X, n_elements, BLOCK_SIZE=BLOCK_SIZE)

# Example usage
import numpy as np

# Create a tensor
X = np.arange(1024, dtype=np.float32)

# Out-of-place multiplication
Z = triton_mul2(X, len(X))
print("Out-of-place result:", Z)

# In-place multiplication
triton_mul2_inplace(X, len(X))
print("In-place result:", X)
