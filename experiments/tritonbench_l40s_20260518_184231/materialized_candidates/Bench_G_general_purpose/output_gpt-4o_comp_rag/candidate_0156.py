import triton
import triton.language as tl

# Define the block size
BLOCK_SIZE = 1024

# Kernel for multiplying elements by 2 and storing in an output tensor
@triton.jit
def mul2_kernel(X, Z, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load, multiply by 2, and store
    x = tl.load(X + offsets, mask=mask)
    tl.store(Z + offsets, x * 2, mask=mask)

# Kernel for multiplying elements by 2 in place
@triton.jit
def mul2_inplace_kernel(X, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load, multiply by 2, and store back to the same location
    x = tl.load(X + offsets, mask=mask)
    tl.store(X + offsets, x * 2, mask=mask)

# Wrapper function to multiply elements by 2 and store in an output tensor
def triton_mul2(input_tensor, output_tensor, n_elements):
    grid = lambda meta: (triton.cdiv(n_elements, BLOCK_SIZE),)
    mul2_kernel[grid](input_tensor, output_tensor, n_elements, BLOCK_SIZE=BLOCK_SIZE)

# Wrapper function to multiply elements by 2 in place
def triton_mul2_inplace(input_tensor, n_elements):
    grid = lambda meta: (triton.cdiv(n_elements, BLOCK_SIZE),)
    mul2_inplace_kernel[grid](input_tensor, n_elements, BLOCK_SIZE=BLOCK_SIZE)
