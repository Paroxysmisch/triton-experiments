import triton
import triton.language as tl

# Define the SiLU activation function
@triton.jit
def silu(x):
    return x * tl.sigmoid(x)

# Define the forward kernel
@triton.jit
def _swiglu_forward_kernel(a_ptr, b_ptr, c_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M * N

    a_offsets = a_ptr + offsets
    b_offsets = b_ptr + offsets
    c_offsets = c_ptr + offsets

    a = tl.load(a_offsets, mask=mask)
    b = tl.load(b_offsets, mask=mask)

    c = b * silu(a)
    tl.store(c_offsets, c, mask=mask)

# Define the backward kernel
@triton.jit
def _swiglu_backward_kernel(a_ptr, b_ptr, c_grad_ptr, a_grad_ptr, b_grad_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M * N

    a_offsets = a_ptr + offsets
    b_offsets = b_ptr + offsets
    c_grad_offsets = c_grad_ptr + offsets
    a_grad_offsets = a_grad_ptr + offsets
    b_grad_offsets = b_grad_ptr + offsets

    a = tl.load(a_offsets, mask=mask)
    b = tl.load(b_offsets, mask=mask)
    c_grad = tl.load(c_grad_offsets, mask=mask)

    silu_a = silu(a)
    a_grad = c_grad * b * (silu_a + a * (1 - silu_a))
    b_grad = c_grad * silu_a

    tl.store(a_grad_offsets, a_grad, mask=mask)
    tl.store(b_grad_offsets, b_grad, mask=mask)

# Helper function to calculate block size and number of warps
def calculate_settings(M, N, max_fused_size=1024, hip=False):
    if hip:
        num_warps = 4
    else:
        num_warps = 8

    block_size = min(max_fused_size, (M * N + num_warps - 1) // num_warps)
    block_size = 2 ** (block_size.bit_length() - 1)
    return block_size, num_warps

# Wrapper function for forward pass
def swiglu_forward(a, b):
    M, N = a.shape
    c = tl.zeros_like(a)
    BLOCK_SIZE, num_warps = calculate_settings(M, N)
    grid = (M * N + BLOCK_SIZE - 1) // BLOCK_SIZE
    _swiglu_forward_kernel[grid, BLOCK_SIZE, num_warps](a, b, c, M, N, BLOCK_SIZE)
    return c

# Wrapper function for backward pass
def swiglu_backward(a, b, c_grad):
    M, N = a.shape
    a_grad = tl.zeros_like(a)
    b_grad = tl.zeros_like(b)
    BLOCK_SIZE, num_warps = calculate_settings(M, N)
    grid = (M * N + BLOCK_SIZE - 1) // BLOCK_SIZE
    _swiglu_backward_kernel[grid, BLOCK_SIZE, num_warps](a, b, c_grad, a_grad, b_grad, M, N, BLOCK_SIZE)
    return a_grad, b_grad
