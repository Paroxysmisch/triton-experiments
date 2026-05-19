import triton
import triton.language as tl

# Define the SiLU function using Triton's JIT capabilities
@triton.jit
def silu(x):
    return x * tl.sigmoid(x)

# Define the forward kernel for SWiGLU
@triton.jit
def _swiglu_forward_kernel(
    a_ptr, b_ptr, c_ptr,
    a_shape, b_shape, c_shape,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.cdiv(a_shape[0], BLOCK_SIZE)
    a_off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    a_off = a_off[:, None] * a_shape[1] + tl.arange(0, a_shape[1])
    a_off = a_off.flatten()
    a = tl.load(a_ptr + a_off, mask=a_off < a_shape[0] * a_shape[1], other=0.0)
    b_off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    b_off = b_off[:, None] * b_shape[1] + tl.arange(0, b_shape[1])
    b_off = b_off.flatten()
    b = tl.load(b_ptr + b_off, mask=b_off < b_shape[0] * b_shape[1], other=0.0)
    c_off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    c_off = c_off[:, None] * c_shape[1] + tl.arange(0, c_shape[1])
    c_off = c_off.flatten()
    a_silu = silu(a)
    c = b * a_silu
    tl.store(c_ptr + c_off, c, mask=c_off < c_shape[0] * c_shape[1])

# Define the forward function for SWiGLU
@triton.jit
def swiglu_forward(
    a, b, c,
    BLOCK_SIZE: tl.constexpr,
):
    a_shape = a.shape
    b_shape = b.shape
    c_shape = c.shape
    num_warps = calculate_settings(a_shape[0], a_shape[1], b_shape[1], BLOCK_SIZE)
    grid_size = tl.cdiv(a_shape[0], BLOCK_SIZE)
    _swiglu_forward_kernel[grid_size, BLOCK_SIZE, num_warps](a, b, c, a_shape, b_shape, c_shape, BLOCK_SIZE)

# Define the backward kernel for SWiGLU
@triton.jit
def _swiglu_backward_kernel(
    a_ptr, b_ptr, c_ptr, grad_c_ptr, grad_a_ptr, grad_b_ptr,
    a_shape, b_shape, c_shape,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.cdiv(c_shape[0], BLOCK_SIZE)
    c_off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    c_off = c_off[:, None] * c_shape[1] + tl.arange(0, c_shape[1])
    c_off = c_off.flatten()
    c = tl.load(c_ptr + c_off, mask=c_off < c_shape[0] * c_shape[1], other=0.0)
    grad_c = tl.load(grad_c_ptr + c_off, mask=c_off < c_shape[0] * c_shape[1], other=0.0)
    a_off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    a_off = a_off[:, None] * a_shape[1] + tl.arange(0, a_shape[1])
    a_off = a_off.flatten()
    a = tl.load(a_ptr + a_off, mask=a_off < a_shape[0] * a_shape[1], other=0.0)
    b_off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    b_off = b_off[:, None] * b_shape[1] + tl.arange(0, b_shape[1])
    b_off = b_off.flatten()
    b = tl.load(b_ptr + b_off, mask=b_off < b_shape[0] * b_shape[1], other=0.0)
    a_silu = silu(a)
    grad_a = grad_c * b * a_silu * (1.0 - a_silu)
    grad_b = grad_c * a_silu
    tl.store(grad_a_ptr + a_off, grad_a, mask=a_off < a_shape[0] * a_shape[1])
    tl.store(grad_b_ptr + b_off, grad_b, mask=b_off < b_shape[0] * b_shape[1])

# Define the backward function for SWiGLU
@triton.jit
def swiglu_backward(
    a, b, c, grad_c, grad_a, grad_b,
    BLOCK_SIZE: tl.constexpr,
):
    a_shape = a.shape
    b_shape = b.shape
    c_shape = c.shape
    num_warps = calculate_settings(c_shape[0], c_shape[1], a_shape[1], BLOCK_SIZE)
    grid_size = tl.cdiv(c_shape[0], BLOCK_SIZE)
    _swiglu_backward_kernel[grid_size, BLOCK_SIZE, num_warps](a, b, c, grad_c, grad_a, grad_b, a_shape, b_shape, c_shape, BLOCK_SIZE)

# Helper function to calculate settings
def calculate_settings(a_rows, a_cols, b_cols, block_size):
    max_fused_size = 1024
    num_warps = 4
    while block_size * num_warps > max_fused_size:
        num_warps //= 2
    return num_warps
