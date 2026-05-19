import triton
import triton.language as tl
import torch

# --------------------------------------------------------------------------------
# Triton kernel for approximate forward pass of GEGLU using a tanh-based approximation
# --------------------------------------------------------------------------------
@triton.jit
def _geglu_tanh_forward_kernel(
    a_ptr, b_ptr, c_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_id = tl.program_id(0)
    row_offset = row_id * n_cols

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load inputs
    a_vals = tl.load(a_ptr + row_offset + col_offsets, mask=mask, other=0.0).to(tl.float32)
    b_vals = tl.load(b_ptr + row_offset + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Approximate GEGLU activation: 0.5 * a * (1 + tanh(s * (a + 0.044715 * a^3))) * b
    s = 0.7978845608  # sqrt(2 / pi)
    x_vals = s * (a_vals + 0.044715 * (a_vals ** 3))
    t_vals = tl.math.tanh(x_vals)
    f_vals = 0.5 * a_vals * (1.0 + t_vals)
    c_vals = f_vals * b_vals

    # Store result
    tl.store(c_ptr + row_offset + col_offsets, c_vals, mask=mask)

# --------------------------------------------------------------------------------
# Python wrapper for the approximate forward GEGLU pass
# --------------------------------------------------------------------------------
def geglu_forward(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Reshape inputs to 2D (n_rows x n_cols), if necessary
    original_shape = a.shape
    n_rows = a.shape[0]
    n_cols = a.shape[-1]

    # Create output tensor
    c = torch.empty_like(a)

    # Constants for kernel launch
    BLOCK_SIZE = 128
    num_warps = 4  # not strictly used here, but can be part of the launch config

    # Launch kernel
    grid = (n_rows,)
    _geglu_tanh_forward_kernel[grid](
        a, b, c,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    return c

# --------------------------------------------------------------------------------
# Triton kernel for approximate backward pass of GEGLU using a tanh-based approximation
# --------------------------------------------------------------------------------
@triton.jit
def _geglu_tanh_backward_kernel(
    dc_ptr, a_ptr, b_ptr, da_ptr, db_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_id = tl.program_id(0)
    row_offset = row_id * n_cols

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load gradients and inputs
    dc_vals = tl.load(dc_ptr + row_offset + col_offsets, mask=mask, other=0.0).to(tl.float32)
    a_vals = tl.load(a_ptr + row_offset + col_offsets, mask=mask, other=0.0).to(tl.float32)
    b_vals = tl.load(b_ptr + row_offset + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Recompute forward function f(a) = 0.5 * a * (1 + tanh(s * (a + 0.044715 * a^3)))
    s = 0.7978845608
    x_vals = s * (a_vals + 0.044715 * (a_vals ** 3))
