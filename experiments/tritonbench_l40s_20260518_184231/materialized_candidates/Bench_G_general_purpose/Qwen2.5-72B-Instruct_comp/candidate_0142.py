import triton
import triton.language as tl

@triton.jit
def _geglu_tanh_forward_kernel(
    a_ptr,  # Pointer to input tensor a
    b_ptr,  # Pointer to input tensor b
    c_ptr,  # Pointer to output tensor c
    n_rows,  # Number of rows in the 2D tensor
    n_cols,  # Number of columns in the 2D tensor
    BLOCK_SIZE: tl.constexpr,  # Block size for parallelism
):
    pid = tl.program_id(axis=0)  # Get the program ID
    block_start = pid * BLOCK_SIZE  # Start index for the block

    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # Offsets for the block
    mask = offsets < n_cols  # Mask to ensure we don't go out of bounds

    a = tl.load(a_ptr + offsets, mask=mask)  # Load a
    b = tl.load(b_ptr + offsets, mask=mask)  # Load b

    # Compute the GEGLU activation using tanh approximation
    x = 0.044715 * a * a * a
    y = a + x
    z = tl.sqrt(2.0 / 3.141592653589793) * y
    tanh_z = tl.tanh(z)
    c = 0.5 * a * (1 + tanh_z)

    tl.store(c_ptr + offsets, c, mask=mask)  # Store the result in c

@triton.jit
def _geglu_tanh_backward_kernel(
    a_ptr,  # Pointer to input tensor a
    b_ptr,  # Pointer to input tensor b
    dc_ptr,  # Pointer to upstream gradient tensor dc
    da_ptr,  # Pointer to gradient tensor for a
    db_ptr,  # Pointer to gradient tensor for b
    n_rows,  # Number of rows in the 2D tensor
    n_cols,  # Number of columns in the 2D tensor
    BLOCK_SIZE: tl.constexpr,  # Block size for parallelism
):
    pid = tl.program_id(axis=0)  # Get the program ID
    block_start = pid * BLOCK_SIZE  # Start index for the block

    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # Offsets for the block
    mask = offsets < n_cols  # Mask to ensure we don't go out of bounds

    a = tl.load(a_ptr + offsets, mask=mask)  # Load a
    b = tl.load(b_ptr + offsets, mask=mask)  # Load b
    dc = tl.load(dc_ptr + offsets, mask=mask)  # Load upstream gradient

    # Recompute necessary intermediates
    x = 0.044715 * a * a * a
    y = a + x
    z = tl.sqrt(2.0 / 3.141592653589793) * y
    tanh_z = tl.tanh(z)
    c = 0.5 * a * (1 + tanh_z)

    # Compute the derivative of the GEGLU activation
    dz = 0.5 * (1 + tanh_z) + 0.5 * a * (1 - tanh_z * tanh_z) * tl.sqrt(2.0 / 3.141592653589793) * (1 + 3 * 0.044715 * a * a)
    da = dz * dc
    db = 0.5 * a * (1 + tanh_z) * dc

    tl.store(da_ptr + offsets, da, mask=mask)  # Store the gradient for a
    tl.store(db_ptr + offsets, db, mask=mask)  # Store the gradient for b

import torch

def geglu_forward(a, b):
    n_rows, n_cols = a.shape[-2], a.shape[-1]
    c = torch.empty_like(a)  # Initialize the output tensor

    BLOCK_SIZE = 128
    num_warps = 4

    _geglu_tanh_forward_kernel[(n_rows,)](
        a, b, c, n_rows, n_cols, BLOCK_SIZE,
        num_warps=num_warps
    )

    return c

def geglu_backward(a, b, dc):
    n_rows, n_cols = a.shape[-2], a.shape[-1]
    da = torch.empty_like(a)  # Initialize the gradient tensor for a
    db = torch.empty_like(b)  # Initialize the gradient tensor for b

    BLOCK_SIZE = 128
    num_warps = 4

    _geglu_tanh_backward_kernel[(n_rows,)](
        a, b, dc, da, db, n_rows, n_cols, BLOCK_SIZE,
        num_warps=num_warps
    )

    return da, db
