import triton
import triton.language as tl

@triton.jit
def rotary_kernel(
    X_ptr,  # Pointer to the input tensor
    cos_ptr,  # Pointer to the cosine values
    sin_ptr,  # Pointer to the sine values
    Y_ptr,  # Pointer to the output tensor
    N,  # Number of elements in the sequence
    D,  # Dimension of the input tensor
    varlen,  # Flag for variable length sequences
    interleaved,  # Flag for interleaved data layout
    conjugate,  # Flag for conjugate transformation
    BLOCK_SIZE: tl.constexpr,  # Block size for parallel processing
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load the input data
    X = tl.load(X_ptr + offsets, mask=offsets < N * D, other=0.0)

    # Compute the indices for cos and sin
    seq_idx = offsets // D
    dim_idx = offsets % D

    # Load the cosine and sine values
    cos = tl.load(cos_ptr + seq_idx, mask=seq_idx < N, other=1.0)
    sin = tl.load(sin_ptr + seq_idx, mask=seq_idx < N, other=0.0)

    # Apply the rotary positional encoding
    if interleaved:
        X_even = X[::2]
        X_odd = X[1::2]
        Y_even = X_even * cos - X_odd * sin
        Y_odd = X_even * sin + X_odd * cos
        Y = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        Y[::2] = Y_even
        Y[1::2] = Y_odd
    else:
        X_even = X[::2]
        X_odd = X[1::2]
        Y_even = X_even * cos - X_odd * sin
        Y_odd = X_even * sin + X_odd * cos
        Y = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        Y[::2] = Y_even
        Y[1::2] = Y_odd

    # Apply conjugate transformation if needed
    if conjugate:
        Y = tl.conj(Y)

    # Store the result
    tl.store(Y_ptr + offsets, Y, mask=offsets < N * D)

import torch

def apply_rotary(X, cos, sin, varlen=False, interleaved=True, conjugate=False):
    N, D = X.shape
    assert cos.shape == (N,)
    assert sin.shape == (N,)

    # Convert tensors to Triton-compatible format
    X_triton = X.contiguous().view(-1)
    cos_triton = cos.contiguous()
    sin_triton = sin.contiguous()
    Y_triton = torch.empty_like(X_triton)

    # Define the grid and block sizes
    BLOCK_SIZE = 1024
    grid_size = (N * D + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the kernel
    rotary_kernel[grid_size, BLOCK_SIZE](
        X_triton, cos_triton, sin_triton, Y_triton, N, D, varlen, interleaved, conjugate, BLOCK_SIZE
    )

    # Reshape the output tensor
    Y = Y_triton.view(N, D)
    return Y
