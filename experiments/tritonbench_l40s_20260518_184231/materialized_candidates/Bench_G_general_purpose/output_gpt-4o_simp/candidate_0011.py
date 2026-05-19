import triton
import triton.language as tl

@triton.jit
def rms_matmul_rbe_kernel(
    x_ptr, weight_ptr, out_ptr, rotary_ptr,
    n, m, k,  # Dimensions
    stride_xn, stride_xk,  # Strides for input matrix x
    stride_wn, stride_wk,  # Strides for weight matrix
    stride_on, stride_om,  # Strides for output matrix
    apply_rotary,  # Flag to apply rotary embeddings
    BLOCK_SIZE: tl.constexpr  # Block size for tiling
):
    # Compute the row and column indices
    row = tl.program_id(0)
    col = tl.program_id(1)

    # Compute the pointers to the current block
    x_ptrs = x_ptr + row * stride_xn + tl.arange(0, BLOCK_SIZE) * stride_xk
    w_ptrs = weight_ptr + col * stride_wk + tl.arange(0, BLOCK_SIZE) * stride_wn

    # Initialize accumulation
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Loop over the k dimension
    for k_block in range(0, k, BLOCK_SIZE):
        # Load x and w
        x = tl.load(x_ptrs + k_block * stride_xk)
        w = tl.load(w_ptrs + k_block * stride_wk)

        # Compute partial product
        acc += x * w

    # RMS normalization
    rms = tl.sqrt(tl.sum(acc * acc) / m)
    acc /= rms

    # Apply rotary embeddings if required
    if apply_rotary:
        rotary_emb = tl.load(rotary_ptr + row * stride_on + col * stride_om)
        acc = acc * rotary_emb

    # Store the result
    out_ptrs = out_ptr + row * stride_on + col * stride_om
    tl.store(out_ptrs, acc)

import torch

def rms_matmul_rbe_wrapper(x, weight, rotary_emb=None, apply_rotary=False):
    # Ensure the data types are compatible
    assert x.dtype in [torch.float16, torch.int8]
    assert weight.dtype in [torch.float16, torch.int8]

    # Get dimensions
    n, k = x.shape
    m, _ = weight.shape

    # Allocate output tensor
    out = torch.empty((n, m), dtype=torch.float16, device=x.device)

    # Convert to pointers
    x_ptr = x.data_ptr()
    weight_ptr = weight.data_ptr()
    out_ptr = out.data_ptr()
    rotary_ptr = rotary_emb.data_ptr() if rotary_emb is not None else 0

    # Define block size
    BLOCK_SIZE = 128  # Example block size, can be tuned

    # Launch kernel
    grid = (n // BLOCK_SIZE, m // BLOCK_SIZE)
    rms_matmul_rbe_kernel[grid](
        x_ptr, weight_ptr, out_ptr, rotary_ptr,
        n, m, k,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        apply_rotary,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out
