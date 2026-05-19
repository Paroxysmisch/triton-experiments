import triton
import triton.language as tl

@triton.jit
def mean_dim_kernel(X, Mean, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    col_start = 0

    # Compute the range of rows and columns for this block
    row_end = min(row_start + BLOCK_M, M)
    col_end = min(col_start + BLOCK_N, N)

    # Initialize the sum and count
    _sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    count = 0

    # Iterate over the columns in this block
    for col in range(col_start, col_end):
        # Load a block of elements from X
        x_block = tl.load(X + row_start * N + col, mask=row_start + tl.arange(0, BLOCK_M) < row_end, other=0.0)
        _sum += x_block
        count += 1

    # Compute the mean
    _mean = _sum / count

    # Store the result in Mean
    tl.store(Mean + row_start, _mean, mask=row_start + tl.arange(0, BLOCK_M) < row_end)

import torch
import triton
import triton.language as tl

def dim_compress(inp, dims):
    # Rearrange the input tensor dimensions for efficient memory access
    perm = [i for i in range(len(inp.shape)) if i not in dims] + dims
    permuted = inp.permute(perm)
    new_shape = [permuted.shape[i] for i in range(len(perm)) if i < len(perm) - len(dims)]
    new_shape.append(-1)
    return permuted.reshape(new_shape)

def mean_dim(x, dim, keepdim=False, BLOCK_M=128, BLOCK_N=128):
    # Convert the input tensor to a format compatible with the kernel
    if isinstance(dim, int):
        dim = [dim]
    permuted = dim_compress(x, dim)
    M = permuted.shape[0]
    N = permuted.shape[-1]

    # Create an empty output tensor
    out_shape = list(permuted.shape[:-1])
    if not keepdim:
        out_shape = [s for i, s in enumerate(out_shape) if i not in dim]
    out = torch.empty(out_shape, device=x.device, dtype=x.dtype)

    # Launch the kernel
    grid = (triton.cdiv(M, BLOCK_M),)
    mean_dim_kernel[grid](permuted, out, M, N, BLOCK_M, BLOCK_N)

    # Squeeze the reduced dimensions if keepdim is False
    if not keepdim:
        out = out.squeeze(dim)

    return out
