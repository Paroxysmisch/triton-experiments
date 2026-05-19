import torch
import triton
import triton.language as tl


@triton.jit
def _log_softmax_linear_fwd_kernel(
    x_ptr,       # [B, I]
    w_ptr,       # [O, I]
    b_ptr,       # [O] or None
    y_ptr,       # [B, O]
    B,           # batch size
    I,           # in_features
    O,           # out_features
    HAS_BIAS: tl.constexpr,  # bool
    BLOCK_SIZE: tl.constexpr # width of the tile in out_features dimension
):
    # Program ids
    batch_id = tl.program_id(0)
    # This kernel processes one row (batch_id) at a time
    # Load indices for output features
    offs_o = tl.arange(0, BLOCK_SIZE)
    # Check valid row in batch
    row_valid = batch_id < B
    # Check valid columns for out_features
    col_valid = offs_o < O

    # Initial output accumulator
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32) - float('inf')

    # Base pointers for input row, weight, and output row
    x_row_ptr = x_ptr + batch_id * I
    # We'll accumulate partial sums of x*w^T using tile loops over I
    # Iterate over in_features in steps of 1; unrolled is for clarity
    # (In practice you'd tile over in_features for performance.)
    for i_off in range(I):
        w_col = tl.load(w_ptr + offs_o * I + i_off, mask=col_valid, other=0.0)
        x_val = tl.load(x_row_ptr + i_off, mask=row_valid, other=0.0)
        # Each out_features element accumulates x_val * w_col
        acc = tl.where(col_valid, tl.where(row_valid, acc, 0.0) + x_val * w_col, acc)

    # Add bias if present
    if HAS_BIAS:
        b_val = tl.load(b_ptr + offs_o, mask=col_valid, other=0.0)
        acc = tl.where(col_valid & row_valid, acc + b_val, acc)

    # Compute max for numerical stability
    # (reduce over BLOCK_SIZE dimension)
    max_elem = tl.max(acc, 0)
    # Sum exp
    sum_exp = tl.sum(tl.exp(acc - max_elem), 0)
    # Log softmax
    log_softmax_res = acc - max_elem - tl.log(sum_exp)

    # Store result
    out_row_ptr = y_ptr + batch_id * O
    tl.store(out_row_ptr + offs_o, log_softmax_res, mask=(col_valid & row_valid))


def log_softmax_linear(input, weight, bias=None, dim=-1, dtype=None):
    # Ensure dim == -1 for this simplified kernel
    if dim != -1:
        raise ValueError("Only dim = -1 is supported by this Triton kernel prototype.")
    # Cast input if dtype is specified
    if dtype is not None:
        input = input.to(dtype)
    # Shapes
    in_shape = input.shape
    B = 1
    for s in in_shape[:-1]:
        B *= s
    I = in_shape[-1]
    W_shape = weight.shape
    O = W_shape[0]

    # Flatten input to [B, I]
    x = input.contiguous().view(B, I)
    w = weight.contiguous()  # [O, I]
    has_bias = (bias is not None)
    if has_bias:
        b = bias.contiguous()
    else:
        b = torch.empty(1, device=x.device, dtype=x.dtype)

    # Allocate output
    y = torch.empty((B, O), device=x.device, dtype=x.dtype)
    
    # Define block size in out_features dimension
    BLOCK_SIZE = triton.next_power_of_2(O) if O > 1 else 1
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)  # clamp for safety
    
    # Launch kernel (one block per batch row)
    grid = (B,)
    _log_softmax_linear_fwd_kernel[grid](
        x, w, b, y,
        B, I, O,
        HAS_BIAS=has_bias,
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Reshape output back
    out = y.view(*in_shape[:-1], O)
    return out
