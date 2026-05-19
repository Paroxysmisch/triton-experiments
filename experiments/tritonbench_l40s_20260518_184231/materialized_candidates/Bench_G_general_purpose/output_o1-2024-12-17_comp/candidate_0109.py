import triton
import triton.language as tl


@triton.jit
def prev_multiple_of(x, y):
    return x - (x % y)


@triton.jit
def softmax_kernel_online_v2(
    input_ptr, 
    output_ptr,
    M, 
    N, 
    TILE_N,
    stride_in_m, 
    stride_in_n, 
    stride_out_m, 
    stride_out_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    row_idx = tl.program_id(0)
    # Guard against out-of-bounds in the row dimension
    if row_idx >= M:
        return

    # 1) Compute row-wise maximum in an online fashion
    row_max = float('-inf')
    boundary_n = prev_multiple_of(N, TILE_N)
    col_off = 0
    while col_off < boundary_n:
        cols = col_off + tl.arange(0, TILE_N)
        # Load input for this tile
        inp = tl.load(
            input_ptr + row_idx * stride_in_m + cols * stride_in_n,
            mask=cols < N,
            other=float('-inf')
        )
        cur_max = tl.max(inp, axis=0)
        row_max = tl.maximum(row_max, cur_max)
        col_off += TILE_N

    # Handle leftover elements if any
    if boundary_n < N:
        leftover_cols = boundary_n + tl.arange(0, N - boundary_n)
        inp = tl.load(
            input_ptr + row_idx * stride_in_m + leftover_cols * stride_in_n,
            mask=leftover_cols < N,
            other=float('-inf')
        )
        cur_max = tl.max(inp, axis=0)
        row_max = tl.maximum(row_max, cur_max)

    # 2) Compute the sum of exponentials using the row-wise maximum (for numerical stability)
    exp_sum = 0.0
    col_off = 0
    while col_off < boundary_n:
        cols = col_off + tl.arange(0, TILE_N)
        inp = tl.load(
            input_ptr + row_idx * stride_in_m + cols * stride_in_n,
            mask=cols < N,
            other=0.0
        )
        # Shift by row_max for stability
        exp_val = tl.exp(inp - row_max)
        exp_sum += tl.sum(exp_val, axis=0)
        col_off += TILE_N

    if boundary_n < N:
        leftover_cols = boundary_n + tl.arange(0, N - boundary_n)
        inp = tl.load(
            input_ptr + row_idx * stride_in_m + leftover_cols * stride_in_n,
            mask=leftover_cols < N,
            other=0.0
        )
        exp_val = tl.exp(inp - row_max)
        exp_sum += tl.sum(exp_val, axis=0)

    # 3) Write out the final normalized values
    col_off = 0
    while col_off < boundary_n:
        cols = col_off + tl.arange(0, TILE_N)
        inp = tl.load(
            input_ptr + row_idx * stride_in_m + cols * stride_in_n,
            mask=cols < N,
            other=0.0
        )
        exp_val = tl.exp(inp - row_max)
        out_val = exp_val / exp_sum
        tl.store(
            output_ptr + row_idx * stride_out_m + cols * stride_out_n,
            out_val,
            mask=cols < N
        )
        col_off += TILE_N

    if boundary_n < N:
        leftover_cols = boundary_n + tl.arange(0, N - boundary_n)
        inp = tl.load(
            input_ptr + row_idx * stride_in_m + leftover_cols * stride_in_n,
            mask=leftover_cols < N,
            other=0.0
        )
        exp_val = tl.exp(inp - row_max)
        out_val = exp_val / exp_sum
        tl.store(
            output_ptr + row_idx * stride_out_m + leftover_cols * stride_out_n,
            out_val,
            mask=leftover_cols < N
        )


def softmax(input_tensor, TILE_N=128):
    import torch
    assert input_tensor.dim() == 2, "Input must be a 2D tensor"
    M, N = input_tensor.shape
    output_tensor = torch.empty_like(input_tensor)

    # Strides for row-major layout
    stride_in_m = input_tensor.stride(0)
    stride_in_n = input_tensor.stride(1)
    stride_out_m = output_tensor.stride(0)
    stride_out_n = output_tensor.stride(1)

    grid = (M,)
    softmax_kernel_online_v2[grid](
        input_tensor, 
        output_tensor,
        M, 
        N, 
        TILE_N,
        stride_in_m, 
        stride_in_n,
        stride_out_m, 
        stride_out_n,
        BLOCK_M=1,
        BLOCK_N=TILE_N
    )
    return output_tensor
