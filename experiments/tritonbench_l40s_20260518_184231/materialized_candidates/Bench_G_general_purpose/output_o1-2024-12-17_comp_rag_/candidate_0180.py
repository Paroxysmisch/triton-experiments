import torch
import triton
import triton.language as tl

# ------------------------------------------------------------------------------
# Heuristic functions to determine optimal tile sizes based on input dimensions
# ------------------------------------------------------------------------------
def heur_tile_k(M, N, K):
    # Simple heuristic that returns a power-of-two size based on K
    # More sophisticated logic can be used to balance resource usage
    return min(triton.next_power_of_2(K), 1024)

def heur_tile_n_non_inner(M, N, K):
    # Heuristic for non-inner dimension tiling in the N dimension
    return min(triton.next_power_of_2(N), 1024)

def heur_tile_n_inner(M, N, K):
    # Heuristic for inner dimension tiling in the N dimension
    return min(triton.next_power_of_2(N), 1024)

# ------------------------------------------------------------------------------
# Forward pass kernels for softmax
# ------------------------------------------------------------------------------

@triton.jit
def softmax_kernel_non_inner(
    output_ptr, input_ptr,
    M, N, K,
    stride_m, stride_n,
    TILE_K: tl.constexpr, TILE_N: tl.constexpr, ONE_TILE_PER_CTA: tl.constexpr
):
    pid = tl.program_id(0)
    # Each program processes a tile in the (K, N) space
    # For non-inner dimensions, we iterate over the K dimension
    k_offset = pid * TILE_K if ONE_TILE_PER_CTA else tl.program_id(1) * TILE_K
    n_offset = tl.program_id(1) * TILE_N if ONE_TILE_PER_CTA else tl.program_id(2) * TILE_N

    k_range = k_offset + tl.arange(0, TILE_K)
    n_range = n_offset + tl.arange(0, TILE_N)
    k_mask = k_range < K
    n_mask = n_range < N

    # Create pointers
    input_offset = (k_range[:, None] * stride_m) + (n_range[None, :] * stride_n)
    output_offset = input_offset  # same shape

    # Load data
    mask = k_mask[:, None] & n_mask[None, :]
    data = tl.load(input_ptr + input_offset, mask=mask, other=-float('inf'))

    # Subtract max
    row_max = tl.max(data, 1)
    data = data - row_max[:, None]

    # Exponential
    data_exp = tl.exp(data)
    data_sum = tl.sum(data_exp, 1)

    # Softmax
    data_out = data_exp / data_sum[:, None]

    # Store
    tl.store(output_ptr + output_offset, data_out, mask=mask)


@triton.jit
def softmax_kernel_inner(
    output_ptr, input_ptr,
    M, N, K,
    stride_m, stride_n,
    TILE_N: tl.constexpr
):
    # This kernel computes softmax over the last dimension (inner dimension)
    # We parallelize across the 'M' dimension
    row_id = tl.program_id(0)
    col_offsets = tl.arange(0, TILE_N)
    row_offset = row_id * stride_m
    col_ptrs = input_ptr + row_offset + col_offsets * stride_n

    mask = col_offsets < N
    row = tl.load(col_ptrs, mask=mask, other=-float('inf'))

    row_max = tl.max(row, axis=0)
    row_minus_max = row - row_max
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    output = numerator / denominator

    out_ptrs = output_ptr + row_offset + col_offsets * stride_n
    tl.store(out_ptrs, output, mask=mask)


# ------------------------------------------------------------------------------
# Backward pass kernels for softmax
# ------------------------------------------------------------------------------

@triton.jit
def softmax_backward_kernel_non_inner(
    in_grad_ptr, out_ptr, grad_out_ptr,
    M, N, K,
    stride_m, stride_n,
    TILE_K: tl.constexpr, TILE_N: tl.constexpr, ONE_TILE_PER_CTA: tl.constexpr
):
    pid = tl.program_id(0)
    k_offset = pid * TILE_K if ONE_TILE_PER_CTA else tl.program_id(1) * TILE_K
    n_offset = tl.program_id(1) * TILE_N if ONE_TILE_PER_CTA else tl.program_id(2) * TILE_N

    k_range = k_offset + tl.arange(0, TILE_K)
    n_range = n_offset + tl.arange(0, TILE_N)
    k_mask = k_range < K
    n_mask = n_range < N

    # Create pointers
    out_offset = (k_range[:, None] * stride_m) + (n_range[None, :] * stride_n)
    grad_out_offset = out_offset
    in_grad_offset = out_offset

    # Load forward output and grad output
    mask = k_mask[:, None] & n_mask[None, :]
    y = tl.load(out_ptr + out_offset, mask=mask, other=0.0)
    dy = tl.load(grad_out_ptr + grad_out_offset, mask=mask, other=0.0)

    # Compute dot(y, dy)
    dot = tl.sum(y * dy, 1)

    # dx = y * (dy - dot(y, dy))
    dx = y * (dy - dot[:, None])
    tl.store(in_grad_ptr + in_grad_offset, dx, mask=mask)


@triton.jit
def softmax_backward_kernel_inner(
    in_grad_ptr, out_ptr, grad_out_ptr,
    M, N, K,
    stride_m, stride_n,
    TILE_N: tl.constexpr
):
    row_id = tl.program_id(0)
    col_offsets = tl.arange(0, TILE_N)
    row_offset = row_id * stride_m
    out_ptrs = out_ptr + row_offset + col_offsets * stride_n
    dy_ptrs = grad_out_ptr + row_offset + col_offsets * stride_n

    mask = col_offsets < N
    y = tl.load(out_ptrs, mask=mask, other=0.0)
    dy = tl.load(dy_ptrs, mask=mask, other=0.0)

    dot = tl.sum(y * dy, axis=0)
    dx = y * (dy - dot)

    in_grad_ptrs = in_grad_ptr + row_offset + col_offsets * stride_n
    tl.store(in_grad_ptrs, dx, mask=mask)


# ------------------------------------------------------------------------------
# Softmax Autograd Function
# ------------------------------------------------------------------------------
class Softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, K: int = 1):
        """
        x:  [M, N] tensor
        K:  dimension size to differentiate between inner / non-inner
        """
        M, N = x.shape
        x_out = torch.empty_like(x)

        # Heuristic determinations
        tile_k = heur_tile_k(M, N, K)
        tile_n_non_inner = heur_tile_n_non_inner(M, N, K)
        tile_n_inner = heur_tile_n_inner(M, N, K)

        # Save tensors for backward
        ctx.save_for_backward(x_out)
        ctx.x_shape = (M, N)
        ctx.K_dim = K

        if K > 1:
            # Launch softmax_kernel_non_inner
            grid = lambda META: (triton.cdiv(K, META['TILE_K']) * triton.cdiv(N, META['TILE_N']),)
            softmax_kernel_non_inner[grid](
                x_out, x,
                M, N, K,
                x.stride(0), x.stride(1),
                TILE_K=tile_k, TILE_N=tile_n_non_inner,
                ONE_TILE_PER_CTA=True
            )
        else:
            # Launch softmax_kernel_inner
            grid = lambda META: (M,)
            softmax_kernel_inner[grid](
                x_out, x,
                M, N, K,
                x.stride(0), x.stride(1),
                TILE_N=tile_n_inner
            )
        return x_out

    @staticmethod
    def backward(ctx, grad_output):
        (x_out,) = ctx.saved_tensors
        M, N = ctx.x_shape
        K = ctx.K_dim

        dx = torch.empty_like(grad_output)

        tile_k = heur_tile_k(M, N, K)
        tile_n_non_inner = heur_tile_n_non_inner(M, N, K)
        tile_n_inner = heur_tile_n_inner(M, N, K)

        if K > 1:
            grid = lambda META: (triton.cdiv(K, META['TILE_K']) * triton.cdiv(N, META['TILE_N']),)
            softmax_backward_kernel_non_inner[grid](
                dx, x_out, grad_output,
                M, N, K,
                x_out.stride(0), x_out.stride(1),
                TILE_K=tile_k, TILE_N=tile_n_non_inner,
                ONE_TILE_PER_CTA=True
            )
        else:
            grid = lambda META: (M,)
            softmax_backward_kernel_inner[grid](
                dx, x_out, grad_output,
                M, N, K,
                x_out.stride(0), x_out.stride(1),
                TILE_N=tile_n_inner
            )
        return dx, None

# ------------------------------------------------------------------------------
# Simple wrapper function to invoke the Softmax forward/backward
# ------------------------------------------------------------------------------
def softmax(x, K=1):
    return Softmax.apply(x, K)
