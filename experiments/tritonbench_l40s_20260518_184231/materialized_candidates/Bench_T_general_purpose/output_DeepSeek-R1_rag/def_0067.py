import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def _pairwise_distance_forward(
    x1_ptr, x2_ptr, dist_ptr, N, M, D, xnumel, rnumel, p, eps,
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)
    x_mask = x_index < xnumel  # xnumel = N * M

    i = x_index // M
    j = x_index % M

    x1_row = i * D
    x2_row = j * D

    acc = tl.zeros((XBLOCK,), tl.float32)

    for roffset in range(0, rnumel, RBLOCK):
        r_index = roffset + tl.arange(0, RBLOCK)
        r_mask = r_index < D

        x1_vals = tl.load(x1_ptr + x1_row + r_index, mask=r_mask & x_mask, other=0.0)
        x2_vals = tl.load(x2_ptr + x2_row + r_index, mask=r_mask & x_mask, other=0.0)

        diff = x1_vals - x2_vals
        abs_diff = tl.abs(diff)
        pow_diff = tl.math.powf(abs_diff, p)
        acc += pow_diff * tl.where(r_mask, 1.0, 0.0)

    sum_pow = acc + eps
    distance = tl.math.powf(sum_pow, 1.0 / p)
    tl.store(dist_ptr + x_index, distance, mask=x_mask)

@triton.jit
def _pairwise_distance_backward(
    grad_dist_ptr, grad_x1_ptr, grad_x2_ptr,
    x1_ptr, x2_ptr, N, M, D, xnumel, rnumel, p, eps,
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)
    x_mask = x_index < xnumel  # xnumel = N * D for x1, M * D for x2

    i = x_index // D
    k = x_index % D

    grad_x1_acc = tl.zeros((XBLOCK,), tl.float32)
    grad_x2_acc = tl.zeros((XBLOCK,), tl.float32)

    for j_offset in range(0, M, RBLOCK):
        j = j_offset + tl.arange(0, RBLOCK)
        j_mask = j < M

        x1_val = tl.load(x1_ptr + i * D + k, mask=x_mask, other=0.0)
        x2_val = tl.load(x2_ptr + j * D + k, mask=j_mask, other=0.0)

        diff = x1_val - x2_val
        abs_diff = tl.abs(diff)
        sum_pow = tl.load(
            (i * M + j).astype(tl.int64),
            mask=j_mask & (i < N),
            other=0.0
        )
        sum_pow_val = tl.math.powf(sum_pow, p)  # Recompute sum_pow from forward

        grad_coeff = tl.math.powf(sum_pow_val + eps, (1.0 / p) - 1.0)
        grad_dist = tl.load(grad_dist_ptr + i * M + j, mask=j_mask & (i < N), other=0.0)

        term = grad_coeff * tl.math.powf(abs_diff, p - 1.0) * tl.sign(diff) * grad_dist
        grad_x1_acc += tl.sum(term * tl.where(j_mask, 1.0, 0.0), axis=0)
        grad_x2_acc -= tl.sum(term * tl.where(j_mask, 1.0, 0.0), axis=0)

    tl.store(grad_x1_ptr + x_index, grad_x1_acc, mask=x_mask)
    tl.store(grad_x2_ptr + x_index, grad_x2_acc, mask=x_mask)

class PairwiseDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, p, eps):
        N, D = x1.shape
        M, D2 = x2.shape
        assert D == D2, "Feature dimensions must match"

        ctx.save_for_backward(x1, x2)
        ctx.p = p
        ctx.eps = eps

        dist = torch.empty((N, M), device=x1.device, dtype=x1.dtype)
        xnumel = N * M
        rnumel = D

        max_xblock = 1024
        XBLOCK = min(triton.next_power_of_2(xnumel), max_xblock)
        RBLOCK = 128

        grid = (triton.cdiv(xnumel, XBLOCK),)
        _pairwise_distance_forward[grid](
            x1, x2, dist, N, M, D, xnumel, rnumel, p, eps,
            XBLOCK=XBLOCK, RBLOCK=RBLOCK
        )

        return dist

    @staticmethod
    def backward(ctx, grad_dist):
        x1, x2 = ctx.saved_tensors
        p = ctx.p
        eps = ctx.eps

        N, D = x1.shape
        M, D2 = x2.shape

        grad_x1 = torch.zeros_like(x1)
        grad_x2 = torch.zeros_like(x2)

        xnumel_x1 = N * D
        xnumel_x2 = M * D

        XBLOCK = 1024
        RBLOCK = 128

        grid_x1 = (triton.cdiv(xnumel_x1, XBLOCK),)
        _pairwise_distance_backward[grid_x1](
            grad_dist, grad_x1, grad_x2, x1, x2, N, M, D, xnumel_x1, M, p, eps,
            XBLOCK=XBLOCK, RBLOCK=RBLOCK
        )

        grid_x2 = (triton.cdiv(xnumel_x2, XBLOCK),)
        _pairwise_distance_backward[grid_x2](
            grad_dist, grad_x2, grad_x1, x2, x1, M, N, D, xnumel_x2, N, p, eps,
            XBLOCK=XBLOCK, RBLOCK=RBLOCK
        )

        return grad_x1, grad_x2, None, None

def fused_pairwise_distance_adaptive_avg_pool2d(
    x1: torch.Tensor,
    x2: torch.Tensor,
    output_size: int or tuple,
    p: float = 2.0,
    eps: float = 1e-6,
    keepdim: bool = False
) -> torch.Tensor:
    # Apply adaptive average pooling
    x1_pooled = F.adaptive_avg_pool2d(x1, output_size)
    x2_pooled = F.adaptive_avg_pool2d(x2, output_size)
    
    # Flatten spatial dimensions
    x1_flat = x1_pooled.flatten(start_dim=1)
    x2_flat = x2_pooled.flatten(start_dim=1)
    
    # Compute pairwise distance using Triton function
    dist = PairwiseDistanceFunction.apply(x1_flat, x2_flat, p, eps)
    
    if keepdim:
        dist = dist.unsqueeze(-1)
    
    return dist
