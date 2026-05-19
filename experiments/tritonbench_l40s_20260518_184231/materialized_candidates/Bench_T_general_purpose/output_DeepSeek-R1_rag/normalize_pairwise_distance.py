import torch
import triton
import triton.language as tl

@triton.jit
def _pairwise_distance_forward(
    x1_ptr,
    x2_ptr,
    distance_ptr,
    N,
    D,
    p_distance,
    eps_distance,
    xnumel,
    rnumel,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel

    row_i = xindex // N
    row_j = xindex % N

    x1_row_start = row_i * D
    x2_row_start = row_j * D

    accumulator = tl.zeros((XBLOCK,), dtype=tl.float32)

    for k in range(0, D, RBLOCK):
        k_offsets = k + tl.arange(0, RBLOCK)
        k_mask = k_offsets < D

        x1_ptrs = x1_row_start[:, None] + k_offsets[None, :]
        x2_ptrs = x2_row_start[:, None] + k_offsets[None, :]

        x1_val = tl.load(x1_ptr + x1_ptrs, mask=k_mask & xmask[:, None], other=0.0)
        x2_val = tl.load(x2_ptr + x2_ptrs, mask=k_mask & xmask[:, None], other=0.0)

        diff = x1_val - x2_val
        diff_abs = tl.abs(diff)
        diff_p = tl.math.pow(diff_abs, p_distance)
        accumulator += tl.sum(diff_p, axis=1)

    distance_val = tl.math.pow(accumulator, 1.0 / p_distance) + eps_distance
    tl.store(distance_ptr + xindex, distance_val, mask=xmask)

class NormalizePairwiseDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, p_distance, eps_distance, keepdim, p_norm, dim_norm, eps_norm):
        assert x1.shape == x2.shape, "x1 and x2 must have the same shape"
        N, D = x1.shape
        distance = torch.empty((N, N), device=x1.device, dtype=x1.dtype)
        xnumel = N * N
        rnumel = D

        XBLOCK = 1024
        RBLOCK = 128
        grid = lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)

        _pairwise_distance_forward[grid](
            x1, x2, distance,
            N, D,
            p_distance,
            eps_distance,
            xnumel, rnumel,
            XBLOCK=XBLOCK,
            RBLOCK=RBLOCK,
        )

        norm = distance.norm(p=p_norm, dim=dim_norm, keepdim=keepdim).clamp_min(eps_norm)
        normalized = distance / norm

        ctx.save_for_backward(x1, x2, distance, norm)
        ctx.p_distance = p_distance
        ctx.eps_distance = eps_distance
        ctx.p_norm = p_norm
        ctx.dim_norm = dim_norm
        ctx.eps_norm = eps_norm
        ctx.keepdim = keepdim

        return normalized

    @staticmethod
    def backward(ctx, grad_output):
        x1, x2, distance, norm = ctx.saved_tensors
        grad_output = grad_output.clone()

        distance.requires_grad_(True)
        with torch.enable_grad():
            norm_recomp = distance.norm(p=ctx.p_norm, dim=ctx.dim_norm, keepdim=ctx.keepdim).clamp_min(ctx.eps_norm)
            normalized_recomp = distance / norm_recomp
            grad_distance = torch.autograd.grad(normalized_recomp, distance, grad_output, retain_graph=True)[0]

        grad_x1 = torch.zeros_like(x1)
        grad_x2 = torch.zeros_like(x2)

        N, D = x1.shape
        xnumel = N * D
        rnumel = N

        XBLOCK = 1024
        RBLOCK = 128
        grid = lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)

        return grad_x1, grad_x2, None, None, None, None, None, None

def normalize_pairwise_distance(x1, x2, p_distance=2.0, eps_distance=1e-6, keepdim=False, p_norm=2, dim_norm=1, eps_norm=1e-12):
    return NormalizePairwiseDistanceFunction.apply(x1, x2, p_distance, eps_distance, keepdim, p_norm, dim_norm, eps_norm)
