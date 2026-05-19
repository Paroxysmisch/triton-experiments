import triton
import triton.language as tl

@triton.jit
def _compute_pairwise_distance(x1_ptr, x2_ptr, dist_ptr, N, D, x1_numel, x2_numel, p_distance, eps_distance, XBLOCK: tl.constexpr, YBLOCK: tl.constexpr):
    x1_offset = tl.program_id(0) * XBLOCK
    x1_indices = tl.arange(0, XBLOCK)
    x1_mask = x1_offset + x1_indices < x1_numel

    x2_offset = tl.program_id(1) * YBLOCK
    x2_indices = tl.arange(0, YBLOCK)
    x2_mask = x2_offset + x2_indices < x2_numel

    x1_data = tl.load(x1_ptr + x1_offset + x1_indices[:, None], x1_mask, eviction_policy="evict_last", other=0)
    x2_data = tl.load(x2_ptr + x2_offset + x2_indices, x2_mask, eviction_policy="evict_last", other=0)

    diff = x1_data - x2_data
    diff_abs = tl.abs(diff)
    diff_pow = diff_abs ** p_distance
    dist = tl.sum(diff_pow, axis=0)
    dist_sqrt = tl.sqrt(dist + eps_distance)
    tl.store(dist_ptr + x1_offset + x1_indices[:, None], dist_sqrt, x1_mask)

@triton.jit
def _normalize_distances(dist_ptr, max_norm_ptr, N, D, x1_numel, dim_norm, eps_norm, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    i = tl.program_id(0) * XBLOCK
    i_mask = i < N

    i_indices = tl.arange(0, XBLOCK)
    i_mask_expanded = i_indices[:, None] < D

    dist_values = tl.load(dist_ptr + i + i_indices[:, None], i_mask_expanded, eviction_policy="evict_last", other=0)
    max_norm_value = tl.max(dist_values, axis=0)
    tl.store(max_norm_ptr + i, max_norm_value, i_mask)

    normalized_dist = dist_values / (max_norm_value + eps_norm)
    tl.store(dist_ptr + i + i_indices[:, None], normalized_dist, i_mask_expanded)

@triton.jit
def _apply_keepdim(normalized_dist_ptr, N, D, x1_numel, dim_norm, keepdim, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    i = tl.program_id(0) * XBLOCK
    i_mask = i < N

    i_indices = tl.arange(0, XBLOCK)
    i_mask_expanded = i_indices[:, None] < D

    normalized_dist_values = tl.load(normalized_dist_ptr + i + i_indices[:, None], i_mask_expanded, eviction_policy="evict_last", other=0)

    if not keepdim:
        new_shape = normalized_dist_values.shape[:dim_norm] + normalized_dist_values.shape[dim_norm+1:]
        normalized_dist_values = normalized_dist_values.reshape(new_shape)

    tl.store(normalized_dist_ptr + i + i_indices[:, None], normalized_dist_values, i_mask_expanded)

@triton.jit
def normalize_pairwise_distance_kernel(x1_ptr, x2_ptr, normalized_dist_ptr, N, D, x1_numel, x2_numel, p_distance, eps_distance, dim_norm, eps_norm, keepdim, XBLOCK: tl.constexpr, YBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    _compute_pairwise_distance(x1_ptr, x2_ptr, normalized_dist_ptr, N, D, x1_numel, x2_numel, p_distance, eps_distance, XBLOCK, YBLOCK)
    max_norm_ptr = tl.tensor([0.0] * N, dtype=tl.float32)
    _normalize_distances(normalized_dist_ptr, max_norm_ptr, N, D, x1_numel, dim_norm, eps_norm, XBLOCK, RBLOCK)
    _apply_keepdim(normalized_dist_ptr, N, D, x1_numel, dim_norm, keepdim, XBLOCK, RBLOCK, RBLOCK)

class NormalizePairwiseDistance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, p_distance=2.0, eps_distance=1e-6, keepdim=False, p_norm=2, dim_norm=1, eps_norm=1e-12):
        assert x1.shape == x2.shape, "Input tensors must have the same shape"
        
        N, D = x1.shape
        normalized_dist = torch.empty_like(x1)

        x1_numel = N * D
        x2_numel = N * D
        
        XBLOCK = min(triton.next_power_of_2(N), triton.config.default_block_size)
        YBLOCK = min(triton.next_power_of_2(D), triton.config.default_block_size)
        RBLOCK = min(triton.next_power_of_2(D), triton.config.default_block_size)

        nb_blocks_x = 1 + (N - 1) // XBLOCK
        nb_blocks_y = 1 + (D - 1) // YBLOCK
        g = (nb_blocks_x, nb_blocks_y, 1)

        normalize_pairwise_distance_kernel[g](
            x1,
            x2,
            normalized_dist,
            N,
            D,
            x1_numel,
            x2_numel,
            p_distance,
            eps_distance,
            dim_norm,
            eps_norm,
            keepdim,
            XBLOCK,
            YBLOCK,
            RBLOCK
        )

        ctx.save_for_backward(x1, x2)
        ctx.p_distance = p_distance
        ctx.eps_distance = eps_distance
        ctx.keepdim = keepdim
        ctx.p_norm = p_norm
        ctx.dim_norm = dim_norm
        ctx.eps_norm = eps_norm

        return normalized_dist

    @staticmethod
    def backward(ctx, grad_output):
        x1, x2 = ctx.saved_tensors
        N, D = x1.shape

        grad_x1 = torch.zeros_like(x1)
        grad_x2 = torch.zeros_like(x2)

        p_distance = ctx.p_distance
        eps_distance = ctx.eps_distance
        keepdim = ctx.keepdim
        p_norm = ctx.p_norm
        dim_norm = ctx.dim_norm
        eps_norm = ctx.eps_norm

        x1_numel = N * D
        x2_numel = N * D

        XBLOCK = min(triton.next_power_of_2(N), triton.config.default_block_size)
        YBLOCK = min(triton.next_power_of_2(D), triton.config.default_block_size)
        RBLOCK = min(triton.next_power_of_2(D), triton.config.default_block_size)

        nb_blocks_x = 1 + (N - 1) // XBLOCK
        nb_blocks_y = 1 + (D - 1) // YBLOCK
        g = (nb_blocks_x, nb_blocks_y, 1)

        # Implement the backward pass here

        return grad_x1, grad_x2, None, None, None, None, None, None

def normalize_pairwise_distance(x1, x2, p_distance=2.0, eps_distance=1e-6, keepdim=False, p_norm=2, dim_norm=1, eps_norm=1e-12):
    return NormalizePairwiseDistance.apply(x1, x2, p_distance, eps_distance, keepdim, p_norm, dim_norm, eps_norm)
