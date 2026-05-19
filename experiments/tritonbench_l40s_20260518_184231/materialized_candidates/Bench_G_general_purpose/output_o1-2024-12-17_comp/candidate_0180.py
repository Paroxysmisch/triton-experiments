import triton
import triton.language as tl
import torch

@triton.jit
def _l2_norm_fwd_1pass_kernel(
    x_ptr, 
    y_ptr, 
    stride_x_row, 
    N, 
    eps, 
    BLOCK_N: tl.constexpr
):
    row_id = tl.program_id(0)
    offsets = row_id * stride_x_row + tl.arange(0, BLOCK_N)
    mask = tl.arange(0, BLOCK_N) < N

    x_data = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    sq_vals = x_data * x_data
    sum_sq = tl.sum(sq_vals, axis=0)
    rstd = 1.0 / tl.sqrt(sum_sq + eps)
    y_data = x_data * rstd
    tl.store(y_ptr + offsets, y_data, mask=mask)

def _l2_norm_fwd(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    original_shape = x.shape
    if x.dim() != 2:
        x = x.view(-1, x.size(-1))
    x = x.contiguous()
    M, N = x.shape

    BLOCK_SIZE_BYTES = 64 * 1024
    dtype_size = x.element_size()
    BLOCK_N = BLOCK_SIZE_BYTES // dtype_size
    if N > BLOCK_N:
        raise ValueError(f"Feature dimension too large: N={N} > BLOCK_N={BLOCK_N}")

    y = torch.empty_like(x)
    grid = (M,)
    _l2_norm_fwd_1pass_kernel[grid](
        x, 
        y, 
        x.stride(0), 
        N, 
        eps, 
        BLOCK_N=BLOCK_N
    )
    return y.view(*original_shape)
