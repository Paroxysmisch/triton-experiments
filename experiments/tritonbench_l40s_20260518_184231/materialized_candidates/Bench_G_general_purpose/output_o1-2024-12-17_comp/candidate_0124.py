import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w)
        for w in [1, 2, 4, 8, 16, 32]
    ],
    key=['N', 'D']
)
@triton.jit
def logsumexp_fwd_kernel(
    x_ptr,        # *Pointer to input x
    scale_ptr,    # *Pointer to scale tensor or dummy pointer (when HAS_SCALE=0)
    z_ptr,        # *Pointer to output buffer for partial results
    N, D, B,      # Dimensions and block size
    stride_xn, stride_xd,
    stride_sn, stride_sd,
    stride_zn, stride_zd,
    HAS_SCALE: tl.constexpr
):
    # Program IDs
    i_n = tl.program_id(0)
    i_nd = tl.program_id(1)

    # Offset in the D dimension
    d_offset = tl.arange(0, B)
    d = i_nd * B + d_offset
    mask = d < D

    # Load x
    x = tl.load(
        x_ptr + i_n * stride_xn + d * stride_xd,
        mask=mask,
        other=-float('inf')
    )

    # Optionally load scale and apply
    if HAS_SCALE:
        s = tl.load(
            scale_ptr + i_n * stride_sn + d * stride_sd,
            mask=mask,
            other=1.0
        )
        x = x * s

    # Get block-wise max
    block_max = tl.max(x, axis=0)
    # Compute exponentials relative to max
    x = tl.exp(x - block_max)
    # Sum and take log
    block_sum = tl.sum(x, axis=0)
    block_lse = tl.log(block_sum) + block_max

    # Store partial result
    tl.store(
        z_ptr + i_n * stride_zn + i_nd * stride_zd,
        block_lse
    )


def logsumexp_fwd(x, scale=None, out_dtype=None, B=128):
    """
    Compute logsumexp over the last dimension of x using Triton.
    x is expected to have shape [N, D].
    """
    # Shape
    N, D = x.shape
    ND = (D + B - 1) // B  # number of blocks along D

    # Allocate output buffer [N, ND] for partial block results
    z = torch.empty((N, ND), device=x.device, dtype=x.dtype)

    # Strides
    stride_xn = x.stride(0)
    stride_xd = x.stride(1)
    if scale is not None:
        stride_sn = scale.stride(0)
        stride_sd = scale.stride(1)
    else:
        stride_sn = 0
        stride_sd = 0
    stride_zn = z.stride(0)
    stride_zd = z.stride(1)

    # Kernel call
    HAS_SCALE = 1 if scale is not None else 0
    grid = (N, ND)
    logsumexp_fwd_kernel[grid](
        x, scale if scale is not None else x, z,
        N, D, B,
        stride_xn, stride_xd,
        stride_sn, stride_sd,
        stride_zn, stride_zd,
        HAS_SCALE=HAS_SCALE
    )

    # Reduce partial results along ND dimension
    # Each block in z now holds a log-sum-exp for a portion of the D-dim.
    # We can combine block-wise log-sum-exp values properly.
    # 1) Find global max to stabilize
    block_max_vals, _ = torch.max(z, dim=-1, keepdim=True)
    # 2) Convert block logs back to exponentials (relative to block max), then sum
    exp_vals = torch.exp(z - block_max_vals)
    sum_vals = torch.sum(exp_vals, dim=-1)
    # 3) Final log-sum-exp
    lse_vals = torch.log(sum_vals) + block_max_vals.squeeze(-1)
    if out_dtype is not None:
        lse_vals = lse_vals.to(out_dtype)
    return lse_vals
