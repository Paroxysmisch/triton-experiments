import triton
import triton.language as tl

@triton.jit
def _fused_add_mul_groupnorm_sum_kernel(
    x_ptr, y_ptr, grp_sum_ptr, grp_sq_sum_ptr,
    B, C, D, group_size, eps,
    # Strides
    stride_xb, stride_xc, stride_xd,
    stride_yb, stride_yc, stride_yd,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr
):
    """
    This kernel computes partial sums and partial squared sums of:
      Z = X + Y
      M = Z * Y
    grouped by (N, group), where group contains 'group_size' consecutive channels.
    We store the sums and squared sums to use in a second pass for normalization.
    """
    pid = tl.program_id(0)
    # Each program handles one block of D (spatial/feature) dimension
    block_start = pid * BLOCK_SIZE
    offs_d = block_start + tl.arange(0, BLOCK_SIZE)
    mask_d = offs_d < D

    # We'll iterate over B * (C // group_size) * group_size channels
    # in a loop to do partial sums at the group level.
    # For stability we do sums in float32.
    for b in range(B):
        for c in range(C):
            # Identify the group index for channel c
            group_idx = c // group_size
            out_idx = b * (C // group_size) + group_idx

            # Load X and Y, apply the fused ops
            x_offset = b * stride_xb + c * stride_xc
            y_offset = b * stride_yb + c * stride_yc
            x_vals = tl.load(x_ptr + x_offset + offs_d * stride_xd, mask=mask_d, other=0.0)
            y_vals = tl.load(y_ptr + y_offset + offs_d * stride_yd, mask=mask_d, other=0.0)

            z_vals = x_vals + y_vals
            m_vals = z_vals * y_vals

            # Accumulate partial sums for M and M^2
            sum_ = tl.sum(m_vals.to(tl.float32), axis=0)
            sq_sum_ = tl.sum((m_vals * m_vals).to(tl.float32), axis=0)

            # Atomic accumulate into grp_sum_ptr, grp_sq_sum_ptr
            tl.atomic_add(grp_sum_ptr + out_idx, sum_)
            tl.atomic_add(grp_sq_sum_ptr + out_idx, sq_sum_)

@triton.jit
def _fused_add_mul_groupnorm_apply_kernel(
    x_ptr, y_ptr, w_ptr, b_ptr, out_ptr,
    grp_mean_ptr, grp_invvar_ptr,
    B, C, D, group_size,
    # Strides
    stride_xb, stride_xc, stride_xd,
    stride_yb, stride_yc, stride_yd,
    stride_w, stride_b,
    stride_ob, stride_oc, stride_od
):
    """
    This kernel finishes the fused operation by:
      Z = X + Y
      M = Z * Y
      then normalizes M using the precomputed mean/invvar from each group,
      and applies the learned weight and bias.
    """
    # Compute global linear index
    pid = tl.program_id(0)
    # Each program handles one element in [B, C, D], in a simplified 1D dispatch
    # for demonstration. For real usage, a 2D or 3D mapping is typical.
    if pid >= B * C * D:
        return

    d = pid % D
    cc = (pid // D) % C
    b = (pid // (D * C))

    # Load x and y
    x_offset = b * stride_xb + cc * stride_xc + d * stride_xd
    y_offset = b * stride_yb + cc * stride_yc + d * stride_yd
    x_val = tl.load(x_ptr + x_offset)
    y_val = tl.load(y_ptr + y_offset)

    # Fused ops
    z_val = x_val + y_val
    m_val = z_val * y_val

    # Group index
    group_idx = cc // group_size
    # Loads group mean and invvar
    mean = tl.load(grp_mean_ptr + b * (C // group_size) + group_idx)
    invvar = tl.load(grp_invvar_ptr + b * (C // group_size) + group_idx)

    # Normalize
    norm_val = (m_val - mean) * invvar

    # Apply weight/bias
    w = tl.load(w_ptr + cc * stride_w)
    b_ = tl.load(b_ptr + cc * stride_b)
    out_val = norm_val * w + b_

    # Store result
    out_offset = b * stride_ob + cc * stride_oc + d * stride_od
    tl.store(out_ptr + out_offset, out_val)


def fused_add_mul_groupnorm(input1, input2, weight, bias, num_groups, eps=1e-5, *, out=None):
    """
    fused_add_mul_groupnorm(input1, input2, weight, bias, num_groups, eps=1e-5, *, out=None) -> Tensor

    Performs the following steps in a fused manner:
      1) Z = input1 + input2
      2) M = Z * input2
      3) GroupNorm(M, weight, bias, num_groups, eps)

    Arguments:
      input1 (Tensor): The first input tensor (X).
      input2 (Tensor): The second input tensor (Y), must be broadcastable to the shape of X.
      weight (Tensor): The learnable weight (gamma) of shape (C,).
      bias (Tensor): The learnable bias (beta) of shape (C,).
      num_groups (int): Number of groups for group normalization.
      eps (float, optional): Epsilon for numerical stability in group norm. Default: 1e-5.
      out (Tensor, optional): Output tensor. Ignored if None. Default: None

    Returns:
      A tensor of the same shape as input1, after the fused operation.
    """
    import torch

    # Broadcast shapes if needed
    X, Y = torch.broadcast_tensors(input1, input2)
    shape = X.shape
    if len(weight.shape) != 1 or len(bias.shape) != 1:
        raise ValueError("weight and bias must be 1D tensors of shape (C,)")
    # Assume shape has B, C, *spatial_dims
    # We flatten the rest of the spatial dimensions into D
    # e.g. input shape [B, C, H, W] -> B, C, D with D = H*W
    if len(shape) < 2:
        raise ValueError("Input must have at least 2 dimensions [N, C, ...].")

    B = shape[0]
    C = shape[1]
    D = 1
    for s in shape[2:]:
        D *= s

    if C != weight.shape[0] or C != bias.shape[0]:
        raise ValueError("weight and bias must have shape (C,) where C is the
