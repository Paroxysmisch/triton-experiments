import torch
import triton
import triton.language as tl


@triton.jit
def _fused_hardsigmoid_batch_norm_kernel(
    x_ptr,                   # [B, C]
    out_ptr,                 # [B, C]
    mean_ptr,                # [C]
    var_ptr,                 # [C]
    weight_ptr,              # [C] or None
    bias_ptr,                # [C] or None
    B,                       # total batch size (rows)
    C,                       # number of channels (cols)
    stride_b,                # stride for batch dimension in x
    stride_c,                # stride for channel dimension in x
    eps,                     # epsilon for numerical stability
    inplace,                 # bool: if True, modify x in place
    BLOCK_M: tl.constexpr,   # block size for row dimension
    BLOCK_N: tl.constexpr    # block size for column dimension
):
    row_id = tl.program_id(0)
    col_id = tl.program_id(1)

    row_range = row_id * BLOCK_M + tl.arange(0, BLOCK_M)
    col_range = col_id * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_row = row_range < B
    mask_col = col_range < C

    # Create 2D mask
    row_mask, col_mask = [tl.broadcast_to(x, [BLOCK_M, BLOCK_N]) for x in [mask_row, mask_col]]
    full_mask = row_mask & col_mask

    # Pointers for row/col blocks
    x_offset = row_range[:, None] * stride_b + col_range[None, :] * stride_c
    out_offset = x_offset

    # Load data
    x_val = tl.load(x_ptr + x_offset, mask=full_mask, other=0.0)

    # Load mean/var/weight/bias (broadcast along the batch dimension)
    mean_val = tl.load(mean_ptr + col_range, mask=mask_col, other=0.0)
    var_val  = tl.load(var_ptr + col_range, mask=mask_col, other=0.0)
    if weight_ptr != 0:
        w_val = tl.load(weight_ptr + col_range, mask=mask_col, other=1.0)
    else:
        w_val = 1.0
    if bias_ptr != 0:
        b_val = tl.load(bias_ptr + col_range, mask=mask_col, other=0.0)
    else:
        b_val = 0.0

    # Broadcast mean, var, weight, bias across rows
    mean_val = tl.broadcast_to(mean_val[None, :], [BLOCK_M, BLOCK_N])
    var_val  = tl.broadcast_to(var_val[None, :],  [BLOCK_M, BLOCK_N])
    w_val    = tl.broadcast_to(w_val[None, :],    [BLOCK_M, BLOCK_N])
    b_val    = tl.broadcast_to(b_val[None, :],    [BLOCK_M, BLOCK_N])

    # Batch normalization
    inv_std = 1.0 / tl.sqrt(var_val + eps)
    bn_out = (x_val - mean_val) * inv_std * w_val + b_val

    # Hardsigmoid activation in a single pass:
    #  0 if x < -3
    #  1 if x > 3
    #  Otherwise (x + 3) / 6
    hs_lower = bn_out + 3.0
    hs_lower = tl.maximum(hs_lower, 0.0)
    hs_lower = tl.minimum(hs_lower, 6.0)
    hs_out = hs_lower * (1.0 / 6.0)

    # Store output
    if inplace != 0:
        tl.store(x_ptr + x_offset, hs_out, mask=full_mask)
    else:
        tl.store(out_ptr + out_offset, hs_out, mask=full_mask)


def fused_hardsigmoid_batch_norm(
    x: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    weight: torch.Tensor = None,
    bias: torch.Tensor = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
    inplace: bool = False
) -> torch.Tensor:
    """
    Fused Batch Normalization + Hardsigmoid in Triton.
    """
    # Ensure x is 2D for simplicity: [Batch, Channels]
    # If x has more dimensions, flatten them except for the channel dim
    if x.dim() > 2:
        # assume channel dim is 1
        N = x.size(0)
        C = x.size(1)
        rest = x.shape[2:]
        x_reshaped = x.contiguous().view(N, C, -1)
        x_reshaped = x_reshaped.transpose(1, 2).contiguous().view(-1, C)
        is_reshaped = True
        original_shape = x.shape
    else:
        x_reshaped = x
        is_reshaped = False

    B, C = x_reshaped.shape
    x_reshaped = x_reshaped.contiguous()

    # Compute current mean/var if training
    if training:
        with torch.no_grad():
            current_mean = x_reshaped.mean(dim=0)
            current_var = x_reshaped.var(dim=0, unbiased=False)
            # Update running estimates
            running_mean.mul_(1 - momentum).add_(current_mean, alpha=momentum)
            running_var.mul_(1 - momentum).add_(current_var, alpha=momentum)
        used_mean = current_mean
        used_var = current_var
    else:
        used_mean = running_mean
        used_var = running_var

    # Prepare output tensor
    if inplace:
        out = x_reshaped
    else:
        out = torch.empty_like(x_reshaped)

    # Convert all to contiguous and float32 for safety
    x_ptr = x_reshaped.data_ptr()
    out_ptr = out.data_ptr()

    mean_ptr = used_mean.contiguous().data_ptr()
    var_ptr = used_var.contiguous().data_ptr()
    weight_ptr = weight.contiguous().data_ptr() if weight is not None else 0
    bias_ptr = bias.contiguous().data_ptr() if bias is not None else 0

    # Strides
    stride_b = x_reshaped.stride(0)
    stride_c = x_reshaped.stride(1)

    # Launch kernel
    BLOCK_M = 64
    BLOCK_N = 64
    grid = ( (B + BLOCK_M - 1) // BLOCK_M, (C + BLOCK_N - 1) // BLOCK_N )

    _fused_hardsigmoid_batch_norm_kernel[grid](
        x_ptr, out_ptr,
        mean_ptr, var_ptr,
        weight_ptr, bias_ptr,
        B, C,
        stride_b, stride_c,
        eps,
        int(inplace),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N
    )

    # Reshape output if needed
    if is_reshaped:
        out_re = out.view(-1, original_shape[1], *original_shape[2:])
        out_re = out_re.view(original_shape[0], -1, *original_shape[2:]).transpose(1, 2).contiguous()
        return out_re.view(original_shape)
    return out
