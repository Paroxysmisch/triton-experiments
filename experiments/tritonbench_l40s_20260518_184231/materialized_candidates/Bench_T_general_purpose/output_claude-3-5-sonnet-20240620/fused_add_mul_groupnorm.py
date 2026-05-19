import triton
import triton.language as tl
import torch

@triton.jit
def _fused_add_mul_groupnorm_kernel(
    # Pointers to matrices
    x_ptr, y_ptr, gamma_ptr, beta_ptr, output_ptr,
    # Matrix dimensions
    batch_size, height, width, channels,
    # Parameters
    num_groups, eps,
    # Strides
    stride_xb, stride_xh, stride_xw, stride_xc,
    stride_yb, stride_yh, stride_yw, stride_yc,
    stride_ob, stride_oh, stride_ow, stride_oc,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    # Compute group size
    channels_per_group = channels // num_groups
    
    # Compute position
    pid = tl.program_id(0)
    batch_idx = pid // (height * width * num_groups)
    group_idx = pid % num_groups
    h_idx = (pid % (height * width * num_groups)) // (width * num_groups)
    w_idx = (pid % (width * num_groups)) // num_groups

    # Compute start channel for this group
    start_channel = group_idx * channels_per_group
    
    # Load block of channels for this group
    channel_offsets = start_channel + tl.arange(0, channels_per_group)
    mask = channel_offsets < channels
    
    # Compute base offsets
    base_x_offset = batch_idx * stride_xb + h_idx * stride_xh + w_idx * stride_xw
    base_y_offset = batch_idx * stride_yb + h_idx * stride_yh + w_idx * stride_yw
    base_o_offset = batch_idx * stride_ob + h_idx * stride_oh + w_idx * stride_ow
    
    # Load input values
    x = tl.load(x_ptr + base_x_offset + channel_offsets * stride_xc, mask=mask)
    y = tl.load(y_ptr + base_y_offset + channel_offsets * stride_yc, mask=mask)
    
    # Compute Z = X + Y
    z = x + y
    
    # Compute M = Z ⊙ Y
    m = z * y
    
    # Group normalization
    # Compute mean
    mean = tl.sum(m, axis=0) / channels_per_group
    
    # Compute variance
    m_centered = m - mean
    var = tl.sum(m_centered * m_centered, axis=0) / channels_per_group
    
    # Normalize
    rstd = 1 / tl.sqrt(var + eps)
    normalized = m_centered * rstd
    
    # Apply affine transformation
    gamma = tl.load(gamma_ptr + channel_offsets, mask=mask)
    beta = tl.load(beta_ptr + channel_offsets, mask=mask)
    output = gamma * normalized + beta
    
    # Store result
    tl.store(output_ptr + base_o_offset + channel_offsets * stride_oc, output, mask=mask)

def fused_add_mul_groupnorm(input1, input2, weight, bias, num_groups, eps=1e-5, *, out=None):
    """
    Performs a fused operation combining element-wise addition, multiplication, and group normalization.
    
    Args:
        input1 (Tensor): The first input tensor X
        input2 (Tensor): The second input tensor Y, must be broadcastable to X
        weight (Tensor): Learnable weight parameter γ of shape (C,)
        bias (Tensor): Learnable bias parameter β of shape (C,)
        num_groups (int): Number of groups for group normalization
        eps (float, optional): Small value for numerical stability. Default: 1e-5
        out (Tensor, optional): Output tensor. Default: None
    
    Returns:
        Tensor: The result of the fused operation
    """
    # Input validation
    assert input1.dim() == 4, "Input1 must be 4D (N,C,H,W)"
    assert input2.dim() == 4, "Input2 must be 4D (N,C,H,W)"
    assert weight.dim() == 1, "Weight must be 1D"
    assert bias.dim() == 1, "Bias must be 1D"
    
    batch_size, channels, height, width = input1.shape
    assert channels % num_groups == 0, "Number of channels must be divisible by num_groups"
    assert weight.shape[0] == channels, "Weight must have shape (C,)"
    assert bias.shape[0] == channels, "Bias must have shape (C,)"
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty_like(input1)
    
    # Launch kernel
    grid = (batch_size * height * width * num_groups,)
    _fused_add_mul_groupnorm_kernel[grid](
        input1, input2, weight, bias, out,
        batch_size, height, width, channels,
        num_groups, eps,
        input1.stride(0), input1.stride(2), input1.stride(3), input1.stride(1),
        input2.stride(0), input2.stride(2), input2.stride(3), input2.stride(1),
        out.stride(0), out.stride(2), out.stride(3), out.stride(1),
        BLOCK_SIZE=32,
    )
    
    return out
