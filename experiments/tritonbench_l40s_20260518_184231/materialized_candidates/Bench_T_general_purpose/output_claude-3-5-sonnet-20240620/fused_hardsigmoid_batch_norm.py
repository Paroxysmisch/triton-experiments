import torch
import triton
import triton.language as tl

@triton.jit
def fused_hardsigmoid_batchnorm_kernel(
    # Pointers to tensors
    x_ptr, out_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
    # Tensor dimensions and properties
    n_elements, channels,
    # BN parameters
    eps, momentum,
    # Strides for memory access
    stride_x, stride_out,
    # Constants
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets * stride_x, mask=mask)
    
    # Calculate channel index
    channel_idx = (offsets // (n_elements // channels)) % channels
    
    # Load BN parameters for current channel
    mean = tl.load(running_mean_ptr + channel_idx)
    var = tl.load(running_var_ptr + channel_idx)
    gamma = tl.load(weight_ptr + channel_idx) if weight_ptr is not None else 1.0
    beta = tl.load(bias_ptr + channel_idx) if bias_ptr is not None else 0.0
    
    # Batch norm computation
    x_normalized = (x - mean) / tl.sqrt(var + eps)
    x_scaled = gamma * x_normalized + beta
    
    # Hardsigmoid: min(max(x * 1/6 + 0.5, 0), 1)
    x_hardsigmoid = tl.minimum(tl.maximum(x_scaled * (1.0/6.0) + 0.5, 0.0), 1.0)
    
    # Store result
    tl.store(out_ptr + offsets * stride_out, x_hardsigmoid, mask=mask)

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
    # Input validation
    assert x.dim() >= 2, "Input tensor must have at least 2 dimensions"
    channels = x.size(1)
    
    # Ensure running stats have correct size
    assert running_mean.size(0) == channels
    assert running_var.size(0) == channels
    
    # Optional weight/bias validation
    if weight is not None:
        assert weight.size(0) == channels
    if bias is not None:
        assert bias.size(0) == channels
    
    # Calculate total elements and prepare output
    n_elements = x.numel()
    out = x if inplace else torch.empty_like(x)
    
    # Update running statistics if in training mode
    if training:
        dims = [0] + list(range(2, x.dim()))
        current_mean = x.mean(dims)
        current_var = x.var(dims, unbiased=False)
        
        running_mean.mul_(1 - momentum).add_(current_mean * momentum)
        running_var.mul_(1 - momentum).add_(current_var * momentum)
    
    # Grid and block sizes
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    fused_hardsigmoid_batchnorm_kernel[grid](
        x.contiguous().data_ptr(),
        out.contiguous().data_ptr(),
        running_mean.contiguous().data_ptr(),
        running_var.contiguous().data_ptr(),
        weight.contiguous().data_ptr() if weight is not None else None,
        bias.contiguous().data_ptr() if bias is not None else None,
        n_elements,
        channels,
        eps,
        momentum,
        x.stride(0),
        out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out
