import triton
import triton.language as tl
import torch

@triton.jit
def silu_batch_norm_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, out_ptr,
    n_elements, eps, is_training,
    BLOCK_SIZE: tl.constexpr
):
    # Compute the block index
    pid = tl.program_id(0)
    
    # Compute the offsets for the current block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=offsets < n_elements, other=0.0)
    
    # Load mean and variance
    mean = tl.load(mean_ptr + offsets, mask=offsets < n_elements, other=0.0)
    var = tl.load(var_ptr + offsets, mask=offsets < n_elements, other=0.0)
    
    # Normalize
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_hat = (x - mean) * inv_std
    
    # Scale and shift
    if weight_ptr is not None:
        weight = tl.load(weight_ptr + offsets, mask=offsets < n_elements, other=1.0)
        x_hat *= weight
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offsets, mask=offsets < n_elements, other=0.0)
        x_hat += bias
    
    # Apply SiLU activation
    sigmoid = 1 / (1 + tl.exp(-x_hat))
    out = x_hat * sigmoid
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=offsets < n_elements)

def silu_batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
    # Ensure input is a torch tensor
    assert isinstance(input, torch.Tensor), "Input must be a torch tensor"
    
    # Prepare output tensor
    output = torch.empty_like(input)
    
    # Number of elements in the input tensor
    n_elements = input.numel()
    
    # Launch the kernel
    BLOCK_SIZE = 1024  # Adjust block size based on your hardware capabilities
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    silu_batch_norm_kernel[grid](
        input, running_mean, running_var, weight, bias, output,
        n_elements, eps, training,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output
