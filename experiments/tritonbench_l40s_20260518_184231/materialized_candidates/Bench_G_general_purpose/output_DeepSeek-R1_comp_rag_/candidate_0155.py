import triton
import triton.language as tl
import torch

@triton.jit
def fused_add_mul_activation_kernel(
    x_ptr,
    in_ptr,
    bias_ptr,
    output_ptr,
    multiplier,
    n_elements,
    activation_type: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    in_val = tl.load(in_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr + offsets, mask=mask)
    
    result = x + bias + in_val * multiplier
    
    if activation_type == 'relu':
        result = tl.maximum(result, 0.0)
    elif activation_type == 'sigmoid':
        result = tl.sigmoid(result)
    
    tl.store(output_ptr + offsets, result, mask=mask)

def fused_add_mul_activation_torch(
    x: torch.Tensor,
    in_tensor: torch.Tensor,
    bias: torch.Tensor,
    multiplier: float = 1.0,
    activation: str = 'sigmoid'
) -> torch.Tensor:
    assert x.shape == in_tensor.shape == bias.shape, "All inputs must have the same shape"
    assert activation in ['sigmoid', 'relu'], "Activation must be 'sigmoid' or 'relu'"
    
    x = x.contiguous()
    in_tensor = in_tensor.contiguous()
    bias = bias.contiguous()
    output = torch.empty_like(x)
    
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    fused_add_mul_activation_kernel[grid](
        x.data_ptr(),
        in_tensor.data_ptr(),
        bias.data_ptr(),
        output.data_ptr(),
        multiplier,
        n_elements,
        activation_type=activation,
        BLOCK_SIZE=1024,
    )
    return output
