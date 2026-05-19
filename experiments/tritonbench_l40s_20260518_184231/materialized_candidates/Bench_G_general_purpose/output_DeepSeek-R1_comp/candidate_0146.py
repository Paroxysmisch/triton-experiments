import torch
import triton
import triton.language as tl

@triton.jit
def fused_add_mul_activation_kernel(
    x_ptr,
    bias_ptr,
    in_ptr,
    output_ptr,
    scale,
    activation_code,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs with eviction policy to optimize cache usage
    x = tl.load(x_ptr + offsets, mask=mask, eviction_policy="evict_first")
    bias = tl.load(bias_ptr + offsets, mask=mask, eviction_policy="evict_first")
    in_val = tl.load(in_ptr + offsets, mask=mask, eviction_policy="evict_first")
    
    # Compute fused operations
    scaled_in = scale * in_val
    add_result = x + bias + scaled_in
    
    # Apply activation function
    if activation_code == 0:  # Sigmoid
        output = tl.sigmoid(add_result)
    elif activation_code == 1:  # ReLU
        output = tl.maximum(add_result, 0.0)
    else:
        output = add_result  # Identity if unknown code
    
    # Store result with eviction policy
    tl.store(output_ptr + offsets, output, mask=mask, eviction_policy="evict_first")

def fused_add_mul_activation_torch(
    in_out_tensor: torch.Tensor,
    bias_tensor: torch.Tensor,
    in_tensor: torch.Tensor,
    scale: float = 1.0,
    activation_type: str = 'sigmoid'
) -> torch.Tensor:
    # Ensure tensors are on CUDA and contiguous
    assert all(t.is_cuda and t.is_contiguous() for t in [in_out_tensor, bias_tensor, in_tensor]), "Tensors must be CUDA-contiguous"
    assert in_out_tensor.shape == bias_tensor.shape == in_tensor.shape, "Tensor shapes must match"
    
    n_elements = in_out_tensor.numel()
    output_tensor = torch.empty_like(in_out_tensor)
    
    # Determine activation code
    activation_code = 0 if activation_type == 'sigmoid' else 1 if activation_type == 'relu' else -1
    if activation_code == -1:
        raise ValueError(f"Unsupported activation type: {activation_type}")
    
    # Grid configuration
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    
    # Launch kernel with default BLOCK_SIZE
    fused_add_mul_activation_kernel[grid](
        in_out_tensor.data_ptr(),
        bias_tensor.data_ptr(),
        in_tensor.data_ptr(),
        output_tensor.data_ptr(),
        scale,
        activation_code,
        n_elements,
        BLOCK_SIZE=1024,
    )
    
    return output_tensor
