import torch
import triton
import triton.language as tl

@triton.jit
def fused_linear_relu_layernorm_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    in_features, out_features,
    eps,
    input_row_stride, weight_row_stride, output_row_stride,
    BLOCK_SIZE_IN: tl.constexpr,
    BLOCK_SIZE_OUT: tl.constexpr,
):
    row_idx = tl.program_id(0)
    
    input_row = input_ptr + row_idx * input_row_stride
    output_row = output_ptr + row_idx * output_row_stride
    
    # Accumulate linear output for each feature in the output
    acc = tl.zeros((BLOCK_SIZE_OUT,), dtype=tl.float32)
    for in_block in range(0, in_features, BLOCK_SIZE_IN):
        in_offsets = in_block + tl.arange(0, BLOCK_SIZE_IN)
        in_mask = in_offsets < in_features
        in_val = tl.load(input_row + in_offsets, mask=in_mask, other=0.0)
        
        for out_block in range(0, out_features, BLOCK_SIZE_OUT):
            out_offsets = out_block + tl.arange(0, BLOCK_SIZE_OUT)
            out_mask = out_offsets < out_features
            weight_ptrs = weight_ptr + out_offsets[:, None] * weight_row_stride + in_offsets[None, :]
            weight_val = tl.load(weight_ptrs, mask=out_mask[:, None] & in_mask[None, :], other=0.0)
            acc_val = tl.sum(in_val[None, :] * weight_val, axis=1)
            acc = tl.where(out_mask, acc + acc_val, acc)
    
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + tl.arange(0, BLOCK_SIZE_OUT), mask=out_mask, other=0.0)
        acc += bias
    
    # Apply ReLU
    acc = tl.maximum(acc, 0.0)
    
    # Compute mean and variance for layer normalization
    mean = tl.sum(acc) / out_features
    variance = tl.sum((acc - mean) ** 2) / out_features
    inv_std = 1.0 / tl.sqrt(variance + eps)
    normalized = (acc - mean) * inv_std
    
    tl.store(output_row + tl.arange(0, BLOCK_SIZE_OUT), normalized, mask=out_mask)

def fused_layer_norm_relu_linear(
    input: torch.Tensor, weight: torch.Tensor, bias=None, normalized_shape=None, eps=1e-5, elementwise_affine=True
) -> torch.Tensor:
    assert input.dim() >= 2, "Input must have at least 2 dimensions"
    in_features = input.size(-1)
    out_features = weight.size(0)
    assert weight.size(1) == in_features, "Weight shape mismatch"
    
    if normalized_shape is None:
        normalized_shape = input.shape[1:]
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    normalized_shape = torch.Size(normalized_shape)
    assert out_features == normalized_shape[-1], "Normalized shape must match the last dimension of the linear output"
    
    output = torch.empty((*input.shape[:-1], out_features), device=input.device, dtype=input.dtype)
    
    has_bias = bias is not None
    if has_bias:
        assert bias.size(0) == out_features, "Bias shape mismatch"
        bias_ptr = bias.data_ptr()
    else:
        bias_ptr = None
    
    BLOCK_SIZE_IN = 32
    BLOCK_SIZE_OUT = 32
    grid = (input.numel() // in_features,)
    
    fused_linear_relu_layernorm_kernel[grid](
        input.data_ptr(), weight.data_ptr(), bias_ptr, output.data_ptr(),
        in_features, out_features,
        eps,
        input.stride(-2), weight.stride(0), output.stride(-2),
        BLOCK_SIZE_IN, BLOCK_SIZE_OUT
    )
    
    if elementwise_affine:
        ln_weight = torch.ones(normalized_shape, device=input.device, dtype=input.dtype)
        ln_bias = torch.zeros(normalized_shape, device=input.device, dtype=input.dtype)
        output = output * ln_weight + ln_bias
    
    return output

# Example usage:
# input = torch.randn(4, 5)
# weight = torch.randn(3, 5)
# bias = torch.randn(3)
# normalized_shape = 3
# output = fused_layer_norm_relu_linear(input, weight, bias, normalized_shape)
# print(output.shape)  # Expected: (4, 3)
