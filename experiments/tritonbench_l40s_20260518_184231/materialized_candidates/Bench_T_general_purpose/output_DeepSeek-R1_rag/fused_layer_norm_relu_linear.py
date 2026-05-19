import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.runtime.jit import get_cuda_stream

@triton.jit
def fused_linear_relu_layernorm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    gamma_ptr,
    beta_ptr,
    output_ptr,
    in_features,
    out_features,
    input_row_stride,
    weight_row_stride,
    eps,
    elementwise_affine: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    out_idx = tl.arange(0, BLOCK_SIZE)
    mask = out_idx < out_features

    # Load input row
    input_offset = row_idx * input_row_stride
    input = tl.load(input_ptr + input_offset + tl.arange(0, in_features), mask=tl.arange(0, in_features) < in_features, other=0.0)

    # Load weights for all output features
    weight = tl.load(weight_ptr + out_idx[:, None] * weight_row_stride + tl.arange(0, in_features)[None, :], 
                     mask=mask[:, None] & (tl.arange(0, in_features) < in_features), other=0.0)

    # Compute linear transformation
    dot = tl.sum(input[None, :] * weight, axis=1)
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + out_idx, mask=mask, other=0.0)
        dot += bias

    # Apply ReLU
    activations = tl.where(dot > 0, dot, 0.0).to(tl.float32)

    # Compute mean and variance for layer normalization
    mean = tl.sum(activations, axis=0) / out_features
    variance = tl.sum((activations - mean) ** 2, axis=0) / out_features

    # Normalize
    normalized = (activations - mean) / tl.sqrt(variance + eps)

    # Apply gamma and beta if elementwise_affine is True
    if elementwise_affine:
        gamma = tl.load(gamma_ptr + out_idx, mask=mask, other=1.0)
        beta = tl.load(beta_ptr + out_idx, mask=mask, other=0.0)
        normalized = normalized * gamma + beta

    # Store the result
    output_offset = row_idx * out_features + out_idx
    tl.store(output_ptr + output_offset, normalized, mask=mask)

@torch.inference_mode()
def fused_layer_norm_relu_linear(input: Tensor, weight: Tensor, bias=None, normalized_shape=None, eps=1e-5, elementwise_affine=True) -> Tensor:
    assert normalized_shape is not None, "normalized_shape must be provided"
    out_features = weight.size(0)
    if isinstance(normalized_shape, int):
        normalized_shape = [normalized_shape]
    assert list(normalized_shape) == [out_features], "normalized_shape must match the output features of the linear layer"
    
    # Create gamma and beta if elementwise_affine is True
    if elementwise_affine:
        gamma = torch.ones(normalized_shape, dtype=input.dtype, device=input.device)
        beta = torch.zeros(normalized_shape, dtype=input.dtype, device=input.device)
    else:
        gamma = None
        beta = None

    # Ensure contiguous tensors
    input_contig = input.contiguous()
    weight_contig = weight.contiguous()
    if bias is not None:
        bias_contig = bias.contiguous()
    else:
        bias_contig = None

    # Reshape input to 2D (batch * ..., in_features)
    original_shape = input_contig.shape
    input_2d = input_contig.view(-1, original_shape[-1])
    batch_rows = input_2d.size(0)

    # Prepare output tensor
    output_2d = torch.empty((batch_rows, out_features), dtype=input.dtype, device=input.device)

    # Kernel configuration
    BLOCK_SIZE = triton.next_power_of_2(out_features)
    grid = (batch_rows,)
    
    # Kernel meta for execution
    device = input_contig.device
    device_idx = device.index
    stream = get_cuda_stream(device_idx)
    kernel_meta = dict(device=device, device_type=device.type, stream=stream)

    fused_linear_relu_layernorm_kernel[grid](
        input_2d, weight_contig, bias_contig,
        gamma, beta, output_2d,
        in_features=input_contig.size(-1),
        out_features=out_features,
        input_row_stride=input_2d.stride(0),
        weight_row_stride=weight_contig.stride(0),
        eps=eps,
        elementwise_affine=elementwise_affine,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
        **kernel_meta
    )

    # Reshape output to original shape (excluding last dimension) + out_features
    output = output_2d.view(original_shape[:-1] + (out_features,))
    return output
