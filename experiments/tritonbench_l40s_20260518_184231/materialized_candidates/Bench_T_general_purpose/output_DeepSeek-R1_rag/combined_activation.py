import torch
import triton
import triton.language as tl

@triton.jit
def combined_activation_kernel(
    input_ptr, weight1_ptr, weight2_ptr, bias_ptr, output_ptr,
    B, N, D_in, D_out,
    stride_input_b, stride_input_n, stride_input_k,
    stride_weight1_k, stride_weight1_dout,
    stride_weight2_b, stride_weight2_n, stride_weight2_dout,
    stride_bias_b, stride_bias_n, stride_bias_dout,
    stride_output_b, stride_output_n, stride_output_dout,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_DOUT: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_dout = tl.program_id(2)
    
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_dout = pid_dout * BLOCK_DOUT + tl.arange(0, BLOCK_DOUT)
    offs_k = tl.arange(0, BLOCK_K)
    
    input_ptrs = input_ptr + (offs_b[:, None, None] * stride_input_b + 
                              offs_n[None, :, None] * stride_input_n + 
                              offs_k[None, None, :] * stride_input_k)
    weight1_ptrs = weight1_ptr + (offs_k[:, None] * stride_weight1_k + 
                                  offs_dout[None, :] * stride_weight1_dout)
    
    acc = tl.zeros((BLOCK_B, BLOCK_N, BLOCK_DOUT), dtype=tl.float32)
    
    for k in range(0, D_in, BLOCK_K):
        mask_input = (offs_b[:, None, None] < B) & (offs_n[None, :, None] < N) & (offs_k[None, None, :] + k < D_in)
        a = tl.load(input_ptrs + k * stride_input_k, mask=mask_input, other=0.0)
        mask_weight1 = (offs_k[:, None] + k < D_in) & (offs_dout[None, :] < D_out)
        b = tl.load(weight1_ptrs + k * stride_weight1_k, mask=mask_weight1, other=0.0)
        acc += tl.sum(a[:, :, :, None] * b[None, None, :, :], axis=2)
    
    sig = 1.0 / (1.0 + tl.exp(-acc))
    tanh_acc = tl.tanh(sig)
    
    weight2_ptrs = weight2_ptr + (offs_b[:, None, None] * stride_weight2_b + 
                                  offs_n[None, :, None] * stride_weight2_n + 
                                  offs_dout[None, None, :] * stride_weight2_dout)
    mask_weight2 = (offs_b[:, None, None] < B) & (offs_n[None, :, None] < N) & (offs_dout[None, None, :] < D_out)
    weight2 = tl.load(weight2_ptrs, mask=mask_weight2, other=0.0)
    
    bias_ptrs = bias_ptr + (offs_b[:, None, None] * stride_bias_b + 
                            offs_n[None, :, None] * stride_bias_n + 
                            offs_dout[None, None, :] * stride_bias_dout)
    mask_bias = (offs_b[:, None, None] < B) & (offs_n[None, :, None] < N) & (offs_dout[None, None, :] < D_out)
    bias = tl.load(bias_ptrs, mask=mask_bias, other=0.0)
    
    output = tanh_acc * weight2 + bias
    output_ptrs = output_ptr + (offs_b[:, None, None] * stride_output_b + 
                                offs_n[None, :, None] * stride_output_n + 
                                offs_dout[None, None, :] * stride_output_dout)
    tl.store(output_ptrs, output, mask=mask_bias)

def combined_activation(input: torch.Tensor, weight1: torch.Tensor, weight2: torch.Tensor, bias: torch.Tensor, *, out=None) -> torch.Tensor:
    assert input.dim() >= 2, "Input must have at least two dimensions"
    assert weight1.dim() == 2, "Weight1 must be 2D"
    D_in, D_out = weight1.shape
    assert input.size(-1) == D_in, "Input last dimension must match D_in of weight1"
    
    original_shape = input.shape
    B = input.numel() // (original_shape[-2] * original_shape[-1])
    N = original_shape[-2]
    input_flat = input.reshape(B, N, D_in)
    
    output_shape = (B, N, D_out)
    weight2_expanded = weight2.expand(output_shape)
    bias_expanded = bias.expand(output_shape)
    
    input_flat = input_flat.contiguous()
    weight1 = weight1.contiguous()
    weight2_expanded = weight2_expanded.contiguous()
    bias_expanded = bias_expanded.contiguous()
    
    output_flat = torch.empty(output_shape, device=input.device, dtype=torch.float32)
    
    BLOCK_B = 16
    BLOCK_N = 32
    BLOCK_K = 64
    BLOCK_DOUT = 32
    
    grid = (
        triton.cdiv(B, BLOCK_B),
        triton.cdiv(N, BLOCK_N),
        triton.cdiv(D_out, BLOCK_DOUT),
    )
    
    combined_activation_kernel[grid](
        input_flat, weight1, weight2_expanded, bias_expanded, output_flat,
        B, N, D_in, D_out,
        input_flat.stride(0), input_flat.stride(1), input_flat.stride(2),
        weight1.stride(0), weight1.stride(1),
        weight2_expanded.stride(0), weight2_expanded.stride(1), weight2_expanded.stride(2),
        bias_expanded.stride(0), bias_expanded.stride(1), bias_expanded.stride(2),
        output_flat.stride(0), output_flat.stride(1), output_flat.stride(2),
        BLOCK_B=BLOCK_B,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        BLOCK_DOUT=BLOCK_DOUT,
    )
    
    output = output_flat.view(*original_shape[:-2], N, D_out)
    if out is not None:
        out.copy_(output)
        return out
    return output
