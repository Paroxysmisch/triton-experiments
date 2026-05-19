import torch
import triton
import triton.language as tl
from typing import Optional

@triton.jit
def elu_linear_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    alpha_ptr,
    output_ptr,
    M, N, K,
    stride_input_m, stride_input_k,
    stride_weight_n, stride_weight_k,
    stride_bias_n,
    stride_output_m, stride_output_n,
    HAS_BIAS: tl.constexpr,
    ALPHA_IS_TENSOR: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    input_ptrs = input_ptr + offs_m[:, None] * stride_input_m + offs_k[None, :] * stride_input_k
    weight_ptrs = weight_ptr + offs_n[:, None] * stride_weight_n + offs_k[None, :] * stride_weight_k
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        input_mask = (k + offs_k) < K
        input = tl.load(input_ptrs, mask=input_mask[None, :], other=0.0)
        weight = tl.load(weight_ptrs, mask=input_mask[:, None], other=0.0)
        acc += tl.dot(input, weight)
        input_ptrs += BLOCK_SIZE_K * stride_input_k
        weight_ptrs += BLOCK_SIZE_K * stride_weight_k
    
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n * stride_bias_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]
    
    if ALPHA_IS_TENSOR:
        alpha = tl.load(alpha_ptr)
    else:
        alpha = alpha_ptr
    
    elu_output = tl.where(acc > 0, acc, alpha * (tl.exp(acc) - 1))
    
    output_ptrs = output_ptr + offs_m[:, None] * stride_output_m + offs_n[None, :] * stride_output_n
    output_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(output_ptrs, elu_output, mask=output_mask)

def elu_linear(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, alpha: float = 1.0, inplace: bool = False) -> torch.Tensor:
    assert input.dim() == 2, "Input must be 2D"
    M, K = input.shape
    N, K_w = weight.shape
    assert K == K_w, f"Input feature size {K} must match weight size {K_w}"
    if inplace:
        raise RuntimeError("In-place operation is not supported for elu_linear")
    
    output = torch.empty((M, N), device=input.device, dtype=input.dtype)
    alpha_tensor = torch.tensor([alpha], device=input.device, dtype=input.dtype)
    
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_SIZE_M']), triton.cdiv(N, meta['BLOCK_SIZE_N']))
    
    elu_linear_kernel[grid](
        input, weight,
        bias if bias is not None else torch.empty(0, device=input.device),
        alpha_tensor,
        output,
        M, N, K,
        input.stride(0), input.stride(1),
        weight.stride(0), weight.stride(1),
        bias.stride(0) if bias is not None else 0,
        output.stride(0), output.stride(1),
        HAS_BIAS=bias is not None,
        ALPHA_IS_TENSOR=True,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_K=32,
    )
    return output

# Example usage
torch.manual_seed(0)
M, K, N = 512, 256, 1024
input = torch.randn((M, K), device='cuda')
weight = torch.randn((N, K), device='cuda')
bias = torch.randn(N, device='cuda')
alpha = 1.0

output_triton = elu_linear(input, weight, bias, alpha)
output_linear = torch.nn.functional.linear(input, weight, bias)
output_linear_elu = torch.nn.functional.elu(output_linear, alpha=alpha)

print(f"Max difference: {torch.max(torch.abs(output_triton - output_linear_elu))}")
