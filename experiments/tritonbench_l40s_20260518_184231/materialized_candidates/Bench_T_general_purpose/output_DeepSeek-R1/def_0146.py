import torch
import triton
import triton.language as tl
import torch.nn.functional as F

@triton.jit
def elu_linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    M, N, K,
    alpha,
    stride_input_m, stride_input_k,
    stride_weight_n, stride_weight_k,
    stride_output_m, stride_output_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    input_ptrs = input_ptr + offs_m[:, None] * stride_input_m + offs_k[None, :] * stride_input_k
    weight_ptrs = weight_ptr + offs_n[:, None] * stride_weight_n + offs_k[None, :] * stride_weight_k
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        input_tile = tl.load(input_ptrs, mask=(offs_k[None, :] < k_remaining), other=0.0)
        weight_tile = tl.load(weight_ptrs, mask=(offs_k[None, :] < k_remaining), other=0.0)
        
        acc += tl.dot(input_tile, weight_tile, allow_tf32=True)
        
        input_ptrs += BLOCK_K * stride_input_k
        weight_ptrs += BLOCK_K * stride_weight_k
    
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]
    
    zero = 0.0
    mask = acc > zero
    elu = tl.where(mask, acc, alpha * (tl.exp(acc) - 1))
    
    output_ptrs = output_ptr + offs_m[:, None] * stride_output_m + offs_n[None, :] * stride_output_n
    tl.store(output_ptrs, elu, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def elu_linear(input, weight, bias=None, alpha=1.0, inplace=False):
    if inplace:
        x = F.linear(input, weight, bias)
        return F.elu_(x, alpha=alpha)
    
    assert input.dim() == 2, "Input must be 2D"
    M, K = input.shape
    N, K_weight = weight.shape
    assert K == K_weight, f"Input feature size {K} must match weight feature size {K_weight}"
    if bias is not None:
        assert bias.dim() == 1, "Bias must be 1D"
        assert bias.size(0) == N, f"Bias size {bias.size(0)} must match weight output size {N}"
    
    output = torch.empty((M, N), device=input.device, dtype=input.dtype)
    
    stride_input_m, stride_input_k = input.stride()
    stride_weight_n, stride_weight_k = weight.stride()
    stride_output_m, stride_output_n = output.stride()
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
    
    elu_linear_kernel[grid](
        input, weight, bias, output,
        M, N, K,
        alpha,
        stride_input_m, stride_input_k,
        stride_weight_n, stride_weight_k,
        stride_output_m, stride_output_n,
        BLOCK_M=32, BLOCK_N=32, BLOCK_K=32
    )
    
    return output
