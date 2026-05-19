import torch
import triton
import triton.language as tl

@triton.jit
def softplus_linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    M, N, K,
    stride_input_m, stride_input_k,
    stride_weight_n, stride_weight_k,
    stride_bias_n,
    stride_output_m, stride_output_n,
    beta,
    threshold,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
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
        mask_k = (k + offs_k) < K
        a = tl.load(input_ptrs, mask=mask_k[None, :] & (offs_m[:, None] < M), other=0.0)
        b = tl.load(weight_ptrs, mask=mask_k[None, :] & (offs_n[:, None] < N), other=0.0)
        a = a.to(tl.float32)
        b = b.to(tl.float32)
        acc += tl.dot(a, b, allow_tf32=True)
        input_ptrs += BLOCK_SIZE_K * stride_input_k
        weight_ptrs += BLOCK_SIZE_K * stride_weight_k
    
    if HAS_BIAS:
        bias_ptrs = bias_ptr + offs_n * stride_bias_n
        b = tl.load(bias_ptrs, mask=offs_n < N, other=0.0).to(tl.float32)
        acc += b[None, :]
    
    x = acc
    scaled_x = beta * x
    mask = scaled_x <= threshold
    safe_scaled_x = tl.where(mask, scaled_x, 0.0)
    exp_term = tl.exp(safe_scaled_x)
    log_term = tl.log(1.0 + exp_term)
    log_term = tl.where(mask, log_term, scaled_x)
    softplus_x = log_term / beta
    
    softplus_x = softplus_x.to(tl.float32)  # Cast to output dtype if needed, adjust based on actual use
    
    offs_out_m = offs_m[:, None]
    offs_out_n = offs_n[None, :]
    output_ptrs = output_ptr + offs_out_m * stride_output_m + offs_out_n * stride_output_n
    output_mask = (offs_out_m < M) & (offs_out_n < N)
    tl.store(output_ptrs, softplus_x, mask=output_mask)

def softplus_linear(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, beta: float = 1, threshold: int = 20) -> torch.Tensor:
    assert input.dim() == 2, "Input must be 2D"
    M, K = input.shape
    N, K_w = weight.shape
    assert K == K_w, f"Input and weight dimensions must match, got {K} and {K_w}"
    output = torch.empty((M, N), device=input.device, dtype=input.dtype)
    
    if bias is not None:
        assert bias.shape == (N,), f"Bias must have shape ({N},), got {bias.shape}"
        bias_ptr = bias
        HAS_BIAS = True
        stride_bias_n = bias.stride(0)
    else:
        bias_ptr = torch.tensor([], device=input.device, dtype=input.dtype)
        HAS_BIAS = False
        stride_bias_n = 0
    
    stride_input_m, stride_input_k = input.stride()
    stride_weight_n, stride_weight_k = weight.stride()
    stride_output_m, stride_output_n = output.stride()
    
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    grid_m = triton.cdiv(M, BLOCK_SIZE_M)
    grid_n = triton.cdiv(N, BLOCK_SIZE_N)
    
    softplus_linear_kernel[(grid_m, grid_n)](
        input, weight, bias_ptr, output,
        M, N, K,
        stride_input_m, stride_input_k,
        stride_weight_n, stride_weight_k,
        stride_bias_n,
        stride_output_m, stride_output_n,
        beta,
        threshold,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        HAS_BIAS=HAS_BIAS,
    )
    return output
