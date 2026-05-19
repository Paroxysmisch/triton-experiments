import torch
import triton
import triton.language as tl

@triton.jit
def tanh_linear_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    in_features,
    out_features,
    stride_input_m,
    stride_input_k,
    stride_weight_n,
    stride_weight_k,
    stride_output_m,
    stride_output_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)
    
    input_block_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(in_features, out_features),
        strides=(stride_input_k, stride_input_m),
        offsets=(0, pid_m * BLOCK_SIZE_M),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_M),
        order=(1, 0)
    )
    
    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(out_features, in_features),
        strides=(stride_weight_n, stride_weight_k),
        offsets=(pid_n * BLOCK_SIZE_N, 0),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K),
        order=(1, 0)
    )
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(in_features, BLOCK_SIZE_K)):
        a = tl.load(input_block_ptr, boundary_check=(0, 1))
        b = tl.load(weight_block_ptr, boundary_check=(0, 1))
        acc += tl.dot(a, b, allow_tf32=True)
        input_block_ptr = tl.advance(input_block_ptr, (BLOCK_SIZE_K, 0))
        weight_block_ptr = tl.advance(weight_block_ptr, (0, BLOCK_SIZE_K))
    
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + rn, mask=rn < out_features, other=0.0)
        acc += bias[None, :]
    
    acc = tl.tanh(acc)
    
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(out_features, in_features),
        strides=(stride_output_n, stride_output_m),
        offsets=(pid_n * BLOCK_SIZE_N, pid_m * BLOCK_SIZE_M),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_M),
        order=(1, 0)
    )
    tl.store(output_block_ptr, acc, boundary_check=(0, 1))

def tanh_linear(input, weight, bias=None) -> torch.Tensor:
    assert input.dim() >= 2, "Input must have at least 2 dimensions"
    *leading_dims, in_features = input.shape
    out_features, weight_in_features = weight.shape
    assert in_features == weight_in_features, f"Expected input features ({in_features}) to match weight's in_features ({weight_in_features})"
    
    input_2d = input.reshape(-1, in_features)
    M, K = input_2d.shape
    N = out_features
    
    output_2d = torch.empty((M, N), device=input.device, dtype=input.dtype)
    
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    tanh_linear_kernel[grid](
        input_2d, weight, bias, output_2d,
        in_features, out_features,
        input_2d.stride(0), input_2d.stride(1),
        weight.stride(0), weight.stride(1),
        output_2d.stride(0), output_2d.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    output = output_2d.reshape(*leading_dims, N)
    return output
