import triton
import triton.language as tl
import torch
import math

@triton.jit
def ff_llama(
    # Pointers to matrices
    x_ptr, w1_ptr, w3_ptr, rms_w_ptr, output_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_xm, stride_xk,
    stride_w1k, stride_w1n,
    stride_w3n, stride_w3k,
    stride_om, stride_on,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    eps: tl.float32
):
    """Kernel for computing the fused feedforward operation in LLaMA."""
    
    # -----------------------------------------------------------
    # Matrix multiplication for W1 computation (x @ w1)
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Offset pointers based on PID
    x_ptr = x_ptr + pid_m * stride_xm
    w1_ptr = w1_ptr + pid_n * stride_w1n
    
    # Initialize accumulator for matrix multiplication
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Load RMS weights
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    rms_w = tl.load(rms_w_ptr + offs_k)

    # Iterate to compute matrix multiplication
    for k in range(0, K, BLOCK_SIZE_K):
        # Load x and normalize with RMS
        x = tl.load(x_ptr + offs_k[None, :] * stride_xk)
        x_squared = x * x
        rms = tl.sqrt(tl.sum(x_squared, axis=1) / K + eps)
        x_normalized = x / rms[:, None]
        x_scaled = x_normalized * rms_w[None, :]
        
        # Load w1
        w1 = tl.load(w1_ptr + offs_k[:, None] * stride_w1k)
        
        # Compute matrix multiplication
        acc += tl.dot(x_scaled, w1)
    
    # Apply SILU activation
    acc_silu = acc * (1 / (1 + tl.exp(-acc)))
    
    # Matrix multiplication with W3
    w3_ptr = w3_ptr + pid_n * stride_w3n
    acc_final = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        w3 = tl.load(w3_ptr + offs_k[:, None] * stride_w3k)
        acc_final += tl.dot(acc_silu, w3)
    
    # Write output
    output_ptr = output_ptr + pid_m * stride_om + pid_n * stride_on
    tl.store(output_ptr, acc_final)

def kernel_ff(x: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor, rms_w: torch.Tensor):
    """Wrapper function for the ff_llama kernel."""
    
    # Check constraints
    assert x.is_cuda and w1.is_cuda and w3.is_cuda and rms_w.is_cuda
    assert x.is_contiguous() and w1.is_contiguous() and w3.is_contiguous() and rms_w.is_contiguous()
    
    # Get matrix dimensions
    M, K = x.shape
    _, N = w1.shape
    
    # Allocate output
    output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    # Set block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    
    # Launch kernel
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    ff_llama[grid](
        x, w1, w3, rms_w, output,
        M, N, K,
        x.stride(0), x.stride(1),
        w1.stride(0), w1.stride(1),
        w3.stride(0), w3.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        eps=1e-6,
    )
    
    return output
