import triton
import triton.language as tl
import torch

@triton.jit
def ff_llama_kernel(
    # Pointers to matrices
    x_ptr, w1_ptr, w3_ptr, rms_w_ptr, output_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1 element
    stride_xm, stride_xk,
    stride_w1k, stride_w1n,
    stride_w3k, stride_w3n,
    stride_rms,
    stride_om, stride_on,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    EPS: tl.constexpr,
):
    """Kernel for computing the fused feed-forward operation."""
    
    # Program ID
    pid = tl.program_id(axis=0)
    
    # Block ID
    # This program is responsible for computing a block of output of size (BLOCK_SIZE_M, BLOCK_SIZE_N)
    block_m = pid // (N // BLOCK_SIZE_N)
    block_n = pid % (N // BLOCK_SIZE_N)
    
    # Initialize pointers to the start of the current block
    x_block_ptr = x_ptr + block_m * stride_xm
    w1_block_ptr = w1_ptr + block_n * stride_w1n
    w3_block_ptr = w3_ptr + block_n * stride_w3n
    rms_block_ptr = rms_w_ptr + block_m * stride_rms
    
    # Initialize accumulator for matrix multiplication
    acc1 = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    
    # Load RMS weights
    rm = tl.load(rms_block_ptr + tl.arange(0, BLOCK_SIZE_M))
    
    # Iterate over k dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load x matrix
        x = tl.load(x_block_ptr + k + tl.arange(0, BLOCK_SIZE_K))
        
        # Apply RMS normalization
        x_squared = x * x
        inv_rms = 1.0 / tl.sqrt(tl.sum(x_squared, axis=1) / K + EPS)
        x = x * inv_rms[:, None] * rm[:, None]
        
        # Load weights
        w1 = tl.load(w1_block_ptr + k * stride_w1k + 
                    tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_w1k +
                    tl.arange(0, BLOCK_SIZE_N)[None, :])
        w3 = tl.load(w3_block_ptr + k * stride_w3k +
                    tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_w3k +
                    tl.arange(0, BLOCK_SIZE_N)[None, :])
        
        # Accumulate matrix multiplications
        acc1 += tl.dot(x, w1)
        acc2 += tl.dot(x, w3)
    
    # Apply activation function (SiLU/Swish) and multiply
    silu_out = acc1 * (1 / (1 + tl.exp(-acc1)))
    output = silu_out * acc2
    
    # Store output
    output_block_ptr = output_ptr + block_m * stride_om + block_n * stride_on
    mask = (block_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:, None] < M
    mask = mask & (block_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None, :] < N
    tl.store(output_block_ptr, output, mask=mask)

def kernel_ff(x: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor, rms_w: torch.Tensor):
    """Wrapper function for the ff_llama_kernel."""
    
    # Check input constraints
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert w1.is_contiguous(), "Weight tensor w1 must be contiguous"
    assert w3.is_contiguous(), "Weight tensor w3 must be contiguous"
    assert rms_w.is_contiguous(), "RMS weight tensor must be contiguous"
    
    # Get dimensions
    M, K = x.shape
    _, N = w1.shape
    
    # Compute grid and block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    # Allocate output
    output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    # Launch kernel
    ff_llama_kernel[grid](
        x, w1, w3, rms_w, output,
        M, N, K,
        x.stride(0), x.stride(1),
        w1.stride(0), w1.stride(1),
        w3.stride(0), w3.stride(1),
        rms_w.stride(0),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        EPS=1e-6,
    )
    
    return output
