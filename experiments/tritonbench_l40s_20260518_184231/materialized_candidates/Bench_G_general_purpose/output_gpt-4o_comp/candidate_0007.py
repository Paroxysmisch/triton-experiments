import triton
import triton.language as tl
import torch

# Define constants for block sizes and epsilon
BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 128
BLOCK_SIZE_K = 32
EPS = 1e-5

@triton.jit
def ff_llama_kernel(x_ptr, w1_ptr, w3_ptr, rms_w_ptr, output_ptr,
                    M, N, K, stride_xm, stride_xk, stride_w1k, stride_w1n,
                    stride_w3k, stride_w3n, stride_rmsk, stride_rmsn,
                    stride_om, stride_on, use_fp8, **meta):
    pid = tl.program_id(0)
    
    # Calculate the start of the block for the current program
    block_start_m = pid // meta['grid_n'] * BLOCK_SIZE_M
    block_start_n = pid % meta['grid_n'] * BLOCK_SIZE_N
    
    # Initialize accumulators
    acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over the K dimension in chunks
    for k in range(0, K, BLOCK_SIZE_K):
        # Load x, w1, w3, and rms weights
        x = tl.load(x_ptr + (block_start_m + tl.arange(0, BLOCK_SIZE_M))[:, None] * stride_xm + (k + tl.arange(0, BLOCK_SIZE_K)) * stride_xk, mask=tl.arange(0, BLOCK_SIZE_M)[:, None] < M)
        w1 = tl.load(w1_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * stride_w1k + (block_start_n + tl.arange(0, BLOCK_SIZE_N)) * stride_w1n, mask=tl.arange(0, BLOCK_SIZE_K)[:, None] < K)
        w3 = tl.load(w3_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * stride_w3k + (block_start_n + tl.arange(0, BLOCK_SIZE_N)) * stride_w3n, mask=tl.arange(0, BLOCK_SIZE_K)[:, None] < K)
        rms_w = tl.load(rms_w_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * stride_rmsk + (block_start_n + tl.arange(0, BLOCK_SIZE_N)) * stride_rmsn, mask=tl.arange(0, BLOCK_SIZE_K)[:, None] < K)
        
        # Perform matrix multiplications
        acc1 += tl.dot(x, w1)
        acc2 += tl.dot(x, w3)
    
    # Apply RMS normalization
    rms_norm = tl.sqrt(tl.sum(acc1 ** 2, axis=1, keepdim=True) / N + EPS)
    acc1 = acc1 / rms_norm
    
    # Apply scaled sigmoid activation
    silu = acc1 * tl.sigmoid(acc1)
    
    # Compute final output
    output = silu * acc2
    
    # Store results conditionally
    mask = (block_start_m + tl.arange(0, BLOCK_SIZE_M))[:, None] < M
    tl.store(output_ptr + (block_start_m + tl.arange(0, BLOCK_SIZE_M))[:, None] * stride_om + (block_start_n + tl.arange(0, BLOCK_SIZE_N)) * stride_on, output, mask=mask)

def kernel_ff(x, w1, w3, rms_w, use_fp8=False):
    # Ensure inputs are of correct types and shapes
    assert x.dtype == torch.float32
    assert w1.dtype == torch.float32
    assert w3.dtype == torch.float32
    assert rms_w.dtype == torch.float32

    M, K = x.shape
    _, N = w1.shape

    # Prepare output tensor
    output = torch.empty((M, N), dtype=torch.float32, device=x.device)

    # Define grid size
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N

    # Launch kernel
    ff_llama_kernel[(grid_m * grid_n,)](x, w1, w3, rms_w, output,
                                        M, N, K,
                                        x.stride(0), x.stride(1),
                                        w1.stride(0), w1.stride(1),
                                        w3.stride(0), w3.stride(1),
                                        rms_w.stride(0), rms_w.stride(1),
                                        output.stride(0), output.stride(1),
                                        use_fp8,
                                        grid_n=grid_n)

    return output
