import torch
import triton
import triton.language as tl

# Define the Triton kernel
@triton.jit
def ff_llama_kernel(
    # Pointers to matrices
    x_ptr, w1_ptr, w3_ptr, rms_w_ptr, output_ptr,
    # Matrix dimensions
    B, S, H, I,  # batch, sequence, hidden, intermediate
    # The strides for the matrices
    stride_xb, stride_xs, stride_xh,  # x strides
    stride_w1i, stride_w1h,  # w1 strides 
    stride_w3h, stride_w3i,  # w3 strides
    stride_ob, stride_os, stride_oh,  # output strides
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Block indices
    bid_m = pid // (I // BLOCK_SIZE_N)  # batch*seq index
    bid_n = pid % (I // BLOCK_SIZE_N)   # hidden index

    # Initialize pointers to blocks
    offs_m = (bid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_n = (bid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator
    acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Load RMS weights
    b_idx = offs_m // S
    s_idx = offs_m % S
    rms_w = tl.load(rms_w_ptr + offs_m)
    
    # Compute L2 norm for input normalization
    x_sum = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for k in range(0, H, BLOCK_SIZE_K):
        x_ptrs = x_ptr + b_idx[:, None] * stride_xb + s_idx[:, None] * stride_xs + \
                 (k + offs_k[None, :]) * stride_xh
        x_block = tl.load(x_ptrs, mask=k + offs_k[None, :] < H, other=0.0)
        x_sum += tl.sum(x_block * x_block, axis=1)
    
    rms_norm = tl.sqrt(x_sum / H) * rms_w
    
    # First matmul: x @ w1
    for k in range(0, H, BLOCK_SIZE_K):
        # Load x block
        x_ptrs = x_ptr + b_idx[:, None] * stride_xb + s_idx[:, None] * stride_xs + \
                 (k + offs_k[None, :]) * stride_xh
        x_block = tl.load(x_ptrs, mask=k + offs_k[None, :] < H, other=0.0)
        x_block = x_block / rms_norm[:, None]
        
        # Load w1 block
        w1_ptrs = w1_ptr + (k + offs_k[:, None]) * stride_w1h + offs_n[None, :] * stride_w1i
        w1_block = tl.load(w1_ptrs, mask=k + offs_k[:, None] < H, other=0.0)
        
        # Compute matmul
        acc1 += tl.dot(x_block, w1_block)
    
    # Apply SiLU activation
    acc1 = acc1 * (1 / (1 + tl.exp(-acc1)))
    
    # Second matmul: activation @ w3
    acc2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, I, BLOCK_SIZE_K):
        w3_ptrs = w3_ptr + offs_k[:, None] * stride_w3h + offs_n[None, :] * stride_w3i
        w3_block = tl.load(w3_ptrs, mask=k + offs_k[:, None] < I, other=0.0)
        acc2 += tl.dot(acc1, w3_block)
    
    # Store output
    output_ptrs = output_ptr + b_idx[:, None] * stride_ob + s_idx[:, None] * stride_os + \
                  offs_n[None, :] * stride_oh
    tl.store(output_ptrs, acc2)

# Wrapper function
def kernel_ff(x: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor, rms_w: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and w1.is_cuda and w3.is_cuda and rms_w.is_cuda
    
    # Extract dimensions
    B, S, H = x.shape  # batch, sequence, hidden
    _, I = w1.shape    # intermediate dimension
    
    # Allocate output
    output = torch.empty((B, S, H), device=x.device, dtype=x.dtype)
    
    # Configure meta-parameters
    BLOCK_SIZE_M = 16  # batch * sequence dimension
    BLOCK_SIZE_N = 32  # hidden dimension
    BLOCK_SIZE_K = 32  # intermediate dimension
    
    # Launch kernel
    grid = lambda meta: (
        triton.cdiv(B * S, meta['BLOCK_SIZE_M']) * triton.cdiv(I, meta['BLOCK_SIZE_N']),
    )
    
    ff_llama_kernel[grid](
        x, w1, w3, rms_w, output,
        B, S, H, I,
        x.stride(0), x.stride(1), x.stride(2),
        w1.stride(0), w1.stride(1),
        w3.stride(0), w3.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return output
