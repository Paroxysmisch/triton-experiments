import triton
import triton.language as tl
import torch

@triton.jit
def mean_dim_kernel(
    X_ptr,  # Pointer to input tensor
    Out_ptr,  # Pointer to output tensor
    stride_xm, stride_xn,  # Strides for input tensor
    stride_om, stride_on,  # Strides for output tensor
    M, N,  # Dimensions of the tensor
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr  # Block sizes
):
    # Calculate program ID for the current thread
    pid = tl.program_id(0)
    
    # Calculate row and column indices
    row_idx = pid // tl.cdiv(N, BLOCK_N)
    col_idx = pid % tl.cdiv(N, BLOCK_N)
    
    # Calculate offsets
    offs_m = row_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = col_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Create mask for valid elements
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    # Load and accumulate values
    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    acc += x
    
    # Calculate mean
    count = M * N
    acc = acc / float(count)
    
    # Store result
    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])

# Python wrapper function
def mean_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda and x.is_contiguous()
    
    # Handle dimension
    if dim < 0:
        dim = x.ndim + dim
    
    # Get input shape
    shape = list(x.shape)
    M = shape[dim]
    N = shape[1] if dim == 0 else shape[0]
    
    # Create output tensor
    out_shape = shape.copy()
    out_shape[dim] = 1
    output = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    
    # Calculate grid and block sizes
    BLOCK_M = 32
    BLOCK_N = 32
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    
    # Calculate strides
    stride_xm = x.stride(0)
    stride_xn = x.stride(1)
    stride_om = output.stride(0)
    stride_on = output.stride(1)
    
    # Launch kernel
    mean_dim_kernel[grid](
        x, output,
        stride_xm, stride_xn,
        stride_om, stride_on,
        M, N,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )
    
    return output
